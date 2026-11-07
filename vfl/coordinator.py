"""Central coordinator: membership, session lifecycle, epoch barriers and telemetry.

Trust boundary: the coordinator handles control state and telemetry only. It never
receives raw features, labels, top-model inputs, representations or gradients; parties
report counts, timings, byte sizes and aggregate metrics.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
import uuid


from . import config as cfgmod
from .proto import vfl_pb2 as pb
from .proto import vfl_pb2_grpc as pbg
import grpc

from . import defense
from . import limits
from .ids import check_party_id, check_session_id, check_ui_url, is_lab_identity, one_line
from .security import TLSFiles, effective_role
from .telemetry import SECURITY, JsonlWriter, ResourceMonitor, environment_info, now_iso, setup_security_log
from .transport import COORDINATOR_POLICY, SecurityGate, make_server, peer_cert, peer_ip, reject

log = logging.getLogger("vfl.coordinator")
ALIVE_S = 15.0
FINAL_GRACE_S = 30.0
ROLES = ("active", "passive")


class Session:
    def __init__(self, sid: str, cfg: dict, participants: list[dict], run_dir: str):
        self.id = sid
        self.config = cfg
        self.participants = participants
        self.party_ids = [p["party_id"] for p in participants]
        self.state = "STARTING"
        self.created = time.time()
        self.started = None
        self.finished = None
        self.undelivered = set(self.party_ids)
        self.stop_requested = False
        self.stop_time = None
        self.stop_reason = ""
        self.barriers: dict[int, set] = collections.defaultdict(set)
        self.barrier_release: dict[int, float] = {}
        self.epoch_done: dict[int, set] = collections.defaultdict(set)
        self.epoch_metrics: list[dict] = []
        self.party_epochs: dict[str, list] = collections.defaultdict(list)
        self.progress: dict[str, dict] = {}
        self.alignment: dict[str, dict] = {}
        self.finals: dict[str, dict] = {}
        self.errors: list[dict] = []
        self.events: list[dict] = []
        self.recent = collections.deque(maxlen=3000)
        self.batch_counts = collections.Counter()
        self.dir = os.path.join(run_dir, sid)
        os.makedirs(self.dir, exist_ok=True)
        self._writers: dict[str, JsonlWriter] = {}
        with open(os.path.join(self.dir, "config.json"), "w") as fh:
            json.dump({"session_id": sid, "config": cfg, "participants": participants}, fh, indent=2)

    def writer(self, name: str) -> JsonlWriter:
        if name not in self._writers:
            self._writers[name] = JsonlWriter(os.path.join(self.dir, f"{name}.jsonl"))
        return self._writers[name]

    def event(self, kind: str, msg: str, **kw):
        ev = {"t": time.time(), "time": now_iso(), "kind": kind, "msg": msg, **kw}
        self.events.append(ev)
        self.writer("events").write(ev)
        log.info("[%s] %s", self.id, msg)

    @property
    def completed_epochs(self) -> int:
        k = int(self.config.get("start_epoch", 0) or 0)
        while set(self.party_ids) <= self.epoch_done.get(k, set()):
            k += 1
        return k

    def brief(self) -> dict:
        last = self.epoch_metrics[-1] if self.epoch_metrics else {}
        return {"session_id": self.id, "name": self.config.get("name"), "state": self.state,
                "created": self.created, "started": self.started, "finished": self.finished,
                "participants": self.party_ids, "epochs": self.config["epochs"],
                "completed_epochs": self.completed_epochs, "batch_size": self.config["batch_size"],
                "inject_rtt_ms": self.config.get("inject_rtt_ms", 0), "codec": self.config.get("codec"),
                "defense": (self.config.get("defense") or {}).get("type"),
                "alignment": self.config.get("alignment"),
                "last_test_auroc": (last.get("test") or {}).get("auroc"),
                "last_val_auroc": (last.get("val") or {}).get("auroc"),
                "stop_reason": self.stop_reason, "errors": self.errors[-3:]}

    def summary(self) -> dict:
        return {"session_id": self.id, "state": self.state, "config": self.config,
                "participants": self.participants, "created": self.created, "started": self.started,
                "finished": self.finished,
                "wall_s": (self.finished - self.started) if self.finished and self.started else None,
                "alignment": self.alignment, "epoch_metrics": self.epoch_metrics,
                "party_epochs": dict(self.party_epochs), "finals": self.finals, "errors": self.errors,
                "events": self.events, "batch_record_counts": dict(self.batch_counts),
                "completed_epochs": self.completed_epochs, "stop_reason": self.stop_reason}

    def persist(self):
        with open(os.path.join(self.dir, "summary.json"), "w") as fh:
            json.dump(self.summary(), fh, indent=2, default=str)

    def close(self):
        for w in self._writers.values():
            w.close()


class CoordinatorState:
    def __init__(self, run_dir: str, default_config: dict, tls: TLSFiles):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.default_config = default_config
        self.tls = tls
        self.id = f"coord-{uuid.uuid4().hex[:6]}"
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.parties: dict[str, dict] = {}
        self.sessions: dict[str, Session] = {}
        self.order: list[str] = []
        self.current: str | None = None
        self.started_at = time.time()
        self.monitor = ResourceMonitor(interval=2.0).start()
        self.index_path = os.path.join(run_dir, "sessions_index.json")
        self.history = self._load_index()
        self.registry = None          # enrollment.IdentityRegistry in pki mode
        self.lab_mode = False
        self.lab = None               # attacks.LabRunner when --lab-mode
        self.gates: list = []

    def revoked(self) -> set:
        return set(self.registry.revoked) if self.registry else set()

    # ------------------------------------------------------------ persistence
    def _load_index(self) -> list[dict]:
        try:
            with open(self.index_path) as fh:
                return json.load(fh)
        except Exception:
            return []

    def _save_index(self):
        known = {h["session_id"]: h for h in self.history}
        for sid in self.order:
            known[sid] = self.sessions[sid].brief()
        self.history = sorted(known.values(), key=lambda h: h.get("created") or 0)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.history, fh, indent=2, default=str)
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------ membership
    def alive(self, p: dict) -> bool:
        return time.time() - p["last_seen"] < ALIVE_S

    def register(self, req, context=None) -> tuple[bool, str]:
        ok, err = self._register(req)
        if not ok and context is not None:
            kind = ("second_active" if "already registered" in err and req.role == "active" else
                    "label_claim" if "label ownership" in err else "bad_registration")
            SECURITY.event(kind, "protocol", "blocked", "coordinator role validation", "server-observed",
                           peer_ip(context), req.party_id, (peer_cert(context) or {}).get("name"), "Register",
                           detail=err)
        return ok, err

    def _register(self, req) -> tuple[bool, str]:
        pid = (req.party_id or "").strip()
        try:
            check_party_id(pid)
        except ValueError:
            return False, "invalid party_id"
        if req.role not in ROLES:
            return False, f"invalid role {req.role!r}; expected one of {ROLES}"
        if is_lab_identity(pid) and not self.lab_mode:
            # lab test identities (attack simulator) must never take part in a real session (RT-09)
            return False, f"{pid} is a lab test identity; it may only register with a coordinator in --lab-mode"
        if req.role == "active" and not req.address:
            return False, "active party must advertise its gRPC address"
        if len(req.address) > 255 or one_line(req.address) != req.address or " " in req.address:
            return False, "invalid address"
        try:
            ui_url = check_ui_url(req.ui_url)           # rendered as a link on the dashboards (RT-04)
        except ValueError as e:
            return False, str(e)
        if len(req.summary_json or "") > limits.MAX_SUMMARY_JSON:
            return False, "summary_json too large"
        try:
            summary = json.loads(req.summary_json or "{}")
        except json.JSONDecodeError:
            return False, "summary_json is not valid JSON"
        if not isinstance(summary, dict):
            return False, "summary_json must be an object"
        with self.lock:
            if req.role == "active":
                for other in self.parties.values():
                    if other["role"] == "active" and other["party_id"] != pid and self.alive(other):
                        return False, f"an active party ({other['party_id']}) is already registered"
            if req.role == "passive" and summary.get("has_labels"):
                return False, "passive party reports label ownership; labels must stay at the active party"
            prev = self.parties.get(pid)
            if prev and prev["role"] != req.role and self.alive(prev):
                return False, f"{pid} already registered with role {prev['role']}"
            self.parties[pid] = {"party_id": pid, "role": req.role, "address": req.address,
                                 "ui_url": ui_url, "hostname": one_line(req.hostname, 128), "summary": summary,
                                 "registered": time.time(), "last_seen": time.time(), "status": "idle",
                                 "progress": {}}
        log.info("registered %s as %s (host=%s)", pid, req.role, one_line(req.hostname, 128) or "?")
        return True, ""

    def heartbeat(self, req) -> pb.HeartbeatReply:
        with self.lock:
            p = self.parties.get(req.party_id)
            if not p:
                return pb.HeartbeatReply(command="NONE", known=False, revoked_serials=sorted(self.revoked()))
            p["last_seen"] = time.time()
            p["status"] = one_line(req.status, 32) or p["status"]
            if len(req.progress_json or "") <= limits.MAX_PROGRESS_JSON:
                try:
                    prog = json.loads(req.progress_json or "{}")
                    if isinstance(prog, dict):
                        p["progress"] = prog
                except json.JSONDecodeError:
                    pass
            s = self.sessions.get(self.current) if self.current else None
            if s and req.party_id in s.party_ids:
                s.progress[req.party_id] = p["progress"]
                if s.stop_requested and p["status"] not in ("idle", "done", "failed"):
                    return pb.HeartbeatReply(command="STOP", session_json=json.dumps({"session_id": s.id}),
                                             known=True, revoked_serials=sorted(self.revoked()))
                if req.party_id in s.undelivered and not s.stop_requested:
                    s.undelivered.discard(req.party_id)
                    p["status"] = "running"
                    if not s.undelivered and s.state == "STARTING":
                        s.state = "RUNNING"
                        s.started = time.time()
                    payload = {"session_id": s.id, "config": s.config, "participants": s.participants}
                    return pb.HeartbeatReply(command="START", session_json=json.dumps(payload), known=True,
                                             revoked_serials=sorted(self.revoked()))
        return pb.HeartbeatReply(command="NONE", known=True, revoked_serials=sorted(self.revoked()))

    # ------------------------------------------------------------ sessions
    def start_session(self, request: dict) -> dict:
        request = dict(request or {})
        resume_from = request.get("resume_from")
        overrides = request.get("config") or {}
        with self.lock:
            if self.current and self.sessions[self.current].state in ("STARTING", "RUNNING"):
                raise ValueError(f"session {self.current} is still running")
            if resume_from:
                prev = self.sessions.get(resume_from)
                base = prev.config if prev else self._load_session_config(resume_from)
                k = prev.completed_epochs if prev else self._completed_epochs_on_disk(resume_from)
                if k <= 0:
                    raise ValueError(f"session {resume_from} has no completed epoch to resume from")
                cfg = cfgmod.deep_merge(base, overrides)
                cfg["resume_from"] = resume_from
                cfg["start_epoch"] = k
                if k >= cfg["epochs"]:
                    raise ValueError(f"session {resume_from} already completed {k} epochs")
            else:
                cfg = cfgmod.deep_merge(self.default_config, overrides)
                cfg["start_epoch"] = 0
                cfg["resume_from"] = None
            cfgmod.validate_session_config(cfg)
            alive = [p for p in self.parties.values() if self.alive(p)]
            if cfg.get("participants"):
                wanted = set(cfg["participants"])
                missing = wanted - {p["party_id"] for p in alive}
                if missing:
                    raise ValueError(f"requested participants not alive: {sorted(missing)}")
                alive = [p for p in alive if p["party_id"] in wanted]
            actives = [p for p in alive if p["role"] == "active"]
            passives = sorted([p for p in alive if p["role"] == "passive"], key=lambda p: p["party_id"])
            if len(actives) != 1:
                raise ValueError(f"need exactly one live active party, found {len(actives)}")
            if len(passives) < int(cfg.get("min_passive", 1)):
                raise ValueError(f"need at least {cfg.get('min_passive', 1)} live passive parties, found {len(passives)}")
            check_data_consistency(cfg, actives + passives)
            busy = [p["party_id"] for p in actives + passives if p["status"] not in ("idle", "done", "failed")]
            if busy:
                raise ValueError(f"parties still busy with a previous session: {busy}")
            participants = [{"party_id": p["party_id"], "role": p["role"], "address": p["address"]}
                            for p in actives + passives]
            sid = time.strftime("s%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            s = Session(sid, cfg, participants, self.run_dir)
            self.sessions[sid] = s
            self.order.append(sid)
            self.current = sid
            s.event("session", f"session created with {len(participants)} parties "
                               f"(bs={cfg['batch_size']}, epochs={cfg['epochs']}, rtt={cfg['inject_rtt_ms']}ms, "
                               f"alignment={cfg['alignment']}, defense={cfg['defense']['type']}"
                               + (f", resume_from={resume_from}@epoch{cfg['start_epoch']}" if resume_from else "") + ")")
            self._save_index()
            return s.brief()

    def _load_session_config(self, sid: str) -> dict:
        check_session_id(sid)
        path = os.path.join(self.run_dir, sid, "config.json")
        if not os.path.exists(path):
            raise ValueError(f"unknown session {sid}")
        with open(path) as fh:
            return json.load(fh)["config"]

    def _completed_epochs_on_disk(self, sid: str) -> int:
        try:
            check_session_id(sid)
            with open(os.path.join(self.run_dir, sid, "summary.json")) as fh:
                return int(json.load(fh).get("completed_epochs", 0))
        except Exception:
            return 0

    def stop_session(self, sid: str, reason: str = "stopped by operator") -> dict:
        with self.cv:
            s = self.sessions.get(sid)
            if not s:
                raise ValueError(f"unknown session {sid}")
            if s.state in ("STARTING", "RUNNING"):
                s.stop_requested = True
                s.stop_time = time.time()
                s.stop_reason = reason
                s.state = "STOPPING"
                s.event("session", f"stop requested: {reason}")
                self.cv.notify_all()
                self._finish_if_done(s)
            return s.brief()

    def barrier(self, req) -> pb.BarrierReply:
        t0 = time.time()
        with self.cv:
            s = self.sessions.get(req.session_id)
            if not s:
                return pb.BarrierReply(proceed=False, reason="unknown session")
            if req.party_id not in s.party_ids:
                return pb.BarrierReply(proceed=False, reason="not a participant")
            e = int(req.epoch)
            s.barriers[e].add(req.party_id)
            self.cv.notify_all()
            deadline = t0 + float(s.config.get("barrier_timeout_s", 300))
            while not (set(s.party_ids) <= s.barriers[e]):
                if s.stop_requested or s.state in ("FAILED", "STOPPED"):
                    return pb.BarrierReply(proceed=False, reason=s.stop_reason or s.state,
                                           waited_ms=(time.time() - t0) * 1000)
                left = deadline - time.time()
                if left <= 0:
                    missing = sorted(set(s.party_ids) - s.barriers[e])
                    self._fail(s, f"epoch {e} barrier timed out waiting for {missing}")
                    return pb.BarrierReply(proceed=False, reason=f"barrier timeout; missing {missing}",
                                           waited_ms=(time.time() - t0) * 1000)
                self.cv.wait(timeout=min(left, 1.0))
            if e not in s.barrier_release:
                s.barrier_release[e] = time.time()
                s.event("epoch", f"epoch {e} started (all {len(s.party_ids)} parties at barrier)")
            return pb.BarrierReply(proceed=True, waited_ms=(time.time() - t0) * 1000)

    def _fail(self, s: Session, reason: str):
        s.errors.append({"t": time.time(), "reason": reason})
        s.stop_time = s.stop_time or time.time()
        s.stop_requested = True
        s.stop_reason = reason
        if s.state in ("STARTING", "RUNNING", "STOPPING"):
            s.state = "FAILING"
        s.event("error", reason)
        self.cv.notify_all()
        self._finish_if_done(s)

    def _finish_if_done(self, s: Session):
        """Close a session when every participant has sent its final report, or when it
        was stopped/failed and no participant is still busy."""
        if s.finished:
            return
        if not set(s.party_ids) <= set(s.finals):
            if not s.stop_requested:
                return
            # Stopped/failed: wait (bounded) for final reports from parties that received START.
            running = [pid for pid in s.party_ids
                       if pid not in s.finals and pid not in s.undelivered and pid in self.parties
                       and self.alive(self.parties[pid])]
            if running and time.time() - (s.stop_time or time.time()) < FINAL_GRACE_S:
                return
        s.finished = time.time()
        if s.state in ("FAILING",) or s.errors:
            s.state = "FAILED"
        elif s.stop_requested:
            s.state = "STOPPED"
        else:
            s.state = "COMPLETED"
        s.event("session", f"session {s.state.lower()} after {s.finished - (s.started or s.created):.1f}s")
        s.persist()
        self._save_index()

    def report(self, req) -> pb.ReportReply:
        if len(req.payload_json or "") > limits.MAX_REPORT_JSON:
            return pb.ReportReply(ok=False, error="report too large")
        if req.kind == "security":          # party security events, inside or outside sessions
            try:
                recs = json.loads(req.payload_json or "{}").get("events", [])
            except (json.JSONDecodeError, AttributeError):
                return pb.ReportReply(ok=False, error="bad json")
            if req.party_id not in self.parties or not isinstance(recs, list):
                return pb.ReportReply(ok=False, error="unknown party")
            for r in recs[:limits.MAX_SECURITY_EVENTS_PER_REPORT]:
                if isinstance(r, dict) and isinstance(r.get("attack_type"), str) and r["attack_type"]:
                    SECURITY.ingest(_party_security_record(r, req.party_id))
            return pb.ReportReply(ok=True)
        with self.cv:
            s = self.sessions.get(req.session_id)
            if not s or req.party_id not in s.party_ids:
                return pb.ReportReply(ok=False, error="unknown session or party")
            try:
                payload = json.loads(req.payload_json or "{}")
            except json.JSONDecodeError:
                return pb.ReportReply(ok=False, error="bad json")
            k = req.kind
            if not isinstance(payload, dict):
                return pb.ReportReply(ok=False, error="payload must be an object")
            is_active = (self.parties.get(req.party_id) or {}).get("role") == "active"
            if k == "batches":
                recs = payload.get("records", [])
                if not isinstance(recs, list):
                    return pb.ReportReply(ok=False, error="records must be a list")
                recs = [r for r in recs[:limits.MAX_RECORDS_PER_REPORT] if isinstance(r, dict)]
                w = s.writer(f"batches_{req.party_id}")
                for r in recs:
                    w.write(r)
                    s.recent.append(r)
                s.batch_counts[req.party_id] += len(recs)
            elif k == "alignment":
                s.alignment[req.party_id] = payload
                s.event("alignment", f"{req.party_id}: aligned {payload.get('n_aligned')} of "
                                     f"{payload.get('n_local')} local rows via {payload.get('mode')} "
                                     f"({'private' if payload.get('private') else 'NON-PRIVATE'}) in "
                                     f"{payload.get('align_ms', 0):.0f} ms")
            elif k == "epoch":
                try:
                    e = int(payload.get("epoch", -1))
                except (TypeError, ValueError):
                    return pb.ReportReply(ok=False, error="bad epoch")
                if not is_active:
                    # utility, leakage and privacy figures are the label owner's to report. v2.2 appended a
                    # passive's val/test/leakage keys to the session results, so any passive could forge the
                    # AUROC and attack AUC shown in the dashboard, inference panel and report (RT-07)
                    payload = {k2: v for k2, v in payload.items()
                               if k2 not in ("val", "test", "leakage", "privacy", "reached_target_s")}
                s.party_epochs[req.party_id].append(payload)
                s.writer("epochs").write({"party": req.party_id, **payload})
                s.epoch_done[e].add(req.party_id)
                if "val" in payload or "test" in payload:
                    s.epoch_metrics.append(payload)
                    t = (payload.get("test") or {}).get("auroc")
                    s.event("metrics", f"epoch {e}: train_loss={payload.get('train_loss', float('nan')):.4f} "
                                       f"val_auroc={_fmt((payload.get('val') or {}).get('auroc'))} "
                                       f"test_auroc={_fmt(t)}")
                s.persist()
            elif k == "final":
                s.finals[req.party_id] = payload
                s.event("final", f"{req.party_id} finished: {payload.get('status', 'ok')}")
                self._finish_if_done(s)
            elif k == "error":
                self._fail(s, f"{req.party_id}: {one_line(payload.get('error', 'unknown error'))}")
            elif k == "event":
                s.event(one_line(payload.get("kind", "party"), 32), f"{req.party_id}: {one_line(payload.get('msg', ''))}")
            else:
                return pb.ReportReply(ok=False, error=f"unknown report kind {k!r}")
            self.cv.notify_all()
        return pb.ReportReply(ok=True)

    def watchdog(self):
        """Fail a running session if a participant stops heartbeating."""
        while True:
            time.sleep(2.0)
            with self.cv:
                s = self.sessions.get(self.current) if self.current else None
                if not s or s.finished:
                    continue
                for pid in s.party_ids:
                    p = self.parties.get(pid)
                    if p and not self.alive(p) and pid not in s.finals:
                        self._fail(s, f"party {pid} stopped heartbeating (> {ALIVE_S:.0f}s)")
                        break
                if s.stop_requested:
                    self._finish_if_done(s)

    # ------------------------------------------------------------ identities
    def revoke_identity(self, serial: str, reason: str = "revoked by operator") -> dict:
        if not self.registry:
            raise ValueError("revocation needs pki mode")
        rec = self.registry.revoke(serial, reason)
        with self.cv:
            s = self.sessions.get(self.current) if self.current else None
            if s and not s.finished and rec["name"] in s.party_ids:
                self._fail(s, f"identity of {rec['name']} (serial {serial}) was revoked mid-session")
            p = self.parties.get(rec["name"])
            if p:
                p["last_seen"] = 0          # must re-register; its gate requests are now rejected
        return rec

    def inference_points(self) -> list[dict]:
        """Attack AUC vs test AUROC per finished session, for the inference-attack panel."""
        out = []
        sids = list(self.order)
        for h in self.history:
            if h["session_id"] not in sids:
                sids.append(h["session_id"])
        for sid in sids[-200:]:
            s = self.sessions.get(sid)
            if s:
                cfg, em, state = s.config, s.epoch_metrics, s.state
            else:
                path = os.path.join(self.run_dir, sid, "summary.json")
                if not os.path.exists(path):
                    continue
                try:
                    d = json.load(open(path))
                except Exception:
                    continue
                cfg, em, state = d.get("config", {}), d.get("epoch_metrics", []), d.get("state")
            if not em:
                continue
            last = em[-1]
            lk = last.get("leakage") or {}
            d = cfg.get("defense") or {}
            sigma = float(d.get("noise_multiplier", 0)) if d.get("type") == "grad_noise" else 0.0
            worst = max([v for v in (lk.get("norm_leak_auc"), lk.get("cosine_leak_auc")) if v is not None], default=None)
            out.append({"session_id": sid, "state": state, "sigma": sigma, "defense": d.get("type", "none"),
                        "label_rr_epsilon": d.get("label_rr_epsilon"),
                        "test_auroc": (last.get("test") or {}).get("auroc"),
                        "norm_leak_auc": lk.get("norm_leak_auc"), "cosine_leak_auc": lk.get("cosine_leak_auc"),
                        "attack_outcome": defense.leak_outcome(worst), "privacy": last.get("privacy")})
        return out

    # ------------------------------------------------------------ UI
    def ui_state(self) -> dict:
        with self.lock:
            parties = []
            for p in sorted(self.parties.values(), key=lambda p: (p["role"] != "active", p["party_id"])):
                parties.append({**{k: v for k, v in p.items() if k != "progress"},
                                "alive": self.alive(p), "age_s": time.time() - p["last_seen"],
                                "progress": p.get("progress", {})})
            cur = self.sessions.get(self.current) if self.current else None
            sessions = [self.sessions[sid].brief() for sid in reversed(self.order)]
            known = {s["session_id"] for s in sessions}
            sessions += [h for h in reversed(self.history) if h["session_id"] not in known][:30]
            return {"role": "coordinator", "node": self.id, "uptime_s": time.time() - self.started_at,
                    "tls": self.tls.enabled, "parties": parties, "sessions": sessions,
                    "current": self.session_detail(cur.id, tail=400) if cur else None,
                    "resources": self.monitor.latest(), "default_config": self.default_config,
                    "security": {**SECURITY.summary(), "recent": SECURITY.recent(150),
                                 "gate": {k: v for g in self.gates for k, v in g.stats.items()}},
                    "lab_mode": self.lab_mode, "lab": self.lab.view() if self.lab else None,
                    "pki": self.registry.view() if self.registry else None,
                    "inference": self.inference_points()}

    def session_detail(self, sid: str, tail: int = 2000) -> dict | None:
        with self.lock:
            s = self.sessions.get(sid)
            if not s:
                try:
                    check_session_id(sid)
                except ValueError:
                    return None
                path = os.path.join(self.run_dir, sid, "summary.json")
                if os.path.exists(path):
                    with open(path) as fh:
                        return {"from_disk": True, **json.load(fh)}
                return None
            return {**s.brief(), "config": s.config, "participants": s.participants,
                    "progress": s.progress, "alignment": s.alignment, "epoch_metrics": s.epoch_metrics,
                    "party_epochs": dict(s.party_epochs), "finals": s.finals,
                    "events": s.events[-200:], "recent_batches": list(s.recent)[-tail:]}


def check_data_consistency(cfg: dict, parties: list[dict]):
    """Refuse a session whose parties hold data from different prepare-data runs, or whose data does
    not match the session's `partition` / `overlap` / `binarize` expectations. Uses only the split
    metadata in each party's own manifest (no data, no column names)."""
    parts = {p["party_id"]: (p.get("summary") or {}).get("partition") for p in parties}
    with_m = {pid: m for pid, m in parts.items() if m}
    hashes = {m.get("manifest_sha256") for m in with_m.values()}
    if len(hashes) > 1:
        raise ValueError("parties hold data from different prepare-data runs (manifest hashes differ: "
                         + ", ".join(f"{pid}={m.get('manifest_sha256', '?')[:8]}" for pid, m in sorted(with_m.items()))
                         + "); redistribute the CSVs and manifests from one run")
    wanted = {k: cfg.get(k) for k in ("partition", "overlap", "binarize") if cfg.get(k) is not None}
    if wanted:
        missing = sorted(pid for pid, m in parts.items() if not m)
        if missing:
            raise ValueError(f"session expects {wanted} but {missing} have no data manifest to check; "
                             "regenerate data with this version of prepare-data or unset these keys")
        for pid, m in sorted(with_m.items()):
            if "partition" in wanted and m.get("strategy") != wanted["partition"]:
                raise ValueError(f"{pid} holds a {m.get('strategy')!r} split, session expects {wanted['partition']!r}")
            if "overlap" in wanted and abs(float(m.get("overlap") or 0) - float(wanted["overlap"])) > 1e-9:
                raise ValueError(f"{pid} holds overlap {m.get('overlap')}, session expects {wanted['overlap']}")
        if "binarize" in wanted:
            act = [m for pid, m in with_m.items() if "num_classes" in m]
            if not act or act[0].get("binarize") != wanted["binarize"]:
                raise ValueError(f"active party's labels were prepared with binarize="
                                 f"{act[0].get('binarize') if act else None!r}, session expects {wanted['binarize']!r}")
    want_k = cfg.get("num_classes")
    for m in with_m.values():
        k = m.get("num_classes")
        if want_k is not None and k is not None and int(want_k) < int(k):
            raise ValueError(f"session sets num_classes={want_k} but the active party's labels have {k} classes")
    # every party's bottom model must be able to use its data (cnn/gru need a shape in the manifest)
    from .models import SHAPED, resolve_bottom_kind
    for p in parties:
        kind = resolve_bottom_kind(cfg, p["party_id"])
        shape = (p.get("summary") or {}).get("shape")
        if kind in SHAPED and (not shape or len(shape) != SHAPED[kind]):
            raise ValueError(f"{p['party_id']} uses bottom_model {kind!r}, which needs a {SHAPED[kind]}-D input "
                             f"shape, but its data manifest has shape={shape}")


SECURITY_FIELDS = ("attack_type", "layer", "outcome", "defence", "source_ip", "claimed_identity", "cert_identity",
                   "method", "session", "detail", "t", "time")


def _party_security_record(r: dict, party_id: str) -> dict:
    """A security event as reported by a party: only known fields, bounded, single-line, attributed to the
    reporting party and marked as party-reported. v2.2 ingested the party's dict verbatim, so a party could
    forge 'server-observed' evidence or another node's events in the coordinator feed (RT-07)."""
    from .telemetry import LAYERS, OUTCOMES
    rec = {k: (one_line(r.get(k), 500 if k == "detail" else 128) if r.get(k) is not None else None)
           for k in SECURITY_FIELDS if k not in ("t",)}
    try:
        rec["t"] = float(r.get("t") or time.time())
    except (TypeError, ValueError):
        rec["t"] = time.time()
    rec["layer"] = rec["layer"] if rec["layer"] in LAYERS else "protocol"
    rec["outcome"] = rec["outcome"] if rec["outcome"] in OUTCOMES else "detected"
    rec["node"] = party_id
    rec["evidence"] = "party-reported"
    return rec


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


class CoordinatorServicer(pbg.CoordinatorServicer):
    def __init__(self, state: CoordinatorState):
        self.s = state

    def Register(self, request, context):
        c = peer_cert(context)
        role = effective_role(c, bool(self.s.tls.pki)) if self.s.tls.enabled else request.role   # once per Register
        if self.s.tls.enabled and role != request.role:
            reject(context, grpc.StatusCode.PERMISSION_DENIED,
                   f"certificate role {role!r} cannot register as {request.role!r}", "bad_role", "auth",
                   "certificate role", request.party_id, method="Register")
        ok, err = self.s.register(request, context)
        return pb.RegisterReply(ok=ok, error=err, coordinator_id=self.s.id)

    def Heartbeat(self, request, context):
        return self.s.heartbeat(request)

    def Barrier(self, request, context):
        return self.s.barrier(request)

    def Report(self, request, context):
        return self.s.report(request)


def _ensure_own_cert(tls: TLSFiles, pki_dir: str, registry, hours: float):
    """pki mode: the coordinator issues (and later renews) its own short-lived leaf from the intermediate."""
    from . import security as sec
    info = tls.leaf_info() if os.path.exists(os.path.join(tls.dir, "coordinator.crt")) else None
    want_dns = sec.dns_name("coordinator", (tls.pki or {}).get("trust_domain", sec.TRUST_DOMAIN), "coordinator")
    if info and info.get("role") == "coordinator" and info["not_after"] - time.time() > hours * 3600 / 3 \
            and want_dns in sec.leaf_dns_names(tls.read()[2]):       # v2.2 certs lack the role-qualified name
        return info
    info = sec.issue_static_pki(pki_dir, "coordinator", "coordinator", hours)
    if registry:
        registry.record_static(info, via="self")
    log.info("issued coordinator certificate serial %s valid until %s", info["serial"],
             time.strftime("%Y-%m-%d %H:%M", time.localtime(info["not_after"])))
    return info


def serve(listen: str, ui_host: str, ui_port: int, tls_dir: str | None, run_dir: str,
          session_config_path: str | None, ui_token: str | None = None, opts: dict | None = None):
    """opts: rate_limit_rps, max_peer_concurrency, max_concurrent_rpcs, join_mode, enroll_listen,
    cert_hours, lab_mode, lab_certs, ui_tls, ui_users, log_dir."""
    o = opts or {}
    setup_security_log("coordinator", os.path.join(o.get("log_dir") or "logs", "coordinator"))
    tls = TLSFiles(tls_dir, "coordinator")
    default_cfg = cfgmod.load_session_config(session_config_path)
    state = CoordinatorState(run_dir, default_cfg, tls)
    enroll_server = None
    if tls.enabled and tls.pki:
        from .enrollment import EnrollmentServicer, IdentityRegistry
        if not os.path.exists(os.path.join(tls_dir, "intermediate.key")):
            raise SystemExit("pki mode: the coordinator needs intermediate.key/.crt in --tls-dir")
        hours = float(o.get("cert_hours") or 24)
        state.registry = IdentityRegistry(tls_dir, os.path.join(run_dir, "pki"), o.get("join_mode") or "static", hours)
        for rec in _static_issued(tls_dir):
            state.registry.record_static(rec)
        _ensure_own_cert(tls, tls_dir, state.registry, hours)

        def renew_loop():
            while True:
                time.sleep(300)
                try:
                    _ensure_own_cert(tls, tls_dir, state.registry, hours)
                except Exception:
                    log.exception("coordinator certificate renewal failed")
        threading.Thread(target=renew_loop, daemon=True, name="cert-renew").start()
        if o.get("enroll_listen"):
            enroll_server, _ = make_server(tls, o["enroll_listen"], require_client_auth=False,
                                           gate=SecurityGate(TLSFiles(None, "coordinator"), {},
                                                             rate_limit_rps=o.get("rate_limit_rps") or 20))
            pbg.add_EnrollmentServicer_to_server(EnrollmentServicer(state.registry), enroll_server)
            enroll_server.start()
            log.info("enrollment service on %s (server-only TLS), join mode %s", o["enroll_listen"],
                     state.registry.join_mode)
    gate = SecurityGate(tls, COORDINATOR_POLICY, state.revoked, o.get("rate_limit_rps"), o.get("max_peer_concurrency"))
    state.gates.append(gate)
    server, port = make_server(tls, listen, gate=gate,
                               max_concurrent_rpcs=o.get("max_concurrent_rpcs") or None)
    pbg.add_CoordinatorServicer_to_server(CoordinatorServicer(state), server)
    server.start()
    threading.Thread(target=state.watchdog, daemon=True, name="watchdog").start()
    if o.get("lab_mode"):
        from .attacks.lab import LabRunner
        state.lab_mode = True
        state.lab = LabRunner(state, o.get("lab_certs") or tls_dir, listen, ui_port, ui_token, o.get("ui_tls"),
                              log_dir=o.get("log_dir") or "logs")
        log.warning("LAB MODE: attack simulator enabled from the dashboard (never use on a production deployment)")
    log.info("coordinator %s gRPC on %s (%s%s); runs -> %s", state.id, listen,
             "mTLS" if tls.enabled else "INSECURE", ", pki" if tls.pki else "", os.path.abspath(run_dir))
    log.info("environment: %s", json.dumps(environment_info()))
    from .dashboard import create_app, resolve_ui_auth, run_app
    if ui_port and ui_host not in ("127.0.0.1", "localhost", "::1") and o.get("ui_public_views"):
        log.warning("SECURITY: --ui-public-views: the coordinator dashboard on %s shows parties, sessions and "
                    "identities to anyone who can reach port %d", ui_host, ui_port)
    ui_token, private = resolve_ui_auth(ui_host, ui_token, o.get("ui_users"), "coordinator", o.get("log_dir"),
                                        bool(o.get("ui_public_views")))
    if state.lab:
        state.lab.token = ui_token
    app = create_app(state, "coordinator", token=ui_token, users=o.get("ui_users"), private_views=private)
    run_app(app, ui_host, ui_port, tls=o.get("ui_tls") and tls)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(1)
        if enroll_server:
            enroll_server.stop(1)


def _static_issued(tls_dir: str) -> list[dict]:
    p = os.path.join(tls_dir, "issued_static.json")
    try:
        return json.load(open(p))
    except Exception:
        return []
