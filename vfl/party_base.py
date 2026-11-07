"""Shared machinery for active and passive party nodes: registration, heartbeats,
command handling, telemetry shipping, epoch barriers and dashboard state."""
from __future__ import annotations

import collections
import json
import logging
import os
import socket
import threading
import time
import traceback

import grpc

from .data import PartyData
from .proto import vfl_pb2 as pb
from .proto import vfl_pb2_grpc as pbg
from .security import TLSFiles
from .telemetry import RING, SECURITY, ResourceMonitor, SessionLog, environment_info, setup_security_log
from .transport import make_channel

log = logging.getLogger("vfl.party")


class StopRequested(Exception):
    pass


class PartyNode:
    role = "party"

    def __init__(self, party_id: str, data_path: str, coordinator: str, tls_dir: str | None,
                 log_root: str = "logs", ckpt_root: str = "checkpoints", ui_url: str = "",
                 advertise: str = "", manifest: str | None = None, enroll_addr: str | None = None,
                 lab_mode: bool = False):
        self.party_id = party_id
        self.data_path = data_path
        self.coordinator_addr = coordinator
        self.tls = TLSFiles(tls_dir, party_id)
        self.log_root = log_root
        self.ckpt_root = ckpt_root
        self.ui_url = ui_url
        self.advertise = advertise
        self.data = PartyData.load(data_path, require_label=(self.role == "active"), manifest_path=manifest)
        self.channel = make_channel(self.tls, coordinator, "coordinator")
        self.coord = pbg.CoordinatorStub(self.channel)
        self.status = "starting"
        self.registered = False
        self.coord_ok = False
        self.coord_error = ""
        self.session_id: str | None = None
        self.session_cfg: dict | None = None
        self.participants: list[dict] = []
        self.progress: dict = {}
        self.counters = collections.Counter()
        self.epoch_history: list[dict] = []
        self.history: list[dict] = []          # finished sessions on this node
        self.slog: SessionLog | None = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self._ship_buf: list[dict] = []
        self._ship_lock = threading.Lock()
        self.monitor = ResourceMonitor(interval=1.0).start()
        self.env = environment_info()
        self.started_at = time.time()
        self._session_thread: threading.Thread | None = None
        self._hb_now = threading.Event()
        self.revoked: set = set()                 # serials from the coordinator's revocation list
        self.enroll_addr = enroll_addr            # pki mode: where to renew certificates
        self.lab_mode = lab_mode
        self._sec_buf: list[dict] = []
        setup_security_log(party_id, os.path.join(log_root, party_id))
        SECURITY.sinks.append(self._queue_security)

    def _queue_security(self, rec: dict):
        if rec.get("evidence") != "attacker-reported":
            with self._ship_lock:
                self._sec_buf.append(rec)

    def _reconnect(self):
        old = self.channel
        self.channel = make_channel(self.tls, self.coordinator_addr, "coordinator")
        self.coord = pbg.CoordinatorStub(self.channel)
        try:
            old.close()
        except Exception:
            pass

    def renew_loop(self):
        """pki mode: renew the short-lived certificate when a third of its lifetime is left."""
        from .enrollment import renew_party
        while True:
            time.sleep(30)
            info = self.tls.leaf_info()
            if not info or not info.get("role") or not self.enroll_addr:
                continue
            life = info["not_after"] - info["not_before"]
            from .security import dns_name, leaf_dns_names
            want = dns_name(self.party_id, (self.tls.pki or {}).get("trust_domain", "oncloud9"), info["role"])
            legacy = want not in leaf_dns_names(self.tls.read()[2])     # v2.2 certificate: renew now (v2.3 names)
            if info["not_after"] - time.time() > life / 3 and not legacy:
                continue
            try:
                new = renew_party(self.tls, self.enroll_addr)
                log.info("renewed certificate: serial %s valid until %s", new["serial"],
                         time.strftime("%Y-%m-%d %H:%M", time.localtime(new["not_after"])))
                self._reconnect()
            except Exception as e:
                log.error("certificate renewal failed: %s", e)

    # ------------------------------------------------------------ coordinator link
    def register(self):
        summary = self.data.summary()
        req = pb.RegisterRequest(party_id=self.party_id, role=self.role, address=self.advertise,
                                 ui_url=self.ui_url, hostname=socket.gethostname(),
                                 summary_json=json.dumps(summary))
        while True:
            try:
                rep = self.coord.Register(req, timeout=10)
                if rep.ok:
                    self.registered = True
                    self.coord_ok = True
                    self.coord_error = ""
                    if self.status == "starting":
                        self.status = "idle"
                    log.info("registered with coordinator %s as %s", rep.coordinator_id, self.role)
                    return
                self.coord_error = rep.error
                log.error("registration rejected: %s", rep.error)
            except grpc.RpcError as e:
                self.coord_ok = False
                self.coord_error = f"{e.code().name}: {e.details()}"
                log.warning("coordinator unreachable (%s); retrying", self.coord_error)
            time.sleep(3)

    def heartbeat_loop(self):
        while True:
            try:
                rep = self.coord.Heartbeat(pb.HeartbeatRequest(
                    party_id=self.party_id, status=self.status,
                    progress_json=json.dumps(dict(self.progress), default=str)), timeout=5)
                self.coord_ok = True
                self.revoked = set(rep.revoked_serials)
                if not rep.known:
                    log.warning("coordinator does not know this party (restarted?); re-registering")
                    self.register()
                elif rep.command == "START":
                    self._on_start(json.loads(rep.session_json))
                elif rep.command == "STOP":
                    if not self.stop_event.is_set():
                        log.warning("coordinator requested STOP")
                        self.stop_event.set()
                        self.on_stop()
            except grpc.RpcError as e:
                if self.coord_ok:
                    log.warning("heartbeat failed: %s", e.code().name)
                self.coord_ok = False
                self.coord_error = f"{e.code().name}: {e.details()}"
            except Exception:  # pragma: no cover
                log.exception("heartbeat loop error")
            self._hb_now.wait(1.0)
            self._hb_now.clear()

    def _on_start(self, payload: dict):
        from .ids import check_session_id
        try:                                   # these become file paths on this machine: refuse anything else
            check_session_id(payload["session_id"])
            if (payload.get("config") or {}).get("resume_from") is not None:
                check_session_id(payload["config"]["resume_from"])
        except (KeyError, ValueError, TypeError) as e:
            log.error("refusing START with an unsafe session id: %s", e)
            SECURITY.event("unsafe_session_id", "protocol", "blocked", "session id validation", "server-observed",
                           None, "coordinator", None, "Heartbeat", None, str(e))
            return
        with self.lock:
            if self._session_thread and self._session_thread.is_alive():
                log.error("START received while a session is still running; ignoring")
                return
            self.stop_event.clear()
            self.session_id = payload["session_id"]
            self.session_cfg = payload["config"]
            self.participants = payload["participants"]
            self.counters = collections.Counter()
            self.epoch_history = []
            self.progress = {"session_id": self.session_id, "phase": "starting"}
            self.status = "running"
            self.slog = SessionLog(self.log_root, self.party_id, self.session_id)
            self._session_thread = threading.Thread(target=self._session_wrapper, daemon=True,
                                                    name=f"session-{self.session_id}")
            self._session_thread.start()

    def _session_wrapper(self):
        sid = self.session_id
        log.info("session %s started", sid)
        t0 = time.time()
        try:
            summary = self.run_session(self.session_cfg)
            status = "stopped" if self.stop_event.is_set() else "ok"
            summary["status"] = status
            self.status = "done"
        except StopRequested:
            summary = {"status": "stopped"}
            self.status = "done"
        except Exception as e:
            if self.stop_event.is_set():
                log.warning("session %s interrupted by stop request (%s)", sid, e)
                summary = {"status": "stopped", "note": str(e)}
                self.status = "done"
                return self._finish(sid, summary, t0)
            log.error("session %s failed: %s\n%s", sid, e, traceback.format_exc())
            self.report("error", {"error": f"{type(e).__name__}: {e}"})
            summary = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            self.status = "failed"
        self._finish(sid, summary, t0)

    def _finish(self, sid, summary, t0):
        summary.setdefault("session_wall_s", time.time() - t0)
        summary["party"] = self.party_id
        summary["role"] = self.role
        self.flush_telemetry()
        if self.slog:
            self.slog.write_json("run_summary.json", summary)
            self.slog.close()
        self.report("final", summary)
        self.history.insert(0, {"session_id": sid, "status": summary.get("status"),
                                "finished": time.time(), "summary": _brief(summary)})
        self.progress = {**self.progress, "phase": summary.get("status")}
        self._hb_now.set()   # tell the coordinator promptly that this node is free again
        log.info("session %s finished (%s); logs in %s", sid, summary.get("status"),
                 self.slog.dir if self.slog else "-")

    def on_stop(self):
        """Hook for subclasses (wake waiting RPC threads)."""

    def run_session(self, cfg: dict) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def report(self, kind: str, payload: dict) -> bool:
        try:
            rep = self.coord.Report(pb.ReportRequest(session_id=self.session_id or "", party_id=self.party_id,
                                                     kind=kind, payload_json=json.dumps(payload, default=_jd)),
                                    timeout=15)
            return rep.ok
        except grpc.RpcError as e:
            log.warning("report(%s) failed: %s", kind, e.code().name)
            return False

    def barrier(self, epoch: int) -> bool:
        t0 = time.time()
        while True:
            try:
                rep = self.coord.Barrier(pb.BarrierRequest(session_id=self.session_id, party_id=self.party_id,
                                                           epoch=epoch, kind="epoch_start"),
                                         timeout=float(self.session_cfg.get("barrier_timeout_s", 300)) + 10)
                if not rep.proceed:
                    log.warning("coordinator did not release epoch %d: %s", epoch, rep.reason)
                self.counters["barrier_wait_ms"] += (time.time() - t0) * 1000
                return rep.proceed
            except grpc.RpcError as e:
                if self.stop_event.is_set() or time.time() - t0 > 60:
                    raise
                log.warning("barrier call failed (%s); retrying", e.code().name)
                time.sleep(1)

    # ------------------------------------------------------------ telemetry
    def record_batch(self, rec: dict):
        rec = {"session": self.session_id, "party": self.party_id, "role": self.role, **rec}
        if self.slog:
            self.slog.batch(rec)
        with self._ship_lock:
            self._ship_buf.append(rec)

    def flush_security(self):
        with self._ship_lock:
            sec, self._sec_buf = self._sec_buf, []
        if sec:
            try:
                self.coord.Report(pb.ReportRequest(session_id=self.session_id or "", party_id=self.party_id,
                                                   kind="security", payload_json=json.dumps({"events": sec}, default=_jd)),
                                  timeout=10)
            except grpc.RpcError:
                pass

    def flush_telemetry(self):
        with self._ship_lock:
            buf, self._ship_buf = self._ship_buf, []
        for i in range(0, len(buf), 500):
            self.report("batches", {"records": buf[i:i + 500]})

    def shipper_loop(self):
        while True:
            time.sleep(float((self.session_cfg or {}).get("telemetry_flush_s", 2.0)))
            if self._ship_buf and self.session_id:
                self.flush_telemetry()
            if self._sec_buf:
                self.flush_security()

    def start_background(self):
        self.register()
        threading.Thread(target=self.heartbeat_loop, daemon=True, name="heartbeat").start()
        threading.Thread(target=self.shipper_loop, daemon=True, name="telemetry").start()
        if self.tls.pki:
            threading.Thread(target=self.renew_loop, daemon=True, name="cert-renew").start()

    def ckpt_path(self, session_id: str, epoch: int) -> str:
        return os.path.join(self.ckpt_root, self.party_id, session_id, f"ckpt_epoch{epoch}.pt")

    # ------------------------------------------------------------ dashboard
    def base_ui_state(self) -> dict:
        cfg = self.session_cfg or {}
        recent = list(self.slog.recent)[-400:] if self.slog else []
        return {
            "role": self.role, "node": self.party_id, "status": self.status,
            "uptime_s": time.time() - self.started_at, "tls": self.tls.enabled,
            "coordinator": {"address": self.coordinator_addr, "connected": self.coord_ok,
                            "registered": self.registered, "error": self.coord_error},
            "data": {**self.data.summary(), "file": os.path.basename(self.data_path),
                     "feature_names": self.data.feature_names[:60]},
            "session": {"session_id": self.session_id, "config": cfg, "participants": self.participants},
            "progress": self.progress, "counters": dict(self.counters),
            "epochs": self.epoch_history, "recent_batches": recent,
            "resources": self.monitor.latest(), "events": list(RING.lines)[-150:],
            "history": self.history[:20], "env": self.env,
            "security": {**SECURITY.summary(), "recent": SECURITY.recent(60)},
            "identity": self.tls.leaf_info() if self.tls.enabled else None,
            "lab_mode": self.lab_mode,
            "log_dir": self.slog.dir if self.slog else None,
        }


def _brief(summary: dict) -> dict:
    keys = ("status", "committed_batches", "optimizer_steps", "throughput_batches_per_s", "final_test",
            "batch_ms_p50", "batch_ms_p95", "training_wall_s", "error")
    return {k: summary.get(k) for k in keys if k in summary}


def _jd(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)
