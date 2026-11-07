"""Passive (feature-owning) party: computes representations of its own feature partition,
sends them to the active party, and updates its bottom model with the returned gradient.
It never holds labels, never sees the loss and never sends raw feature columns (FR-4).
"""
from __future__ import annotations

import logging
import time

import grpc
import numpy as np

from . import codec
from . import passive_guard as guard
from . import psi_wire
from .alignment import ROUND_WORK, DHPSIParty, ExperimentalAlignment, run_psi_member
from .config import alignment_salt
from .core import PassiveCore, aligned_features, load_checkpoint, save_checkpoint
from .data import batch_digest, epoch_batches, eval_batches
from .models import configure_torch
from .party_base import PartyNode, StopRequested
from .proto import vfl_pb2 as pb
from .proto import vfl_pb2_grpc as pbg
from .telemetry import SECURITY, latency_summary, percentile
from .transport import make_channel

log = logging.getLogger("vfl.passive")
RETRY_CODES = (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE)


class PassiveNode(PartyNode):
    role = "passive"

    def __init__(self, *a, malicious: str | None = None, **kw):
        super().__init__(*a, **kw)
        if malicious and not self.lab_mode:
            raise ValueError("--malicious is a lab-only attack simulation; it requires --lab-mode")
        self.malicious = malicious
        self.core: PassiveCore | None = None
        self.align_report = None
        self.guard: dict = {}
        self.gmon: guard.GradientMonitor | None = None

    # ------------------------------------------------------------ helpers
    def _active(self):
        act = [p for p in self.participants if p["role"] == "active"][0]
        ch = make_channel(self.tls, act["address"], act["party_id"])
        return act, ch, pbg.ActivePartyStub(ch)

    def _header(self, epoch=0, batch=0, attempt=1, msg_type="activation", phase="train", digest=""):
        return pb.MsgHeader(session_id=self.session_id, epoch=epoch, batch=batch, attempt=attempt,
                            sender=self.party_id, msg_type=msg_type, phase=phase, batch_digest=digest,
                            sent_unix_ns=time.time_ns())

    def _call(self, fn, req, cfg, what: str):
        """Unary call with deadline + bounded retries; returns (reply, attempts, timeouts)."""
        attempt, timeouts = 1, 0
        while True:
            if self.stop_event.is_set():
                raise StopRequested()
            if hasattr(req, "header"):
                req.header.attempt = attempt
            try:
                return fn(req, timeout=float(cfg["rpc_deadline_s"])), attempt, timeouts
            except grpc.RpcError as e:
                code = e.code()
                if code in RETRY_CODES and attempt <= int(cfg["max_retries"]) and not self.stop_event.is_set():
                    timeouts += code == grpc.StatusCode.DEADLINE_EXCEEDED
                    self.counters["rpc_timeouts" if code == grpc.StatusCode.DEADLINE_EXCEEDED else "rpc_unavailable"] += 1
                    self.counters["retries"] += 1
                    log.warning("%s attempt %d failed with %s (%s) after deadline %.1fs; retrying",
                                what, attempt, code.name, e.details(), float(cfg["rpc_deadline_s"]))
                    self.report("event", {"kind": "timeout",
                                          "msg": f"{what} attempt {attempt}: {code.name}; retrying"})
                    attempt += 1
                    time.sleep(float(cfg["retry_backoff_s"]) * attempt)
                    continue
                if self.stop_event.is_set() or code == grpc.StatusCode.CANCELLED:
                    # the active party cancels in-flight RPCs only when the session is stopped
                    self.stop_event.set()
                    raise StopRequested() from e
                raise RuntimeError(f"{what} failed: {code.name}: {e.details()}") from e

    # ------------------------------------------------------------ alignment
    def align(self, stub, cfg) -> dict:
        t0 = time.time()
        ids = self.data.ids
        up = down = 0
        if cfg["alignment"] == "experimental":
            ad = ExperimentalAlignment(alignment_salt(cfg))
            req = pb.AlignRequest(header=self._header(msg_type="align"), mode="experimental", round=1,
                                  items=ad.encode(ids))
            rep, _, _ = self._call(stub.Align, req, {**cfg, "rpc_deadline_s": cfg["slot_timeout_s"] * 2}, "align")
            if not rep.ok:
                raise RuntimeError(f"alignment rejected: {rep.error}")
            up, down = req.ByteSize(), rep.ByteSize()
            res = ad.rows_for(ids, list(rep.items))
        elif sum(1 for p in self.participants if p.get("role") == "passive") >= 2:
            res, up, down = self._align_multi_psi(stub, cfg, ids)
        else:
            psi = DHPSIParty(ids)
            t_b = time.time()
            blinded = psi.blind_own()
            req1 = pb.AlignRequest(header=self._header(msg_type="psi_r1"), mode="dh_psi", round=1, items=blinded)
            rep1, _, _ = self._call(stub.Align, req1, {**cfg, "rpc_deadline_s": max(600.0, cfg["rpc_deadline_s"])},
                                    "psi round 1")
            if not rep1.ok:
                raise RuntimeError(f"PSI round 1 rejected: {rep1.error}")
            peer_double = psi.reblind(list(rep1.peer_items))
            res, positions = psi.finish_initiator(list(rep1.items), peer_double)
            req2 = pb.AlignRequest(header=self._header(msg_type="psi_r2"), mode="dh_psi", round=2, indices=positions)
            rep2, _, _ = self._call(stub.Align, req2, cfg, "psi round 2")
            if not rep2.ok:
                raise RuntimeError(f"PSI round 2 rejected: {rep2.error}")
            up, down = req1.ByteSize() + req2.ByteSize(), rep1.ByteSize() + rep2.ByteSize()
            log.info("DH-PSI local crypto %.1fs", time.time() - t_b)
        self.counters["align_bytes_up"] += up
        self.counters["align_bytes_down"] += down
        rep = {**res.report(), "align_ms": (time.time() - t0) * 1000, "app_bytes_up": up, "app_bytes_down": down}
        return res, rep

    def _align_multi_psi(self, stub, cfg, ids):
        """Three or more parties: this party's side of multi-party DH-PSI (vfl/alignment.run_psi_member), every
        message relayed through the active party. Returns (result, bytes up, bytes down)."""
        n = {"up": 0, "down": 0}
        slow = {**cfg, "rpc_deadline_s": max(600.0, float(cfg["rpc_deadline_s"]))}
        poll = {**cfg, "rpc_deadline_s": max(float(cfg["rpc_deadline_s"]), psi_wire.WORK_WAIT_S + 30.0)}

        def call(rnd, items=None, task="", wait_s=0.0):
            if self.stop_event.is_set():
                raise StopRequested()
            req = psi_wire.request(self._header(msg_type=f"psi_r{rnd}"), rnd, items, task)
            rep, _, _ = self._call(stub.Align, req, poll if rnd == ROUND_WORK else slow, f"psi round {rnd}")
            n["up"] += req.ByteSize()
            n["down"] += rep.ByteSize()
            return psi_wire.reply_dict(rep)

        t_b = time.time()
        res, stats = run_psi_member(ids, call, deadline_s=max(3600.0, float(cfg["slot_timeout_s"]) * 3))
        log.info("multi-party DH-PSI (%s parties): re-blinded %d sets / %d items in %.1fs", stats.get("n_parties"),
                 stats["tasks"], stats["items_reblinded"], time.time() - t_b)
        return res, n["up"], n["down"]

    # ------------------------------------------------------------ session
    def run_session(self, cfg: dict) -> dict:
        env = configure_torch(cfg.get("torch_threads", 1), cfg.get("deterministic", True))
        t_setup = time.time()
        act, ch, stub = self._active()
        self.progress.update({"phase": "aligning"})
        res, arep = self.align(stub, cfg)
        if self.stop_event.is_set():
            raise StopRequested()
        X, splits, scaler = aligned_features(self.data, res.rows, cfg, return_scaler=True)
        arep["splits"] = {k: int(len(v)) for k, v in splits.items()}
        self.align_report = arep
        self.report("alignment", arep)
        log.info("aligned %d of %d local rows via %s (%s)", res.n_aligned, len(self.data.ids), res.mode,
                 "private" if res.private else "NON-PRIVATE experimental")
        self.core = core = PassiveCore(cfg, X, self.party_id, shape=self.data.shape, scaler=scaler)
        log.info("bottom model %s (%d params)", core.kind, core.n_params)
        self.guard = guard.guard_config(cfg)                 # v2.3: defences against a malicious active party
        self.gmon = guard.GradientMonitor() if self.guard["grad_monitor"] != "off" else None
        if any((self.guard["grad_clip"], self.guard["max_drift"], self.gmon, self.guard["leakage_audit"] != "off")):
            log.info("passive guard: grad_clip=%s max_param_drift=%s grad_monitor=%s leakage_audit=%s (r2_max %.2f)",
                     self.guard["grad_clip"], self.guard["max_drift"], self.guard["grad_monitor"],
                     self.guard["leakage_audit"], self.guard["r2_max"])
        start_epoch = int(cfg.get("start_epoch", 0))
        recovery_t0 = t_setup
        if cfg.get("resume_from"):
            core.load(load_checkpoint(self.ckpt_path(cfg["resume_from"], start_epoch - 1)))
            log.info("resumed bottom model from %s epoch %d", cfg["resume_from"], start_epoch - 1)
        evals = eval_batches(splits, cfg["eval_batch_size"])
        setup_s = time.time() - t_setup
        rtt_half = float(cfg.get("inject_rtt_ms", 0.0)) / 2000.0
        fault = cfg.get("fault") or {}
        dup_set = {tuple(x) for x in fault.get("duplicate", [])}
        steps_at_start = core.optimizer_steps
        batch_ms_all, epoch_rows = [], []
        t_train = time.time()
        first_commit_latency = None
        for epoch in range(start_epoch, cfg["epochs"]):
            if not self.barrier(epoch) or self.stop_event.is_set():
                break
            te = time.time()
            batches = epoch_batches(splits["train"], cfg["batch_size"], cfg["seed"], epoch)
            self.progress.update({"phase": "train", "epoch": epoch, "epochs": cfg["epochs"],
                                  "n_batches": len(batches), "batch": 0})
            ep_bytes = {"tensor_up": 0, "tensor_down": 0, "app_up": 0, "app_down": 0}
            ep_batch_ms, ep_wait = [], []
            for b, idx in enumerate(batches):
                if self.stop_event.is_set():
                    raise StopRequested()
                rec = self._train_batch(stub, cfg, epoch, b, idx, rtt_half, (epoch, b) in dup_set)
                if first_commit_latency is None and cfg.get("resume_from"):
                    first_commit_latency = time.time() - recovery_t0
                batch_ms_all.append(rec["batch_ms"])
                ep_batch_ms.append(rec["batch_ms"])
                ep_wait.append(rec["server_wait_ms"] + rec["network_ms"] + rec["injected_rtt_ms"] + rec["server_delay_ms"])
                for k1, k2 in (("tensor_up", "tensor_bytes_up"), ("tensor_down", "tensor_bytes_down"),
                               ("app_up", "app_bytes_up"), ("app_down", "app_bytes_down")):
                    ep_bytes[k1] += rec[k2]
                self.progress.update({"batch": b + 1, "last_batch_ms": rec["batch_ms"]})
            train_s = time.time() - te
            # evaluation forward passes (no gradients flow back)
            self.progress.update({"phase": "eval"})
            for j, (split, idx) in enumerate(evals):
                h = self._tamper(core.embed(idx), idx, cfg)
                t = codec.encode_cfg(h, cfg)
                req = pb.ExchangeRequest(header=self._header(epoch, j, phase="eval", digest=batch_digest(idx)),
                                         activation=t)
                rep, _, _ = self._call(stub.Exchange, req, cfg, f"eval e{epoch} b{j}")
                ep_bytes["tensor_up"] += len(t.data)
                ep_bytes["app_up"] += req.ByteSize()
                ep_bytes["app_down"] += rep.ByteSize()
            row = {"epoch": epoch, "epoch_s": time.time() - te, "train_s": train_s, "train_batches": len(batches),
                   "batch_ms_p50": percentile(ep_batch_ms, 50), "batch_ms_p95": percentile(ep_batch_ms, 95),
                   "wait_ms_mean": float(np.mean(ep_wait)) if ep_wait else None,
                   "throughput_batches_per_s": len(batches) / train_s if train_s > 0 else None,
                   "bytes": ep_bytes, "optimizer_steps": core.optimizer_steps,
                   "mib_per_epoch_app": (ep_bytes["app_up"] + ep_bytes["app_down"]) / 2 ** 20,
                   "mib_per_epoch_tensor": (ep_bytes["tensor_up"] + ep_bytes["tensor_down"]) / 2 ** 20}
            if self.guard["leakage_audit"] != "off":
                row["feature_leakage"] = self._leakage_audit(core, splits, cfg, epoch)
            if self.guard["grad_clip"] or self.guard["max_drift"] or self.gmon:
                row["guard"] = {"grad_rows_clipped": int(self.counters["guard_rows_clipped"]),
                                "drift_projections": int(core.drift_projections),
                                "param_drift": round(float(core.last_drift), 4),
                                "grad_flags": int(self.gmon.flags) if self.gmon else 0}
            if cfg.get("checkpoint", True):
                save_checkpoint(self.ckpt_path(self.session_id, epoch), core.state(epoch))
            row["resources"] = {**self.monitor.latest(), "t": time.time()}    # v2.1: CPU/RSS over time
            epoch_rows.append(row)
            self.epoch_history.append(row)
            self.slog.epochs.write(row)
            self.report("epoch", row)
            log.info("epoch %d: %d batches in %.2fs (p50 %.1f ms, p95 %.1f ms), %.3f MiB app traffic",
                     epoch, len(batches), train_s, row["batch_ms_p50"] or 0, row["batch_ms_p95"] or 0,
                     row["mib_per_epoch_app"])
        train_wall = time.time() - t_train
        committed = core.optimizer_steps - steps_at_start
        n_train = len(splits["train"])
        tot = {k: sum(r["bytes"][k] for r in epoch_rows) for k in ("tensor_up", "tensor_down", "app_up", "app_down")}
        n_ep = max(1, len(epoch_rows))
        summary = {
            "session_id": self.session_id, "setup_s": setup_s, "training_wall_s": train_wall,
            "alignment": arep, "committed_batches": committed, "optimizer_steps": core.optimizer_steps,
            "optimizer_steps_this_session": core.optimizer_steps - steps_at_start,
            "scheduled_batches": int(sum(r["train_batches"] for r in epoch_rows)),
            "throughput_batches_per_s": committed / train_wall if train_wall > 0 else None,
            "throughput_train_only": committed / sum(r["train_s"] for r in epoch_rows) if epoch_rows else None,
            **{k: v for k, v in latency_summary(batch_ms_all).items() if k != "n"},
            "epoch_s_median": float(np.median([r["epoch_s"] for r in epoch_rows])) if epoch_rows else None,
            "bytes_total": tot,
            "mib_per_epoch_tensor": (tot["tensor_up"] + tot["tensor_down"]) / 2 ** 20 / n_ep,
            "mib_per_epoch_app": (tot["app_up"] + tot["app_down"]) / 2 ** 20 / n_ep,
            "mib_per_entity_epoch_app": (tot["app_up"] + tot["app_down"]) / 2 ** 20 / n_ep / max(1, n_train),
            "n_train_entities": n_train, "epochs": epoch_rows,
            "recovery_first_commit_s": first_commit_latency,
            "model_params_passive": core.n_params, "bottom_model": core.kind, "codec": cfg["codec"],
            "counters": dict(self.counters), "resources": self.monitor.summary(),
            "env": {**self.env, **env}, "tls": self.tls.enabled,
        }
        ch.close()
        return summary

    def _tamper(self, h, idx, cfg):
        """LAB ONLY (--lab-mode --malicious ...): a deviating passive party (outside the core threat model)."""
        mode = self.malicious
        if not mode:
            return h
        self.counters["malicious_batches"] += 1
        if mode == "free_rider":
            return np.zeros_like(h)
        if mode.startswith("scale:"):
            return (h * float(mode.split(":")[1])).astype(np.float32)
        if mode.startswith("noise:"):
            rng = np.random.default_rng(self.counters["malicious_batches"])
            return rng.normal(0, float(mode.split(":")[1]), size=h.shape).astype(np.float32)
        if mode == "nan":
            out = h.copy()
            out[0, 0] = np.nan
            return out
        raise ValueError(f"unknown malicious mode {mode!r}")

    def _leakage_audit(self, core, splits, cfg, epoch) -> dict:
        """How much of this party's raw features the active party could recover from what was sent (ridge)."""
        ev = splits["val"] if len(splits["val"]) >= 10 else splits["test"]
        res = guard.leakage_audit(lambda idx: core.embed(np.asarray(idx)), core.X, splits["train"], ev,
                                  seed=int(cfg["seed"]) + epoch)
        r2, limit = res["ridge_r2"], self.guard["r2_max"]
        res["limit"] = limit
        if np.isfinite(r2) and r2 > limit:
            action = self.guard["leakage_audit"]
            msg = (f"epoch {epoch}: features recoverable from the representations sent (ridge R^2 {r2:.3f} > "
                   f"{limit:.2f}); {'leaving the session' if action == 'stop' else 'continuing (warn)'}")
            SECURITY.event("feature_leakage", "inference", "detected", "passive leakage audit (ridge decoder)",
                           "server-observed", None, "active", None, "Exchange", self.session_id, msg)
            self.report("event", {"kind": "leakage", "msg": msg})
            log.warning("LEAKAGE AUDIT: %s", msg)
            if action == "stop":
                raise guard.GuardStop(msg)
        return res

    def _validated_gradient(self, rep, shape, cfg):
        sec = cfg.get("security") or {}
        try:
            g = codec.decode(rep.gradient, expected_shape=tuple(shape))
            if sec.get("validate_tensors", True):
                codec.validate(g, tuple(shape), sec.get("max_grad_row_norm"))
            gd = self.guard or {}
            if self.gmon is not None:
                flag = self.gmon.check(g)                 # on the gradient as SENT, before clipping
                if flag:
                    msg = (f"gradient norm {flag['norm']:.3g} is {flag['z']:.1f} sd above the running trend "
                           f"({flag['consecutive']} in a row)")
                    SECURITY.event("suspicious_gradient", "integrity", "detected", "passive gradient monitor",
                                   "server-observed", None, "active", None, "Exchange", self.session_id, msg)
                    if gd.get("grad_monitor") == "stop" and self.gmon.should_stop():
                        raise guard.GuardStop(f"leaving the session: {msg}")
            if gd.get("grad_clip"):
                g, n = guard.clip_rows(g, gd["grad_clip"])
                self.counters["guard_rows_clipped"] += n
            return g
        except codec.TensorRejected as e:
            SECURITY.event({"non_finite": "non_finite_tensor", "norm_bound": "norm_bound"}.get(e.reason, "malformed_tensor"),
                           "data", "blocked", "tensor validation (shape, finite, norm bound)", "server-observed",
                           None, "active", "active", "Exchange", self.session_id, f"gradient rejected: {e}")
            raise RuntimeError(f"received an invalid gradient from the active party ({e}); stopping") from e

    def _train_batch(self, stub, cfg, epoch, b, idx, rtt_half, inject_duplicate) -> dict:
        core = self.core
        t0 = time.perf_counter()
        h = self._tamper(core.forward(idx), idx, cfg)
        t1 = time.perf_counter()
        tens = codec.encode_cfg(h, cfg)
        req = pb.ExchangeRequest(header=self._header(epoch, b, digest=batch_digest(idx)), activation=tens)
        t2 = time.perf_counter()
        if rtt_half:
            time.sleep(rtt_half)                       # emulated uplink propagation delay
        t3 = time.perf_counter()
        rep, attempts, timeouts = self._call(stub.Exchange, req, cfg, f"exchange e{epoch} b{b}")
        t4 = time.perf_counter()
        if rtt_half:
            time.sleep(rtt_half)                       # emulated downlink propagation delay
        t5 = time.perf_counter()
        if rep.dropped:
            # straggler policy: the active party committed this batch without us - no local update
            core.skip_step()
            self.counters["dropped_batches"] += 1
            grad, tm = None, {"bwd_ms": 0.0, "step_ms": 0.0}
            t6 = t7 = time.perf_counter()
        else:
            grad = self._validated_gradient(rep, h.shape, cfg)
            t6 = time.perf_counter()
            tm = core.backward_step(grad)              # exactly one local step per committed batch
            t7 = time.perf_counter()
            self.counters["committed_batches"] += 1
        self.counters["tensor_bytes_up"] += len(tens.data)
        self.counters["tensor_bytes_down"] += len(rep.gradient.data)
        dup_ack = None
        if inject_duplicate:
            # FR-8 / E5: replay the already-committed message; the active party must answer with
            # its cached gradient (duplicate=True) and neither side may take another step.
            steps_before = core.optimizer_steps
            rep_dup = stub.Exchange(req, timeout=float(cfg["rpc_deadline_s"]))
            dup_ack = bool(rep_dup.duplicate)
            self.counters["duplicates_injected"] += 1
            self.counters["duplicate_acks"] += int(dup_ack)
            msg = (f"injected duplicate for e{epoch} b{b}: server duplicate={dup_ack}, "
                   f"passive steps unchanged={core.optimizer_steps == steps_before}")
            log.warning("FAULT INJECTION: %s", msg)
            self.report("event", {"kind": "fault", "msg": msg})
        rpc_ms = (t4 - t3) * 1000
        rec = {
            "phase": "train", "epoch": epoch, "batch": b, "n": int(len(idx)), "t": time.time(),
            "attempts": attempts, "timeouts": timeouts,
            "fwd_ms": (t1 - t0) * 1000, "encode_ms": (t2 - t1) * 1000, "rpc_ms": rpc_ms,
            "server_wait_ms": rep.server_wait_ms, "server_compute_ms": rep.server_compute_ms,
            "server_delay_ms": rep.server_delay_ms,
            "network_ms": max(0.0, rpc_ms - rep.server_wait_ms - rep.server_compute_ms - rep.server_delay_ms),
            "injected_rtt_ms": ((t3 - t2) + (t5 - t4)) * 1000 if rtt_half else 0.0, "decode_ms": (t6 - t5) * 1000,
            "bwd_ms": tm["bwd_ms"], "step_ms": tm["step_ms"], "batch_ms": (t7 - t0) * 1000,
            "tensor_bytes_up": len(tens.data), "tensor_bytes_down": len(rep.gradient.data),
            "expected_tensor_bytes": codec.expected_tensor_bytes(len(idx), cfg["embed_dim"], cfg["codec"],
                                                                 float(cfg.get("topk_fraction", 0.25))),
            "dropped": bool(rep.dropped),
            "app_bytes_up": req.ByteSize(), "app_bytes_down": rep.ByteSize(),
            "duplicate_reply": bool(rep.duplicate), "duplicate_injected_ack": dup_ack,
            "optimizer_steps": core.optimizer_steps,
        }
        self.record_batch(rec)
        return rec

    def ui_state(self) -> dict:
        st = self.base_ui_state()
        cfg = self.session_cfg or {}
        st["privacy"] = {
            "holds": ["raw passive-party features", f"bottom model f_{self.party_id}"],
            "receives": ["dL/dH (gradient of its own representation) from the active party"],
            "sends": [f"representations H = f(X) [batch x {cfg.get('embed_dim', '?')}] ({cfg.get('codec', 'fp32')})"],
            "never_leaves": ["raw feature columns", "entity ids (only salted hashes / PSI blinds during alignment)"],
            "never_receives": ["labels", "loss", "top model"],
            "alignment": self.align_report,
        }
        return st


def serve(party_id: str, data: str, coordinator: str, tls_dir: str | None, ui_host: str, ui_port: int,
          log_root: str, ckpt_root: str, ui_token: str | None = None, public_ui_url: str | None = None,
          manifest: str | None = None, opts: dict | None = None):
    o = opts or {}
    scheme = "https" if o.get("ui_tls") else "http"
    ui_url = public_ui_url or f"{scheme}://{ui_host if ui_host not in ('0.0.0.0', '') else 'localhost'}:{ui_port}"
    node = PassiveNode(party_id, data, coordinator, tls_dir, log_root=log_root, ckpt_root=ckpt_root, ui_url=ui_url,
                       manifest=manifest, enroll_addr=o.get("enroll_addr"), lab_mode=bool(o.get("lab_mode")),
                       malicious=o.get("malicious"))
    log.info("passive party %s: %s; data rows=%d features=%d", party_id,
             "mTLS" if node.tls.enabled else "INSECURE", len(node.data.ids), node.data.X.shape[1])
    if node.malicious:
        log.warning("LAB MODE: this passive party is MALICIOUS (%s)", node.malicious)
    from .dashboard import create_app, resolve_ui_auth, run_app
    ui_token, private = resolve_ui_auth(ui_host, ui_token, o.get("ui_users"), party_id, log_root,
                                        bool(o.get("ui_public_views")))
    run_app(create_app(node, "passive", token=ui_token, users=o.get("ui_users"), private_views=private), ui_host, ui_port,
            tls=o.get("ui_tls") and node.tls)
    node.start_background()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
