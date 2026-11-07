"""Active (label-owning) party: owns labels, its bottom model, the top model and the loss.

Serves the ActiveParty gRPC service. For each aligned batch it waits for every passive
party's representation, runs top-model forward/backward/step once, and returns each
party's representation gradient. Duplicate or retried messages for a committed batch
receive the cached gradient and never trigger another optimizer step (FR-8).
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import grpc
import numpy as np

from . import codec, defense
from . import psi_wire
from .limits import check_slot, psi_item_cap
from .alignment import DHPSIParty, ExperimentalAlignment, MultiPSIHub
from .config import alignment_salt
from .core import ActiveCore, aligned_features, load_checkpoint, save_checkpoint, utility_metrics
from .data import batch_digest, epoch_batches, eval_batches
from .models import configure_torch
from .party_base import PartyNode, StopRequested
from .proto import vfl_pb2 as pb
from .proto import vfl_pb2_grpc as pbg
from .telemetry import SECURITY, percentile
from .transport import ACTIVE_POLICY, SecurityGate, make_server, peer_cert, peer_ip, reject

log = logging.getLogger("vfl.active")
KEEP_SLOTS = 64


class AnomalyDetector:
    """Lab defence against MALICIOUS passive parties (outside the core threat model, v2 item 16):
    flags near-constant representations (free-riders) and row norms far above that party's running
    average (scaled / poisoned representations). Detection only - the session continues."""

    def __init__(self, z: float = 6.0, warmup: int = 5):
        self.z, self.warmup = z, warmup
        self.stats: dict = {}
        self.flagged = collections.Counter()

    def check(self, party: str, arr: np.ndarray, session: str, context):
        norms = np.linalg.norm(arr, axis=1)
        st = self.stats.setdefault(party, {"n": 0, "mean": 0.0, "m2": 0.0})
        spread = float(np.mean(np.std(arr, axis=0))) if len(arr) > 1 else 1.0
        if st["n"] >= self.warmup:
            mu = st["mean"]
            sd = max((st["m2"] / max(st["n"] - 1, 1)) ** 0.5, 1e-6 + 0.05 * abs(mu))
            zmax = float((norms.mean() - mu) / sd)
            kind = None
            if spread < 1e-6:
                kind, why = "free_rider", "representation does not vary across entities (constant/zero)"
            elif zmax > self.z:
                kind, why = "poisoned_representation", f"mean row norm {norms.mean():.3g} is {zmax:.1f} sd above {mu:.3g}"
            if kind:
                self.flagged[(party, kind)] += 1
                if self.flagged[(party, kind)] in (1, 10, 100) or self.flagged[(party, kind)] % 500 == 0:
                    SECURITY.event(kind, "integrity", "detected", "representation anomaly detector",
                                   "server-observed", peer_ip(context), party, (peer_cert(context) or {}).get("name"),
                                   "Exchange", session, f"{why} (flag #{self.flagged[(party, kind)]})")
                return kind
        x = float(norms.mean())
        st["n"] += 1
        d = x - st["mean"]
        st["mean"] += d / st["n"]
        st["m2"] += d * (x - st["mean"])
        return None


class Slot:
    __slots__ = ("key", "idx", "acts", "senders", "dropped", "arrived", "req_bytes", "tensor_in", "first", "last", "computing", "done",
                 "error", "grads", "result", "delay_ms", "replied")

    def __init__(self, key, idx):
        self.key = key
        self.idx = idx
        self.acts: dict[str, np.ndarray] = {}
        self.senders: set = set()
        self.dropped: set = set()
        self.arrived: dict = {}
        self.req_bytes: dict[str, int] = {}
        self.tensor_in: dict[str, int] = {}
        self.first = None
        self.last = None
        self.computing = False
        self.done = False
        self.error = None
        self.grads: dict[str, pb.Tensor] = {}
        self.result: dict = {}
        self.delay_ms = 0.0
        self.replied: set = set()


class ActiveSession:
    """Per-session state shared between RPC threads and the session thread."""

    def __init__(self, node: "ActiveNode", sid: str, cfg: dict, participants: list[dict]):
        self.node = node
        self.id = sid
        self.cfg = cfg
        self.passive_ids = sorted(p["party_id"] for p in participants if p["role"] == "passive")
        self.cv = threading.Condition()
        # alignment
        self.align_inputs: dict[str, list[bytes]] = {}
        self.align_order: list[bytes] | None = None
        self.align_result = None
        self.align_error = None
        self.aligned = threading.Event()
        self.psi: DHPSIParty | None = None
        self.mpsi: MultiPSIHub | None = None             # >= 3 parties (v2.1)
        self.mpsi_lock = threading.Lock()
        self.align_t0 = None
        # training
        self.core: ActiveCore | None = None
        self.ready = threading.Event()
        self.splits = None
        self._batches: dict[int, list[np.ndarray]] = {}
        self.evals = None
        self.slots: collections.OrderedDict = collections.OrderedDict()
        self.evicted: set = set()
        self.train_committed = collections.Counter()
        self.eval_committed = collections.Counter()
        self.epoch_events: dict[int, threading.Event] = collections.defaultdict(threading.Event)
        self.eval_preds: dict[int, dict[str, list]] = {}
        self.epoch_losses: dict[int, list] = collections.defaultdict(list)
        self.last_commit_t = None
        self.dup_suppressed = 0
        self.delayed = set()
        self.digest_mismatch = 0
        self.anomaly = AnomalyDetector()
        self.current_epoch: int | None = None       # set when the epoch barrier releases (slot window)
        self.psi_round1: dict = {}                    # sender -> (items digest, reply)

    # -------------------------------------------------------------- schedule
    def batches(self, epoch: int) -> list[np.ndarray]:
        if epoch not in self._batches:
            self._batches = {k: v for k, v in self._batches.items() if k >= epoch - 1}
            self._batches[epoch] = epoch_batches(self.splits["train"], self.cfg["batch_size"],
                                                 self.cfg["seed"], epoch)
        return self._batches[epoch]

    # -------------------------------------------------------------- alignment
    def handle_align(self, req, context) -> pb.AlignReply:
        sender = req.header.sender
        if sender not in self.passive_ids:
            self._reject(context, grpc.StatusCode.PERMISSION_DENIED, f"{sender} is not a passive participant",
                         "not_participant", "protocol", "session participant list", sender, method="Align")
        if req.mode != self.cfg["alignment"]:
            return pb.AlignReply(ok=False, error=f"alignment mode mismatch: {req.mode} vs {self.cfg['alignment']}")
        ids = self.node.data.ids
        cap = psi_item_cap(len(ids), self.cfg)
        if len(req.items) > cap:
            # every item costs modular exponentiations (PSI) or memory; and in two-party PSI each submitted item
            # is a membership query against this party's ids (RT-08)
            self._reject(context, grpc.StatusCode.RESOURCE_EXHAUSTED,
                         f"{len(req.items)} alignment items exceed the limit {cap} (security.max_align_items)",
                         "flood", "protocol", "alignment input bound", sender, method="Align")
        with self.cv:
            if self.align_t0 is None:
                self.align_t0 = time.time()
        if req.mode == "experimental":
            ad = ExperimentalAlignment(alignment_salt(self.cfg))
            with self.cv:
                self.align_inputs[sender] = list(req.items)
                if set(self.passive_ids) <= set(self.align_inputs) and self.align_order is None:
                    order = ad.intersect([ad.encode(ids)] + [self.align_inputs[p] for p in self.passive_ids])
                    self.align_order = order
                    self.align_result = ad.rows_for(ids, order)
                    self.align_inputs.clear()   # do not retain peers' hashes
                    self.aligned.set()
                    self.cv.notify_all()
                deadline = time.time() + float(self.cfg["slot_timeout_s"])
                while self.align_order is None:
                    if self.node.stop_event.is_set():
                        return pb.AlignReply(ok=False, error="session stopped")
                    if time.time() > deadline:
                        return pb.AlignReply(ok=False, error="timed out waiting for other parties' alignment input")
                    self.cv.wait(0.5)
                return pb.AlignReply(ok=True, items=self.align_order, n_aligned=len(self.align_order))
        if len(self.passive_ids) >= 2:                  # three or more parties: multi-party DH-PSI (v2.1)
            return psi_wire.hub_reply(self._multi_psi(ids), sender, req)
        # two-party DH-PSI (v2 protocol, unchanged)
        if req.round == 1:
            import hashlib
            digest = hashlib.sha256(b"".join(req.items)).hexdigest()
            with self.cv:
                prev = self.psi_round1.get(sender)
                if prev is not None:
                    if prev[0] == digest and prev[1] is not None:
                        return prev[1]                  # an RPC retry of the same message: same answer
                    # v2.2 answered round 1 any number of times, each with fresh blinds: an unlimited membership
                    # oracle over candidate ids plus a CPU denial of service (RT-08)
                    return pb.AlignReply(ok=False, error="PSI round 1 was already answered in this session")
                self.psi_round1[sender] = (digest, None)
                self.psi = DHPSIParty(ids)
                psi = self.psi
            reblinded = psi.reblind(list(req.items))
            own = psi.blind_own()
            rep = pb.AlignReply(ok=True, items=reblinded, peer_items=own)
            with self.cv:
                self.psi_round1[sender] = (digest, rep)
            return rep
        if req.round == 2:
            if self.psi is None:
                return pb.AlignReply(ok=False, error="PSI round 2 before round 1")
            try:
                res = self.psi.finish_responder(list(req.indices))
            except ValueError as e:
                return pb.AlignReply(ok=False, error=str(e))
            with self.cv:
                self.align_result = res
                self.psi = None
                self.aligned.set()
                self.cv.notify_all()
            return pb.AlignReply(ok=True, n_aligned=res.n_aligned)
        return pb.AlignReply(ok=False, error=f"bad PSI round {req.round}")

    def _multi_psi(self, ids) -> MultiPSIHub:
        """The session's multi-party PSI hub, created on the first PSI request (its own blinding is the slow part,
        so it is built under a dedicated lock, never under self.cv)."""
        with self.mpsi_lock:
            if self.mpsi is None:
                def finished(res):
                    with self.cv:
                        self.align_result = res
                        self.aligned.set()
                        self.cv.notify_all()
                self.mpsi = MultiPSIHub(ids, self.node.party_id, self.passive_ids, on_finish=finished,
                                        item_cap=psi_item_cap(len(ids), self.cfg))
            return self.mpsi

    # -------------------------------------------------------------- exchange
    def _reject(self, context, code, msg, attack_type, layer, defence, sender, method="Exchange", outcome="blocked"):
        reject(context, code, msg, attack_type, layer, defence, sender, self.id, method, outcome)

    def handle_exchange(self, req, context) -> pb.ExchangeReply:
        h = req.header
        sender = h.sender
        if sender not in self.passive_ids:
            self._reject(context, grpc.StatusCode.PERMISSION_DENIED, f"{sender} is not a passive participant of {self.id}",
                         "not_participant", "protocol", "session participant list", sender)
        if not self.ready.wait(timeout=float(self.cfg["slot_timeout_s"])):
            context.abort(grpc.StatusCode.UNAVAILABLE, "active party not ready (alignment/setup pending)")
        phase = h.phase or "train"
        epoch, b = int(h.epoch), int(h.batch)
        try:
            # the honest schedule only: no negative indices, no far-future epochs or batches. v2.2 created a slot
            # for any (epoch, batch), so one passive could evict the live slot and park server threads (RT-08)
            epochs = int(self.cfg["epochs"])
            check_slot(phase, epoch, max(b, 0), 1 << 30, self.current_epoch, epochs)
            n_b = len(self.batches(epoch)) if phase == "train" else len(self.evals)
            done = (self.train_committed if phase == "train" else self.eval_committed)[epoch]
            check_slot(phase, epoch, b, n_b, self.current_epoch, epochs, committed=done)
            idx = self.batches(epoch)[b] if phase == "train" else self.evals[b][1]
        except (ValueError, IndexError) as e:
            self._reject(context, grpc.StatusCode.OUT_OF_RANGE, f"no {phase} batch {b} in epoch {epoch}: {e}",
                         "bad_batch", "protocol", "deterministic schedule", sender)
        if h.batch_digest != batch_digest(idx):
            self.digest_mismatch += 1
            self._reject(context, grpc.StatusCode.FAILED_PRECONDITION,
                         f"batch membership digest mismatch for {phase} e{epoch} b{b}: "
                         f"parties disagree on alignment/schedule", "wrong_digest", "protocol", "batch digest", sender)
        sec_cfg = self.cfg.get("security") or {}
        try:
            arr = codec.decode(req.activation, expected_shape=(len(idx), int(self.cfg["embed_dim"])))
            if sec_cfg.get("validate_tensors", True):
                codec.validate(arr, (len(idx), int(self.cfg["embed_dim"])), sec_cfg.get("max_row_norm"))
        except codec.TensorRejected as e:
            kind = {"non_finite": "non_finite_tensor", "norm_bound": "norm_bound"}.get(e.reason, "malformed_tensor")
            self._reject(context, grpc.StatusCode.INVALID_ARGUMENT, str(e), kind, "data",
                         "tensor validation (shape, finite, norm bound)", sender)
        if sec_cfg.get("anomaly_detection") and phase == "train":
            self.anomaly.check(sender, arr, self.id, context)
        key = (phase, epoch, b)
        t_arr = time.perf_counter()
        duplicate = False
        with self.cv:
            slot = self.slots.get(key)
            if slot is None:
                if key in self.evicted:
                    self._reject(context, grpc.StatusCode.FAILED_PRECONDITION, f"batch {key} is too old (evicted)",
                                 "replay", "protocol", "slot window", sender)
                slot = Slot(key, idx)
                self.slots[key] = slot
                while len(self.slots) > KEEP_SLOTS:
                    old, _ = self.slots.popitem(last=False)
                    self.evicted.add(old)
            if sender in slot.senders:
                duplicate = True
                self.dup_suppressed += 1
                self.node.counters["duplicates_suppressed"] += 1
                log.info("duplicate %s message from %s for e%d b%d (attempt %d) - no extra optimizer step",
                         phase, sender, epoch, b, h.attempt)
                SECURITY.event("retry_duplicate" if h.attempt > 1 else "duplicate_or_replay", "protocol", "detected",
                               "idempotent slot cache (no second optimizer step)", "server-observed",
                               peer_ip(context), sender, (peer_cert(context) or {}).get("name"), "Exchange", self.id,
                               f"{phase} e{epoch} b{b} attempt {h.attempt}")
            elif slot.done:
                slot.senders.add(sender)            # arrived after the slot closed (straggler policy)
            else:
                slot.senders.add(sender)
                slot.acts[sender] = arr
                slot.req_bytes[sender] = req.ByteSize()
                slot.tensor_in[sender] = len(req.activation.data)
                slot.first = slot.first or t_arr
                slot.last = t_arr
                slot.arrived[sender] = t_arr
            compute_now = (not slot.done and not slot.computing and set(self.passive_ids) <= slot.senders)
            if compute_now:
                slot.computing = True
        if compute_now:
            self._compute(slot)
        st = self.cfg.get("straggler") or {}
        drop_policy = st.get("policy", "wait") == "drop" and phase == "train"
        slot_deadline = float(st.get("slot_deadline_s") or 5.0)
        deadline = time.time() + float(self.cfg["slot_timeout_s"])
        while True:
            dropped_now = None
            with self.cv:
                if slot.done or slot.error is not None:
                    break
                if self.node.stop_event.is_set():
                    context.abort(grpc.StatusCode.CANCELLED, "session stopped")
                if drop_policy and not slot.computing and slot.first and \
                        time.perf_counter() - slot.first > slot_deadline:
                    slot.computing = True
                    dropped_now = sorted(set(self.passive_ids) - set(slot.acts))
                elif time.time() > deadline:
                    missing = sorted(set(self.passive_ids) - slot.senders)
                    context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                                  f"timed out waiting for activations from {missing} for {phase} e{epoch} b{b}")
                else:
                    self.cv.wait(0.05 if drop_policy else 0.25)
            if dropped_now is not None:
                self._compute(slot, dropped=dropped_now)
        with self.cv:
            if slot.error is not None:
                context.abort(grpc.StatusCode.INTERNAL, slot.error)
            if sender in slot.replied:
                duplicate = True
            slot.replied.add(sender)
        was_dropped = sender in slot.dropped
        reply = pb.ExchangeReply(
            header=pb.MsgHeader(session_id=self.id, epoch=epoch, batch=b, attempt=h.attempt,
                                sender=self.node.party_id, msg_type="gradient" if phase == "train" else "ack",
                                phase=phase, batch_digest=h.batch_digest, sent_unix_ns=time.time_ns()),
            duplicate=duplicate, dropped=was_dropped,
            # time this message waited at the active party for the other parties' activations
            server_wait_ms=max(0.0, (slot.result.get("t_start", t_arr) - t_arr) * 1000),
            server_compute_ms=0.0 if duplicate else slot.result.get("compute_ms", 0.0),
            server_delay_ms=0.0 if duplicate else slot.delay_ms)
        if phase == "train" and not was_dropped:
            reply.gradient.CopyFrom(slot.grads[sender])
        self.node.counters["app_bytes_out"] += reply.ByteSize()
        return reply

    def _compute(self, slot: Slot, dropped=()):
        phase, epoch, b = slot.key
        t0 = time.perf_counter()
        rec = {"phase": phase, "epoch": epoch, "batch": b, "n": int(len(slot.idx)), "t": time.time()}
        try:
            if phase == "train":
                with self.cv:
                    slot.dropped = set(dropped)
                if dropped:
                    rec["dropped"] = list(dropped)
                    for p in dropped:
                        self.node.counters[f"dropped:{p}"] += 1
                    log.warning("straggler policy: e%d b%d committed without %s", epoch, b, ", ".join(dropped))
                loss, grads, tm = self.core.train_step(slot.idx, slot.acts, epoch, b, dropped=set(dropped))
                if self.node.malicious:
                    grads = self.node.tamper_gradients(grads, epoch, b)
                enc = {p: codec.encode_cfg(g, self.cfg) for p, g in grads.items() if g is not None}
                rec.update(tm)
                rec["loss"] = loss
                rec["tensor_bytes_out"] = int(sum(len(t.data) for t in enc.values()))
                rec["optimizer_steps"] = self.core.optimizer_steps
            else:
                prob = self.core.predict(slot.idx, slot.acts)
                enc = {}
            compute_ms = (time.perf_counter() - t0) * 1000
            delay_ms = 0.0
            fault = self.cfg.get("fault") or {}
            if phase == "train" and [epoch, b] in [list(x) for x in fault.get("delay", [])] \
                    and (epoch, b) not in self.delayed:
                self.delayed.add((epoch, b))
                d = float(fault.get("delay_s", 2.0))
                log.warning("FAULT INJECTION: delaying gradient reply for e%d b%d by %.2fs", epoch, b, d)
                self.node.report("event", {"kind": "fault", "msg": f"injected {d:.2f}s reply delay at e{epoch} b{b}"})
                time.sleep(d)
                delay_ms = d * 1000
            now = time.perf_counter()
            with self.cv:
                slot.grads = enc
                slot.delay_ms = delay_ms
                slot.result = {"compute_ms": compute_ms, "wait_parties_ms": (slot.last - slot.first) * 1000,
                               "t_start": t0, "t_done": now}
                rec.update({"compute_ms": compute_ms, "delay_ms": delay_ms,
                            "wait_parties_ms": (slot.last - slot.first) * 1000,
                            "arrivals_ms": {p: (t - slot.first) * 1000 for p, t in slot.arrived.items()},
                            "idle_ms": ((slot.first - self.last_commit_t) * 1000) if self.last_commit_t else None,
                            "tensor_bytes_in": int(sum(slot.tensor_in.values())),
                            "app_bytes_in": int(sum(slot.req_bytes.values())),
                            "slot_ms": (now - slot.first) * 1000})
                self.last_commit_t = now
                if phase == "train":
                    self.train_committed[epoch] += 1
                    self.epoch_losses[epoch].append(loss)
                    self.node.counters["committed_batches"] += 1
                    self.node.progress.update({"phase": "train", "epoch": epoch, "batch": b + 1,
                                               "n_batches": len(self.batches(epoch)), "last_loss": loss})
                else:
                    split, _ = self.evals[b]
                    self.eval_preds.setdefault(epoch, {}).setdefault(split, []).append((slot.idx, prob))
                    self.eval_committed[epoch] += 1
                    self.node.progress.update({"phase": "eval", "epoch": epoch, "eval_batch": b + 1,
                                               "n_eval_batches": len(self.evals)})
                    if self.eval_committed[epoch] == len(self.evals):
                        self.epoch_events[epoch].set()
                slot.done = True
                slot.acts = {}          # free representations once committed
                self.cv.notify_all()
            self.node.counters["tensor_bytes_in"] += rec["tensor_bytes_in"]
            self.node.counters["app_bytes_in"] += rec["app_bytes_in"]
            self.node.counters["tensor_bytes_out"] += rec.get("tensor_bytes_out", 0)
            self.node.record_batch(rec)
        except Exception as e:
            log.exception("compute failed for %s", slot.key)
            with self.cv:
                slot.error = f"{type(e).__name__}: {e}"
                self.cv.notify_all()
            self.node.report("error", {"error": f"active compute failed at {slot.key}: {e}"})


class ActiveNode(PartyNode):
    role = "active"

    def __init__(self, *a, malicious: str | None = None, **kw):
        super().__init__(*a, **kw)
        if malicious and not self.lab_mode:
            raise ValueError("--malicious is a lab-only attack simulation; it requires --lab-mode")
        if malicious and not (malicious == "probe" or malicious.startswith(("scale:", "drift:"))):
            raise ValueError(f"unknown malicious mode {malicious!r}: scale:K | drift:K | probe")
        self.malicious = malicious
        self.sess: ActiveSession | None = None
        self.sess_cv = threading.Condition()

    def tamper_gradients(self, grads: dict, epoch: int, batch: int) -> dict:
        """LAB ONLY (--lab-mode --malicious ...): a deviating active party, to demonstrate the passive guard.
          scale:K   every gradient multiplied by K from epoch 1 on (sabotage; caught by grad_monitor / clipping)
          drift:K   a constant push of size K in a fixed direction on every row (steers the bottom model; bounded
                    by max_param_drift, visible to the leakage audit)
          probe     a huge gradient on the first row of every batch (targeted probing; defeated by clipping)"""
        mode = self.malicious
        if not mode:
            return grads
        self.counters["malicious_batches"] += 1
        out = {}
        for p, g in grads.items():
            if g is None:
                out[p] = g
                continue
            g = np.array(g, dtype=np.float32, copy=True)
            if mode.startswith("scale:") and epoch >= 1:
                g *= float(mode.split(":")[1])
            elif mode.startswith("drift:"):
                d = np.ones(g.shape[1], np.float32) / np.sqrt(g.shape[1])
                g += float(mode.split(":")[1]) * d
            elif mode == "probe":
                g[0] *= 1e4
            out[p] = g
        return out

    def on_stop(self):
        s = self.sess
        if s:
            with s.cv:
                s.cv.notify_all()
            for ev in list(s.epoch_events.values()):
                ev.set()
            s.aligned.set()

    def session_for(self, sid: str, context, claimed: str | None = None, method: str = "") -> ActiveSession:
        """RPCs may arrive slightly before this node has processed START, so wait a little."""
        deadline = time.time() + 10
        with self.sess_cv:
            while not (self.sess and self.sess.id == sid):
                left = deadline - time.time()
                if left <= 0:
                    reject(context, grpc.StatusCode.FAILED_PRECONDITION, f"active party has no session {sid!r}",
                           "unknown_session", "protocol", "session id check", claimed, sid, method)
                self.sess_cv.wait(min(left, 0.5))
            return self.sess

    def run_session(self, cfg: dict) -> dict:
        env = configure_torch(cfg.get("torch_threads", 1), cfg.get("deterministic", True))
        s = ActiveSession(self, self.session_id, cfg, self.participants)
        with self.sess_cv:
            self.sess = s
            self.sess_cv.notify_all()
        t_setup = time.time()
        self.progress.update({"phase": "aligning"})
        log.info("waiting for alignment (%s mode, %d passive parties)", cfg["alignment"], len(s.passive_ids))
        # DH-PSI is public-key work proportional to (parties x rows): give it more time than hashing
        align_budget = float(cfg["slot_timeout_s"]) * 3
        if cfg["alignment"] == "dh_psi":
            align_budget = max(align_budget, 3600.0)
        if not s.aligned.wait(timeout=align_budget):
            if s.mpsi is not None:
                s.mpsi.abort("alignment timed out at the active party")
            raise TimeoutError("alignment did not complete")
        if self.stop_event.is_set():
            raise StopRequested()
        res = s.align_result
        align_ms = (time.time() - (s.align_t0 or t_setup)) * 1000
        if res.n_aligned < 10:
            raise ValueError(f"only {res.n_aligned} aligned entities; check ids/overlap")
        X, splits, scaler = aligned_features(self.data, res.rows, cfg, return_scaler=True)
        y = self.data.y[np.asarray(res.rows, dtype=np.int64)]
        s.splits = splits
        s.evals = eval_batches(splits, cfg["eval_batch_size"])
        s.core = ActiveCore(cfg, X, y, self.party_id, s.passive_ids, shape=self.data.shape, scaler=scaler)
        log.info("bottom model %s (%d params incl. top), %d classes (%s)%s", s.core.kind, s.core.n_params,
                 s.core.num_classes, "BCE" if s.core.binary else "cross-entropy",
                 f", label RR flips {s.core.label_flip_rate:.1%} of labels" if s.core.label_flip_rate else "")
        start_epoch = int(cfg.get("start_epoch", 0))
        for e in range(cfg["epochs"]):
            s.epoch_events[e] = threading.Event()
        restore_s = None
        if cfg.get("resume_from"):
            t_r = time.time()
            st = load_checkpoint(self.ckpt_path(cfg["resume_from"], start_epoch - 1))
            s.core.load(st)
            restore_s = time.time() - t_r
            log.info("resumed from %s epoch %d (optimizer steps %d)", cfg["resume_from"], start_epoch - 1,
                     s.core.optimizer_steps)
        areport = {**res.report(), "align_ms": align_ms, "splits": {k: int(len(v)) for k, v in splits.items()},
                   "positive_rate_train": float(y[splits["train"]].mean())}
        self.report("alignment", areport)
        log.info("aligned %d entities (%s); train/val/test = %d/%d/%d", res.n_aligned, res.mode,
                 len(splits["train"]), len(splits["val"]), len(splits["test"]))
        s.ready.set()
        setup_s = time.time() - t_setup

        t_train = time.time()
        epoch_rows, first_target_s = [], None
        for epoch in range(start_epoch, cfg["epochs"]):
            if not self.barrier(epoch) or self.stop_event.is_set():
                break
            s.current_epoch = epoch
            te = time.time()
            n_b = len(s.batches(epoch))
            self.progress.update({"phase": "train", "epoch": epoch, "epochs": cfg["epochs"], "n_batches": n_b,
                                  "batch": 0})
            while not s.epoch_events[epoch].wait(timeout=1.0):
                if self.stop_event.is_set():
                    break
            if self.stop_event.is_set():
                break
            if s.train_committed[epoch] != n_b:
                raise RuntimeError(f"epoch {epoch}: {s.train_committed[epoch]} of {n_b} batches committed")
            row = {"epoch": epoch, "epoch_s": time.time() - te, "train_batches": n_b,
                   "train_loss": float(np.mean(s.epoch_losses[epoch])),
                   "optimizer_steps": s.core.optimizer_steps}
            for split, parts in s.eval_preds.pop(epoch, {}).items():
                idx = np.concatenate([p[0] for p in parts])
                prob = np.concatenate([p[1] for p in parts])
                row[split] = utility_metrics(s.core.y[idx], prob)
            if cfg.get("attack_eval", True):
                row["leakage"] = s.core.epoch_leakage()
            pr = defense.privacy_report(cfg, epoch + 1, len(s.passive_ids))
            if pr["label_dp"] or pr["grad_channel"]:
                row["privacy"] = {"label_dp_epsilon": (pr["label_dp"] or {}).get("epsilon"),
                                  "grad_channel_epsilon": (pr["grad_channel"] or {}).get("epsilon"),
                                  "delta": (pr["grad_channel"] or {}).get("delta")}
            va = (row.get("val") or {}).get("auroc")
            tgt = cfg.get("target_val_auroc")
            if tgt and first_target_s is None and va is not None and va >= tgt:
                first_target_s = time.time() - t_train
                row["reached_target_s"] = first_target_s
            if cfg.get("checkpoint", True):
                save_checkpoint(self.ckpt_path(self.session_id, epoch), s.core.state(epoch))
            row["resources"] = {**self.monitor.latest(), "t": time.time()}    # v2.1: CPU/RSS over time
            epoch_rows.append(row)
            self.epoch_history.append(row)
            self.slog.epochs.write(row)
            self.report("epoch", row)
            log.info("epoch %d done in %.2fs: loss=%.4f val_auroc=%s test_auroc=%s", epoch, row["epoch_s"],
                     row["train_loss"], _f((row.get("val") or {}).get("auroc")),
                     _f((row.get("test") or {}).get("auroc")))
        train_wall = time.time() - t_train
        self.flush_telemetry()
        recs = [r for r in (self.slog.recent if self.slog else []) if r.get("phase") == "train"]
        committed = sum(s.train_committed.values())
        last = epoch_rows[-1] if epoch_rows else {}
        summary = {
            "session_id": self.session_id, "setup_s": setup_s, "training_wall_s": train_wall,
            "resumed_from": cfg.get("resume_from"), "start_epoch": start_epoch, "checkpoint_restore_s": restore_s,
            "alignment": areport, "committed_batches": committed,
            "optimizer_steps": s.core.optimizer_steps,
            "optimizer_steps_this_session": s.core.optimizer_steps - (
                0 if not cfg.get("resume_from") else st["optimizer_steps"]),
            "scheduled_batches": int(sum(r["train_batches"] for r in epoch_rows)),
            "duplicates_suppressed": s.dup_suppressed, "digest_mismatches": s.digest_mismatch,
            "throughput_batches_per_s": committed / train_wall if train_wall > 0 else None,
            "server_compute_ms_p50": percentile([r["compute_ms"] for r in recs], 50),
            "server_compute_ms_p95": percentile([r["compute_ms"] for r in recs], 95),
            "wait_parties_ms_mean": float(np.mean([r["wait_parties_ms"] for r in recs])) if recs else None,
            "epochs": epoch_rows, "final_val": last.get("val"), "final_test": last.get("test"),
            "final_train_loss": last.get("train_loss"), "reached_target_s": first_target_s,
            "model_params_active": s.core.n_params, "defense": defense.describe(cfg.get("defense")),
            "privacy": defense.privacy_report(cfg, len(epoch_rows), len(s.passive_ids)), "bottom_model": s.core.kind,
            "num_classes": s.core.num_classes, "label_flip_rate": s.core.label_flip_rate,
            "codec": cfg["codec"], "counters": dict(self.counters), "resources": self.monitor.summary(),
            "env": {**self.env, **env}, "tls": self.tls.enabled,
        }
        return summary

    def ui_state(self) -> dict:
        st = self.base_ui_state()
        s = self.sess
        st["privacy"] = {
            "holds": ["raw active-party features", "labels", "bottom model f_A", "top model g", "loss"],
            "receives": [f"representations H_i [batch x {(self.session_cfg or {}).get('embed_dim', '?')}] "
                         f"from {', '.join(s.passive_ids) if s else 'passive parties'}"],
            "sends": ["dL/dH_i (gradients) back to each passive party"],
            "never_leaves": ["labels", "raw features", "top-model inputs/outputs", "loss"],
            "alignment": (s.align_result.report() if s and s.align_result else None),
            "defense": defense.describe((self.session_cfg or {}).get("defense")),
            "accounting": defense.privacy_report(self.session_cfg or {}, len(self.epoch_history),
                                                 len(s.passive_ids) if s else 1),
            "label_flip_rate": s.core.label_flip_rate if s and s.core else None,
        }
        st["gate"] = dict(getattr(self, "gate", None).stats) if getattr(self, "gate", None) else {}
        st["model"] = {"bottom_model": s.core.kind, "num_classes": s.core.num_classes,
                       "params": s.core.n_params} if s and s.core else None
        return st


class ActiveServicer(pbg.ActivePartyServicer):
    def __init__(self, node: ActiveNode):
        self.node = node

    def Align(self, request, context):
        s = self.node.session_for(request.header.session_id, context, request.header.sender, "Align")
        return s.handle_align(request, context)

    def Exchange(self, request, context):
        s = self.node.session_for(request.header.session_id, context, request.header.sender, "Exchange")
        return s.handle_exchange(request, context)

    def Ping(self, request, context):
        return pb.PingReply(node=self.node.party_id, unix_ns=time.time_ns())


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def serve(party_id: str, data: str, coordinator: str, listen: str, advertise: str, tls_dir: str | None,
          ui_host: str, ui_port: int, log_root: str, ckpt_root: str, ui_token: str | None = None,
          public_ui_url: str | None = None, manifest: str | None = None, opts: dict | None = None):
    o = opts or {}
    scheme = "https" if o.get("ui_tls") else "http"
    ui_url = public_ui_url or f"{scheme}://{ui_host if ui_host not in ('0.0.0.0', '') else 'localhost'}:{ui_port}"
    node = ActiveNode(party_id, data, coordinator, tls_dir, log_root=log_root, ckpt_root=ckpt_root,
                      ui_url=ui_url, advertise=advertise, manifest=manifest, enroll_addr=o.get("enroll_addr"),
                      lab_mode=bool(o.get("lab_mode")), malicious=o.get("malicious"))
    if node.malicious:
        log.warning("LAB MODE: this active party is MALICIOUS (%s)", node.malicious)
    gate = SecurityGate(node.tls, ACTIVE_POLICY, lambda: node.revoked, o.get("rate_limit_rps"),
                        o.get("max_peer_concurrency"))
    node.gate = gate
    server, port = make_server(node.tls, listen, gate=gate, max_concurrent_rpcs=o.get("max_concurrent_rpcs") or None)
    pbg.add_ActivePartyServicer_to_server(ActiveServicer(node), server)
    server.start()
    log.info("active party %s: gRPC on %s (advertised as %s, %s); data rows=%d features=%d",
             party_id, listen, advertise, "mTLS" if node.tls.enabled else "INSECURE",
             len(node.data.ids), node.data.X.shape[1])
    from .dashboard import create_app, resolve_ui_auth, run_app
    ui_token, private = resolve_ui_auth(ui_host, ui_token, o.get("ui_users"), party_id, log_root,
                                        bool(o.get("ui_public_views")))
    run_app(create_app(node, "active", token=ui_token, users=o.get("ui_users"), private_views=private), ui_host, ui_port,
            tls=o.get("ui_tls") and node.tls)
    node.start_background()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(1)
