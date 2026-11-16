"""Live check for the stale-session-id fix, against a running local deployment (bash scripts/run_local.sh).

    python3 scripts/verify_stale_session.py              # the red team's reproduction, with pass/fail checks
    python3 scripts/verify_stale_session.py --sessions 3 # also: real sessions must still train (START race),
                                                         # including one flooded with stale calls while it runs
Exit status 0 = every check passed.
"""
import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grpc  # noqa: E402

from vfl.proto import vfl_pb2 as pb  # noqa: E402
from vfl.proto import vfl_pb2_grpc as pbg  # noqa: E402
from vfl.transport import TLSFiles, make_channel  # noqa: E402

FAILED = []


def check(name, ok, info=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {info}")
    if not ok:
        FAILED.append(name)


def exchange(stub, sid, timeout):
    t0 = time.time()
    hdr = pb.MsgHeader(session_id=sid, epoch=0, batch=0, attempt=1, sender="passive-1")
    try:
        stub.Exchange(pb.ExchangeRequest(header=hdr), timeout=timeout)
        return "OK", "", time.time() - t0
    except grpc.RpcError as e:
        return e.code().name, e.details(), time.time() - t0


def reporter_steps(stub):
    print(">> 1. stale session id, 3 s client deadline")
    code, details, dt = exchange(stub, "stale-1", 3.0)
    check("fails fast with a diagnostic", code == "FAILED_PRECONDITION" and dt < 1.0, f"{code} in {dt:.2f}s: {details}")

    print(">> 2. 16 concurrent stale calls (12 s deadline each)")
    t0 = time.time()
    with cf.ThreadPoolExecutor(20) as pool:
        futs = [pool.submit(exchange, stub, f"stale-{i}", 12.0) for i in range(16)]
        time.sleep(0.5)
        print(">> 3. 17th call while they are (no longer) waiting")
        code17, details17, _ = exchange(stub, "real-session", 2.0)
        res = [f.result() for f in futs]
    codes = sorted({r[0] for r in res})
    check("all 16 answered quickly", time.time() - t0 < 3.0 and codes == ["FAILED_PRECONDITION"],
          f"{time.time() - t0:.2f}s, codes={codes}")
    check("17th call is not RESOURCE_EXHAUSTED", code17 != "RESOURCE_EXHAUSTED", f"{code17}: {details17}")


def real_sessions(stub, n, url, epochs):
    from vfl.cli import _http, _wait
    token = os.environ.get("VFL_UI_TOKEN")
    for i in range(n):
        flood = i == n - 1                     # last session runs under a stale-call flood
        print(f">> session {i + 1}/{n}{' (with stale-call flood)' if flood else ''}")
        r = _http("POST", f"{url.rstrip('/')}/api/sessions", {"config": {"epochs": epochs}, "resume_from": None}, token)
        stop, sent = threading.Event(), [0]

        def hammer():
            while not stop.is_set():
                exchange(stub, f"flood-{sent[0]}", 5.0)
                sent[0] += 1
        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(8)] if flood else []
        for t in threads:
            t.start()
        d = _wait(url, r["session_id"], 1800)
        stop.set()
        for t in threads:
            t.join()
        check(f"session {r['session_id']} completed", d["state"] == "COMPLETED",
              f"state={d['state']}" + (f", {sent[0]} stale calls during training" if flood else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", default="certs")
    ap.add_argument("--active", default="127.0.0.1:50051")
    ap.add_argument("--url", default="http://localhost:8080", help="coordinator dashboard")
    ap.add_argument("--sessions", type=int, default=0, help="real sessions to run after the reproduction")
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    stub = pbg.ActivePartyStub(make_channel(TLSFiles(a.certs, "passive-1"), a.active, "active", "active"))
    reporter_steps(stub)
    if a.sessions:
        real_sessions(stub, a.sessions, a.url, a.epochs)
    print("\nALL CHECKS PASSED" if not FAILED else f"\n{len(FAILED)} CHECK(S) FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
