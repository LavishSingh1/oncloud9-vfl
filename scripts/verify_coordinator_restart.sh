#!/usr/bin/env bash
# Issue #13 live check: kill -9 the coordinator in the middle of a session, restart it, and check that the session
# is closed (FAILED with its real epoch count), that `stop` is a clear no-op, and that resume still works.
#
#   bash scripts/verify_coordinator_restart.sh        # exit status 0 = resolved
#
# Restarts the local deployment from scripts/run_local.sh (data in ./data, certificates in ./certs) and stops it at
# the end. Takes about 2.5 minutes, most of it waiting for the parties to give up on the dead coordinator.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
RT="${RUNTIME:-.runtime}"
PARTY_WAIT="${PARTY_WAIT:-75}"   # parties retry the coordinator for about 60 s before ending the session
fail=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
parties_alive() { $PY -m vfl.cli status 2>/dev/null | grep -cE "^  (active|passive-1) +[a-z]+ +alive"; }

echo ">> fresh local deployment"
bash scripts/stop_local.sh >/dev/null 2>&1; sleep 1
bash scripts/run_local.sh >/dev/null
for _ in $(seq 1 60); do [ "$(parties_alive)" = 2 ] && break; sleep 1; done

SID=$($PY -m vfl.cli start --set epochs=40 --set inject_rtt_ms=200 \
      | $PY -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
echo ">> started $SID (40 epochs); waiting for the first epoch report"
for _ in $(seq 1 240); do [ -f "runs/$SID/summary.json" ] && break; sleep 0.5; done
sleep 1
kill -9 "$(cat "$RT/coordinator.pid")"
echo ">> coordinator killed with kill -9; waiting ${PARTY_WAIT}s for the parties to give up"
sleep "$PARTY_WAIT"

echo ">> restarting the coordinator"
$PY -m vfl.cli coordinator --listen 127.0.0.1:50050 --tls-dir certs --config configs/session.yaml \
    > "$RT/coordinator.out" 2>&1 &
echo $! > "$RT/coordinator.pid"
for _ in $(seq 1 60); do $PY -m vfl.cli status >/dev/null 2>&1 && break; sleep 1; done

LINE=$($PY -m vfl.cli status | grep "$SID")
STOP=$($PY -m vfl.cli stop "$SID" 2>&1)
DISK=$($PY -c "import json; s=json.load(open('runs/$SID/summary.json')); print(s['state'], s['completed_epochs'])")
echo "   status:       $LINE"
echo "   stop:         $(echo "$STOP" | grep -E '"note"|error' | head -1)"
echo "   summary.json: $DISK"
check "status no longer lists the session as STARTING/RUNNING" '! echo "$LINE" | grep -qE "STARTING|RUNNING"'
check "status lists it as FAILED with its completed epochs" 'echo "$LINE" | grep -qE "FAILED +epochs [1-9]"'
check "stop is a clear no-op, not 'unknown session'" \
      'echo "$STOP" | grep -q "already ended" && ! echo "$STOP" | grep -q "unknown session"'
check "summary.json says FAILED with at least one completed epoch" 'echo "$DISK" | grep -qE "^FAILED [1-9]"'
check "checkpoints are kept" 'ls "checkpoints/active/$SID"/ckpt_epoch0.pt >/dev/null 2>&1'

echo ">> resuming from $SID once both parties have re-registered"
for _ in $(seq 1 120); do [ "$(parties_alive)" = 2 ] && break; sleep 1; done
K=$(echo "$DISK" | awk '{print $2}')
NEW=$($PY -m vfl.cli start --resume-from "$SID" --set epochs=$((K + 1)) --set inject_rtt_ms=0 \
      | $PY -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
$PY -m vfl.cli wait "$NEW" --timeout 300 >/dev/null 2>&1
RES=$($PY -c "import json; s=json.load(open('runs/$NEW/summary.json')); print(s['state'], s['config'].get('start_epoch'), s['config'].get('resume_from'))")
echo "   resumed:      $NEW -> $RES"
check "resume continues from the last completed epoch and completes" 'test "$RES" = "COMPLETED $K $SID"'

bash scripts/stop_local.sh >/dev/null 2>&1
if [ "$fail" = 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit "$fail"
