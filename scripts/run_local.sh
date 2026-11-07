#!/usr/bin/env bash
# Launch a complete VFL deployment on ONE machine as separate processes:
#   coordinator (gRPC :50050, UI :8080), active (gRPC :50051, UI :8081),
#   passive-1..N (UI :8082..). Uses mTLS certificates in ./certs (created if missing).
#
#   bash scripts/run_local.sh                  # 1 passive, breast_cancer data in ./data
#   PASSIVES=2 DATA=data2 bash scripts/run_local.sh
#   INSECURE=1 bash scripts/run_local.sh       # no TLS (debugging only)
#   bash scripts/stop_local.sh                 # stop everything
set -euo pipefail
cd "$(dirname "$0")/.."

PASSIVES="${PASSIVES:-1}"
DATA="${DATA:-data}"
CERTS="${CERTS:-certs}"
RUNTIME="${RUNTIME:-.runtime}"
PY="${PYTHON:-python3}"
mkdir -p "$RUNTIME"

if [[ ! -f "$DATA/active.csv" ]]; then
  echo ">> no data in $DATA - generating breast_cancer split with $PASSIVES passive part(ies)"
  $PY -m vfl.cli prepare-data --dataset breast_cancer --passives "$PASSIVES" --out "$DATA"
fi

TLS_ARGS=(--tls-dir "$CERTS")
if [[ "${INSECURE:-0}" == "1" ]]; then
  TLS_ARGS=(--insecure)
elif [[ ! -f "$CERTS/ca.crt" || ! -f "$CERTS/passive-$PASSIVES.crt" ]]; then
  echo ">> generating mTLS certificates in $CERTS"
  $PY -m vfl.cli certs --out "$CERTS" --passives "$PASSIVES"
fi

start() {  # name, command...
  local name="$1"; shift
  "$@" > "$RUNTIME/$name.out" 2>&1 &
  echo $! > "$RUNTIME/$name.pid"
  echo "   started $name (pid $(cat "$RUNTIME/$name.pid")) -> $RUNTIME/$name.out"
}

echo ">> starting coordinator"
# one machine: every gRPC port on loopback only (v2.3; --insecure refuses anything else)
start coordinator $PY -m vfl.cli coordinator --listen 127.0.0.1:50050 "${TLS_ARGS[@]}" --config "${SESSION_CONFIG:-configs/session.yaml}"
sleep 2
echo ">> starting active party"
start active $PY -m vfl.cli active --data "$DATA/active.csv" --coordinator localhost:50050 \
  --listen 127.0.0.1:50051 --advertise localhost:50051 "${TLS_ARGS[@]}"
for i in $(seq 1 "$PASSIVES"); do
  echo ">> starting passive-$i"
  start "passive-$i" $PY -m vfl.cli passive --id "passive-$i" --data "$DATA/passive-$i.csv" \
    --coordinator localhost:50050 --ui-port $((8081 + i)) "${TLS_ARGS[@]}"
done
sleep 3
echo
echo "Dashboards:"
echo "   coordinator  http://localhost:8080   (start/stop sessions here)"
echo "   active       http://localhost:8081"
for i in $(seq 1 "$PASSIVES"); do echo "   passive-$i    http://localhost:$((8081 + i))"; done
echo
echo "Start a session from the coordinator dashboard, or:"
echo "   $PY -m vfl.cli start --set epochs=10 --wait"
