PY ?= python3
PASSIVES ?= 1

.PHONY: install proto data data-synthetic certs local stop status session test test-fast bench bench-quick security analyze clean config-docs ui attack bench-ui

install:            ## install dependencies (CPU PyTorch)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

proto:              ## regenerate gRPC stubs from proto/vfl.proto
	bash scripts/gen_proto.sh

data:               ## breast-cancer vertical split -> data/
	$(PY) -m vfl.cli prepare-data --dataset breast_cancer --passives $(PASSIVES) --out data

data-synthetic:     ## 20k x 40 synthetic task -> data/
	$(PY) -m vfl.cli prepare-data --dataset synthetic --n 20000 --features 40 --passives $(PASSIVES) --out data

certs:              ## private CA + node certificates -> certs/
	$(PY) -m vfl.cli certs --out certs --passives $(PASSIVES)

local:              ## run coordinator + active + passive(s) on this machine
	PASSIVES=$(PASSIVES) bash scripts/run_local.sh

stop:
	bash scripts/stop_local.sh

status:
	$(PY) -m vfl.cli status

session:            ## start a 10-epoch session and wait
	$(PY) -m vfl.cli start --set epochs=10 --wait

test:               ## unit + end-to-end tests
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

config-docs:        ## regenerate docs/CONFIG.md and the README table from vfl/config_schema.py
	$(PY) -m vfl.cli config docs --out docs/CONFIG.md --readme README.md

bench:              ## full PRD evaluation plan incl. sweep + security conformance suite
	$(PY) bench/run_benchmarks.py --out results/bench --suites e1,e2,eff,def,psi,e5,sweep,security
	$(PY) bench/security_test.py --out results/security --plaintext-control
	$(PY) bench/analyze.py results/bench --security results/security/security_report.json --cost configs/cloud_cost.example.yaml

ui:                 ## unified UI on 127.0.0.1:8090 (set VFL_UI_TOKEN to enable actions safely)
	$(PY) -m vfl.cli ui

bench-ui:           ## same unified UI (kept for compatibility)
	$(PY) -m vfl.cli bench-ui --results results

attack:             ## lab-only conformance suite against the local deployment (no dashboard needed)
	$(PY) -m vfl.cli attack all --lab-mode --coordinator 127.0.0.1:50050 --active 127.0.0.1:50051 \
		--certs certs --data-dir data --log-dirs logs --out results/attacks

bench-quick:        ## smaller sweep for a smoke check (~3 min)
	$(PY) bench/run_benchmarks.py --out results/bench-quick --seeds 1,2 --epochs 5 --rtts 0,50 --batch-sizes 32,128 --noise 1
	$(PY) bench/analyze.py results/bench-quick

security:
	$(PY) bench/security_test.py --out results/security --plaintext-control

analyze:
	$(PY) bench/analyze.py results/bench --security results/security/security_report.json

clean:
	rm -rf .runtime runs logs checkpoints results
