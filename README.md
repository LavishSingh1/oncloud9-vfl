# Vertical Federated Learning over Heterogeneous Networked Systems

OnCloud9 · DS252 PRD v3 implementation — a reproducible, networked VFL system with a central
coordinator, an active (label-owning) party, one or more passive (feature-owning) parties,
mutual-TLS gRPC transport, per-node web dashboards, and a benchmark/test harness that measures
every functional and non-functional requirement in the PRD.

```
                    ┌──────────────────────────────┐
                    │  Coordinator  :50050 / UI :8080│  membership, sessions, epoch barriers,
                    │  (never sees data or tensors)  │  telemetry, run summaries
                    └───────▲──────────▲──────────▲──┘
              control/telemetry        │          control/telemetry
                    ┌───────┴──┐  ┌────┴─────────┐  ┌──┴────────┐
                    │ Passive 1│  │ Active  :50051│  │ Passive N │
                    │ X1, f1   │──│ X_A, f_A, Y   │──│ XN, fN    │
                    │ UI :8082 │H→│ top g + loss  │←H│ UI :808x  │
                    └──────────┘←∂│ UI :8081      │∂→└───────────┘
                                  └───────────────┘
   H = representation f_i(X_i) [batch × embed]       ∂ = dL/dH_i returned by the active party
```

## Documentation

| Read this | For |
|---|---|
| this README | install and use: quick start, every command, configuration, UI |
| [PRD.md](PRD.md) | the original project plan (PRD v3) |
| [dev.md](dev.md) | developer docs: what each part of the codebase does and how to extend it |
| [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) ([PDF](docs/PROJECT_DOCUMENTATION.pdf)) | architecture diagrams, PRD requirement traceability, privacy design |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | separate machines, VPN, cloud, PKI enrollment |
| [docs/CONFIG.md](docs/CONFIG.md) | every session key (generated) |
| [CONTRIBUTORS.md](CONTRIBUTORS.md) | team and ownership of each part of the code |

Team OnCloud9 (DS252): Amisha Sadhukan, Atrayee Samanta, Lavish Singh, M. Divya.

## Contents

```
oncloud9-vfl/
├── README.md · PRD.md · dev.md · CONTRIBUTORS.md · CHANGELOG-v2.md
├── proto/vfl.proto            wire protocol (Coordinator + ActiveParty services)
├── vfl/
│   ├── cli.py                 single entry point: python -m vfl.cli <command>
│   ├── config_schema.py       every session key: type, default, presets, help (form, --set, docs)
│   ├── coordinator.py         session manager: registration, role validation, START/STOP,
│   │                          epoch barriers, watchdog, telemetry aggregation, resume
│   ├── active.py              active party: alignment server, per-batch slot barrier,
│   │                          top model/loss, duplicate suppression, fault injection
│   ├── passive.py             passive party: alignment client, forward/backward, retries,
│   │                          RTT emulation, per-batch telemetry
│   ├── party_base.py          shared party machinery (heartbeats, barriers, telemetry shipping)
│   ├── core.py                transport-independent training core (used by network + local sim)
│   ├── models.py              split PyTorch models (bottom / top / centralized reference)
│   ├── alignment.py           replaceable alignment: non-private experimental + DH-PSI
│   ├── defense.py             gradient-perturbation defence + label-leakage attack metrics
│   ├── data.py                datasets, vertical partitioning, splits, deterministic batches
│   ├── codec.py               tensor <-> protobuf (fp32 / fp16) and byte accounting
│   ├── security.py            private CA + per-node certificates (mTLS, role in the SAN)
│   ├── websec.py              HTTP hardening for every web UI (Host/Origin/CSRF checks, CSP)
│   ├── ids.py, limits.py      validation of ids/strings and protocol limits on authorised peers (v2.3)
│   ├── passive_guard.py       passive-side defences against a malicious active party (v2.3.1)
│   ├── transport.py           gRPC server/channel construction, certificate identity checks
│   ├── telemetry.py           JSONL per-batch/epoch/run logs, CPU/RSS sampling
│   ├── local_sim.py           B0 centralized, B1 active-only, single-process VFL references
│   ├── netem_proxy.py         TCP relay: adds RTT, counts wire bytes, captures for inspection
│   ├── dashboard/             Flask API + one role-aware web UI per node (no external JS/CSS)
│   ├── webui/                 unified UI `vfl.cli ui` :8090 (data prep, deployment, training, attacks,
│   │                          tests & benchmarks, analytics, reports) + background jobs
│   ├── attacks/               LAB-ONLY conformance suite: standalone CLI, report writer, UI launchers
│   ├── metrics_curves.py      down-sampled ROC points + confusion counts (aggregates only)
│   └── proto/                 generated protobuf/gRPC stubs
├── bench/
│   ├── run_benchmarks.py      PRD evaluation plan E1/E2/E5 + efficiency, defence, PSI suites
│   ├── security_test.py       E3: 100 invalid-cert attempts, impersonation, packet scan, log audit
│   ├── analyze.py             figures, NFR pass/fail table, requirement matrix, report.md
│   └── harness.py             launch local deployments / drive remote ones
├── tests/                     pytest: alignment, data, core, coordinator, end-to-end, schema, UI
├── docs/                      PROJECT_DOCUMENTATION.md/.pdf, DEPLOYMENT.md, CONFIG.md (generated), diagrams/, images/
├── configs/                   session.yaml (default), synthetic.yaml, cloud_cost.example.yaml
├── scripts/                   run_local.sh, stop_local.sh, gen_proto.sh
├── docker/                    Dockerfile + docker-compose.yml
└── Makefile
```

## 1. Install

Python 3.10–3.12 recommended (3.13 works if wheels exist for your platform).

```bash
cd oncloud9-vfl                  # the repository root (the vfl-system folder)
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU PyTorch is enough
pip install -r requirements.txt
bash scripts/gen_proto.sh        # optional: regenerate stubs with your grpcio-tools version
pytest -q                        # 170 tests incl. end-to-end runs over mTLS
```

The same code runs on x86-64 and ARM (e.g. a Raspberry Pi 4/5 or Apple Silicon as a passive
party) — that is how you get a heterogeneous deployment.

## 2. Quick start on one machine (separate processes)

```bash
python -m vfl.cli prepare-data --dataset breast_cancer --passives 1 --out data
python -m vfl.cli certs --out certs --passives 1
bash scripts/run_local.sh              # coordinator + active + passive-1 in the background
```

Prefer clicking? `python -m vfl.cli ui` does all of this from one page (see [4d](#4d-the-unified-ui-v21)).

Open the dashboards:

| node | dashboard | what you do there |
|---|---|---|
| coordinator | http://localhost:8080 | see parties, configure and **start/stop sessions**, live loss/AUROC/latency/leakage, history |
| active | http://localhost:8081 | label-owner view: privacy boundary, loss, AUROC per epoch, slot timing |
| passive-1 | http://localhost:8082 | feature-owner view: what leaves the node, batch-time breakdown, traffic |

Start a session from the coordinator dashboard form, or from the terminal:

```bash
python -m vfl.cli start --set epochs=10 --wait
python -m vfl.cli status
bash scripts/stop_local.sh             # stop all nodes
```

### Running each node yourself (one terminal per node)

```bash
# Terminal 1 — session management node (coordinator)
python -m vfl.cli coordinator --listen 0.0.0.0:50050 --tls-dir certs --config configs/session.yaml

# Terminal 2 — active party (labels + top model)
python -m vfl.cli active --id active --data data/active.csv \
    --coordinator localhost:50050 --listen 0.0.0.0:50051 --advertise localhost:50051 --tls-dir certs

# Terminal 3 — passive party
python -m vfl.cli passive --id passive-1 --data data/passive-1.csv \
    --coordinator localhost:50050 --tls-dir certs

# Terminal 4 — start a session (or use the dashboard)
python -m vfl.cli start --set epochs=10 --set batch_size=64 --wait
```

Two passive parties: `prepare-data --passives 2`, `certs --passives 2`, then run a second passive
with `--id passive-2 --data data/passive-2.csv --ui-port 8083`. The coordinator includes every
live party in the next session (or the list in `participants:`).

`--insecure` disables TLS for local debugging only; dashboards bind to 127.0.0.1 unless you pass
`--ui-host 0.0.0.0` (add `--ui-token SECRET` to protect the start/stop buttons).

## 3. Multi-machine LAN deployment

> Full guide for separate machines on a LAN, over a VPN, on cloud VMs or through port forwarding — including how the scenarios differ and which settings to change — is in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

The unified UI's **Deployment → Multi-machine bundles** builds one zip per machine (that node's certificate
and key, the CA certificate, its data file and manifest, and the exact run command; never `ca.key`) and then
shows each node as it registers.

Example: coordinator on 192.168.1.10, active on 192.168.1.20, passive on 192.168.1.30.

```bash
# on any one machine: create CA + certs, then copy ca.crt + <node>.crt/.key to each node
python -m vfl.cli certs --out certs --passives 1
#   coordinator  -> certs/{ca.crt, coordinator.crt, coordinator.key}
#   active       -> certs/{ca.crt, active.crt, active.key}
#   passive-1    -> certs/{ca.crt, passive-1.crt, passive-1.key}     (never copy ca.key)

# each party keeps ONLY its own CSV (data/active.csv on the active machine, etc.)

# 192.168.1.10
python -m vfl.cli coordinator --listen 0.0.0.0:50050 --tls-dir certs --ui-host 0.0.0.0
# 192.168.1.20
python -m vfl.cli active --data data/active.csv --coordinator 192.168.1.10:50050 \
    --listen 0.0.0.0:50051 --advertise 192.168.1.20:50051 --tls-dir certs --ui-host 0.0.0.0
# 192.168.1.30
python -m vfl.cli passive --id passive-1 --data data/passive-1.csv \
    --coordinator 192.168.1.10:50050 --tls-dir certs --ui-host 0.0.0.0
```

Then open `http://192.168.1.10:8080`. Certificates are bound to node names, not IPs (clients
pin the expected peer name), so machines can change addresses without re-issuing certificates.
Open TCP 50050 (coordinator) and 50051 (active) in the firewalls; 808x only if you want remote
dashboards. Your own data: `prepare-data --dataset csv --csv mydata.csv --label target` or hand
each party a CSV with an `id` column (and `label` only at the active party).

Docker (all roles in containers on one host): `docker compose -f docker/docker-compose.yml up --build`.

## 3a. Data partitioning (v2 partitioner)

`prepare-data` has two generators. **v1 is the default** and reproduces earlier results byte-for-byte;
`--partitioner v2` uses `vfl/partition.py` (vendored, unmodified from `vfl_partition.py`).

```bash
# 3 parties, uneven feature split, 80 % entity overlap, opaque hashed ids
python -m vfl.cli prepare-data --partitioner v2 --dataset breast_cancer --parties 3 \
    --partition dirichlet --alpha 0.3 --overlap 0.8 --out data
# image strips (8x8 digits) for 3 parties, labels made binary (odd vs even)
python -m vfl.cli prepare-data --partitioner v2 --dataset digits --parties 3 \
    --partition image_strips --binarize odd_even --out data
```

| v2 strategies | `sequential`, `random`, `dirichlet` (`--alpha`), `importance` (`--top-frac`, `--informative-party`), `correlated`, `manual` (`--columns-json`), `image_strips`, `image_grid` (`--grid 2x2`) |
|---|---|
| v2 datasets | `breast_cancer`, `digits`, `synthetic`, `csv` offline; `adult`, `bank`, `credit_default`, `mnist`, `fashion_mnist` download from openml.org |
| v1 names in v2 | `importance_skewed` → `importance` at passive-1; `importance_balanced` is v1-only |

Output per party: `<name>.csv` + `<name>.manifest.json`. **Give each machine only those two files.**
The party manifest holds only that party's role, row count, `model_input_dim`, its own column order,
`shape` (image tiles) and split metadata; label-derived facts (number of classes, binarize rule) appear
only in the active party's file. `manifest.json` / `data_report.json` (all parties' columns, label
balance, importance, reference AUROC) stay with whoever runs the experiment. Parties load their
manifest automatically (or `--manifest PATH`) and refuse to start if role, rows, input width, column
order or shape disagree with the CSV.

Session keys `partition`, `overlap`, `binarize` (default `null` = no check): when set, the coordinator
refuses to start unless every party's manifest matches; it always refuses when parties hold data from
different `prepare-data` runs. The benchmark harness takes the same options
(`run_benchmarks.py --partitioner v2 --partition dirichlet --overlap 0.8`).

Datasets with more than two classes (`digits`, `mnist`, …) can be generated now; training them needs
v2 item 4 (multi-class). Until then the coordinator and the active party refuse them with a clear
message — use `--binarize odd_even | one_vs_rest:K | pair:A,B`.

## 4. Session configuration (FR-9)

Defaults live in `configs/session.yaml`. Override them from the session form (coordinator dashboard or the
unified UI's Training page) or with `--set key=value`. Every key the training code reads is defined once in
`vfl/config_schema.py`: the form, `--set` validation, this table and `docs/CONFIG.md` are generated from it, and
`tests/test_config_schema.py` fails if the code reads a key the schema does not have.

```bash
python -m vfl.cli config validate --set lr=0.003 --set 'bottom_model={active: cnn, default: mlp}'
python -m vfl.cli config validate --set cnn_kernal=5     # refused: did you mean 'cnn_kernel'?
python -m vfl.cli config docs --out docs/CONFIG.md --readme README.md   # regenerate the tables
```

The "Dropdown presets" column is what the form offers; `--set` accepts any value that passes the type/range
check. Keys marked *new in v2.1* default to the previous behaviour.

<!-- config-table:start (generated by `python -m vfl.cli config docs --readme README.md`) -->
| Key | Type | Default | Dropdown presets | Applies to | Description |
|---|---|---|---|---|---|
| **Training** | | | | | |
| `seed` | int (>= 0) | `42` | `0`, `1`, `2`, `3`, `4`, `5`, `7`, `42`, `123`, `2026` | all | Seed for model init, batch order, splits and defence noise. |
| `epochs` | int (>= 1) | `10` | `1`, `2`, `3`, `5`, `10`, `15`, `20`, `30`, `50`, `100` | all | Training epochs. |
| `batch_size` | int (>= 1) | `64` | `16`, `32`, `64`, `128`, `256`, `512`, `1024` | all | Training mini-batch size (entities per exchanged batch). |
| `lr` | float (> 0) | `0.001` | `0.0001`, `0.0003`, `0.001`, `0.003`, `0.01`, `0.03`, `0.1` | all | Learning rate (every party's optimizer uses the same value). |
| `optimizer` | enum | `adam` | `adam`, `sgd` | all | Optimizer at every party. |
| `momentum` | float (>= 0, <= 1) | `0.0` | `0.0`, `0.5`, `0.9`, `0.95`, `0.99` | all; when optimizer = sgd | SGD momentum (ignored by adam). |
| `target_val_auroc` | float or null (>= 0, <= 1) | `null` | `0.8`, `0.85`, `0.9`, `0.95`, `0.97`, `0.98`, `0.99` | all | Validation AUROC that counts as 'converged' for the time-to-target metric (training does not stop). |
| `split.val` | float (>= 0, <= 1) | `0.15` | `0.0`, `0.1`, `0.15`, `0.2`, `0.25` | all | Validation fraction of the aligned entities. |
| `split.test` | float (> 0, <= 1) | `0.15` | `0.1`, `0.15`, `0.2`, `0.25`, `0.3` | all | Test fraction of the aligned entities. |
| `eval_batch_size` | int (>= 1) | `512` | `128`, `256`, `512`, `1024`, `2048` | all | Batch size for validation/test passes. |
| **Model sizes** | | | | | |
| `hidden_dim` | int (>= 0) | `64` | `0`, `16`, `32`, `64`, `128`, `256` | mlp | MLP bottom hidden width (0 = a single linear layer). |
| `embed_dim` | int (>= 1) | `16` | `2`, `4`, `8`, `16`, `32`, `64`, `128` | all | Representation width each party sends: the wire contract is [batch x embed_dim]. |
| `top_hidden` | int (>= 0) | `32` | `0`, `16`, `32`, `64`, `128`, `256` | all | Top-model hidden width at the active party (0 = linear top). |
| `num_classes` | int or null (>= 2) | `null` | `2`, `3`, `4`, `5`, `6`, `8`, `10`, `20`, `100` | all | Number of classes. Empty = inferred from the labels (2 -> one logit + BCE, k > 2 -> cross-entropy). |
| **Bottom models** | | | | | |
| `bottom_model` | kind or {party: kind} | `mlp` | `mlp`, `linear`, `cnn`, `cnn1d`, `gru`, `text` | all | Bottom model per party. One kind for everyone, or a mapping {party_id: kind, default: kind}. |
| `cnn_channels` | list of int | `[16, 32]` | `[8, 16]`, `[16, 32]`, `[32, 64]`, `[16, 32, 64]`, `[32, 64, 128]`, `[16]` | cnn, cnn1d | Conv widths, one conv layer per entry (cnn and cnn1d). The default [16, 32] is the v2 network. |
| `cnn_kernel` | int (>= 1, odd) | `3` | `3`, `5`, `7` | cnn | 2-D conv kernel size (odd; 'same' padding keeps the tile size). *(new in v2.1; default = previous behaviour)* |
| `cnn_pool` | enum | `auto` | `auto`, `max`, `avg`, `none` | cnn | Pooling after the 2-D convs. auto = 2x2 max-pool when the tile is at least 4x4 (v2 behaviour). *(new in v2.1; default = previous behaviour)* |
| `cnn1d_kernel` | int (>= 1, odd) | `5` | `3`, `5`, `7`, `9` | cnn1d | 1-D conv kernel size over time (odd). *(new in v2.1; default = previous behaviour)* |
| `cnn1d_pool` | enum | `mean` | `mean`, `max` | cnn1d | How the 1-D conv features are pooled over time (mean is v2). *(new in v2.1; default = previous behaviour)* |
| `gru_hidden` | int (>= 1) | `32` | `16`, `32`, `64`, `128` | gru | GRU hidden size. |
| `gru_layers` | int (>= 1, <= 4) | `1` | `1`, `2`, `3` | gru | Stacked GRU layers (the last layer's final state feeds the head). *(new in v2.1; default = previous behaviour)* |
| `text_hidden` | int (>= 0) | `0` | `0`, `32`, `64`, `128` | text | Text bottom: hidden width of the trainable projection over the frozen encoder embeddings (0 = linear projection to embed_dim, v2). The encoder itself is chosen at data preparation. *(new in v2.1; default = previous behaviour)* |
| **Alignment & codec** | | | | | |
| `alignment` | enum | `experimental` | `experimental`, `dh_psi` | all | Entity alignment. experimental is NON-PRIVATE (salted hashes reveal the intersection); dh_psi is Diffie-Hellman PSI for any number of parties (two-party protocol with one passive, multi-party through the active party with more). |
| `alignment_salt` | str or null | `null` | derived / random | all | Salt for experimental alignment hashes. Empty = derived from the seed (reproducible order). |
| `psi_group_bits` | int | `2048` | `2048` | all; when alignment = dh_psi | Recorded for the run log only: DH-PSI (two- and multi-party) always uses the RFC 3526 2048-bit group (not read by the code). *(not read by the code)* |
| `codec` | enum | `fp32` | `fp32`, `fp16`, `int8`, `topk` | all | Wire encoding of representations and gradients. |
| `topk_fraction` | float (> 0, <= 1) | `0.25` | `0.05`, `0.1`, `0.25`, `0.5`, `0.75` | all; when codec = topk | top-k codec: fraction of entries kept per row. |
| **Network & timing** | | | | | |
| `inject_rtt_ms` | float (>= 0) | `0.0` | `0.0`, `5.0`, `10.0`, `20.0`, `50.0`, `100.0`, `200.0`, `500.0` | all | Application-level RTT emulated at passive parties. |
| `rpc_deadline_s` | float (> 0) | `30.0` | `5.0`, `10.0`, `30.0`, `60.0`, `120.0`, `300.0` | all | Per-RPC deadline before a retry. |
| `max_retries` | int (>= 0) | `3` | `0`, `1`, `2`, `3`, `5`, `10` | all | Retries per RPC on UNAVAILABLE / DEADLINE_EXCEEDED. |
| `retry_backoff_s` | float (>= 0) | `0.2` | `0.05`, `0.1`, `0.2`, `0.5`, `1.0` | all | Linear back-off step between retries. |
| `slot_timeout_s` | float (> 0) | `120.0` | `30.0`, `60.0`, `120.0`, `300.0`, `600.0` | all | How long the active party waits for every passive activation. |
| `barrier_timeout_s` | float (> 0) | `300.0` | `60.0`, `120.0`, `300.0`, `600.0`, `1800.0` | all | Epoch-barrier timeout at the coordinator. |
| **Defences** | | | | | |
| `defense.type` | enum | `none` | `none`, `grad_noise` | all | Gradient perturbation on dL/dH (clip + Gaussian noise). Not a formal end-to-end DP guarantee. |
| `defense.clip` | float (> 0) | `0.05` | `0.005`, `0.01`, `0.02`, `0.05`, `0.1`, `0.5`, `1.0` | all; when defense.type = grad_noise | Per-row L2 clip C on dL/dH (rows scale like 1/batch_size). |
| `defense.noise_multiplier` | float (>= 0) | `1.0` | `0.1`, `0.25`, `0.5`, `1.0`, `2.0`, `4.0`, `8.0` | all; when defense.type = grad_noise | Gaussian noise std as a multiple of C. |
| `defense.delta` | float (> 0, <= 1) | `1e-05` | `1e-06`, `1e-05`, `0.0001` | all; when defense.type = grad_noise | delta for the (eps, delta) report of the gradient channel. |
| `defense.label_rr_epsilon` | float or null (> 0) | `null` | `0.5`, `1.0`, `2.0`, `3.0`, `4.0`, `8.0` | all | Randomized response on training labels: the ONLY formal guarantee here (the trained model and the gradients sent to passives are eps-label-DP; validation/test metrics use true labels). |
| `defense.rep_noise_sigma` | float (>= 0) | `0.0` | `0.0`, `0.01`, `0.05`, `0.1`, `0.5`, `1.0` | all | Passive-side representation noise (clip to rep_clip, add N(0, (sigma*rep_clip)^2)); no formal guarantee. |
| `defense.rep_clip` | float (> 0) | `1.0` | `0.5`, `1.0`, `2.0`, `5.0` | all; when defense.rep_noise_sigma ≠ 0.0 | Row clip applied before representation noise. |
| `defense.noise_source` | enum | `secret` | `secret`, `seed` | all | Where defence randomness (gradient noise, randomized-response flips, representation noise) comes from. secret: a secret each party draws locally and never sends. seed: v2 behaviour, derived from the shared session seed - reproducible, but every party can recompute and REMOVE the noise, so it is not a defence. *(new in v2.1; default = previous behaviour)* |
| `attack_eval` | bool | `true` | toggle | all | Evaluator-side label-leakage metrics (norm / cosine attack AUC) on the gradients actually sent. |
| **Straggler** | | | | | |
| `straggler.policy` | enum | `wait` | `wait`, `drop` | all | wait = every batch waits for all parties (v1); drop = proceed without parties that miss the deadline. |
| `straggler.slot_deadline_s` | float (> 0) | `5.0` | `0.5`, `1.0`, `2.0`, `5.0`, `10.0`, `30.0` | all; when straggler.policy = drop | drop policy: per-batch wait at the active party. |
| **Fault injection** | | | | | |
| `fault.duplicate` | list of [epoch, batch] | `[]` | epoch/batch rows | all | Passive parties re-send the message for these (epoch, batch) slots; the active party must not step twice. |
| `fault.delay` | list of [epoch, batch] | `[]` | epoch/batch rows | all | The active party delays its reply for these (epoch, batch) slots. |
| `fault.delay_s` | float (>= 0) | `2.0` | `0.5`, `1.0`, `2.0`, `5.0`, `10.0` | all | Length of each injected reply delay. |
| **Resume & data checks** | | | | | |
| `resume_from` | str or null | `null` | past sessions | all | Resume a past session from its last complete epoch. |
| `partition` | str or null | `null` | `random`, `sequential`, `importance_balanced`, `importance_skewed`, `dirichlet`, `importance`, `correlated`, `manual`, `image_strips`, `image_grid`, `sensor_streams`, `multimodal` | all | Refuse to start unless every party's manifest reports this split strategy (empty = don't check). |
| `overlap` | float or null (> 0, <= 1) | `null` | `0.5`, `0.6`, `0.7`, `0.8`, `0.9`, `0.95`, `1.0` | all | Refuse to start unless the manifests report this entity overlap. |
| `binarize` | str or null | `null` | `odd_even`, `one_vs_rest:0`, `one_vs_rest:1`, `one_vs_rest:2`, `one_vs_rest:3`, `one_vs_rest:4`, `one_vs_rest:5`, `one_vs_rest:6`, `one_vs_rest:7`, `one_vs_rest:8`, `one_vs_rest:9`, `pair:3,8`, `pair:1,7`, `pair:4,9` | all | Refuse to start unless the active party's labels used this binarize rule. |
| `participants` | list of party ids or null | `null` | detected parties | all | Parties to include. Empty = every live party. |
| `min_passive` | int (>= 1) | `1` | `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8` | all | Minimum live passive parties needed to start. |
| `checkpoint` | bool | `true` | toggle | all | Save a checkpoint at every party after each epoch (needed to resume). |
| `start_epoch` | int (>= 0) | `0` |  | all | First epoch of this run (set by the coordinator when resuming). *(system-set; not in the form)* |
| **Tensor validation** | | | | | |
| `security.validate_tensors` | bool | `true` | toggle | all | Shape + finiteness checks on every H and dL/dH. |
| `security.max_row_norm` | float or null (> 0) | `null` | `1.0`, `5.0`, `10.0`, `50.0`, `100.0` | all | Reject representation rows with a larger L2 norm (active party). |
| `security.max_grad_row_norm` | float or null (> 0) | `null` | `0.01`, `0.1`, `1.0`, `10.0` | all | Reject gradient rows with a larger L2 norm (passive parties). |
| `security.private_init` | bool | `false` | toggle | all | Passive parties initialise their bottom models from a local secret instead of the shared seed, so the active party cannot compute their starting weights (which helps it invert representations). Off keeps runs reproducible from the seed. *(new in v2.1; default = previous behaviour)* |
| `security.anomaly_detection` | bool | `false` | toggle | all | Flag near-constant / outlier representations (free-riders, poisoning). |
| `security.max_align_items` | int or null (>= 1) | `null` | `10000`, `100000`, `1000000`, `5000000` | all | Most ids a party may submit for alignment. Empty = 4 x the active party's rows + 10000 (PSI makes every party work per submitted id, and each one is a membership query). *(new in v2.3; default = previous behaviour)* |
| `security.passive_grad_clip` | float or null (> 0) | `null` | `0.01`, `0.1`, `1.0`, `10.0` | all | Passive parties clip every received gradient row to this L2 norm (bounds how much a malicious active party can move them per step; defeats scaled / targeted gradients). *(new in v2.3; default = previous behaviour)* |
| `security.max_param_drift` | float or null (> 0) | `null` | `0.5`, `1.0`, `2.0`, `5.0`, `10.0` | all | Passive bottom models stay within this relative L2 distance of their initialisation (a trust region: no gradient sequence can steer them further). *(new in v2.3; default = previous behaviour)* |
| `security.grad_monitor` | enum | `off` | `off`, `warn`, `stop` | all | Passive parties flag gradients whose norm jumps far above the running trend; stop = leave the session after 3 consecutive flags. *(new in v2.3; default = previous behaviour)* |
| `security.leakage_audit` | enum | `off` | `off`, `warn`, `stop` | all | After each epoch every passive measures how well its own raw features can be reconstructed (ridge decoder) from the representations it sent; warn or stop above leakage_r2_max. *(new in v2.3; default = previous behaviour)* |
| `security.leakage_r2_max` | float (>= 0, <= 1) | `0.9` | `0.3`, `0.5`, `0.7`, `0.8`, `0.9`, `0.95` | all; when security.leakage_audit = warn/stop | Largest acceptable feature-reconstruction R^2 for the leakage audit (1 = features fully recoverable). *(new in v2.3; default = previous behaviour)* |
| **Runtime & telemetry** | | | | | |
| `torch_threads` | int (>= 1) | `1` | `1`, `2`, `4`, `8` | all | PyTorch CPU threads per party process. |
| `deterministic` | bool | `true` | toggle | all | Deterministic kernels so CPU runs repeat exactly. |
| `telemetry_flush_s` | float (> 0) | `2.0` | `0.5`, `1.0`, `2.0`, `5.0` | all | How often parties flush batch telemetry to the coordinator. |
| **Run labels** | | | | | |
| `name` | str | `vfl-run` | suggested names | all | Session name recorded in logs and the run index. |
| `dataset_tag` | str | `unspecified` | datasets in manifests | all | Informational dataset label recorded in logs. The form offers the datasets named in the parties' manifests. |
<!-- config-table:end -->

## 4b. Models and modalities (v2)

Each party's bottom model is chosen with `bottom_model` — one name, or a per-party map:

```bash
# a CNN at the active party (image tiles), MLPs elsewhere
python -m vfl.cli start --set 'bottom_model={"active":"cnn","default":"mlp"}' --wait
```

| bottom_model | input it expects | from |
|---|---|---|
| `mlp` (default), `linear` | any feature vector | tabular data |
| `cnn` | image tile `[C,H,W]` (manifest `shape`) | `image_strips` / `image_grid`, `npz`, `multimodal_digits` |
| `cnn1d`, `gru` | sensor window `[C,T]` | `synthetic_sensors`, `har` |
| `text` | frozen-encoder embeddings | `vfl.cli embed-text`, `multimodal_digits` |

Every bottom maps to `[batch, embed_dim]`, so the wire contract and top model are unchanged. Multi-class
is automatic: `num_classes` is inferred (2 → sigmoid + BCE, k>2 → softmax + cross-entropy + macro
one-vs-rest AUROC), or set it explicitly. The B0/B1 baselines use the same model family and at least the
split model's capacity. CPU runs are deterministic (`torch.use_deterministic_algorithms(True)`, fixed
threads).

New datasets: `digits`, `npz` (large images, e.g. CIFAR saved as .npz), `synthetic_sensors`, `har`
(UCI HAR), `multimodal_digits` (image + tabular + text parties). The text encoder is **frozen** and runs
at the party (offline); only the projection after it is trained — this is a feature extractor, not split
learning through a language model. Install `requirements-optional.txt` for MiniLM/DistilBERT, or use
`--text-encoder hashing` (offline, no extra deps).

Efficiency codecs: `fp16`, `int8` (per-row scale), `topk` (sparse). Straggler policy `drop` lets training
proceed without a party that misses its slot deadline (`straggler.policy=drop`).

## 4c. Security, identity and privacy (v2)

**Live security gate.** Every gRPC call passes a server interceptor that enforces per-peer rate limits
(`--rate-limit-rps`), per-peer and server-wide connection caps, certificate revocation and expiry, and
binds the message's claimed identity (and, in PKI mode, role) to the certificate. Representations and
gradients are validated (shape, finite values, optional norm bound). Every rejection is written to
`logs/<node>/security.jsonl` with: time, source IP, claimed identity, certificate identity, attack type,
layer, outcome (blocked/detected/succeeded), defence, and evidence source (server-observed vs
attacker-reported). Each node also serves Prometheus metrics at `/metrics`.

**PKI and enrollment** (`vfl.cli certs --mode pki`): an offline root plus a name-constrained intermediate;
short-lived certificates (24 h, auto-renewed) carrying role and name in the SAN
(`spiffe://oncloud9/vfl/<role>/<name>`). Parties generate their own keys and enrol with a single-use,
expiring token (`vfl.cli token create`) or operator approval; revoked serials are pushed to every node.
The coordinator dashboard gains **Security**, **Inference** and **Identities** tabs (join requests,
revoke, join-mode switch, and a lab attack launcher).

**Privacy.** `defense.label_rr_epsilon` applies randomized response to the training labels before training. The
trained model and every gradient sent to passive parties are then formally ε-label-DP (δ=0); this is the setting
that satisfies NFR-8 strictly. Validation/test metrics are computed on the true labels and are outside that scope.
`grad_noise` is reported as gradient perturbation, with an RDP (ε, δ) for the gradient channel only (not
end-to-end), plus the budget if all passives collude. `defense.rep_noise_sigma` adds representation noise against
feature reconstruction. Since v2.1 every defence draws its randomness from a secret local to the party adding it
(`defense.noise_source: secret`). In v2 it came from the shared session seed, so the protected-against party could
recompute and remove it; `noise_source: seed` keeps that reproducible behaviour for lab comparisons only.
`security.private_init` keeps passive bottom-model initialisation secret from the active party.

**DH-PSI for any number of parties.** `alignment: dh_psi` uses the v2 two-party protocol with one passive party,
and with two or more passives a multi-party protocol through the active party (v2.1): every set is blinded by every
party's key and only fully blinded values are compared. Everyone learns the global intersection and the set sizes;
the active party additionally learns which passives hold each of its own ids and sub-intersection sizes; passives
learn nothing more. Semi-honest parties only; see docs/PROJECT_DOCUMENTATION.md §6.

**Security conformance suite** (lab only, standalone since v2.1). It needs only gRPC endpoints, certificates
and a target config, with no dashboard or Flask process:

```bash
python -m vfl.cli attack all --lab-mode --coordinator HOST:50050 --active HOST:50051 \
    --certs certs/ --config deployments/lab/deployment.json --out results/attacks/
```

It runs a client for each class of bad request (missing/other-CA/expired/mismatched certificate,
non-participant, unknown session, replay, wrong digest, non-finite tensor, bad enrollment token) and
confirms the gate refuses it, plus the inference measurements (label-leakage, membership) and the
deviating-party monitor. Outputs in `--out`: `report.json` and `report.csv` (attack, layer, expected
outcome, observed outcome, evidence source), `summary.txt` (PASS/FAIL), and `security.jsonl`. When the
target config lists the nodes' log folders, rows the nodes also logged are marked *server-observed*.
Exit code 0 = all as designed, 1 = not, 2 = refused by a safeguard.

Safeguards: `--lab-mode` is required, and targets outside the config's hosts are refused (without
`--config`, loopback only). A target with nothing listening is reported as an **error**, never as
"blocked". The same command runs from three places with identical verdicts: the CLI, the coordinator's
Security tab (`coordinator --lab-mode`, operator only) and the unified UI's Security & Attacks page
(`vfl.cli attack-ui --token T`, needs an explicit acknowledgement per run). It only verifies the
defences already built, mints no capability, and is for your own test deployment only.

> Scope: experimental alignment is a non-private reference; only `label_rr_epsilon` is a formal guarantee;
> mTLS authenticates peers but does not hide tensors from their authorised recipient. See
> docs/PROJECT_DOCUMENTATION.md §8.

## 4d. The unified UI (v2.1)

```bash
python -m vfl.cli ui --token "$(openssl rand -hex 16)"     # http://127.0.0.1:8090/?token=...
python -m vfl.cli attack-ui --token T                       # same UI, opened on Security & Attacks (lab mode)
python -m vfl.cli bench-ui                                  # kept for compatibility: the same UI
```

One page with seven sections. It links to the node dashboards (:8080, :8081, ...) and does not replace them.
Every action runs the same `python -m vfl.cli ...` or bench command as a background job, so the UI and the CLI
give the same results. `tests/test_webui.py` checks that every CLI subcommand has a UI route.

| Section | What it does | CLI equivalent |
|---|---|---|
| **Data preparation** | every `prepare-data` option as dropdowns, toggles and pickers: partitioner v1/v2, dataset (built-in, CSV upload with a label-column dropdown, `.npz`, HAR zip, synthetic sensors, multimodal), parties, strategy with alpha / top-frac / informative party, a column-to-party table for `manual`, overlap, id style, binarize, grid, n / features, one-hot, evaluate, sentinel. **Preview** shows per-party columns/shape, rows, intersection, label balance, importance share and the experiment-only reference AUROC before anything is saved. Saved datasets are listed with their manifests. Certificates (static or PKI) for N parties; text embeddings | `prepare-data`, `certs`, `embed-text` |
| **Deployment** | local mode: one click starts coordinator + active + 1-8 passives with a saved dataset and certificates, with per-node status (starting / live / failed), log tails, dashboard links, and stop / restart. Multi-machine mode: one zip per machine (its own cert + key, CA cert, data file and manifest; never `ca.key`) plus the exact run command, then shows nodes as they register | `coordinator`, `active`, `passive`, `netem`, `scripts/run_local.sh` |
| **Training / Sessions** | the schema-generated session form, start / stop / resume, baselines, join tokens, forwarded to the chosen coordinator | `start`, `stop`, `status`, `wait`, `config`, `baseline`, `token` |
| **Security & Attacks** | the standalone attack simulator against a deployment from the Deployment panel: pass/fail table, report downloads | `attack` |
| **Tests & Benchmarks** | pytest (all / not-slow / one file) with live output and a pass/fail table; the benchmark launcher for every suite (e1, e2, eff, def, psi, e5, sweep, lopo, security) with grids chosen by dropdown or multi-select, a queue with progress, cancel and rerun | `pytest`, `bench/run_benchmarks.py` |
| **Analytics** | every figure from `bench/analyze.py` (the same functions as `report.md`) as PNG / SVG / CSV. Model: AUROC/accuracy/F1/loss vs epoch with seed bands for B0/B1/C1, ROC + confusion matrix (macro one-vs-rest for multi-class), time-to-target, leave-one-party-out contribution, privacy vs utility, per-bottom-model and per-partition comparisons. Systems: epoch time and p50/p95 vs RTT and batch size, throughput, stacked batch-time breakdown, bytes per epoch and per entity, CPU/RSS per node over time, slot wait and stragglers, scaling, codec comparison, PSI cost, E5 fault timeline, security pass/fail. Filter / group by any config key, mean ± std with 95% CI, side-by-side comparison of two runs | `bench/analyze.py` |
| **Reports** | the NFR scorecard linked to its evidence runs, `report.md`, and a report bundle (zip) | `bench/analyze.py` |

Access: views are read-only. Every action needs the operator token when one is set. The UI binds 127.0.0.1
by default and refuses a non-loopback `--host` without `--token`, because it can start processes. The
attack launcher additionally needs `--lab-mode` and a token. All chart and UI code is vendored; nothing
loads from a CDN.

Workspace (`--workspace`, default `.`): `datasets/<name>/`, `uploads/`, `certs/<name>/`,
`deployments/<name>/` (node logs, `deployment.json`, bundles), `results/` (benchmarks, attack reports).

### UI walkthrough (viva)

A 10-minute demo on one laptop. Everything below is clicks; the CLI line for each step is in brackets.

1. **Start the UI**: `python -m vfl.cli ui --token demo --lab-mode`, then open
   `http://127.0.0.1:8090/?token=demo`.
2. **Data preparation**: partitioner v2, dataset `breast_cancer`, parties 3, strategy `dirichlet`,
   alpha 0.5, overlap 0.8. Press **Preview** and point out the intersection (< all rows, so alignment is
   real), the uneven importance shares, and the reference AUROC labelled experiment-only. Press **Generate**
   (`prepare-data --partitioner v2 ... --out NAME`). Under Certificates, pick 2 passives, static, and
   generate (`certs --passives 2`).
3. **Deployment**: pick the dataset, 2 passive parties and the certificate set. Note the active-party
   dropdown is fixed at 1 (FR-1). **Launch**, wait for four green "live" nodes, and open a node dashboard
   from its link. Optionally show a log tail and **restart** passive-2.
4. **Training / Sessions**: the form is generated from the schema. Set epochs 10, bottom model `cnn` for one
   party to show the CNN fields appear, and set `defense.label_rr_epsilon` to 2 (formal label-DP). **Preview**,
   then **Start**. Follow the epochs on the coordinator dashboard; it is the same session.
5. **Security & Attacks**: target this deployment, case `all`, tick the acknowledgement, **Run**. Walk
   through the table: auth/protocol cases *blocked* (gate works); `membership` *succeeded* (experimental
   alignment is non-private, as documented); label-leakage measured, not prevented by mTLS. Download the CSV.
   The coordinator's Security tab shows the same run.
6. **Tests & Benchmarks**: run `not-slow` pytest and show the pass/fail table. Queue a small `e2` grid
   (RTT 0, 50 × batch 32, 128) with seeds 1, 2.
7. **Analytics**: pick the benchmark, group by `batch_size`, show seed bands, ROC/confusion, latency vs RTT and
   the bytes figures. Compare two runs side by side. Download a PNG and a CSV.
8. **Reports**: the NFR scorecard with evidence links, then **Export** the report bundle.
9. **Deployment → Stop**: every node and its children exit (no orphans).

## 5. Tests, benchmarks and the report

```bash
pytest -q                                   # unit + end-to-end (pytest -m "not slow" skips e2e)
make bench-quick                            # ~3 min smoke version of the evaluation plan
make bench                                  # full plan: E1, E2, NFR-6, FR-13, FR-12, E5 + security
```

`make bench` runs, in order:

1. `bench/run_benchmarks.py` — launches a local deployment (separate processes, mTLS) and runs
   * **E1** B0 centralized, B1 active-only, local VFL and networked C1 over seeds 1–3, plus a
     same-seed repeat (NFR-1, NFR-2);
   * **E2** RTT ∈ {0, 20, 50, 100} ms × batch ∈ {32, 64, 128} through a delay relay that also
     counts wire bytes (NFR-3, NFR-4, NFR-5);
   * **eff** fp16 vs fp32 paired seeds (NFR-6); **def** defence off / σ ∈ {0.5, 1, 2} with
     label-leakage attacks (FR-13, NFR-8); **psi** DH-PSI vs experimental alignment (FR-12);
   * **E5** duplicate + 2 s delay with 5 s and 1 s deadlines, stop + checkpoint resume (NFR-10).
2. `bench/security_test.py` — 100 invalid-certificate attempts, certificate/identity binding,
   capture-and-scan of party traffic (with a plaintext control run), sentinel log audit (NFR-7).
3. `bench/analyze.py` — writes `results/bench/report/`:
   `report.md`, `nfr_table.md/.csv` (measured value + pass/fail/not attempted), `requirements_matrix.md`,
   `summary.json` and figures (utility vs epoch, epoch/p95 time and throughput vs RTT, bytes per
   entity, batch-time breakdown, privacy–utility, fp16 efficiency).

Against an already running LAN/cloud deployment:

```bash
python bench/run_benchmarks.py --url http://192.168.1.10:8080 --data-dir data --suites e2,e5 --out results/lan
python bench/analyze.py results/lan --cost configs/cloud_cost.example.yaml   # NFR-9 with your price snapshot
```

(remote mode emulates RTT at the passive parties; `--data-dir` with every CSV is only needed for
the baselines, which by design read all parties' data in one process — experiments only.)
For kernel-level delay instead: `sudo tc qdisc add dev eth0 root netem delay 25ms`, or put
`python -m vfl.cli netem --listen 0.0.0.0:60051 --target ACTIVE:50051 --rtt-ms 50 --stats net.json`
in front of the active party and start it with `--advertise RELAY_HOST:60051`.

## 6. Logs and telemetry (FR-10)

| where | contents |
|---|---|
| `logs/<node>/<node>.events.log` | human-readable node log (also shown live in the dashboard) |
| `logs/<party>/<session>/batches.jsonl` | one record per batch: forward/encode/RPC/network/wait/compute/backward ms, tensor & serialized bytes, attempts, duplicates |
| `logs/<party>/<session>/epochs.jsonl`, `resources.jsonl`, `run_summary.json` | per-epoch metrics, CPU/RSS samples, run summary with versions & determinism settings |
| `runs/<session>/` (coordinator) | `config.json`, `events.jsonl`, `batches_<party>.jsonl`, `epochs.jsonl`, `summary.json` |
| `runs/sessions_index.json` | all sessions and their outcome |
| `checkpoints/<party>/<session>/ckpt_epoch<k>.pt` | epoch checkpoints used by resume |

Records contain timings, counts, sizes and aggregate metrics only — never entity ids, feature
values or labels (verified by the sentinel audit). JSON endpoints: `GET /api/state` on every
node, `GET/POST /api/sessions[...]` on the coordinator.

## 7. How the PRD requirements map to the code

| ID | where |
|---|---|
| FR-1 | `coordinator.register/start_session` (role checks, single active, label-holding passive rejected) |
| FR-2 | `alignment.py` adapter interface; experimental mode labelled non-private everywhere |
| FR-3 | `data.make_splits/epoch_batches` from (seed, aligned order); `batch_digest` checked per message |
| FR-4 | passive sends only `f_i(X_i)`; the protocol has no feature/label fields |
| FR-5 | `core.ActiveCore.train_step`; one slot per batch waits for all passive parties |
| FR-6 | synchronous slot barrier at the active party + epoch barriers at the coordinator |
| FR-7 | `PassiveCore.backward_step`, one optimizer step per committed batch |
| FR-8 | `MsgHeader` (session/epoch/batch/attempt/sender/type); cached replies for duplicates/retries |
| FR-9 | `config.py`, `configs/*.yaml`, dashboard form, `--set` |
| FR-10 | `telemetry.py`, coordinator `runs/` |
| FR-11 | `security.py`, `transport.py` (mTLS + certificate CN must equal claimed sender) |
| FR-12 | `alignment.DHPSIParty` (two parties) and `alignment.MultiPSIHub` + `run_psi_member` (three or more, v2.1): semi-honest DH-PSI; everyone learns the intersection and set sizes |
| FR-13 | `defense.py` (clip + Gaussian noise on returned gradients; norm and cosine label-leakage attacks) |
| FR-14 | not attempted (Flotilla adapter is a stretch goal) |

## 7a. Hardened profile

For any run with real data, start the coordinator with the hardened profile (DH-PSI, ε = 2 label-DP, secret
passive initialisation, norm bounds, anomaly detection and the passive guard):

```bash
python -m vfl.cli coordinator --config configs/session.secure.yaml --tls-dir certs --ui-token "$(openssl rand -hex 16)"
```

Operational notes:
- Dashboards reached through a DNS name (not an IP or localhost) need `VFL_UI_ALLOWED_HOSTS=name1,name2`.
- Scripted clients must send JSON (or the header `X-VFL-Request: 1`), with the token in `Authorization: Bearer`.
- `--insecure` works on loopback only.
- Dashboards on a non-loopback `--ui-host` need the token for every view. Without `--ui-token` one is generated
  into `logs/<node>/ui_token`; open the page with `?token=...`.
- Re-run `vfl.cli certs` if you have certificates from before v2.3, so they carry their role.

## 8. Scope (as required by the PRD)

* `alignment: experimental` (the default) is a reproducible, explicitly **non-private** reference alignment. Use
  `alignment: dh_psi` for private alignment. DH-PSI is a semi-honest protocol implemented for this study.
* Formal label-DP (`label_rr_epsilon`) covers the trained model and the gradients sent to passives; validation/test
  metrics use true labels. Runs with a defence on are not bit-reproducible unless `defense.noise_source: seed`
  (lab comparisons only).
* `grad_noise` and representation noise are reported as perturbation with their parameters and measured attack AUC.
* mTLS protects against network observers and unauthenticated peers, not inference by an authorised recipient.
* Duplicate suppression covers an uninterrupted run; restart is epoch-checkpoint based, not exactly-once.
* One active (label-owning) party per session (FR-1). Several active parties run as several deployments.
* Data-preparation previews and the `importance` / `correlated` strategies read every party's columns in one place:
  experiment set-up only, labelled as such.
* FR-14 (Flotilla adapter) was not attempted.

## Troubleshooting

* *Party shows "coordinator unreachable"* — check `--coordinator HOST:PORT`, firewall, and that the
  coordinator uses the same `--tls-dir` CA.
* *`PERMISSION_DENIED … certificate identity`* — the node's `--id` must match its certificate name.
* *`batch membership digest mismatch`* — parties hold different data versions or ids; regenerate data.
* *`parties still busy`* — the previous session is still shutting down; wait a few seconds.
* *Need more detail* — `--log-level DEBUG`, then read `logs/<node>/<node>.events.log`.
