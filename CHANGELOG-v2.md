# VFL v2 changes

All v1 behaviour and defaults are preserved; every v2 feature is opt-in (config, `--set`, dashboard, or a CLI flag).

## Data & models
- Integrated `vfl/partition.py` (8 strategies, overlap, opaque ids, manifests); `prepare-data --partitioner v2`.
- Per-party manifests (`<name>.manifest.json`); the coordinator checks every party holds the session's split.
- New datasets: `digits`, `npz` (large images), `synthetic_sensors`, `har`, `multimodal_digits`; `.npz` party files.
- Bottom-model registry: `mlp`, `linear`, `cnn`, `cnn1d`, `gru`, `text` (per-party selectable).
- Multi-class training (cross-entropy + macro one-vs-rest AUROC); binary remains the default.
- B0/B1 baselines use the VFL run's model family and >= its capacity; deterministic CPU runs.
- Frozen text encoder (`embed-text`, MiniLM/DistilBERT/hashing) + a multimodal run.

## Security, identity, privacy
- Live gRPC security gate: identity/role/revocation/expiry checks, rate limits, connection caps.
- Tensor validation (shape, finite, norm bound) on every representation and gradient.
- `security.jsonl` structured security events; Prometheus `/metrics` on every node.
- PKI mode (offline root + name-constrained intermediate, 24 h certs, SPIFFE SAN); enrollment service
  (static/token/approval); certificate renewal and a revocation list.
- Security conformance suite (`vfl.cli attack`, lab-only) + `bench/security_test.run_conformance`.
- Formal label-DP via randomized response (`defense.label_rr_epsilon`); RDP accountant for the gradient
  channel; representation noise.
- Codecs `int8`, `topk`; straggler `drop` policy.

## Dashboards & benchmarks
- Coordinator Security / Inference / Identities tabs + lab attack launcher; new session-form fields;
  optional dashboard logins (viewer/operator) and HTTPS.
- New `bench-ui` analytics dashboard (launcher, runs explorer, analytics, NFR scorecard, export).
- Sweeps over parties × partition × overlap; analyze emits PNG + SVG + CSV and a scaling figure.

## Not attempted (documented future work)
- FR-14 Flotilla adapter; asynchronous training; batch-level WAL crash recovery; dynamic membership;
  cloud C2 cost table (the NFR-9 machinery exists; no cloud run was performed here).

## v2.1 (in progress)

### Item 2: complete training form from one config schema
- `vfl/config_schema.py` is the single source of truth for every session key: type, default, dropdown presets,
  group, help text, and the bottom models it applies to. `DEFAULT_SESSION`, `validate_session_config`, `--set`
  checks, the coordinator form (`GET /api/config/schema`) and `docs/CONFIG.md` are all generated from it.
- The coordinator's "Start a session" form now covers every key the training code reads, with no free-text inputs.
  Numbers and enums are preset dropdowns and booleans are toggles. `bottom_model` gets one dropdown per detected
  party, with incompatible kinds disabled by the party's manifest shape. Fault slots are epoch/batch rows,
  `resume_from` lists past sessions (and loads that session's config), and `participants` lists detected parties.
  Fields show or hide with the choices made (CNN fields only when a party uses `cnn`, clip/noise only with
  `grad_noise`, ...). Fields are grouped in collapsible sections, with "Reset to defaults" and "Preview"
  (server-side validation) buttons.
- `--set` refuses unknown keys (with a "did you mean" hint), system-set keys (`start_epoch`) and out-of-range values
  before anything is sent. YAML files keep working; unknown YAML keys only warn.
- New CLI: `python -m vfl.cli config {docs,schema,defaults,validate}`.
- New opt-in model options, whose defaults rebuild the v2 networks exactly: `cnn_kernel`, `cnn_pool`,
  `cnn1d_kernel`, `cnn1d_pool`, `gru_layers`, `text_hidden`. `cnn_channels` may now list 1-6 layers.
- Config validation no longer imports torch, so the CLI client (`start`, `config validate`) runs without it.
- `tests/test_config_schema.py` fails if the code reads a config key that has no schema entry (AST scan of `vfl/`
  and `bench/`). It also checks that the v2 defaults are unchanged, that every preset validates, and that the form
  round-trips to the same config (in Python and through the dashboard's JS).

### Item 3: Data preparation in the UI
- Every `prepare-data` option is a dropdown, toggle or picker (`vfl/webui/dataprep.py`, one description per
  option, with show/hide rules). There is a CSV upload with a label-column dropdown, `.npz` and HAR-zip uploads
  (zip-slip safe), and a column-to-party table for the `manual` strategy instead of typing JSON.
- The UI builds a `prepare-data` argv checked by the real CLI parser and runs the CLI as a job, so outputs are
  byte-identical to the CLI (tested for v1, v2 dirichlet + evaluate, image strips + binarize, manual columns).
- **Preview** runs the same command into a scratch folder, shows per-party columns/shape, rows, intersection,
  label balance, importance share and the experiment-only reference AUROC, then deletes the folder.
- Saved datasets are listed with their manifests and offered in Deployment, Training and the benchmark
  launcher. Certificates (static or PKI) are generated for the chosen number of parties. `embed-text` is
  available from the same page.

### Item 4: Deployment panel
- Local mode reuses `bench/harness.LocalDeployment`, now extended with `plan()`, `status()`, `log_tail()`,
  `restart()` and `stop_node()`, the Python twin of `scripts/run_local.sh`. One click starts the coordinator,
  the active party and 1-8 passives. Each node shows starting/live/failed, a log tail, its dashboard link, and
  stop/restart. Every node runs in its own process group, so stop leaves no orphans (tested). Port "slots" let
  deployments run side by side. An optional `netem` RTT relay is available.
- Multi-machine mode builds one zip per machine (its own cert + key, CA cert, data file + manifest, the exact
  run command). It never bundles `ca.key`, and shows nodes as they register.
- One active party per session (FR-1) is shown fixed at 1, with the reason. Several active parties = several
  deployments; a multi-session coordinator is documented as future work.

### Item 1: Standalone attack simulator
- `python -m vfl.cli attack <case|family|all> --lab-mode --coordinator H:P --active H:P --certs DIR --config
  targets.json --out DIR` needs no dashboard: grpc/protobuf load lazily and Flask is never imported.
- Outputs: `report.json` and `report.csv` (attack, layer, expected, observed, evidence source), `summary.txt`
  (PASS/FAIL), and `security.jsonl`. Rows are marked server-observed when the target's node logs recorded the
  same attack.
- Safeguards: `--lab-mode`, a target allowlist (loopback only without `--config`), and in the UIs the operator
  token plus a per-run acknowledgement.
- Fix: a probe against a port with nothing listening used to count as "blocked" (a false PASS for a stopped
  deployment). It is now an error.
- The coordinator's Security tab (`LabRunner`) and the unified UI's attack page run the same CLI command
  through `vfl/attacks/jobs.py`. Identical verdicts are tested.

### Item 5: Tests, benchmarks and analytics
- pytest from the UI (all / not-slow / one file) with live output and a JUnit-based pass/fail table. One file
  that fails to import no longer hides the other results (`--continue-on-collection-errors`).
- The benchmark launcher covers every `run_benchmarks.py` flag and suite. It adds `lopo` (leave-one-party-out)
  and `--codecs` for the efficiency suite (fp32/fp16/int8/top-k). Jobs run in a serial queue with progress,
  cancel and rerun.
- `bench/analyze.py` has a figure registry (`FIGURES`) used by both `report.md` and the UI. It covers every
  requested model and systems graph, each as PNG, SVG and CSV, with filters, grouping by any config key,
  95% CI, two-run comparison, the NFR scorecard with evidence links, and a report bundle.
- Parties now report a down-sampled ROC curve and confusion counts (aggregates only) and CPU/RSS per epoch.
  These feed the ROC/confusion and resources figures. Training is unchanged.

### Item 6: One UI
- `python -m vfl.cli ui` (127.0.0.1:8090) has seven sections: Data preparation · Deployment · Training/Sessions
  · Security & Attacks · Tests & Benchmarks · Analytics · Reports. It links to the node dashboards.
  `attack-ui` opens it on the attack page in lab mode, and `bench-ui` is kept as an alias.
- Every CLI subcommand maps to a UI route (`CLI_ROUTES`, checked against the parser by `tests/test_webui.py`).
  Every POST/DELETE route needs the operator token (tested). A non-loopback bind without `--token` is refused.

### Item 7: Docs
- README: the `ui` command and its sections, the generated hyperparameter table (kept current by a test), the
  standalone attack CLI, the data-prep and deployment panels, a UI walkthrough for the viva, and new honest
  limits (one active party per session, lab-only attacks, experiment-only previews). `docs/DEPLOYMENT.md`:
  multi-machine bundles from the UI.

### Multi-party DH-PSI and the privacy audit
- `alignment: dh_psi` now works for any number of parties. One passive uses the unchanged v2 two-party protocol;
  two or more use a multi-party protocol through the active party (`vfl/alignment.MultiPSIHub`,
  `run_psi_member`, `vfl/psi_wire.py`; new backward-compatible Align fields `task`, `indices`, `status`,
  `n_parties`). The coordinator no longer refuses dh_psi with more than two parties, and the `psi` benchmark suite
  runs for any party count. Tests: `tests/test_multiparty_psi.py` (correctness for 3-6 parties, serialised
  protobuf flow with retries, privacy properties, protocol abuse).
- Privacy fix (critical): gradient noise, randomized-response flips and representation noise were derived from the
  session seed that every party receives, so they could be removed or inverted by the party they protect against.
  Each party now uses a local secret (`defense.noise_source: secret`, the default; `seed` = the v2 behaviour, not
  private), and the active checkpoints it so a resumed run keeps the same noisy labels.
  Tests: `tests/test_defence_randomness.py`.
- Privacy report: the label-DP scope is stated correctly (model + gradients; evaluation metrics use true labels), and
  an ε for colluding passives was added. New opt-in `security.private_init` hides passive bottom-model
  initialisation from the active party.
- Deployment bundles: the coordinator gets a generated `--ui-token`, the token is redacted from the UI's open
  views, and bundle downloads (which hold private keys) need the operator token. The coordinator warns when its
  dashboard is exposed beyond loopback without auth.
- What multi-party PSI discloses is documented in `docs/PROJECT_DOCUMENTATION.md` §6.2.

## v2.3 (security hardening after the red-team review)

Every fix has a regression test in `tests/test_redteam.py`.

- **Identity and roles (RT-01, RT-05, RT-06, RT-09):** static certificates carry their role (SAN URI), and the gate
  and `Register` always enforce it, so a passive can no longer register as the active party. pki DNS names are
  role-qualified (`<name>.<role>.vfl.<domain>`) and pinned by clients. Enrollment reserves `coordinator` and lab
  identities, and refuses a name that already holds a valid certificate. Revocation covers the whole identity,
  including renewed certificates. `rogue-node` test certificates are opt-in, and only a `--lab-mode` coordinator
  accepts them.
- **Code execution (RT-02, RT-19):** `resume_from`, `participants` and session ids must be safe path components
  (coordinator, config validator, each party). Checkpoints load with `weights_only=True`. Requires `torch>=2.4`.
- **Web UIs (RT-03, RT-04, RT-18):** `vfl/websec.py` adds a Host allow-list (DNS-rebinding defence), refuses
  cross-site Origins, requires JSON or `X-VFL-Request: 1` for actions, and sets CSP, framing, nosniff and referrer
  headers. Party links are http(s)-only on the server and in the JS. Tokens are compared in constant time, are
  header-only for actions, and are stripped from the address bar. Logins are throttled.
- **Protocol abuse by an authorised party (RT-07, RT-08, RT-14, RT-15, RT-17):** only the active party reports
  utility, leakage and privacy figures. Party security events are sanitised and marked `party-reported`. The slot
  window follows the honest schedule. PSI round 1 is answered once and alignment inputs are bounded
  (`security.max_align_items`, new). Tensors are shape-checked before decoding. Party strings and payloads are
  bounded and single-line. Messages must name their sender. There is a per-identity concurrency cap (default 16)
  and 64 server workers.
- **Enrollment (RT-12, RT-13):** CSRs are validated before a token is spent, pending requests are capped, request
  ids are 128-bit, and token lifetime is at most 7 days. A root CA fetched from the dashboard must be pinned
  (token or `--ca-sha256`).
- **Operations (RT-10, RT-11, RT-16, RT-20):**
  - the lab attack allow-list covers only this deployment's private/loopback addresses;
  - the UI token reaches nodes through `VFL_UI_TOKEN`, never argv;
  - bundles are `0600`, always mTLS, and set the token through the environment;
  - UI names are validated before path joins, and zip extraction is bounded;
  - `--insecure` refuses non-loopback listeners (`--allow-insecure-remote` for a throw-away lab), and
    `run_local.sh` binds to loopback.
- New `configs/session.secure.yaml`: DH-PSI, ε=2 label-DP, `private_init`, norm bounds and anomaly detection, for
  the red/blue-team exercise and real data.
- Compatibility: v1/v2 training defaults are unchanged. Re-run `vfl.cli certs` so static certificates carry roles
  (legacy ones still work, with the role derived from the name). Existing pki leaves are re-issued automatically.
  UI clients must send `X-VFL-Request: 1` on uploads and bodyless POSTs, as the bundled UIs do.

## v2.3.1 (closing the residual risks)

- **Malicious active party:** `vfl/passive_guard.py` adds opt-in passive-side defences, all off by default so the
  attack demos still work:
  - gradient-row clipping (`security.passive_grad_clip`);
  - a parameter-drift trust region (`security.max_param_drift`);
  - a gradient anomaly monitor (`security.grad_monitor`: warn/stop);
  - a per-epoch feature-leakage audit: a ridge decoder H → X scored on held-out rows, reported as
    `feature_leakage.ridge_r2` (`security.leakage_audit`, `security.leakage_r2_max`).

  The lab-only `active --malicious scale:K | drift:K | probe` demonstrates them. `configs/session.secure.yaml`
  turns on clip, monitor and audit.
- **Dashboards:** a dashboard on a non-loopback address requires the token or a login for every view, not just for
  actions. With no `--ui-token` a token is generated into `logs/<node>/ui_token` (0600). `--ui-public-views` keeps
  the v2.2 behaviour. Multi-machine bundles give every machine its own dashboard token, through the environment,
  shown once.
