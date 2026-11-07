# Contributors and ownership

Team **OnCloud9** · DS252 · VFL system `v2.3.1`

The work is split four ways along the PRD's two workstreams (ML: Lavish and Divya; Systems: Atrayee and Amisha).
Each person owns about a quarter of the repository (≈5,700–6,000 lines of code, tests and docs each) and is the
person who pushes those files and fixes issues opened against them. As in the PRD, each pair reviews the other
member's work, so no one is the only person who knows an area.

| Member | Workstream | Area | Files | Lines (code / tests / docs) | Reviewer |
|---|---|---|---|---|---|
| **Lavish Singh** (26318) | ML | Alignment and PSI, privacy defences, passive guard, conformance suite, benchmarks and analysis, PRD | 28 | 4,317 / 1,041 / 349 | Divya |
| **M. Divya** (26454) | ML | Data, partitioning, models, training core, baselines, configuration schema, data-prep UI, project documentation and diagrams | 38 | 4,370 / 1,199 / 399 | Lavish |
| **Atrayee Samanta** (26437) | Systems | Wire protocol, coordinator, active/passive runtime, transport, codec, telemetry, node dashboards | 28 | 5,287 / 206 / 426 | Amisha |
| **Amisha Sadhukan** (26386) | Systems | Certificates and PKI, enrollment, security gate, web hardening, CLI, unified UI back end, deployment, packaging, README, dev.md | 28 | 3,602 / 957 / 1,151 | Atrayee |

## Who pushes what

Paths are relative to the repository root. Every file in the repository appears in exactly one list.

### Lavish Singh — branch `feat/privacy-eval`

**Requirements:** FR-12, FR-13 · NFR-1 to NFR-6 and NFR-8 to NFR-10 (benchmark suites and report)

```
PRD.md
bench/__init__.py
bench/analyze.py
bench/run_benchmarks.py
bench/ui.py
bench/ui_static/app.js
bench/ui_static/index.html
bench/ui_static/style.css
configs/cloud_cost.example.yaml
docs/images/prd_figure1_architecture.png
tests/bench_fixture.py
tests/test_alignment.py
tests/test_analytics.py
tests/test_attacks_standalone.py
tests/test_bench_ui.py
tests/test_defence_randomness.py
tests/test_defense.py
tests/test_multiparty_psi.py
tests/test_passive_guard.py
vfl/alignment.py
vfl/attacks/__init__.py
vfl/attacks/jobs.py
vfl/attacks/lab.py
vfl/attacks/runner.py
vfl/attacks/suite.py
vfl/defense.py
vfl/passive_guard.py
vfl/psi_wire.py
```

### M. Divya — branch `feat/data-models-core`

**Requirements:** FR-2 (data side), FR-3, FR-5, FR-7, FR-9 · NFR-1, NFR-2

```
configs/session.yaml
configs/synthetic.yaml
docs/CONFIG.md
docs/PROJECT_DOCUMENTATION.md
docs/PROJECT_DOCUMENTATION.pdf
docs/diagrams/architecture.mmd
docs/diagrams/code_layers.mmd
docs/diagrams/data_pipeline.mmd
docs/diagrams/deployment.mmd
docs/diagrams/session_lifecycle.mmd
docs/diagrams/training_sequence.mmd
docs/images/architecture.png
docs/images/code_layers.png
docs/images/data_pipeline.png
docs/images/deployment.png
docs/images/session_lifecycle.png
docs/images/training_sequence.png
tests/config_scan.py
tests/test_config_schema.py
tests/test_core.py
tests/test_data.py
tests/test_dataprep.py
tests/test_models.py
tests/test_models_v21.py
tests/test_partition.py
vfl/config.py
vfl/config_schema.py
vfl/core.py
vfl/data.py
vfl/datasets.py
vfl/local_sim.py
vfl/metrics_curves.py
vfl/models.py
vfl/partition.py
vfl/webui/dataprep.py
vfl/webui/static/app.js
vfl/webui/static/index.html
vfl/webui/static/webui.css
```

### Atrayee Samanta — branch `feat/protocol-runtime`

**Requirements:** FR-1, FR-4, FR-6, FR-8, FR-10 · NFR-3, NFR-4, NFR-5, NFR-10 (runtime side)

```
CHANGELOG-v2.md
CONTRIBUTORS.md
proto/vfl.proto
scripts/gen_proto.sh
tests/conftest.py
tests/fake_node.py
tests/test_codec.py
tests/test_coordinator.py
tests/test_integration.py
vfl/__init__.py
vfl/active.py
vfl/codec.py
vfl/coordinator.py
vfl/dashboard/__init__.py
vfl/dashboard/static/app.js
vfl/dashboard/static/config_form.js
vfl/dashboard/static/index.html
vfl/dashboard/static/style.css
vfl/ids.py
vfl/limits.py
vfl/netem_proxy.py
vfl/party_base.py
vfl/passive.py
vfl/proto/__init__.py
vfl/proto/vfl_pb2.py
vfl/proto/vfl_pb2_grpc.py
vfl/telemetry.py
vfl/transport.py
```

### Amisha Sadhukan — repository setup + branch `feat/security-ui-deploy`

**Requirements:** FR-11 · NFR-7, NFR-9 (deployment side)

First commit on `main` (repository skeleton; `PRD.md` in it is Lavish's file, pushed here so the plan is in the
repository from the start):

```
.gitignore
.dockerignore
README.md
PRD.md
requirements.txt
requirements-optional.txt
pytest.ini
Makefile
```

Then on `feat/security-ui-deploy`:

```
bench/harness.py
bench/security_test.py
configs/session.secure.yaml
dev.md
docker/Dockerfile
docker/docker-compose.yml
docs/DEPLOYMENT.md
scripts/run_local.sh
scripts/stop_local.sh
tests/test_deploy.py
tests/test_enrollment.py
tests/test_redteam.py
tests/test_security_gate.py
tests/test_webui.py
vfl/cli.py
vfl/enrollment.py
vfl/security.py
vfl/websec.py
vfl/webui/__init__.py
vfl/webui/deploy.py
vfl/webui/jobs.py
```

## Push workflow

The repository root is the contents of the `vfl-system/` folder. Suggested name: `oncloud9-vfl` (public).

1. **Amisha creates the public repository** on GitHub, adds the other three as collaborators, and pushes the
   skeleton as the first commit on `main`. `.gitignore` is in this first commit so that `certs/`, `data/`, `logs/`,
   `runs/`, `checkpoints/`, `results/` and private keys can never be committed. Then protect `main`: pull request
   required, one approving review.

   ```bash
   # next to the unzipped release bundle (folders vfl-system/ and push-lists/)
   git init oncloud9-vfl && cd oncloud9-vfl
   tar -C ../vfl-system -cf - -T ../push-lists/0-skeleton-main.txt | tar -xf -
   git add . && git commit -m "Repository skeleton: packaging, README, PRD"
   git branch -M main
   git remote add origin git@github.com:<org-or-user>/oncloud9-vfl.git
   git push -u origin main
   ```

2. **Each member clones and pushes their own branch** (Amisha too, for `feat/security-ui-deploy`). Copy only the
   files in your list (`push-lists/<name>.txt` holds the same list as above):

   ```bash
   git clone git@github.com:<org-or-user>/oncloud9-vfl.git && cd oncloud9-vfl
   git checkout -b feat/protocol-runtime                                   # your branch
   tar -C ../vfl-system -cf - -T ../push-lists/atrayee.txt | tar -xf -     # your list
   git status                                   # only your files; nothing from certs/ data/ logs/
   git add . && git commit -m "Protocol, coordinator and party runtime"
   git push -u origin feat/protocol-runtime
   ```

   Open a pull request into `main`; your reviewer (table above) approves it.

3. **Merge order:** `feat/protocol-runtime` (Atrayee) → `feat/data-models-core` (Divya) → `feat/privacy-eval`
   (Lavish) → `feat/security-ui-deploy` (Amisha). `main` does not import cleanly until all four are merged.

4. **Release check** on a machine with the requirements installed:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   pytest -q
   make data certs local && python -m vfl.cli start --set epochs=3 --wait && make stop
   git status                   # must be clean: generated files are git-ignored
   ```

5. **Tag** the submitted commit: `git tag -a v2.3.1 -m "VFL system v2.3.1" && git push origin v2.3.1`.

6. **Issues:** the owner of the affected file fixes it on a branch `fix/<issue-number>`, the pair partner reviews,
   and it merges into `main` with `Fixes #<issue-number>` in the pull request.

## Before every push

* Never commit `certs/`, `*.key`, `data/`, `datasets/`, `uploads/`, `deployments/`, `logs/`, `runs/`,
  `checkpoints/`, `results/`, `.venv/` or `ui_users.json`. They are git-ignored; check `git status` anyway.
* Never put a UI token, password or private key in a commit message, issue or README.
* Run the tests for your own area (`pytest -q tests/test_<area>.py`) before opening the pull request.
