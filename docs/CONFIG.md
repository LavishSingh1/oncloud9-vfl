# Session configuration reference

<!-- generated from vfl/config_schema.py by `python -m vfl.cli config docs`; do not edit by hand -->

Every key the training code reads. Set them in `configs/session.yaml`, with `python -m vfl.cli start --set key=value`, or in the coordinator dashboard's session form (dropdowns of the presets below). Presets are the dropdown values, not limits: `--set` accepts any value that passes the type and range check.

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

Honest limits: only `defense.label_rr_epsilon` gives a formal (end-to-end) label-DP guarantee. `defense.type: grad_noise` reports an (eps, delta) for the dL/dH channel only. `alignment: experimental` is non-private. The text encoder is chosen when the data is prepared (`prepare-data --text-encoder`, `embed-text --encoder`), not here.
