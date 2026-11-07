"""v2.1 item 2: one config schema drives the session form, --set validation and docs/CONFIG.md.

None of these tests need torch or grpc (the schema, the form and the coordinator dashboard routes are torch-free).
"""
import copy
import glob
import json
import os
import shutil
import subprocess

import pytest

from config_scan import scan
from vfl import config_schema as S
from vfl.config import DEFAULT_SESSION, deep_merge, load_session_config, validate_session_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORM_JS = os.path.join(ROOT, "vfl", "dashboard", "static", "config_form.js")

# Every key the v2 DEFAULT_SESSION had, with its v2 value (copied from v2's vfl/config.py). Pinned so that a
# schema edit cannot silently change a v1/v2 run.
V2_DEFAULTS = {
    "name": "vfl-run", "dataset_tag": "unspecified", "seed": 42, "epochs": 10, "batch_size": 64,
    "eval_batch_size": 512, "lr": 0.001, "optimizer": "adam", "momentum": 0.0, "bottom_model": "mlp",
    "hidden_dim": 64, "cnn_channels": [16, 32], "gru_hidden": 32, "num_classes": None, "embed_dim": 16,
    "top_hidden": 32, "split": {"val": 0.15, "test": 0.15}, "alignment": "experimental", "alignment_salt": None,
    "psi_group_bits": 2048, "rpc_deadline_s": 30.0, "max_retries": 3, "retry_backoff_s": 0.2,
    "slot_timeout_s": 120.0, "barrier_timeout_s": 300.0, "inject_rtt_ms": 0.0, "codec": "fp32",
    "topk_fraction": 0.25,
    "defense": {"type": "none", "clip": 0.05, "noise_multiplier": 1.0, "delta": 1e-5, "label_rr_epsilon": None,
                "rep_noise_sigma": 0.0, "rep_clip": 1.0},
    "straggler": {"policy": "wait", "slot_deadline_s": 5.0},
    "security": {"validate_tensors": True, "max_row_norm": None, "max_grad_row_norm": None,
                 "anomaly_detection": False},
    "attack_eval": True, "fault": {"duplicate": [], "delay": [], "delay_s": 2.0}, "checkpoint": True,
    "resume_from": None, "min_passive": 1, "participants": None, "torch_threads": 1, "deterministic": True,
    "target_val_auroc": None, "telemetry_flush_s": 2.0, "partition": None, "overlap": None, "binarize": None,
}
# v2.1 opt-in keys and the defaults that reproduce the v2 networks
V21_NEW = {"cnn_kernel": 3, "cnn_pool": "auto", "cnn1d_kernel": 5, "cnn1d_pool": "mean", "gru_layers": 1,
           "text_hidden": 0, "defense.noise_source": "secret", "security.private_init": False}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k in S.SECTIONS:
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out

RICH = deep_merge(DEFAULT_SESSION, {
    "name": "digits-3party", "dataset_tag": "digits", "seed": 7, "epochs": 20, "lr": 0.003, "optimizer": "sgd",
    "momentum": 0.9, "bottom_model": {"active": "cnn", "passive-2": "gru", "default": "mlp"},
    "cnn_channels": [16, 32, 64], "cnn_kernel": 5, "cnn_pool": "avg", "gru_layers": 2, "text_hidden": 64,
    "num_classes": 4, "alignment_salt": "a1b2c3d4e5f6", "codec": "topk", "topk_fraction": 0.1,
    "defense": {"type": "grad_noise", "clip": 0.02, "noise_multiplier": 2.0, "label_rr_epsilon": 2.0,
                "rep_noise_sigma": 0.1, "rep_clip": 2.0},
    "straggler": {"policy": "drop", "slot_deadline_s": 1.0},
    "fault": {"duplicate": [[0, 3]], "delay": [[1, 5], [2, 0]], "delay_s": 5.0},
    "resume_from": "s20261102-101500-ab12", "participants": ["active", "passive-1"], "target_val_auroc": 0.95,
    "partition": "image_strips", "overlap": 0.9, "binarize": "odd_even",
    "security": {"max_row_norm": 10.0, "anomaly_detection": True}, "inject_rtt_ms": 50.0,
})


def _code_files():
    files = [f for f in glob.glob(os.path.join(ROOT, "vfl", "**", "*.py"), recursive=True)
             if os.sep + "proto" + os.sep not in f and not f.endswith("config_schema.py")]
    return files + glob.glob(os.path.join(ROOT, "bench", "*.py"))


# ------------------------------------------------------------------ coverage: code <-> schema
def test_every_config_key_the_code_reads_has_a_schema_entry():
    reads = scan(_code_files(), S.SECTIONS)
    missing = {k: v[:3] for k, v in reads.items() if k not in S.BY_KEY and k not in S.SECTIONS}
    assert not missing, ("config keys read by the code but missing from vfl/config_schema.py "
                         f"(add a Param so they reach the form, --set validation and docs): {missing}")


def test_scanner_finds_known_reads():
    """Guards the guard: if the scanner silently stopped matching, the test above would pass vacuously."""
    reads = scan(_code_files(), S.SECTIONS)
    for k in ("cnn_channels", "gru_hidden", "defense.clip", "defense.rep_clip", "split.val", "fault.delay_s",
              "straggler.slot_deadline_s", "security.max_grad_row_norm", "topk_fraction", "start_epoch",
              "partition", "overlap", "binarize", "retry_backoff_s", "telemetry_flush_s", "cnn_kernel", "gru_layers",
              "text_hidden", "dataset_tag"):
        assert k in reads, k
    assert len(reads) >= 60


def test_every_schema_key_is_read_or_marked_unread():
    reads = scan(_code_files(), S.SECTIONS)
    stale = [p.key for p in S.PARAMS if p.read and p.key not in reads]
    assert not stale, f"schema entries the code never reads (remove them or mark read=False): {stale}"
    for p in S.PARAMS:
        if not p.read:
            assert p.key not in reads, f"{p.key} is marked read=False but the code reads it"


# ------------------------------------------------------------------ defaults
def test_v2_defaults_unchanged_and_new_keys_are_opt_in():
    old, now = _flat(V2_DEFAULTS), _flat(DEFAULT_SESSION)
    for k, v in old.items():
        assert now[k] == v and type(now[k]) is type(v), k
    v23_new = {"security.max_align_items": None,           # v2.3 hardening: all off / automatic by default
               "security.passive_grad_clip": None, "security.max_param_drift": None,
               "security.grad_monitor": "off", "security.leakage_audit": "off", "security.leakage_r2_max": 0.9}
    assert set(now) - set(old) == set(V21_NEW) | set(v23_new)
    for k, v in V21_NEW.items():
        assert now[k] == v
        assert S.BY_KEY[k].since == "v2.1"
    for k, v in v23_new.items():
        assert now[k] == v and S.BY_KEY[k].since == "v2.3"
    validate_session_config(DEFAULT_SESSION)


def test_schema_is_well_formed():
    assert len(S.BY_KEY) == len(S.PARAMS)
    for p in S.PARAMS:
        assert p.group in S.GROUP_IDS and p.widget in S.WIDGETS and p.type in S.TYPES, p.key
        assert p.help and len(p.help) > 10, p.key
        assert S.check_value(p, p.default) is None, f"default of {p.key} fails its own check"
        for k in p.applies_to:
            assert k in S.BOTTOM_MODELS, p.key
        for c in p.show_if:
            ref = c[1] if c[0] == "!" else c[0]
            assert ref in S.BY_KEY, f"{p.key} show_if refers to unknown {ref}"


def test_no_free_text_inputs():
    """Every form field is a dropdown (with real options), a toggle, or one of the structured widgets."""
    for p in S.form_params():
        if p.widget == "select":
            assert len(S.option_values(p)) >= 2 or p.choices, f"{p.key} has no dropdown options"
        if p.type == "bool":
            assert p.widget == "toggle", p.key
    assert S.BY_KEY["bottom_model"].widget == "party_models"
    assert S.BY_KEY["fault.duplicate"].widget == S.BY_KEY["fault.delay"].widget == "pairs"
    assert S.BY_KEY["resume_from"].widget == "session"


def test_every_preset_passes_validation():
    for p in S.form_params():
        for v in S.option_values(p):
            assert S.check_value(p, v) is None, (p.key, v)
            if p.widget == "select":
                cfg = copy.deepcopy(DEFAULT_SESSION)
                S.set_path(cfg, p.key, v)
                validate_session_config(cfg)


def test_bad_values_rejected_by_schema():
    bad = {"epochs": 0, "lr": -1, "cnn_kernel": 4, "cnn_pool": "median", "codec": "int4", "defense.clip": 0,
           "overlap": 1.5, "gru_layers": 9, "fault.delay": [[0]], "bottom_model": {"active": "resnet"},
           "attack_eval": "yes", "cnn_channels": [16, 0], "num_classes": 1}
    for k, v in bad.items():
        assert S.check_value(S.BY_KEY[k], v), (k, v)


# ------------------------------------------------------------------ --set validation
def test_set_rejects_unknown_internal_and_bad_values():
    with pytest.raises(ValueError, match="did you mean 'cnn_kernel'"):
        load_session_config(None, ["cnn_kernal=5"])
    with pytest.raises(ValueError, match="set by the system"):
        load_session_config(None, ["start_epoch=2"])
    with pytest.raises(ValueError, match="must be odd"):
        load_session_config(None, ["cnn_kernel=4"])
    cfg = load_session_config(None, ["lr=0.002", "cnn_kernel=5", "bottom_model={active: cnn, default: mlp}",
                                     "fault.delay=[[1, 5]]"])
    assert cfg["lr"] == 0.002 and cfg["cnn_kernel"] == 5 and cfg["fault"]["delay"] == [[1, 5]]
    assert load_session_config(None, ["lr=1e-3"])["lr"] == "1e-3"      # YAML string, accepted as in v2


def test_cli_config_command(capsys):
    from vfl.cli import main
    with pytest.raises(SystemExit, match="unknown config key 'defense.clipp'"):
        main(["config", "validate", "--set", "defense.clipp=0.1"])
    main(["config", "validate", "--config", os.path.join(ROOT, "configs", "session.yaml"), "--set", "epochs=3"])
    out = json.loads(capsys.readouterr().out)
    assert out["epochs"] == 3 and out["dataset_tag"] == "breast_cancer"
    main(["config", "schema"])
    sch = json.loads(capsys.readouterr().out)
    assert {p["key"] for p in sch["params"]} == {p.key for p in S.form_params()}


def test_docs_table_is_generated_and_current():
    path = os.path.join(ROOT, "docs", "CONFIG.md")
    assert open(path).read() == S.render_markdown(), \
        "docs/CONFIG.md is stale: run `python -m vfl.cli config docs --out docs/CONFIG.md`"
    for p in S.PARAMS:
        assert f"`{p.key}`" in S.render_markdown()
    readme = open(os.path.join(ROOT, "README.md")).read()
    block = readme.split(S.README_START, 1)[1].split(S.README_END, 1)[0].strip()
    assert block == S.render_table(), \
        "README hyperparameter table is stale: run `python -m vfl.cli config docs --readme README.md`"


# ------------------------------------------------------------------ form round-trip (python)
@pytest.mark.parametrize("cfg", [DEFAULT_SESSION, RICH], ids=["defaults", "rich"])
def test_form_round_trips_to_the_same_config(cfg):
    validate_session_config(cfg)
    form = S.form_values(cfg)
    assert set(form) == {p.key for p in S.form_params()}
    back = S.config_from_form(json.loads(json.dumps(form)))          # through JSON, as the browser sends it
    assert back == cfg
    assert json.dumps(back, sort_keys=True) == json.dumps(cfg, sort_keys=True)   # types too (2.0 stays 2.0)


def test_form_round_trips_session_yaml():
    cfg = load_session_config(os.path.join(ROOT, "configs", "session.yaml"))
    assert S.config_from_form(json.loads(json.dumps(S.form_values(cfg)))) == cfg


def test_form_rejects_unknown_and_bad_fields():
    with pytest.raises(ValueError, match="unknown config key"):
        S.config_from_form({"epochz": 3})
    with pytest.raises(ValueError, match="set by the system"):
        S.config_from_form({"start_epoch": 3})
    with pytest.raises(ValueError, match="epochs"):
        S.config_from_form({"epochs": "ten"})


# ------------------------------------------------------------------ form round-trip (the real JS)
def _node(script: str, payload: dict, tmp_path) -> dict:
    data = tmp_path / "in.json"
    data.write_text(json.dumps(payload))
    js = (f"const F = require({json.dumps(FORM_JS)}); const I = JSON.parse(require('fs').readFileSync("
          f"{json.dumps(str(data))}, 'utf8')); const out = (() => {{ {script} }})(); "
          "process.stdout.write(JSON.stringify(out));")
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@needs_node
@pytest.mark.parametrize("cfg", [DEFAULT_SESSION, RICH], ids=["defaults", "rich"])
def test_js_form_prefill_matches_python_and_round_trips(cfg, tmp_path):
    out = _node("return F.toPayload(F.initialValues(I.schema, I.cfg));",
                {"schema": S.schema_json(), "cfg": cfg}, tmp_path)
    assert out["form"] == json.loads(json.dumps(S.form_values(cfg)))
    assert S.config_from_form(out["form"]) == cfg
    assert out["resume_from"] == cfg.get("resume_from")


@needs_node
def test_js_dropdown_options_match_python(tmp_path):
    out = _node("const r = {}; for (const p of I.schema.params) r[p.key] = F.optionValues(p, p.default); return r;",
                {"schema": S.schema_json()}, tmp_path)
    for p in S.form_params():
        assert out[p.key] == json.loads(json.dumps(S.option_values(p))), p.key      # JS writes 0.0 as 0: compare values


@needs_node
def test_js_show_hide_rules(tmp_path):
    parties = [{"party_id": "active", "role": "active"}, {"party_id": "passive-1", "role": "passive"}]
    out = _node("""
      const P = Object.fromEntries(I.schema.params.map((p) => [p.key, p]));
      const v0 = F.initialValues(I.schema, I.defaults);
      const vis = (key, patch) => F.visible(P[key], { ...v0, ...patch }, I.parties);
      return {
        cnn_default: vis("cnn_kernel", {}),
        cnn_one_party: vis("cnn_kernel", { bottom_model: { active: "cnn", default: "mlp" } }),
        cnn1d_with_cnn: vis("cnn1d_kernel", { bottom_model: { active: "cnn", default: "mlp" } }),
        channels_cnn1d: vis("cnn_channels", { bottom_model: "cnn1d" }),
        hidden_all_gru: vis("hidden_dim", { bottom_model: "gru" }),
        hidden_default_used: vis("hidden_dim", { bottom_model: { active: "gru", default: "mlp" } }),
        hidden_all_overridden: vis("hidden_dim",
          { bottom_model: { active: "gru", "passive-1": "gru", default: "mlp" } }),
        clip_none: vis("defense.clip", {}),
        clip_noise: vis("defense.clip", { "defense.type": "grad_noise" }),
        topk: [vis("topk_fraction", {}), vis("topk_fraction", { codec: "topk" })],
        momentum: [vis("momentum", {}), vis("momentum", { optimizer: "sgd" })],
        rep_clip: [vis("defense.rep_clip", {}), vis("defense.rep_clip", { "defense.rep_noise_sigma": 0.1 })],
        slot: [vis("straggler.slot_deadline_s", {}), vis("straggler.slot_deadline_s", { "straggler.policy": "drop" })],
      };""", {"schema": S.schema_json(), "defaults": DEFAULT_SESSION, "parties": parties}, tmp_path)
    assert out["cnn_default"] is False and out["cnn_one_party"] is True and out["cnn1d_with_cnn"] is False
    assert out["channels_cnn1d"] is True and out["hidden_all_gru"] is False
    assert out["hidden_default_used"] is True and out["hidden_all_overridden"] is False
    assert out["clip_none"] is False and out["clip_noise"] is True
    for k in ("topk", "momentum", "rep_clip", "slot"):
        assert out[k] == [False, True], k


# ------------------------------------------------------------------ coordinator dashboard routes
class _Backend:
    party_id = "coord-test"
    lab = None
    lab_mode = False
    registry = None

    def __init__(self):
        self.default_config = load_session_config(os.path.join(ROOT, "configs", "session.yaml"))
        self.started = []

    def start_session(self, body):
        self.started.append(body)
        return {"session_id": "s-test", "state": "STARTING"}

    def ui_state(self):
        return {"role": "coordinator", "sessions": []}


@pytest.fixture
def dash():
    pytest.importorskip("flask")
    from vfl.dashboard import create_app
    be = _Backend()
    return be, create_app(be, "coordinator", token="tok").test_client()


def test_schema_endpoint_prefills_coordinator_defaults(dash):
    be, c = dash
    r = c.get("/api/config/schema")
    assert r.status_code == 200
    j = r.get_json()
    assert {p["key"] for p in j["params"]} == {p.key for p in S.form_params()}
    assert j["current"] == json.loads(json.dumps(S.form_values(be.default_config)))
    assert j["current"]["epochs"] == 20                       # from configs/session.yaml, not the built-in 10


def test_start_from_form_and_legacy_body(dash):
    be, c = dash
    H = {"Authorization": "Bearer tok"}
    form = S.form_values(RICH)
    assert c.post("/api/sessions", json={"form": form}).status_code == 401          # still needs the token
    r = c.post("/api/sessions", json={"form": json.loads(json.dumps(form))}, headers=H)
    assert r.status_code == 200, r.get_json()
    body = be.started[-1]
    assert body["resume_from"] == RICH["resume_from"] and "resume_from" not in body["config"]
    assert body["config"] == {k: v for k, v in RICH.items() if k != "resume_from"}
    r = c.post("/api/sessions", json={"form": {"epochs": 0}}, headers=H)
    assert r.status_code == 400 and "epochs" in r.get_json()["error"]
    c.post("/api/sessions", json={"config": {"epochs": 2}}, headers=H)               # v2 / CLI body unchanged
    assert be.started[-1] == {"config": {"epochs": 2}}


def test_validate_endpoint(dash):
    _, c = dash
    ok = c.post("/api/config/validate", json={"form": S.form_values(DEFAULT_SESSION)}).get_json()
    assert ok["ok"] and ok["config"]["epochs"] == DEFAULT_SESSION["epochs"]
    bad = c.post("/api/config/validate", json={"form": {**S.form_values(DEFAULT_SESSION), "split.val": 0.25,
                                                        "split.test": 0.8}})
    assert bad.status_code == 400 and "val+test" in bad.get_json()["error"]
    bad = c.post("/api/config/validate", json={"config": {"cnn_kernal": 3}})
    assert bad.status_code == 400 and "did you mean" in bad.get_json()["error"]
