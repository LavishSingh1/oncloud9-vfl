"""v2.1 item 3: Data preparation from the unified UI.

The key property: the UI path and `vfl.cli prepare-data` produce byte-identical outputs for the same options,
because the UI builds a prepare-data argv (checked by the real parser) and runs the CLI as a job.
Torch and grpc are not needed.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

import pytest

pytest.importorskip("flask")
from vfl.cli import build_parser  # noqa: E402
from vfl.webui import create_app  # noqa: E402
from vfl.webui import dataprep as D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"Authorization": "Bearer tok", "X-VFL-Request": "1"}       # what the UI sends (CSRF header, v2.3)


def _app(tmp_path):
    app = create_app(str(tmp_path / "ws"), token="tok")
    return app, app.test_client(), app.extensions["vfl"]


def _wait(ext, jid, timeout=300):
    j = ext["jobs"].wait(jid, timeout)
    assert j["status"] == "done", j["log"][-2000:]
    return j


def _hashes(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def _cli(cwd, argv):
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, "-m", "vfl.cli"] + argv, cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ coverage of the CLI options
def test_every_prepare_data_option_is_in_the_form():
    sub = next(a for a in build_parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    dests = {a.dest for a in sub.choices["prepare-data"]._actions if a.dest != "help"}
    missing = dests - set(D.BY_DEST) - set(D.NOT_IN_FORM)
    assert not missing, f"prepare-data options with no Data preparation field: {missing}"
    for o in D.PREP_OPTIONS:
        assert o["widget"] in ("select", "toggle", "upload", "column", "dir", "columns_table"), o["dest"]


def test_fields_show_and_hide_with_choices():
    keys = lambda v: {f["dest"] for f in D.form(v)["fields"]}            # noqa: E731
    assert "alpha" not in keys({"partitioner": "v2", "partition": "random"})
    assert "alpha" in keys({"partitioner": "v2", "partition": "dirichlet"})
    assert "columns_json" in keys({"partitioner": "v2", "partition": "manual"})
    assert {"csv", "label"} <= keys({"partitioner": "v2", "dataset": "csv"})
    assert "ratios" in keys({"partitioner": "v1"}) and "ratios" not in keys({"partitioner": "v2"})
    assert {"channels", "timesteps", "classes"} <= keys({"partitioner": "v2", "dataset": "synthetic_sensors"})
    assert "text_encoder" in keys({"partitioner": "v2", "dataset": "multimodal_digits"})
    assert D.form({"partitioner": "v2", "dataset": "multimodal_digits", "parties": 5})["values"]["parties"] == 3
    grid = next(f for f in D.form({"partitioner": "v2", "dataset": "digits", "partition": "image_grid",
                                   "parties": 4})["fields"] if f["dest"] == "grid")
    assert [o["value"] for o in grid["options"]] == [None, "1x4", "2x2", "4x1"]


def test_bad_options_are_rejected_by_the_real_parser():
    with pytest.raises(ValueError, match="csv is required"):
        D.build_argv({"partitioner": "v2", "dataset": "csv"}, "x")
    with pytest.raises(ValueError, match="each of the 2 parties"):
        D.build_argv({"partitioner": "v2", "partition": "manual", "columns_json": [["a"]]}, "x", "/tmp")


# ------------------------------------------------------------------ byte-identical UI vs CLI
CASES = {
    "v1-default": {"partitioner": "v1", "dataset": "breast_cancer", "parties": 2},
    "v1-3p-skewed": {"partitioner": "v1", "dataset": "synthetic", "parties": 3, "partition": "importance_skewed",
                     "n": 1000, "features": 10, "overlap": 0.8, "seed": 3},
    "v2-dirichlet-eval": {"partitioner": "v2", "dataset": "breast_cancer", "parties": 3, "partition": "dirichlet",
                          "alpha": 0.5, "overlap": 0.9, "evaluate": True, "id_style": "int"},
    "v2-digits-strips": {"partitioner": "v2", "dataset": "digits", "parties": 2, "partition": "image_strips",
                         "binarize": "odd_even"},
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_ui_and_cli_outputs_are_byte_identical(tmp_path, case):
    app, c, ext = _app(tmp_path)
    r = c.post("/api/prep/generate", json={"values": CASES[case], "name": "ds"}, headers=H)
    assert r.status_code == 200, r.get_json()
    j = _wait(ext, r.get_json()["job"])
    argv = r.get_json()["argv"]
    ui_dir = tmp_path / "ws" / "datasets" / "ds"
    cli_cwd = tmp_path / "cli" / "datasets"
    cli_cwd.mkdir(parents=True)
    _cli(str(cli_cwd), argv)                       # `cd datasets && python -m vfl.cli prepare-data ... --out ds`
    a, b = _hashes(ui_dir), _hashes(cli_cwd / "ds")
    assert a and a == b
    assert j["result"]["n_parties"] == CASES[case]["parties"]
    listed = c.get("/api/prep/datasets").get_json()
    assert [d["name"] for d in listed] == ["ds"] and listed[0]["options"]["dataset"] == CASES[case]["dataset"]


def test_manual_column_assignment_matches_cli(tmp_path):
    app, c, ext = _app(tmp_path)
    cols = c.post("/api/prep/columns", json={"values": {"partitioner": "v2", "partition": "manual"}},
                  headers=H).get_json()["columns"]
    assert len(cols) == 30
    groups = [cols[:10], cols[10:22], cols[22:]]
    v = {"partitioner": "v2", "partition": "manual", "parties": 3, "columns_json": groups}
    r = c.post("/api/prep/generate", json={"values": v, "name": "man"}, headers=H).get_json()
    _wait(ext, r["job"])
    m = json.load(open(tmp_path / "ws" / "datasets" / "man" / "manifest.json"))
    assert [p["columns"] for p in m["parties"]] == groups
    cli_cwd = tmp_path / "cli"
    cli_cwd.mkdir()
    _cli(str(cli_cwd), r["argv"])
    assert _hashes(tmp_path / "ws" / "datasets" / "man") == _hashes(cli_cwd / "man")


# ------------------------------------------------------------------ preview, uploads, certificates
def test_preview_summarises_without_saving(tmp_path):
    app, c, ext = _app(tmp_path)
    v = {"partitioner": "v2", "dataset": "breast_cancer", "parties": 3, "partition": "importance", "overlap": 0.8}
    j = _wait(ext, c.post("/api/prep/preview", json={"values": v}, headers=H).get_json()["job"])
    s = j["result"]
    assert s["n_parties"] == 3 and len(s["parties"]) == 3
    assert 0 < s["intersection_size"] < s["entities_total"]
    assert s["label_balance"] is not None and set(s["label_distribution"]) == {"0", "1"}
    assert abs(sum(p["importance_share"] for p in s["parties"]) - 1) < 0.01
    assert "all_features" in s["reference_auroc"] and "experiment-only" in s["reference_auroc_note"]
    assert c.get("/api/prep/datasets").get_json() == []                       # nothing saved
    assert os.listdir(tmp_path / "ws" / "datasets" / ".preview") == []        # scratch folder cleaned


def test_csv_upload_label_dropdown_and_generate(tmp_path):
    app, c, ext = _app(tmp_path)
    csv = "id,age,income,city,y\n" + "".join(f"{i},{20 + i % 30},{1000 + 7 * i},{'abc'[i % 3]},{i % 2}\n"
                                              for i in range(300))
    r = c.post("/api/uploads", data={"file": (io.BytesIO(csv.encode()), "people.csv")}, headers=H,
               content_type="multipart/form-data")
    assert r.get_json()["file"] == "people.csv"
    assert c.get("/api/uploads/columns?file=people.csv").get_json()["columns"] == ["age", "income", "city", "y"]
    f = c.post("/api/prep/form", json={"values": {"partitioner": "v2", "dataset": "csv", "csv": "people.csv"}},
               headers=H).get_json()
    label = next(x for x in f["fields"] if x["dest"] == "label")
    assert label["widget"] == "column" and "y" in label["options"]
    v = {"partitioner": "v2", "dataset": "csv", "csv": "people.csv", "label": "y", "parties": 2}
    j = _wait(ext, c.post("/api/prep/generate", json={"values": v, "name": "people"}, headers=H).get_json()["job"])
    assert j["result"]["entities_total"] == 300


def test_zip_upload_cannot_escape_uploads(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "x")
    p = tmp_path / "bad.zip"
    p.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="unsafe path"):
        D.safe_extract(str(p), str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("mode", ["static", "pki"])
def test_certificates_for_the_chosen_parties(tmp_path, mode):
    app, c, ext = _app(tmp_path)
    r = c.post("/api/prep/certs", json={"name": f"c-{mode}", "passives": 2, "mode": mode, "hosts": ["127.0.0.1"]},
               headers=H)
    assert r.status_code == 200, r.get_json()
    _wait(ext, r.get_json()["job"])
    sets = {s["name"]: s for s in c.get("/api/prep/certs").get_json()["sets"]}
    assert sets[f"c-{mode}"]["mode"] == mode
    assert {"active", "passive-1", "passive-2"} <= set(sets[f"c-{mode}"]["nodes"])
    assert c.post("/api/prep/certs", json={"name": f"c-{mode}", "passives": 2, "mode": mode},
                  headers=H).status_code == 400                                # exists: no silent overwrite


def test_actions_need_the_token(tmp_path):
    app, c, ext = _app(tmp_path)
    assert c.post("/api/prep/generate", json={"values": {}, "name": "x"}).status_code == 401
    assert c.post("/api/prep/certs", json={"name": "x"}).status_code == 401
    assert c.get("/api/prep/datasets").status_code == 200
