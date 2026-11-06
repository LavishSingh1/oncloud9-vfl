"""v2 item 25: bench-ui run index and API."""
import json
import os

from bench.ui import BenchUI, create_app


def _bench(dirp, runs):
    os.makedirs(dirp)
    json.dump(runs, open(os.path.join(dirp, "index.json"), "w"))


def test_index_and_grouping(tmp_path):
    res = tmp_path / "results"
    _bench(str(res / "b1"), [
        {"run_id": "s1", "suite": "e1", "seed": 1, "test_auroc": 0.90, "config": {"dataset_tag": "bc", "bottom_model": "mlp"}},
        {"run_id": "s2", "suite": "e1", "seed": 2, "test_auroc": 0.92, "config": {"dataset_tag": "bc", "bottom_model": "mlp"}},
        {"run_id": "s3", "suite": "e1", "seed": 1, "test_auroc": 0.70, "config": {"dataset_tag": "bc", "bottom_model": "cnn"}},
    ])
    ui = BenchUI(str(res), [])
    assert len(ui.runs()) == 3
    g = ui.grouped(["bottom_model"], {})
    mlp = next(x for x in g if x["group"]["bottom_model"] == "mlp")
    assert mlp["n_runs"] == 2 and abs(mlp["auroc_mean"] - 0.91) < 1e-6 and mlp["auroc_std"] > 0
    filt = ui.grouped(["dataset_tag"], {"bottom_model": "cnn"})
    assert filt[0]["n_runs"] == 1


def test_api_endpoints(tmp_path):
    res = tmp_path / "results"
    _bench(str(res / "b1"), [{"run_id": "s1", "suite": "e1", "seed": 1, "test_auroc": 0.9, "config": {"dataset_tag": "bc"}}])
    app = create_app(BenchUI(str(res), []), token=None)
    c = app.test_client()
    assert c.get("/api/runs").get_json()[0]["run_id"] == "s1"
    assert "b1" in c.get("/api/benches").get_json()
    assert c.get("/api/jobs").status_code == 200


def test_launch_needs_token(tmp_path):
    app = create_app(BenchUI(str(tmp_path), []), token="secret")
    c = app.test_client()
    assert c.post("/api/launch", json={"suite": "e1"}).status_code == 401
