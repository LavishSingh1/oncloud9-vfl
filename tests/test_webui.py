"""v2.1 item 6: one entry point (`vfl.cli ui`, port 8090) with every CLI capability reachable from the UI."""
import json
import os
import re
import socket
import threading
import types

import pytest

pytest.importorskip("flask")
from vfl.cli import build_parser  # noqa: E402
from vfl.config import load_session_config  # noqa: E402
from vfl.webui import CLI_ROUTES, PAGES, create_app, serve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = {"Authorization": "Bearer tok", "X-VFL-Request": "1"}       # what the UI sends (CSRF header, v2.3)


def _subcommands():
    sub = next(a for a in build_parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    return sub.choices


@pytest.fixture
def app(tmp_path):
    a = create_app(str(tmp_path / "ws"), token="tok")
    yield a
    a.extensions["vfl"]["deploys"].stop_all()


def test_every_cli_subcommand_has_a_ui_route(app):
    cmds = set(_subcommands())
    assert cmds == set(CLI_ROUTES), f"no UI route for {cmds - set(CLI_ROUTES)}; stale {set(CLI_ROUTES) - cmds}"
    rules = {(r.rule, m) for r in app.url_map.iter_rules() for m in r.methods}
    for cmd, (page, method, route) in CLI_ROUTES.items():
        assert page in PAGES, cmd
        assert (route, method) in rules, f"{cmd}: {method} {route} is not served"


def test_shell_navigates_every_section(app):
    html = app.test_client().get("/").get_data(as_text=True)
    for label in ("Data preparation", "Deployment", "Training / Sessions", "Security &amp; Attacks",
                  "Tests &amp; Benchmarks", "Analytics", "Reports"):
        assert label in html, label
    assert not re.search(r"<script[^>]+src=\"https?://", html) and "cdn" not in html.lower()   # vendored only
    js = open(os.path.join(ROOT, "vfl", "webui", "static", "app.js")).read()
    assert "dashboard" in js and "target=\"_blank\"" in js        # links out to node dashboards, not a copy


def test_every_action_needs_the_token(app):
    c = app.test_client()
    for r in app.url_map.iter_rules():
        for m in r.methods & {"POST", "DELETE", "PUT"}:
            url = re.sub(r"<[^>]+>", "x", r.rule)
            resp = c.open(url, method=m, json={})
            assert resp.status_code == 401, f"{m} {r.rule} answered {resp.status_code} without the token"


def test_defaults_and_bind_safety(tmp_path):
    p = _subcommands()["ui"]
    a = p.parse_args([])
    assert a.host == "127.0.0.1" and a.port == 8090
    assert _subcommands()["attack-ui"].parse_args([]).lab_mode is True
    with pytest.raises(SystemExit, match="without --token"):
        serve(str(tmp_path), host="0.0.0.0", port=0, token=None)


def _stub_coordinator():
    from vfl.dashboard import create_app as dash
    from werkzeug.serving import make_server
    seen = []
    be = types.SimpleNamespace(party_id="coord", lab=None, lab_mode=False, registry=None,
                               default_config=load_session_config(os.path.join(ROOT, "configs", "session.yaml")))
    be.start_session = lambda body: (seen.append(body), {"session_id": "s1", "state": "STARTING"})[1]
    be.ui_state = lambda: {"role": "coordinator", "parties": [], "sessions": []}
    be.session_detail = lambda sid: {"session_id": sid, "config": be.default_config}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, dash(be, "coordinator", token="tok"), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}", seen, srv


def test_training_page_drives_a_coordinator(tmp_path):
    url, seen, srv = _stub_coordinator()
    try:
        app = create_app(str(tmp_path / "ws"), token="tok", coordinator_url=url, coordinator_token="tok")
        c = app.test_client()
        assert [x["url"] for x in c.get("/api/train/coordinators").get_json()] == [url]
        sch = c.get("/api/train/schema").get_json()
        assert sch["current"]["epochs"] == 20 and not sch.get("offline")       # the coordinator's own defaults
        form = {**sch["current"], "epochs": 3, "lr": 0.003}
        r = c.post("/api/train/start", json={"form": form}, headers=H)
        assert r.status_code == 200, r.get_json()
        assert seen[-1]["config"]["epochs"] == 3 and seen[-1]["config"]["lr"] == 0.003
        bad = c.post("/api/train/start", json={"form": {**form, "epochs": 0}}, headers=H)
        assert bad.status_code == 400 and "epochs" in bad.get_json()["error"]
        assert c.get("/api/train/sessions/s1").get_json()["config"]["epochs"] == 20
    finally:
        srv.shutdown()


def test_training_schema_works_offline(tmp_path):
    c = create_app(str(tmp_path / "ws"), token="tok").test_client()
    sch = c.get("/api/train/schema").get_json()
    assert sch["offline"] is True and {p["key"] for p in sch["params"]} >= {"epochs", "cnn_kernel", "defense.clip"}
    r = c.post("/api/train/start", json={"form": {}}, headers=H)
    assert r.status_code == 400 and "no coordinator" in r.get_json()["error"]


def test_bench_ui_alias_still_serves_the_unified_ui():
    a = _subcommands()["bench-ui"].parse_args([])
    assert a.port == 8090 and a.host == "127.0.0.1"
    assert json.dumps(CLI_ROUTES["bench-ui"]) == json.dumps(["tests", "GET", "/"])
