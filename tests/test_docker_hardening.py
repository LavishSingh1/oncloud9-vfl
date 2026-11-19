"""Static checks on the container / packaging setup (no Docker daemon needed).

Red-team finding: containers ran as root on a mutable base tag, dependencies were floor-pinned only,
and the single-host compose put ca.key (plus every node's key, CSV and checkpoints) in one volume
mounted by every party container.
"""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = os.path.join(ROOT, "docker", "Dockerfile")
COMPOSE = os.path.join(ROOT, "docker", "docker-compose.yml")
LOCK = os.path.join(ROOT, "requirements.lock")
NODES = ("coordinator", "active", "passive-1")


def _dockerfile_lines():
    with open(DOCKERFILE) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def _compose():
    with open(COMPOSE) as fh:
        return yaml.safe_load(fh)


def _volume_names(svc: dict) -> set[str]:
    out = set()
    for v in svc.get("volumes", []):
        src = (v.split(":", 1)[0] if isinstance(v, str) else v.get("source", ""))
        out.add(src)
    return out


def test_dockerfile_runs_as_non_root():
    users = [ln.split(None, 1)[1] for ln in _dockerfile_lines() if ln.upper().startswith("USER ")]
    assert users, "Dockerfile has no USER directive"
    final = users[-1].split(":")[0]
    assert final not in ("0", "root"), f"final USER is {users[-1]!r}"


def test_base_image_pinned_by_digest():
    froms = [ln for ln in _dockerfile_lines() if ln.upper().startswith("FROM ")]
    assert froms
    for ln in froms:
        assert re.search(r"@sha256:[0-9a-f]{64}\b", ln), \
            f"base image not pinned by digest: {ln!r} (run scripts/pin_deps.sh)"


def test_dependencies_locked_exactly():
    assert os.path.exists(LOCK), "requirements.lock missing (run scripts/pin_deps.sh)"
    with open(LOCK) as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    assert lines and all("==" in ln for ln in lines), "every line of requirements.lock must be name==version"
    locked = {re.split(r"==", ln)[0].lower().replace("_", "-") for ln in lines}
    with open(os.path.join(ROOT, "requirements.txt")) as fh:
        wanted = {re.split(r"[<>=!~ ;\[]", ln.strip())[0].lower().replace("_", "-")
                  for ln in fh if ln.strip() and not ln.lstrip().startswith("#")}
    assert wanted <= locked, f"not locked: {sorted(wanted - locked)}"
    with open(DOCKERFILE) as fh:
        assert "-c requirements.lock" in fh.read(), "Dockerfile does not install with the lock as constraints"


def test_compose_services_non_root_and_no_caps():
    services = _compose()["services"]
    for name, svc in services.items():
        assert str(svc.get("user", "")).split(":")[0] not in ("", "0", "root"), f"{name} runs as root"
        assert "ALL" in svc.get("cap_drop", []), f"{name} keeps Linux capabilities"


def test_ca_key_volume_only_in_setup():
    services = _compose()["services"]
    assert "pki" in _volume_names(services["setup"])
    for name in NODES:
        assert "pki" not in _volume_names(services[name]), f"{name} mounts the CA volume"
        assert "shared" not in _volume_names(services[name]), f"{name} mounts the old shared volume"


def test_no_volume_shared_between_nodes():
    services = _compose()["services"]
    seen: dict[str, str] = {}
    for name in NODES:
        for vol in _volume_names(services[name]):
            assert vol not in seen, f"volume {vol!r} is shared by {seen[vol]} and {name}"
            seen[vol] = name


def test_setup_copies_only_per_node_material():
    cmd = "\n".join(_compose()["services"]["setup"]["command"])
    copies = [ln.strip() for ln in cmd.splitlines() if ln.strip().startswith("cp ") or " cp " in ln]
    assert copies
    for ln in copies:
        assert "ca.key" not in ln, f"setup copies ca.key to a node: {ln}"
        assert "data_report" not in ln, f"setup hands the full data report to a node: {ln}"
