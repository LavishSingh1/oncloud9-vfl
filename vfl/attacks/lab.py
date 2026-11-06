"""Security-tab launcher for the conformance suite (coordinator --lab-mode only).

Runs the same command as the standalone CLI (`vfl.cli attack <case> --lab-mode ...`) through
vfl/attacks/jobs.AttackJobs, the code path the unified UI's attack page uses too, so all three entry points
give identical verdicts. The target config is built from this coordinator's own state (its listen address, the
registered active party and the parties' hosts), so the Security tab can only aim at this deployment. It
refuses to exist unless the coordinator was started with --lab-mode, and the dashboard only offers it to an
operator. It changes no training behaviour.
"""
from __future__ import annotations

import json
import os

from . import CASES, FAMILIES
from .jobs import AttackJobs


class LabRunner:
    def __init__(self, state, certs_dir, coord_listen, ui_port, ui_token, ui_tls, log_dir: str | None = None):
        self.state = state
        self.certs = certs_dir
        self.coord = coord_listen.replace("0.0.0.0", "127.0.0.1")
        self.url = f"{'https' if ui_tls else 'http'}://127.0.0.1:{ui_port}"
        self.token = ui_token
        self.log_dir = os.path.abspath(log_dir) if log_dir else None
        self.runner = AttackJobs(os.path.join(state.run_dir, "lab"))

    def view(self) -> dict:
        return {"enabled": True, "cases": list(CASES), "families": {k: list(v) for k, v in FAMILIES.items()},
                "certs": self.certs, "jobs": self.runner.view()[:50]}

    def targets(self) -> str:
        """This deployment only: the coordinator itself and the registered active party's address, and only when
        that address is loopback or a private-network IP. v2.2 also added every party's self-reported hostname
        and address, so any registered party could aim the lab attack suite at a third-party host (RT-10)."""
        active = _guess_active(self.state)
        hosts = {"127.0.0.1", "localhost"}
        coord_host = self.coord.rsplit(":", 1)[0].strip("[]")
        if _lab_host_ok(coord_host):
            hosts.add(coord_host)
        if active:
            h = active.rsplit(":", 1)[0].strip("[]")
            if not _lab_host_ok(h):
                raise ValueError(f"the registered active party advertises {active!r}: lab attacks only target "
                                 "loopback or private-network addresses")
            hosts.add(h)
        cfg = {"coordinator": self.coord, "active": active, "hosts": sorted(hosts),
               "log_dirs": [self.log_dir] if self.log_dir else []}
        path = os.path.join(self.state.run_dir, "lab", "targets.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return path

    def launch(self, name: str, args: dict) -> dict:
        if name != "all" and name not in CASES and name not in FAMILIES:
            raise ValueError(f"unknown case {name!r}")
        cfg = self.targets()
        return self.runner.launch(name, coordinator=self.coord, active=_guess_active(self.state), certs=self.certs,
                                  config=cfg, session=args.get("session"), data_dir=args.get("data_dir"),
                                  lab_certs=args.get("lab_certs"), url=self.url, token=self.token,
                                  log_dirs=[self.log_dir] if self.log_dir else None)

    def cancel(self, jid: str) -> dict:
        return self.runner.cancel(jid)


def _lab_host_ok(h: str) -> bool:
    import ipaddress
    if h in ("localhost",):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False                                     # names can resolve anywhere: IP literals only
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _guess_active(state) -> str | None:
    for p in state.parties.values():
        if p["role"] == "active" and p.get("address"):
            return p["address"].replace("0.0.0.0", "127.0.0.1")
    return None
