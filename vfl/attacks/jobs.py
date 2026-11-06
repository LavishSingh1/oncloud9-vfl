"""One way to launch the conformance suite from a UI (v2.1 item 1).

Both UI entry points - the coordinator's Security tab (LabRunner, coordinator --lab-mode) and the unified UI's
Security & Attacks page - call AttackJobs.launch(), which runs exactly the CLI command
`python -m vfl.cli attack <case> --lab-mode ...` as a subprocess. So CLI, Security tab and standalone page
share one code path (vfl/attacks/suite.py) and give identical verdicts for the same attack.
"""
from __future__ import annotations

import json
import os

from . import CASES, FAMILIES


class AttackJobs:
    def __init__(self, out_root: str, manager=None):
        from ..webui.jobs import JobManager
        self.out_root = os.path.abspath(out_root)
        self.jobs = manager or JobManager()

    @staticmethod
    def argv(case: str, coordinator: str, active: str | None, certs: str, out: str, config: str | None = None,
             session: str | None = None, data_dir: str | None = None, lab_certs: str | None = None,
             enroll: str | None = None, url: str | None = None, token: str | None = None,
             log_dirs: list[str] | None = None) -> list[str]:
        if case != "all" and case not in CASES and case not in FAMILIES:
            raise ValueError(f"unknown case {case!r}")
        argv = ["attack", case, "--lab-mode", "--coordinator", coordinator, "--certs", certs, "--out", out]
        for flag, v in (("--active", active), ("--config", config), ("--session", session), ("--data-dir", data_dir),
                        ("--lab-certs", lab_certs), ("--enroll", enroll), ("--url", url), ("--token", token)):
            if v:
                argv += [flag, str(v)]
        if log_dirs:
            argv += ["--log-dirs", *log_dirs]
        return argv

    def launch(self, case: str, label: str | None = None, **kw) -> dict:
        from ..webui.jobs import cli_cmd
        jid_dir = os.path.join(self.out_root, f"{case}-{os.urandom(3).hex()}")
        argv = self.argv(case, out=jid_dir, **kw)

        def done(job):
            p = os.path.join(jid_dir, "report.json")
            if not os.path.exists(p):
                return None
            rep = json.load(open(p))
            return {"summary": rep["summary"], "verdicts": rep["verdicts"], "rows": rep["rows"], "out": jid_dir}

        j = self.jobs.submit("attack", cli_cmd(argv), label=label or case, meta={"case": case, "out": jid_dir},
                             on_done=done, timeout=600, ok_codes=(0, 1))     # 1 = ran, not all as expected
        return {"job": j.id, "name": case, "out": jid_dir}

    def view(self) -> list[dict]:
        out = []
        for j in self.jobs.list("attack", tail=40):
            res = j.get("result") or {}
            out.append({"id": j["id"], "name": j["meta"].get("case"), "t": j["queued"], "status": j["status"],
                        "summary": res.get("summary"), "verdicts": res.get("verdicts"), "rows": res.get("rows"),
                        "out": j["meta"].get("out"), "log": j["log"], "returncode": j["returncode"]})
        return out

    def cancel(self, jid: str) -> dict:
        return self.jobs.cancel(jid)
