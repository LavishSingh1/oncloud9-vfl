"""Validation of identifiers and strings that cross a trust boundary (v2.3 red-team hardening).

Everything here is torch- and grpc-free so the coordinator, the parties and the UIs can share it.

* party and session ids end up in file paths (logs/<party>/<session>/, checkpoints/..., runs/<session>/), so
  they must be single path components: no separators, no '..', no absolute paths, no control characters;
* strings a party sends (hostname, status, progress, report messages) are shown in dashboards and written to
  logs, so they are cut to a bounded length and kept on one line (no forged log lines);
* links a party supplies (its dashboard URL) are rendered as <a href>, so only http(s) URLs are accepted.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_TEXT = 500
_CTRL = re.compile(r"[\x00-\x1f\x7f  ]")
LAB_ONLY_PREFIXES = ("rogue",)         # identities minted for the attack simulator (lab certificate sets only)


def _check(kind: str, value) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value) or ".." in value:
        raise ValueError(f"invalid {kind} {value!r}: use 1-64 letters, digits, '.', '_' or '-' "
                         "(not starting with '.' or '-', no '..')")
    return value


def check_party_id(pid) -> str:
    return _check("party id", pid)


def check_session_id(sid) -> str:
    return _check("session id", sid)


def is_lab_identity(pid: str) -> bool:
    return any(pid == p or pid.startswith(p + "-") for p in LAB_ONLY_PREFIXES)


def one_line(value, limit: int = MAX_TEXT) -> str:
    """Printable, single-line, bounded: safe to log and to show (HTML escaping is still the UI's job)."""
    if value is None:
        return ""
    return _CTRL.sub(" ", str(value))[:limit]


def check_ui_url(url) -> str:
    """'' or an absolute http(s) URL without credentials. Anything else (javascript:, data:, //host, ...) is
    refused because the coordinator dashboard renders it as a link."""
    if not url:
        return ""
    if not isinstance(url, str) or len(url) > 256 or _CTRL.search(url):
        raise ValueError("ui_url must be a short single-line http(s) URL")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"ui_url must be an http(s) URL, got {url[:60]!r}")
    return url
