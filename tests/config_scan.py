"""Static scan of the training code for every session-config key it reads (used by test_config_schema.py).

A "read" is `X["k"]`, `X.get("k", ...)` or `X.get(k)` with `k` iterating over string constants, where X is a
session config or a section of one. X is recognised as:
  * a name containing "cfg" or "config" (cfg, self.cfg, session_cfg, self.session_cfg, self.config, ...);
  * an alias assigned from a read, e.g. `d = cfg.get("defense") or {}`  ->  later `d.get("clip")` is defense.clip;
  * a parameter that is documented to receive one section (SECTION_PARAMS below).
Statements are visited in source order inside each function, so re-binding a name drops the alias.
Nested reads are resolved to dotted paths: `cfg["split"]["val"]` -> split.val.
"""
from __future__ import annotations

import ast
import os
import re

ROOT_RE = re.compile(r"(^|_)(cfg|config)$|^cfg|cfg_|session_cfg|default_config")

# functions whose parameter is ONE section of the session config, not the whole config
SECTION_PARAMS = {
    ("defense.py", "apply_defense"): {"cfg": "defense"},
    ("defense.py", "randomized_response"): {"cfg": "defense"},
    ("defense.py", "apply_rep_noise"): {"cfg": "defense"},
    ("defense.py", "describe"): {"cfg": "defense"},
}
# names that look config-ish but hold something else (documented, keep short)
NOT_CONFIG = {
    "cfg_or_frac",          # codec._topk_k: either the topk fraction or the full config (handled: reads topk_fraction)
}


def _const_str(n):
    return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None


class _FnScanner(ast.NodeVisitor):
    def __init__(self, fname, func, reads, sections):
        self.fname, self.func = fname, func
        self.reads = reads
        self.alias: dict[str, tuple] = {}          # name -> path tuple ("" root = ())
        self.sections = sections
        self.loop_consts: dict[str, list] = {}
        for pname, sec in SECTION_PARAMS.get((fname, func), {}).items():
            self.alias[pname] = (sec,)

    # ----- resolution
    def path_of(self, n):
        """Config path the expression evaluates to, or None if it is not (part of) the session config."""
        if isinstance(n, ast.Name):
            if n.id in self.alias:
                return self.alias[n.id]
            if n.id == "cfg_or_frac":
                return ()
            if ROOT_RE.search(n.id) and n.id not in NOT_CONFIG:
                return ()
            return None
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id in ("self", "s", "state", "prev"):
                if ROOT_RE.search(n.attr):
                    return ()
            return None
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
            return self.path_of(n.values[0])
        if isinstance(n, ast.Subscript):
            base = self.path_of(n.value)
            k = _const_str(n.slice)
            if base is not None and k is not None:
                return base + (k,)
            return None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args:
            base = self.path_of(n.func.value)
            k = _const_str(n.args[0])
            if base is not None and k is not None:
                return base + (k,)
        return None

    def record(self, base, key, node):
        path = base + (key,)
        # only section dicts are walked into; anything deeper is the value of one key (e.g. bottom_model map)
        if len(path) > 2 or (len(path) == 2 and path[0] not in self.sections):
            return
        self.reads.setdefault(".".join(path), []).append(f"{self.fname}:{node.lineno}")

    # ----- visitors
    def visit_FunctionDef(self, node):
        if node.name != self.func:          # nested function: scan with a fresh scope
            _FnScanner(self.fname, node.name, self.reads, self.sections).run(node)
            return
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def run(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        self.visit(node.value)
        p = self.path_of(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name):
                if p is not None and p != ():
                    self.alias[t.id] = p
                else:
                    self.alias.pop(t.id, None)
            else:
                self.visit(t)

    def visit_For(self, node):
        consts = None
        if isinstance(node.iter, (ast.Tuple, ast.List)):
            vals = [_const_str(e) for e in node.iter.elts]
            if vals and all(vals):
                consts = vals
        if consts and isinstance(node.target, ast.Name):
            self.loop_consts[node.target.id] = consts
        self.generic_visit(node)
        if consts and isinstance(node.target, ast.Name):
            self.loop_consts.pop(node.target.id, None)

    def _comp(self, node):
        added = []
        for g in node.generators:
            if isinstance(g.iter, (ast.Tuple, ast.List)) and isinstance(g.target, ast.Name):
                vals = [_const_str(e) for e in g.iter.elts]
                if vals and all(vals):
                    self.loop_consts[g.target.id] = vals
                    added.append(g.target.id)
        self.generic_visit(node)
        for a in added:
            self.loop_consts.pop(a, None)

    visit_DictComp = visit_ListComp = visit_SetComp = visit_GeneratorExp = _comp

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            base = self.path_of(node.value)
            k = _const_str(node.slice)
            if base is not None and k is not None:
                self.record(base, k, node)
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and node.args:
            base = self.path_of(f.value)
            if base is not None:
                k = _const_str(node.args[0])
                if k is not None:
                    self.record(base, k, node)
                elif isinstance(node.args[0], ast.Name) and node.args[0].id in self.loop_consts:
                    for k in self.loop_consts[node.args[0].id]:
                        self.record(base, k, node)
        self.generic_visit(node)


def scan_file(path: str, sections, reads: dict | None = None) -> dict:
    reads = {} if reads is None else reads
    fname = os.path.basename(path)
    tree = ast.parse(open(path).read(), path)
    top = _FnScanner(fname, "<module>", reads, sections)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _FnScanner(fname, node.name, reads, sections).run(node)
        elif isinstance(node, ast.ClassDef):
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _FnScanner(fname, b.name, reads, sections).run(b)
        else:
            top.visit(node)
    return reads


def scan(paths, sections) -> dict:
    reads: dict = {}
    for p in paths:
        scan_file(p, sections, reads)
    return reads
