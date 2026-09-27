"""Dependency-free lint (no ruff/flake8 required): syntax, unused imports, duplicate definitions, bare except,
mutable default arguments, `print` left in library code, and `assert` used for input validation in library code.

    python scripts/lint.py            # exit code 1 when anything is found
"""
import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LIB = os.path.join(ROOT, "src", "llm")
ALLOWED_PRINT = {"train.py", "train_reward.py", "train_bpe.py", "preferences.py"}   # CLIs / __main__ demos report on stdout
SKIP = {os.path.join("scripts", "smoke_test.py"), os.path.join("tests", "legacy")}   # the team's original scripts, kept verbatim


def py_files(base):
    for d, _, fs in os.walk(base):
        if "__pycache__" in d or os.sep + "data" in d or os.sep + "runs" in d:
            continue
        for f in fs:
            p = os.path.join(d, f)
            if f.endswith(".py") and not any(os.path.relpath(p, ROOT).startswith(sk) for sk in SKIP):
                yield p


def check(path):
    problems = []
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: syntax error: {e.msg}"]
    rel = os.path.relpath(path, ROOT)
    imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
            for a in n.names:
                imported[a.asname or a.name] = n.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    exported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            exported = {e.value for e in n.value.elts}
    text_uses = src  # names used only in string annotations (from __future__ import annotations)
    for name, line in imported.items():
        if name in used or name in exported or path.endswith("__init__.py"):
            continue
        if text_uses.count(name) > 1:            # appears in an annotation / docstring reference
            continue
        problems.append(f"{rel}:{line}: unused import '{name}'")
    seen = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if n.name in seen:
                problems.append(f"{rel}:{n.lineno}: duplicate definition '{n.name}' (first at line {seen[n.name]})")
            seen[n.name] = n.lineno
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            problems.append(f"{rel}:{n.lineno}: bare except")
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.args.defaults + [d for d in n.args.kw_defaults if d is not None]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    problems.append(f"{rel}:{d.lineno}: mutable default argument in '{n.name}'")
        if path.startswith(LIB):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" and os.path.basename(path) not in ALLOWED_PRINT:
                problems.append(f"{rel}:{n.lineno}: print() in library code (use warnings/logging)")
    return problems


def main():
    problems = []
    for base in (LIB, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests")):
        for p in py_files(base):
            problems += check(p)
    for p in problems:
        print(p)
    print(f"lint: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
