"""Anti-reward-hacking checks, ported from CUDA-Agent/KernelBench discipline.

Static (AST on the candidate source):
  S1 forward must call the declared HIP extension entry point (no pure-torch fallback).
  S2 no torch.compile / cudagraph / stream construction / event self-timing in the candidate.
  S3 no import of this harness (the eval must stay invisible to the candidate).

Runtime (eval_protected.py calls these):
  R1 input immutability — inputs hashed before/after candidate forward.
  R2 recompute check — A,B,A input pattern: outputs for the second A must match
     the first A (catches compute-once-and-stash caching).
  R3 no dependence on reference outputs — candidate module dict never contains
     names from the reference evaluation (enforced by construction: separate
     module dicts + candidate exec'd with a restricted global set).
"""
from __future__ import annotations
import ast

BANNED_CALLS = {
    # S2: timing/stream/graph tricks
    "compile", "Stream", "Event", "graph", "CUDAGraph", "make_graphed_callables",
    "set_stream", "current_stream", "synchronize",
}
BANNED_MODULES = {"kernelbench", "harness", "eval_protected", "runner", "antihack", "problems"}


def static_check(src: str, entry_symbol: str) -> dict:
    """Return {ok: bool, violations: [..]} for the candidate source."""
    violations = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"ok": False, "violations": [f"syntax-error: {e}"]}

    forward_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    forward_fn = item
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in names:
                if any(m == b or m.startswith(b + ".") for b in BANNED_MODULES):
                    violations.append(f"S3 imports harness module: {m}")

    if forward_fn is None:
        violations.append("S1 no Model.forward found")
        return {"ok": False, "violations": violations}

    calls_in_forward = []
    for node in ast.walk(forward_fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            calls_in_forward.append(name)
            if name in BANNED_CALLS:
                violations.append(f"S2 banned call in forward: {name}()")

    if not any(entry_symbol in c for c in calls_in_forward):
        violations.append(
            f"S1 forward never calls the HIP entry '{entry_symbol}' — pure-torch fallback"
        )

    return {"ok": not violations, "violations": violations}
