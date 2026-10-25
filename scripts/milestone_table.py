"""Milestone table: verdicts + baselines -> results/milestone-table.md + .json.

Every row names its arms and verdict file; attempts that failed name the
failure; problems with no candidate yet are listed as not-attempted with the
reason. Skipped is never reported as passed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "results" / "verdicts"
BASE = json.loads((ROOT / "results" / "baselines.json").read_text())

NOT_ATTEMPTED = {
    "5_Matrix_scalar_multiplication.py": "not in subset",
    "45_Average_Pooling_2D.py": "pilot budget: pooling covered by 42 (max); avg is the same kernel class",
    "50_conv_standard_2D__square_input__square_kernel.py": "rocDNN-backed; direct conv kernel out of pilot budget",
    "87_conv_pointwise_2D.py": "rocDNN-backed; out of pilot budget",
    "89_cumsum.py": "scan kernel class; torch's optimized scan expected to hold; out of pilot budget",
    "95_CrossEntropyLoss.py": "CARD-INFEASIBLE: eager baseline itself OOMs the 24 GiB card (baselines.json)",
    "97_ScaledDotProductAttention.py": "FA-class kernel; out of pilot budget (matmul core is its prerequisite)",
}


def main() -> int:
    rows = []
    for vp in sorted(V.glob("*.json")):
        verdict = json.loads(vp.read_text())
        name = verdict["problem"]
        base = BASE.get(name, {})
        e = base.get("eager")
        c = base.get("compiled")
        rows.append({
            "problem": name,
            "candidate": verdict.get("candidate", ""),
            "milestone": verdict.get("milestone"),
            "parity": (verdict.get("parity") or {}).get("parity"),
            "cand_ms": (verdict.get("cand_time") or {}).get("median_ms"),
            "eager_ms": (e or {}).get("median_ms") if isinstance(e, dict) else None,
            "compiled_ms": (c or {}).get("median_ms") if isinstance(c, dict) else None,
            "speedup_eager": verdict.get("speedup_eager"),
            "speedup_compiled": verdict.get("speedup_compiled"),
            "why": verdict.get("why"),
            "verdict_file": vp.name,
        })

    attempted = {r["problem"] for r in rows}
    untouched = [n for n in NOT_ATTEMPTED if n not in attempted]

    lines = [
        "# KernelBench-HIP pilot — milestone table",
        "",
        "Milestones (CUDA-Agent paper): -1 fail · 1 parity · 2 beats eager · 3 beats torch.compile.",
        "Every number: median ms over 100 L2-thrashed event-timed trials; arms named; verdict file cited.",
        "",
        "| problem | milestone | cand ms | eager ms | compiled ms | vs eager | vs compiled |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        m = r["milestone"]
        lines.append(
            f"| {r['problem'].replace('.py','')} | {m} | {r['cand_ms']} | {r['eager_ms']} | "
            f"{r['compiled_ms']} | {r['speedup_eager']} | {r['speedup_compiled']} |")

    if rows:
        bad = [r for r in rows if r["milestone"] == -1]
        if bad:
            lines += ["", "## Failures (verdicts, not dropped)", ""]
            for r in bad:
                lines.append(f"- {r['problem']}: {str(r['why'])[:220]} ({r['verdict_file']})")
    if untouched:
        lines += ["", "## Not attempted (with reasons — skipped is not passed)", ""]
        for n in untouched:
            lines.append(f"- {n}: {NOT_ATTEMPTED[n]}")

    out_md = ROOT / "results" / "milestone-table.md"
    out_md.write_text("\n".join(lines) + "\n")
    (ROOT / "results" / "milestone-table.json").write_text(
        json.dumps({"rows": rows, "not_attempted": untouched}, indent=1) + "\n")
    print("\n".join(lines[:14]))
    print(f"... full table: {out_md} ({len(rows)} attempts, {len(untouched)} not attempted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
