"""Evaluate a batch of candidates, one SUBPROCESS per candidate.

Why subprocesses: a wedged HIP kernel pins the GPU at 100% and blocks the host
inside a C-level sync where SIGALRM cannot fire (measured twice, 2026-10-26).
Only killing the process destroys the HIP context and frees the card. The
driver spawns `eval_protected.py` per candidate with a wall-clock timeout,
SIGKILLs the process group on timeout, and records an honest timeout verdict.

Usage: run_batch.py PROBLEM CANDIDATE OUT [triple...]
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "harness" / "eval_protected.py"
TIMEOUT_S = 900


def run_one(problem: str, cand: str, out: str) -> None:
    cmd = [str(ROOT / ".venv" / "bin" / "python"), str(EVAL),
           "--problem", problem, "--candidate", cand, "--out", out]
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        proc.wait(timeout=TIMEOUT_S)
        if proc.returncode != 0:
            v = json.loads(Path(out).read_text()) if Path(out).exists() else {}
            v.setdefault("milestone", -1)
            v.setdefault("why", f"eval exited {proc.returncode}")
            Path(out).write_text(json.dumps(v, indent=1, default=str) + "\n")
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        Path(out).write_text(json.dumps({
            "problem": problem, "candidate": cand, "milestone": -1,
            "why": f"eval killed at {TIMEOUT_S}s wall (wedged kernel suspected; HIP context destroyed)",
        }, indent=1) + "\n")
        print(f"TIMEOUT {problem}: killed after {TIMEOUT_S}s", flush=True)
        return
    v = json.loads(Path(out).read_text())
    print(f"DONE {problem}: milestone={v['milestone']} "
          f"se={v.get('speedup_eager')} sc={v.get('speedup_compiled')} "
          f"why={str(v.get('why'))[:80]}", flush=True)


def main() -> int:
    args = sys.argv[1:]
    if len(args) % 3 != 0:
        print("usage: run_batch.py PROBLEM CANDIDATE OUT [triple...]")
        return 2
    for i in range(0, len(args), 3):
        run_one(args[i], args[i + 1], args[i + 2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
