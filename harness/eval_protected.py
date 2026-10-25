"""Protected eval — the one place verdicts are computed.

Protocol, ported verbatim from KernelBench (eval.py, timing.py) and the
CUDA-Agent paper's robust milestone reward:

  correctness : 5 trials, seeds 0..4; per trial seed get_inputs() and Model
                construction; fp32; pass = shape match AND
                torch.allclose(ref, cand, atol=1e-2, rtol=1e-2) on ALL 5.
  timing      : cuda events on the default stream; warmup 5, trials 100,
                discard first 1; L2 thrashed (256 MB fill) before every trial;
                stats = median (verdict), mean, std, min.
  milestones  : -1 compile/runtime failure or anti-hack trip
                 1 parity 5/5
                 2 parity and median strictly beats eager reference
                 3 parity and median strictly beats torch.compile(inductor)
                   reference (tier null -> capped at 2, disclosed)
  anti-hack   : static (antihack.py) + runtime R1 input-immutability and
                R2 A,B,A recompute check.

This file must never import or be imported by candidate code. Every verdict
record embeds this file's sha256 (PROTECTED_SHA) so a result names the exact
eval that produced it.
"""
from __future__ import annotations
import hashlib
import json
import math
import signal
import statistics
import sys
import time
from pathlib import Path

import torch

# Fedora keeps all of ROCm at /usr; cpp_extension's search prefers /opt/rocm.
# PYTORCH_ROCM_ARCH pins device compilation to this card (13-arch default is slow).
import os
os.environ.setdefault("ROCM_PATH", "/usr")
os.environ.setdefault("HIP_PATH", "/usr")
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1100")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import antihack  # noqa: E402
import problems  # noqa: E402

PROTECTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CORRECTNESS_SEEDS = [0, 1, 2, 3, 4]
TOL = 1e-2
WARMUP = 5
TRIALS = 100
DISCARD_FIRST = 1
STALL_RATIO = 3.0


class PhaseTimeout(Exception):
    pass


def _alarm(sig, frame):
    raise PhaseTimeout()


def _with_timeout(seconds):
    """Arm SIGALRM; a wedged kernel (no completion, GPU pinned at 100%) must
    become an honest timeout verdict instead of a hung window (measured:
    3_Batched v1 wedged 19+ min, killed by hand 2026-10-26)."""
    def deco(fn):
        def wrapper(*a, **k):
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(seconds)
            try:
                return fn(*a, **k)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        return wrapper
    return deco


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_l2(device) -> None:
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


def time_fn(fn, args, device) -> dict:
    """KernelBench cuda_event protocol: warmup, discard-first, L2 thrash per trial."""
    for _ in range(WARMUP):
        fn(*args)
        torch.cuda.synchronize(device=device)
    torch.cuda.empty_cache()

    times = []
    for trial in range(TRIALS + DISCARD_FIRST):
        torch.cuda.synchronize(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        clear_l2(device)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize(device=device)
        if trial >= DISCARD_FIRST:
            times.append(start.elapsed_time(end))

    med = statistics.median(times)
    stats = {
        "median_ms": round(med, 6),
        "mean_ms": round(statistics.fmean(times), 6),
        "std_ms": round(statistics.pstdev(times), 6),
        "min_ms": round(min(times), 6),
        "n": len(times),
        "stall_flag": med > STALL_RATIO * min(times),
    }
    return stats


def make_inputs(problem, device, seed: int):
    set_seed(seed)
    inputs = problem["get_inputs"]()
    set_seed(seed)
    return [x.to(device=device, dtype=torch.float32) if isinstance(x, torch.Tensor) else x
            for x in inputs]


def evaluate(problem_name: str, candidate_path: str, out_path: str) -> dict:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    Model_ref, get_inputs, get_init_inputs, ref_src = problems.load_problem(problem_name)
    init_inputs = get_init_inputs()

    cand_src = Path(candidate_path).read_text()

    verdict = {
        "problem": problem_name,
        "candidate": str(candidate_path),
        "protected_eval_sha": PROTECTED_SHA,
        "tol": TOL, "warmup": WARMUP, "trials": TRIALS,
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
    }

    # -- static anti-hack ---------------------------------------------------
    entry = _candidate_entry(cand_src)
    s = antihack.static_check(cand_src, entry)
    verdict["static_check"] = s
    if not s["ok"]:
        verdict["milestone"] = -1
        verdict["why"] = "static anti-hack violation"
        _write(out_path, verdict)
        return verdict

    # -- build candidate ----------------------------------------------------
    t0 = time.time()
    try:
        cand_model_cls, entry = _with_timeout(900)(build_candidate)(cand_src)
        verdict["compile_s"] = round(time.time() - t0, 1)
    except PhaseTimeout:
        verdict["milestone"] = -1
        verdict["why"] = "compile timeout (900 s)"
        verdict["compile_s"] = round(time.time() - t0, 1)
        _write(out_path, verdict)
        return verdict
    except Exception as e:
        verdict["milestone"] = -1
        verdict["why"] = f"compile failed: {type(e).__name__}: {e}"
        verdict["compile_s"] = round(time.time() - t0, 1)
        _write(out_path, verdict)
        return verdict

    # -- correctness (R2 recompute check folded in) -------------------------
    try:
        parity = _with_timeout(420)(_parity_with_recompute)(
            Model_ref, cand_model_cls, get_inputs, init_inputs, device)
    except PhaseTimeout:
        verdict.update({"milestone": -1, "why": "correctness phase timeout (420 s)"})
        _write(out_path, verdict)
        return verdict
    except torch.OutOfMemoryError as e:
        verdict.update({"milestone": -1, "why": f"parity OOM: {e}"[:400]})
        _write(out_path, verdict)
        return verdict
    verdict["parity"] = parity
    if not parity.get("parity"):
        verdict["milestone"] = -1
        verdict["why"] = "parity failed"
        _write(out_path, verdict)
        return verdict

    # -- timing: candidate, eager ref, compiled ref -------------------------
    cand_model = cand_model_cls(*init_inputs).to(device=device, dtype=torch.float32)
    ref_model = Model_ref(*init_inputs).to(device=device, dtype=torch.float32)

    inputs = [x.to(device=device, dtype=torch.float32) if isinstance(x, torch.Tensor) else x
              for x in get_inputs()]

    def timed(model, is_compiled=False):
        m = model
        if is_compiled:
            torch._dynamo.reset()
            m = torch.compile(model, backend="inductor", mode="default")
        with torch.no_grad():
            stats = time_fn(m, tuple(inputs), device)
        if stats["stall_flag"]:  # retry-until-clean, keep both (strata pattern)
            with torch.no_grad():
                stats["retry"] = time_fn(m, tuple(inputs), device)
            if not stats["retry"]["stall_flag"]:
                stats = stats["retry"]
        return stats

    try:
        verdict["cand_time"] = _with_timeout(600)(timed)(cand_model, False)
    except PhaseTimeout:
        verdict.update({"milestone": -1, "why": "candidate timing timeout (600 s) — wedged kernel?"})
        _write(out_path, verdict)
        return verdict
    except torch.OutOfMemoryError as e:
        verdict.update({"milestone": -1, "why": f"candidate timing OOM: {e}"[:400]})
        _write(out_path, verdict)
        return verdict
    try:
        verdict["eager_time"] = _with_timeout(600)(timed)(ref_model, False)
    except (torch.OutOfMemoryError, PhaseTimeout) as e:
        verdict["eager_time"] = None
        verdict["eager_oom"] = str(e)[:200]
    try:
        verdict["compiled_time"] = _with_timeout(900)(timed)(ref_model, True)
    except Exception as e:
        verdict["compiled_time"] = None
        verdict["compiled_error"] = repr(e)[:500]

    # -- milestones ---------------------------------------------------------
    c = verdict["cand_time"]["median_ms"]
    e = verdict["eager_time"]["median_ms"]
    verdict["speedup_eager"] = round(e / c, 4) if c > 0 else None
    if verdict["compiled_time"]:
        k = verdict["compiled_time"]["median_ms"]
        verdict["speedup_compiled"] = round(k / c, 4) if c > 0 else None
    milestone = 1
    if verdict["speedup_eager"] and verdict["speedup_eager"] > 1.0:
        milestone = 2
    if verdict.get("speedup_compiled") and verdict["speedup_compiled"] > 1.0:
        milestone = 3
    verdict["milestone"] = milestone

    _write(out_path, verdict)
    return verdict


def _candidate_entry(cand_src: str) -> str:
    """The ENTRY constant convention: candidates declare ENTRY = "<fn name>"."""
    import re
    m = re.search(r'^ENTRY\s*=\s*["\']([\w]+)["\']', cand_src, re.M)
    return m.group(1) if m else ""


def _patched_load_inline():
    """load_inline with the Fedora include-chain fix.

    torch's cpp_extension passes `-isystem /usr/include`, which breaks GCC/clang
    `#include_next` (the deduped real /usr/include is skipped, so <stdlib.h> /
    <math.h> vanish). Verified minimal repro 2026-10-25. Filtering it out of
    include_paths() restores the default chain. Recorded in docs/gfx1100-notes.md.
    """
    import torch.utils.cpp_extension as ce
    if not getattr(ce, "_kbhip_patched", False):
        orig = ce.include_paths

        def filtered(*a, **k):
            return [p for p in orig(*a, **k) if os.path.realpath(p) != "/usr/include"]

        ce.include_paths = filtered
        ce._kbhip_patched = True
    return ce.load_inline


def build_candidate(cand_src: str):
    """Exec candidate with a restricted global set; returns (ModelCls, entry)."""
    import torch.nn as nn

    g = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": nn,
        "load_inline": _patched_load_inline(),
        "math": math,
    }
    exec(compile(cand_src, "<candidate>", "exec"), g)
    if "Model" not in g:
        raise RuntimeError("candidate defines no Model")
    return g["Model"], g.get("ENTRY", "")


def _chunked_allclose(a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 26) -> bool:
    """allclose without the full-size abs temp. 1<<28 (1 GiB) chunks OOM'd the
    7.5 GiB three-tensor parity phase (RMSNorm/Frobenius, 2026-10-26)."""
    fa = a.reshape(-1)
    fb = b.reshape(-1)
    for i in range(0, fa.numel(), chunk_elems):
        ca = fa[i:i + chunk_elems]
        cb = fb[i:i + chunk_elems]
        if not torch.allclose(ca, cb, atol=TOL, rtol=TOL):
            return False
    return True


def _digest(t: torch.Tensor) -> list:
    """Fingerprint for the R2 recompute check: (shape, numel, min, max).

    No whole-tensor sum: sums of 10^8 near-zero-mean values jitter at ~1e-2
    absolute between runs from atomic order alone (measured: LayerNorm R2
    false-fail 2026-10-26). A cached/stashed output is BITWISE identical, so
    min/max at 1e-4-of-range tolerance still detects it and any input-state
    dependence, without false-failing legitimate fp32 nondeterminism."""
    lo = t.min().item()
    hi = t.max().item()
    return [tuple(t.shape), t.numel(), f"{lo:.6e}", f"{hi:.6e}"]


def _digest_close(d1: list, d2: list) -> bool:
    if d1[:2] != d2[:2]:
        return False
    lo1, hi1 = float(d1[2]), float(d1[3])
    lo2, hi2 = float(d2[2]), float(d2[3])
    tol = 1e-4 * max(1e-30, hi1 - lo1, hi2 - lo2)
    return abs(lo1 - lo2) <= tol and abs(hi1 - hi2) <= tol


def _parity_with_recompute(ref_cls, cand_cls, get_inputs, init_inputs, device) -> dict:
    """Correctness 5/5 + R1 immutability + R2 recompute digest.

    Sliced path for oversized problems: when input + ref-out + cand-out exceed
    the card (RMSNorm/Frobenius-class: 3 x 7.5 GiB + the reference's own x**2
    temp), parity runs in leading-dim slices with streaming min/max. Same
    seeds, same allclose, bounded memory. Skipped is never passed: a slice
    failure fails the trial."""
    ref_model = ref_cls(*init_inputs).to(device=device, dtype=torch.float32)
    cand_model = cand_cls(*init_inputs).to(device=device, dtype=torch.float32)

    def fresh_inputs(seed):
        set_seed(seed)
        return [x.to(device=device, dtype=torch.float32) if isinstance(x, torch.Tensor) else x
                for x in get_inputs()]

    # decide on size from shapes only, then route: oversized single-input ->
    # sliced parity. Multi-input problems keep whole-tensor (slice routing
    # would feed the reference one tensor; measured: 4_Matvec TypeError).
    probe = fresh_inputs(0)
    big = (len(probe) == 1 and isinstance(probe[0], torch.Tensor)
           and probe[0].nbytes > 5e9)
    del probe
    torch.cuda.empty_cache()

    if big:
        return _parity_sliced(ref_model, cand_model, get_inputs, fresh_inputs, device)

    trials, passed = [], 0
    digests = {}
    for seed in (0, 1, 2, 3, 4):
        inputs = fresh_inputs(seed)
        pre = [_hash_tensor(x) for x in inputs if isinstance(x, torch.Tensor)]
        with torch.no_grad():
            out_ref = ref_model(*inputs)
            torch.cuda.synchronize(device=device)
            out_cand = cand_model(*inputs)
            torch.cuda.synchronize(device=device)
        post = [_hash_tensor(x) for x in inputs if isinstance(x, torch.Tensor)]
        row = {"seed": seed}
        try:
            if pre != post:
                row.update({"pass": False, "why": "R1 input mutated"})
                trials.append(row)
                continue
            if out_ref.shape != out_cand.shape:
                row.update({"pass": False,
                            "why": f"shape {tuple(out_ref.shape)} vs {tuple(out_cand.shape)}"})
                trials.append(row)
                continue
            if not _chunked_allclose(out_ref, out_cand):
                max_diff = (out_ref - out_cand).abs().max().item()
                row.update({"pass": False, "max_diff": round(max_diff, 6)})
                trials.append(row)
                continue
            digests[seed] = _digest(out_cand)
            row["pass"] = True
            trials.append(row)
            passed += 1
        finally:
            del out_ref, out_cand, inputs
            torch.cuda.empty_cache()

    # R2: re-run seed 0 after other inputs flowed; digest must reproduce.
    inputs0 = fresh_inputs(0)
    with torch.no_grad():
        out0 = cand_model(*inputs0)
        torch.cuda.synchronize(device=device)
    r2 = _digest_close(digests[0], _digest(out0))
    del out0, inputs0
    torch.cuda.empty_cache()
    return {"parity": passed == 5 and r2, "trials_passed": passed,
            "recompute_check": r2, "trials": trials,
            "mode": "whole-tensor"}


def _parity_sliced(ref_model, cand_model, get_inputs, fresh_inputs, device) -> dict:
    """Row-sliced parity: per slice, ref forward + cand forward + allclose;
    streaming (min, max) digest per seed. Bounded to ~2 slices of memory.

    The device input is allocated ONCE and refilled via copy_ per seed: a
    fresh .to(device) per seed churns a 7.5 GiB block against a fragmented
    heap and OOMs (measured 2026-10-26, verdicts 36_RMSNorm v4-v6)."""
    trials, passed = [], 0
    digests = {}

    def load_device_input(seed):
        """Fresh seeded CPU inputs -> the persistent device buffer."""
        set_seed(seed)
        cpu_inputs = get_inputs()
        set_seed(seed)
        devs = []
        for x in cpu_inputs:
            if isinstance(x, torch.Tensor):
                if not hasattr(load_device_input, "_bufs"):
                    load_device_input._bufs = {}
                key = tuple(x.shape)
                buf = load_device_input._bufs.get(key)
                if buf is None:
                    buf = torch.empty(x.shape, device=device, dtype=torch.float32)
                    load_device_input._bufs[key] = buf
                buf.copy_(x)
                devs.append(buf)
            else:
                devs.append(x)
        del cpu_inputs
        return devs

    for seed in (0, 1, 2, 3, 4):
        inputs = load_device_input(seed)
        x = inputs[0]
        pre = [_hash_tensor(x) for x in inputs if isinstance(x, torch.Tensor)]
        n0 = x.size(0)
        slice_rows = max(1, int(4e9 // (x.nbytes / n0)))
        lo, hi = float("inf"), float("-inf")
        ok = True
        why = None
        with torch.no_grad():
            for s0 in range(0, n0, slice_rows):
                xs = x[s0:s0 + slice_rows]
                try:
                    out_ref = ref_model(xs)
                    torch.cuda.synchronize(device=device)
                    out_cand = cand_model(xs)
                    torch.cuda.synchronize(device=device)
                except torch.OutOfMemoryError as e:
                    raise RuntimeError(
                        f"sliced parity OOM at slice {s0} (slice_rows={slice_rows}, "
                        f"slice shape {tuple(xs.shape)}, bytes {xs.nbytes}): {e}") from e
                if out_ref.shape != out_cand.shape:
                    ok, why = False, f"slice shape {tuple(out_ref.shape)} vs {tuple(out_cand.shape)}"
                    break
                if not _chunked_allclose(out_ref, out_cand):
                    ok, why = False, f"slice @{s0}: allclose failed"
                    break
                lo = min(lo, out_cand.min().item())
                hi = max(hi, out_cand.max().item())
                del out_ref, out_cand, xs
        post = [_hash_tensor(x) for x in inputs if isinstance(x, torch.Tensor)]
        row = {"seed": seed}
        if pre != post:
            ok, why = False, "R1 input mutated"
        if ok:
            digests[seed] = (lo, hi)
            row["pass"] = True
            trials.append(row)
            passed += 1
        else:
            row.update({"pass": False, "why": why})
            trials.append(row)
        del inputs, x
        torch.cuda.empty_cache()

    inputs0 = load_device_input(0)
    x0 = inputs0[0]
    n0 = x0.size(0)
    slice_rows = max(1, int(4e9 // (x0.nbytes / n0)))
    lo, hi = float("inf"), float("-inf")
    with torch.no_grad():
        for s0 in range(0, n0, slice_rows):
            out0 = cand_model(x0[s0:s0 + slice_rows])
            torch.cuda.synchronize(device=device)
            lo = min(lo, out0.min().item())
            hi = max(hi, out0.max().item())
            del out0
    del inputs0, x0
    torch.cuda.empty_cache()
    d0 = digests.get(0)
    r2 = d0 is not None and abs(d0[0] - lo) <= 1e-4 * max(1e-30, d0[1] - d0[0], hi - lo) \
        and abs(d0[1] - hi) <= 1e-4 * max(1e-30, d0[1] - d0[0], hi - lo)
    return {"parity": passed == 5 and r2, "trials_passed": passed,
            "recompute_check": r2, "trials": trials, "mode": "sliced"}


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _write(out_path, verdict):
    Path(out_path).write_text(json.dumps(verdict, indent=1, default=str) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    v = evaluate(a.problem, a.candidate, a.out)
    print(json.dumps({k: v.get(k) for k in
                      ("problem", "milestone", "speedup_eager", "speedup_compiled",
                       "why", "parity")}, default=str))
