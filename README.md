# KernelBench-HIP

An independent port of the [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
evaluation to **HIP / ROCm**, run on an **AMD Radeon RX 7900 XTX (gfx1100)**,
scored with the milestone-reward environment from
[CUDA-Agent (arXiv 2602.24286)](https://arxiv.org/html/2602.24286v1):
**−1 fail → +1 parity → +2 beats eager → +3 beats torch.compile**,
with a training-free agent loop drafting the HIP kernels (no RL, no training
anywhere in this work).

To our knowledge (searched 2026-10-25/26) this is the first public
KernelBench-protocol evaluation on ROCm hardware. AMD's
[AIG-Datasets](https://huggingface.co/datasets/amd/AIG-Datasets) ships HIP
kernel-generation *training* corpora (including a HIP-ported CUDA-Agent-Ops-6K);
this repo complements that with the missing *evaluation* side.

## Results (24 of 30 Level-1 problems attempted)

Full per-problem numbers with arms and verdict files:
[`results/milestone-table.md`](results/milestone-table.md).

| tier | problems | highlights |
| --- | --- | --- |
| M3 (beats eager **and** compiled) | 7 | MinGPT-GELU 8.6×, Swish 2.6×, matvec K=1M 2.0×, huge-K split-K GEMM 1.03× |
| M2 (beats eager) | 4 | MSELoss 2.7×, LayerNorm 2.1×, RMSNorm 1.3×, MaxPool 1.01× |
| M1 (parity) | 11 | torch wins reductions/softmax; rocBLAS square GEMM 3× faster than our tiled kernel |
| open | 1 | batched GEMM candidate wedges the GPU (recorded, not hidden) |

**The one finding that generalizes:** the hand kernels win where the baseline
runs *multi-pass chains with materialized temporaries* (eager's 5-op GELU chain,
sigmoid+mul, 3-pass norms) or where the library is weak on a shape (rocBLAS at
78 GB/s on a huge-K skinny GEMM, 350 GB/s on an 8.6 GiB matvec). Where the
baseline is already a single fused pass at the HBM roof (ReLU-class, GELU,
square SGEMM), hand kernels tie or lose. Compute the traffic floor first; it
predicts the outcome.

## Honest limits

- Level 1 only, 30 of 100 problems (family-representative subset), one consumer
  card, stock clocks, desktop co-tenancy excluded per window (stop/restore of
  the resident engine) but not a lab environment.
- **Single-shot, training-free, n=1 per problem.** This is an evaluation
  substrate + baseline dataset, NOT a capability or SOTA claim. The paper's RL
  numbers are not comparable to ours and we make no such comparison.
- Protocol fidelity: KernelBench's 5-seeded-trials fp32 correctness
  (atol=rtol=1e-2) and event timing (5 warmup / 100 trials / discard-first /
  256 MiB L2 thrash) are ported as-is; medians are verdicts; stall-flag +
  retry-until-clean keeps both attempts.
- 6 problems not attempted, each with a named reason (incl. one that is
  card-infeasible: the eager baseline itself OOMs 24 GiB). Skipped is not
  passed — see the table footer.

## Anti-reward-hacking

Ported from the paper's protocol + KernelBench's EVAL.md guidance: a
hash-pinned protected eval (every verdict embeds the eval's sha256), static
checks that the candidate's `forward` really calls the declared HIP entry (no
torch fallback) and uses no streams/events/graphs/timing tricks, runtime input
immutability hashing and an A,B,A recompute check, and honest verdicts for OOM
and wedge failures (a wedged kernel is killed by subprocess isolation and
recorded, never dropped).

## Repo layout

    harness/eval_protected.py   the protected eval (verdicts are computed here only)
    harness/antihack.py         static checks on candidates
    harness/problems.py         the 30-problem L1 subset + loader
    scripts/run_batch.py        subprocess-per-candidate driver (wedge isolation)
    scripts/run_baselines.py    eager + torch.compile(inductor) reference sweep
    scripts/milestone_table.py  regenerate the table from verdicts
    scripts/gpu_window.sh       batched GPU window wrapper (stop workloads → run → restore)
    candidates/                 agent-drafted HIP kernels (one dir per problem)
    results/verdicts/           one verdict per problem per attempt (full trail)
    results/baselines.json      eager + compiled reference sweep
    results/records/            session records: gates, decisions, claims, non-claims
    docs/gfx1100-notes.md       measured Fedora-ROCm/gfx1100 knowledge base

## Reproducing

Requires: a ROCm 7.x system with userspace at `/usr` (Fedora tested), an amdgpu
GPU, and Python 3.12–3.14. Torch: `pip install torch --index-url
https://download.pytorch.org/whl/rocm7.0` (verified: 2.10.0+rocm7.0,
triton-rocm 3.6.0; inductor works on gfx1100).

```bash
pip install numpy ninja
git clone --depth 1 https://github.com/ScalingIntelligence/KernelBench third_party/KernelBench
./scripts/py.sh harness/problems.py                 # subset self-check (CPU)
# one GPU window (stops/resumes your GPU workloads around it):
./scripts/gpu_window.sh results/baselines.log ./scripts/py.sh scripts/run_baselines.py
./scripts/py.sh scripts/milestone_table.py
```

Known Fedora/ROCm pitfalls (include-chain break from `-isystem /usr/include`,
arch pinning, fp64-upcast OOMs on big tensors) are documented with minimal
repros in [`docs/gfx1100-notes.md`](docs/gfx1100-notes.md); the include-chain
bug is filed upstream as
[pytorch#202007](https://github.com/pytorch/pytorch/issues/202007).

## License

MIT for everything in this repo (see [LICENSE](LICENSE)). KernelBench
(ScalingIntelligence, MIT) is referenced as the upstream benchmark; fetch it
separately — it is not redistributed here.
