# gfx1100 / ROCm HIP notes — knowledge base for HIP kernel work

## 0. Portability map — what transfers to YOUR card

Findings sort into three layers. Before reusing anything below, know which
layer it lives in:

| layer | applies to | items here |
| --- | --- | --- |
| **distro/toolchain** | every AMD card on a similar OS stack | §1 (python/venv poisoning, PYTHONPATH leak), §2 (the `-isystem /usr/include` include-chain bug, extension-cache staleness, `ROCM_PATH`), §5's window/hygiene pattern |
| **VRAM-class** | scales with GiB, not ISA | §3 (fp64-upcast OOM, chunked allclose, digest rules, sliced parity) — thresholds should scale with `torch.cuda.mem_get_info()`, constants here are 24 GiB-sized |
| **arch-specific** | gfx1100 (RDNA3) ONLY | §4's achieved bandwidth numbers and every candidates/ verdict, wave32 default, 64 KB LDS, the rocBLAS/inductor behavior observed here |

**Arch families people run (torch-ROCm supports):**

| family | cards (gfx) | what differs vs gfx1100 |
| --- | --- | --- |
| RDNA3 | 7900 XTX/XT/GRE (gfx1100), 7800 XT (gfx1101), 7600 (gfx1102), iGPU gfx1150/1151 | these notes; wave32 |
| RDNA4 | 9070-class (gfx1200/gfx1201) | different WMMA generation — different kernels win; re-measure everything |
| RDNA2 | 6800/6900 (gfx1030) | no f32 WMMA; rocBLAS-artifact class behavior; different arms entirely |
| CDNA | MI-series (gfx90a/942/950) | wave64, MFMA — different kernel culture, different rooflines |

**Rule that follows:** a candidate or optimization measured only on one arch is
evidence for that arch alone — it must not become a default elsewhere (this is
also the discipline Strata's arch-dispatch + opt-in pattern enforces upstream).

Set `PYTORCH_ROCM_ARCH` to *your* arch wherever this doc says gfx1100. Per-card
baseline submissions: see CONTRIBUTING.md.

---

Everything below was **measured on this box** (7900 XTX, gfx1100, Fedora 44,
ROCm 7.1.1 system + torch 2.10.0+rocm7.0 wheel), 2026-10-25/26, unless a
source says otherwise. Each item names how it was proven. Several items
transfer 1:1 to any HIP build on this machine (including strata-amd).

## 1. Python / torch-ROCm stack

- **The zcode AppImage shadows `python3.14` in PATH.** `python3.14 -m venv`
  created a venv whose `bin/python` symlinks to `zcode.appimage` (pyvenv.cfg:
  `home = ~/AppImages`). Every later "python" run launched the desktop
  app. Rule: create and run venvs with the absolute `/usr/bin/python3.14` and
  `env -i PATH=/usr/bin:/bin HOME=$HOME`. (`scripts/py.sh` wraps this.)
- **The tool shell injects `PYTHONPATH`** (`~/.local/lib/python3.14/site-packages`
  + `~/projects/sinter/src`), which preempts venv site-packages: `import torch`
  resolved a user-site `torch 2.14.0+cpu` over the venv's rocm build. `env -i`
  is the only reliable route.
- **pip silently no-ops on a poisoned interpreter**: the AppImage python's pip
  saw user-site torch 2.14.0 satisfying `torch`, skipped the rocm7.0 install,
  exit 0. Always verify `torch.__version__ == '2.10.0+rocm7.0'` post-install.
- **torch wheel**: `torch 2.10.0+rocm7.0` has cp314 wheels on
  `download.pytorch.org/whl/rocm7.0`; bundles HIP 7.0.51831 userspace; works
  against the system amdgpu driver; `triton-rocm 3.6.0` included. **inductor
  compiles gelu/softmax/matmul+add on gfx1100** (sanity-gpu-w1.json).

## 2. Building HIP extensions via `torch.utils.cpp_extension` on Fedora

- **THE include-chain trap**: torch emits `-isystem /usr/include` for HIP
  extensions because `include_paths()` in
  `torch/utils/cpp_extension.py` appends `_join_rocm_home('include')`
  unconditionally in the HIP branch. The CUDA branch right below it *guards*
  against exactly this (`if cuda_home_include != '/usr/include'` — "gcc doesn't
  like having /usr/include passed explicitly"). Effect: every `#include_next
  <stdlib.h>` / `<math.h>` from libstdc++ fails ("No such file or directory")
  on BOTH the host c++ pass and the hipcc device pass. Minimal repro verified:
  `c++ -isystem /usr/include -c` a file including `<cstdlib>` fails; without
  the flag it passes. **Fix (no torch patch): monkeypatch
  `torch.utils.cpp_extension.include_paths` to drop `/usr/include`** —
  `harness/eval_protected.py::_patched_load_inline`. This is upstream-worthy
  as a pytorch/ROCm issue.
- **Arch pinning**: torch passes no `--offload-arch` for HIP extensions;
  Fedora hipcc then compiles for **13 arches** (gfx900…gfx1151). Set
  `PYTORCH_ROCM_ARCH=gfx1100` (read at call time by `_get_rocm_arch_flags`),
  or pass `--offload-arch=gfx1100` in `extra_cuda_cflags` (if the candidate
  passes its own offload-arch, torch steps aside).
- **Extension cache staleness**: `~/.cache/torch_extensions/py314_*/<name>` —
  build.ninja is only regenerated when *sources* change; env-flag changes
  (e.g. arch pinning) replay stale ninja files. `rm -rf` the extension dir
  when changing env. (The dir tag says "cpu" on ROCm builds — cosmetic.)
- `ROCM_PATH=/usr`, `HIP_PATH=/usr` setdefaults; Fedora keeps all of ROCm at
  `/usr`, torch prefers `/opt/rocm`.
- `load_inline` on ROCm hipifies `.cu` sources; plain HIP (`hip/hip_runtime.h`,
  `<<<>>>` launches, `at::cuda::getCurrentCUDAStream()`) compiles clean.

## 3. Memory discipline on 24 GB (KernelBench L1 sizes bite)

- `19_ReLU` allocates 4096×393216 fp32 = **6.44 GiB per tensor**. Parity with
  ref+cand outputs live plus any full-size temp OOMs the card.
- **fp64 upcast is the silent killer**: `t.sum(dtype=torch.float64)` on the
  1.6G-element tensor allocs a **12 GiB** fp64 temp. Digests must use chunked
  fp32 sums (see `_digest`).
- Chunked `allclose` (1 GiB slices) bounds parity temps; `del` + `empty_cache`
  per trial; R2 recompute check uses digests, never device clones.
- inductor's compile-time benchmarking on the big tensor can transiently hold
  several 6 GiB buffers — the compiled tier is wrapped: an OOM there is
  recorded (`compiled_oom`), it never fails the verdict silently.

## 4. Measurement protocol (ported: KernelBench timing.py + CUDA-Agent paper)

- CUDA(=HIP) events, default stream; **warmup 5, trials 100, discard first 1**;
  **L2 thrashed before every trial** (256 MiB int64 fill); stats = median
  (verdict), mean/std/min; **stall flag = median > 3×min**, then
  retry-until-clean keeping both attempts (strata-amd pattern).
- Correctness: **5 seeded trials** (seeds 0..4; seed drives `get_inputs()` AND
  model init), fp32, pass = shape match AND `allclose(atol=rtol=1e-2)` on all
  5 (KernelBench v0/v0.1 convention).
- Milestones (CUDA-Agent paper): −1 fail / +1 parity / +2 beats eager /
  +3 beats torch.compile — "beats" = strictly faster median.
- Anti-hack: static (no torch-op fallback in `forward` — the declared HIP entry
  must be called; no streams/events/graphs/compile in candidate; no harness
  imports) + runtime (input-immutability hashing; A,B,A recompute digest).
  Eval script is hash-pinned into every verdict (`protected_eval_sha`).

## 5. GPU window protocol on this box (shared card)

- `strata-8081.service` (our engine, ~22-24 GiB VRAM resident) is **stopped for
  each window and restored + health-checked after** (`scripts/gpu_window.sh`);
  startup to /health 200 takes ~65 s — health checks must poll, not sleep once.
- Sandboxed shells hide `/dev/kfd`/`/dev/dri`: GPU runs always go through the
  widened-sandbox window wrapper, **batched** (one window per work group).
- Python inside windows: `env -i` (see §1); logs are written to files by the
  workload, not trusted to stdout (desktop-hook noise pollutes both stdout and
  redirected fds on this box).

## 6. Related work landscape (2026-10-25)

- **CUDA-Agent paper (arXiv 2602.24286)**: RL-trained CUDA kernel writer;
  its RL legs are NOT reproducible locally (128-GPU pool) — we adopt its
  environment/eval/milestone protocol with a training-free agent loop.
- **AMD AIG-Datasets v0.2** (huggingface.co/datasets/amd/AIG-Datasets): 60K+
  HIP records — HIP-CudaAgent (5,388 PyTorch→HIP — the paper's recipe, HIP-ified
  by AMD), HIP-GPUMode (5,910 tasks), HIP-to-HIP optimization (34,368),
  AICUDA Engineer (14,282), ROCm-lib QA, Triton sets. Their VALIDATION_PROTOCOL
  is the same generate-evaluate-reflect loop; tolerance there is 1e-4 (ours is
  KernelBench's 1e-2). **They shipped the training corpus; an independent
  KernelBench-protocol evaluation on real gfx1100 remains unoccupied.** Local
  reference copy: `third_party/aig/` (cuda_agent_ops_6k subset + docs).
- No official or community KernelBench port to ROCm/HIP found (searched
  2026-10-25).
