---
name: Baseline report
about: Per-card KernelBench-HIP results (one directory per card — see CONTRIBUTING.md)
title: "[baseline] <gfx-arch> <card name>"
labels: ''
assignees: ''
---

<!-- One issue per card. Attach the files from your results/<slug>/ directory
     (zip or gist). See CONTRIBUTING.md for the procedure and the ground rules. -->

**GPU**

- Marketing name:
- gfx arch (from the harness output, e.g. gfx1201):
- VRAM:

**Software stack**

- OS / distro:
- ROCm userspace version and install location (/opt/rocm or /usr):
- torch (`pip show torch` — exact version + index, e.g. 2.10.0+rocm7.0):
- python version:
- GPU driver / kernel:

**Conditions**

- What else was running on the card during the windows?
- Clocks stock / pinned? Laptop (thermals) or desktop?
- Anything about memory pressure (other VRAM tenants, iGPU carve-outs)?

**What ran**

Commands used (copy-paste), and which of the 30 subset problems completed.

**Attachments**

- [ ] env.json
- [ ] baselines.json
- [ ] milestone-table.md
- [ ] verdicts/ (zip)
- [ ] NOTES.md

**What failed to run, and why**

(skipped is not passed — failed and skipped problems are results too)

**Anything surprising**

Quirks worth adding to the portability map (say which layer you think they
belong to: distro-level / VRAM-class / arch-level — guessing wrong is fine).
