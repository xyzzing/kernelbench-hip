# Contributing results for other AMD cards

The harness is card-agnostic: it runs wherever PyTorch-ROCm sees a GPU. What is
**gfx1100-specific** is every number in `results/` — so per-card results are a
first-class contribution. The procedure below exists so rows from different
cards are comparable and honestly qualified.

## Ground rules

1. **The protocol is fixed.** Don't change tolerances, trial counts, timing
   parameters, or the milestone definition to make numbers look better. If
   something can't run on your card, that is a *result* — record it (skipped
   is not passed).
2. **Compute the traffic floor before drafting a candidate**: bytes moved ×
   your card's HBM rate. Record expected-vs-measured in your notes; a "win"
   with no floor story is how reward hacking sneaks in.
3. **Honest bad numbers are the point.** We review for protocol fidelity, not
   for good results. Ties and losses are rows too.
4. State your co-tenancy conditions: what else was on the card, whether clocks
   were stock, laptop vs desktop.

## Layout: one directory per card

    results/<gfx-arch>-<short-card-name>/     e.g. results/gfx1201-rx9070xt/
    ├── env.json            GPU name (marketing + gfx arch), VRAM, ROCm/torch/driver/OS versions
    ├── baselines.json      from scripts/run_baselines.py
    ├── milestone-table.md  from scripts/milestone_table.py
    ├── verdicts/           one JSON per problem per attempt (the full trail)
    └── NOTES.md            quirks, what failed and why, traffic-floor notes

The pilot's gfx1100 results intentionally stay at the repo root (they are the
first row); new cards go in their own directory.

## Submitting

Open an issue using the **Baseline report** template with the directory's files
attached (zip or gist). Quirks that survive review get folded into
`docs/gfx1100-notes.md`'s portability map with your card named.

## What we will fold in

- New per-card baseline rows (any RDNA2/RDNA3/RDNA4/CDNA card torch-ROCm supports).
- Kernel candidates — provided they pass the same anti-hack checks and carry
  their traffic-floor reasoning.
- Portability findings: anything that breaks differently on your stack
  (distro-level, VRAM-class, or arch-level — the notes doc sorts findings into
  these three layers).

## Licensing

MIT, same as the repo. You keep authorship of your result directories (named
in the PR); by contributing you agree it may be distributed under the repo
license.
