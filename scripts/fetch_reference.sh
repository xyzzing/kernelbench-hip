#!/bin/bash
# Fetch the reference benchmark (not redistributed; MIT, ScalingIntelligence).
# Optionally also the AMD AIG-Datasets sample docs (see docs/gfx1100-notes.md §6).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/third_party"
git clone --depth 1 https://github.com/ScalingIntelligence/KernelBench "$HERE/third_party/KernelBench"
echo "KernelBench cloned. The pilot subset is in KernelBench/level1/."
