"""Load KernelBench problems from the vendored reference clone.

A problem file defines: Model (nn.Module), get_inputs(), get_init_inputs(),
and module-level size constants. We exec the source with a fresh module dict
and pull those four names out. No candidate code passes through here.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

KB_LEVEL1 = Path(__file__).resolve().parent.parent / "third_party" / "KernelBench" / "KernelBench" / "level1"

# Pilot subset: 30 Level-1 problems, representative of every kernel family in
# Level 1 (matmul 8, activations 6, norms 4, reductions 4, pooling 2, conv 2,
# attention 1, losses 2, + 1 structured matmul). Conv/pooling deliberately
# included though library-dominated — the +3 tier there is expected to be hard
# and that is part of the honest result.
SUBSET = [
    # matmul family
    "1_Square_matrix_multiplication_.py",
    "2_Standard_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "4_Matrix_vector_multiplication_.py",
    "6_Matmul_with_large_K_dimension_.py",
    "7_Matmul_with_small_K_dimension_.py",
    "12_Matmul_with_diagonal_matrices_.py",
    "16_Matmul_with_transposed_A.py",
    # activations
    "19_ReLU.py",
    "23_Softmax.py",
    "24_LogSoftmax.py",
    "25_Swish.py",
    "26_GELU_.py",
    "88_MinGPTNewGelu.py",
    # norms
    "33_BatchNorm.py",
    "36_RMSNorm_.py",
    "37_FrobeniusNorm_.py",
    "40_LayerNorm.py",
    # reductions
    "47_Sum_reduction_over_a_dimension.py",
    "48_Mean_reduction_over_a_dimension.py",
    "51_Argmax_over_a_dimension.py",
    "89_cumsum.py",
    # pooling
    "42_Max_Pooling_2D.py",
    "45_Average_Pooling_2D.py",
    # conv
    "50_conv_standard_2D__square_input__square_kernel.py",
    "87_conv_pointwise_2D.py",
    # attention
    "97_ScaledDotProductAttention.py",
    # losses
    "94_MSELoss.py",
    "95_CrossEntropyLoss.py",
    # structured matmul
    "13_Matmul_for_symmetric_matrices.py",
]


def load_problem(name: str):
    """Exec a problem file, return (Model, get_inputs, get_init_inputs, src)."""
    path = KB_LEVEL1 / name
    if not path.exists():
        raise FileNotFoundError(f"problem not found: {path}")
    src = path.read_text()
    mod = {}
    code = compile(src, name, "exec")
    exec(code, mod)
    for required in ("Model", "get_inputs", "get_init_inputs"):
        if required not in mod:
            raise KeyError(f"{name}: missing {required}")
    return mod["Model"], mod["get_inputs"], mod["get_init_inputs"], src


def subset_names(only: list[str] | None = None) -> list[str]:
    if only:
        return only
    # dedupe, keep file order of the SUBSET list
    seen, out = set(), []
    for n in SUBSET:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


if __name__ == "__main__":
    names = subset_names(sys.argv[1:] or None)
    ok = 0
    for n in names:
        try:
            load_problem(n)
            ok += 1
        except Exception as e:
            print(f"LOAD-FAIL {n}: {e!r}")
    print(f"loaded {ok}/{len(names)}")
