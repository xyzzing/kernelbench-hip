# KernelBench-HIP pilot — milestone table

Milestones (CUDA-Agent paper): -1 fail · 1 parity · 2 beats eager · 3 beats torch.compile.
Every number: median ms over 100 L2-thrashed event-timed trials; arms named; verdict file cited.

| problem | milestone | cand ms | eager ms | compiled ms | vs eager | vs compiled |
| --- | --- | --- | --- | --- | --- | --- |
| 12_Matmul_with_diagonal_matrices_ | 3 | 0.230922 | 0.276283 | 0.269663 | 1.234 | 1.1707 |
| 13_Matmul_for_symmetric_matrices | 1 | 15.41881 | 5.021258 | 5.021898 | 0.3226 | 0.3209 |
| 16_Matmul_with_transposed_A | 1 | 15.128525 | 5.134873 | 5.162553 | 0.3373 | 0.3372 |
| 19_ReLU | 3 | 14.836503 | 15.267159 | 15.110098 | 1.0163 | 1.0305 |
| 1_Square_matrix_multiplication_ | 1 | 15.421633 | 4.980609 | 4.970269 | 0.3227 | 0.322 |
| 23_Softmax | 1 | 39.478573 | 33.170933 | 25.118628 | 0.8383 | 0.6347 |
| 24_LogSoftmax | 1 | 39.840513 | 34.071064 | 25.09813 | 0.8516 | 0.6276 |
| 25_Swish | 3 | 14.824902 | 39.250666 | 15.084384 | 2.641 | 1.0156 |
| 26_GELU_ | 3 | 14.824712 | 15.255887 | 15.111365 | 1.0149 | 1.0277 |
| 33_BatchNorm | 1 | 315.866486 | 20.638468 | 14.828204 | 0.0654 | 0.0468 |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | -1 | None | 46.245913 | 20.256826 | None | None |
| 36_RMSNorm_ | 2 | 34.030701 | 46.245913 | 20.256826 | 1.3447 | 0.5935 |
| 37_FrobeniusNorm_ | -1 | None | 26.74844 | 25.84489 | None | None |
| 37_FrobeniusNorm_ | -1 | None | 26.74844 | 25.84489 | None | None |
| 37_FrobeniusNorm_ | 1 | 27.262226 | 26.74844 | 25.84489 | 0.9565 | 0.9566 |
| 3_Batched_matrix_multiplication | -1 | None | 10.441704 | 10.439084 | None | None |
| 3_Batched_matrix_multiplication | -1 | None | 10.441704 | 10.439084 | None | None |
| 40_LayerNorm | -1 | None | 2.629869 | 1.157393 | None | None |
| 40_LayerNorm | 2 | 1.229972 | 2.629869 | 1.157393 | 2.1371 | 0.9419 |
| 42_Max_Pooling_2D | 2 | 20.427053 | 20.886578 | 8.495217 | 1.0111 | 0.4105 |
| 47_Sum_reduction_over_a_dimension | 1 | 13.027383 | 10.612439 | 10.919703 | 0.8139 | 0.8419 |
| 48_Mean_reduction_over_a_dimension | 1 | 13.011418 | 10.61466 | 10.793842 | 0.8148 | 0.8492 |
| 4_Matrix_vector_multiplication_ | -1 | None | 24.733712 | 24.788571 | None | None |
| 4_Matrix_vector_multiplication_ | 3 | 11.408542 | 24.733712 | 24.788571 | 2.0351 | 2.0414 |
| 51_Argmax_over_a_dimension | -1 | None | 10.626341 | 10.935544 | None | None |
| 51_Argmax_over_a_dimension | -1 | None | 10.626341 | 10.935544 | None | None |
| 51_Argmax_over_a_dimension | 1 | 15.74893 | 10.626341 | 10.935544 | 0.6747 | 0.6959 |
| 6_Matmul_with_large_K_dimension_ | 3 | 11.572277 | 12.873952 | 12.839051 | 1.0306 | 1.0294 |
| 7_Matmul_with_small_K_dimension_ | 1 | 42.859703 | 6.188204 | 6.195204 | 0.1396 | 0.1393 |
| 88_MinGPTNewGelu | 3 | 0.693927 | 5.945485 | 0.742548 | 8.6115 | 1.0654 |
| 94_MSELoss | 2 | 11.430781 | 31.21652 | 9.727873 | 2.7231 | 0.8493 |

## Failures (verdicts, not dropped)

- 36_RMSNorm_.py: parity OOM: HIP out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of 23.98 GiB of which 480.00 MiB is free. Of the allocated memory 23.25 GiB is allocated by PyTorch, and 0 bytes is reserved by PyT (36_RMSNorm-v1.json)
- 36_RMSNorm_.py: parity OOM: HIP out of memory. Tried to allocate 7.00 GiB. GPU 0 has a total capacity of 23.98 GiB of which 2.75 GiB is free. Of the allocated memory 21.00 GiB is allocated by PyTorch, and 0 bytes is reserved by PyTorch  (36_RMSNorm-v2.json)
- 36_RMSNorm_.py: parity OOM: HIP out of memory. Tried to allocate 7.00 GiB. GPU 0 has a total capacity of 23.98 GiB of which 2.75 GiB is free. Of the allocated memory 21.00 GiB is allocated by PyTorch, and 0 bytes is reserved by PyTorch  (36_RMSNorm-v3.json)
- 36_RMSNorm_.py: compile failed: NameError: name 'signal' is not defined (36_RMSNorm-v4.json)
- 36_RMSNorm_.py: parity OOM: HIP out of memory. Tried to allocate 7.00 GiB. GPU 0 has a total capacity of 23.98 GiB of which 6.06 GiB is free. Of the allocated memory 14.38 GiB is allocated by PyTorch, and 3.31 GiB is reserved by PyTorch (36_RMSNorm-v5.json)
- 36_RMSNorm_.py: parity OOM: HIP out of memory. Tried to allocate 7.00 GiB. GPU 0 has a total capacity of 23.98 GiB of which 6.06 GiB is free. Of the allocated memory 14.38 GiB is allocated by PyTorch, and 3.31 GiB is reserved by PyTorch (36_RMSNorm-v6.json)
- 36_RMSNorm_.py: parity failed (36_RMSNorm-v7.json)
- 37_FrobeniusNorm_.py: compile failed: RuntimeError: Error building extension 'frobenius_hip_v1': [1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=frobenius_hip_v1 -DTORCH_API_INCLUDE_EXTENSION_H -isystem ~/projects/kernelbench-hip/.venv/lib (37_FrobeniusNorm-v1.json)
- 37_FrobeniusNorm_.py: static anti-hack violation (37_FrobeniusNorm-v2.json)
- 3_Batched_matrix_multiplication.py: compile failed: NameError: name 'signal' is not defined (3_Batched-v3.json)
- 3_Batched_matrix_multiplication.py: eval killed at 900s wall (wedged kernel suspected; HIP context destroyed) (3_Batched-v5.json)
- 40_LayerNorm.py: parity failed (40_LayerNorm-v1.json)
- 4_Matrix_vector_multiplication_.py: compile failed: NameError: name 'signal' is not defined (4_Matvec-v1.json)
- 51_Argmax_over_a_dimension.py: compile failed: ImportError: ~/.cache/torch_extensions/py314_cpu/argmax_hip_v1/argmax_hip_v1.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIxEEPT_v (51_Argmax-v1.json)
- 51_Argmax_over_a_dimension.py: compile failed: ImportError: ~/.cache/torch_extensions/py314_cpu/argmax_hip_v1/argmax_hip_v1.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIxEEPT_v (51_Argmax-v2.json)

## Not attempted (with reasons — skipped is not passed)

- 5_Matrix_scalar_multiplication.py: not in subset
- 45_Average_Pooling_2D.py: pilot budget: pooling covered by 42 (max); avg is the same kernel class
- 50_conv_standard_2D__square_input__square_kernel.py: rocDNN-backed; direct conv kernel out of pilot budget
- 87_conv_pointwise_2D.py: rocDNN-backed; out of pilot budget
- 89_cumsum.py: scan kernel class; torch's optimized scan expected to hold; out of pilot budget
- 95_CrossEntropyLoss.py: CARD-INFEASIBLE: eager baseline itself OOMs the 24 GiB card (baselines.json)
- 97_ScaledDotProductAttention.py: FA-class kernel; out of pilot budget (matmul core is its prerequisite)
