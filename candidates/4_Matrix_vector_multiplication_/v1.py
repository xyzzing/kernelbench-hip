# 4_Matvec v1 — C = A·B, A (2048 x 1048576) fp32 = 8.6 GiB, B (K x 1).
# Traffic-bound: floor ~9.6 ms at HBM rate. Eager/compiled 24.7 ms = 350 GB/s
# (rocBLAS split for K=1M is poor) — one coalesced block-per-row-chunk kernel
# with smem reduce + fp32 atomicAdd into the row scalar.
ENTRY = "gemv_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHUNK 32768

__global__ void gemv_kernel(const float* A, const float* B, float* C, long K) {
  long row = blockIdx.x / ((K + CHUNK - 1) / CHUNK);
  long chunk = blockIdx.x % ((K + CHUNK - 1) / CHUNK);
  long start = chunk * CHUNK;
  long end = min(start + (long)CHUNK, K);
  const float* Ar = A + row * K + start;
  const float* b = B + start;

  float acc = 0.0f;
  for (long i = threadIdx.x; i < end - start; i += blockDim.x) {
    acc += Ar[i] * b[i];
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&C[row], sm[0]);
}

void gemv_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  long M = A.size(0), K = A.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipMemsetAsync(C.data_ptr<float>(), 0, M * sizeof(float), stream);
  long blocks = M * ((K + CHUNK - 1) / CHUNK);
  gemv_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K);
}
'''

EXT = load_inline(
    name="gemv_hip_v1",
    cpp_sources="void gemv_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C);",
    cuda_sources=_SRC,
    functions=["gemv_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, A, B):
        C = torch.empty(A.size(0), 1, device=A.device, dtype=torch.float32)
        EXT.gemv_entry(A, B, C)
        return C
