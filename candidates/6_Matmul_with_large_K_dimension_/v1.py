# 6_LargeK v1 — M=256, N=256, K=524288. rocBLAS runs this at ~78 GB/s effective
# (12.87 ms); the shape is split-K territory. Grid = (M/64, N/64, SPLIT=8);
# each block accumulates a 64x64 tile over K/8 then atomicAdds into C
# (zeroed per call). Compute ~68.7 GFLOP at a realistic ~10-15 TFLOPS hand
# kernel -> 4.6-6.9 ms vs eager/compiled ~12.8 ms.
ENTRY = "splitk_gemm_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define TM 64
#define TN 64
#define TK 16

__global__ void splitk_kernel(const float* A, const float* B, float* C,
                              long M, long N, long K, long kchunk) {
  long k_begin = blockIdx.z * kchunk;
  long k_end = min(k_begin + kchunk, K);
  long tile_m = blockIdx.x;
  long tile_n = blockIdx.y;
  int lr = threadIdx.x % TM;
  int lk = threadIdx.x / TM;
  __shared__ float As[TM * TK];
  __shared__ float Bs[TK * TN];

  float acc[4][4] = {};
  long row0 = tile_m * TM + (threadIdx.x / 16) * 4;
  long col0 = tile_n * TN + (threadIdx.x % 16) * 4;

  for (long k0 = k_begin; k0 < k_end; k0 += TK) {
    for (int i = 0; i < 4; ++i) {
      long r = tile_m * TM + lr;
      long c = k0 + lk + i * 4;
      As[lr * TK + lk + i * 4] = (r < M && c < k_end) ? A[r * K + c] : 0.0f;
      Bs[(lk + i * 4) * TN + lr] =
          (c < k_end && tile_n * TN + lr < N) ? B[c * N + tile_n * TN + lr] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < TK; ++kk) {
      int ty = threadIdx.x / 16;
      int tx = threadIdx.x % 16;
      float a_0 = As[(ty * 4 + 0) * TK + kk];
      float a_1 = As[(ty * 4 + 1) * TK + kk];
      float a_2 = As[(ty * 4 + 2) * TK + kk];
      float a_3 = As[(ty * 4 + 3) * TK + kk];
      float b0 = Bs[kk * TN + tx * 4 + 0];
      float b1 = Bs[kk * TN + tx * 4 + 1];
      float b2 = Bs[kk * TN + tx * 4 + 2];
      float b3 = Bs[kk * TN + tx * 4 + 3];
      acc[0][0] += a_0*b0; acc[0][1] += a_0*b1; acc[0][2] += a_0*b2; acc[0][3] += a_0*b3;
      acc[1][0] += a_1*b0; acc[1][1] += a_1*b1; acc[1][2] += a_1*b2; acc[1][3] += a_1*b3;
      acc[2][0] += a_2*b0; acc[2][1] += a_2*b1; acc[2][2] += a_2*b2; acc[2][3] += a_2*b3;
      acc[3][0] += a_3*b0; acc[3][1] += a_3*b1; acc[3][2] += a_3*b2; acc[3][3] += a_3*b3;
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      atomicAdd(&C[(row0 + i) * N + col0 + j], acc[i][j]);
    }
  }
}

void splitk_gemm_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                       long split) {
  long M = A.size(0), K = A.size(1), N = B.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipMemsetAsync(C.data_ptr<float>(), 0, M * N * sizeof(float), stream);
  long kchunk = (K + split - 1) / split;
  kchunk = (kchunk + TK - 1) / TK * TK;
  dim3 grid((unsigned)(M / TM), (unsigned)(N / TN), (unsigned)split);
  splitk_kernel<<<grid, dim3(256), 0, stream>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
      M, N, K, kchunk);
}
'''

EXT = load_inline(
    name="splitk_gemm_hip_v1",
    cpp_sources="void splitk_gemm_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C, long split);",
    cuda_sources=_SRC,
    functions=["splitk_gemm_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, A, B):
        M, K = A.shape
        N = B.size(1)
        if M % 64 or N % 64 or K % 16:
            raise RuntimeError("shape not divisible by tile (v1 contract)")
        C = torch.empty(M, N, device=A.device, dtype=torch.float32)
        EXT.splitk_gemm_entry(A, B, C, 8)
        return C
