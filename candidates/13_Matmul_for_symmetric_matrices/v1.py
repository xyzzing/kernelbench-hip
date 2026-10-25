# 1_Square (and family) v1 — classic tiled SGEMM, no tensor cores.
# 64x64 C tile per block, 16x16 threads, 4x4 register micro-tile per thread,
# A/B k-tiles (64x16, 16x64) staged in LDS. rocBLAS does 27.5 TFLOPS here —
# parity is the expected tier for a hand kernel; the point is the harness
# measuring the gap honestly. Requires M,N % 64 == 0 and K % 16 == 0
# (raises otherwise — never falls back to torch).
ENTRY = "sgemm_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define TM 64
#define TN 64
#define TK 16

// C (M x N) = op(A) (M x K) * B (K x N); op = transpose if trans_a.
__global__ void sgemm_kernel(const float* A, const float* B, float* C,
                             long M, long N, long K, long lda, long ldb, long ldc,
                             int trans_a, long batch_stride_a,
                             long batch_stride_b, long batch_stride_c) {
  A += blockIdx.z * batch_stride_a;
  B += blockIdx.z * batch_stride_b;
  C += blockIdx.z * batch_stride_c;

  long tile_m = blockIdx.x;  // M/64
  long tile_n = blockIdx.y;  // N/64
  int tx = threadIdx.x % 16;
  int ty = threadIdx.x / 16;

  __shared__ float As[TM * TK];
  __shared__ float Bs[TK * TN];

  float acc[4][4] = {};

  long row0 = tile_m * TM + ty * 4;
  long col0 = tile_n * TN + tx * 4;

  for (long k0 = 0; k0 < K; k0 += TK) {
    // thread t: tile-row (t%64), k-rows (t/64) + {0,4,8,12}; 4 elems each for A and B
    int lr = threadIdx.x % TM;
    int lk = threadIdx.x / TM;
    for (int i = 0; i < 4; ++i) {
      long r = tile_m * TM + lr;
      long c = k0 + lk + i * 4;
      As[lr * TK + lk + i * 4] =
          (r < M && c < K) ? (trans_a ? A[c * lda + r] : A[r * lda + c]) : 0.0f;
      long br = c;
      long bc = tile_n * TN + lr;
      Bs[(lk + i * 4) * TN + lr] = (br < K && bc < N) ? B[br * ldb + bc] : 0.0f;
    }
    __syncthreads();

    for (int kk = 0; kk < TK; ++kk) {
      float a0 = As[(ty * 4 + 0) * TK + kk];
      float a1 = As[(ty * 4 + 1) * TK + kk];
      float a2 = As[(ty * 4 + 2) * TK + kk];
      float a3 = As[(ty * 4 + 3) * TK + kk];
      float b0 = Bs[kk * TN + tx * 4 + 0];
      float b1 = Bs[kk * TN + tx * 4 + 1];
      float b2 = Bs[kk * TN + tx * 4 + 2];
      float b3 = Bs[kk * TN + tx * 4 + 3];
      acc[0][0] += a0*b0; acc[0][1] += a0*b1; acc[0][2] += a0*b2; acc[0][3] += a0*b3;
      acc[1][0] += a1*b0; acc[1][1] += a1*b1; acc[1][2] += a1*b2; acc[1][3] += a1*b3;
      acc[2][0] += a2*b0; acc[2][1] += a2*b1; acc[2][2] += a2*b2; acc[2][3] += a2*b3;
      acc[3][0] += a3*b0; acc[3][1] += a3*b1; acc[3][2] += a3*b2; acc[3][3] += a3*b3;
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    long r = row0 + i;
    if (r >= M) break;
    for (int j = 0; j < 4; ++j) {
      long c = col0 + j;
      if (c < N) C[r * ldc + c] = acc[i][j];
    }
  }
}

void sgemm_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                 long trans_a, long batches) {
  long M, K, N, lda, ldb;
  if (trans_a) {
    K = A.size(0); M = A.size(1); lda = A.size(1) ? A.stride(0) : 0;
  } else {
    M = A.size(0); K = A.size(1); lda = A.stride(0);
  }
  N = B.size(1); ldb = B.stride(0);
  long ldc = C.stride(0);
  long bsa = trans_a ? K * lda : M * lda;
  long bsb = K * ldb;
  long bsc = M * ldc;
  dim3 grid((unsigned)(M / TM), (unsigned)(N / TN), (unsigned)batches);
  sgemm_kernel<<<grid, dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
      M, N, K, lda, ldb, ldc, (int)trans_a, bsa, bsb, bsc);
}
'''

EXT = load_inline(
    name="sgemm_hip_v13",
    cpp_sources="void sgemm_entry(torch::Tensor A, torch::Tensor B, torch::Tensor C, long trans_a, long batches);",
    cuda_sources=_SRC,
    functions=["sgemm_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    """torch.matmul(A, B); A may be given as (K, M) meaning A.T (problem 16)."""
    TRANS_A = False
    BATCHED = False

    def forward(self, A, B):
        if self.BATCHED:
            bM, bK, bN = A.size(1), A.size(2), B.size(2)
            if bM % 64 or bN % 64 or bK % 16:
                raise RuntimeError("shape not divisible by tile (v1 contract)")
            C = torch.empty(A.size(0), bM, bN, device=A.device, dtype=torch.float32)
        else:
            M = A.size(1) if self.TRANS_A else A.size(0)
            K = A.size(0) if self.TRANS_A else A.size(1)
            N = B.size(1)
            if M % 64 or N % 64 or K % 16:
                raise RuntimeError("shape not divisible by tile (v1 contract)")
            C = torch.empty(M, N, device=A.device, dtype=torch.float32)
        EXT.sgemm_entry(A, B, C, 1 if self.TRANS_A else 0,
                        A.size(0) if self.BATCHED else 1)
        return C
