# 51_Argmax v1 — dim=1 over (128 x 4096 x 4095), middle-dim reduction to int64
# indices. Same coalesced column-walk as the 47/48 kernel; strict > keeps the
# FIRST max index (torch's tie rule; ties are measure-zero on torch.rand).
ENTRY = "argmax_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void argmax_kernel(const float* x, int64_t* out, long B, long I,
                              long J) {
  long b = blockIdx.x;
  long j0 = (long)blockIdx.y * blockDim.x;
  long j = j0 + threadIdx.x;
  if (j >= J) return;
  const float* xb = x + b * I * J + j;
  float m = -3.402823466e+38f;
  int64_t mi = 0;
  for (long i = 0; i < I; ++i) {
    float v = xb[i * J];
    if (v > m) { m = v; mi = i; }
  }
  out[b * J + j] = mi;
}

void argmax_entry(torch::Tensor x, torch::Tensor out) {
  long B = x.size(0), I = x.size(1), J = x.size(2);
  dim3 grid((unsigned)B, (unsigned)((J + 255) / 256));
  argmax_kernel<<<grid, dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), out.data_ptr<int64_t>(), B, I, J);
}
'''

EXT = load_inline(
    name="argmax_hip_v2",
    cpp_sources="void argmax_entry(torch::Tensor x, torch::Tensor out);",
    cuda_sources=_SRC,
    functions=["argmax_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        B, _, J = x.shape
        out = torch.empty(B, J, device=x.device, dtype=torch.int64)
        EXT.argmax_entry(x, out)
        return out
