# 47_Sum over dim=1 (128 x 4096 x 4095, middle-dim reduction, keepdim=True).
# One kernel: block = 256 threads mapped onto a 256-wide j-chunk; the block
# loops i = 0..4095 reading x[b, i, j0+t] — adjacent threads read adjacent
# addresses (coalesced), each thread accumulates its column privately, then
# one coalesced write. 8.4 GiB read total, no temps.
ENTRY = "colreduce_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void colreduce_kernel(const float* x, float* out, long B, long I,
                                 long J, long mean_div, int mean_mode) {
  long b = blockIdx.x;
  long j0 = (long)blockIdx.y * blockDim.x;
  long j = j0 + threadIdx.x;
  if (j >= J) return;
  const float* xb = x + b * I * J + j;
  float acc = 0.0f;
  for (long i = 0; i < I; ++i) {
    acc += xb[i * J];
  }
  if (mean_mode) acc /= mean_div;
  out[b * J + j] = acc;
}

void colreduce_entry(torch::Tensor x, torch::Tensor out, long mean_mode) {
  long B = x.size(0), I = x.size(1), J = x.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)B, (unsigned)((J + 255) / 256));
  colreduce_kernel<<<grid, dim3(256), 0, stream>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), B, I, J,
      mean_mode ? I : 1, (int)mean_mode);
}
'''

EXT = load_inline(
    name="colreduce_hip_v1",
    cpp_sources="void colreduce_entry(torch::Tensor x, torch::Tensor out, long mean_mode);",
    cuda_sources=_SRC,
    functions=["colreduce_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    """torch.sum(x, dim=1, keepdim=True)"""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        B, _, J = x.shape
        out = torch.empty(B, 1, J, device=x.device, dtype=torch.float32)
        EXT.colreduce_entry(x, out.view(B, J), 0)
        return out
