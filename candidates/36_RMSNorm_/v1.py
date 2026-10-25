# 36_RMSNorm v1 — one pass. Eager: mean pass (reads+temp) + div pass = ~3 passes
# over the 7.5 GiB tensor (46.2 ms); compiled 20.3 ms; ours reads 7.5 + writes 7.5
# once. eps goes INSIDE the sqrt exactly like the reference: x / sqrt(mean(x^2, dim=1) + eps).
# Layout 112 x 64 x 512 x 512, reduce over dim=1 (64 contiguous-adjacent rows of
# 512*512): thread per (b, d1, d2) row of 64 values -> coalesced if threads map
# to d2 fastest (contiguous). 29.36M rows, 64 elems each.
ENTRY = "rmsnorm_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// x: (B=112, F=64, D1=512, D2=512) contiguous. Row of F values starts at
// ((b*F + f)*D1 + d1)*D2 + d2 -> for fixed (b,d1,d2), f-stride = D1*D2 = 262144.
// Thread per (b,d1,d2): 64 strided loads (stride 1 MB) — uncoalesced per thread
// BUT adjacent threads read adjacent d2 -> each warp iteration is fully coalesced.
__global__ void rmsnorm_kernel(const float* x, float* y, float eps,
                               long rows, int F, long inner) {
  long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  long base = r;
  // r = ((b*F? no: r indexes (b, d1, d2) = b*inner*? -> base = r over B*D1*D2 rows
  const float* xr = x + base;
  float acc = 0.0f;
  for (int f = 0; f < F; ++f) {
    float v = xr[(long)f * inner];
    acc += v * v;
  }
  float rms = sqrtf(acc / (float)F + eps);
  float* yr = y + base;
  for (int f = 0; f < F; ++f) {
    yr[(long)f * inner] = xr[(long)f * inner] / rms;
  }
}

void rmsnorm_entry(torch::Tensor x, torch::Tensor y, double eps, long rows,
                   long F, long inner) {
  long blocks = (rows + 255) / 256;
  rmsnorm_kernel<<<dim3(blocks), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), (float)eps, rows, (int)F, inner);
}
'''

EXT = load_inline(
    name="rmsnorm_hip_v1",
    cpp_sources="void rmsnorm_entry(torch::Tensor x, torch::Tensor y, double eps, long rows, long F, long inner);",
    cuda_sources=_SRC,
    functions=["rmsnorm_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def __init__(self, num_features: int = 64, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        B, F, D1, D2 = x.shape
        rows = B * D1 * D2
        inner = D1 * D2
        y = torch.empty_like(x)
        EXT.rmsnorm_entry(x, y, self.eps, rows, F, inner)
        return y
