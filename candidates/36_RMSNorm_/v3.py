# 36_RMSNorm v2 — split kernels to bound memory: pass 1 computes per-row RMS
# into a 117 MB side buffer (29.36M rows), pass 2 normalizes. v1's single-pass
# design was fine but the eval phase had input + output + the reference's own
# 7 GiB x**2 temp live at once; splitting drops the candidate's peak from
# in+out+temp to in+out. eps inside sqrt, exact reference contract.
ENTRY = "rmsnorm_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void rms_stats_kernel(const float* x, float* rms, float eps,
                                 long rows, int F, long inner) {
  long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  long b = r / inner;         // batch index
  long rem = r % inner;       // (d1, d2) offset
  const float* xr = x + b * (long)F * inner + rem;
  float acc = 0.0f;
  for (int f = 0; f < F; ++f) {
    float v = xr[(long)f * inner];
    acc += v * v;
  }
  rms[r] = rsqrtf(acc / (float)F + eps);
}

__global__ void rms_apply_kernel(const float* x, float* y, const float* rms,
                                 long rows, int F, long inner) {
  long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  long b = r / inner;
  long rem = r % inner;
  const float* xr = x + b * (long)F * inner + rem;
  float* yr = y + b * (long)F * inner + rem;
  float inv = rms[r];
  for (int f = 0; f < F; ++f) {
    yr[(long)f * inner] = xr[(long)f * inner] * inv;
  }
}

void rmsnorm_entry(torch::Tensor x, torch::Tensor y, torch::Tensor rms,
                   double eps, long rows, long F, long inner) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long blocks = (rows + 255) / 256;
  rms_stats_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
      x.data_ptr<float>(), rms.data_ptr<float>(), (float)eps, rows, (int)F, inner);
  rms_apply_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), rms.data_ptr<float>(),
      rows, (int)F, inner);
}
'''

EXT = load_inline(
    name="rmsnorm_hip_v3",
    cpp_sources=("void rmsnorm_entry(torch::Tensor x, torch::Tensor y, torch::Tensor rms, "
                 "double eps, long rows, long F, long inner);"),
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
        rms = torch.empty(rows, device=x.device, dtype=torch.float32)
        EXT.rmsnorm_entry(x, y, rms, self.eps, rows, F, inner)
        return y
