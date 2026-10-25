# 88_MinGPTNewGelu v1 — tanh-form GELU fused to ONE pass. Eager runs a 5-op
# chain (mul/add/pow/tanh/mul) with 3 materialized temps (39.3ms baseline was
# the 4096x393216 family; this problem is 8192x8192, compiled fused it to
# 0.743ms — the bar to beat).
ENTRY = "mingpt_gelu_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void mingpt_gelu_kernel(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

void mingpt_gelu_entry(torch::Tensor x, torch::Tensor y) {
  long n = x.numel();
  long blocks = (n + 255) / 256;
  mingpt_gelu_kernel<<<dim3(blocks), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
}
'''

EXT = load_inline(
    name="mingpt_gelu_hip_v1",
    cpp_sources="void mingpt_gelu_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["mingpt_gelu_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        EXT.mingpt_gelu_entry(x, y)
        return y
