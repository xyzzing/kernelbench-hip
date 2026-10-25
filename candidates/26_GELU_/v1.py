# 26_GELU v1 — exact erf formula (matches F.gelu default), single pass.
# eager aten gelu is already one fused kernel: expect a tie at the bandwidth roof.
ENTRY = "gelu_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void gelu_kernel(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
  }
}

void gelu_entry(torch::Tensor x, torch::Tensor y) {
  long n = x.numel();
  long blocks = (n + 255) / 256;
  gelu_kernel<<<dim3(blocks), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
}
'''

EXT = load_inline(
    name="gelu_hip_v1",
    cpp_sources="void gelu_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["gelu_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        EXT.gelu_entry(x, y)
        return y
