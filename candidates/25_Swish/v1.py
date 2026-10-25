# 25_Swish v1 — single fused pass; eager = sigmoid kernel + mul kernel (2 passes
# + a 6.4 GiB temp), ours = one 12.8 GiB streaming pass.
ENTRY = "swish_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void swish_kernel(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

void swish_entry(torch::Tensor x, torch::Tensor y) {
  long n = x.numel();
  long blocks = (n + 255) / 256;
  swish_kernel<<<dim3(blocks), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
}
'''

EXT = load_inline(
    name="swish_hip_v1",
    cpp_sources="void swish_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["swish_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        EXT.swish_entry(x, y)
        return y
