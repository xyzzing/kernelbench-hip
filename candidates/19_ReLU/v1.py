# Candidate: 19_ReLU v1 — flat elementwise kernel, 256 threads, grid-stride free
# (n fits long; one thread per element, grid covers exactly).
ENTRY = "relu_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void relu_kernel(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(x[i], 0.0f);
}

void relu_entry(torch::Tensor x, torch::Tensor y) {
  long n = x.numel();
  long blocks = (n + 255) / 256;
  relu_kernel<<<dim3(blocks), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
}
'''

EXT = load_inline(
    name="relu_hip_v1",
    cpp_sources="void relu_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["relu_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        EXT.relu_entry(x, y)
        return y
