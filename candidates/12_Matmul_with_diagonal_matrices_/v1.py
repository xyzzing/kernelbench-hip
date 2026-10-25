# 12_Diagonal v1 — the reference is NOT a matmul: A.unsqueeze(1) * B is a
# broadcast row scale, A (4096,), B (4096 x 4096). One elementwise pass.
ENTRY = "diagmul_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void diagmul_kernel(const float* a, const float* b, float* y,
                               long rows, long cols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n = rows * cols;
  if (i < n) {
    y[i] = a[i / cols] * b[i];
  }
}

void diagmul_entry(torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  long n = b.numel();
  long cols = b.size(1);
  long blocks = (n + 255) / 256;
  diagmul_kernel<<<dim3(blocks), dim3(256), 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(),
      b.size(0), cols);
}
'''

EXT = load_inline(
    name="diagmul_hip_v1",
    cpp_sources="void diagmul_entry(torch::Tensor a, torch::Tensor b, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["diagmul_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, A, B):
        y = torch.empty_like(B)
        EXT.diagmul_entry(A, B, y)
        return y
