# 24_LogSoftmax v1 — dim=1 over (4096 x 393216). Block per row: max, sum(exp),
# then y = (x - max) - log(sum). __expf/__logf fast intrinsics: parity bound is
# atol=1e-2 vs values ~1e-6..1 — approximation error is far inside it.
ENTRY = "logsoftmax_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void logsoftmax_kernel(const float* x, float* y, long rows, long cols) {
  long row = blockIdx.x;
  const float* xr = x + row * cols;
  float* yr = y + row * cols;

  float tmax = -3.402823466e+38f;
  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    tmax = fmaxf(tmax, xr[c]);
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = tmax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  float m = sm[0];

  float tsum = 0.0f;
  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    tsum += __expf(xr[c] - m);
  }
  sm[threadIdx.x] = tsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  float l = __logf(sm[0]);

  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    yr[c] = (xr[c] - m) - l;
  }
}

void logsoftmax_entry(torch::Tensor x, torch::Tensor y) {
  long rows = x.size(0), cols = x.numel() / rows;
  logsoftmax_kernel<<<dim3((unsigned)rows), dim3(256), 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
}
'''

EXT = load_inline(
    name="logsoftmax_hip_v1",
    cpp_sources="void logsoftmax_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["logsoftmax_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.empty_like(x)
        EXT.logsoftmax_entry(x, y)
        return y
