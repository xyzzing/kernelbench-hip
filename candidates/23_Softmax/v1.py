# 23_Softmax v1 — dim=1 over (4096 x 393216). Block per row, 256 threads.
# Pass 1: block max. Pass 2: sum of exp. Pass 3: write. The 1.57 MB row is
# re-read from L2 on passes 2-3 (gfx1100 L2 = 6 MB). Eager 33.2 ms,
# compiled 25.1 ms; traffic floor ~6.4 GiB read + 6.4 write ~ 15 ms.
ENTRY = "softmax_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void softmax_kernel(const float* x, float* y, long rows, long cols) {
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
  float inv = 1.0f / sm[0];

  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    yr[c] = __expf(xr[c] - m) * inv;
  }
}

void softmax_entry(torch::Tensor x, torch::Tensor y) {
  long rows = x.size(0), cols = x.numel() / rows;
  softmax_kernel<<<dim3((unsigned)rows), dim3(256), 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
}
'''

EXT = load_inline(
    name="softmax_hip_v1",
    cpp_sources="void softmax_entry(torch::Tensor x, torch::Tensor y);",
    cuda_sources=_SRC,
    functions=["softmax_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        EXT.softmax_entry(x, y)
        return y
