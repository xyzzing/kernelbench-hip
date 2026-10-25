# 94_MSELoss v1 — scalar mean((p - t)^2) over 1.07G elems. Eager runs a 3-pass
# chain (sub + pow + mean = ~21 GB traffic, 31.2 ms); compiled fuses to 9.7 ms.
# Ours: one grid-stride reduce pass (8.6 GB read), atomicAdd per block into a
# scalar, then a divide micro-kernel. fp32 atomics vs torch tree reduce: the
# parity bound atol=1e-2 on a ~0.33-magnitude scalar is far above fp32 noise.
ENTRY = "mse_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void mse_sum_kernel(const float* p, const float* t, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < n; i += stride) {
    float d = p[i] - t[i];
    acc += d * d;
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, sm[0]);
}

__global__ void mse_div_kernel(float* out, float inv_n) {
  out[0] *= inv_n;
}

void mse_entry(torch::Tensor p, torch::Tensor t, torch::Tensor out) {
  long n = p.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  hipMemsetAsync(out.data_ptr<float>(), 0, sizeof(float), stream);
  long blocks = (n + 255) / 256;
  if (blocks > 8192) blocks = 8192;
  mse_sum_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
      p.data_ptr<float>(), t.data_ptr<float>(), out.data_ptr<float>(), n);
  mse_div_kernel<<<dim3(1), dim3(1), 0, stream>>>(
      out.data_ptr<float>(), 1.0f / (float)n);
}
'''

EXT = load_inline(
    name="mse_hip_v1",
    cpp_sources="void mse_entry(torch::Tensor p, torch::Tensor t, torch::Tensor out);",
    cuda_sources=_SRC,
    functions=["mse_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, predictions, targets):
        out = torch.empty((), device=predictions.device, dtype=torch.float32)
        EXT.mse_entry(predictions, targets, out)
        return out
