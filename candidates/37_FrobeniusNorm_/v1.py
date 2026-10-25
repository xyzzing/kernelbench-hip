# 37_FrobeniusNorm v1 — sum-of-squares reduce (block reduce + one atomic per
# block) then a scale-by-rsqrt pass. Declaration order fixed: the scale kernel
# is defined before the entry that launches it. Eager 26.7 ms / compiled 25.8 ms;
# traffic is identical (read 7.5 + read 7.5 + write 7.5 GiB) — expected tie.
ENTRY = "frobenius_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void scale_with_norm_kernel(const float* x, float* y,
                                       const float* sumsq, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * rsqrtf(*sumsq);
}

__global__ void sumsq_kernel(const float* x, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < n; i += stride) {
    float v = x[i];
    acc += v * v;
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

void frobenius_entry(torch::Tensor x, torch::Tensor y, torch::Tensor scratch) {
  long n = x.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  hipMemsetAsync(scratch.data_ptr<float>(), 0, sizeof(float), stream);
  long blocks = (n + 255) / 256;
  if (blocks > 4096) blocks = 4096;  // grid-stride inside; atomic cap
  sumsq_kernel<<<dim3((unsigned)blocks), dim3(256), 0, stream>>>(
      x.data_ptr<float>(), scratch.data_ptr<float>(), n);
  scale_with_norm_kernel<<<dim3((unsigned)((n + 255) / 256)), dim3(256), 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), scratch.data_ptr<float>(), n);
}
'''

EXT = load_inline(
    name="frobenius_hip_v1",
    cpp_sources="void frobenius_entry(torch::Tensor x, torch::Tensor y, torch::Tensor scratch);",
    cuda_sources=_SRC,
    functions=["frobenius_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        scratch = torch.zeros(1, device=x.device, dtype=torch.float32)
        EXT.frobenius_entry(x, y, scratch)
        return y
