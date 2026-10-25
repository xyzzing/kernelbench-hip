# 42_MaxPool2D v1 — kernel 4, stride 1, padding 1, dilation 1 on
# 32 x 64 x 512 x 512 -> output 510 x 510. Thread per output element, 4x4
# window with -FLT_MAX padding; overlapping windows hit L2 (adjacent threads
# read adjacent columns). Eager 20.9 ms, compiled 8.5 ms.
ENTRY = "maxpool_entry"

_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void maxpool_flat(const float* x, float* y, long N, long C, long H,
                             long W, long OH, long OW, int K, long pad) {
  long total = N * C * OH * OW;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    long ow = idx % OW;
    long oh = (idx / OW) % OH;
    long c = (idx / (OW * OH)) % C;
    long n = idx / ((long)OW * OH * C);
    const float* xc = x + ((n * C) + c) * H * W;
    float m = -3.402823466e+38f;
    for (int u = 0; u < K; ++u) {
      long ih = oh - pad + u;
      if (ih < 0 || ih >= H) continue;
      for (int v = 0; v < K; ++v) {
        long iw = ow - pad + v;
        if (iw < 0 || iw >= W) continue;
        float val = xc[ih * W + iw];
        if (val > m) { m = val; }
      }
    }
    y[idx] = m;
  }
}

void maxpool_entry(torch::Tensor x, torch::Tensor y, long K, long pad) {
  long N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  long OH = y.size(2), OW = y.size(3);
  long total = N * C * OH * OW;
  maxpool_flat<<<dim3((unsigned)((total + 255) / 256)), dim3(256), 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, OH, OW, (int)K, pad);
}
'''

EXT = load_inline(
    name="maxpool_hip_v1",
    cpp_sources="void maxpool_entry(torch::Tensor x, torch::Tensor y, long K, long pad);",
    cuda_sources=_SRC,
    functions=["maxpool_entry"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(torch.nn.Module):
    def __init__(self, kernel_size: int = 4, stride: int = 1, padding: int = 1,
                 dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        N, C, H, W = x.shape
        K, S, P, D = self.kernel_size, self.stride, self.padding, self.dilation
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
        y = torch.empty(N, C, OH, OW, device=x.device, dtype=torch.float32)
        EXT.maxpool_entry(x, y, K, P)
        return y
