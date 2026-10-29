"""Exact SHA-256 of actual device bytes, optional and never speed-qualified.

The CUDA kernel deliberately implements the ordinary sequential SHA-256, not a
tree/checksum substitute. It returns only 32 bytes. Build/warm this opt-in path
before measuring; large messages may be slower than OpenSSL on a CPU. No remote
digest is accepted by this module as a substitute for reading a tensor.
"""
from functools import lru_cache
import hashlib

import torch


CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__device__ __forceinline__ unsigned r(unsigned x, int n) {return (x>>n)|(x<<(32-n));}
__device__ __constant__ unsigned K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__global__ void sha(const unsigned char* p, unsigned long long n, unsigned char* out) {
 unsigned h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 unsigned long long total=((n+9+63)/64)*64;
 for(unsigned long long base=0;base<total;base+=64) {
  unsigned w[64];
  for(int j=0;j<16;j++){unsigned v=0;for(int k=0;k<4;k++){
   auto i=base+j*4+k;unsigned b=i<n?p[i]:i==n?128:i>=total-8?((n*8)>>(8*(total-1-i)))&255:0;
   v=(v<<8)|b;}w[j]=v;}
  for(int j=16;j<64;j++)w[j]=w[j-16]+(r(w[j-15],7)^r(w[j-15],18)^(w[j-15]>>3))+w[j-7]+(r(w[j-2],17)^r(w[j-2],19)^(w[j-2]>>10));
  unsigned a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
  for(int j=0;j<64;j++){unsigned t=z+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+K[j]+w[j];
   unsigned u=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));z=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
 }
 for(int j=0;j<32;j++)out[j]=(h[j/4]>>(24-8*(j%4)))&255;
}
torch::Tensor device_sha256(torch::Tensor x) {
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kUInt8 && x.is_contiguous(),"contiguous CUDA bytes required");
 c10::cuda::CUDAGuard guard(x.device());
 auto out=torch::empty({32},x.options());
 sha<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<unsigned char>(),x.numel(),out.data_ptr<unsigned char>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
'''


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("device SHA-256 requires CUDA; CPU reference cannot qualify it")
    from torch.utils.cpp_extension import load_inline
    return load_inline(name="flyai_sha256_" + hashlib.sha256(CUDA_SOURCE.encode()).hexdigest()[:12],
        cpp_sources="torch::Tensor device_sha256(torch::Tensor x);", cuda_sources=CUDA_SOURCE,
        functions=["device_sha256"], extra_cuda_cflags=["-O3"], verbose=False)


def tensor_digest(parts, *, device=None, max_bytes=1 << 30, reference=False):
    """SHA256(concatenated bytes/tensors) with explicit bounded temporary storage."""
    if not isinstance(parts, (list, tuple)) or not parts or type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("bounded nonempty digest segments required")
    count = 0
    for part in parts:
        if not isinstance(part, bytes) and not torch.is_tensor(part):
            raise TypeError("digest segments must be bytes or actual tensors")
        count += len(part) if isinstance(part, bytes) else part.numel() * part.element_size()
    if count > max_bytes:
        raise ValueError("device digest input exceeds configured bound")
    if reference:
        if any(torch.is_tensor(p) and p.device.type != "cpu" for p in parts):
            raise ValueError("CPU digest reference accepts only CPU tensors")
        hasher = hashlib.sha256()
        for part in parts:
            hasher.update(part if isinstance(part, bytes) else part.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return hasher.digest()
    target = torch.device(device) if device is not None else next(
        (p.device for p in parts if torch.is_tensor(p) and p.is_cuda), torch.device("cpu"))
    if target.type != "cuda":
        raise ValueError("device digest needs an explicit CUDA device")
    if any(torch.is_tensor(p) and p.is_cuda and p.device != target for p in parts):
        raise ValueError("device digest cannot copy another GPU's input implicitly")
    with torch.cuda.device(target):
        pieces = [(torch.tensor(list(p), dtype=torch.uint8, device=target) if isinstance(p, bytes)
                   else p.detach().contiguous().reshape(-1).view(torch.uint8).to(target)) for p in parts]
        payload = torch.cat(pieces)
        return bytes(_extension().device_sha256(payload).cpu().tolist())


try:
    from receipt import ActualByteDigest
except ImportError:
    from shard.receipt import ActualByteDigest
