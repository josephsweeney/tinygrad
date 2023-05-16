import os
os.environ["METAL"] = "1"
import time
import numpy as np
from tinygrad.helpers import dtypes, getenv
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram

BS = 64
CIN = 1
COUT = 1
HW = 32
K = 3
assert(K==3) # This implementation only supports 3x3 filters.
assert(CIN==COUT) # For simplicity for now, can add groups later.
assert(CIN==1) # @BUG: I think pytorch has a different model for channels. Investigate further.
FLOPS = BS*K*K*CIN*HW*HW*COUT*2

A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)
B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
G = np.array([[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]], dtype=np.float32)
n_data = np.random.default_rng().standard_normal(size=(BS,CIN,HW,HW), dtype=np.float32)
n_conv = np.random.default_rng().standard_normal(size=(COUT,CIN,K,K), dtype=np.float32)
n_filter = np.array([[np.matmul(np.matmul(G, conv), G.T) for conv in c_conv] for c_conv in n_conv], dtype=np.float32)
padded = np.pad(n_data, [(0,0), (0,0), (1,1), (1,1)])
INHW = HW+2 # Due to padding
data = RawMetalBuffer.fromCPU(padded)
filter_buf = RawMetalBuffer.fromCPU(n_filter)
output = RawMetalBuffer(BS*COUT*HW*HW, dtypes.float32)

def shader_str(arr):
    return f"float{arr.shape[1]}x{arr.shape[0]}(" + ','.join(map(str, arr.T.flatten())) + ")" # .T is needed because shader is column major

prog = MetalProgram("conv", f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
using namespace metal;

constant float4x2 A =  {shader_str(A)};
constant float2x4 AT = {shader_str(A.T)};
constant float4x4 B =  {shader_str(B)};
constant float4x4 BT = {shader_str(B.T)};

kernel void conv(device float *out, device const float *data, device const float *filter, uint3 gid [[thread_position_in_grid]]) {{
  uint x = (gid.y % {HW//2}) * 2;
  uint y = (gid.y / {HW//2}) * 2;
  if (x > {INHW-3} || y > {INHW-3}) {{return;}}
  uint cin = gid.x % {CIN};
  uint cout = gid.x / {CIN} % {COUT};
  uint batch = gid.x / {CIN*COUT};
  device const float *f = filter + (cout * {CIN*K*K}) + (cin * {K*K});
  float4x4 F_t = float4x4(f[0],  f[1],  f[2],  f[3],
                          f[4],  f[5],  f[6],  f[7], 
                          f[8],  f[9],  f[10], f[11], 
                          f[12], f[13], f[14], f[15]);
  device const float *d = data + (batch * {CIN*INHW*INHW}) + (cin * {INHW*INHW}) + (y * {INHW}) + x;
  float4x4 D;
  for (int i = 0; i < 4; ++i) {{
      D[i] = float4(d[0], d[1], d[2], d[3]);
      d += {INHW};
  }}
  float4x4 D_t = ((B * D) * BT);
  float4x4 O_t = float4x4((F_t[0]*D_t[0]), (F_t[1]*D_t[1]), (F_t[2]*D_t[2]), (F_t[3]*D_t[3]));
  float2x2 O = (A * O_t) * AT;
  device float *output = out + (batch * {COUT*HW*HW}) + (cout * {HW*HW}) + (y * {HW}) + x;
  *output          = O[0][0];
  *(output+1)      = O[0][1];
  *(output+{HW})   = O[1][0];
  *(output+{HW}+1) = O[1][1];
}}""")
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(lambda: prog([BS*COUT*CIN, HW*HW//4, 1], [1,1,1], output, data, filter_buf, wait=True)) for _ in range(20)])
local_output = output.toCPU().reshape(BS,COUT,HW,HW)
print(f"{tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS conv in tinygrad")

import time, torch, torch.mps
b = torch.from_numpy(n_data).to('mps')
c = torch.from_numpy(n_conv).to('mps')
def torch_prog(b, c):
  st = time.perf_counter()
  a = torch.nn.functional.conv2d(b, c, padding=1)
  torch.mps.synchronize()
  np.testing.assert_allclose(local_output, a.cpu(), atol=1e-3)
  return time.perf_counter() - st
tm = min([torch_prog(b, c) for _ in range(20)])
print(f"{tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS conv in torch")