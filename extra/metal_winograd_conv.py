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
assert(CIN==COUT) # For simplicity for now, can add groups later.
FLOPS = BS*K*K*CIN*HW*HW*COUT*2

output = RawMetalBuffer(BS*COUT*HW*HW, dtypes.float32)

n_data = np.random.default_rng().standard_normal(size=(BS,CIN,HW,HW), dtype=np.float32)

# Box Blur
# n_conv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) * (1/9)
n_conv = np.random.default_rng().standard_normal(size=(COUT,CIN,K,K), dtype=np.float32)

A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)
B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
G = np.array([[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]], dtype=np.float32)
n_filter = np.array([[np.matmul(np.matmul(G, conv), G.T) for conv in c_conv] for c_conv in n_conv], dtype=np.float32)
#TODO: Add padding before making metal buffer
data = RawMetalBuffer.fromCPU(n_data)
filter_buf = RawMetalBuffer.fromCPU(n_filter)

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
  uint batch = gid.x;
  uint cout = gid.y/{CIN};
  uint cin = gid.y%{COUT};
  filter += (cout * {CIN} * {K*K}) + (cin * {K*K});
  float4x4 F_t = float4x4(filter[0],  filter[1],  filter[2],  filter[3],
                          filter[4],  filter[5],  filter[6],  filter[7], 
                          filter[8],  filter[9],  filter[10], filter[11], 
                          filter[12], filter[13], filter[14], filter[15]);
  uint x = (gid.z%{HW})*2;
  uint y = (gid.z/{HW})*2;
  data += batch * ({CIN*HW*HW}); // Get to the right batch
  data += cin * ({HW*HW}); // Assume parallel channels rather than interspersed for simplicity
  data += x + (y * {HW});
  float4x4 D;
  for (int i = 0; i < 4; ++i) {{
      D[i] = float4(data[0], data[1], data[2], data[3]); 
      data += {HW};
  }}
  float4x4 D_t = ((B * D) * BT);
  float4x4 O_t = float4x4((F_t[0]*D_t[0]), (F_t[1]*D_t[1]), (F_t[2]*D_t[2]), (F_t[3]*D_t[3]));
  float2x2 O = (A * O_t) * AT;
  out += (batch * {COUT*HW*HW}) + (cout * {HW*HW});
  uint index = x + (y * {HW});
  out[index]        = O[0][0];
  out[index+1]      = O[0][1];
  out[index+{HW}]   = O[1][0];
  out[index+{HW}+1] = O[1][1];
}}""")
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(lambda: prog([BS, COUT*CIN, HW*HW], [1,1,1], output, data, filter_buf, wait=True)) for _ in range(20)])
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