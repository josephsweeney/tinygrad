import os
os.environ["METAL"] = "1"
import time
import numpy as np
from tinygrad.helpers import dtypes, getenv
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram

"""
The idea is to have the each kernel call be for a single tile.

D_t = (BD)B^T <- This will be done for each kernel call.

F_t = (GF)G^T <- This filter can be computed once and used everywhere.

O_t = F_t (.) D_t <- Elementwise multiply, just a for loop.

O = (AO_t)A^T

"""

XSIZE = 2048
YSIZE = 1024

# TODO: Get data from image
output = RawMetalBuffer((XSIZE - 2)*(YSIZE - 2), dtypes.float32)
debug = RawMetalBuffer(16, dtypes.float32)

n_data = np.random.default_rng().standard_normal(size=(YSIZE, XSIZE), dtype=np.float32) #.astype(np.int32).astype(np.float32)

# Box Blur
n_conv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) * (1/9)

A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)
B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
G = np.array([[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]], dtype=np.float32)
n_filter = np.matmul(np.matmul(G, n_conv), G.T)
data = RawMetalBuffer.fromCPU(n_data)

# FLOPS = N*N*N*2
# BW = N*N*3*4

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
constant float4x4 F_t = {shader_str(n_filter)};

kernel void conv(device float *out, device const float *data, uint3 gid [[thread_position_in_grid]]) {{
  data += (gid.x * 2) + ({XSIZE} * gid.y * 2);
  float4x4 D;
  for (int i = 0; i < 4; ++i) {{
    D[i] = float4(data[0], data[1], data[2], data[3]); 
    data += {XSIZE};
  }}
  float4x4 D_t = ((B * D) * BT);
  float4x4 O_t = float4x4((F_t[0]*D_t[0]), (F_t[1]*D_t[1]), (F_t[2]*D_t[2]), (F_t[3]*D_t[3]));
  float2x2 O = (A * O_t) * AT;
  uint index = (gid.x * 2) + ({XSIZE - 2} * gid.y * 2);
  out[index]               = O[0][0];
  out[index+1]             = O[0][1];
  out[index+{XSIZE - 2}]   = O[1][0];
  out[index+{XSIZE - 2}+1] = O[1][1];
}}""")
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(lambda: prog([((XSIZE-2)//2), ((YSIZE-2)//2), 1], [1, 1, 1], output, data, wait=True)) for _ in range(20)])
local_output = output.toCPU().reshape(YSIZE-2,XSIZE-2)
print(f"time: {tm}")

def winograd_np(data):
    output = np.zeros(((YSIZE - 2), (XSIZE - 2)), dtype=np.float32)
    for y in range(0, YSIZE-3, 2):
        for x in range(0, XSIZE-3, 2):
            O_t = np.multiply(np.matmul(np.matmul(B, data[y:y+4, x:x+4]), B.T), n_filter)
            O = np.matmul(np.matmul(A, O_t), A.T)
            output[y:y+2, x:x+2] = O
    return output

cpu_out = winograd_np(n_data)
np.testing.assert_allclose(local_output, cpu_out, atol=1e-3)
# TODO: Compute FLOPs, and estimate optimal FLOPS.
# TODO: Imitate tinygrad and pytorch API for shader