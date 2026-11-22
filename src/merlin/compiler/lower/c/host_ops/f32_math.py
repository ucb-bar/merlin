"""Checked binary32 SiLU, last-axis sum, and reciprocal square root.

SiLU and rsqrt reproduce the arithmetic the pinned torch 2.11.0+cpu wheel runs for
the source operation; the sequences are fixed binary32 operations, so results do
not depend on the CPU that builds or runs them. SiLU is x / RN(1 + e) with e from
the wheel's vector exponential (SLEEF expf, 1.0 ULP, fused steps). rsqrt is
RN(1 / RN(sqrt(x))). The sum is the declared order torch-2.11-cpu-f32-sum-v1, an
eight-lane cascade over the last axis, exact only against that order: torch eager
equals it when its operand is contiguous along the reduced axis and is not one row
of 32,768 or more elements, because eager's order depends on layout and threads.
Inputs or results outside the declared domain refuse before any store.
"""

from math import prod
from struct import pack, unpack

from merlin.compiler.semantic.linalg.source_compositions import FINITE_F32, SILU_SEMANTICS, f32_sum_shape_admitted
from merlin.compiler.semantic.linalg.linalg_recognition import _f32_reduce_maps
from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps

from ..dispatch import c_source_pair
from .gelu import ENVIRONMENT_CHECK, literal

RSQRT_SEMANTICS = "source-f32-rsqrt-unqualified-v1"
SUM_SEMANTICS = "source-f32-add-unqualified-v1"
SUM_ORDER = "torch-2.11-cpu-f32-sum-v1"
_MAX_ROW = 1 << 24


def _bits(value):
    return unpack("<I", pack("<f", value))[0]


_EXP = {key: literal(_bits(value)) for key, value in {
    "inverse_log2": 1.442695040888963407359924681001892137426645954152985934135449406931,
    "log2_upper": 0.693145751953125, "log2_lower": 1.428606765330187045e-06,
    "c5": 0.000198527617612853646278381, "c4": 0.00139304355252534151077271, "c3": 0.00833336077630519866943359,
    "c2": 0.0416664853692054748535156, "c1": 0.166666671633720397949219, "half": 0.5, "one": 1.0,
    "low": -104.0, "high": 100.0}.items()}


def _prelude(what, largest):
    return f"""#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__)
#error {what} requires strict floating point
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_MANT_DIG==24 && FLT_RADIX==2 &&
               FLT_MAX_EXP==128 && SIZE_MAX>={largest}ULL,"binary32 host representation required");
"""


def _load(name):
    return f"""static float {name}_load(const unsigned char *p) {{ float x; memcpy(&x,p,4); return x; }}
static uint32_t {name}_bits(const unsigned char *p) {{ uint32_t x; memcpy(&x,p,4); return x; }}
"""


def _guards(input_bytes, output_bytes):
    return f"""  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{input_bytes}ULL || b>UINTPTR_MAX-{output_bytes}ULL ||
      (a<b+{output_bytes}ULL && b<a+{input_bytes}ULL)) return -1;
{ENVIRONMENT_CHECK}
  const unsigned char *input=(const unsigned char *)inputs[0];
  unsigned char *output=(unsigned char *)outputs[0];"""


def _single(source, operation):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError(f"{operation} needs one dense logical computation")
    node = pure[0]
    if node.op != operation or len(node.inputs) != 1 or source.outputs != (node.id,) or len(source.nodes) != 2:
        raise ValueError(f"{operation} needs one declared source operation")
    operand = source.node(node.inputs[0])
    if (operand.effect != "input" or operand.attrs or operand.index_maps or source.constants or not node.type.shape
            or node.type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
            or operand.type.dtype != "f32" or operand.type.numerical_policy != "raw-bit-pattern"):
        raise ValueError(f"{operation} type, operand, or constants differ")
    return node, operand


def _result(source, name, node, operand, semantics, domain, filename, code):
    record = dict(source_request_digest=source.digest(), implementation=name, inputs=[operand.id], outputs=[node.id],
                  numeric_semantics=semantics, input_domain=domain, qualification="requires-exact-receipt")
    return record, c_source_pair(name, public_headers={}, private_headers={}, sources={filename: code},
                                 declarations=f"int {name}(void *, const void *const *, void *const *);\n")


def emit_f32_silu_c(source, name):
    node, operand = _single(source, "f32_silu")
    if (operand.type != node.type or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            or dict(node.attrs) != {"input_domain": FINITE_F32, "numeric_semantics": SILU_SEMANTICS}):
        raise ValueError("f32 SiLU type, indexing, or semantics differ")
    count = prod(node.type.shape)
    c = _EXP
    code = _prelude("f32 SiLU", count * 4) + _load(name) + f"""static float {name}_fma(float a,float b,float c) {{
#if defined(__riscv) && __riscv_xlen == 64
  float r; __asm__ ("fmadd.s %0, %1, %2, %3, rne" : "=f" (r) : "f" (a), "f" (b), "f" (c)); return r;
#elif defined(__x86_64__) && defined(__FMA__)
  __asm__ ("vfmadd231ss %2, %1, %0" : "+x" (c) : "x" (a), "x" (b)); return c;
#else
  return __builtin_fmaf(a,b,c);
#endif
}}
static float {name}_pow2(int32_t q) {{ uint32_t b=(uint32_t)(q+127)<<23; float x; memcpy(&x,&b,4); return x; }}
static float {name}_exp(float d) {{ /* SLEEF expf, 1.0 ULP, as the vector kernel runs it */
  if (d<{c["low"]}) return 0.0f;
  if (d>{c["high"]}) {{ uint32_t b=0x7f800000U; float x; memcpy(&x,&b,4); return x; }}
  volatile float p=d*{c["inverse_log2"]};
  int32_t q=(int32_t)p;
  volatile float f=p-(float)q;
  if (f>{c["half"]} || (f=={c["half"]} && (q&1))) ++q;
  else if (f<-{c["half"]} || (f==-{c["half"]} && (q&1))) --q;
  float n=(float)q;
  float s={name}_fma(n,-{c["log2_upper"]},d);
  s={name}_fma(n,-{c["log2_lower"]},s);
  float u={c["c5"]};
  u={name}_fma(u,s,{c["c4"]});
  u={name}_fma(u,s,{c["c3"]});
  u={name}_fma(u,s,{c["c2"]});
  u={name}_fma(u,s,{c["c1"]});
  u={name}_fma(u,s,{c["half"]});
  volatile float square=s*s;
  volatile float tail={name}_fma(square,u,s);
  volatile float y={c["one"]}+tail;
  int32_t h=q>>1;
  y=y*{name}_pow2(h);
  y=y*{name}_pow2(q-h);
  return y;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
{_guards(count * 4, count * 4)}
  for (size_t i=0;i<{count}ULL;++i) if (({name}_bits(input+4*i)&0x7f800000U)==0x7f800000U) return -2;
  for (size_t i=0;i<{count}ULL;++i) {{
    float x={name}_load(input+4*i);
    volatile float e={name}_exp(-x);
    volatile float d={c["one"]}+e;
    volatile float y=x/d;
    float result=y; memcpy(output+4*i,&result,4);
  }}
  return 0;
}}
"""
    return _result(source, name, node, operand, SILU_SEMANTICS, FINITE_F32, "f32_silu.c", code)


def emit_f32_rsqrt_c(source, name):
    node, operand = _single(source, "source_f32_rsqrt_unqualified")
    if (operand.type != node.type or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            or dict(node.attrs) != {"numeric_semantics": RSQRT_SEMANTICS, "source_operation": "aten.rsqrt.default"}):
        raise ValueError("f32 rsqrt type, indexing, or semantics differ")
    count = prod(node.type.shape)
    code = _prelude("f32 rsqrt", count * 4) + _load(name) + f"""static float {name}_sqrt(float x) {{
  float r;
#if defined(__riscv)
  __asm__ ("fsqrt.s %0, %1, rne" : "=f" (r) : "f" (x));
#elif defined(__x86_64__)
  __asm__ ("sqrtss %1, %0" : "=x" (r) : "x" (x));
#else
#error f32 rsqrt needs an admitted IEEE square root
#endif
  return r;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
{_guards(count * 4, count * 4)}
  for (size_t i=0;i<{count}ULL;++i) {{
    uint32_t bits={name}_bits(input+4*i);
    if ((bits&0x7f800000U)==0x7f800000U || ((bits>>31) && (bits&0x7fffffffU))) return -2;
  }}
  for (size_t i=0;i<{count}ULL;++i) {{
    volatile float root={name}_sqrt({name}_load(input+4*i));
    volatile float y=1.0f/root;
    float result=y; memcpy(output+4*i,&result,4);
  }}
  return 0;
}}
"""
    return _result(source, name, node, operand, RSQRT_SEMANTICS, "finite-nonnegative-f32", "f32_rsqrt.c", code)


def emit_f32_row_sum_c(source, name):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("f32 sum needs one dense logical computation")
    node = pure[0]
    if node.op != "source_f32_reduce_add_unqualified" or len(node.inputs) != 2 or source.outputs != (node.id,):
        raise ValueError("f32 sum needs one declared source reduction")
    operand, seed = (source.node(key) for key in node.inputs)
    shape = operand.type.shape
    rank = len(shape)
    binding = next((row for row in source.constants if row.node_id == seed.id), None)
    if (operand.effect != "input" or operand.attrs or operand.index_maps or len(source.nodes) != 3
            or operand.type != TensorType(shape, "f32", "raw-bit-pattern") or rank < 2
            or node.type != TensorType(shape[:-1], "f32", "raw-bit-pattern") or not 1 <= shape[-1] <= _MAX_ROW
            or not f32_sum_shape_admitted(shape)
            or seed.type != node.type or seed.effect != "constant" or binding is None
            or binding.encoding != "f32-le-raw" or any(bytes.fromhex(binding.data_hex))
            or node.index_maps != _f32_reduce_maps(rank, (rank - 1,))
            or dict(node.attrs) != {"axis": rank - 1, "reduction_order": "unresolved",
                                    "numeric_semantics": SUM_SEMANTICS, "scalar_add_order": "input-accumulator",
                                    "source_operation": "aten.mean.dim"}):
        raise ValueError("f32 sum needs a +0 seed over the last axis of a dense tensor, not one row of 32,768 or more")
    rows, length = prod(shape[:-1]), shape[-1]
    code = _prelude("f32 sum", rows * length * 4) + _load(name) + f"""static void {name}_partials(const unsigned char *p,size_t stride,size_t size,float out[4]) {{
  unsigned log=0; while (((size_t)1<<log)<size) ++log;
  unsigned power=log/4>4?log/4:4;
  size_t step=(size_t)1<<power,mask=step-1,i=0;
  float s[4][4]={{{{0.0f}}}};
  while (i+step<=size) {{
    for (size_t j=0;j<step;++j,++i) for (unsigned k=0;k<4;++k) s[0][k]=s[0][k]+{name}_load(p+(i*4+k)*stride);
    for (unsigned level=1;level<4;++level) {{
      for (unsigned k=0;k<4;++k) {{ s[level][k]=s[level][k]+s[level-1][k]; s[level-1][k]=0.0f; }}
      if (i&(mask<<(level*power))) break;
    }}
  }}
  for (;i<size;++i) for (unsigned k=0;k<4;++k) s[0][k]=s[0][k]+{name}_load(p+(i*4+k)*stride);
  for (unsigned level=1;level<4;++level) for (unsigned k=0;k<4;++k) s[level][k]=s[level][k]+s[level-1][k];
  for (unsigned k=0;k<4;++k) out[k]=s[3][k];
}}
static float {name}_cascade(const unsigned char *p,size_t stride,size_t n) {{
  float q[4]; size_t blocks=n/4;
  {name}_partials(p,stride,blocks,q);
  for (size_t i=blocks*4;i<n;++i) q[0]=q[0]+{name}_load(p+i*stride);
  volatile float left=q[0]+q[1]; left=left+q[2]; left=left+q[3];
  return left;
}}
static float {name}_sum(const unsigned char *row) {{
  size_t count={length}ULL/8;
  if (!count) return {name}_cascade(row,4,{length}ULL);
  float lane[8];
  for (unsigned j=0;j<8;++j) lane[j]={name}_cascade(row+4*j,32,count);
  if ({length}ULL%8) {{
    volatile float tail={name}_load(row+count*32);
    for (size_t t=count*8+1;t<{length}ULL;++t) tail=tail+{name}_load(row+4*t);
    lane[0]=lane[0]+tail;
  }}
  volatile float total=lane[0];
  for (unsigned j=1;j<8;++j) total=total+lane[j];
  return total;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
{_guards(rows * length * 4, rows * 4)}
  for (size_t i=0;i<{rows * length}ULL;++i) if (({name}_bits(input+4*i)&0x7f800000U)==0x7f800000U) return -2;
  for (size_t r=0;r<{rows}ULL;++r) {{
    float value={name}_sum(input+r*{length * 4}ULL); uint32_t bits; memcpy(&bits,&value,4);
    if ((bits&0x7f800000U)==0x7f800000U) return -2;
  }}
  for (size_t r=0;r<{rows}ULL;++r) {{
    float result={name}_sum(input+r*{length * 4}ULL); memcpy(output+4*r,&result,4);
  }}
  return 0;
}}
"""
    return _result(source, name, node, operand, SUM_ORDER, FINITE_F32, "f32_row_sum.c", code)
