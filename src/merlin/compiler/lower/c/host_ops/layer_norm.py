"""Checked BF16 layer norm as the pinned torch 2.11.0+cpu wheel's eager CPU kernel computes it.

On AVX2 and AVX512 hosts that wheel dispatches the BF16 layer_norm kernel with
8-lane binary32 halves of 16-lane BF16 vectors and fuses 7 multiply-adds: the
inner Welford m1 and m2, the AddMomentsVec m2, the scalar AddMoments m1 and sum,
and both output steps. The AddMomentsVec m1 update and the scalar tail m2 update
round the product and the sum separately. This emitter computes RowwiseMoments'
chunked Welford moments, cascade and lane combination, var = m2 / N,
rstd = 1 / sqrt(var + eps) with two binary32 roundings, and
fma(fma(x, rstd, -mean * rstd), weight, bias), where an absent weight is 1 and an
absent bias is 0, then one BF16 RNE. Each fused step is one IEEE fused
multiply-add instruction; every other step rounds once. torch's DEFAULT capability
and other builds are not matched. sqrt is the IEEE instruction the host-math
supplier also uses. A first pass evaluates every row without stores and refuses
non-finite inputs, statistics or binary32 results; the second pass stores.
"""

from math import prod
from struct import pack

from merlin.compiler.semantic.linalg.source_compositions import FINITE_BF16, LAYER_NORM_SEMANTICS, row_reduction_maps
from merlin.compiler.semantic.model import TensorType

from ..dispatch import c_source_pair
from .gelu import ENVIRONMENT_CHECK, literal

ROLES = ("weight", "bias")


LANES = 8  # 8-lane binary32 halves; a 16-lane BF16 vector holds two
CHUNK = 16  # torch kChunkSize


def emit_bf16_layer_norm_c(source, name):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 layer norm needs one dense logical computation")
    node = pure[0]
    attrs = dict(node.attrs)
    roles = tuple(role for role in str(attrs.get("affine_operands", "")).split(",") if role)
    shape, rank = node.type.shape, len(node.type.shape)
    k = attrs.get("normalized_rank")
    if (
        node.op != "bf16_layer_norm"
        or source.outputs != (node.id,)
        or set(attrs) != {"affine_operands", "epsilon_f32_bits", "input_domain", "normalized_rank", "numeric_semantics"}
        or attrs["numeric_semantics"] != LAYER_NORM_SEMANTICS
        or attrs["input_domain"] != FINITE_BF16
        or roles not in {(), ("weight",), ("bias",), ROLES}
        or type(k) is not int or not 0 < k < rank
        or len(node.inputs) != 1 + len(roles)
        or len(set(node.inputs)) != len(node.inputs)
        or node.type != TensorType(shape, "bf16", "raw-bit-pattern")
    ):
        raise ValueError("BF16 layer norm source type, operands, or policy differs")
    if node.index_maps != row_reduction_maps(rank, k, len(roles)):
        raise ValueError("BF16 layer norm indexing differs from whole-row and affine maps")
    epsilon = attrs["epsilon_f32_bits"]
    if type(epsilon) is not int or not 0 <= epsilon < 0x7F800000:
        raise ValueError("BF16 layer norm epsilon must be a finite nonnegative binary32")
    x = source.node(node.inputs[0])
    if x.type != node.type or x.effect != "input" or x.attrs or x.index_maps:
        raise ValueError("BF16 layer norm activation operand differs")
    width, rows = prod(shape[rank - k:]), prod(shape[:rank - k])
    count_bits = int.from_bytes(pack("<f", float(width)), "little")
    constants = {binding.node_id: binding for binding in source.constants}
    inputs, pointers, statics = [x.id], {}, []
    for role, key in zip(roles, node.inputs[1:]):
        term = source.node(key)
        if term.type != TensorType(shape[rank - k:], "bf16", "raw-bit-pattern") or term.attrs or term.index_maps:
            raise ValueError("BF16 layer norm affine terms must be BF16 tensors of the normalized shape")
        if term.effect == "constant":
            binding = constants.get(key)
            if binding is None or binding.encoding != "bf16-le-raw" or len(binding.data_hex) != 4 * width:
                raise ValueError("BF16 layer norm constant term encoding differs")
            data = bytes.fromhex(binding.data_hex)
            if any((data[i] | data[i + 1] << 8) & 0x7F80 == 0x7F80 for i in range(0, len(data), 2)):
                raise ValueError("BF16 layer norm constant terms must be finite")
            statics.append(f"static const unsigned char {name}_{role}[]={{{','.join(map(str, data))}}};")
            pointers[role] = f"{name}_{role}"
        elif term.effect == "input":
            pointers[role] = f"((const unsigned char *)inputs[{len(inputs)}])"
            inputs.append(term.id)
        else:
            raise ValueError("BF16 layer norm affine term must be an input or a constant")
    size = 2 * rows * width
    if size > 2**63 - 1:
        raise ValueError("BF16 layer norm exceeds the signed byte-size domain")
    vectors = width // (2 * LANES)
    chunks = -(-vectors // CHUNK)
    depth = (chunks - 1).bit_length() if chunks > 1 else 0
    levels = max(depth, 1)
    weights = ",".join(literal(int.from_bytes(pack("<f", 1 / (j + 1)), "little")) for j in range(CHUNK))
    weight = f"{name}_load({pointers['weight']},j)" if "weight" in pointers else "1.0f"
    bias = f"{name}_load({pointers['bias']},j)" if "bias" in pointers else "0.0f"
    overlap = "\n".join(
        f"  {{ uintptr_t t=(uintptr_t)inputs[{i}]; if (!t || t>UINTPTR_MAX-{2 * width}u || "
        f"(t<b+{size}u && b<t+{2 * width}u)) return -1; }}"
        for i in range(1, len(inputs)))
    code = f"""#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#if defined(__FAST_MATH__)
#error BF16 layer norm requires strict floating point
#endif
_Static_assert(CHAR_BIT==8 && sizeof(float)==4 && FLT_MANT_DIG==24 && FLT_RADIX==2 &&
               FLT_MAX_EXP==128 && SIZE_MAX>={size}ULL,"binary32 host representation required");
{chr(10).join(statics)}
static const float {name}_c[{CHUNK}]={{{weights}}};
static float {name}_load(const unsigned char *p,size_t i) {{
  uint32_t bits=((uint32_t)p[2*i]|((uint32_t)p[2*i+1]<<8))<<16; float v; memcpy(&v,&bits,4); return v;
}}
static int {name}_finite(float v) {{ uint32_t bits; memcpy(&bits,&v,4); return (bits&0x7f800000u)!=0x7f800000u; }}
static float {name}_sqrt(float x) {{ /* the supplier's IEEE sqrtf: one correctly rounded instruction */
  float r;
#if defined(__riscv) && __riscv_xlen == 64
  __asm__ ("fsqrt.s %0, %1, rne" : "=f" (r) : "f" (x));
#elif defined(__x86_64__)
  __asm__ ("sqrtss %1, %0" : "=x" (r) : "x" (x));
#else
#error BF16 layer norm needs an admitted IEEE square root
#endif
  return r;
}}
static float {name}_fma(float a,float b,float c) {{ /* RN(a*b+c), one IEEE fused instruction */
#if defined(__riscv) && __riscv_xlen == 64
  float r; __asm__ ("fmadd.s %0, %1, %2, %3, rne" : "=f" (r) : "f" (a), "f" (b), "f" (c)); return r;
#elif defined(__x86_64__)
  __asm__ ("vfmadd231ss %2, %1, %0" : "+x" (c) : "x" (a), "x" (b)); return c;
#else
#error BF16 layer norm needs an admitted fused multiply-add
#endif
}}
/* torch AddMomentsVec as dispatched: m1 = RN(m1 + RN(c*delta)), m2 = fma(RN(delta*m0), RN(c*delta), RN(m2+m2_add)). */
static void {name}_add_vec(long long add,const float *m1a,const float *m2a,long long *m0,float *m1,float *m2) {{
  long long n=*m0+add; volatile float c=n?(float)add/(float)n:0.0f;
  for (int l=0;l<{LANES};++l) {{
    volatile float delta=m1a[l]-m1[l], tmp=m2[l]+m2a[l], cd=c*delta, md=delta*(float)*m0;
    m1[l]=m1[l]+cd; m2[l]={name}_fma(md,cd,tmp);
  }}
  *m0=n;
}}
/* torch scalar AddMoments as dispatched: m1 = fma(c, delta, m1), m2 = RN(m2 + fma(RN(RN(delta*delta)*c), m0, m2_add)). */
static void {name}_add(long long add,float m1a,float m2a,long long *m0,float *m1,float *m2) {{
  long long n=*m0+add; volatile float c=n?(float)add/(float)n:0.0f;
  volatile float delta=m1a-*m1, dd=delta*delta, ddc=dd*c, sum={name}_fma(ddc,(float)*m0,m2a);
  *m1={name}_fma(c,delta,*m1); *m2=*m2+sum; *m0=n;
}}
/* One row: torch RowwiseMoments, rstd, and the affine update; stores only when y is not null. */
static int {name}_row(const void *const *inputs,size_t r,unsigned char *y) {{
  const unsigned char *x=(const unsigned char *)inputs[0]+2u*{width}u*r;
  long long m0s[{levels}]={{0}}; float m1s[{levels}][{LANES}]={{{{0}}}}, m2s[{levels}][{LANES}]={{{{0}}}};
  for (long long i=0;i<{chunks};++i) {{
    long long count={vectors}-i*{CHUNK}; if (count>{CHUNK}) count={CHUNK};
    float a1[2][{LANES}]={{{{0}}}}, a2[2][{LANES}]={{{{0}}}};
    for (long long j=0;j<count;++j) for (int h=0;h<2;++h) for (int l=0;l<{LANES};++l) {{
      float v={name}_load(x,(size_t)((i*{CHUNK}+j)*{2 * LANES}+h*{LANES}+l));
      volatile float delta=v-a1[h][l]; a1[h][l]={name}_fma(delta,{name}_c[j],a1[h][l]);
      volatile float rest=v-a1[h][l]; a2[h][l]={name}_fma(delta,rest,a2[h][l]);
    }}
    for (int h=0;h<2;++h) {name}_add_vec(count,a1[h],a2[h],&m0s[0],m1s[0],m2s[0]);
    long long mask=i+1;
    for (int j=1;j<{depth} && !(mask&1);++j) {{
      {name}_add_vec(m0s[j-1],m1s[j-1],m2s[j-1],&m0s[j],m1s[j],m2s[j]);
      m0s[j-1]=0; for (int l=0;l<{LANES};++l) m1s[j-1][l]=m2s[j-1][l]=0.0f;
      mask>>=1;
    }}
  }}
  for (int j=1;j<{depth};++j) {name}_add_vec(m0s[j],m1s[j],m2s[j],&m0s[0],m1s[0],m2s[0]);
  long long m0=0; float m1=0.0f, m2=0.0f;
  for (size_t i={vectors * 2 * LANES}u;i<{width}u;++i) {{
    float v={name}_load(x,i); volatile float delta=v-m1; ++m0;
    volatile float step=delta/(float)m0; m1=m1+step; volatile float rest=v-m1, grow=delta*rest; m2=m2+grow;
  }}
  for (int l=0;l<{LANES};++l) {name}_add({vectors * 2},m1s[0][l],m2s[0][l],&m0,&m1,&m2);
  volatile float variance=m2/{literal(count_bits)}, shifted=variance+{literal(epsilon)};
  if (!{name}_finite(m1) || !{name}_finite(shifted) || !(shifted>0.0f)) return -2;
  volatile float root={name}_sqrt(shifted), rstd=1.0f/root, offset=(-m1)*rstd;
  if (!{name}_finite(rstd) || !{name}_finite(offset)) return -2;
  for (size_t j=0;j<{width}u;++j) {{
    volatile float centered={name}_fma({name}_load(x,j),rstd,offset), value={name}_fma(centered,{weight},{bias});
    float out=value;
    if (!{name}_finite(out)) return -2;
    if (y) {{
      uint32_t bits; memcpy(&bits,&out,4);
      bits=(bits+0x7fffU+((bits>>16)&1U))>>16;
      y[2u*({width}u*r+j)]=(unsigned char)bits;
      y[2u*({width}u*r+j)+1u]=(unsigned char)(bits>>8);
    }}
  }}
  return 0;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
{overlap}
{ENVIRONMENT_CHECK}
  const unsigned char *x=(const unsigned char *)inputs[0];
  for (size_t i=0;i<{size}u;i+=2u) if ((((unsigned)x[i]|((unsigned)x[i+1]<<8))&32640u)==32640u) return -2;
  for (size_t r=0;r<{rows}u;++r) if ({name}_row(inputs,r,0)) return -2;
  for (size_t r=0;r<{rows}u;++r) (void){name}_row(inputs,r,(unsigned char *)outputs[0]);
  return 0;
}}
"""
    record = dict(
        source_request_digest=source.digest(),
        implementation=name,
        inputs=inputs,
        outputs=[node.id],
        numeric_semantics=LAYER_NORM_SEMANTICS,
        input_domain=FINITE_BF16,
        selected_order="pinned torch 2.11.0+cpu AVX2/AVX512 dispatch: chunked Welford moments with 7 fused and 2 rounded multiply-adds, 1/sqrt(var+eps), fma(fma(x,rstd,-mean*rstd),weight,bias)",
        host_math_symbols=[],
        scratch_policy="constant-space-two-pass-recomputation",
        workspace_bytes=0,
        qualification="requires-exact-receipt",
    )
    return record, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"layer_norm.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
