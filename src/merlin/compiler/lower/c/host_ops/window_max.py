"""Scalar BF16 ordered window maximum (max pooling) on integer carriers."""

from math import prod

from merlin.compiler.frontend.patterns.linalg_window_max import INPUT_DOMAIN, NUMERIC_SEMANTICS
from merlin.compiler.semantic.model import IndexMap, TensorType

from ..dispatch import c_source_pair


def emit_bf16_window_max_c(source, name):
    """Emit the declared scan: per window in loop order, replace when strictly greater.

    Ordering compares BF16 values through a signed magnitude key, so -0 and +0
    tie and the first is kept. Reads outside the input are -inf padding and are
    skipped. Any NaN input returns -2 before an output write.
    """
    pure = [n for n in source.nodes if n.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("BF16 window maximum needs one dense logical computation")
    node = pure[0]
    attrs = dict(node.attrs)
    output_map = IndexMap(6, tuple(tuple(int(a == b) for b in range(6)) for a in range(4)), (0,) * 4)
    if (
        node.op != "bf16_window_max"
        or len(node.inputs) != 1
        or source.outputs != (node.id,)
        or len(source.nodes) != 2
        or set(attrs) != {"input_domain", "numeric_semantics", "window_h", "window_w"}
        or attrs["numeric_semantics"] != NUMERIC_SEMANTICS
        or attrs["input_domain"] != INPUT_DOMAIN
        or len(node.type.shape) != 4
        or node.type != TensorType(node.type.shape, "bf16", "raw-bit-pattern")
        or len(node.index_maps) != 2
        or node.index_maps[1] != output_map
        or node.index_maps[0].loop_rank != 6
        or len(node.index_maps[0].coefficients) != 4
    ):
        raise ValueError("BF16 window maximum source type, indexing, or policy differs")
    operand = source.node(node.inputs[0])
    shape = operand.type.shape
    if (
        operand.type != TensorType(shape, "bf16", "raw-bit-pattern")
        or len(shape) != 4
        or operand.effect not in {"input", "constant"}
        or operand.attrs
        or operand.index_maps
        or shape[:2] != node.type.shape[:2]
    ):
        raise ValueError("BF16 window maximum operand differs")
    rows, offsets = node.index_maps[0].coefficients, node.index_maps[0].offsets
    sh, dh, sw, dw = rows[2][2], rows[2][4], rows[3][3], rows[3][5]
    ph, pw = -offsets[2], -offsets[3]
    expected = ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, sh, 0, dh, 0), (0, 0, 0, sw, 0, dw))
    if rows != expected or offsets[:2] != (0, 0) or min(sh, sw, dh, dw) < 1 or min(ph, pw) < 0:
        raise ValueError("BF16 window maximum needs a strided, dilated, padded NCHW window map")
    if source.constants:
        raise ValueError("BF16 window maximum admits only a runtime input")
    n, c, height, width = shape
    oh, ow = node.type.shape[2:]
    kh, kw = attrs["window_h"], attrs["window_w"]
    in_bytes, out_bytes = prod(shape) * 2, prod(node.type.shape) * 2
    if (
        max(in_bytes, out_bytes) > 2**31
        or any(type(v) is not int or v < 1 for v in (kh, kw))
        or max(kh, kw, sh, sw, dh, dw, ph, pw) > 2**31
        or (oh - 1) * sh + (kh - 1) * dh > 2**31
        or (ow - 1) * sw + (kw - 1) * dw > 2**31
    ):
        raise ValueError("BF16 window maximum exceeds the 32-bit index domain")
    code = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={max(in_bytes, out_bytes)}ULL,"BF16 window maximum host types");
static int32_t {name}_key(unsigned bits) {{
  int32_t magnitude=(int32_t)(bits&32767u);
  return (bits&32768u)?-magnitude:magnitude;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{in_bytes}u || b>UINTPTR_MAX-{out_bytes}u || (a<b+{out_bytes}u && b<a+{in_bytes}u)) return -1;
  const unsigned char *x=(const unsigned char *)inputs[0];
  unsigned char *y=(unsigned char *)outputs[0];
  for (size_t i=0;i<{in_bytes}u;i+=2u) if ((((unsigned)x[i]|((unsigned)x[i+1]<<8))&32767u)>32640u) return -2;
  for (long plane=0;plane<{n * c}L;++plane) for (long r=0;r<{oh}L;++r) for (long q=0;q<{ow}L;++q) {{
    unsigned best=65408u;
    for (long i=0;i<{kh}L;++i) for (long j=0;j<{kw}L;++j) {{
      long row=r*{sh}L+i*{dh}L-{ph}L,col=q*{sw}L+j*{dw}L-{pw}L;
      if (row<0 || row>={height}L || col<0 || col>={width}L) continue;
      size_t at=2u*(size_t)((plane*{height}L+row)*{width}L+col);
      unsigned bits=(unsigned)x[at]|((unsigned)x[at+1]<<8);
      if ({name}_key(bits)>{name}_key(best)) best=bits;
    }}
    size_t out=2u*(size_t)((plane*{oh}L+r)*{ow}L+q);
    y[out]=(unsigned char)(best&255u);
    y[out+1]=(unsigned char)(best>>8);
  }}
  return 0;
}}
"""
    return dict(
        source_request_digest=source.digest(),
        implementation=name,
        inputs=[operand.id],
        outputs=[node.id],
        numeric_semantics=NUMERIC_SEMANTICS,
        input_domain=INPUT_DOMAIN,
    ), c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"window_max.c": code},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
