"""Scalar BF16 ReLU as the checked ordered select x < +0 ? +0 : x."""

from math import prod

from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps
from ..dispatch import c_source_pair


SELECT_ATTRS = {
    "input_domain": "all-bf16-bit-patterns",
    "numeric_semantics": "bf16-ordered-lt-positive-zero-select-v1",
}

def emit_bf16_relu_c(source, name):
    pure = [n for n in source.nodes if n.effect == 'pure']
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('BF16 ReLU needs one dense logical computation')
    node = pure[0]
    if (node.op != 'relu' or len(node.inputs) != 1 or source.outputs != (node.id,)
            or not node.type.shape or node.type != TensorType(node.type.shape, 'bf16', 'raw-bit-pattern')
            or dict(node.attrs) != SELECT_ATTRS
            or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
            or len(source.nodes) != 2):
        raise ValueError('BF16 ReLU source type, indexing, or policy differs')
    operand = source.node(node.inputs[0])
    if (operand.type != node.type or operand.effect not in {'input', 'constant'}
            or operand.attrs or operand.index_maps):
        raise ValueError('BF16 ReLU operand differs')
    count = prod(node.type.shape)
    size = count * 2
    if size > 2**63 - 1:
        raise ValueError('BF16 ReLU exceeds signed byte-size domain')
    for binding in source.constants:
        if (binding.node_id != operand.id or binding.encoding != 'bf16-le-raw'
                or len(bytes.fromhex(binding.data_hex)) != size):
            raise ValueError('BF16 ReLU constant encoding differs')
    code = f'''#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={size}ULL,"BF16 ReLU host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
  const unsigned char *x=(const unsigned char *)inputs[0];
  unsigned char *y=(unsigned char *)outputs[0];

  for (size_t i=0;i<{count}u;++i) {{
    unsigned magnitude=((unsigned)x[2*i]|((unsigned)x[2*i+1]<<8))&32767u;
    int clamp=(x[2*i+1]&128u) && magnitude>0u && magnitude<=32640u;
    y[2*i]=clamp?0:x[2*i];
    y[2*i+1]=clamp?0:x[2*i+1];
  }}
  return 0;
}}
'''
    return dict(source_request_digest=source.digest(), implementation=name,
                inputs=[operand.id], outputs=[node.id], **SELECT_ATTRS), c_source_pair(
        name, public_headers={}, private_headers={}, sources={'relu.c': code},
        declarations=f'int {name}(void *, const void *const *, void *const *);\n')
