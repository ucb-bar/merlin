"""Compile explicitly selected CPU sine/cosine calls to the fixed host-math interface."""

from math import prod

from merlin.compiler.semantic.model import TensorType, pointwise_identity_maps
from merlin.development.targetgen.execution.host.admission import validate_linkage_contract

from .dispatch import c_source_pair
from merlin.runtime.host_math_supplier import HEADER_PATH, load_policy

OPERATIONS = {
    f"source_{dtype}_{operation}_unqualified": (dtype, symbol)
    for dtype, suffix in (("f32", "f"), ("f64", ""))
    for operation, symbol in (("sin", "sin" + suffix), ("cos", "cos" + suffix))
}


FIXED_FIELDS = {
    "schema": "merlin.compiled_unary_math.v1",
    "policy": "selected-host-math-rne-finite-v1",
    "isa": "rv64gc",
    "abi": "lp64d",
    "effects": "enabled-fpu-rne-unobserved-flags-and-errno-v1",
}


def validate_policy(value):
    """A library supplier pin and explicit numerical/effect permission are separate."""
    refusal = "compiled unary math needs explicit operations, RV64GC/lp64d and numerical/effect permissions"
    if not isinstance(value, dict) or set(value) != {*FIXED_FIELDS, "operations", "linkage"}:
        raise ValueError(f"{refusal}; fields must be exactly {sorted({*FIXED_FIELDS, 'operations', 'linkage'})}")
    differing = [f"{key} is {value[key]!r}, requires {expected!r}" for key, expected in FIXED_FIELDS.items()
                 if value[key] != expected]
    operations = value["operations"]
    if (
        not isinstance(operations, list)
        or not operations
        or any(not isinstance(op, str) or op not in OPERATIONS for op in operations)
        or len(set(operations)) != len(operations)
    ):
        differing.append(f"operations must be distinct names from {sorted(OPERATIONS)}")
    if differing:
        raise ValueError(f"{refusal}; " + "; ".join(differing))
    supplier = load_policy(required=[OPERATIONS[op][1] for op in value["operations"]])
    linkage = validate_linkage_contract(value["linkage"])
    expected_symbols = sorted(
        ["merlin_host_check_fp_env"]
        + [supplier["functions"][OPERATIONS[op][1]]["symbol"] for op in value["operations"]]
    )
    if linkage["symbols"] != expected_symbols:
        raise ValueError("compiled unary math library symbols differ from selected operations")
    return {**value, "operations": list(value["operations"]), "linkage": linkage}


def emit_unary_math_c(source, name, policy):
    """Keep source types and call a selected library; framework agreement is unproved.

    The caller enables the FPU. A non-RNE rounding mode returns an error, and all
    input lanes are checked for finiteness before any result is written. Library
    exception flags and errno are explicitly unobserved by this interface.
    """
    policy = validate_policy(policy)
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("compiled unary math needs one logical computation and dense buffers")
    node = pure[0]
    if node.op not in policy["operations"] or len(node.inputs) != 1 or source.outputs != (node.id,):
        raise ValueError("source operation is not selected for compiled CPU math")
    operand = source.node(node.inputs[0])
    dtype, function = OPERATIONS[node.op]
    supplier = load_policy(required=[function])
    symbol = supplier["functions"][function]["symbol"]
    if (
        not node.type.shape
        or node.type != TensorType(node.type.shape, dtype, "raw-bit-pattern")
        or operand.type != node.type
        or operand.effect not in {"input", "constant"}
        or operand.index_maps
        or {n.id for n in source.nodes} != {operand.id, node.id}
        or node.index_maps != pointwise_identity_maps(len(node.type.shape), 1)
        or dict(node.attrs) != {"numeric_semantics": f"source-{dtype}-{function.rstrip('f')}-unqualified-v1"}
    ):
        raise ValueError("compiled unary math source types, indexing or intrinsic attributes differ")
    count = prod(node.type.shape)
    width, ctype, infinity = (4, "float", 0x7F800000) if dtype == "f32" else (8, "double", 0x7FF0000000000000)
    magnitude = (1 << (width * 8 - 1)) - 1
    size = count * width
    if size > 2**63 - 1:
        raise ValueError("compiled unary math exceeds its signed byte-size domain")
    for binding in source.constants:
        if binding.encoding != dtype + "-le-raw":
            raise ValueError("compiled unary math constant has a different source encoding")
        data = bytes.fromhex(binding.data_hex)
        if any(
            int.from_bytes(data[i : i + width], "little") & magnitude >= infinity for i in range(0, len(data), width)
        ):
            raise ValueError("compiled unary math requires finite constant inputs")
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include "host_math.h"
#if !defined(__riscv) || __riscv_xlen!=64 || !defined(__riscv_flen) || __riscv_flen<64
#error selected unary math requires RV64GC and an enabled double-precision FPU
#endif
#if !defined(__riscv_float_abi_double) || defined(__FAST_MATH__)
#error selected unary math requires lp64d without fast-math
#endif
_Static_assert(CHAR_BIT==8 && FLT_RADIX==2 && FLT_MANT_DIG==24 && DBL_MANT_DIG==53 &&
               sizeof(float)==4 && sizeof(double)==8 && SIZE_MAX>={size}ULL,"unary math host representation");
static {ctype} (*volatile const {name}_call)({ctype})={symbol};
static uint64_t {name}_load(const unsigned char *p) {{
  uint64_t bits=0;for (unsigned j=0;j<{width}u;++j) bits|=(uint64_t)p[j]<<(8u*j);return bits;
}}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-{size}u || b>UINTPTR_MAX-{size}u || (a<b+{size}u && b<a+{size}u)) return -1;
  if (merlin_host_check_fp_env()!=1) return -3;
  for (size_t i=0;i<{count}u;++i)
    if (({name}_load((const unsigned char *)inputs[0]+{width}u*i)&UINT64_C({magnitude}))
          >=UINT64_C({infinity})) return -2;
  for (size_t i=0;i<{count}u;++i) {{
    {"uint32_t" if width == 4 else "uint64_t"} bits=({"uint32_t" if width == 4 else "uint64_t"})
      {name}_load((const unsigned char *)inputs[0]+{width}u*i);
    {ctype} x,y;memcpy(&x,&bits,{width}u);y={name}_call(x);memcpy(&bits,&y,{width}u);
    for (unsigned j=0;j<{width}u;++j) ((unsigned char *)outputs[0])[{width}u*i+j]=(unsigned char)(bits>>(8u*j));
  }}
  return 0;
}}
"""
    linkage = {**policy["linkage"], "symbols": sorted(["merlin_host_check_fp_env", symbol])}
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": [operand.id],
        "outputs": [node.id],
        "arithmetic": "selected-library-source-math",
        "numeric_policy": policy["policy"],
        "source_intrinsic": node.op,
        "host_isa": policy["isa"],
        "host_abi": policy["abi"],
        "host_linkage": linkage,
        "host_math_symbols": [symbol],
        "effects": policy["effects"],
        "framework_equivalence": "not_established",
        "domain": "finite source encodings; enabled FPU; RNE; reject invalid inputs before writes",
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={"host_math.h": HEADER_PATH.read_text()},
        sources={"unary_math.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
