"""Compile explicitly placed source predicates with checked dense-buffer access."""

from hashlib import sha256
from itertools import product
from math import prod

from merlin.semantic_compiler.model import TensorType, pointwise_identity_maps
from merlin.semantic_compiler.predicate_values import predicate_element

from .c_sources import c_source_pair
from .layout_c import _element_bytes, _flat_map


def _float_comparison_c(name, width):
    magnitude = (1 << (width * 8 - 1)) - 1
    infinity = magnitude - ((1 << (23 if width == 4 else 52)) - 1)
    return f"""static int {name}_compare(uint64_t a,uint64_t b,unsigned predicate) {{
  const uint64_t magnitude=UINT64_C({magnitude}), infinity=UINT64_C({infinity});
  if ((a&magnitude)>infinity || (b&magnitude)>infinity) return 0;
  unsigned equal=a==b || (!(a&magnitude) && !(b&magnitude));
  unsigned sign_a=(unsigned)(a>>{width * 8 - 1}), sign_b=(unsigned)(b>>{width * 8 - 1});
  unsigned less=!equal && (sign_a!=sign_b ? sign_a : sign_a ? a>b : a<b);
  switch (predicate) {{
    case 0: return equal;
    case 1: return !equal;
    case 2: return less;
    case 3: return less || equal;
    case 4: return !(less || equal);
    default: return !less;
  }}
}}
"""


def _constant_result(source, node, operands, widths, output_width, limit):
    if any(item.effect != "constant" for item in operands):
        return None
    if prod(node.type.shape) + sum(prod(item.type.shape) for item in operands) > limit:
        return None
    bindings = {binding.node_id: binding for binding in source.constants}
    data = [bytes.fromhex(bindings[item.id].data_hex) for item in operands]
    values = [
        tuple(int.from_bytes(payload[i : i + width], "little") for i in range(0, len(payload), width))
        for payload, width in zip(data, widths, strict=True)
    ]
    result = bytearray()
    for index in product(*(range(extent) for extent in node.type.shape)):
        arguments = []
        for item, mapping, elements in zip(operands, node.index_maps[:-1], values, strict=True):
            flat = 0
            for coordinate, extent in zip(mapping.apply(index), item.type.shape, strict=True):
                flat = flat * extent + coordinate
            arguments.append(elements[flat])
        value = predicate_element(node, arguments, widths)
        result.extend(value.to_bytes(output_width, "little"))
    return bytes(result), {
        "operand_sha256": [sha256(value).hexdigest() for value in data],
        "result_sha256": sha256(result).hexdigest(),
        "element_work_limit": limit,
    }


def emit_predicate_c(source, name, *, constant_fold_limit=65536):
    """Preserve IEEE truth/ordered comparisons and signed-i64/Boolean predicates.

    This emits CPU computation only when the placement policy selects it.
    Floating arithmetic and unordered comparisons require separate lowerings.
    Boolean inputs must contain canonical zero/one bytes.
    """
    if type(constant_fold_limit) is not int or constant_fold_limit < 0:
        raise ValueError("constant predicate folding needs a nonnegative element budget")
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) == 1 and computations[0].op == "bucketize":
        from .c_bucketize import emit_bucketize_c

        return emit_bucketize_c(source, name)
    if len(computations) != 1 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError("predicate C needs one computation and dense logical buffers")
    node = computations[0]
    operands = tuple(source.node(key) for key in node.inputs)
    if (
        source.outputs != (node.id,)
        or not operands
        or {item.id for item in source.nodes} != {*node.inputs, node.id}
        or any(item.effect not in {"input", "constant"} or item.index_maps for item in operands)
    ):
        raise ValueError("predicate C needs complete source operand and result accounting")
    count, rank = prod(node.type.shape), len(node.type.shape)
    if not rank:
        raise ValueError("predicate C needs a positive-rank tensor")
    maps = node.index_maps
    if len(maps) != len(operands) + 1 or maps[-1] != pointwise_identity_maps(rank, 1)[0]:
        raise ValueError("predicate C needs an identity output map")
    truth, helper = node.op == "f32_nonzero", ""
    if node.op == "select":
        if (
            len(operands) != 3
            or node.attrs
            or operands[0].type != TensorType(node.type.shape, "i1", "boolean")
            or any(item.type != node.type for item in operands[1:])
            or maps != pointwise_identity_maps(rank, 3)
        ):
            raise ValueError("select needs an exact Boolean condition and unchanged value types")
        output_width = _element_bytes(node.type.dtype)
        if output_width not in {1, 2, 4, 8}:
            raise ValueError("select needs an admitted integer bit carrier")
        widths = (1, output_width, output_width)
        c_types = tuple(f"uint{width * 8}_t" for width in widths)
        expression = "v0 ? v1 : v2"
        boolean_inputs = tuple(i for i, item in enumerate(operands) if item.type.dtype == "i1")
    elif truth:
        if (
            len(operands) != 1
            or operands[0].type != TensorType(node.type.shape, "f32", "raw-bit-pattern")
            or node.type != TensorType(node.type.shape, "i1", "boolean")
            or node.attrs != (("numeric_semantics", "f32-une-zero-v1"),)
            or maps != pointwise_identity_maps(rank, 1)
        ):
            raise ValueError("nonzero predicate has incompatible source semantics")
        c_type, width, output_width, expression, boolean = "uint32_t", 4, 1, "(v0 & UINT32_C(2147483647)) != 0", False
    elif node.op == "mask":
        attrs = dict(node.attrs)
        if set(attrs) != {"operation", "predicate"}:
            raise ValueError("mask has unknown semantic attributes")
        operation, predicate = attrs["operation"], attrs["predicate"]
        comparisons = {"eq": "==", "ne": "!=", "slt": "<", "sle": "<=", "sgt": ">", "sge": ">="}
        if operation == "arith.cmpi" and predicate in comparisons and len(operands) == 2:
            expected, c_type, width, output_width, boolean = ("i64", "signed-i64"), "int64_t", 8, 1, False
            expression = f"v0 {comparisons[predicate]} v1"
        elif (
            operation == "arith.cmpf" and predicate in {"oeq", "one", "olt", "ole", "ogt", "oge"} and len(operands) == 2
        ):
            dtype = operands[0].type.dtype
            if dtype not in {"f32", "f64"}:
                raise ValueError("floating predicate needs an IEEE f32 or f64 source type")
            width = 4 if dtype == "f32" else 8
            expected, c_type, output_width, boolean = (dtype, "raw-bit-pattern"), f"uint{width * 8}_t", 1, False
            helper = _float_comparison_c(name, width)
            index = ("oeq", "one", "olt", "ole", "ogt", "oge").index(predicate)
            expression = f"{name}_compare(v0,v1,{index}u)"
        elif operation in {"arith.andi", "arith.xori", "arith.extui"} and predicate is None:
            if len(operands) != (1 if operation == "arith.extui" else 2):
                raise ValueError("Boolean predicate has an incompatible arity")
            expected, c_type, width, boolean = ("i1", "boolean"), "uint8_t", 1, True
            output_width = 8 if operation == "arith.extui" else 1
            expression = "v0" if output_width == 8 else "v0 & v1" if operation == "arith.andi" else "v0 ^ v1"
        else:
            raise ValueError("mask operation or predicate has no compiled implementation")
        if any(
            (item.type.dtype, item.type.numerical_policy) != expected for item in operands
        ) or node.type != TensorType(
            node.type.shape, "i64" if output_width == 8 else "i1", "signed-i64" if output_width == 8 else "boolean"
        ):
            raise ValueError("mask source or output types differ from its operation")
    else:
        raise ValueError("source operation has no compiled predicate implementation")
    if node.op != "select":
        widths, c_types = (width,) * len(operands), (c_type,) * len(operands)
        boolean_inputs = tuple(range(len(operands))) if boolean else ()
    sizes = [prod(item.type.shape) * width for item, width in zip(operands, widths, strict=True)]
    output_bytes = count * output_width
    if max(*sizes, output_bytes) > 2**63 - 1:
        raise ValueError("predicate C exceeds its signed index or byte-size domain")
    reads = []
    for i, (operand, mapping, width, c_type) in enumerate(zip(operands, maps[:-1], widths, c_types, strict=True)):
        axes = []
        for row, offset, extent in zip(mapping.coefficients, mapping.offsets, operand.type.shape, strict=True):
            selected = [axis for axis, coefficient in enumerate(row) if coefficient]
            if offset or len(selected) > 1 or any(coefficient not in (0, 1) for coefficient in row):
                raise ValueError("predicate input map is not a singleton projection")
            if not selected:
                if extent != 1:
                    raise ValueError("predicate constant projection needs a singleton extent")
            elif selected[0] in axes or extent != node.type.shape[selected[0]]:
                raise ValueError("predicate projection has incompatible extents")
            else:
                axes.append(selected[0])
        index = _flat_map(mapping, node.type.shape, operand.type.shape)
        reads.append(
            f"{c_type} v{i}; __builtin_memcpy(&v{i},(const unsigned char *)inputs[{i}]+{width}u*({index}),{width}u);"
        )
    counters = " ".join(f"for (size_t d{i}=0;d{i}<{extent}u;++d{i})" for i, extent in enumerate(node.type.shape))
    output_index = _flat_map(maps[-1], node.type.shape, node.type.shape)
    check = [f"b>UINTPTR_MAX-{output_bytes}u"]
    for i, size in enumerate(sizes):
        check += [
            f"!inputs[{i}]",
            f"(uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}u",
            f"((uintptr_t)inputs[{i}]<b+{output_bytes}u && b<(uintptr_t)inputs[{i}]+{size}u)",
        ]
    validate = ""
    if boolean_inputs:
        conditions = " || ".join(f"v{i}>1" for i in boolean_inputs)
        validate = f"{counters} {{ {' '.join(reads[i] for i in boolean_inputs)} if ({conditions}) return -2; }}"
    folded = _constant_result(source, node, operands, widths, output_width, constant_fold_limit)
    if folded is not None:
        payload, evidence = folded
        words = [int.from_bytes(payload[i : i + output_width], "little") for i in range(0, len(payload), output_width)]
        initializer = ",".join(f"UINT{output_width * 8}_C({value})" for value in words)
        reads_constant = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={output_bytes}ULL, "constant predicate exceeds host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context; (void)inputs;
  static const uint{output_width * 8}_t values[{count}]={{{initializer}}};
  if (!outputs || !outputs[0] || (uintptr_t)outputs[0]>UINTPTR_MAX-{output_bytes}u) return -1;
  __builtin_memcpy(outputs[0],values,{output_bytes}u);
  return 0;
}}
"""
        record = {
            "source_request_digest": source.digest(),
            "implementation": name,
            "inputs": [],
            "outputs": [node.id],
            "arithmetic": "exact-source-predicate",
            "constant_folding": evidence,
        }
        return record, c_source_pair(
            name,
            public_headers={},
            private_headers={},
            sources={"predicate.c": reads_constant},
            declarations=f"int {name}(void *, const void *const *, void *const *);\n",
        )
    implementation = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && sizeof(int64_t)==8 && sizeof(uint32_t)==4 && SIZE_MAX>={max(*sizes, output_bytes)}ULL,
               "predicate storage differs from the source ABI");
{helper}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if ({" || ".join(check)}) return -1;
  {validate}
  {counters} {{ {" ".join(reads)}
    uint{output_width * 8}_t result=(uint{output_width * 8}_t)({expression});
    __builtin_memcpy((unsigned char *)outputs[0]+{output_width}u*({output_index}),&result,{output_width}u);
  }}
  return 0;
}}
"""
    return {
        "source_request_digest": source.digest(),
        "implementation": name,
        "inputs": list(node.inputs),
        "outputs": [node.id],
        "arithmetic": "exact-source-predicate",
    }, c_source_pair(
        name,
        public_headers={},
        private_headers={},
        sources={"predicate.c": implementation},
        declarations=f"int {name}(void *, const void *const *, void *const *);\n",
    )
