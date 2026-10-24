"""Prove a uniform rational zero from a finite integral floating image, read from the IR.

arith.remf(convert(n), d), with n integer storage converted to a floating
format and d a closed literal whose reciprocal is an integer (d = +-2^-k), is
an exact zero for every n: the converted image is a finite integral value and
every integer is a multiple of d. IEEE gives it the dividend's sign; at the
declared quantized output boundary the dequantized zero is canonical +0.
The rest of the scalar body is then evaluated exactly with that zero of
either sign; only a program whose result is the same closed literal for both
signs, independent of every runtime value, becomes a uniform fill. The
divisor and the evaluation come only from the live IR operations; no case
record, overload name or provenance label participates.
"""

import hashlib
import json
import math
from fractions import Fraction

INTEGER_CONVERSIONS = {"arith.sitofp", "arith.uitofp"}


def _remainders(root, argument):
    """remf(convert(argument), d) operations with a reciprocal-integral literal divisor."""
    from merlin.targetgen.closed_scalar import literal

    found = []
    from merlin.targetgen.native_payload_carriers import op_name

    for op in root.body.block.ops:
        if op_name(op) != "arith.remf":
            continue
        dividend, divisor = op.operands
        conversion = dividend.owner
        if op_name(conversion) not in INTEGER_CONVERSIONS or conversion.operands[0] is not argument:
            continue
        d = literal(divisor)
        if not isinstance(d, float) or not math.isfinite(d) or d == 0 or (1 / abs(Fraction(d))).denominator != 1:
            continue
        found.append((op, d))
    return found


def specialize(directory, report, software_spec):
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerType, TensorType
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.common.yaml import load_yaml
    from merlin.llvmlower.quantized_native import parameters
    from merlin.targetgen.closed_scalar import UNKNOWN, evaluate
    from merlin.targetgen.source_power_specialization import _elementwise_root
    from merlin.targetgen.source_primitive_effects import pure_tensor_source, single_forward

    if software_spec is None:
        return None
    found = single_forward(directory, report)
    if found is None:
        return None
    record, original, module, function = found
    if len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    if not pure_tensor_source(function):
        return None
    source = function.body.block.args[0].type
    if not isinstance(source, TensorType) or not isinstance(source.element_type, IntegerType):
        return None
    bits = source.element_type.width.data
    if bits not in {1, 8, 16, 32, 64}:
        return None
    found = _elementwise_root(function, function.body.block.args[0])
    if found is None:
        return None
    root, argument = found
    remainders = _remainders(root, argument)
    if not remainders:
        return None
    if str(root.results[0].type.element_type) not in {"f32", "f64"}:
        return None
    yielded = root.body.block.last_op.operands[0]
    results = []
    for zero in (0.0, -0.0):
        value = evaluate(yielded, {op.results[0]: zero for op, _ in remainders})
        results.append(value)
    if any(v is UNKNOWN or not isinstance(v, float) or not math.isfinite(v) for v in results):
        return None
    if results[0] != results[1]:
        return None
    fill = results[0]
    if fill == 0:
        fill = 0.0  # Canonical positive zero at the quantized output boundary.
    indices = record.get("output_indices", ())
    posts = record.get("semantic_boundary", {}).get("post_indices", ())
    types = list(function.function_type.outputs)
    if len(indices) != 1 or len(posts) != 1 or set((*indices, *posts)) != set(range(len(types))):
        return None
    result = types[indices[0]]
    if not isinstance(result, TensorType) or result != root.results[0].type:
        return None
    if types[posts[0]] != source:
        return None
    contract = parameters(load_yaml(software_spec))
    block = Block(arg_types=[source])
    constant = arith.ConstantOp(FloatAttr(fill, result.element_type))
    filled = tensor.SplatOp(constant.result, [], result)
    values = [None] * len(types)
    values[indices[0]] = filled.result
    values[posts[0]] = block.args[0]
    block.add_ops([constant, filled, func.ReturnOp(*values)])
    replacement = function.clone_without_regions()
    replacement.body.add_block(block)
    Rewriter.replace_op(function, replacement)
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="ir_finite_integral_float_image_mod_exact_reciprocal",
        input_signed_storage_bits=bits,
        opmath=str(result.element_type),
        divisors=[d for _, d in remainders],
        fill=fill,
        zero_sign="canonical positive at quantized output boundary",
        parameters="live IR operations and literals only; no case record or tensor payload read",
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        quantized_contract=contract,
    )
    (directory / "native-source-remainder.json").write_text(json.dumps(proof, indent=2) + "\n")
    record["source_specialization"] = proof
    return proof, original
