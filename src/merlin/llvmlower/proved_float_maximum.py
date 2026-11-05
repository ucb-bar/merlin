"""Replace floating maximum only with proved NaN-free, negative-zero-free inputs.

The ordinary maximumNumber form can select a different signed zero and has
different NaN behavior. A typed producer proof excludes both differences; no
fast-math flag, sampled input, provenance identifier or library claim is used.
See https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxnumf-arithmaxnumfop.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

FEATURE = "lower_proved_float_maximum"


@dataclass(frozen=True)
class _Domain:
    # Every value is non-NaN and cannot be negative zero. Infinities are allowed.
    unit_or_zero: bool = False  # Nonzero magnitude is at least one, including infinity.
    constant: float | None = None


def _ordinary(op, allowed=()):
    for name, value in (*op.attributes.items(), *op.properties.items()):
        if name == "fastmath" and str(value) == "#arith.fastmath<none>":
            continue
        if name not in allowed and not name.startswith("prov."):
            return False
    return True


def _strict(op):
    while op is not None:
        if any("strictfp" in name or "strictfp" in str(value)
               for name, value in (*op.attributes.items(), *op.properties.items())):
            return True
        op = op.parent_op()
    return False


def _constant(number):
    number = float(number)
    if math.isnan(number) or number == 0 and math.copysign(1., number) < 0:
        return None
    return _Domain(number == 0 or abs(number) >= 1, number)


class _Proof:
    def __init__(self):
        self.memo = {}

    def tensor(self, value, depth=0):
        from xdsl.dialects.builtin import Float32Type, NoneAttr, TensorType
        from xdsl.ir import Operation

        if value in self.memo:
            return self.memo[value]
        self.memo[value] = None
        typ, op = value.type, value.owner
        if (depth > 64 or len(self.memo) > 8192 or not isinstance(typ, TensorType)
                or not isinstance(typ.element_type, Float32Type) or not isinstance(typ.encoding, NoneAttr)
                or any(d <= 0 for d in typ.get_shape()) or not isinstance(op, Operation) or _strict(op)):
            return None
        domain = None
        if op.name == "arith.constant" and _ordinary(op, {"value"}):
            from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

            attr = op.properties.get("value", op.attributes.get("value"))
            if isinstance(attr, DenseIntOrFPElementsAttr) and attr.is_splat():
                domain = _constant(attr.get_values()[0])
        elif op.name == "tensor.splat" and len(op.operands) == 1 and _ordinary(op):
            domain = self.scalar(op.operands[0], {}, depth + 1)
        elif op.name in {"tensor.extract_slice", "tensor.expand_shape", "tensor.collapse_shape", "tensor.cast"}:
            allowed = {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes",
                       "reassociation", "static_output_shape"}
            if op.operands and _ordinary(op, allowed):
                domain = self.tensor(op.operands[0], depth + 1)
        elif op.name == "linalg.generic":
            domain = self.generic(op, depth + 1)
        elif op.name == "linalg.reduce":
            proof = self.reduction(op, depth + 1)
            if proof is not None:
                domain = proof[1]
        self.memo[value] = domain
        return domain

    def scalar(self, value, bindings, depth):
        from xdsl.dialects.builtin import Float32Type, FloatAttr, IntegerType
        from xdsl.ir import Operation

        if value in bindings:
            return bindings[value]
        op = value.owner
        if depth > 64 or not isinstance(value.type, Float32Type) or not isinstance(op, Operation):
            return None
        if op.name == "arith.constant" and _ordinary(op, {"value"}):
            attr = op.properties.get("value", op.attributes.get("value"))
            return _constant(attr.value.data) if isinstance(attr, FloatAttr) else None
        if op.name in {"arith.sitofp", "arith.uitofp"} and _ordinary(op) and len(op.operands) == 1:
            typ = op.operands[0].type
            if isinstance(typ, IntegerType) and 1 <= typ.width.data <= 64:
                # All <=64-bit integers convert to finite binary32; zero converts
                # to +0 and every nonzero converted magnitude is at least one.
                return _Domain(True)
            return None
        if op.name == "arith.select" and _ordinary(op) and len(op.operands) == 3:
            a, b = (self.scalar(v, bindings, depth + 1) for v in op.operands[1:])
            return None if a is None or b is None else _Domain(a.unit_or_zero and b.unit_or_zero)
        if op.name not in {"arith.mulf", "arith.divf"} or not _ordinary(op):
            return None
        a, b = (self.scalar(v, bindings, depth + 1) for v in op.operands)
        if op.name == "arith.mulf" and a is not None and a.constant is not None:
            a, b = b, a
        if (a is None or b is None or b.constant is None
                or not math.isfinite(b.constant) or b.constant <= 0):
            return None
        scale = b.constant
        grows = scale >= 1 if op.name == "arith.mulf" else scale <= 1
        if not grows and not a.unit_or_zero:
            return None
        # A nonzero converted integer has magnitude >=1. Multiplication by a
        # positive finite binary32 cannot shrink it below 2^-149; division by
        # such a value cannot shrink it below 1/F32MAX > 2^-128. Thus rounding
        # in any mode cannot create negative zero. Growing preserves this too.
        return _Domain(a.unit_or_zero and grows)

    def generic(self, op, depth):
        from xdsl.dialects import tensor
        from xdsl.dialects.builtin import TensorType
        from xdsl.dialects.linalg.attrs import IteratorType
        from xdsl.ir.affine import AffineMap

        if (not _ordinary(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"})
                or len(op.results) != 1 or len(op.outputs) != 1 or not isinstance(op.outputs[0].owner, tensor.EmptyOp)
                or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
                or any(not isinstance(v.type, TensorType) for v in op.operands)):
            return None
        rank = len(op.results[0].type.get_shape())
        if (op.outputs[0].type != op.results[0].type
                or tuple(i.data for i in op.iterator_types) != (IteratorType.PARALLEL,) * rank
                or op.indexing_maps.data[-1].data != AffineMap.identity(rank)):
            return None
        block = op.body.block
        body = tuple(block.ops)
        allowed = {"arith.constant", "arith.sitofp", "arith.uitofp", "arith.mulf", "arith.divf", "arith.select"}
        if (len(block.args) != len(op.operands) or block.args[-1].uses or not body
                or body[-1].name != "linalg.yield" or len(body[-1].operands) != 1
                or any(child.name not in allowed or child.regions for child in body[:-1])):
            return None
        bindings = {arg: self.tensor(value, depth + 1) for arg, value in zip(block.args, op.inputs)}
        return self.scalar(body[-1].operands[0], bindings, depth + 1)

    def reduction(self, op, depth=0):
        if (not _ordinary(op, {"dimensions"}) or _strict(op) or len(op.results) != 1
                or len(op.operands) != 2 or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
            return None
        block = op.regions[0].block
        body = tuple(block.ops)
        if (len(block.args) != 2 or len(body) != 2 or body[0].name != "arith.maximumf"
                or not _ordinary(body[0]) or tuple(body[0].operands) not in (tuple(block.args), tuple(block.args)[::-1])
                or body[1].name != "linalg.yield" or tuple(body[1].operands) != tuple(body[0].results)):
            return None
        a, b = (self.tensor(value, depth + 1) for value in op.operands)
        if a is None or b is None:
            return None
        return body[0], _Domain(a.unit_or_zero and b.unit_or_zero)


def rewrite_proved_maxima(module) -> dict:
    from xdsl.dialects import arith

    proof, planned = _Proof(), []
    for op in module.walk():
        if op.name == "linalg.reduce":
            match = proof.reduction(op)
            if match is not None:
                planned.append(match[0])
    for old in planned:
        new = arith.MaxnumfOp.create(operands=old.operands, result_types=old.result_types,
                                    attributes=dict(old.attributes), properties=dict(old.properties))
        new.location = old.location
        block = old.parent_block()
        block.insert_op_before(new, old)
        old.results[0].replace_all_uses_with(new.results[0])
        block.erase_op(old)
    module.verify()
    return {"proved_nan_and_negative_zero_free_maximum_reductions": len(planned)} if planned else {}


def rewrite_text(source: str) -> tuple[str, dict]:
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    module.verify()
    report = rewrite_proved_maxima(module)
    if not report:
        return source, report
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue(), report


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Use maximumNumber only for typed binary32 reductions proved free of NaNs and "
                "negative zero; preserve every source arithmetic operation and flags, default off."
            ),
        ))
    return FEATURE
