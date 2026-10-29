"""Explicitly split an original entry quantization into a separate ABI.

The preparer retains the original quantization operation and its constant
operands. The forward receives that operation's exact result type. This is a
caller-selected source/ABI transformation, not a numerical or timing policy.
"""

from dataclasses import dataclass
from hashlib import sha256
import math

from xdsl.dialects import arith, func, tensor
from xdsl.dialects.builtin import (
    FloatAttr, FunctionType, IntegerAttr, IntegerType, ModuleOp, NoneAttr,
    TensorType, f16, f32, f64,
)
from xdsl.ir import Block, Operation, Region

from merlin.xdsl_dialects._common import text


def _name(op):
    return getattr(getattr(op, "op_name", None), "data", op.name)


def _digest(module):
    return sha256(text(module, generic=True).encode()).hexdigest()


@dataclass(frozen=True)
class EntryQuantizationSplit:
    preparer: ModuleOp
    forward: ModuleOp
    entry: str
    preparer_entry: str
    argument_index: int
    source_sha256: str
    original_type: str
    prepared_type: str
    quantization_sha256: str

    def record(self):
        return dict(
            schema="merlin.entry_quantization_split.v1",
            entry=self.entry, preparer_entry=self.preparer_entry,
            argument_index=self.argument_index,
            source_sha256=self.source_sha256,
            preparer_sha256=_digest(self.preparer), forward_sha256=_digest(self.forward),
            original_type=self.original_type, prepared_type=self.prepared_type,
            quantization_sha256=self.quantization_sha256,
            scope="Exact original operation and constant operands moved to an explicit preparer; "
                  "forward ABI changed at the selected argument only. Original preparation "
                  "must execute once per input; excluded preparation must be reported separately. "
                  "The caller preserves the original floating environment between both calls. "
                  "No runtime, numerical, ownership or performance qualification is supplied.",
        )


def split_entry_quantization(module, *, entry, argument_index, preparer_entry):
    """Return independent preparer/forward modules without changing ``module``.

    The selected static, dense floating tensor must have exactly one direct
    use: the data operand of an original per-tensor quantization. Its scale and
    zero point must be rank-zero constants or splats of scalar constants. All
    other forward arguments, results, operations and their order are retained.
    Unknown quantization properties, source aliases/callers and parameter DAGs
    refuse rather than being inferred or approximately reconstructed.
    """
    if type(module) is not ModuleOp or not isinstance(entry, str) or not isinstance(preparer_entry, str):
        raise ValueError("entry quantization split needs a typed module and explicit symbols")
    if not entry or not preparer_entry or entry == preparer_entry:
        raise ValueError("preparer needs a distinct explicit symbol")
    if type(argument_index) is not int or argument_index < 0:
        raise ValueError("entry quantization split needs an explicit argument index")
    module.verify()
    functions = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    if len(functions) != 1 or functions[0].sym_name.data != entry:
        raise ValueError("entry quantization split requires a sole original function")
    original = functions[0]
    if len(original.body.blocks) != 1 or argument_index >= len(original.body.block.args):
        raise ValueError("entry quantization split requires a single-block original entry")
    if any(_name(op) == "func.call" for op in original.walk()):
        raise ValueError("entry quantization split cannot change an internal call ABI")
    if original.arg_attrs is not None or original.res_attrs is not None:
        raise ValueError("entry quantization split requires an unannotated tensor ABI")
    argument = original.body.block.args[argument_index]
    source_type = argument.type
    if (not isinstance(source_type, TensorType) or source_type.element_type not in (f16, f32, f64)
            or not isinstance(source_type.encoding, NoneAttr) or any(d <= 0 for d in source_type.get_shape())):
        raise ValueError("entry quantization split requires a static dense floating tensor")
    uses = tuple(argument.uses)
    if len(uses) != 1 or uses[0].index != 0:
        raise ValueError("selected input must be used only by its original quantization")
    quant = uses[0].operation
    if (_name(quant) != "quant_ext.quantize_per_tensor" or quant.parent is not original.body.block
            or len(quant.operands) != 3 or len(quant.results) != 1 or quant.regions or quant.successors):
        raise ValueError("selected input does not have a direct original per-tensor quantization")
    result_type = quant.results[0].type
    if (not isinstance(result_type, TensorType) or not isinstance(result_type.element_type, IntegerType)
            or result_type.get_shape() != source_type.get_shape() or not isinstance(result_type.encoding, NoneAttr)
            or not quant.results[0].uses):
        raise ValueError("quantization must retain the complete dense tensor coordinates")
    properties = {**quant.attributes, **quant.properties}
    if any(not key.startswith("prov.") and key not in {
            "op_name__", "quant_min", "quant_max", "output_dtype"} for key in properties):
        raise ValueError("unknown quantization semantics")
    if any(not isinstance(properties.get(key), IntegerAttr) for key in ("quant_min", "quant_max")):
        raise ValueError("quantization needs original explicit integer bounds")
    qmin, qmax = (properties[key].value.data for key in ("quant_min", "quant_max"))
    width = result_type.element_type.width.data
    if not -(1 << (width - 1)) <= qmin <= qmax < 1 << (width - 1):
        raise ValueError("quantization bounds exceed the original signed result")

    constants = []

    def constant(value, expected_element):
        typ, owner = value.type, value.owner
        if (not isinstance(typ, TensorType) or typ.get_shape() != () or typ.element_type != expected_element
                or not isinstance(typ.encoding, NoneAttr) or not isinstance(owner, Operation)
                or owner.parent is not original.body.block):
            raise ValueError("quantization parameter must be an original scalar tensor constant")
        if isinstance(owner, tensor.SplatOp):
            scalar = owner.input.owner
            if (not isinstance(scalar, arith.ConstantOp) or scalar.parent is not original.body.block
                    or scalar.operands or scalar.regions):
                raise ValueError("quantization parameter splat needs its original scalar constant")
            constants.extend(op for op in (scalar, owner) if op not in constants)
            return scalar.value
        if isinstance(owner, arith.ConstantOp) and not owner.operands and not owner.regions:
            value = owner.value
            values = tuple(getattr(value, "get_values", lambda: ())())
            if len(values) != 1:
                raise ValueError("quantization parameter must contain one constant")
            if owner not in constants:
                constants.append(owner)
            return values[0]
        raise ValueError("quantization parameter is not an original constant")

    scale_type = quant.operands[1].type
    if not isinstance(scale_type, TensorType) or scale_type.element_type not in (f16, f32, f64):
        raise ValueError("quantization scale needs an original floating scalar tensor")
    scale = constant(quant.operands[1], scale_type.element_type)
    scale_value = scale.value.data if isinstance(scale, FloatAttr) else scale
    if not isinstance(scale_value, float) or not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("quantization scale must be positive and finite")
    zp_type = quant.operands[2].type
    if not isinstance(zp_type, TensorType) or not isinstance(zp_type.element_type, IntegerType):
        raise ValueError("quantization zero point needs an original integer scalar tensor")
    constant(quant.operands[2], zp_type.element_type)
    for predecessor in original.body.block.ops:
        if predecessor is quant:
            break
        if predecessor not in constants:
            raise ValueError("entry quantization must precede all other source operations")
    ordered = [op for op in original.body.block.ops if op in constants or op is quant]
    prep_block = Block(arg_types=[source_type])
    mapping = {argument: prep_block.args[0]}
    for op in ordered:
        cloned = op.clone(value_mapper=mapping)
        prep_block.add_op(cloned)
        mapping.update(zip(op.results, cloned.results, strict=True))
    prep_block.add_op(func.ReturnOp(mapping[quant.results[0]]))
    preparer = ModuleOp([func.FuncOp(preparer_entry, ([source_type], [result_type]), Region(prep_block))])
    forward = module.clone()
    selected = next(op for op in forward.body.block.ops if isinstance(op, func.FuncOp))
    block = selected.body.block
    cloned_ops = dict(zip(original.body.block.ops, block.ops, strict=True))
    old_argument = block.args[argument_index]
    old_quant = next(iter(old_argument.uses)).operation
    prepared = block.insert_arg(result_type, argument_index)
    old_quant.results[0].replace_all_uses_with(prepared)
    block.erase_op(old_quant)
    block.erase_arg(old_argument)
    # Remove only the proven constant cone, and only when it has no remaining use.
    for source_constant in reversed(ordered[:-1]):
        candidate = cloned_ops[source_constant]
        if not any(result.uses for result in candidate.results):
            block.erase_op(candidate)
    selected.function_type = FunctionType.from_lists([a.type for a in block.args], selected.function_type.outputs)
    preparer.verify()
    forward.verify()
    return EntryQuantizationSplit(preparer, forward, entry, preparer_entry, argument_index,
                                  _digest(module), str(original.function_type), str(selected.function_type),
                                  sha256(text(quant, generic=True).encode()).hexdigest())
