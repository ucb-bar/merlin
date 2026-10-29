"""Public typed LLVM parsing with preserved upstream optimization metadata.

Extend missing standard LLVM annotations on local operation classes.
Original xDSL operand, successor and custom checks still run. Metadata payloads
remain opaque: the stock compiler verifier owns their internal vocabulary;
these annotations never establish numerical or transformation correctness.
"""

from copy import copy
from dataclasses import replace

from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    UnitAttr,
    UnregisteredAttr,
)
from xdsl.ir import Dialect, Operation
from xdsl.irdl import BaseAttr, OperandDef, OptPropertyDef, ParsePropInAttrDict, VarConstraint
from xdsl.irdl.operations import get_accessors_from_op_def
from xdsl.utils.exceptions import VerifyException


def _annotated_operation(base, property_name, constraint, metadata_name):
    original = base.get_irdl_definition()
    if property_name in original.properties and metadata_name is not None:
        return base
    definition = replace(
        original,
        properties={**original.properties, property_name: OptPropertyDef(constraint)},
        options=[
            *original.options,
            *(
                []
                if any(isinstance(option, ParsePropInAttrDict) for option in original.options)
                else [ParsePropInAttrDict()]
            ),
        ],
    )

    def verify(operation):
        metadata = operation.properties.get(property_name)
        if metadata is not None and metadata_name == "builtin.string_array":
            if not isinstance(metadata, ArrayAttr) or any(not isinstance(entry, StringAttr) for entry in metadata):
                raise VerifyException("expected an LLVM no-builtins array of function names")
        elif metadata is not None and metadata_name == "builtin.dictionary_array":
            if not isinstance(metadata, ArrayAttr) or any(not isinstance(entry, DictionaryAttr) for entry in metadata):
                raise VerifyException("expected an LLVM argument dictionary array")
        elif metadata is not None and metadata_name not in {None, "builtin.unit"}:
            entries = metadata if isinstance(metadata, ArrayAttr) else [metadata]
            if not entries or any(
                not isinstance(entry, UnregisteredAttr) or entry.attr_name.data != metadata_name or entry.is_type.data
                for entry in entries
            ):
                raise VerifyException(f"expected {metadata_name} metadata")
        # Run original custom checks on a shallow property view: SSA, CFG,
        # regions and all nonmetadata properties stay identical. Neither the
        # parsed operation nor upstream xDSL classes are mutated.
        view = copy(operation)
        # The extended IRDL definition already checked traits against the
        # actual parent above. A duplicate terminator-trait check must not
        # mistake this detached verification view for the block's last op.
        view.parent = operation.parent if isinstance(operation, llvm.CallOp) else None
        view.properties = dict(operation.properties)
        if metadata_name is None:
            # LLVM's generic property form stores the same two overflow bits
            # as i32; xDSL's truncation schema expects its enum attribute.
            # from_int rejects unknown bits; preserve the original property.
            if isinstance(metadata, IntegerAttr):
                view.properties[property_name] = llvm.OverflowAttr.from_int(metadata.value.data)
        else:
            view.properties.pop(property_name, None)
        base.verify_(view)

    accessors = get_accessors_from_op_def(definition, verify)
    if metadata_name is None:
        # The upstream custom trunc printer accepts only its enum attribute.
        # Generic printing preserves either representation without rewriting
        # the original numeric property or losing its overflow bits.
        accessors["print"] = Operation.print
    return type(f"_Annotated{base.__name__}", (base,), accessors)


def _pointer_icmp_operation():
    """Retain stock ICmp checks while accepting LLVM's scalar pointer operand."""
    original = llvm.ICmpOp.get_irdl_definition()
    operands = []
    for name, operand in original.operands:
        scalar = operand.constr.constraint.constr
        selected = VarConstraint(scalar.name, scalar.constraint | BaseAttr(llvm.LLVMPointerType))
        operands.append((name, OperandDef(selected)))
    definition = replace(original, operands=operands)

    def verify(operation):
        if isinstance(operation.lhs.type, llvm.LLVMPointerType):
            if operation.res.type != builtin.i1 or not 0 <= operation.predicate.value.data <= 9:
                raise VerifyException("scalar LLVM pointer comparison requires i1 and a defined predicate")
        else:
            llvm.ICmpOp.verify_(operation)

    accessors = get_accessors_from_op_def(definition, verify)
    return type("_PointerICmpOp", (llvm.ICmpOp,), accessors)


_OPERATIONS = {
    llvm.ICmpOp: _pointer_icmp_operation(),
    llvm.CallOp: _annotated_operation(
        _annotated_operation(
            _annotated_operation(
                _annotated_operation(
                    _annotated_operation(llvm.CallOp, "arg_attrs", BaseAttr(ArrayAttr), "builtin.dictionary_array"),
                    "no_unwind",
                    BaseAttr(UnitAttr),
                    "builtin.unit",
                ),
                "nobuiltin",
                BaseAttr(UnitAttr),
                "builtin.unit",
            ),
            "nobuiltins",
            BaseAttr(ArrayAttr),
            "builtin.string_array",
        ),
        "memory_effects",
        BaseAttr(UnregisteredAttr),
        "llvm.memory_effects",
    ),
    llvm.BrOp: _annotated_operation(llvm.BrOp, "loop_annotation", BaseAttr(UnregisteredAttr), "llvm.loop_annotation"),
    llvm.CondBrOp: _annotated_operation(
        llvm.CondBrOp, "loop_annotation", BaseAttr(UnregisteredAttr), "llvm.loop_annotation"
    ),
    llvm.LoadOp: _annotated_operation(llvm.LoadOp, "tbaa", BaseAttr(ArrayAttr), "llvm.tbaa_tag"),
    llvm.StoreOp: _annotated_operation(llvm.StoreOp, "tbaa", BaseAttr(ArrayAttr), "llvm.tbaa_tag"),
    llvm.FuncOp: _annotated_operation(
        _annotated_operation(
            _annotated_operation(llvm.FuncOp, "dso_local", BaseAttr(UnitAttr), "builtin.unit"),
            "nobuiltins",
            BaseAttr(ArrayAttr),
            "builtin.string_array",
        ),
        "memory_effects",
        BaseAttr(UnregisteredAttr),
        "llvm.memory_effects",
    ),
    llvm.TruncOp: _annotated_operation(
        llvm.TruncOp, "overflowFlags", BaseAttr(llvm.OverflowAttr) | IntegerAttr.constr(type=IntegerType(32)), None
    ),
}


# Standard stock-imported control/effect annotations stay exact. These local
# schema extensions do not grant caller effects or suppress base verification.
for _base, _properties in ((llvm.FuncOp, ("will_return", "noreturn")), (llvm.CallOp, ("noreturn",))):
    for _property in _properties:
        _OPERATIONS[_base] = _annotated_operation(_OPERATIONS[_base], _property, BaseAttr(UnitAttr), "builtin.unit")


def make_llvm_context() -> Context:
    """Return a local context; never patch xDSL classes or discard metadata."""
    dialect = Dialect(
        "llvm",
        [_OPERATIONS.get(operation, operation) for operation in llvm.LLVM.operations],
        list(llvm.LLVM.attributes),
    )
    context = Context(allow_unregistered=True)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(func.Func)
    context.load_dialect(dialect)
    return context
