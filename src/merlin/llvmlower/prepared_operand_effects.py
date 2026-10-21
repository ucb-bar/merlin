"""Conservative typed source immutability checks for prepared operand lifetimes.

This proves properties of the original tensor program. An external physical
producer/consumer still needs its independently validated borrowed-buffer proof;
function names or proof hashes alone do not supply that permission.
"""

from __future__ import annotations

from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, LinalgStructuredOperation, ReduceOp, YieldOp
from xdsl.traits import get_effects

from .tensor_preparation_identity import validate_tensor_preparation_opportunity


def validate_pure_tensor_function(function, functions, *, active=(), allow_named_linalg=False):
    """Require a complete nonrecursive tensor/scalar body with no opaque calls.

    Tensor linalg operations have value semantics when every shaped operand is a
    tensor and their scalar regions are pure. Buffer linalg, unknown effects,
    address conversion, external functions, recursion and captured buffers refuse.
    """
    if not isinstance(function, func.FuncOp) or len(function.body.blocks) != 1:
        raise ValueError("prepared source requires a complete single-block tensor function")
    if type(allow_named_linalg) is not bool:
        raise ValueError("named tensor operation admission requires an explicit boolean")
    if function in active:
        raise ValueError("recursive prepared source effects are unknown")
    for value in (*function.body.block.args, *function.function_type.outputs):
        _value_type(value.type if hasattr(value, "type") else value)
    for operation in function.body.block.ops:
        _operation(operation, functions, (*active, function), allow_named_linalg=allow_named_linalg)


def _value_type(typ):
    if isinstance(typ, builtin.TensorType):
        if not isinstance(typ.encoding, builtin.NoneAttr):
            raise ValueError("prepared source tensor encoding has unknown effects")
        _value_type(typ.element_type)
    elif not isinstance(typ, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
        raise ValueError("prepared source may not expose buffer or address values")


def _operation(operation, functions, active, *, allow_named_linalg=False):
    for value in (*operation.operands, *operation.results):
        _value_type(value.type)
    if isinstance(operation, func.CallOp):
        if operation.callee.nested_references:
            raise ValueError("prepared source call has unsupported nested symbol resolution")
        parent = operation.parent_op()
        while parent is not None and not isinstance(parent, func.FuncOp):
            parent = parent.parent_op()
        if parent not in functions.values():
            raise ValueError("prepared source call is outside the selected root function scope")
        callee = functions.get(operation.callee.root_reference.data)
        if callee is None:
            raise ValueError("prepared source call has unknown external effects")
        validate_pure_tensor_function(callee, functions, active=active, allow_named_linalg=allow_named_linalg)
        return
    if type(operation) in (scf.ForOp, scf.IfOp, scf.YieldOp):
        if operation.properties or any(not key.startswith("prov.") for key in operation.attributes):
            raise ValueError("prepared source control operation has unknown execution attributes")
        # These exact serial control operations have only the effects of their
        # complete nested bodies, checked below. No generic region wrapper or
        # unknown loop inherits this fact. Verification checks carried types,
        # bounds operands and terminator structure before recursive inspection.
        operation.verify()
    elif isinstance(operation, (func.ReturnOp, YieldOp)):
        return
    elif isinstance(operation, (GenericOp, ReduceOp)) or (
        allow_named_linalg and isinstance(operation, LinalgStructuredOperation)
    ):
        if "library_call" in operation.attributes or "library_call" in operation.properties:
            raise ValueError("prepared source library override has unknown external effects")
        # The only admitted effect-free structured special case: immutable
        # tensor values with pure scalar regions. No buffer outputs allowed.
        if not operation.results or any(not isinstance(v.type, builtin.TensorType) for v in operation.results):
            raise ValueError("prepared source structured operation requires tensor results")
    elif type(operation) is IndexOp:
        # This xDSL version gives linalg.index its parent constraint but no
        # effect trait. It reads only the enclosing generic's logical iteration
        # coordinate. Admit that exact typed operation and a valid axis; no
        # opaque operation or buffer obtains a purity assumption from its name.
        parent = operation.parent_op()
        if (
            not isinstance(parent, GenericOp)
            or not parent.results
            or any(not isinstance(value.type, builtin.TensorType) for value in parent.operands)
            or any(not key.startswith("prov.") for key in operation.attributes)
            or set(operation.properties) != {"dim"}
            or not 0 <= operation.dim.value.data < len(parent.iterator_types)
        ):
            raise ValueError("prepared source index requires an ordinary tensor generic and a valid axis")
        operation.verify()
    elif (
        isinstance(operation, arith.ExtSIOp)
        and not operation.properties
        and all(key.startswith("prov.") for key in operation.attributes)
    ):
        # The supported xDSL version omits Pure on this standard typed integer
        # widening operation. Its exact sign extension has no memory effects;
        # no unknown arithmetic operation inherits this specific fact.
        pass
    elif get_effects(operation) != set():
        raise ValueError("prepared source operation has unknown or nonempty effects")
    for region in operation.regions:
        for block in region.blocks:
            for argument in block.args:
                _value_type(argument.type)
            for nested in block.ops:
                _operation(nested, functions, active, allow_named_linalg=allow_named_linalg)


def validate_prepared_source_lifetime(opportunity, module):
    """Check every operation from first borrow through the last source consumer.

    This is deliberately conservative: even an unrelated unknown memory effect
    refuses. The result grants no physical cached-storage or external ABI proof.
    """
    validate_tensor_preparation_opportunity(opportunity)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    operations = tuple(opportunity.owner_block.ops)
    positions = [operations.index(call) for call in opportunity.consumers]
    if positions != sorted(set(positions)):
        raise ValueError("prepared source consumers are not in unique source order")
    for operation in operations[positions[0] : positions[-1] + 1]:
        _operation(operation, functions, ())
