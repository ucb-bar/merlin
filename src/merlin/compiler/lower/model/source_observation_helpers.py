"""Reify closed scalar observation helpers from their current typed source proof.

The helpers clone the proved arithmetic, including original operation order,
precision, attributes and literals. No expression grammar, workload label or
retained compiled body selects the implementation. Coordinates, ownership,
numeric-mode guards and subsequent helper-call binding remain caller obligations.
"""

from __future__ import annotations

from merlin.compiler.lower.model.source_expression_interval import (
    ClosedScalarObserver,
    IntervalEffectContract,
    validate_closed_scalar_observer,
)


def reify_closed_scalar_observer_helpers(
    proof: ClosedScalarObserver,
    *,
    effects: IntervalEffectContract,
    expression_symbol: str,
    observer_symbol: str,
):
    """Return fresh ``f32→f32`` expression and ``f32→i8`` observation functions.

    The observer accepts the original second rounded finishing multiply's result;
    neither finishing multiply is moved, folded or reproduced here. The current
    source witness and all enclosing numeric conditions are checked before and
    after cloning. A stale proof, missing dependency or invalid ABI symbol refuses.
    The source operations and their use lists are unchanged.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp, StringAttr, f32, i8
    from xdsl.ir import Block, Region

    if not isinstance(proof, ClosedScalarObserver):
        raise ValueError("current typed ClosedScalarObserver proof required")
    if not isinstance(effects, IntervalEffectContract):
        raise ValueError("explicit IntervalEffectContract required")
    effects.validate()
    validate_closed_scalar_observer(proof)
    for symbol in (expression_symbol, observer_symbol):
        if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
            raise ValueError("explicit nonempty ASCII helper ABI symbol required")
    if expression_symbol == observer_symbol:
        raise ValueError("distinct expression and observer ABI symbols required")

    def trace(value):
        result = {}
        operation = value.owner
        while hasattr(operation, "parent_op"):
            for key, attribute in operation.attributes.items():
                if key.startswith("prov."):
                    result.setdefault(key, attribute)
            operation = operation.parent_op()
            if operation is None:
                break
        # Preserve enclosing trace data alongside a derived transformation record.
        additions = {
            "prov.reification": StringAttr("closed_scalar_observer_helper"),
            "prov.reified_expression_sha256": StringAttr(proof.expression.canonical_sha256),
        }
        for key, value in additions.items():
            if key in result and result[key] != value:
                raise ValueError("source provenance conflicts with helper reification record")
            result[key] = value
        return result

    def function(symbol, source_input, source_output, operations):
        block = Block(arg_types=[f32])
        mapping = {source_input: block.args[0]}
        for operation in operations:
            if any(operand not in mapping for operand in operation.operands):
                raise ValueError("source helper dependency is not inside its proved scalar DAG")
            block.add_op(operation.clone(mapping))
        if source_output not in mapping:
            raise ValueError("proved scalar helper output is not materialized")
        returned = func.ReturnOp(mapping[source_output])
        block.add_op(returned)
        result = func.FuncOp(symbol, ([f32], [source_output.type]), Region(block))
        result.attributes.update(trace(source_output))
        return result

    expression = function(expression_symbol, proof.cut, proof.endpoint, proof.expression_operations)
    scaled = proof.observer_operations[1].results[0]
    observer = function(observer_symbol, scaled, proof.integer_result, proof.observer_operations[2:])
    if expression.function_type.outputs.data != (f32,) or observer.function_type.outputs.data != (i8,):
        raise ValueError("proved scalar helper ABI changed")
    module = ModuleOp([expression, observer])
    module.verify()
    validate_closed_scalar_observer(proof)
    return module
