"""Discover and validate source readout before canonical Interface materialization.

Matching stays independent of target encodings or schedules. One shared matcher
checks the scalar SSA chain and exact parameter maps. The
canonical record retains original scalar/channel constants and f32 bias order.
A target must separately prove its readout implements that arithmetic. Dynamic
parameters, unsupported maps and other consumers produce a clear refusal.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import DictionaryAttr
from xdsl.ir import Operation

from merlin.compiler.ir.readout import constant_parameter, parameter_readout_contract, readout_contract
from merlin.compiler.lower.quant.requantization import match
from merlin.compiler.lower.quant.readout_match import (
    F32ReadoutMatch as F32ReadoutMatch,
    _tensor_spec as _tensor_spec,
    match_f32_readout as match_f32_readout,
)


def match_integer_saturating_cast(op):
    """Prove pointwise max(x,-128), min(x,127), trunc-to-i8 exactly."""
    if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    source = op.inputs[0]
    if _tensor_spec(source)["dtype"] != "i32" or _tensor_spec(op.results[0]) != {**_tensor_spec(source), "dtype": "i8"}:
        return None
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    rank = len(_tensor_spec(source)["shape"])
    ident = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank)))
    if tuple(a.data for a in op.indexing_maps) != (ident, ident) or any(
        str(a.data.value) != "parallel" for a in op.iterator_types
    ):
        return None
    body = op.regions[0].block
    ops = list(body.ops)
    if [o.name for o in ops] != ["arith.maxsi", "arith.minsi", "arith.trunci", "linalg.yield"]:
        return None

    def constant(v, expected):
        owner = v.owner
        attr = getattr(owner, "properties", {}).get("value", getattr(owner, "attributes", {}).get("value"))
        return (
            getattr(owner, "name", "") == "arith.constant"
            and getattr(getattr(attr, "value", None), "data", None) == expected
        )

    lo, hi, cast, yld = ops
    if body.args[0] not in lo.operands or not any(constant(v, -128) for v in lo.operands):
        return None
    if lo.results[0] not in hi.operands or not any(constant(v, 127) for v in hi.operands):
        return None
    if list(cast.operands) != [hi.results[0]] or list(yld.operands) != [cast.results[0]]:
        return None
    return source


@dataclass(frozen=True)
class ReadoutMatch:
    operations: tuple[Operation, ...]
    contract: DictionaryAttr
    stages: tuple[str, ...]
    proof: dict
    parameters: tuple = ()


def _parameters(epilogue, contraction):
    """Account for immutable scalar parameters and unused output storage."""
    covered = set()
    scalar_members = set(epilogue.walk())
    pending = [value for operation in scalar_members for value in operation.operands]
    while pending:
        value = pending.pop()
        owner = value.owner
        if owner is contraction or not isinstance(owner, Operation) or owner is epilogue or owner in covered:
            continue
        if owner.parent is not contraction.parent:
            # Scalar operations inside the matched epilogue are already checked
            # as part of its exact SSA pattern, rather than extra top-level work.
            if owner in scalar_members:
                continue
            raise ValueError("readout parameter escapes its complete source function")
        if owner.name not in ("arith.constant", "tensor.empty") or owner.operands or owner.regions:
            raise ValueError("readout requires immutable constants and static output storage")
        covered.add(owner)
        pending.extend(owner.operands)
    return covered


def match_readout(contraction, returned_value):
    """Match -> check exact source legality -> return materialization inputs.

    None means a raw contraction result. Every other returned value must match
    one complete supported readout; failures never silently keep a partial cut.
    """
    if returned_value is contraction.results[0]:
        return None
    epilogue = returned_value.owner
    if (
        not isinstance(epilogue, Operation)
        or epilogue.parent is not contraction.parent
        or tuple(epilogue.results) != (returned_value,)
    ):
        raise ValueError("returned source value is not one complete local readout")
    if len(list(contraction.results[0].uses)) != 1:
        raise ValueError("readout fusion has another live accumulator consumer")
    retained = ()
    source = match_integer_saturating_cast(epilogue)
    if source is contraction.results[0]:
        contract = readout_contract(integer=True)
        stages = ()
        proof = {"source": "exact signed-i32 clamp [-128,127] then i8 truncation"}
    else:
        selected = match_f32_readout(epilogue)
        if selected.contraction is not contraction:
            raise ValueError("readout matched a different contraction")
        parameterized = any(
            kind == "bias" or constant_parameter(selected.parameters[index])[1] for kind, index in selected.steps
        )
        if parameterized:
            contract = parameter_readout_contract(
                selected.steps,
                selected.parameters,
                output_shape=tuple(epilogue.results[0].type.get_shape()),
                relu=selected.relu,
            )
            proof = {
                "source": "exact mapped binary32 scalar SSA chain; target implementation not selected",
                "steps": [kind for kind, _ in selected.steps],
            }
            retained = selected.parameters
            stages = ("acc_scale",) + (("relu",) if selected.relu else ())
        else:
            uniform = match(epilogue)
            contract = readout_contract(uniform.proof["source_scales"], relu=uniform.relu)
            stages = ("acc_scale",) + (("relu",) if uniform.relu else ())
            proof = uniform.proof
    parameters = _parameters(epilogue, contraction)
    members = tuple(
        operation for operation in contraction.parent.ops if operation is epilogue or operation in parameters
    )
    owned = {contraction, *members}
    scalar_members = set(epilogue.walk())
    if any(
        value is not returned_value
        and any(use.operation not in owned and use.operation not in scalar_members for use in value.uses)
        for operation in owned
        for value in operation.results
    ):
        raise ValueError("readout fusion has another live intermediate output")
    return ReadoutMatch(members, contract, stages, proof, retained)
