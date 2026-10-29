"""Compare a whole-model output with the reference its device groups' declared contracts admit.

A device group can declare that its integer output lies within ``bound_lsb`` output steps of the
captured program's single rounding: an operand sum whose scaled load rounds each operand before an
exact add, or a window mean that rounds an exact integer sum. The torch-eager golden applies the
single rounding, so a later contraction can amplify an admitted one-step difference past the
model's tolerance even though the candidate computed exactly what the group declared.

This module reads, from the captured program, the exact linear map from each such group's integer
output to the model outputs (views, transposes, dequantizes, integer or float contractions and
scalings by stored constants). It then looks for one integer offset per group element, within the
group's declared bound, that explains the observed outputs, and reruns the unchanged comparison
(same policy, same atol/rtol) against the eager golden plus that offset's effect. Accepting
requires that witness; anything else leaves the eager verdict standing:

* an output policy other than a float tolerance (integer outputs are compared exactly);
* a group declared exact (bound zero), or one the candidate did not place on the device;
* a group whose path to an output is not linear in its integers (a requantize, a norm, a softmax);
* an offset outside a bound, or a residual outside the tolerance after the offset.

The witness is checked, never assumed: the least-squares estimate is rounded and clipped to the
bounds, and only the comparison of its reference decides.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Largest Jacobian (group elements times model-output elements, or times one intermediate's
#: elements) this check materializes; a larger one gets no allowance rather than a slow grade.
MAX_JACOBIAN_ENTRIES = 1 << 24

_RESHAPES = frozenset({"tensor.collapse_shape", "tensor.expand_shape", "tensor.reshape", "tensor.cast"})
_CASTS = frozenset({"arith.extsi", "arith.sitofp", "arith.extf"})
_MULS = frozenset({"arith.mulf", "arith.muli"})
_ADDS = frozenset({"arith.addf", "arith.addi"})
_SUBS = frozenset({"arith.subf", "arith.subi"})


class NotLinear(ValueError):
    """The group's output reaches a model output through something other than a linear map."""


def declared_bound(group) -> int:
    """The output steps a device group declares it may lie from the single rounding, else zero."""
    for contract in (getattr(group, "operand_sum", None), getattr(group, "window_mean", None)):
        bound = contract.get("bound_lsb") if isinstance(contract, Mapping) else None
        if type(bound) is int and bound > 0:
            return bound
    return 0


def group_sink(group):
    """The one value a group hands to work outside it."""
    inside = {id(member) for member in group.members}
    sinks = [
        value
        for member in group.members
        for value in member.results
        if any(id(use.operation) not in inside for use in value.uses)
    ]
    if len(sinks) != 1:
        raise NotLinear(f"the group hands {len(sinks)} values to later work, not one")
    return sinks[0]


# ------------------------------------------------------------------------------------- constants


def _shape(value) -> tuple[int, ...]:
    from merlin.common import mlir_query as mq

    shape, _ = mq.type_shape_dtype(value.type)
    if any(not isinstance(extent, int) or extent < 0 for extent in shape):
        raise NotLinear("a tensor on the path has a dynamic extent")
    return tuple(shape)


class Constants:
    """Concrete values of stored arguments and compile-time constants, by structure only."""

    def __init__(self, capture: str | Path | None = None):
        from merlin.xdsl_dialects.lowering.output_fusion import Values

        self._stored = Values(capture)

    def array(self, value):
        import numpy as np
        from xdsl.ir import BlockArgument

        from merlin.common import mlir_query as mq

        shape = _shape(value)
        if isinstance(value, BlockArgument):
            if self._stored.manifest is None:
                raise NotLinear("a model argument on the path has no stored value")
            from merlin.xdsl_dialects.lowering import group_prepack as GP

            try:
                _name, raw, _dtype = GP._stored_raw(int(value.index), self._stored.manifest, self._stored.weights)  # noqa: SLF001
            except Exception as error:  # noqa: BLE001 -- an unreadable argument is simply not a constant
                raise NotLinear(f"model argument {value.index} is not a stored constant: {error}") from error
            return np.asarray(raw, dtype=np.float64).reshape(shape)
        owner = value.owner
        name = mq.op_name(owner)
        if name == "arith.constant":
            attribute = owner.properties.get("value") or owner.attributes.get("value")
            data = getattr(getattr(attribute, "value", None), "data", None)
            if isinstance(data, (int, float)) and not isinstance(data, bool):
                return np.full(shape, float(data))
            listed = list(attribute.get_values()) if callable(getattr(attribute, "get_values", None)) else []
            if len(listed) == 1:
                return np.full(shape, float(listed[0]))
            if listed and int(np.prod(shape)) == len(listed):
                return np.asarray(listed, dtype=np.float64).reshape(shape)
            raise NotLinear("a constant on the path is not a dense number")
        if name in ("tensor.splat", "linalg.fill") and owner.operands:
            scalar = self.array(owner.operands[0])
            if scalar.size != 1:
                raise NotLinear(f"{name} fills from a non-scalar")
            return np.full(shape, float(scalar.reshape(-1)[0]))
        if name in _RESHAPES and owner.operands:
            return self.array(owner.operands[0]).reshape(shape)
        if name == "linalg.transpose" and owner.operands:
            return np.transpose(self.array(owner.operands[0]), _permutation(owner))
        raise NotLinear(f"{name} on the path is not a compile-time or stored constant")


def _permutation(op) -> tuple[int, ...]:
    attribute = op.properties.get("permutation") or op.attributes.get("permutation")
    return tuple(int(axis) for axis in attribute.get_values())


# ---------------------------------------------------------------------------------- linalg body


def _body_form(op, perturbed: int) -> tuple[float, tuple[int, ...], bool]:
    """``(coefficient, factor operands, accumulates)`` when a generic's result is linear in operand
    ``perturbed``: ``coefficient * x * prod(factors)`` per point, summed over reduction dimensions
    into the output when ``accumulates``. Anything else raises :class:`NotLinear`."""
    from xdsl.ir import BlockArgument

    from merlin.common import mlir_query as mq

    block = op.regions[0].blocks[0]
    n_inputs = len(op.inputs)
    tags: dict[int, tuple] = {}

    def tag(value):
        if isinstance(value, BlockArgument) and value.owner is block:
            index = int(value.index)
            if index == perturbed:
                return ("lin", 1.0, ())
            return ("acc",) if index >= n_inputs else ("const", 1.0, (index,))
        if id(value) in tags:
            return tags[id(value)]
        # A value captured from outside the body: only a scalar constant has a known number.
        try:
            number = Constants().array(value)
        except NotLinear:
            return ("opaque",)
        return ("const", float(number.reshape(-1)[0]), ()) if number.size == 1 else ("opaque",)

    for inner in block.ops:
        name = mq.op_name(inner)
        operands = [tag(operand) for operand in inner.operands]
        kinds = [entry[0] for entry in operands]
        if name in ("linalg.yield",):
            if len(operands) != 1:
                raise NotLinear("the body yields more than one value")
            result = operands[0]
            if result[0] == "lin":
                return result[1], result[2], False
            if result[0] == "accsum":
                return result[1], result[2], True
            raise NotLinear("the body's result does not depend linearly on the group's integers")
        if "lin" not in kinds and "accsum" not in kinds:
            if "acc" in kinds:
                raise NotLinear(f"{name} reads the accumulator other than to add one linear term")
            if name == "arith.constant":
                number = Constants().array(inner.results[0])
                out = ("const", float(number.reshape(-1)[0]), ())
            elif name in _MULS and all(kind == "const" for kind in kinds):
                out = ("const", operands[0][1] * operands[1][1], operands[0][2] + operands[1][2])
            elif name in _CASTS and kinds == ["const"]:
                out = operands[0]
            else:
                out = ("opaque",)
        elif name in _CASTS and kinds == ["lin"]:
            out = operands[0]
        elif name in _MULS and sorted(kinds) == ["const", "lin"]:
            lin, const = (operands[0], operands[1]) if kinds[0] == "lin" else (operands[1], operands[0])
            out = ("lin", lin[1] * const[1], lin[2] + const[2])
        elif name in _ADDS and kinds.count("lin") == 1 and set(kinds) <= {"lin", "const", "opaque"}:
            out = operands[kinds.index("lin")]
        elif name in _ADDS and sorted(kinds) == ["acc", "lin"]:
            lin = operands[kinds.index("lin")]
            out = ("accsum", lin[1], lin[2])
        elif name in _SUBS and kinds[0] == "lin" and kinds[1] in ("const", "opaque"):
            out = operands[0]
        elif name in _SUBS and kinds[1] == "lin" and kinds[0] in ("const", "opaque"):
            out = ("lin", -operands[1][1], operands[1][2])
        elif name == "arith.negf" and kinds == ["lin"]:
            out = ("lin", -operands[0][1], operands[0][2])
        elif name == "arith.divf" and kinds[0] == "lin" and kinds[1] == "const" and not operands[1][2]:
            out = ("lin", operands[0][1] / operands[1][1], operands[0][2])
        else:
            raise NotLinear(f"{name} is not linear in the group's integers")
        for result in inner.results:
            tags[id(result)] = out
    raise NotLinear("the body has no yield")


def _subscripts(affine_map, letters: str, shape: tuple[int, ...]) -> tuple[str, tuple]:
    """Einsum letters of one operand, and the index that selects its constant (broadcast) axes."""
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    out, select = "", []
    for axis, expression in enumerate(affine_map.results):
        if isinstance(expression, AffineDimExpr):
            out += letters[expression.position]
            select.append(slice(None))
        elif isinstance(expression, AffineConstantExpr) and 0 <= expression.value < shape[axis]:
            select.append(int(expression.value))
        else:
            raise NotLinear("an indexing map is not a projection of the loop dimensions")
    return out, tuple(select)


def _generic(op, perturbed: int, delta, constants: Constants):
    import numpy as np

    coefficient, factors, accumulates = _body_form(op, perturbed)
    maps = [attribute.data for attribute in (op.properties.get("indexing_maps") or op.attributes["indexing_maps"]).data]
    iterators = [str(getattr(kind, "data", kind)).rsplit(".", 1)[-1].strip("<>") for kind in op.iterator_types.data]
    if len(op.outputs) != 1 or len(op.results) != 1:
        raise NotLinear("the generic does not produce exactly one tensor")
    dims = maps[0].num_dims
    if dims > 25:
        raise NotLinear("the generic has more loop dimensions than this check names")
    letters = "abcdefghijklmnopqrstuvwxy"[:dims]
    operands = [*op.inputs, *op.outputs]
    out_letters, out_select = _subscripts(maps[len(op.inputs)], letters, _shape(op.outputs[0]))
    if any(item != slice(None) for item in out_select) or len(set(out_letters)) != len(out_letters):
        raise NotLinear("the generic writes its result through a non-permutation map")
    reduced = [letters[d] for d in range(dims) if letters[d] not in out_letters]
    if any("parallel" in iterators[letters.index(d)] for d in reduced) or (bool(reduced) and not accumulates):
        raise NotLinear("the generic sums dimensions without accumulating into its output")
    if accumulates and not reduced:
        raise NotLinear("the generic adds its output to a linear term without a reduction")
    if accumulates and _depends_on_init(op):
        raise NotLinear("the reduction's initial value is not independent of the group")
    lhs, select = _subscripts(maps[perturbed], letters, _shape(operands[perturbed]))
    terms, arrays = ["Z" + lhs], [delta[(slice(None), *select)]]
    for index in factors:
        letters_f, select_f = _subscripts(maps[index], letters, _shape(operands[index]))
        terms.append(letters_f)
        arrays.append(constants.array(operands[index])[select_f])
    return coefficient * np.einsum(",".join(terms) + "->Z" + out_letters, *arrays, optimize=True)


def _depends_on_init(op) -> bool:
    """Whether a reduction's initial value is anything but a fill of a constant (zero is not required:
    a constant start adds the same amount to both references)."""
    from merlin.common import mlir_query as mq

    init = op.outputs[0]
    owner = getattr(init, "owner", None)
    return owner is None or mq.op_name(owner) not in ("tensor.splat", "linalg.fill", "arith.constant", "tensor.empty")


# --------------------------------------------------------------------------------- propagation


def output_jacobians(sink, constants: Constants) -> list:
    """For each returned value of the function holding ``sink``, ``d(output)/d(sink)`` with shape
    ``(sink elements, *output shape)``; raises :class:`NotLinear` when some path is not linear."""
    import numpy as np

    from merlin.common import mlir_query as mq

    elements = int(np.prod(_shape(sink)))
    if elements * elements > MAX_JACOBIAN_ENTRIES:
        raise NotLinear("the group's output is too large to map element by element")
    delta = {id(sink): np.eye(elements).reshape((elements, *_shape(sink)))}
    block = sink.owner.parent
    started = False
    for op in block.ops:
        if not started:
            started = op is sink.owner
            continue
        moved = [index for index, operand in enumerate(op.operands) if id(operand) in delta]
        if not moved:
            continue
        name = mq.op_name(op)
        if name == "func.return":
            return [
                delta[id(value)] if id(value) in delta else np.zeros((elements, *_shape(value)))
                for value in op.operands
            ]
        if len(op.results) != 1:
            raise NotLinear(f"{name} reads the group's integers and does not produce one tensor")
        result = op.results[0]
        if elements * int(np.prod(_shape(result))) > MAX_JACOBIAN_ENTRIES:
            raise NotLinear("the map from the group to the outputs is too large to materialize")
        source = delta[id(op.operands[moved[0]])]
        if name in _RESHAPES and moved == [0]:
            delta[id(result)] = source.reshape((elements, *_shape(result)))
        elif name == "linalg.transpose" and moved == [0]:
            delta[id(result)] = np.transpose(source, (0, *(1 + axis for axis in _permutation(op))))
        elif name.rpartition(".")[2] == "dequantize_per_tensor" and moved == [0]:
            delta[id(result)] = source * float(constants.array(op.operands[1]).reshape(-1)[0])
        elif name == "linalg.generic" and len(moved) == 1 and moved[0] < len(op.inputs):
            delta[id(result)] = _generic(op, moved[0], source, constants)
        else:
            raise NotLinear(f"{name} is not linear in the group's integers")
    raise NotLinear("the function returns nothing the group reaches")


# ------------------------------------------------------------------------------------- witness


def _flat(outputs: Mapping[str, Any], order: Sequence[str]):
    import numpy as np

    return np.concatenate([np.asarray(outputs[name], dtype=np.float64).reshape(-1) for name in order])


def explain(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    order: Sequence[str],
    groups: Sequence[Any],
    constants: Constants,
    golden_source: str,
    compare: Callable[..., dict],
) -> dict:
    """The eager ``report`` unchanged, or a passing one against the declared-contract reference.

    ``groups`` are the device groups the candidate placed; ``order`` names the function's results in
    return order. A group that declares no bound contributes nothing.
    """
    import numpy as np

    if report.get("status") == "pass" or not _float_policy(policy):
        return dict(report)
    bounded, unexplained = [], []
    for group in groups:
        bound = declared_bound(group)
        if not bound:
            continue
        label = _label(group)
        try:
            jacobians = output_jacobians(group_sink(group), constants)
        except NotLinear as error:
            unexplained.append({"group": label, "bound_lsb": bound, "reason": str(error)})
            continue
        if len(jacobians) != len(order):
            unexplained.append({"group": label, "bound_lsb": bound, "reason": "output roster differs"})
            continue
        rows = jacobians[0].shape[0]
        bounded.append((label, bound, np.concatenate([j.reshape(rows, -1) for j in jacobians], axis=1)))
    if not bounded and not unexplained:
        return dict(report)
    contracts: dict[str, Any] = {"status": "not_explained", "groups": [], "unexplained_groups": unexplained}
    if not bounded:
        contracts["reason"] = "no declared-bound group reaches an output through a linear map"
        return {**report, "declared_group_contracts": contracts}
    eager, seen = _flat(expected, order), _flat(observed, order)
    effect = np.concatenate([matrix for _label_, _bound, matrix in bounded], axis=0)
    if effect.size > MAX_JACOBIAN_ENTRIES or not np.isfinite(seen).all():
        contracts["reason"] = "the observed outputs are not finite or the map is too large"
        return {**report, "declared_group_contracts": contracts}
    estimate, *_ = np.linalg.lstsq(effect.T, seen - eager, rcond=None)
    steps, start = np.rint(estimate), 0
    for label, bound, matrix in bounded:
        part = steps[start : start + matrix.shape[0]]
        np.clip(part, -bound, bound, out=part)
        changed = int(np.count_nonzero(part))
        contracts["groups"].append(
            {
                "group": label,
                "bound_lsb": bound,
                "elements": int(matrix.shape[0]),
                "offset_elements": changed,
                "max_offset_steps": int(np.abs(part).max()) if changed else 0,
            }
        )
        start += matrix.shape[0]
    reference = eager + steps @ effect
    # What is left once the offsets are applied; the policy's own tolerance is still what judges it.
    contracts["max_residual"] = float(np.abs(seen - reference).max()) if seen.size else 0.0
    shaped, offset = {}, 0
    for name in order:
        shape = np.asarray(expected[name]).shape
        size = int(np.prod(shape))
        shaped[name] = reference[offset : offset + size].reshape(shape).tolist()
        offset += size
    try:
        verdict = compare(shaped, dict(observed), dict(policy), golden_source=golden_source)
    except ValueError as error:
        contracts["reason"] = f"the contract reference cannot be compared: {error}"
        return {**report, "declared_group_contracts": contracts}
    if verdict.get("status") != "pass":
        contracts["reason"] = (
            "no offset within the declared bounds explains the observed outputs within the policy tolerance"
        )
        contracts["contract_reference"] = _summary(verdict)
        return {**report, "declared_group_contracts": contracts}
    contracts["status"] = "explained"
    moved = [row for row in contracts["groups"] if row["offset_elements"]]
    contracts["note"] = "matches the declared group contracts" + (
        "; differs from eager by "
        + ", ".join(
            f"<= {row['max_offset_steps']} step(s) on {row['offset_elements']}/{row['elements']} "
            f"elements at {row['group']}"
            for row in moved
        )
        if moved
        else ""
    )
    return {
        **verdict,
        "reference": "eager_with_declared_group_offsets",
        "eager": _summary(report),
        "declared_group_contracts": contracts,
    }


def _float_policy(policy: Mapping[str, Any]) -> bool:
    if policy.get("compare") == "tolerance_float":
        return True
    outputs = policy.get("outputs")
    return (
        policy.get("compare") == "per_output"
        and isinstance(outputs, Mapping)
        and any(isinstance(child, Mapping) and child.get("compare") == "tolerance_float" for child in outputs.values())
    )


def _summary(report: Mapping[str, Any]) -> dict:
    keys = ("status", "mismatch_count", "max_abs_error", "max_rel_error", "first_mismatch")
    return {key: report.get(key) for key in keys}


def _label(group) -> str:
    from merlin.common import mlir_query as mq

    region = mq.attr_str(group.root, "prov.region_id") if getattr(group, "root", None) is not None else None
    return region or f"group {getattr(group, 'index', '?')}"


# ------------------------------------------------------------------------------ owned-model use


def placed_groups(groups, join: Mapping[str, Any] | None, *, host: str) -> list:
    """The device groups whose source identity the candidate's linked build actually routed."""
    if not isinstance(join, Mapping) or not isinstance(join.get("source"), Mapping):
        return []
    source = join["source"]
    routed_optional = set(join.get("optional_source_groups_routed") or ())
    rows = [*(source.get("eligible_group_provenance") or ())]
    rows += [row for row in source.get("optional_group_provenance") or () if row.get("source_group") in routed_optional]
    regions = {row.get("source_region_id") for row in rows if isinstance(row, Mapping)}
    return [
        group
        for group in groups
        if group.placement != host and getattr(group, "root", None) is not None and _label(group) in regions
    ]


def owned_model_reference(report, expected, observed, policy, *, record, order, golden_source, compare) -> dict:
    """:func:`explain` for an owned-host model record: re-form the build's exact source groups."""
    if report.get("status") == "pass" or not _float_policy(policy):
        return dict(report)
    join, inputs = record.get("source_device_join"), (record.get("build") or {}).get("inputs")
    if not isinstance(join, Mapping) or not isinstance(inputs, Mapping) or "runtime_inputs" not in record:
        return dict(report)  # not a source-joined whole-model build: no placed group to read
    try:
        from merlin.common import mlir_query as mq
        from merlin.llvmlower.source_compute_groups import form_exact_groups
        from merlin.xdsl_dialects.lowering import compute_groups as CG

        bundle, target = Path(record["runtime_inputs"]), inputs["target"]
        module = mq.parse((bundle / "model.mlir").read_text())
        groups, _ = form_exact_groups(module, target, bundle, CG.form_groups)
        placed = placed_groups(groups, join, host=CG.HOST)
        return explain(
            report,
            expected,
            observed,
            policy,
            order=order,
            groups=placed,
            constants=Constants(bundle),
            golden_source=golden_source,
            compare=compare,
        )
    except Exception as error:  # noqa: BLE001 -- no allowance is ever granted by an analysis that failed
        reason = f"{type(error).__name__}: {error}"
        return {**report, "declared_group_contracts": {"status": "not_explained", "reason": reason}}
