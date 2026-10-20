"""Deterministic capsule entries for captured compute groups, and the one binding they are built under.

Restates compiler grouping decisions in the shared capsule vocabulary. No corpus
writes, golden generation, grading, promotion, or optional experiment imports.

Every consumer that turns ONE captured group into an interface capsule -- the derived model-form
and form-perf capsules, the capsule a whole-program build asks a package for, a grading harness --
goes through :func:`group_binding` and :func:`interface_capsule`. So the capsule that certifies a
group and the request a model build sends for that group are the same bytes under the same binding:
one grouping (``compute_groups`` + ``group_command``), one binding (``corpus_spec.derive_binding``).
Writing a capsule to disk (golden, scrub, instruction classes) belongs to the Phase 0 writer.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from math import prod
from typing import Any

SCHEMA = "group_capsules_v1"
#: The schema's own role for a capsule whose shape is a real model's and deliberately not tile-relative.
SOURCE_ROLE = "model_derived"
#: Entry keys that make two groups the same program. The name and the multiplier's VALUE do not:
#: a unit's command stream is the same for every positive multiplier.
_IDENTITY_DROPS = ("name", "acc_scale", "lhs_scale", "rhs_scale", "comment", "source_reference")


def _identity(entry: Mapping[str, Any]) -> str:
    return json.dumps({k: v for k, v in entry.items() if k not in _IDENTITY_DROPS}, sort_keys=True)


def _numerical_identity(entry: Mapping[str, Any]) -> str:
    """Keep multiplier values when the consumer will bind an exact golden."""
    return json.dumps(
        {k: v for k, v in entry.items() if k not in {"name", "comment", "source_reference"}},
        sort_keys=True,
    )


def _label(entry: Mapping[str, Any]) -> str:
    stages = "_".join(str(s) for s in entry.get("epilogue") or ()) or "raw"
    if entry["op"] == "conv2d":
        extent = (
            f"c{entry['ci']}x{entry['Himg']}x{entry['Wimg']}_"
            f"k{entry['kh']}x{entry['kw']}s{entry['stride'][0]}_n{entry['N']}"
        )
    elif "K" in entry:
        extent = f"m{entry['M']}k{entry['K']}n{entry['N']}"
    else:
        # An elementwise entry reduces over nothing. Its declared bound is part of what is asked:
        # the same extents under two bounds are two demands.
        extent = f"m{entry['M']}n{entry['N']}" + (f"_b{entry['bound_lsb']}" if "bound_lsb" in entry else "")
    return f"G_{entry['op']}_{extent}_{stages}"


def _element_range(dtype: str) -> tuple[int, int]:
    """The whole range of an integer element format, from the format registry."""
    from merlin.common import quant_formats as qf
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE

    try:
        fmt = qf.get(dtype)
    except (KeyError, ValueError):
        return tuple(SIGNED_STIMULUS_RANGE)
    if fmt.kind != "int_affine":
        return tuple(SIGNED_STIMULUS_RANGE)
    return -(1 << (fmt.element_bits - 1)), (1 << (fmt.element_bits - 1)) - 1


class UnderivedAccumulator(ValueError):
    """Neither the declared numeric policy, the contract nor the RTL facts state an accumulator."""


def group_binding(
    te,
    datapath: Mapping[str, Any] | None,
    *,
    operand_dtype: str | None = None,
    contract: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
    taxonomy: Mapping[str, Any] | None = None,
    defer_classes: bool = False,
):
    """The ``CorpusBinding`` a captured group is stated under -- the same one the corpus is built with.

    ``datapath`` is the explicitly loaded recipe's ``datapath`` block, exactly as the corpus generator
    receives it (tiers, requantization and tolerances included), so a group capsule and every other
    corpus member share one binding. ``operand_dtype`` overlays the group's own operand format. The ACCUMULATOR is never supplied by the caller: it is the declared policy's
    ``accum_dtype`` when the recipe states one, else what the contract or the RTL facts state
    (:func:`~merlin.targetgen.corpus_spec.derived_accumulator`). When none of them does, this raises
    :class:`UnderivedAccumulator` rather than assuming a widening default.
    """
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.target_experiment import load_capability_manifest

    selected = CS.profile_datapath({"datapath": dict(datapath or {})})
    if operand_dtype:
        selected["operand_dtype"] = str(operand_dtype)
    if contract is not None:
        effective = dict(contract)
    else:
        try:
            effective = load_capability_manifest(te.target).contract
        except Exception:  # noqa: BLE001 -- derive_binding reads the same manifest and raises its own error
            effective = None
    if effective is not None and not selected.get("accum_dtype"):
        units = effective.get("compute_units") or [{}]
        operand = selected.get("operand_dtype") or ((units[0] or {}).get("dtypes") or [None])[0]
        if facts is None:
            try:
                from merlin.targetgen.rtl.facts import load_facts

                facts = load_facts(te.target)
            except Exception:  # noqa: BLE001 -- no facts: the contract alone must state it
                facts = None
        accumulator = (
            CS.derived_accumulator(effective, str(operand), dict(facts) if facts else None) if operand else None
        )
        if accumulator is None:
            raise UnderivedAccumulator(
                f"{te.target}: no declared accum_dtype, contract accumulate row or RTL accumulator datapath "
                f"for operand {operand!r}; a group capsule cannot be stated under an assumed accumulator"
            )
        selected["accum_dtype"] = accumulator
    evidence = {
        "contract": effective if contract is not None else None,
        "facts": dict(facts) if facts else None,
        "taxonomy": dict(taxonomy) if taxonomy is not None else None,
    }
    return CS.derive_binding(te, selected, defer_classes=defer_classes,
                             **{key: value for key, value in evidence.items() if value is not None})


def interface_entry(
    entry: Mapping[str, Any],
    *,
    name: str | None = None,
    source_reference: str | None = None,
) -> dict[str, Any]:
    """A group's generator entry with the declarations every capsule carries, defaulted not invented."""
    out = dict(entry)
    if name is not None:
        out["name"] = str(name)
    if not out.get("name"):
        raise ValueError("a group interface capsule needs a name")
    out.setdefault("kind", "op")
    out.setdefault("source_role", SOURCE_ROLE)
    out.setdefault("source_reference", source_reference or "a compute group of a captured model")
    out.setdefault("label", "dev")
    return out


def interface_capsule(entry: Mapping[str, Any], binding) -> tuple[dict[str, Any], str]:
    """``(capsule, interface MLIR)`` for one stated group, under ``binding`` routed to its regime.

    Exactly what the Phase 0 writer builds for the same entry and binding
    (:func:`~merlin.targetgen.corpus_spec.entry_binding` then :func:`~merlin.targetgen.corpus_spec.build`),
    so a request made from this and a capsule written from it cannot differ.
    """
    from merlin.targetgen import corpus_spec as CS

    stated = interface_entry(entry)
    _regime, routed = CS.entry_binding(stated, binding)
    return CS.build(stated, routed)


def entries(
    target: str,
    module,
    *,
    weight_args: Collection[int] | None = None,
    model: str = "",
    with_raw: bool = True,
    numerical_variants: bool = False,
    oracle=None,
    groups: Sequence | None = None,
) -> dict[str, Any]:
    """Generator entries for accelerator groups of ``module`` on ``target``.

    Form consumers may merge groups with different positive multiplier values,
    because they share a command shape. Consumers binding an exact numerical
    golden must request ``numerical_variants`` so those groups stay distinct.
    """
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    found: dict[str, dict[str, Any]] = {}
    unstated: Counter = Counter()
    groups = groups if groups is not None else CG.form_groups(module, target, oracle=oracle)
    for group in groups:
        if group.placement == CG.HOST or group.root is None:
            continue
        try:
            stated = GC.program(group, weight_args=weight_args)
        except CG.NoCapsuleForm as error:
            unstated[str(error)] += 1
            continue
        entry = dict(stated.entry)
        key = _numerical_identity(entry) if numerical_variants else _identity(entry)
        row = found.setdefault(
            key,
            {
                "entry": entry,
                "count": 0,
                "groups": [],
                "program": stated,
                "group_batch_shapes": [],
                "slice_instances": 0,
            },
        )
        row["count"] += 1
        row["groups"].append(group.index)
        slices = prod(stated.batch_shape) if stated.batch_shape else 1
        row["slice_instances"] += slices
        row["group_batch_shapes"].append(
            {"group": group.index, "batch_shape": list(stated.batch_shape), "slices": slices}
        )
    out: list[dict[str, Any]] = []
    for row in sorted(found.values(), key=lambda r: (-r["count"], _label(r["entry"]))):
        entry = row["entry"]
        name = _label(entry)
        entry.update(
            {
                "name": name,
                "cat": "layers",
                "kind": "layer",
                "label": "dev",
                "source_role": SOURCE_ROLE,
                "source_reference": (
                    f"{row['count']} compute group(s) of {model or 'a captured model'} on {target}: "
                    f"groups {row['groups'][:8]}{'...' if len(row['groups']) > 8 else ''}"
                ),
            }
        )
        if "bound_lsb" in entry:
            # A declared bound is a claim about rounding and saturation at the edges of the type,
            # and an elementwise sum cannot overflow an accumulator the way a contraction's small
            # stimulus guards against: only the whole element range can fail it.
            entry["stimulus_range"] = list(_element_range(str(entry.get("operand_dtype") or "")))
        elif entry.get("epilogue"):
            # Every fused stage behaves differently below zero; a stimulus that cannot go there
            # cannot fail it.
            entry["stimulus_range"] = list(SIGNED_STIMULUS_RANGE)
        entry.pop("scale_granularity", None)
        # The capsule and its representative program describe one slice. Batch multiplicity belongs
        # to each group, not to the first group selected by deduplication.
        slice_program = row["program"].to_dict()
        slice_program.pop("batch_shape", None)
        out.append(
            {
                "name": name,
                "count": row["count"],
                "groups": row["groups"],
                "slice_instances": row["slice_instances"],
                "group_batch_shapes": row["group_batch_shapes"],
                "entry": entry,
                "program": slice_program,
            }
        )
        if with_raw and entry.get("epilogue"):
            raw = {
                k: v
                for k, v in entry.items()
                if k not in ("epilogue", "acc_scale", "stimulus_range") and not k.startswith("pool_")
            }
            raw["epilogue"] = []
            raw["name"] = _label(raw)
            raw["source_reference"] = f"the bare-accumulator sibling of {name}"
            if not any(other["name"] == raw["name"] for other in out):
                out.append(
                    {
                        "name": raw["name"],
                        "count": row["count"],
                        "groups": row["groups"],
                        "slice_instances": row["slice_instances"],
                        "group_batch_shapes": row["group_batch_shapes"],
                        "entry": raw,
                        "raw_of": name,
                    }
                )
    return {
        "schema": SCHEMA,
        "target": target,
        "model": model,
        "accelerator_groups": sum(1 for g in groups if g.placement != CG.HOST and g.root is not None),
        "stated": sum(row["count"] for row in found.values()),
        "distinct": len(found),
        "unstated": dict(unstated),
        "entries": out,
    }


#: The sum bodies a trailing-window reduction may carry. Structural (the op spellings of MLIR's own
#: ``arith`` dialect), never a target fact.
_SUM_BODIES = (["arith.addf"], ["arith.addi"])
_DIVIDE_BODIES = (["arith.divf"], ["arith.divsi"], ["arith.divui"])


def _through_views(value, members: Collection[Any]):
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    owner = getattr(value, "owner", None)
    while owner is not None and mq.op_name(owner) in CG._VIEW_OPS and getattr(owner, "operands", None):  # noqa: SLF001
        value = owner.operands[0]
        owner = getattr(value, "owner", None)
    return owner if any(owner is member for member in members) else None


def host_reduction_forms(groups) -> list[dict[str, Any]]:
    """Trailing-window SUM reductions that host regions compute, stated in the device's window form.

    A region the target does not take (a mean in floating point after a dequantize, a layer-norm or
    softmax denominator) still has a device form when its core is a sum over the trailing dims of a
    tensor: each kept position is a row, the window is what a contraction reduces over, and the stored
    operand is a constant one -- the same form ``compute_groups`` states an admitted window mean in.
    A divide by a compile-time count after the sum makes it a MEAN (the readout carries the reciprocal).
    Read from the IR only; a reduction over non-trailing dims or with dynamic extents is skipped, with
    no form invented for it.
    """
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_numerics as GN

    out: list[dict[str, Any]] = []
    for group in groups:
        if group.placement != CG.HOST:
            continue
        members = list(group.members)
        for reduce in members:
            if mq.op_name(reduce) != "linalg.reduce":
                continue
            body = [name for name in CG._body_ops(reduce) if name not in CG._BODY_NOISE]  # noqa: SLF001
            if body not in _SUM_BODIES:
                continue
            in_shape, in_dtype = mq.type_shape_dtype(reduce.operands[0].type)
            reduced = CG._int_array(reduce, "dimensions")  # noqa: SLF001
            if not in_shape or reduced is None or any(not isinstance(e, int) or e <= 0 for e in in_shape):
                continue
            kept = len(in_shape) - len(reduced)
            if kept < 1 or sorted(reduced) != list(range(kept, len(in_shape))):
                continue
            rows, window = CG._product(in_shape[:kept]), CG._product(in_shape[kept:])  # noqa: SLF001
            count = None
            for member in members:
                names = [name for name in CG._body_ops(member) if name not in CG._BODY_NOISE]  # noqa: SLF001
                if names in _DIVIDE_BODIES and member.operands and _through_views(member.operands[0], [reduce]):
                    divisor = GN._constant_of(member.operands[1]) if len(member.operands) > 1 else None  # noqa: SLF001
                    if isinstance(divisor, (int, float)) and divisor:
                        count = float(divisor)
                        break
            out.append(
                {
                    "group": group.index,
                    "kind": "window_mean" if count is not None else "row_sum",
                    "rows": int(rows),
                    "window": int(window),
                    "input_rank": len(in_shape),
                    "reduced_dims": len(reduced),
                    "divisor": count,
                    "source_dtype": str(in_dtype or ""),
                    "region_dtype": group.in_dtype,
                }
            )
    return out


def reduction_entry(form: Mapping[str, Any], *, operand_dtype: str, name: str | None = None) -> dict[str, Any]:
    """A host reduction's device-form generator entry: ``[rows, window] @ ones[window, 1]``.

    The same statement ``compute_groups.capsule_entry`` makes for an admitted window mean; a mean
    carries the reciprocal of its count on the readout scale, a plain row sum commits the bare sum.
    """
    entry: dict[str, Any] = {
        "name": name or f"group_{form['group']}",
        "kind": "op",
        "op": "matmul",
        "M": int(form["rows"]),
        "K": int(form["window"]),
        "N": 1,
        "epilogue": [],
        "operand_dtype": operand_dtype,
    }
    if form.get("divisor"):
        entry.update(
            {"epilogue": ["acc_scale"], "acc_scale": 1.0 / float(form["divisor"]), "scale_granularity": "tensor"}
        )
    return entry


def activation_source(group, stored_operand: int | None) -> str:
    """Whether a contraction's STREAMED operand is the program's input or something it computed.

    ``model_input`` when the activation reaches a function argument through nothing but quantization,
    padding, movement and views -- the first layer on a raw input (a stem), whose operand the host
    has to put on the grid and gather -- else ``intermediate``. Read from the IR; no layer is named.
    """
    from xdsl.ir import BlockArgument

    from merlin.xdsl_dialects.lowering import compute_groups as CG

    root = getattr(group, "root", None)
    if root is None or not getattr(root, "operands", None):
        return "intermediate"
    candidates = [i for i in range(min(2, len(root.operands))) if i != stored_operand]
    if not candidates:
        return "intermediate"
    passing = (CG.VIEW, CG.MOVEMENT, CG.PAD, CG.QUANTIZE, CG.DEQUANTIZE)
    value = root.operands[candidates[0]]
    for _ in range(64):
        if isinstance(value, BlockArgument):
            return "model_input"
        owner = getattr(value, "owner", None)
        if owner is None or not getattr(owner, "operands", None):
            return "intermediate"
        stage = CG.classify(owner)
        if stage is None or stage.kind not in passing:
            return "intermediate"
        value = owner.operands[0]
    return "intermediate"
