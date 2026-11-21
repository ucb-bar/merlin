"""What a unit's readout can absorb, derived from the target and audited against what it declares.

A contraction leaves a wide accumulator. Whether the scale, the rounding, the clamp and the
activation that turn it into a stored element happen ON the unit or on the host is decided by one
question nothing in the compiler used to ask: at what granularity can this hardware hold a scale?

The cost of not asking is a whole model's performance. A quantizer that emits one scale per output
channel, lowered onto a store path that holds one scale per command, leaves the compiler two
choices: read the raw accumulator out and finish every element on the host, or compute something
other than what the model says. The first is what happened; it is correct, it passes every gate, and
it is two orders of magnitude slower than the hardware.

So the granularity a readout accepts is a FACT, derived here and compared with the contract's
``scaling:`` declaration, which until now nothing checked.

**Evidence ladder.** Each field records the rung that produced it and the thing observed.

``format``
    A unit whose operand format carries its own scale structure (a block-scaled format) fixes the
    granularity by definition: the block length and the scale dtype are the format's.
``rtl_datapath``
    Extracted element and accumulator dtypes, and whether the accumulator is an addressable memory
    or exists only inside the datapath.
``register_layout``
    The target's own command-register layouts. A field whose name carries the ``scale`` role and
    that is a register field of bounded width holds ONE value per command, so every store it
    configures has one scale. Roles are read from the tokens of the target's own field names
    against a closed, target-neutral vocabulary; nothing here names a target, a register or a
    command.
``isa_role``
    On a self-hosted ISA, the structural role census. A scaled-readout class proves a scale is an
    instruction OPERAND. It does not say how much of the accumulator one such instruction drains,
    so this rung records the carrier and leaves the granularity unknown with that as the reason.
``scalar_abi``
    A scalar readout contract the target's backend derived from its generated parameter header
    (accumulator, element and scale types, clamp bounds, round-half-to-even). Supplied by the
    backend through an optional ``readout_scalar_abi`` hook, because the header is the target's.
``backend_declared``
    Which stages each readout selector applies (``readout_epilogue_capability``), weakest because
    it is declared, and the only source for it.

**Three states.** A field no rung could decide is ``None`` with a reason under ``unknown``. It is
never defaulted: a granularity assumed to be per-tensor on a target that can do better loses
accuracy silently, and one assumed to be per-channel on a target that cannot is the defect above.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from merlin.common import target_data as qf
from merlin.target.capabilities.readout_facet import ReadoutFacet

# ---- readout_facet -----------------------------------------------------------------------------


def _contracts(unit: Mapping[str, Any] | None) -> bool:
    """Whether ``unit`` (``None``: the target's one unit-less facet) is one a contraction leaves."""
    if unit is None:
        return True
    families = {str((cap or {}).get("family")) for cap in unit.get("semantic_capabilities") or ()}
    return "contraction" in families or "matmul" in (unit.get("ops") or ())


def _from_output_epilogue(
    facet: ReadoutFacet, record: Mapping[str, Any] | None, unit: Mapping[str, Any] | None
) -> None:
    """Fold the contract's derived output-epilogue record into a contraction unit's facet."""
    from merlin.target.capabilities.readout_facet import ReadoutFacet

    if not record or not _contracts(unit):
        return
    from merlin.target.rtl.firrtl import output_epilogue as OE

    checked = OE.validate(record)
    facet.derived_stages = tuple(stage for stage, applies in sorted(checked["stages"].items()) if applies)
    facet.accumulates = checked["accumulate"]["status"] == OE.DERIVED
    if facet.derived_stages:
        facet.note(
            "derived_stages",
            "rtl_derived",
            f"the derived output epilogue applies {list(facet.derived_stages)} on move-out",
        )
    scale = checked["scale"]
    if scale["status"] == OE.DERIVED and facet.scale_dtype is None:
        facet.scale_dtype = str(scale["dtype"])
        facet.note("scale_dtype", "rtl_derived", f"the readout decodes its scale as {scale['format']['format']}")


_NO_RUNG = "no rung produced it"


_SCALED_READOUT_ROLE = "acc_readout_scaled"


_PLAIN_READOUT_ROLE = "acc_readout"


def derive(
    target: str,
    *,
    facts: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
    scalar_abi: Mapping[str, Any] | None = None,
    readouts: Sequence[Mapping[str, Any]] | None = None,
    stage_routes: Sequence[Mapping[str, Any]] | None = None,
    taxonomy: Mapping[str, Any] | None = None,
    operand_sum: Mapping[str, Any] | None = None,
    output_epilogue: Mapping[str, Any] | None = None,
) -> ReadoutFacet:
    """Derive one unit's readout facet from supplied evidence. Pure: it loads nothing."""
    from merlin.target.capabilities.readout_facet import (
        ReadoutFacet,
        _from_address_space,
        _from_datapath,
        _from_formats,
        _from_isa_roles,
        _from_operand_sum,
        _from_registers,
        _from_scalar_abi,
        _register_bundles,
    )

    facet = ReadoutFacet(target=target, unit=str(unit.get("name")) if unit else None)
    for name in (
        "element_dtype",
        "accumulator_dtype",
        "accumulator_kind",
        "scale_granularities",
        "scale_dtype",
        "rounding",
        "clamp",
        "zero_point_carried",
    ):
        facet.unknown[name] = _NO_RUNG
    body = facts.get("facts", facts) if isinstance(facts, Mapping) else {}
    _from_formats(facet, tuple((unit or {}).get("dtypes") or ()))
    _from_datapath(facet, body)
    if facts:
        _from_address_space(facet, target, facts)
    _from_registers(facet, body)
    _from_scalar_abi(facet, scalar_abi)
    _from_operand_sum(facet, operand_sum)
    _from_output_epilogue(facet, output_epilogue, unit)
    # An unknown says which rungs were silent and why, because "not derived" tells the next person
    # nothing about whether to extract more or to stop looking.
    silent = []
    if not body.get("datapaths"):
        silent.append("the target's facts carry no extracted datapath")
    if _register_bundles(body) is None:
        silent.append("its facts carry no command-register layouts")
    if not scalar_abi:
        silent.append("its backend supplies no scalar readout contract")
    _from_isa_roles(facet, taxonomy)
    for name, why in list(facet.unknown.items()):
        if why == _NO_RUNG:
            facet.unknown[name] = "; ".join(silent) or "every rung ran and none decides this field"
    if readouts:
        facet.readouts = tuple(dict(readout) for readout in readouts)
        facet.note(
            "readouts",
            "backend_declared",
            f"{len(facet.readouts)} readout selector(s) declared by the target's backend",
        )
    if stage_routes:
        facet.stage_routes = tuple(dict(route) for route in stage_routes)
        facet.note("stage_routes", "backend_declared", f"{len(facet.stage_routes)} scoped stage route(s) declared")
    return facet


class ReadoutSupportError(RuntimeError):
    """The selected support declares readout semantics but cannot serve them.

    Raised instead of returning ``None``: an absent declaration reads as UNKNOWN downstream, and a
    selected provider that promised readout facts and then failed to load, lacks the hook, or raised
    from it must not degrade into "this target declares nothing" without anyone being told.
    """


def _readout_required(target: str) -> str | None:
    """The selected provider root when its contract DECLARES ``readout_semantics``, else None.

    Only a selected support provider can owe an answer; a target with no selected support (or one
    whose contract declares no readout block) keeps the historical "no hook" meaning.
    """
    try:
        from merlin.target.registry.plugins import resolve_support

        selected = resolve_support(target)
    except Exception:  # noqa: BLE001 -- nothing selected: nothing is owed
        return None
    try:
        contract = selected.load_contract()
    except Exception as exc:  # noqa: BLE001 -- a selected provider whose contract will not load
        raise ReadoutSupportError(
            f"{target}: selected support {selected.base} has an unreadable contract: {type(exc).__name__}: {exc}"
        ) from exc
    return (
        str(selected.base)
        if isinstance(contract, Mapping) and isinstance(contract.get("readout_semantics"), Mapping)
        else None
    )


def _backend_hook(target: str, name: str) -> Callable[[], Any] | None:
    from merlin.runtime.backends import registry as base

    required = _readout_required(target)
    try:
        backend = base.get_backend(target)
    except Exception as exc:  # noqa: BLE001 -- a target with no runtime backend has no hook, not an error
        if required:
            raise ReadoutSupportError(
                f"{target}: selected support {required} declares readout_semantics, but its backend did not "
                f"load ({type(exc).__name__}: {exc}); refusing to treat the readout as undeclared"
            ) from exc
        return None
    hook = getattr(backend, name, None)
    if not callable(hook):
        if required:
            raise ReadoutSupportError(
                f"{target}: selected support {required} declares readout_semantics, but its backend "
                f"{getattr(backend, '__name__', backend)!r} has no {name}() hook to serve it"
            )
        return None
    if not required:
        return hook

    def strict():
        try:
            return hook()
        except Exception as exc:  # noqa: BLE001 -- surfaced, never swallowed into "absent"
            raise ReadoutSupportError(
                f"{target}: selected support {required} could not serve {name}(): {type(exc).__name__}: {exc}"
            ) from exc

    return strict


def epilogue_readouts(target: str):
    """The target's own ``readout_epilogue_capability`` declaration, as rule-module objects.

    ONE accessor, so the grade-time check and the capsule GENERATOR read the same declaration. They did
    not before: the check consulted it and the generator did not, which is how a corpus came to hold
    capsules whose declared epilogue no readout applies -- found, one at a time, by the check that runs
    hours later instead of by the writer that had the answer in hand.

    ``None`` means the target declares nothing, which is UNKNOWN and never "applies everything".
    """
    hook = _backend_hook(target, "readout_epilogue_capability")
    if hook is None:
        return None
    try:
        raw = hook()
    except ReadoutSupportError:
        raise
    except Exception:  # noqa: BLE001 -- an unreadable declaration is an absent one, not a crash
        return None
    if not raw:
        return None
    return validated_readouts(raw)


def validated_readouts(raw):
    """Parse supplied readout data without discovering an execution backend."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    from merlin.verification.references.epilogue_applicability import ReadoutCapability

    return [
        ReadoutCapability(
            selector=str(r.get("selector")),
            applies=frozenset(str(s) for s in (r.get("applies") or ())),
            evidence=str(r.get("evidence") or ""),
        )
        for r in raw
        if isinstance(r, Mapping) and r.get("selector")
    ]


def epilogue_stage_routes(target: str):
    """Parse the selected backend's non-readout stage routes; malformed rows license nothing."""
    hook = _backend_hook(target, "epilogue_stage_routes")
    if hook is None:
        return ()
    try:
        raw = hook()
    except ReadoutSupportError:
        raise
    except Exception:  # noqa: BLE001 -- a failed declaration grants nothing
        return ()
    return validated_stage_routes(raw)


def validated_stage_routes(raw):
    """Parse supplied route data with the same rules as the grade-time accessor."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    from merlin.runtime.engine.command_buffer import EPILOGUE_STAGE_SET
    from merlin.verification.references.epilogue_applicability import StageRoute

    routes = []
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        stages = row.get("stages")
        vector_keys = ("readouts", "producer_opcodes", "consumer_opcodes")
        scalar_keys = ("site", "composed_with", "operand_attribute", "operand_role", "evidence")
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not stages:
            continue
        if any(not isinstance(row.get(key), str) or not row[key] for key in scalar_keys):
            continue
        if any(
            not isinstance(row.get(key), Sequence)
            or isinstance(row[key], (str, bytes))
            or not row[key]
            or any(not isinstance(x, str) or not x for x in row[key])
            for key in vector_keys
        ):
            continue
        for stage in stages:
            if not isinstance(stage, str) or stage not in EPILOGUE_STAGE_SET:
                continue
            routes.append(
                StageRoute(
                    stage=str(stage),
                    site=str(row["site"]),
                    composed_with=str(row["composed_with"]),
                    readouts=frozenset(str(x) for x in row["readouts"]),
                    producer_opcodes=frozenset(str(x) for x in row["producer_opcodes"]),
                    consumer_opcodes=frozenset(str(x) for x in row["consumer_opcodes"]),
                    operand_attribute=str(row["operand_attribute"]),
                    operand_role=str(row["operand_role"]),
                    evidence=str(row["evidence"]),
                )
            )
    return tuple(routes)


def capture_inputs(target: str, *, facts: Mapping[str, Any], include_taxonomy: bool = True) -> dict[str, Any]:
    """Observe optional readout declarations once, without loading RTL facts."""
    from merlin.target.capabilities.readout_facet import _register_bundles

    inputs: dict[str, Any] = {}
    for field_name, hook_name in (
        ("scalar_abi", "readout_scalar_abi"),
        ("readouts", "readout_epilogue_capability"),
        ("operand_sum", "readout_operand_sum"),
    ):
        hook = _backend_hook(target, hook_name)
        try:
            inputs[field_name] = hook() if hook is not None else None
        except ReadoutSupportError:
            raise
        except Exception as exc:  # noqa: BLE001 -- an unreadable declaration is unknown
            inputs[field_name] = None
            inputs.setdefault("unknown", {})[field_name] = f"{type(exc).__name__}: {exc}"
    # The fact view and the grade must use the same validated routes. A raw malformed backend row
    # must not become a software-spec fact merely because the grade's parser would later reject it.
    inputs["stage_routes"] = [route.to_dict() for route in epilogue_stage_routes(target)]
    inputs["taxonomy"] = None
    if include_taxonomy and _register_bundles(facts.get("facts", facts)) is None:
        try:
            from merlin.target.isa import taxonomy as isa_taxonomy

            inputs["taxonomy"] = isa_taxonomy.taxonomy_for_target(target)
        except Exception as exc:  # noqa: BLE001 -- no ISA evidence is an unknown rung
            inputs.setdefault("unknown", {})["taxonomy"] = f"{type(exc).__name__}: {exc}"
    return inputs


def for_target(
    target: str,
    *,
    contract: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
    readout_inputs: Mapping[str, Any] | None = None,
) -> list[ReadoutFacet]:
    """One facet per compute unit the target declares (one unit-less facet if it declares none)."""
    from merlin.target.capabilities.readout_facet import ReadoutFacet, with_current_register_layouts

    if contract is None:
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        contract = load_capability_manifest(target).contract
    if facts is None:
        from merlin.target.rtl import facts as rtl_facts

        try:
            facts = with_current_register_layouts(rtl_facts.load_facts(target))
        except Exception:  # noqa: BLE001 -- underivable facts leave every RTL rung silent
            facts = {}
    if readout_inputs is None and (contract.get("runner") or {}).get("backend") == "chipyard_rocc":
        from merlin.runtime.backends.rocc.chipyard_backend import readout_metadata

        readout_inputs = readout_metadata(contract)
    inputs = capture_inputs(target, facts=facts) if readout_inputs is None else readout_inputs
    units = [u for u in contract.get("compute_units") or () if isinstance(u, Mapping)] or [None]
    return [
        derive(
            target,
            facts=facts,
            unit=u,
            scalar_abi=inputs.get("scalar_abi"),
            readouts=inputs.get("readouts"),
            stage_routes=inputs.get("stage_routes"),
            taxonomy=inputs.get("taxonomy"),
            operand_sum=inputs.get("operand_sum"),
            output_epilogue=contract.get("output_epilogue"),
        )
        for u in units
    ]


@dataclass(frozen=True)
class TargetReadout:
    """Every unit's facet, asked as one: can ANY readout of this target hold the granularity?

    ``True`` if one can, ``False`` only if every facet was derived and none can, else ``None``.
    An underived facet beside a derived refusal is unknown, not a refusal: the unit nobody could
    read may be the one that holds it.
    """

    facets: tuple[ReadoutFacet, ...]

    def applies_composition(
        self,
        stages: Sequence[str],
        *,
        unit: str | None = None,
        selector: str | None = None,
        routes=(),
    ) -> bool | None:
        """Whether one selected unit/readout carries the complete stage set.

        This establishes physical composition only. Stage ordering and numeric
        equivalence remain separate checks. Stages offered by different units
        or different readouts cannot be combined into an invented pipeline.
        """
        from merlin.verification.references.epilogue_applicability import selectors_applying

        answers = []
        for facet in self.facets:
            if unit is not None and facet.unit != unit:
                continue
            readouts = facet.capabilities()
            if not readouts:
                answers.append(None)
                continue
            if selector is not None:
                readouts = tuple(row for row in readouts if row.selector == selector)
                if not readouts:
                    answers.append(False)
                    continue
            answers.append(bool(selectors_applying(readouts, stages, routes=routes, composition="contraction")))
        if any(answer is True for answer in answers):
            return True
        return False if answers and all(answer is False for answer in answers) else None

    @property
    def scale_granularities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(g for f in self.facets for g in f.scale_granularities or ()))

    @property
    def unknown(self) -> dict[str, str]:
        return {key: why for f in self.facets for key, why in f.unknown.items()}

    @property
    def accumulates(self) -> bool:
        """Whether some unit's derived output path adds an operand into the accumulator on load."""
        return any(f.accumulates for f in self.facets)

    def operand_sum(self, multipliers: Sequence[float]) -> tuple[ReadoutFacet | None, str | None]:
        """``(facet, None)`` for the first unit that can sum operands under ``multipliers``, else
        ``(None, why)`` with the first refusal."""
        refusals = []
        for facet in self.facets:
            refusal = facet.operand_sum_refusal(multipliers)
            if refusal is None:
                return facet, None
            refusals.append(refusal)
        return None, (refusals[0] if refusals else "the target has no unit")

    def admits_granularity(self, granularity: str) -> bool | None:
        answers = [f.admits_granularity(granularity) for f in self.facets]
        if any(answer is True for answer in answers):
            return True
        return False if answers and all(answer is False for answer in answers) else None

    def applies_stage(self, stage: str) -> bool | None:
        """Can ANY readout of this target apply ``stage``? Same tri-state as
        :meth:`admits_granularity`: ``True`` if one can, ``False`` only when every unit declared its
        readouts and none applies it, ``None`` when no unit declared any -- the unit nobody described
        may be the one that applies it."""
        answers = [f.applies_stage(stage) for f in self.facets]
        if any(answer is True for answer in answers):
            return True
        return False if answers and all(answer is False for answer in answers) else None


def reconcile(unit: Mapping[str, Any], facet: ReadoutFacet) -> list[dict[str, Any]]:
    """Compare a unit's declared ``scaling`` with what its readout was derived to hold.

    ``scaling_exceeds_readout`` is the defect this module exists for: the contract promises a
    granularity the hardware cannot hold, so everything generated from the contract (capsules,
    goldens, quantization) asks for arithmetic that can only happen on the host.
    """
    from merlin.target.capabilities.readout_facet import GRANULARITY_OF_SCALE_KIND, ReadoutFacet

    declared = unit.get("scaling")
    if not declared or str(declared) == "none":
        return []
    wanted = GRANULARITY_OF_SCALE_KIND.get(str(declared))
    base = {
        "unit": unit.get("name"),
        "declared": declared,
        "declared_granularity": wanted,
        "derived": (list(facet.scale_granularities) if facet.scale_granularities is not None else None),
    }
    if wanted is None:
        return [{**base, "kind": "unmapped_scaling", "why": f"scaling {declared!r} has no facet granularity"}]
    admitted = facet.admits_granularity(wanted)
    if admitted is None:
        return [
            {
                **base,
                "kind": "unaudited_scaling",
                "why": facet.unknown.get("scale_granularities", "granularity not derived"),
            }
        ]
    if admitted:
        return []
    return [
        {
            **base,
            "kind": "scaling_exceeds_readout",
            "rungs": list(facet.rungs_for("scale_granularities")),
            "why": "; ".join(e.observed for e in facet.evidence if e.field == "scale_granularities"),
        }
    ]


def epilogue_capability(facet: ReadoutFacet, *, name: str | None = None):
    """The quantized-epilogue capability this facet licenses, or a ``ValueError`` naming the gap.

    Every limit is a derived one. A facet with an unknown in any field the capability needs
    produces no capability: a planner handed invented limits would plan against them.
    """
    from merlin.compiler.quantization.contract import QuantizedEpilogueCapability
    from merlin.runtime.engine.command_buffer import BIAS_STAGES
    from merlin.target.capabilities.readout_facet import EPILOGUE_GRANULARITY, ReadoutFacet

    needed = {
        "accumulator_dtype": facet.accumulator_dtype,
        "element_dtype": facet.element_dtype,
        "scale_granularities": facet.scale_granularities,
        "scale_dtype": facet.scale_dtype,
        "rounding": facet.rounding,
        "clamp": facet.clamp,
        "zero_point_carried": facet.zero_point_carried,
    }
    missing = sorted(key for key, value in needed.items() if value is None)
    if missing:
        raise ValueError(
            f"readout facet of {facet.target!r} cannot license an epilogue capability: "
            + ", ".join(f"{key} ({facet.unknown.get(key, 'not derived')})" for key in missing)
        )
    granularities = tuple(EPILOGUE_GRANULARITY[g] for g in facet.scale_granularities if g in EPILOGUE_GRANULARITY)
    if not granularities:
        raise ValueError(
            f"readout facet of {facet.target!r} holds scales at "
            f"{list(facet.scale_granularities)}, which the epilogue contract cannot "
            f"express"
        )
    if facet.scale_dtype != "f32" or facet.rounding != "half_even":
        raise ValueError(
            f"the epilogue contract evaluates an f32 scale rounded half to even; "
            f"{facet.target!r} derives scale {facet.scale_dtype} / {facet.rounding}"
        )
    if facet.zero_point_carried:
        raise ValueError(
            f"{facet.target!r} carries an output zero point in a register; its domain "
            f"is not derived, so no capability is licensed"
        )
    applied = {stage for readout in facet.readouts for stage in readout.get("applies") or ()}
    has_bias, has_relu = bool(applied & set(BIAS_STAGES)), "relu" in applied
    core = ("scale_f32", "round_to_nearest_even", "clamp")
    templates = tuple(
        (("bias_i32",) if bias else ()) + core + (("relu",) if relu else ())
        for bias in ((False, True) if has_bias else (False,))
        for relu in ((False, True) if has_relu else (False,))
    )
    return QuantizedEpilogueCapability(
        name=name or f"{facet.target}:{facet.unit or 'readout'}",
        accumulator_dtypes=(facet.accumulator_dtype,),
        output_dtypes=(facet.element_dtype,),
        scale_granularities=granularities,
        bias_domains=("none", "accumulator") if has_bias else ("none",),
        output_zero_points=(0,),
        roundings=("round_to_nearest_even",),
        saturations=(facet.clamp,),
        ordered_stage_templates=templates,
        activations=("none", "relu") if has_relu else ("none",),
    )


# ---- output_scale_partition --------------------------------------------------------------------
# Bind a logical column-scale obligation without assigning a physical schedule.
#
# Independent contraction output columns may use different scalar readout gains.
# The hardware vector width still describes the elements sharing one physical
# gain. This record preserves the source gain vector and that width; target
# packing, commands, addresses, legality and execution remain compiler obligations.

SCHEMA = "logical_output_column_partition_v1"


def interface_components(source):
    """Scalar-scale logical components of an explicit independent-column contraction.

    This proves only source arithmetic and exhaustive disjoint column coverage.
    Every component still needs the ordinary SW/HW declaration screen; packing,
    addresses, physical commands and execution remain candidate obligations.
    Unsupported source forms return None, never a permissive scalar default.
    """
    from merlin.runtime.engine.command_buffer import declared_acc_scale_by_column

    if not isinstance(source, dict) or source.get("operation") not in {"matmul", "conv2d"}:
        return None
    lhs, rhs, attrs = (source.get(key) for key in ("lhs", "rhs", "attributes"))
    if not isinstance(attrs, dict) or not all(isinstance(item, dict) for item in (lhs, rhs)):
        return None
    shapes = [item.get("shape") for item in (lhs, rhs)]
    if any(
        not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape) for shape in shapes
    ):
        return None
    if lhs.get("dtype") not in {"i8", "int8"} or rhs.get("dtype") != lhs.get("dtype"):
        return None
    a, w = shapes
    if len(w) != 2 or source.get("disjoint") is not True:
        return None
    if source["operation"] == "matmul":
        if len(a) != 2 or a[1] != w[0]:
            return None
        rows = a[0]
    else:
        kernel = attrs.get("kernel")
        if (
            len(a) != 4
            or not isinstance(kernel, list)
            or len(kernel) != 4
            or any(type(n) is not int or n < 1 for n in kernel)
            or kernel[2] != a[-1]
            or math.prod(kernel[:3]) != w[0]
            or kernel[3] != w[1]
        ):
            return None
        from merlin.runtime.engine.command_buffer import conv_out_dims

        geometry = [attrs.get(key) for key in ("stride", "padding", "dilation")]
        if any(
            not isinstance(value, list) or len(value) != arity or any(type(n) is not int or n < minimum for n in value)
            for value, arity, minimum in zip(geometry, (2, 4, 2), (1, 0, 1), strict=True)
        ):
            return None
        ho, wo = conv_out_dims(a[1], a[2], *kernel[:2], *geometry)
        if ho < 1 or wo < 1 or str(attrs.get("layout", "nhwc")).lower() != "nhwc":
            return None
        rows = a[0] * ho * wo
    stages = attrs.get("epilogue")
    if (
        not isinstance(stages, list)
        or "acc_scale" not in stages
        or any(stage not in {"bias", "bias_add", "acc_scale", "relu"} for stage in stages)
        or attrs.get("output_dtype") not in {"i8", "int8"}
    ):
        return None
    geometry = {"kernel", "stride", "padding", "dilation"}
    if set(attrs) - geometry - {"epilogue", "acc_scale_by_column", "output_dtype", "bias", "layout", "name"}:
        return None
    result = source.get("result")
    if result is not None and (result.get("shape") != [rows, w[1]] or result.get("dtype") != attrs["output_dtype"]):
        return None
    if any(stage in {"bias", "bias_add"} for stage in stages):
        bias = source.get("bias") or {}
        if bias.get("shape") != [w[1]] or bias.get("dtype") not in {"i32", "int32"}:
            return None
    try:
        gains = declared_acc_scale_by_column(attrs, w[1], op="logical output-column partition")
    except (ValueError, TypeError, OverflowError):
        return None
    # Unknown vector-valued attributes could couple columns. Geometry is shared
    # unchanged; all other arrays/mappings keep the partition unavailable.
    if any(
        isinstance(value, (list, dict))
        for key, value in attrs.items()
        if key not in geometry | {"epilogue", "acc_scale_by_column"}
    ):
        return None
    parts = []
    start = 0
    while start < len(gains):
        end = start + 1
        while end < len(gains) and gains[end] == gains[start]:
            end += 1
        parts.append(
            {"start": start, "stop": end, "gain": gains[start], "dimensions": {"M": rows, "K": w[0], "N": end - start}}
        )
        start = end
    return {
        "scope": "source_arithmetic_only",
        "columns": w[1],
        "parts": parts,
        "preserved": ["complete_reduction", "bias_lane", "stage_order", "rounding", "saturation"],
        "physical_implementation": "unverified",
    }


def obligation(gains, physical_columns):
    """A source-arithmetic obligation for gains varying within a physical block.

    Return None for an already block-uniform vector. Callers establish that
    these are independent source output columns; this function grants no such
    independence, hardware admission, implementation or timing authority.
    """
    if type(physical_columns) is not int or physical_columns < 1:
        raise ValueError("the physical scale-vector width must be a positive integer")
    if len(gains) == 0:
        raise ValueError("a column-scale vector must be nonempty")
    digest = hashlib.sha256()
    values = []
    for value in gains:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("column gains must be positive finite numeric values")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("column gains must be positive finite numeric values")
        # The caller retains the selected numeric format in the complete proof.
        # Bind its exact already-typed values without imposing another format.
        digest.update(value.hex().encode("ascii") + b"\n")
        values.append(value)
    if all(
        len(set(values[start : start + physical_columns])) == 1 for start in range(0, len(values), physical_columns)
    ):
        return None
    return {
        "schema": SCHEMA,
        "scope": "source_arithmetic_only",
        "columns": len(values),
        "physical_columns_per_scale": physical_columns,
        "gains_sha256": digest.hexdigest(),
        "requires": "independent_output_column_implementation",
    }
