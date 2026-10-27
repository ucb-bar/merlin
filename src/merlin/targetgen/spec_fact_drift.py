"""Compare an authored software spec with the capability the target's own facts establish.

Phase 0 admits only what the hardware facts AND the authored spec both allow, so an authored field
narrower than the facts silently removes hardware capability from every later phase. The existing
audits look in one direction only: a claimed capability must have evidence. Nothing flagged a
capability the facts establish and the authored spec forbids. Measured on gemmini: the spec
hand-authored ``elementwise_map: placement fused_accelerator, composed_with [contraction]`` and
``quantization.eligible_operations: [contraction]`` while the facts establish a standalone operand
sum on the load path, so the standalone residual add was refused although nothing about the
hardware refused it.

This module compares the two field by field, in both directions, per operation family:

``ok``
    The authored field equals what the facts establish.
``restricted``
    The authored field is narrower, and a top-level ``restrictions`` entry names the family, the
    field, the values it declines and WHY. A software choice, reported but not blocking.
``forbids_established``
    The authored field is narrower with no recorded reason. BLOCKING: the authored spec forbids
    what the hardware facts establish.
``exceeds_facts``
    The authored field claims values the facts decide against. BLOCKING, as admission already is.
``unconfirmed``
    The authored field claims values a fact source that cannot decide that axis did not report.
    A review obligation, not a refusal (the capability ladder decides families, not every axis).
``undetermined``
    No fact source could decide the field at all.

Nothing here names a target, an opcode or an op spelling: the family vocabulary, the dtype
registry, the readout stage vocabulary and the quantization families are shared data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA = "merlin.phase0.spec_fact_drift.v1"
OK = "ok"
RESTRICTED = "restricted"
FORBIDS = "forbids_established"
EXCEEDS = "exceeds_facts"
UNCONFIRMED = "unconfirmed"
UNDETERMINED = "undetermined"
BLOCKING = frozenset({FORBIDS, EXCEEDS})

#: Placement as an ordered capability: a standalone accelerator form subsumes a fused one, which
#: subsumes none. ``host`` is the absence of an accelerator form, never a hardware fact against one.
_PLACEMENT_RANK = {"host": 0, "unknown": 0, "fused_accelerator": 1, "accelerator": 2}
_PLACEMENT_NAME = {0: "host", 1: "fused_accelerator", 2: "accelerator"}
#: Fields a restriction may name. ``quantization`` is per quantizable family; its values are format ids.
RESTRICTABLE_FIELDS = frozenset(
    {"placement", "composed_with", "dtypes", "epilogues", "scale_granularity", "ranks", "quantization"}
)


@dataclass(frozen=True)
class FactField:
    """What the facts establish for one field, whether that is the whole answer, and why."""

    values: tuple
    complete: bool
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "complete": self.complete, "evidence": list(self.evidence)}


def _dtype(value: Any) -> str:
    """The format registry's name, so ``i8`` and ``int8`` are one dtype; an unknown spelling stays as is."""
    from merlin.common import quant_formats

    try:
        return quant_formats.get(str(value)).name
    except (KeyError, ValueError, TypeError):
        return str(value)


def _stage(value: Any) -> str:
    """The command-buffer ABI's one spelling of a bias stage, shared with SW admission."""
    from merlin.runtime.commandbuffer import BIAS_STAGES

    return "bias_add" if value in BIAS_STAGES else str(value)


def _sorted(values: Iterable) -> tuple:
    return tuple(sorted(set(values), key=str))


# --- the fact side --------------------------------------------------------------------------------


def fact_capabilities(
    *,
    target: str,
    contract: Mapping[str, Any],
    raw_facts: Mapping[str, Any] | None,
    readout_facets: Iterable[Mapping[str, Any]] = (),
    quantization_candidates: Iterable[Mapping[str, Any]] = (),
    taxonomy: Mapping[str, Any] | None = None,
    prohibited_roles: Iterable[str] = (),
    data_only: bool = False,
    semantic_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-family capability from the derived ladder, the evidence-checked contract and the facets.

    ``placement`` reads the same standalone verdict lowering uses (:mod:`capability_roles`, under
    the experiment's prohibited roles) together with the evidence ladder
    (:func:`capability_derive.derive`); a contract capability counts as standalone only through an
    evidence path, never as a bare declaration.
    """
    from merlin.targetgen import capability_derive as CD
    from merlin.targetgen import capability_roles as CR
    from merlin.targetgen import compute_units as CU

    derived = CD.derive(
        target,
        dict(contract),
        dict(raw_facts or {}),
        taxonomy=dict(taxonomy or {}) or None,
        **({"data_only": data_only} if data_only is not False else {}),
    )
    units = list(CU.compute_units(dict(contract)))
    declared: dict[str, list] = {}
    for unit in units:
        for capability in unit.semantic_capabilities:
            declared.setdefault(capability.family, []).append(capability)
    from merlin.targetgen import isa_taxonomy as IT
    from merlin.targetgen.semantic_families import ISA_ROLE_FAMILY, from_op

    roles = IT._classes_by_role(dict(taxonomy or {})) if taxonomy else {}
    facets = [dict(facet) for facet in readout_facets or () if isinstance(facet, Mapping)]
    body = (raw_facts or {}).get("facts") or {}
    families: dict[str, dict[str, FactField]] = {}
    # A stage a readout applies is fused work of that stage's family, observed on the readout path.
    staged: dict[str, list[str]] = {}
    for facet in facets:
        for readout in facet.get("readouts") or ():
            for stage in (readout or {}).get("applies") or ():
                family = from_op(str(stage))
                if family is not None:
                    staged.setdefault(family, []).append(
                        f"{facet.get('unit')} readout {readout.get('selector')!r} applies {stage!r}"
                    )

    for family in sorted({*derived.supported, *declared, *derived.unknown}):
        evidence = derived.supported.get(family)
        capabilities = declared.get(family, [])
        verdict = (
            CR.family_admission(units, family, prohibited_roles=prohibited_roles, semantic_facts=semantic_facts)
            if capabilities
            else None
        )
        notes: list[str] = []
        composed: set[str] = set()
        mode = None
        if verdict is not None and verdict["standalone"] and verdict["basis"] == "evidence":
            mode = 2
            notes.append(f"contract evidence path: {verdict['reason']}")
        if evidence is not None and not evidence.composed_with and evidence.source != "unit_intent":
            mode = 2
            notes.append(f"{evidence.source}: {evidence.evidence}")
            # The ladder keeps its first rung's evidence string while composition intersects across
            # rungs, so name every structural role that observed this family standalone.
            notes.extend(
                f"ISA role {role!r} -> {classes[:3]} (standalone)"
                for role, classes in sorted(roles.items())
                if ISA_ROLE_FAMILY.get(role) == (family, ())
            )
        decided = mode is not None
        if mode is None:
            for capability in capabilities:
                composed.update(capability.composed_with)
            if evidence is not None:
                composed.update(evidence.composed_with)
            if composed or evidence is not None:
                mode = 1
                notes.append(
                    f"fused only, composed with {sorted(composed)}"
                    + (f" ({evidence.source}: {evidence.evidence})" if evidence is not None else "")
                )
                notes.extend(staged.get(family, ()))
                # A fused form stands on a fact when a ladder rung or a readout stage observed it; a
                # bare contract composition is a declaration the comparison cannot hold the spec to.
                decided = evidence is not None or family in staged
                if not decided:
                    notes.append("contract declaration with no evidence path or observed readout stage")
        row = {
            "placement": FactField(
                (_PLACEMENT_NAME[mode if mode is not None else 0],),
                decided,
                tuple(notes) or ((derived.unknown[family].evidence,) if family in derived.unknown else ()),
            )
        }
        if mode == 1:
            row["composed_with"] = FactField(_sorted(composed), True, tuple(notes))
        dtypes: set[str] = set()
        dtype_notes: list[str] = []
        if evidence is not None and evidence.dtypes:
            dtypes.update(_dtype(value) for value in evidence.dtypes)
            dtype_notes.append(f"{evidence.source}: {list(evidence.dtypes)}")
        for capability in capabilities:
            if capability.dtypes:
                dtypes.update(_dtype(value) for value in capability.dtypes)
                dtype_notes.append(f"contract capability dtypes {list(capability.dtypes)}")
        # Only a rung's observed ranks are facts; a contract's declared ranks are what is audited.
        # The ladder decides families, not the rank axis (capability_derive's standing limit), so an
        # authored rank beyond these is unconfirmed rather than refused.
        if evidence is not None and evidence.ranks:
            row["ranks"] = FactField(
                _sorted(int(rank) for rank in evidence.ranks), False, (f"{evidence.source}: {evidence.evidence}",)
            )
        families[family] = row
        families[family]["_dtype_sources"] = (dtypes, dtype_notes)  # type: ignore[assignment]

    # Movement moves every element format the storage holds; the extracted datapaths say which.
    storage = [row for row in body.get("storage_datapaths") or body.get("datapaths") or () if isinstance(row, Mapping)]
    if storage and "movement" in families:
        dtypes, notes = families["movement"]["_dtype_sources"]  # type: ignore[misc]
        dtypes.update(_dtype(row["dtype"]) for row in storage if row.get("dtype"))
        notes.append("storage datapaths " + ", ".join(f"{row.get('name')}={row.get('dtype')}" for row in storage))

    # The readout's own stages: the elementwise work a unit applies on the way out of a contraction.
    if facets and "elementwise_map" in families:
        stages: set[str] = set()
        granularities: set[str] = set()
        stage_notes, granularity_notes = [], []
        stages_complete, granularity_complete = True, True
        dtypes, notes = families["elementwise_map"]["_dtype_sources"]  # type: ignore[misc]
        for facet in facets:
            readouts = [row for row in facet.get("readouts") or () if isinstance(row, Mapping)]
            stages_complete = stages_complete and bool(readouts) and not facet.get("unknown")
            for readout in readouts:
                # Only this family's own stages: a pooling stage on the same readout is fused
                # reduction work, compared under that family's placement, not an elementwise epilogue.
                applied = [
                    _stage(stage)
                    for stage in readout.get("applies") or ()
                    if from_op(str(stage)) in (None, "elementwise_map")
                ]
                stages.update(applied)
                if applied and readout.get("selector"):
                    dtypes.add(_dtype(readout["selector"]))
                    notes.append(f"readout {readout['selector']!r} applies {sorted(set(applied))}")
                stage_notes.append(
                    f"{facet.get('unit')} readout {readout.get('selector')!r}: {readout.get('evidence')}"
                )
            for route in facet.get("stage_routes") or ():
                if (
                    not isinstance(route, Mapping)
                    or route.get("composed_with") != "contraction"
                    or route.get("site") != "accumulator_seed"
                ):
                    continue
                routed = [
                    _stage(stage)
                    for stage in route.get("stages") or ()
                    if from_op(str(stage)) in (None, "elementwise_map")
                ]
                stages.update(routed)
                dtypes.update(_dtype(selector) for selector in route.get("readouts") or ())
                if routed:
                    stage_notes.append(
                        f"{facet.get('unit')} {route.get('site')} route applies {sorted(set(routed))} "
                        f"with contraction: {route.get('evidence')}"
                    )
            scale = facet.get("scale") or {}
            granularities.update(str(value) for value in scale.get("granularities") or ())
            granularity_complete = granularity_complete and scale.get("granularities_complete") is True
            granularity_notes.append(f"{facet.get('unit')} scale carriers: {scale.get('carriers')}")
        if stages:
            families["elementwise_map"]["epilogues"] = FactField(_sorted(stages), stages_complete, tuple(stage_notes))
        if granularities:
            families["elementwise_map"]["scale_granularity"] = FactField(
                _sorted(granularities), granularity_complete, tuple(granularity_notes)
            )
    for family, row in families.items():
        dtypes, notes = row.pop("_dtype_sources")  # type: ignore[misc]
        if dtypes:
            row["dtypes"] = FactField(_sorted(dtypes), True, tuple(notes))

    quantizable: dict[str, list[str]] = {}
    for candidate in quantization_candidates or ():
        if not isinstance(candidate, Mapping) or candidate.get("status") != "derived":
            continue
        recipe = candidate.get("recipe") or {}
        for family in recipe.get("families") or ():
            quantizable.setdefault(str(family), []).append(
                f"{candidate.get('unit')} {candidate.get('format')}: {(recipe.get('why') or {}).get(family, 'derived')}"
            )
    forms = _forms(
        families=families, derived=derived, declared=declared, contract=contract, facets=facets, storage=storage
    )
    # The whole-family view covers every established form, so a declaration of one form never reads
    # as wider than the family.
    for family, by_form in forms.items():
        values = {
            value for form in by_form.values() for value in (form.get("dtypes") or FactField((), True, ())).values
        }
        flat = families[family].get("dtypes")
        if values - set(flat.values if flat else ()):
            families[family]["dtypes"] = FactField(
                _sorted(values | set(flat.values if flat else ())),
                True,
                (
                    *(flat.evidence if flat else ()),
                    *(e for form in by_form.values() for e in form.get("dtypes", FactField((), True, ())).evidence),
                ),
            )
    declarations = {}
    for family, capabilities in declared.items():
        verdict = CR.family_admission(
            units, family, prohibited_roles=prohibited_roles, semantic_facts=semantic_facts, review_status="reviewed"
        )
        if verdict["standalone"] and verdict["basis"] == "declared_unchecked":
            dtypes = {_dtype(dtype) for cap in capabilities if not cap.composed_with for dtype in cap.dtypes}
            declarations[family] = {"standalone": {"placement": ["accelerator"], "dtypes": sorted(dtypes)}}
    return {
        "declared_unchecked_forms": declarations,
        "families": families,
        "forms": forms,
        "quantization": {
            family: FactField(("quantizable",), True, tuple(notes)) for family, notes in sorted(quantizable.items())
        },
    }


def _forms(*, families, derived, declared, contract, facets, storage) -> dict[str, dict[str, dict[str, FactField]]]:
    """Each family's established FORMS, field by field: ``standalone`` and ``fused``.

    The flat per-family view above is what the drift check compares a whole spec against; a single
    declaration states ONE form, and the two forms of one family differ (gemmini's standalone operand
    sum loads int8 operands, while its fused readout also writes the i32 accumulator). Only facts
    enter: a standalone form needs the flat view's decided standalone placement; a fused form needs a
    ladder rung or an observed readout stage, never a bare contract composition.
    """
    from merlin.targetgen import capability_derive as CD
    from merlin.targetgen.semantic_families import from_op

    intent = CD.DerivedCapabilities()
    CD._from_unit_intent(dict(contract), intent)
    out: dict[str, dict[str, dict[str, FactField]]] = {}
    granularities, granularity_complete, granularity_notes = set(), bool(facets), []
    for facet in facets:
        scale = facet.get("scale") or {}
        granularities.update(str(value) for value in scale.get("granularities") or ())
        granularity_complete = granularity_complete and scale.get("granularities_complete") is True
        granularity_notes.append(f"{facet.get('unit')} scale carriers: {scale.get('carriers')}")
    for family, flat in families.items():
        evidence = derived.supported.get(family)
        capabilities = declared.get(family, [])
        forms: dict[str, dict[str, FactField]] = {}
        placement = flat.get("placement")
        if placement is not None and placement.values == ("accelerator",) and placement.complete:
            dtypes, notes = set(), []
            for capability in capabilities:
                if not capability.composed_with and capability.dtypes:
                    dtypes.update(_dtype(value) for value in capability.dtypes)
                    notes.append(f"standalone contract capability dtypes {list(capability.dtypes)}")
            if evidence is not None and not evidence.composed_with and evidence.source != "unit_intent":
                dtypes.update(_dtype(value) for value in evidence.dtypes)
                notes.append(f"{evidence.source}: {list(evidence.dtypes)}")
            if family == "movement":
                dtypes.update(_dtype(row["dtype"]) for row in storage if row.get("dtype"))
                notes.extend(f"storage datapath {row.get('name')}={row.get('dtype')}" for row in storage)
            form = {"placement": FactField(("accelerator",), True, placement.evidence)}
            if dtypes:
                form["dtypes"] = FactField(_sorted(dtypes), True, tuple(notes))
            if family == "elementwise_map" and granularities:
                form["scale_granularity"] = FactField(
                    _sorted(granularities), granularity_complete, tuple(granularity_notes)
                )
            forms["standalone"] = form
        partners, dtypes, notes, stages, stage_notes = set(), set(), [], set(), []
        for capability in capabilities:
            if capability.composed_with:
                partners.update(capability.composed_with)
                dtypes.update(_dtype(value) for value in capability.dtypes)
                notes.append(f"contract composed_with {list(capability.composed_with)}")
        rungs = [row for row in (evidence, intent.supported.get(family)) if row is not None and row.composed_with]
        for rung in rungs:
            partners.update(rung.composed_with)
            dtypes.update(_dtype(value) for value in rung.dtypes)
            notes.append(f"{rung.source}: {rung.evidence}")
        for facet in facets:
            for readout in facet.get("readouts") or ():
                applied = {_stage(stage) for stage in readout.get("applies") or () if from_op(str(stage)) == family}
                if applied:
                    stages.update(applied)
                    if readout.get("selector"):
                        dtypes.add(_dtype(readout["selector"]))
                    # A fused readout stage reads the accumulator it drains, so the accumulator's
                    # format is an operand format of the stage (the SW screen observes it so).
                    if facet.get("accumulator_dtype"):
                        dtypes.add(_dtype(facet["accumulator_dtype"]))
                        stage_notes.append(f"{facet.get('unit')} accumulator {facet['accumulator_dtype']!r}")
                    stage_notes.append(
                        f"{facet.get('unit')} readout {readout.get('selector')!r} applies {sorted(applied)}"
                    )
            for route in facet.get("stage_routes") or ():
                if (
                    not isinstance(route, Mapping)
                    or route.get("composed_with") != "contraction"
                    or route.get("site") != "accumulator_seed"
                ):
                    continue
                applied = {_stage(stage) for stage in route.get("stages") or () if from_op(str(stage)) == family}
                if applied:
                    stages.update(applied)
                    dtypes.update(_dtype(selector) for selector in route.get("readouts") or ())
                    stage_notes.append(
                        f"{facet.get('unit')} {route.get('site')} route with contraction applies "
                        f"{sorted(applied)}: {route.get('evidence')}"
                    )
        if partners and (rungs or stages):
            form = {
                "placement": FactField(("fused_accelerator",), True, tuple(notes + stage_notes)),
                "composed_with": FactField(_sorted(partners), True, tuple(notes)),
            }
            if dtypes:
                form["dtypes"] = FactField(_sorted(dtypes), True, tuple(notes + stage_notes))
            if family == "elementwise_map" and stages:
                form["epilogues"] = FactField(_sorted(stages), True, tuple(stage_notes))
            if family == "elementwise_map" and granularities:
                form["scale_granularity"] = FactField(
                    _sorted(granularities), granularity_complete, tuple(granularity_notes)
                )
            forms["fused"] = form
        # A standalone operand sum writes an accumulator that can use the SAME unit's readout.
        # This is a distinct composition from a contraction epilogue: joining their partners and
        # stages in one form would incorrectly license a cross-product of unrelated readout paths.
        if family == "elementwise_map" and "standalone" in forms:
            from merlin.xdsl_dialects.lowering.compute_groups import RESIDUAL_ADD

            eligible = []
            for facet in facets:
                operand_sum = facet.get("operand_sum") or {}
                operand_count = operand_sum.get("operands") if isinstance(operand_sum, Mapping) else None
                if facet.get("unknown") or type(operand_count) is not int or operand_count < 2:
                    continue
                operand_dtype = operand_sum.get("operand_dtype")
                if not operand_dtype:
                    continue
                for readout in facet.get("readouts") or ():
                    if not isinstance(readout, Mapping) or _dtype(readout.get("selector")) != _dtype(operand_dtype):
                        continue
                    applied = {
                        _stage(stage) for stage in readout.get("applies") or () if from_op(str(stage)) in (None, family)
                    }
                    if applied:
                        eligible.append((facet, readout, applied))
            if eligible:
                # A declaration is safe across alternative units/readouts only for stages every
                # eligible path carries. A wider union would admit a stage the selected path drops.
                common_stages = set.intersection(*(applied for _facet, _readout, applied in eligible))
                common_granularities = set.intersection(
                    *(set((facet.get("scale") or {}).get("granularities") or ()) for facet, _readout, _ in eligible)
                )
                if common_stages:
                    evidence = tuple(
                        f"{facet.get('unit')} operand sum on load -> readout {readout.get('selector')!r}: "
                        f"{readout.get('evidence')}"
                        for facet, readout, _applied in eligible
                    )
                    form = {
                        "placement": FactField(("fused_accelerator",), True, evidence),
                        "composed_with": FactField((RESIDUAL_ADD,), True, evidence),
                        "dtypes": FactField(
                            _sorted(_dtype(readout["selector"]) for _facet, readout, _applied in eligible),
                            True,
                            evidence,
                        ),
                        "epilogues": FactField(_sorted(common_stages), True, evidence),
                    }
                    if common_granularities:
                        complete = all(
                            (facet.get("scale") or {}).get("granularities_complete") is True
                            for facet, _readout, _ in eligible
                        )
                        form["scale_granularity"] = FactField(
                            _sorted(common_granularities),
                            complete,
                            evidence,
                        )
                    forms["fused_operand_sum"] = form
        if forms:
            out[family] = forms
    return out


# --- filling an authored spec's hardware-shaped fields from the facts ------------------------------

#: The fields a ``hardware:`` declaration may not author: the selected facts fill them.
DERIVED_FIELDS = ("placement", "composed_with", "dtypes", "operand_dtypes", "epilogues", "scale_granularity")
FORMS = ("standalone", "fused", "fused_operand_sum")
FROM_FACTS = "from_facts"
RESOLUTION_SCHEMA = "merlin.phase0.software_spec_derivation.v1"


def semantic_family(row: Mapping[str, Any]) -> str | None:
    """The one semantic family a ``hardware:`` declaration describes, or ``None`` when ambiguous."""
    from merlin.targetgen.semantic_families import FAMILIES

    found = {family for family in _row_families(row) if family in FAMILIES}
    if not found and row.get("id") in FAMILIES:
        found = {row["id"]}
    return next(iter(found)) if len(found) == 1 else None


def resolve_spec(spec: Mapping[str, Any], facts: Mapping[str, Any]) -> tuple[dict, dict]:
    """Fill every ``hardware:`` declaration and ``eligible_operations: from_facts`` from ``facts``.

    Returns the resolved spec and a record of what was filled from which evidence. A form the facts do
    not establish resolves to placement ``unknown`` and is listed under ``unresolved`` -- the
    declaration cannot be admitted, and the selection reports it rather than inventing a value. A
    reasoned ``restrictions`` entry removes the values it declines from a derived list; one that would
    remove every value, or a whole form's placement, is an authoring error.
    """
    import copy

    validate_restrictions(spec)
    resolved = copy.deepcopy(dict(spec))
    forms = facts.get("forms") or {}
    rows, unresolved = [], []
    for row in resolved.get("operations") or ():
        form_name = row.pop("hardware", None)
        if form_name is None:
            continue
        family = semantic_family(row)
        form = (forms.get(family) or {}).get(form_name)
        record = {"id": row["id"], "family": family, "form": form_name, "fields": {}, "evidence": {}}
        signature = row.setdefault("signature", {})
        declared_form = ((facts.get("declared_unchecked_forms") or {}).get(family) or {}).get(form_name)
        if form is None and declared_form and spec.get("status") == "reviewed" and row.get("status") == "reviewed":
            row["placement"] = declared_form["placement"][0]
            for field, values in declared_form.items():
                if field != "placement" and values:
                    # Authored constraints may narrow this declaration but never widen it.
                    selected = signature.get(field, values)
                    signature[field] = [value for value in selected if value in values]
            row["admission_basis"] = "declared_unchecked"
            record.update(status="declared_unchecked", basis="declared_unchecked", fields=declared_form)
            rows.append(record)
            continue
        if form is None:
            row["placement"] = "unknown"
            record["status"] = "unresolved"
            record["reason"] = f"the selected facts establish no {form_name} form of {family}"
            unresolved.append(record)
            rows.append(record)
            continue
        row["derived_from_facts"] = {"form": form_name, "family": family}
        for field, fact in sorted(form.items()):
            values = list(fact.values)
            if field == "placement":
                if _covering(spec, family, "placement", set(values), declaration=row["id"])[0]:
                    raise ValueError(f"{row['id']}: declares a {form_name} form its own restriction declines")
                row["placement"] = values[0]
            else:
                declined, used = _covering(spec, family, field, set(values), declaration=row["id"])
                values = [value for value in values if value not in declined]
                if not values:
                    raise ValueError(f"{row['id']}: restrictions decline every fact-derived {field}")
                signature[field] = values
                if used:
                    record.setdefault("restrictions", {})[field] = used
            record["fields"][field] = values
            record["evidence"][field] = list(fact.evidence)
        record["status"] = "resolved"
        rows.append(record)
    quantizable = set(facts.get("quantization") or {})
    eligible_record = []
    for declaration in (resolved.get("quantization") or {}).get("formats") or ():
        if declaration.get("eligible_operations") != FROM_FACTS:
            continue
        format_id = str(
            declaration.get("id")
            or f"{_dtype(declaration.get('operand_dtype'))}__{_dtype(declaration.get('accumulator_dtype'))}"
        )
        selected = []
        for row in resolved.get("operations") or ():
            if row.get("placement") != "accelerator":
                continue
            covered = _row_families(row) & quantizable
            covered = {f for f in covered if not _covering(spec, f, "quantization", {format_id})[0]}
            if covered:
                selected.append(row["id"])
        if not selected:
            unresolved.append({"format": format_id, "reason": "no accelerator declaration covers a quantizable family"})
        declaration["eligible_operations"] = selected
        eligible_record.append({"format": format_id, "eligible_operations": selected})
    record = {
        "schema": RESOLUTION_SCHEMA,
        "status": "resolved" if not unresolved else "unresolved",
        "declarations": rows,
        "quantization": eligible_record,
        "unresolved": unresolved,
    }
    if rows or eligible_record:
        resolved["fact_derivation"] = {"schema": RESOLUTION_SCHEMA, "status": record["status"]}
    return resolved, record


# --- the authored side ----------------------------------------------------------------------------


def _row_families(row: Mapping[str, Any]) -> set[str]:
    from merlin.targetgen.semantic_families import from_op

    found = set(row.get("families") or ())
    found.update(family for op in row.get("ops") or () if (family := from_op(str(op))))
    return found


def _authored(spec: Mapping[str, Any], family: str) -> dict[str, Any]:
    """The union of every authored declaration covering ``family``: its widest placement and the
    accelerator-side constraints. A field no accelerator declaration names is unconstrained."""
    rows = [row for row in spec.get("operations") or () if isinstance(row, Mapping) and family in _row_families(row)]
    mode = max((_PLACEMENT_RANK.get(str(row.get("placement")), 0) for row in rows), default=0)
    device = [row for row in rows if _PLACEMENT_RANK.get(str(row.get("placement")), 0) > 0]
    fields: dict[str, set | None] = {}
    for field, keys in (
        ("dtypes", ("operand_dtypes", "dtypes")),
        ("composed_with", ("composed_with",)),
        ("epilogues", ("epilogues",)),
        ("scale_granularity", ("scale_granularity",)),
        ("ranks", ("ranks",)),
    ):
        seen = None
        for row in device:
            signature = row.get("signature") or {}
            for key in keys:
                if key in signature:
                    value = signature[key]
                    values = value if isinstance(value, list) else [value]
                    seen = (seen or set()) | {
                        _dtype(v) if field == "dtypes" else _stage(v) if field == "epilogues" else v for v in values
                    }
        fields[field] = seen
    unknown_only = bool(rows) and all(str(row.get("placement")) == "unknown" for row in rows)
    return {"declarations": [row.get("id") for row in rows], "mode": mode, "unknown_only": unknown_only, **fields}


def _authored_placement(authored: Mapping[str, Any]) -> str:
    return "unknown" if authored["unknown_only"] else _PLACEMENT_NAME[authored["mode"]]


def _restrictions(spec: Mapping[str, Any]) -> list[dict]:
    rows = spec.get("restrictions") or []
    return [row for row in rows if isinstance(row, Mapping)]


def validate_restrictions(spec: Mapping[str, Any]) -> list[dict]:
    """Restrict a whole family or one fact-derived declaration, with an explicit reason."""
    rows = spec.get("restrictions")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("restrictions must be a list")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) not in (
            {"family", "field", "values", "reason"},
            {"family", "field", "values", "reason", "declaration"},
        ):
            raise ValueError("each restriction needs exactly family, field, values and reason, optionally declaration")
        if row["field"] not in RESTRICTABLE_FIELDS:
            raise ValueError(f"restriction field {row['field']!r} is not one of {sorted(RESTRICTABLE_FIELDS)}")
        if not isinstance(row["family"], str) or not row["family"].strip():
            raise ValueError("restriction family must be a nonempty string")
        if "declaration" in row:
            matching = [
                declaration
                for declaration in spec.get("operations") or ()
                if isinstance(declaration, Mapping) and declaration.get("id") == row["declaration"]
            ]
            if (
                len(matching) != 1
                or (matching[0].get("hardware") is None and not matching[0].get("derived_from_facts"))
                or semantic_family(matching[0]) != row["family"]
            ):
                raise ValueError("declaration-specific restriction must name a fact-derived declaration of its family")
        if not isinstance(row["reason"], str) or len(row["reason"].split()) < 3:
            raise ValueError(f"restriction of {row['family']}.{row['field']} needs a stated reason")
        values = row["values"]
        if not isinstance(values, list) or not values or any(isinstance(v, (dict, list)) for v in values):
            raise ValueError(f"restriction of {row['family']}.{row['field']} must list the declined values")
    return list(rows)


def _canonical(field: str, value: Any) -> Any:
    return _dtype(value) if field == "dtypes" else _stage(value) if field == "epilogues" else value


def _covering(
    spec: Mapping[str, Any], family: str, field: str, missing: set, *, declaration: str | None = None
) -> tuple[set, list[dict]]:
    """The missing values a recorded restriction declines, and the restrictions that do it."""
    covered, used = set(), []
    for row in _restrictions(spec):
        if row.get("family") != family or row.get("field") != field:
            continue
        if "declaration" in row and row["declaration"] != declaration:
            continue
        declined = {_canonical(field, value) for value in row.get("values") or ()} & missing
        if declined:
            covered |= declined
            used.append(dict(row))
    return covered, used


def _finding(family, field, authored, fact: FactField, *, spec) -> list[dict]:
    """Classify one field. A field can be narrower in some values and wider in others."""
    base = {"family": family, "field": field, "authored": sorted(authored, key=str), "facts": fact.to_dict()}
    established = set(fact.values)
    missing = established - set(authored)
    excess = set(authored) - established
    out = []
    if missing:
        covered, used = _covering(spec, family, field, missing)
        if covered:
            out.append({**base, "classification": RESTRICTED, "values": sorted(covered, key=str), "restrictions": used})
        if missing - covered:
            out.append(
                {
                    **base,
                    "classification": FORBIDS,
                    "values": sorted(missing - covered, key=str),
                    "reason": "authored spec forbids what the hardware facts establish",
                }
            )
    if excess:
        out.append(
            {
                **base,
                "classification": EXCEEDS if fact.complete else UNCONFIRMED,
                "values": sorted(excess, key=str),
                "reason": (
                    "authored spec claims what the hardware facts decide against"
                    if fact.complete
                    else "authored spec claims values no fact source could confirm for this axis"
                ),
            }
        )
    return out or [{**base, "classification": OK, "values": []}]


def spec_fact_drift(spec: Mapping[str, Any], facts: Mapping[str, Any], *, target: str | None = None) -> dict[str, Any]:
    """Every compared (family, field) with its classification; blocking findings are counted."""
    validate_restrictions(spec)
    findings: list[dict] = []
    families = facts.get("families") or {}
    from merlin.targetgen.semantic_families import FAMILIES

    # Placement and shape compare per SEMANTIC family; a row may also name a quantization selector
    # (a recipe family), which is compared below under quantization only.
    authored_families = {
        family for row in spec.get("operations") or () for family in _row_families(row) if family in FAMILIES
    }
    for family in sorted({*families, *authored_families}):
        fact_row = families.get(family) or {}
        authored = _authored(spec, family)
        placement = fact_row.get("placement")
        if placement is None:
            if authored["mode"] > 0:
                findings.append(
                    {
                        "family": family,
                        "field": "placement",
                        "authored": [_authored_placement(authored)],
                        "facts": None,
                        "classification": UNDETERMINED,
                        "values": [_PLACEMENT_NAME[authored["mode"]]],
                        "reason": "no fact source reports this family",
                    }
                )
            continue
        fact_mode = _PLACEMENT_RANK[placement.values[0]]
        if fact_mode == 0 and authored["mode"] == 0:
            continue  # host-only on both sides: nothing about the accelerator is claimed or forbidden
        if not placement.complete and fact_mode == 0:
            findings.append(
                {
                    "family": family,
                    "field": "placement",
                    "authored": [_authored_placement(authored)],
                    "facts": placement.to_dict(),
                    "classification": UNDETERMINED,
                    "values": [],
                    "reason": "no fact source could decide this family",
                }
            )
            continue
        # Placement compares capability RANK: authoring fused where the facts establish standalone is
        # narrower even though both are accelerator placements.
        if authored["mode"] == fact_mode:
            findings.append(
                {
                    "family": family,
                    "field": "placement",
                    "authored": [_authored_placement(authored)],
                    "facts": placement.to_dict(),
                    "classification": OK,
                    "values": [],
                    "declarations": authored["declarations"],
                }
            )
        else:
            narrower = authored["mode"] < fact_mode
            values = {placement.values[0]} if narrower else {_PLACEMENT_NAME[authored["mode"]]}
            if narrower and (not placement.complete or authored["unknown_only"]):
                # An unevidenced fact cannot hold the spec to it, and an authored ``unknown`` is an
                # undecided placement rather than a refusal: both are open decisions, reported.
                used, classification = [], UNDETERMINED
            elif narrower:
                covered, used = _covering(spec, family, "placement", values)
                classification = RESTRICTED if covered else FORBIDS
            else:
                used, classification = [], EXCEEDS if placement.complete else UNCONFIRMED
            findings.append(
                {
                    "family": family,
                    "field": "placement",
                    "authored": [_authored_placement(authored)],
                    "facts": placement.to_dict(),
                    "classification": classification,
                    "values": sorted(values),
                    "declarations": authored["declarations"],
                    **({"restrictions": used} if used else {}),
                    **(
                        {"reason": "authored spec forbids what the hardware facts establish"}
                        if classification == FORBIDS
                        else {"reason": "authored spec claims what the hardware facts decide against"}
                        if classification == EXCEEDS
                        else {
                            "reason": "the authored placement is unknown while the facts establish an accelerator form"
                        }
                        if classification == UNDETERMINED and authored["unknown_only"]
                        else {
                            "reason": "the narrower authored placement is measured against an unevidenced declaration"
                        }
                        if classification == UNDETERMINED
                        else {}
                    ),
                }
            )
        if authored["mode"] == 0 or fact_mode == 0:
            continue
        for field in ("dtypes", "epilogues", "scale_granularity", "ranks", "composed_with"):
            fact = fact_row.get(field)
            authored_values = authored.get(field)
            if field == "composed_with" and (fact is None or authored["mode"] != 1):
                continue  # composition partners compare only where both sides are fused
            if authored_values is None or fact is None:
                continue  # unconstrained on the authored side, or nothing the facts can say
            findings.extend(_finding(family, field, authored_values, fact, spec=spec))

    from merlin.targetgen import quant_recipe as QR

    # The recipe's own family vocabulary; a declaration's other families are never quantized by a
    # recipe (``capture_recipe_candidates`` intersects with it), so they claim nothing here.
    vocabulary = {QR.CONTRACTION, QR.OPERAND_SUM, QR.WINDOW_MEAN}
    quantizable = facts.get("quantization") or {}
    for declaration in (spec.get("quantization") or {}).get("formats") or ():
        if not isinstance(declaration, Mapping):
            continue
        format_id = str(
            declaration.get("id")
            or f"{_dtype(declaration.get('operand_dtype'))}__{_dtype(declaration.get('accumulator_dtype'))}"
        )
        operations = {row.get("id"): row for row in spec.get("operations") or () if isinstance(row, Mapping)}
        eligible = set()
        for identity in declaration.get("eligible_operations") or ():
            row = operations.get(identity) or {}
            if _PLACEMENT_RANK.get(str(row.get("placement")), 0) > 0:
                eligible |= _row_families(row) & vocabulary
        for family in sorted({*quantizable, *eligible}):
            fact = quantizable.get(family)
            if fact is None:
                if family in eligible and quantizable:
                    findings.append(
                        {
                            "family": family,
                            "field": "quantization",
                            "authored": [format_id],
                            "facts": {"values": [], "complete": True, "evidence": []},
                            "classification": EXCEEDS,
                            "values": [format_id],
                            "reason": "authored spec quantizes a family no derived hardware recipe quantizes",
                        }
                    )
                continue
            authored_values = {format_id} if family in eligible else set()
            findings.extend(
                _finding(
                    family, "quantization", authored_values, FactField((format_id,), True, fact.evidence), spec=spec
                )
            )
    blocking = [row for row in findings if row["classification"] in BLOCKING]
    return {
        "schema": SCHEMA,
        "target": target or spec.get("target"),
        "status": "consistent" if not blocking else "drift",
        "n_blocking": len(blocking),
        "counts": {
            kind: sum(row["classification"] == kind for row in findings)
            for kind in (OK, RESTRICTED, FORBIDS, EXCEEDS, UNCONFIRMED, UNDETERMINED)
        },
        "findings": findings,
        "qualification": (
            "static comparison of authored declarations with selected fact-derived capability; "
            "neither side is executed here"
        ),
    }


def blockers(report: Mapping[str, Any] | None) -> list[dict]:
    """Readiness blockers for a drift report; an absent report is itself a blocker."""
    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        return [{"component": "software_spec_drift", "reason": "spec-vs-facts drift check is absent"}]
    return [
        {
            "component": "software_spec_drift",
            "family": row["family"],
            "field": row["field"],
            "classification": row["classification"],
            "values": row.get("values"),
            "reason": row.get("reason"),
        }
        for row in report.get("findings") or ()
        if row.get("classification") in BLOCKING
    ]
