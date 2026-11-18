"""Host-owned Phase 0 sweeps implementation."""

from __future__ import annotations

import copy
import hashlib

from merlin.analysis.perf.contract import workload_facts as _merlin_workload_facts_1
from merlin.analysis.perf.contract import workload_programs as WG  # noqa: E402
from merlin.analysis.perf.contract.profile import TRAITS, derive_profile  # noqa: E402
from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1
from merlin.development.testgen.generation.provenance import _document_digest
from merlin.development.testgen.performance.capture_residency import (
    _bound_capture_inventory,
    _capture_residency_sweep,
)
from merlin.development.testgen.profiles.loading import (
    _DERIVED_AXES,
    _comparison_roles,
    _validate_performance_block,
    pending_host_source_basis,
)
from merlin.development.testgen.profiles.synthesis_selection import _cross_product, _render_variant
from merlin.runtime.backends.registry import EXECUTION_CAPABILITIES, execution_capability_facts  # noqa: E402

# ---- sweeps ------------------------------------------------------------------------------------


def _materialize_performance_entry(entry: dict, binding, *, oracle_selection: dict[str, str] | None = None) -> dict:
    """Resolve a performance member onto a runnable direct corpus path.

    Dtypes come from workload_gen's capability-manifest accessor and must agree
    with the corpus binding selected for the same target.  The shared template
    therefore contains neither a target dtype nor a frontend choice.
    """
    from merlin.development.testgen.profiles.sweep_axes import _accum_for_encoding, _resolve_target_oracle_evidence

    target = str(getattr(binding, "target", "") or "")
    if not target:
        raise ValueError("performance materialization needs binding.target")
    binding_operand = getattr(binding, "operand_dtype", None)
    binding_accum = getattr(binding, "accum_dtype", None)
    declared = entry.pop("_encoding_variant", None)
    derived_axes = entry.pop("_derived_axes", None)
    if declared:
        # AN ENCODING FAMILY'S WHOLE CONTENT IS THAT ITS MEMBERS DIFFER HERE. This function otherwise
        # overwrites `operand_dtype` with the target's single corpus binding, which is right for every
        # other family -- they compare geometry or fusion at ONE datapath and must not drift off it --
        # and fatal for an L6_global comparison: both members would collapse onto the same encoding and
        # the differential would read "no effect" for a lever that was never exercised.
        #
        # The accumulate format is DERIVED for that operand rather than carried over from the binding:
        # an accumulator belongs to the datapath the operand feeds, and pairing one encoding's operand
        # with another's accumulator describes a machine that does not exist.
        operand_dtype = str(declared)
        accum_dtype = _accum_for_encoding(target, operand_dtype, binding_accum)
        datatype_basis = "declared_encoding_variant"
    elif binding_operand and binding_accum:
        # This is the binding _write_capsule will actually consume, derived by
        # corpus_spec from the target experiment + numeric profile. Prefer it
        # over re-deriving through a second capability-manifest representation.
        operand_dtype, accum_dtype = str(binding_operand), str(binding_accum)
        datatype_basis = "corpus_binding"
    else:
        operand_dtype, accum_dtype = _merlin_workload_facts_1.datapath_formats(target, accum_dtype=binding_accum)
        datatype_basis = "merlin.analysis.perf.contract.workload_facts.datapath_formats"
    op = str(entry.get("op") or "")
    if op not in _merlin_model_builders_1.BUILDERS:
        raise ValueError(
            f"performance member {entry.get('name', '?')}: direct corpus source has no runnable builder for {op!r}"
        )
    entry["source"] = "direct"
    entry["operand_dtype"] = operand_dtype
    performance = copy.deepcopy(entry["performance"])
    if oracle_selection is None:
        performance = _resolve_target_oracle_evidence(performance, target)
    else:
        performance = _resolve_target_oracle_evidence(performance, target, oracle_selection=oracle_selection)
    performance["emitter"] = copy.deepcopy(performance["emitter"])
    performance["emitter"]["resolved"] = {
        "source": "direct",
        "operand_dtype": operand_dtype,
        "accum_dtype": accum_dtype,
        "datatype_basis": datatype_basis,
        "builder": "merlin.development.targetgen.corpus.model_builders.build",
    }
    if derived_axes:
        # THE UNREACHABLE REGIMES TRAVEL WITH THE CAPSULE. The derivation knows which bands it could
        # not reach and why, and that answer is worth exactly as much as the points it did reach: a
        # regime nothing covers because nothing CAN is a result, while one that is merely absent from
        # the corpus is a hole. Recorded on every member so a corpus-side gate can read it out of the
        # tracked capsule rather than out of a generation report that may not have been rewritten.
        performance["emitter"]["derived_axes"] = copy.deepcopy(derived_axes)
    entry["performance"] = performance
    return entry


def _scope_requirement_sweeps(
    sweeps: list[dict],
    requirement: dict | None,
    digest: str | None,
    skipped: list | None,
    blocked: list | None,
) -> list[dict]:
    """Derive one exact claim cohort per supported captured scope signature.

    The template declares a family *pattern*. Each requirement row becomes its
    own digest-named family so an affine analyzer never mixes different chains.
    Eight regions is this builder's current cost/census cap, not a device limit.
    Unsupported rows are explicit debt, never silently truncated to one match.
    """
    expanded = []
    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError(f"sweep entry {sweep!r} is not a mapping")
        pattern = sweep.get("requires_scope_pattern")
        if pattern is None:
            expanded.append(sweep)
            continue
        family = str(sweep.get("id") or "")
        if pattern != {"prefix": ["movement", "contraction"], "repeated_tail": "elementwise_map", "min_tail": 1}:
            raise ValueError(f"performance sweep {family}: unsupported scope pattern")
        scope = (requirement or {}).get("scope") or {}
        performance_scope = scope.get("performance") or {}
        if performance_scope.get("schema") != "merlin.phase0.performance_scope.v1":
            if blocked is not None:
                blocked.append(
                    {
                        "family": family,
                        "sweep": family,
                        "status": "blocked_unimplemented",
                        "reason": "selected requirement lacks exact SW/emitter-derived Phase 2 scope",
                        "requirement_sha256": digest,
                    }
                )
            continue
        from merlin.development.testgen.performance.scope import validate_performance_scope

        performance_scope = validate_performance_scope(scope)
        required = performance_scope.get("required") or []
        matched = 0
        seen_families: set[str] = set()
        for row in sorted(required, key=lambda item: str(item.get("signature")) if isinstance(item, dict) else ""):
            if not isinstance(row, dict) or not isinstance(row.get("signature"), str):
                continue
            signature = row["signature"]
            families = signature.split(" -> ")
            if not (
                len(families) >= 3
                and families[:2] == pattern["prefix"]
                and all(part == pattern["repeated_tail"] for part in families[2:])
            ):
                continue
            matched += 1
            if row.get("length") != len(families) or not digest:
                reason = "scope requirement length or frozen digest is invalid"
            elif len(families) > 8:
                reason = "scope chain exceeds this builder's eight-region cost/census cap, not a hardware limit"
            else:
                reason = None
            if reason is not None:
                if blocked is not None:
                    blocked.append(
                        {
                            "family": family,
                            "sweep": family,
                            "status": "blocked_unimplemented",
                            "reason": reason,
                            "signature": signature,
                            "requirement_sha256": digest,
                        }
                    )
                continue
            derived_family = f"{family}_{hashlib.sha256(signature.encode()).hexdigest()[:12]}"
            if derived_family in seen_families:
                raise ValueError(f"scope sweep {family}: duplicate or colliding signature {signature!r}")
            seen_families.add(derived_family)
            selected = copy.deepcopy(sweep)
            del selected["requires_scope_pattern"]
            selected["id"] = derived_family
            selected["base"]["scope_families"] = families
            performance = selected["base"]["performance"]
            performance["family"] = derived_family
            performance["requirement_basis"] = {
                "sha256": digest,
                "axis": "scope.performance.required",
                "pattern_family": family,
                "signature": signature,
                "occurrences": row.get("occurrences"),
            }
            selected["source_reference"] = (
                str(selected.get("source_reference") or "") + f"; selected scope.performance.required: {signature}"
            )
            expanded.append(selected)
        if matched == 0:
            if performance_scope.get("status") == "unresolved" and blocked is not None:
                blocked.append(
                    {
                        "family": family,
                        "sweep": family,
                        "status": "blocked_unimplemented",
                        "reason": "exact source/SW/emitter Phase 2 scope remains unresolved",
                        "required_pattern": pattern,
                        "requirement_sha256": digest,
                    }
                )
            elif skipped is not None:
                skipped.append(
                    {
                        "family": family,
                        "sweep": family,
                        "status": "skipped_inapplicable",
                        "reason": "selected frozen requirement has no eligible scope-chain signature",
                        "required_pattern": pattern,
                        "requirement_sha256": digest,
                    }
                )
    return expanded


def _capture_shape_sweeps(
    sweeps: list[dict],
    binding,
    basis: dict | None,
    basis_sha256: str | None,
    requirement: dict | None,
    requirement_sha256: str | None,
    skipped: list | None,
    evidence=None,
) -> list[dict]:
    """Add the smallest capture-shaped law cohorts preserving the original K grid.

    This selects *shape relevance*, not source equivalence, compiler placement,
    numerical safety, or achieved cycles. Every other independent contraction
    remains visible in the generation audit, including off-grid shapes.
    """
    expanded = []
    for sweep in sweeps:
        pattern = sweep.get("capture_shape_pattern")
        if isinstance(pattern, dict) and pattern.get("kind") == "observed_mn_for_memory_regime":
            expanded.extend(
                _capture_residency_sweep(
                    sweep, binding, basis, basis_sha256, requirement, requirement_sha256, evidence, skipped
                )
            )
            continue
        expanded.append(sweep)
        if pattern is None:
            continue
        family = str(sweep.get("id") or "")
        if (
            not isinstance(pattern, dict)
            or pattern.get("kind") != "observed_mn_on_existing_k_grid"
            or type(pattern.get("max_cohorts")) is not int
            or pattern["max_cohorts"] < 1
            or (sweep.get("base") or {}).get("op") != "matmul"
        ):
            raise ValueError(f"performance sweep {family}: unsupported capture-shape pattern")
        if basis is None:
            if skipped is not None:
                skipped.append(
                    {
                        "family": f"{family}.capture_shape",
                        "status": "skipped_inapplicable",
                        "reason": "no frozen selected-capture performance basis",
                    }
                )
            continue
        inventory_digest = _bound_capture_inventory(
            basis, basis_sha256, requirement, requirement_sha256, str(getattr(binding, "target", ""))
        )
        tile = int(getattr(binding, "tile_dim", 0) or 0)
        if tile < 1 or sweep.get("axes", {}).get("K") != ["tile", "2*tile", "4*tile", "8*tile"]:
            raise ValueError(f"performance sweep {family}: capture-shape law needs its unchanged four-point K grid")
        k_grid = {tile, 2 * tile, 4 * tile, 8 * tile}
        operand_dtype = getattr(binding, "operand_dtype", None)
        from merlin.common import target_data as quant_formats

        wanted_dtype = quant_formats.get(operand_dtype).name

        def canonical_format(value: object) -> str | None:
            try:
                return quant_formats.get(value).name if isinstance(value, str) else None
            except KeyError:
                return None

        eligible, remainder = [], []
        for application, app in sorted((basis.get("applications") or {}).items()):
            for index, row in enumerate(app.get("rows") or []):
                if row.get("independent_compute_demand") is not True or row.get("semantic_family") != "contraction":
                    continue
                shape = row.get("contraction_shape") or {}
                m, k, n = (shape.get(axis) for axis in ("M", "K", "N"))
                total = (row.get("macs") or {}).get("total")
                witness = {
                    "application": application,
                    "capture_sha256": app.get("capture_sha256"),
                    "signature_index": index,
                    "operation": row.get("operation"),
                    "shape": {"M": m, "K": k, "N": n},
                    "count": row.get("count"),
                    "known_static_macs": total,
                    "source_mlir_operation": row.get("mlir_operation"),
                    "source_operand_format": row.get("operand_format"),
                    "source_result_dtypes": copy.deepcopy(row.get("result_dtypes")),
                    "source_shape_confidence": row.get("shape_confidence"),
                }
                if any(type(axis) is not int or axis < 1 for axis in (m, k, n)) or type(total) is not int or total < 1:
                    reason = "no recognized exact static single-MAC shape"
                elif k not in k_grid:
                    reason = "observed K is outside the unchanged four-point law grid"
                elif canonical_format(row.get("operand_format")) is None:
                    reason = "captured operand format is not a known selected encoding"
                elif canonical_format(row.get("operand_format")) != wanted_dtype:
                    reason = "captured operand format differs from selected corpus binding"
                else:
                    reason = None
                if reason is None:
                    eligible.append(witness)
                else:
                    remainder.append({**witness, "reason": reason})
        groups: dict[tuple[int, int], list[dict]] = {}
        for witness in eligible:
            shape = witness["shape"]
            groups.setdefault((shape["M"], shape["N"]), []).append(witness)
        ranked = sorted(
            groups.items(),
            key=lambda item: (-sum(row["known_static_macs"] for row in item[1]), item[0]),
        )
        if not ranked:
            if skipped is not None:
                skipped.append(
                    {
                        "family": f"{family}.capture_shape",
                        "status": "skipped_inapplicable",
                        "reason": "no selected contraction fits the unchanged law grid and dtype",
                        "performance_basis_sha256": basis_sha256,
                        "remainder": remainder,
                    }
                )
            continue
        max_cohorts = pattern["max_cohorts"]
        for _, witnesses in ranked[max_cohorts:]:
            remainder.extend(
                {**row, "reason": f"deferred by shared-template max_cohorts={max_cohorts}"} for row in witnesses
            )
        remainder.sort(key=lambda row: (row["application"], row["signature_index"]))
        if skipped is not None and remainder:
            skipped.append(
                {
                    "family": f"{family}.capture_shape.remainder",
                    "status": "skipped_inapplicable",
                    "reason": (
                        "captured contraction demands excluded by the existing K grid, dtype, or declared cohort cap"
                    ),
                    "performance_basis_sha256": basis_sha256,
                    "remainder": remainder,
                }
            )
        for (m, n), witnesses in ranked[:max_cohorts]:
            witnesses = sorted(witnesses, key=lambda row: (row["application"], row["signature_index"]))
            identity = hashlib.sha256(f"{basis_sha256}:{m}:{n}".encode()).hexdigest()[:12]
            derived = copy.deepcopy(sweep)
            del derived["capture_shape_pattern"]
            derived["id"] = f"{family}_capture_{identity}"
            derived["axes"] = {"M": [m], "N": [n], "K": list(sweep["axes"]["K"])}
            derived["base"]["performance"]["family"] = derived["id"]
            derived["base"]["performance"]["requirement_basis"] = {
                "axis": "coverage.performance-basis.capture_shape",
                "sha256": requirement_sha256,
                "performance_basis_sha256": basis_sha256,
                "selected_inventory_sha256": inventory_digest,
                "pattern_family": family,
                "observed_mn": {"M": m, "N": n},
                "observed_k": sorted({row["shape"]["K"] for row in witnesses}),
                "source_witnesses": witnesses,
                "source_match": "capture_shape_candidate",
                "qualification": (
                    "observed M/N and listed K points in the unchanged law grid; no source-body equivalence, "
                    "compiler placement, source range safety, timing, or speedup is asserted"
                ),
            }
            derived["source_reference"] = (
                f"selected frozen performance basis {basis_sha256}: observed M={m}, N={n}, "
                f"K={sorted({row['shape']['K'] for row in witnesses})}; shape-only diagnostic"
            )
            expanded.append(derived)
    return expanded


def expand_sweeps(
    profile: dict,
    binding,
    *,
    trait_facts: dict | None = None,
    skipped: list | None = None,
    blocked_unimplemented: list | None = None,
    errors: list | None = None,
    traits: dict | None = None,
    evidence=None,
    selected_requirement: dict | None = None,
    requirement_sha256: str | None = None,
    selected_performance_basis: dict | None = None,
    performance_basis_sha256: str | None = None,
    source_preparation: str | None = None,
) -> list[dict]:
    """Return the profile's capsule entries with any ``sweeps:`` block expanded.

    A performance family's ``performance.gate.traits`` are evaluated against
    :func:`derive_profile`'s canonical tri-state records. Every trait must be
    exactly True. False and None both skip admission but remain distinct, with
    their evidence and evidence tier, in ``skipped``.

    A sweep is a cross-product over named axes plus a shared ``base``, producing
    exactly the flat entry dicts the per-capsule pipeline already consumes — so
    ``_write_capsule`` and every golden path are untouched by this feature.

    Two rules are enforced rather than documented:

    * **Every fitted axis needs at least two distinct points.** K is always
      fitted when present because one reduction depth cannot separate a tiled
      unit's rate from fixed overhead. Other axes opt in through ``fit_axes``;
      a one-point fit prices a parameter confidently and wrongly.
    * **Names must be unique across generated and hand-authored entries.** A
      collision would have one capsule overwrite another's directory, silently
      shrinking the corpus.

    Hand-written entries in ``capsules:`` are kept verbatim and come first, so a
    profile can mix a sweep with cases whose prose is worth writing by hand.
    """
    from merlin.development.testgen.profiles.sweep_axes import (
        _MAX_SWEEP_CAPSULES,
        AxisDerivationUnavailable,
        OracleSelectionError,
        _performance_facts,
        _resolve_derived_axis,
        evaluate_gate,
        resolve_encoding,
        resolve_extent,
        target_encodings,
    )

    entries = list(profile.get("capsules") or [])
    if evidence is not None:
        if evidence.target != binding.target:
            raise ValueError("selected evidence target differs from binding target")
        if trait_facts is None:
            trait_facts = evidence.performance_facts
    if source_preparation is not None:
        from merlin.development.testgen.components import source_performance

        source_performance.require_selection(source_preparation, source_components=binding.tile_dim is None)
        if not isinstance(trait_facts, dict):
            raise ValueError("source preparation requires explicit pending gate facts without target discovery")
    sweeps = (
        profile.get("sweeps") or []
        if source_preparation is not None
        else _scope_requirement_sweeps(
            profile.get("sweeps") or [],
            selected_requirement,
            requirement_sha256,
            skipped,
            blocked_unimplemented,
        )
    )
    sweeps = (
        sweeps
        if source_preparation is not None
        else _capture_shape_sweeps(
            sweeps,
            binding,
            selected_performance_basis,
            performance_basis_sha256,
            selected_requirement,
            requirement_sha256,
            skipped,
            evidence,
        )
    )
    if not sweeps:
        return entries
    # Compatibility for the public/holdout disjointness checker, which passes
    # this old keyword even for purely functional profiles. It is never used to
    # admit performance: the first performance sweep below rejects it.
    legacy_traits_supplied = traits is not None

    tile = int(getattr(binding, "tile_dim", 0) or 0)
    if tile < 1 and source_preparation is None:
        raise ValueError("sweeps need a tile edge; the binding reports none")

    seen = {e.get("name") for e in entries if isinstance(e, dict)}
    generated: list[dict] = []

    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError(f"sweep entry {sweep!r} is not a mapping")
        sweep_id = str(sweep.get("id") or "").strip()
        if not sweep_id:
            raise ValueError("every sweep needs an `id` (it prefixes the generated names)")
        if source_preparation is not None:
            refusal = source_performance.sweep_refusal(sweep)
            if refusal is not None:
                if blocked_unimplemented is None:
                    raise ValueError(refusal)
                blocked_unimplemented.append({"family": sweep_id, "status": "source_unavailable", "reason": refusal})
                continue
        withdrawal = (profile.get("_performance_withdrawals") or {}).get(sweep_id)
        if withdrawal is not None:
            if skipped is not None:
                skipped.append(
                    {
                        "family": sweep_id,
                        "sweep": sweep_id,
                        "status": "withdrawn",
                        "reason": withdrawal["reason"],
                        "decided_by": withdrawal["decided_by"],
                        "basis": "target recipe performance_withdrawals",
                        "fit_axes": list(sweep.get("fit_axes") or []),
                        "comparison_roles": _comparison_roles(sweep),
                    }
                )
            continue
        base = dict(sweep.get("base") or {})
        variant_performance = [
            i
            for i, variant in enumerate(sweep.get("variants") or [])
            if isinstance(variant, dict) and "performance" in variant
        ]
        if variant_performance:
            raise ValueError(
                f"sweep {sweep_id!r}: performance blocks belong on `base`, not variants "
                f"{variant_performance}; every member of a family shares one claim contract"
            )
        performance = base.get("performance")
        is_performance = base.get("cat") in {"perf", "_perf"} or performance is not None
        gate_decision = None
        if is_performance:
            if base.get("op") == "scope_chain":
                mlir_dtype = getattr(binding, "mlir_dtype", None)
                operand = mlir_dtype(binding.operand_dtype) if callable(mlir_dtype) else ""
                accumulator = mlir_dtype(binding.accum_dtype) if callable(mlir_dtype) else ""
                if not (
                    operand.startswith("i")
                    and operand[1:].isdigit()
                    and accumulator.startswith("i")
                    and accumulator[1:].isdigit()
                    and int(accumulator[1:]) > int(operand[1:])
                ):
                    if skipped is not None:
                        skipped.append(
                            {
                                "family": sweep_id,
                                "sweep": sweep_id,
                                "status": "skipped_inapplicable",
                                "reason": (
                                    "scope-chain builder requires signed integer operands and a wider accumulator"
                                ),
                                "operand_dtype": getattr(binding, "operand_dtype", None),
                                "accum_dtype": getattr(binding, "accum_dtype", None),
                                "requirement_basis": performance.get("requirement_basis"),
                            }
                        )
                    continue
            if legacy_traits_supplied:
                raise ValueError(
                    f"performance sweep {sweep_id}: legacy ad-hoc `traits` cannot gate performance; "
                    "pass canonical derive_profile(target) records through `trait_facts`"
                )
            performance = _validate_performance_block(performance, owner=f"performance sweep {sweep_id}")
            if performance["family"] != sweep_id:
                raise ValueError(f"performance sweep {sweep_id}: performance.family must equal the sweep id")
            facts = trait_facts
            if facts is None:
                target = str(getattr(binding, "target", "") or "")
                if not target:
                    raise ValueError(
                        f"performance sweep {sweep_id}: no trait facts supplied and binding.target is absent"
                    )
                facts = _performance_facts(target)
            ok, decision = evaluate_gate(performance["gate"], facts)
            gate_decision = decision
            if not ok and source_preparation is None:
                if skipped is not None:
                    skipped.append(
                        {
                            "family": sweep_id,
                            "sweep": sweep_id,
                            "status": "skipped_inapplicable",
                            "gate": decision,
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue
            emitter_status = str(performance["emitter"]["status"])
            if emitter_status != "existing":
                if blocked_unimplemented is not None:
                    blocked_unimplemented.append(
                        {
                            "family": sweep_id,
                            "status": "blocked_unimplemented",
                            "reason": f"declared emitter {emitter_status!r} is not implemented",
                            "emitter": copy.deepcopy(performance["emitter"]),
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue
        if sweep.get("requires_form_scope") is not None or sweep.get("requires_capacity_scope") is not None:
            # Form-perf and capacity members are derived (form classes / RTL facts), never from axes.
            from merlin.development.testgen.performance.capacity_families import scope_members

            for entry in scope_members(
                sweep,
                binding=binding,
                evidence=evidence,
                requirement=(selected_requirement, requirement_sha256),
                entries=entries,
                skipped=skipped,
                blocked=blocked_unimplemented,
            ):
                if entry["name"] in seen:
                    raise ValueError(f"sweep {sweep_id!r} generated duplicate capsule name {entry['name']!r}")
                seen.add(entry["name"])
                try:
                    if "_performance_oracles" in profile:
                        _materialize_performance_entry(entry, binding, oracle_selection=profile["_performance_oracles"])
                    else:
                        _materialize_performance_entry(entry, binding)
                except OracleSelectionError:
                    raise  # Selection is a run prerequisite, not a per-member generation error.
                except Exception as exc:  # noqa: BLE001 - persisted as a generation error
                    if errors is None:
                        raise
                    errors.append(
                        {
                            "family": sweep_id,
                            "member": entry["name"],
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "detail": str(exc)[:500],
                        }
                    )
                    continue
                generated.append(entry)
            continue
        axes = sweep.get("axes") or {}
        if not isinstance(axes, dict) or not axes:
            raise ValueError(f"sweep {sweep_id!r} declares no axes")
        encodings = (
            []
            if source_preparation is not None
            else target_encodings(
                str(getattr(binding, "target", "") or ""),
                **({"contract": evidence.contract} if evidence is not None else {}),
            )
        )

        # Resolve each axis to concrete extents, preserving declaration order so
        # the generated corpus is reproducible.
        resolved: dict[str, list[int]] = {}
        derived_axes = {a: t for a, t in axes.items() if isinstance(t, dict)}
        for axis, tokens in axes.items():
            if axis in derived_axes:
                continue
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"sweep {sweep_id!r} axis {axis!r} must be a non-empty list")
            resolved[axis] = [resolve_extent(t, tile) for t in tokens]
        # DERIVED axes come second, because a derivation reads the literal axes it is banded against
        # (a residency band is a property of one parallel shape). The order is a dependency, not a
        # convenience: resolving them together would make the M/N a derivation sees depend on dict
        # iteration order.
        axis_labels: dict[str, dict] = {}
        axis_records: dict[str, dict] = {}
        unhostable = None
        for axis, spec in derived_axes.items():
            try:
                values, labels, record = _resolve_derived_axis(
                    spec,
                    owner=f"sweep {sweep_id!r}",
                    axis=axis,
                    target=str(getattr(binding, "target", "") or ""),
                    tile=tile,
                    dtype=getattr(binding, "operand_dtype", None),
                    fixed=resolved,
                    evidence=evidence,
                )
            except AxisDerivationUnavailable as exc:
                unhostable = {"axis": axis, "derive": str(spec.get("derive")), "detail": str(exc)}
                if exc.record is not None:
                    unhostable["record"] = exc.record
                break
            resolved[axis] = values
            axis_labels[axis] = labels
            axis_records[axis] = record
        if unhostable is not None:
            if skipped is not None:
                skipped.append(
                    {
                        "family": sweep_id,
                        "sweep": sweep_id,
                        "status": "skipped_inapplicable",
                        "reason": unhostable["detail"],
                        "axis_derivation": unhostable,
                        "gate": gate_decision,
                        "fit_axes": list(sweep.get("fit_axes") or []),
                        "comparison_roles": _comparison_roles(sweep),
                    }
                )
            continue

        declared_fit_axes = sweep.get("fit_axes") or []
        if not isinstance(declared_fit_axes, list) or any(
            not isinstance(axis, str) or not axis for axis in declared_fit_axes
        ):
            raise ValueError(f"sweep {sweep_id!r}: `fit_axes` must be a list of axis names")
        fitted_axes = list(dict.fromkeys((["K"] if "K" in resolved else []) + declared_fit_axes))
        unknown_fit_axes = [axis for axis in fitted_axes if axis not in resolved]
        if unknown_fit_axes:
            raise ValueError(
                f"sweep {sweep_id!r} fits undeclared axis/axes {unknown_fit_axes}; declared axes are {sorted(resolved)}"
            )
        for axis in fitted_axes:
            if len(set(resolved[axis])) < 2:
                detail = (
                    "a tiled unit needs at least TWO distinct K points, because one cannot "
                    "separate the rate from the per-tile overhead"
                    if axis == "K"
                    else "every fitted parameter needs at least TWO distinct points"
                )
                raise ValueError(f"sweep {sweep_id!r} fits {axis} over {resolved[axis]} — {detail}")

        combos = _cross_product(resolved)
        if len(combos) > _MAX_SWEEP_CAPSULES:
            raise ValueError(
                f"sweep {sweep_id!r} would generate {len(combos)} capsules (cap {_MAX_SWEEP_CAPSULES}); "
                f"narrow the axes rather than letting it be truncated"
            )

        # A sweep may also cross the extents with a list of NON-EXTENT overrides. The motivating case is
        # a fusion comparison: three capsules that must sit at the IDENTICAL shape and differ only in
        # which op they ask for, so that `cycles(fused)` and `cycles(part) + cycles(part)` are about the
        # same work. Expressing that as an axis is impossible -- an axis value is an extent, resolved
        # against the tile -- and expressing it as three hand-authored entries would hardcode the shape
        # in three places, where the whole point is that the three shapes are the same one.
        variants = sweep.get("variants") or [{}]
        if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
            raise ValueError(f"sweep {sweep_id!r}: `variants` must be a list of mappings")

        # A comparison group whose members cannot be compared is a declaration with no content. The
        # field was carried on four shipped capsules for a year while every one of them sat alone in
        # its group, so nothing could ever do the arithmetic it exists for.
        _groups: dict[str, int] = {}
        for v in variants:
            g = v.get("comparison_group")
            gname = g.get("name") if isinstance(g, dict) else g
            if gname:
                _groups[str(gname)] = _groups.get(str(gname), 0) + 1
        _lonely = sorted(g for g, n in _groups.items() if n < 2)
        if _lonely:
            raise ValueError(
                f"sweep {sweep_id!r} declares comparison group(s) {_lonely} with a single member; a "
                "group of one cannot be compared to anything, so declare the other members or drop "
                "the group"
            )

        # A group whose standalone part the declarations refuse cannot be compared on this target.
        if evidence is not None and _groups:
            from merlin.development.testgen.profiles.screens import refused_part

            refusal = refused_part(variants, base, software_spec=evidence.software_spec, binding=binding)
            if refusal is not None:
                if skipped is not None:
                    skipped.append(
                        {
                            "family": sweep_id,
                            "sweep": sweep_id,
                            "status": "skipped_inapplicable",
                            "reason": (
                                f"comparison part {refusal['member']!r} is refused standalone by the selected "
                                f"declarations: {refusal['reason']}"
                            ),
                            "refused_part": refusal,
                            "gate": gate_decision,
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue

        template = str(sweep.get("name") or "{id}_{i:02d}")
        index = 0
        sweep_generated: list[dict] = []
        for combo in combos:
            for variant in variants:
                entry = copy.deepcopy(base)
                entry.update(combo)
                rendered = _render_variant(variant, combo)
                # An `operand_dtype: encoding<N>` variant names an encoding POSITIONALLY, so the shared
                # template never writes a dtype down. Resolved here, where the target is known, and
                # stashed under a private key so `_materialize_performance_entry` can tell "this member
                # chose its encoding" from "this member inherited the corpus binding".
                enc_token = rendered.get("operand_dtype")
                if isinstance(enc_token, str) and enc_token.startswith("encoding"):
                    rendered["operand_dtype"] = resolve_encoding(enc_token, encodings)
                    rendered["_encoding_variant"] = rendered["operand_dtype"]
                entry.update(rendered)
                # A derived axis contributes a LABEL as well as a value, so a member can be named by
                # the band it was derived for (`{K_label}` -> "spills") rather than only by an extent
                # whose meaning a reader would have to recompute against the store.
                name_ns = {**combo, **variant}
                for _axis, _labels in axis_labels.items():
                    name_ns[f"{_axis}_label"] = _labels.get(combo.get(_axis), "unknown")
                entry["name"] = template.format(id=sweep_id, i=index, **name_ns)
                index += 1
                entry.setdefault("source_role", "derived_sweep")
                # Provenance survives generation: say which sweep and which point.
                reference = sweep.get("source_reference") or f"generated by sweep {sweep_id!r}"
                axis_note = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
                if variant:
                    axis_note += "; " + ", ".join(
                        f"{k}={v}" for k, v in sorted(variant.items()) if not isinstance(v, dict)
                    )
                if axis_records:
                    entry["_derived_axes"] = {
                        a: {
                            "derive": str(axes[a].get("derive")),
                            "value": combo.get(a),
                            "label": axis_labels[a].get(combo.get(a)),
                            "derivation": r,
                        }
                        for a, r in axis_records.items()
                    }
                    axis_note += "; " + ", ".join(
                        f"{a}={combo.get(a)} is {axis_labels[a].get(combo.get(a))}" for a in sorted(axis_records)
                    )
                entry["source_reference"] = f"{reference} (tile={tile}; {axis_note})"
                if entry["name"] in seen:
                    raise ValueError(f"sweep {sweep_id!r} generated duplicate capsule name {entry['name']!r}")
                seen.add(entry["name"])
                if is_performance:
                    try:
                        if source_preparation is not None:
                            entry = source_performance.materialize(
                                entry, binding, point=combo, gate=gate_decision, selection=source_preparation
                            )
                        elif "_performance_oracles" in profile:
                            _materialize_performance_entry(
                                entry, binding, oracle_selection=profile["_performance_oracles"]
                            )
                        else:
                            _materialize_performance_entry(entry, binding)
                    except OracleSelectionError:
                        raise  # Refuse before any later capsule capture work.
                    except Exception as exc:  # noqa: BLE001 - persisted as a generation error
                        if errors is None:
                            raise
                        errors.append(
                            {
                                "family": sweep_id,
                                "member": entry["name"],
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "detail": str(exc)[:500],
                            }
                        )
                        continue
                sweep_generated.append(entry)

        if evidence is not None and _groups and sweep_generated:
            from merlin.development.testgen.profiles.screens import refused_emitted_part

            refusal = refused_emitted_part(sweep_generated, binding=binding, evidence=evidence)
            if refusal is not None:
                if skipped is not None:
                    skipped.append(
                        {
                            "family": sweep_id,
                            "sweep": sweep_id,
                            "status": "skipped_inapplicable",
                            "reason": (
                                f"comparison part {refusal['member']!r} is refused by its builder-emitted "
                                f"typed interface: {refusal['reason']}"
                            ),
                            "refused_part": refusal,
                            "gate": gate_decision,
                            "fit_axes": list(sweep.get("fit_axes") or []),
                            "comparison_roles": _comparison_roles(sweep),
                        }
                    )
                continue
        generated.extend(sweep_generated)

    # A GROUP REDUCED TO ONE MEMBER IS NOT A COMPARISON. The declaration-time check above refuses a
    # group AUTHORED with a single member, but a member can also disappear afterwards: materialization
    # is allowed to fail per member and record an error, and the survivors were shipped anyway. A lone
    # member then carries a `comparison_group` and a DIFFERENTIAL claim with nothing to difference
    # against -- which is the same silent vacuity the single-member check exists to prevent, arriving by
    # a different route. Measured while opening the encoding family: one target's second encoding could
    # not resolve its accumulator, and the first shipped alone.
    if generated:
        members: dict[str, int] = {}
        for e in generated:
            g = e.get("comparison_group")
            name = g.get("name") if isinstance(g, dict) else g
            if name:
                members[str(name)] = members.get(str(name), 0) + 1
        lonely = {g for g, n in members.items() if n < 2}
        if lonely:
            kept = []
            for e in generated:
                g = e.get("comparison_group")
                name = str(g.get("name") if isinstance(g, dict) else g or "")
                if name in lonely:
                    fam = (e.get("performance") or {}).get("family")
                    if errors is not None:
                        errors.append(
                            {
                                "family": fam,
                                "member": e.get("name", "?"),
                                "status": "error",
                                "error_type": "IncompleteComparisonGroup",
                                "detail": (
                                    f"comparison group {name!r} kept only this member; a "
                                    f"differential needs both, so it is dropped rather than "
                                    f"shipped as a comparison against nothing"
                                ),
                            }
                        )
                    continue
                kept.append(e)
            generated = kept

    return entries + generated
