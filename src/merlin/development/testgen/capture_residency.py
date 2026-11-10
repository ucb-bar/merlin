"""Capture-bound residency constraint selection for Phase 0 performance probes."""

from __future__ import annotations

import copy


def _bound_capture_inventory(
    basis: dict,
    basis_sha256: str | None,
    requirement: dict | None,
    requirement_sha256: str | None,
    target: str,
) -> str:
    """Return the selected inventory digest only when both frozen views agree."""
    if (
        basis.get("schema") != "merlin.phase0.performance_basis.v1"
        or basis.get("target") != target
        or not isinstance(basis_sha256, str)
        or len(basis_sha256) != 64
        or any(character not in "0123456789abcdef" for character in basis_sha256)
        or not isinstance(requirement_sha256, str)
        or len(requirement_sha256) != 64
        or any(character not in "0123456789abcdef" for character in requirement_sha256)
    ):
        raise ValueError("capture-shape selection lacks a bound basis, target, or requirement")
    requirement_demands = (requirement or {}).get("application_demands") or {}
    selected_inventory = basis.get("selected_inventory") or {}
    inventory_digest = requirement_demands.get("full_inventory_sha256")
    basis_inventory_digest = (basis.get("sources") or {}).get("selected_inventory_sha256")
    if (
        (requirement or {}).get("target") != target
        or not isinstance(inventory_digest, str)
        or len(inventory_digest) != 64
        or inventory_digest != basis_inventory_digest
        or inventory_digest != selected_inventory.get("content_sha256")
        or selected_inventory.get("declared_roster_matches") is not True
    ):
        raise ValueError("capture-shape basis and selected requirement name different application inventories")
    demand_apps = requirement_demands.get("applications") or {}
    basis_apps = basis.get("applications") or {}
    if (
        not isinstance(demand_apps, dict)
        or set(demand_apps) != set(basis_apps)
        or any(
            demand_apps[label].get("capture_sha256") != basis_apps[label].get("capture_sha256") for label in demand_apps
        )
    ):
        raise ValueError("capture-shape basis and requirement disagree on capture roster or bytes")
    from merlin.targetgen import claim_models

    if any(claim_models.is_claim_bundle(label) for label in basis_apps):
        raise ValueError("capture-shape basis contains a held-out validation workload")
    return inventory_digest


def _measurable_residency_boundaries(record: dict, tile: int) -> list[dict]:
    """Prove emitted endpoint points straddle one RTL-sized band boundary."""
    from merlin.targetgen import memory_regime as MR

    pairs = []
    capacity = record.get("capacity_rows")
    for lower, upper, threshold in (
        (MR.FITS_DOUBLE, MR.FITS_SINGLE, "twice_live_rows_exceed_capacity"),
        (MR.FITS_SINGLE, MR.SPILLS, "live_rows_exceed_capacity"),
    ):
        a, b = record["by_regime"].get(lower) or {}, record["by_regime"].get(upper) or {}
        a_points, b_points = a.get("points") or [], b.get("points") or []
        if len(a_points) < 3 or len(b_points) < 3:
            continue
        before, after = a_points[-1], b_points[0]
        if (
            before["K"] + tile != after["K"]
            or before["K_tiles"] + 1 != after["K_tiles"]
            or a.get("band_tile_multiples", [None, None])[1] != before["K_tiles"]
            or b.get("band_tile_multiples", [None, None])[0] != after["K_tiles"]
        ):
            continue
        crosses = (
            2 * before["rows"] <= capacity < 2 * after["rows"]
            if lower == MR.FITS_DOUBLE
            else before["rows"] <= capacity < after["rows"]
        )
        if crosses:
            pairs.append(
                {
                    "from_regime": lower,
                    "to_regime": upper,
                    "predicate": threshold,
                    "below": {"K": before["K"], "rows": before["rows"]},
                    "above": {"K": after["K"], "rows": after["rows"]},
                    "capacity_rows": capacity,
                    "step": "one derived tile in K; fixed exact captured M/N",
                }
            )
    return pairs


def _capture_residency_sweep(
    sweep: dict,
    binding,
    basis: dict | None,
    basis_sha256: str | None,
    requirement: dict | None,
    requirement_sha256: str | None,
    evidence,
    skipped: list | None,
) -> list[dict]:
    """Anchor one existing residency-law cohort to a selected development shape.

    A PR analyzer requires one fixed M/N cohort. Rank independent, exact static
    contractions by observed static MAC count (a deterministic tie-break, not
    an execution frequency or performance weight) and take the first shape on
    which the selected RTL operand store proves one adjacent two-band crossing.
    This is a shape candidate, never a source-body or placement witness.
    """
    family = str(sweep.get("id") or "")
    pattern = sweep.get("capture_shape_pattern") or {}
    if (
        pattern.get("kind") != "observed_mn_for_memory_regime"
        or type(pattern.get("max_candidates")) is not int
        or pattern["max_candidates"] < 1
        or (sweep.get("base") or {}).get("op") != "matmul"
        or (sweep.get("axes") or {}).get("K", {}).get("derive") != "memory_regime_reduction_depth"
    ):
        raise ValueError(f"performance sweep {family}: unsupported captured residency pattern")
    if basis is None:
        # Legacy and direct template inspection retain the generic law. A
        # verified, capture-bound run always supplies its frozen basis.
        if skipped is not None:
            skipped.append(
                {
                    "family": f"{family}.capture_shape",
                    "status": "skipped_inapplicable",
                    "reason": "no frozen selected-capture performance basis; generic machine law only",
                }
            )
        retained = copy.deepcopy(sweep)
        del retained["capture_shape_pattern"]
        return [retained]
    inventory_digest = _bound_capture_inventory(
        basis, basis_sha256, requirement, requirement_sha256, str(getattr(binding, "target", ""))
    )
    if evidence is None:
        if skipped is not None:
            skipped.append(
                {
                    "family": family,
                    "status": "skipped_inapplicable",
                    "reason": "selected RTL facts are unavailable for a capture-anchored residency boundary",
                    "performance_basis_sha256": basis_sha256,
                }
            )
        return []
    raw_rtl_sha256 = (basis.get("sources") or {}).get("raw_rtl_facts_sha256")
    if (
        not isinstance(raw_rtl_sha256, str)
        or len(raw_rtl_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_rtl_sha256)
        or getattr(evidence, "raw_facts_sha256", None) != raw_rtl_sha256
    ):
        raise ValueError("captured residency selection differs from the frozen selected RTL facts")
    from merlin.common import quant_formats
    from merlin.targetgen import address_space as AS
    from merlin.targetgen import memory_regime as MR

    try:
        wanted_dtype = quant_formats.get(getattr(binding, "operand_dtype", None)).name
        space = AS.derive_address_space(binding.target, facts=evidence.refreshed_facts)
        selected = AS.operand_store(space, dtype=binding.operand_dtype)
        store, capacity = selected.store, selected.capacity_rows(binding.operand_dtype)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        store, capacity = None, None
        store_reason = f"selected operand-store derivation failed: {type(exc).__name__}: {exc}"
    else:
        store_reason = "selected RTL facts do not establish a sized operand store"
    if store is None or not capacity:
        if skipped is not None:
            skipped.append(
                {
                    "family": family,
                    "status": "skipped_inapplicable",
                    "reason": store_reason,
                    "performance_basis_sha256": basis_sha256,
                }
            )
        return []

    groups: dict[tuple[int, int], list[dict]] = {}
    remainder = []
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
                "source_operand_format": row.get("operand_format"),
            }
            if any(type(axis) is not int or axis < 1 for axis in (m, k, n)) or type(total) is not int or total < 1:
                reason = "no recognized exact static single-MAC shape"
            else:
                try:
                    source_dtype = quant_formats.get(row.get("operand_format")).name
                except (KeyError, TypeError):
                    source_dtype = None
                reason = (
                    None if source_dtype == wanted_dtype else "captured operand format differs from selected binding"
                )
            if reason is None:
                groups.setdefault((m, n), []).append(witness)
            else:
                remainder.append({**witness, "reason": reason})
    ranked = sorted(groups.items(), key=lambda item: (-sum(w["known_static_macs"] for w in item[1]), item[0]))
    selected_shape = None
    selected_record = None
    for (m, n), witnesses in ranked[: pattern["max_candidates"]]:
        record = MR.reduction_depth_regimes(
            binding.target,
            (MR.FITS_DOUBLE, MR.FITS_SINGLE, MR.SPILLS),
            tile_dim=int(binding.tile_dim),
            dtype=binding.operand_dtype,
            m_extent=m,
            n_extent=n,
            points_per_regime=int(sweep["axes"]["K"].get("points_per_regime", 3)),
            spills_max_fraction=float(sweep["axes"]["K"].get("spills_max_fraction_of_capacity", 2.0)),
            store=store,
            capacity=capacity,
        )
        boundary_pairs = _measurable_residency_boundaries(record, int(binding.tile_dim))
        if boundary_pairs:
            selected_shape, selected_record = (m, n, witnesses), record
            break
        remainder.extend(
            {**w, "reason": f"no adjacent three-point RTL-derived residency boundary at M={m}, N={n}"}
            for w in witnesses
        )
    for _, witnesses in ranked[pattern["max_candidates"] :]:
        remainder.extend({**w, "reason": f"deferred by max_candidates={pattern['max_candidates']}"} for w in witnesses)
    if skipped is not None and remainder:
        skipped.append(
            {
                "family": f"{family}.capture_shape.remainder",
                "status": "skipped_inapplicable",
                "reason": "selected captured contractions not represented by this one fixed-shape residency cohort",
                "performance_basis_sha256": basis_sha256,
                "remainder": sorted(remainder, key=lambda row: (row["application"], row["signature_index"])),
            }
        )
    if selected_shape is None:
        if skipped is not None:
            skipped.append(
                {
                    "family": family,
                    "status": "skipped_inapplicable",
                    "reason": "no selected development shape spans an adjacent measurable RTL-derived residency boundary",
                    "performance_basis_sha256": basis_sha256,
                }
            )
        return []
    m, n, witnesses = selected_shape
    numeric_obligation = {"status": "not_applicable", "reason": "selected datapath is not integer"}
    if getattr(binding, "integer", False):
        from merlin.targetgen.operation_numerics import integer_split_k_limit

        software = getattr(evidence, "software_spec", None) or {}
        semantics = software.get("numerical_semantics") or {}
        max_k = max(point["K"] for band in selected_record["by_regime"].values() for point in band["points"])
        try:
            limit = integer_split_k_limit(
                {"numerical_semantics": semantics}, tile_dim=int(binding.tile_dim), full_k=max_k
            )
        except ValueError as exc:
            limit = None
            numeric_reason = str(exc)
        else:
            numeric_reason = (
                "selected signed internal MAC/aggregation contract is incomplete" if limit is None else None
            )
        if limit is None or limit["tile_k"] < int(binding.tile_dim):
            if skipped is not None:
                skipped.append(
                    {
                        "family": family,
                        "status": "skipped_inapplicable",
                        "reason": (
                            "capture-anchored residency depths lack a safe selected integer split-K route: "
                            f"{numeric_reason}"
                        ),
                        "performance_basis_sha256": basis_sha256,
                        "observed_mn": {"M": m, "N": n},
                    }
                )
            return []
        needs_split = max_k > limit["max_padded_k"]
        numeric_obligation = {
            "status": "requires_split_k" if needs_split else "single_primitive_safe",
            "selected_software_spec_sha256": (basis.get("sources") or {}).get("selected_software_spec_sha256"),
            "selected_software_review": software.get("status"),
            "maximum_generated_k": max_k,
            "max_unpartitioned_k": limit["max_padded_k"],
            "max_tile_aligned_primitive_k": limit["tile_k"],
            "full_result_bound": limit["full_result_bound"],
            "mac_result_bits": limit["mac_result_bits"],
            "signed_operand_bits": limit["signed_operand_bits"],
            "aggregation": limit["aggregation"],
            "qualification": "selected arithmetic declaration; review and execution require separate evidence",
            "verification": (
                "Phase 0 states the worst-case requirement only; Phase 1 must prove every bounded primitive "
                "dispatch and exact i32 aggregation before Phase 2 may treat a measurement as generalizable"
            ),
        }
    derived = copy.deepcopy(sweep)
    del derived["capture_shape_pattern"]
    derived["axes"]["M"], derived["axes"]["N"] = [m], [n]
    derived["base"]["performance"]["requirement_basis"] = {
        "axis": "coverage.performance-basis.residency_shape",
        "sha256": requirement_sha256,
        "performance_basis_sha256": basis_sha256,
        "selected_inventory_sha256": inventory_digest,
        "raw_rtl_facts_sha256": raw_rtl_sha256,
        "selected_hardware_spec_sha256": (basis.get("sources") or {}).get("selected_hardware_spec_sha256"),
        "selected_store": {"name": store.name, "capacity_rows": int(capacity)},
        "observed_mn": {"M": m, "N": n},
        "observed_k": sorted({w["shape"]["K"] for w in witnesses}),
        "selection_order": "descending captured static MAC count, then ascending exact M/N; not a runtime weight",
        "reachable_bands": sorted(
            name for name, band in selected_record["by_regime"].items() if len(band["points"]) >= 3
        ),
        "boundary_pairs": _measurable_residency_boundaries(selected_record, int(binding.tile_dim)),
        "numeric_obligation": numeric_obligation,
        "source_witnesses": sorted(witnesses, key=lambda row: (row["application"], row["signature_index"])),
        "source_match": "capture_shape_candidate",
        "qualification": (
            "selected development capture fixes exact M/N; RTL-derived operand capacity selects K band "
            "endpoints and spread points; no source-body equivalence, compiler placement, usable "
            "working-set proof, measured timing or speedup is asserted"
        ),
    }
    derived["source_reference"] = (
        f"selected frozen performance basis {basis_sha256}: observed M={m}, N={n}; "
        "K residency boundaries derived from selected RTL operand store; shape-only diagnostic"
    )
    return [derived]
