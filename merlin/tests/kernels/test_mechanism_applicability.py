"""Mechanism differences, catalog route mapping and library admission on synthetic occurrence records."""

import pytest

from merlin.compiler.schedule.api_catalog import CAPABILITY_EFFECTS
from merlin.compiler.schedule.api_catalog import capability_effects
from merlin.compiler.schedule.api_catalog import scheduling_capabilities
from merlin.kernels.mechanism_applicability import applicability, mechanism_differences

# Neutral synthetic geometry: a 4-wide block, 64 operand rows and 32 accumulator rows.
GEOMETRY = dict(block=4, operand_rows=64, accumulator_rows=32, operand_bank_rows=16, operand_store="left_sram",
                accumulator_store="sums", operand_row_bytes=4, accumulator_row_bytes=16, operand_element_bits=8,
                accumulator_element_bits=32)


def _fact(value, status="observed"):
    return dict(status=status, value=value, evidence=[], scope={})


def _record(name, *, roles, loops=1, tiles=None, broadcast=None, loop_sites=0):
    facts = dict(
        transfer_tiles=_fact(dict(by_role=dict(operand_load=tiles or []))) if tiles else _fact(None, "unknown"),
        broadcast_loads=_fact(dict(broadcast_movement_sites=broadcast or {}, nonzero_by_state={"0": 1},
                                   stride_zero_by_state={"1": 1} if broadcast else {})),
        hardware_loop=_fact(dict(loop_descriptor_sites=loop_sites)),
        operand_retention=_fact(None, "unknown"),
    )
    return dict(stable_id=name, schedule_model=dict(role_counts=roles), cfg=dict(natural_loops=loops),
                derived_effects=dict(evidence_id=name, facts=facts, inputs=dict(structure_sha256="s")))


REFERENCE = _record("ref", roles=dict(accumulate=16, operand_load=4, weight_load=16, config=2, readout=4),
                    tiles=[dict(rows=4, cols=16, sites=4)], broadcast={"LOAD": 2}, loop_sites=2)
CANDIDATE = _record("cand", roles=dict(accumulate=2, operand_load=4, weight_load=2, config=4, readout=2, sync=1),
                    tiles=[dict(rows=4, cols=4, sites=4)])


def _row(report, effect):
    return next(row for row in report["differences"] if row["effect"] == effect)


def test_every_declared_effect_names_a_catalog_capability():
    names = {row["name"] for row in scheduling_capabilities(24)["capabilities"]}
    assert set(CAPABILITY_EFFECTS) <= names
    assert [row["name"] for row in capability_effects(24)] == [n for n in CAPABILITY_EFFECTS if n in names]
    for effect in CAPABILITY_EFFECTS.values():
        assert set(effect["changes"].values()) <= {"reduces", "increases"}


def test_differences_carry_status_evidence_and_direction():
    rows = mechanism_differences(REFERENCE, CANDIDATE)
    by = {row["effect"]: row for row in rows}
    reload = by["operand_load_per_compute"]
    assert reload["difference"] == "differs" and reload["needed_change"] == "reduces"
    assert reload["status"] == "inferred" and reload["limits"]  # loops: a static ratio, not a dynamic rate
    assert by["weight_load_per_compute"]["difference"] == "same"
    assert by["operand_transfer_extent"]["needed_change"] == "increases"
    assert by["broadcast_load_sites"]["needed_change"] == "increases"
    assert by["sync_sites"]["needed_change"] == "reduces"
    assert by["operand_retention"]["difference"] == "unknown"
    assert all(row["evidence"] and row["status"] in ("observed", "observed_partial", "inferred", "unknown")
               for row in rows)
    straight = mechanism_differences(dict(REFERENCE, cfg=dict(natural_loops=0)),
                                     dict(CANDIDATE, cfg=dict(natural_loops=0)))
    assert next(r for r in straight if r["effect"] == "operand_load_per_compute")["status"] == "observed"


def test_routes_follow_catalog_order_and_carry_requirements_without_ranking():
    report = applicability(REFERENCE, CANDIDATE, shape=dict(m=8, k=8, n=8), geometry=GEOMETRY)
    routes = _row(report, "operand_load_per_compute")["routes"]
    names = [route["capability"] for route in routes]
    order = [row["name"] for row in capability_effects(24)]
    assert names == sorted(names, key=order.index)
    assert {"segmented_k_retained_operand_pipeline", "capacity_derived_m_panels_with_rhs_reuse"} <= set(names)
    for route in routes:
        assert route["usage"] and route["composition_and_scheduling_requirements"] and route["symbols"]
        assert not {"rank", "score", "recommended", "best"} & set(route)
    assert report["performance"] == report["semantic_equivalence"] == "unknown"


def test_admission_satisfied_violated_and_unknown():
    satisfied = applicability(REFERENCE, CANDIDATE, shape=dict(m=8, k=8, n=8), geometry=GEOMETRY)
    m_panel = next(r for r in _row(satisfied, "operand_load_per_compute")["routes"]
                   if r["capability"] == "capacity_derived_m_panels_with_rhs_reuse")["admission"]
    assert m_panel["status"] == "satisfied" and m_panel["satisfied_n_tiles"]
    single_m = applicability(REFERENCE, CANDIDATE, shape=dict(m=4, k=8, n=8), geometry=GEOMETRY)
    violated = next(r for r in _row(single_m, "operand_load_per_compute")["routes"]
                    if r["capability"] == "capacity_derived_m_panels_with_rhs_reuse")["admission"]
    assert violated["status"] == "violated" and "multiple_m_tiles" in violated["violated_bounds"]
    no_shape = applicability(REFERENCE, CANDIDATE, geometry=GEOMETRY)
    unknown = next(r for r in _row(no_shape, "operand_load_per_compute")["routes"]
                   if r["capability"] == "segmented_k_retained_operand_pipeline")["admission"]
    assert unknown["status"] == "unknown" and "candidate shape" in unknown["missing"][0]
    no_facts = applicability(REFERENCE, CANDIDATE, shape=dict(m=8, k=8, n=8), geometry_reason="no facts")
    assert next(r for r in _row(no_facts, "operand_load_per_compute")["routes"]
                if r["capability"] == "segmented_k_retained_operand_pipeline")["admission"]["missing"] == [
        "selected RTL geometry"]
    law_only = next(r for r in _row(satisfied, "operand_load_per_compute")["routes"]
                    if r["capability"] == "source_axis_loop_scheduling")["admission"]
    assert law_only["status"] == "not_declared"


@pytest.mark.parametrize("channels, status", [(4, "satisfied"), (16, "violated")])
def test_convolution_rhs_residency_capacity(channels, status):
    report = applicability(REFERENCE, CANDIDATE, shape=dict(kh=3, kw=3, c=channels, oc=4), geometry=GEOMETRY)
    route = next(r for r in _row(report, "operand_load_per_compute")["routes"]
                 if r["capability"] == "capacity_phased_convolution_rhs_residency")
    assert route["family_match"] is True and route["admission"]["status"] == status
    dense = next(r for r in _row(report, "operand_load_per_compute")["routes"]
                 if r["capability"] == "capacity_derived_m_panels_with_rhs_reuse")
    assert dense["family_match"] is False and dense["admission"]["status"] == "not_applicable"


def test_missing_capability_names_the_unaddressed_effect():
    report = applicability(REFERENCE, CANDIDATE, shape=dict(m=8, k=8, n=8), geometry=GEOMETRY)
    loop = _row(report, "loop_descriptor_sites")
    assert loop["routes"] == [] and loop["missing_capability"]["effect"] == "loop_descriptor_sites"
    assert loop["missing_capability"]["needed_change"] == "increases"
    assert loop["missing_capability"] in report["missing_capabilities"]
    # A route exists for broadcast loads, but only for dense sources: a convolution reports the gap.
    conv = applicability(REFERENCE, CANDIDATE, shape=dict(kh=1, kw=1, c=4, oc=4), geometry=GEOMETRY)
    assert _row(conv, "broadcast_load_sites")["missing_capability"]["family"] == "convolution"


# --- facts of a proved dynamic command stream (contract shape, synthetic values) -----------------

def _stream(width, buffering, overlap):
    return dict(role_transfer_width=_fact(dict(by_role=width, preload=[dict(rows=4, cols=4, count=64)])),
                operand_buffering=_fact(dict(by_role=buffering)), issue_overlap=_fact(overlap))


def _with_stream(record, facts):
    effects = dict(record["derived_effects"], facts=dict(record["derived_effects"]["facts"], **facts))
    return dict(record, derived_effects=effects)


WIDE = [dict(rows=4, cols=16, count=8, elements=64)]
NARROW = [dict(rows=4, cols=4, count=32, elements=16)]
DOUBLE = dict(windows=[[0, 32], [32, 64]], disjoint=2, alternating=True)
OVERLAPPED = dict(loads_with_compute_before_barrier=dict(weight=dict(count=48, total=64),
                                                         input=dict(count=0, total=8)),
                  compute_run_median=16, load_run_median=4, fences=4, computes_per_fence=64,
                  output_group_extent=dict(rows=4, cols=64), loads_per_compute_run=0.25,
                  compute_run_between_load_bands=16)
SERIAL = dict(OVERLAPPED, loads_with_compute_before_barrier=dict(weight=dict(count=0, total=64),
                                                                 input=dict(count=0, total=8)),
              compute_run_median=64, load_run_median=16, compute_run_between_load_bands=64)
STREAM_REF = _with_stream(REFERENCE, _stream(dict(input=WIDE, weight=WIDE), dict(weight=DOUBLE, input=DOUBLE),
                                             OVERLAPPED))


def _rows(report, effect):
    return {row.get("role"): row for row in report["differences"] if row["effect"] == effect}


def test_stream_width_is_compared_per_role_and_partial_recipe_lists_what_differs():
    candidate = _with_stream(CANDIDATE, _stream(dict(input=NARROW, weight=WIDE), dict(weight=DOUBLE, input=DOUBLE),
                                                SERIAL))
    report = applicability(STREAM_REF, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    width = _rows(report, "role_transfer_width")
    assert width["weight"]["direction"] == "candidate_matches" and width["weight"]["difference"] == "same"
    assert width["input"]["direction"] == "candidate_differs" and width["input"]["needed_change"] == "increases"
    assert width["input"]["reference"]["elements_per_transfer"] == 64
    assert width["input"]["candidate"]["elements_per_transfer"] == 16
    assert "preload" in width["input"] and "preload" not in width  # kept separate, never a role
    assert [r["capability"] for r in width["input"]["routes"]] == ["exact_final_wide_loads"]
    overlap = _rows(report, "issue_overlap")
    assert overlap["weight"]["direction"] == "candidate_differs" and overlap["weight"]["needed_change"] == "increases"
    # Equal per-role fractions match; the run-structure difference is the pooled and interleave rows' fact.
    assert overlap["input"]["direction"] == "candidate_matches" and overlap["input"]["needed_change"] is None
    prefetch = _rows(report, "prefetch_interleave")["all"]
    assert prefetch["direction"] == "candidate_differs" and prefetch["needed_change"] == "reduces"
    assert _rows(report, "output_group_extent")["all"]["direction"] == "candidate_matches"
    assert _rows(report, "operand_buffering")["weight"]["direction"] == "candidate_matches"
    recipe = report["partial_recipe"]
    assert recipe["status"] is True
    assert any(r["effect"] == "operand_buffering" and r["role"] == "weight" and r["direction"] == "candidate_matches"
               for r in recipe["matched"])
    differs = {(r["effect"], r["role"]) for r in recipe["still_differs"]}
    assert {("role_transfer_width", "input"), ("issue_overlap", "weight"), ("prefetch_interleave", "all")} <= differs


def test_a_candidate_past_the_reference_states_its_numbers_and_is_offered_no_route():
    reference = _with_stream(REFERENCE, _stream(dict(input=NARROW), dict(input=dict(disjoint=1, alternating=False)),
                                                SERIAL))
    candidate = _with_stream(CANDIDATE, _stream(dict(input=WIDE), dict(input=DOUBLE), OVERLAPPED))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    for effect in ("role_transfer_width", "operand_buffering"):
        row = _rows(report, effect)["input"]
        assert row["direction"] == "candidate_differs" and row["beyond_reference"] is True
        assert row["needed_change"] == "reduces" and row["comparison"].endswith("candidate higher")
        assert row["routes"] == [] and "missing_capability" not in row and "further away" in row["routes_basis"]
    assert _rows(report, "issue_overlap")["weight"]["beyond_reference"] is True
    interleave = _rows(report, "prefetch_interleave")["all"]
    # Shorter compute runs between load bands: lower is finer interleaving, past the reference.
    assert interleave["comparison"] == "reference 64, candidate 16, candidate lower"
    assert (interleave["needed_change"], interleave["beyond_reference"]) == ("increases", True)
    assert interleave["routes"] == [] and "coarser" in interleave["larger_means"]
    assert any(r["effect"] == "prefetch_interleave" for r in report["partial_recipe"]["beyond_reference"])
    # A summary rate below the reference's, where fewer loads per compute is more of the mechanism.
    reload = next(r for r in applicability(CANDIDATE, REFERENCE, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
                  ["differences"] if r["effect"] == "operand_load_per_compute")
    assert reload["comparison"] == "reference 2.0, candidate 0.25, candidate lower"
    assert (reload["direction"], reload["needed_change"], reload["beyond_reference"]) == (
        "candidate_differs", "increases", True)
    assert reload["routes"] == [] and "missing_capability" not in reload


def test_stream_facts_stay_unknown_without_a_proved_stream():
    report = applicability(REFERENCE, CANDIDATE, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    for effect in ("role_transfer_width", "operand_buffering", "issue_overlap"):
        (row,) = _rows(report, effect).values()
        assert row["direction"] == "unknown" and "no_proved_command_stream" in row["reason"]
    assert report["partial_recipe"]["status"] == "unknown"
    one_side = applicability(STREAM_REF, CANDIDATE, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    assert "candidate side" in _rows(one_side, "role_transfer_width")[None]["reason"]


def test_static_transfer_extent_is_compared_per_operand_class():
    def tiles(by_class):
        rows = [row for shapes in by_class.values() for row in shapes]
        return dict(stable_id="x", schedule_model=dict(role_counts=dict(accumulate=1)), cfg=dict(natural_loops=0),
                    derived_effects=dict(evidence_id="x", facts=dict(transfer_tiles=_fact(dict(
                        by_role=dict(operand_load=rows), by_class=by_class)))))

    structure = dict(instruction_roles=dict(LD=["operand_load"], LD2=["operand_load"], ST=["readout"]))
    ref = tiles(dict(LD=[dict(rows=4, cols=16, sites=1)], LD2=[dict(rows=4, cols=16, sites=1)]))
    cand = tiles(dict(LD=[dict(rows=4, cols=4, sites=1)], LD2=[dict(rows=4, cols=16, sites=1)]))
    rows = {r["operand"]: r for r in mechanism_differences(ref, cand, structure=structure)
            if r["effect"] == "operand_transfer_extent"}
    assert rows["LD2"]["difference"] == "same" and rows["LD"]["needed_change"] == "increases"


def _further(rate, config, placement, used, same_bank, contiguous, on_accelerator):
    return dict(host_issue_rate=_fact(dict(rate=rate)),
                reconfiguration=_fact(dict(config_per_block=config, placement=placement)),
                local_capacity_use=_fact(dict(by_role=dict(weight=dict(rows=used, capacity=64,
                                                                        same_bank_halves=same_bank)))),
                dram_contiguity=_fact(dict(by_role=dict(input=dict(contiguous_fraction=contiguous)))),
                epilogue_placement=_fact(dict(on_accelerator=on_accelerator)))


def test_further_stream_facts_are_labelled_only_when_reported():
    reference = _with_stream(STREAM_REF, _further(0.5, 1, "hoisted", 64, False, 1.0, True))
    candidate = _with_stream(STREAM_REF, _further(0.25, 4, "in_loop", 32, True, 0.5, False))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    labels = {(r["effect"], r.get("role")): (r["direction"], r["needed_change"]) for r in report["differences"]}
    assert labels[("host_issue_rate", "all")] == ("candidate_differs", "increases")
    assert labels[("reconfiguration", "all")] == ("candidate_differs", "reduces")
    assert labels[("reconfiguration_in_loop", "all")] == ("candidate_differs", "reduces")
    assert labels[("local_capacity_use", "weight")] == ("candidate_differs", "increases")
    assert labels[("same_bank_halves", "weight")] == ("candidate_differs", "reduces")
    assert labels[("dram_contiguity", "input")] == ("candidate_differs", "increases")
    assert labels[("epilogue_placement", "all")] == ("candidate_differs", "increases")
    assert report["partial_recipe"]["status"] is True
    assert ("dram_contiguity", "input") in {(r["effect"], r["role"]) for r in report["partial_recipe"]["still_differs"]}
    plain = applicability(STREAM_REF, STREAM_REF, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    assert not {r["effect"] for r in plain["differences"]} & {"host_issue_rate", "dram_contiguity"}


def _deriver_shape(input_cols, weight_hit, run):
    """The decoded-effects deriver's value layout: data_movement/by_operand, staging kept apart."""
    def width(cols, count):
        return dict(transfers=count, shapes=[dict(rows=4, cols=cols, count=count, elements=4 * cols)])

    return dict(
        role_transfer_width=_fact(dict(data_movement=dict(operand_load=dict(
            width(16, 8), by_operand=dict(a0=width(16, 8), a1=width(input_cols, 64 // input_cols)))),
            local_staging=dict(weight_load=dict(issues=64)))),
        operand_buffering=_fact(dict(operand_load=dict(a0=dict(disjoint_windows=2, successive_phases_alternate=True)))),
        issue_overlap=_fact(dict(loads_with_compute_before_barrier=weight_hit / 2, loads=16, median_compute_run=run,
                                 median_load_run=4, fences=4, computes_per_fence=64,
                                 by_operand=dict(a0=dict(loads=8, loads_with_compute_before_barrier=weight_hit)))))


def test_deriver_layout_compares_per_role_and_operand():
    reference = _with_stream(REFERENCE, _deriver_shape(16, 0.75, 64))
    candidate = _with_stream(CANDIDATE, _deriver_shape(4, 0.0, 256))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    width = _rows(report, "role_transfer_width")
    assert width["operand_load:a0"]["direction"] == "candidate_matches"
    assert width["operand_load:a1"]["needed_change"] == "increases"
    assert width["operand_load:a1"]["preload"]["reference"] == dict(weight_load=dict(issues=64))
    assert _rows(report, "operand_buffering")["operand_load:a0"]["direction"] == "candidate_matches"
    overlap = _rows(report, "issue_overlap")
    assert overlap["a0"]["reference"]["fraction"] == 0.75 and overlap["a0"]["needed_change"] == "increases"
    assert overlap["all"]["candidate"]["fraction"] == 0.0
    assert _rows(report, "prefetch_interleave")["all"]["needed_change"] == "reduces"
    assert report["partial_recipe"]["status"] is True


def _attributed_shape(operands, *, overlap_hit):
    """Deriver layout with the role's loads attributed to ``operands`` (``{operand: (cols, count)}``)."""
    def width(cols, count):
        return dict(transfers=count, shapes=[dict(rows=4, cols=cols, count=count, elements=4 * cols)])

    shapes = [shape for cols, count in operands.values() for shape in width(cols, count)["shapes"]]
    loads = sum(count for _, count in operands.values())
    return dict(
        role_transfer_width=_fact(dict(data_movement=dict(operand_load=dict(
            transfers=loads, shapes=shapes,
            by_operand={k: width(cols, count) for k, (cols, count) in operands.items()})))),
        operand_buffering=_fact({"operand_load": {
            k: dict(disjoint_windows=1, successive_phases_alternate=False, windows=[dict(lo=16 * i, hi=16 * i + 7)])
            for i, k in enumerate(operands)}}),
        issue_overlap=_fact(dict(loads_with_compute_before_barrier=overlap_hit, loads=loads, median_compute_run=4,
                                 by_operand={k: dict(loads=count, loads_with_compute_before_barrier=overlap_hit)
                                             for k, (_, count) in operands.items()})))


def test_mismatched_operand_attribution_falls_back_to_role_level_rows():
    reference = _with_stream(REFERENCE, _attributed_shape(dict(a0=(16, 8), a1=(16, 8)), overlap_hit=1.0))
    candidate = _with_stream(CANDIDATE, _attributed_shape(dict(unattributed=(4, 4)), overlap_hit=0.0))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    width = _rows(report, "role_transfer_width")
    role = width["operand_load"]
    assert role["attribution"] == "role_level_fallback" and role["needed_change"] == "increases"
    assert role["operand_attribution"] == dict(reference=["a0", "a1"], candidate=["unattributed"])
    for key in ("operand_load:a0", "operand_load:a1", "operand_load:unattributed"):
        assert width[key]["status"] == "unknown" and width[key]["attribution"] == "argument_register"
        assert "no consuming compute port is derived" in width[key]["reason"]
        assert "compared at role level in role operand_load" in width[key]["reason"]
    # No pooled buffering entry exists on either side: one is aggregated from the per-operand windows.
    buffering = _rows(report, "operand_buffering")
    pooled = buffering["operand_load:all"]
    assert pooled["attribution"] == "role_level_fallback" and pooled["status"] == "observed"
    assert (pooled["reference"]["disjoint_windows"], pooled["candidate"]["disjoint_windows"]) == (2, 1)
    assert buffering["operand_load:a0"]["status"] == "unknown"
    overlap = [r for r in report["differences"] if r["effect"] == "issue_overlap"]
    pooled = [r for r in overlap if r["role"] == "all" and "routes" in r]
    assert len(pooled) == 1 and pooled[0]["attribution"] == "role_level_fallback"
    assert pooled[0]["reference"]["fraction"] == 1.0 and pooled[0]["candidate"]["fraction"] == 0.0
    assert pooled[0]["direction"] == "candidate_differs"
    assert {r["role"]: r["status"] for r in overlap if r["role"] in ("a0", "unattributed")} == dict(
        a0="unknown", unattributed="unknown")


def test_role_level_overlap_is_aggregated_when_one_side_reports_no_pooled_value():
    reference = _with_stream(REFERENCE, _attributed_shape(dict(a0=(16, 6), a1=(16, 2)), overlap_hit=0.5))
    candidate = _with_stream(CANDIDATE, _attributed_shape(dict(unattributed=(4, 4)), overlap_hit=0.0))
    for record in (reference, candidate):
        record["derived_effects"]["facts"]["issue_overlap"]["value"].pop("loads_with_compute_before_barrier")
    overlap = [r for r in mechanism_differences(reference, candidate) if r["effect"] == "issue_overlap"]
    pooled = [r for r in overlap if r["role"] == "all"]
    assert len(pooled) == 1 and pooled[0]["attribution"] == "role_level_fallback"
    assert pooled[0]["reference"] == dict(count=None, total=8, fraction=0.5)
    assert pooled[0]["candidate"]["fraction"] == 0.0


def test_matched_register_attribution_states_its_basis_on_per_operand_rows():
    reference = _with_stream(REFERENCE, _attributed_shape(dict(a0=(16, 8), a1=(16, 8)), overlap_hit=1.0))
    candidate = _with_stream(CANDIDATE, _attributed_shape(dict(a0=(16, 8), a1=(4, 4)), overlap_hit=1.0))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    rows = [r for r in report["differences"] if r["effect"] in ("role_transfer_width", "operand_buffering",
                                                                 "issue_overlap")]
    assert not any(r.get("attribution") == "role_level_fallback" for r in rows)
    per_operand = [r for r in rows if r["role"] in ("a0", "a1", "operand_load:a0", "operand_load:a1")]
    assert per_operand and all(r["attribution"] == "argument_register" for r in per_operand)
    assert all("no consuming compute port" in r["attribution_basis"] for r in per_operand)
    assert not any("attribution" in r for r in rows if r["role"] in ("all", "operand_load"))
    width = _rows(report, "role_transfer_width")
    assert width["operand_load:a0"]["direction"] == "candidate_matches"
    assert width["operand_load:a1"]["needed_change"] == "increases"
    assert "operand_load:all" not in _rows(report, "operand_buffering")


# --- operands matched by consuming compute port; prefetch distance; output groups from their own fact ---

PORTED = "consumption_rounds_v2"


def _ported(operands, *, columns=2048, definition=PORTED, ports=True):
    """Deriver layout whose per-operand entries name the compute port reading their rows.

    ``operands``: ``{register: (port, cols, count, overlap, median distance, compute-ahead fraction)}``."""
    def width(cols, count):
        return dict(transfers=count, shapes=[dict(rows=4, cols=cols, count=count, elements=4 * cols)])

    def port(name):
        return dict(consumer_port=name) if ports else {}

    loads = sum(spec[2] for spec in operands.values())
    distance = {k: dict(median_loads_between=d, min_loads_between=0, fraction_with_compute_ahead=ahead, loads=n,
                        consumed_loads=n, unconsumed_loads=0, **port(p))
                for k, (p, _, n, _, d, ahead) in operands.items()}
    return dict(
        role_transfer_width=_fact(dict(data_movement=dict(operand_load=dict(
            transfers=loads, shapes=[s for _, c, n, *_ in operands.values() for s in width(c, n)["shapes"]],
            by_operand={k: dict(width(c, n), **port(p)) for k, (p, c, n, *_) in operands.items()})))),
        operand_buffering=_fact({"operand_load": {
            k: dict(disjoint_windows=2, alternating=True, windows=[dict(lo=0, hi=7), dict(lo=8, hi=15)],
                    **({"definition": definition} if definition else {}), **port(p))
            for k, (p, *_) in operands.items()}}),
        issue_overlap=_fact(dict(loads_with_compute_before_barrier=1.0, loads=loads, median_compute_run=4,
                                 by_operand={k: dict(loads=n, loads_with_compute_before_barrier=o, **port(p))
                                             for k, (p, _, n, o, *_) in operands.items()})),
        prefetch_distance=_fact(dict(by_operand=distance, all=dict(
            median_loads_between=max(d for *_, d, _ in operands.values()), min_loads_between=0,
            fraction_with_compute_ahead=1.0, loads=loads, consumed_loads=loads, unconsumed_loads=0))),
        output_group_extent=_fact(dict(blocks=1, median_output_columns=columns)))


EXPERT = dict(a1=("accumulate.a", 16, 8, 1.0, 141.5, 0.97), a2=("weight_load.w", 16, 64, 1.0, 63.0, 0.99))
JUST_IN_TIME = dict(a0=("weight_load.w", 16, 64, 1.0, 0.5, 0.5), a1=("accumulate.a", 16, 8, 1.0, 173.5, 0.99))


def test_operands_are_matched_by_the_compute_port_that_reads_them():
    reference = _with_stream(REFERENCE, _ported(EXPERT))
    candidate = _with_stream(CANDIDATE, _ported(JUST_IN_TIME, columns=128))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    width = _rows(report, "role_transfer_width")["operand_load:weight_load.w"]
    assert width["attribution"] == "consumer_port" and width["direction"] == "candidate_matches"
    assert width["operand_attribution"] == dict(reference=["a2"], candidate=["a0"])
    assert _rows(report, "operand_buffering")["operand_load:weight_load.w"]["direction"] == "candidate_matches"
    assert _rows(report, "issue_overlap")["weight_load.w"]["operand_attribution"]["candidate"] == ["a0"]
    distance = _rows(report, "prefetch_distance")["weight_load.w"]
    assert (distance["reference"]["median_loads_between"], distance["candidate"]["median_loads_between"]) == (63.0, 0.5)
    assert distance["direction"] == "candidate_differs" and distance["attribution"] == "consumer_port"
    ahead = _rows(report, "prefetch_compute_ahead")["weight_load.w"]
    assert ahead["direction"] == "candidate_differs" and "routes_if_established" not in ahead
    groups = _rows(report, "output_group_extent")["all"]
    assert (groups["reference"], groups["candidate"], groups["needed_change"]) == (2048, 128, "increases")
    assert not any(r.get("role") in ("operand_load:a0", "operand_load:a2", "a0", "a2") for r in report["differences"])


def test_a_side_without_consumer_ports_keys_the_other_side_by_port_and_says_why_nothing_compares():
    reference = _ported(EXPERT, ports=False)
    reference["operand_buffering"]["value"]["operand_load"]["a2"] = dict(
        disjoint_windows=None, reason="64 loads: load local address or transfer extent unresolved")
    reference = _with_stream(REFERENCE, reference)
    candidate = _with_stream(CANDIDATE, _ported(JUST_IN_TIME))
    width = _rows(applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY),
                  "role_transfer_width")
    # The candidate's rows are keyed by the port reading them; its registers are labels only.
    port = width["operand_load:weight_load.w"]
    assert port["attribution"] == "consumer_port" and port["operand_attribution"]["candidate"] == ["a0"]
    assert port["status"] == "unknown" and port["candidate"]["elements_per_transfer"] == 64
    assert "no reference load of this role is attributed to port weight_load.w" in port["reason"]
    assert "local address or transfer extent unresolved" in port["reason"]
    assert "operand_load:a0" not in width
    # The reference's rows keep their registers, unknown, with the reason the port is missing.
    register = width["operand_load:a2"]
    assert register["attribution"] == "argument_register" and register["status"] == "unknown"
    assert "no consuming compute port is derived for a2 on the reference side" in register["reason"]
    assert "not matched to any candidate operand" in register["reason"]
    # The role level still compares every load of the role.
    assert width["operand_load"]["status"] == "observed"


def test_two_operands_sharing_a_port_fall_back_to_registers():
    shared = dict(a0=("accumulate.a", 16, 8, 1.0, 1.0, 1.0), a1=("accumulate.a", 16, 8, 1.0, 1.0, 1.0))
    reference = _with_stream(REFERENCE, _ported(shared))
    candidate = _with_stream(CANDIDATE, _ported(shared))
    rows = _rows(applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY),
                 "prefetch_distance")
    assert rows["a0"]["attribution"] == "argument_register" and "share one consuming port" in \
        rows["a0"]["attribution_basis"]


def test_buffering_counted_under_different_definitions_is_unknown():
    reference = _with_stream(REFERENCE, _ported(EXPERT, definition=None))
    candidate = _with_stream(CANDIDATE, _ported(JUST_IN_TIME))
    row = _rows(applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY),
                "operand_buffering")["operand_load:weight_load.w"]
    assert row["status"] == "unknown" and "different definitions" in row["reason"]


def test_stream_facts_a_record_predates_are_named_and_output_groups_need_their_columns():
    older = _ported(EXPERT)
    older.pop("prefetch_distance")
    older["output_group_extent"] = _fact(dict(blocks=1, column_extents={"2048": 1}))
    reference = _with_stream(REFERENCE, older)
    candidate = _with_stream(CANDIDATE, _ported(JUST_IN_TIME))
    report = applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
    distance = [r for r in report["differences"] if r["effect"] == "prefetch_distance"]
    assert all(r["status"] == "unknown" and "derived before this fact existed" in r["reason"] for r in distance)
    # The candidate still describes its own operands; nothing is compared with the reference.
    own = {r["role"]: r for r in distance if r["role"]}
    assert own["a0"]["candidate"]["median_loads_between"] == 0.5 and own["a0"]["consumer_port"] == "weight_load.w"
    assert "reference" not in own["a0"] and own["all"]["direction"] == "unknown"
    groups = _rows(report, "output_group_extent")["all"]
    assert groups["status"] == "unknown" and groups["reason"].startswith("reference: output_group_extent")


# --- comparison semantics: one definition per effect; numbers, needed_change and label agree --------

#: One side's facts making the compared quantity of ``effect`` grow with ``x`` (an int).
_STREAM_INPUTS = dict(
    role_transfer_width=lambda x: dict(role_transfer_width=_fact(dict(by_role=dict(
        input=[dict(rows=1, cols=4 * x, count=4, elements=4 * x)])))),
    operand_buffering=lambda x: dict(operand_buffering=_fact(dict(by_role=dict(input=dict(
        disjoint=x, alternating=False))))),
    issue_overlap=lambda x: dict(issue_overlap=_fact(dict(loads_with_compute_before_barrier=dict(
        input=dict(count=x, total=4))))),
    prefetch_distance=lambda x: dict(prefetch_distance=_fact(dict(all=dict(
        median_loads_between=x, fraction_with_compute_ahead=0.5)))),
    prefetch_compute_ahead=lambda x: dict(prefetch_distance=_fact(dict(all=dict(
        median_loads_between=1, fraction_with_compute_ahead=x / 4)))),
    output_group_extent=lambda x: dict(output_group_extent=_fact(dict(median_output_columns=16 * x))),
    prefetch_interleave=lambda x: dict(issue_overlap=_fact(dict(compute_run_between_load_bands=16 * x))),
)
#: The value key each further fact's metric reads (filled with ``x``), plus fixed companions.
_EXTRA_INPUTS = dict(
    host_issue_rate=dict(rate=None), reconfiguration=dict(config_per_block=None),
    reconfiguration_in_loop=dict(in_loop=None), local_capacity_use=dict(fraction=None),
    same_bank_halves=dict(same_bank_halves=None), dram_contiguity=dict(contiguous_fraction=None),
    epilogue_placement=dict(on_accelerator=None), queue_occupancy=dict(max_outstanding_before_dependency=None),
    queue_points_past_entries=dict(points_exceeding_dispatch_entries=None),
    transfer_exceeds_bound=dict(exceeds_bound=None), post_store_reuse=dict(read_again=None, accumulated_into=0))


def _inputs(effect, x):
    from merlin.kernels.mechanism_applicability import EXTRA_FACTS

    if effect in _STREAM_INPUTS:
        return _STREAM_INPUTS[effect](x)
    fact = next(name for name, compared, *_ in EXTRA_FACTS if compared == effect)
    key = next(iter(_EXTRA_INPUTS[effect]))
    return {fact: _fact(dict(_EXTRA_INPUTS[effect], **{key: x}))}


def test_every_compared_effect_has_one_explicit_comparison_semantics():
    from merlin.kernels.mechanism_applicability import COMPARISON_SEMANTICS, EXTRA_FACTS, STREAM_EFFECTS

    summary = {r["effect"] for r in mechanism_differences(REFERENCE, CANDIDATE)} - {
        "operand_retention", "local_address_windows", "dataflow", "operation"} - set(STREAM_EFFECTS)
    assert summary and summary <= set(COMPARISON_SEMANTICS)
    assert set(STREAM_EFFECTS) <= set(COMPARISON_SEMANTICS)
    assert set(_STREAM_INPUTS) | set(_EXTRA_INPUTS) == set(STREAM_EFFECTS)
    assert {compared for _, compared, *_ in EXTRA_FACTS} == set(_EXTRA_INPUTS)
    for larger, more_when in COMPARISON_SEMANTICS.values():
        assert larger and more_when in ("increases", "reduces", None)
    # The catalog's routes move each effect the way the table says adds more of the mechanism.
    for capability in CAPABILITY_EFFECTS.values():
        for effect, change in capability["changes"].items():
            assert COMPARISON_SEMANTICS[effect][1] == change, effect


@pytest.mark.parametrize("effect", sorted(set(_STREAM_INPUTS) | set(_EXTRA_INPUTS)))
@pytest.mark.parametrize("lower", [True, False])
def test_stream_and_further_facts_state_numbers_change_and_label_consistently(effect, lower):
    from merlin.kernels.mechanism_applicability import COMPARISON_SEMANTICS

    ref_x, cand_x = (2, 1) if lower else (1, 2)
    reference = _with_stream(REFERENCE, _inputs(effect, ref_x))
    candidate = _with_stream(CANDIDATE, _inputs(effect, cand_x))
    rows = [r for r in applicability(reference, candidate, shape=dict(m=8, k=16, n=64), geometry=GEOMETRY)
            ["differences"] if r["effect"] == effect and r["direction"] != "unknown"]
    assert rows, effect
    larger, more_when = COMPARISON_SEMANTICS[effect]
    for row in rows:
        assert row["direction"] == "candidate_differs" and row["difference"] == "differs"
        assert row["comparison"].endswith("candidate lower" if lower else "candidate higher"), row["comparison"]
        assert row["needed_change"] == ("increases" if lower else "reduces")
        assert row["larger_means"] == larger
        assert row["beyond_reference"] is (row["needed_change"] != more_when)
        if row["beyond_reference"]:
            assert row["routes"] == [] and "missing_capability" not in row and row["routes_basis"]


@pytest.mark.parametrize("effect,reference,candidate,word,change,beyond", [
    ("operand_load_per_compute", 5.25, 2.0, "lower", "increases", True),
    ("operand_load_per_compute", 1.0, 2.0, "higher", "reduces", False),
    ("weight_load_per_compute", 1.0, 0.5, "lower", "increases", True),
    ("operand_transfer_extent", 224, 16, "lower", "increases", False),
    ("broadcast_load_sites", 4, 0, "lower", "increases", False),
    ("config_per_transfer", 0.2, 0.833333, "higher", "reduces", False),
    ("sync_sites", 0, 1, "higher", "reduces", False),
    ("loop_descriptor_sites", 2, 0, "lower", "increases", None),
    ("configured_load_states", 3, 2, "lower", "increases", None),
])
def test_summary_effects_read_in_plain_words(effect, reference, candidate, word, change, beyond):
    from merlin.kernels.mechanism_applicability import _order

    fields = _order(effect, reference, candidate)
    assert fields["comparison"] == f"reference {reference}, candidate {candidate}, candidate {word}"
    assert (fields["needed_change"], fields["beyond_reference"]) == (change, beyond)
    assert _order(effect, reference, reference)["direction"] == "candidate_matches"
