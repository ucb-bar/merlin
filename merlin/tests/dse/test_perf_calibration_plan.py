from __future__ import annotations

from merlin.perf.calibration_plan import (
    DerivedFact,
    Disposition,
    MIN_POINTS_PER_PARAMETER,
    REQUIRED_FACT_KEYS,
    build_calibration_plan,
    build_calibration_plan_from_rtl,
    derive_resource_traits,
    plan_calibration,
)


def _fact(value, path: str):
    return {
        "value": value,
        "derived": True,
        "source": f"rtl extraction at {path}",
        "path": path,
    }


def _complete_traits():
    return {
        "dma.read.sizes_bytes": _fact([32, 128, 512, 2048], "facts.ports.read.legal_sizes"),
        "dma.write.sizes_bytes": _fact([16, 64, 256, 1024], "facts.ports.write.legal_sizes"),
        "dma.copy.sizes_bytes": _fact([64, 192, 768, 3072], "facts.paths.copy.legal_sizes"),
        "dma.measurement_protocols": _fact(
            ["fresh-process", "predecessor-run"], "facts.instrument.measurement_protocols"),
        "compute.tile_shape": _fact([7, 7, 13], "facts.arrays.primary.tile_shape"),
        "compute.tile_multiples": _fact([1, 3, 6, 10], "facts.resources.tile_multiples"),
    }


def _synthetic_rtl_facts():
    return {
        "generator": {"tool": "structural-fixture", "revision": "abc123"},
        "facts": {
            "source": "synthetic_chip/rtl/top.sv",
            "arrays": [{
                "rows": 6,
                "cols": 10,
                "source": "structural array discovery",
            }],
            "memories": [{
                "bytes": 24_576,
                "depth": 768,
                "source": "structural memory discovery",
            }],
        },
    }


def _synthetic_capabilities():
    return {
        "dma.directions": {
            "value": ["read", "write", "copy"],
            "derived_from_tool": True,
            "source": "harness operation inventory",
        },
        "dma.measurement_protocols": {
            "value": ["fresh-process", "predecessor-run"],
            "derived_from_tool": True,
            "source": "runner measurement protocol probe",
        },
        "dma.read.sizes_bytes": {
            "value": [24, 96, 384, 1536],
            "derived_from_tool": True,
            "source": "compiler-generated read descriptor probes",
        },
        "dma.write.sizes_bytes": {
            "value": [40, 160, 640, 2560],
            "derived_from_tool": True,
            "source": "compiler-generated write descriptor probes",
        },
        "dma.copy.sizes_bytes": {
            "value": [56, 224, 896, 3584],
            "derived_from_tool": True,
            "source": "compiler-generated copy descriptor probes",
        },
        "compute.workload_emitter": {
            "value": True,
            "derived_from_tool": True,
            "source": "workload emitter capability probe",
        },
        "compute.tile_multiples": {
            "value": [1, 3, 7, 15],
            "derived_from_tool": True,
            "source": "compiler-generated compute workload probes",
        },
    }


def test_complete_plan_has_six_dma_sweeps_and_one_compute_sweep():
    plan = build_calibration_plan(_complete_traits())

    assert plan.ready
    assert plan.status == "READY"
    assert len(plan.sweeps) == 7
    assert {sweep.sweep_id for sweep in plan.sweeps} == {
        "dma.read.fresh-process", "dma.read.predecessor-run",
        "dma.write.fresh-process", "dma.write.predecessor-run",
        "dma.copy.fresh-process", "dma.copy.predecessor-run",
        "compute.saturation",
    }
    assert all(sweep.disposition is Disposition.READY for sweep in plan.sweeps)
    assert all(sweep.fit.points_per_parameter == MIN_POINTS_PER_PARAMETER
               for sweep in plan.sweeps)
    assert all(len(sweep.points) >= sweep.fit.required_points for sweep in plan.sweeps)
    assert all("fixed_term_separation" in sweep.objective for sweep in plan.sweeps)


def test_dma_coordinates_preserve_derived_sizes_protocols_and_fact_provenance():
    sweep = next(s for s in build_calibration_plan(_complete_traits()).sweeps
                 if s.sweep_id == "dma.copy.predecessor-run")

    assert [point.to_dict() for point in sweep.points] == [
        {"transfer_bytes": 64, "measurement_protocol": "predecessor-run"},
        {"transfer_bytes": 192, "measurement_protocol": "predecessor-run"},
        {"transfer_bytes": 768, "measurement_protocol": "predecessor-run"},
        {"transfer_bytes": 3072, "measurement_protocol": "predecessor-run"},
    ]
    assert [(fact.semantic, fact.path) for fact in sweep.facts] == [
        ("dma.copy.sizes_bytes", "facts.paths.copy.legal_sizes"),
        ("dma.measurement_protocols", "facts.instrument.measurement_protocols"),
    ]
    assert all(fact.provenance for fact in sweep.facts)


def test_compute_points_use_supplied_geometry_and_multiples_without_defaults():
    sweep = build_calibration_plan(_complete_traits()).sweeps[-1]

    assert [point.to_dict() for point in sweep.points] == [
        {"tile_multiple": 1, "tile_shape": [7, 7, 13]},
        {"tile_multiple": 3, "tile_shape": [7, 7, 13]},
        {"tile_multiple": 6, "tile_shape": [7, 7, 13]},
        {"tile_multiple": 10, "tile_shape": [7, 7, 13]},
    ]
    assert {fact.path for fact in sweep.facts} == {
        "facts.arrays.primary.tile_shape",
        "facts.resources.tile_multiples",
    }


def test_empty_bundle_emits_explicit_unknowns_for_every_required_sweep():
    plan = plan_calibration({})

    assert not plan.ready
    assert len(plan.unknown) == 4
    assert not plan.refusals
    assert all(not sweep.points for sweep in plan.sweeps)
    assert {issue.code for sweep in plan.unknown for issue in sweep.issues} == {"MISSING_FACT"}
    assert set(REQUIRED_FACT_KEYS) == {
        issue.fact_paths[0]
        for sweep in plan.unknown
        for issue in sweep.issues
    }


def test_unproven_and_underived_values_stay_unknown():
    traits = _complete_traits()
    traits["dma.read.sizes_bytes"] = [1, 2, 3, 4]
    traits["compute.tile_shape"] = DerivedFact(
        value=(4, 4),
        path="declared.compute.shape",
        provenance=("descriptor declaration",),
        derived_from_rtl=False,
    )

    plan = build_calibration_plan(traits)
    read = [sweep for sweep in plan.sweeps if sweep.mechanism == "dma_read"]
    compute = plan.sweeps[-1]

    assert all(sweep.disposition is Disposition.UNKNOWN for sweep in read)
    assert {issue.code for sweep in read for issue in sweep.issues} == {"UNPROVEN_FACT"}
    assert compute.disposition is Disposition.UNKNOWN
    assert compute.issues[0].code == "UNKNOWN_FACT"
    assert "positive RTL-derived standing" in compute.issues[0].reason


def test_rate_plus_fixed_term_refuses_fewer_than_two_points_per_parameter():
    traits = _complete_traits()
    traits["dma.write.sizes_bytes"] = _fact([8, 64, 512], "facts.write.sizes")
    traits["compute.tile_multiples"] = _fact([1, 2, 4], "facts.compute.multiples")

    plan = build_calibration_plan(traits)
    write = [sweep for sweep in plan.sweeps if sweep.mechanism == "dma_write"]
    compute = plan.sweeps[-1]

    assert all(sweep.disposition is Disposition.REFUSED for sweep in write)
    assert all(sweep.issues[0].code == "INSUFFICIENT_FIT_POINTS" for sweep in write)
    assert compute.disposition is Disposition.REFUSED
    assert compute.issues[0].code == "INSUFFICIENT_FIT_POINTS"
    assert not write[0].points
    assert not compute.points


def test_one_observable_protocol_creates_one_honestly_labelled_sweep_per_direction():
    traits = _complete_traits()
    traits["dma.measurement_protocols"] = _fact(
        ["one-observed-protocol"], "facts.instrument.measurement_protocols")

    plan = build_calibration_plan(traits)
    dma = plan.sweeps[:-1]

    assert len(dma) == 3
    assert all(sweep.ready and sweep.condition == "one-observed-protocol" for sweep in dma)
    assert plan.sweeps[-1].ready


def test_target_bundle_fields_shape_is_accepted_and_serialises_fail_closed_status():
    bundle = {"fields": _complete_traits()}

    payload = build_calibration_plan(bundle).to_dict()

    assert payload["status"] == "READY"
    assert payload["ready_sweeps"] == payload["required_sweeps"] == 7
    assert payload["sweeps"][0]["facts"][0]["path"] == "facts.ports.read.legal_sizes"
    assert payload["sweeps"][0]["fit"]["required_points"] == 4


def test_invalid_coordinates_refuse_instead_of_coercing_or_guessing():
    traits = _complete_traits()
    traits["dma.copy.sizes_bytes"] = _fact([64, 64, 256, 1024], "facts.copy.sizes")
    traits["compute.tile_shape"] = _fact([8, 0, 8], "facts.compute.shape")

    plan = build_calibration_plan(traits)
    copy = [sweep for sweep in plan.sweeps if sweep.mechanism == "dma_copy"]

    assert all(sweep.disposition is Disposition.REFUSED for sweep in copy)
    assert all(sweep.issues[0].code == "INVALID_FACT" for sweep in copy)
    assert plan.sweeps[-1].disposition is Disposition.REFUSED
    assert not plan.sweeps[-1].points


def test_raw_rtl_adapter_uses_rtl_geometry_and_tool_proven_legal_coordinates():
    traits = derive_resource_traits(_synthetic_rtl_facts(), _synthetic_capabilities())
    plan = build_calibration_plan(traits)

    assert plan.ready
    assert traits["dma.read.sizes_bytes"].value == [24, 96, 384, 1536]
    assert traits["compute.tile_shape"].value == (6, 10)
    assert traits["compute.tile_multiples"].value == [1, 3, 7, 15]
    compute = plan.sweeps[-1]
    assert [point.to_dict() for point in compute.points] == [
        {"tile_multiple": 1, "tile_shape": [6, 10]},
        {"tile_multiple": 3, "tile_shape": [6, 10]},
        {"tile_multiple": 7, "tile_shape": [6, 10]},
        {"tile_multiple": 15, "tile_shape": [6, 10]},
    ]
    assert any(fact.path == "facts.arrays[0].{rows,cols}" for fact in compute.facts)
    assert all(fact.provenance for fact in compute.facts)


def test_tool_derived_protocol_is_admitted_without_a_cache_state_claim():
    plan = build_calibration_plan_from_rtl(_synthetic_rtl_facts(), _synthetic_capabilities())
    dma = plan.sweeps[:-1]

    assert all(sweep.ready for sweep in dma)
    protocol_evidence = [fact for fact in dma[0].facts
                         if fact.semantic == "dma.measurement_protocols"]
    assert len(protocol_evidence) == 1
    assert protocol_evidence[0].standing == "tool_derived"
    assert protocol_evidence[0].provenance == ("runner measurement protocol probe",)


def test_tool_standing_does_not_authorise_hardware_coordinates():
    traits = _complete_traits()
    traits["dma.read.sizes_bytes"] = {
        "value": [32, 64, 128, 256],
        "derived_from_tool": True,
        "source": "harness guess",
    }
    traits["dma.measurement_protocols"] = {
        "value": ["fresh-process", "predecessor-run"],
        "derived_from_tool": True,
        "source": "runner protocol probe",
    }

    plan = build_calibration_plan(traits)
    read = [sweep for sweep in plan.sweeps if sweep.mechanism == "dma_read"]

    assert all(sweep.ready for sweep in read)
    assert all(sweep.facts[0].standing == "tool_derived" for sweep in read)


def test_missing_tool_capabilities_leave_every_adapter_sweep_unknown():
    plan = build_calibration_plan_from_rtl(_synthetic_rtl_facts(), {})

    assert len(plan.unknown) == 4
    assert not plan.refusals
    assert all(not sweep.points for sweep in plan.sweeps)
    assert all(any(path.startswith("capabilities.")
                   for issue in sweep.issues for path in issue.fact_paths)
               for sweep in plan.sweeps)


def test_adapter_refuses_to_choose_between_multiple_unmarked_compute_arrays():
    facts = _synthetic_rtl_facts()
    facts["facts"]["arrays"].append({
        "rows": 3,
        "cols": 19,
        "source": "second structural array discovery",
    })

    plan = build_calibration_plan_from_rtl(facts, _synthetic_capabilities())

    assert all(sweep.ready for sweep in plan.sweeps[:-1])
    assert plan.sweeps[-1].disposition is Disposition.UNKNOWN
    assert not plan.sweeps[-1].points
    assert {path for issue in plan.sweeps[-1].issues for path in issue.fact_paths} == {
        "facts.arrays",
    }


def test_memory_geometry_cannot_change_or_create_a_dma_transfer_ladder():
    facts = _synthetic_rtl_facts()
    facts["facts"]["memories"].append({
        "bytes": 12_288,
        "depth": 192,
        "source": "second structural memory discovery",
    })

    traits = derive_resource_traits(facts, _synthetic_capabilities())

    assert traits["dma.copy.sizes_bytes"].value == [56, 224, 896, 3584]
    assert traits["dma.copy.sizes_bytes"].path == "dma.copy.sizes_bytes"


def test_memory_geometry_alone_never_establishes_legal_dma_sizes():
    capabilities = _synthetic_capabilities()
    for direction in ("read", "write", "copy"):
        del capabilities[f"dma.{direction}.sizes_bytes"]

    plan = build_calibration_plan_from_rtl(_synthetic_rtl_facts(), capabilities)

    assert all(sweep.disposition is Disposition.UNKNOWN for sweep in plan.sweeps[:-1])
    assert all(not sweep.points for sweep in plan.sweeps[:-1])


def test_compute_array_geometry_never_fabricates_executable_multiples():
    capabilities = _synthetic_capabilities()
    del capabilities["compute.tile_multiples"]

    plan = build_calibration_plan_from_rtl(_synthetic_rtl_facts(), capabilities)

    assert plan.sweeps[-1].disposition is Disposition.UNKNOWN
    assert not plan.sweeps[-1].points
