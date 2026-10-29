"""The capacity cohort: large Phase 2 shapes derived from a target's facts, decided per regime."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import capacity_families as CF
from merlin_experiments.phase0 import profiles
from merlin_experiments.phase2.claims import capacity as CP
from merlin_experiments.phase2.claims import dispatch

from merlin.common.paths import repo_root
from merlin.targetgen import corpus_spec as CS

#: A synthetic machine, deliberately not any real target's numbers: an 8x8 array, an operand store of
#: four banks and an accumulator of two, read/write movement ports, three command classes and two
#: runtime-selected dataflow modes.
_PORT = {
    "request_channels": 1,
    "beat_bytes": 8,
    "max_request_bytes": 32,
    "in_flight": {"status": "derived", "value": 4},
}
_FACTS = {
    "facts": {
        "arrays": [{"name": "mesh", "rows": 8, "cols": 8, "mac_idiom": {"muls": 1}}],
        "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
        "memories": [
            {"name": "scratchpad", "banks": 4, "bytes": 32768, "depth": 1024, "row_elems": 8},
            {"name": "accumulator", "banks": 2, "bytes": 8192, "depth": 128, "lanes": 8},
        ],
        "timing_structure": {
            "memory_clients": {
                "status": "derived",
                "clients": {
                    "reader": {**_PORT, "request_data_bits": 64, "response_data_bits": 64},
                    "writer": {**_PORT, "request_data_bits": 64, "response_data_bits": None},
                },
            },
            "command_classes": {
                "status": "derived",
                "classes": {
                    name: {
                        "entries": {"status": "derived", "value": entries},
                        "command_queue": {"status": "derived", "entries_total": queue},
                    }
                    for name, entries, queue in (("a", 8, 4), ("b", 4, 2), ("c", 2, 1))
                },
            },
            "dataflow": {
                "status": "derived",
                "modes": {"0": "output_stationary", "1": "weight_stationary"},
                "select_runtime_driven": True,
            },
        },
    }
}


def _template() -> dict:
    return yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())


def _px() -> dict:
    return copy.deepcopy(next(s for s in _template()["sweeps"] if s["id"] == "PX"))


def _entries(skipped=None):
    return CF.capacity_family_entries(_px(), target="synthetic", facts=_FACTS, skipped=skipped)


def test_the_basis_is_read_from_the_facts():
    basis = CF.derive_basis("synthetic", _FACTS)
    assert (basis.tile_rows, basis.operand_bytes, basis.accumulate_bytes, basis.ways) == (8, 1, 4, 2)
    assert basis.operand_store_bytes == 32768 and basis.accumulator_bytes == 8192
    assert basis.read_bytes_per_cycle == basis.write_bytes_per_cycle == 8.0
    assert basis.command_windows == {"a": 12, "b": 6, "c": 3}
    body = copy.deepcopy(_FACTS)
    del body["facts"]["timing_structure"]["memory_clients"]
    with pytest.raises(CF.CapacityUnavailable, match="movement ports"):
        CF.derive_basis("synthetic", body)


def test_fractional_capacity_points_produce_valid_distinct_capsule_names():
    import jsonschema

    from merlin.targetgen.contract.schemas import load_schema

    entries = _entries()
    name_schema = load_schema("capsule")["properties"]["name"]
    for entry in entries:
        jsonschema.validate(entry["name"], name_schema)
    assert len({entry["name"] for entry in entries}) == len(entries)
    assert any("0.2x" in entry["source_reference"] for entry in entries)


def test_encoded_labels_cannot_alias_valid_literals():
    labels = ["x0.2", "x0_2", "encoded_78302e32", "x1e-06", "x1e+06", "x1e_06"]
    assert len({CF._identifier_label(label) for label in labels}) == len(labels)


def test_every_declared_family_lands_at_least_two_points_per_regime_and_scales_with_capacity():
    skipped: list = []
    entries = _entries(skipped)
    assert not skipped
    families = {e["performance"]["capacity"]["family"] for e in entries}
    assert families == {spec["name"] for spec in _px()["requires_capacity_scope"]["families"]}
    groups: dict = {}
    for entry in entries:
        block = entry["performance"]["capacity"]
        groups.setdefault((block["family"], block["variant"], block["regime"]), set()).add(block["x"])
        assert block["why"] and block["exercises"] and block["roofline"]["floor_cycles"] > 0
    assert all(len(xs) >= 2 for xs in groups.values())
    knee = {e["performance"]["capacity"]["regime"] for e in entries if e["performance"]["capacity"]["family"] == "knee"}
    assert knee == {CF.FITS_DOUBLE, CF.FITS_SINGLE, CF.SPILLS}
    largest = max(e.get("M", 0) * e.get("K", 0) * e.get("N", 0) for e in entries if e.get("op") == "matmul")
    # The working sets reach past the stores: no member of the old cohort came near them.
    assert largest * 1 > 4 * 32768


def test_larger_stores_scale_the_derived_extents():
    big = copy.deepcopy(_FACTS)
    for memory in big["facts"]["memories"]:
        memory["bytes"] *= 4
        memory["depth"] *= 4

    def deepest(entries):
        return max(e["K"] for e in entries if e["performance"]["capacity"]["family"] == "deep_k")

    assert deepest(CF.capacity_family_entries(_px(), target="synthetic", facts=big)) > deepest(_entries())


def test_the_regimes_say_why_with_the_derived_numbers():
    entries = _entries()
    short = next(e for e in entries if e["performance"]["capacity"]["family"] == "short_m" and e["M"] == 1)
    block = short["performance"]["capacity"]
    assert "MAC per weight byte" in block["why"]
    # Logical byte counts do not establish this fixture's physical traffic.
    # The known compute floor cannot identify its complete runtime bottleneck.
    assert block["roofline"]["status"] == "partial"
    assert "movement" in block["roofline"]["unresolved"]
    assert all(value is None for value in block["roofline"]["movement"]["physical_bytes"].values())
    queue = [e["performance"]["capacity"] for e in entries if e["performance"]["capacity"]["family"] == "queue"]
    assert {b["regime"] for b in queue} == {"within_window", "back_pressure"}
    assert all("'c'" in b["why"] for b in queue), "the smallest live window decides back-pressure"


def test_a_target_without_a_runtime_dataflow_choice_skips_that_family_with_its_reason():
    facts = copy.deepcopy(_FACTS)
    facts["facts"]["timing_structure"]["dataflow"]["select_runtime_driven"] = False
    skipped: list = []
    entries = CF.capacity_family_entries(_px(), target="synthetic", facts=facts, skipped=skipped)
    assert [row["family"] for row in skipped] == ["dataflow"]
    assert all(e["performance"]["capacity"]["family"] != "dataflow" for e in entries)


def test_members_build_on_the_direct_corpus_path():
    binding = CS.CorpusBinding(
        target="synthetic",
        tile_dim=8,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L2"],
        compare="exact_int",
    )
    built = set()
    for entry in _entries():
        kind = entry["performance"]["capacity"]["kind"]
        if kind in built or "requant" in entry.get("epilogue", []):
            continue
        capsule, _ = CS.build({**entry, "operand_dtype": "int8"}, binding)
        assert capsule["operation"]["op"] == entry["op"]
        built.add(kind)
    assert {"capacity_knee", "conv", "residency_chain", "epilogue"} <= built


@pytest.mark.parametrize("integer_shift_admitted", [False, True])
def test_capacity_software_refusals_remain_explicit_unmet_points(integer_shift_admitted):
    stages = ["acc_scale", "relu"] + (["requant"] if integer_shift_admitted else [])
    spec = {
        "operations": [
            {"id": "contract", "families": ["contraction"], "placement": "accelerator", "signature": {}},
            {"id": "readout", "families": ["elementwise_map"], "placement": "fused_accelerator",
             "signature": {"composed_with": ["contraction"], "epilogues": stages}},
        ],
    }
    binding = CS.CorpusBinding(
        target="synthetic", tile_dim=8, operand_dtype="int8", accum_dtype="i32", integer=True,
        tiers=["L2"], compare="exact_int", requant_output_dtype="i8", requant_shift=4,
    )
    blocked = []
    arguments = dict(
        binding=binding, evidence=SimpleNamespace(refreshed_facts=_FACTS, software_spec=spec),
        requirement=None, entries=[], skipped=[],
    )
    members = CF.scope_members(_px(), **arguments, blocked=blocked)
    shifts = [row for row in _entries() if "requant" in row.get("epilogue", [])]
    assert shifts and all(row["requant_shift"] == 4 for row in shifts)
    selected_shifts = [row for row in members if "requant" in row.get("epilogue", [])]
    assert len(selected_shifts) == (len(shifts) if integer_shift_admitted else 0)
    assert len(blocked) == (0 if integer_shift_admitted else len(shifts))
    assert any(row.get("epilogue") == ["acc_scale", "relu"] for row in members)
    for refusal in blocked:
        assert refusal["operation_admission"]["status"] == "unsupported"
        assert refusal["entry"] in shifts
        assert refusal["capacity_point"] == refusal["entry"]["performance"]["capacity"]
        assert refusal["status"] == "blocked_unimplemented"
    if not integer_shift_admitted:
        with pytest.raises(ValueError, match="retained unmet-obligation"):
            CF.scope_members(_px(), **arguments, blocked=None)


def test_the_shared_template_declares_the_capacity_family_and_it_dispatches():
    profile: dict = {"capsules": []}
    profiles._merge_shared_perf(
        profile,
        source=repo_root() / "recipe.yaml",
        performance_template=repo_root() / "experiments/templates/phase0/performance.yaml",
    )
    families = {row["family"]: row for row in profile["_performance_template"]["families"]}
    assert families["PX"]["claim"] == "PREDICTS"
    resolved = dispatch.resolve(_entries())
    assert resolved.analyze is CP.analyze_capacity_claim
    assert CP.preflight_capacity_claim(_entries())["verdict"] == "ADMISSIBLE"
    with pytest.raises(ValueError, match="attainment_min"):
        sweep = _px()
        sweep["requires_capacity_scope"]["roofline_target"] = {}
        CF.validate_capacity_scope_declaration(sweep, owner="PX")


def _rows(entries, cycles):
    return [
        {"capsule": e["name"], "replicate": r, "calibrated_timing_cycles": cycles(e)}
        for e in entries
        for r in ("r000", "r001")
    ]


def _affine(entry):
    block = entry["performance"]["capacity"]
    floor = block["roofline"]["floor_cycles"]
    # Twice the floor plus a fixed cost, linear in the regime variable within each group.
    return 2 * floor + 1000


def test_a_per_regime_affine_cohort_above_its_floors_is_accepted():
    entries = [e for e in _entries() if e["performance"]["capacity"]["family"] in ("deep_k", "chain")]
    verdict = CP.analyze_capacity_claim(entries, _rows(entries, _affine))
    assert verdict["verdict"] == CP.ACCEPTED, verdict
    assert all(0 < a <= 1 for a in verdict["measured"]["attainment_by_capsule"].values())
    assert verdict["measured"]["regime_slope_ratios"]["chain/w2"] == []


def test_beating_a_floor_or_missing_a_replicate_is_never_accepted():
    entries = [e for e in _entries() if e["performance"]["capacity"]["family"] == "deep_k"]
    below = CP.analyze_capacity_claim(
        entries, _rows(entries, lambda e: e["performance"]["capacity"]["roofline"]["floor_cycles"] / 2)
    )
    assert below["verdict"] == CP.REFUTED and below["measured"]["floor_refutations"]
    partial = _rows(entries, _affine)[:-1]
    assert CP.analyze_capacity_claim(entries, partial)["verdict"] == CP.REFUSED
