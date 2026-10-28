"""Fence density and static overlap reach the author: per case, in the brief, the grade and the headline.

The fixture holds anonymized excerpts of one archived attempt: a self-check's decoded instruction stream
whose compiler fenced the whole device after every output tile, the same command classes re-ordered as
a software-pipelined stream, and the timing model's command census of a large GEMM (131 fences, 128
output stores). Roles come from the fixture target's own endpoint declaration, never from this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import fence_overlap as FO
from merlin_experiments.phase1.feedback import mechanism_hints as MH
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import performance_headline as PH
from merlin_experiments.phase1.feedback import performance_headroom as PR

FIXTURE = json.loads((Path(__file__).with_name("fence_overlap_streams.json")).read_text(encoding="utf-8"))


def _roles():
    roles = FO._endpoint_roles(FIXTURE["target"])
    if roles is None:
        pytest.skip("the fixture target's compute-endpoint declaration is not resolvable here")
    return roles


def _case(workspace: Path, name: str, stream: str) -> None:
    folder = workspace / "selfcheck_out" / name
    folder.mkdir(parents=True)
    (folder / FO.TRACE).write_text(json.dumps(FIXTURE[stream]))
    (folder / FO.BUFFER).write_text(json.dumps({"target": FIXTURE["target"], "commands": []}))


def _suggestions(workspace: Path) -> None:
    rows = [
        {
            "id": "readout_frontier",
            "family": "overlap",
            "api": "lib.ready",
            "symbol": "propose",
            "status": "supported",
            "requirements": {"memory.accumulator": "supported"},
            "mechanism": "Read out each accumulator tile at its final frontier",
            "evidence": "readout issue points",
        },
        {
            "id": "prefetch",
            "family": "overlap",
            "api": "lib.copies",
            "symbol": "prefetch",
            "status": "unknown",
            "requirements": {"workload.independent_buffers": "unknown"},
            "evidence": "copy overlap",
        },
        {
            "id": "fence_cleanup",
            "family": "synchronization",
            "api": "lib.effects",
            "symbol": "cleanup",
            "status": "unknown",
            "requirements": {"scheduling.queues": "unknown"},
            "mechanism": "Remove redundant fences and barriers",
            "evidence": "fence count",
        },
        {
            "id": "coalesce",
            "family": "movement",
            "api": "lib.copies",
            "symbol": "merge",
            "status": "supported",
            "requirements": {"memory.explicit_dma": "supported"},
            "evidence": "transfer count",
        },
        {
            "id": "loops",
            "family": "issue",
            "api": "lib.loops",
            "symbol": "compact",
            "status": "supported",
            "requirements": {"submission.cpu_issued": "supported"},
            "evidence": "command count",
        },
    ]
    (workspace / MH.SUGGESTIONS).write_text(json.dumps({"all_mechanisms": rows}))


def _performance(cycles: int, census: dict | None = None) -> dict:
    return {
        "status": "PROVISIONAL",
        "complete_event_accounting": True,
        "estimated_whole_soc_cycles": cycles,
        "structural_roofline": {
            "useful_macs": 4096,
            "structural_peak_macs_per_cycle": 256,
            "structural_compute_floor_cycles": 16,
        },
        "feature_counts": dict(census or {}),
    }


def test_fence_after_every_tile_is_counted_from_the_decoded_stream():
    profile = FO.static_profile(FIXTURE["fence_per_tile"], _roles())
    assert profile["fences"] == 37 and profile["output_tiles"] == 36
    assert profile["epochs_with_readout"] >= 30 and profile["median_tiles_per_readout_epoch"] == 1
    assert profile["static_overlap_ratio"] == 0.0 and profile["loads_after_earlier_readout_in_epoch"] == 0
    assert profile["commands_without_declared_role"] == 0
    whole = {"static": profile}
    whole["fence_per_tile"] = FO.fence_per_tile(whole)
    assert whole["fence_per_tile"] is True
    text = FO.reading(whole)
    assert text.startswith("a full-device fence after every output tile (37 fences, 0.97 tile/fence, static sites)")
    assert "0% of loads issue while compute is outstanding" in text


def test_overlapped_stream_reads_as_overlapped():
    profile = FO.static_profile(FIXTURE["overlapped"], _roles())
    assert profile["fences"] == profile["boundary_fences"] == 2 and profile["epochs"] == 1
    assert profile["output_tiles"] == 6 and profile["static_overlap_ratio"] == pytest.approx(10 / 12, abs=1e-3)
    assert profile["loads_after_earlier_readout_in_epoch"] == 8
    whole = {"static": profile}
    assert FO.fence_per_tile(whole) is False
    assert FO.reading({**whole, "fence_per_tile": False}).startswith("2 fences (static sites), 3.0 tile/fence")


def test_executed_census_of_a_large_gemm_names_the_fence_storm():
    census = FIXTURE["large_gemm_census"]
    dynamic = FO.dynamic_profile({**census, "estimated_whole_soc_cycles": 191_520})
    assert (dynamic["fences"], dynamic["output_stores"], dynamic["loads"]) == (131, 128, 45915)
    assert dynamic["computes"] == 45824 and dynamic["output_stores_per_fence"] == 0.98
    assert dynamic["modeled_fence_wait_cycles"] == 4788 and dynamic["fence_wait_share_of_cycles"] == 0.025
    profile = {"dynamic": dynamic}
    profile["fence_per_tile"] = FO.fence_per_tile(profile)
    assert profile["fence_per_tile"] is True
    assert FO.reading(profile) == (
        "a full-device fence after every output tile (131 fences, 0.98 tile/fence, executed): loads cannot "
        "overlap compute across tiles; the timing model spends 2.5% of cycles waiting on fences"
    )
    # A program with only bracketing fences around a few tiles is not called fence-bound.
    few = FO.dynamic_profile({"feature_counts": {"target.fences": 3, "target.class.store": 2}})
    assert FO.fence_per_tile({"dynamic": few, "static": {"boundary_fences": 2}}) is False


def test_fence_hints_put_fence_and_readout_mechanisms_first_and_list_unconfirmed_apart(tmp_path):
    _suggestions(tmp_path)
    available, unconfirmed = MH.fence_hints(MH.load_selection(tmp_path), why="fenced per tile")
    assert [hint["id"] for hint in available] == ["readout_frontier", "loops", "coalesce", "prefetch"]
    assert available[0]["why"].startswith("names a fence/barrier/readout; fenced per tile")
    assert unconfirmed == [
        {
            "id": "fence_cleanup",
            "api": "lib.effects:cleanup",
            "facet": "synchronization",
            "capability_unconfirmed": ["scheduling.queues"],
        }
    ]
    assert MH.bound_facets(MH.FENCE_BOUND) == ("synchronization", "overlap", "issue", "movement")


def test_brief_entry_reading_and_fence_bound_list(tmp_path):
    _roles()
    _suggestions(tmp_path)
    _case(tmp_path, "fenced", "fence_per_tile")
    _case(tmp_path, "pipelined", "overlapped")
    census = FIXTURE["large_gemm_census"]["feature_counts"]
    rows = [
        {"capsule": "fenced", "pass": True, "performance": _performance(4800)},
        {"capsule": "pipelined", "pass": True, "performance": _performance(900)},
        {"capsule": "large", "pass": True, "performance": {"feature_counts": census}},
    ]
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'large_kernel'} for name in ('fenced', 'pipelined', 'large')}}))
    brief = PB.build(rows, workspace=tmp_path, artifact_root="selfcheck_out")
    fenced, pipelined = brief["largest_headroom"]
    block = fenced["fence_overlap"]
    assert block["fence_per_tile"] is True and block["fences"] == 37 and block["static_overlap_ratio"] == 0.0
    assert block["reading"] in fenced["reading"]
    assert [m["id"] for m in block["suggested_mechanisms"]][:1] == ["readout_frontier"]
    assert pipelined["fence_overlap"]["fence_per_tile"] is False
    assert "fence after every output tile" not in pipelined["reading"]
    listed = brief["fence_bound"]
    assert [case["test_case"] for case in listed["cases"]] == ["large", "fenced"]  # untimed cases too
    assert listed["cases"][0]["reading"].startswith("a full-device fence after every output tile (131 fences")
    assert listed["related_capability_unconfirmed"][0]["id"] == "fence_cleanup"
    print(json.dumps({"entry": block, "fence_bound": listed}, indent=1))


def test_grade_headroom_and_headline_carry_the_fence_reading(tmp_path):
    _roles()
    _suggestions(tmp_path)
    _case(tmp_path, "fenced", "fence_per_tile")
    verdict = {
        "per_capsule": [{"capsule": "fenced", "tiers": {"L2": "pass"}, "performance": _performance(4800)}],
        "feedback_brief": {},
    }
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {'fenced': {'workload_role': 'large_kernel'}}}))
    block = PR.build(tmp_path, verdict)
    (entry,) = block["largest_headroom"]
    assert entry["fence_overlap"]["fence_per_tile"] is True
    assert block["fence_bound"]["cases"][0]["test_case"] == "fenced"
    (top,) = PH._top(block)
    assert top["fences"].startswith("a full-device fence after every output tile (37 fences")
    assert top["mechanisms"][0] == "readout_frontier"
    rendered = "\n".join(PH.render({"schema": PH.SCHEMA, "top_headroom": [top]}))
    assert "fence after every output tile" in rendered


def test_unresolvable_target_and_absent_artifacts_add_nothing(tmp_path):
    folder = tmp_path / "selfcheck_out" / "x"
    folder.mkdir(parents=True)
    (folder / FO.TRACE).write_text(json.dumps(FIXTURE["fence_per_tile"]))
    (folder / FO.BUFFER).write_text(json.dumps({"target": "no_such_target_declared"}))
    assert FO.analyze(folder) is None
    assert FO.for_entry("x", tmp_path) is None and FO.for_entry("../x", tmp_path) is None


def test_console_cases_columns_show_tiles_per_fence_and_overlap():
    from merlin_experiments.tracking.console.sources import cases

    assert {"tile/fence", "overlap"} <= set(cases.COLUMNS)
    grade = {
        "feedback_brief": {
            "performance_priority": {
                "development_headroom": {
                    "largest_headroom": [
                        {
                            "test_case": "a",
                            "fence_overlap": {
                                "tiles_per_fence": 0.97,
                                "fence_per_tile": True,
                                "static_overlap_ratio": 0.0,
                                "reading": "r",
                            },
                        }
                    ],
                    "fence_bound": {"cases": [{"test_case": "b", "tiles_per_fence": 0.98, "reading": "s"}]},
                }
            }
        }
    }
    rows = {"c": {"performance": {"feature_counts": {"target.fences": 4, "target.class.store": 8}}}}
    found = cases._fences(grade, rows)
    assert found["a"] == {"per_fence": 0.97, "overlap": 0.0, "reading": "r", "per_tile": True}
    assert found["b"]["per_tile"] is True and found["c"] == {"per_fence": 2.0}
    assert cases._per_fence(found["a"]) == "0.97!" and cases._per_fence(found["c"]) == "2.00"
