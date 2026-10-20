"""Full simultaneous buffers, row rounding and retained-source custody controls."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.phase0 import development_blocked_reference as B
from merlin_experiments.phase0 import development_capacity_materialize as M
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_envelopes as E
from merlin_experiments.phase0 import development_joint_capacity as J
from merlin_experiments.phase0 import development_sources as D


def facts():
    return {
        "facts": {
            "memories": [
                dict(
                    name="inputs",
                    banks=4,
                    depth=4096,
                    elem_bits=8,
                    row_bits_rtl=128,
                    bytes=262144,
                    view_kind="address_space",
                ),
                dict(
                    name="sums", banks=2, depth=512, elem_bits=32, row_bytes=64, bytes=65536, view_kind="address_space"
                ),
            ]
        }
    }


def prove(m, k, n, *, selected=None, granules=None, reserve=None):
    return J.prove(
        E.declaration("matmul", dict(M=m, K=k, N=n), ordinal=0),
        stores=selected or J.selected_stores(facts(), operand_memory="inputs", accumulator_memory="sums"),
        axis_granules=granules or dict(M=16, N=16, K=16),
        reserved_bytes=reserve or dict(operand=0, accumulator=0),
    )


def test_adjacent_retained_scale_boundaries_cover_all_three_joint_regimes():
    double, single = prove(512, 240, 16), prove(512, 256, 16)
    last, spill = prove(1024, 240, 16), prove(1024, 256, 16)
    assert [x["regime"] for x in (double, single, last, spill)] == [
        "fits_double",
        "fits_single",
        "fits_single",
        "spills",
    ]
    assert double["placements"]["2"]["at_last_fit"]
    assert single["placements"]["2"]["at_first_spill"]
    assert last["placements"]["1"]["at_last_fit"]
    assert spill["placements"]["1"]["at_first_spill"]
    assert double["placements"]["2"]["stores"]["accumulator"]["required_bytes"] == 65536
    assert spill["placements"]["1"]["stores"]["operand"]["overflow_bytes"] == 4096
    for result in (double, single, last, spill):
        assert not result["physical_placement_proven"] and not result["double_buffer_execution_proven"]
        for variant in result["placements"].values():
            for store in variant["stores"].values():
                intervals = store["intervals"]
                assert all(a["end"] == b["begin"] for a, b in zip(intervals, intervals[1:]))
                assert intervals[-1]["end"] == store["required_bytes"]


def test_rectangular_rows_round_each_buffer_and_search_exact_k_boundary():
    selected = dict(
        operand=dict(capacity_bytes=640, row_bytes=8, element_bytes=1),
        accumulator=dict(capacity_bytes=1024, row_bytes=32, element_bytes=4),
    )
    result = prove(7, 5, 9, selected=selected, granules=dict(M=4, N=8, K=2))
    assert result["padded_shape"] == dict(M=8, K=6, N=16)
    assert result["useful_products"] == 7 * 5 * 9
    assert result["full_padded_products"] == 8 * 6 * 16
    variant = result["placements"]["2"]
    assert variant["stores"]["operand"]["required_bytes"] == 320
    assert variant["maximum_aligned_K_same_M_N"] == 12
    assert variant["next_aligned_K"] == 14
    assert prove(7, 12, 9, selected=selected, granules=dict(M=4, N=8, K=2))["placements"]["2"]["fits"]
    assert not prove(7, 14, 9, selected=selected, granules=dict(M=4, N=8, K=2))["placements"]["2"]["fits"]


def test_accumulator_and_fixed_reservation_can_prevent_double_fit_independently():
    result = prove(512, 240, 16, reserve=dict(operand=16, accumulator=64))
    assert result["regime"] == "fits_single"
    two = result["placements"]["2"]
    assert two["stores"]["operand"]["fits"]
    assert two["stores"]["operand"]["required_bytes"] == 253440 + 16
    assert two["stores"]["accumulator"]["overflow_bytes"] == 64
    assert two["maximum_aligned_K_same_M_N"] == 0


def test_physical_census_aliases_are_retained_but_never_summed_into_logical_capacity():
    document = facts()
    document["facts"]["census"] = {
        "physical_memories": [
            dict(
                name="backing",
                backs="sums",
                bytes=32768,
                copies=2,
                copies_note="original physical representation has independently scoped copies",
            ),
        ]
    }
    selected = J.selected_stores(document, operand_memory="inputs", accumulator_memory="sums")
    accumulator = selected["accumulator"]
    assert accumulator["capacity_bytes"] == 65536
    assert accumulator["fact_path"] == "facts.memories[1]"
    assert accumulator["physical_census_views"][0]["selected_fact"]["copies"] == 2
    assert accumulator["representation_discrepancies"][0]["physical_view_bytes"] == 32768
    assert accumulator["independent_ports_and_overlap"].startswith("UNKNOWN")


@pytest.mark.parametrize("change", ["duplicate", "physical_only", "wrong_bytes", "bad_width", "bool_banks"])
def test_unproved_or_inconsistent_selected_memory_geometry_refuses(change):
    document = facts()
    row = document["facts"]["memories"][0]
    if change == "duplicate":
        document["facts"]["memories"].append(dict(row))
    elif change == "physical_only":
        row["view_kind"] = "physical_census_alias"
    elif change == "wrong_bytes":
        row["bytes"] += 1
    elif change == "bad_width":
        row["elem_bits"] = 20
    else:
        row["banks"] = True
    with pytest.raises(ValueError):
        J.selected_stores(document, operand_memory="inputs", accumulator_memory="sums")


@pytest.mark.parametrize("reserve", [dict(operand=1, accumulator=0), dict(operand=0, accumulator=-1)])
def test_nonrow_or_negative_reservations_refuse(reserve):
    with pytest.raises(ValueError, match="complete rows"):
        prove(7, 5, 9, reserve=reserve)


def fixture_source(tmp_path, *, reference=True):
    source = tmp_path / "source"
    manifest = F.freeze(
        [
            dict(declaration=E.declaration("matmul", dict(M=7, K=17, N=11), ordinal=0), role="development_control"),
            dict(status="source_refused", reason="original unsupported numerical contract", role="functional_guard"),
        ],
        source,
        provenance={},
        budgets=dict(
            max_members=3,
            max_source_bytes=65536,
            max_reference_elements=100000 if reference else 1,
            max_reference_work=100000 if reference else 1,
            max_total_reference_work=100000,
            max_total_reference_elements=100000,
        ),
    )
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts()))
    return source, manifest, path


def run(source, facts_path, output, **changed):
    kwargs = dict(
        manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        ordinals=[0],
        facts_path=facts_path,
        facts_sha256=hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        operand_memory="inputs",
        accumulator_memory="sums",
        axis_granules=dict(M=4, N=8, K=2),
        reserved_bytes=dict(operand=0, accumulator=0),
        output=output,
        block=dict(M=3, N=4, K=5),
        budgets=dict(
            max_member_work=100000,
            max_total_work=100000,
            max_input_bytes=100000,
            max_output_bytes=100000,
            max_workspace_bytes=8 << 20,
            member_timeout_s=10,
        ),
    )
    kwargs.update(changed)
    return M.materialize(source, **kwargs)


def test_actual_source_inputs_complete_reference_and_original_roster_are_retained(tmp_path):
    source, original, selected_facts = fixture_source(tmp_path)
    report = run(source, selected_facts, tmp_path / "view")
    row = report["members"][0]
    assert row["status"] == "complete_source_inputs_reference_materialized"
    assert row["computed_products"] == 7 * 17 * 11 and row["complete_values"] == 77
    assert row["original_request_role"] == "development_control"
    assert report["members"][1]["original_reason"] == "original unsupported numerical contract"
    assert report["members"][1]["status"] == "NOT_SELECTED"
    assert len(report["members"]) == len(original["members"])
    assert Path(row["capsule_entry"]["loader"]).is_file()
    with np.load(tmp_path / "view/member-0000/original-inputs-and-reference.npz") as values:
        assert np.unique(values["input_0"]).size > 1
        np.testing.assert_array_equal(
            values["output_0"], np.load(tmp_path / "view/member-0000/blocked-reference/output-0.npy")
        )
    assert F.verify(source, manifest_sha256=report["original_manifest"]["sha256"]) == original
    assert not any(
        report[k]
        for k in (
            "phase0_admitted",
            "performance_objective_changed",
            "roles_changed",
            "physical_capacity_qualified",
            "timing_qualified",
        )
    )


def test_incomplete_or_cost_denied_member_is_not_dropped(tmp_path):
    source, original, selected_facts = fixture_source(tmp_path, reference=False)
    report = run(source, selected_facts, tmp_path / "view")
    assert report["members"][0]["original_status"] == original["members"][0]["status"]
    assert report["members"][0]["status"] == "PREPARATION_REFUSED"
    assert "reference absent" in report["members"][0]["reason"]
    assert len(report["members"]) == 2


def test_explicit_work_budget_denial_preserves_source_and_conditional_proof(tmp_path):
    source, original, selected_facts = fixture_source(tmp_path)
    report = run(
        source,
        selected_facts,
        tmp_path / "view",
        budgets=dict(
            max_member_work=1,
            max_total_work=1,
            max_input_bytes=100000,
            max_output_bytes=100000,
            max_workspace_bytes=8 << 20,
            member_timeout_s=10,
        ),
    )
    row = report["members"][0]
    assert row["status"] == "PREPARATION_REFUSED"
    assert "cost admission denied" in row["reason"]
    assert row["joint_capacity"]["useful_products"] == original["members"][0]["description"]["scalar_products"]
    assert not list((tmp_path / "view").glob("member-*"))
    assert len(report["members"]) == len(original["members"])


def test_independent_complete_output_difference_cannot_materialize_as_success(tmp_path, monkeypatch):
    source, _, selected_facts = fixture_source(tmp_path)
    original = B.materialize

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        output = np.load(result["outputs"][0], mmap_mode="r+")
        output[-1, -1] += 1
        output.flush()
        return result

    monkeypatch.setattr(B, "materialize", changed)
    report = run(source, selected_facts, tmp_path / "view")
    assert report["members"][0]["status"] == "PREPARATION_REFUSED"
    assert "complete blocked output differs" in report["members"][0]["reason"]


def test_stale_original_source_or_facts_pin_refuses_before_output(tmp_path):
    source, _, selected_facts = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="facts differ"):
        run(source, selected_facts, tmp_path / "view", facts_sha256="f" * 64)
    loader = next(source.rglob("loader.py"))
    loader.chmod(0o644)
    loader.write_text("changed")
    with pytest.raises(ValueError, match="source/reference bytes changed"):
        run(source, selected_facts, tmp_path / "view")


def test_complete_original_inputs_are_checked_independently_of_matching_output(tmp_path):
    source, original, _ = fixture_source(tmp_path)
    member = original["members"][0]
    reference = source / member["reference"]
    with np.load(reference) as archive:
        values = {key: archive[key] for key in archive.files}
    values["input_1"][-1, -1] += 1
    changed = tmp_path / "changed-input.npz"
    np.savez(changed, **values)
    with pytest.raises(ValueError, match="complete original seed"):
        M._check_reference_inputs(changed, member["description"], block=dict(M=3, N=4, K=5))
    np.savez(changed, **values, foreign=np.zeros((1,), dtype=np.int8))
    with pytest.raises(ValueError, match="complete typed payload"):
        M._check_reference_inputs(changed, member["description"], block=dict(M=3, N=4, K=5))


def test_factory_evolution_never_rewrites_the_pinned_original_source_epoch(tmp_path, monkeypatch):
    source, original, selected_facts = fixture_source(tmp_path)
    render = D.render
    monkeypatch.setattr(D, "render", lambda declaration: render(declaration) + "\n# later factory revision\n")
    report = run(source, selected_facts, tmp_path / "view")
    row = report["members"][0]
    assert row["status"] == "complete_source_inputs_reference_materialized"
    assert not row["current_factory_render_matches_retained"]
    staged = Path(row["capsule_entry"]["loader"]).read_bytes()
    assert hashlib.sha256(staged).hexdigest() == original["members"][0]["loader_identity"]["sha256"]
    assert not staged.endswith(b"# later factory revision\n")
    assert row["source_execution_agreement"].startswith("UNKNOWN")
