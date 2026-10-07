"""Fused regions are on by default for a closed model, with an operator opt-out; the decision is the
prepared run's receipt, and coverage counts a claimed region's members only while its kernel answers."""

from __future__ import annotations

from pathlib import Path

import pytest
from merlin_experiments.phase2.whole_model_measured import cli as MCLI
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import feedback as F
from merlin_experiments.phase2.whole_model_measured import gates as G

from merlin.perf import whole_model_build as WMB


@pytest.fixture()
def closure(monkeypatch):
    """The model's closure, as the test sets it (``state["open"]``); every capsule loads."""
    from merlin.perf import whole_model_capsule, whole_model_open

    state = {"open": False}
    monkeypatch.setattr(whole_model_open, "is_open_model", lambda capsule, target: state["open"])
    monkeypatch.setattr(whole_model_capsule, "load_model_capsule", lambda path: path)
    return state


def _document(tmp_path: Path, **options) -> dict:
    capsule = tmp_path / "capsule"
    capsule.mkdir(exist_ok=True)
    return {
        "schema": C.CONFIG_SCHEMA,
        "store": str(tmp_path / "store"),
        "screen": {"build_options": {"model_capsule": str(capsule), **options}},
    }


def test_a_closed_models_package_may_claim_fused_regions_by_default(tmp_path, closure):
    prepared = C.prepare_document(_document(tmp_path), target="toy")
    assert prepared["screen"]["build_options"]["allow_regions"] is True
    decision = prepared["fused_region_decision"]
    assert decision["default"] == "allowed" and decision["sections"]["screen"]["allowed"] is True
    assert "allow_passes" not in prepared["screen"]["build_options"]  # passes stay what the policy derives


def test_the_operator_opts_out_for_the_run_or_for_one_section(tmp_path, closure):
    run_wide = C.prepare_document({**_document(tmp_path), C.FUSED_REGIONS: False}, target="toy")
    assert run_wide["screen"]["build_options"]["allow_regions"] is False
    assert run_wide["fused_region_decision"]["sections"]["screen"]["why"] == "opted out by the operator"
    one_section = C.prepare_document(_document(tmp_path, allow_regions=False), target="toy")
    assert one_section["screen"]["build_options"]["allow_regions"] is False
    with pytest.raises(C.ConfigError, match="true or false"):
        C.prepare_document({**_document(tmp_path), C.FUSED_REGIONS: "maybe"}, target="toy")


def test_an_open_model_never_claims_a_region_and_a_declaration_otherwise_is_refused(tmp_path, closure):
    closure["open"] = True
    prepared = C.prepare_document(_document(tmp_path), target="toy")
    assert prepared["screen"]["build_options"]["allow_regions"] is False
    assert "open" in prepared["fused_region_decision"]["sections"]["screen"]["why"]
    with pytest.raises(C.ConfigError, match="contradicts"):
        C.prepare_document(_document(tmp_path, allow_regions=True), target="toy")


def test_an_underivable_closure_leaves_regions_off_and_says_why(tmp_path, monkeypatch):
    from merlin.perf import whole_model_capsule

    def refuse(path):
        raise WMB.WholeModelBuildError("capsule 'x' declares no weights, so it is not a model capsule")

    monkeypatch.setattr(whole_model_capsule, "load_model_capsule", refuse)
    prepared = C.prepare_document(_document(tmp_path), target="toy")
    assert "allow_regions" not in prepared["screen"]["build_options"]
    assert "could not be derived" in prepared["fused_region_decision"]["sections"]["screen"]["why"]


def test_the_command_line_opt_out_reaches_the_prepared_config():
    assert MCLI._regions_opt_out({"a": 1}, True) == {"a": 1, C.FUSED_REGIONS: False}
    assert MCLI._regions_opt_out({"a": 1}, False) == {"a": 1}
    parser = MCLI._parser()
    assert parser.parse_args(
        ["prepare", "--target", "t", "--method", "m", "--why", "w", "--no-fused-regions"]
    ).no_fused_regions


# ------------------------------------------------------------------------------------- coverage


def _build(*rows):
    return {"groups": [dict(r) for r in rows]}


REGION = {"member_groups": [1, 2], "role": "internal", "id": "r"}


def test_a_linked_regions_members_all_count_as_package_authored():
    build = _build(
        {"group": "1", "op": "conv2d", "on": "package", "region": REGION, "graded_at": 2},
        {"group": "2", "op": "conv2d", "on": "package", "region": {**REGION, "role": "boundary"}},
        {"group": "3", "op": "add", "on": "vendor"},
    )
    reference = {"verdict": {"groups": [{"group": g, "cycles": c} for g, c in (("1", 50), ("2", 30), ("3", 20))]}}
    authored = F.package_authored({"build": build}, reference)
    assert authored["groups"] == ["1", "2"] and authored["priced_share"] == 0.8
    gate = {"price": {"1": 50, "2": 30, "3": 20}, "floor_groups": ["1", "2"], "floor_share": 0.8}
    assert G.coverage_gate({"coverage_gate": gate, "role": "candidate"}, build, None) is None


def test_a_region_whose_kernel_is_not_linked_counts_none_of_its_members():
    """MUTATION of the above: the same claim with the boundary's kernel unlinked credits nothing -- the
    builder settles every internal member back to the library, and the coverage gate then refuses."""
    rows = [
        {"group": 1, "op": "conv2d", "on": "package", "region": REGION, "graded_at": 2},
        {
            "group": 2,
            "op": "conv2d",
            "on": "vendor",
            "cause": "object_failed",
            "region": {**REGION, "role": "boundary"},
        },
    ]
    WMB.settle_regions(rows)
    assert rows[0]["on"] == "vendor" and rows[0]["cause"] == WMB.REGION_UNLINKED
    build = _build(*({**r, "group": str(r["group"])} for r in rows))
    gate = {"price": {"1": 50, "2": 30, "3": 20}, "floor_groups": ["1", "2"], "floor_share": 0.8}
    job = {"coverage_gate": gate, "role": "candidate", "package_sha256": "p" * 64}
    refused = G.coverage_gate(job, build, None)
    assert refused is not None and refused["coverage_regression"]["declined_groups"] == ["1", "2"]


def test_a_region_is_never_offered_across_a_declined_group():
    """A caller-declined group cannot sit inside a claimed region (the statement's own rule)."""
    import inspect

    from merlin.llvmlower import whole_program as WP

    source = inspect.getsource(WP.whole_program_buffer)
    assert '_is_declined(int(ctx["group"].index), str(ctx["entry"].get("op")), decline) for ctx in window' in source


def test_a_one_group_programs_region_is_linked_only_while_its_boundary_is_the_packages():
    from merlin.perf.whole_model_group_timing import linked_region

    member = {"group": 1, "on": "package", "region": {**REGION}}
    assert linked_region(member) == {"members": [1, 2], "boundary": 2, "id": "r"}
    assert linked_region({**member, "on": "vendor"}) is None
    assert linked_region({"group": 1, "on": "package"}) is None
    assert linked_region({**member, "region": {"member_groups": [1]}}) is None


def test_a_region_step_reads_only_what_its_members_read_from_outside(tmp_path):
    import numpy as np

    from merlin.perf.whole_model_group_timing import _one_group_model

    model = {
        "steps": [
            {"group": 0, "out": "A", "lhs": "IN"},
            {
                "group": 2,
                "kind": "region",
                "out": "C",
                "members": [{"group": 1, "out": "B", "lhs": "A"}, {"group": 2, "out": "C", "lhs": "B"}],
            },
        ],
        "buffers": [{"name": n, "ctype": "elem_t", "elements": 2} for n in ("IN", "A", "B", "C")],
        "arrays": {},
    }
    one = _one_group_model(model, 2, {"A": [1, 2]}, {"elem_t": "<i1"})
    assert one["embedded_inputs"] == ["A"] and set(one["arrays"]) == {"A"}
    assert np.asarray(one["arrays"]["A"]).tolist() == [1, 2]
