"""What the agent reads from a whole-model measurement: per-group ours-vs-reference on ONE machine, the
instruction census beside each gap holder, the derived roofline, and the transfer check."""

from __future__ import annotations

from merlin_experiments.phase2.whole_model_measured import feedback as F
from merlin_experiments.phase2.whole_model_measured import transfer as T

from merlin.perf import whole_model_verdict as V


def _result(cycles, *, device="d" * 64, census=None, status=V.TIMING_MEASURED, kinds=("conv", "matmul")):
    groups = [
        {"group": str(i + 1), "kind": kind, "cycles": c, "correct": True, "state": "correct"}
        for i, (kind, c) in enumerate(zip(kinds, cycles, strict=True))
    ]
    return {
        "package_sha256": "p",
        "timing_status": status,
        "objective_cycles": sum(cycles),
        "device": {"machine": "firesim", "binary_sha256": device, "artifact": "board"},
        "machine": {"target": ""},
        "verdict": {"timing_status": status, "whole_window_cycles": sum(cycles), "groups": groups},
        "build": {
            "groups": [{"group": g["group"], "op": g["kind"], "on": "package"} for g in groups],
            "isa_census": {"per_group": census or {}},
        },
    }


def test_a_reference_from_another_device_is_never_a_divisor():
    document = F.compare(_result([300, 200]), _result([100, 100], device="e" * 64))
    assert document["reference"]["admissible"] is False and "distance_to_bar" not in document


def test_gap_holders_carry_the_census_and_are_ranked_by_their_gap():
    census = {"1": {"total": 12, "by_kind": {"compute": 9, "mvin": 3}}}
    document = F.compare(_result([300, 200], census=census), _result([100, 150]))
    holders = document["gap_holders"]
    assert [h["group"] for h in holders] == ["1", "2"] and holders[0]["instruction_census"]["total"] == 12
    assert document["distance_to_bar"]["gap_cycles"] == 250
    text = F.render(document)
    assert "issued 12 instructions" in text and "compute=9" in text


def test_a_gap_holder_carries_no_compute_only_bound():
    """The derived roofline (``roofline_gaps``) is the one bound the feedback shows the agent. An
    older MACs-over-array-peak figure beside it disagreed with the roofline for the same group (an
    FC layer: 8,000 vs 128,128 cycles) and was dropped rather than shown alongside a second answer."""
    document = F.compare(_result([900, 200]), _result([100, 150]))
    assert "headroom" not in document["gap_holders"][0]
    assert not any("derived bound" in line for line in F.render(document).splitlines())


def test_an_invalid_measurement_shows_its_distance_but_never_as_an_achievement():
    document = F.compare(_result([50, 50], status=V.TIMING_MEASURED_INVALID), _result([100, 100]))
    assert (
        document["distance_to_bar"]["achieved"] is False
        and document["distance_to_bar"]["counts_as_achievement"] is False
    )


def test_a_gain_that_does_not_carry_to_a_held_out_model_is_flagged_by_kind():
    primary = _result([80, 200])
    reference = _result([100, 200])
    held = {"heldout": (_result([105, 190]), _result([100, 200]))}
    report = T.build_transfer_report(
        primary_name="primary", primary_result=primary, primary_reference=reference, held_out_results=held
    )
    assert report["overfit_suspected"] and [r["kind"] for r in report["flagged"]] == ["conv"]
    untested = T.transfer_report("primary", [{"kind": "conv", "ratio": 0.5}], {"heldout": []})
    assert untested["checked"][0]["status"] == "untested" and not untested["overfit_suspected"]


def test_feedback_ranks_every_group_by_its_roofline_gap_even_one_faster_than_the_reference():
    """The reference is another implementation's cycles; the roofline is the machine's. A group that
    beats the reference stays in view, ranked by how far it still is from the machine's floor."""

    def result(cycles, diagnostics=None):
        groups = [{"group": g, "kind": "matmul", "cycles": c, "correct": True, "state": "correct"} for g, c in cycles]
        return {
            "timing_status": V.TIMING_MEASURED,
            "verdict": {"groups": groups, "whole_window_cycles": sum(c for _, c in cycles)},
            **({"diagnostics": diagnostics} if diagnostics else {}),
        }

    def roof(bound):
        return {"roofline": {"status": "derived", "roofline_cycles": bound, "limiter": "compute"}}

    ours = result([("1", 300), ("2", 900)], {"per_group": {"1": roof(100), "2": roof(800)}})
    document = F.compare(ours, result([("1", 400), ("2", 850)]))
    gaps = document["roofline_gaps"]
    # g1 beats the reference (300 < 400) and is still the larger distance from the machine's floor.
    assert [g["group"] for g in gaps] == ["1", "2"] and gaps[0]["over_roofline"] == 3.0
    assert document["distance_to_roofline"] == {
        "ours_cycles": 1200,
        "roofline_cycles": 900,
        "ratio": 1.333,
        "groups": 2,
    }
    assert "DISTANCE TO THE MACHINE'S ROOFLINE" in "\n".join(F.roofline_lines(document))
    # A refuted roofline is not a bar, and a result with no diagnostics has no block at all.
    assert "roofline_gaps" not in F.compare(result([("1", 300)]), None)


# ----------------------------------------------------------------------- the defaults the agent reads


def test_every_cycle_count_in_the_feedback_carries_its_package_authored_share():
    reference = _result([100, 150])
    document = F.compare(_result([300, 200]), reference)
    text = F.render(document)
    assert "package-authored: 2/2 groups, 100.0% of the work" in text
    assert "whole-window 500 cycles (package-authored 2/2 groups, 100.0%)" in text
    declined = _result([300, 200])
    declined["build"]["groups"][0]["on"] = "vendor"
    assert "whole-window 500 cycles (package-authored 1/2 groups, 60.0%)" in F.render(F.compare(declined, reference))


def test_the_vendor_reference_is_labelled_context_only_never_a_target():
    document = F.compare(_result([300, 200]), _result([100, 150]))
    assert document["reference"]["role"] == "context_only"
    assert document["distance_to_bar"]["role"].startswith("context_only")
    text = F.render(document)
    assert "vendor reference, same machine (context only, not a target) 250" in text
    assert "VENDOR reference (context only" in text and "bar (reference, same machine)" not in text


class _Screen:
    target = "toy"
    machine = {"kind": "spike"}
    build_options: dict = {}
    root = None

    def jobs(self):
        return []


def test_the_whole_model_feedback_carries_the_derived_rooflines_by_default(tmp_path, monkeypatch):
    from merlin_experiments.phase2.whole_model_measured import objective as O
    from merlin_experiments.phase2.whole_model_measured import roofline as ROOF

    from merlin.perf import whole_model_capsule

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "iface.mlir").write_text("module {}")
    monkeypatch.setattr(
        whole_model_capsule, "load_model_capsule", lambda path: type("C", (), {"interface": capsule / "iface.mlir"})()
    )
    monkeypatch.setattr(
        ROOF, "roofline_machine", lambda target: {"array_rows": 4, "array_cols": 4, "memory": {}, "unresolved": {}}
    )
    monkeypatch.setattr(
        ROOF, "group_shapes", lambda capsule, target: {"1": {"op": "matmul", "extents": {"M": 8, "K": 8, "N": 8}}}
    )
    objective = O.WholeModelObjective(screen=_Screen(), screen_reference=None)
    objective.screen_reference = _result([100, 150])
    objective.config = {"screen": {"build_options": {"model_capsule": str(capsule)}}}
    document = objective.feedback(_result([300, 200]))
    assert document["roofline_gaps"][0]["group"] == "1" and document["roofline_gaps"][0]["roofline"] == 32
    assert "DISTANCE TO THE MACHINE'S ROOFLINE" in F.render(document)


def test_rooflines_that_cannot_be_derived_are_said_never_silently_absent():
    from merlin_experiments.phase2.whole_model_measured import objective as O

    objective = O.WholeModelObjective(screen=_Screen(), screen_reference=None)
    objective.screen_reference = _result([100, 150])
    objective.config = {}
    document = objective.feedback(_result([300, 200]))
    assert "no model capsule" in document["roofline_unavailable"]
    assert "no derived roofline in this feedback: the screen names no model capsule" in F.render(document)
    opted_out = O.WholeModelObjective(screen=_Screen(), screen_reference=None)
    opted_out.config = {"roofline_feedback": False}
    assert "turned roofline feedback off" in opted_out.feedback(_result([300, 200]))["roofline_unavailable"]
