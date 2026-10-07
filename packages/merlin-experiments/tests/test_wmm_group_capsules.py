"""The cell machine's real measurer: one-group programs for both arms, each arm's recipe the whole-model
build options of that arm, every package program's whole ELF held to the rule, timed on the emulator.
(The build and the emulator are faked here; the GSIM validation of the real path is recorded in the
commit that ports it.)"""

from __future__ import annotations

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cells as CELLS
from merlin_experiments.phase2.whole_model_measured import group_capsules as GC
from merlin_experiments.phase2.whole_model_measured import worker as W

from merlin.perf import whole_model_verdict as V

OPTIONS = {
    "machine": "emu",
    "header": "/h/params.h",
    "prohibited_roles": ["loop_descriptor"],
    "phase0_recipe": "/r/recipe.yaml",
    "harness_overrides": ["/h/vendor.h"],
}


def test_an_arm_is_its_whole_model_build_options_and_the_bar_carries_no_rule():
    arm = GC.arm_from_options(OPTIONS, name=GC.ARM_PACKAGE)
    assert arm.prohibited_roles == ("loop_descriptor",) and arm.phase0_recipe == "/r/recipe.yaml"
    assert arm.harness_overrides == ("/h/vendor.h",)
    with pytest.raises(GC.GroupCapsuleError, match="not the bar"):
        GC.arm_from_options(OPTIONS, name=GC.ARM_REFERENCE)
    with pytest.raises(GC.GroupCapsuleError, match="header"):
        GC.arm_from_options({"machine": "emu"}, name=GC.ARM_PACKAGE)
    measurer = GC.GroupProgramMeasurer({"target": "toy", "build_options": OPTIONS})
    assert measurer.arm(GC.ARM_REFERENCE).prohibited_roles == ()  # the package recipe without its rule
    assert measurer.arm(GC.ARM_REFERENCE).header == "/h/params.h"


@pytest.fixture
def faked(monkeypatch):
    built = []

    def build(arm, groups, *, package_dir, model_capsule, target, out, **kw):
        built.append((arm.name, tuple(groups), arm.phase0_recipe, str(package_dir) if package_dir else None))
        return {
            int(g): {
                "group": int(g),
                "arm": arm.name,
                "linked": "submission" if arm.name == GC.ARM_PACKAGE and g != 9 else "vendor",
                "cause": "package_declined" if g == 9 else None,
                "elf": f"/g{g}.elf",
                "elf_sha256": "e" * 64,
            }
            for g in groups
        }

    monkeypatch.setattr(GC, "build_arm_programs", build)
    monkeypatch.setattr(
        GC, "isa_scan", lambda record, *, target, roles: {"clean": True, "summary": {}, "prohibited": {"8": "LOOP_0"}}
    )
    monkeypatch.setattr(
        GC,
        "time_on_gsim",
        lambda records, **kw: {
            label: {"status": "graded", "cycles": 10 * int(r["group"]), "correct": True} for label, r in records.items()
        },
    )
    return built


def _cell_job(tmp_path, cell):
    spec = {
        "kind": "cell",
        "target": "toy",
        "measurer": "merlin_experiments.phase2.whole_model_measured.group_capsules:cell_measurer",
        "cell": cell,
        "timing": {"registry_machine": "emu", "max_cycles": 1000},
    }
    return FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=spec,
        builder=FX.write_builder(tmp_path),
        build_options=dict(OPTIONS),
        instruction_policy=FX.sealed_policy(),
    )


def test_a_cell_is_measured_by_the_jobs_own_recipe_through_the_real_measurer(tmp_path, faked):
    result = W.work(_cell_job(tmp_path, {"id": "c", "model_capsule": "/m", "groups": [1, 2]}))
    assert result["timing_status"] == V.TIMING_MEASURED and result["objective_cycles"] == 30
    (call,) = faked
    assert call[0] == GC.ARM_PACKAGE and call[1] == (1, 2) and call[2] == "/r/recipe.yaml" and call[3]


def test_a_group_the_package_declines_is_refused_untimed_through_the_real_measurer(tmp_path, faked):
    result = W.work(_cell_job(tmp_path, {"id": "c", "model_capsule": "/m", "groups": [1, 9]}))
    assert result["timing_status"] == V.TIMING_REFUSED and "g9" in result["refusal"]
    (g9,) = [g for g in result["verdict"]["groups"] if g["group"] == "9"]
    assert "package_declined" in g9["refusal"] and g9.get("cycles") is None


def test_the_reference_arm_builds_with_the_library_and_no_rule(tmp_path, faked):
    spec = {
        "kind": "cell",
        "target": "toy",
        "measurer": "merlin_experiments.phase2.whole_model_measured.group_capsules:cell_measurer",
        "cell": {"id": "c", "model_capsule": "/m", "groups": [1]},
        "reference_build_options": {k: v for k, v in OPTIONS.items() if k != "prohibited_roles"},
    }
    document = CELLS.reference_cell(spec, target="toy", out=tmp_path / "ref")
    assert document["timing_status"] == V.TIMING_MEASURED and faked[0][0] == GC.ARM_REFERENCE


def test_the_validation_path_refuses_an_arm_with_no_enforceable_sealed_policy(tmp_path, faked):
    """Arms from two whole-model jobs: the package arm is held to the job's sealed policy, and an arm
    under roles no sealed policy resolved is refused before any program is built."""
    import json

    jobs = {}
    for name, policy in (("package", FX.sealed_policy()), ("reference", None)):
        options = dict(OPTIONS) if name == "package" else {k: v for k, v in OPTIONS.items() if k != "prohibited_roles"}
        jobs[name] = tmp_path / f"{name}.json"
        jobs[name].write_text(json.dumps({"build_options": options, "instruction_policy": policy}))
    arms = GC.arms_from_jobs(jobs["package"], jobs["reference"])
    assert arms[GC.ARM_PACKAGE].instruction_policy == FX.sealed_policy()
    document = GC.measure_on_gsim(arms, [1], package_dir=tmp_path, model_capsule="/m", target="toy", out=tmp_path / "o")
    assert all(r["status"] == "graded" for r in document["rows"])
    unsealed = {**arms, GC.ARM_PACKAGE: GC.arm_from_options(OPTIONS, name=GC.ARM_PACKAGE)}
    faked.clear()
    with pytest.raises(GC.GroupCapsuleError, match="sealed instruction policy"):
        GC.measure_on_gsim(unsealed, [1], package_dir=tmp_path, model_capsule="/m", target="toy", out=tmp_path / "u")
    assert faked == []  # nothing was built


def test_the_validation_table_keeps_each_signed_offset_and_flags_a_systematic_one():
    measured = [
        {"group": "1", "arm": "package", "device": "elaborated_rtl", "cycles": 102, "correct": True},
        {"group": "2", "arm": "package", "device": "elaborated_rtl", "cycles": 210, "correct": True},
    ]
    in_model = {("package", "elaborated_rtl"): {"groups": {"1": 100, "2": 200}, "source": "r.json"}}
    table = GC.validation_table(measured, in_model, tolerance=0.03)
    assert [r["offset_cycles"] for r in table["rows"]] == [2, 10]
    assert table["per_device"]["elaborated_rtl"]["all_same_sign"] is True
    assert table["passed"] is False  # 5% is outside 3%
