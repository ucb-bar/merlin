"""Cell mode: form-perf groups timed alone on the emulator rank a candidate; a declined group is refused
before any emulator time; a win is transplanted to the whole model, where the board adjudicates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cells as CELLS
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import worker as W

from merlin.perf import whole_model_verdict as V


class FakeMeasurer:
    """Builds one record per group; ``declined`` groups are answered by the library; ``dirty`` groups
    issue a prohibited instruction; records what it was asked to time."""

    instances: list[FakeMeasurer] = []

    def __init__(self, spec):
        self.spec = spec
        self.timed: list[str] = []
        self.declined = set(spec.get("fake", {}).get("declined", ()))
        self.dirty = set(spec.get("fake", {}).get("dirty", ()))
        self.slow = {int(g): c for g, c in spec.get("fake", {}).get("slow", {}).items()}
        self.wrong = set(spec.get("fake", {}).get("wrong", ()))
        self.bounds: dict[str, int] = {}
        FakeMeasurer.instances.append(self)

    def programs(self, arm, groups, *, package_dir, member, out):
        model = member.get("label") or "objective"
        return {
            f"{arm}:{model}:g{g}": {
                "group": g,
                "linked": "vendor" if (arm == "reference" or g in self.declined) else "submission",
                "cause": "package_declined" if g in self.declined else None,
                "why": "NameError: name 'tile' is not defined" if g in self.declined else None,
                "kind": "conv2d",
                "elf": f"/{g}.elf",
            }
            for g in groups
        }

    def scan(self, record, *, roles):
        return {
            "clean": record["group"] not in self.dirty,
            "summary": {f"LOOP in g{record['group']}": 1},
            "census": {"total": 3},
            "prohibited": dict(self.spec.get("fake", {}).get("prohibited", {"8": "LOOP_A"})),
        }

    def time(self, records, *, member, max_cycles, out):
        self.timed += list(records)
        self.bounds[member.get("label") or "objective"] = max_cycles
        return {
            label: {
                "status": "graded",
                "cycles": self.slow.get(int(rec["group"]), 100 * int(rec["group"])),
                "correct": int(rec["group"]) not in self.wrong,
            }
            for label, rec in records.items()
        }


def make(spec):
    return FakeMeasurer(spec)


def _spec(**fake):
    return {
        "kind": "cell",
        "target": "toy",
        "measurer": f"{__name__}:make",
        "cell": {
            "id": "stem",
            "model_capsule": "/m",
            "groups": [1, 2],
            "held_out": [{"label": "heldout", "model_capsule": "/h", "groups": [5]}],
        },
        "timing": {"registry_machine": "emu", "max_cycles": 1000},
        "fake": fake,
    }


@pytest.fixture(autouse=True)
def _reset():
    FakeMeasurer.instances = []


def _job(tmp_path, spec, roles=("loop_descriptor",), *, sealed=True):
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=spec,
        builder=FX.write_builder(tmp_path),
        build_options={"prohibited_roles": list(roles)},
        instruction_policy=FX.sealed_policy(roles) if roles and sealed else None,
    )
    return job_dir


def test_a_clean_cell_sums_only_the_objective_groups(tmp_path):
    result = W.work(_job(tmp_path, _spec()))
    assert result["timing_status"] == V.TIMING_MEASURED and result["objective_cycles"] == 300
    assert [g["model"] for g in result["verdict"]["held_out"]] == ["heldout"]
    assert (
        result["device"]["timing_basis"] == "cell_sum" and result["build"]["isa_census"]["per_group"]["1"]["total"] == 3
    )


def test_a_declined_cell_group_is_refused_before_any_emulator_time(tmp_path):
    """A declined stem convolution once ran the library's host fallback to the emulator's cycle bound,
    holding both slots for hours, for a number nothing may use."""
    result = W.work(_job(tmp_path, _spec(declined=[1])))
    measurer = FakeMeasurer.instances[-1]
    assert not any(label.endswith(":objective:g1") for label in measurer.timed)
    assert result["timing_status"] == V.TIMING_REFUSED and "g1" in result["refusal"]
    assert "package_declined" in next(g for g in result["verdict"]["groups"] if g["group"] == "1")["refusal"]


def test_a_held_out_group_answered_by_the_library_is_reported_not_refused(tmp_path):
    result = W.work(_job(tmp_path, _spec(declined=[5])))
    assert result["timing_status"] == V.TIMING_MEASURED
    assert result["verdict"]["held_out"][0]["on"] == "vendor"


def test_a_cell_program_with_a_prohibited_instruction_is_refused_untimed(tmp_path):
    result = W.work(_job(tmp_path, _spec(dirty=[2])))
    assert result["isa_prohibited"]["summary"] == {"g2": 1} and result["refusal"].startswith("isa_prohibited")
    assert not any(label.endswith(":objective:g2") for label in FakeMeasurer.instances[-1].timed)


def test_a_cell_under_roles_no_sealed_policy_resolved_is_refused_before_any_program_is_built(tmp_path):
    result = W.work(_job(tmp_path, _spec(), sealed=False))
    assert result["timing_status"] == V.TIMING_REFUSED and "sealed instruction policy" in result["refusal"]
    assert not FakeMeasurer.instances  # refused before the measurer was even made


def test_a_cell_scan_that_prohibits_less_than_phase0_sealed_is_refused_untimed(tmp_path):
    """A scan whose prohibited set is empty (or misses a sealed instruction) checked nothing."""
    for name, prohibited in (("empty", {}), ("weaker", {"9": "OTHER"})):
        result = W.work(_job(tmp_path / name, _spec(prohibited=prohibited)))
        assert result["timing_status"] == V.TIMING_REFUSED and result["refusal"].startswith("isa_prohibited")
        assert not any(":objective:" in label for label in FakeMeasurer.instances[-1].timed)


def test_the_reference_arm_is_measured_unrestricted_by_the_same_path(tmp_path):
    document = CELLS.reference_cell(_spec(), target="toy", out=tmp_path / "ref")
    assert document["role"] == J.ROLE_REFERENCE and document["timing_status"] == V.TIMING_MEASURED
    assert json.loads((tmp_path / "ref" / "result.json").read_text())["objective_cycles"] == 300


def test_a_cell_win_is_transplanted_to_the_whole_model_screen_with_its_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: SimpleNamespace(pid=999_999_999))
    spec, pin = FX.write_builder(tmp_path)
    screen = S.MeasurementService(
        tmp_path / "screen", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )
    record = CELLS.transplant(
        screen, FX.package(tmp_path, "winner"), cell_id="stem", cell_digest="c" * 64, why="the cell's best"
    )
    log = [json.loads(line) for line in (tmp_path / "screen" / "transplants.jsonl").read_text().splitlines()]
    assert log[0]["cell"] == "stem" and log[0]["whole_model_job"] == record["whole_model_job"]
    with pytest.raises(CELLS.CellError):
        CELLS.transplant(screen, FX.package(tmp_path, "w2"), cell_id="stem", cell_digest="c", why="")


def test_a_cell_without_groups_is_refused():
    with pytest.raises(CELLS.CellError):
        CELLS.members({"model_capsule": "/m"})


# ------------------------------------------------------------------------- collateral (other forms)
def _collateral_spec(**fake):
    spec = _spec(**fake)
    spec["cell"]["collateral"] = {
        "tolerance": 0.01,
        "groups": [
            {"group": 7, "form": "1x1 conv", "cycles": 700, "on": "package"},
            {"group": 9, "form": "residual add", "cycles": 900, "on": "vendor"},
        ],
    }
    return spec


def test_collateral_regressions_refuse_and_a_clean_candidate_passes(tmp_path):
    """A stem-conv win once made every 1x1 contraction decline, and the whole model 700x slower,
    while the cell's own groups reported the win."""
    clean = W.work(_job(tmp_path / "a", _collateral_spec()))
    assert clean["timing_status"] == V.TIMING_MEASURED and clean["objective_cycles"] == 300  # never summed in
    assert {r["group"]: r["state"] for r in clean["verdict"]["collateral"]} == {"7": "ok", "9": "ok"}
    # The bound is derived from the worst baseline, not the cell's own bound.
    assert FakeMeasurer.instances[-1].bounds["collateral"] == int(900 * 3.0 + 2_000_000)

    slower = W.work(_job(tmp_path / "b", _collateral_spec(slow={"7": 720})))
    assert slower["timing_status"] == V.TIMING_REFUSED
    assert "COLLATERAL REGRESSION" in slower["refusal"] and "720 cycles against the baseline's 700" in slower["refusal"]

    wrong = W.work(_job(tmp_path / "c", _collateral_spec(wrong=[9])))
    assert wrong["timing_status"] == V.TIMING_REFUSED and "g9: output is wrong" in wrong["refusal"]


def test_a_representative_the_package_stopped_answering_is_refused_untimed_quoting_its_own_error(tmp_path):
    result = W.work(_job(tmp_path, _collateral_spec(declined=[7])))
    assert result["timing_status"] == V.TIMING_REFUSED
    assert "NameError: name 'tile' is not defined" in result["refusal"]
    assert not any(label.endswith(":collateral:g7") for label in FakeMeasurer.instances[-1].timed)
    # g9 was the library's in the baseline too: answered by the library now is not a regression.
    assert next(r for r in result["verdict"]["collateral"] if r["group"] == "9")["state"] == "ok"


def test_a_declined_objective_group_quotes_the_packages_own_reason(tmp_path):
    result = W.work(_job(tmp_path, _spec(declined=[1])))
    assert (
        "NameError: name 'tile' is not defined"
        in next(g for g in result["verdict"]["groups"] if g["group"] == "1")["refusal"]
    )


def test_representatives_are_one_per_other_form_by_largest_share():
    forms = [
        {"group": 1, "form_text": "stem"},
        {"group": 2, "form_text": "1x1"},
        {"group": 3, "form_text": "1x1"},
        {"group": 4, "form_text": "add"},
    ]
    picked = CELLS.collateral_representatives(forms, [1], {"2": 10, "3": 30})
    assert picked == [{"group": 3, "form": "1x1", "members": 2}, {"group": 4, "form": "add", "members": 1}]


def test_the_baseline_block_is_measured_from_a_baseline_package_and_refuses_an_untimed_representative(tmp_path):
    reps = [{"group": 7, "form": "1x1"}, {"group": 9, "form": "add"}]
    block = CELLS.measure_collateral_baseline(
        _spec(declined=[9]),
        reps,
        baseline_package=FX.package(tmp_path, "seed"),
        target="toy",
        out=tmp_path / "b",
        tolerance=0.01,
    )
    assert block["tolerance"] == 0.01 and block["baseline_package_sha256"]
    # g9 is the library's in the baseline: it measures no code a candidate can change, so it is left out.
    assert {r["group"]: (r["cycles"], r["on"]) for r in block["groups"]} == {7: (700, "package")}
    assert [(r["group"], r["cause"]) for r in block["not_package"]] == [(9, "package_declined")]
    with pytest.raises(CELLS.CellError, match="g7"):
        CELLS.measure_collateral_baseline(
            _spec(wrong=[7]),
            reps,
            baseline_package=FX.package(tmp_path, "s2"),
            target="toy",
            out=tmp_path / "c",
            tolerance=0.01,
        )


def test_a_candidates_objective_programs_are_bounded_from_the_cells_own_baseline(tmp_path):
    """A deadlocked candidate whose groups take at most 0.6M cycles ran every program to the flat 60M
    bound (5.5 h each) before it could be refused; the bound now comes from the cell's own baseline.
    Held-out groups and the reference arm, which have no baseline, keep the flat bound."""
    spec = _spec()
    spec["cell"][CELLS.BASELINE] = {"groups": [{"group": 1, "cycles": 100}, {"group": 2, "cycles": 600}]}
    result = W.work(_job(tmp_path, spec))
    assert result["timing_status"] == V.TIMING_MEASURED
    bounds = FakeMeasurer.instances[-1].bounds
    assert bounds["objective"] == int(600 * 3.0 + 2_000_000) and bounds["heldout"] == 1000
    rows = json.loads(next((tmp_path / "store").rglob("cell_rows.json")).read_text())
    basis = {(r["model"], r["group"]): (r["max_cycles"], r["max_cycles_basis"]) for r in rows}
    assert basis[(None, 2)] == (2_001_800, "derived from cell.baseline: worst group x factor + setup")
    assert basis[("heldout", 5)][0] == 1000 and basis[("heldout", 5)][1].startswith("flat")
    # The cell's baseline block is the bound's: it never reaches a member as the collateral's bar.
    assert CELLS.BASELINE not in CELLS.members(spec["cell"])[0]

    reference = CELLS.reference_cell(spec, target="toy", out=tmp_path / "ref")
    assert FakeMeasurer.instances[-1].bounds["objective"] == 1000
    assert reference["objective_cycles"] == 300


def test_an_older_cell_spec_with_no_baseline_keeps_the_flat_bound(tmp_path):
    W.work(_job(tmp_path, _spec()))
    assert FakeMeasurer.instances[-1].bounds["objective"] == 1000
    rows = json.loads(next((tmp_path / "store").rglob("cell_rows.json")).read_text())
    assert {r["max_cycles_basis"] for r in rows if r["model"] is None} == {
        "flat: the cell carries no baseline of its own groups"
    }


def test_the_cell_baseline_is_every_group_timed_correctly_or_none(tmp_path):
    block = CELLS.measure_cell_baseline(
        _spec(), baseline_package=FX.package(tmp_path, "seed"), target="toy", out=tmp_path / "b"
    )
    assert [(r["group"], r["cycles"], r["on"]) for r in block["groups"]] == [(1, 100, "package"), (2, 200, "package")]
    assert block["baseline_package_sha256"]
    # A bound read off a subset would understate the group it is missing: no block, the flat bound.
    assert (
        CELLS.measure_cell_baseline(
            _spec(wrong=[2]), baseline_package=FX.package(tmp_path, "s2"), target="toy", out=tmp_path / "c"
        )
        is None
    )
    assert CELLS.cell_baseline([1, 2], [{"group": 1, "status": "graded", "correct": True, "cycles": 5}]) is None


def test_a_held_out_member_may_not_take_the_collateral_label():
    spec = _spec()
    spec["cell"]["held_out"][0]["label"] = CELLS.COLLATERAL
    with pytest.raises(CELLS.CellError, match="collateral"):
        CELLS.members(spec["cell"])


class DiagnosingMeasurer(FakeMeasurer):
    """A FakeMeasurer that records which member carried the cell's diagnostics, and gives each program
    it timed with them an efficiency row."""

    seen: dict = {}

    def time(self, records, *, member, max_cycles, out):
        DiagnosingMeasurer.seen[member.get("label") or "objective"] = member.get(CELLS.DIAGNOSTICS)
        timed = super().time(records, member=member, max_cycles=max_cycles, out=out)
        if member.get(CELLS.DIAGNOSTICS):
            timed = {label: {**row, "efficiency": {"computes_issued": 4}} for label, row in timed.items()}
        return timed


def diagnosing(spec):
    return DiagnosingMeasurer(spec)


def test_the_objective_alone_carries_diagnostics_and_the_result_publishes_them(tmp_path):
    """The cell's diagnostics reach the OBJECTIVE member only -- never a held-out or a collateral
    member -- and the result carries each objective group's roofline beside what its program issued."""
    roofline = {"status": "derived", "roofline_cycles": 50, "limiter": "compute"}
    spec = _collateral_spec()
    spec["measurer"] = f"{__name__}:diagnosing"
    spec["cell"][CELLS.DIAGNOSTICS] = {"functional_model": {"command": ["fm"]}, "rooflines": {"1": roofline}}
    DiagnosingMeasurer.seen = {}
    result = W.work(_job(tmp_path, spec))
    seen = DiagnosingMeasurer.seen
    assert seen["objective"]["rooflines"] == {"1": roofline}
    assert seen["heldout"] is None and seen["collateral"] is None
    per_group = result["diagnostics"]["per_group"]
    assert per_group["1"] == {"roofline": roofline, "efficiency": {"computes_issued": 4}}
    assert per_group["2"]["roofline"] is None


def test_the_measurer_takes_an_efficiency_census_of_correct_objective_package_programs_only(tmp_path, monkeypatch):
    from merlin_experiments.phase2.whole_model_measured import group_capsules as GC

    timed = {
        "package_g1": {"status": "graded", "correct": True, "cycles": 9},
        "package_g2": {"status": "graded", "correct": False, "cycles": 9},
        "reference_g1": {"status": "graded", "correct": True, "cycles": 9},
    }
    monkeypatch.setattr(GC, "time_on_gsim", lambda records, **kw: {k: dict(v) for k, v in timed.items()})
    monkeypatch.setattr(GC, "efficiency_row", lambda record, row, diagnostics, **kw: {"for": record["group"]})
    records = {
        "package_g1": {"group": 1, "arm": GC.ARM_PACKAGE},
        "package_g2": {"group": 2, "arm": GC.ARM_PACKAGE},
        "reference_g1": {"group": 1, "arm": GC.ARM_REFERENCE},
    }
    measurer = GC.GroupProgramMeasurer({"target": "toy"})
    diagnostics = {"rooflines": {"1": {}}}
    rows = measurer.time(records, member={"label": None, "diagnostics": diagnostics}, max_cycles=1, out=tmp_path)
    assert rows["package_g1"]["efficiency"] == {"for": 1}
    assert "efficiency" not in rows["package_g2"] and "efficiency" not in rows["reference_g1"]
    held = measurer.time(records, member={"label": "heldout", "diagnostics": diagnostics}, max_cycles=1, out=tmp_path)
    assert not any("efficiency" in row for row in held.values())


# --------------------------------------------------------------- fused regions in a cell


class RegionMeasurer(FakeMeasurer):
    """The package answers groups 1 and 2 as ONE fused region (boundary 2): one program, timed once."""

    def programs(self, arm, groups, *, package_dir, member, out):
        records = super().programs(arm, groups, package_dir=package_dir, member=member, out=out)
        if arm != "package" or member.get("label") is not None:
            return records
        region = {"members": [1, 2], "boundary": 2, "id": "r1"}
        for label, record in records.items():
            if record["group"] in (1, 2):
                record["region"] = region
            if record["group"] == 1:
                record["timed_with"] = 2
                record.pop("elf")
                if 1 in self.spec.get("fake", {}).get("declined_inside", ()):
                    record["linked"] = "vendor"
        return records

    def time(self, records, *, member, max_cycles, out):
        assert not any(r.get("timed_with") for r in records.values()), "an internal member is never timed alone"
        timed = super().time(records, member=member, max_cycles=max_cycles, out=out)
        for label, record in records.items():
            if record.get("region"):
                timed[label]["cycles"] = 250  # the region's one program
        return timed


def make_region(spec):
    return RegionMeasurer(spec)


def _region_spec(**fake):
    spec = _spec(**fake)
    spec["measurer"] = f"{__name__}:make_region"
    return spec


def test_a_fused_region_is_timed_once_and_every_member_is_the_packages(tmp_path):
    result = W.work(_job(tmp_path, _region_spec()))
    assert result["timing_status"] == V.TIMING_MEASURED and result["objective_cycles"] == 250  # not 100 + 250
    groups = {g["group"]: g for g in result["verdict"]["groups"]}
    assert groups["1"]["timed_with"] == 2 and groups["1"]["cycles"] == 0 and groups["2"]["cycles"] == 250
    assert {r["group"]: r["on"] for r in result["build"]["groups"] if r["group"] in ("1", "2")} == {
        "1": "package",
        "2": "package",
    }


def test_a_member_handed_back_inside_a_claimed_region_is_refused_like_any_decline(tmp_path):
    """Claiming a region and declining inside it cannot keep the claim's coverage."""
    result = W.work(_job(tmp_path, _region_spec(declined_inside=[1])))
    assert result["timing_status"] == V.TIMING_REFUSED and "g1" in result["refusal"]
    row = next(g for g in result["verdict"]["groups"] if g["group"] == "1")
    assert "inside its claimed region" in row["refusal"]


def test_a_refused_region_boundary_refuses_its_members(tmp_path):
    result = W.work(_job(tmp_path, _region_spec(dirty=[2])))
    assert result["timing_status"] == V.TIMING_REFUSED
    row = next(g for g in result["verdict"]["groups"] if g["group"] == "1")
    assert "boundary g2 was refused" in row["refusal"]


def test_a_collateral_region_is_held_to_its_members_summed_baselines():
    rows = [
        {"group": 7, "model": CELLS.COLLATERAL, "linked": "submission", "status": "graded", "correct": True,
         "cycles": 0, "timed_with": 8, "region": {"members": [7, 8], "boundary": 8}},
        {"group": 8, "model": CELLS.COLLATERAL, "linked": "submission", "status": "graded", "correct": True,
         "cycles": 290, "region": {"members": [7, 8], "boundary": 8}},
    ]  # fmt: skip
    baseline = {"7": {"cycles": 100, "on": "package"}, "8": {"cycles": 200, "on": "package"}}
    verdict = CELLS.compose_cell(rows, machine="emu", collateral=baseline, tolerance=0.01)
    assert verdict["collateral"][1]["baseline_cycles"] == 300 and not verdict.get("collateral_regression")
    # MUTATION: held to the boundary's own baseline alone, the same region would read as a 45% regression.
    alone = CELLS.collateral_problem(rows[1], baseline["8"], 0.01)
    assert alone and "+45.0%" in alone
