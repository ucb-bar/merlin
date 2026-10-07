"""A measured run is graded under the exactness contract it was launched with -- by value, recorded in every
verdict, cell and group-capsule grade, enforced at champion export -- and a result graded under another
contract is never its best."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import wmm_fixtures as FX
import yaml
from merlin_experiments.phase2.whole_model_measured import cells as CELLS
from merlin_experiments.phase2.whole_model_measured import config as CFG
from merlin_experiments.phase2.whole_model_measured import group_capsules as GC
from merlin_experiments.phase2.whole_model_measured import ledger as L
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import worker as W

from merlin.perf import exactness as EX
from merlin.perf import whole_model_verdict as V
from merlin.targetgen import champions

BOUNDED_MATMUL = {
    "schema": EX.SCHEMA,
    "target": "toy",
    "default": {"mode": "exact"},
    "forms": [
        {
            "name": "device_requant_matmul",
            "match": {"op": "matmul"},
            "exactness": {"mode": "bounded", "bound": {"max_abs_lsb": 1}, "reason": "half-even device rounding"},
        }
    ],
}


def _contract_file(tmp_path: Path, document=BOUNDED_MATMUL) -> Path:
    path = tmp_path / "exactness.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_a_measured_result_records_the_contract_it_was_graded_under(tmp_path):
    contract = EX.load(_contract_file(tmp_path))
    groups = {
        "1": {"cycles": 100, "sum": 7, "fnv": 9, "want_sum": 7, "want_fnv": 9},
        "2": {"cycles": 200, "sum": 6, "fnv": 6, "want_sum": 5, "want_fnv": 6},  # its digest differs
    }
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p", groups=groups),
        machine=FX.spike_machine(tmp_path),
        builder=FX.write_builder(tmp_path),
        exactness={"contract": contract.to_document(), "forms": {}},
    )
    result = W.work(job_dir)
    verdict = result["verdict"]
    assert verdict["exactness"]["per_group"] == {"1": "bounded(<=1 LSB)", "2": "bounded(<=1 LSB)"}
    assert verdict["exactness"]["contract"]["semantics_sha256"] == contract.semantics_sha256
    # A differing digest says THAT the group differs, not by how much: it cannot pass a bound, and says so.
    row = next(r for r in verdict["groups"] if r["group"] == "2")
    assert row["state"] == V.GROUP_FAILED and row["exactness"]["verifiable"] is False
    assert result["timing_status"] == V.TIMING_MEASURED_INVALID


def test_without_a_contract_a_result_is_graded_exact_and_says_so(tmp_path):
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=FX.spike_machine(tmp_path),
        builder=FX.write_builder(tmp_path),
    )
    result = W.work(job_dir)
    assert result["timing_status"] == V.TIMING_MEASURED
    assert EX.label_summary(result["verdict"]["exactness"]) == "exact"
    assert EX.graded_semantics(result["verdict"]) == EX.DEFAULT_SEMANTICS_SHA256


def test_a_run_seals_its_contract_by_value_at_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    path = _contract_file(tmp_path)
    spec, pin = FX.write_builder(tmp_path)
    config = {
        "schema": CFG.CONFIG_SCHEMA,
        "store": str(tmp_path / "store"),
        "builder": {"spec": spec, "sha256": pin},
        "exactness": str(path),
        "screen": {"machine": FX.spike_machine(tmp_path)},
    }
    prepared = RUNS.prepare(
        target="toy",
        method="m",
        objective_config=config,
        seed=FX.package(tmp_path, "seed"),
        prohibited_roles=[],
        why="test",
        run_factory=lambda **kw: tmp_path / "run",
    )
    sealed = json.loads(prepared.config_path.read_text())["exactness"]
    assert sealed["document"] == BOUNDED_MATMUL and sealed["sha256"] == EX.load(path).sha256
    path.write_text(yaml.safe_dump({**BOUNDED_MATMUL, "forms": []}))  # an edit after the launch...
    objective = CFG.from_config(json.loads(prepared.config_path.read_text()), target="toy")
    carried = EX.Contract.from_value(objective.screen.exactness["contract"])
    assert carried.forms and carried.semantics_sha256 != EX.DEFAULT_SEMANTICS_SHA256  # ...does not reach the run
    with pytest.raises(CFG.ConfigError, match="exactness contract"):
        CFG.seal_exactness({**config, "exactness": str(tmp_path / "absent.yaml")}, target="toy")


class _Screen:
    target = "toy"
    machine = {"kind": "spike"}
    build_options: dict = {}

    def __init__(self, exactness):
        self.exactness = exactness

    def jobs(self):
        return []


def test_a_result_graded_under_another_contract_is_never_the_best():
    bounded = EX.Contract(BOUNDED_MATMUL)
    objective = O.WholeModelObjective(screen=_Screen({"contract": bounded.to_document()}), screen_reference=None)
    graded_exact = {"verdict": {"timing_status": V.TIMING_MEASURED}}  # recorded no contract: the default
    assert not objective.eligible(graded_exact)
    assert "graded under exactness contract" in objective.ineligible_text(graded_exact)
    graded_here = {"verdict": {"timing_status": V.TIMING_MEASURED, "exactness": {"contract": bounded.record()}}}
    assert objective.eligible(graded_here)
    default_run = O.WholeModelObjective(screen=_Screen(None), screen_reference=None)
    assert default_run.eligible(graded_exact) and not default_run.eligible(graded_here)


def test_champion_export_requires_the_contract_a_measurement_was_graded_under():
    unrecorded = {"verdict": {"timing_status": V.TIMING_MEASURED}}
    assert L.exactness_record(unrecorded) is None
    assert champions.exactness_problems({"exactness": None}) == [
        "measurements.exactness (the measurement recorded no exactness contract)"
    ]
    bounded = EX.Contract(BOUNDED_MATMUL)
    applied = {"contract": bounded.record(), "per_group": {"1": "bounded(<=1 LSB)"}, "summary": {"bounded(<=1 LSB)": 1}}
    record = L.exactness_record({"verdict": {"exactness": applied}})
    assert record["label"] == "bounded(<=1 LSB) x1" and record["bounded_forms"][0]["name"] == "device_requant_matmul"
    assert champions.exactness_problems({"exactness": record}) == []


class _LabelledMeasurer:
    """A cell measurer whose grades say which contract they applied (as the real one's do)."""

    def __init__(self, spec):
        self.spec = spec

    def programs(self, arm, groups, *, package_dir, member, out):
        return {f"{arm}:g{g}": {"group": g, "linked": "submission", "kind": "matmul", "elf": f"/{g}"} for g in groups}

    def scan(self, record, *, roles):
        return {"clean": True, "summary": {}}

    def time(self, records, *, member, max_cycles, out):
        label = EX.Contract.from_value(self.spec.get("exactness")).resolve({"op": "matmul"}).label()
        return {k: {"status": "graded", "cycles": 10, "correct": True, "exactness": label} for k in records}


def labelled(spec):
    return _LabelledMeasurer(spec)


def test_a_cell_result_records_the_contract_its_programs_were_graded_under(tmp_path):
    spec = {
        "kind": "cell",
        "target": "toy",
        "measurer": f"{__name__}:labelled",
        "cell": {"id": "c", "model_capsule": "/m", "groups": [1, 2]},
        "timing": {"registry_machine": "emu", "max_cycles": 1000},
    }
    bounded = EX.Contract(BOUNDED_MATMUL)
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=spec,
        builder=FX.write_builder(tmp_path),
        exactness={"contract": bounded.to_document(), "forms": {}},
    )
    fields = CELLS.measure_cell(
        json.loads((job_dir / "job.json").read_text()), job_dir, job_dir / "package", target="toy"
    )
    assert fields["exactness"]["label"] == "bounded(<=1 LSB) x2"
    assert fields["exactness"]["contract"]["semantics_sha256"] == bounded.semantics_sha256


def test_group_programs_are_built_with_each_groups_contract(tmp_path, monkeypatch):
    from merlin.perf import whole_model_group_timing as T

    seen = {}

    def fake_build(*args, exactness=None, **kwargs):
        seen["package"] = exactness(1, {"op": "matmul", "operand_dtype": "i8"})
        seen["residual"] = exactness(2, {"op": "residual_add", "bound_lsb": 1})
        return {}

    monkeypatch.setattr(T, "build_group_programs", fake_build)
    measurer = GC.GroupProgramMeasurer(
        {
            "target": "toy",
            "build_options": {"machine": "m", "header": "h"},
            "exactness": EX.Contract(BOUNDED_MATMUL).to_document(),
        }
    )
    measurer.programs("package", [1], package_dir=tmp_path, member={"model_capsule": "/m"}, out=tmp_path)
    assert (
        seen["package"]["label"] == "bounded(<=1 LSB)"
        and seen["package"]["declared_by"] == "form:device_requant_matmul"
    )
    assert seen["residual"]["declared_by"] == EX.DECLARED_OP
