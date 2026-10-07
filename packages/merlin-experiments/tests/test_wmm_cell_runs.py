"""A cell run end to end from the command line: prepared from a measured loop run (its method, roles
and store base carried over, its seed the current champion record unless one is named), launched
detached, and read back -- with the per-model cells file refused against any other model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cell_prep as CP
from merlin_experiments.phase2.whole_model_measured import cell_runs as CR
from merlin_experiments.phase2.whole_model_measured import cells as CELLS
from merlin_experiments.phase2.whole_model_measured import cli as MCLI
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import forms as FORMS
from merlin_experiments.phase2.whole_model_measured import launch as LAUNCH
from merlin_experiments.phase2.whole_model_measured.identity import package_digest

from merlin.common import oot_repo
from merlin.common.paths import repo_root

FORMS_OF_MODEL = [
    {"group": "1", "form_text": "stem", "shape": {"H": 224}},
    {"group": "2", "form_text": "1x1", "shape": {"H": 56}},
    {"group": "3", "form_text": "1x1", "shape": {"H": 56}},
    {"group": "4", "form_text": "1x1", "shape": {"H": 28}},
]


@pytest.fixture
def loop(tmp_path, monkeypatch, capsys):
    """A prepared measured loop run (no measurement runs), and the cell measurements replaced."""
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    spec, pin = FX.write_builder(tmp_path)
    capsule = tmp_path / "capsules" / "SY_model_toy"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: SY_model_toy\n")
    options = {"model_capsule": "{input:model_capsule}", "machine": "emu", "header": str(tmp_path / "h.h")}
    config = tmp_path / "objective.json"
    config.write_text(
        json.dumps(
            {
                "schema": C.CONFIG_SCHEMA,
                "builder": {"spec": spec, "sha256": pin},
                "store": str(tmp_path / "store"),
                "screen": {"machine": FX.spike_machine(tmp_path), "build_options": dict(options)},
                "certifier": {"machine": FX.spike_machine(tmp_path), "build_options": dict(options)},
            }
        )
    )
    MCLI.main(
        [
            "prepare",
            "--target",
            "toy",
            "--method",
            "loop_nofsm",
            "--why",
            "the loop",
            "--objective-config",
            str(config),
            "--seed",
            str(FX.package(tmp_path, "seed")),
            "--input",
            f"model_capsule={capsule}",
            "--prohibited-instruction-role",
            "loop_descriptor",
            "--phase0-manifest",
            str(FX.write_phase0_manifest(tmp_path)),
        ]
    )
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    seen: dict = {}
    monkeypatch.setattr(FORMS, "statement_forms", lambda capsule, target: FORMS_OF_MODEL)

    def reference(spec, *, target, out):
        out.mkdir(parents=True, exist_ok=True)
        document = {"timing_status": "MEASURED", "objective_cycles": 5000}
        (out / "result.json").write_text(json.dumps(document))
        return document

    def own(spec, *, baseline_package, target, out):
        seen["baseline_package"] = str(baseline_package)
        return {"groups": [{"group": g, "cycles": 40, "on": "package"} for g in spec["cell"]["groups"]]}

    monkeypatch.setattr(CELLS, "reference_cell", reference)
    monkeypatch.setattr(CELLS, "measure_cell_baseline", own)
    monkeypatch.setattr(CP, "cell_diagnostics", lambda groups, **kw: {"rooflines": {}})
    return SimpleNamespace(run_dir=run_dir, tmp=tmp_path, seen=seen)


def test_the_example_cells_file_is_valid_data_naming_its_model():
    cells = CR.load_cells(repo_root() / "examples" / "gemmini" / "phase2" / "cells.yaml")
    assert cells["model"] == "SY_model_resnet50"
    assert all(body.get("groups") or body.get("form_of") for body in cells["cells"].values())


def test_a_cell_named_by_form_is_prepared_with_the_loops_method_and_roles(loop):
    seed = FX.package(loop.tmp, "cell_seed", argmax=4)
    document = CR.prepare_cell_run(loop.run_dir, cell_id="mm1x1", why="the 1x1 form", form_of=[3], seed=seed)
    run_dir = Path(document["run_dir"])
    record = json.loads((run_dir / "run.json").read_text())
    assert record["method"] == "loop_nofsm_cell_mm1x1" and record["prohibited_instruction_roles"] == ["loop_descriptor"]
    assert document["groups"] == [2, 4]  # one per distinct shape of the anchor's form
    assert document["seed"] == {
        "kind": "explicit",
        "package": str(seed.resolve()),
        "package_sha256": package_digest(seed),
    }
    assert (
        document["baseline_package"] is None and "baseline_package" not in loop.seen
    )  # an explicit seed is not verified
    config = json.loads((run_dir / "whole_model_objective_config.json").read_text())
    assert config["screen"]["machine"]["kind"] == "cell" and config["store"].endswith("/cells")
    assert json.loads((run_dir / CR.CELL_RECORD).read_text())["cell_id"] == "mm1x1"


def test_the_seed_defaults_to_the_loops_confirmed_best_and_is_the_baseline(loop):
    with pytest.raises(CR.CellRunError, match="name --seed"):
        CR.prepare_cell_run(loop.run_dir, cell_id="g1", why="no champion yet", groups=[1])
    repo = loop.run_dir / "oot"
    oot_repo.tag(repo, oot_repo.BEST_TAG, oot_repo.resolve(repo, "HEAD"))
    document = CR.prepare_cell_run(loop.run_dir, cell_id="g1", why="from the best", groups=[1])
    best = package_digest(loop.run_dir / "seed" / "submission")
    assert document["seed"]["kind"] == "loop_best" and document["seed"]["package_sha256"] == best
    assert package_digest(Path(document["run_dir"]) / "seed" / "submission") == best
    assert loop.seen["baseline_package"] == document["seed"]["package"] == document["baseline_package"]


def test_a_cells_file_is_refused_against_another_model(loop, tmp_path):
    cells = tmp_path / "cells.yaml"
    cells.write_text(f"schema: {CR.CELLS_SCHEMA}\nmodel: SY_model_other\ncells:\n  c: {{groups: [1]}}\n")
    with pytest.raises(CR.CellRunError, match="SY_model_other"):
        CR.prepare_cell_run(loop.run_dir, cell_id="c", why="x", cells_file=cells, seed=FX.package(tmp_path, "s"))
    cells.write_text(f"schema: {CR.CELLS_SCHEMA}\nmodel: SY_model_toy\ncells:\n  c: {{groups: [1, 3]}}\n")
    document = CR.prepare_cell_run(loop.run_dir, cell_id="c", why="x", cells_file=cells, seed=FX.package(tmp_path, "s"))
    assert document["groups"] == [1, 3] and document["definition"] == {"groups": [1, 3]}


def test_the_command_line_prepares_launches_and_reads_a_cell_run(loop, monkeypatch, capsys):
    from merlin_experiments import cli as TOP

    seed = FX.package(loop.tmp, "cli_seed")
    assert (
        TOP.main(
            ["cell", "prepare", str(loop.run_dir), "--cell", "g1", "--groups", "1", "--seed", str(seed), "--why", "cli"]
        )
        == 0
    )
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]
    with pytest.raises(SystemExit, match="not a cell run"):
        TOP.main(["cell", "launch", str(loop.run_dir), "--profile", "codex-gpt-6-sol-cell"])
    spawned = []
    monkeypatch.setattr(LAUNCH, "spawn_process", lambda argv, **kw: spawned.append(kw) or SimpleNamespace(pid=4321))
    assert TOP.main(["cell", "launch", run_dir, "--profile", "codex-gpt-6-sol-cell"]) == 0
    assert spawned[0]["start_new_session"] is True and json.loads(capsys.readouterr().out)["pid"] == 4321
    assert TOP.main(["cell", "status", run_dir, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["cell"]["id"] == "g1" and status["cell"]["reference_cycles"] == 5000
    assert status["launcher"]["pid"] == 4321
    assert TOP.main(["cell", "status", "--target", "toy", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["cell"] == "g1" and row["run_dir"] == run_dir


def test_the_cell_profile_is_a_valid_launch_profile():
    from merlin_experiments.phase2.whole_model_measured import profiles as P

    profile = P.load("codex-gpt-6-sol-cell")
    assert profile["round_seconds"] == 2700 and profile["max_sessions"] == 48


def test_board_validation_runs_both_arms_on_the_package_jobs_own_machine(tmp_path, monkeypatch, capsys):
    """The arms are the two jobs' own build options and the board, functional model and control the
    package job's own machine -- minus the run owner's host preparation step; the output is a product."""
    from merlin_experiments import cli as TOP
    from merlin_experiments.phase2.whole_model_measured import group_capsules_board as GCB

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    timing = {"hw_config": "hw", "lock_path": "/l", "queue_command": ["q"], "prepare_command": ["glue"]}
    machine = {"kind": "batched", "timing": timing, "local": {"kind": "spike"}, "control": {"solo_result": "/s"}}
    jobs = {}
    for name in ("package", "reference"):
        job_dir = tmp_path / name
        job_dir.mkdir()
        options = {"machine": "m", "header": "/h", "model_capsule": "/cap"}
        (job_dir / "job.json").write_text(json.dumps({"target": "toy", "machine": machine, "build_options": options}))
        jobs[name] = job_dir / "job.json"
    seen = {}
    monkeypatch.setattr(
        GCB, "measure_on_board", lambda arms, groups, **kw: seen.update(arms=arms, groups=groups, **kw) or {"rows": []}
    )
    argv = ["cell", "board", "--package-job", str(jobs["package"]), "--reference-job", str(jobs["reference"])]
    assert TOP.main([*argv, "--groups", "1,70", "--prepare-only"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == []
    assert sorted(seen["arms"]) == ["package", "reference"] and seen["groups"] == [1, 70] and seen["submit"] is False
    assert seen["machine"] == {k: v for k, v in timing.items() if k != "prepare_command"}
    assert seen["control"] == machine["control"] and seen["package_dir"] == tmp_path / "package" / "package"
    assert (tmp_path / "out" / "artifacts" / "perf-studies" / "group-capsules" / "toy") in seen["out"].parents
