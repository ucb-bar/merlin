"""Champion evidence profiles by phase: a phase-1 compiler is certified on capsules, a phase-2 one on a
whole-model program, and the phase is declared -- never read off the target."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from merlin.common import oot_repo as O
from merlin.common.paths import phase_runs_root
from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import champions as C
from merlin.targetgen import target_index as TI

TARGET = "fixture"
PROHIBITED = {"8": "LOOP_A", "9": "LOOP_B"}


@pytest.fixture()
def out_root(tmp_path, monkeypatch, pytestconfig):
    from merlin.targetgen import publish

    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.setattr(publish.paths, "repo_root", lambda: pytestconfig.rootpath)
    return root


def _package(root: Path, body: str) -> Path:
    root.mkdir(parents=True)
    write_yaml(
        root / "manifest.yaml",
        {
            "artifact_type": "mlir_oot_target_backend",
            "target": TARGET,
            "package_id": "fixture_oot_v0",
            "language": "python",
            "authoring": {"mode": "hand_curated"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "fixture-opt"},
            "commands": {
                name: {"argv": ["{tool}", "{input_mlir}"]}
                for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
            },
        },
    )
    tool = root / "fixture-opt"
    tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n")
    tool.chmod(0o755)
    (root / "transforms.py").write_text(body)
    return root


@pytest.fixture()
def phase1(tmp_path, out_root):
    """A phase-1 run's harness history: two graded rounds, the second tagged ``frozen``."""
    pkg = _package(tmp_path / "workspace" / "pkg", "x = 1\n")
    run = phase_runs_root(TARGET, 1) / "20260929T100000Z_merlin_assisted_abc1234"
    repo = O.init(run / "oot", sandbox_roots=[tmp_path / "workspace"])
    O.commit_candidate(repo, pkg, label="round 1", when="20260929T100500Z", run_id=run.name)
    (pkg / "transforms.py").write_text("x = 2\n")
    frozen = O.commit_candidate(repo, pkg, label="round 2", when="20260929T110000Z", run_id=run.name)
    O.tag(repo, O.FROZEN_TAG, frozen.commit)
    return run, repo, frozen


def _capsule_evidence(run: Path, frozen) -> dict:
    return {
        "provenance": {
            "phase1": {"run": run.name, "frozen_commit": frozen.commit},
            "corpus_seal_digest": "a" * 64,
            "phase0_evidence_digest": "b" * 64,
        },
        "measurements": {"package_digest": frozen.package_digest},
        "certification": {
            "capsules": {
                "tier": "L3",
                "grader": {"commit": "c" * 40},
                "engine": {"name": "fixture-rtl-sim", "binary_sha256": "d" * 64},
                "public": {"graded": 180, "at_tier": 121, "measured_now": 121, "carried": 0},
                "hidden": {"graded": 14, "at_tier": 14, "measured_now": 10, "carried": 4},
            }
        },
        "isa_prohibition": {
            "scope": "whole_elf",
            "verdict": "clean",
            "prohibited_roles": ["loop_descriptor"],
            "prohibited_instructions": dict(PROHIBITED),
            "coverage": {"elfs": 175, "measured": 175, "clean": 175, "unmeasured": 0},
        },
    }


def _whole_model_evidence(run: Path, frozen) -> dict:
    """The phase-2 evidence for the same bytes, as the exporter has always required it."""
    evidence = _capsule_evidence(run, frozen)
    evidence["measurements"] = {
        "package_digest": frozen.package_digest,
        "firesim": {"cycles": 37_455_758, "machine": "fixture-board", "header": "h", "control": {"in_batch": True}},
        "exactness": {"contract_sha256": "e" * 64, "label": "exact"},
    }
    evidence["certification"] = {"gsim": {"verdict": "pass"}}
    return evidence


def _export(repo: Path, evidence: dict, package_id: str, **kwargs) -> Path:
    return C.export_champion(TARGET, repo, package_id=package_id, **copy.deepcopy(evidence), **kwargs)


def test_a_phase1_champion_passes_on_capsule_evidence_alone(phase1, out_root):
    run, repo, frozen = phase1
    evidence = _capsule_evidence(run, frozen)
    assert C.evidence_problems(evidence, phase=C.PHASE1) == []
    dest = _export(repo, evidence, "p1_champion", phase=C.PHASE1)  # rev defaults to frozen for phase 1

    assert C.layout_problems(dest) == []
    provenance = json.loads((dest / ".merlin/provenance.json").read_text())
    assert provenance["phase"] == C.PHASE1 and provenance["evidence_profile"] == "phase1"
    assert provenance["phase1"]["history"]["commit"] == frozen.commit
    assert "phase2" not in provenance
    measurements = json.loads((dest / ".merlin/measurements.json").read_text())
    assert "firesim" not in measurements and "exactness" not in measurements
    note = (dest / C.PUBLICATION_NOTE).read_text()
    assert "NOT CERTIFIED" not in note and "Phase-1 champion" in note
    assert "public capsules at L3: 121/180 (121 executed now, 0 carried)" in note
    assert "hidden capsules at L3: 14/14 (10 executed now, 4 carried)" in note
    (row,) = load_yaml(TI.index_path(TARGET))["champions"]
    assert row["phase"] == C.PHASE1 and row["package_digest"] == frozen.package_digest


def test_the_record_may_declare_the_phase_instead_of_the_call(phase1, out_root):
    run, repo, frozen = phase1
    evidence = _capsule_evidence(run, frozen)
    evidence["provenance"]["phase"] = C.PHASE1
    dest = _export(repo, evidence, "p1_by_record")
    assert json.loads((dest / ".merlin/provenance.json").read_text())["phase"] == C.PHASE1


def _no_capsules(e):
    e["certification"] = {}


def _empty_prohibited_set(e):
    e["isa_prohibition"]["prohibited_instructions"] = {}


def _no_grader(e):
    del e["certification"]["capsules"]["grader"]


def _no_engine_digest(e):
    e["certification"]["capsules"]["engine"].pop("binary_sha256")


def _split_does_not_add_up(e):
    e["certification"]["capsules"]["hidden"]["carried"] = 5


def _more_passes_than_graded(e):
    e["certification"]["capsules"]["public"].update(at_tier=181, measured_now=181)


def _nothing_graded(e):
    e["certification"]["capsules"]["public"].update(graded=0, at_tier=0, measured_now=0)


def _unmeasured_elf(e):
    e["isa_prohibition"]["coverage"].update(measured=174, unmeasured=1)


def _no_coverage(e):
    del e["isa_prohibition"]["coverage"]


def _not_clean(e):
    e["isa_prohibition"]["verdict"] = "violations"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (_no_capsules, r"certification\.capsules\.tier"),
        (_empty_prohibited_set, r"isa_prohibition\.prohibited_instructions"),
        (_no_grader, r"capsules\.grader\.commit"),
        (_no_engine_digest, r"capsules\.engine\.binary_sha256"),
        (_split_does_not_add_up, r"capsules\.hidden: measured_now 10 \+ carried 5 is not at_tier 14"),
        (_more_passes_than_graded, r"capsules\.public\.at_tier: 181 passes of 180 graded"),
        (_nothing_graded, r"capsules\.public\.graded"),
        (_unmeasured_elf, r"coverage: 174 of 175 ELFs measured, 1 unmeasured"),
        (_no_coverage, r"isa_prohibition\.coverage"),
        (_not_clean, r"isa_prohibition\.verdict"),
    ],
)
def test_a_phase1_champion_without_its_capsule_evidence_is_refused(phase1, out_root, mutate, match):
    run, repo, frozen = phase1
    evidence = _capsule_evidence(run, frozen)
    mutate(evidence)
    with pytest.raises(C.ChampionError, match=match):
        _export(repo, evidence, "refused", phase=C.PHASE1)
    assert not C.champion_dir(TARGET, "refused").exists()


def test_phase1_asks_no_whole_model_number_and_phase2_still_does(phase1, out_root):
    run, repo, frozen = phase1
    capsules = _capsule_evidence(run, frozen)
    assert C.evidence_problems(capsules, phase=C.PHASE1) == []
    # Undeclared means phase 2, exactly as before: the same capsule evidence is refused there.
    undeclared = C.evidence_problems(capsules)
    assert undeclared == C.evidence_problems(capsules, phase=C.PHASE2)
    assert {"measurements.firesim.cycles", "certification.gsim.verdict"} <= set(undeclared)
    assert any(p.startswith("measurements.exactness") for p in undeclared)
    with pytest.raises(C.ChampionError, match="phase-2 champion evidence"):
        _export(repo, capsules, "refused", rev=O.FROZEN_TAG)
    # ... and the whole-model evidence still passes the phase-2 profile with no phase declared.
    whole = _whole_model_evidence(run, frozen)
    assert C.missing_evidence(whole) == [] and C.evidence_problems(whole) == []
    dest = _export(repo, whole, "p2_shape", rev=O.FROZEN_TAG)
    provenance = json.loads((dest / ".merlin/provenance.json").read_text())
    assert provenance["phase"] == C.PHASE2 and provenance["phase2"]["best_commit"] == frozen.commit
    assert "Phase-2 champion" in (dest / C.PUBLICATION_NOTE).read_text()


@pytest.mark.parametrize(
    "call, record, match",
    [
        (C.PHASE1, C.PHASE2, "the export declares phase 1, the provenance record phase 2"),
        (3, None, r"phase: 3 is not a champion phase"),
        (None, "1", r"provenance\.phase: '1' is not a champion phase"),
    ],
)
def test_the_phase_is_declared_consistently(phase1, out_root, call, record, match):
    run, repo, frozen = phase1
    evidence = _capsule_evidence(run, frozen)
    if record is not None:
        evidence["provenance"]["phase"] = record
    with pytest.raises(C.ChampionError, match=match):
        _export(repo, evidence, "refused", phase=call)


def test_a_phase1_champion_is_its_frozen_commit(phase1, out_root, tmp_path):
    run, repo, frozen = phase1
    pkg = tmp_path / "workspace" / "pkg"
    (pkg / "transforms.py").write_text("x = 3\n")
    later = O.commit_candidate(repo, pkg, label="round 3", when="20260929T120000Z", run_id=run.name)
    O.tag(repo, O.BEST_TAG, later.commit)
    evidence = _capsule_evidence(run, frozen)
    evidence["measurements"]["package_digest"] = later.package_digest
    with pytest.raises(C.ChampionError, match="a phase-1 champion is its frozen commit"):
        _export(repo, evidence, "refused", phase=C.PHASE1, rev=O.BEST_TAG)


def test_export_cli_preserves_declared_service_and_original_evidence(phase1, out_root, tmp_path, capsys):
    from merlin_experiments.cli import main

    run, _, _ = phase1
    package = tmp_path / "workspace" / "pkg"
    project = (
        '[project]\nname = "fixture-compiler"\nversion = "0.1.0"\n'
        '[project.entry-points."merlin.backends"]\nfixture = "unloaded_fixture_provider"\n'
    )
    (package / "pyproject.toml").write_text(project)
    guard = 'raise AssertionError("export must not import compiler code")\n'
    (package / "unloaded_fixture_provider.py").write_text(guard)
    repo = O.init(run / "service-oot", sandbox_roots=[tmp_path / "workspace"])
    frozen = O.commit_candidate(repo, package, label="declared service", when="20260929T120000Z", run_id=run.name)
    O.tag(repo, O.FROZEN_TAG, frozen.commit)
    evidence = _capsule_evidence(run, frozen)
    arguments = ["export", "--target", TARGET, "--repo", str(repo), "--phase", "1",
                 "--package-id", "cli_champion", "--run-dir", str(run),
                 "--stage-root", str(run / "explicit-exports"), "--artifacts-root", str(out_root / "artifacts")]
    files = {}
    for name, document in evidence.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        files[path] = path.read_bytes()
        arguments.extend(("--" + name.replace("_", "-"), str(path)))
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    destination = C.champion_dir(TARGET, "cli_champion")
    assert Path(report["directory"]) == destination
    assert (destination / "pyproject.toml").read_bytes() == (package / "pyproject.toml").read_bytes()
    assert (destination / "unloaded_fixture_provider.py").read_text() == guard
    assert report["services"] == {
        "scope": "preserved_project_metadata_only", "status": "declared", "compatibility": "not_checked",
        "installed_by_export": False, "code_loaded_by_inspection": False,
        "providers": [{"name": "fixture", "entry_point": "unloaded_fixture_provider"}],
    }
    assert O.tree_digest(repo, O.FROZEN_TAG) == frozen.package_digest
    assert all(path.read_bytes() == payload for path, payload in files.items())

    # The same public command retains the champion owner's original evidence refusal.
    (tmp_path / "measurements.json").write_text(json.dumps({"package_digest": "0" * 64}))
    arguments[arguments.index("cli_champion")] = "cli_refused"
    with pytest.raises(SystemExit) as stopped:
        main(arguments)
    assert stopped.value.code == 2
    assert "but the measurements are of" in capsys.readouterr().err
    assert not C.champion_dir(TARGET, "cli_refused").exists()
    for malformed in ("[]", "{", '{"package_digest": NaN}', '{"package_digest": "a", "package_digest": "b"}'):
        (tmp_path / "measurements.json").write_text(malformed)
        with pytest.raises(SystemExit) as stopped:
            main(arguments)
        assert stopped.value.code == 2
        assert "error:" in capsys.readouterr().err
        assert not C.champion_dir(TARGET, "cli_refused").exists()


@pytest.mark.parametrize(
    "project,status",
    [
        (None, "missing_metadata"),
        ("[tool.custom]\nvalue = 1\n", "missing_project_metadata"),
        ("[project]\nname = 'fixture'\n", "absent"),
        ("[project]\ndynamic = ['entry-points']\n", "dynamic"),
        ("[project\n", "invalid_metadata"),
        ("project = 1\n", "invalid_metadata"),
        ("[project]\ndynamic = 'entry-points'\n", "invalid_metadata"),
        ("[project]\nentry-points = 1\n", "invalid_metadata"),
        ("[project.entry-points]\n'merlin.backends' = 1\n", "invalid_metadata"),
        ("[project.entry-points.'merlin.backends']\nfixture = 1\n", "invalid_metadata"),
        ("[project]\ndynamic = ['entry-points']\n[project.entry-points]\n", "invalid_metadata"),
    ],
)
def test_export_service_metadata_never_infers_an_installed_compiler(tmp_path, project, status):
    from merlin_experiments.backend_export import service_declarations

    if project is not None:
        (tmp_path / "pyproject.toml").write_text(project)
    report = service_declarations(tmp_path)
    assert report["status"] == status
    assert report["providers"] == []
    assert report["compatibility"] == "not_checked"
    assert report["installed_by_export"] is report["code_loaded_by_inspection"] is False
