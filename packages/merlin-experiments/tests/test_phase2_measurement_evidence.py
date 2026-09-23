"""Real paired publication and report replay with explicit upstream fixtures.

Real: generated corpus freeze, holdout loading, functional fork/hash checks, paired
plan/schedule/raw store/manifest writer, coordinator admission, statistics and report.
Substituted: prequalified candidate handoff and target/certificate/RTL inputs,
engine execution and auxiliary roofline qualification. No paid agent, hardware or
full candidate-authoring qualification is claimed. CLI refusal uses the real verifier.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import measurement_support as MS
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PME

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir, python_import_roots


@pytest.fixture
def produced(tmp_path, monkeypatch):
    scripts = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    pair = importlib.import_module("merlin_experiments.phase2.paired_cli")
    coordinator = importlib.import_module("merlin_experiments.phase2.checkpoint_controller")
    reporting = importlib.import_module("merlin_experiments.phase2.reporting")
    renderer = importlib.import_module("gen_perf_report")
    statistics = importlib.import_module("merlin_experiments.phase2.statistics")
    descriptor = {
        "name": "case",
        "label": "dev",
        "source_role": "derived_sweep",
        "performance": {"family": "PK", "claim": "RECOVERS"},
        "operation": {"op": "matmul", "attributes": {"lhs": "X", "weight": "W", "out": "Y", "output_dtype": "i32"}},
        "inputs": [
            {"name": "X", "role": "input", "shape": [16, 17], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [17, 16], "dtype": "i8"},
        ],
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    public = tmp_path / "live/public"
    public.mkdir(parents=True)
    member = public.parent / "_tuning/case"
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(json.dumps(descriptor))
    (public.parent / "MANIFEST.yaml").write_text(
        json.dumps(
            {
                "generated": ["_tuning/case"],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {
                            "category": "_tuning",
                            "label": "dev",
                            "included_in_functional_grade": False,
                        },
                    }
                },
            }
        )
    )
    target = SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])
    tuning = CORPUS.freeze_performance_corpus(CORPUS.discover_performance_corpus(target), tmp_path / "tuning")
    holdout_root = tmp_path / "heldout"
    holdout_member = holdout_root / "_perf/case"
    holdout_member.mkdir(parents=True)
    (holdout_member / "capsule.yaml").write_text(json.dumps(descriptor))
    holdout_tree = PI.holdout_tree_record(holdout_root)
    holdout_manifest = holdout_root / "holdout_manifest.json"
    holdout_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "generated_performance_holdout_reveal",
                "domain": {"target": "fixture"},
                "cohorts": {"PK_predictor": {"family": "PK", "member_count": 1}},
                "members": [
                    {
                        "name": "case",
                        "path": "_perf/case",
                        "family": "PK",
                        "cohort": "PK_predictor",
                        "M": 16,
                        "N": 16,
                        "K": 17,
                    }
                ],
                "corpus": holdout_tree,
            }
        )
    )
    for path in reversed([holdout_root, *holdout_root.rglob("*")]):
        path.chmod(0o555 if path.is_dir() else 0o444)
    holdout = PI.load_holdout_corpus(
        holdout_root,
        holdout_manifest,
        manifest_sha256=C.sha256_file(holdout_manifest),
        capsules_sha256=holdout_tree["sha256"],
        expected_target="fixture",
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# frozen functional bytes\n")
    baseline_sha = hash_tree(baseline)["sha256"]
    functional = SimpleNamespace(
        run_id="functional",
        digest=baseline_sha,
        frozen_at=None,
        public_score={"per_capsule": [{"capsule": "public", "tiers": {"L3": "pass"}}]},
        hidden_score={"per_capsule": [{"capsule": "hidden", "tiers": {"L3": "pass"}}]},
    )
    workload = PME.gsim_workload(tuning.capsules[0])
    certificate = SimpleNamespace(
        sha256="c" * 64,
        path=tmp_path / "certificate.json",
        unresolved={},
        pins={},
        members={GATE.workload_sha256(workload): {}},
        to_dict=lambda: {"sha256": "c" * 64},
    )

    def load_descriptor(path, *, source_root):
        assert source_root == tmp_path
        return target

    monkeypatch.setattr(pair, "load_target_experiment", load_descriptor)
    monkeypatch.setattr(GATE, "load_certificate", lambda *_a, **_k: certificate)
    monkeypatch.setattr(MS, "load_rtl_identity", lambda *_a: {"fixture": "external RTL facts"})
    monkeypatch.setattr(MS, "roofline_auxiliary_requirements", lambda *_a: {"fixture": "not qualified"})
    runs = tmp_path / "runs"
    runs.mkdir()

    execute = PME.execute_schedule

    def engine(spec, *_args, **_kwargs):
        return {
            "execution": spec.as_dict(),
            "measurement": {
                "status": "pass",
                "numeric": "pass",
                "gsim_qualification": {"admitted": True},
                "per_sim": {
                    "spike": {"correct": True},
                    "gsim": {
                        "correct": True,
                        "cycles": 100 if spec.arm == "baseline" else 80,
                        "provenance": {
                            "tier": "L3",
                            "simulator": "gsim",
                            "oracle_kind": "rtl_gsim",
                            "derived_from_rtl": True,
                            "cycle_accurate": True,
                            "elf_sha256": "e" * 64,
                            "reused_measurement": False,
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(PME, "execute_schedule", lambda *a, **kw: execute(*a, **kw, executor=engine))
    trials = [f"trial_{index:02d}" for index in range(3)]
    handoffs, receipts, verified_children = {}, [], []
    for index, trial in enumerate(trials):
        candidate = tmp_path / trial
        candidate.mkdir()
        (candidate / "compiler.py").write_text(f"# synthetic admitted candidate {index}\n")
        handoff = SimpleNamespace(
            record_sha256=f"{index + 1}" * 64,
            candidate_sha256=hash_tree(candidate)["sha256"],
            functional_run_id="functional",
            functional_submission_sha256=baseline_sha,
            corpus_manifest_sha256=tuning.manifest_sha256,
            corpus_sha256=tuning.capsules_sha256,
            agent_contract={"trial": trial},
        )
        handoffs[trial] = handoff
        for phase, corpus in (("tuning", tuning), ("held_out", holdout)):
            inputs = PME.PairedInputs(
                functional,
                handoff,
                corpus,
                phase,
                baseline,
                baseline_sha,
                candidate,
                handoff.candidate_sha256,
                certificate,
            )
            monkeypatch.setattr(PI, "load_paired_inputs", lambda *_a, _inputs=inputs, **_k: _inputs)
            run_id = f"{trial}__{phase}"
            assert (
                pair.main(
                    [
                        "--descriptor",
                        str(tmp_path / "explicit-target.yaml"),
                        "--source-root",
                        str(tmp_path),
                        "--measurement-root",
                        str(runs),
                        "--functional-runs-root",
                        str(tmp_path / "functional-runs"),
                        "--contract-root",
                        str(tmp_path / "contract"),
                        "--functional-run-id",
                        "functional",
                        "--functional-submission-sha256",
                        baseline_sha,
                        "--candidate-record",
                        str(tmp_path / "upstream-admitted.json"),
                        "--corpus-root",
                        str(corpus.root),
                        "--corpus-manifest",
                        str(corpus.manifest_path),
                        "--corpus-manifest-sha256",
                        corpus.manifest_sha256,
                        "--corpus-capsules-sha256",
                        corpus.capsules_sha256,
                        "--phase",
                        phase,
                        "--gsim-certificate",
                        str(certificate.path),
                        "--gsim-certificate-sha256",
                        certificate.sha256,
                        "--rtl-facts",
                        str(tmp_path / "rtl.json"),
                        "--run-id",
                        run_id,
                    ]
                )
                == 0
            )
            path = runs / run_id / "campaign_manifest.json"
            binding = ME.MeasurementBinding(
                phase,
                "functional",
                baseline_sha,
                handoff.record_sha256,
                handoff.candidate_sha256,
                corpus.manifest_sha256,
                corpus.capsules_sha256,
                certificate.sha256,
            )
            verified = ME.verify_paired_measurement(path, expected=binding)
            adopted = coordinator._verify_measurement_manifest(
                path,
                phase=phase,
                functional_run_id="functional",
                functional_submission_sha256=baseline_sha,
                handoff=handoff,
                corpus_manifest_sha256=corpus.manifest_sha256,
                corpus_capsules_sha256=corpus.capsules_sha256,
                certificate_sha256=certificate.sha256,
            )
            assert adopted == verified.receipt()
            receipts.append({"trial": trial, **adopted})
            verified_children.append((binding, verified))
    trial_evidence = [
        {"trial": trial, "agent_run_id": f"agent-{trial}", "agent_evidence_sha256": handoffs[trial].record_sha256}
        for trial in trials
    ]
    declaration = statistics.predeclare(
        trials=[{key: row[key] for key in ("trial", "agent_run_id")} for row in trial_evidence],
        capsules=[{"family": f"{phase}:PK", "capsule": "case"} for phase in ME.PHASES],
        replicates=ME.REPLICATES,
    )
    stats_path = tmp_path / "statistics_predeclaration.json"
    stats_path.write_text(statistics.canonical_json(declaration))
    stats_path.chmod(0o444)
    rows = [
        row
        for receipt, (_, verified) in zip(receipts, verified_children, strict=True)
        for row in ME.statistics_rows(verified.rows, trial=receipt["trial"])
    ]
    result = statistics.evaluate(declaration, rows, trial_evidence=trial_evidence)
    assert result["status"] == "admitted"
    parent = {
        "schema": "merlin.agentic-performance-experiment.v1",
        "status": "GO",
        "selection": "all_three_trials_all_predeclared_cells_no_best_of_no_drop",
        "declaration": {
            "experiment_id": "fixture",
            "trials": trials,
            "measurement_phases": list(ME.PHASES),
            "trial_contracts": {trial: handoffs[trial].agent_contract for trial in trials},
            "gsim_certificate_sha256": certificate.sha256,
        },
        "trials": trial_evidence,
        "measurement_manifests": receipts,
        "holdout": {"manifest_sha256": holdout.manifest_sha256, "capsules_sha256": holdout.capsules_sha256},
        "heldout_gsim_certificate": {"sha256": certificate.sha256},
        "statistics_predeclaration_sha256": C.sha256_file(stats_path),
        "statistics": result,
    }
    parent_path = coordinator._seal_final(tmp_path, parent)
    return SimpleNamespace(
        parent=parent,
        path=parent_path,
        sha=C.sha256_file(parent_path),
        handoffs=handoffs,
        reporting=reporting,
        renderer=renderer,
        children=verified_children,
        rows=rows,
        scripts=scripts,
    )


def test_real_paired_writer_coordinator_and_report_replay(produced):
    parent, rows, result = produced.reporting.load_current_experiment(
        produced.path, expected_sha256=produced.sha, handoffs=produced.handoffs
    )
    assert rows == produced.rows
    assert len(rows) == 24
    assert result["aggregate"]["median_speedup"] == 1.25
    report = produced.renderer.render_current_report(parent, rows, result, produced.sha)
    assert "GSIM supplies RTL timing" in report
    assert "PK CLAIM ESTABLISHED" not in report
    with pytest.raises(produced.reporting.ReportingGateError, match="parent experiment manifest"):
        importlib.import_module("perf_reporting").load_reportable_run(produced.children[0][1].manifest_path.parent)


@pytest.mark.parametrize("change", ["manifest", "results", "binding", "duplicate", "missing", "timing", "fork"])
def test_producer_evidence_negative_controls(produced, change):
    binding, verified = produced.children[0]
    path = verified.manifest_path
    document = json.loads(path.read_bytes())
    if change == "binding":
        binding = dataclasses.replace(binding, candidate_sha256="f" * 64)
        produced.handoffs["trial_00"].candidate_sha256 = "f" * 64
    elif change in {"results", "duplicate", "missing"}:
        result_path = Path(document["raw_results"]["paired_cells"])
        result_path.chmod(0o644)
        results = json.loads(result_path.read_bytes())
        if change == "results":
            results["cells"][0]["correct"] = False
        elif change == "duplicate":
            results["cells"].append(results["cells"][0])
        else:
            results["cells"].pop()
        result_path.write_text(json.dumps(results))
        if change != "results":
            document["raw_results"]["paired_cells_sha256"] = C.sha256_file(result_path)
            path.write_text(json.dumps(document))
    else:
        if change == "timing":
            document["engine_policy"]["timing_authority"] = "verilator"
        elif change == "fork":
            document["fork_after"]["ok"] = False
        else:
            document["candidate_sha256"] = "f" * 64
        path.write_text(json.dumps(document))
    with pytest.raises(ME.MeasurementEvidenceError):
        ME.verify_paired_measurement(path, expected=binding)
    with pytest.raises(produced.reporting.ReportingGateError):
        produced.reporting.load_current_experiment(
            produced.path, expected_sha256=produced.sha, handoffs=produced.handoffs
        )


def test_hash_and_parse_share_result_bytes(produced, monkeypatch):
    binding, verified = produced.children[0]
    result_path = Path(verified.manifest["raw_results"]["paired_cells"])
    original = Path.read_bytes
    calls = []

    def read(path):
        payload = original(path)
        if path == result_path:
            calls.append(path)
            path.chmod(0o644)
            path.write_bytes(b'{"cells": []}')
        return payload

    monkeypatch.setattr(Path, "read_bytes", read)
    observed = ME.verify_paired_measurement(
        verified.manifest_path, expected=binding, manifest_sha256=verified.manifest_sha256
    )
    assert observed.rows == verified.rows
    assert calls == [result_path]
    with pytest.raises(ME.MeasurementEvidenceError, match="changed"):
        ME.verify_paired_measurement(verified.manifest_path, expected=binding)


@pytest.mark.parametrize("pin_matches", [False, True])
def test_current_cli_uses_real_candidate_verifier_and_never_writes_on_refusal(produced, tmp_path, pin_matches):
    bad = tmp_path / "invalid-candidate.json"
    bad.write_text("{}")
    bad.chmod(0o444)
    if pin_matches:
        produced.parent["trials"][0]["agent_evidence_sha256"] = C.sha256_file(bad)
        produced.path.chmod(0o644)
        produced.path.write_text(json.dumps(produced.parent))
        produced.path.chmod(0o444)
        produced.sha = C.sha256_file(produced.path)
    output = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(produced.scripts / "gen_perf_report.py"),
            "--experiment-manifest",
            str(produced.path),
            "--manifest-sha256",
            produced.sha,
            "--candidate-record",
            f"trial_00={bad}",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots()))),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "candidate record" in result.stderr
    assert ("verify_candidate_handoff" in result.stderr) is pin_matches
    assert not output.exists()


@pytest.mark.parametrize("owner", ["campaign", "plan", "results"])
@pytest.mark.parametrize("schema", [None, "unknown_future_schema"])
def test_current_schemas_cannot_be_guessed(produced, owner, schema):
    binding, verified = produced.children[0]
    path = verified.manifest_path
    document = json.loads(path.read_bytes())
    if owner == "results":
        result_path = Path(document["raw_results"]["paired_cells"])
        result = json.loads(result_path.read_bytes())
        result["schema"] = schema
        result_path.chmod(0o644)
        result_path.write_text(json.dumps(result))
        document["raw_results"]["paired_cells_sha256"] = C.sha256_file(result_path)
    elif owner == "plan":
        document["measurement_plan"]["schema"] = schema
        import hashlib

        document["measurement_plan_sha256"] = hashlib.sha256(
            ME._canonical_bytes(document["measurement_plan"])
        ).hexdigest()
    else:
        document["schema"] = schema
    path.write_text(json.dumps(document))
    with pytest.raises(ME.MeasurementEvidenceError, match="schema"):
        ME.verify_paired_measurement(path, expected=binding)


@pytest.mark.parametrize("change", ["missing-child", "duplicate-child", "statistics", "trial"])
def test_parent_bound_report_refuses_incomplete_or_changed_evidence(produced, change):
    if change == "missing-child":
        produced.parent["measurement_manifests"].pop()
    elif change == "duplicate-child":
        produced.parent["measurement_manifests"][1] = dict(produced.parent["measurement_manifests"][0])
    elif change == "statistics":
        produced.parent["statistics"]["aggregate"]["median_speedup"] = 1000
    else:
        produced.parent["trials"][0]["agent_evidence_sha256"] = "f" * 64
    produced.path.chmod(0o644)
    produced.path.write_text(json.dumps(produced.parent))
    produced.path.chmod(0o444)
    with pytest.raises(produced.reporting.ReportingGateError):
        produced.reporting.load_current_experiment(
            produced.path, expected_sha256=C.sha256_file(produced.path), handoffs=produced.handoffs
        )


def test_packaged_measurement_owner_imports_without_native_engines(tmp_path):
    program = """
import importlib.abc, json, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in {'perf_agent_stage', 'run_paired_perf_bench', 'run_agentic_perf_experiment',
                    'perf_reporting', 'perf_experiment_stats', '_pbcommon', '_common'}:
            raise AssertionError('native engine import: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase2 import measurement_evidence
from merlin_experiments.phase2 import reporting, statistics
assert callable(measurement_evidence.verify_paired_measurement)
assert callable(measurement_evidence.completion_report)
assert callable(reporting.load_current_experiment)
assert callable(statistics.predeclare) and callable(statistics.evaluate)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, json.dumps([str(path) for path in python_import_roots()])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cold_packaged_report_replays_without_native_modules(produced, tmp_path):
    """The reader uses already-verified handoffs, never a replacement candidate verifier."""
    handoffs = tmp_path / "reader-handoffs.json"
    handoffs.write_text(json.dumps({name: vars(handoff) for name, handoff in produced.handoffs.items()}))
    program = """
import importlib.abc, json, pathlib, sys
from types import SimpleNamespace
sys.path[:0] = json.loads(sys.argv[1])
forbidden = {'perf_agent_stage', 'run_paired_perf_bench', 'run_agentic_perf_experiment',
             'perf_reporting', 'perf_experiment_stats', '_pbcommon', '_common'}
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in forbidden:
            raise AssertionError('native engine import: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase2 import reporting
handoffs = {key: SimpleNamespace(**value) for key, value in json.loads(pathlib.Path(sys.argv[4]).read_bytes()).items()}
parent, rows, result = reporting.load_current_experiment(
    pathlib.Path(sys.argv[2]), expected_sha256=sys.argv[3], handoffs=handoffs)
assert len(rows) == 24 and result['aggregate']['median_speedup'] == 1.25
assert not forbidden.intersection(sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            json.dumps([str(path) for path in python_import_roots()]),
            str(produced.path),
            produced.sha,
            str(handoffs),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_reporting_json_refuses_nonregular_inputs_before_reading(tmp_path, kind):
    path = tmp_path / "parent.json"
    if kind == "fifo":
        os.mkfifo(path, 0o444)
    else:
        path.mkdir(mode=0o555)
    program = """
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from merlin_experiments.phase2 import reporting
path = pathlib.Path(sys.argv[2])
# Frozen directories still use the generic guard legitimately.
reporting.require_read_only(path, label='frozen input')
try:
    reporting.read_immutable_json(path, '0' * 64, label='parent experiment manifest')
except reporting.ReportingGateError as exc:
    assert 'ordinary file' in str(exc), str(exc)
else:
    raise AssertionError('nonregular JSON input accepted')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"), str(path)],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots()))),
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        path.chmod(0o700 if kind == "directory" else 0o600)
    assert list(tmp_path.iterdir()) == [path]
