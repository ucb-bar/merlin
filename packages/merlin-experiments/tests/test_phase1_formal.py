"""Formal grading lifecycle through package-owned entrypoints, without hardware qualification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest
import yaml

from merlin.common.paths import data_path, python_import_roots


@pytest.mark.parametrize(
    "fault", ["wrong_firrtl", "wrong_roster", "wrong_digest", "changed_in_scope", "changed_reader", "raises"]
)
def test_private_facts_refuse_drift_and_restore_public_environment(tmp_path, monkeypatch, fault):
    from merlin_experiments.phase1.feedback import private_full_models as models
    from merlin_experiments.phase1.feedback.private_facts import selected_input_facts

    firrtl = tmp_path / "input.fir"
    firrtl.write_text("synthetic selected RTL\n")
    digest = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    document = {
        "inputs": {
            "target": "fixture",
            "fir_sha256": digest,
            "firrtl_inputs": [{"path": str(firrtl), "sha256": digest}],
        },
        "facts": {"source": {"config": "Fixture"}},
    }
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(document))
    effective = tmp_path / "effective.json"
    if fault == "wrong_firrtl":
        other = tmp_path / "other.fir"
        other.write_text("another RTL\n")
        other_digest = hashlib.sha256(other.read_bytes()).hexdigest()
        document["inputs"].update(fir_sha256=other_digest, firrtl_inputs=[{"path": str(other), "sha256": other_digest}])
    effective.write_text(json.dumps(dict(document, host_view=True)))
    spec = tmp_path / "private.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema": models.SCHEMA,
                "target": "fixture",
                "models": [
                    {
                        "id": "different" if fault == "wrong_roster" else "model",
                        "rtl_facts": str(raw),
                        "rtl_config": "Fixture",
                        "rtl_facts_sha256": "0" * 64
                        if fault == "wrong_digest"
                        else hashlib.sha256(raw.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(effective))
    before = dict(os.environ)
    with pytest.raises((ValueError, RuntimeError)):
        with selected_input_facts(spec, target="fixture", required_models=("model",)):
            assert os.environ["MERLIN_RTL_FACTS"] == str(raw)
            if fault == "changed_in_scope":
                raw.write_text(raw.read_text() + " ")
            elif fault == "changed_reader":
                os.environ["MERLIN_RTL_FACTS"] = str(effective)
            elif fault == "raises":
                os.environ["IN_SCOPE_ONLY"] = "must not leak"
                raise RuntimeError("synthetic build refusal")
    assert dict(os.environ) == before


@pytest.mark.parametrize("owner,flag", [("formal", "--descriptor"), ("freeze", "--repo")])
def test_installed_formal_help_has_no_native_import_or_target_selection(tmp_path, owner, flag):
    program = """
import importlib.abc, os, runpy, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in {'_common', 'grade_agent_run', 'freeze_run'}:
            raise AssertionError('native import: ' + name)
sys.meta_path.insert(0, NoNative())
os.environ['MERLIN_TARGET_EXPERIMENT'] = '/missing/descriptor.yaml'
sys.argv = ['merlin_experiments.phase1.feedback.' + sys.argv[1], '--help']
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    result = subprocess.run(
        [sys.executable, "-c", program, owner], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout


_HOST = r"""
import importlib.abc, json, os, runpy, sys
from pathlib import Path
from types import SimpleNamespace
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in {'_common', 'grade_agent_run', 'freeze_run', 'run_baseline_qa_loop'}:
            raise AssertionError('native import: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin.targetgen import capsule_runner as CR
from merlin.common import provenance
from merlin_experiments.phase1.feedback import freeze
CR._config_for_target = lambda *args: SimpleNamespace(rtl_tiers={'L3'}, tier_sim={'L2': 'spike'})
CR._rtl_tiers_of = lambda target: {'L3'}
CR.describe_l3_engine = lambda target: {
    'available': not bool(os.environ.get('MISSING_ENGINE')),
    'engine': 'fixture', 'fidelity': 'elaborated_rtl', 'reason': 'synthetic protocol fixture'
}
CR.oracle_adapters = lambda *args: {'L2': object(), 'L3': object()}
CR.suite_for = lambda target: target + '-capsule-bench'
# Target capability discovery is external to this lifecycle test. No cohort is removed.
CR._split_ineligible = lambda caps, target: (caps, [])
provenance.load_pins = lambda: {}
run = Path(sys.argv[sys.argv.index('--run-dir') + 1])
def event(value):
    with (run / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
real_freeze = freeze.freeze
def freeze_and_observe(root, *, repo):
    record = real_freeze(root, repo=repo)
    event('freeze')
    if os.environ.get('MUTATE_AFTER_FREEZE'):
        (root / 'submission/compiler.py').write_text('changed after freeze\n')
    return record
freeze.freeze = freeze_and_observe
def external_execution(caps, package_dir, *, runs_root, oracle_adapters, target, **kwargs):
    if '--qa-timeout' in sys.argv:
        assert kwargs['timeout'] == int(sys.argv[sys.argv.index('--qa-timeout') + 1])
    labels = {cap['label'] for cap in caps}
    hidden = labels == {'hidden'}
    functional_hidden = hidden and '--hidden-capsule-mode' in sys.argv
    if functional_hidden:
        assert set(oracle_adapters) == {'L2'}
        executable = [cap for cap in caps if (cap.get('expected') or {}).get('outcome') != 'refuse']
        assert all(cap['oracle_tier_ceiling'] == 'L2' for cap in executable)
        assert all(cap['required_oracle_tiers'] == ['L2'] for cap in executable)
    assert bool((run / 'freeze.json').exists()) == hidden
    if hidden:
        record = json.loads((run / 'freeze.json').read_text())
        assert record['submission_sha256'] == record['submission_sha256_recheck']
        assert not record['workspace_mutable_after_freeze']
    event('hidden' if hidden else 'public')
    rows = []
    for cap in caps:
        tiers = {name: {'status': 'pass', 'derived_from_rtl': name == 'L3',
                        'engine': 'spike' if name == 'L2' else 'fixture',
                        'fidelity': 'functional_model' if name == 'L2' else 'elaborated_rtl'}
                 for name in oracle_adapters}
        missing = not tiers
        row = {
            'capsule': cap['name'], 'label': cap['label'], 'kind': 'isa',
            'status': 'not_gradeable_no_oracle' if missing else 'pass', 'tiers': tiers,
            'highest_tier': max(tiers) if tiers else None,
            'numeric': {'status': 'not_gradeable_no_oracle' if missing else 'pass', 'mismatch_count': 0},
            'trace_check': {'status': 'pass', 'violations': []}
        }
        if (cap.get('expected') or {}).get('outcome') == 'refuse':
            from merlin.targetgen.expected_refusal import verdict
            declined = {'reason': 'synthetic unsupported operation'}
            status, failure, record = verdict(cap, 'declined', None, declined)
            row.update(status=status, failure=failure, declined=declined, expected_refusal=record,
                       tiers={}, numeric={'status': 'skipped'}, trace_check={'status': 'skipped'})
            if os.environ.get('MALFORMED_REFUSAL'):
                row['expected_refusal']['expected'] = {'reason': 'different', 'evidence': 'different'}
        result = Path(runs_root) / 'runs' / CR.suite_for(target) / cap['name']
        result.mkdir(parents=True)
        (result / 'capsule_result.json').write_text(json.dumps(row))
        rows.append(row)
    return rows
CR.run_suite = external_execution
sys.argv[0] = 'merlin_experiments.phase1.feedback.formal'
runpy.run_module(sys.argv[0], run_name='__main__')
"""


def _run_formal(tmp_path, *, flags=(), fault=None, refusals=False):
    from test_phase1_feedback import _inputs

    run, public, descriptor, host, env, _ = _inputs(tmp_path)
    hidden = tmp_path / "hidden/H"
    hidden.mkdir(parents=True)
    capsule = json.loads((public / "A/capsule.yaml").read_text())
    capsule.update(name="H", label="hidden")
    if '--hidden-capsule-mode' in flags:
        from merlin_experiments.phase1.feedback import hidden_capsule_policy

        mode = flags[flags.index('--hidden-capsule-mode') + 1]
        environment_path = run / 'environment.yaml'
        environment = yaml.safe_load(environment_path.read_text()) if environment_path.is_file() else {}
        environment['hidden_capsule_validation'] = hidden_capsule_policy.record(mode)
        environment_path.write_text(yaml.safe_dump(environment))
        capsule['required_oracle_tiers'] = ['L2', 'L3']
    (hidden / "capsule.yaml").write_text(json.dumps(capsule))
    if refusals:
        expected = {**capsule.get('expected', {}), 'outcome': 'refuse',
                    'refusal': {'reason': 'unsupported operation', 'evidence': 'fixture'}}
        for root, label in ((public, 'public'), (hidden.parent, 'hidden')):
            refused = {**capsule, 'name': 'R', 'label': label, 'expected': expected,
                       'required_oracle_tiers': ['L3']}
            (root / 'R').mkdir()
            (root / 'R/capsule.yaml').write_text(json.dumps(refused))
    host.write_text(_HOST)
    if fault:
        env[fault] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(host),
            "--descriptor",
            str(descriptor),
            "--repo",
            str(tmp_path),
            "--contract",
            str(data_path("contract")),
            "--run-dir",
            str(run),
            "--arm",
            "synthetic",
            "--capsules",
            str(public),
            "--hidden-capsules",
            str(hidden.parent),
            *flags,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return run, result


def test_explicit_hidden_spike_policy_runs_after_freeze_without_relaxing_public_or_private(tmp_path):
    run, result = _run_formal(tmp_path, flags=('--hidden-capsule-mode', 'functional_spike'))
    assert result.returncode == 1, result.stderr + result.stdout  # No private model gate supplied by fixture.
    manifest = yaml.safe_load((run / 'run_manifest.yaml').read_text())
    assert [json.loads(row) for row in (run / 'events.jsonl').read_text().splitlines()] == [
        'public', 'freeze', 'hidden',
    ]
    assert manifest['public_dev']['tier_reached']['L3'] == 1
    assert manifest['public_dev']['formal_complete'] is True
    hidden = manifest['hidden']
    assert hidden['formal_complete'] is True and hidden['tier_reached']['L2'] == 1
    assert hidden['tier_reached']['L3'] == 0 and hidden['rtl_certified'] is False
    assert hidden['claim'] == 'functional_corroboration'
    assert manifest['completion']['required_tier'] == 'L3'
    assert manifest['completion']['hidden_required_tier'] == 'L2'
    assert 'private_full_models:incomplete' in manifest['completion']['failures']
    assert manifest['model_mesh_sim']['fidelity'] == 'elaborated_rtl'
    assert json.loads((run / 'freeze.json').read_text())['hidden_capsule_validation']['mode'] == 'functional_spike'


@pytest.mark.parametrize('malformed', [False, True])
def test_real_projection_keeps_declared_refusal_out_of_execution_denominators(tmp_path, malformed):
    run, result = _run_formal(tmp_path, flags=('--hidden-capsule-mode', 'functional_spike'),
                              refusals=True, fault='MALFORMED_REFUSAL' if malformed else None)
    assert result.returncode == 1, result.stderr + result.stdout  # Private gate deliberately absent.
    manifest = yaml.safe_load((run / 'run_manifest.yaml').read_text())
    for name, tier in (('public_dev', 'L3'), ('hidden', 'L2')):
        phase = manifest[name]
        assert phase['n_capsules'] == phase['n_passed'] == 2
        assert phase['tier_reached'][tier] == 1
        assert phase['expected_refusal_scope']['n_declared_refusals'] == 1
        assert phase['expected_refusal_scope']['n_validated_refusals'] == (0 if malformed else 1)
        assert phase['formal_complete'] is (not malformed)
    assert manifest['public_dev']['pass_evidence']['rtl_backed'] == 1
    assert manifest['hidden']['pass_evidence']['rtl_backed'] == 0
    assert manifest['hidden']['functional_hidden_evidence']['n_spike_passed'] == 1


def test_real_public_freeze_rehash_hidden_lifecycle_outside_checkout(tmp_path):
    run, result = _run_formal(tmp_path)
    # Public/hidden L3 alone is no longer a Phase-1 completion claim.  This
    # lifecycle fixture intentionally supplies no private complete-network gate.
    assert result.returncode == 1, result.stderr + result.stdout
    assert [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()] == [
        "public",
        "freeze",
        "hidden",
    ]
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    assert manifest["completion"]["formal_grade_complete"] is False
    assert "private_full_models:incomplete" in manifest["completion"]["failures"]
    assert manifest["public_dev"]["n_capsules"] == manifest["hidden"]["n_capsules"] == 1
    assert manifest["hidden"]["cohort_admission"]["policy"] == "frozen_target_capability_operand_dtype"
    assert (run / "iterations/iteration_000/notes.md").is_file()
    assert (run / "final_report.md").is_file()


def test_selected_step_timeout_reaches_public_and_hidden_grade(tmp_path):
    run, result = _run_formal(tmp_path, flags=("--qa-timeout", "3217"))
    assert result.returncode == 1, result.stderr + result.stdout
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    assert manifest["grading_budget"]["step_timeout_s"] == 3217
    assert manifest["public_dev"]["n_passed"] == manifest["hidden"]["n_passed"] == 1


@pytest.mark.parametrize("flags", [("--no-oracle",), ("--skip-hidden",)])
def test_diagnostic_flags_cannot_report_formal_completion(tmp_path, flags):
    run, result = _run_formal(tmp_path, flags=flags)
    assert result.returncode == 1, result.stderr + result.stdout
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    assert not manifest["completion"]["formal_grade_complete"]
    if "--skip-hidden" in flags:
        assert manifest["hidden"]["completion_failures"] == ["hidden_grade_skipped"]
        assert not (run / "grading_hidden").exists()
    else:
        assert not manifest["gradeable"]
        assert manifest["model_mesh_sim"]["simulator"] == "fixture"


@pytest.mark.parametrize("flags", [(), ("--no-oracle",)])
def test_missing_formal_engine_refuses_even_no_oracle_before_grading(tmp_path, flags):
    run, result = _run_formal(tmp_path, flags=flags, fault="MISSING_ENGINE")
    assert result.returncode != 0
    assert "cannot resolve formal whole-model simulator" in result.stderr
    assert not (run / "events.jsonl").exists()
    assert not (run / "freeze.json").exists()


def test_post_freeze_mutation_refuses_hidden_execution(tmp_path):
    run, result = _run_formal(tmp_path, fault="MUTATE_AFTER_FREEZE")
    assert result.returncode != 0
    assert "FREEZE VIOLATION" in result.stderr
    assert [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()] == ["public", "freeze"]
    assert json.loads((run / "freeze.json").read_text())["workspace_mutable_after_freeze"]
    assert not (run / "grading_hidden").exists()
    assert not (run / "run_manifest.yaml").exists()


def test_repo_identity_uses_explicit_root_and_preserves_ambient_default(tmp_path, monkeypatch):
    from merlin import benchharness

    calls = []
    monkeypatch.setattr(benchharness, "sh", lambda argv, cwd=None: calls.append((argv, cwd)) or "fixture-sha")
    assert benchharness.repo_sha(repo=tmp_path) == "fixture-sha"
    assert benchharness.repo_sha() == "fixture-sha"
    assert calls == [(["git", "rev-parse", "HEAD"], tmp_path), (["git", "rev-parse", "HEAD"], None)]


def test_canonical_freeze_cli_uses_shared_source_hash(tmp_path):
    from merlin.common.tree_hash import hash_tree

    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "compiler.py").write_text("print('fixture')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "merlin_experiments.phase1.feedback.freeze",
            "--repo",
            str(tmp_path),
            "--run-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "freeze.json").read_text())
    expected = hash_tree(submission)
    assert record["submission_sha256"] == expected["sha256"]
    assert record["submission_files"] == expected["n_files"]
    assert record["repo_sha"] == "unknown"


@pytest.mark.parametrize("missing", ["--capsules", "--hidden-capsules", "--contract"])
def test_canonical_cli_refuses_implicit_roots_before_grading(tmp_path, missing):
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: synthetic\n")
    flags = {"--capsules": "public", "--hidden-capsules": "hidden", "--contract": "contract"}
    del flags[missing]
    run = tmp_path / "never-created"
    command = [
        sys.executable,
        "-m",
        "merlin_experiments.phase1.feedback.formal",
        "--descriptor",
        str(descriptor),
        "--repo",
        str(tmp_path),
        "--run-dir",
        str(run),
        "--arm",
        "synthetic",
    ]
    for key, value in flags.items():
        command += [key, value]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "installed formal grading requires" in result.stderr
    assert missing in result.stderr
    assert "cannot resolve formal whole-model simulator" not in result.stderr
    assert not run.exists()


def test_phase_manifest_keeps_host_correctness_and_verified_device_coverage_distinct():
    from merlin_experiments.phase1.feedback import formal

    score = {
        "n_capsules": 2,
        "n_passed": 2,
        "functional_pass": True,
        "host_lane_scored_pass": 1,
        "device_lane_scored_pass": 1,
        "device_evidence_count": 1,
        "spec_qualified_device_pass": 0,
        "device_unqualified_pass": 1,
        "host_guard_pass": 0,
        "host_guard_total": 0,
    }
    manifest = formal._phase_manifest(score, required_tier="L3")
    assert manifest["functional_pass"] is True and manifest["n_passed"] == 2
    assert manifest["host_lane_scored_pass"] == 1 and manifest["device_lane_scored_pass"] == 1
    assert manifest["spec_qualified_device_pass"] == 0 and manifest["device_unqualified_pass"] == 1
    assert manifest["formal_complete"] is False
