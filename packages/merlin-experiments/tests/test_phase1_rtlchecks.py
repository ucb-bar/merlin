"""Advisory treatment inputs without native imports, FileCheck or simulators."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import source_inputs, treatments
from merlin_experiments.phase1.feedback import rtlchecks


def test_import_does_not_resolve_machine_or_corpus_defaults(tmp_path):
    program = """
import importlib.abc, sys
from merlin.common import paths
from merlin.targetgen import corpora
def forbidden(*args, **kwargs): raise AssertionError('ambient discovery during import')
paths.ext_path = forbidden
corpora.capsule_corpus_roots = forbidden
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'_common', 'qa_check_rtlchecks', 'run_baseline_qa_loop'}:
            raise AssertionError('native import: '+fullname)
sys.meta_path.insert(0, DenyNative())
from merlin_experiments.phase1.feedback import rtlchecks
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_callbacks_use_frozen_roots_and_preserve_target_interleaving(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(rtlchecks.RUN, "load_facts", lambda target: {"target": target})
    monkeypatch.setattr(rtlchecks.RUN, "find_filecheck", lambda candidates: None)

    def screen(directory, facts, index, fc, *, write, target):
        observed.append((target, index["same"], write, facts["target"]))
        return {"capsule": "same", "verdict": "reject", "screen": {"verdict": "reject"}}

    monkeypatch.setattr(rtlchecks.RUN, "screen_run", screen)
    callbacks = {}
    roots = {}
    for target in ("alpha", "beta"):
        root = tmp_path / target / "frozen/same"
        root.mkdir(parents=True)
        (root / "capsule.yaml").write_text(yaml.safe_dump({"name": "same", "label": "public"}))
        roots[target] = root.parent
        generated = tmp_path / target / "evidence/runs" / f"{target}-capsule-bench/same/generated"
        generated.mkdir(parents=True)
        # Both endpoint artifact forms must discover the same directory only once.
        (generated / "instruction_trace.json").write_text("{}")
        (generated / "kernel.S").write_text("synthetic")
        callbacks[target] = rtlchecks.treatment(SimpleNamespace(target=target))
    for target in ("alpha", "beta", "alpha"):
        result = callbacks[target].checkpoint_feedback(tmp_path / target / "evidence", capsule_roots=(roots[target],))
        assert result[0]["verdict"] == "reject"
    assert observed == [
        (target, roots[target] / "same/capsule.yaml", True, target) for target in ("alpha", "beta", "alpha")
    ]


@pytest.mark.parametrize("all_pass", [True, False])
def test_round_feedback_is_advisory_on_failure_and_uses_actual_qa_root(tmp_path, monkeypatch, all_pass):
    seen = []
    context = SimpleNamespace(target="fixture")

    def qa(*args, **kwargs):
        seen.append((args[1], kwargs))
        return {"all_pass": all_pass, "n_capsules": 1}

    monkeypatch.setattr(rtlchecks._base, "run", qa)

    def unavailable(root, **kwargs):
        assert kwargs["capsule_roots"] == (tmp_path / "frozen",)
        assert kwargs["context"] is context
        raise RuntimeError("synthetic unavailable structural checker")

    monkeypatch.setattr(rtlchecks, "feedback", unavailable)
    treatment = rtlchecks.treatment(context)
    verdict = treatment.qa_runner(
        "candidate",
        str(tmp_path / "frozen"),
        tmp_path,
        {"public"},
        True,
        1,
        contract=tmp_path / "contract",
        additional_forbidden=("marker",),
    )
    assert verdict["all_pass"] is all_pass
    assert "synthetic unavailable" in verdict["rtl_checks_error"]
    assert seen == [
        (
            str(tmp_path / "frozen"),
            dict(context=context, contract=tmp_path / "contract", additional_forbidden=("marker",)),
        )
    ]


def test_treatment_callbacks_and_rtl_policy_owners_are_in_existing_inventory(tmp_path):
    entrypoint = Path(source_inputs.__file__).with_name("controller.py")
    record = source_inputs.record(repo=tmp_path, entrypoint=entrypoint)
    selected = rtlchecks.treatment(SimpleNamespace(target="fixture"))
    identity = treatments.record(selected, record)
    for callback in ("qa_runner", "checkpoint_feedback"):
        assert identity["callbacks"][callback]["source_input"] == "phase1:source:feedback/rtlchecks.py"
    for owner in ("rtl_check_runner", "rtl_check_compiler", "rtl_checks", "circt_gate"):
        assert "phase1:startup:" + owner in record["inputs"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--bundle", "raw_baseline_fixture"],
        ["--bundle", "merlin_assisted_rtlchecks_fixture", "--bundle=merlin_assisted_rtlchecks_fixture"],
        ["--bundle"],
        ["--bundle", "../merlin_assisted_rtlchecks_fixture"],
        ["--bundle", "merlin_assisted_rtlchecks_fixture", "--arm", "raw_baseline"],
    ],
)
def test_strict_bundle_and_arm_refusal(tmp_path, arguments):
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("bundle_id: merlin_assisted_rtlchecks_fixture\narm: merlin_rtlchecks\n")
    with pytest.raises(ValueError):
        rtlchecks.prepare_arguments(arguments, bundle_manifest=manifest)


def test_valid_explicit_bundle_injects_only_missing_arm(tmp_path):
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("bundle_id: merlin_assisted_rtlchecks_fixture\narm: merlin_rtlchecks\n")
    args = ["--bundle=merlin_assisted_rtlchecks_fixture"]
    assert rtlchecks.prepare_arguments(args, bundle_manifest=manifest) == args + ["--arm", "merlin_assisted"]
    assert args == ["--bundle=merlin_assisted_rtlchecks_fixture"]


def test_explicit_capsule_index_preserves_first_name_wins_without_ambient_roots(tmp_path):
    roots = []
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "same").mkdir(parents=True)
        (root / "same/capsule.yaml").write_text("name: same\n")
        roots.append(root)
    assert rtlchecks.RUN.capsule_index(tuple(roots)) == {"same": roots[0] / "same/capsule.yaml"}
    assert rtlchecks.RUN.capsule_index(tuple(reversed(roots))) == {"same": roots[1] / "same/capsule.yaml"}
    assert rtlchecks.RUN.capsule_index(()) == {}


def test_filecheck_uses_explicit_candidates_then_path_without_execution(tmp_path, monkeypatch):
    executable = tmp_path / "FileCheck"
    executable.write_text("not executed")
    calls = []
    monkeypatch.setattr(rtlchecks.RUN.shutil, "which", lambda name: calls.append(name) or "PATH/FileCheck")
    assert rtlchecks.RUN.find_filecheck((tmp_path / "absent", executable)) == str(executable)
    assert calls == []
    assert rtlchecks.RUN.find_filecheck() == "PATH/FileCheck"
    assert calls == ["FileCheck"]


@pytest.mark.parametrize("explicit", [False, True])
def test_core_cli_uses_explicit_filecheck_or_path_never_chipyard(tmp_path, monkeypatch, explicit):
    from merlin.common import paths

    def forbidden(*args, **kwargs):
        raise AssertionError("generic CLI must not discover a machine-specific target checkout")

    monkeypatch.setattr(paths, "ext_path", forbidden)
    monkeypatch.setattr(paths, "repo_root", forbidden)
    monkeypatch.setattr(rtlchecks.RUN, "load_facts", lambda target: {})
    monkeypatch.setattr(rtlchecks.RUN, "capsule_corpus_roots", lambda: [])
    monkeypatch.setattr(rtlchecks.RUN, "iter_run_dirs", lambda root: [])
    path_queries = []
    monkeypatch.setattr(rtlchecks.RUN.shutil, "which", lambda name: path_queries.append(name) or "PATH/FileCheck")
    arguments = [str(tmp_path), "--target", "fixture"]
    if explicit:
        executable = tmp_path / "chosen-FileCheck"
        executable.write_text("never executed")
        arguments += ["--filecheck", str(executable)]
    assert rtlchecks.RUN.main(arguments) == 0
    assert path_queries == ([] if explicit else ["FileCheck"])


def test_core_cli_refuses_missing_explicit_filecheck_before_facts(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("invalid explicit executable must not resolve facts or fall back")

    monkeypatch.setattr(rtlchecks.RUN, "load_facts", forbidden)
    monkeypatch.setattr(rtlchecks.RUN.shutil, "which", forbidden)
    with pytest.raises(SystemExit) as exc:
        rtlchecks.RUN.main([str(tmp_path), "--target", "fixture", "--filecheck", str(tmp_path / "absent")])
    assert exc.value.code == 2


def test_redaction_retains_advisory_failures_and_skips_only():
    result = rtlchecks._redact_rtl(
        {
            "capsule": "fixture",
            "verdict": "reject",
            "private": "not exposed",
            "filecheck": {"trace": {"ok": False, "diag": "first line\nnot exposed"}},
            "screen": {
                "verdict": "warn",
                "checks": [
                    {"id": "passed", "status": "pass"},
                    {"id": "failed", "status": "fail", "expected": 2, "got": 1, "private": "not exposed"},
                ],
                "skipped": [{"id": "not_run", "reason": "missing artifact", "private": "not exposed"}],
            },
        }
    )
    assert result["filecheck"] == {"trace": {"ok": False, "diag": "first line"}}
    assert [row["id"] for row in result["findings"]] == ["failed"]
    assert result["not_run"] == [{"id": "not_run", "reason": "missing artifact"}]
    assert "not exposed" not in repr(result)
