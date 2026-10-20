"""Candidate whole-model selfcheck feedback never borrows the host-graph result."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import qa, selfcheck

from merlin.common.paths import data_path

SENTINEL = "PRIVATE_OR_LEGACY_PROGRAM_SENTINEL"


def test_oracle_timeout_reason_does_not_exculpate_or_blame_emitted_program():
    reason = selfcheck.CR._oracle_timeout_reason("selected simulator", "timed out after 1800 seconds")
    assert "UNMEASURED" in reason
    assert "cause is undetermined" in reason
    assert "not a defect" not in reason
    assert "budget/size fact" not in reason
    assert "timed out after 1800 seconds" in reason


def _candidate_result(*, passed=False):
    status = "pass" if passed else "incomplete"
    required = {tier: {"status": "pass" if passed or tier != "L3" else "unverified"} for tier in ("L0", "L1", "L3")}
    return {
        "capsule": "model_case",
        "kind": "model",
        "status": status,
        "numeric": {"status": "pass", "first_mismatch": {"expected": SENTINEL}},
        "tiers": {"L3": {"status": "pass", "cycles": 999}, "L2": {"status": "pass"}},
        "trace_check": {"status": "pass", "summary": SENTINEL},
        "failure": {"plane": "legacy_model", "category": "FUNCTIONAL_MISMATCH", "detail": SENTINEL},
        "candidate_native_execution": {
            "numeric": {"status": "pass", "mismatch_count": 0, "first_mismatch": {"expected": SENTINEL}}
        },
        "candidate_native_model_check": {
            "schema": "merlin_candidate_native_model_check_v1",
            "status": status,
            "violations": [] if passed else ["candidate_required_tiers_unverified"],
            "emitted_host_compute": {"status": "clean"},
            "source_placement": {"status": "clean"},
            "completed_dispatch": {"status": "verified" if passed else "unverified"},
            "candidate_required_tiers": {"status": status, "required_tiers": ["L0", "L1", "L3"], "tiers": required},
            "candidate_source_coverage": (
                {"status": "verified", "n_source_operations": 2, "n_eligible": 1, "n_completed_eligible": 1}
                if passed
                else {"status": "unverified"}
            ),
            "native_status": "numeric_match_diagnostic",
        },
    }


def _write_result(runs_root: Path, result: dict):
    parent = runs_root / "runs" / "fixture-suite" / "model_case"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "capsule_result.json").write_text(json.dumps(result))
    generated = parent / "generated"
    generated.mkdir()
    (generated / "instruction_trace.json").write_text(json.dumps({"summary": SENTINEL}))
    (generated / "l2_engine_binding.json").write_text(json.dumps({"private": SENTINEL}))
    artifacts = parent / "artifacts"
    artifacts.mkdir()
    (artifacts / "legacy_console.log").write_text(SENTINEL)


def _context(root: Path):
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
    )


def _write_public_capsule(corpus: Path, *, kind: str):
    capsule = corpus / "model_case"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        json.dumps(
            {
                "name": "model_case",
                "kind": kind,
                "source_role": "handauthored_compiler_test",
                "label": "public",
                "operation": {"op": "movement"},
                "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
                "expected": {"instruction_classes": []},
                "required_oracle_tiers": ["L2"],
            }
        )
    )


def test_candidate_row_uses_closed_qa_projection_and_never_screen_passes(tmp_path):
    result = _candidate_result()
    _write_result(tmp_path, result)
    closed = qa._per_capsule_from_results(tmp_path)["model_case"]
    row, certified = selfcheck._candidate_selfcheck_row(result, closed, name="model_case", barrier_tier="L3")
    assert not certified and not row["pass"]
    assert row["tiers"] == {"L0": "pass", "L1": "pass", "L3": "unverified"}
    assert row["numeric"] == {"status": "pass", "mismatch_count": 0}
    assert row["execution_digest"] is None
    assert row["failure"]["detail"] == "candidate_required_tiers_unverified"
    assert SENTINEL not in json.dumps(row)
    assert not any(
        key in row
        for key in (
            "trace_summary",
            "trace_check",
            "your_artifacts",
            "sim_console_tail",
            "barrier_cycles",
            "barrier_engine_binding",
        )
    )


def test_candidate_row_only_passes_complete_required_tiers(tmp_path):
    result = _candidate_result(passed=True)
    _write_result(tmp_path, result)
    closed = qa._per_capsule_from_results(tmp_path)["model_case"]
    row, certified = selfcheck._candidate_selfcheck_row(result, closed, name="model_case", barrier_tier="L3")
    assert certified and row["pass"]
    assert row["candidate_native_verification"]["tiers"] == {"L0": "pass", "L1": "pass", "L3": "pass"}
    assert SENTINEL not in json.dumps(row)
    missing, certified = selfcheck._candidate_selfcheck_row(result, None, name="model_case", barrier_tier="L3")
    assert not certified and not missing["pass"]


def test_main_selfcheck_does_not_count_legacy_screen_or_publish_legacy_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    corpus = tmp_path / "public"
    _write_public_capsule(corpus, kind="model")
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a: ({"L3": object()}, "gsim"))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: "fixture-suite")
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)

    def grade(_submission, *, runs_root, **_kwargs):
        _write_result(Path(runs_root), _candidate_result())
        return {"n_capsules": 1, "n_passed": 0, "per_capsule": [{"capsule": "model_case", "status": "incomplete"}]}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck.main(
        ["--sim", "gsim", "--submission", str(submission)],
        context=_context(tmp_path),
        capsules_root=corpus,
        contract=data_path("contract"),
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["n_passed"] == report["n_screened_only"] == report["n_certified"] == 0
    assert report["per_capsule"][0]["candidate_native_verification"]["status"] == "incomplete"
    assert SENTINEL not in json.dumps(report)


@pytest.mark.parametrize(
    ("l2_status", "l3_status", "l3_log", "expected_tier", "expected_tail", "unreadable"),
    [
        ("pass", "fail", None, "L3", None, False),
        ("pass", "fail", "l3_console.log", "L3", "L3_PROGRESS\n", False),
        ("pass", "fail", "../private_console.log", "L3", None, False),
        ("fail", None, None, "L2", "L2_DONE\n", False),
        ("pass", None, None, "L2", None, False),
        ("pass", "pass", "l3_console.log", "L3", None, False),
        # A console that exists but cannot be read is named as unreadable, never as an empty tail.
        ("pass", "fail", "l3_console.log", "L3", None, True),
    ],
)
def test_main_selfcheck_uses_only_selected_tier_console(
    tmp_path, monkeypatch, capsys, l2_status, l3_status, l3_log, expected_tier, expected_tail, unreadable
):
    monkeypatch.chdir(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    corpus = tmp_path / "public"
    _write_public_capsule(corpus, kind="isa")
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a: ({"L3": object()}, "gsim"))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: "fixture-suite")
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)
    passing_case = l2_status == "pass" and l3_status in (None, "pass")

    def grade(_submission, *, runs_root, **_kwargs):
        l3 = {"status": l3_status}
        if l3_log is not None:
            l3["console_log"] = l3_log
        result = {
            "capsule": "model_case",
            "kind": "op",
            "status": "pass" if passing_case else "fail",
            "numeric": {"status": "pass"},
            # This console-selection control supplies its original independent
            # structural verdict; numerical screens alone cannot certify it.
            "trace_check": {"status": "pass", "trace_requirement_status": "pass"},
            "tiers": {"L2": {"status": l2_status, "console_log": "l2_console.log"}},
            "failure": {"category": "TESTBENCH_TIMEOUT", "tier": "L3"},
        }
        if l3_status is not None:
            result["tiers"]["L3"] = l3
        parent = Path(runs_root) / "runs" / "fixture-suite" / "model_case"
        parent.mkdir(parents=True)
        (parent / "capsule_result.json").write_text(json.dumps(result))
        artifacts = parent / "artifacts"
        artifacts.mkdir()
        (artifacts / "l2_console.log").write_text("L2_DONE\n")
        if l3_log is not None and Path(l3_log).name == l3_log:
            (artifacts / l3_log).write_text("L3_PROGRESS\n")
            if unreadable:
                (artifacts / l3_log).chmod(0)
        elif l3_log is not None:
            (artifacts.parent / "private_console.log").write_text("UNRELATED_PRIVATE_CONSOLE\n")
        return {
            "n_capsules": 1,
            "n_passed": int(passing_case),
            "per_capsule": [{"capsule": "model_case", "status": "pass" if passing_case else "fail"}],
        }

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck.main(
        ["--sim", "gsim", "--submission", str(submission)],
        context=_context(tmp_path),
        capsules_root=corpus,
        contract=data_path("contract"),
    )
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if passing_case else 1)
    row = report["per_capsule"][0]
    assert row["barrier_tier"] == expected_tier
    if expected_tail is None and passing_case:
        assert "sim_console_tail" not in row
    elif unreadable:
        # a console that exists and cannot be read is named, not shown as an absent console
        assert row["sim_console_tail"] is None and row["sim_console_tail_unreadable"].startswith("PermissionError")
    else:
        assert row["sim_console_tail"] == expected_tail
        assert "sim_console_tail_unreadable" not in row
    if expected_tier == "L3":
        assert "L2_DONE" not in json.dumps(report)
        assert "UNRELATED_PRIVATE_CONSOLE" not in json.dumps(report)


def test_model_layers_uses_candidate_projection_not_legacy_functional_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "layers"
    capsule = corpus / "model_case"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("{}")
    experiment = SimpleNamespace(model_layer_roots=lambda: [corpus], sim_via="fixture")
    monkeypatch.setattr("merlin.targetgen.target_experiment.load_target_experiment", lambda *_: experiment)
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda *a: {"L3": object()})
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: "fixture-suite")

    def grade(_submission, *, runs_root, **_kwargs):
        _write_result(Path(runs_root), _candidate_result())
        return {"failure": {"detail": SENTINEL}}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck._model_layers(tmp_path / "submission", "", timeout=1, workers=1, context=_context(tmp_path))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["n_passed_functional_tier"] == 0
    assert report["grade_failure"] is None
    assert report["per_capsule"][0]["candidate_native_verification"]["status"] == "incomplete"
    assert SENTINEL not in json.dumps(report)


def test_candidate_emission_string_becomes_actionable_bounded_failure():
    failure = selfcheck._public_candidate_emission_failure({
        'candidate_emission_failure': 'eligible source operation lacks unavoidable attributed contraction work; '
                                      'fused transforms need independent evidence',
    })
    assert failure['plane'] == 'candidate_compilation'
    assert failure['category'] == 'CANDIDATE_EMISSION_FAILED'
    assert 'unavoidable attributed contraction work' in failure['detail']
    assert len(failure['detail']) <= 800


def test_candidate_emission_feedback_cannot_copy_reference_payload():
    result = {'candidate_emission_failure': {
        'detail': 'compiler mismatch expected=123456.75 observed=98765.25',
        'reference': SENTINEL, 'golden': [SENTINEL], 'capture': {'weights': SENTINEL},
    }, 'numeric': {'first_mismatch': {'expected': SENTINEL}}, 'status': 'incomplete'}
    row, certified = selfcheck._candidate_selfcheck_row(result, None, name='public_model', barrier_tier='L2')
    serialized = json.dumps(row)
    assert not certified
    assert row['failure']['category'] == 'CANDIDATE_EMISSION_FAILED'
    assert SENTINEL not in serialized
    assert '123456.75' not in serialized and '98765.25' not in serialized
    assert row['numeric'] == {}


def test_public_model_linker_symbol_survives_long_argv_without_copying_private_payload():
    from merlin_experiments.phase1.feedback import selfcheck_results

    original = {
        "type": "SpikeModelError", "stage": "owned_host_execution",
        "detail": "command failed (rc 1): " + "/public/build/object.o " * 50
        + "\nSTDOUT:\nSTDERR:/private/build/device_shim.o: in function `device_call':\n"
        + "device_shim.c:(.text+0x124): undefined reference to `logical_kernel_0'\n"
        + "expected 123456.75 observed 98765.25\ncollect2: error: ld returned 1 exit status\n",
        "reference": SENTINEL, "golden": [SENTINEL],
        "recipe": {"path": "/operator/private/compilation_recipe.json", "sha256": "a" * 64},
    }
    before = json.dumps(original, sort_keys=True)
    failure = selfcheck_results._public_candidate_emission_failure({"candidate_emission_failure": original})
    assert failure["category"] == "CANDIDATE_EMISSION_FAILED"
    assert failure["detail"].startswith("Linker undefined symbols: logical_kernel_")
    assert "selected executable entry and per-group symbol binding" in failure["detail"]
    assert len(failure["detail"]) <= 800
    assert all(text not in json.dumps(failure) for text in (SENTINEL, "123456.75", "98765.25", "/private/build"))
    assert json.dumps(original, sort_keys=True) == before  # Original diagnostic/recipe pins are retained unchanged.


def test_nonpublic_model_stage_does_not_promote_a_linker_tail():
    failure = selfcheck._public_candidate_emission_failure({"candidate_emission_failure": {
        "stage": "prepare_runtime_bundle", "detail": "command failed: " + "x" * 1000
        + "\nSTDERR:undefined reference to `private_reference_symbol'\n",
    }})
    assert "private_reference_symbol" not in failure["detail"]
