"""Completed host artifacts may remove duplicate work, never missing evidence."""

import json
import shutil
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import authoring
from merlin_experiments.phase1.feedback import compiler_profile_runtime as profile
from merlin_experiments.phase1.feedback import grade_reuse as reuse

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import tier_cache


@pytest.fixture
def grade(tmp_path, monkeypatch):
    run = tmp_path / "run"
    candidate = run / "_qa_work/cand_00/submission"
    candidate.mkdir(parents=True)
    (candidate / "compiler.py").write_text("print('compiler')\n")
    runs = run / "_qa_work/runs_00"
    result = runs / "runs/fixture/operator/capsule_result.json"
    result.parent.mkdir(parents=True)
    (result.parent / "program.elf").write_bytes(b"\x7fELF\x02\x01" + b"\0" * 10 + b"\x02\0" + b"original executable")
    public = tmp_path / "public"
    public.mkdir()
    (public / "capsule.yaml").write_text("original capsule\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture\n")
    instrument = tmp_path / "instrument.py"
    instrument.write_text("original grader\n")
    cap = {"name": "operator", "required_oracle_tiers": ["L2", "L3"]}
    owner = {"profile_sha256": "p" * 64, "snapshot_sha256": "s" * 64, "archive": "original"}
    inputs = SimpleNamespace(
        compiler_profile_service=object(),
        verify_grade_inputs=lambda: None,
        context=SimpleNamespace(target="fixture", descriptor=descriptor),
        contract=None,
        additional_forbidden=(),
    )
    monkeypatch.setattr(profile, "compiler_profile_binding", lambda *a, **kw: dict(owner))
    monkeypatch.setattr(tier_cache, "grading_path", lambda _: (instrument,))
    monkeypatch.setattr(CR, "discover_capsules", lambda *a, **kw: [cap])
    row = {
        "capsule": "operator",
        "kind": "operator",
        "status": "pass",
        "failure": None,
        "tiers": {t: {"status": "pass"} for t in ("L0", "L1", "L2", "L3")},
        "original_compiler_child": {
            "schema": "merlin.original_model_child.v1",
            "candidate_sha256": reuse._source(candidate),
            **owner,
        },
    }
    result.write_text(json.dumps(row))
    submission = tmp_path / "workspace/submission"
    shutil.copytree(candidate, submission)
    verdict = {"all_pass": True, "gradeable": True, "n_capsules": 1, "n_passed": 1}
    return SimpleNamespace(
        run=run,
        candidate=candidate,
        runs=runs,
        result=result,
        public=public,
        inputs=inputs,
        cap=cap,
        owner=owner,
        row=row,
        verdict=verdict,
        submission=submission,
        instrument=instrument,
    )


def retain(g):
    return reuse.retain_driver_grade(
        candidate=g.candidate, runs=g.runs, public=g.public, verdict=g.verdict, inputs=g.inputs, run_dir=g.run
    )


def checkpoint(g, receipt):
    return reuse.reuse_checkpoint(
        receipt,
        submission=g.submission,
        inputs=g.inputs,
        run_dir=g.run,
        eligible=["operator"],
        held=[],
        cert_tier="L3",
        attempt=1,
    )


def test_same_completed_original_grade_reuses_all_required_tiers(grade):
    receipt = retain(grade)
    assert receipt is not None
    result = checkpoint(grade, receipt)
    assert result["all_pass"] and result["n_rtl_passed"] == 1
    assert result["evidence_reuse"]["measured_now"] is False
    assert result["evidence_reuse"]["runs_root"] == str(grade.runs)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "retained_source",
        "owner",
        "artifact",
        "missing_artifact",
        "missing_executable",
        "cohort",
        "instrument",
        "environment",
        "admission",
    ],
)
def test_changed_identity_never_reuses(grade, monkeypatch, change):
    receipt = retain(grade)
    if change in {"source", "retained_source"}:
        root = grade.submission if change == "source" else grade.candidate
        (root / "compiler.py").write_text("changed compiler\n")
    elif change == "owner":
        grade.owner["profile_sha256"] = "new owner"
    elif change == "artifact":
        grade.result.write_text(json.dumps({**grade.row, "status": "fail"}))
    elif change == "missing_artifact":
        grade.result.unlink()
    elif change == "missing_executable":
        (grade.result.parent / "program.elf").unlink()
    elif change == "cohort":
        (grade.public / "capsule.yaml").write_text("different obligation\n")
    elif change == "instrument":
        grade.instrument.write_text("different grader\n")
    elif change == "environment":
        monkeypatch.setenv("MERLIN_NATIVE_SIMULATOR_PROTOCOL", "changed")
    else:
        grade.inputs.verify_grade_inputs = lambda: (_ for _ in ()).throw(ValueError("owner refused"))
    assert checkpoint(grade, receipt) is None


@pytest.mark.parametrize("change", ["tier", "child", "owner", "source", "partial", "ungradeable"])
def test_screen_or_missing_proof_is_not_retained(grade, change):
    if change == "tier":
        grade.row["tiers"].pop("L3")
    elif change == "child":
        grade.row.pop("original_compiler_child")
    elif change == "owner":
        grade.row["original_compiler_child"]["snapshot_sha256"] = "wrong"
    elif change == "source":
        grade.row["original_compiler_child"]["candidate_sha256"] = "wrong"
    elif change == "partial":
        grade.result.unlink()
    else:
        grade.verdict["gradeable"] = False
    if grade.result.exists():
        grade.result.write_text(json.dumps(grade.row))
    assert retain(grade) is None


def test_refusal_bound_to_original_declaration_keeps_separate_scope(grade):
    expected = {"reason": "unsupported", "evidence": "original hardware declaration"}
    decline = {"reason": "unsupported"}
    grade.cap["expected"] = {"outcome": "refuse", "refusal": expected}
    grade.row.update(
        tiers={}, declined=decline, expected_refusal={"status": "met", "expected": expected, "declined": decline}
    )
    grade.result.write_text(json.dumps(grade.row))
    result = checkpoint(grade, retain(grade))
    assert result["all_pass"] and result["n_expected_refusal_passed"] == 1
    assert result["n_rtl_passed"] == 0
    grade.row["expected_refusal"]["expected"] = {"reason": "different", "evidence": "invented"}
    grade.result.write_text(json.dumps(grade.row))
    assert retain(grade) is None


def test_no_original_owner_retains_legacy_grade_path(grade):
    grade.inputs.compiler_profile_service = None
    assert retain(grade) is None
    assert checkpoint(grade, {"all_pass": True, "n_passed": 146}) is None


def test_partial_or_held_cohort_does_not_skip_checkpoint(grade):
    receipt = retain(grade)
    for eligible, held in (([], []), (["operator"], [{"capsule": "missing"}]), (["operator", "operator"], [])):
        assert (
            reuse.reuse_checkpoint(
                receipt,
                submission=grade.submission,
                inputs=grade.inputs,
                run_dir=grade.run,
                eligible=eligible,
                held=held,
                cert_tier="L3",
                attempt=1,
            )
            is None
        )


def test_model_legacy_tiers_cannot_replace_candidate_qualification(grade):
    grade.row["kind"] = "model"
    grade.cap["kind"] = "model"
    grade.result.write_text(json.dumps(grade.row))
    assert retain(grade) is None
    grade.row["model_execution_check"] = {"kind": "owned_host_model_execution", "status": "fail"}
    grade.result.write_text(json.dumps(grade.row))
    assert retain(grade) is None


def test_actual_checkpoint_consumer_reuses_original_run_and_advisory(grade, monkeypatch):
    import ast
    import inspect

    from merlin.targetgen import capsule_grade as CG

    receipt = retain(grade)
    body = ast.parse(inspect.getsource(authoring.execute)).body[0].body
    fn = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == "_verilator_grade")
    calls = []
    scope = dict(
        vars(authoring),
        ws=grade.submission.parent,
        run_dir=grade.run,
        grading_inputs=grade.inputs,
        _verify_implementation_sources=lambda: None,
        treatment=SimpleNamespace(checkpoint_feedback=lambda root, **kw: calls.append((root, kw)) or {"ok": True}),
    )
    monkeypatch.setattr(authoring.CERT, "_cert_tier_name", lambda **kw: "L3")
    monkeypatch.setattr(authoring.CERT, "_pilot_capsule_dirs", lambda **kw: {"operator": grade.public})
    monkeypatch.setattr(CG, "grade", lambda *a, **kw: pytest.fail("duplicate numerical grade"))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "checkpoint-reuse", "exec"), scope)
    result = scope["_verilator_grade"](1, ["operator"], [], driver_receipt=receipt)
    assert result["all_pass"] and result["rtl_checks"] == {"ok": True}
    assert calls == [(grade.runs, {"capsule_roots": (grade.public,)})]


@pytest.mark.parametrize("valid, supplied, expected_calls", [(True, True, 0), (False, True, 1), (True, False, 1)])
def test_codex_noop_finalization_reuses_only_current_host_receipt(grade, monkeypatch, valid, supplied, expected_calls):
    receipt = retain(grade)
    assert checkpoint(grade, receipt)["all_pass"]
    if not valid:
        (grade.submission / "compiler.py").write_text("compiler edited after grade\n")
    ws = grade.submission.parent
    (ws / "qa").mkdir()
    calls = []
    monkeypatch.setattr(authoring.EX, "resolve_driver", lambda *a, **kw: "codex")
    audit = SimpleNamespace(audit_transcript=lambda *a, **kw: {"clean": True, "hits": []})
    result = authoring.finalize_report(
        ws,
        grade.run,
        "codex",
        "high",
        "none",
        {},
        "merlin_assisted",
        grade.verdict,
        30,
        provider=None,
        context=None,
        audit=audit,
        grade_callback=lambda *a: calls.append(a) or grade.verdict,
        **(
            {
                "reuse_noop_grade": lambda: receipt.current(
                    submission=grade.submission, inputs=grade.inputs, run_dir=grade.run
                )
            }
            if supplied
            else {}
        ),
    )
    assert len(calls) == expected_calls
    assert result.get("regrade_reused", False) == (expected_calls == 0)
