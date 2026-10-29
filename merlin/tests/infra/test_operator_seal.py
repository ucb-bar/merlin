"""An explicit incomplete seal stops spend without pretending the functional run converged."""

from __future__ import annotations

import importlib.util
import sys

import pytest
from merlin_experiments.phase1 import run_inputs as RI

from merlin.common.paths import merlin_dir


def _loop():
    harness = merlin_dir() / "experiments" / "capsule_bench" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    path = harness / "run_baseline_qa_loop.py"
    spec = importlib.util.spec_from_file_location("operator_seal_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_seal_is_resume_only_and_rejects_the_legacy_progress_path():
    RI.validate_seal_current_request(seal_current=False, resume=False, legacy_continuous=False)
    RI.validate_seal_current_request(seal_current=True, resume=True, legacy_continuous=False)

    with pytest.raises(RuntimeError, match="requires --resume"):
        RI.validate_seal_current_request(seal_current=True, resume=False, legacy_continuous=False)
    with pytest.raises(RuntimeError, match="certified --schedule"):
        RI.validate_seal_current_request(seal_current=True, resume=True, legacy_continuous=True)


def test_operator_seal_runs_official_grade_but_can_never_report_formal_completion():
    import inspect

    from merlin_experiments.phase1 import authoring as loop

    body = inspect.getsource(loop.execute)
    assert "or incomplete_operator_seal" in body
    import ast

    selected = next(
        node
        for node in ast.walk(ast.parse(body))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_completion_for_scope"
    )
    assert any(
        arg.arg == "checkpoint" and isinstance(arg.value, ast.Name) and arg.value.id == "incomplete_operator_seal"
        for arg in selected.keywords
    )
    for scope in ("full", "capsules"):
        result = loop.CERT._completion_for_scope(
            validation_scope=scope,
            numeric_all_pass=True,
            workflow_conformant=True,
            official_grade={"complete": True},
            telemetry={"complete": True},
            checkpoint=True,
        )
        assert result["converged"] is result["formal_complete"] is False
    assert "if a.seal_current:" in body
    assert "return False" in body
