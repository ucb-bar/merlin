"""An explicit incomplete seal stops spend without pretending the functional run converged."""
from __future__ import annotations

import importlib.util
import sys

import pytest

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
    loop = _loop()
    loop._validate_seal_current_request(
        seal_current=False, resume=False, legacy_continuous=False)
    loop._validate_seal_current_request(
        seal_current=True, resume=True, legacy_continuous=False)

    with pytest.raises(RuntimeError, match="requires --resume"):
        loop._validate_seal_current_request(
            seal_current=True, resume=False, legacy_continuous=False)
    with pytest.raises(RuntimeError, match="certified --schedule"):
        loop._validate_seal_current_request(
            seal_current=True, resume=True, legacy_continuous=True)


def test_operator_seal_runs_official_grade_but_can_never_report_formal_completion():
    loop = _loop()
    import inspect

    body = inspect.getsource(loop.main)
    assert "or incomplete_operator_seal" in body
    assert "formal_complete = (not incomplete_operator_seal)" in body
    assert "if a.seal_current:" in body
    assert "return False" in body
