"""Every ``qa_check`` helper the self-check calls must actually be there.

``agent_selfcheck`` assembles each verdict row through ``_qc.<helper>``. When one of those helpers went
missing -- a branch integration reverted ``qa_check`` to a side that never had
``_execution_digest_from_result`` while ``agent_selfcheck`` kept calling it -- the self-check raised
``AttributeError`` *after* the grade had already built, run and compared every capsule. Nothing was
printed but the oracle banner, so the JSON verdict never appeared on stdout.

That failure is invisible where it matters. Every consumer parses the verdict out of stdout, so all of
them reported "no capsules": readiness section G showed ``n=None/None`` on the from-clean C++ build, on
spike-L2 and on verilator-L3 at once, and the whole gemmini launch read as NO-GO with three broken
oracles -- when the oracles had in fact run to completion (the verilator probe spent 148 s doing it).

A helper reference is not exercised by importing either module, so this test resolves them the way the
running self-check does: it reads the attribute names off ``agent_selfcheck``'s own source and requires
each one to exist on ``qa_check``. Parsed structurally (``partition``/``split``), no regex.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _referenced_qa_check_helpers() -> set[str]:
    """The ``_qc.<name>`` attributes ``agent_selfcheck`` reaches for, read off its source."""
    src = (HARNESS / "agent_selfcheck.py").read_text(encoding="utf-8")
    names: set[str] = set()
    for chunk in src.split("_qc.")[1:]:
        ident = ""
        for ch in chunk:
            if ch.isalnum() or ch == "_":
                ident += ch
            else:
                break
        if ident:
            names.add(ident)
    return names


def _qa_check():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("qa_check", HARNESS / "qa_check.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"qa_check not importable here: {type(e).__name__}: {e}")
    return mod


def test_the_selfcheck_references_at_least_one_helper():
    """Guard the guard: an empty set would make the check below pass vacuously."""
    assert _referenced_qa_check_helpers()


def test_every_helper_the_selfcheck_calls_exists_on_qa_check():
    qc = _qa_check()
    missing = sorted(n for n in _referenced_qa_check_helpers() if not hasattr(qc, n))
    assert not missing, (
        f"agent_selfcheck calls qa_check.{missing} which qa_check does not define; the self-check will "
        f"raise AttributeError after grading and print no verdict JSON at all")


def test_the_execution_digest_bridge_degrades_to_none_rather_than_raising(tmp_path):
    """The promotion identity is OPTIONAL, so its absence must cost a key, never the verdict."""
    qc = _qa_check()
    assert qc._execution_digest_from_result(tmp_path / "no_such_capsule_result.json") is None
