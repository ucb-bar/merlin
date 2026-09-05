"""The per-op profiler's WIRING: asking for a profile must either instrument, or say it did not.

Board-free. Every case below stops before any build or deploy — the point is exactly that the
refusal happens before a board slot is spent.

WHY THIS FILE EXISTS. ``run_on_k1(..., op_profile=True)`` routes a version-2 multi-program session
to ``build_k1_session_binary``, which has no ``op_profile`` parameter. The flag was therefore
DROPPED: the build never instrumented the IR, no ``opprof_table.json`` was written, no ``PROF``
lines reached the console, and the result came back with no ``op_profile`` key. Downstream that read
as an empty op table, and the driver reported ``ok: true`` with a breakdown of nothing — a profile
saying a model has no attributable work, which is not the same statement as "the profiler was never
linked in". A silently un-instrumented run also costs a full build + board slot to produce it.
"""
from __future__ import annotations

import pytest

from merlin.mining import k1


def _session_bundle(tmp_path, version: int):
    d = tmp_path / f"bundle_v{version}"
    d.mkdir()
    (d / "session_contract.yaml").write_text(
        f"version: {version}\nkind: action_chunk\nstages:\n- only\n", encoding="utf-8")
    return d


def test_a_multi_program_session_refuses_a_profile_instead_of_dropping_the_flag(tmp_path,
                                                                               monkeypatch):
    monkeypatch.setattr(k1, "K1_HOST", "unreachable.invalid")
    bundle = _session_bundle(tmp_path, 2)
    with pytest.raises(k1.K1Error) as e:
        k1.run_on_k1(bundle, tmp_path / "work", object(), op_profile=True)
    msg = str(e.value)
    assert "op_profile" in msg
    # and it says what to do instead, so the refusal is actionable rather than a dead end
    assert "stages" in msg


def test_the_same_session_without_a_profile_is_untouched(tmp_path, monkeypatch):
    """The refusal is scoped to the profile request: an ordinary run must not start failing."""
    monkeypatch.setattr(k1, "K1_HOST", "unreachable.invalid")
    bundle = _session_bundle(tmp_path, 2)
    with pytest.raises(Exception) as e:                     # noqa: PT011 - it fails LATER, in build
        k1.run_on_k1(bundle, tmp_path / "work", object(), op_profile=False)
    assert "op_profile" not in str(e.value)


def test_a_version_1_session_still_profiles(tmp_path, monkeypatch):
    """A version-1 session is a SINGLE program (the harness loops its steps internally), so it goes
    through build_k1_binary and IS instrumentable. Refusing it too would have cost the study its
    resnet50/lstmnetvit profiles."""
    monkeypatch.setattr(k1, "K1_HOST", "unreachable.invalid")
    bundle = _session_bundle(tmp_path, 1)
    with pytest.raises(Exception) as e:                     # noqa: PT011 - fails later, in build
        k1.run_on_k1(bundle, tmp_path / "work", object(), op_profile=True)
    assert "op_profile" not in str(e.value)


def test_no_board_no_run(tmp_path, monkeypatch):
    """Fail-closed precedence: an unset host is reported as such, not as a profile refusal."""
    monkeypatch.setattr(k1, "K1_HOST", "")
    with pytest.raises(k1.K1Error) as e:
        k1.run_on_k1(tmp_path, tmp_path / "work", object(), op_profile=True)
    assert "MERLIN_K1_HOST" in str(e.value)


# =================================================================================================
# The other half of the same defect: a run that comes back carrying nothing must be a BLOCKER.
# =================================================================================================

from merlin.llvmlower import op_profile as opf  # noqa: E402


def test_an_empty_op_table_is_a_blocker_not_an_empty_profile():
    b = opf.table_blocker([])
    assert b and "Nothing was measured" in b
    assert opf.table_blocker(None)


def test_a_table_whose_ops_never_ran_is_a_blocker():
    b = opf.table_blocker([{"id": 0, "mlir_op": "linalg.matmul", "ticks": 0, "hits": 0}])
    assert b and "ZERO recorded hits" in b


def test_a_real_table_is_not_blocked():
    assert opf.table_blocker([{"id": 0, "mlir_op": "linalg.matmul", "ticks": 700, "hits": 7}]) is None
