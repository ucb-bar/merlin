"""A live submission and its frozen or snapshot copy have one source digest.

Measured on fresh7refcca: the live ``submission/`` and ``joint_frozen_submission`` differed only in six
``build/tiny/*`` files, yet their ledger digests differed (``3759389286b7b590`` vs ``9e4fcf9b10d6583d``),
because the digest counted ``build/`` while every grade, snapshot and frozen copy drops it. A certificate
keyed on the live tree could then never match the frozen bytes it described.
"""

import shutil

from merlin_experiments.phase1.feedback import promotion as P
from merlin_experiments.phase1.run_inputs import SNAPSHOT_IGNORED, strip_build_state


def _live(tmp_path):
    ws = tmp_path / "ws"
    sub = ws / "submission"
    (sub / "compiler").mkdir(parents=True)
    (sub / "compiler" / "lower.py").write_text("def lower():\n    return 1\n")
    (sub / "compiler" / "build.py").write_text("STEPS = 2\n")  # a source file NAMED like build state
    (sub / "manifest.yaml").write_text("name: x\n")
    return ws, sub


def _with_build_state(sub):
    for rel in ("build/tiny/kernel.o", "build/tiny/report.json", "compiler/build/kernel.ll",
                "native/CMakeCache.txt", "native/CMakeFiles/rules.make", ".git/HEAD"):
        (sub / rel).parent.mkdir(parents=True, exist_ok=True)
        (sub / rel).write_bytes(rel.encode())


def _frozen(sub, destination):
    # The joint completion's freeze rule (and a grade snapshot's): ignore, then strip build state.
    shutil.copytree(sub, destination, ignore=shutil.ignore_patterns(*SNAPSHOT_IGNORED))
    strip_build_state(destination)
    return destination


def test_live_tree_with_build_state_has_its_frozen_copys_digest(tmp_path):
    ws, sub = _live(tmp_path)
    clean = P.submission_tree_digest(sub)
    _with_build_state(sub)
    frozen = _frozen(sub, tmp_path / "frozen")
    assert not (frozen / "build").exists() and (frozen / "compiler" / "build.py").is_file()
    assert P.submission_tree_digest(sub) == P.submission_tree_digest(frozen) == clean
    assert P.submission_digests(ws)[0] == clean


def test_source_named_like_build_state_still_moves_the_digest(tmp_path):
    ws, sub = _live(tmp_path)
    _with_build_state(sub)
    before = P.submission_tree_digest(sub)
    (sub / "compiler" / "build.py").write_text("STEPS = 3\n")
    assert P.submission_tree_digest(sub) != before
    after = P.submission_tree_digest(sub)
    (sub / "build" / "tiny" / "kernel.o").write_bytes(b"rebuilt")
    assert P.submission_tree_digest(sub) == after


def test_component_digests_ignore_build_state(tmp_path):
    from test_current_source_promotions import workspace

    ws = workspace(tmp_path / "ws")
    whole, components, _ = P._submission_digests(ws)
    assert components
    _with_build_state(ws / "submission")
    assert P._submission_digests(ws)[:2] == (whole, components)
