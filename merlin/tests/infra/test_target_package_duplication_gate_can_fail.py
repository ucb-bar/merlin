"""The duplication tripwire fires on a planted copy.

This gate reports zero on the live tree, which is the profile of a tripwire and also the profile of a
gate that does nothing. The difference is provable only by planting the thing it claims to catch, so
that is what this file does -- including a copy that renamed its variables and changed its constants,
because renaming is exactly what someone does while copying.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")

GATE_PATH = repo_root() / "build_tools" / "scripts" / "check_target_package_duplication.py"

ORIGINAL = """
def pick_tiles(rows, cols, depth, budget):
    best = None
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = i * depth + j * depth
            if cost <= budget and (best is None or cost > best[0]):
                best = (cost, i, j)
    return best
"""

#: The same function after a copier has been at it: every identifier renamed, the constants moved.
#: Text comparison misses this; shape comparison must not.
RENAMED_COPY = """
def choose_blocks(m, n, k, ceiling):
    winner = None
    for a in range(2, m + 2):
        for b in range(2, n + 2):
            price = a * k + b * k
            if price <= ceiling and (winner is None or price > winner[0]):
                winner = (price, a, b)
    return winner
"""


def _gate():
    spec = importlib.util.spec_from_file_location("_dup_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = tmp_path / "merlin" / "targets" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


def test_one_target_implementing_it_once_is_clean(tmp_path):
    """The control. Without it every test below passes for a gate that flags everything."""
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/sched.py": ORIGINAL})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert problems == [] and rc == 0


def test_the_same_function_in_two_target_packages_is_flagged(tmp_path):
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/sched.py": ORIGINAL, "beta/backend/sched.py": ORIGINAL})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert rc == 1, "a verbatim copy across target packages was not flagged"
    assert any("alpha/sched.py:pick_tiles" in p and "beta/sched.py:pick_tiles" in p for p in problems)
    assert any("move it to the core" in p for p in problems), "the finding does not say what to do"


def test_a_copy_that_renamed_everything_is_still_flagged(tmp_path):
    """The case text comparison misses, and the one that actually happens."""
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/sched.py": ORIGINAL, "beta/backend/planner.py": RENAMED_COPY})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert rc == 1, "a renamed copy was not recognised; the comparison is matching text, not shape"
    assert any("pick_tiles" in p and "choose_blocks" in p for p in problems)


def test_two_copies_in_ONE_package_are_not_flagged(tmp_path):
    """Duplication inside one target is that target's business.

    The rule is about generic logic crossing package boundaries; widening it to intra-package repetition
    would flood the gate with findings nobody asked for, which is how a gate stops being read.
    """
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/a.py": ORIGINAL, "alpha/backend/b.py": RENAMED_COPY})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert problems == [] and rc == 0


def test_a_trivially_short_function_is_not_a_copy(tmp_path):
    """Two short accessors are the same obvious line written twice, not a copied implementation."""
    gate = _gate()
    short = "\ndef rows(f):\n    return f['rows']\n"
    root = _tree(tmp_path, {"alpha/backend/a.py": short, "beta/backend/b.py": short})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert problems == [] and rc == 0


def test_a_ratcheted_shape_is_accepted_but_the_ledger_may_only_shrink(tmp_path):
    """Accepted debt is declared, not silent -- and the shrink-only meta-gate polices the ledger."""
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/sched.py": ORIGINAL, "beta/backend/sched.py": ORIGINAL})
    found = gate.shapes(root)
    shape = next(s for s, sites in found.items() if len({p for p, _, _ in sites}) > 1)
    problems, rc = gate.verdict(found, {shape})
    assert problems == [] and rc == 0
    assert gate.RATCHET.name.endswith("_ratchet.txt"), (
        "the ledger must be named *_ratchet.txt so check_ratchets_shrink.py holds it to shrinking"
    )


def test_an_unreadable_target_tree_is_a_refusal_not_a_pass(monkeypatch, tmp_path):
    gate = _gate()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert gate.main([]) == 2, "an absent target tree exited clean; it examined nothing"


def test_the_live_tree_has_a_recorded_extraction_not_an_unexamined_package():
    """An empty scan is valid only after an explicit OOT move with no in-tree Python left."""
    gate = _gate()
    found = gate.shapes()
    assert found == {}
    assert not list((repo_root() / "merlin" / "targets").rglob("*.py"))
    assert gate._recorded_extraction(repo_root())
    assert gate.main([]) == 0


def test_a_remaining_in_tree_module_cannot_claim_extraction(monkeypatch, tmp_path):
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/empty.py": "VALUE = 1\n"})
    manifest = root / "build_tools" / "upstreams" / "target_support.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"canonical_sources_removed": true, "companions": [{}]}', encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", root)
    assert gate.main([]) == 2
