"""No tracked artifact may name a held-out capsule — the generalized form of two same-day leaks.

A held-out capsule's NAME is an answer key: knowing which shapes are graded privately is most of the
advantage the holdout exists to deny. Masking the golden VALUES from the running agent, which the
sandbox does, is a different and weaker guarantee.

Two leaks were found on one day, in two subsystems, by two people, with the same root cause:
`CostFit.to_dict()` serialized the run file each cost sample came from (10 names into the conformance
spec, 60 more through per-class fits), and a performance rate table harvested from the same runs
published 8 names and 497 absolute paths. **Anything harvested from certification runs republishes
their provenance, because those runs include the grading passes over the hidden set.** That is a class,
not two incidents, and three existing gates missed both: `check_no_answer_keys` asks whether golden
surfaces are tracked, `check_holdout_disjointness` asks whether the sets overlap, `verify_no_cheat` asks
whether the GRANTED tree leaks into a running experiment. None asks whether any tracked file, anywhere,
contains a holdout name.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root

_spec = importlib.util.spec_from_file_location(
    "check_no_holdout_names", repo_root() / "build_tools" / "scripts" / "check_no_holdout_names.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def test_the_holdout_set_is_derived_from_the_corpus_not_listed():
    """A gate carrying its own copy of the hidden set would BE the leak it prevents, and would go stale
    the first time somebody adds a capsule."""
    import inspect

    src = inspect.getsource(G)
    names = G.holdout_names()
    if not names:
        pytest.skip("no hidden corpus in this checkout (a worktree carries none)")
    # No holdout name may appear as a literal in the gate's own logic. The docstring is exempt by the
    # gate's own allowlist, so check the code below it.
    body = src.split('"""', 2)[-1]
    leaked = sorted(n for n in names if n in body)
    assert not leaked, f"the gate hardcodes holdout names: {leaked}"


def test_the_tree_is_clean_or_every_offender_is_declared_debt():
    rep = G.scan(limit=200)
    if rep["status"] == "undeterminable":
        pytest.skip(rep["detail"])
    assert rep["leaks"] == [], (
        f"tracked file(s) name a held-out capsule and are not in the ratchet: "
        f"{[r['file'] for r in rep['leaks']]}. Fix by WITHHOLDING (keep the auditable counts and the "
        f"refusal reasons, drop the identity-bearing fields), or declare it in "
        f"build_tools/scripts/holdout_name_ratchet.txt with a reason.")


def test_the_ratchet_has_no_stale_entries():
    """An entry that no longer leaks must be REMOVED. A ratchet that keeps fixed debt can only grow,
    which is the opposite of what a ratchet is for."""
    rep = G.scan(limit=200)
    if rep["status"] == "undeterminable":
        pytest.skip(rep["detail"])
    stale = rep["ratchet_declared"] - rep["n_ratcheted"]
    assert stale <= 0, (
        f"{stale} ratchet entr(y/ies) no longer name a holdout and must be deleted from "
        f"build_tools/scripts/holdout_name_ratchet.txt — this list may only shrink")


def test_every_ratchet_entry_states_a_reason():
    """A reasonless entry is indistinguishable from one somebody added to silence the gate."""
    ratchet = G._ratchet()
    if not ratchet:
        pytest.skip("no ratchet file")
    reasonless = sorted(p for p, why in ratchet.items() if not why.strip())
    assert not reasonless, f"ratchet entries with no reason: {reasonless}"


def test_every_allowlist_entry_states_a_reason():
    """`_ALLOWED` means "naming a holdout HERE is correct" — a strictly stronger claim than the ratchet's
    "we have not fixed this yet" — so it needs a reason even more."""
    reasonless = sorted(p for p, why in G._ALLOWED.items() if not str(why).strip())
    assert not reasonless, f"allowlist entries with no reason: {reasonless}"


def test_a_planted_name_is_detected(tmp_path, monkeypatch):
    """⚠️ The gate must be able to FAIL. Mutation-checked against the real tree during development:
    planting a holdout name in a tracked doc turned it red, removing a still-leaking ratchet entry
    turned it red, and a ratchet entry that no longer leaks was reported stale."""
    names = G.holdout_names()
    if not names:
        pytest.skip("no hidden corpus in this checkout")
    victim = sorted(names)[0]
    planted = tmp_path / "leaky.md"
    planted.write_text(f"the hidden capsule {victim} is graded privately\n", encoding="utf-8")
    monkeypatch.setattr(G, "_tracked_files", lambda: [planted])
    monkeypatch.setattr(G, "REPO", tmp_path)
    rep = G.scan()
    assert rep["n_leaks"] == 1, "a planted holdout name must be detected"
    assert victim in rep["leaks"][0]["names"]


def test_a_binary_file_is_skipped_by_content_not_extension(tmp_path, monkeypatch):
    """An extension list is a way to miss the next format."""
    blob = tmp_path / "weights.bin"
    blob.write_bytes(b"\x00\x01\x02" * 100)
    monkeypatch.setattr(G, "_tracked_files", lambda: [blob])
    monkeypatch.setattr(G, "REPO", tmp_path)
    rep = G.scan()
    assert rep["n_binary_skipped"] == 1
    assert rep["n_leaks"] == 0


def test_an_empty_holdout_set_is_undeterminable_never_a_pass(monkeypatch):
    """A worktree has no hidden corpus. "The question could not be asked" must not print [ ok]."""
    monkeypatch.setattr(G, "holdout_names", lambda: set())
    rep = G.scan()
    assert rep["status"] == "undeterminable"
    assert "could not be evaluated" in rep["detail"]
