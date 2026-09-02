"""Which host board a target's lane compiles for, and what happens when nothing says.

The omission this closes was silent in the worst way. ``system_for`` returns ``System(host=None, ...)``
unless a board is passed; ``place.host_units(None)`` then synthesizes a SCALAR-ONLY host; so work that
belongs on the host VECTOR lane was placed on a scalar one with no error at all -- because "no board
declared" and "a board that has no vector unit" were the same value. No descriptor declared a board and
no production caller passed one, so every real System carried host=None. The only caller with a host was
a test, which hardcoded the board name.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.system import place as P
from merlin.system.derive import host_board_for_experiment, system_for_experiment


def _targets() -> list[str]:
    root = merlin_dir() / "experiments/capsule_bench/targets"
    return sorted(p.name for p in root.iterdir() if (p / "target_experiment.yaml").is_file())


@pytest.mark.parametrize("target", _targets())
def test_a_declared_board_resolves_and_an_undeclared_one_says_so(target):
    """Every target answers, and the answer is either a real host or a sentence explaining its absence.
    What must never happen is a fabricated host: a made-up one is worse than none, because placement
    would then be measured against hardware nobody has."""
    board, why = host_board_for_experiment(target)
    system, why2 = system_for_experiment(target)
    assert why and why2, "the answer must always carry its reason"
    if board is None:
        assert system.host is None, "no board declared, so no host may be invented"
        assert "declares no" in why or "could not read" in why or "no experiment descriptor" in why
    else:
        assert system.host is not None, f"{target} declares board {board!r} but no host resolved: {why2}"


def test_the_declared_boards_give_their_targets_a_vector_host():
    """The point of declaring it. A target whose host_lane pins an RVV package needs a host that HAS a
    vector unit, or the lane it is pinned to cannot exist."""
    declared = [t for t in _targets() if host_board_for_experiment(t)[0]]
    assert declared, "at least one target must declare a board, or this closes nothing"
    for target in declared:
        system, why = system_for_experiment(target)
        assert system.host.vector_capable() is True, (
            f"{target} declares an RVV host lane, so its board must be vector-capable ({why})")
        kinds = {u.kind for u in P.host_units(system.host)}
        assert "vector" in kinds, f"{target}: host_units synthesized no vector lane ({sorted(kinds)})"


def test_an_undeclared_target_does_not_silently_become_scalar_only():
    """The failure mode itself: without a board, host_units yields ONLY a scalar unit. That is the right
    output for an unknown host -- what was wrong was that nothing said the host was unknown."""
    undeclared = [t for t in _targets() if host_board_for_experiment(t)[0] is None]
    if not undeclared:
        pytest.skip("every target declares a board")
    for target in undeclared:
        system, why = system_for_experiment(target)
        assert {u.kind for u in P.host_units(system.host)} == {"scalar"}
        assert "unknown" in why or "declares no" in why, (
            f"{target} has no host and no explanation: {why!r}")


def test_a_bad_board_name_is_reported_not_raised(tmp_path, monkeypatch):
    """A descriptor naming a board that does not exist must degrade to 'no host, here is why' rather
    than crash every caller that builds a System."""
    import merlin.system.derive as D

    monkeypatch.setattr(D, "host_board_for_experiment",
                        lambda t: ("definitely_not_a_board", "declared by a test"))
    system, why = D.system_for_experiment("gemmini")
    assert system.host is None
    assert "did not resolve" in why


def test_the_descriptor_override_decides_the_board(tmp_path, monkeypatch):
    """``MERLIN_TARGET_EXPERIMENT`` must select the board, not be silently ignored.

    The first version of this resolver built the convention path by hand
    (``merlin_dir() / "experiments/capsule_bench/targets" / target / ...``), which reads the IN-TREE
    descriptor no matter what the override says. That is the failure mode the whole indirection in
    ``targetgen.corpora.descriptor_path`` exists to prevent: a caller pointed at an out-of-tree
    descriptor got some of its fields from there and the board from somewhere else, with nothing
    printed. Asserting a board DIFFERENT from the in-tree one is what makes this test discriminating --
    a resolver that ignores the override returns the tracked board and still looks like a pass.
    """
    from merlin.runtime.boards import BOARDS
    from merlin.system.derive import host_board_for_experiment

    tracked, _ = host_board_for_experiment("gemmini")
    other = next(b for b in sorted(BOARDS) if b != tracked)
    desc = tmp_path / "target_experiment.yaml"
    desc.write_text(f"target: gemmini\nhost:\n  board: {other}\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))

    got, why = host_board_for_experiment("gemmini")
    assert got == other, (
        f"the override named {other} but the resolver returned {got!r}; it is reading the in-tree "
        f"descriptor ({tracked!r}) and ignoring MERLIN_TARGET_EXPERIMENT")
    assert "declared by" in why
