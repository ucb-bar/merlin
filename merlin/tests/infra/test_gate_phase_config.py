"""A gate's phase is declared in one tracked file, or it raises.

WHY THIS EXISTS. `gate_phase` already made the phase a parameter rather than a comment, which was the
right half of the idea. The other half was missing: both live callers passed `PHASE_REPORT` as a
LITERAL, so two gates -- the cost plane and the offload verdict -- sat blocking nothing for months
with no record anywhere that this was a decision. "We will turn it on later" written in source is
not a state anything can read, and it is not a state anyone reviews.

The tests below are about the two ways that can go wrong again: a gate nobody declared resolving to
a quiet `report`, and a declaration that is read but cannot actually change what blocks.
"""

from __future__ import annotations

import ast

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf import gate_phase as G

DECLARATION = merlin_dir() / "contract" / "gate_phases.yaml"


@pytest.fixture(autouse=True)
def _fresh_cache():
    """The declaration is cached for the process; every test here reads it fresh."""
    G._declared.cache_clear()
    yield
    G._declared.cache_clear()


def _with_declaration(monkeypatch, tmp_path, body):
    """Point the module at a throwaway declaration."""
    contract = tmp_path / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    (contract / "gate_phases.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    monkeypatch.setattr("merlin.common.paths.merlin_dir", lambda: tmp_path)
    G._declared.cache_clear()


class TestTheShippedDeclaration:
    def test_it_loads_and_every_phase_is_one_of_the_two_real_ones(self):
        assert G.declared_gates(), "the repo's own declaration must name at least one gate"
        for gate in G.declared_gates():
            assert G.configured_phase(gate) in G.PHASES

    def test_every_gate_a_caller_asks_for_is_declared(self):
        """Derived from the call sites, not listed here -- a list would drift the same way.

        A rename on one side and not the other is exactly how a gate stops being graded while
        still looking wired, so the callers are the source of truth for what must be declared.
        """
        asked: set[str] = set()
        for path in sorted(repo_root().joinpath("merlin", "python", "merlin").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # not this test's subject
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name == "configured_phase" and isinstance(node.args[0], ast.Constant):
                    value = node.args[0].value
                    if isinstance(value, str):
                        asked.add(value)
        missing = sorted(asked - set(G.declared_gates()))
        assert not missing, f"asked for but never declared: {missing}"


class TestThereIsNoDefaultPhase:
    def test_an_undeclared_gate_raises_rather_than_reporting(self, monkeypatch, tmp_path):
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {"known": "report"}})
        with pytest.raises(G.GatePhaseError) as excinfo:
            G.configured_phase("forgotten")
        assert "known" in str(excinfo.value), "the refusal should say what IS declared"

    def test_an_empty_declaration_is_not_everything_reports(self, monkeypatch, tmp_path):
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {}})
        with pytest.raises(G.GatePhaseError):
            G.configured_phase("anything")

    def test_a_missing_declaration_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("merlin.common.paths.merlin_dir", lambda: tmp_path)
        G._declared.cache_clear()
        with pytest.raises(G.GatePhaseError):
            G.configured_phase("anything")

    def test_incomplete_is_refused_as_a_phase_by_name(self, monkeypatch, tmp_path):
        """`incomplete` is a STATUS, orthogonal to phase. Reading it as `report` would make the
        file say something it does not, and would quietly invent a three-valued phase."""
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {"g": G.STATUS_INCOMPLETE}})
        with pytest.raises(G.GatePhaseError) as excinfo:
            G.configured_phase("g")
        assert G.STATUS_INCOMPLETE in str(excinfo.value)


class TestTheDeclarationActuallyDecidesWhatBlocks:
    """The mutation. Without this, the declaration could be read correctly and change nothing."""

    FAILING = ("over", "below_floor")

    def test_at_report_a_decided_failure_does_not_block(self, monkeypatch, tmp_path):
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {"cost_plane": "report"}})
        assert G.blocks(G.configured_phase("cost_plane"), "over", failing=self.FAILING) is False

    def test_flipping_the_declaration_to_fail_makes_that_same_failure_block(self, monkeypatch, tmp_path):
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {"cost_plane": "fail"}})
        assert G.blocks(G.configured_phase("cost_plane"), "over", failing=self.FAILING) is True

    def test_incomplete_still_does_not_block_even_at_fail(self, monkeypatch, tmp_path):
        """The property the whole module exists to protect: "we could not measure this" is never
        turned into a failure a submission cannot fix."""
        _with_declaration(monkeypatch, tmp_path, {"version": 1, "gates": {"cost_plane": "fail"}})
        phase = G.configured_phase("cost_plane")
        assert G.blocks(phase, G.STATUS_INCOMPLETE, failing=self.FAILING) is False


def test_explicit_declaration_never_discovers_checkout_and_is_not_cached(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("explicit policy must not discover a checkout")

    monkeypatch.setattr("merlin.common.paths.merlin_dir", forbidden)
    path = tmp_path / "selected-policy.yaml"
    path.write_text("gates: {cost_plane: report}\n")
    assert G.configured_phase("cost_plane", declaration=path) == "report"
    assert G.declared_gates(declaration=path) == ("cost_plane",)
    path.write_text("gates: {cost_plane: fail}\n")
    assert G.configured_phase("cost_plane", declaration=path) == "fail"


@pytest.mark.parametrize("body", [None, "gates: {}", "gates: {cost_plane: incomplete}"])
def test_invalid_explicit_declaration_never_falls_back_to_cached_default(tmp_path, monkeypatch, body):
    _with_declaration(monkeypatch, tmp_path, {"gates": {"cost_plane": "report"}})
    assert G.configured_phase("cost_plane") == "report"
    selected = tmp_path / "explicit.yaml"
    if body is not None:
        selected.write_text(body)
    with pytest.raises(G.GatePhaseError):
        G.configured_phase("cost_plane", declaration=selected)
    assert G.configured_phase("cost_plane") == "report"
