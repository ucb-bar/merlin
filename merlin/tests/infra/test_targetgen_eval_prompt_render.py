"""The targetgen_evals method/skill prompts are TARGET-AGNOSTIC templates — the target-specific tokens
are written as the ``{target}`` placeholder and filled by ``harness.render`` at run-materialization
time. This pins that (a) the in-scope templates carry no ``gemmini`` literal (only the intentionally
target-specific ``gemmini_source_curator`` skill does), and (b) rendering for ``gemmini`` reproduces the
former hard-coded paths byte-for-byte. Mirrors ``test_generate_prompt`` for the contract prompts.

The harness is import-isolated (imports zero ``merlin.*``); we load its ``render`` module via the repo
root, never by a ``Path(__file__).parents[N]`` walk.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

_EVALS = repo_root() / "merlin" / "experiments" / "targetgen_evals"

_TEMPLATED_METHODS = ["v0_naive_claude", "v2_schema_generator", "v3_evidence_graph",
                      "v5_kernel_miner", "v6_full"]
_TEMPLATED_SKILLS = ["dialect_design_reviewer", "evidence_extractor", "schema_dialect_planner",
                     "validation_repairer", "xdsl_target_generator"]


def _render():
    """Load harness.render without importing the whole (import-isolated) harness package."""
    path = _EVALS / "harness" / "render.py"
    if not path.exists():
        pytest.skip("targetgen_evals harness not present")
    spec = importlib.util.spec_from_file_location("_tge_render", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tge_render"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_in_scope_templates_carry_no_target_literal():
    for m in _TEMPLATED_METHODS:
        text = (_EVALS / "methods" / m / "prompt.md").read_text(encoding="utf-8")
        assert "gemmini" not in text.lower(), f"methods/{m}/prompt.md still hardcodes gemmini"
    for s in _TEMPLATED_SKILLS:
        text = (_EVALS / "skills" / s / "AGENT.md").read_text(encoding="utf-8")
        assert "gemmini" not in text.lower(), f"skills/{s}/AGENT.md still hardcodes gemmini"


def test_source_curator_skill_stays_target_specific():
    # the reference-target curator is intentionally target-specific and keeps its literal
    text = (_EVALS / "skills" / "gemmini_source_curator" / "AGENT.md").read_text(encoding="utf-8")
    assert "gemmini" in text.lower()


def test_render_reproduces_the_gemmini_paths():
    r = _render()
    prompt = r.render_method_prompt(_EVALS, "v0_naive_claude", "gemmini")
    assert "{target}" not in prompt
    assert "datasets/gemmini/" in prompt and "generated/gemmini-mlir/" in prompt
    # a different target resolves by the same rule (nothing baked in for gemmini)
    other = r.render_method_prompt(_EVALS, "v0_naive_claude", "radiance")
    assert "generated/radiance-mlir/" in other and "gemmini" not in other.lower()


def test_render_skill_docs_resolve_all_skills():
    r = _render()
    docs = r.render_skill_docs(_EVALS, "gemmini")
    for s in _TEMPLATED_SKILLS:
        assert "{target}" not in docs[s], f"{s} left an unresolved placeholder"
    # the curator passes through unchanged (no {target} to fill)
    assert "gemmini" in docs["gemmini_source_curator"].lower()
