"""Which grading path an MX capsule takes must not depend on how the submission spells a dtype.

Measured live on `R5_mx_tile_mxfp8`: one arm wrote its operand dtype as `f8E4M3FN` and another as
`mxfp8`. `is_mx_cb` matched only the first, so the second silently took the fork-free path and failed a
capsule no submission can win there — and both arms reported an identical 26/35 for entirely different
reasons.

The string test was also wrong in both directions. `merlin.common.quant_formats` classes `mxfp8` as
kind `mx_block` / scale `block_e8m0` (genuinely block-scaled) and `f8E4M3FN` as `fp_ocp` / `per_tensor`
(not block-scaled at all) — so the prefixes matched a non-MX format and missed every real one.
"""
from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base as _bk

try:
    _muon = _bk.get_backend("muon")
    _mx = importlib.import_module(_muon.__name__ + ".muon_mx_codegen")
except Exception:  # noqa: BLE001 — SIMT backend absent in this env
    _muon = _mx = None

pytestmark = pytest.mark.skipif(_mx is None, reason="SIMT backend not present in this env")


def _cb(*dtypes: str) -> dict:
    return {"tensors": {f"t{i}": {"dtype": d} for i, d in enumerate(dtypes)}}


@pytest.mark.parametrize("spelling", ["mxfp8", "mxfp6", "mxfp4"])
def test_the_mx_block_spellings_resolve(spelling):
    """These are the spellings the capsules are NAMED after, and every one was missed."""
    assert _mx.is_mx_cb(_cb("bf16", spelling)) is True


@pytest.mark.parametrize("spelling", ["f8E4M3FN", "f6E3M2FN", "f4E2M1FN"])
def test_the_previously_matched_spellings_still_resolve(spelling):
    """Kept as a union: nothing that resolved before may stop resolving."""
    assert _mx.is_mx_cb(_cb("bf16", spelling)) is True


def test_a_plain_float_capsule_is_not_mx():
    assert _mx.is_mx_cb(_cb("f32")) is False
    assert _mx.is_mx_cb(_cb("bf16", "f32")) is False


def test_the_registry_is_what_decides():
    """Derivation, not a prefix list: the format registry must agree with the predicate."""
    from merlin.common import quant_formats as qf

    assert qf.get("mxfp8").kind == "mx_block", "registry no longer classes mxfp8 as block-scaled"
    assert qf.get("f8E4M3FN").kind != "mx_block", (
        "f8E4M3FN is per-tensor OCP fp8; if the registry now calls it mx_block the predicate needs review")


def test_the_oracle_gates_on_the_golden_not_the_cb_dtype():
    """Structural: the MX route must key off `mx_operands` (attached from the capsule's own golden),
    never off the agent-written dtype, or the submission picks its own grading path again."""
    import ast
    import inspect

    src = inspect.getsource(importlib.import_module(_muon.__name__ + ".muon_oracles"))
    tree = ast.parse(src)
    gate = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_mxprog" for t in node.targets):
            gate = node
            break
    assert gate is not None, "the MX route disappeared from the oracle adapter"
    # the guarding `if` must mention mx_operands and must NOT re-introduce the dtype test
    guard_src = src[src.index("_mxprog = None"):src.index("_mxprog = None") + 600]
    assert "mx_operands" in guard_src, "the MX route must gate on the golden-derived operand bundle"
    assert "is_mx_cb" not in guard_src.split("if cb.get")[0] + guard_src.split("_mxprog = _mx")[0][:0], (
        "the agent's dtype spelling must not gate the grading path")
