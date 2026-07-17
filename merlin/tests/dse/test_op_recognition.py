"""C6 held-out extraction eval: does the structural op-recognizer KNOW WHAT a region is?

The recognizer (``dse_guidance.attribution.recognize_regions``) folds per-op prov tags into a
region-level identity (attention / linear / mlp / conv / norm / softmax) by grouping at the nn.Module
boundary and reading op-topology. This harness checks it against the 9 committed real-architecture
captures with THREE invariants — coverage, attention recall, precision — plus an anti-overfit guard.

Honesty of the split: the recognizer's rules were written by inspecting the raw MLIR of the DEV set
{openvla, xr0} only. The other seven are HELD OUT — the invariants must hold on them with no per-model
tuning. Ground truth is not hand-labeled: it is derived from the prov tags the capture already carries
(a softmax/sdpa under an attention-module fqn ⇒ that region must be recognized ATTENTION).
"""
from __future__ import annotations

import inspect

import pytest

from merlin.common.paths import merlin_dir
from merlin.dse_guidance import attribution as A

_RECAP = merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures"
_ALL = ("bitvla", "groot_n1d7", "molmoact", "openvla", "rdt", "rdt2",
        "small_llama", "tiny_llama", "xr0")
_DEV = {"openvla", "xr0"}                       # the only captures whose raw MLIR was inspected
_HELD_OUT = tuple(w for w in _ALL if w not in _DEV)

_present = [w for w in _ALL if (_RECAP / w / "model.mlir").is_file()]
pytestmark = pytest.mark.skipif(len(_present) < 2, reason="op-recognition corpus not present")


def _capture(w: str) -> str:
    return str(_RECAP / w)


def _prov_pairs(mlir_text: str):
    """(prov.op, prov.fqn) for every tagged line — structured scan, no regex."""
    for line in mlir_text.splitlines():
        op = _attr(line, 'prov.op = "')
        fqn = _attr(line, 'prov.fqn = "')
        if op is not None or fqn is not None:
            yield op, fqn


def _attr(line: str, marker: str) -> str | None:
    i = line.find(marker)
    if i == -1:
        return None
    j = line.find('"', i + len(marker))
    return line[i + len(marker):j] if j != -1 else None


def _expected_attention_groups(w: str) -> set[str]:
    """Self-checking ground truth: the attention regions a capture MUST have — the fqn-groups of any
    softmax/sdpa op sitting under an attention-module fqn (derived purely from prov tags)."""
    text = (_RECAP / w / "model.mlir").read_text(encoding="utf-8")
    expected: set[str] = set()
    for op, fqn in _prov_pairs(text):
        if op in ("softmax", "sdpa") and fqn:
            key, token = A._region_group_key(fqn)
            if token == "attn" and key:
                expected.add(key)
    return expected


@pytest.mark.parametrize("w", _present)
def test_coverage_every_compute_op_lands_in_a_recognized_region(w):
    """No compute op (contraction) is orphaned into an unrecognized ('other') region."""
    regs = A.recognize_regions(_capture(w))
    total = sum(r.contraction_count for r in regs)
    recognized = sum(r.contraction_count for r in regs if r.region_label != A.REGION_OTHER)
    assert total > 0, w
    assert recognized == total, (w, [(r.region_label, r.contraction_count) for r in regs
                                     if r.region_label == A.REGION_OTHER and r.contraction_count])


@pytest.mark.parametrize("w", _present)
def test_attention_recall_is_total(w):
    """Every attention-fqn-with-softmax group (ground truth from prov tags) is recognized ATTENTION."""
    regs = A.recognize_regions(_capture(w))
    got = {r.fqn_group for r in regs if r.region_label == A.REGION_ATTENTION}
    expected = _expected_attention_groups(w)
    assert expected <= got, (w, "missed", expected - got)


@pytest.mark.parametrize("w", _present)
def test_attention_precision_requires_softmax_or_sdpa(w):
    """No region is labeled ATTENTION without an actual softmax/sdpa member (never from fqn alone)."""
    regs = A.recognize_regions(_capture(w))
    for r in regs:
        if r.region_label == A.REGION_ATTENTION:
            assert r.has_softmax or (A.OPC_ATTENTION in r.op_labels), (w, r.fqn_group, r.op_labels)


def test_held_out_models_are_recognized_without_tuning():
    """The 7 HELD-OUT captures each recognize ≥1 real region and satisfy the ground-truth attention
    recall — the recognizer generalizes beyond the two DEV models its rules were written against."""
    held = [w for w in _HELD_OUT if w in _present]
    assert held, "expected held-out captures present"
    for w in held:
        regs = A.recognize_regions(_capture(w))
        assert any(r.region_label != A.REGION_OTHER for r in regs), w
        got = {r.fqn_group for r in regs if r.region_label == A.REGION_ATTENTION}
        assert _expected_attention_groups(w) <= got, (w, "attention recall")


def test_recognizer_source_is_not_overfit():
    """Anti-overfit guard: the recognition logic names NO specific workload and hard-codes NO shape
    literal — it keys only on architectural tokens (attn/mlp/norm/conv) + op-topology."""
    src = "".join(inspect.getsource(fn) for fn in
                  (A.recognize_regions, A._region_group_key, A._region_token_type))
    low = src.lower()
    for name in _ALL:
        assert name not in low, f"recognizer names a specific workload: {name}"
    # No TENSOR-SHAPE literal: the only numbers allowed are tiny structural constants (contraction
    # thresholds / list indexing, i.e. 0/1/2). Any literal ≥ 4 would smell like a hard-coded dim/shape.
    nums, run = [], ""
    for ch in src + " ":
        if ch.isdigit():
            run += ch
        elif run:
            nums.append(int(run))
            run = ""
    assert all(n <= 2 for n in nums), f"recognizer hard-codes a shape-like literal: {sorted(set(nums))}"
