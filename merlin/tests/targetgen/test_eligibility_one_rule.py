"""The suite and its whole-model gate must score the SAME corpus, and a piece of a pattern must not be
judged as the whole pattern.

Two defects, one incident. `_split_ineligible` (what the suite grades) and `_gate_counts` (what the
whole-model gate counts) ran different eligibility tests: the first withholds only on a hard structural
fact, the second read the raw family verdict. And the family taxonomy mapped `attention_qk` -- one
reduce-over-k product -- to the full `attention` composite, importing a `reduction` primitive the
computation does not contain.

MEASURED on gemmini (`merlincirct_defcal1`, 2026-08-29): `C7_attention_qk_i8` was graded and counted in
the score, but dropped from the gate denominator, lifting the fraction from 18/23 = 0.78 to 18/22 = 0.82
past a 0.8 gate. The whole-model capstone launched and ran a cycle-accurate simulation for 5 h 30 m
inside a 4 h round. The identical computation spelled `op: matmul` (`C5_attention_qk_matmul`) counted
normally and passed at the RTL tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import coverage_report as _cr
from merlin.targetgen import eligibility as _el
from merlin.targetgen import semantic_families as sf

PUBLIC = merlin_dir() / "experiments/capsule_bench/harness/full_public_capsules"
CONTRACT = merlin_dir() / "contract/capsules"


def _load(d: Path) -> dict:
    c = yaml.safe_load((d / "capsule.yaml").read_text()) or {}
    c["__dir__"] = str(d)
    c.setdefault("name", d.name)
    return c


def _corpus(root: Path) -> list[dict]:
    return [_load(f.parent) for f in sorted(root.rglob("capsule.yaml"))]


# --------------------------------------------------------------- one rule, two callers

@pytest.mark.parametrize("target,root", [
    ("gemmini", PUBLIC), ("atlas", CONTRACT / "atlas"), ("radiance", CONTRACT / "radiance"),
])
def test_the_suite_and_the_gate_agree_on_every_capsule(target, root):
    caps = _corpus(root)
    ops = [c for c in caps if c.get("kind") != "model"]
    if not ops:
        pytest.skip(f"no op capsules under {root}")
    kept = {c["name"] for c in CR._split_ineligible(list(ops), target)[0]}
    disagree = [c["name"] for c in ops
                if (c["name"] in kept)
                != CR._gate_counts({"capsule": c["name"], "status": "fail"}, caps, target)]
    assert disagree == [], (
        "the suite grades these and the gate does not count them (or vice versa) — the score and the "
        "gate are then measuring different corpora")


def test_only_an_absent_operand_dtype_is_a_hard_fact():
    """Family and rank must never withhold: families compose, and rank is what a compiler changes."""
    cmap = _el.capability_map_for_target("gemmini")            # int8 contraction/elementwise/movement
    from merlin.targetgen.eligibility import RegionDescriptor as RD

    absent, why = CR._dtype_has_no_datapath(
        RD(source="x", op="matmul", family="contraction", in_dtype="bf16"), cmap, _el)
    assert absent and "bf16" in why

    for region in (RD(source="x", op="sdpa", family="attention", in_dtype="i8"),   # undeclared family
                   RD(source="x", op="conv2d", family="contraction", in_dtype="i8", rank=4)):  # rank
        absent, _ = CR._dtype_has_no_datapath(region, cmap, _el)
        assert not absent, "only a dtype with no datapath is hard; family and rank are not"


# --------------------------------------------------------------- a piece is not the pattern

def test_the_qk_and_pv_pieces_are_contractions():
    assert sf.from_op("attention_qk") == "contraction"
    assert sf.from_op("attention_pv") == "contraction"
    # the FUSED pattern keeps the composite — it really does carry a reduction and a per-element map
    assert sf.from_op("sdpa") == "attention"
    assert sf.from_op("attention_full") == "attention"
    assert sf.primitives_of("attention") == ("contraction", "reduction", "elementwise_map")


def test_no_capsule_declares_a_family_coarser_than_its_op():
    """The invariant is kept in the DATA, not patched at read time.

    A reconciler in `_capsule_region` would have papered over exactly the inconsistency the corpus-wide
    gate in `test_semantic_coverage` exists to surface, so the declarations were corrected instead —
    five capsules across three targets, one of them in a HIDDEN set where the cost would have been an
    invisible held-out point.
    """
    seen = 0
    for root in (CONTRACT, PUBLIC):
        for f in sorted(root.rglob("capsule.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            fam = (c.get("semantic") or {}).get("semantic_family")
            op = (c.get("operation") or {}).get("op")
            derived = sf.from_op(op)
            if fam and derived:
                seen += 1
                assert fam == derived, f"{f}: declares {fam!r}, op {op!r} derives {derived!r}"
    assert seen > 50, f"the scan collapsed to {seen} capsules"


def test_the_same_computation_classifies_the_same_way_however_it_is_spelled():
    """`C5` declares `op: matmul` with K pre-transposed; `C7` declares `op: attention_qk` with K raw.
    Both are Q@K^T on int8. A taxonomy that separates them makes one unrunnable on paper."""
    c5, c7 = _load(PUBLIC / "C5_attention_qk_matmul"), _load(PUBLIC / "C7_attention_qk_i8")
    r5, r7 = _cr._capsule_region(c5), _cr._capsule_region(c7)
    assert r5.family == r7.family == "contraction"
    cmap = _el.capability_map_for_target("gemmini")
    assert _el.is_eligible(r5, cmap).eligible and _el.is_eligible(r7, cmap).eligible


def test_every_qk_capsule_of_every_target_is_eligible():
    """The same mislabel shipped in four capsules across three targets — fixed once, checked here."""
    seen = 0
    for target, root in (("gemmini", PUBLIC), ("atlas", CONTRACT / "atlas"),
                         ("radiance", CONTRACT / "radiance")):
        cmap = _el.capability_map_for_target(target)
        for c in _corpus(root):
            if (c.get("operation") or {}).get("op") not in ("attention_qk", "attention_pv"):
                continue
            seen += 1
            region = _cr._capsule_region(c)
            assert region.family == "contraction", c["name"]
            v = _el.is_eligible(region, cmap)
            assert v.eligible or "dtype" in (v.reason or ""), (
                f"{c['name']} on {target}: {v.reason} — a QK piece may only be refused on its DTYPE")
    assert seen >= 4, f"expected the known QK capsules, found {seen}"


# ------------------------------------------------------- the rank a contract admits must be the one it loads

def test_the_batched_contraction_rank_is_admitted_on_the_path_that_actually_loads():
    """A rank the target can execute must not be declared away, and the fix must land on the LIVE copy.

    Measured twice, and the second time is why this test reads the loader rather than a file. The
    contract declared ``ranks: [2, 4]``, so every batched matmul was ruled ineligible before it reached
    the device rewrite -- while the rewrite already supported the (B,M,N,K) signature, the device
    builder already produced the (M,N,K) kernel, and the shim already looped the batch calling that
    kernel per slice. The machinery existed and the contract forbade its use.

    It was then repaired in the PACKAGED copy of the contract while the copy the registry actually
    loads kept saying [2, 4], so the bug stayed live behind a fix that looked applied. Asserting
    through ``capability_map_for_target`` is the whole point: it answers for whichever file the loader
    resolves, which a grep over either path cannot.

    Rank 5 is checked alongside so this cannot pass by admitting everything.
    """
    from merlin.targetgen.eligibility import (RegionDescriptor, capability_map_for_target,
                                              is_eligible)

    caps = capability_map_for_target("gemmini")

    def verdict(rank):
        region = RegionDescriptor(source="t", op="matmul", in_dtype="int8", weight_dtype="int8",
                                  rank=rank, m=16, k=32, n=16)
        return is_eligible(region, caps)

    assert verdict(3).eligible, (
        "a batched contraction is ineligible, so every rank-3 region is refused before the device "
        f"rewrite sees it: {verdict(3).reason}")
    assert verdict(2).eligible and verdict(4).eligible
    assert not verdict(5).eligible, "admitting every rank would make this assertion vacuous"
