"""A whole-model verdict must say how much of the output it looked at.

The board harness dumps at most ``MERLIN_DUMP_CAP`` (4096) output elements over the console
(``merlin.mining.k1.main_linux_c``), and ``zephyr_model._gate`` scores that prefix against the
leading elements of the reference. For every model whose output fits under the cap that is the whole
tensor. For the ones that do not it is a slice, and nothing in cos/rel distinguishes the two.

MEASURED, 2026-09-04, ``tiny_llama_int8_consistent`` (TinyLlama-1.1B W8A8) on the x86 dispatch
runtime, graded against the bundle's INDEPENDENT torchao W8A8 reference:

    first 4096 of 256000 (1.6%)   w8a8_cos = 0.8925040364265442   rel = 0.6207035875961355
    all    256000                 w8a8_cos = 0.995999             rel = 0.2740

The first number is bit-for-bit the one a K1 board run reported and that was read as a codegen
defect. It is not: the 4096-element prefix is exactly token 0's logits, and token 0 is the position
W8A8 quantization destroys for *everyone* -- the independent torchao reference itself scores
cos -0.320 against fp32 there, while tokens 1-7 all sit above 0.988.

So the gate now reports its coverage on every verdict, and a caller that needs a whole-output verdict
can declare a floor and get a refusal instead of a prefix score.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime.backends.zephyr_model import _gate

#: the console cap the board harness compiles in (`k1.main_linux_c(dump_cap=4096)`)
DUMP_CAP = 4096
#: tiny_llama's real output: 8 positions x 32000 logits
SEQ, VOCAB = 8, 32000


def _tiny_llama_shaped(seed: int = 0):
    """A reference/run pair with tiny_llama's measured structure: position 0 is destroyed by
    quantization and every later position is faithful. Nothing here is model-specific beyond that
    shape -- it is the smallest thing that makes a prefix score and a whole-output score disagree."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(SEQ, VOCAB)).astype(np.float32)
    # position 0 carries a much smaller norm than the rest, as it does on the real model
    # (measured |logits| 267 at position 0 against 640-900 at positions 1-7)
    ref[0] *= 0.2
    run = ref.copy()
    run[0] = 0.2 * rng.normal(size=VOCAB).astype(np.float32)   # position 0: unrelated
    run[1:] += 1e-3 * rng.normal(size=(SEQ - 1, VOCAB))        # the rest: faithful
    return ref.ravel(), run.ravel()


def test_gate_reports_how_much_of_the_output_it_compared():
    ref, run = _tiny_llama_shaped()
    g = _gate(run[:DUMP_CAP], {"fp32": ref})
    assert g["n_compared"] == DUMP_CAP
    assert g["n_reference"] == SEQ * VOCAB
    assert g["compared_fraction"] == pytest.approx(DUMP_CAP / (SEQ * VOCAB))
    assert g["comparison_complete"] is False


def test_a_whole_output_comparison_is_marked_complete():
    ref, run = _tiny_llama_shaped()
    g = _gate(run, {"fp32": ref})
    assert g["n_compared"] == g["n_reference"] == SEQ * VOCAB
    assert g["compared_fraction"] == 1.0
    assert g["comparison_complete"] is True


def test_the_reported_score_depends_entirely_on_where_the_console_stopped():
    """The defect this file exists for: the same run scores two very different numbers, and without
    the coverage fields a reader cannot tell which one they were handed."""
    ref, run = _tiny_llama_shaped()
    part = _gate(run[:DUMP_CAP], {"fp32": ref})
    whole = _gate(run, {"fp32": ref})
    assert part["fp32_cos"] < 0.5 < 0.99 < whole["fp32_cos"]
    # ... and the ONLY thing in the record that separates them is the coverage.
    assert part["comparison_complete"] is False and whole["comparison_complete"] is True


def test_the_longest_reference_sets_the_denominator():
    """A short reference under one tier must not make a truncated run look complete."""
    ref, run = _tiny_llama_shaped()
    g = _gate(run[:DUMP_CAP], {"w8a8": ref[:DUMP_CAP], "fp32": ref})
    assert g["n_reference"] == SEQ * VOCAB
    assert g["comparison_complete"] is False


def test_a_declared_coverage_floor_refuses_a_prefix_only_verdict():
    """The knob has to be able to FAIL, or it is documentation. A prefix that passes every tier on
    its own numbers is still not a verdict about the model's output."""
    ref = np.random.default_rng(1).normal(size=SEQ * VOCAB).astype(np.float32)
    run = ref[:DUMP_CAP].copy()                              # a perfect prefix: every tier clears
    passes = _gate(run, {"fp32": ref})
    assert passes["ok"] is True and passes["tier_ok"] is not None
    assert passes["coverage_ok"] is True                     # default floor is off

    vetoed = _gate(run, {"fp32": ref}, min_coverage=1.0)
    assert vetoed["coverage_ok"] is False
    assert vetoed["ok"] is False
    assert vetoed["tier_ok"] is None
    assert vetoed["per_element_guarded"] is False
    # the scores are still reported -- the veto withholds the VERDICT, it does not hide the numbers
    assert vetoed["fp32_cos"] == pytest.approx(passes["fp32_cos"])


def test_a_floor_a_complete_comparison_meets_changes_nothing():
    ref = np.random.default_rng(2).normal(size=SEQ * VOCAB).astype(np.float32)
    full = _gate(ref, {"fp32": ref}, min_coverage=1.0)
    assert full["coverage_ok"] is True
    assert full["ok"] is True
