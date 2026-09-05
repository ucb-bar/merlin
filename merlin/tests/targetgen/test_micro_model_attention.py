"""The composed micro model must be able to write an ATTENTION layer, and write the right op.

``statement_for`` raises rather than skipping, so a family with no entry in ``_STATEMENT`` does not
degrade the model -- it stops the capsule being generated at all. Measured 2026-09-05: radiance was the
only roster target with no ``SY_micro_model``, and the whole cause was this one missing entry
(``no emittable statement for family 'attention'``).

Two near misses make this worth pinning rather than trusting:

* ``attention_qk`` is classified as a CONTRACTION -- Q@K^T is exactly that -- so adding it would leave
  attention unwritable AND add a second candidate to ``contraction``;
* ``attention_mx`` IS in the family and IS materializable, and it is the one ``op_for_family`` picks by
  cost out of the full op set -- but its golden exists only in the block-scaled engine, which is how
  radiance's fp16/bf16/f32 cells previously died with "no SIMT golden for op 'attention_mx'".

``_STATEMENT`` membership is the writability filter (``available_ops() & set(_STATEMENT)``), so these
tests check the SELECTION, which is the thing that silently goes wrong.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import corpus_synth as CS
from merlin.targetgen import micro_model as MM


def _statement_pool() -> set[str]:
    return CS.available_ops() & set(MM._STATEMENT)


class TestTheAttentionLayerIsWritable:
    def test_the_family_resolves_to_an_op_with_a_statement(self):
        op, (_init, fwd) = MM.statement_for("attention")
        assert op in MM._STATEMENT
        assert fwd, "a family that resolves must carry a forward line"

    def test_it_is_not_the_block_scaled_only_op(self):
        """``attention_mx`` is what cost-ranking picks unfiltered; its golden is block-scaled only."""
        op, _ = MM.statement_for("attention")
        assert op != "attention_mx"

    def test_the_trap_op_is_still_the_unfiltered_choice(self):
        """VACUITY GUARD. If ``attention_mx`` ever stops being the cheapest attention op, the test
        above starts passing for a reason unrelated to what it is protecting."""
        assert "attention_mx" in CS.available_ops(), "the trap op vanished; re-derive this guard"
        assert CS.op_for_family("attention", admitted_ops=CS.available_ops()) == "attention_mx"

    def test_a_contraction_spelling_is_not_mistaken_for_the_family(self):
        """``attention_qk`` reduces to a contraction and must not be the attention representative."""
        assert CS._op_family_map().get("attention_qk") == "contraction"
        op, _ = MM.statement_for("attention")
        assert op != "attention_qk"


class TestTheOtherFamiliesDidNotMove:
    """Adding a statement changes a SHARED pool, so every other family's choice is pinned.

    Five targets already ship a ``SY_micro_model`` generated from this table; a changed representative
    would silently rewrite their models and invalidate goldens that are graded, not regenerated.
    """

    @pytest.mark.parametrize("family,expected", [
        ("contraction", "matmul"),
        ("elementwise_map", "gelu"),
        ("reduction", "reduce_sum"),
        ("movement", "movement"),
        ("normalization", "rmsnorm"),
    ])
    def test_representative_is_unchanged(self, family, expected):
        assert CS.op_for_family(family, admitted_ops=_statement_pool()) == expected


class TestTheEmittedLayerComposes:
    def test_it_preserves_the_square_extent_every_other_statement_preserves(self):
        """The composed input is ``(E, E)`` and each statement maps ``(E,E) -> (E,E)``; a layer that
        changed the extent would break every layer after it in the composition."""
        torch = pytest.importorskip("torch")
        _op, (init, fwd) = MM.statement_for("attention")
        assert init, "this layer carries parameters, so it must emit an init line"
        E = 32
        scope: dict = {"torch": torch, "nn": torch.nn, "E": E}
        exec(f"import torch.nn as nn\nclass _M(nn.Module):\n"
             f"    def __init__(self):\n        super().__init__()\n        {init.format(i=0)}\n"
             f"    def forward(self, x):\n        {fwd.format(i=0)}\n        return x\n", scope)
        torch.manual_seed(0)
        model = scope["_M"]().eval()
        x = torch.randn(E, E)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape
        assert bool(torch.isfinite(y).all()), "a non-finite layer makes every downstream golden junk"
