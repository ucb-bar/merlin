"""The int8 contraction's OP NAME decides whether the transform-schedule levers exist.

``passes_quant_int`` built the i8xi8->i32 contraction as a ``linalg.generic``, which erases the
named form the whole transform layer keys on. ``impr_features`` matches ``linalg.matmul`` /
``linalg.batch_matmul`` in 39 places, and ``transform.structured.match`` on a name nothing carries
returns an EMPTY handle -- every op downstream of it becomes a vacuous no-op. Measured on
small_llama_int8: 15 matmuls before the pass, 0 after. So the entire register-blocking family did
nothing on int8 while still reporting as applied, and an 87-fork beam over those levers emitted only
21 distinct binaries and could not beat the two generic-level levers.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root


def test_the_canonical_predicate_matches_what_linalg_matmul_promises():
    """A named op ASSERTS an indexing convention. mlir-opt --linalg-generalize-named-ops on a
    mixed-type matmul yields maps (d0,d2),(d2,d1)->(d0,d1) with parallel/parallel/reduction; the
    predicate must accept exactly that and nothing else, or we would be relabelling a contraction
    into a convention it does not have -- a correctness bug, not a missed optimization."""
    from merlin.llvmlower.passes_quant_int import _is_canonical_matmul as ok
    assert ok(3, [[0, 2], [2, 1]], [0, 1], [False, False, True])
    # a transposed B is NOT linalg.matmul
    assert not ok(3, [[0, 2], [1, 2]], [0, 1], [False, False, True])
    # batched (4-D) is not, even with otherwise-canonical inner maps
    assert not ok(4, [[0, 1, 3], [0, 3, 2]], [0, 1, 2], [False, False, False, True])
    # reduction on the wrong axis is not
    assert not ok(3, [[0, 2], [2, 1]], [0, 1], [False, True, False])
    # swapped operands are not
    assert not ok(3, [[2, 1], [0, 2]], [0, 1], [False, False, True])


def test_the_feature_is_registered_and_default_off():
    from merlin.llvmlower.impr_features import NAMED_INT8_CONTRACTION_NAME as N, get, known
    assert N in known()
    f = get(N)
    assert f.action_class == "PASS"
    # it must NOT edit the schedule or the pipeline -- it changes only which op the quant pass emits
    assert f.edit_schedule is None and f.edit_pipeline is None


def test_apply_quant_only_hands_the_flag_to_the_contraction_pass():
    """Coupling every quant pass to a decision that is not theirs is how a flag becomes a mystery."""
    src = (repo_root() / "merlin" / "python" / "merlin" / "llvmlower" / "quant_passes.py").read_text()
    assert 'n == "contraction_int8"' in src
    assert "named_contraction: bool = False" in src


@pytest.mark.slow
def test_the_flag_restores_the_named_ops_and_the_default_does_not():
    """End-to-end on the real bundle: 15 matmuls erased by default, restored by the feature."""
    import collections
    bundle = repo_root() / "out/artifacts/recaptures/small_llama_int8_consistent/model.mlir"
    if not bundle.is_file():
        pytest.skip("small_llama int8 bundle not on disk")
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul, lower_quant_ext
    from merlin.llvmlower.quant_passes import apply_quant
    from merlin.runtime.dispatch_runtime import _propagate_quant_inner

    def census(named):
        m = parse_mlir_file(bundle)
        collapse_overrank_matmul(m)
        _propagate_quant_inner(m)
        apply_quant(m, named_contraction=named)
        lower_quant_ext(m)
        return collections.Counter(op.name for op in m.walk())

    off, on = census(False), census(True)
    assert off["linalg.matmul"] == 0, "the default datapath must stay byte-identical"
    assert on["linalg.matmul"] == 15
    # the named ops REPLACE generics one-for-one; nothing is duplicated or dropped
    assert off["linalg.generic"] - on["linalg.generic"] == 15
