"""The int8 rewrite must not destroy the identity of the ops it rewrites.

A capture stamps every region with ``prov.*`` — most importantly ``prov.fqn``, the module path that
joins a Merlin region to the same layer in a board profile or another frontend. The int8 datapath
replaces one captured contraction with an integer contraction PLUS a requant epilogue, and it used to
carry the provenance to the epilogue only. The contraction was then anonymous, and since a profile's
join key falls back to the MLIR op name, every contraction in an int8 model collapsed into one
``linalg.generic`` bucket: no per-layer cost attribution was possible for exactly the ops that dominate
a model's arithmetic.

These tests pin the repair and the property that makes it useful — that the two pieces sharing an fqn
stay distinguishable, so restoring the key does not just trade one imprecision for another.
"""
from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.kernels.shapes import observe_contractions

_F32_MATMUL = """
builtin.module {
  func.func @forward(%a: tensor<8x64xf32>, %b: tensor<64x32xf32>) -> tensor<8x32xf32> {
    %e = tensor.empty() : tensor<8x32xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<8x32xf32>) -> tensor<8x32xf32>
    %0 = linalg.matmul {prov.fqn = "enc.l0", prov.op = "linear", prov.region_id = "mm_0"}
         ins(%a, %b : tensor<8x64xf32>, tensor<64x32xf32>)
         outs(%f : tensor<8x32xf32>) -> tensor<8x32xf32>
    func.return %0 : tensor<8x32xf32>
  }
}
"""


@pytest.fixture
def rewritten():
    """The module after the real contraction pass — not a stand-in for it."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    module = parse_mlir_text(_F32_MATMUL)
    n = lower_contraction_int8(module)
    assert n == 1, "the fixture must actually be rewritten for these assertions to mean anything"
    return module


def _by_role(module):
    return {mq.provenance(op).get("prov.role"): op
            for op in module.walk() if mq.provenance(op).get("prov.role")}


class TestProvenanceSurvives:
    def test_the_integer_contraction_keeps_the_source_fqn(self, rewritten):
        contractions = observe_contractions(rewritten)
        assert len(contractions) == 1
        prov = mq.provenance(contractions[0][0])
        assert prov.get("prov.fqn") == "enc.l0", "the profile join key must survive the rewrite"

    def test_every_prov_key_is_carried_not_just_the_join_key(self, rewritten):
        prov = mq.provenance(observe_contractions(rewritten)[0][0])
        assert prov.get("prov.op") == "linear"
        assert prov.get("prov.region_id") == "mm_0"

    def test_the_contraction_reads_as_int8_with_an_i32_accumulator(self, rewritten):
        # The dtype triple is what a legality question is asked in; if the rewrite ran, it is int8.
        assert observe_contractions(rewritten)[0][1].dtypes == ("i8", "i8", "i32")


class TestRolesDistinguishThePieces:
    def test_both_pieces_are_tagged_and_the_tags_differ(self, rewritten):
        roles = _by_role(rewritten)
        assert set(roles) == {"contraction", "requant"}

    def test_the_two_pieces_share_the_source_fqn(self, rewritten):
        roles = _by_role(rewritten)
        fqns = {r: mq.provenance(op).get("prov.fqn") for r, op in roles.items()}
        assert fqns == {"contraction": "enc.l0", "requant": "enc.l0"}, (
            "they are one captured op split in two, so one fqn is correct -- the role is what "
            "separates their costs")

    def test_the_role_tagged_contraction_is_the_one_the_shape_observer_finds(self, rewritten):
        observed = observe_contractions(rewritten)[0][0]
        assert mq.provenance(observed).get("prov.role") == "contraction"

    def test_the_profiler_lifts_the_role_into_its_op_table(self):
        # The census joins on fqn+role, which only works if the profile table carries the role at all.
        from merlin.llvmlower.op_profile import _PROV_KEYS
        assert "prov.role" in _PROV_KEYS


class TestCensusJoinIsRestored:
    def test_a_census_of_the_rewritten_module_attributes_each_contraction_to_its_layer(self, rewritten):
        from merlin.kernels import census as cs
        got = cs.census(rewritten, model="m", stage="prepared")
        assert [r.key for r in got.rows] == ["enc.l0"]
        assert [r.role for r in got.rows] == ["contraction"]
        assert [r.family for r in got.rows] == ["linear"]
