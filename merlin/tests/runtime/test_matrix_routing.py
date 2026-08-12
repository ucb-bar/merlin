"""Routing contractions to a matrix unit inside the whole-model build.

The build is where the two halves of the path have to agree: the IR rewrite decides which contractions
move, and the object build defines the symbols those calls need. A disagreement between them is a link
error a long way from its cause, so these tests assert the agreement rather than either half.

They also pin the fail-closed direction. Enabling the feature with nothing to route to would produce a
model that grades correctly and reports a capability it never used, which is the failure mode that is
hardest to notice and easiest to cite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.llvmlower import passes_opu as PO
from merlin.llvmlower.impr_features import OPU_MATMUL_NAME
from merlin.runtime.backends.zephyr_model import MatrixRouting, prepare_for_lowering

#: A rank-2 int8 contraction in the form the int8 rewrite leaves behind. Two extents: one that fills a
#: 32-lane tile in both directions and one that does not, so a tile-filling selector has something to
#: decline.
_MODEL = """
builtin.module {
  func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x64xi8>,
                     %c: tensor<8x32xi8>, %d: tensor<32x8xi8>) -> tensor<64x64xi32> {
    %z = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<64x64xi32>
    %f0 = linalg.fill ins(%z : i32) outs(%e0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %big = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d2, d1)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x32xi8>, tensor<32x64xi8>)
        outs(%f0 : tensor<64x64xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<64x64xi32>
    %e1 = tensor.empty() : tensor<8x8xi32>
    %f1 = linalg.fill ins(%z : i32) outs(%e1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %small = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d2, d1)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%c, %d : tensor<8x32xi8>, tensor<32x8xi8>)
        outs(%f1 : tensor<8x8xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<8x8xi32>
    func.return %big : tensor<64x64xi32>
  }
}
"""


def _model(tmp_path: Path) -> Path:
    p = tmp_path / "model.mlir"
    p.write_text(_MODEL, encoding="utf-8")
    return p


class TestTheRoutingIsInertUnlessAsked:
    def test_without_the_feature_nothing_is_routed_and_no_sidecar_appears(self, tmp_path):
        # The whole-model build must be byte-identical when the feature is off; a sidecar left behind
        # would make a later build think something had been routed.
        prepared, _feats = prepare_for_lowering(_model(tmp_path), tmp_path, features=frozenset(),
                                               blocking=False)
        assert "merlin_opu_gemm_i8" not in prepared.read_text()
        assert not (tmp_path / PO.SIDECAR_NAME).exists()

    def test_the_feature_without_a_routing_target_is_refused(self, tmp_path):
        # Silently not routing would be indistinguishable from a feature that did nothing, and the model
        # would grade correctly while reporting a capability it never used.
        with pytest.raises(ValueError, match="no `matrix=` routing"):
            prepare_for_lowering(_model(tmp_path), tmp_path,
                                 features=frozenset({OPU_MATMUL_NAME}), blocking=False)


class TestTheGeometryHasOneSource:
    """The selector's tile edge and the compiled kernel's tile edge must be the same number.

    Deriving one from the caller's ``vlen`` and the other from the configuration's Scala would be two
    statements of the same fact, and a disagreement would be silent: the selector would choose
    contractions for a geometry the kernel does not have.
    """

    @pytest.fixture
    def routing(self):
        from merlin.common.paths import env as _env
        if not _env("MERLIN_CHIPYARD"):
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        return MatrixRouting(unit="saturn_opu", config="OPUV256D128ShuttleConfig")

    def test_the_edge_comes_from_the_named_configuration(self, routing):
        assert routing.tile_edge() == 32

    def test_a_wider_configuration_gives_a_wider_edge(self, routing):
        from dataclasses import replace
        assert replace(routing, config="OPUV512D256ShuttleConfig").tile_edge() == 64

    def test_the_default_selector_declines_a_contraction_narrower_than_a_tile(self, routing):
        select = routing.selector()

        class _Big:
            parallel, reduction = (64, 64), (32,)

        class _Small:
            parallel, reduction = (8, 8), (32,)

        assert select(_Big()) and not select(_Small())

    def test_a_supplied_selector_overrides_the_default(self, routing):
        # This is the seam the cost model and the e-graph plug into; nothing here decides profitability.
        from dataclasses import replace
        assert replace(routing, select=lambda _s: False).selector()(object()) is False


class TestTheRewriteAndTheSidecarAgree:
    @pytest.fixture
    def routed(self, tmp_path):
        from merlin.common.paths import env as _env
        if not _env("MERLIN_CHIPYARD"):
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        prepared, _feats = prepare_for_lowering(
            _model(tmp_path), tmp_path, features=frozenset({OPU_MATMUL_NAME}), blocking=False,
            matrix=MatrixRouting(unit="saturn_opu", config="OPUV256D128ShuttleConfig"))
        return prepared, PO.load_sidecar(tmp_path)

    def test_only_the_tile_filling_contraction_moves(self, routed):
        prepared, sigs = routed
        text = prepared.read_text()
        assert text.count("func.call @merlin_opu_gemm_i8") == 1
        assert list(sigs.values()) == [(64, 64, 32)]

    def test_every_symbol_the_module_calls_is_in_the_sidecar(self, routed):
        # THE agreement: a call with no sidecar entry is a link error at image-build time, a long way
        # from the rewrite that created it.
        prepared, sigs = routed
        text = prepared.read_text()
        for sym in sigs:
            assert f"func.call @{sym}" in text
        for line in text.splitlines():
            if "func.call @merlin_opu_gemm_i8" in line:
                called = line.split("func.call @", 1)[1].split("(", 1)[0].strip()
                assert called in sigs, f"{called} is called but not recorded for the build"

    def test_the_declarations_keep_their_access_attributes(self, routed):
        # Without these one-shot-bufferize copies the weight operand of every routed contraction. The
        # printer drops them silently, so the rewrite repairs the text and refuses to write it if it
        # cannot -- this is the assertion that the repair happened.
        prepared, sigs = routed
        text = prepared.read_text()
        assert PO.unpatched_declarations(text, PO.OpuRewrite(signatures=dict(sigs))) == ()
