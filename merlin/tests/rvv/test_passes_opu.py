"""The contraction→matrix-unit rewrite, tested on what it must REFUSE as much as what it moves.

Routing a contraction to a unit that cannot compute it produces a call the C side has no definition for,
or worse a definition that computes the wrong thing. So most of these tests hand the pass something it
must decline: a float contraction, a batch contraction, a rank-3 shape, a candidate the selector rejects.

The last class runs the real prepared spectformer module, because the only forms that matter are the ones
the int8 rewrite actually produces — a contraction that has become a `linalg.generic` with `(i8, i8, i32)`
operands and is not renamed back to `linalg.matmul` until later in the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import passes_opu as PO

#: A rank-2 int8 contraction in exactly the form `passes_quant_int` leaves behind: a generic with
#: (i8, i8) inputs, an i32 accumulator initialised by linalg.fill, and extsi/muli/addi in the body.
_INT8_MM = """
builtin.module {
  func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x16xi8>) -> tensor<64x16xi32> {
    %e = tensor.empty() : tensor<64x16xi32>
    %z = arith.constant 0 : i32
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<64x16xi32>) -> tensor<64x16xi32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d2, d1)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x32xi8>, tensor<32x16xi8>)
        outs(%f : tensor<64x16xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<64x16xi32>
    func.return %r : tensor<64x16xi32>
  }
}
"""

#: The same shape in f32 — legal linalg, illegal on an int8 unit.
_F32_MM = _INT8_MM.replace("xi8>", "xf32>").replace("xi32>", "xf32>").replace(
    "%x: i8, %y: i8, %acc: i32", "%x: f32, %y: f32, %acc: f32").replace(
    "arith.extsi %x : i8 to i32", "arith.mulf %x, %x : f32").replace(
    "arith.extsi %y : i8 to i32", "arith.mulf %y, %y : f32").replace(
    "arith.muli %xe, %ye : i32", "arith.mulf %xe, %ye : f32").replace(
    "arith.addi %acc, %m : i32", "arith.addf %acc, %m : f32").replace(
    "%z = arith.constant 0 : i32", "%z = arith.constant 0.0 : f32").replace(
    "ins(%z : i32)", "ins(%z : f32)").replace("linalg.yield %s : i32", "linalg.yield %s : f32")

_ALL = lambda _s: True                                              # noqa: E731


def _module(text: str):
    return parse_mlir_text(text)


class TestItIsInertWithoutADecision:
    def test_no_selector_routes_nothing(self):
        # The pass must not contain the routing policy; that lives in the cost model / e-graph.
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod)
        assert got.count == 0 and got.signatures == {}
        assert any("no selector" in why for _w, why in got.skipped)

    def test_a_selector_that_declines_everything_routes_nothing(self):
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=lambda _s: False)
        assert got.count == 0
        assert any("none selected" in why for _w, why in got.skipped)

    def test_enumeration_does_not_mutate(self):
        mod = _module(_INT8_MM)
        before = len(PO.routable_contractions(mod))
        assert len(PO.routable_contractions(mod)) == before == 1


class TestWhatItRefuses:
    def test_a_float_contraction_is_not_a_candidate(self):
        # Extents alone cannot answer legality on a unit that computes int8 and not fp32.
        mod = _module(_F32_MM)
        assert PO.routable_contractions(mod) == []
        assert PO.rewrite_contractions_to_opu(mod, select=_ALL).count == 0

    def test_a_batch_contraction_is_not_routed(self):
        # A contract declaring `matmul` gaps batch_matmul; folding the batch dim into the tile would
        # compute something the contract never promised.
        text = _INT8_MM.replace(
            "affine_map<(d0, d1, d2) -> (d0, d2)>,\n                                          "
            "affine_map<(d0, d1, d2) -> (d2, d1)>,\n                                          "
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,\n                                          "
            "affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,\n                                          "
            "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>").replace(
            '["parallel", "parallel", "reduction"]',
            '["parallel", "parallel", "parallel", "reduction"]').replace(
            "tensor<64x32xi8>", "tensor<4x64x32xi8>").replace(
            "tensor<32x16xi8>", "tensor<4x32x16xi8>").replace(
            "tensor<64x16xi32>", "tensor<4x64x16xi32>")
        mod = _module(text)
        assert all(sh.op == PO.ROUTABLE_OP for _op, sh in PO.routable_contractions(mod)), \
            "a batch contraction must not be classified as the rank-2 class"

    def test_the_routable_op_class_is_the_rank_two_one(self):
        assert PO.ROUTABLE_OP == "linalg.matmul"
        assert PO.INT8_DTYPES == ("i8", "i8", "i32")

    def test_a_nonzero_init_is_not_routed(self):
        # `linalg.matmul` computes C_init + A@B; the microkernel WRITES its output. Those agree only when
        # C_init is zero, so routing this would silently drop the addend — a wrong answer, not a slow one.
        mod = _module(_INT8_MM.replace("%z = arith.constant 0 : i32", "%z = arith.constant 7 : i32"))
        assert PO.routable_contractions(mod) == []
        assert PO.rewrite_contractions_to_opu(mod, select=_ALL).count == 0

    def test_a_zero_fill_init_is_routed(self):
        # The other side of the same condition, so the check above cannot pass by declining everything.
        mod = _module(_INT8_MM)
        assert len(PO.routable_contractions(mod)) == 1
        assert PO.rewrite_contractions_to_opu(mod, select=_ALL).count == 1

    def test_an_init_that_is_not_a_fill_at_all_is_declined(self):
        # Fail closed: an init this cannot recognise might carry live values. Being wrong in this direction
        # leaves a contraction on the vector path, which is merely slower.
        mod = _module(_INT8_MM.replace(
            "%f = linalg.fill ins(%z : i32) outs(%e : tensor<64x16xi32>) -> tensor<64x16xi32>",
            "%f = tensor.empty() : tensor<64x16xi32>"))
        assert PO.routable_contractions(mod) == []

    def test_zero_initialised_rejects_a_block_argument_init(self):
        # A contraction accumulating into a function argument has an init whose `owner` is a Block rather
        # than an Operation; reaching for `.name` on it would raise instead of declining.
        mod = _module(_INT8_MM.replace(
            "func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x16xi8>) -> tensor<64x16xi32> {",
            "func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x16xi8>, "
            "%c: tensor<64x16xi32>) -> tensor<64x16xi32> {").replace(
            "outs(%f : tensor<64x16xi32>) {", "outs(%c : tensor<64x16xi32>) {"))
        assert PO.routable_contractions(mod) == []


class TestWhatItEmits:
    def test_the_contraction_becomes_a_call_to_the_kernel(self):
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        assert got.count == 1
        from merlin.xdsl_dialects._common import text as to_text
        out = to_text(mod)
        assert f"{PO.SYMBOL_PREFIX}_0" in out
        assert "linalg.generic" not in out, "the contraction itself must be gone"

    def test_one_symbol_per_distinct_signature(self):
        # MLIR function types are monomorphic, so two shapes cannot share a callee.
        two = _INT8_MM.replace(
            "func.return %r : tensor<64x16xi32>",
            """%e2 = tensor.empty() : tensor<8x16xi32>
    %f2 = linalg.fill ins(%z : i32) outs(%e2 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %a2 = tensor.empty() : tensor<8x32xi8>
    %r2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a2, %b : tensor<8x32xi8>, tensor<32x16xi8>)
        outs(%f2 : tensor<8x16xi32>) {
    ^bb1(%x2: i8, %y2: i8, %acc2: i32):
      %xe2 = arith.extsi %x2 : i8 to i32
      %ye2 = arith.extsi %y2 : i8 to i32
      %m2 = arith.muli %xe2, %ye2 : i32
      %s2 = arith.addi %acc2, %m2 : i32
      linalg.yield %s2 : i32
    } -> tensor<8x16xi32>
    func.return %r : tensor<64x16xi32>""")
        mod = _module(two)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        assert got.count == 2
        assert len(got.signatures) == 2, f"expected two callees, got {got.signatures}"

    def test_repeated_shapes_share_one_symbol(self):
        # spectformer calls the same two shapes 12 times each; 12 callees would be 12 copies of a kernel.
        mod = _module(_INT8_MM.replace(
            "func.return %r : tensor<64x16xi32>",
            """%e2 = tensor.empty() : tensor<64x16xi32>
    %f2 = linalg.fill ins(%z : i32) outs(%e2 : tensor<64x16xi32>) -> tensor<64x16xi32>
    %r2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x32xi8>, tensor<32x16xi8>)
        outs(%f2 : tensor<64x16xi32>) {
    ^bb1(%x2: i8, %y2: i8, %acc2: i32):
      %xe2 = arith.extsi %x2 : i8 to i32
      %ye2 = arith.extsi %y2 : i8 to i32
      %m2 = arith.muli %xe2, %ye2 : i32
      %s2 = arith.addi %acc2, %m2 : i32
      linalg.yield %s2 : i32
    } -> tensor<64x16xi32>
    func.return %r : tensor<64x16xi32>"""))
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        assert got.count == 2 and len(got.signatures) == 1

    def test_the_printer_drops_declaration_arg_attrs(self):
        # Documents the xDSL limitation this path has to work around: arg_attrs are stored correctly and
        # print fine on a func WITH a body, but a bodyless declaration prints only its types. If xDSL
        # ever fixes that, this test fails and the patch below can go.
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        from merlin.xdsl_dialects._common import text as to_text
        raw = to_text(mod)
        assert "bufferization.access" not in raw
        assert PO.unpatched_declarations(raw, got) == tuple(got.signatures)

    def test_the_patch_restores_them(self):
        # Not cosmetic: without them one-shot-bufferize copies the weight operand of every routed
        # contraction.
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        from merlin.xdsl_dialects._common import text as to_text
        fixed = PO.patch_declaration_arg_attrs(to_text(mod), got)
        assert fixed.count('bufferization.access = "read"') == 2
        assert fixed.count('bufferization.access = "write"') == 1
        assert PO.unpatched_declarations(fixed, got) == ()

    def test_the_patch_keeps_the_types_intact(self):
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        from merlin.xdsl_dialects._common import text as to_text
        fixed = PO.patch_declaration_arg_attrs(to_text(mod), got)
        assert "tensor<64x32xi8>" in fixed and "tensor<32x16xi8>" in fixed
        assert "-> tensor<64x16xi32>" in fixed

    def test_the_patch_is_idempotent(self):
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        from merlin.xdsl_dialects._common import text as to_text
        once = PO.patch_declaration_arg_attrs(to_text(mod), got)
        twice = PO.patch_declaration_arg_attrs(once, got)
        # Applying it again must not double-annotate; the arg list no longer splits into 3 bare types.
        assert twice.count('bufferization.access') == once.count('bufferization.access')

    def test_the_declaration_is_private(self):
        mod = _module(_INT8_MM)
        PO.rewrite_contractions_to_opu(mod, select=_ALL)
        from merlin.xdsl_dialects._common import text as to_text
        assert "private" in to_text(mod)

    def test_the_signature_records_m_n_k(self):
        mod = _module(_INT8_MM)
        got = PO.rewrite_contractions_to_opu(mod, select=_ALL)
        (sym, mnk), = got.signatures.items()
        assert mnk == (64, 16, 32)
        r = got.routed[0]
        assert (r.m, r.n, r.k) == (64, 16, 32) and r.symbol == sym

    def test_the_report_serialises(self):
        import json
        mod = _module(_INT8_MM)
        json.dumps(PO.rewrite_contractions_to_opu(mod, select=_ALL).to_dict())


_PREPARED = Path("out/artifacts/target-evolution/saturn_opu/v1/latest/prepared/"
                 "spectformer_int8_full/model.prepared.mlir")


@pytest.mark.skipif(not _PREPARED.is_file(), reason=f"needs the prepared capture at {_PREPARED}")
class TestOnTheRealPreparedModel:
    """The only forms that matter are the ones the int8 rewrite actually produces."""

    @pytest.fixture(scope="class")
    def prepared(self):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        return parse_mlir_file(_PREPARED)

    def test_it_finds_exactly_the_census_count(self, prepared):
        # The workload census records 90 linalg.matmul + 16 linalg.batch_matmul for spectformer; the
        # batch ones must not appear here.
        assert len(PO.routable_contractions(prepared)) == 90

    def test_every_candidate_is_int8_rank_two(self, prepared):
        for _op, sh in PO.routable_contractions(prepared):
            assert tuple(sh.dtypes) == PO.INT8_DTYPES
            assert len(sh.parallel) == 2 and len(sh.reduction) == 1

    def test_a_tile_filling_selector_splits_the_work_dominant_shapes_out(self, prepared):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        mod = parse_mlir_file(_PREPARED)
        got = PO.rewrite_contractions_to_opu(mod, select=lambda s: min(s.parallel) >= 32)
        # 41 of 90: the matmul/im2col families that can fill a 32-edge tile. The remaining 49 are the
        # FFT-family N in {8,14} plus the 1x1000 classifier, together ~1% of the arithmetic.
        assert got.count == 41
        assert len(PO.routable_contractions(mod)) == 49
        assert got.skipped == ()

    def test_the_dominant_shapes_are_among_the_signatures(self, prepared):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        mod = parse_mlir_file(_PREPARED)
        got = PO.rewrite_contractions_to_opu(mod, select=lambda s: min(s.parallel) >= 32)
        mnks = set(got.signatures.values())
        assert (196, 1024, 256) in mnks and (196, 256, 1024) in mnks


class TestTheFileSeam:
    """The whole-model build applies this to a module on disk and reads the signatures in another process.

    Both halves are tested here because the failure mode is asymmetric: the rewrite half is visible in the
    module it writes, while the sidecar half is only visible as a link error a long way downstream.
    """

    def test_it_routes_and_records(self, tmp_path):
        prepared = tmp_path / "model.prepared.mlir"
        prepared.write_text(_INT8_MM, encoding="utf-8")
        got = PO.rewrite_prepared_file(prepared, tmp_path, select=_ALL)
        assert got.count == 1
        text = prepared.read_text()
        assert "func.call @merlin_opu_gemm_i8_0" in text
        assert 'bufferization.access = "read"' in text
        assert PO.load_sidecar(tmp_path) == {"merlin_opu_gemm_i8_0": (64, 16, 32)}

    def test_nothing_selected_leaves_the_module_byte_identical(self, tmp_path):
        # The module is only rewritten when something moved, so an enabled-but-declining build cannot
        # perturb the IR through a parse/print round trip.
        prepared = tmp_path / "model.prepared.mlir"
        prepared.write_text(_INT8_MM, encoding="utf-8")
        before = prepared.read_bytes()
        assert PO.rewrite_prepared_file(prepared, tmp_path, select=lambda _s: False).count == 0
        assert prepared.read_bytes() == before
        assert PO.load_sidecar(tmp_path) == {}

    def test_an_absent_sidecar_reads_as_nothing_routed(self, tmp_path):
        assert PO.load_sidecar(tmp_path) == {}

    def test_a_malformed_sidecar_is_an_error_not_an_empty_set(self, tmp_path):
        # Reading a broken sidecar as "nothing routed" would emit a translation unit missing the symbols
        # the module calls, and the failure would surface as an unattributable link error.
        (tmp_path / PO.SIDECAR_NAME).write_text('{"signatures": {"s": [1, 2]}}', encoding="utf-8")
        with pytest.raises(ValueError, match="triple"):
            PO.load_sidecar(tmp_path)


class TestTheTileFillingSelector:
    def test_it_takes_the_edge_as_a_parameter(self):
        # A threshold baked in would be right on one configuration of the unit and wrong on every other.
        class _S:
            parallel, reduction = (16, 64), (32,)
        assert PO.tile_filling_selector(16)(_S()) is True
        assert PO.tile_filling_selector(32)(_S()) is False

    def test_a_nonsense_edge_is_refused(self):
        with pytest.raises(ValueError, match="lane count"):
            PO.tile_filling_selector(0)
