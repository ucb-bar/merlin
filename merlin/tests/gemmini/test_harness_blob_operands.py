"""A constant operand too large to write as C must still be buildable, and must change nothing else.

The corpus this harness serves is supposed to represent captured models, and 99.4% of their MAC mass
sits in shapes whose operands are hundreds of thousands of elements. Written as C initializer lists
those capsules are not slow to build, they are unbuildable -- the sibling SIMT harness measured 124.7
MB of C and a compiler that ran 45+ minutes without finishing. So the blob form is what makes a
census-anchored member exist at all.

These tests pin the emitter, not the toolchain, so they run without a RISC-V compiler. The end-to-end
claim they support was measured separately: at M=1024 K=256 N=128 (the census's own `tall_skinny`
extents, 33.6M MACs) the harness goes from 590,522 bytes to 664, and spike returns the correct result.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _codegen():
    """Import the backend as part of its package (``merlin/targets`` is not importable)."""
    root = Path(__file__).resolve()
    while root.name != "merlin" or not (root / "targets").is_dir():
        if root.parent == root:
            pytest.skip("cannot locate merlin/targets")
        root = root.parent
    pkg = types.ModuleType("gback")
    pkg.__path__ = [str(root / "targets" / "gemmini" / "backend")]
    sys.modules["gback"] = pkg
    return importlib.import_module("gback.gemmini_codegen_mlir")


def _cb(*, m: int, k: int, n: int) -> dict:
    return {"abi_version": "0.1", "target": "gemmini",
            "tensors": {"W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
                        "A0": {"shape": [m, k], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [m, n], "dtype": "i32", "role": "output"}},
            "commands": [
                {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
                 "attributes": {"layout": "packed_rhs"}},
                {"opcode": "MATMUL_RESIDENT",
                 "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
                {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
                 "attributes": {"epilogue": [], "output_dtype": "i32"}},
                {"opcode": "EVICT", "operands": {"handle": "W_res"}}]}


class TestTheGradedDefaultDoesNotMove:
    def test_a_caller_that_offers_no_blobs_gets_the_form_it_always_got(self):
        """This harness is on the graded L0/L1/L3 path.

        A change to every run would make a round's verdicts incomparable with the rounds before it,
        so the opt-out is the DEFAULT and a caller that cannot link an extra object still builds.
        """
        gm = _codegen()
        small = _cb(m=16, k=16, n=16)
        assert gm._harness_c(small) == gm._harness_c(small, blobs=None)
        assert "static const elem_t T_W" in gm._harness_c(small)
        assert "extern const" not in gm._harness_c(small)

    def test_a_small_operand_stays_inline_even_when_blobs_are_available(self):
        """Below the threshold the blob buys nothing and would cost a relocation, so it is not taken."""
        gm = _codegen()
        blobs: dict = {}
        source = gm._harness_c(_cb(m=16, k=16, n=16), blobs=blobs)
        assert blobs == {}
        assert "static const elem_t T_W" in source

    def test_a_large_operand_emitted_without_anywhere_to_put_it_is_still_correct(self):
        """`blobs=None` must degrade to a slow build, never to an undefined symbol."""
        gm = _codegen()
        source = gm._harness_c(_cb(m=64, k=64, n=64))
        assert "extern const" not in source
        assert "static const elem_t T_W[4096]" in source


class TestALargeOperandMovesOutOfLine:
    def test_it_becomes_an_extern_filled_from_a_blob(self):
        gm = _codegen()
        blobs: dict = {}
        source = gm._harness_c(_cb(m=64, k=64, n=64), blobs=blobs)

        assert set(blobs) == {"T_W", "T_A0"}
        assert "extern const elem_t T_W[4096];" in source
        assert "static const elem_t T_W" not in source, "the operand must not be defined twice"

    def test_the_blob_carries_one_byte_per_element_for_an_i8_operand(self):
        """The payload is the operand, not a re-encoding of it: a width bug here is silent."""
        gm = _codegen()
        blobs: dict = {}
        gm._harness_c(_cb(m=64, k=64, n=64), blobs=blobs)
        assert blobs["T_W"]["elems"] == 4096
        assert len(blobs["T_W"]["bytes"]) == 4096

    def test_the_alignment_is_derived_from_the_same_expression_the_c_attribute_used(self):
        """`row_align(1)` is `aligned(1 * DIM * sizeof(elem_t))`; an under-aligned blob is not a build
        error, it is a DMA reading from an address the row stride does not expect."""
        gm = _codegen()
        blobs: dict = {}
        gm._harness_c(_cb(m=64, k=64, n=64), blobs=blobs)
        assert blobs["T_W"]["align"] == gm._ceil_dim(1) * 1      # tile edge x sizeof(i8)

    def test_the_stub_incbins_rather_than_listing_bytes(self):
        """A `.byte` list would move the one-statement-per-element cost from the compiler to the
        assembler, which is not a fix."""
        gm = _codegen()
        stub = gm._blob_asm("T_W", Path("/tmp/T_W.bin"), align=16, elems=4096)
        assert ".incbin" in stub and ".byte" not in stub
        assert ".balign 16" in stub and ".globl T_W" in stub

    def test_the_harness_collapses_by_orders_of_magnitude_at_a_census_shape(self):
        """The point of the change, stated as a number rather than an intention."""
        gm = _codegen()
        shape = {"m": 1024, "k": 256, "n": 128}                  # the census's own tall_skinny
        inline = gm._harness_c(_cb(**shape))
        blobs: dict = {}
        blobbed = gm._harness_c(_cb(**shape), blobs=blobs)
        assert len(inline) > 500_000
        assert len(blobbed) < 2_000
        assert blobs["T_A0"]["elems"] == 1024 * 256
