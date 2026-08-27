"""Large operands are LINKED IN, not materialized element-wise, on the object-kernel path.

The self-contained C harness writes one C statement per operand element. That is deliberate on the
inline-source path, where the flat-``.text`` re-map cannot carry a relocation, so a constant pool or a
``.rodata`` array would be unbuildable. But it does not scale: clang is superlinear in statement count,
and the array is a stack local besides. Measured before this path existed, a 2048x2048 f32 operand
produced 4,194,305 statements / 124.7 MB of C and clang ran 45+ minutes without finishing -- which is
why no whole-model capsule had ever been graded.

The object-kernel path (:func:`muon.compile_mlir_forkfree`) transcodes reloc-PRESERVING, so it can carry
the HI20/LO12 pair a ``.rodata`` reference costs. Past a threshold the operand becomes a blob the linker
supplies and the kernel reads in place -- no initializer, no copy, no stack.

These tests pin all three properties that matter: small operands are untouched, large ones stop being
emitted as text, and the linked result still computes the right answer.
"""
import numpy as np
import pytest

from merlin.runtime.backends.base import get_backend

_B = get_backend("muon")
muon = _B.muon
H = _B.muon_harness
CG = _B.muon_codegen_mlir

needs_cyclotron = pytest.mark.skipif(
    not muon.available("cyclotron"),
    reason="cyclotron SIMT oracle / stock LLVM not available",
)


def _model():
    from merlin.targetgen.isa_model import isa_model_from_encoding
    from merlin.targetgen.rtl import mlc_bridge

    fact = mlc_bridge.isa_encoding_for("radiance")
    if not fact:
        pytest.skip("no derived ISA encoding fact for radiance")
    return isa_model_from_encoding("radiance", fact)


def _arg(name, r, c, fill=0.5):
    return H.TensorArg(name=name, rows=r, cols=c, values=[fill] * (r * c), dtype="f32")


def _matmul_cb(n: int, a: np.ndarray, w: np.ndarray) -> dict:
    return {
        "commands": [
            {"opcode": "MATMUL", "operands": {"dst": "acc", "lhs": "A", "rhs": "B"}},
            {"opcode": "COMMIT", "operands": {"dst": "OUT", "src": "acc"}},
        ],
        "tensors": {
            "A": {"shape": [n, n], "dtype": "f32", "role": "input"},
            "B": {"shape": [n, n], "dtype": "f32", "role": "weight"},
        },
        "canonical_inputs": {
            "A": {"values": a.ravel().tolist()},
            "B": {"values": w.ravel().tolist()},
        },
    }


def test_small_operand_stays_element_wise():
    """Below the threshold nothing changes, so no capsule that passes today can regress."""
    model = _model()
    h = H.build_external_kernel_main(
        [_arg("A", 8, 8)], [_arg("Y", 8, 8, 0.0)], kernel_symbol="k", model=model
    )
    assert h.blobs == {}, "a small operand must not become a blob"
    assert "_in_A[0]=" in h.source, "a small operand must stay element-wise on the stack"


def test_large_operand_becomes_a_blob_and_leaves_the_source_small():
    """Past the threshold the operand leaves the C text entirely.

    The assertion is on SOURCE SIZE, not merely on the blob existing: the defect being fixed was that
    the operand was emitted as text at all.
    """
    model = _model()
    n = 2048  # the shape that produced 124.7 MB of C and never finished compiling
    h = H.build_external_kernel_main(
        [_arg("W", n, n)], [_arg("Y", n, 1, 0.0)], kernel_symbol="k", model=model
    )
    assert set(h.blobs) == {"_in_W"}
    assert len(h.blobs["_in_W"]) == n * n * 4, "blob must hold every element as a 32-bit word"
    assert "_in_W[0]=" not in h.source, "the operand must not be materialized element-wise"
    assert len(h.source) < 100_000, (
        f"harness source is {len(h.source)} bytes; the element-wise form was 124.7 MB and the "
        "whole point is that operand size no longer reaches the C text"
    )


def test_large_output_moves_off_the_stack():
    """A big output is the same stack hazard as a big input and must land in .bss."""
    model = _model()
    h = H.build_external_kernel_main(
        [_arg("A", 4, 4)], [_arg("Y", 4096, 1, 0.0)], kernel_symbol="k", model=model
    )
    assert "static volatile uint32_t _out_Y[4096];" in h.source
    assert "  volatile uint32_t _out_Y[4096];" not in h.source


@needs_cyclotron
def test_blob_operands_build_and_compute_the_right_answer(tmp_path):
    """The end-to-end claim: a blob-linked build still produces the correct numbers.

    Shrinking the C text is worthless if the linked ELF reads the operand from the wrong place, so this
    checks the result bit-exactly against numpy rather than checking that the build merely succeeded.
    """
    n = 40  # 1600 elements per operand -- over the threshold, small enough to simulate
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(n, n)).astype("f4")
    w = rng.integers(-3, 4, size=(n, n)).astype("f4")
    cb = _matmul_cb(n, a, w)

    elf = muon.compile_mlir_forkfree(
        CG.emit_kernel_mlir(cb, target="radiance"), cb, tmp_path, target="radiance"
    )
    main_c = (tmp_path / "main.c").read_text()
    assert "_in_A[0]=" not in main_c, "operands should have been linked in, not emitted as text"

    console, cycles, _ = muon.run_elf(elf, simulator="cyclotron", timeout=900)
    outs, _raw = muon.parse_output(console, cycles)
    assert outs, f"kernel produced no OUT record; console tail: {console[-400:]!r}"
    got = np.array(next(iter(outs.values())), dtype="f4").reshape(n, n)
    assert np.array_equal(got, a @ w), (
        f"blob-linked operands gave the wrong result; max abs err {np.abs(got - a @ w).max()}"
    )
