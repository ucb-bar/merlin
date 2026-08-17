"""``@triton.jit`` -> TTIR through a stock wheel: what comes out, and what must never come out.

The seam this package is built on is that TTIR is *machine-independent*. That is an empirical claim
about triton's frontend, not a definition, so it is tested three ways: TTIR is produced with no GPU
present, it carries no GPU-layout or target dialect, and it does not change when the nominal backend
does. If any of those stopped holding, TTIR would no longer be a legal place to enter Merlin and the
architecture would need revisiting rather than patching.
"""
from __future__ import annotations

import pytest
import triton_kernels as K

from merlin.triton import source
from merlin.triton.spec import KernelArg, TritonKernelSpec

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")

# Dialect prefixes that must not survive into TTIR. `ttg` is Triton's GPU dialect (layouts, warps,
# shared memory) and is the boundary INV-4 forbids; `gpu`/`nvvm`/`rocdl` are device-specific.
FORBIDDEN_PREFIXES = ("ttg.", "triton_gpu.", "gpu.", "nvvm.", "rocdl.")


@pytest.fixture(scope="module")
def add_ttir():
    return source.make_ttir(K.vector_add_spec())


@pytest.fixture(scope="module")
def dot_ttir():
    return source.make_ttir(K.matmul_one_tile_spec())


def test_ttir_is_produced_without_a_gpu(add_ttir):
    """No device, no driver, no `triton.compile()` stage chain — the whole point of the seam."""
    assert add_ttir.kernel_name == "vector_add"
    assert add_ttir.text.strip()
    assert add_ttir.triton_version == source.require().installed


def test_vector_add_ttir_carries_the_expected_ops(add_ttir):
    for op in ("tt.func", "tt.load", "tt.store", "tt.get_program_id", "tt.return"):
        assert add_ttir.has_op(op), f"{op} missing from {add_ttir.ops}"


def test_the_masked_tail_is_visible_in_the_ir(add_ttir):
    """The mask is a real value in TTIR, so the bridge can fail closed instead of ignoring it.

    A dropped mask is the quietest possible miscompile: it reads and writes past the end and still
    produces the right answer whenever the extent divides the block.
    """
    assert "tt.load" in add_ttir.ops
    loads = [op for op in source.walk_ops(add_ttir) if op.get_name() == "tt.load"]
    assert loads, "no tt.load to inspect"
    assert any(op.get_num_operands() > 1 for op in loads), (
        "every tt.load is single-operand, so the mask is not in the IR the bridge sees")


def test_matmul_ttir_carries_a_dot(dot_ttir):
    assert dot_ttir.has_op("tt.dot"), dot_ttir.ops
    assert dot_ttir.has_op("tt.load") and dot_ttir.has_op("tt.store")


def test_no_gpu_dialect_survives_into_ttir(add_ttir, dot_ttir):
    """INV-4: TTGIR is never a Merlin boundary. TTIR must be above the layout assignment."""
    for ttir in (add_ttir, dot_ttir):
        offenders = [op for op in ttir.ops if op.startswith(FORBIDDEN_PREFIXES)]
        assert not offenders, f"{ttir.kernel_name} carries {offenders}"
        assert "#ttg." not in ttir.text and "#triton_gpu." not in ttir.text, (
            f"{ttir.kernel_name} carries a GPU layout attribute")


def test_ttir_is_reproducible():
    """Same source + signature + constexprs -> same digest (INV-10)."""
    a = source.make_ttir(K.vector_add_spec())
    b = source.make_ttir(K.vector_add_spec())
    assert a.digest == b.digest
    assert a.text == b.text
    assert "loc(" not in a.text, "debug locations would make the digest depend on the checkout path"


def test_constexprs_change_the_digest():
    """The digest identifies the compiled program, not just the function — tiles are baked in."""
    small = source.make_ttir(K.vector_add_spec(block=64))
    large = source.make_ttir(K.vector_add_spec(block=256))
    assert small.digest != large.digest


def test_ttir_is_independent_of_the_nominal_backend(monkeypatch):
    """A GPUTarget must be supplied; it must not change the IR.

    This is the load-bearing claim of the whole design. If TTIR differed per backend it would be a
    target-specific IR, and entering Merlin there would mean the frontend had already chosen a
    target — exactly what the architecture forbids.
    """
    baseline = source.make_ttir(K.vector_add_spec())
    monkeypatch.setattr(source, "_NOMINAL_BACKEND", "hip")
    monkeypatch.setattr(source, "_NOMINAL_ARCH", "gfx942")
    monkeypatch.setattr(source, "_NOMINAL_WARP_SIZE", 64)
    try:
        other = source.make_ttir(K.vector_add_spec())
    except source.TritonFrontendError as exc:
        pytest.skip(f"no second backend in this wheel: {exc}")
    assert other.text == baseline.text, (
        "TTIR changed with the nominal backend — it is not machine-independent, and the TTIR seam "
        "is no longer a valid place to enter Merlin")


def test_a_non_jit_function_is_rejected_with_a_useful_message():
    """Fail closed (INV-8): naming the cause beats an AttributeError from inside triton."""
    spec = TritonKernelSpec(
        function=lambda x_ptr: None,
        args=(KernelArg("x_ptr", "pointer", "fp32", shape=(8,), effect="write"),),
        grid=K.GridSpec(dims=(1,)))
    with pytest.raises(source.TritonFrontendError) as exc:
        source.make_ttir(spec)
    assert "@triton.jit" in str(exc.value)


def test_a_kernel_triton_rejects_is_attributed_to_the_kernel():
    """`tl.dot` enforces K >= 32; the rejection must name the kernel, not surface raw."""
    spec = K.matmul_one_tile_spec()
    spec = TritonKernelSpec(function=spec.function, args=spec.args, grid=spec.grid,
                            constexprs={"BM": 16, "BN": 16, "BK": 16})
    with pytest.raises(source.TritonFrontendError) as exc:
        source.make_ttir(spec)
    assert "matmul_one_tile" in str(exc.value)


def test_the_module_can_be_walked_structurally(dot_ttir):
    """The bridge reads the live IR, not MLIR text — no regex, no tokenizer (repo mandate).

    Pins the in-memory op/value API the bridge depends on, so a triton upgrade that removes it
    fails here with a clear cause instead of somewhere inside the bridge.
    """
    ops = source.walk_ops(dot_ttir)
    assert len(ops) > 5
    dot = next(op for op in ops if op.get_name() == "tt.dot")
    assert dot.get_num_operands() >= 2
    assert dot.get_num_results() == 1
    result_type = dot.get_result(0).get_type()
    assert "tensor<" in str(result_type), result_type
    assert str(K.TILE_M) in str(result_type) and str(K.TILE_N) in str(result_type)
