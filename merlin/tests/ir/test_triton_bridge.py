"""TTIR -> linalg-on-tensors: what the bridge produces, and what it refuses to produce.

Two thirds of these tests are refusals, on purpose. A kernel frontend that compiles everything is
not covering more ground than one that refuses — it is guessing, and a wrong guess here emits a
program that runs, verifies at every later stage, and computes something else. So the acceptance
tests pin the output shape, and the refusal tests pin the boundary.

The sharpest pair is `vector_add_unmasked` accepted at n=1024 and refused at n=1000. The same kernel,
the same bridge, the same grid arithmetic: only the declared extent differs. A bridge that ignored
masks would accept both, and would be wrong exactly when the block size does not divide the tensor —
which is the case nobody tests by hand.
"""
from __future__ import annotations

import pytest
import triton_kernels as K

from merlin.triton import source
from merlin.triton.bridge import CORE_DIALECTS, BridgeError, to_linalg

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def bridge(spec):
    return to_linalg(source.make_ttir(spec), spec)


# --------------------------------------------------------------------------------- acceptance


def test_vector_add_becomes_a_whole_tensor_elementwise_add():
    """The 4-program grid disappears: what comes out is the operation, not a tiling of it."""
    result = bridge(K.vector_add_spec(n=1024))
    assert "linalg.add" in result.text
    assert "tensor<1024xf32>" in result.text
    assert result.entry == "vector_add"
    assert result.arg_names == ["x_ptr", "y_ptr"]
    assert result.result_names == ["out_ptr"]
    assert result.report.grid == (4, 1, 1)
    assert "scf." not in result.text and "affine." not in result.text, (
        "the grid was lowered to a loop instead of normalized away — that is a scheduling decision "
        "and it belongs to Merlin, not to the frontend")


def test_a_write_only_pointer_becomes_a_result_not_an_argument():
    """Re-raising is what makes the output ingestible: operands must be block arguments."""
    result = bridge(K.vector_add_spec(n=256))
    fn = next(op for op in result.module.walk() if op.name == "func.func")
    assert len(fn.body.blocks[0].args) == 2, "the output pointer is still an argument"
    assert [str(t) for t in fn.function_type.outputs] == ["tensor<256xf32>"]


def test_the_one_tile_matmul_becomes_the_idiom_merlin_already_ingests():
    """i8 x i8 -> i32 must land on linalg.quantized_matmul, the same op Merlin's own frontend emits.

    Emitting a different-but-equivalent form here would work and would still break the point of the
    exercise: Triton has to converge on Merlin's existing input, not grow a parallel one.
    """
    result = bridge(K.matmul_one_tile_spec())
    assert "linalg.quantized_matmul" in result.text
    assert "tensor<16x32xi8>" in result.text and "tensor<16x16xi32>" in result.text
    assert "linalg.fill" in result.text, "the accumulator must be zeroed, not left undefined"


def test_the_float_matmul_uses_the_plain_matmul_op():
    result = bridge(K.matmul_one_tile_spec(dtype="fp32", acc_dtype="fp32"))
    assert "linalg.matmul" in result.text and "quantized" not in result.text


@pytest.mark.parametrize("n", [1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1000, 1023, 1024, 1025, 4099])
def test_every_masked_tail_size_is_re_raised_to_the_full_tensor(n):
    """The tail sizes are the point: a dropped mask is invisible whenever BLOCK divides n."""
    result = bridge(K.vector_add_spec(n=n))
    assert f"tensor<{n}xf32>" in result.text
    for name in ("x_ptr", "y_ptr", "out_ptr"):
        assert f"whole tensor[{n}]" in result.report.pointer_patterns[name]


def test_only_core_dialects_survive():
    """INV-6: linalg-on-tensors is the convergence point, so nothing else may reach it."""
    for spec in (K.vector_add_spec(), K.matmul_one_tile_spec()):
        result = bridge(spec)
        assert set(result.report.output_dialects) <= CORE_DIALECTS


def test_the_output_round_trips_through_merlins_own_parser():
    """The text has to be ingestible by the LLVM path, which parses it rather than sharing objects."""
    from merlin.xdsl_dialects._common import text

    result = bridge(K.vector_add_spec(n=512))
    assert text(result.module) == result.text
    assert result.text.count("func.func") == 1


def test_every_ttir_op_is_accounted_for():
    """Seen == lowered + deliberately discarded. An unexplained remainder aborts the translation."""
    for spec in (K.vector_add_spec(), K.matmul_one_tile_spec()):
        report = bridge(spec).report
        assert not report.unaccounted, report.as_dict()
        assert sum(report.ttir_ops_seen.values()) == (
            sum(report.ttir_ops_lowered.values()) + sum(report.ttir_ops_discarded.values()))
        assert report.ttir_ops_discarded, "triton's dead range checks should have been discarded"


def test_the_bridge_is_deterministic():
    a, b = bridge(K.vector_add_spec()), bridge(K.vector_add_spec())
    assert a.text == b.text


# ----------------------------------------------------------------------------------- refusal


def test_an_unmasked_kernel_is_accepted_when_the_block_tiles_the_tensor_exactly():
    result = bridge(K.vector_add_unmasked_spec(1024))
    assert "linalg.add" in result.text


def test_the_same_unmasked_kernel_is_refused_when_it_would_run_past_the_end():
    """The bug this exists to catch: correct at n=1024, out of bounds at n=1000, same source."""
    with pytest.raises(BridgeError) as exc:
        bridge(K.vector_add_unmasked_spec(1000))
    message = str(exc.value)
    assert "not covered exactly" in message
    assert "outside the declared shape" in message and "1000" in message


def test_a_full_but_reordered_access_is_reported_as_a_permutation():
    """Covering every element is not enough — the bridge must also say they are in order."""
    with pytest.raises(BridgeError) as exc:
        bridge(K.transposed_store_spec())
    assert "permutation" in str(exc.value)


def test_an_untranslated_op_names_itself():
    with pytest.raises(BridgeError) as exc:
        bridge(K.atomic_add_spec())
    assert exc.value.op == "tt.atomic_rmw"
    assert "no translation" in str(exc.value)


def test_a_runtime_scalar_with_no_declared_value_is_refused():
    """A mask bound that is unknown at compile time cannot be checked, so it is not assumed."""
    from merlin.triton.spec import TritonKernelSpec

    base = K.vector_add_spec(n=1024)
    blind = TritonKernelSpec(function=base.function, args=base.args,
                             grid=K.GridSpec(dims=(4,)), constexprs=base.constexprs)
    with pytest.raises(BridgeError) as exc:
        bridge(blind)
    assert "n_elements" in str(exc.value) and "assumptions" in str(exc.value)


def test_a_mis_declared_effect_is_refused():
    """A kernel that writes an argument the caller believes is read-only is a miscompile."""
    from merlin.triton.spec import KernelArg, TritonKernelSpec

    base = K.matmul_one_tile_spec()
    args = list(base.args)
    args[2] = KernelArg("c_ptr", "pointer", "i32", shape=(16, 16), effect="read")
    args[0] = KernelArg("a_ptr", "pointer", "i8", shape=(16, 32), effect="write")
    lying = TritonKernelSpec(function=base.function, args=tuple(args), grid=base.grid,
                             constexprs=base.constexprs)
    with pytest.raises(BridgeError) as exc:
        bridge(lying)
    assert "effect" in str(exc.value) or "write-only" in str(exc.value)


def test_a_declared_shape_that_disagrees_with_the_kernel_is_refused():
    from merlin.triton.spec import KernelArg, TritonKernelSpec

    base = K.matmul_one_tile_spec()
    args = list(base.args)
    args[0] = KernelArg("a_ptr", "pointer", "i8", shape=(16, 64), effect="read")
    wrong = TritonKernelSpec(function=base.function, args=tuple(args), grid=base.grid,
                             constexprs=base.constexprs)
    with pytest.raises(BridgeError) as exc:
        bridge(wrong)
    assert "a_ptr" in str(exc.value)


def test_a_spec_paired_with_the_wrong_ttir_is_refused():
    """A declared dtype disagreeing with the kernel is normally caught by Triton itself.

    The spec drives the TTIR signature, so `*fp32` never reaches a kernel written for `*i8`. The
    bridge's own check therefore guards the remaining way to get there — pairing a module with a
    spec it was not built from — which is exactly what a stale on-disk artifact would do.
    """
    from merlin.triton.spec import KernelArg, TritonKernelSpec

    base = K.matmul_one_tile_spec()
    ttir = source.make_ttir(base)
    args = list(base.args)
    args[0] = KernelArg("a_ptr", "pointer", "fp32", shape=(16, 32), effect="read")
    other = TritonKernelSpec(function=base.function, args=tuple(args), grid=base.grid,
                             constexprs=base.constexprs)
    with pytest.raises(BridgeError) as exc:
        to_linalg(ttir, other)
    assert "a_ptr" in str(exc.value) and "fp32" in str(exc.value)


def test_a_contraction_under_a_multi_program_grid_is_refused():
    """A batched matmul covers every argument exactly once and still must not be normalized.

    Each program computes an independent product, so folding the grid away would turn a stack of
    small matmuls into one large one. The addressing analysis cannot catch this — coverage is
    perfect — which is why the payload has its own guard.
    """
    with pytest.raises(BridgeError) as exc:
        bridge(K.batched_matmul_spec(batch=2))
    assert exc.value.op == "tt.dot"
    assert "grid" in str(exc.value)


def test_that_same_batched_kernel_has_perfect_coverage_at_batch_one():
    """Proof the refusal above is about the contraction, not about a coverage failure."""
    result = bridge(K.batched_matmul_spec(batch=1))
    assert "linalg.matmul" in result.text
