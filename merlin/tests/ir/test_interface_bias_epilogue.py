"""A bias add fused into ``interface.commit`` — and everything that must still be refused.

``interface.CommitOp`` has always declared a bias epilogue: ``KNOWN_EPILOGUE`` holds
``bias``/``bias_add``, the op carries an optional ``bias`` StringAttr, and its verifier insists on
one whenever a bias stage is present. The runtime engines execute it (``BIAS_STAGES`` in
``merlin.runtime.commandbuffer``, read by both the simulator and the reference). What was missing was
the only thing that could ever populate it: interface materialization built EVERY commit with a
hard-coded empty ``epilogue``, so a matmul whose result was consumed by a bias-add
``linalg.generic`` had that generic left unaccounted, and the completeness guard refused the whole
program.

That guard is right and stays. The fix is to make the bias-add a thing the rebuild ACCOUNTS for —
one ``bias_add`` stage on its contraction's commit — while every other epilogue shape stays exactly
as unaccounted as before. The refusal tests below are the load-bearing half of this file: the danger
in teaching a fail-closed guard a new pattern is widening it into a fail-open one, and a bias dropped
in silence is indistinguishable from an arithmetic defect (every output element off by its own
column's bias).

The bias is referenced BY NAME because the engine has no SSA: its environment is keyed by the names
in the command buffer's resource table. Those names are minted two stages later, in
``runtime_lowering``. ``test_bias_name_is_the_one_the_engine_resolves`` pins the two together by
lowering all the way to the emitted buffer and reading the name back out of it, so a change to that
naming rule turns this red instead of producing a commit that names a tensor no engine has heard of.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xdsl")

from merlin.frontends.linalg_mlir import parse_mlir_text  # noqa: E402
from merlin.xdsl_dialects.lowering.interface_lowering import (  # noqa: E402
    LoweringError, find_bias_epilogues, lower_to_interface)
from merlin.xdsl_dialects.lowering.pipeline import execute, lower_module  # noqa: E402


def _mm_then(epilogue_body: str, *, extra_args: str = "", extra_ops: str = "",
             returns: str = "%s") -> str:
    """``@forward(%a, %b[, ...]) -> a 16x16 matmul %r, then whatever ``epilogue_body`` spells."""
    return f"""
module {{
  func.func @forward(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>{extra_args})
      -> tensor<16x16xf32> {{
    %e = tensor.empty() : tensor<16x16xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
{extra_ops}
{epilogue_body}
    return {returns} : tensor<16x16xf32>
  }}
}}
"""


#: The bias add the interface layer now carries: all-parallel, identity in/out maps, the bias
#: broadcast over the output's trailing (column) axis, body = one ``arith.addf`` of the two inputs.
BIAS_ADD = """
    %o = tensor.empty() : tensor<16x16xf32>
    %s = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%r, %bias : tensor<16x16xf32>, tensor<16xf32>) outs(%o : tensor<16x16xf32>) {
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        linalg.yield %y : f32
    } -> tensor<16x16xf32>
"""

MM_BIAS = _mm_then(BIAS_ADD, extra_args=", %bias: tensor<16xf32>")

MM_ONLY = """
module {
  func.func @forward(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<16x16xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %r : tensor<16x16xf32>
  }
}
"""


def _commits(module):
    return [op for op in module.walk() if op.name == "interface.commit"]


def _stages(commit) -> list[str]:
    return [entry.data for entry in commit.epilogue]


# --------------------------------------------------------------------------------------------
# What the pattern ACCEPTS
# --------------------------------------------------------------------------------------------

def test_bias_add_generic_becomes_a_commit_epilogue_stage():
    """The whole point: the stage is declared AND populated, with the bias named."""
    out = lower_to_interface(parse_mlir_text(MM_BIAS))
    commits = _commits(out)
    assert len(commits) == 1
    assert _stages(commits[0]) == ["bias_add"]
    assert commits[0].bias is not None
    # No leftover linalg in the rebuilt body — the generic became the stage, it was not emitted
    # a second time (which would apply the bias twice).
    assert not [op for op in out.walk() if op.name.startswith("linalg.")]
    out.verify()


def test_the_generics_init_tensor_is_accounted_for_too():
    """``tensor.empty`` feeding only the fused generic is support, so nothing is left over.

    The bug report's error named ``tensor.empty`` rather than the generic, because in that module
    the generic was dead and the empty was returned. Once the generic is payload its init feeds only
    payload, so ``support_ops`` absorbs it — this asserts the whole payload is accounted, not just
    the op that was obviously missing.
    """
    module = parse_mlir_text(MM_BIAS)
    block = [op for op in module.walk() if op.name == "func.func"][0].body.blocks[0]
    from merlin.xdsl_dialects.lowering.input_workload import find_matmuls
    from merlin.xdsl_dialects.lowering.interface_lowering import payload_ops, unaccounted_ops

    payload = payload_ops(block, find_matmuls(module))
    assert [op.name for op in unaccounted_ops(block, payload)] == []


def test_bias_name_is_the_one_the_engine_resolves():
    """MEASURED end to end: the name on the commit is a real tensor of the emitted buffer.

    This is the drift pin between this stage and ``runtime_lowering``'s naming pre-pass. If that
    pre-pass ever renames arguments, the name minted here stops appearing in the resource table and
    this fails — rather than a ``bias_add`` reaching an engine that cannot look the tensor up.
    """
    res = lower_module(parse_mlir_text(MM_BIAS))
    cb = res.command_buffer
    commits = [c for c in cb["commands"] if c["opcode"] == "COMMIT"]
    assert len(commits) == 1
    assert commits[0]["attributes"]["epilogue"] == ["bias_add"]
    name = commits[0]["operands"]["bias"]
    assert name in cb["tensors"], (name, sorted(cb["tensors"]))
    assert cb["tensors"][name]["role"] == "bias"
    assert cb["tensors"][name]["shape"] == [16]


def test_the_lowered_program_computes_matmul_plus_bias():
    """Numerics through the real host engine, against numpy — not merely 'it stopped raising'."""
    rng = np.random.default_rng(20260904)
    a = rng.standard_normal((16, 16))
    w = rng.standard_normal((16, 16))
    bias = rng.standard_normal((16,))

    res = lower_module(parse_mlir_text(MM_BIAS))
    cb = res.command_buffer
    bias_name = [c for c in cb["commands"]
                 if c["opcode"] == "COMMIT"][0]["operands"]["bias"]
    inputs = {"A0": a.tolist(), "W": w.tolist(), bias_name: bias.tolist()}
    got = np.array(execute(res, inputs)["outputs"][cb["outputs"][0]], dtype=float)

    assert np.allclose(got, a @ w + bias, atol=1e-9)
    # And it is NOT the un-biased program: a test that only checked matmul would pass on the bug.
    assert not np.allclose(got, a @ w, atol=1e-6)


def test_a_payload_with_no_epilogue_is_untouched():
    """Nothing else changes: a plain matmul still commits with an empty epilogue and no bias."""
    out = lower_to_interface(parse_mlir_text(MM_ONLY))
    commits = _commits(out)
    assert len(commits) == 1
    assert _stages(commits[0]) == []
    assert commits[0].bias is None


# --------------------------------------------------------------------------------------------
# What the pattern still REFUSES — the fail-closed half
# --------------------------------------------------------------------------------------------

ROW_BIAS = BIAS_ADD.replace("affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(d0)>")

MUL_BODY = BIAS_ADD.replace("arith.addf", "arith.mulf")

TWO_OP_BODY = """
    %o = tensor.empty() : tensor<16x16xf32>
    %s = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%r, %bias : tensor<16x16xf32>, tensor<16xf32>) outs(%o : tensor<16x16xf32>) {
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        %y2 = arith.mulf %y, %bb : f32
        linalg.yield %y2 : f32
    } -> tensor<16x16xf32>
"""

#: A masked store: the epilogue writes a slice of a destination rather than a whole tensor. Nothing
#: about it is a bias, and the guard must say so.
MASKED_STORE = """
    %o = tensor.empty() : tensor<16x16xf32>
    %s = "tensor.insert_slice"(%r, %o) <{static_offsets = array<i64: 0, 0>,
          static_sizes = array<i64: 16, 16>, static_strides = array<i64: 1, 1>,
          operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}>
        : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
"""


@pytest.mark.parametrize("label, module_text", [
    # A ROW bias. The engine's bias stage adds a length-n vector to every row (per COLUMN); a row
    # bias is a different computation no engine here implements, so matching it would emit a stage
    # executed against the wrong axis.
    ("row bias", _mm_then(ROW_BIAS, extra_args=", %bias: tensor<16xf32>")),
    # An elementwise epilogue that is not an add at all.
    ("multiply body", _mm_then(MUL_BODY, extra_args=", %bias: tensor<16xf32>")),
    # A body that does more than the add.
    ("two-op body", _mm_then(TWO_OP_BODY, extra_args=", %bias: tensor<16xf32>")),
    # A masked store.
    ("masked store", _mm_then(MASKED_STORE)),
])
def test_an_epilogue_that_is_not_a_bias_add_is_still_refused(label, module_text):
    with pytest.raises(LoweringError):
        lower_to_interface(parse_mlir_text(module_text))


def test_a_second_consumer_of_the_contraction_result_is_refused():
    """Fusing rewrites the contraction's only readout.

    If something else also reads the raw accumulation, folding the bias into the commit would hand
    that consumer the BIASED tensor instead. The pattern requires a single use, so this stays
    unaccounted and is refused rather than quietly miscompiled.
    """
    text = """
module {
  func.func @forward(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %bias: tensor<16xf32>)
      -> (tensor<16x16xf32>, tensor<16x16xf32>) {
    %e = tensor.empty() : tensor<16x16xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
    %o = tensor.empty() : tensor<16x16xf32>
    %s = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%r, %bias : tensor<16x16xf32>, tensor<16xf32>) outs(%o : tensor<16x16xf32>) {
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        linalg.yield %y : f32
    } -> tensor<16x16xf32>
    return %s, %r : tensor<16x16xf32>, tensor<16x16xf32>
  }
}
"""
    module = parse_mlir_text(text)
    block = [op for op in module.walk() if op.name == "func.func"][0].body.blocks[0]
    from merlin.xdsl_dialects.lowering.input_workload import find_matmuls

    assert find_bias_epilogues(block, find_matmuls(module)) == {}
    with pytest.raises(LoweringError):
        lower_to_interface(module)


def test_a_bias_that_is_not_a_function_argument_is_refused_by_name():
    """A constant bias has no command-buffer name, so it is REFUSED, never invented.

    The resource table declares the function's own tensor arguments; a bias folded into the module
    as a constant is never materialized by any engine. Naming it something plausible would produce a
    commit whose bias lookup fails at run time (or, worse, silently resolves to another tensor).
    """
    text = """
module {
  func.func @forward(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %e = tensor.empty() : tensor<16x16xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%a, %b : tensor<16x16xf32>, tensor<16x16xf32>)
                       outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
    %cb = arith.constant dense<1.0> : tensor<16xf32>
    %o = tensor.empty() : tensor<16x16xf32>
    %s = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%r, %cb : tensor<16x16xf32>, tensor<16xf32>) outs(%o : tensor<16x16xf32>) {
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        linalg.yield %y : f32
    } -> tensor<16x16xf32>
    return %s : tensor<16x16xf32>
  }
}
"""
    with pytest.raises(LoweringError) as excinfo:
        lower_to_interface(parse_mlir_text(text))
    assert "function argument" in str(excinfo.value)


def test_a_bias_add_on_an_integer_contraction_carries_the_stage_too():
    """``arith.addi`` is the integer spelling of the same stage — matched structurally, not by name
    of the float op."""
    text = """
module {
  func.func @forward(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>, %bias: tensor<16xi32>)
      -> tensor<16x16xi32> {
    %e = tensor.empty() : tensor<16x16xi32>
    %z = arith.constant 0 : i32
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<16x16xi32>) -> tensor<16x16xi32>
    %r = linalg.quantized_matmul ins(%a, %b, %z, %z
             : tensor<16x16xi8>, tensor<16x16xi8>, i32, i32)
             outs(%f : tensor<16x16xi32>) -> tensor<16x16xi32>
    %o = tensor.empty() : tensor<16x16xi32>
    %s = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    } ins(%r, %bias : tensor<16x16xi32>, tensor<16xi32>) outs(%o : tensor<16x16xi32>) {
      ^bb0(%x: i32, %bb: i32, %oo: i32):
        %y = arith.addi %x, %bb : i32
        linalg.yield %y : i32
    } -> tensor<16x16xi32>
    return %s : tensor<16x16xi32>
  }
}
"""
    out = lower_to_interface(parse_mlir_text(text))
    commits = _commits(out)
    assert len(commits) == 1
    assert _stages(commits[0]) == ["bias_add"]
    assert commits[0].bias is not None
    out.verify()
