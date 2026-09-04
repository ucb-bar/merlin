"""`fold_weight_transpose`: sink a loop-invariant weight transpose into its consumers' maps.

Four defects are pinned here.

1. THE MATCH THAT NEVER FIRES. ``fuse_transpose_b`` matches ``linalg.matmul``. The integer (W8A8)
   datapath emits its contraction as a ``linalg.generic``, so on a quantized model the lowered
   module contains NO ``linalg.matmul`` and that fold applies to nothing -- while
   ``linalg.transpose`` is 45.9% of the attributed op time in that model's board profile. The
   whole-model test below asserts the generalized fold removes the weight transposes the
   matmul-only one cannot see.

2. THE PERMUTATION DIRECTION. ``linalg.transpose`` gives ``T[i] = W[f]`` with ``f[P[t]] = i[t]``,
   so the composed map is ``M'[P[t]] = M[t]`` -- NOT ``M'[t] = M[P[t]]``. The two agree for every
   2-D ``[1, 0]``, which is every permutation in the models measured so far, so a 2-D fixture cannot
   tell them apart. ``test_non_involutive_permutation_*`` uses a 3-cycle, where they disagree.

3. THE RUNNER VARIANT THAT DROPS THE STAGE. ``erase_self_copy`` read as an inert lever for seven
   beam rounds because one of the three runner variants drove the PassManager itself. Every variant
   is asserted to splice this rewrite and to read its gate.

4. THE LEVER THE BEAM CANNOT SEE. ``_composes`` swallows an unregistered name as "does not
   compose", so a lever registered lazily in a module the proposer never imports is silently never
   proposed. Resolution is asserted in a fresh interpreter.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir, merlin_dir
from merlin.llvmlower import lower as _lower_mod  # noqa: F401  (registers the feature)
from merlin.llvmlower import toolchain
from merlin.llvmlower.impr_features import get, known, normalize
from merlin.llvmlower.transpose_maps import FEATURE, run_source

BUNDLE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent"

_needs_m2m = pytest.mark.skipif(not toolchain.available(),
                                reason="m2m venv / clang not configured")


# ---------------------------------------------------------------------------------------------
# 1. registration + the runner wiring
# ---------------------------------------------------------------------------------------------

def test_feature_is_registered_and_default_off():
    assert FEATURE in known()
    assert get(FEATURE).action_class == "PASS"
    # A runner-gated rewrite must not touch the pass list at all.
    assert get(FEATURE).edit_pipeline is None
    assert normalize(None) == frozenset()
    assert normalize([FEATURE]) == frozenset({FEATURE})


def _runner_variants() -> dict[str, str]:
    from merlin.llvmlower.accum_microkernel import run_source as scalarize_source
    from merlin.llvmlower.pipeline import (
        EMIT_TRANSLATE,
        _RUNNER,
        _activation_poly_runner,
    )
    return {"plain": _RUNNER,
            "act_poly": _activation_poly_runner(EMIT_TRANSLATE),
            "scalarize": scalarize_source().replace("__MERLIN_EMIT__", EMIT_TRANSLATE)}


@pytest.mark.parametrize("variant", sorted(_runner_variants()))
def test_every_runner_variant_runs_the_fold(variant):
    src = _runner_variants()[variant]
    assert "_fold_weight_transposes(" in src, f"{variant} runner never calls the fold"
    assert "_FOLD_WEIGHT_TRANSPOSE" in src, f"{variant} runner never reads the gate"
    assert "sys.argv[7]" in src, f"{variant} runner never receives the gate"


def test_the_gate_is_threaded_from_the_feature_set():
    """argv[7] must come from the normalized feature set, so an unnamed feature stays off."""
    import inspect

    from merlin.llvmlower import pipeline as P
    src = inspect.getsource(P.lower_to_llvm_ir)
    assert "_FOLD_WEIGHT_TRANSPOSE_FEATURE in feats" in src
    assert "_fold_wt" in src


def test_the_beam_can_reach_the_lever():
    """A lever whose NAME does not resolve is swallowed by `_composes` as "does not compose", so it
    is silently never proposed rather than rejected -- the defect that kept `perop_register_block`
    out of the search. Resolution is checked in a FRESH interpreter, because a name another test
    already registered in this one would hide it."""
    import subprocess
    import sys

    code = (
        "from merlin.mining.wholemodel_proposer import RANKED_LEVERS, _composes\n"
        "names = [n for n, _ in RANKED_LEVERS]\n"
        "assert 'fold_weight_transpose' in names, names\n"
        "assert _composes(['fold_weight_transpose'])\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(merlin_dir() / "python")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, env=env)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------------------------
# 2. the rewrite itself, driven through the SHIPPED prelude
# ---------------------------------------------------------------------------------------------

def _fold(tmp_path, mlir_text: str, name: str = "f") -> tuple[str, str]:
    """Run the shipped rewrite on `mlir_text`. Returns (stdout, rewritten module text)."""
    driver = tmp_path / f"{name}_driver.py"
    driver.write_text(run_source(), encoding="utf-8")
    src = tmp_path / f"{name}.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    out = tmp_path / f"{name}.out.mlir"
    proc = subprocess.run([str(toolchain.m2m_python()), str(driver), str(src), str(out)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, out.read_text(encoding="utf-8")


#: The SHAPE of every weight relayout in the int8 whole model: a transpose of a function argument
#: with TWO consumers -- the dequantize generic the integer datapath leaves behind, and the i8
#: contraction generic that replaced the matmul. Neither is a `linalg.matmul`.
_INT8_WEIGHT = """
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#acc = affine_map<(d0, d1, d2) -> (d0, d1)>
#par = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%a: tensor<8x16xi8>, %w: tensor<32x16xi8>, %s: tensor<32xf32>)
      -> (tensor<8x32xi32>, tensor<16x32xf32>) {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x32xi8>
    %t = linalg.transpose ins(%w:tensor<32x16xi8>) outs(%e:tensor<16x32xi8>) permutation = [1, 0]
    %de = tensor.empty() : tensor<16x32xf32>
    %dq = linalg.generic {indexing_maps = [#par, #col, #par],
                          iterator_types = ["parallel", "parallel"]}
        ins(%t, %s : tensor<16x32xi8>, tensor<32xf32>) outs(%de : tensor<16x32xf32>) {
    ^bb0(%x: i8, %sc: f32, %o: f32):
      %xf = arith.sitofp %x : i8 to f32
      %m = arith.mulf %xf, %sc : f32
      linalg.yield %m : f32
    } -> tensor<16x32xf32>
    %ce = tensor.empty() : tensor<8x32xi32>
    %cf = linalg.fill ins(%z : i32) outs(%ce : tensor<8x32xi32>) -> tensor<8x32xi32>
    %mm = linalg.generic {indexing_maps = [#lhs, #rhs, #acc],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %t : tensor<8x16xi8>, tensor<16x32xi8>) outs(%cf : tensor<8x32xi32>) {
    ^bb0(%p: i8, %q: i8, %ac: i32):
      %pe = arith.extsi %p : i8 to i32
      %qe = arith.extsi %q : i8 to i32
      %pr = arith.muli %pe, %qe : i32
      %su = arith.addi %pr, %ac : i32
      linalg.yield %su : i32
    } -> tensor<8x32xi32>
    return %mm, %dq : tensor<8x32xi32>, tensor<16x32xf32>
  }
}
"""


@_needs_m2m
def test_folds_a_weight_transpose_into_both_generic_consumers(tmp_path):
    stdout, text = _fold(tmp_path, _INT8_WEIGHT, "int8")
    assert "FOLDED 1" in stdout, stdout
    assert "into 2 consumer operand(s)" in stdout, stdout
    assert "linalg.transpose" not in text, "the transpose survived the fold"
    # The contraction now reads B as (m, n, k) -> (n, k): the un-transposed [N, K] weight, contiguous
    # along the reduction. That is the transpose-b access an expert GEMM uses.
    assert "affine_map<(d0, d1, d2) -> (d1, d2)>" in text, text


@_needs_m2m
def test_folded_module_verifies(tmp_path):
    _stdout, text = _fold(tmp_path, _INT8_WEIGHT, "verify")
    opt = toolchain.mlir_translate().parent / "mlir-opt"
    if not opt.is_file():
        pytest.skip("standalone mlir-opt not present")
    src = tmp_path / "verify.folded.mlir"
    src.write_text(text, encoding="utf-8")
    proc = subprocess.run([str(opt), str(src), "-o", str(tmp_path / "v.out.mlir")],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------------------------
# 3. fail closed -- each refusal is counted, never guessed
# ---------------------------------------------------------------------------------------------

_COMPUTED_SOURCE = """
#par = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%w: tensor<4x2xf32>) -> tensor<2x4xf32> {
    %e0 = tensor.empty() : tensor<4x2xf32>
    %c = linalg.generic {indexing_maps = [#par, #par], iterator_types = ["parallel", "parallel"]}
        ins(%w : tensor<4x2xf32>) outs(%e0 : tensor<4x2xf32>) {
    ^bb0(%x: f32, %o: f32):
      %y = arith.addf %x, %x : f32
      linalg.yield %y : f32
    } -> tensor<4x2xf32>
    %e = tensor.empty() : tensor<2x4xf32>
    %t = linalg.transpose ins(%c:tensor<4x2xf32>) outs(%e:tensor<2x4xf32>) permutation = [1, 0]
    return %t : tensor<2x4xf32>
  }
}
"""

_SLICE_CONSUMER = """
module {
  func.func @forward(%w: tensor<4x2xf32>) -> tensor<1x4xf32> {
    %e = tensor.empty() : tensor<2x4xf32>
    %t = linalg.transpose ins(%w:tensor<4x2xf32>) outs(%e:tensor<2x4xf32>) permutation = [1, 0]
    %s = tensor.extract_slice %t[0, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
    return %s : tensor<1x4xf32>
  }
}
"""

_OUTS_CONSUMER = """
#par = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%w: tensor<4x2xf32>, %a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %e = tensor.empty() : tensor<2x4xf32>
    %t = linalg.transpose ins(%w:tensor<4x2xf32>) outs(%e:tensor<2x4xf32>) permutation = [1, 0]
    %g = linalg.generic {indexing_maps = [#par, #par], iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<2x4xf32>) outs(%t : tensor<2x4xf32>) {
    ^bb0(%x: f32, %o: f32):
      %y = arith.addf %x, %o : f32
      linalg.yield %y : f32
    } -> tensor<2x4xf32>
    return %g : tensor<2x4xf32>
  }
}
"""


@_needs_m2m
@pytest.mark.parametrize("text,needle", [
    pytest.param(_COMPUTED_SOURCE, "not loop-invariant", id="computed-source"),
    pytest.param(_SLICE_CONSUMER, "states no per-operand indexing_maps", id="unmappable-consumer"),
    pytest.param(_OUTS_CONSUMER, "`outs` operand", id="written-operand"),
])
def test_refusals_are_counted_and_leave_the_transpose(tmp_path, text, needle):
    stdout, out = _fold(tmp_path, text, "refuse")
    assert "FOLDED 0" in stdout, stdout
    assert needle in stdout, stdout
    assert "linalg.transpose" in out, "a refused transpose was removed anyway"


# ---------------------------------------------------------------------------------------------
# 4. the permutation direction -- a 3-cycle, where the two candidate compositions disagree
# ---------------------------------------------------------------------------------------------

#: perm [1, 2, 0]: dim(res, i) = dim(in, perm[i]), so a 2x3x4 input gives a 3x4x2 result and
#: res[i, j, k] = in[k, i, j] -- the same rule numpy's `transpose` follows.
_PERM3 = """
#par = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @forward(%w: tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
    %e = tensor.empty() : tensor<3x4x2xf32>
    %t = linalg.transpose ins(%w:tensor<2x3x4xf32>) outs(%e:tensor<3x4x2xf32>)
         permutation = [1, 2, 0]
    %o = tensor.empty() : tensor<3x4x2xf32>
    %g = linalg.generic {indexing_maps = [#par, #par],
                         iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%t : tensor<3x4x2xf32>) outs(%o : tensor<3x4x2xf32>) {
    ^bb0(%x: f32, %y: f32):
      %z = arith.mulf %x, %x : f32
      linalg.yield %z : f32
    } -> tensor<3x4x2xf32>
    return %g : tensor<3x4x2xf32>
  }
}
"""


@_needs_m2m
def test_non_involutive_permutation_composes_the_right_way(tmp_path):
    stdout, text = _fold(tmp_path, _PERM3, "perm3")
    assert "FOLDED 1" in stdout, stdout
    # M = (d0,d1,d2)->(d0,d1,d2); M'[P[t]] = M[t] with P = [1,2,0] gives (d2,d0,d1).
    # The other candidate, M'[t] = M[P[t]], would give (d1,d2,d0) -- a different array.
    assert "affine_map<(d0, d1, d2) -> (d2, d0, d1)>" in text, text
    assert "affine_map<(d0, d1, d2) -> (d1, d2, d0)>" not in text, text


@_needs_m2m
def test_non_involutive_permutation_keeps_the_numbers(tmp_path):
    """The same 3-cycle, EXECUTED. A map composed the other way still verifies and still lowers --
    only the values tell the two apart."""
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    w = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 0.5
    expect = np.transpose(w, (1, 2, 0)) ** 2

    got = {}
    for tag, feats in (("off", None), ("on", frozenset({FEATURE}))):
        res = lower_model(_PERM3, tmp_path / tag, targets=("host",), textual=True,
                          vectorize=False, features=feats)
        model = HostModel.load(str(res.host_so))
        src = np.ascontiguousarray(w)
        out = np.zeros((3, 4, 2), dtype=np.float32)
        model([(src.ctypes.data, list(src.shape)), (out.ctypes.data, list(out.shape))])
        got[tag] = out.copy()

    assert np.array_equal(got["off"], expect), got["off"]
    assert np.array_equal(got["on"], got["off"]), "the fold changed the values"


# ---------------------------------------------------------------------------------------------
# 5. whole model -- counts, digest, and BOTH goldens under their own tier keys
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                    reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@_needs_m2m
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_folds_every_weight_transpose_and_gates(tmp_path):
    import hashlib

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.runtime.backends.zephyr_model import _gate, prepare_for_lowering
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    prep = tmp_path / "prep"
    prep.mkdir(parents=True, exist_ok=True)
    prepared, _ = prepare_for_lowering(BUNDLE / "model.mlir", prep, int8_compute=True,
                                       features=frozenset(), blocking=False)
    upstream, _stats = preprocess_text_textual(prepared.read_text(encoding="utf-8"))

    # The premise: the matmul-only fold has nothing to match on a quantized model.
    assert upstream.count("linalg.matmul") == 0
    before = upstream.count("linalg.transpose")
    assert before == 25, before

    stdout, folded_text = _fold(tmp_path, upstream, "model")
    assert "FOLDED 15" in stdout, stdout
    assert folded_text.count("linalg.transpose") == 10, "the 10 activation permutes must remain"

    args = resolve_forward_args(BUNDLE)
    golden = np.load(BUNDLE / "golden.npy")
    golden_w8a8 = np.load(BUNDLE / "golden_w8a8.npy")

    outs, digests, gates = {}, {}, {}
    for tag, feats in (("off", frozenset()), ("on", frozenset({FEATURE}))):
        work = tmp_path / tag
        work.mkdir(parents=True, exist_ok=True)
        ll = work / "model.ll"
        ll.write_text(lower_to_llvm_ir(upstream, workdir=work, features=feats), encoding="utf-8")
        so = build_host_shared(ll, work / "model_host.so")
        digests[tag] = hashlib.sha256(so.read_bytes()).hexdigest()
        out = np.zeros(golden.shape, dtype=np.float32)
        bufs = ([(a.ctypes.data, list(a.shape)) for a in args]
                + [(out.ctypes.data, list(out.shape))])
        HostModel.load(str(so), n_args=len(bufs))(bufs)
        outs[tag] = out.copy()
        gates[tag] = _gate(out, {"fp32": golden, "w8a8": golden_w8a8})

    assert digests["off"] != digests["on"], "the feature did not change the emitted object"
    # A transpose fold moves no arithmetic: every value must be bit-identical.
    assert np.array_equal(outs["off"], outs["on"]), "the fold changed the model's output"
    for tag, g in gates.items():
        assert g["tiers"] == ["fp32", "w8a8"], (tag, g)
        assert g["ok"], (tag, g)
    assert gates["off"]["tier_ok"] == gates["on"]["tier_ok"]
