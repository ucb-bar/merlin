"""``fuse_epilogue_loops``: the int8 requant epilogue lands inside the contraction's own loop nest.

Three layers, cheapest first:

1. PIPELINE — the empty feature set leaves every pass list byte-identical, the stage replaces the
   loop-generation anchor in both the scalar and the RVV pipelines, and a pipeline without that
   anchor fails closed instead of guessing.
2. IR — on a real contraction-plus-requant pair lowered through the shipping pre-stage, the
   ``sitofp``/``mulf`` epilogue ends up in the SAME nest as the ``extsi``/``muli`` reduction, and the
   control (the same affine loop form with the fusion pass removed) leaves them in two nests. That
   control is the point: it is what makes the fused result attributable to the fusion rather than to
   the change of loop dialect.
3. NUMERICS — the whole int8 capture, lowered both ways to a host object and run: the outputs must be
   BIT-IDENTICAL and both must gate against the fp32 AND w8a8 goldens under their own tier keys.
   Slow (two whole-model lowerings + two runs), so it is behind ``MERLIN_RUN_SLOW``.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir
from merlin.llvmlower import lower as _lower_mod  # noqa: F401  (registers the feature)
from merlin.llvmlower import epilogue_fusion as EF
from merlin.llvmlower.impr_features import apply_pipeline, get, known, normalize
from merlin.llvmlower.pipeline import _UPSTREAM_PASSES, _splice
from merlin.llvmlower.toolchain import available as _toolchain_available, mlir_translate

BUNDLE = artifacts_dir() / "recaptures" / "small_llama_int8_consistent"


def _mlir_opt():
    """``mlir-opt`` from the same standalone LLVM install that provides ``mlir-translate`` — derived
    from the resolved tool, never a hardcoded path."""
    return mlir_translate().parent / "mlir-opt"


# ---------------------------------------------------------------------------------------------
# 1. pipeline
# ---------------------------------------------------------------------------------------------

def test_feature_is_registered_and_default_off():
    assert EF.FEATURE in known()
    assert get(EF.FEATURE).action_class == "PASS"
    # empty set in => empty set out => the baseline pass list is the SAME list content.
    base = list(_UPSTREAM_PASSES)
    assert normalize(None) == frozenset()
    assert _splice(apply_pipeline(base, normalize(None))) == _splice(base)


def test_stage_replaces_the_loop_anchor():
    base = list(_UPSTREAM_PASSES)
    out = apply_pipeline(base, normalize([EF.FEATURE]))
    i = base.index(EF.LOOP_ANCHOR)
    assert out[:i] == base[:i]
    assert out[i:i + len(EF.fusion_stage())] == EF.fusion_stage()
    assert out[i + len(EF.fusion_stage()):] == base[i + 1:]
    # the anchor survives as the tail of the stage: ops the affine conversion cannot take must
    # still become loops.
    assert EF.fusion_stage()[-1] == EF.LOOP_ANCHOR
    # and the fusion is the zero-tolerance producer-consumer one, not upstream's default.
    fuse = [p for p in EF.fusion_stage() if "affine-loop-fusion" in p]
    assert len(fuse) == 1
    assert "mode=producer" in fuse[0]
    assert f"compute-tolerance={EF.COMPUTE_TOLERANCE}" in fuse[0]


def test_missing_anchor_fails_closed():
    with pytest.raises(ValueError) as e:
        EF.edit_pipeline(["canonicalize", "cse"])
    assert EF.LOOP_ANCHOR in str(e.value)


def test_stage_applies_to_the_rvv_pipeline_too(tmp_path):
    """The board path builds its pass list with ``build_rvv_pipeline``; the anchor must be there."""
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE, build_rvv_pipeline

    sched = tmp_path / "sched.mlir"
    sched.write_text(RVV_TRANSFORM_SCHEDULE, encoding="utf-8")
    plain = build_rvv_pipeline(sched)
    feat = build_rvv_pipeline(sched, features=normalize([EF.FEATURE]))
    assert EF.LOOP_ANCHOR in plain
    assert "affine-loop-fusion" not in plain          # default-off: baseline untouched
    assert "affine-loop-fusion" in feat
    assert "func.func(convert-linalg-to-affine-loops)" in feat


# ---------------------------------------------------------------------------------------------
# 2. IR — does the epilogue actually land in the reduction's nest?
# ---------------------------------------------------------------------------------------------

#: One i8 x i8 -> i32 contraction plus the requant epilogue the int8 datapath emits for it
#: (``out = sitofp(acc) * s_lhs * s_rhs``) — the exact pair ``lower_contraction_int8`` produces.
_PAIR = """
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#acc = affine_map<(d0, d1, d2) -> (d0, d1)>
#par = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
func.func @f(%a: tensor<8x64xi8>, %b: tensor<64x32xi8>,
             %sa: tensor<8xf32>, %sb: tensor<32xf32>) -> tensor<8x32xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<8x32xi32>
  %f = linalg.fill ins(%z : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
  %mm = linalg.generic {indexing_maps = [#lhs, #rhs, #acc],
                        iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<8x64xi8>, tensor<64x32xi8>) outs(%f : tensor<8x32xi32>) {
  ^bb0(%x: i8, %y: i8, %ac: i32):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %p = arith.muli %xe, %ye : i32
    %q = arith.addi %p, %ac : i32
    linalg.yield %q : i32
  } -> tensor<8x32xi32>
  %oe = tensor.empty() : tensor<8x32xf32>
  %rq = linalg.generic {indexing_maps = [#par, #row, #col, #par],
                        iterator_types = ["parallel", "parallel"]}
      ins(%mm, %sa, %sb : tensor<8x32xi32>, tensor<8xf32>, tensor<32xf32>)
      outs(%oe : tensor<8x32xf32>) {
  ^bb0(%ac: i32, %s1: f32, %s2: f32, %o: f32):
    %cf = arith.sitofp %ac : i32 to f32
    %m1 = arith.mulf %cf, %s1 : f32
    %m2 = arith.mulf %m1, %s2 : f32
    linalg.yield %m2 : f32
  } -> tensor<8x32xf32>
  return %rq : tensor<8x32xf32>
}
"""


def _prefix_passes() -> list[str]:
    """The shipping scalar pipeline up to (and including) the loop-generation stage — everything the
    fusion needs to have run, and nothing that would destroy the loop structure we inspect."""
    stop = _UPSTREAM_PASSES.index("convert-scf-to-cf")
    return list(_UPSTREAM_PASSES[:stop])


def _run_opt(tmp_path, passes: list[str], name: str) -> str:
    src = tmp_path / f"{name}.in.mlir"
    src.write_text(_PAIR, encoding="utf-8")
    out = tmp_path / f"{name}.out.mlir"
    proc = subprocess.run(
        [str(_mlir_opt()), str(src), f"--pass-pipeline=builtin.module({_splice(passes)})",
         "-o", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return out.read_text(encoding="utf-8")


def _loop_nests(text: str) -> list[str]:
    """Top-level loop nests of the function body, split on brace depth (structural, no regex)."""
    nests: list[str] = []
    depth = 0
    cur: list[str] | None = None
    start = 0
    for line in text.split("\n"):
        head = line.strip()
        opens = line.count("{") - line.count("}")
        if cur is None and (head.startswith("affine.for") or head.startswith("scf.for")):
            cur = [line]
            start = depth
            depth += opens
            continue
        if cur is not None:
            cur.append(line)
            depth += opens
            if depth <= start:
                nests.append("\n".join(cur))
                cur = None
            continue
        depth += opens
    return nests


def _classify(nests: list[str]) -> dict[str, int]:
    """How many nests carry the reduction, the epilogue, or both."""
    out = {"reduction_only": 0, "epilogue_only": 0, "fused": 0}
    for n in nests:
        red = "arith.muli" in n and "i8 to i32" in n
        epi = "arith.sitofp" in n and "i32 to f32" in n
        if red and epi:
            out["fused"] += 1
        elif red:
            out["reduction_only"] += 1
        elif epi:
            out["epilogue_only"] += 1
    return out


@pytest.mark.skipif(not _mlir_opt().is_file(), reason="standalone mlir-opt not present")
def test_epilogue_fuses_into_the_reduction_nest(tmp_path):
    fused = _classify(_loop_nests(
        _run_opt(tmp_path, apply_pipeline(_prefix_passes(), normalize([EF.FEATURE])), "fused")))
    assert fused["fused"] == 1, fused
    assert fused["reduction_only"] == 0, fused
    assert fused["epilogue_only"] == 0, fused


@pytest.mark.skipif(not _mlir_opt().is_file(), reason="standalone mlir-opt not present")
def test_affine_loop_form_alone_does_not_fuse(tmp_path):
    """CONTROL. The same stage with the fusion pass removed must leave the epilogue in its own nest —
    otherwise a passing fused case would not be evidence that the FUSION did it."""
    stage = [p for p in EF.fusion_stage() if "affine-loop-fusion" not in p]
    passes = list(_prefix_passes())
    i = passes.index(EF.LOOP_ANCHOR)
    passes[i:i + 1] = stage
    ctl = _classify(_loop_nests(_run_opt(tmp_path, passes, "control")))
    assert ctl["fused"] == 0, ctl
    assert ctl["reduction_only"] == 1, ctl
    assert ctl["epilogue_only"] == 1, ctl


# ---------------------------------------------------------------------------------------------
# 3. numerics — whole model, both goldens, under their own tier keys
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"), reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_output_is_bit_identical_and_gates(tmp_path):
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

    variants = {"base": list(_UPSTREAM_PASSES),
                "fused": apply_pipeline(list(_UPSTREAM_PASSES), normalize([EF.FEATURE]))}
    args = resolve_forward_args(BUNDLE)
    golden = np.load(BUNDLE / "golden.npy")
    golden_w8a8 = np.load(BUNDLE / "golden_w8a8.npy")

    outs = {}
    for name, passes in variants.items():
        work = tmp_path / name
        work.mkdir(parents=True, exist_ok=True)
        ll = work / "model.ll"
        ll.write_text(lower_to_llvm_ir(upstream, workdir=work, pipeline=_splice(passes)),
                      encoding="utf-8")
        so = build_host_shared(ll, work / "model_host.so")
        out = np.zeros(golden.shape, dtype=np.float32)
        bufs = [(a.ctypes.data, list(a.shape)) for a in args] + [(out.ctypes.data, list(out.shape))]
        HostModel.load(str(so), n_args=len(bufs))(bufs)
        outs[name] = out.copy()

    # The fusion reorders loops; it must not touch a single bit of arithmetic.
    assert np.array_equal(outs["base"], outs["fused"]), "epilogue fusion changed the output"
    for name, arr in outs.items():
        g = _gate(arr, {"fp32": golden, "w8a8": golden_w8a8})
        assert set(g["tiers"]) == {"fp32", "w8a8"}, (name, g["tiers"])
        assert g["ok"], (name, g)
        assert g["tier_ok"], (name, g)
