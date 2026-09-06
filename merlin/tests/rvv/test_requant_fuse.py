"""``fuse_requant_into_contraction``: the W8A8 requant epilogue runs inside the contraction's tile loop.

Four layers, cheapest first:

1. REGISTRATION — the name resolves in a process that imported no proposer (the lowering subprocess
   re-imports ``impr_features`` fresh, and a lever that is only registered by the proposer raises
   "unknown impr feature" there), and the empty feature set leaves the tagger source, the schedule
   text and the concrete per-op feature byte-identical.
2. PAIRING — the predicate that decides which epilogues are fusable, run over the exact IR the int8
   datapath emits, plus each way it must REFUSE.
3. IR — on that pair, the generated arms put the epilogue inside the tile loop and delete the
   model-sized i32 accumulator; the CONTROL (the same arms without the two fusions) leaves it.
4. NUMERICS — the whole int8 capture, lowered both ways to a host object and run: BIT-IDENTICAL
   output, and both arms gating against the fp32 AND w8a8 goldens. Slow, behind ``MERLIN_RUN_SLOW``.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from merlin.common.paths import artifacts_dir
from merlin.llvmlower import perop_blocks as PB
from merlin.llvmlower import requant_fuse as RF
from merlin.llvmlower.impr_features import ensure_perop_block, get, known
from merlin.llvmlower.toolchain import available as _toolchain_available, mlir_translate

BUNDLE = artifacts_dir() / "recaptures" / "lstmnetvit_int8_consistent"

#: One (fill, i8 x i8 -> i32 contraction, requant) triple, exactly as ``lower_contraction_int8``
#: emits it once ``linalg-specialize-generic-ops`` has recovered the named contraction, with the
#: block tag the per-op tagger applies.
_TRIPLE = """
#par = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
func.func @f(%a: tensor<8x64xi8>, %b: tensor<64x32xi8>,
             %sa: tensor<8xf32>, %sb: tensor<32xf32>) -> tensor<8x32xf32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<8x32xi32>
  %fl = linalg.fill ins(%z : i32) outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
  %mm = linalg.matmul {merlin.blk_mm_4x8} ins(%a, %b : tensor<8x64xi8>, tensor<64x32xi8>)
        outs(%fl : tensor<8x32xi32>) -> tensor<8x32xi32>
  %oe = tensor.empty() : tensor<8x32xf32>
  %rq = linalg.generic {indexing_maps = [#par, #row, #col, #par],
                        iterator_types = ["parallel", "parallel"]}
      ins(%mm, %sa, %sb : tensor<8x32xi32>, tensor<8xf32>, tensor<32xf32>)
      outs(%oe : tensor<8x32xf32>) {
  ^bb0(%ac: i32, %s1: f32, %s2: f32, %o: f32):
    %fv = arith.sitofp %ac : i32 to f32
    %m1 = arith.mulf %fv, %s1 : f32
    %m2 = arith.mulf %m1, %s2 : f32
    linalg.yield %m2 : f32
  } -> tensor<8x32xf32>
  return %rq : tensor<8x32xf32>
}
"""

_TABLE = {"linalg.matmul:8x32:64": (4, 8)}


def _mlir_opt():
    """``mlir-opt`` from the same standalone LLVM install that provides ``mlir-translate`` — derived
    from the resolved tool, never a hardcoded path."""
    return mlir_translate().parent / "mlir-opt"


# ---------------------------------------------------------------------------------------------
# 1. registration / default-off
# ---------------------------------------------------------------------------------------------

def test_feature_is_registered_and_is_a_pass():
    assert RF.ensure_registered() == RF.FEATURE
    assert RF.FEATURE in known()
    assert get(RF.FEATURE).action_class == "PASS"


def test_both_points_are_registered_and_neither_implies_the_other():
    """The lever has two points and they do not agree in sign on the emitted code, so a build must
    say WHICH one it is: the plain point removes a traversal and leaves the epilogue's loop form to
    clang; `_vec` also reshapes that loop into a fixed MR x NR tile."""
    RF.ensure_registered()
    assert RF.VEC_FEATURE in known()
    assert RF.VEC_FEATURE != RF.FEATURE
    assert not get(RF.FEATURE).implies and not get(RF.VEC_FEATURE).implies


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_naming_both_points_is_refused(tmp_path):
    """Two spellings of one fusion in one feature set describe two builds; there is no correct way to
    pick one, so the preparation must refuse rather than let sorted order decide."""
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering
    with pytest.raises(ValueError) as e:
        prepare_for_lowering(BUNDLE / "model.mlir", tmp_path, int8_compute=True,
                             features=frozenset(["perop_register_block", RF.FEATURE,
                                                 RF.VEC_FEATURE]), harts=1, vlen=256)
    assert RF.VEC_FEATURE in str(e.value)


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_either_point_without_the_block_request_is_refused(tmp_path):
    """The pair tags come from the per-op tagger; named alone the lever would tag nothing, build the
    baseline and report as applied."""
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering
    for name in (RF.FEATURE, RF.VEC_FEATURE):
        with pytest.raises(ValueError) as e:
            prepare_for_lowering(BUNDLE / "model.mlir", tmp_path, int8_compute=True,
                                 features=frozenset([name]), harts=1, vlen=256)
        assert "perop_register_block" in str(e.value)


def test_only_the_epilogue_vectorize_differs_between_the_points():
    """The CONTRACTION is vectorized either way — that is the shape the per-op block arm already gave
    it. The knob must move the epilogue's shape and nothing else, or the two points are not an
    attribution of the reshape."""
    plain = RF.fused_arms([[0, 4, 8, 2]])
    vec = RF.fused_arms([[0, 4, 8, 2]], True)
    assert plain.count("transform.structured.vectorize") == 1
    assert vec.count("transform.structured.vectorize") == 2
    # everything else is identical: the vec text is the plain text plus one line
    extra = [l for l in vec.split("\n") if l not in plain.split("\n")]
    assert len(extra) == 1 and "vectorize" in extra[0], extra
    from merlin.llvmlower.impr_features import ensure_perop_block as _e
    assert _e(_TABLE, 16, [[0, 4, 8, 2]]) != _e(_TABLE, 16, [[0, 4, 8, 2]], True)


def test_name_resolves_without_importing_the_proposer():
    """The lowering runs in a SUBPROCESS that re-imports `impr_features` and imports no proposer, so
    a lever registered only by `wholemodel_proposer` raises there while resolving fine in the parent.
    Asserted in a real fresh interpreter rather than by reading the hook."""
    import sys
    for name in (RF.FEATURE, RF.VEC_FEATURE):
        proc = subprocess.run(
            [sys.executable, "-c",
             "from merlin.llvmlower.impr_features import normalize;"
             f"print(sorted(normalize([{name!r}])))"],
            capture_output=True, text=True)
        assert proc.returncode == 0, (name, proc.stderr)
        assert name in proc.stdout


def test_default_off_leaves_the_tagger_and_the_schedule_byte_identical():
    assert PB.runner_rewrite_src(_TABLE, pair_fuse=False) == PB.runner_rewrite_src(_TABLE)
    assert "tag_requant_pairs" not in PB.runner_rewrite_src(_TABLE)
    assert PB.schedule_text(_TABLE, 16, []) == PB.schedule_text(_TABLE, 16)
    assert "rqfuse" not in PB.schedule_text(_TABLE, 16)
    # and the concrete per-op feature registered for an unfused build is a DIFFERENT name from the
    # fused one, so a fused build can never be handed the unfused schedule.
    assert ensure_perop_block(_TABLE, 16) != ensure_perop_block(_TABLE, 16, [[0, 4, 8, 2]])


def test_pair_table_reaches_the_schedule():
    text = PB.schedule_text(_TABLE, 16, [[0, 4, 8, 2]])
    assert RF.tag_for(0, RF.ROLE_REQUANT) in text
    assert RF.tag_for(0, RF.ROLE_CONTRACTION) in text
    assert RF.tag_for(0, RF.ROLE_FILL) in text
    assert text.count("transform.structured.fuse_into_containing_op") == 2
    assert "transform.apply_patterns.tensor.fold_tensor_empty" in text


def test_tiling_is_derived_from_the_parallel_rank_and_refuses_anything_else():
    assert RF._tile_spec(2, 4, 8)[0] == "[4, 8]"
    assert RF._tile_spec(3, 4, 8)[0] == "[1, 4, 8]"
    assert RF._tile_spec(2, 4, 8)[1] == "[0, 0, 1]"
    assert RF._tile_spec(3, 4, 8)[1] == "[0, 0, 0, 1]"
    with pytest.raises(ValueError):
        RF._tile_spec(4, 4, 8)


# ---------------------------------------------------------------------------------------------
# 2. the pairing predicate, over the real IR shape
# ---------------------------------------------------------------------------------------------

def _pair_probe(module_text: str) -> tuple[list, dict]:
    """Run the TAGGER's own pairing phase in the m2m venv and return ``(pairs, refused)``.

    The predicate under test is the one that is actually spliced into the tagger, not a copy.
    """
    import json
    import tempfile
    from pathlib import Path

    from merlin.llvmlower.toolchain import m2m_python

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        td = Path(td)
        (td / "m.mlir").write_text(module_text, encoding="utf-8")
        (td / "probe.py").write_text(
            "import sys, json\n"
            "from torch_mlir import ir\n"
            "from torch_mlir.passmanager import PassManager\n"
            + PB.runner_rewrite_src(_TABLE, pair_fuse=True) +
            "\nctx = ir.Context()\nctx.allow_unregistered_dialects = True\n"
            "mod = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
            "with ctx, ir.Location.unknown():\n"
            "    pairs, refused = tag_requant_pairs(mod, ctx)\n"
            "print('PAIRS', json.dumps([pairs, refused]))\n", encoding="utf-8")
        proc = subprocess.run([str(m2m_python()), str(td / "probe.py"), str(td / "m.mlir")],
                              capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("PAIRS "))
    return tuple(json.loads(line[len("PAIRS "):]))


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv not configured")
def test_the_int8_triple_pairs():
    pairs, refused = _pair_probe(_TRIPLE)
    assert pairs == [[0, 4, 8, 2]], (pairs, refused)
    assert refused == {}, refused


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv not configured")
def test_a_second_consumer_of_the_accumulator_refuses():
    """Fusing an accumulator that something else also reads would either duplicate the contraction or
    leave it materialized anyway — the pass must decline, and SAY which reason."""
    text = _TRIPLE.replace(
        "  return %rq : tensor<8x32xf32>",
        "  %oe2 = tensor.empty() : tensor<8x32xf32>\n"
        "  %rq2 = linalg.generic {indexing_maps = [#par, #par],\n"
        "                         iterator_types = [\"parallel\", \"parallel\"]}\n"
        "      ins(%mm : tensor<8x32xi32>) outs(%oe2 : tensor<8x32xf32>) {\n"
        "  ^bb0(%ac: i32, %o: f32):\n"
        "    %fv = arith.sitofp %ac : i32 to f32\n"
        "    linalg.yield %fv : f32\n"
        "  } -> tensor<8x32xf32>\n"
        "  %s = arith.addf %rq, %rq2 : tensor<8x32xf32>\n"
        "  return %s : tensor<8x32xf32>")
    pairs, refused = _pair_probe(text)
    assert pairs == [], pairs
    assert refused == {"accumulator_has_2_uses": 1}, refused


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv not configured")
def test_a_reducing_consumer_refuses():
    """A consumer that REDUCES the accumulator has a different iteration space; tiling it and fusing
    the contraction in would change which reductions are complete when the consumer runs."""
    text = _TRIPLE.replace(
        'iterator_types = ["parallel", "parallel"]}\n'
        '      ins(%mm, %sa, %sb : tensor<8x32xi32>, tensor<8xf32>, tensor<32xf32>)\n'
        '      outs(%oe : tensor<8x32xf32>)',
        'iterator_types = ["parallel", "reduction"]}\n'
        '      ins(%mm, %sa, %sb : tensor<8x32xi32>, tensor<8xf32>, tensor<32xf32>)\n'
        '      outs(%oe : tensor<8x32xf32>)')
    pairs, refused = _pair_probe(text)
    assert pairs == [], pairs
    assert refused == {"consumer_has_reduction": 1}, refused


@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv not configured")
def test_a_shared_accumulator_fill_refuses():
    """The fill is pulled into the tile loop too; if another op reads the same filled tensor, doing
    so would change what that op sees."""
    text = _TRIPLE.replace(
        "  return %rq : tensor<8x32xf32>",
        "  %keep = linalg.generic {indexing_maps = [#par, #par],\n"
        "                          iterator_types = [\"parallel\", \"parallel\"]}\n"
        "      ins(%fl : tensor<8x32xi32>) outs(%oe : tensor<8x32xf32>) {\n"
        "  ^bb0(%ac: i32, %o: f32):\n"
        "    %fv = arith.sitofp %ac : i32 to f32\n"
        "    linalg.yield %fv : f32\n"
        "  } -> tensor<8x32xf32>\n"
        "  %s = arith.addf %rq, %keep : tensor<8x32xf32>\n"
        "  return %s : tensor<8x32xf32>")
    pairs, refused = _pair_probe(text)
    assert pairs == [], pairs
    assert refused == {"fill_shared": 1}, refused


# ---------------------------------------------------------------------------------------------
# 3. IR — the epilogue lands in the tile loop and the model-sized i32 tensor is gone
# ---------------------------------------------------------------------------------------------

def _tagged_triple() -> str:
    """``_TRIPLE`` with the three pair attributes applied by hand, in the tagger's own spelling.

    Hand-applied rather than tagger-applied so this layer needs only ``mlir-opt``: what it is testing
    is the ARMS, and the tagger's own output is what layer 2 asserts.
    """
    text = _TRIPLE.replace("{merlin.blk_mm_4x8}",
                           "{" + RF.tag_for(0, RF.ROLE_CONTRACTION) + "}")
    text = text.replace("%fl = linalg.fill ins",
                        "%fl = linalg.fill {" + RF.tag_for(0, RF.ROLE_FILL) + "} ins")
    return text.replace("%rq = linalg.generic {indexing_maps",
                        "%rq = linalg.generic {" + RF.tag_for(0, RF.ROLE_REQUANT)
                        + ", indexing_maps")


def _run_sched(tmp_path, arms: str, tag: str) -> str:
    src = tmp_path / f"{tag}.mlir"
    src.write_text(_tagged_triple(), encoding="utf-8")
    lib = tmp_path / f"{tag}_sched.mlir"
    lib.write_text("module attributes {transform.with_named_sequence} {\n"
                   "  transform.named_sequence @__transform_main"
                   "(%arg0: !transform.any_op {transform.readonly}) {\n"
                   f"{arms}"
                   "    transform.yield\n  }\n}\n", encoding="utf-8")
    out = tmp_path / f"{tag}.out.mlir"
    proc = subprocess.run(
        [str(_mlir_opt()), str(src),
         f"--transform-preload-library=transform-library-paths={lib}",
         "--transform-interpreter", "--canonicalize", "--cse", "-o", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return out.read_text(encoding="utf-8")


@pytest.mark.skipif(not _mlir_opt().is_file(), reason="standalone mlir-opt not present")
@pytest.mark.parametrize("vec", [False, True])
def test_the_accumulator_is_tile_sized_and_the_epilogue_is_in_the_tile_loop(tmp_path, vec):
    text = _run_sched(tmp_path, RF.fused_arms([[0, 4, 8, 2]], vec), f"fused{int(vec)}")
    # the model-sized i32 accumulator is gone entirely; what is left is the MR x NR tile
    assert "tensor<8x32xi32>" not in text
    assert "tensor<4x8xi32>" in text
    # and the epilogue's arithmetic is inside the tile loop, after the K loop
    body = text[text.index("scf.for"):]
    assert "arith.sitofp" in body and "arith.mulf" in body
    # the contraction is vectorized either way; only the epilogue's own shape follows the knob
    assert "vector<4x8xi32>" in text
    assert ("vector<4x8xf32>" in text) is vec


@pytest.mark.skipif(not _mlir_opt().is_file(), reason="standalone mlir-opt not present")
def test_control_without_the_two_fusions_keeps_the_model_sized_accumulator(tmp_path):
    """CONTROL. Tiling the epilogue ALONE must leave the whole i32 accumulator in place — otherwise a
    passing fused case would be evidence of the tiling, not of the fusion."""
    arms = "\n".join(l for l in RF.fused_arms([[0, 4, 8, 2]]).split("\n")
                     if "fuse_into_containing_op" not in l
                     and "%q0ck" not in l and "%q0cf" not in l) + "\n"
    text = _run_sched(tmp_path, arms, "control")
    assert "tensor<8x32xi32>" in text


# ---------------------------------------------------------------------------------------------
# 4. numerics — whole model, both goldens, bit-identical
# ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"), reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "golden_w8a8.npy").is_file(), reason="int8 capture bundle absent")
def test_whole_model_output_is_bit_identical_and_gates(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual
    from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE, lower_to_llvm_ir
    from merlin.runtime.backends.zephyr_model import _gate, prepare_for_lowering
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    golden = np.load(BUNDLE / "golden.npy")
    golden_w8a8 = np.load(BUNDLE / "golden_w8a8.npy")
    args = resolve_forward_args(BUNDLE)
    outs = {}
    for name, feats in (("base", ["perop_register_block"]),
                        ("fused", ["perop_register_block", RF.FEATURE])):
        work = tmp_path / name
        work.mkdir(parents=True, exist_ok=True)
        prepared, concrete = prepare_for_lowering(BUNDLE / "model.mlir", work, int8_compute=True,
                                                  features=frozenset(feats), harts=1, vlen=256)
        upstream, _ = preprocess_text_textual(prepared.read_text(encoding="utf-8"))
        ll = work / "model.ll"
        ll.write_text(lower_to_llvm_ir(upstream, workdir=work, vectorize=True,
                                       transform_schedule=RVV_TRANSFORM_SCHEDULE,
                                       features=frozenset(concrete or ())), encoding="utf-8")
        so = build_host_shared(ll, work / "model_host.so")
        out = np.zeros(golden.shape, dtype=np.float32)
        bufs = [(a.ctypes.data, list(a.shape)) for a in args] + [(out.ctypes.data, list(out.shape))]
        HostModel.load(str(so), n_args=len(bufs))(bufs)
        outs[name] = out.copy()

    assert np.array_equal(outs["base"], outs["fused"]), "the fused epilogue changed the output"
    for name, arr in outs.items():
        g = _gate(arr, {"fp32": golden, "w8a8": golden_w8a8})
        assert g["ok"], (name, g)
        assert sorted(g["tiers"]) == ["fp32", "w8a8"], (name, g["tiers"])


@pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"), reason="whole-model lowering; MERLIN_RUN_SLOW=1")
@pytest.mark.skipif(not _toolchain_available(), reason="m2m venv / clang not configured")
@pytest.mark.skipif(not (BUNDLE / "model.mlir").is_file(), reason="int8 capture bundle absent")
def test_every_contraction_of_the_int8_capture_pairs(tmp_path):
    """The census this lever is ranked on, asserted rather than quoted: on a real int8 capture every
    tagged contraction pairs, so a later change that quietly stops pairing them is a test failure and
    not a lever that measures as a no-op."""
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering
    work = tmp_path / "prep"
    work.mkdir(parents=True, exist_ok=True)
    _prepared, _ = prepare_for_lowering(
        BUNDLE / "model.mlir", work, int8_compute=True,
        features=frozenset(["perop_register_block", RF.FEATURE]), harts=1, vlen=256)
    tagged = (work / "model.perop_tagged.mlir").read_text(encoding="utf-8")
    n_pairs = tagged.count(RF.TAG_PREFIX) // 3
    assert n_pairs >= 40, n_pairs
    # nothing is left carrying BOTH a block tag and a pair tag: two arms tiling one contraction
    # would not be a slower build, it would be a wrong one.
    for line in tagged.split("\n"):
        if RF.TAG_PREFIX in line:
            assert PB.TAG_PREFIX not in line, line
