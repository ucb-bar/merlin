"""`concat_dps`: a `tensor.concat` is a placement statement, not a reason to move bytes.

Bufferizing a concat gives every operand a `memref.subview` destination and a `memref.copy` into
it, even when the operand's producer already takes "where to write" as an operand and could have
written there in the first place. This pins the rewrite that changes that, and — just as
importantly — the two orderings without which it is worse than useless: it must run AFTER the
pipeline's opening `canonicalize,cse` (running before it un-merges the rotary embedding's identical
transcendental generics) and it must imply `erase_self_copy` (whose post-bufferization
canonicalize/cse is what turns the in-place `insert_slice`'s copy into a deletable self-copy).
"""
from __future__ import annotations

import subprocess

import pytest

import merlin.llvmlower.lower  # noqa: F401 — the production import that registers the feature
from merlin.llvmlower import toolchain
from merlin.llvmlower.concat_dps import FEATURE, RUNNER_PRELUDE
from merlin.llvmlower.impr_features import apply_pipeline, apply_schedule, get, known, normalize
from merlin.llvmlower.selfcopy import FEATURE as SELF_COPY_FEATURE

#: `cat(cos(f), cos(f))` in the shape the rotary embedding reaches the pipeline in: both operands
#: come from a `linalg.generic` writing into its own `tensor.empty`, so both are retargetable.
TWO_PRODUCERS = """
module {
  func.func @forward(%a: tensor<8x16xf32>) -> tensor<8x32xf32> {
    %e0 = tensor.empty() : tensor<8x16xf32>
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<8x16xf32>) outs(%e0 : tensor<8x16xf32>) attrs = {prov.region_id = "cos_0"} {
    ^bb0(%in: f32, %out: f32):
      %c = math.cos %in : f32
      linalg.yield %c : f32
    } -> tensor<8x16xf32>
    %e1 = tensor.empty() : tensor<8x16xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<8x16xf32>) outs(%e1 : tensor<8x16xf32>) attrs = {prov.region_id = "sin_0"} {
    ^bb0(%in: f32, %out: f32):
      %c = math.sin %in : f32
      linalg.yield %c : f32
    } -> tensor<8x32xf32>
    %2 = tensor.concat dim(1) %0, %1 {prov.region_id = "cat_0"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    return %2 : tensor<8x32xf32>
  }
}
"""
# the second generic's declared result type above is a typo magnet; state it correctly
TWO_PRODUCERS = TWO_PRODUCERS.replace(
    "      linalg.yield %c : f32\n    } -> tensor<8x32xf32>",
    "      linalg.yield %c : f32\n    } -> tensor<8x16xf32>")

#: `cat(x, x)` — one value, used twice by the same concat. Only the FIRST position may be
#: retargeted: the second is a genuine second placement of the same bytes.
REPEATED_OPERAND = """
module {
  func.func @forward(%a: tensor<8x16xf32>) -> tensor<8x32xf32> {
    %e0 = tensor.empty() : tensor<8x16xf32>
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<8x16xf32>) outs(%e0 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c = math.cos %in : f32
      linalg.yield %c : f32
    } -> tensor<8x16xf32>
    %1 = tensor.concat dim(1) %0, %0 : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    return %1 : tensor<8x32xf32>
  }
}
"""

#: No destination-passing producer at all: both operands are slices of a function argument. The
#: concat must be LEFT ALONE — decomposing it would move the same bytes under another op name.
NO_PRODUCER = """
module {
  func.func @forward(%a: tensor<8x32xf32>) -> tensor<8x32xf32> {
    %0 = tensor.extract_slice %a[0, 16] [8, 16] [1, 1] : tensor<8x32xf32> to tensor<8x16xf32>
    %1 = tensor.extract_slice %a[0, 0] [8, 16] [1, 1] : tensor<8x32xf32> to tensor<8x16xf32>
    %2 = tensor.concat dim(1) %0, %1 : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    return %2 : tensor<8x32xf32>
  }
}
"""


def test_the_feature_is_registered_by_the_lowering_entry_point_and_off_by_default():
    """`wholemodel_proposer._composes` swallows the KeyError an unregistered name raises and returns
    False, so a lazily-registered feature is not rejected — it is silently never proposed."""
    assert FEATURE in known()
    assert normalize(None) == frozenset()


def test_it_implies_erase_self_copy_because_without_it_the_rewrite_is_worse_than_baseline():
    """MEASURED at the post-bufferization point of the RVV pipeline on small_llama_int8_consistent:
    strided-destination `memref.copy` 12 -> 31 and their elements 4,608 -> 5,216 WITHOUT the erase,
    versus 12 -> 8 and 4,608 -> 2,560 with it. The implication is a correctness-of-measurement
    property, not a convenience."""
    assert get(FEATURE).implies == frozenset({SELF_COPY_FEATURE})
    assert normalize([FEATURE]) == frozenset({FEATURE, SELF_COPY_FEATURE})


def test_it_edits_neither_the_pass_list_nor_the_transform_schedule():
    """It is a pre-pipeline module rewrite gated on argv, so the frozen baseline's pipeline string
    and schedule text must be untouched even when the feature is named."""
    assert get(FEATURE).edit_pipeline is None
    assert get(FEATURE).edit_schedule is None
    assert get(FEATURE).edit_cflags is None
    passes = ["canonicalize", "cse", "one-shot-bufferize", "convert-func-to-llvm"]
    assert apply_pipeline(list(passes), frozenset({FEATURE})) == passes
    sched = "module attributes {transform.with_named_sequence} {}\n"
    assert apply_schedule(sched, frozenset({FEATURE})) == sched


def test_every_lowering_runner_carries_the_rewrite_and_the_same_argv_gate():
    """`erase_self_copy` read as an inert lever for seven beam rounds because ONE runner variant
    (act_poly) drove the PassManager itself and quietly skipped it. Every variant must splice this
    prelude, and they must all read the same argv slot."""
    from merlin.llvmlower import accum_microkernel, pipeline

    runners = [pipeline._RUNNER_SRC, pipeline._RUNNER_ACT_POLY_TAIL, accum_microkernel.run_source()]
    for src in runners:
        assert "_concat_dps(module, ctx)" in src, "a runner variant does not run the rewrite"
        assert "_CONCAT_DPS = len(sys.argv) > 8" in src, "a runner variant has its own gate"
    # ...and the caller passes that slot, so the gate is reachable at all.
    lowering = (pipeline.__file__ and open(pipeline.__file__, encoding="utf-8").read())
    assert "_concat_dps_gate" in lowering and "_fold_wt, _concat_dps_gate" in lowering


def _rewrite(mlir_text: str, tmp_path) -> tuple[str, int, list[str]]:
    """Run the rewrite exactly as the lowering runner does (m2m venv owns the MLIR bindings).

    Returns (printed module, concats rewritten, report kinds)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "in.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = tmp_path / "_rewrite.py"
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n" + RUNNER_PRELUDE + "\n"
        "ctx = ir.Context()\n"
        "mod = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        "n, rep = _concat_dps(mod, ctx)\n"
        "mod.operation.verify()\n"
        "print('N', n)\n"
        "for k, v in rep: print('R', k, v)\n"
        "print('MODULE')\n"
        "print(str(mod.operation))\n", encoding="utf-8")
    proc = subprocess.run([str(toolchain.m2m_python()), str(script), str(src)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    head, _, module = proc.stdout.partition("MODULE\n")
    lines = head.splitlines()
    n = int(next(ln for ln in lines if ln.startswith("N ")).split()[1])
    return module, n, [ln.split(maxsplit=1)[1] for ln in lines if ln.startswith("R ")]


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_both_producers_write_into_the_concatenated_buffer(tmp_path):
    """The payoff shape: no concat survives, each producer's destination is a slice of the result,
    and the module still verifies."""
    module, n, report = _rewrite(TWO_PRODUCERS, tmp_path)
    assert n == 1
    assert "tensor.concat" not in module
    assert module.count("tensor.insert_slice") == 2
    assert module.count("tensor.extract_slice") == 2, "a producer was not retargeted"
    assert [r for r in report if r.startswith("retargeted")] == ["retargeted 2"]
    # the arithmetic is untouched
    assert module.count("math.cos") == 1 and module.count("math.sin") == 1


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_value_used_twice_by_one_concat_is_retargeted_once(tmp_path):
    """`cse_through_provenance` turns `cat(cos, cos)` into `cat(x, x)`. The first position may be
    produced in place; the second is a real second placement of the same bytes and keeps its copy."""
    module, n, report = _rewrite(REPEATED_OPERAND, tmp_path)
    assert n == 1
    assert "tensor.concat" not in module
    assert [r for r in report if r.startswith("retargeted")] == ["retargeted 1"]
    assert module.count("tensor.insert_slice") == 2


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_concat_with_no_destination_passing_producer_is_left_alone(tmp_path):
    """Decomposing it anyway would move the same bytes under a different op name and make the
    feature read as if it had fired."""
    module, n, report = _rewrite(NO_PRODUCER, tmp_path)
    assert n == 0
    assert "tensor.concat" in module
    assert [r.split()[0] for r in report] == ["skip_no_dps_producer", "retargeted"]
