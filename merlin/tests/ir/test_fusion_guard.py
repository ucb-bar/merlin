"""Elementwise fusion never re-evaluates a producer once per element of a broadcast.

Upstream ``linalg-fuse-elementwise-ops`` fuses any single-use all-parallel producer into its consumer,
including through a broadcast, where the fused body then runs once per CONSUMER element: a per-row
``log``/``exp`` chain read by every element of its row ran row-length times over. ``fusion_guard``
adds the control function (refuse when the consumer iterates more points than the operand has
elements, unless the producer's body computes nothing). These tests drive the SAME runner prelude the
lowering installs, through ``pipeline.apply_passes``, and pin both sides: what is refused and what
still fuses.
"""

from __future__ import annotations

import pytest

from merlin.llvmlower import fusion_guard as FG
from merlin.llvmlower import pipeline as P
from merlin.llvmlower import toolchain

m2m = pytest.mark.skipif(
    not toolchain.m2m_python().is_file(), reason="model2MLIR venv (torch-mlir pass registry) not present"
)

FUSE = "canonicalize,cse," + FG.FUSE_PASS

MODULE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#r1 = affine_map<(d0) -> (d0)>
#r1b = affine_map<(d0, d1) -> (d0)>
func.func @f(%x: tensor<4x8xf32>, %s: tensor<4x1xf32>, %v: tensor<4xf32>, %t: tensor<4xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) {
  %e1 = tensor.empty() : tensor<4x1xf32>
  %e = tensor.empty() : tensor<4x8xf32>
  %e4 = tensor.empty() : tensor<4xf32>
  // A per-row chain (log, then exp) read through a size-1 broadcast.
  %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%s : tensor<4x1xf32>) outs(%e1 : tensor<4x1xf32>) {
  ^bb0(%i: f32, %o: f32):
    %l = math.log %i : f32
    linalg.yield %l : f32
  } -> tensor<4x1xf32>
  %b = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<4x1xf32>) outs(%e1 : tensor<4x1xf32>) {
  ^bb0(%i: f32, %o: f32):
    %p = math.exp %i : f32
    linalg.yield %p : f32
  } -> tensor<4x1xf32>
  %c = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x, %b : tensor<4x8xf32>, tensor<4x1xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%i: f32, %r: f32, %o: f32):
    %m = arith.mulf %i, %r : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>
  // A per-element producer read through the identity.
  %n = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%i: f32, %o: f32):
    %q = math.sqrt %i : f32
    linalg.yield %q : f32
  } -> tensor<4x8xf32>
  %d = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%n : tensor<4x8xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%i: f32, %o: f32):
    %q = arith.addf %i, %i : f32
    linalg.yield %q : f32
  } -> tensor<4x8xf32>
  // A rank-1 per-row value read through a projection.
  %w = linalg.generic {indexing_maps = [#r1, #r1], iterator_types = ["parallel"]}
      ins(%v : tensor<4xf32>) outs(%e4 : tensor<4xf32>) {
  ^bb0(%i: f32, %o: f32):
    %q = math.powf %i, %i : f32
    linalg.yield %q : f32
  } -> tensor<4xf32>
  %y = linalg.generic {indexing_maps = [#id, #r1b, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x, %w : tensor<4x8xf32>, tensor<4xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%i: f32, %r: f32, %o: f32):
    %m = arith.addf %i, %r : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>
  // A producer that computes nothing (a copy) read through a projection: fusing it is free.
  %u = linalg.generic {indexing_maps = [#r1, #r1], iterator_types = ["parallel"]}
      ins(%t : tensor<4xf32>) outs(%e4 : tensor<4xf32>) {
  ^bb0(%i: f32, %o: f32):
    linalg.yield %i : f32
  } -> tensor<4xf32>
  %z = linalg.generic {indexing_maps = [#id, #r1b, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x, %u : tensor<4x8xf32>, tensor<4xf32>) outs(%e : tensor<4x8xf32>) {
  ^bb0(%i: f32, %r: f32, %o: f32):
    %m = arith.subf %i, %r : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>
  return %c, %d, %y, %z : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
}
"""


def _generics(text: str) -> list[tuple[str, list[str]]]:
    """``(result type, body op names)`` of each linalg.generic in printed MLIR."""
    lines, out = text.splitlines(), []
    for i, line in enumerate(lines):
        if "linalg.generic" not in line:
            continue
        ops, j = [], i + 1
        while j < len(lines) and not lines[j].strip().startswith("} ->"):
            ops += [w for w in lines[j].split() if w.startswith(("arith.", "math."))]
            j += 1
        out.append((lines[j].strip()[len("} ->") :].strip(), ops))
    return out


def _per_element(text: str) -> list[list[str]]:
    return [ops for result, ops in _generics(text) if result == "tensor<4x8xf32>"]


@m2m
def test_a_per_row_chain_stays_per_row(monkeypatch):
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "1")
    fused = P.apply_passes(MODULE, FUSE)
    assert "merlin.fusion_pin" not in fused
    bodies = _per_element(fused)
    assert bodies and not any({"math.log", "math.exp", "math.powf"} & set(ops) for ops in bodies), bodies
    # The chain itself still collapsed: one per-row op holds both log and exp.
    rows = [ops for result, ops in _generics(fused) if result == "tensor<4x1xf32>"]
    assert any({"math.log", "math.exp"} <= set(ops) for ops in rows), rows


@m2m
def test_what_does_not_multiply_work_still_fuses(monkeypatch):
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "1")
    bodies = _per_element(P.apply_passes(MODULE, FUSE))
    # sqrt fused into its identity-mapped consumer; the copy fused into its projected one.
    assert any({"math.sqrt", "arith.addf"} <= set(ops) for ops in bodies), bodies
    assert len(bodies) == 4, bodies  # c, d (with n), y, z (with u): nothing per-element left unfused


@m2m
def test_without_the_guard_upstream_recomputes_per_element(monkeypatch):
    """The baseline this guards against, so the tests above cannot pass vacuously."""
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    bodies = _per_element(P.apply_passes(MODULE, FUSE))
    assert any({"math.log", "math.exp"} <= set(ops) for ops in bodies), bodies
    assert any("math.powf" in ops for ops in bodies), bodies


def test_every_runner_variant_carries_the_guard():
    src = P._select_runner(P._upstream_pipeline(), frozenset(), emit=P.EMIT_TRANSLATE)
    assert src.index("_fg_pm_module.PassManager = _FGPassManager") < src.index(
        "from torch_mlir.passmanager import PassManager"
    )
    # The inspection binder rewrites that spelling; the guard must never contain it.
    assert "PassManager.parse(" not in FG.RUNNER_PRELUDE


def test_the_guard_reads_its_switch_per_build(monkeypatch):
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    assert "_FG_ENABLED = False" in FG.runner_prelude()
    monkeypatch.delenv("MERLIN_FUSION_GUARD")
    assert "_FG_ENABLED = True" in FG.runner_prelude()


@m2m
@pytest.mark.parametrize("enabled",["0","1"])
@pytest.mark.parametrize("nested",[False,True])
def test_captured_mutable_reads_keep_their_snapshot_before_overlapping_store(monkeypatch,enabled,nested):
    monkeypatch.setenv("MERLIN_FUSION_GUARD",enabled)
    observe = "%value = memref.load %state[%i] : memref<16xi64>"
    if nested:
        observe = """%value = scf.if %allow -> (i64) {
          %read = memref.load %state[%i] : memref<16xi64>
          scf.yield %read : i64
        } else {
          scf.yield %zero : i64
        }"""
    source = """#id = affine_map<(d0)->(d0)>
    func.func @snapshot(%state: memref<16xi64>, %allow: i1) {
      %four = arith.constant 4 : index
      %zero = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %empty = tensor.empty() : tensor<8xi64>
      %values = linalg.generic {indexing_maps=[#id], iterator_types=["parallel"]}
          outs(%empty: tensor<8xi64>) {
      ^bb0(%unused: i64):
        %i = linalg.index 0 : index
        OBSERVE
        linalg.yield %value : i64
      } -> tensor<8xi64>
      // The pass removes this unused destination use before fusing. The
      // snapshot must be protected even while it initially has two uses.
      %updated = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel"]}
          ins(%values: tensor<8xi64>) outs(%values: tensor<8xi64>) {
      ^bb0(%value: i64, %unused: i64):
        %next = arith.addi %value, %one : i64
        linalg.yield %next : i64
      } -> tensor<8xi64>
      linalg.generic {indexing_maps=[#id], iterator_types=["parallel"]}
          ins(%updated: tensor<8xi64>) {
      ^bb0(%value: i64):
        %i = linalg.index 0 : index
        %destination = arith.addi %i, %four : index
        memref.store %value, %state[%destination] : memref<16xi64>
        linalg.yield
      }
      return
    }""".replace("OBSERVE",observe)
    result = P.apply_passes(source,FUSE)
    assert "merlin.fusion_pin" not in result
    assert result.count("linalg.generic") == 2
    assert result.index("memref.load") < result.rindex("linalg.generic") < result.index("memref.store")
