"""`cse_through_provenance`: the capture's `prov.*` tags stop `cse` from seeing duplicate work.

The defect this pins is not a pass bug — it is that METADATA changes which arithmetic executes.
MLIR's `cse` compares whole attribute dictionaries, and `prov.region_id` is unique per captured
region, so two ops that compute the identical value from the identical operands are never common
subexpressions. On the small_llama int8 capture that is 33 extra `linalg.generic` ops, including the
rotary embedding's 8 identical `math.cos` and 8 identical `math.sin` — 1024 libm `cosf` and 1024
`sinf` calls per inference where 128 each suffice.
"""
from __future__ import annotations

import subprocess

import pytest

import merlin.llvmlower.lower  # noqa: F401 — the production import that registers the feature
from merlin.llvmlower import toolchain
from merlin.llvmlower.impr_features import apply_pipeline, apply_schedule, known, normalize
from merlin.llvmlower.pipeline import _upstream_pipeline
from merlin.llvmlower.prov_cse import FEATURE, PROV_PREFIX, rewrite_prepared_file

#: Two structurally identical `math.cos` generics over the same operand, differing ONLY in their
#: provenance tags — the shape the rotary embedding's `cat(cos(f), cos(f))` reaches the pipeline in.
DUPES = """
module {
  func.func @forward(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %e0 = tensor.empty() : tensor<4x8xf32>
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf32>) outs(%e0 : tensor<4x8xf32>) attrs = {prov.region_id = "cos_0", prov.op = "cos", prov.fqn = "blocks.0.attn"} {
    ^bb0(%in: f32, %out: f32):
      %c = math.cos %in : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>
    %e1 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<4x8xf32>) outs(%e1 : tensor<4x8xf32>) attrs = {prov.region_id = "cos_1", prov.op = "cos", prov.fqn = "blocks.0.attn"} {
    ^bb0(%in: f32, %out: f32):
      %c = math.cos %in : f32
      linalg.yield %c : f32
    } -> tensor<4x8xf32>
    %2 = linalg.add ins(%0, %1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%e0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
  }
}
"""

#: The same module with no provenance at all — the fail-closed case.
NO_PROV = DUPES.replace(
    'attrs = {prov.region_id = "cos_0", prov.op = "cos", prov.fqn = "blocks.0.attn"} ', "").replace(
    'attrs = {prov.region_id = "cos_1", prov.op = "cos", prov.fqn = "blocks.0.attn"} ', "")


def test_feature_is_registered_by_the_lowering_entry_point_and_off_by_default():
    """Registration rides `llvmlower.lower`, which every whole-model backend imports; a package that
    names the feature must be able to resolve it without importing this module first."""
    assert FEATURE in known()
    assert normalize(None) == frozenset()
    assert normalize([FEATURE]) == frozenset({FEATURE})


def test_the_feature_edits_neither_the_pass_list_nor_the_schedule():
    """It is a PREPARED-MODULE rewrite, so the frozen baseline pipeline must be untouched by it —
    naming it may not perturb a build that does not reach the preparation layer."""
    passes = _upstream_pipeline().split(",")
    assert apply_pipeline(list(passes), frozenset({FEATURE})) == passes
    sched = "module attributes {transform.with_named_sequence} {}\n"
    assert apply_schedule(sched, frozenset({FEATURE})) == sched


def _count_generics(mlir_text: str, tmp_path, *, strip: bool) -> int:
    """`linalg.generic` ops left after canonicalize+cse, with or without the provenance tags."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / ("stripped.mlir" if strip else "tagged.mlir")
    src.write_text(mlir_text, encoding="utf-8")
    script = tmp_path / "_count.py"
    script.write_text(
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        "ctx = ir.Context()\n"
        "ctx.allow_unregistered_dialects = True\n"
        "mod = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        "with ctx:\n"
        "    PassManager.parse('builtin.module(canonicalize,cse)', ctx).run(mod.operation)\n"
        "print('GENERICS', str(mod.operation).count('linalg.generic'))\n", encoding="utf-8")
    proc = subprocess.run([str(toolchain.m2m_python()), str(script), str(src)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("GENERICS"))
    return int(line.split()[1])


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_provenance_is_what_hides_the_duplicate_from_cse(tmp_path):
    """The whole reason the feature exists: same IR, same passes, and the tags decide the op count."""
    tagged = _count_generics(DUPES, tmp_path / "a", strip=False)
    src = tmp_path / "in.mlir"
    src.write_text(DUPES, encoding="utf-8")
    stripped_path = rewrite_prepared_file(src, tmp_path / "w")
    stripped = _count_generics(stripped_path.read_text(encoding="utf-8"), tmp_path / "b", strip=True)
    assert tagged == 2, "fixture no longer contains a duplicate cse could collapse"
    assert stripped == 1, "stripping provenance did not let cse collapse the duplicate"


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_the_strip_removes_provenance_and_only_provenance(tmp_path):
    src = tmp_path / "in.mlir"
    src.write_text(DUPES, encoding="utf-8")
    out = rewrite_prepared_file(src, tmp_path / "w").read_text(encoding="utf-8")
    assert PROV_PREFIX not in out
    # the arithmetic, the maps and the iterator types are untouched
    assert out.count("math.cos") == 2
    assert out.count("linalg.generic") == 2
    assert "iterator_types" in out and "linalg.add" in out


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_module_with_no_provenance_fails_closed(tmp_path):
    """Enabling the feature on a module it cannot act on must be an error, not a silent no-op —
    otherwise it reports as applied and measures as an inert lever."""
    src = tmp_path / "in.mlir"
    src.write_text(NO_PROV, encoding="utf-8")
    with pytest.raises(RuntimeError, match="carries no"):
        rewrite_prepared_file(src, tmp_path / "w")


def test_the_strip_runs_after_every_provenance_consuming_derivation():
    """The ordering defect, pinned at the source.

    The strip round-trips the module through the MLIR printer, and that print is not parseable by the
    xDSL reader `kernels.shapes.observe_contractions` uses -- which returns an EMPTY list on an
    unparseable module instead of raising. Placed before the per-op block table is derived, it
    therefore took the observed contraction count 19 -> 0, emptied the table, silently dropped
    `perop_register_block`, and left every contraction to convert-linalg-to-loops: MEASURED as
    `vwmacc` 152 -> 0 in the emitted `forward`, while the build, the numerics and the gate all still
    passed. Nothing detects that except the order itself.
    """
    from merlin.common.paths import merlin_dir

    src = (merlin_dir() / "python" / "merlin" / "runtime" / "backends"
           / "zephyr_model.py").read_text(encoding="utf-8")
    prepare = src[src.index("def prepare_for_lowering("):src.index("def _strip_provenance(")]
    uses = [ln.strip() for ln in prepare.splitlines() if "_strip_provenance(" in ln]
    assert uses, "prepare_for_lowering no longer strips provenance at all"
    # Only ever ON THE WAY OUT. A mid-function `prepared = _strip_provenance(...)` would put the
    # unparseable module in front of a derivation that still has to read it.
    assert all(ln.startswith("return ") for ln in uses), (
        f"the provenance strip is not a terminal rewrite: {uses}")
    assert len(uses) == len([ln for ln in prepare.splitlines()
                             if ln.strip().startswith("return ") and "features" in ln]), (
        "prepare_for_lowering has a return path that does not strip provenance")
