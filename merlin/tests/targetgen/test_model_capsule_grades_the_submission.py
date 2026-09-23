"""A whole-model capsule must be graded against the SUBMISSION, and must not hide tile padding.

Two defects, found together, both of which made whole-model evidence describe something other than the
thing it appeared to judge.

**1. The submission never arrived.** ``run_capsule`` is handed the package under test, but
``_grade_model_capsule`` did not take it and called ``compile_model(package=None)``. Every whole-model
number — the numeric verdict, the layers-on-mesh accounting, the tile certification — was therefore a
statement about the DEFAULT flow. Measured: mesh verification reported ``n_tiles: 0`` with reason
"no default OOT backend package for target" while a perfectly good submission sat in the caller's hand;
threading it through turned that into ``n_tiles: 15, n_passed: 15`` and a tier map that was finally
non-empty. It goes to ``mesh_package`` (the OOT accelerator backend), NOT ``package`` (the RVV
whole-model codegen package) — two different things one name would conflate.

**2. The certified tile was not the layer.** ``_mesh_verify`` rounds each extent UP to the mesh edge and
recorded only the rounded value, under the name ``layer_extent``. A model whose every matmul is M=8 on a
16-row mesh had 15/15 tiles "pass" at M=16 while the model-execution path declined the real M=8 on all
15 — the same layer, two paths, opposite verdicts, with only the flattering one written down. A padded
tile is evidence that the PADDED shape runs; that is a weaker claim and must say so.
"""

from __future__ import annotations

import ast
import inspect

from merlin.common.paths import merlin_dir


def _fn_src(path, name: str) -> str:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn) or ""


from merlin.common.paths import module_source_path

_RUNNER = module_source_path("merlin.targetgen.capsule_runner")
_MESH = merlin_dir() / "python/merlin/compile/mesh.py"


def test_model_grading_accepts_the_package_under_test():
    from merlin.targetgen.capsule_runner import _grade_model_capsule

    assert "package_dir" in inspect.signature(_grade_model_capsule).parameters, (
        "the submission being graded must reach whole-model grading"
    )


def test_the_package_is_threaded_as_the_accelerator_backend_not_the_codegen_package():
    # the grade itself; `_grade_model_capsule` is the wall-clock budget wrapper around it
    seg = _fn_src(_RUNNER, "_grade_model_capsule_inline")
    assert "mesh_package=_pkg" in seg, (
        "the submission is the OOT accelerator backend that certifies tiles -> mesh_package"
    )
    assert "package=_host_package_arg" in seg, (
        "`package` is the descriptor-pinned RVV host lane; it must not fall back to a certified/default "
        "package merely because the submitted accelerator package is a different artifact"
    )


def test_run_capsule_passes_the_package_to_the_model_path():
    src = _RUNNER.read_text(encoding="utf-8")
    assert "_grade_model_capsule(capsule, target=eff_target, timeout=timeout,\n" in src
    assert "package_dir=package_dir, budget_s=_budget)" in src


def test_a_padded_tile_records_the_layer_extent_it_did_not_certify():
    seg = _fn_src(_MESH, "_mesh_verify")
    assert "declared_layer_extent" in seg, "the layer's own extent must survive the rounding"
    assert "padded_to_mesh_edge" in seg
    assert "evidence_note" in seg


def test_the_padding_note_states_the_weaker_claim():
    seg = _fn_src(_MESH, "_mesh_verify")
    assert "PADDED shape runs" in seg, (
        "a padded tile proves the padded shape runs, not the layer's own extent -- say so where a "
        "reader of the record will see it"
    )


# --- 3. A tile is synthesized from the EXTENT, not from how the op is spelled -----------------------
# Third defect of the same shape. `_mesh_verify` looked a mesh op's name up in `corpus_spec.BUILDERS`,
# aliasing only matmul/linear. torch-MLIR spells a convolution `convolution_im2col_matmul`, which is not
# in that table, so every such op was counted `n_unsynthesizable` -- and the grader turned a non-zero
# count into a cert-tier failure for the whole model. Measured on M2_microvit_gemmini: the model ran
# BIT-EXACT (cos 1.0, max_rel 0.0) with 12 of 12 certifiable tiles passing the L2 screen and the L3 RTL
# cert, and failed anyway, for a builder table the submission does not own.

from merlin.compile.mesh import tile_builder_op  # noqa: E402
from merlin.targetgen.capsule_grade import model_execution_check as _mec  # noqa: E402
from merlin.targetgen.routing import OpDemand  # noqa: E402


def _demand(op: str, *, family: str | None, shaped: bool = True) -> OpDemand:
    mkn = {"m": 16, "k": 32, "n": 64} if shaped else {}
    return OpDemand(op=op, in_fmt="i8", weight_fmt="i8", site="s0", family=family, **mkn)


def test_a_matmul_is_still_chosen_by_name():
    assert tile_builder_op(_demand("matmul", family="contraction")) == ("matmul", "name")
    assert tile_builder_op(_demand("linear", family="contraction")) == ("matmul", "name")


def test_a_contraction_under_another_spelling_gets_a_matmul_tile():
    """M2's shape: the router placed it on the mesh as a contraction and it carries an extent."""
    op, via = tile_builder_op(_demand("convolution_im2col_matmul", family="contraction"))
    assert (op, via) == ("matmul", "family"), "a contraction with a shape has a matmul tile"


def test_an_unknown_op_is_still_unsynthesizable():
    """The mutation: family is UNKNOWN-safe and must never widen."""
    assert tile_builder_op(_demand("mystery_op", family=None)) == (None, "unsynthesizable")
    assert tile_builder_op(_demand("mystery_op", family="elementwise_map")) == (None, "unsynthesizable")


def test_a_contraction_without_an_extent_is_still_unsynthesizable():
    """Nothing to certify: the tile comes from m/k/n, so an unshaped demand has no tile."""
    assert tile_builder_op(_demand("convolution_im2col_matmul", family="contraction", shaped=False)) == (
        None,
        "unsynthesizable",
    )


def _tiles(n_certified: int, n_stub: int) -> dict:
    per = [{"op": "matmul", "site": f"s{i}", "status": "pass"} for i in range(n_certified)]
    per += [{"op": "mystery", "site": f"u{i}", "status": "no_tile_synthesizer"} for i in range(n_stub)]
    return {
        "n_tiles": n_certified,
        "n_passed": n_certified,
        "n_failed": 0,
        "n_unavailable": 0,
        "n_unsynthesizable": n_stub,
        "n_screened": n_certified,
        "n_screen_passed": n_certified,
        "per_tile": per,
    }


def test_an_unsynthesizable_op_is_reported_once_not_three_times():
    """`per_tile` holds a record per ROUTED op; `n_tiles` counts only certified tiles. Comparing the two
    made one condition produce three violation names, two of them wrong."""
    v = _mec({"mesh_tile_verification": _tiles(2, 1)}, {}).get("violations", [])
    assert "model_tile_unsynthesizable" in v, "the real condition is still reported"
    assert "model_tile_evidence_missing_or_malformed" not in v
    assert "model_tile_record_not_pass" not in v


def test_evidence_is_still_judged_over_the_records_that_claim_to_be_tiles():
    """The mutation: a certified record that did not pass must still be caught past the stubs."""
    t = _tiles(2, 1)
    t["per_tile"][0]["status"] = "fail"
    v = _mec({"mesh_tile_verification": t}, {}).get("violations", [])
    assert "model_tile_record_not_pass" in v


def test_a_short_evidence_list_is_still_malformed():
    """The other mutation: fewer tile records than certified tiles is a real hole."""
    t = _tiles(2, 1)
    t["n_tiles"] = 3
    v = _mec({"mesh_tile_verification": t}, {}).get("violations", [])
    assert "model_tile_evidence_missing_or_malformed" in v
