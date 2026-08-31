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


_RUNNER = merlin_dir() / "python/merlin/targetgen/capsule_runner.py"
_CLI = merlin_dir() / "python/merlin/compile_cli.py"


def test_model_grading_accepts_the_package_under_test():
    from merlin.targetgen.capsule_runner import _grade_model_capsule
    assert "package_dir" in inspect.signature(_grade_model_capsule).parameters, (
        "the submission being graded must reach whole-model grading")


def test_the_package_is_threaded_as_the_accelerator_backend_not_the_codegen_package():
    # the grade itself; `_grade_model_capsule` is the wall-clock budget wrapper around it
    seg = _fn_src(_RUNNER, "_grade_model_capsule_inline")
    assert "mesh_package=_pkg" in seg, (
        "the submission is the OOT accelerator backend that certifies tiles -> mesh_package")
    assert "package=_host_package_arg" in seg, (
        "`package` is the descriptor-pinned RVV host lane; it must not fall back to a certified/default "
        "package merely because the submitted accelerator package is a different artifact")


def test_run_capsule_passes_the_package_to_the_model_path():
    src = _RUNNER.read_text(encoding="utf-8")
    assert "_grade_model_capsule(capsule, target=eff_target, timeout=timeout,\n" in src
    assert "package_dir=package_dir, budget_s=_budget)" in src


def test_a_padded_tile_records_the_layer_extent_it_did_not_certify():
    seg = _fn_src(_CLI, "_mesh_verify")
    assert "declared_layer_extent" in seg, "the layer's own extent must survive the rounding"
    assert "padded_to_mesh_edge" in seg
    assert "evidence_note" in seg


def test_the_padding_note_states_the_weaker_claim():
    seg = _fn_src(_CLI, "_mesh_verify")
    assert "PADDED shape runs" in seg, (
        "a padded tile proves the padded shape runs, not the layer's own extent -- say so where a "
        "reader of the record will see it")
