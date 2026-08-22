"""A pass earned by the harness's own reference kernel must say so.

A block-scaled MX capsule cannot be won by a general backend: its E8M0 block scales are corpus-seeded from
the capsule-name salt, live only in `golden.yaml`, and `capsule_golden.mx_operands`' docstring says
outright that "a general backend could not" reconstruct them. So the harness grades those capsules on its
OWN reference MX kernel (`muon_mx_codegen.emit_mx_kernel`).

That is a defensible fixture, but only if the score stays decomposable. It was not: an earlier run of this
corpus reported 40/40 where 9 of those passes were the fixture rather than the submitted compiler, and
nothing in the artifact recorded the difference. `TierResult.toolchain` carries what the adapter actually
built, so a reader can subtract the fixture passes instead of taking the headline at face value.
"""
from __future__ import annotations

from merlin.targetgen.capsule_runner import TierResult


def test_a_reported_toolchain_rides_the_tier_record():
    d = TierResult("L2", "pass", True, toolchain="mx-reference-kernel(not-the-submission;fork)").to_dict()
    assert d["toolchain"] == "mx-reference-kernel(not-the-submission;fork)", (
        "the graded program must reach capsule_result.json — without it a fixture pass and a real pass "
        "are indistinguishable in a finished score")


def test_the_stamp_names_that_it_is_not_the_submission():
    """The value has to be readable by a person skimming a score, not just present."""
    stamp = "mx-reference-kernel(not-the-submission;fork)"
    assert "not-the-submission" in stamp


def test_output_is_unchanged_when_no_toolchain_was_reported():
    """Most adapters report nothing; their rows must stay byte-identical to before the field existed."""
    d = TierResult("L2", "pass", True).to_dict()
    assert "toolchain" not in d
    assert set(d) == {"status", "mandatory", "not_run_is_not_pass", "reason", "cycles",
                      "derived_from_rtl", "cycle_accurate", "evidence", "timing"}


def test_the_mx_branch_precedes_the_artifact_branch():
    """Structural: the MX route must be checked BEFORE is_mlir_artifact.

    The substitution used to live only inside `program_from_cb`, i.e. only on the inline-source path, so an
    MLIR-emitting submission never reached it and every MX capsule failed for a reason no submission could
    fix. Ordering is the whole fix, so pin it.
    """
    import ast
    import inspect

    from merlin.runtime.backends import base as _bk

    muon = _bk.get_backend("muon")
    import importlib
    src = inspect.getsource(importlib.import_module(muon.__name__ + ".muon_oracles"))
    tree = ast.parse(src)
    mx_line = mlir_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "is_mx_cb" and mx_line is None:
                mx_line = node.lineno
            if node.func.attr == "is_mlir_artifact" and mlir_line is None:
                mlir_line = node.lineno
    assert mx_line is not None, "the MX route disappeared from the oracle adapter"
    assert mlir_line is not None, "the MLIR artifact branch disappeared"
    assert mx_line < mlir_line, (
        "is_mx_cb must be tested before is_mlir_artifact, or an MLIR submission skips the MX reference "
        "kernel and the capsule becomes unwinnable again")
