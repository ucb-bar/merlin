"""Shape-generalization and code-generation probes folded into a loop grade.

Split out of :mod:`loop_grading` so that module stays reviewable. ``loop_grading`` re-exports every
name here and :func:`loop_grading.grade` resolves them through its own module globals at call time,
so a caller (or a test) that replaces ``loop_grading._attach_shape_generalization`` still replaces
the probe the grade runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from merlin.common.cpu_pool import AUXILIARY, with_priority
from merlin.common.tree_hash import hash_tree
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback.public_object_build import projection as _public_object_build_projection


def _shape_skip_reason(verdict: dict) -> str | None:
    """Why this grade's shape probes need not run, or None when its gate needs them.

    Only a grade that is not gradeable and cannot converge skips: nothing it reports may terminate
    authoring, so the probes wait for a gradeable grade. A structure-only grade whose ``all_pass`` is
    still true keeps its probes, because that gate depends on them.
    """
    if verdict.get("gradeable") is False and verdict.get("all_pass") is not True:
        return "grade is not gradeable; required shape probes run on the next gradeable grade"
    return None


def _skip_shape_generalization(verdict: dict, reason: str) -> None:
    verdict["shape_coverage"] = {
        "ran": False,
        "skipped": True,
        "reason": reason,
        "note": "the shape-coverage probes did NOT run for this grade; this is NOT a pass.",
    }
    if verdict.get("all_pass"):
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = "the required shape-coverage probes did not run; not a pass"


# The public suite is the set of shapes an agent can see, so a backend that keys on those shapes passes
# it by construction and the loop has nothing left to say. Measured: a submission converged at 14/26 with
# every failure a shape it had never implemented, its self-check clean on everything it HAD, and no round
# feedback able to point at the gap -- the first instrument to see it was a post-freeze holdout.
#
# These probes are derived from the target's own declared capability closure and its DERIVED tile edge
# (never the corpus), so they leak nothing an agent could not compute itself, and they run at the cheap
# loop tier. Restricted to the multi-tile corners: the question is "does this backend generalize past ONE
# tile, and along WHICH axis" -- per-axis, because a backend that loops over K and N but not M passes two
# of the three, and only naming the axis makes the result actionable.
@with_priority(AUXILIARY)  # Advisory probe CPU yields to author requests and grade capsules.
def _attach_shape_generalization(
    verdict: dict,
    cand,
    run_dir,
    rnd: int,
    *,
    timeout: int,
    artifact_key: str | None = None,
    context: InvocationContext,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
    compiler_profile_service=None,
    before_probe: Callable[[], None] | None = None,
) -> None:
    """Probe whether this round's candidate LOWERS shapes past a single tile, and fold it into the gate.

    Structural, not numerical: it runs only the emit half of the contract, so it costs no oracle,
    needs no golden, and works on an operand format that has no CPU reference. Artifact size is
    reported separately as an advisory observation; a larger shape may use a smaller runtime loop.

    Failure to run is RECORDED, never treated as clean -- a probe that did not run reading as a pass is
    the same class of bug as an unavailable oracle scoring as one.

    ``MERLIN_SHAPE_COVERAGE_WORKERS`` runs corners concurrently and ``MERLIN_SHAPE_COVERAGE_CACHE=1``
    reuses a sweep of byte-identical inputs (see :func:`shape_coverage.measure`). ``before_probe`` runs
    only when probes are actually about to run, so a caller can publish its pending capsule verdict first.
    """
    from . import shape_coverage

    key = artifact_key or f"{rnd:02d}"
    out = run_dir / "qa_history" / f"shape_coverage_round_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    workers, use_cache = shape_coverage.selected_workers(), shape_coverage.cache_selected()
    try:
        cov = shape_coverage.measure(
            cand,
            run_dir,
            context=context,
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
            timeout=min(timeout, 300),
            additional_forbidden=additional_forbidden,
            service=compiler_profile_service,
            before_probe=before_probe,
            workers=workers,
            use_cache=use_cache,
        )
        out.write_text(json.dumps(cov, indent=2))
    except Exception as e:  # noqa: BLE001 -- record it; never let it read as clean
        verdict["shape_coverage"] = {
            "ran": False,
            "error": f"{type(e).__name__}: {e}",
            "note": "the shape-coverage probes did NOT run this round; this is NOT a pass.",
        }
        if verdict.get("all_pass"):
            verdict["all_pass"] = False
            verdict["not_converged_reason"] = (
                "every public capsule passes, but the required shape-coverage probes did not run; "
                "shape generalization is unmeasured, not a pass"
            )
        return

    verdict["shape_coverage"] = {
        "ran": True,
        "scope": "public_emit_only",
        "tile_edge": cov.get("tile_edge"),
        "baseline_tile_lowered": cov.get("baseline_tile_lowered"),
        "per_corner": {c["corner"]: c["outcome"] for c in cov.get("corners", [])},
        "emitted_work": cov.get("emitted_work"),
        "smaller_emitted_artifacts": cov.get("smaller_emitted_artifacts") or [],
        "multi_tile_axes_uncovered": cov.get("multi_tile_axes_uncovered") or [],
        "tail_cases_uncovered": cov.get("tail_cases_uncovered") or [],
        "tail_axes_uncovered": cov.get("tail_axes_uncovered") or [],
        "all_covered": bool(cov.get("all_covered")),
        "unmeasured": cov.get("unmeasured"),
        "detail": {c["corner"]: c.get("detail") for c in cov.get("corners", []) if c.get("detail")},
        "note": (
            "DERIVED shape probes, not corpus capsules: the SAME contraction at one tile and at two "
            "tiles in each of M, K and N, at this target's derived tile edge. `emitted_work` is the "
            "size of the program you emitted for each. Smaller text is advisory: a runtime loop can "
            "correctly be more compact. An uncovered axis or tail means its emit probe declined, "
            "errored or returned no commands; it is not a numerical verdict. If you cannot lower a "
            "shape, DECLARE it (`declined` on the command buffer) instead of emitting a terminator. "
            "Compilation and native numerical checks remain separate."
        ),
    }
    # THE GATE. Passing every public capsule while lowering only the shapes they happen to use is exactly
    # the state that shipped at 14/26, so the loop must not call that converged.
    if verdict.get("all_pass") and not verdict["shape_coverage"]["all_covered"]:
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = (
            "every public capsule passes, but the derived shape probes did not confirm lowering "
            + (
                f"past one tile on axis/axes {verdict['shape_coverage']['multi_tile_axes_uncovered']}"
                if verdict["shape_coverage"]["multi_tile_axes_uncovered"]
                else (
                    f"for tail case(s) {verdict['shape_coverage']['tail_cases_uncovered']}"
                    if verdict["shape_coverage"]["tail_cases_uncovered"]
                    else "the baseline tile itself (nothing about shape can be concluded yet)"
                )
            )
        )


@with_priority(AUXILIARY)  # Advisory probe CPU yields to author requests and grade capsules.
def _attach_codegen_scalability(
    verdict: dict,
    cand,
    run_dir,
    rnd: int,
    *,
    timeout: int,
    context: InvocationContext,
    artifact_key: str | None = None,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
    build_service=None,
    build_selection: dict | None = None,
    build_budget_s: int = 0,
    candidate_sha256: str | None = None,
) -> None:
    """Publish answer-free public size observations without changing scientific grades."""
    from . import codegen_scalability

    key = artifact_key or f"{rnd:02d}"
    out = run_dir / "qa_history" / f"codegen_scalability_round_{key}.json"
    public_out = run_dir / "qa_history" / f"codegen_scalability_public_round_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if build_service is not None:
            if (
                type(build_budget_s) is not int
                or not 0 < build_budget_s <= 120
                or not isinstance(build_selection, dict)
                or not isinstance(candidate_sha256, str)
                or len(candidate_sha256) != 64
                or hash_tree(Path(cand))["sha256"] != candidate_sha256
            ):
                raise ValueError("frozen public object-build selection or candidate changed")
            if codegen_scalability._selected_build_inputs(build_service, context.target) != build_selection:
                raise ValueError("frozen public object-build source/tool selection changed")
        observation = codegen_scalability.run(
            cand,
            target=context.target,
            contract=contract,
            timeout=min(timeout, 30),
            additional_forbidden=additional_forbidden,
            **(
                {"build_service": build_service, "build_timeout_s": build_budget_s} if build_service is not None else {}
            ),
        )
        if build_service is not None:
            observation["graded_candidate_sha256"] = candidate_sha256
        out.write_text(json.dumps(observation, indent=2))
        if build_service is None:
            verdict["codegen_scalability"] = {"ran": True, **observation}
        else:
            if (
                hash_tree(Path(cand))["sha256"] != candidate_sha256
                or codegen_scalability._selected_build_inputs(build_service, context.target) != build_selection
                or any(
                    build.get("build_selection") != build_selection
                    for row in observation.get("samples", [])
                    if isinstance(row, dict)
                    for build in (row.get("build_only"),)
                    if isinstance(build, dict) and build.get("status") == "compiled"
                )
            ):
                raise ValueError("frozen public object-build selection changed during observation")
            visible = _public_object_build_projection(
                observation, budget_s=build_budget_s, candidate_sha256=candidate_sha256
            )
            public_out.write_text(json.dumps(visible, indent=2))
            verdict["codegen_scalability"] = visible
    except Exception as exc:  # noqa: BLE001 -- missing advisory measurements are not numerical failures
        if build_service is None:
            verdict["codegen_scalability"] = {
                "ran": False,
                "error": f"{type(exc).__name__}: {exc}",
                "scope": "public_emit_only",
                "note": "No scalability measurement was obtained; this is not a pass or a numerical verdict.",
            }
        else:
            if not out.is_file():
                out.write_text(
                    json.dumps(
                        {
                            "scope": "host_private_public_build_diagnostic",
                            "candidate_sha256": candidate_sha256,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        indent=2,
                    )
                )
            visible = {
                "ran": False,
                "schema": "merlin.public_object_build_feedback.v1",
                "scope": "public_emit_and_build_only_advisory",
                "candidate_sha256": candidate_sha256,
                "error": "public object-build measurement unavailable",
                "note": "No object-build result was obtained; the numerical grade is unchanged.",
            }
            public_out.write_text(json.dumps(visible, indent=2))
            verdict["codegen_scalability"] = visible
