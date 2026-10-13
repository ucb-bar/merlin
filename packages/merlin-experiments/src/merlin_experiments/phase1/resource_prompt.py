"""Runtime resource and scheduling guidance served with the sealed experiment task."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.cpu_pool import grading_cpus, parse_roster, shared_compute_enabled


def render_resource_block(options, *, environ=None, manager_cpus=None, grade_workers=None, boundary=None) -> str:
    environment = os.environ if environ is None else environ
    manager = tuple(sorted(os.sched_getaffinity(0))) if manager_cpus is None else tuple(manager_cpus)
    author = parse_roster(environment.get("MERLIN_CODEX_CPUS", ""))
    shared = shared_compute_enabled(environment)
    compute = grading_cpus(environment)
    if boundary is None:
        from merlin_experiments.phase2.external_resources import active_resource_boundary

        boundary = active_resource_boundary()
    if boundary is not None:
        boundary.verify()
        if not set((*manager, *author, *compute)) <= set(boundary.cpu_ids):
            raise ValueError("task CPU selections exceed the verified resource boundary")
    if grade_workers is None:
        from merlin.targetgen.capsule_grade import default_grade_workers

        grade_workers = default_grade_workers()
    workers = min(grade_workers, len(compute)) if compute else grade_workers
    native = getattr(options, "sim_max_jobs", 0)
    helpers = getattr(options, "codex_max_subagents", None)

    def roster(values):
        return ",".join(map(str, values)) if values else "inherited allocation"

    lines = [
        "\n\n## Resource allocation and scheduling\n",
        f"- Manager CPU affinity observed at task staging: `{roster(manager)}`.",
        f"- Configured Codex author/helper CPU pool: `{roster(author)}`. "
        "Helpers share this pool with the author; they do not receive additional CPUs.",
        f"- Configured {'shared compute' if shared else 'grading'} CPU pool: `{roster(compute)}`. "
        + (
            "Preparation, compilation and simulator engines acquire stages from the same per-core lease pool."
            if shared
            else "Actual build and simulator allocations follow the selected runtime policy."
        ),
        f"- Configured grading worker ceiling: **{workers}**. "
        + (
            f"Configured native broker concurrency ceiling: **{native}**. "
            if native
            else "Native concurrency is engine-derived. "
        )
        + (
            f"Concurrent author subagent ceiling: **{helpers}**."
            if helpers is not None
            else "Author subagent concurrency follows the selected provider's limit."
        ),
        "- These are allocations and ceilings, not active-session counts. The scheduler runs only ready work. "
        "Use current selfcheck progress and broker status to report active/ready/queued work, "
        "operator L2/L3 coverage, and public/private whole-model L3 qualifications separately.",
        f"- Batch independent cases with the documented selfcheck `--workers {workers}` option. "
        "Keep the compiler snapshot stable until that batch returns; source edits invalidate its attribution.",
        "- Obtain focused L2 feedback first, then use immediate automatic L3 promotion or a documented "
        "full required-tier check. Whole-model requests need their same-request operator gate evidence "
        "or a harness-issued continuation; a model-only request cannot supply that evidence.",
        "- Keep one full-cohort request in flight for a compiler snapshot. Use its incremental progress "
        "instead of submitting a duplicate full sweep; the harness yields advisory background work "
        "to an authorized full selfcheck of the same source and cohort.",
        "- For a repeated native timeout, request native progress/liveness diagnostics from the experiment "
        "operator before another long retry. A requested timeout does not override an executor's lower limit. "
        "A timeout remains unqualified, without a numerical correctness verdict.",
        "- Keep builds and simulations on the documented harness routes. Reuse only evidence/cache hits "
        "whose source, executable, owner and instrument identity the harness authorizes.",
    ]
    return "\n".join(lines) + "\n"


def append_resource_guidance(ws: Path, run_dir: Path, options) -> None:
    """Append before provenance, including when a treatment supplied its own task."""
    task = Path(ws) / "TASK.md"
    task.write_text(task.read_text() + render_resource_block(options))
    (Path(run_dir) / "TASK.md").write_bytes(task.read_bytes())
