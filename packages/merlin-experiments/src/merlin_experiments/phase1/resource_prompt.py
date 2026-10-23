"""Runtime resource and scheduling guidance served with the sealed experiment task."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.cpu_pool import grading_cpus, parse_roster, preparation_cpus, shared_compute_enabled
from .telemetry.token_budget import guidance as token_guidance


def _public_codegen_policy_line(selection):
    from merlin.targetgen.public_codegen_budget import validate_policy

    policy = validate_policy(selection)
    limits = ", ".join(f"{key}={value}" for key, value in policy["device_ir_budget"].items()
                       if key.startswith("max_") and value is not None)
    return (
        f"- Original public scale-emission resource policy: mode `{policy['device_ir_budget']['mode']}`, "
        f"limits `{limits}`; mandatory for completion: **{policy['required']}**. "
        f"Use `python3 agent_selfcheck.py --compile-only --scale-probe --capsules "
        f"{','.join(policy['capsules'])} --timeout {policy['timeout_s']}`. "
        "The harness supplies these limits; requests cannot relax them. The selected ladder also runs "
        "before background capsule simulations and publishes `qa/public_codegen_resources.json` with its "
        "candidate digest. Resource acceptance is separate from object compilation, semantic placement, "
        "numerical and trace qualification."
    )


def render_resource_block(
    options, *, environ=None, manager_cpus=None, grade_workers=None, boundary=None, device_ir_budget=None,
    public_codegen_policy=None,
) -> str:
    environment = os.environ if environ is None else environ
    manager = tuple(sorted(os.sched_getaffinity(0))) if manager_cpus is None else tuple(manager_cpus)
    author = parse_roster(environment.get("MERLIN_CODEX_CPUS", ""))
    shared = shared_compute_enabled(environment)
    compute = grading_cpus(environment)
    preparation = preparation_cpus(environment)
    if boundary is None:
        from merlin_experiments.phase2.external_resources import active_resource_boundary

        boundary = active_resource_boundary()
    if boundary is not None:
        boundary.verify()
        if not set((*manager, *author, *compute, *preparation)) <= set(boundary.cpu_ids):
            raise ValueError("task CPU selections exceed the verified resource boundary")
    if grade_workers is None:
        from merlin.targetgen.capsule_grade import default_grade_workers

        grade_workers = default_grade_workers()
    workers = min(grade_workers, len(compute)) if compute else grade_workers
    native = getattr(options, "sim_max_jobs", 0)
    helpers = getattr(options, "codex_max_subagents", None)
    object_budget = getattr(options, "public_object_build_budget_s", 0)
    if type(object_budget) is not int or not 0 <= object_budget <= 120:
        raise ValueError("public object-build prompt requires the selected 0..120-second budget")

    def roster(values):
        return ",".join(map(str, values)) if values else "inherited allocation"

    lines = [
        "\n\n## Resource allocation and scheduling\n",
        f"- Manager CPU affinity observed at task staging: `{roster(manager)}`.",
        f"- Configured Codex author/helper CPU pool: `{roster(author)}`. "
        "Helpers share this pool with the author; they do not receive additional CPUs.",
        f"- Unleased preparation CPU affinity: `{roster(preparation)}`. "
        "Build and simulator steps use their own explicit CPU leases.",
        f"- Configured {'shared compute' if shared else 'grading'} CPU pool: `{roster(compute)}`. "
        + (
            "CPU-intensive preparation, compilation and simulation steps acquire per-core leases here."
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
        f"- Submit independent cases with the documented selfcheck `--workers {workers}` option. "
        "Once the harness acknowledges its immutable snapshot, continue authoring. Results belong to "
        "that captured revision; do not mutate a source while it is being captured.",
        "- Obtain focused L2 feedback first, then use immediate automatic L3 promotion or a documented "
        "full required-tier check. Whole-model requests need their same-request operator gate evidence "
        "or a harness-issued continuation; a model-only request cannot supply that evidence.",
        "- Keep one full-cohort request in flight for a compiler snapshot. Use its incremental progress "
        "instead of submitting a duplicate full sweep; the harness yields advisory background work "
        "to an authorized full selfcheck of the same source and cohort.",
        "- A completed full public grade already includes the source-matched offload census from its "
        "graded artifacts. Read that result first; a standalone offload-census request recompiles the "
        "cohort and is only useful when the grade has no usable emission evidence. Model capstones use "
        "their separate placement gate, and host-only slices use provided-host qualification.",
        "- For a repeated native timeout, request native progress/liveness diagnostics from the experiment "
        "operator before another long retry. A requested timeout does not override an executor's lower limit. "
        "A timeout remains unqualified, without a numerical correctness verdict.",
        "- Keep builds and simulations on the documented harness routes. Reuse only evidence/cache hits "
        "whose source, executable, owner and instrument identity the harness authorizes.",
    ]
    lines.extend([
        "- Before expensive simulation or full-model builds, stress one or two original public members: "
        "`python3 agent_selfcheck.py --compile-only --capsules PUBLIC_ID_1,PUBLIC_ID_2 --timeout 120`. "
        "Replace those IDs with declared public member IDs; keep this diagnostic small. "
        "The harness supplies the original source-only scale ladder and frozen compiler owner. "
        "No captured tensor values, reference outputs or private model shapes are inputs.",
        "- General large/unseen dimensions should usually change runtime loop bounds rather than expand "
        "static instructions per tensor element. Check emitted IR growth and the selected build limits "
        "early. Preserve all semantics, placement and outputs; a small artifact or compile-only result "
        "does not pass functional, placement or required trace gates.",
    ])
    from .feedback import compile_only

    if getattr(compile_only, "SCALE_PROBE_FLAG", None) == "--scale-probe":
        lines.append(
            "- Obtain cheaper emission-only feedback first: "
            "`python3 agent_selfcheck.py --compile-only --scale-probe --capsules PUBLIC_ID_1 --timeout 120`. "
            "This stops after actual emitted LLVM and reports size/CFG growth; it does not build, link or simulate."
        )
    if object_budget:
        from .feedback.codegen_scalability import _MAX_BUILD_PROBE_BYTES

        lines.append(
            f"- Selected public object-build advisory budget: **{object_budget} seconds per sample**. "
            f"Its actual pre-parse emitted-source cap is **{_MAX_BUILD_PROBE_BYTES:,} bytes**. "
            "This probe-local cap is separate from any selected device-build IR budget. "
            "Budget refusals and timeouts remain unqualified operational results."
        )
    else:
        lines.append("- Public object-build advisory probes are unselected; current scalability feedback is emit-only.")
    if device_ir_budget is not None:
        from merlin.llvmlower.device_ir_scalability import DeviceIRBudget

        if type(device_ir_budget) is not DeviceIRBudget:
            raise ValueError("task device IR limits require the exact operator-selected budget")
        selected = device_ir_budget.record()
        limits = ", ".join(f"{name}={value}" for name, value in selected.items()
                           if name.startswith("max_") and value is not None)
        lines.append(f"- Selected device IR budget mode: `{selected['mode']}`; limits: `{limits or 'none selected'}`. "
                     "These limits constrain build resources and issue no correctness or execution credit.")
    if public_codegen_policy is not None:
        lines.append(_public_codegen_policy_line(public_codegen_policy))
    return "\n".join(lines) + "\n"


def append_resource_guidance(ws: Path, run_dir: Path, options, *, device_ir_budget=None,
                             public_codegen_policy=None, task_staging_config=None) -> None:
    """Append before provenance, including when a treatment supplied its own task."""
    task = Path(ws) / "TASK.md"
    if task.read_text().startswith(
        "# Phase 1: Gemmini compiler for complete models"  # target-ok: selected Gemmini full-model Phase 1 path
    ):
        from . import joint_control

        joint = joint_control.selected() is not None
        if joint and task_staging_config is None:
            raise ValueError("joint resource guidance requires verified task staging")
        selected_block = ""
        if task_staging_config is not None:
            from merlin.targetgen.generate_prompt import shared_pass_prompt_block

            from .task_staging import TaskStagingConfig, _shared_pass_record, verify_shared_pass_prompt

            if type(task_staging_config) is not TaskStagingConfig:
                raise ValueError("resource guidance requires the original task staging configuration")
            requirement = verify_shared_pass_prompt(task_staging_config, ws, run_dir)
            if requirement is not None and requirement["schema"] == "merlin.shared_pass_requirement.v2":
                selected_block = shared_pass_prompt_block(_shared_pass_record(task_staging_config, run_dir))
                if task.read_text().count(selected_block) != 1:
                    raise ValueError("selected shared-pass task block changed before resource staging")
        if public_codegen_policy is not None:
            task.write_text(task.read_text() + "\n\n## Selected public compilation resources\n\n"
                            + _public_codegen_policy_line(public_codegen_policy) + "\n")
        # The joint task includes verified scalability and campaign instructions
        # in addition to the compact legacy task and selected pass contract.
        # Keep an explicit bounded budget, without truncating required guidance.
        if joint:
            task.write_text(task.read_text() + render_joint_resource_block(options) + token_guidance(options))
        limit = 64 * 1024 if joint else 12 * 1024
        if task.stat().st_size - len(selected_block.encode("utf-8")) > limit:
            raise RuntimeError(f"full-model TASK.md exceeds {limit // 1024} KB")
        (Path(run_dir) / "TASK.md").write_bytes(task.read_bytes())
        return
    selected = {} if device_ir_budget is None else {"device_ir_budget": device_ir_budget}
    if public_codegen_policy is not None:
        selected["public_codegen_policy"] = public_codegen_policy
    task.write_text(task.read_text() + render_resource_block(options, **selected) + token_guidance(options))
    (Path(run_dir) / "TASK.md").write_bytes(task.read_bytes())


def render_joint_resource_block(options) -> str:
    """Describe actual native admission classes, not a fictitious user knob."""
    compute = grading_cpus()
    author = parse_roster(os.environ.get("MERLIN_CODEX_CPUS", ""))
    return (
        "\n\n## Joint shared CPU scheduling\n\n"
        f"Author/helper affinity: `{','.join(map(str, author))}`; "
        f"shared compute affinity: `{','.join(map(str, compute))}`. "
        "Builds, Spike, gSIM and preparation may use any available core in the shared pool; "
        "there is no core partition by engine. Expensive harness steps acquire per-core leases. "
        "Author processes share their affinity; helpers receive no additional cores. "
        f"Native simulation concurrency ceiling: {getattr(options, 'sim_max_jobs', 0)}; "
        f"author/helper ceiling: {getattr(options, 'codex_max_subagents', None)}. "
        "These are ceilings, not counts of running work. The separate gSIM witness and "
        "two-job limits still apply.\n\n"
        "The native scheduler serves author-requested checks first (priority 1), "
        "background grading second (2), preparation third (3), and advisory probes last (4). "
        "Waiting jobs are FIFO within a class; running leases are not preempted. "
        "These classes are assigned by the harness, not a selfcheck priority flag. "
        "Prioritize critical work by requesting its focused check through the native broker; "
        "submit independent cases to rolling admission, reuse exact-input caches and avoid duplicate full sweeps. "
        "Continue compiler development after immutable snapshot acknowledgment; consume each finished result "
        "without waiting for unrelated slow cases. "
        "Keep heavy work on the documented leased harness routes, not unmanaged shell jobs.\n"
    )
