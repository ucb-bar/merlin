from __future__ import annotations

import json
import re
import textwrap
from dataclasses import asdict
from pathlib import Path

import yaml

from .model import (
    ExecutionBundle,
    GeneratedScaffoldFile,
    GenerationBundle,
    MutationBundle,
    MutationCandidate,
    SupportPlan,
    TargetCapabilities,
    TaskNode,
)


def emit_generation_artifacts(
    *,
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task_graph: list[TaskNode],
    target_dir: Path,
    compile_view: dict,
    deployment_view: dict | None,
) -> GenerationBundle:
    generated_root = target_dir / "generated" / "tree"
    generated_root.mkdir(parents=True, exist_ok=True)
    inputs_dir = target_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    raw_specs = _build_scaffold_specs(
        capabilities=capabilities,
        task_graph=task_graph,
        compile_view=compile_view,
        deployment_view=deployment_view,
    )
    specs = _merge_duplicate_specs(raw_specs)

    files: list[GeneratedScaffoldFile] = []
    for spec in specs:
        repo_path = spec["repo_path"]
        staged_path = generated_root / repo_path
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(spec["content"], encoding="utf-8")
        files.append(
            GeneratedScaffoldFile(
                repo_path=repo_path,
                staged_path=str(staged_path),
                category=spec["category"],
                rationale=spec["rationale"],
                task_ids=spec["task_ids"],
            )
        )

    bundle = GenerationBundle(
        target=capabilities.identity.name,
        primary_integration=support_plan.primary_integration,
        integration_styles=support_plan.integration_styles,
        generated_root=str(generated_root),
        files=files,
    )

    _copy_input_snapshot(Path(capabilities.capability_path), inputs_dir / "capability.yaml")
    if capabilities.deployment is not None:
        _copy_input_snapshot(Path(capabilities.deployment.source_path), inputs_dir / "overlay.yaml")

    (target_dir / "generation_bundle.json").write_text(
        json.dumps(asdict(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "generation_summary.md").write_text(
        _render_generation_summary(
            capabilities=capabilities,
            support_plan=support_plan,
            files=files,
        ),
        encoding="utf-8",
    )
    return bundle


def emit_mutation_artifacts(
    *,
    target_dir: Path,
    generation_bundle: GenerationBundle,
    execution_bundle: ExecutionBundle,
) -> MutationBundle:
    mutation_dir = target_dir / "mutation"
    proposed_root = mutation_dir / "proposed_tree"
    proposed_root.mkdir(parents=True, exist_ok=True)

    task_by_id = {task.id: task for task in execution_bundle.tasks}
    candidates: list[MutationCandidate] = []
    staged_files_by_repo_path = {item.repo_path: item for item in generation_bundle.files}

    for repo_path, staged_file in staged_files_by_repo_path.items():
        matching_tasks = [
            task_by_id[task_id]
            for task_id in staged_file.task_ids
            if task_id in task_by_id and task_by_id[task_id].mutation_policy != "planner_generated_only"
        ]
        if not matching_tasks:
            continue
        chosen_task = matching_tasks[0]
        source_path = Path(staged_file.staged_path)
        proposed_path = proposed_root / repo_path
        proposed_path.parent.mkdir(parents=True, exist_ok=True)
        proposed_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        candidates.append(
            MutationCandidate(
                repo_path=repo_path,
                staged_path=str(proposed_path),
                action="create_or_edit",
                rationale=staged_file.rationale,
                task_ids=staged_file.task_ids,
                mutation_policy=chosen_task.mutation_policy,
            )
        )

    branch_name = f"targetgen/{execution_bundle.target}-staged"
    worktree_root = mutation_dir / "worktree_root"
    bundle = MutationBundle(
        target=execution_bundle.target,
        branch_name=branch_name,
        worktree_root=str(worktree_root),
        generated_root=generation_bundle.generated_root,
        proposed_root=str(proposed_root),
        candidates=candidates,
        blocking_gates=[
            "branch_switch_required",
            "manual_review_required",
            "validation_not_run",
        ],
        next_steps=[
            "Review generation_summary.md and mutation/proposal_brief.md before applying any change.",
            "Create an isolated worktree or branch before enabling live mutation.",
            "Promote only the staged files that still match the target capability spec and support plan.",
            "Run validation after promotion; this staging phase intentionally does not run tests.",
        ],
    )
    mutation_dir.mkdir(parents=True, exist_ok=True)
    (mutation_dir / "mutation_bundle.json").write_text(
        json.dumps(asdict(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (mutation_dir / "proposal_brief.md").write_text(
        _render_mutation_summary(bundle),
        encoding="utf-8",
    )
    (mutation_dir / "worktree_plan.md").write_text(
        _render_worktree_plan(bundle),
        encoding="utf-8",
    )
    return bundle


def _build_scaffold_specs(
    *,
    capabilities: TargetCapabilities,
    task_graph: list[TaskNode],
    compile_view: dict,
    deployment_view: dict | None,
) -> list[dict[str, object]]:
    target_name = capabilities.identity.name
    task_ids = {task.id for task in task_graph}
    repo_specs: list[dict[str, object]] = []
    compile_task_id = "derive_compile_view"
    if compile_task_id in task_ids:
        repo_specs.append(
            {
                "repo_path": f"models/{target_name}.yaml",
                "category": "compile_view",
                "rationale": "Derived compile-target view staged from the canonical capability spec.",
                "task_ids": [compile_task_id],
                "content": _render_yaml_scaffold(
                    title="Derived compile-target view",
                    origin=capabilities.capability_path,
                    payload=compile_view,
                ),
            }
        )

    if deployment_view is not None and capabilities.deployment is not None:
        deployment_task_id = "derive_deployment_overlay"
        if deployment_task_id in task_ids:
            recipe_path = capabilities.deployment.hardware_recipe or f"build_tools/hardware/{target_name}.yaml"
            repo_specs.append(
                {
                    "repo_path": recipe_path,
                    "category": "deployment_view",
                    "rationale": "Derived deployment and hardware-backend view staged from the deployment overlay.",
                    "task_ids": [deployment_task_id],
                    "content": _render_yaml_scaffold(
                        title="Derived deployment view",
                        origin=capabilities.deployment.source_path,
                        payload=deployment_view,
                    ),
                }
            )

    for task in task_graph:
        if task.family == "shared":
            continue
        repo_specs.extend(
            _family_scaffold_specs(
                capabilities=capabilities,
                task=task,
            )
        )
    return repo_specs


def _family_scaffold_specs(
    *,
    capabilities: TargetCapabilities,
    task: TaskNode,
) -> list[dict[str, object]]:
    target_name = capabilities.identity.name
    target_title = _pascal_case(target_name)
    target_lower = target_name.replace("-", "_")
    if task.family == "post_global_plugin":
        return [
            _text_spec(
                repo_path=f"compiler/plugins/target/{target_title}/CMakeLists.txt",
                category="plugin_scaffold",
                rationale="Post-global-optimization plugin registration and build entrypoint.",
                task=task,
                content=_render_cmake_scaffold(
                    target_title=target_title,
                    task=task,
                    lines=[
                        f"add_subdirectory({target_title})",
                        "",
                        f"# TODO: register the {target_title} plugin target and its pass library.",
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"compiler/plugins/target/{target_title}/PluginRegistration.cpp",
                category="plugin_scaffold",
                rationale="Plugin registration scaffold for TargetGen-generated post-global-opt work.",
                task=task,
                content=_render_cpp_scaffold(
                    target_title=target_title,
                    task=task,
                    body=f"""
                    namespace merlin {{
                    namespace {target_title} {{

                    // TODO: register passes, dialects, and post-global-opt hooks.
                    void register{target_title}Plugin() {{}}

                    }}  // namespace {target_title}
                    }}  // namespace merlin
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"compiler/plugins/target/{target_title}/Options.h",
                category="plugin_scaffold",
                rationale="Plugin options surface derived from the capability spec.",
                task=task,
                content=_render_cpp_scaffold(
                    target_title=target_title,
                    task=task,
                    body=f"""
                    namespace merlin {{
                    namespace {target_title} {{

                    struct {target_title}Options {{
                      // TODO: add planner-derived CLI or pass options.
                    }};

                    }}  // namespace {target_title}
                    }}  // namespace merlin
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"compiler/plugins/target/{target_title}/Options.cpp",
                category="plugin_scaffold",
                rationale="Out-of-line option plumbing for the target plugin.",
                task=task,
                content=_render_cpp_scaffold(
                    target_title=target_title,
                    task=task,
                    body="""
                    // TODO: define option parsing or registration when this scaffold is promoted.
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"compiler/src/merlin/Dialect/{target_title}/README.md",
                category="dialect_scaffold",
                rationale="Dialect-facing recovery notes for the target family.",
                task=task,
                content=_render_markdown_scaffold(
                    title=f"{target_title} Dialect Scaffold",
                    task=task,
                    bullets=[
                        "Recover target-native semantics immediately after global optimization.",
                        "Define fallback boundaries for unsupported patterns.",
                        "List the first dialect ops to add and the passes that materialize them.",
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"compiler/src/merlin/Dialect/{target_title}/Transforms/CMakeLists.txt",
                category="dialect_scaffold",
                rationale="Transform library entrypoint for target recovery passes.",
                task=task,
                content=_render_cmake_scaffold(
                    target_title=target_title,
                    task=task,
                    lines=[
                        f"# TODO: add the {target_title} transform library once passes are implemented.",
                    ],
                ),
            ),
        ]
    if task.family == "structured_text_isa":
        return [
            _text_spec(
                repo_path=f"compiler/src/merlin/Dialect/{target_title}/README.md",
                category="dialect_scaffold",
                rationale="Kernel, schedule, and ISA stage notes for the structured text ISA flow.",
                task=task,
                content=_render_markdown_scaffold(
                    title=f"{target_title} Structured ISA Pipeline",
                    task=task,
                    bullets=[
                        "Separate kernel, schedule, and ISA responsibilities explicitly.",
                        "Call out verifier and memory-planning stages before late export.",
                        "Record the expected textual or structured ISA sink.",
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"compiler/src/merlin/Dialect/{target_title}/IR/{target_title}Ops.td",
                category="dialect_scaffold",
                rationale="Initial operation definition surface for the structured ISA family.",
                task=task,
                content=_render_tablegen_scaffold(
                    target_title=target_title,
                    task=task,
                    body=f"""
                    // TODO: define kernel, schedule, and ISA-stage ops for {target_title}.
                    include "mlir/IR/OpBase.td"
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"compiler/src/merlin/Dialect/{target_title}/Transforms/CMakeLists.txt",
                category="dialect_scaffold",
                rationale="Transform library entrypoint for structured ISA lowering.",
                task=task,
                content=_render_cmake_scaffold(
                    target_title=target_title,
                    task=task,
                    lines=[
                        f"# TODO: add {target_title} kernel/schedule lowering targets.",
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"compiler/src/merlin/Translation/{target_title}/CMakeLists.txt",
                category="translation_scaffold",
                rationale="Exporter entrypoint for the target ISA text or packet format.",
                task=task,
                content=_render_cmake_scaffold(
                    target_title=target_title,
                    task=task,
                    lines=[
                        f"# TODO: add the {target_title} translation/export target.",
                    ],
                ),
            ),
        ]
    if task.family == "runtime_hal":
        return [
            _text_spec(
                repo_path=f"runtime/src/iree/hal/drivers/{target_lower}/README.md",
                category="runtime_scaffold",
                rationale="Runtime and device contract notes for the HAL backend.",
                task=task,
                content=_render_markdown_scaffold(
                    title=f"{target_title} HAL Driver Scaffold",
                    task=task,
                    bullets=[
                        "Describe driver registration, device creation, and executable contract separately.",
                        "Keep queue, synchronization, and transport responsibilities explicit.",
                        "Record smoke-test assumptions and deployment prerequisites before implementation.",
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"runtime/src/iree/hal/drivers/{target_lower}/api.h",
                category="runtime_scaffold",
                rationale="Driver-facing public API surface for the staged HAL backend.",
                task=task,
                content=_render_c_header_scaffold(
                    target_upper=_macro_case(target_name),
                    task=task,
                    declarations=[
                        (
                            f"iree_status_t iree_hal_{target_lower}_driver_module_register("
                            "iree_hal_driver_registry_t* registry);"
                        ),
                    ],
                ),
            ),
            _text_spec(
                repo_path=f"runtime/src/iree/hal/drivers/{target_lower}/driver.c",
                category="runtime_scaffold",
                rationale="Driver registration and enumeration entrypoint.",
                task=task,
                content=_render_c_scaffold(
                    target_title=target_title,
                    task=task,
                    body="""
                    // TODO: implement driver registration, device enumeration, and capability reporting.
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"runtime/src/iree/hal/drivers/{target_lower}/device.c",
                category="runtime_scaffold",
                rationale="Device model and executable contract staging file.",
                task=task,
                content=_render_c_scaffold(
                    target_title=target_title,
                    task=task,
                    body="""
                    // TODO: implement device creation, executable loading, queueing, and synchronization.
                    """,
                ),
            ),
            _text_spec(
                repo_path=f"runtime/src/iree/hal/drivers/{target_lower}/registration/driver_module.c",
                category="runtime_scaffold",
                rationale="Driver module registration hook for Merlin packaging and tests.",
                task=task,
                content=_render_c_scaffold(
                    target_title=target_title,
                    task=task,
                    body="""
                    // TODO: wire the driver module into Merlin and IREE runtime registration.
                    """,
                ),
            ),
        ]
    isa_prefix = _llvm_intrinsics_prefix(capabilities)
    target_dir = _llvm_target_dir(capabilities)
    return [
        _text_spec(
            repo_path=("third_party/iree_bar/compiler/src/iree/compiler/Codegen/" f"{target_title}TargetGen.md"),
            category="llvm_ukernel_scaffold",
            rationale="Planner-derived lowering and tiling design note for the LLVM/IREE path.",
            task=task,
            content=_render_markdown_scaffold(
                title=f"{target_title} LLVM/Ukernel Scaffold",
                task=task,
                bullets=[
                    "Record tile selection, packing, and datatype legality decisions.",
                    "List the IREE codegen seams and lowering strategy hooks that must change.",
                    "Keep the LLVM exposure and ukernel registration story aligned in one place.",
                ],
            ),
        ),
        _text_spec(
            repo_path=("third_party/iree_bar/runtime/src/iree/builtins/ukernel/" f"{target_lower}_targetgen.md"),
            category="llvm_ukernel_scaffold",
            rationale="Ukernel registration and packing/unpacking notes for the target.",
            task=task,
            content=_render_markdown_scaffold(
                title=f"{target_title} Ukernel Notes",
                task=task,
                bullets=[
                    "Capture preferred tile families and pack/unpack requirements.",
                    "Record datatype triples and layout assumptions from the hardware spec.",
                    "Keep benchmark and verification expectations adjacent to the ukernel plan.",
                ],
            ),
        ),
        _text_spec(
            repo_path=(
                "third_party/iree_bar/third_party/llvm-project/llvm/include/llvm/IR/"
                f"Intrinsics{isa_prefix}{target_title}.td"
            ),
            category="llvm_ukernel_scaffold",
            rationale="Intrinsic staging surface for target ISA exposure.",
            task=task,
            content=_render_tablegen_scaffold(
                target_title=target_title,
                task=task,
                body=f"""
                // TODO: stage intrinsic declarations for {target_title}.
                let TargetPrefix = "{capabilities.isa.base}" in {{
                  // Add planner-derived intrinsic declarations here.
                }}
                """,
            ),
        ),
        _text_spec(
            repo_path=(
                "third_party/iree_bar/third_party/llvm-project/llvm/lib/Target/"
                f"{target_dir}/{target_title}InstrInfo.td"
            ),
            category="llvm_ukernel_scaffold",
            rationale="Instruction-pattern staging surface for LLVM backend changes.",
            task=task,
            content=_render_tablegen_scaffold(
                target_title=target_title,
                task=task,
                body=f"""
                // TODO: stage instruction patterns, feature bits, and legalization notes for {target_title}.
                """,
            ),
        ),
    ]


def _text_spec(
    *,
    repo_path: str,
    category: str,
    rationale: str,
    task: TaskNode,
    content: str,
) -> dict[str, object]:
    return {
        "repo_path": repo_path,
        "category": category,
        "rationale": rationale,
        "task_ids": [task.id],
        "content": content,
    }


def _merge_duplicate_specs(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for spec in specs:
        repo_path = str(spec["repo_path"])
        if repo_path not in merged:
            merged[repo_path] = {
                **spec,
                "task_ids": list(spec["task_ids"]),
            }
            continue
        current = merged[repo_path]
        current["task_ids"] = list(dict.fromkeys([*current["task_ids"], *spec["task_ids"]]))
        if spec["rationale"] not in current["rationale"]:
            current["rationale"] = f"{current['rationale']} {spec['rationale']}"
        if spec["content"] != current["content"]:
            if repo_path.endswith(".md"):
                current["content"] = (
                    f"{current['content'].rstrip()}\n\n"
                    "<!-- Additional TargetGen staged content from a second integration family. -->\n\n"
                    f"{spec['content']}"
                )
    return list(merged.values())


def _render_generation_summary(
    *,
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    files: list[GeneratedScaffoldFile],
) -> str:
    lines = [
        "# TargetGen Generation Summary",
        "",
        f"- Target: `{capabilities.identity.name}`",
        f"- Display name: `{capabilities.identity.display_name}`",
        f"- Primary integration: `{support_plan.primary_integration}`",
        f"- Integration styles: {', '.join(f'`{style}`' for style in support_plan.integration_styles)}",
        f"- Generated scaffold count: `{len(files)}`",
        "",
        "## What This Is",
        "",
        "This directory is a non-live staging area. The generated files mirror likely repo",
        "paths, but they are not applied to repo-tracked source trees and they are not",
        "validated yet.",
        "",
        "## Staged Repo Paths",
        "",
    ]
    for item in files:
        lines.append(
            f"- `{item.repo_path}`: {item.category} for task(s) {', '.join(f'`{task}`' for task in item.task_ids)}"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            "- Capability spec snapshot: `inputs/capability.yaml`",
            "- Deployment overlay snapshot: `inputs/overlay.yaml` when an overlay was provided",
            "",
            "## Next Step",
            "",
            "Run `merlin targetgen stage-mutation ...` to assemble a proposed tree for",
            "review before any branch switch or live mutation.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_mutation_summary(bundle: MutationBundle) -> str:
    lines = [
        "# TargetGen Mutation Staging",
        "",
        f"- Target: `{bundle.target}`",
        f"- Proposed branch: `{bundle.branch_name}`",
        f"- Proposed worktree root: `{bundle.worktree_root}`",
        f"- Candidate count: `{len(bundle.candidates)}`",
        "",
        "## Blocking Gates",
        "",
    ]
    for gate in bundle.blocking_gates:
        lines.append(f"- `{gate}`")
    lines.extend(
        [
            "",
            "## Proposed Candidates",
            "",
        ]
    )
    for candidate in bundle.candidates:
        lines.append(
            f"- `{candidate.repo_path}` via `{candidate.mutation_policy}` for task(s) "
            f"{', '.join(f'`{task}`' for task in candidate.task_ids)}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The proposed tree mirrors candidate repo paths under `mutation/proposed_tree/`.",
            "Nothing in this staging phase modifies repo-tracked files.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_worktree_plan(bundle: MutationBundle) -> str:
    lines = [
        "# Worktree Plan",
        "",
        "This file is advisory. It describes how a later live-mutation phase should be",
        "prepared once the staged proposals are accepted.",
        "",
        f"- Branch name: `{bundle.branch_name}`",
        f"- Suggested worktree root: `{bundle.worktree_root}`",
        f"- Proposed tree root: `{bundle.proposed_root}`",
        "",
        "## Recommended Sequence",
        "",
    ]
    for step in bundle.next_steps:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def _render_yaml_scaffold(*, title: str, origin: str, payload: dict) -> str:
    header = [
        f"# {title}",
        "# Generated by Merlin TargetGen.",
        "# This file is staged under build/generated/targetgen and has not been promoted.",
        f"# Source: {origin}",
        "",
    ]
    return "\n".join(header) + yaml.safe_dump(payload, sort_keys=False)


def _render_markdown_scaffold(*, title: str, task: TaskNode, bullets: list[str]) -> str:
    lines = [
        f"# {title}",
        "",
        "> Generated by Merlin TargetGen. This is a staged scaffold, not a promoted source file.",
        "",
        f"- Task: `{task.id}`",
        f"- Phase: `{task.phase}`",
        f"- Family: `{task.family}`",
        f"- Repo root: `{task.repo_root}`",
        "",
        "## Focus",
        "",
    ]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.extend(
        [
            "",
            "## Acceptance To Close Before Promotion",
            "",
        ]
    )
    for check in task.acceptance_checks:
        lines.append(f"- {check}")
    lines.append("")
    return "\n".join(lines)


def _render_cmake_scaffold(*, target_title: str, task: TaskNode, lines: list[str]) -> str:
    body = "\n".join(lines)
    return textwrap.dedent(
        f"""\
        # Generated by Merlin TargetGen for {target_title}.
        # Task: {task.id}
        # This scaffold is staged only and is not part of the live build yet.

        {body}
        """
    )


def _render_cpp_scaffold(*, target_title: str, task: TaskNode, body: str) -> str:
    return (
        textwrap.dedent(
            f"""\
        // Generated by Merlin TargetGen for {target_title}.
        // Task: {task.id}
        // This file is staged only and has not been compiled or validated yet.

        {textwrap.dedent(body).strip()}
        """
        ).rstrip()
        + "\n"
    )


def _render_c_scaffold(*, target_title: str, task: TaskNode, body: str) -> str:
    return (
        textwrap.dedent(
            f"""\
        // Generated by Merlin TargetGen for {target_title}.
        // Task: {task.id}
        // This file is staged only and has not been compiled or validated yet.

        {textwrap.dedent(body).strip()}
        """
        ).rstrip()
        + "\n"
    )


def _render_c_header_scaffold(
    *,
    target_upper: str,
    task: TaskNode,
    declarations: list[str],
) -> str:
    body = "\n".join(declarations)
    return textwrap.dedent(
        f"""\
        #ifndef MERLIN_TARGETGEN_{target_upper}_API_H_
        #define MERLIN_TARGETGEN_{target_upper}_API_H_

        // Generated by Merlin TargetGen.
        // Task: {task.id}
        // This header is staged only and has not been compiled or validated yet.

        {body}

        #endif  // MERLIN_TARGETGEN_{target_upper}_API_H_
        """
    )


def _render_tablegen_scaffold(*, target_title: str, task: TaskNode, body: str) -> str:
    return (
        textwrap.dedent(
            f"""\
        // Generated by Merlin TargetGen for {target_title}.
        // Task: {task.id}
        // This file is staged only and has not been compiled or validated yet.

        {textwrap.dedent(body).strip()}
        """
        ).rstrip()
        + "\n"
    )


def _copy_input_snapshot(source: Path, destination: Path) -> None:
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\\-]+", value) if part)


def _macro_case(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).upper()


def _llvm_intrinsics_prefix(capabilities: TargetCapabilities) -> str:
    if capabilities.isa.base.startswith("riscv"):
        return "RISCVX"
    return _pascal_case(capabilities.isa.base)


def _llvm_target_dir(capabilities: TargetCapabilities) -> str:
    if capabilities.isa.base.startswith("riscv"):
        return "RISCV"
    return _pascal_case(capabilities.isa.base)
