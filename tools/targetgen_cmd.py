#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml
from raycp import normalize_state_root, submit_job
from targetgen import (
    answer_operator_request,
    build_execution_bundle,
    build_execution_state,
    build_support_plan,
    build_task_graph,
    emit_generation_artifacts,
    emit_mutation_artifacts,
    execute_bundle,
    load_capability_spec,
    load_deployment_overlay,
    load_execution_environment_from_dir,
    load_provider_config,
    persist_execution_environment,
    prepare_execution_environment,
    render_execution_status,
    render_explain_text,
    render_prompt_packets,
    render_task_briefs,
)

import utils


def setup_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="targetgen_command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate TargetGen capability specs and overlays",
    )
    _add_common_args(validate_parser)
    validate_parser.set_defaults(_handler=cmd_validate)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Emit support-plan and task-graph artifacts",
    )
    _add_common_args(plan_parser)
    plan_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Output directory for generated planner artifacts",
    )
    plan_parser.set_defaults(_handler=cmd_plan)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Emit non-live scaffold files under build/generated/targetgen without touching repo-tracked sources",
    )
    _add_common_args(generate_parser)
    generate_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Output directory for generated scaffold artifacts",
    )
    generate_parser.set_defaults(_handler=cmd_generate)

    explain_parser = subparsers.add_parser(
        "explain",
        help="Print a human-readable TargetGen explanation",
    )
    _add_common_args(explain_parser)
    explain_parser.set_defaults(_handler=cmd_explain)

    orchestrate_parser = subparsers.add_parser(
        "orchestrate",
        help="Emit an execution bundle and LLM-oriented task briefs from the task graph",
    )
    _add_common_args(orchestrate_parser)
    orchestrate_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Output directory for execution-bundle artifacts",
    )
    orchestrate_parser.add_argument(
        "--prompt-backend",
        choices=["none", "manualllm", "provider"],
        default="manualllm",
        help="Prompt packet backend to prepare for orchestration output",
    )
    orchestrate_parser.add_argument(
        "--agent",
        help="Optional mlirAgent provider config name to attach to prompt packets",
    )
    orchestrate_parser.add_argument(
        "--prompts-dir",
        help="Optional output directory for prompt_NNN.md packets; defaults to <out-dir>/<target>/prompts",
    )
    orchestrate_parser.set_defaults(_handler=cmd_orchestrate)

    execute_parser = subparsers.add_parser(
        "execute",
        help="Advance execution state, emit prompts, ingest responses, and stop on operator gates",
    )
    execute_parser.add_argument(
        "capability",
        nargs="?",
        help="Path to a canonical TargetGen capability spec",
    )
    execute_parser.add_argument(
        "--overlay",
        help="Optional deployment overlay that augments the capability spec",
    )
    execute_parser.add_argument(
        "--from-dir",
        help="Existing target output directory to resume from",
    )
    execute_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Base output directory for execution artifacts",
    )
    execute_parser.add_argument(
        "--prompt-backend",
        choices=["none", "manualllm", "provider"],
        default="manualllm",
        help="Prompt packet backend to use for execution",
    )
    execute_parser.add_argument(
        "--agent",
        help="Optional mlirAgent provider config name to attach to prompt packets",
    )
    execute_parser.add_argument(
        "--prompts-dir",
        help="Optional output directory for prompt_NNN.md packets; defaults to <out-dir>/<target>/prompts",
    )
    execute_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing execution_state.json when present",
    )
    execute_parser.add_argument(
        "--engine",
        choices=["local", "ray"],
        default="local",
        help=(
            "Execution backend. `local` runs the in-process executor, `ray` submits "
            "the existing local executor as a Ray job."
        ),
    )
    execute_parser.add_argument(
        "--ray-state-root",
        default="build/generated/ray",
        help="Merlin-owned state directory for Ray cluster and run metadata when --engine ray is used.",
    )
    execute_parser.set_defaults(_handler=cmd_execute)

    stage_parser = subparsers.add_parser(
        "stage-mutation",
        help="Stage a proposed mutation tree under build/generated/targetgen without applying repo-tracked edits",
    )
    stage_parser.add_argument(
        "capability",
        nargs="?",
        help="Path to a canonical TargetGen capability spec",
    )
    stage_parser.add_argument(
        "--overlay",
        help="Optional deployment overlay that augments the capability spec",
    )
    stage_parser.add_argument(
        "--from-dir",
        help="Existing generated TargetGen target directory with inputs snapshots",
    )
    stage_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Base output directory for generation and mutation artifacts",
    )
    stage_parser.set_defaults(_handler=cmd_stage_mutation)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Record an operator choice for an open executor request",
    )
    answer_parser.add_argument(
        "--target-dir",
        required=True,
        help="Absolute or repo-relative path to a generated TargetGen target directory",
    )
    answer_parser.add_argument(
        "--question-id",
        required=True,
        help="Stable operator request id to answer",
    )
    answer_parser.add_argument(
        "--choice",
        required=True,
        help="Chosen option id from the operator request",
    )
    answer_parser.set_defaults(_handler=cmd_answer)

    status_parser = subparsers.add_parser(
        "status",
        help="Show executor task states and open operator requests",
    )
    status_parser.add_argument(
        "--target-dir",
        required=True,
        help="Absolute or repo-relative path to a generated TargetGen target directory",
    )
    status_parser.set_defaults(_handler=cmd_status)


def main(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    if handler is None:
        return 2
    return int(handler(args))


def cmd_validate(args: argparse.Namespace) -> int:
    capabilities = load_capability_spec(args.capability)
    overlay = load_deployment_overlay(args.overlay) if args.overlay else None
    if overlay is not None:
        capabilities.deployment = overlay
    print(f"Validated capability spec: {capabilities.identity.name}")
    if overlay is not None:
        print(f"Validated deployment overlay: {overlay.name}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    capabilities = _load_inputs(args)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    out_dir = _target_out_dir(args.out_dir, capabilities.identity.name)
    compile_view = _build_compile_view(capabilities, support_plan.integration_styles)
    deployment_view = _build_deployment_view(capabilities) if capabilities.deployment is not None else None
    if not getattr(args, "dry_run", False):
        _write_plan_artifacts(
            out_dir=out_dir,
            support_plan=support_plan,
            task_graph=task_graph,
            compile_view=compile_view,
            deployment_view=deployment_view,
        )
    print(f"Planned target: {support_plan.target}")
    print(f"Primary integration: {support_plan.primary_integration}")
    print(f"Task count: {len(task_graph)}")
    print(f"Output directory: {out_dir}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    capabilities = _load_inputs(args)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    target_dir = _target_out_dir(args.out_dir, capabilities.identity.name)
    compile_view = _build_compile_view(capabilities, support_plan.integration_styles)
    deployment_view = _build_deployment_view(capabilities) if capabilities.deployment is not None else None
    if not getattr(args, "dry_run", False):
        _write_plan_artifacts(
            out_dir=target_dir,
            support_plan=support_plan,
            task_graph=task_graph,
            compile_view=compile_view,
            deployment_view=deployment_view,
        )
        generation_bundle = emit_generation_artifacts(
            capabilities=capabilities,
            support_plan=support_plan,
            task_graph=task_graph,
            target_dir=target_dir,
            compile_view=compile_view,
            deployment_view=deployment_view,
        )
    else:
        generation_bundle = None
    print(f"Generated target scaffold: {support_plan.target}")
    print(f"Primary integration: {support_plan.primary_integration}")
    if generation_bundle is not None:
        print(f"Generated scaffold files: {len(generation_bundle.files)}")
        print(f"Generated root: {generation_bundle.generated_root}")
        print(f"Generation summary: {target_dir / 'generation_summary.md'}")
    print(f"Output directory: {target_dir}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    capabilities = _load_inputs(args)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    print(render_explain_text(capabilities, support_plan, task_graph))
    return 0


def cmd_orchestrate(args: argparse.Namespace) -> int:
    capabilities = _load_inputs(args)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    out_dir = _target_out_dir(args.out_dir, capabilities.identity.name)
    briefs_dir = out_dir / "briefs"
    prompts_dir = _prompts_out_dir(args.prompts_dir, out_dir)
    agent = args.agent or ("codex" if args.prompt_backend == "provider" else None)
    provider_config = load_provider_config(agent) if args.prompt_backend == "provider" else {}
    bundle = build_execution_bundle(
        capabilities,
        support_plan,
        task_graph,
        prompt_backend=args.prompt_backend,
        agent=agent,
        provider_config=provider_config,
    )
    prompt_packets = {}
    if args.prompt_backend != "none":
        bundle.prompts_dir = str(prompts_dir)
        prompt_packets = render_prompt_packets(bundle)
    briefs = render_task_briefs(bundle)
    execution_state = build_execution_state(bundle)
    task_states_dir = out_dir / "task_states"
    if not getattr(args, "dry_run", False):
        out_dir.mkdir(parents=True, exist_ok=True)
        briefs_dir.mkdir(parents=True, exist_ok=True)
        task_states_dir.mkdir(parents=True, exist_ok=True)
        if args.prompt_backend != "none":
            prompts_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "execution_bundle.json", asdict(bundle))
        _write_json(out_dir / "execution_state.json", asdict(execution_state))
        if provider_config:
            _write_json(out_dir / "provider_backend.json", provider_config)
        for filename, content in briefs.items():
            (briefs_dir / filename).write_text(content, encoding="utf-8")
        for filename, content in prompt_packets.items():
            (prompts_dir / filename).write_text(content, encoding="utf-8")
        for task_id, state in execution_state.tasks.items():
            _write_json(task_states_dir / f"{task_id}.json", asdict(state))
    print(f"Prepared execution bundle for: {bundle.target}")
    print(f"Workflow: {bundle.workflow}")
    print(f"Prompt backend: {bundle.prompt_backend}")
    print(f"Brief count: {len(briefs)}")
    print(f"Execution state file: {out_dir / 'execution_state.json'}")
    if args.prompt_backend != "none":
        print(f"Prompt packet count: {len(prompt_packets)}")
        print(f"Prompts directory: {prompts_dir}")
    print(f"Output directory: {out_dir}")
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    target_dir = _resolve_execute_target_dir(args)
    if args.capability:
        capabilities = _load_execute_inputs(args, target_dir)
        prompts_dir = _prompts_out_dir(args.prompts_dir, target_dir)
        agent = args.agent or ("codex" if args.prompt_backend == "provider" else None)
        bundle, execution_state, briefs, prompt_packets, operator_requests = prepare_execution_environment(
            capabilities,
            out_dir=target_dir.parent,
            prompt_backend=args.prompt_backend,
            agent=agent,
            prompts_dir=prompts_dir,
            resume=args.resume or (target_dir / "execution_state.json").exists(),
        )
    else:
        bundle, execution_state, briefs, prompt_packets, operator_requests = load_execution_environment_from_dir(
            target_dir=target_dir
        )
    if args.engine == "ray":
        target_dir = persist_execution_environment(
            bundle,
            execution_state,
            out_dir=target_dir.parent,
            briefs=briefs,
            prompt_packets=prompt_packets,
            operator_requests=operator_requests,
        )
        run_record = submit_job(
            state_root=normalize_state_root(args.ray_state_root),
            target=bundle.target,
            workflow="targetgen_execute",
            source="targetgen_execute",
            command=[
                "conda",
                "run",
                "-n",
                "merlin-dev",
                "uv",
                "run",
                "tools/merlin.py",
                "targetgen",
                "execute",
                "--engine",
                "local",
                "--from-dir",
                str(target_dir),
                "--resume",
            ],
            target_dir=target_dir,
            metadata={
                "prompt_backend": bundle.prompt_backend,
                "workflow": bundle.workflow,
            },
            dry_run=getattr(args, "dry_run", False),
        )
        print(f"Submitted target: {bundle.target}")
        print("Engine: ray")
        print(f"Run ID: {run_record.run_id}")
        print(f"Run status: {run_record.status}")
        if run_record.submission_id:
            print(f"Submission ID: {run_record.submission_id}")
        if run_record.message:
            print(f"Message: {run_record.message}")
        print(f"Target directory: {target_dir}")
        print(f"Ray state root: {normalize_state_root(args.ray_state_root)}")
        return 0
    execution_state, operator_requests, outcome = execute_bundle(
        bundle,
        execution_state,
        out_dir=target_dir.parent,
        briefs=briefs,
        prompt_packets=prompt_packets,
        operator_requests=operator_requests,
    )
    print(f"Executed target: {bundle.target}")
    print("Engine: local")
    print(f"Outcome: {outcome}")
    print(f"Current task: {execution_state.current_task or 'none'}")
    print(f"Current stage: {execution_state.current_stage or 'none'}")
    if execution_state.open_question_id:
        print(f"Open question: {execution_state.open_question_id}")
    print(f"Target directory: {target_dir}")
    return 0


def cmd_stage_mutation(args: argparse.Namespace) -> int:
    target_dir = _resolve_stage_target_dir(args)
    capabilities = _load_stage_inputs(args, target_dir)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    compile_view = _build_compile_view(capabilities, support_plan.integration_styles)
    deployment_view = _build_deployment_view(capabilities) if capabilities.deployment is not None else None
    if not getattr(args, "dry_run", False):
        _write_plan_artifacts(
            out_dir=target_dir,
            support_plan=support_plan,
            task_graph=task_graph,
            compile_view=compile_view,
            deployment_view=deployment_view,
        )
        generation_bundle = emit_generation_artifacts(
            capabilities=capabilities,
            support_plan=support_plan,
            task_graph=task_graph,
            target_dir=target_dir,
            compile_view=compile_view,
            deployment_view=deployment_view,
        )
        execution_bundle = build_execution_bundle(
            capabilities,
            support_plan,
            task_graph,
            prompt_backend="none",
            agent=None,
            provider_config={},
        )
        mutation_bundle = emit_mutation_artifacts(
            target_dir=target_dir,
            generation_bundle=generation_bundle,
            execution_bundle=execution_bundle,
        )
    else:
        mutation_bundle = None
    print(f"Staged mutation bundle for: {support_plan.target}")
    print("Mode: non-live")
    if mutation_bundle is not None:
        print(f"Mutation candidates: {len(mutation_bundle.candidates)}")
        print(f"Proposed branch: {mutation_bundle.branch_name}")
        print(f"Proposed tree: {mutation_bundle.proposed_root}")
        print(f"Proposal brief: {target_dir / 'mutation' / 'proposal_brief.md'}")
    print(f"Target directory: {target_dir}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    target_dir = _resolve_target_dir_arg(args.target_dir)
    request = answer_operator_request(
        target_dir=target_dir,
        question_id=args.question_id,
        choice=args.choice,
    )
    print(f"Answered request: {request.id}")
    print(f"Selected option: {request.selected_option}")
    print(f"Target directory: {target_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    target_dir = _resolve_target_dir_arg(args.target_dir)
    print(render_execution_status(target_dir=target_dir))
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "capability",
        help="Path to a canonical TargetGen capability spec",
    )
    parser.add_argument(
        "--overlay",
        help="Optional deployment overlay that augments the capability spec",
    )


def _load_inputs(args: argparse.Namespace):
    capabilities = load_capability_spec(args.capability)
    if args.overlay:
        capabilities.deployment = load_deployment_overlay(args.overlay)
    return capabilities


def _load_execute_inputs(args: argparse.Namespace, target_dir: Path):
    if args.capability:
        capabilities = load_capability_spec(args.capability)
        if args.overlay:
            capabilities.deployment = load_deployment_overlay(args.overlay)
        return capabilities
    if (target_dir / "execution_bundle.json").exists():
        return None
    raise ValueError(
        "execute requires either a capability spec path or an existing " "--from-dir with execution artifacts"
    )


def _target_out_dir(base: str, target_name: str) -> Path:
    return utils.REPO_ROOT / base / target_name


def _prompts_out_dir(base: str | None, target_out_dir: Path) -> Path:
    if not base:
        return target_out_dir / "prompts"
    path = Path(base)
    return path if path.is_absolute() else utils.REPO_ROOT / path


def _resolve_target_dir_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else utils.REPO_ROOT / path


def _resolve_execute_target_dir(args: argparse.Namespace) -> Path:
    if args.from_dir:
        return _resolve_target_dir_arg(args.from_dir)
    if not args.capability:
        raise ValueError("execute requires either a capability spec path or --from-dir")
    capability_name = Path(args.capability).resolve().parent.name
    return _target_out_dir(args.out_dir, capability_name)


def _resolve_stage_target_dir(args: argparse.Namespace) -> Path:
    if args.from_dir:
        return _resolve_target_dir_arg(args.from_dir)
    if not args.capability:
        raise ValueError("stage-mutation requires either a capability spec path or --from-dir")
    capability_name = Path(args.capability).resolve().parent.name
    return _target_out_dir(args.out_dir, capability_name)


def _load_stage_inputs(args: argparse.Namespace, target_dir: Path):
    if args.capability:
        return _load_inputs(args)
    capability_snapshot = target_dir / "inputs" / "capability.yaml"
    if not capability_snapshot.exists():
        raise ValueError(
            "stage-mutation could not find inputs/capability.yaml in the target directory; "
            "run targetgen generate first or provide a capability spec path"
        )
    capabilities = load_capability_spec(str(capability_snapshot))
    overlay_snapshot = target_dir / "inputs" / "overlay.yaml"
    if overlay_snapshot.exists():
        capabilities.deployment = load_deployment_overlay(str(overlay_snapshot))
    return capabilities


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_compile_view(capabilities, integration_styles: list[str]) -> dict:
    generic_flags = []
    if capabilities.platform.host_isa.startswith("riscv"):
        generic_flags.append(f"--iree-llvmcpu-target-triple=" f"{capabilities.platform.host_isa}-unknown-linux-gnu")
    features = ",".join(capabilities.isa.features)
    hw_key = (
        capabilities.deployment.compile_hw
        if capabilities.deployment and capabilities.deployment.compile_hw
        else "default"
    )
    target_name = (
        capabilities.deployment.compile_target
        if capabilities.deployment and capabilities.deployment.compile_target
        else capabilities.identity.name
    )
    compile_view = {
        "target_name": target_name,
        "default_hw": hw_key,
        "generic": generic_flags,
        "targets": {hw_key: [f"--target-isa-features={features}"] if features else []},
        "plugin_flags": [],
    }
    if "post_global_plugin" in integration_styles:
        compile_view["plugin_flags"] = [f"--iree-plugin={capabilities.identity.name.replace('_', '-')}"]
    return compile_view


def _build_deployment_view(capabilities) -> dict:
    assert capabilities.deployment is not None
    return {
        "name": capabilities.deployment.name,
        "mode": capabilities.deployment.mode,
        "build_profile": capabilities.deployment.build_profile,
        "compile_target": capabilities.deployment.compile_target,
        "compile_hw": capabilities.deployment.compile_hw,
        "hardware_recipe": capabilities.deployment.hardware_recipe,
        "chipyard": capabilities.deployment.chipyard,
        "runtime": capabilities.deployment.runtime,
        "extra": capabilities.deployment.extra,
    }


def _write_plan_artifacts(
    *,
    out_dir: Path,
    support_plan,
    task_graph: list,
    compile_view: dict,
    deployment_view: dict | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "support_plan.json", asdict(support_plan))
    _write_json(out_dir / "task_graph.json", [asdict(task) for task in task_graph])
    _write_json(
        out_dir / "verification_manifest.json",
        asdict(support_plan.verification_manifest),
    )
    _write_yaml(out_dir / "compile_view.yaml", compile_view)
    if deployment_view is not None:
        _write_yaml(out_dir / "deployment_view.yaml", deployment_view)
