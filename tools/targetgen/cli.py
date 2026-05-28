#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ray import normalize_state_root, submit_job

import utils
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

from .io import _build_compile_view, _build_deployment_view, _write_json, _write_plan_artifacts
from .paths import (
    _prompts_out_dir,
    _resolve_execute_target_dir,
    _resolve_stage_target_dir,
    _resolve_target_dir_arg,
    _target_out_dir,
)

# Underscore helpers extracted to siblings
from .render import _build_evidence_graph, _render_capability_draft, _render_modification_map_md, _resolve_source_arg
from .spec import _load_execute_inputs, _load_inputs, _load_stage_inputs


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

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Scan an external target source tree and emit a SourceInventory",
    )
    ingest_parser.add_argument(
        "--target-name",
        required=True,
        help="Logical target name used for the output subdirectory",
    )
    ingest_parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Path to a target source tree (may be repeated)",
    )
    ingest_parser.add_argument(
        "--scanner",
        action="append",
        help="Restrict to a specific scanner (may be repeated). Defaults to all scanners.",
    )
    ingest_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Output directory for ingestion artifacts",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing artifacts",
    )
    ingest_parser.set_defaults(_handler=cmd_ingest)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a previously ingested SourceInventory into integration styles",
    )
    classify_parser.add_argument(
        "--target-name",
        required=True,
        help="Logical target name; selects subdirectory under --from-dir / --out-dir",
    )
    classify_parser.add_argument(
        "--from-dir",
        default="build/generated/targetgen",
        help="Directory containing source_inventory.json (default: build/generated/targetgen)",
    )
    classify_parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (defaults to --from-dir)",
    )
    classify_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing artifacts",
    )
    classify_parser.set_defaults(_handler=cmd_classify)

    modmap_parser = subparsers.add_parser(
        "modification-map",
        help="Emit a per-stage patch-surface modification map for a capability spec",
    )
    _add_common_args(modmap_parser)
    modmap_parser.add_argument(
        "--out-dir",
        default="build/generated/targetgen",
        help="Output directory for modification-map artifacts",
    )
    modmap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing artifacts",
    )
    modmap_parser.set_defaults(_handler=cmd_modification_map)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Launch the TargetGen MCP server over stdio for Claude Code",
    )
    mcp_parser.set_defaults(_handler=cmd_mcp)


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


def cmd_ingest(args: argparse.Namespace) -> int:
    from targetgen.intake import build_source_inventory

    sources = [_resolve_source_arg(value) for value in args.source]
    inventory = build_source_inventory(
        target=args.target_name,
        sources=sources,
        scanners=args.scanner,
    )
    print(f"Target: {inventory.target}")
    print(f"Repositories scanned: {len(inventory.repositories)}")
    print(f"Findings: {len(inventory.findings)}")
    print(f"Detected source kinds: {inventory.detected_source_kinds}")
    if inventory.missing_information:
        print(f"Missing information: {inventory.missing_information}")
    if args.dry_run:
        return 0

    out_dir = _target_out_dir(args.out_dir, args.target_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "source_inventory.json"
    inventory_path.write_text(
        json.dumps(asdict(inventory), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    evidence_path = out_dir / "evidence_graph.json"
    evidence_path.write_text(
        json.dumps(_build_evidence_graph(inventory), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {inventory_path}")
    print(f"Wrote {evidence_path}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from targetgen.intake import classify_inventory
    from targetgen.model import (
        SourceFinding,
        SourceInventory,
        SourceRepository,
    )

    in_dir = _target_out_dir(args.from_dir, args.target_name)
    inventory_path = in_dir / "source_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"source_inventory.json not found at {inventory_path}. " "Run `./merlin targetgen ingest` first."
        )
    raw = json.loads(inventory_path.read_text())
    inventory = SourceInventory(
        target=raw["target"],
        repositories=[SourceRepository(**r) for r in raw["repositories"]],
        findings=[SourceFinding(**f) for f in raw["findings"]],
        detected_source_kinds=list(raw["detected_source_kinds"]),
        missing_information=list(raw.get("missing_information", [])),
    )
    classification = classify_inventory(inventory)
    print(f"Target: {classification.target}")
    print(f"Source styles: {classification.source_styles}")
    print(f"TargetGen styles: {classification.targetgen_styles}")
    print(f"Primary integration: {classification.primary_integration}")
    print(f"Confidence: {classification.confidence}")
    if classification.missing_information:
        print("Missing information:")
        for item in classification.missing_information:
            print(f"  - {item}")

    if args.dry_run:
        return 0
    out_dir = _target_out_dir(args.out_dir or args.from_dir, args.target_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    classification_path = out_dir / "classification.json"
    classification_path.write_text(
        json.dumps(asdict(classification), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    draft_path = out_dir / "capability.draft.yaml"
    draft_path.write_text(_render_capability_draft(classification), encoding="utf-8")
    print(f"Wrote {classification_path}")
    print(f"Wrote {draft_path}")
    return 0


def cmd_modification_map(args: argparse.Namespace) -> int:
    from targetgen.stage_map import build_modification_map

    capabilities = _load_inputs(args)
    targetgen_styles = build_support_plan(capabilities).integration_styles
    modmap = build_modification_map(capabilities, targetgen_styles=targetgen_styles)
    target = capabilities.identity.name
    print(f"Target: {target}")
    print(f"Primary integration: {modmap.primary_integration}")
    print(f"Integration styles: {modmap.integration_styles}")
    print("Stages:")
    for stage in modmap.stages:
        print(f"  {stage.stage:38s} applies={stage.applies}  writes={len(stage.write_paths)}")

    if args.dry_run:
        return 0
    out_dir = _target_out_dir(args.out_dir, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "modification_map.json"
    json_path.write_text(
        json.dumps(asdict(modmap), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "modification_map.md"
    md_path.write_text(_render_modification_map_md(modmap), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """`./merlin targetgen mcp` — start the targetgen MCP server.

    Historical entry point; the modern equivalent is `./merlin mcp targetgen`.
    Both route to `tools/mcp_servers/targetgen.py`.
    """
    from mcp import scaffold
    from mcp import targetgen as registry

    _, run_stdio = scaffold.make_stdio_server(
        server_name="merlin-targetgen",
        logger_name="targetgen.mcp",
        list_tool_definitions=registry.list_tool_definitions,
        dispatch_tool=registry.dispatch_tool,
        ToolError=registry.ToolError,
    )
    run_stdio()
    return 0


def _resolve_source_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (utils.REPO_ROOT / path).resolve()


def _build_evidence_graph(inventory) -> dict:
    by_kind: dict[str, list[dict]] = {}
    for finding in inventory.findings:
        by_kind.setdefault(finding.kind, []).append(
            {
                "path": finding.path,
                "symbol": finding.symbol,
                "evidence": finding.evidence,
                "confidence": finding.confidence,
            }
        )
    return {
        "target": inventory.target,
        "repositories": [asdict(repo) for repo in inventory.repositories],
        "findings_by_kind": by_kind,
        "detected_source_kinds": inventory.detected_source_kinds,
        "missing_information": inventory.missing_information,
    }


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "capability",
        help="Path to a canonical TargetGen capability spec",
    )
    parser.add_argument(
        "--overlay",
        help="Optional deployment overlay that augments the capability spec",
    )
