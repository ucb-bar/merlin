from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import utils

from .model import (
    EvidenceItem,
    ExecutionBundle,
    ExecutionState,
    ExecutionTask,
    OperatorOption,
    OperatorRequest,
    PromptFragment,
    PromptSection,
    ResolvedPromptBundle,
    TaskExecutionState,
)
from .orchestrator import build_execution_bundle, build_execution_state, render_prompt_packets, render_task_briefs
from .planner import build_support_plan, build_task_graph
from .prompts import load_provider_config

TASK_STATES = {
    "planned",
    "preflight_ready",
    "preflight_failed",
    "validating",
    "awaiting_operator",
    "prompt_emitted",
    "awaiting_response",
    "proposal_ready",
    "completed",
    "blocked",
}

REQUEST_STATUS_OPEN = "open"
REQUEST_STATUS_ANSWERED = "answered"
REQUEST_STATUS_RESOLVED = "resolved"

QUESTION_KIND_CREDENTIAL = "credential"
QUESTION_KIND_DEVICE = "device"
QUESTION_KIND_BRANCH = "branch_gate"
QUESTION_KIND_PROVIDER = "provider_disabled"

NON_MUTATING_POLICY = "planner_generated_only"


def prepare_execution_environment(
    capabilities,
    *,
    out_dir: Path,
    prompt_backend: str,
    agent: str | None,
    prompts_dir: Path,
    resume: bool,
) -> tuple[ExecutionBundle, ExecutionState, dict[str, str], dict[str, str], dict[str, OperatorRequest]]:
    target_dir = out_dir / capabilities.identity.name
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    provider_config = load_provider_config(agent) if prompt_backend == "provider" and agent else {}
    bundle = build_execution_bundle(
        capabilities,
        support_plan,
        task_graph,
        prompt_backend=prompt_backend,
        agent=agent,
        provider_config=provider_config,
    )
    prompt_packets = {}
    if prompt_backend != "none":
        bundle.prompts_dir = str(prompts_dir)
        prompt_packets = render_prompt_packets(bundle)
    briefs = render_task_briefs(bundle)
    state = build_execution_state(bundle)
    if resume:
        previous_state = _load_json(target_dir / "execution_state.json")
        if previous_state:
            state = _merge_execution_state(bundle, previous_state)
    requests = _load_operator_requests(target_dir / "operator_requests")
    return bundle, state, briefs, prompt_packets, requests


def load_execution_environment_from_dir(
    *,
    target_dir: Path,
) -> tuple[ExecutionBundle, ExecutionState, dict[str, str], dict[str, str], dict[str, OperatorRequest]]:
    bundle_payload = _load_json(target_dir / "execution_bundle.json")
    if bundle_payload is None:
        raise ValueError(f"No execution_bundle.json found in {target_dir}")
    bundle = _execution_bundle_from_payload(bundle_payload)
    state_payload = _load_json(target_dir / "execution_state.json")
    if state_payload is None:
        raise ValueError(f"No execution_state.json found in {target_dir}")
    state = _execution_state_from_payload(bundle, state_payload)
    briefs = {item.name: item.read_text(encoding="utf-8") for item in sorted((target_dir / "briefs").glob("*.md"))}
    prompt_packets = {
        item.name: item.read_text(encoding="utf-8") for item in sorted((target_dir / "prompts").glob("prompt_*.md"))
    }
    requests = _load_operator_requests(target_dir / "operator_requests")
    return bundle, state, briefs, prompt_packets, requests


def execute_bundle(
    bundle: ExecutionBundle,
    execution_state: ExecutionState,
    *,
    out_dir: Path,
    briefs: dict[str, str],
    prompt_packets: dict[str, str],
    operator_requests: dict[str, OperatorRequest],
) -> tuple[ExecutionState, dict[str, OperatorRequest], str]:
    target_dir = out_dir / bundle.target
    prompts_dir = target_dir / "prompts"
    task_states_dir = target_dir / "task_states"
    requests_dir = target_dir / "operator_requests"
    briefs_dir = target_dir / "briefs"
    target_dir.mkdir(parents=True, exist_ok=True)
    task_states_dir.mkdir(parents=True, exist_ok=True)
    briefs_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)
    if bundle.prompt_backend != "none":
        prompts_dir.mkdir(parents=True, exist_ok=True)

    if bundle.prompt_backend == "provider":
        first_task = next((task for task in bundle.tasks if task.execution_state.status != "completed"), None)
        if first_task is not None:
            existing = operator_requests.get(f"{first_task.id}-{QUESTION_KIND_PROVIDER}")
            if existing is not None and existing.status == REQUEST_STATUS_RESOLVED:
                _set_task_status(first_task.execution_state, "blocked", QUESTION_KIND_PROVIDER)
                first_task.execution_state.logs.append(
                    "Provider execution remains disabled until a manual prompt backend is selected."
                )
                execution_state.current_task = first_task.id
                execution_state.current_stage = QUESTION_KIND_PROVIDER
                execution_state.open_question_id = None
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "blocked"
            request = _ensure_operator_request(
                operator_requests,
                target=bundle.target,
                task=first_task,
                stage=QUESTION_KIND_PROVIDER,
                question="Live provider execution is disabled in this milestone.",
                options=[
                    OperatorOption(
                        id="switch_to_manualllm",
                        label="Switch to ManualLLM",
                        description="Regenerate or rerun with --prompt-backend manualllm.",
                    ),
                    OperatorOption(
                        id="pause_execution",
                        label="Pause execution",
                        description="Stop here and keep the execution state unchanged.",
                    ),
                ],
                recommended_option="switch_to_manualllm",
            )
            _set_task_status(first_task.execution_state, "awaiting_operator", QUESTION_KIND_PROVIDER)
            first_task.execution_state.open_question_id = request.id
            execution_state.current_task = first_task.id
            execution_state.current_stage = QUESTION_KIND_PROVIDER
            execution_state.open_question_id = request.id
            _persist_runtime_artifacts(
                bundle,
                execution_state,
                briefs,
                prompt_packets,
                operator_requests,
                target_dir,
            )
            return execution_state, operator_requests, "awaiting_operator"

    for task in bundle.tasks:
        task_state = task.execution_state
        execution_state.current_task = task.id
        execution_state.current_stage = task_state.current_stage
        if task_state.status == "completed":
            continue
        if task_state.status in {"blocked", "preflight_failed"}:
            _persist_runtime_artifacts(
                bundle,
                execution_state,
                briefs,
                prompt_packets,
                operator_requests,
                target_dir,
            )
            return execution_state, operator_requests, "blocked"
        if _has_unresolved_dependency(bundle, task):
            execution_state.current_stage = "dependency_wait"
            _persist_runtime_artifacts(
                bundle,
                execution_state,
                briefs,
                prompt_packets,
                operator_requests,
                target_dir,
            )
            return execution_state, operator_requests, "blocked"
        if task_state.status == "awaiting_operator":
            request = operator_requests.get(task_state.open_question_id or "")
            if request is None or request.status != REQUEST_STATUS_ANSWERED:
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_operator"
            outcome = _apply_operator_answer(task, task_state, request, execution_state)
            request.status = REQUEST_STATUS_RESOLVED
            request.updated_at = _timestamp()
            if outcome != "continue":
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, outcome
        if task_state.status in {"planned", "preflight_ready", "preflight_failed", "validating"}:
            preflight = _run_preflight_checks(task, bundle.target)
            task_state.validation_results.extend(preflight)
            if any(result.startswith("FAIL") for result in preflight):
                _set_task_status(task_state, "preflight_failed", "preflight")
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "blocked"
            _set_task_status(task_state, "preflight_ready", "preflight")
            credential_request = _maybe_require_credential(
                task,
                task_state,
                operator_requests,
                bundle.target,
            )
            if credential_request is not None:
                execution_state.open_question_id = credential_request.id
                execution_state.current_stage = QUESTION_KIND_CREDENTIAL
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_operator"
            device_request = _maybe_require_device_confirmation(
                task,
                task_state,
                operator_requests,
                bundle.target,
            )
            if device_request is not None:
                execution_state.open_question_id = device_request.id
                execution_state.current_stage = QUESTION_KIND_DEVICE
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_operator"
            _record_deferred_validations(task_state)
            if task.mutation_policy == NON_MUTATING_POLICY:
                _set_task_status(task_state, "completed", "completed")
                task_state.logs.append("Auto-completed non-mutating task.")
                continue
            if bundle.prompt_backend != "none" and task.prompt_packet in prompt_packets:
                prompt_path = prompts_dir / task.prompt_packet
                if not prompt_path.exists():
                    prompt_path.write_text(prompt_packets[task.prompt_packet], encoding="utf-8")
                _set_task_status(task_state, "awaiting_response", "prompt_emitted")
                task_state.logs.append(f"Prompt emitted at {prompt_path}.")
                execution_state.current_stage = "prompt_emitted"
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_response"
            _set_task_status(task_state, "blocked", "prompt_backend_disabled")
            task_state.logs.append("Prompt backend disabled for a task that requires a proposal.")
            _persist_runtime_artifacts(
                bundle,
                execution_state,
                briefs,
                prompt_packets,
                operator_requests,
                target_dir,
            )
            return execution_state, operator_requests, "blocked"
        if task_state.status == "awaiting_response":
            response_path = prompts_dir / (task.response_packet or "")
            if not response_path.exists():
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_response"
            task_state.response_file = str(response_path)
            task_state.proposal_summary = _summarize_response(response_path)
            _set_task_status(task_state, "proposal_ready", "proposal_review")
        if task_state.status == "proposal_ready":
            if task.mutation_policy != NON_MUTATING_POLICY and not task_state.live_mutation_enabled:
                branch_request = _ensure_operator_request(
                    operator_requests,
                    target=bundle.target,
                    task=task,
                    stage=QUESTION_KIND_BRANCH,
                    question=(
                        "Task reached a mutation boundary. Switch to a dedicated branch "
                        "before enabling live mutation."
                    ),
                    options=[
                        OperatorOption(
                            id="pause_for_branch_switch",
                            label="Pause for branch switch",
                            description="Stop here and resume after switching to a dedicated branch.",
                        ),
                        OperatorOption(
                            id="continue_without_mutation",
                            label="Continue without mutation",
                            description=("Record the proposal but do not mutate repo-tracked files."),
                        ),
                        OperatorOption(
                            id="mark_blocked",
                            label="Mark blocked",
                            description=("Leave the task blocked until mutation is explicitly enabled."),
                        ),
                    ],
                    recommended_option="pause_for_branch_switch",
                )
                task_state.branch_gate_required = True
                task_state.open_question_id = branch_request.id
                _set_task_status(task_state, "awaiting_operator", QUESTION_KIND_BRANCH)
                execution_state.open_question_id = branch_request.id
                execution_state.current_stage = QUESTION_KIND_BRANCH
                _persist_runtime_artifacts(
                    bundle, execution_state, briefs, prompt_packets, operator_requests, target_dir
                )
                return execution_state, operator_requests, "awaiting_operator"
            _set_task_status(task_state, "completed", "completed")
            task_state.logs.append("Proposal recorded and task marked complete.")
            continue

    execution_state.current_task = None
    execution_state.current_stage = None
    execution_state.open_question_id = None
    _persist_runtime_artifacts(
        bundle,
        execution_state,
        briefs,
        prompt_packets,
        operator_requests,
        target_dir,
    )
    return execution_state, operator_requests, "completed"


def persist_execution_environment(
    bundle: ExecutionBundle,
    execution_state: ExecutionState,
    *,
    out_dir: Path,
    briefs: dict[str, str],
    prompt_packets: dict[str, str],
    operator_requests: dict[str, OperatorRequest] | None = None,
) -> Path:
    operator_requests = operator_requests or {}
    target_dir = out_dir / bundle.target
    _persist_runtime_artifacts(
        bundle,
        execution_state,
        briefs,
        prompt_packets,
        operator_requests,
        target_dir,
    )
    return target_dir


def answer_operator_request(*, target_dir: Path, question_id: str, choice: str) -> OperatorRequest:
    requests_dir = target_dir / "operator_requests"
    requests = _load_operator_requests(requests_dir)
    request = requests.get(question_id)
    if request is None:
        raise ValueError(f"Unknown operator request: {question_id}")
    option_ids = {option.id for option in request.options}
    if choice not in option_ids:
        raise ValueError(f"Choice {choice!r} is not valid for request {question_id}")
    request.selected_option = choice
    request.status = REQUEST_STATUS_ANSWERED
    request.updated_at = _timestamp()
    _write_json(requests_dir / f"{request.id}.json", asdict(request))
    return request


def render_execution_status(*, target_dir: Path) -> str:
    state_payload = _load_json(target_dir / "execution_state.json")
    if state_payload is None:
        raise ValueError(f"No execution_state.json found in {target_dir}")
    requests = _load_operator_requests(target_dir / "operator_requests")
    lines = [
        f"Target: {state_payload['target']}",
        f"Workflow: {state_payload['workflow']}",
        f"Prompt backend: {state_payload['prompt_backend']}",
        f"Current task: {state_payload.get('current_task') or 'none'}",
        f"Current stage: {state_payload.get('current_stage') or 'none'}",
        f"Open question: {state_payload.get('open_question_id') or 'none'}",
        "",
        "Tasks:",
    ]
    for task_id in state_payload["task_order"]:
        task_state = state_payload["tasks"][task_id]
        lines.append(
            f"- {task_id}: {task_state['status']} "
            f"(stage={task_state.get('current_stage') or 'none'}, "
            f"adapter={task_state['execution_adapter']})"
        )
    if requests:
        lines.append("")
        lines.append("Operator requests:")
        for request in sorted(requests.values(), key=lambda item: item.id):
            lines.append(
                f"- {request.id}: {request.status} "
                f"(task={request.task_id}, "
                f"recommended={request.recommended_option}, "
                f"selected={request.selected_option or 'none'})"
            )
    return "\n".join(lines)


def _execution_bundle_from_payload(payload: dict[str, Any]) -> ExecutionBundle:
    tasks = [_execution_task_from_payload(task_payload) for task_payload in payload["tasks"]]
    return ExecutionBundle(
        target=payload["target"],
        workflow=payload["workflow"],
        mode=payload["mode"],
        prompt_backend=payload["prompt_backend"],
        agent=payload.get("agent"),
        tasks=tasks,
        prompts_dir=payload.get("prompts_dir"),
        provider_config=payload.get("provider_config", {}),
    )


def _execution_task_from_payload(payload: dict[str, Any]) -> ExecutionTask:
    prompt_bundle_payload = payload["prompt_bundle"]
    default_state = TaskExecutionState(
        status="planned",
        repo_root=payload["repo_root"],
        execution_adapter=payload["execution_adapter"],
        mutation_policy=payload["mutation_policy"],
        artifacts_in=payload["artifacts_in"],
        artifacts_out=payload["artifacts_out"],
        validation_commands=payload["validation_commands"],
        credential_requirements=payload["credential_requirements"],
        handoff_contract=payload["handoff_contract"],
    )
    return ExecutionTask(
        id=payload["id"],
        title=payload["title"],
        phase=payload["phase"],
        family=payload["family"],
        depends_on=payload.get("depends_on", []),
        tool_profile=payload["tool_profile"],
        prompt=payload["prompt"],
        prompt_bundle=ResolvedPromptBundle(
            backend=prompt_bundle_payload["backend"],
            tool_profile=prompt_bundle_payload["tool_profile"],
            agent=prompt_bundle_payload.get("agent"),
            sections=[
                PromptSection(
                    name=section["name"],
                    heading=section["heading"],
                    content=section["content"],
                    sources=section["sources"],
                )
                for section in prompt_bundle_payload["sections"]
            ],
            fragments=[
                PromptFragment(
                    source=fragment["source"],
                    scope=fragment["scope"],
                    section=fragment["section"],
                    merge_mode=fragment["merge_mode"],
                )
                for fragment in prompt_bundle_payload["fragments"]
            ],
            flattened_prompt=prompt_bundle_payload["flattened_prompt"],
        ),
        evidence=[
            EvidenceItem(
                kind=item["kind"],
                value=item["value"],
                reason=item["reason"],
            )
            for item in payload["evidence"]
        ],
        write_scope=payload["write_scope"],
        acceptance_checks=payload["acceptance_checks"],
        repo_root=payload["repo_root"],
        execution_adapter=payload["execution_adapter"],
        mutation_policy=payload["mutation_policy"],
        artifacts_in=payload["artifacts_in"],
        artifacts_out=payload["artifacts_out"],
        validation_commands=payload["validation_commands"],
        credential_requirements=payload["credential_requirements"],
        handoff_contract=payload["handoff_contract"],
        execution_state=_task_state_from_payload(default_state, payload.get("execution_state", {})),
        prompt_packet=payload.get("prompt_packet"),
        response_packet=payload.get("response_packet"),
    )


def _merge_execution_state(bundle: ExecutionBundle, payload: dict[str, Any]) -> ExecutionState:
    for task in bundle.tasks:
        saved = payload.get("tasks", {}).get(task.id)
        if saved:
            task.execution_state = _task_state_from_payload(task.execution_state, saved)
    merged = build_execution_state(bundle)
    merged.current_task = payload.get("current_task")
    merged.current_stage = payload.get("current_stage")
    merged.live_mutation_enabled = bool(payload.get("live_mutation_enabled", False))
    merged.open_question_id = payload.get("open_question_id")
    return merged


def _execution_state_from_payload(bundle: ExecutionBundle, payload: dict[str, Any]) -> ExecutionState:
    default_state = build_execution_state(bundle)
    tasks: dict[str, TaskExecutionState] = {}
    for task in bundle.tasks:
        saved = payload.get("tasks", {}).get(task.id)
        if saved:
            task.execution_state = _task_state_from_payload(task.execution_state, saved)
        tasks[task.id] = task.execution_state
    return ExecutionState(
        target=payload.get("target", default_state.target),
        workflow=payload.get("workflow", default_state.workflow),
        prompt_backend=payload.get("prompt_backend", default_state.prompt_backend),
        task_order=payload.get("task_order", default_state.task_order),
        tasks=tasks,
        current_task=payload.get("current_task"),
        current_stage=payload.get("current_stage"),
        live_mutation_enabled=bool(payload.get("live_mutation_enabled", False)),
        open_question_id=payload.get("open_question_id"),
    )


def _task_state_from_payload(default: TaskExecutionState, payload: dict[str, Any]) -> TaskExecutionState:
    data = asdict(default)
    data.update(payload)
    return TaskExecutionState(**data)


def _has_unresolved_dependency(bundle: ExecutionBundle, task: ExecutionTask) -> bool:
    completed = {item.id for item in bundle.tasks if item.execution_state.status == "completed"}
    return any(dep not in completed for dep in task.depends_on)


def _run_preflight_checks(task: ExecutionTask, target_name: str) -> list[str]:
    results: list[str] = []
    repo_root = utils.REPO_ROOT / task.repo_root
    if repo_root.exists():
        results.append(f"PASS repo_root_exists {repo_root}")
    else:
        results.append(f"FAIL repo_root_missing {repo_root}")
    for artifact in task.artifacts_in:
        normalized = artifact.replace("<target>", target_name)
        if normalized.startswith("build/generated/"):
            results.append(f"SKIP generated_artifact_check {normalized}")
            continue
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = utils.REPO_ROOT / normalized
        if candidate.exists():
            results.append(f"PASS artifact_exists {normalized}")
        else:
            results.append(f"SKIP artifact_not_found {normalized}")
    return results


def _record_deferred_validations(task_state: TaskExecutionState) -> None:
    task_state.validation_results.extend(
        f"DEFERRED validation_command {command}"
        for command in task_state.validation_commands
        if command and not command.startswith("#")
    )


def _maybe_require_credential(
    task: ExecutionTask,
    task_state: TaskExecutionState,
    operator_requests: dict[str, OperatorRequest],
    target: str,
) -> OperatorRequest | None:
    requirement = task.credential_requirements
    if requirement in {"", "none"}:
        return None
    existing = operator_requests.get(f"{task.id}-{QUESTION_KIND_CREDENTIAL}")
    if existing and existing.status == REQUEST_STATUS_RESOLVED:
        if existing.selected_option == "credential_ready":
            return None
        if existing.selected_option in {"credential_missing", "pause_execution"}:
            _set_task_status(task_state, "blocked", QUESTION_KIND_CREDENTIAL)
            return None
    if _credential_is_available(requirement):
        task_state.logs.append(f"Credential requirement satisfied for {requirement}.")
        return None
    request = _ensure_operator_request(
        operator_requests,
        target=target,
        task=task,
        stage=QUESTION_KIND_CREDENTIAL,
        question=f"Credential requirement `{requirement}` is not currently satisfied.",
        options=[
            OperatorOption(
                id="credential_ready",
                label="Credential is ready",
                description="Resume after making the credential available out of band.",
            ),
            OperatorOption(
                id="credential_missing",
                label="Credential is missing",
                description="Keep the task blocked because the credential is unavailable.",
            ),
            OperatorOption(
                id="pause_execution",
                label="Pause execution",
                description="Stop here without changing the task outcome.",
            ),
        ],
        recommended_option="pause_execution",
    )
    task_state.open_question_id = request.id
    _set_task_status(task_state, "awaiting_operator", QUESTION_KIND_CREDENTIAL)
    return request


def _maybe_require_device_confirmation(
    task: ExecutionTask,
    task_state: TaskExecutionState,
    operator_requests: dict[str, OperatorRequest],
    target: str,
) -> OperatorRequest | None:
    needs_device = any("device_query" in command for command in task.validation_commands)
    if not needs_device or task.family != "runtime_hal":
        return None
    existing = operator_requests.get(f"{task.id}-{QUESTION_KIND_DEVICE}")
    if existing and existing.status == REQUEST_STATUS_RESOLVED:
        if existing.selected_option == "device_available":
            return None
        if existing.selected_option in {"device_unavailable", "pause_execution"}:
            _set_task_status(task_state, "blocked", QUESTION_KIND_DEVICE)
            return None
    request = _ensure_operator_request(
        operator_requests,
        target=target,
        task=task,
        stage=QUESTION_KIND_DEVICE,
        question="Device availability must be confirmed before runtime execution continues.",
        options=[
            OperatorOption(
                id="device_available",
                label="Device available",
                description="Continue with prompt emission and proposal collection.",
            ),
            OperatorOption(
                id="device_unavailable",
                label="Device unavailable",
                description="Mark the task blocked because the target device is unavailable.",
            ),
            OperatorOption(
                id="pause_execution",
                label="Pause execution",
                description="Stop here and keep the task awaiting operator input.",
            ),
        ],
        recommended_option="device_available",
    )
    task_state.open_question_id = request.id
    _set_task_status(task_state, "awaiting_operator", QUESTION_KIND_DEVICE)
    return request


def _apply_operator_answer(
    task: ExecutionTask,
    task_state: TaskExecutionState,
    request: OperatorRequest,
    execution_state: ExecutionState,
) -> str:
    choice = request.selected_option
    task_state.logs.append(f"Operator answered {request.id} with {choice}.")
    task_state.open_question_id = None
    execution_state.open_question_id = None
    if choice in {"credential_ready", "device_available"}:
        _set_task_status(task_state, "preflight_ready", "preflight")
        return "continue"
    if choice in {"credential_missing", "device_unavailable", "mark_blocked"}:
        _set_task_status(task_state, "blocked", request.stage)
        return "blocked"
    if choice == "continue_without_mutation":
        _set_task_status(task_state, "blocked", QUESTION_KIND_BRANCH)
        task_state.logs.append("Proposal retained without mutation.")
        return "blocked"
    if choice in {"pause_execution", "pause_for_branch_switch", "switch_to_manualllm"}:
        _set_task_status(task_state, "blocked", request.stage)
        return "blocked"
    return "blocked"


def _ensure_operator_request(
    operator_requests: dict[str, OperatorRequest],
    *,
    target: str,
    task: ExecutionTask,
    stage: str,
    question: str,
    options: list[OperatorOption],
    recommended_option: str,
) -> OperatorRequest:
    request_id = f"{task.id}-{stage}"
    existing = operator_requests.get(request_id)
    if existing is not None:
        return existing
    request = OperatorRequest(
        id=request_id,
        target=target,
        task_id=task.id,
        stage=stage,
        question=question,
        options=options,
        recommended_option=recommended_option,
        status=REQUEST_STATUS_OPEN,
        created_at=_timestamp(),
        updated_at=_timestamp(),
    )
    operator_requests[request_id] = request
    return request


def _persist_runtime_artifacts(
    bundle: ExecutionBundle,
    execution_state: ExecutionState,
    briefs: dict[str, str],
    prompt_packets: dict[str, str],
    operator_requests: dict[str, OperatorRequest],
    target_dir: Path,
) -> None:
    (target_dir / "briefs").mkdir(parents=True, exist_ok=True)
    (target_dir / "task_states").mkdir(parents=True, exist_ok=True)
    (target_dir / "operator_requests").mkdir(parents=True, exist_ok=True)
    if bundle.prompt_backend != "none":
        (target_dir / "prompts").mkdir(parents=True, exist_ok=True)
    _write_json(target_dir / "execution_bundle.json", asdict(bundle))
    _write_json(target_dir / "execution_state.json", asdict(execution_state))
    if bundle.provider_config:
        _write_json(target_dir / "provider_backend.json", bundle.provider_config)
    for filename, content in briefs.items():
        (target_dir / "briefs" / filename).write_text(content, encoding="utf-8")
    for filename, content in prompt_packets.items():
        if bundle.prompt_backend != "none":
            (target_dir / "prompts" / filename).write_text(content, encoding="utf-8")
    for task_id, task_state in execution_state.tasks.items():
        _write_json(target_dir / "task_states" / f"{task_id}.json", asdict(task_state))
    for request in operator_requests.values():
        _write_json(target_dir / "operator_requests" / f"{request.id}.json", asdict(request))


def _load_operator_requests(path: Path) -> dict[str, OperatorRequest]:
    if not path.exists():
        return {}
    requests: dict[str, OperatorRequest] = {}
    for item in sorted(path.glob("*.json")):
        payload = _load_json(item)
        if payload is None:
            continue
        requests[payload["id"]] = OperatorRequest(
            id=payload["id"],
            target=payload["target"],
            task_id=payload["task_id"],
            stage=payload["stage"],
            question=payload["question"],
            options=[OperatorOption(**option) for option in payload["options"]],
            recommended_option=payload["recommended_option"],
            status=payload["status"],
            selected_option=payload.get("selected_option"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )
    return requests


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_response(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "Empty response file."
    summary = " ".join(lines[:4])
    return summary[:240]


def _credential_is_available(requirement: str) -> bool:
    if requirement in {"", "none"}:
        return True
    env_name = requirement.upper()
    return bool(os.environ.get(env_name))


def _set_task_status(task_state: TaskExecutionState, status: str, stage: str) -> None:
    if status not in TASK_STATES:
        raise ValueError(f"Unknown task status: {status}")
    task_state.status = status
    task_state.current_stage = stage
    task_state.updated_at = _timestamp()
    task_state.attempt_count += 1
    if task_state.opened_at is None:
        task_state.opened_at = task_state.updated_at


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()
