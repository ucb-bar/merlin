from __future__ import annotations

from .model import (
    ExecutionBundle,
    ExecutionState,
    ExecutionTask,
    SupportPlan,
    TargetCapabilities,
    TaskExecutionState,
    TaskNode,
)
from .prompts import render_sectioned_markdown, resolve_prompt_bundle


def build_execution_bundle(
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task_graph: list[TaskNode],
    *,
    prompt_backend: str = "manualllm",
    agent: str | None = None,
    provider_config: dict | None = None,
) -> ExecutionBundle:
    tasks = []
    for task in task_graph:
        prompt_bundle = resolve_prompt_bundle(
            capabilities,
            support_plan,
            task,
            prompt_backend=prompt_backend,
            agent=agent,
        )
        tasks.append(
            ExecutionTask(
                id=task.id,
                title=task.title,
                phase=task.phase,
                family=task.family,
                depends_on=task.depends_on,
                tool_profile=prompt_bundle.tool_profile,
                prompt=prompt_bundle.flattened_prompt,
                prompt_bundle=prompt_bundle,
                evidence=task.evidence,
                write_scope=task.write_scope,
                acceptance_checks=task.acceptance_checks,
                repo_root=task.repo_root,
                execution_adapter=task.execution_adapter,
                mutation_policy=task.mutation_policy,
                artifacts_in=task.artifacts_in,
                artifacts_out=task.artifacts_out,
                validation_commands=task.validation_commands,
                credential_requirements=task.credential_requirements,
                handoff_contract=task.handoff_contract,
                execution_state=TaskExecutionState(
                    status="planned",
                    repo_root=task.repo_root,
                    execution_adapter=task.execution_adapter,
                    mutation_policy=task.mutation_policy,
                    artifacts_in=task.artifacts_in,
                    artifacts_out=task.artifacts_out,
                    validation_commands=task.validation_commands,
                    credential_requirements=task.credential_requirements,
                    handoff_contract=task.handoff_contract,
                ),
            )
        )
    return ExecutionBundle(
        target=capabilities.identity.name,
        workflow="evidence_first_target_enablement",
        mode="agentic_contextual_edit",
        prompt_backend=prompt_backend,
        agent=agent,
        provider_config=provider_config or {},
        tasks=tasks,
    )


def render_task_briefs(bundle: ExecutionBundle) -> dict[str, str]:
    briefs: dict[str, str] = {}
    for index, task in enumerate(bundle.tasks, start=1):
        metadata = {
            "Task ID": task.id,
            "Phase": task.phase,
            "Family": task.family,
            "Tool Profile": task.tool_profile,
            "Prompt Backend": bundle.prompt_backend,
            "Execution Adapter": task.execution_adapter,
            "Repo Root": task.repo_root,
            "Mutation Policy": task.mutation_policy,
        }
        if task.prompt_packet:
            metadata["Prompt Packet"] = task.prompt_packet
        if task.response_packet:
            metadata["Response Packet"] = task.response_packet
        briefs[f"{index:02d}-{task.id}.md"] = render_sectioned_markdown(
            title=task.title,
            sections=task.prompt_bundle.sections,
            metadata=metadata,
        )
    return briefs


def render_prompt_packets(bundle: ExecutionBundle) -> dict[str, str]:
    packets: dict[str, str] = {}
    for index, task in enumerate(bundle.tasks, start=1):
        prompt_filename = f"prompt_{index:03d}.md"
        response_filename = f"prompt_{index:03d}.response.md"
        task.prompt_packet = prompt_filename
        task.response_packet = response_filename
        packets[prompt_filename] = render_sectioned_markdown(
            title=f"TargetGen Task {index:03d}: {task.title}",
            sections=task.prompt_bundle.sections,
            metadata={
                "Target": bundle.target,
                "Task ID": task.id,
                "Phase": task.phase,
                "Family": task.family,
                "Tool Profile": task.tool_profile,
                "Prompt Backend": bundle.prompt_backend,
                "Execution Adapter": task.execution_adapter,
                "Repo Root": task.repo_root,
                "Expected Response File": response_filename,
                "Agent": bundle.agent or "",
            },
        )
    return packets


def build_execution_state(bundle: ExecutionBundle) -> ExecutionState:
    return ExecutionState(
        target=bundle.target,
        workflow=bundle.workflow,
        prompt_backend=bundle.prompt_backend,
        task_order=[task.id for task in bundle.tasks],
        tasks={task.id: task.execution_state for task in bundle.tasks},
    )
