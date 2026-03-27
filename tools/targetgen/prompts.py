from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from .model import (
    PromptFragment,
    PromptSection,
    ResolvedPromptBundle,
    SupportPlan,
    TargetCapabilities,
    TaskNode,
    to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_LIBRARY_ROOT = REPO_ROOT / "tools" / "targetgen" / "prompt_library"
MLIR_AGENT_CONFIGS_ROOT = REPO_ROOT / "projects" / "mlirAgent" / "configs"

PROMPT_SECTION_ORDER = (
    "system",
    "goal",
    "target_facts",
    "implementation_focus",
    "evidence",
    "write_scope",
    "acceptance",
    "response_contract",
)

PROMPT_SECTION_HEADINGS = {
    "system": "System",
    "goal": "Goal",
    "target_facts": "Target Facts",
    "implementation_focus": "Implementation Focus",
    "evidence": "Evidence",
    "write_scope": "Write Scope",
    "acceptance": "Acceptance",
    "response_contract": "Response Contract",
}

PROMPT_MERGE_MODES = {"append", "prepend", "replace"}
MLIR_AGENT_FAMILIES = {"post_global_plugin", "structured_text_isa"}
ALLOWED_FRONTMATTER_KEYS = {
    "section",
    "merge",
    "families",
    "phases",
    "task_ids",
    "targets",
    "deployment_profiles",
    "integration_styles",
    "prompt_backends",
    "tool_profiles",
}
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass(slots=True)
class _PromptFragmentFile:
    path: Path
    scope: str
    metadata: dict[str, Any]
    template: str


def default_tool_profile(task: TaskNode) -> str:
    if task.family in MLIR_AGENT_FAMILIES:
        return "mlir_agent"
    return "generic_contextual_edit"


def resolve_prompt_bundle(
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task: TaskNode,
    *,
    prompt_backend: str,
    agent: str | None = None,
) -> ResolvedPromptBundle:
    tool_profile = default_tool_profile(task)
    context = _build_context(
        capabilities=capabilities,
        support_plan=support_plan,
        task=task,
        prompt_backend=prompt_backend,
        agent=agent,
        tool_profile=tool_profile,
    )
    section_chunks: dict[str, list[str]] = {name: [] for name in PROMPT_SECTION_ORDER}
    section_sources: dict[str, list[str]] = {name: [] for name in PROMPT_SECTION_ORDER}
    fragments: list[PromptFragment] = []
    for fragment_file in _collect_prompt_fragment_files(capabilities, task):
        if not _fragment_applies(
            fragment_file.metadata,
            capabilities,
            support_plan,
            task,
            prompt_backend,
            tool_profile,
        ):
            continue
        section = fragment_file.metadata["section"]
        merge_mode = fragment_file.metadata.get("merge", "append")
        rendered = _render_template(fragment_file.template, context).strip()
        if not rendered:
            continue
        if merge_mode == "replace":
            section_chunks[section] = [rendered]
            section_sources[section] = [_repo_relative(fragment_file.path)]
        elif merge_mode == "prepend":
            section_chunks[section] = [rendered, *section_chunks[section]]
            section_sources[section] = [_repo_relative(fragment_file.path), *section_sources[section]]
        else:
            section_chunks[section].append(rendered)
            section_sources[section].append(_repo_relative(fragment_file.path))
        fragments.append(
            PromptFragment(
                source=_repo_relative(fragment_file.path),
                scope=fragment_file.scope,
                section=section,
                merge_mode=merge_mode,
            )
        )

    sections = [
        PromptSection(
            name=section,
            heading=PROMPT_SECTION_HEADINGS[section],
            content="\n\n".join(chunk for chunk in section_chunks[section] if chunk).strip(),
            sources=section_sources[section],
        )
        for section in PROMPT_SECTION_ORDER
        if section_chunks[section]
    ]
    flattened_prompt = render_sectioned_markdown(
        title=task.title,
        sections=sections,
        metadata={
            "Task ID": task.id,
            "Phase": task.phase,
            "Family": task.family,
            "Tool Profile": tool_profile,
            "Prompt Backend": prompt_backend,
        },
    )
    return ResolvedPromptBundle(
        backend=prompt_backend,
        tool_profile=tool_profile,
        agent=agent,
        sections=sections,
        fragments=fragments,
        flattened_prompt=flattened_prompt,
    )


def render_sectioned_markdown(
    *,
    title: str,
    sections: list[PromptSection],
    metadata: dict[str, str] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    if metadata:
        for key, value in metadata.items():
            if value:
                lines.append(f"- **{key}:** {value}")
        lines.append("")
    for section in sections:
        lines.append(f"## {section.heading}")
        lines.append(section.content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_provider_config(agent_name: str) -> dict[str, Any]:
    loader = _try_mlir_agent_provider_loader()
    if loader is not None:
        config = loader(agent_name, str(MLIR_AGENT_CONFIGS_ROOT))
    else:
        path = MLIR_AGENT_CONFIGS_ROOT / "agents" / f"{agent_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Agent config not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key_env = config.get("api_key_env", "")
        config["api_key"] = os.environ.get(key_env, "")
    config.pop("api_key", None)
    return config


def _build_context(
    *,
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task: TaskNode,
    prompt_backend: str,
    agent: str | None,
    tool_profile: str,
) -> dict[str, Any]:
    return {
        "target": to_dict(capabilities.identity),
        "platform": to_dict(capabilities.platform),
        "execution": to_dict(capabilities.execution),
        "isa": to_dict(capabilities.isa),
        "operations": to_dict(capabilities.operations),
        "tiles": to_dict(capabilities.tiles),
        "memory": to_dict(capabilities.memory),
        "numeric": to_dict(capabilities.numeric),
        "runtime": to_dict(capabilities.runtime),
        "verification": to_dict(capabilities.verification),
        "deployment": to_dict(capabilities.deployment) if capabilities.deployment is not None else None,
        "capabilities": to_dict(capabilities),
        "support_plan": to_dict(support_plan),
        "task": to_dict(task),
        "task_actions": task.actions,
        "task_evidence": [to_dict(item) for item in task.evidence],
        "task_write_scope": task.write_scope,
        "task_acceptance_checks": task.acceptance_checks,
        "prompt_backend": prompt_backend,
        "agent": agent,
        "tool_profile": tool_profile,
        "prompt_section_order": list(PROMPT_SECTION_ORDER),
    }


def _collect_prompt_fragment_files(
    capabilities: TargetCapabilities,
    task: TaskNode,
) -> list[_PromptFragmentFile]:
    target_root = Path(capabilities.capability_path).resolve().parent / "prompts"
    candidate_groups = [
        ("base", PROMPT_LIBRARY_ROOT / "base"),
        (f"family:{task.family}", PROMPT_LIBRARY_ROOT / "families" / task.family),
        (f"phase:{task.phase}", PROMPT_LIBRARY_ROOT / "phases" / task.phase),
        (f"task:{task.id}", PROMPT_LIBRARY_ROOT / "tasks" / task.id),
        (f"task:{task.id}", PROMPT_LIBRARY_ROOT / "tasks" / f"{task.id}.md"),
        ("target", target_root / "target"),
    ]
    if capabilities.deployment is not None:
        candidate_groups.append(
            (
                f"overlay:{capabilities.deployment.name}",
                target_root / "overlays" / capabilities.deployment.name,
            )
        )
    fragments: list[_PromptFragmentFile] = []
    for scope, path in candidate_groups:
        fragments.extend(_load_fragment_group(path, scope))
    return fragments


def _load_fragment_group(path: Path, scope: str) -> list[_PromptFragmentFile]:
    if path.is_file():
        return [_load_fragment_file(path, scope)]
    if not path.exists():
        return []
    return [_load_fragment_file(fragment_path, scope) for fragment_path in sorted(path.glob("*.md"))]


def _load_fragment_file(path: Path, scope: str) -> _PromptFragmentFile:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError(f"{path}: expected YAML frontmatter delimited by --- markers")
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: prompt frontmatter must be a YAML mapping")
    unknown_keys = sorted(set(metadata) - ALLOWED_FRONTMATTER_KEYS)
    if unknown_keys:
        raise ValueError(f"{path}: unknown prompt frontmatter keys: {', '.join(unknown_keys)}")
    section = metadata.get("section")
    if section not in PROMPT_SECTION_ORDER:
        raise ValueError(f"{path}: section must be one of {', '.join(PROMPT_SECTION_ORDER)}")
    merge_mode = metadata.get("merge", "append")
    if merge_mode not in PROMPT_MERGE_MODES:
        raise ValueError(f"{path}: merge must be one of {', '.join(sorted(PROMPT_MERGE_MODES))}")
    return _PromptFragmentFile(
        path=path,
        scope=scope,
        metadata={"section": section, "merge": merge_mode, **metadata},
        template=match.group(2).strip(),
    )


def _fragment_applies(
    metadata: dict[str, Any],
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task: TaskNode,
    prompt_backend: str,
    tool_profile: str,
) -> bool:
    deployment_name = capabilities.deployment.name if capabilities.deployment is not None else None
    return all(
        [
            _matches_selector(metadata.get("families"), task.family),
            _matches_selector(metadata.get("phases"), task.phase),
            _matches_selector(metadata.get("task_ids"), task.id),
            _matches_selector(metadata.get("targets"), capabilities.identity.name),
            _matches_selector(metadata.get("deployment_profiles"), deployment_name),
            _matches_selector(metadata.get("prompt_backends"), prompt_backend),
            _matches_selector(metadata.get("tool_profiles"), tool_profile),
            _matches_overlap(metadata.get("integration_styles"), support_plan.integration_styles),
        ]
    )


def _matches_selector(value: Any, actual: str | None) -> bool:
    options = _as_string_list(value)
    if not options:
        return True
    if actual is None:
        return False
    return actual in options


def _matches_overlap(value: Any, actual_values: list[str]) -> bool:
    options = _as_string_list(value)
    if not options:
        return True
    return bool(set(options) & set(actual_values))


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("Prompt frontmatter selectors must be a string or list of strings")


def _render_template(template: str, context: dict[str, Any]) -> str:
    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["bullet_list"] = _bullet_list
    env.filters["code_list"] = _code_list
    env.filters["comma_list"] = _comma_list
    return env.from_string(template).render(**context)


def _bullet_list(items: list[Any]) -> str:
    rendered = []
    for item in items:
        rendered.append(f"- {item}")
    return "\n".join(rendered)


def _code_list(items: list[Any]) -> str:
    rendered = []
    for item in items:
        rendered.append(f"- `{item}`")
    return "\n".join(rendered)


def _comma_list(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _try_mlir_agent_provider_loader():
    module_path = REPO_ROOT / "projects" / "mlirAgent" / "src" / "mlirAgent" / "evolve" / "providers.py"
    if not module_path.exists():
        return None
    package_root = module_path.parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    spec = importlib.util.spec_from_file_location("mlirAgent.evolve.providers", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "load_agent_config", None)
