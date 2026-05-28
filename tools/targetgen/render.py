"""TargetGen presentation helpers — capability-draft and modification-map rendering.

Pure functions over loaded TargetGen models; no side effects.
"""

from __future__ import annotations

from pathlib import Path


def _render_capability_draft(classification) -> str:
    """Render a *loadable* capability draft.

    Delegates to ``targetgen.intake.draft.render_loadable_draft_yaml`` so the
    CLI emits the same draft schema the MCP `targetgen_create_capability_draft`
    tool produces. The earlier sketch-only renderer (just `target` +
    `_targetgen_intake`) was rejected by `load_capability_spec`, blocking
    the modification-map and plan steps when chained from the CLI.
    """
    from targetgen.intake.draft import render_loadable_draft_yaml

    return render_loadable_draft_yaml(classification)


def _render_modification_map_md(modmap) -> str:
    lines: list[str] = []
    lines.append(f"# Modification map: {modmap.target}")
    lines.append("")
    lines.append(f"**Primary integration:** `{modmap.primary_integration}`  ")
    styles = ", ".join(f"`{s}`" for s in modmap.integration_styles)
    lines.append(f"**Integration styles:** {styles}")
    lines.append("")
    for idx, stage in enumerate(modmap.stages, start=1):
        lines.append(f"## {idx}. {stage.stage}")
        lines.append("")
        lines.append(f"- **applies:** `{stage.applies}`")
        lines.append(f"- **repo_root:** `{stage.repo_root}`")
        lines.append(f"- **reason:** {stage.reason}")
        if stage.read_paths:
            lines.append("- **read paths:**")
            for path in stage.read_paths:
                lines.append(f"  - `{path}`")
        if stage.write_paths:
            lines.append("- **write paths:**")
            for path in stage.write_paths:
                lines.append(f"  - `{path}`")
        if stage.validation_commands:
            lines.append("- **validation:**")
            for cmd in stage.validation_commands:
                lines.append(f"  - `{cmd}`")
        if stage.blocking_questions:
            lines.append("- **blocking questions:**")
            for q in stage.blocking_questions:
                lines.append(f"  - {q}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
