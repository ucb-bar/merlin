"""TargetGen orchestration: ingest -> evidence -> synthesize -> generate -> validate.

Deterministic and LLM-free. ``build`` returns a :class:`BuildResult` and (when ``out`` is
given) writes the generated target repo to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.artifacts import Artifact, write_all, yaml_artifact
from .ingest import build_manifest
from .evidence import build_evidence, render_markdown
from .synthesize import (
    synthesize_target_contract,
    synthesize_dialect_plan,
    synthesize_runtime_adapter_plan,
    synthesize_zephyr_plan,
    synthesize_llvm_extension_plan,
)
from .generate import target_repo, xdsl, mlir_scaffold, zephyr_module, runtime_adapter, llvm_plan
from .validate import validate_plans, render_validation_report

# Layers that --emit can select. "contract-only" suppresses all of them.
EMIT_LAYERS = {"xdsl", "mlir", "zephyr", "runtime", "llvm-plan"}
DEFAULT_EMIT = ["xdsl", "mlir", "zephyr", "llvm-plan", "runtime"]


@dataclass
class BuildResult:
    """Result of a TargetGen build."""

    target: str
    out: Path | None
    plans: dict[str, Any]
    schema_problems: list[str]
    emit: list[str]
    written: list[Path] = field(default_factory=list)
    evidence_concepts: list[str] = field(default_factory=list)


def _normalize_emit(emit: list[str] | None) -> tuple[list[str], bool]:
    """Return (selected_layers, contract_only)."""
    if not emit:
        return list(DEFAULT_EMIT), False
    items = [e.strip() for e in emit if e.strip()]
    if "contract-only" in items:
        return [], True
    selected = [e for e in items if e in EMIT_LAYERS]
    return selected, False


def build(
    target_name: str,
    source_dir: str | None = None,
    examples_dir: str | None = None,
    scala_root: str | None = None,
    out: str | Path | None = None,
    emit: list[str] | None = None,
    source_urls: list[str] | None = None,
    branch: str | None = None,
    commit: str | None = None,
) -> BuildResult:
    """Run the full TargetGen pipeline for ``target_name``."""
    selected, contract_only = _normalize_emit(emit)

    # 1. ingest -> manifest ; 2. evidence
    manifest = build_manifest(
        target_name, source_dir=source_dir, examples_dir=examples_dir,
        scala_root=scala_root, source_urls=source_urls, branch=branch, commit=commit,
    )
    evidence = build_evidence(manifest)

    # 3. synthesize the five plans
    tc = synthesize_target_contract(evidence, target_name)
    dp = synthesize_dialect_plan(evidence, tc)
    rap = synthesize_runtime_adapter_plan(evidence, tc)
    zp = synthesize_zephyr_plan(evidence, tc)
    lp = synthesize_llvm_extension_plan(evidence, tc)
    plans = {
        "target_contract": tc,
        "dialect_plan": dp,
        "runtime_adapter_plan": rap,
        "zephyr_plan": zp,
        "llvm_extension_plan": lp,
    }

    # 4. validate (pre-write)
    schema_problems = validate_plans(plans)

    # 5. reports
    evidence_md = render_markdown(evidence)
    validation_md = render_validation_report(
        target_name, plans, schema_problems,
        emit=(["contract-only"] if contract_only else selected),
    )

    # 6. assemble artifacts
    artifacts: list[Artifact] = []
    artifacts += target_repo.generate_skeleton(target_name)
    artifacts += target_repo.emit_contracts(plans)
    artifacts += target_repo.emit_docs(evidence_md, validation_md, target_name)
    artifacts.append(yaml_artifact("contracts/target_source_manifest.yaml", manifest.to_dict(),
                                   header="Source manifest (target_source_manifest.schema.yaml)."))
    artifacts.append(yaml_artifact("docs/evidence_index.yaml", evidence.to_index_dict(),
                                   header="Evidence index (evidence_report.schema.yaml)."))

    if not contract_only:
        if "xdsl" in selected:
            artifacts += xdsl.generate(dp)
        if "mlir" in selected:
            artifacts += mlir_scaffold.generate(dp)
        if "zephyr" in selected:
            artifacts += zephyr_module.generate(zp)
        if "runtime" in selected:
            artifacts += runtime_adapter.generate(rap)
        if "llvm-plan" in selected:
            artifacts += llvm_plan.generate(lp)

    # 7-8. write + backfill AGENT.md
    written: list[Path] = []
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        written = write_all(artifacts, out)
        written += target_repo.backfill_agent_md(out)

    return BuildResult(
        target=target_name,
        out=Path(out) if out is not None else None,
        plans=plans,
        schema_problems=schema_problems,
        emit=(["contract-only"] if contract_only else selected),
        written=written,
        evidence_concepts=sorted(evidence.concept_names()),
    )
