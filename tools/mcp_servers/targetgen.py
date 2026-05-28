"""Pure-Python TargetGen tool implementations consumed by the MCP server.

Each tool is a synchronous function returning a JSON-serialisable dict. The
MCP layer (``server.py``) wraps these in async tool handlers. Splitting keeps
the planner logic testable without an MCP client.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from targetgen import build_support_plan, load_capability_spec, load_deployment_overlay
from targetgen.audit import TARGETGEN_STYLE_EVIDENCE, audit_claim
from targetgen.baseline import compare_to_deterministic
from targetgen.explore import explore_target
from targetgen.intake import build_source_inventory, classify_inventory
from targetgen.intake.draft import render_loadable_draft, render_loadable_draft_yaml
from targetgen.model import (
    PIPELINE_STAGES,
    Classification,
    SourceFinding,
    SourceInventory,
    SourceRepository,
)
from targetgen.stage_map import build_modification_map


class ToolError(RuntimeError):
    """Raised when an MCP tool invocation fails for a recoverable reason."""


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _ingest_source(args: dict[str, Any]) -> dict[str, Any]:
    target = _require_str(args, "target_name")
    sources = _require_list(args, "source_paths")
    inv = build_source_inventory(target=target, sources=[Path(s) for s in sources])
    return asdict(inv)


def _classify_target(args: dict[str, Any]) -> dict[str, Any]:
    target = _require_str(args, "target_name")
    inv: SourceInventory
    if "inventory_path" in args:
        raw = json.loads(Path(args["inventory_path"]).read_text())
        inv = SourceInventory(
            target=raw["target"],
            repositories=[SourceRepository(**r) for r in raw["repositories"]],
            findings=[SourceFinding(**f) for f in raw["findings"]],
            detected_source_kinds=list(raw["detected_source_kinds"]),
            missing_information=list(raw.get("missing_information", [])),
        )
    elif "source_paths" in args:
        sources = _require_list(args, "source_paths")
        inv = build_source_inventory(target=target, sources=[Path(s) for s in sources])
    else:
        raise ToolError("classify_target requires either 'inventory_path' or 'source_paths'")
    classification = classify_inventory(inv)
    return asdict(classification)


def _create_capability_draft(args: dict[str, Any]) -> dict[str, Any]:
    """Classify a target's source tree and write a *loadable* capability
    draft to disk so the downstream tools can chain off it.

    The draft is intentionally minimal — every operation, memory space,
    numeric type triple, and verification oracle is a stub the agent
    must replace before promotion. The header of the file lists the
    unresolved fields explicitly.
    """
    target = _require_str(args, "target_name")
    out_path = Path(_require_str(args, "out_path"))
    overwrite = bool(args.get("overwrite", False))

    if "inventory_path" in args:
        raw = json.loads(Path(args["inventory_path"]).read_text())
        inv = SourceInventory(
            target=raw["target"],
            repositories=[SourceRepository(**r) for r in raw["repositories"]],
            findings=[SourceFinding(**f) for f in raw["findings"]],
            detected_source_kinds=list(raw["detected_source_kinds"]),
            missing_information=list(raw.get("missing_information", [])),
        )
    elif "source_paths" in args:
        sources = _require_list(args, "source_paths")
        inv = build_source_inventory(target=target, sources=[Path(s) for s in sources])
    else:
        raise ToolError("create_capability_draft requires 'inventory_path' or 'source_paths'")

    if out_path.exists() and not overwrite:
        raise ToolError(f"out_path already exists: {out_path}. Pass overwrite=true to replace it.")

    classification = classify_inventory(inv)
    yaml_text = render_loadable_draft_yaml(classification)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    # Round-trip through the loader so the agent gets an immediate sanity
    # signal — if the draft is somehow not loadable, this raises a
    # ToolError instead of leaving a broken file behind.
    try:
        load_capability_spec(out_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(f"draft was written but failed loader validation: {exc}") from exc

    return {
        "target": target,
        "capability_path": str(out_path),
        "primary_integration": classification.primary_integration,
        "targetgen_styles": list(classification.targetgen_styles),
        "source_styles": list(classification.source_styles),
        "confidence": classification.confidence,
        "rationales": list(classification.rationales),
        "missing_information": list(classification.missing_information),
        "loader_validated": True,
        "next_step": (
            f"Refine the stubs marked UNRESOLVED in {out_path}, then call "
            f"`targetgen_plan_target(capability_path={out_path!s})` and "
            f"`targetgen_get_modification_map(capability_path={out_path!s})`."
        ),
    }


def _plan_target(args: dict[str, Any]) -> dict[str, Any]:
    capabilities = _load_capabilities(args)
    plan = build_support_plan(capabilities)
    return asdict(plan)


def _get_modification_map(args: dict[str, Any]) -> dict[str, Any]:
    capabilities = _load_capabilities(args)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
    return asdict(modmap)


def _get_allowed_patch_surfaces(args: dict[str, Any]) -> dict[str, Any]:
    stage_name = _require_str(args, "stage")
    if stage_name not in PIPELINE_STAGES:
        raise ToolError(f"Unknown stage {stage_name!r}; expected one of {list(PIPELINE_STAGES)}")
    capabilities = _load_capabilities(args)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
    stage = next((s for s in modmap.stages if s.stage == stage_name), None)
    if stage is None:  # defensive — PIPELINE_STAGES enforces presence
        raise ToolError(f"Stage {stage_name!r} not present in modification map")
    return {
        "target": modmap.target,
        "stage": stage.stage,
        "applies": stage.applies,
        "repo_root": stage.repo_root,
        "allowed_write_paths": stage.write_paths,
        "read_paths": stage.read_paths,
        "validation_commands": stage.validation_commands,
        "blocking_questions": stage.blocking_questions,
        "forbidden_unless_approved": _forbidden_paths(stage),
    }


def _get_validation_commands(args: dict[str, Any]) -> dict[str, Any]:
    capabilities = _load_capabilities(args)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
    return {
        "target": modmap.target,
        "validation_commands": [
            {"stage": s.stage, "commands": s.validation_commands} for s in modmap.stages if s.validation_commands
        ],
    }


def _list_pipeline_stages(_: dict[str, Any]) -> dict[str, Any]:
    return {"stages": list(PIPELINE_STAGES)}


def _explore_target(args: dict[str, Any]) -> dict[str, Any]:
    """Return raw scanner evidence + file pointers for agent-driven reasoning.

    Unlike ``classify_target``, this tool does not collapse evidence into
    integration styles. The agent reads the returned ``key_files`` with its
    own ``Read``/``Grep``/``Glob`` tools, then proposes a classification
    via ``targetgen_propose_modifications``.
    """
    target = _require_str(args, "target_name")
    sources = _require_list(args, "source_paths")
    report = explore_target(target, [Path(s) for s in sources])
    return {
        "target": report.target,
        "source_paths": report.source_paths,
        "directory_summaries": [asdict(s) for s in report.directory_summaries],
        "key_files": asdict(report.key_files),
        "findings_by_kind": report.findings_by_kind,
        "detected_source_kinds": report.detected_source_kinds,
        "findings": [asdict(f) for f in report.findings],
        "next_steps": report.next_steps,
    }


def _propose_modifications(args: dict[str, Any]) -> dict[str, Any]:
    """Build a modification map driven by the agent's classification, not the
    deterministic classifier, then audit the claim against scanner evidence.

    Required arguments:
      target_name              : str
      source_paths             : list[str]   (for the audit pass)
      targetgen_styles         : list[str]   (subset of the four canonical)
      source_styles            : list[str]   (any of the eight source-facing)
      primary_integration      : str         (one of targetgen_styles)

    Optional:
      rationale                : str
      confidence               : float (0.0-1.0)
    """
    target = _require_str(args, "target_name")
    sources = _require_list(args, "source_paths")
    tg_styles = _require_list(args, "targetgen_styles")
    src_styles = args.get("source_styles") or []
    if not isinstance(src_styles, list):
        raise ToolError("source_styles must be a list of strings")
    primary = _require_str(args, "primary_integration")
    rationale = args.get("rationale", "")
    confidence_raw = args.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"confidence must be a number, got {confidence_raw!r}") from exc

    if primary not in tg_styles:
        raise ToolError(f"primary_integration {primary!r} must appear in targetgen_styles {tg_styles!r}")
    unknown_tg = [s for s in tg_styles if s not in TARGETGEN_STYLE_EVIDENCE]
    if unknown_tg:
        raise ToolError(
            f"unknown targetgen_styles {unknown_tg!r}; valid: " f"{sorted(TARGETGEN_STYLE_EVIDENCE.keys())}"
        )

    inventory = build_source_inventory(target=target, sources=[Path(s) for s in sources])
    audit = audit_claim(
        claimed_targetgen_styles=tg_styles,
        claimed_source_styles=src_styles,
        inventory=inventory,
    )

    # Synthesize a TargetCapabilities object from the agent's classification by
    # rendering a loadable draft and round-tripping through the loader. This
    # reuses the existing draft synthesizer's defaults for fields the agent
    # does not provide (memory, numeric, verification stubs).
    synthetic = Classification(
        target=target,
        source_styles=list(src_styles),
        targetgen_styles=list(tg_styles),
        primary_integration=primary,
        confidence=confidence,
        missing_information=[],
        rationales=[rationale] if rationale else [],
    )
    draft_payload = render_loadable_draft(synthetic)
    draft_payload.pop("_unresolved_intake_notes", None)

    out_path_arg = args.get("out_path")
    persisted_path: str | None = None
    if out_path_arg:
        out_path = Path(out_path_arg)
        if out_path.exists() and not bool(args.get("overwrite", False)):
            raise ToolError(f"out_path already exists: {out_path}. Pass overwrite=true to replace it.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(draft_payload, sort_keys=False), encoding="utf-8")
        capabilities = load_capability_spec(out_path)
        persisted_path = str(out_path)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(draft_payload, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)
        try:
            capabilities = load_capability_spec(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    plan = build_support_plan(capabilities)
    # The agent's targetgen_styles override the planner's — that is the whole
    # point of this tool. We pass the agent's list explicitly.
    modmap = build_modification_map(capabilities, targetgen_styles=tg_styles)

    baseline = compare_to_deterministic(
        target_name=target,
        source_paths=[Path(s) for s in sources],
        agent_targetgen_styles=tg_styles,
        agent_source_styles=src_styles,
        agent_primary_integration=primary,
        agent_modification_map=modmap,
    )

    return {
        "target": target,
        "agent_classification": {
            "targetgen_styles": list(tg_styles),
            "source_styles": list(src_styles),
            "primary_integration": primary,
            "confidence": confidence,
            "rationale": rationale,
        },
        "capability_path": persisted_path,
        "modification_map": asdict(modmap),
        "support_plan": asdict(plan),
        "audit": {
            "overall_status": audit.overall_status,
            "findings": [asdict(f) for f in audit.findings],
            "unused_evidence": audit.unused_evidence,
        },
        "baseline_comparison": asdict(baseline),
        "next_step": (
            "Pick the first applies==true stage in modification_map.stages. "
            "Address audit findings of severity 'error' before editing. "
            f"Current audit status: {audit.overall_status}. "
            f"Baseline-vs-agent style agreement: "
            f"targetgen={baseline.targetgen_styles_agreement}, "
            f"primary_match={baseline.primary_integration_match}."
        ),
    }


_XPURT_HINT_VOCABULARY = frozenset(
    {
        "prefer_coarser",
        "prefer_finer",
        "consider_fuse_with_pred",
        "consider_split_backend",
    }
)


def _is_xpurt_hint(value: str) -> bool:
    """Closed-set hint check that allows the parameterised pin_target=<m>."""
    if value in _XPURT_HINT_VOCABULARY:
        return True
    return value.startswith("pin_target=") and len(value) > len("pin_target=")


def _ingest_xpurt_feedback(args: dict[str, Any]) -> dict[str, Any]:
    """Persist XPU-RT scheduler feedback into a Merlin output dir.

    The companion ``scripts/merlin_adapter.py --emit-feedback`` in XPU-RT
    produces ``xpurt_feedback.json`` next to ``schedule.json``. This tool
    reads that payload (or accepts it inline) and writes the canonical
    persisted form to ``<merlin_dir>/breakdowns/feedback.json``, where
    Merlin's QNN / SpaceMit compile paths read it (inert if absent — see
    docs/merlin_integration.md).

    Merge semantics: if a previously persisted feedback.json carries the
    same ``run_id``, this call accumulates per-dispatch hints (set union)
    rather than overwriting. A new ``run_id`` replaces the file
    wholesale. This is what makes incremental streaming feedback from the
    on-board scheduler_runner safe to call repeatedly.

    Required arguments:
      merlin_dir      : str — path to the merlin output directory
                              (the one with breakdowns/profiled_manifest.json).
    One of:
      payload_path    : str — path to xpurt_feedback.json on disk.
      payload         : dict — inline payload (when streaming over MCP).
    """
    merlin_dir = Path(_require_str(args, "merlin_dir"))
    if not merlin_dir.is_dir():
        raise ToolError(f"merlin_dir does not exist: {merlin_dir}")

    payload: dict[str, Any]
    if "payload_path" in args:
        payload_path = Path(_require_str(args, "payload_path"))
        if not payload_path.is_file():
            raise ToolError(f"payload_path does not exist: {payload_path}")
        try:
            payload = json.loads(payload_path.read_text())
        except json.JSONDecodeError as exc:
            raise ToolError(f"payload_path is not valid JSON: {exc}") from exc
    elif "payload" in args:
        payload = args["payload"]
        if not isinstance(payload, dict):
            raise ToolError("payload must be a JSON object")
    else:
        raise ToolError("ingest_xpurt_feedback requires either 'payload_path' or 'payload'")

    if payload.get("schema_version") != 1:
        raise ToolError(f"unsupported schema_version {payload.get('schema_version')!r}; " f"expected 1")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ToolError("payload.run_id must be a non-empty string")
    dispatches = payload.get("dispatches", {})
    if not isinstance(dispatches, dict):
        raise ToolError("payload.dispatches must be an object")

    # Validate hints + cross-check dispatch IDs against profiled_manifest
    # if present (a soft warning rather than a hard error so streaming
    # feedback that names dispatches from a different replica still flows
    # through).
    manifest_path = merlin_dir / "breakdowns" / "profiled_manifest.json"
    known_dispatches: set[str] = set()
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            known_dispatches = set(manifest.get("dispatches", {}).keys())
        except json.JSONDecodeError:
            pass  # don't block on a malformed manifest
    unknown_dispatches: list[str] = []
    for d_id, entry in dispatches.items():
        if not isinstance(entry, dict):
            raise ToolError(f"dispatches.{d_id} must be an object")
        hints = entry.get("hints") or []
        if not isinstance(hints, list):
            raise ToolError(f"dispatches.{d_id}.hints must be a list")
        bad = [h for h in hints if not isinstance(h, str) or not _is_xpurt_hint(h)]
        if bad:
            raise ToolError(
                f"dispatches.{d_id}.hints contains unknown values {bad!r}; "
                f"vocabulary: {sorted(_XPURT_HINT_VOCABULARY)} + "
                f"'pin_target=<machine>'"
            )
        if known_dispatches and d_id not in known_dispatches:
            unknown_dispatches.append(d_id)

    out_path = merlin_dir / "breakdowns" / "feedback.json"
    merged_payload = payload
    merged = False
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and existing.get("run_id") == run_id:
            merged_payload = _merge_xpurt_payloads(existing, payload)
            merged = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_payload, indent=2, sort_keys=True))

    # Hint-count summary for the LLM caller / streaming daemon.
    hint_counts: dict[str, int] = {}
    for entry in merged_payload.get("dispatches", {}).values():
        for h in entry.get("hints") or []:
            key = "pin_target" if h.startswith("pin_target=") else h
            hint_counts[key] = hint_counts.get(key, 0) + 1

    return {
        "merlin_dir": str(merlin_dir),
        "feedback_path": str(out_path),
        "run_id": run_id,
        "merged_with_existing": merged,
        "n_dispatches_with_hints": len(merged_payload.get("dispatches", {})),
        "hint_counts": hint_counts,
        "model_signals": merged_payload.get("model_signals", {}),
        "unknown_dispatches": unknown_dispatches,
    }


def _merge_xpurt_payloads(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Set-union the per-dispatch hint lists; later model_signals wins."""
    merged = dict(existing)
    merged["model_signals"] = incoming.get("model_signals", existing.get("model_signals", {}))
    merged["source_schedule"] = incoming.get("source_schedule", existing.get("source_schedule"))
    out_dispatches: dict[str, Any] = dict(existing.get("dispatches", {}))
    for d_id, new_entry in (incoming.get("dispatches") or {}).items():
        if d_id not in out_dispatches:
            out_dispatches[d_id] = dict(new_entry)
            continue
        cur = dict(out_dispatches[d_id])
        cur_hints = list(cur.get("hints") or [])
        for h in new_entry.get("hints") or []:
            if h not in cur_hints:
                cur_hints.append(h)
        cur["hints"] = cur_hints
        # Numeric fields: take the most recent reading (incoming wins).
        for k in ("current_target", "idle_fraction", "transfer_cost_ratio", "deadline_slack_us", "rationale"):
            if k in new_entry:
                cur[k] = new_entry[k]
        out_dispatches[d_id] = cur
    merged["dispatches"] = out_dispatches
    return merged


def _forbidden_paths(stage) -> list[str]:
    """Paths that require explicit operator approval before edit."""
    forbidden = ["third_party/iree_bar/", "third_party/iree_bar/third_party/llvm-project/"]
    # If the stage's allowed write paths already include submodule edits, those
    # are intentionally allowed for that stage and removed from the forbidden list.
    return [p for p in forbidden if not any(w.startswith(p) for w in stage.write_paths)]


def _load_capabilities(args: dict[str, Any]):
    capability_path = _require_str(args, "capability_path")
    if not Path(capability_path).is_file():
        raise ToolError(f"capability_path does not exist or is not a file: {capability_path!r}")
    try:
        capabilities = load_capability_spec(capability_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(f"failed to load capability spec {capability_path!r}: {exc}") from exc
    overlay = args.get("overlay_path")
    if overlay:
        if not Path(overlay).is_file():
            raise ToolError(f"overlay_path does not exist or is not a file: {overlay!r}")
        try:
            capabilities.deployment = load_deployment_overlay(overlay)
        except (FileNotFoundError, ValueError) as exc:
            raise ToolError(f"failed to load overlay {overlay!r}: {exc}") from exc
    return capabilities


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"missing or empty required argument {key!r}")
    return value


def _require_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key)
    if not isinstance(value, list) or not value:
        raise ToolError(f"missing or empty required argument {key!r}")
    if not all(isinstance(v, str) for v in value):
        raise ToolError(f"argument {key!r} must be a list of strings")
    return value


_CAPABILITY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "capability_path": {
            "type": "string",
            "description": "Path to a TargetGen capability.yaml",
        },
        "overlay_path": {
            "type": "string",
            "description": "Optional deployment overlay path",
        },
    },
    "required": ["capability_path"],
}


TOOL_REGISTRY: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="targetgen_ingest_source",
        description=(
            "Walk one or more source paths and return a SourceInventory of "
            "detected MLIR/CMake/HAL/Chipyard/RTL/SystemC/docs findings. "
            "Read-only — does not mutate the repo."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "source_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["target_name", "source_paths"],
        },
        handler=_ingest_source,
    ),
    ToolDefinition(
        name="targetgen_classify_target",
        description=(
            "Classify a target into source-facing styles and the four canonical "
            "TargetGen styles (runtime_hal, structured_text_isa, "
            "post_global_plugin, llvm_ukernel)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "source_paths": {"type": "array", "items": {"type": "string"}},
                "inventory_path": {"type": "string"},
            },
            "required": ["target_name"],
        },
        handler=_classify_target,
    ),
    ToolDefinition(
        name="targetgen_create_capability_draft",
        description=(
            "Classify a held-out target and write a loadable capability "
            "draft YAML to ``out_path``. Use this BEFORE any tool that "
            "requires a capability_path (plan_target, "
            "get_modification_map, get_allowed_patch_surfaces, "
            "get_validation_commands). The draft is a sketch — every "
            "stub flagged 'UNRESOLVED' in the file header must be "
            "filled in before the spec is promoted, but downstream "
            "tools accept it as-is so the agent can iterate."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "out_path": {
                    "type": "string",
                    "description": (
                        "Where to write capability.yaml. Recommended: "
                        "build/generated/targetgen/<target>/capability.draft.yaml"
                    ),
                },
                "source_paths": {"type": "array", "items": {"type": "string"}},
                "inventory_path": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["target_name", "out_path"],
        },
        handler=_create_capability_draft,
    ),
    ToolDefinition(
        name="targetgen_plan_target",
        description=(
            "Run the existing TargetGen planner against a capability.yaml and "
            "return the SupportPlan (integration styles, patch surfaces, "
            "verification manifest, task node ids)."
        ),
        input_schema=_CAPABILITY_INPUT_SCHEMA,
        handler=_plan_target,
    ),
    ToolDefinition(
        name="targetgen_get_modification_map",
        description=(
            "Return the per-stage ModificationMap for a capability.yaml. "
            "Each of the nine pipeline stages reports applies/reason/"
            "read_paths/write_paths/validation_commands/blocking_questions."
        ),
        input_schema=_CAPABILITY_INPUT_SCHEMA,
        handler=_get_modification_map,
    ),
    ToolDefinition(
        name="targetgen_get_allowed_patch_surfaces",
        description=(
            "Return the allowed write paths and validation commands for a "
            "single pipeline stage. Use this BEFORE editing compiler/, "
            "runtime/, third_party/iree_bar/, or LLVM."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "capability_path": {"type": "string"},
                "overlay_path": {"type": "string"},
                "stage": {
                    "type": "string",
                    "enum": list(PIPELINE_STAGES),
                },
            },
            "required": ["capability_path", "stage"],
        },
        handler=_get_allowed_patch_surfaces,
    ),
    ToolDefinition(
        name="targetgen_get_validation_commands",
        description=(
            "Return validation commands per stage for a capability.yaml. "
            "Always invoke through ./merlin (no direct cmake/ninja/uv calls)."
        ),
        input_schema=_CAPABILITY_INPUT_SCHEMA,
        handler=_get_validation_commands,
    ),
    ToolDefinition(
        name="targetgen_list_pipeline_stages",
        description=(
            "Return the canonical nine-stage Merlin pipeline used by the "
            "modification map and allowed-patch-surfaces tools."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_pipeline_stages,
    ),
    ToolDefinition(
        name="targetgen_explore_target",
        description=(
            "Walk a target's source tree and return raw scanner evidence + "
            "file pointers (README, build files, representative source files). "
            "Unlike `classify_target`, this tool does NOT collapse evidence "
            "into integration styles — it returns the data for the agent to "
            "reason about. Use the agent's own Read/Grep/Glob tools to drill "
            "into the returned `key_files`. After exploring, call "
            "`targetgen_propose_modifications` with your structured judgment."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "source_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["target_name", "source_paths"],
        },
        handler=_explore_target,
    ),
    ToolDefinition(
        name="targetgen_propose_modifications",
        description=(
            "Build a modification map from the AGENT's classification (not the "
            "deterministic classifier) and audit the claim against scanner "
            "evidence. Use this when `classify_target` mis-classifies a novel "
            "target, or when the agent's reading of README + source justifies "
            "a different integration. The audit cross-checks each claimed "
            "style against the scanner kinds that should support it; "
            "severity='error' findings should be resolved before editing. "
            "Optionally writes the synthesized capability YAML to `out_path` "
            "for chaining with `targetgen_get_allowed_patch_surfaces`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "source_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Source paths the audit will re-scan for evidence.",
                },
                "targetgen_styles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(TARGETGEN_STYLE_EVIDENCE.keys()),
                    },
                    "minItems": 1,
                    "description": "Subset of the four canonical TargetGen styles the agent claims apply.",
                },
                "source_styles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source-facing styles (chipyard_generator, rocc_accelerator, etc.).",
                },
                "primary_integration": {
                    "type": "string",
                    "enum": list(TARGETGEN_STYLE_EVIDENCE.keys()),
                },
                "rationale": {
                    "type": "string",
                    "description": "Short prose justifying the classification (recorded for audit).",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "out_path": {
                    "type": "string",
                    "description": "Optional path to persist the synthesized capability YAML.",
                },
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": [
                "target_name",
                "source_paths",
                "targetgen_styles",
                "primary_integration",
            ],
        },
        handler=_propose_modifications,
    ),
    ToolDefinition(
        name="ingest_xpurt_feedback",
        description=(
            "Ingest XPU-RT scheduler feedback into a Merlin output dir. "
            "Persists the payload to <merlin_dir>/breakdowns/feedback.json, "
            "where Merlin's QNN / SpaceMit compile paths read it (inert if "
            "absent). Same `run_id` accumulates per-dispatch hints (set "
            "union); a new `run_id` replaces the file. Pass either a "
            "`payload_path` (e.g. xpurt_feedback.json emitted by "
            "scripts/merlin_adapter.py --emit-feedback) or an inline "
            "`payload` dict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "merlin_dir": {
                    "type": "string",
                    "description": ("Path to merlin output dir (containing " "breakdowns/profiled_manifest.json)."),
                },
                "payload_path": {
                    "type": "string",
                    "description": "Path to xpurt_feedback.json on disk.",
                },
                "payload": {
                    "type": "object",
                    "description": (
                        "Inline xpurt_feedback payload (use when streaming " "feedback from a long-running daemon)."
                    ),
                },
            },
            "required": ["merlin_dir"],
        },
        handler=_ingest_xpurt_feedback,
    ),
)


def list_tool_definitions() -> tuple[ToolDefinition, ...]:
    return TOOL_REGISTRY


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"Unknown tool: {name!r}")
