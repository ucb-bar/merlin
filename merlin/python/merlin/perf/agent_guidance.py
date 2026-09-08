"""Expose optimization evidence and real compiler edit surfaces to an experiment agent.

The harness, not the agent, joins a whole-model bottleneck to the compiler source that is allowed to
change it.  A package may declare semantic ``optimization_surfaces`` in its manifest.  Every declared
file and symbol is checked against the package's actual Python AST and command component graph before
it is shown.  When the declaration is absent, a structural source index is still provided, but the
semantic join remains explicitly UNKNOWN rather than guessed from a filename.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .whole_model_report import OptimizationOpportunity, WholeModelReport


SCOPES = frozenset({"flag", "knob", "heuristic", "pass", "codegen"})
EFFECTS = frozenset({
    "placement", "layout", "dtype", "encoding", "quantization", "movement", "residency",
    "fusion", "synchronization", "issue", "tiling", "latency_hiding",
})
_OPPORTUNITY_EFFECTS = {
    "missing_occupancy": frozenset({"latency_hiding", "issue", "tiling"}),
    "accelerator_bubbles": frozenset({
        "synchronization", "issue", "tiling", "latency_hiding", "residency"}),
    "exposed_movement": frozenset({
        "movement", "residency", "layout", "encoding", "latency_hiding"}),
    "encoding_transitions": frozenset({"encoding", "layout", "fusion"}),
    "unknown_latency_hiding": frozenset({"latency_hiding", "issue"}),
    "unknown_encoding_transitions": frozenset({"encoding", "layout", "fusion"}),
}

_STRUCTURAL_EFFECTS = {
    "residency_restaged": frozenset({"movement", "residency"}),
    "memory_round_trip": frozenset({"movement", "residency", "fusion"}),
    "unfused_single_consumer": frozenset({"fusion", "movement"}),
}

# CCA axes are more precise than broad effects.  They let an agent move from "movement is high" to
# "this exact surface controls whether an intermediate materializes" without baking a target opcode or
# compiler filename into the harness.  The effect match remains as a compatibility path for packages
# frozen before this optional field existed.
_FINDING_AXES = {
    "host_memory_hotspot": frozenset({
        "communication.intermediate_materialized", "coverage.non_contraction_op_fraction",
        "layout.transpose_materialized", "compute.epilogue", "compute.activation_vectorization"}),
    "whole_model_lowering_declined": frozenset({
        "coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes",
        "envelope.calls_in_loop", "envelope.runtime_calls",
        "communication.intermediate_materialized", "dispatch.loop_offloaded"}),
    "inter_lane_boundary_pressure": frozenset({
        "communication.host_device_bytes", "communication.engine_engine_bytes",
        "communication.intermediate_materialized", "communication.copy_compute_overlap",
        "coverage.non_contraction_op_fraction"}),
    "heterogeneous_model_placement": frozenset({
        "coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes",
        "coverage.non_contraction_op_fraction"}),
    "memory_regime_pressure": frozenset({
        "memory.capacity_fit", "memory.onchip_resident",
        "dispatch.double_buffered_banks", "dispatch.dma_overlap"}),
    "arithmetic_expansion": frozenset({
        "compute.contraction_form", "coverage.claimed_mac_fraction",
        "coverage.unclaimed_op_classes"}),
    "movement_regression": frozenset({
        "communication.host_device_bytes", "communication.engine_engine_bytes",
        "communication.intermediate_materialized", "communication.resident_across_calls",
        "communication.copy_compute_overlap", "dispatch.dma_overlap",
        "memory.onchip_resident", "layout.transpose_materialized", "layout.operand_major"}),
    "unpriced_movement": frozenset({
        "communication.host_device_bytes", "communication.engine_engine_bytes",
        "memory.dma_pattern", "memory.onchip_resident"}),
    "synchronization_regression": frozenset({
        "communication.fences", "dispatch.dma_overlap", "simt.barriers_in_loop"}),
    "residency_restaged": frozenset({
        "compute.accumulator_resident", "spatial.accumulator_resident",
        "communication.resident_across_calls", "memory.onchip_resident"}),
    "memory_round_trip": frozenset({
        "communication.intermediate_materialized", "communication.resident_across_calls",
        "compute.accumulator_resident", "memory.onchip_resident"}),
    "unfused_single_consumer": frozenset({
        "compute.epilogue", "communication.intermediate_materialized",
        "layout.transpose_materialized"}),
    "unverified_encoding_transitions": frozenset({
        "layout.transpose_materialized", "layout.operand_major", "memory.access_pattern"}),
    "unknown_occupancy_and_overlap": frozenset({
        "dispatch.dma_overlap", "dispatch.double_buffered_banks",
        "dispatch.descriptor_reuse", "dispatch.loop_offloaded",
        "communication.copy_compute_overlap"}),
    "issued_movement_regression": frozenset({
        "memory.onchip_resident", "communication.resident_across_calls",
        "communication.intermediate_materialized", "layout.transpose_materialized"}),
    "unresolved_target_encoding": frozenset({
        "layout.operand_major", "layout.transpose_materialized", "memory.access_pattern"}),
    "dispatch_configuration_regression": frozenset({
        "dispatch.descriptor_reuse", "dispatch.loop_offloaded"}),
    "loop_offload_regression": frozenset({"dispatch.loop_offloaded"}),
    "artifact_synchronization_regression": frozenset({
        "communication.fences", "dispatch.dma_overlap", "simt.barriers_in_loop"}),
    "trace_regression": frozenset({
        "communication.resident_across_calls", "memory.onchip_resident",
        "communication.fences", "dispatch.descriptor_reuse", "compute.epilogue"}),
}

_OPPORTUNITY_AXES = {
    "missing_occupancy": _FINDING_AXES["unknown_occupancy_and_overlap"],
    "accelerator_bubbles": _FINDING_AXES["unknown_occupancy_and_overlap"],
    "exposed_movement": _FINDING_AXES["movement_regression"],
    "encoding_transitions": _FINDING_AXES["unverified_encoding_transitions"],
    "unknown_latency_hiding": _FINDING_AXES["unknown_occupancy_and_overlap"],
    "unknown_encoding_transitions": _FINDING_AXES["unverified_encoding_transitions"],
}

# The generic mechanism checklist for a complete-model optimization loop.  This is not a list of
# target tricks: every row is expressed in the cross-target CCA vocabulary.  Its purpose is to make
# omissions visible.  A row is covered only when the current artifact exposes evidence *and* the
# candidate package declares a real AST surface for one of the controlling axes.
_GAP_AXES = {
    "whole_model_placement_and_coverage": frozenset({
        "coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes",
        "coverage.non_contraction_op_fraction", "envelope.calls_in_loop",
        "envelope.runtime_calls"}),
    "arithmetic_lowering": frozenset({
        "compute.contraction_form", "compute.reduction_form", "compute.epilogue",
        "compute.activation_vectorization"}),
    "encoding_and_layout": frozenset({
        "layout.operand_major", "layout.transpose_materialized", "memory.access_pattern"}),
    "movement_and_materialization": frozenset({
        "communication.host_device_bytes", "communication.engine_engine_bytes",
        "communication.intermediate_materialized", "memory.onchip_resident"}),
    "residency_across_operations": frozenset({
        "compute.accumulator_resident", "spatial.accumulator_resident",
        "communication.resident_across_calls", "memory.onchip_resident"}),
    "fusion_and_host_boundaries": frozenset({
        "compute.epilogue", "communication.intermediate_materialized",
        "coverage.non_contraction_op_fraction", "envelope.calls_in_loop"}),
    "dispatch_and_loop_offload": frozenset({
        "dispatch.descriptor_reuse", "dispatch.loop_offloaded"}),
    "latency_hiding_and_double_buffering": frozenset({
        "dispatch.dma_overlap", "dispatch.double_buffered_banks",
        "communication.copy_compute_overlap"}),
    "synchronization": frozenset({
        "communication.fences", "simt.barriers_in_loop", "dispatch.dma_overlap"}),
    "capacity_and_contention": frozenset({
        "memory.capacity_fit", "dispatch.double_buffered_banks"}),
    "exact_quantized_epilogues_and_residuals": frozenset({
        "compute.epilogue", "compute.activation_vectorization",
        "coverage.non_contraction_op_fraction"}),
    "arena_lifetimes_and_reuse": frozenset({
        "memory.capacity_fit", "communication.intermediate_materialized"}),
    "entry_abi_and_runtime_overhead": frozenset({
        "envelope.runtime_calls", "envelope.calls_in_loop"}),
}


# Phase 2 is a whole-program optimization loop. A flat list of actionable findings can make a
# cheap local rewrite look as important as deleting a model-wide representation boundary. Keep
# the ordering target-neutral: tiers name compiler/dataflow scope, never a target opcode or model.
_MACRO_OPTIMIZATION_LADDER = (
    {
        "tier": 0,
        "name": "correctness_and_regression_repair",
        "gaps": (),
        "purpose": "repair emission, correctness, work, movement, synchronization, or trace regressions",
    },
    {
        "tier": 1,
        "name": "whole_program_work_deletion",
        "gaps": (
            "whole_model_placement_and_coverage",
            "fusion_and_host_boundaries",
            "exact_quantized_epilogues_and_residuals",
            "arena_lifetimes_and_reuse",
            "entry_abi_and_runtime_overhead",
        ),
        "purpose": "delete whole-model tasks, materializations, boundaries, redundant representations, and runtime work",
    },
    {
        "tier": 2,
        "name": "global_dataflow_and_representation",
        "gaps": (
            "encoding_and_layout",
            "movement_and_materialization",
            "residency_across_operations",
        ),
        "purpose": "select encodings globally and keep values resident instead of converting, spilling, or reloading them",
    },
    {
        "tier": 3,
        "name": "global_execution_and_latency_hiding",
        "gaps": (
            "dispatch_and_loop_offload",
            "latency_hiding_and_double_buffering",
            "synchronization",
            "capacity_and_contention",
        ),
        "purpose": "keep engines occupied by issuing independent work early and minimizing exposed waits",
    },
    {
        "tier": 4,
        "name": "operator_layer_and_tile_efficiency",
        "gaps": ("arithmetic_lowering",),
        "purpose": "improve an operator or tile only after larger whole-program opportunities are terminal for this revision",
    },
    {
        "tier": 5,
        "name": "local_scalar_cleanup",
        "gaps": (),
        "purpose": "apply peepholes and scalar cleanup only when no higher-tier mechanism remains actionable",
    },
)


# Quantized-region plans name semantic ownership roles, never paths.  This host-owned table maps a
# known role onto CCA/effect vocabulary; the real editable coordinates still have to resolve through
# ``inspect_compiler_package`` and the immutable edit contract below.  A candidate receipt therefore
# cannot grant itself authority by inventing a path, effect, or new role.
_QUANTIZED_REGION_EDIT_ROLES = {
    "source_epilogue_semantics": {
        "effects": frozenset({"quantization", "fusion", "dtype"}),
        "cca_axes": frozenset({
            "compute.epilogue", "coverage.non_contraction_op_fraction",
        }),
        "purpose": "recognize exact source quantization stages and ownership",
    },
    "global_quant_domain_planner": {
        "effects": frozenset({"quantization", "encoding", "residency", "fusion"}),
        "cca_axes": frozenset({
            "compute.epilogue", "communication.intermediate_materialized",
            "communication.resident_across_calls",
        }),
        "purpose": "choose compatible domains and legal producer/consumer regions globally",
    },
    "target_epilogue_emitter": {
        "effects": frozenset({"quantization", "dtype", "encoding", "fusion"}),
        "cca_axes": frozenset({"compute.epilogue", "layout.operand_major"}),
        "purpose": "lower an admitted epilogue through a target-supported physical readout",
    },
    "target_residual_region_emitter": {
        "effects": frozenset({"quantization", "fusion", "residency", "movement"}),
        "cca_axes": frozenset({
            "compute.epilogue", "communication.intermediate_materialized",
            "communication.resident_across_calls", "memory.onchip_resident",
        }),
        "purpose": "lower admitted wide-domain residual operations without host materialization",
    },
    "target_encoding_and_residency": {
        "effects": frozenset({"encoding", "layout", "residency", "movement"}),
        "cca_axes": frozenset({
            "layout.operand_major", "communication.resident_across_calls",
            "memory.onchip_resident", "memory.capacity_fit",
        }),
        "purpose": "realize direct encoded boundaries under target-derived capacity",
    },
}


def macro_optimization_order() -> dict[str, Any]:
    """Return the host-owned largest-to-smallest Phase-2 selection contract."""
    return {
        "schema": "macro_optimization_order_v1",
        "selection": (
            "choose the lowest numbered nonterminal tier; within it prefer quantified dynamic "
            "extent, then cross-model coverage; use a matched reduced cycle witness when unlike "
            "costs trade off"
        ),
        "descent_gate": (
            "every higher tier must have a retained structural change or an explicit source/plan-"
            "bound refusal or no-op for the current compiler revision"
        ),
        "prohibited_shortcut": (
            "an easy local rewrite, capsule win, or unknown cost cannot bypass an actionable "
            "whole-program or global-dataflow mechanism"
        ),
        "tiers": [dict(row) for row in _MACRO_OPTIMIZATION_LADDER],
    }


@dataclass(frozen=True)
class SourceSymbol:
    path: str
    symbol: str
    kind: str
    line: int
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "symbol": self.symbol, "kind": self.kind,
                "line": self.line, "commands": list(self.commands)}


@dataclass(frozen=True)
class EditSurface:
    id: str
    scope: str
    path: str
    symbol: str
    line: int
    effects: tuple[str, ...]
    cca_axes: tuple[str, ...]
    cca_axis_status: tuple[tuple[str, str], ...]
    mechanism: str
    emitted_delta: str
    validation: str
    abandonment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "scope": self.scope, "path": self.path, "symbol": self.symbol,
            "line": self.line,
            "effects": list(self.effects), "cca_axes": list(self.cca_axes),
            "cca_axis_status": dict(self.cca_axis_status), "mechanism": self.mechanism,
            "emitted_delta": self.emitted_delta, "validation": self.validation,
            "abandonment": self.abandonment,
        }


@dataclass(frozen=True)
class PackageOptimizationInventory:
    symbols: tuple[SourceSymbol, ...]
    surfaces: tuple[EditSurface, ...]
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"symbols": [row.to_dict() for row in self.symbols],
                "surfaces": [row.to_dict() for row in self.surfaces],
                "missing": list(self.missing)}


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty package-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} escapes the compiler package")
    return path.as_posix()


def _component_paths(manifest: Mapping[str, Any], package: Path) -> dict[str, set[str]]:
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        return {}
    result: dict[str, set[str]] = {}
    for command, raw_paths in components.items():
        if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)):
            continue
        found: set[str] = set()
        for raw in raw_paths:
            relative = _safe_relative(raw, f"component {command}")
            path = package / relative
            if path.is_dir():
                found.update(item.relative_to(package).as_posix()
                             for item in path.rglob("*.py") if item.is_file())
            elif path.is_file() and path.suffix == ".py":
                found.add(relative)
        result[str(command)] = found
    return result


def _symbols(path: Path, relative: str, commands: tuple[str, ...]) -> list[SourceSymbol]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(f"cannot index compiler source {relative}: {exc}") from exc
    result: list[SourceSymbol] = []

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                result.append(SourceSymbol(relative, name, "function", node.lineno, commands))
            elif isinstance(node, ast.ClassDef):
                name = f"{prefix}.{node.name}" if prefix else node.name
                result.append(SourceSymbol(relative, name, "class", node.lineno, commands))
                visit(node.body, name)

    visit(tree.body)
    return result


def inspect_compiler_package(package: str | Path, *,
        host_surface_declarations: Sequence[Mapping[str, Any]] | None = None
        ) -> PackageOptimizationInventory:
    """Resolve declarations against real AST/component ownership, without granting edits.

    A host may supply separately pinned semantic descriptions instead of the candidate's
    manifest. The controller must intersect these with its immutable edit authority.
    Source symbols, line numbers and CCA classifications are always derived here.
    """
    root = Path(package)
    manifest_path = root / "manifest.yaml"
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() \
            or not manifest_path.is_file():
        raise ValueError("compiler package and manifest must be real files")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, Mapping):
        raise ValueError("compiler manifest must be a mapping")
    component_paths = _component_paths(manifest, root)
    missing: list[str] = []
    if not component_paths:
        missing.append(
            "manifest.yaml declares no usable components; command ownership and source edit "
            "surfaces cannot be established")
    commands_by_path: dict[str, list[str]] = {}
    for command, paths in component_paths.items():
        for path in paths:
            commands_by_path.setdefault(path, []).append(command)
    symbols = tuple(sorted((
        symbol for relative, commands in commands_by_path.items()
        for symbol in _symbols(root / relative, relative, tuple(sorted(commands)))
    ), key=lambda row: (row.path, row.line, row.symbol)))
    symbol_index = {(row.path, row.symbol): row for row in symbols}

    raw_surfaces = (manifest.get("optimization_surfaces") if host_surface_declarations is None
                    else host_surface_declarations)
    if raw_surfaces is None:
        missing.append(
            "manifest.yaml declares no optimization_surfaces; source symbols are indexed, but "
            "the harness cannot guess which one changes an encoding, movement, or schedule")
        raw_surfaces = ()
    if not isinstance(raw_surfaces, Sequence) or isinstance(raw_surfaces, (str, bytes)):
        raise ValueError("optimization_surfaces must be a sequence")
    surfaces: list[EditSurface] = []
    from merlin.kernels import cca_contract
    editable_cca_classes = {cca_contract.LEVER, cca_contract.BACKEND_STUB}
    for index, raw in enumerate(raw_surfaces):
        if not isinstance(raw, Mapping):
            raise ValueError(f"optimization surface {index} is not a mapping")
        ident = str(raw.get("id") or "").strip()
        scope = str(raw.get("scope") or "").strip()
        relative = _safe_relative(raw.get("path"), f"optimization surface {ident!r}")
        symbol = str(raw.get("symbol") or "").strip()
        effects = tuple(sorted(str(value) for value in (raw.get("effects") or ())))
        raw_axes = raw.get("cca_axes") or ()
        if (not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes))
                or any(not isinstance(axis, str) or not axis for axis in raw_axes)):
            raise ValueError(f"optimization surface {ident!r} has invalid cca_axes")
        cca_axes = tuple(sorted(set(raw_axes)))
        unknown_axes = [axis for axis in cca_axes if axis not in cca_contract.FIELD_REGISTRY]
        if unknown_axes:
            raise ValueError(
                f"optimization surface {ident!r} names unknown CCA axes {unknown_axes}")
        noneditable_axes = [
            axis for axis in cca_axes
            if cca_contract.FIELD_REGISTRY[axis].classification not in editable_cca_classes]
        if noneditable_axes:
            raise ValueError(
                f"optimization surface {ident!r} maps non-editable CCA axes {noneditable_axes}")
        cca_axis_status = tuple(
            (axis, cca_contract.FIELD_REGISTRY[axis].classification) for axis in cca_axes)
        if not ident or scope not in SCOPES or not symbol:
            raise ValueError(f"optimization surface {index} has an invalid id/scope/symbol")
        if not effects or any(effect not in EFFECTS for effect in effects):
            raise ValueError(f"optimization surface {ident!r} has invalid effects {effects}")
        resolved_symbol = symbol_index.get((relative, symbol))
        if resolved_symbol is None:
            raise ValueError(
                f"optimization surface {ident!r} does not resolve to AST symbol "
                f"{relative}:{symbol}")
        texts = tuple(str(raw.get(name) or "").strip() for name in (
            "mechanism", "emitted_delta", "validation", "abandonment"))
        if not all(texts):
            raise ValueError(
                f"optimization surface {ident!r} must state mechanism, emitted delta, "
                "validation, and abandonment")
        surfaces.append(EditSurface(
            ident, scope, relative, symbol, resolved_symbol.line, effects, cca_axes,
            cca_axis_status, *texts))
    ids = [surface.id for surface in surfaces]
    if len(ids) != len(set(ids)):
        raise ValueError("optimization surface ids must be unique")
    return PackageOptimizationInventory(symbols, tuple(surfaces), tuple(missing))


def build_compiler_edit_contract(
        inventory: PackageOptimizationInventory, *,
        helper_extensions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Describe a host-frozen edit boundary; candidate manifests cannot expand it.

    The launcher must bind this object before authoring and enforce it before
    executing submitted compiler code. Returning a catalog alone is not an edit
    sandbox. Optional new-helper directories are explicit host inputs, never
    inferred from an untrusted candidate's newly declared optimization surfaces.
    """
    symbols = {(row.path, row.symbol): row for row in inventory.symbols}
    existing = []
    for surface in inventory.surfaces:
        owner = symbols.get((surface.path, surface.symbol))
        if owner is None:
            raise ValueError('edit surface must resolve to the frozen source inventory')
        existing.append({'surface_id': surface.id, 'path': surface.path,
                         'symbol': surface.symbol, 'kind': owner.kind, 'line': owner.line})
    ids = {row['surface_id'] for row in existing}
    directories = {str(parent) for row in inventory.symbols
                   for parent in PurePosixPath(row.path).parents if str(parent) != '.'}
    extensions = []
    for extension in helper_extensions:
        directory = _safe_relative(extension.get('directory'), 'helper extension directory')
        owners = extension.get('surface_ids')
        reason = extension.get('reason')
        if directory not in directories:
            raise ValueError('helper extension must be an existing scoped compiler component directory')
        if (not isinstance(owners, Sequence) or isinstance(owners, (str, bytes))
                or not owners or any(owner not in ids for owner in owners)):
            raise ValueError('helper extension requires frozen owning surface ids')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('helper extension requires a host-declared reason')
        extensions.append({'directory': directory, 'surface_ids': sorted(set(owners)), 'reason': reason})
    contract = {
        'schema': 'compiler_edit_contract_v1',
        'authority': 'host-frozen initial inventory; candidate declarations are descriptive only',
        'existing_symbols': existing,
        'helper_extensions': extensions,
        'protected_controls': [
            'frozen source models, inputs, weights and Phase 1 outcomes',
            'reference semantics, correctness checks and waivers',
            'measurement harness, warm/ROI contract and counter interpretation',
            'target facts, hardware pins and derived ISA constants',
            'evaluator, sandbox policy, launch commands and scoring',
            'edit contract itself; candidate manifest changes grant no permissions',
        ],
        'work_order_required_fields': [
            'surface_ids', 'source_operation_ids', 'current_plan_digest', 'hypothesis',
            'expected_emitted_delta', 'semantic_obligations', 'cheap_validation',
            'stop_or_revert_condition',
        ],
        'enforcement': 'requires host pre-execution submitted-change gate; catalog is not enforcement',
        'unmapped_mechanism': 'record missing edit surface and request an explicit host-approved contract extension',
    }
    contract['sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return contract


def guidance_for_quantized_region_plan(
        plan: Mapping[str, Any], inventory: PackageOptimizationInventory) -> dict[str, Any]:
    """Join a quantized-region plan to host-verified, contract-bound edit coordinates.

    The plan may request only a known semantic role.  Its text is never trusted as a path or an
    effect classification.  Returned surfaces come exclusively from ``inventory`` and are bound to
    the generated edit-contract SHA, so candidate-authored metadata cannot enlarge the sandbox.
    Accuracy/corpus policy is intentionally listed as a protected host input rather than an edit
    surface.
    """
    if plan.get("schema") != "target_neutral_quantized_region_plan_v1":
        raise ValueError("quantized region guidance requires a canonical framework plan")
    requirements = plan.get("agent_edit_requirements")
    if not isinstance(requirements, Sequence) or isinstance(requirements, (str, bytes)):
        raise ValueError("quantized region plan has malformed agent edit requirements")
    contract = build_compiler_edit_contract(inventory)
    authorized = {
        (owner["surface_id"], owner["path"], owner["symbol"])
        for owner in contract["existing_symbols"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(requirements):
        if not isinstance(raw, Mapping):
            raise ValueError(f"quantized edit requirement {index} is not an object")
        role = raw.get("role")
        if role not in _QUANTIZED_REGION_EDIT_ROLES or role in seen:
            raise ValueError(f"quantized edit requirement role {role!r} is unknown or repeated")
        seen.add(role)
        reason_codes = raw.get("reason_codes")
        if (not isinstance(reason_codes, Sequence) or isinstance(reason_codes, (str, bytes))
                or any(not isinstance(reason, str) or not reason for reason in reason_codes)):
            raise ValueError(f"quantized edit requirement {role!r} has malformed reasons")
        metadata = _QUANTIZED_REGION_EDIT_ROLES[role]
        axis_matches = [
            surface for surface in inventory.surfaces
            if metadata["cca_axes"].intersection(surface.cca_axes)
            and (surface.id, surface.path, surface.symbol) in authorized
        ]
        effect_matches = [
            surface for surface in inventory.surfaces
            if metadata["effects"].intersection(surface.effects)
            and (surface.id, surface.path, surface.symbol) in authorized
        ]
        matches = axis_matches or effect_matches
        rows.append({
            "role": role,
            "status": "authorized" if matches else "missing_verified_surface",
            "purpose": metadata["purpose"],
            "reason_codes": sorted(set(reason_codes)),
            "required_effects": sorted(metadata["effects"]),
            "required_cca_axes": sorted(metadata["cca_axes"]),
            "mapping_basis": (
                "exact CCA axis" if axis_matches else
                "broad effect compatibility" if effect_matches else None
            ),
            "authorized_edit_surfaces": [surface.to_dict() for surface in matches],
        })
    protected = plan.get("protected_policy_blockers")
    if not isinstance(protected, Sequence) or isinstance(protected, (str, bytes)):
        raise ValueError("quantized region plan has malformed protected policy blockers")
    return {
        "schema": "agent_quantized_region_guidance_v1",
        "plan_source_sha256": plan.get("normalized_source_sha256"),
        "plan_status": plan.get("status"),
        "edit_contract_sha256": contract["sha256"],
        "requirements": rows,
        "protected_inputs": [{
            "name": "accuracy_and_corpus_policy",
            "editable_by_candidate": False,
            "reason_codes": sorted(set(str(reason) for reason in protected)),
            "required_action": (
                "the experiment owner must provide a reviewed, source-bound policy and corpus; "
                "compiler edits cannot self-authorize approximation"
            ),
        }],
        "authority": (
            "only surfaces resolved from the host-frozen compiler inventory and named by the "
            "bound compiler_edit_contract_v1 are returned"
        ),
        "candidate_manifest_grants_authority": False,
        "compiler_edit_contract": contract,
    }


def guidance_for_report(report: WholeModelReport,
                        inventory: PackageOptimizationInventory) -> dict[str, Any]:
    """Join ranked whole-model problems to declared, verified compiler edit surfaces."""
    actions: list[dict[str, Any]] = []
    for opportunity in report.opportunities:
        effects = _OPPORTUNITY_EFFECTS.get(opportunity.kind, frozenset())
        axes = _OPPORTUNITY_AXES.get(opportunity.kind, frozenset())
        axis_matches = [surface for surface in inventory.surfaces
                        if axes.intersection(surface.cca_axes)]
        effect_matches = [surface for surface in inventory.surfaces
                          if effects.intersection(surface.effects)]
        matches = axis_matches or effect_matches
        actions.append({
            "opportunity": opportunity.to_dict(),
            "required_effects": sorted(effects),
            "required_cca_axes": sorted(axes),
            "edit_surfaces": [surface.to_dict() for surface in matches],
            "status": "actionable" if matches else "unmapped",
            "mapping_basis": ("exact CCA axis" if axis_matches else
                              "broad effect compatibility" if effect_matches else None),
            "missing": (None if matches else
                        "no verified manifest surface declares an effect that addresses this gap"),
        })
    return {
        "schema": "agent_global_optimization_brief_v1",
        "objective": "minimize warm complete-model compute cycles",
        "optimization_order": macro_optimization_order(),
        "success_gates": dict(report.gates),
        "ranked_actions": actions,
        "package_inventory": inventory.to_dict(),
        "compiler_edit_contract_template": build_compiler_edit_contract(inventory),
        "method": (
            "edit a verified compiler surface, confirm its emitted representation/activity delta, "
            "run one warm reduced witness within 600 seconds, then re-plan the complete model; "
            "full-size execution is optional post-freeze validation and is not required by Phase 2"),
    }


def guidance_for_emission_analysis(
        analysis: Mapping[str, Any],
        inventory: PackageOptimizationInventory) -> dict[str, Any]:
    """Turn one host-owned whole-model emission comparison into exact edit instructions.

    This is deliberately a structural brief, not a performance verdict.  It ranks regressions and
    visible inefficiencies in the emitted complete-model program, then joins each one to manifest
    declarations that resolve to real Python AST symbols.  A package without such declarations gets
    an explicit manifest edit instruction and the complete source index; semantic ownership is never
    guessed from a filename or function name.
    """

    arms = analysis.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    baseline = arms.get("baseline")
    baseline = baseline if isinstance(baseline, Mapping) else {}
    candidate = arms.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    findings: list[dict[str, Any]] = []

    def add(kind: str, detail: str, effects: Sequence[str], *,
            evidence: Mapping[str, Any] | None = None, priority: int,
            cca_axes: Sequence[str] | None = None) -> None:
        findings.append({
            "kind": kind,
            "detail": detail,
            "required_effects": sorted(set(str(effect) for effect in effects)),
            "required_cca_axes": sorted(set(cca_axes) if cca_axes is not None
                                        else _FINDING_AXES.get(kind, ())),
            "evidence": dict(evidence or {}),
            "priority": priority,
        })

    if candidate.get("status") == "declined":
        decline = candidate.get("declined")
        decline = decline if isinstance(decline, Mapping) else {}
        add("whole_model_lowering_declined",
            "the candidate emitted no executable whole-model command stream",
            ("placement", "fusion", "movement", "residency", "issue", "latency_hiding"),
            evidence={"op": decline.get("op"), "reason": decline.get("reason"),
                      "shape": decline.get("shape")}, priority=-1)

    candidate_representation = candidate.get("representation_activity")
    candidate_representation = (candidate_representation
                                if isinstance(candidate_representation, Mapping) else {})
    candidate_placement = candidate_representation.get("placement")
    candidate_placement = candidate_placement if isinstance(candidate_placement, Mapping) else {}
    lane_transitions = candidate_placement.get("adjacent_lane_transitions")
    if isinstance(lane_transitions, int) and not isinstance(lane_transitions, bool) \
            and lane_transitions > 0:
        add("inter_lane_boundary_pressure",
            "the complete model crosses compiler-declared lane boundaries that require explicit pricing",
            ("placement", "movement", "residency", "fusion", "latency_hiding"),
            evidence={"adjacent_lane_transitions": lane_transitions,
                      "lane_counts": candidate_placement.get("lane_counts")}, priority=1)

    weighted = analysis.get("model_contraction_placement")
    weighted = weighted if isinstance(weighted, Mapping) else {}
    weighted_candidate = weighted.get("candidate")
    weighted_candidate = weighted_candidate if isinstance(weighted_candidate, Mapping) else {}
    by_lane = weighted_candidate.get("mac_fraction_by_lane")
    if isinstance(by_lane, Mapping) and len(by_lane) > 1:
        add("heterogeneous_model_placement",
            "captured contraction MAC work is split across multiple declared lanes",
            ("placement", "fusion", "movement"),
            evidence={"mac_fraction_by_lane": dict(by_lane),
                      "total_contraction_macs": weighted_candidate.get("total_contraction_macs"),
                      "unresolved_placement_macs": weighted_candidate.get(
                          "unresolved_placement_macs")}, priority=1)
    memory_regime = weighted_candidate.get("memory_regime")
    memory_regime = memory_regime if isinstance(memory_regime, Mapping) else {}
    regime_counts = memory_regime.get("region_counts")
    if isinstance(regime_counts, Mapping) and regime_counts:
        add("memory_regime_pressure",
            "captured contractions occupy different on-chip capacity regimes; overlap legality and "
            "reload policy must follow the regime rather than one universal schedule",
            ("residency", "movement", "tiling", "latency_hiding"),
            evidence={"region_counts": dict(regime_counts),
                      "mac_fraction_by_regime": memory_regime.get("mac_fraction_by_regime"),
                      "double_buffer_eligible_regime": memory_regime.get(
                          "double_buffer_eligible_regime")}, priority=1)

    verified_plan = analysis.get("verified_global_plan_emission")
    verified_plan = verified_plan if isinstance(verified_plan, Mapping) else {}
    host_activity = verified_plan.get("host_activity")
    host_activity = host_activity if isinstance(host_activity, Mapping) else {}
    host_tasks = host_activity.get("top_tasks_by_scalar_memory_payload")
    host_allocations = host_activity.get("top_allocations_by_static_payload")
    host_buffers = host_activity.get("top_buffers_by_scalar_memory_payload")
    if verified_plan.get("status") == "verified" and any(
            isinstance(rows, list) and rows for rows in (host_tasks, host_allocations, host_buffers)):
        add("host_memory_hotspot",
            "source-derived allocations and host loops expose materialization and scalar-work hotspots; "
            "smaller emitted IR alone does not delete this dynamic work",
            ("fusion", "movement", "placement", "layout"),
            evidence={"status": host_activity.get("status", "UNKNOWN"),
                      "top_tasks": host_tasks[:5] if isinstance(host_tasks, list) else [],
                      "top_allocations": host_allocations[:3] if isinstance(host_allocations, list) else [],
                      "top_buffers": host_buffers[:3] if isinstance(host_buffers, list) else [],
                      "artifact_sha256": host_activity.get("artifact_sha256"),
                      "allocation_identity_scope": host_activity.get("memory_hotspot_identity_scope", "UNKNOWN"),
                      "load_payload_bytes": host_activity.get("load_payload_bytes"),
                      "store_payload_bytes": host_activity.get("store_payload_bytes"),
                      "scope": "pre-optimization LLVM scalar payload, not DRAM or CPU cycles; "
                               "static allocation payload is not stack-frame size or live-memory peak"},
            priority=1)

    baseline_macs, candidate_macs = baseline.get("macs"), candidate.get("macs")
    if (isinstance(baseline_macs, (int, float)) and not isinstance(baseline_macs, bool)
            and isinstance(candidate_macs, (int, float)) and not isinstance(candidate_macs, bool)
            and candidate_macs > baseline_macs):
        add("arithmetic_expansion", "candidate emits more arithmetic for the same frozen model",
            ("fusion", "tiling"), evidence={"baseline_macs": baseline_macs,
                                             "candidate_macs": candidate_macs}, priority=0)

    baseline_movement = baseline.get("movement")
    baseline_movement = baseline_movement if isinstance(baseline_movement, Mapping) else {}
    candidate_movement = candidate.get("movement")
    candidate_movement = candidate_movement if isinstance(candidate_movement, Mapping) else {}
    base_bytes, cand_bytes = (baseline_movement.get("known_bytes"),
                              candidate_movement.get("known_bytes"))
    if (isinstance(base_bytes, (int, float)) and not isinstance(base_bytes, bool)
            and isinstance(cand_bytes, (int, float)) and not isinstance(cand_bytes, bool)
            and cand_bytes > base_bytes):
        add("movement_regression", "candidate declares more command-buffer movement than baseline",
            ("movement", "residency", "layout", "encoding"),
            evidence={"baseline_known_bytes": base_bytes,
                      "candidate_known_bytes": cand_bytes}, priority=0)
    if candidate_movement.get("is_lower_bound") is True:
        add("unpriced_movement", "some candidate movement is not represented by an exact byte count",
            ("movement", "residency", "encoding"),
            evidence={"refusals": list(candidate_movement.get("refusals") or ())}, priority=1)

    barriers = analysis.get("barriers")
    barriers = barriers if isinstance(barriers, Mapping) else {}
    if barriers.get("status") == "counted":
        removed = barriers.get("removed")
        if isinstance(removed, int) and not isinstance(removed, bool) and removed < 0:
            add("synchronization_regression",
                "candidate added completion points to the complete-model command stream",
                ("synchronization", "issue", "latency_hiding"),
                evidence={"baseline_barriers": barriers.get("baseline_barriers"),
                          "candidate_barriers": barriers.get("candidate_barriers")}, priority=0)

    levels = analysis.get("structural_levels")
    levels = levels if isinstance(levels, Mapping) else {}
    candidate_levels = levels.get("candidate")
    candidate_levels = candidate_levels if isinstance(candidate_levels, Mapping) else {}
    for raw in candidate_levels.get("findings") or ():
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "structural_inefficiency")
        effects = _STRUCTURAL_EFFECTS.get(kind, frozenset({"movement", "fusion"}))
        add(kind, str(raw.get("detail") or "candidate has a structural inefficiency"), effects,
            evidence={key: raw[key] for key in ("level", "value", "at_command") if key in raw},
            priority=1)

    representation = candidate.get("representation_activity")
    representation = representation if isinstance(representation, Mapping) else {}
    emitted = representation.get("emitted_encoding_transitions")
    emitted = emitted if isinstance(emitted, Mapping) else {}
    if emitted.get("status") != "measured":
        add("unverified_encoding_transitions",
            "declared encodings are visible, but executed conversions have not been counted",
            ("encoding", "layout", "fusion"),
            evidence={"declared_directives": emitted.get("declared_directives"),
                      "status": emitted.get("status", "UNKNOWN")}, priority=2)
    occupancy = representation.get("occupancy")
    occupancy = occupancy if isinstance(occupancy, Mapping) else {}
    if occupancy.get("status") != "measured":
        add("unknown_occupancy_and_overlap",
            "compute occupancy and movement/compute overlap require an event adapter or warm counters",
            ("latency_hiding", "issue", "tiling", "movement", "residency"),
            evidence={"status": occupancy.get("status", "UNKNOWN")}, priority=2)

    artifact = analysis.get("target_artifact_activity")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    artifact_baseline = artifact.get("baseline")
    artifact_baseline = artifact_baseline if isinstance(artifact_baseline, Mapping) else {}
    artifact_candidate = artifact.get("candidate")
    artifact_candidate = artifact_candidate if isinstance(artifact_candidate, Mapping) else {}
    base_issued = artifact_baseline.get("issued")
    base_issued = base_issued if isinstance(base_issued, Mapping) else {}
    cand_issued = artifact_candidate.get("issued")
    cand_issued = cand_issued if isinstance(cand_issued, Mapping) else {}

    def _numeric_delta(field: str) -> tuple[int | float, int | float] | None:
        left, right = base_issued.get(field), cand_issued.get(field)
        if (isinstance(left, (int, float)) and not isinstance(left, bool)
                and isinstance(right, (int, float)) and not isinstance(right, bool)):
            return left, right
        return None

    delta = _numeric_delta("movement_instructions")
    if delta is not None and delta[1] > delta[0]:
        add("issued_movement_regression",
            "candidate issues more movement instructions in the lowered target artifact",
            ("movement", "residency", "layout", "encoding"),
            evidence={"baseline": delta[0], "candidate": delta[1]}, priority=0)
    delta = _numeric_delta("configuration_instructions")
    if delta is not None and delta[1] > delta[0]:
        add("dispatch_configuration_regression",
            "candidate reconfigures the target endpoint more often",
            ("issue", "latency_hiding", "synchronization"),
            evidence={"baseline": delta[0], "candidate": delta[1]}, priority=0)
    delta = _numeric_delta("synchronization_instructions")
    if delta is not None and delta[1] > delta[0]:
        add("artifact_synchronization_regression",
            "candidate issues more synchronization in the lowered target artifact",
            ("synchronization", "issue", "latency_hiding"),
            evidence={"baseline": delta[0], "candidate": delta[1]}, priority=0)
    base_loop = base_issued.get("loop_descriptor_instructions")
    cand_loop = cand_issued.get("loop_descriptor_instructions")
    if (isinstance(base_loop, int) and not isinstance(base_loop, bool)
            and isinstance(cand_loop, int) and not isinstance(cand_loop, bool)
            and base_loop > 0 and cand_loop == 0):
        add("loop_offload_regression",
            "candidate removed the target's emitted loop-descriptor path",
            ("issue", "tiling", "latency_hiding"),
            evidence={"baseline": base_loop, "candidate": cand_loop}, priority=0)

    encoding = artifact_candidate.get("encoding_resolution")
    encoding = encoding if isinstance(encoding, Mapping) else {}
    if artifact_candidate and encoding.get("status") != "complete":
        add("unresolved_target_encoding",
            "the emitted artifact contains instructions without a complete semantic-role decode",
            ("encoding", "layout"), evidence={
                "status": encoding.get("status", "UNKNOWN"),
                "unknown_instruction_indices": list(
                    encoding.get("unknown_instruction_indices") or ()),
                "named_without_role": list(encoding.get("named_without_role") or ()),
            }, priority=1)

    for raw in artifact_candidate.get("artifact_opportunities") or ():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("axis"), str):
            continue
        axis = str(raw["axis"])
        prefix = axis.partition(".")[0]
        effects = {
            "compute": ("fusion", "tiling"),
            "memory": ("movement", "residency", "layout"),
            "layout": ("layout", "encoding", "movement"),
            "dispatch": ("issue", "latency_hiding", "synchronization"),
            "communication": ("movement", "residency", "latency_hiding"),
            "simt": ("synchronization", "latency_hiding"),
            "envelope": ("fusion", "issue"),
        }.get(prefix, ("issue",))
        add("artifact_cca_opportunity", str(raw.get("observation") or
                                             "emitted role shape exposes an optimization"),
            effects, evidence={key: raw[key] for key in (
                "axis", "change", "seam", "forkable_now", "status", "confidence") if key in raw},
            priority=1, cca_axes=(axis,))

    trace = analysis.get("trace_conformance")
    trace = trace if isinstance(trace, Mapping) else {}
    introduced = trace.get("introduced_candidate_findings")
    if isinstance(introduced, Sequence) and not isinstance(introduced, (str, bytes)) and introduced:
        add("trace_regression",
            "candidate introduced decoded trace findings absent from the frozen Phase-1 baseline",
            ("residency", "movement", "synchronization", "encoding", "issue"),
            evidence={"introduced_findings": list(introduced)}, priority=0)

    ranked: list[dict[str, Any]] = []
    for rank, finding in enumerate(sorted(
            findings, key=lambda row: (int(row["priority"]), str(row["kind"]))), start=1):
        required = set(finding["required_effects"])
        required_axes = set(finding["required_cca_axes"])
        axis_matches = [surface for surface in inventory.surfaces
                        if required_axes.intersection(surface.cca_axes)]
        effect_matches = [surface for surface in inventory.surfaces
                          if required.intersection(surface.effects)]
        matches = axis_matches or effect_matches
        row = dict(finding)
        row.pop("priority", None)
        row.update({
            "rank": rank,
            "status": "actionable" if matches else "unmapped",
            "edit_surfaces": [surface.to_dict() for surface in matches],
            "mapping_basis": ("exact CCA axis" if axis_matches else
                              "broad effect compatibility" if effect_matches else None),
            "next_step": (
                "edit one listed path/symbol, rerun inspect-optimization-surfaces, then rerun "
                "analyze-whole-model; use a short mechanism-equivalent probe only for uncertain costs"
                if matches else
                "add an optimization_surfaces entry to manifest.yaml for the real owning AST "
                "symbol; the host will verify path and symbol ownership, then analyze whether the "
                "declared emitted delta actually appears"),
        })
        ranked.append(row)

    movement_status = ("unavailable: lowering declined" if candidate.get("status") == "declined"
                       else "lower_bound" if candidate_movement.get("is_lower_bound") is True
                       else "declared_exact" if candidate_movement else "UNKNOWN")
    trace_status = str(trace.get("status") or "UNKNOWN")
    artifact_status = str(artifact_candidate.get("status") or "UNKNOWN")
    mechanism_coverage = {
        "arithmetic_demand": "observed" if candidate_macs is not None else "UNKNOWN",
        "whole_model_lowering": str(candidate.get("status") or "UNKNOWN"),
        "declared_movement_volume": movement_status,
        "issued_movement": artifact_status,
        "host_scalar_work": str(host_activity.get("status") or "UNKNOWN"),
        "target_encoding": str(encoding.get("status") or "UNKNOWN"),
        "dispatch_shape": artifact_status,
        "synchronization": artifact_status,
        "residency_reload_defect": trace_status,
        "occupancy": "UNKNOWN for full model; bound reduced profiles are reported separately as scoped evidence",
        "contention": "UNKNOWN for full model; executed reduced counters do not establish model-wide contention",
        "cycle_ordering": "UNMEASURED: cheap signals are not validated schedule rankers",
    }

    placement = representation.get("placement")
    placement = placement if isinstance(placement, Mapping) else {}
    candidate_cca = artifact_candidate.get("program_cca")
    candidate_cca = candidate_cca if isinstance(candidate_cca, Mapping) else {}
    candidate_dispatch = candidate_cca.get("dispatch")
    candidate_dispatch = candidate_dispatch if isinstance(candidate_dispatch, Mapping) else {}
    overlap = candidate_dispatch.get("dma_overlap")
    def needs_evidence(status: str) -> bool:
        normalized = status.strip().casefold()
        return ("unknown" in normalized or "unmeasured" in normalized
                or normalized.startswith(("refused", "declined", "unavailable", "failed", "error",
                                          "invalid", "unsupported", "missing", "unresolved", "not_", "not ")))

    level_status = str(candidate_levels.get("status") or "UNKNOWN")
    structural_status = (level_status if candidate_levels.get("status") and needs_evidence(level_status)
                         else "screened" if isinstance(candidate_levels.get("findings"), Sequence)
                         and not isinstance(candidate_levels.get("findings"), (str, bytes)) else "UNKNOWN")
    observability = {
        "whole_model_placement_and_coverage": str(placement.get("status") or "UNKNOWN"),
        "arithmetic_lowering": mechanism_coverage["arithmetic_demand"],
        "encoding_and_layout": mechanism_coverage["target_encoding"],
        "movement_and_materialization": mechanism_coverage["issued_movement"],
        "residency_across_operations": mechanism_coverage["residency_reload_defect"],
        "fusion_and_host_boundaries": structural_status,
        "dispatch_and_loop_offload": mechanism_coverage["dispatch_shape"],
        "latency_hiding_and_double_buffering": (
            "structurally_observed" if isinstance(overlap, bool)
            else "UNKNOWN: emitted DMA/wait evidence and scoped warm profiles do not alone prove full-model latency hiding"),
        "synchronization": mechanism_coverage["synchronization"],
        "capacity_and_contention": (
            "capacity_regime_derived; dynamic_contention=UNKNOWN"
            if memory_regime.get("status") == "derived" else
            "UNKNOWN: target capacity/resource evidence is not joined"),
        "exact_quantized_epilogues_and_residuals": "UNKNOWN: operation counts do not establish staged rounding equivalence or second-operand offload",
        "arena_lifetimes_and_reuse": "UNKNOWN: static allocation totals do not prove aliasing, live ranges or asynchronous completion",
        "entry_abi_and_runtime_overhead": "UNKNOWN: task counts do not establish entry pointer binding or runtime overhead",
    }
    cheap_validation = {
        "whole_model_placement_and_coverage": "re-emit fixed full-model capture and compare placement",
        "arithmetic_lowering": "re-emit and compare exact command/artifact work",
        "encoding_and_layout": "decode emitted target artifact and count unresolved transitions",
        "movement_and_materialization": "compare declared bytes and issued movement instructions",
        "residency_across_operations": "run exact decoded reload/liveness checks",
        "fusion_and_host_boundaries": "compare materializations, host boundaries, and producer uses",
        "dispatch_and_loop_offload": "compare role-tagged config and loop-descriptor issue",
        "latency_hiding_and_double_buffering": "static DMA/wait shape, then one reduced warm counter run",
        "synchronization": "compare command completion points and emitted sync instructions",
        "capacity_and_contention": "requires a target resource adapter plus reduced pressure witness",
        "exact_quantized_epilogues_and_residuals": "clone the actual typed epilogue/residual chain; preserve each rounding stage and compare a short independent-reference witness",
        "arena_lifetimes_and_reuse": "prove buffer aliases, last uses and device completion on the full plan; compare planned live bytes and check a short reuse hazard witness",
        "entry_abi_and_runtime_overhead": "verify emitted entry signature and base/offset bindings against the full CFG; qualify a short address-binding mechanism",
    }
    gap_coverage: list[dict[str, Any]] = []
    for gap, axes in _GAP_AXES.items():
        matches = [surface for surface in inventory.surfaces if axes.intersection(surface.cca_axes)]
        observed = observability[gap]
        unresolved = needs_evidence(observed)
        gap_coverage.append({
            "gap": gap,
            "evidence_status": observed,
            "cca_axes": sorted(axes),
            "edit_surfaces": [surface.to_dict() for surface in matches],
            "edit_status": "mapped" if matches else "missing_verified_surface",
            "cheap_validation": cheap_validation[gap],
            "coverage_status": ("needs_evidence" if unresolved else
                                "needs_edit_surface" if not matches else "ready"),
        })

    plan_identity_fields = ("candidate_sha256", "source_sha256", "logical_dispatch_digest", "plan_digest",
                            "candidate_lowered_sha256", "candidate_command_buffer_sha256")
    plan_identities = {key: verified_plan.get(key) for key in plan_identity_fields}
    bound_plan = (verified_plan.get("status") == "verified" and all(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in plan_identities.values()))
    owned_counts = verified_plan.get("emitted_operations_by_task")
    temporaries = verified_plan.get("compiler_temporaries")
    plan_cfg = verified_plan.get("control_flow")
    plan_cfg = plan_cfg if isinstance(plan_cfg, Mapping) else {}
    compiler_plan = {
        "status": "verified" if bound_plan else (
            "UNKNOWN: verified plan lacks complete artifact identities" if verified_plan.get("status") == "verified"
            else str(verified_plan.get("status") or "UNKNOWN")),
        "identities": plan_identities,
        "source_operations": verified_plan.get("source_operations"), "tasks": verified_plan.get("tasks"),
        "owned_emitted_operation_count": sum(owned_counts.values())
            if bound_plan and isinstance(owned_counts, Mapping)
            and all(type(value) is int and value >= 0 for value in owned_counts.values()) else None,
        "compiler_temporary_count": len(temporaries) if bound_plan and isinstance(temporaries, list) else None,
        "control_flow": {key: plan_cfg.get(key, "UNKNOWN") for key in
                         ("status", "reachable_blocks", "return_blocks", "proof_scope", "loop_trip_count_equivalence")},
        "proof_scope": verified_plan.get("proof_scope", "UNKNOWN"),
        "numerical_equivalence": verified_plan.get("numeric_equivalence", "UNPROVEN"),
        "costs": "UNKNOWN", "optimality": "UNPROVEN",
    }

    return {
        "schema": "agent_emission_optimization_brief_v1",
        "objective": "minimize warm complete-model compute cycles",
        "optimization_order": macro_optimization_order(),
        "timing_status": "UNMEASURED",
        "ranked_actions": ranked,
        "package_inventory": inventory.to_dict(),
        "compiler_edit_contract_template": build_compiler_edit_contract(inventory),
        "mechanism_coverage": mechanism_coverage,
        "gap_coverage": gap_coverage,
        "global_planner_wiring": {
            "status": "compiler_owned_plan_verified" if bound_plan else "UNKNOWN",
            "compiler_owned_plan": compiler_plan,
            "shared_exact_cover_search": {
                "status": "UNKNOWN", "proof": None,
                "scope": "a verified compiler-owned plan does not establish use of the shared exact-cover solver"},
            "required": (
                "optimize and re-verify the existing compiler-owned full-model plan; use focused reduced "
                "witnesses for changed semantics and uncertain costs. If adopting shared exact-cover search, "
                "bind actual solver selection through a target planning adapter and emitted-plan receipt"
                if bound_plan else
                "provide a source/artifact-bound compiler-owned plan with complete task, ABI and CFG coverage; "
                "shared TargetPlanningAdapter integration is a separate optional search route"
            ),
            "available_shared_blocks": [
                "GlobalPlan exact-cover search", "explicit ValueRepresentation transitions",
                "ActivityTimeline dependency/resource scheduling",
                "PipelineProjection from reduced stage calibration",
                "verified GlobalPlanEmission accounting",
                "OutlinedGlobalPlanEmitter with expanded full-model IR equivalence proof",
            ],
            "captured_logical_graph": {
                key: value for key, value in (analysis.get("captured_logical_graph") or {}).items()
                if key != "dispatch_program"
            },
            "warning": (
                "shared planner classes existing in Merlin does not prove this compiler package "
                "selects or emits their plan"
            ),
        },
        "edit_protocol": [
            "select declared surface ids and state a source/plan-bound hypothesis, expected delta, semantic obligations and stopping condition",
            "edit only the host-frozen permitted package paths and AST symbols; a changed candidate manifest cannot expand permission",
            "add helpers only within explicit host-approved extension directories; otherwise report the missing surface",
            "refresh the optimization-surface inventory after the edit",
            "re-emit the fixed complete-model witness and verify the intended structural delta",
            "check changed semantics using a relative proof or a focused independent correctness witness",
            "only when a selection depends on uncertain cost, run one warm and one measured "
            "short mechanism-equivalent probe; never execute a full layer or model",
            "reject absent structural deltas or incorrect edits; retain uncalibrated structural "
            "improvements as unmeasured, never as demonstrated cycle speedups",
        ],
    }
