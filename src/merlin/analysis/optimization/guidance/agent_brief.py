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

from merlin.analysis.optimization.evaluation.model_report import OptimizationOpportunity, WholeModelReport
from merlin.analysis.optimization.guidance.vocabulary import FINDING_AXES as _FINDING_AXES
from merlin.analysis.optimization.guidance.vocabulary import GAP_AXES as _GAP_AXES
from merlin.analysis.optimization.guidance.vocabulary import (
    MACRO_OPTIMIZATION_LADDER as _MACRO_OPTIMIZATION_LADDER,
)
from merlin.analysis.optimization.guidance.vocabulary import OPPORTUNITY_AXES as _OPPORTUNITY_AXES
from merlin.analysis.optimization.guidance.vocabulary import OPPORTUNITY_EFFECTS as _OPPORTUNITY_EFFECTS
from merlin.analysis.optimization.guidance.vocabulary import (
    QUANTIZED_REGION_EDIT_ROLES as _QUANTIZED_REGION_EDIT_ROLES,
)
from merlin.analysis.optimization.guidance.vocabulary import STRUCTURAL_EFFECTS as _STRUCTURAL_EFFECTS
from merlin.analysis.optimization.guidance.vocabulary import (
    _declared_surface_vocabulary,
    macro_optimization_order,
)

# ---- agent_guidance ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSymbol:
    path: str
    symbol: str
    kind: str
    line: int
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "line": self.line,
            "commands": list(self.commands),
        }


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
            "id": self.id,
            "scope": self.scope,
            "path": self.path,
            "symbol": self.symbol,
            "line": self.line,
            "effects": list(self.effects),
            "cca_axes": list(self.cca_axes),
            "cca_axis_status": dict(self.cca_axis_status),
            "mechanism": self.mechanism,
            "emitted_delta": self.emitted_delta,
            "validation": self.validation,
            "abandonment": self.abandonment,
        }


@dataclass(frozen=True)
class PackageOptimizationInventory:
    symbols: tuple[SourceSymbol, ...]
    surfaces: tuple[EditSurface, ...]
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": [row.to_dict() for row in self.symbols],
            "surfaces": [row.to_dict() for row in self.surfaces],
            "missing": list(self.missing),
        }


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
                found.update(item.relative_to(package).as_posix() for item in path.rglob("*.py") if item.is_file())
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


def inspect_compiler_package(
    package: str | Path,
    *,
    host_surface_declarations: Sequence[Mapping[str, Any]] | None = None,
    contract: str | Path | None = None,
) -> PackageOptimizationInventory:
    """Resolve declarations against real AST/component ownership, without granting edits.

    A host may supply separately pinned semantic descriptions instead of the candidate's
    manifest. The controller must intersect these with its immutable edit authority.
    Source symbols, line numbers and CCA classifications are always derived here.
    """
    root = Path(package)
    manifest_path = root / "manifest.yaml"
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("compiler package and manifest must be real files")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, Mapping):
        raise ValueError("compiler manifest must be a mapping")
    component_paths = _component_paths(manifest, root)
    missing: list[str] = []
    if not component_paths:
        missing.append(
            "manifest.yaml declares no usable components; command ownership and source edit "
            "surfaces cannot be established"
        )
    commands_by_path: dict[str, list[str]] = {}
    for command, paths in component_paths.items():
        for path in paths:
            commands_by_path.setdefault(path, []).append(command)
    symbols = tuple(
        sorted(
            (
                symbol
                for relative, commands in commands_by_path.items()
                for symbol in _symbols(root / relative, relative, tuple(sorted(commands)))
            ),
            key=lambda row: (row.path, row.line, row.symbol),
        )
    )
    symbol_index = {(row.path, row.symbol): row for row in symbols}

    raw_surfaces = (
        manifest.get("optimization_surfaces") if host_surface_declarations is None else host_surface_declarations
    )
    if raw_surfaces is None:
        missing.append(
            "manifest.yaml declares no optimization_surfaces; source symbols are indexed, but "
            "the harness cannot guess which one changes an encoding, movement, or schedule"
        )
        raw_surfaces = ()
    if not isinstance(raw_surfaces, Sequence) or isinstance(raw_surfaces, (str, bytes)):
        raise ValueError("optimization_surfaces must be a sequence")
    surfaces: list[EditSurface] = []
    from merlin.analysis.kernels.cca import action_contract as cca_contract

    editable_cca_classes = {cca_contract.LEVER, cca_contract.BACKEND_STUB}
    declared_scopes, declared_effects = _declared_surface_vocabulary(contract=contract)
    for index, raw in enumerate(raw_surfaces):
        if not isinstance(raw, Mapping):
            raise ValueError(f"optimization surface {index} is not a mapping")
        ident = str(raw.get("id") or "").strip()
        scope = str(raw.get("scope") or "").strip()
        relative = _safe_relative(raw.get("path"), f"optimization surface {ident!r}")
        symbol = str(raw.get("symbol") or "").strip()
        effects = tuple(sorted(str(value) for value in (raw.get("effects") or ())))
        raw_axes = raw.get("cca_axes") or ()
        if (
            not isinstance(raw_axes, Sequence)
            or isinstance(raw_axes, (str, bytes))
            or any(not isinstance(axis, str) or not axis for axis in raw_axes)
        ):
            raise ValueError(f"optimization surface {ident!r} has invalid cca_axes")
        cca_axes = tuple(sorted(set(raw_axes)))
        unknown_axes = [axis for axis in cca_axes if axis not in cca_contract.FIELD_REGISTRY]
        if unknown_axes:
            raise ValueError(f"optimization surface {ident!r} names unknown CCA axes {unknown_axes}")
        noneditable_axes = [
            axis for axis in cca_axes if cca_contract.FIELD_REGISTRY[axis].classification not in editable_cca_classes
        ]
        if noneditable_axes:
            raise ValueError(f"optimization surface {ident!r} maps non-editable CCA axes {noneditable_axes}")
        cca_axis_status = tuple((axis, cca_contract.FIELD_REGISTRY[axis].classification) for axis in cca_axes)
        if not ident or scope not in declared_scopes or not symbol:
            raise ValueError(f"optimization surface {index} has an invalid id/scope/symbol")
        if not effects or any(effect not in declared_effects for effect in effects):
            raise ValueError(f"optimization surface {ident!r} has invalid effects {effects}")
        resolved_symbol = symbol_index.get((relative, symbol))
        if resolved_symbol is None:
            raise ValueError(f"optimization surface {ident!r} does not resolve to AST symbol {relative}:{symbol}")
        texts = tuple(
            str(raw.get(name) or "").strip() for name in ("mechanism", "emitted_delta", "validation", "abandonment")
        )
        if not all(texts):
            raise ValueError(
                f"optimization surface {ident!r} must state mechanism, emitted delta, validation, and abandonment"
            )
        surfaces.append(
            EditSurface(
                ident, scope, relative, symbol, resolved_symbol.line, effects, cca_axes, cca_axis_status, *texts
            )
        )
    ids = [surface.id for surface in surfaces]
    if len(ids) != len(set(ids)):
        raise ValueError("optimization surface ids must be unique")
    return PackageOptimizationInventory(symbols, tuple(surfaces), tuple(missing))


def build_compiler_edit_contract(
    inventory: PackageOptimizationInventory,
    *,
    helper_extensions: Sequence[Mapping[str, Any]] = (),
    new_python_components: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe a host-frozen edit boundary; candidate manifests cannot expand it.

    The launcher must bind this object before authoring and enforce it before
    executing submitted compiler code. Returning a catalog alone is not an edit
    sandbox. Optional new-helper directories are explicit host inputs, never
    inferred from an untrusted candidate's newly declared optimization surfaces.
    Explicit ``new_python_components`` selects V2 authority for absent Python
    component directories. The inventory checks ownership and scoped intent;
    ``validate_edit_contract`` checks actual absence and real parents before
    freezing. Omitting the argument preserves the V1 document and its identity.
    """
    symbols = {(row.path, row.symbol): row for row in inventory.symbols}
    existing = []
    for surface in inventory.surfaces:
        owner = symbols.get((surface.path, surface.symbol))
        if owner is None:
            raise ValueError("edit surface must resolve to the frozen source inventory")
        existing.append(
            {
                "surface_id": surface.id,
                "path": surface.path,
                "symbol": surface.symbol,
                "kind": owner.kind,
                "line": owner.line,
            }
        )
    ids = {row["surface_id"] for row in existing}
    directories = {
        str(parent) for row in inventory.symbols for parent in PurePosixPath(row.path).parents if str(parent) != "."
    }
    extensions = []
    for extension in helper_extensions:
        directory = _safe_relative(extension.get("directory"), "helper extension directory")
        owners = extension.get("surface_ids")
        reason = extension.get("reason")
        if directory not in directories:
            raise ValueError("helper extension must be an existing scoped compiler component directory")
        if (
            not isinstance(owners, Sequence)
            or isinstance(owners, (str, bytes))
            or not owners
            or any(owner not in ids for owner in owners)
        ):
            raise ValueError("helper extension requires frozen owning surface ids")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("helper extension requires a host-declared reason")
        extensions.append({"directory": directory, "surface_ids": sorted(set(owners)), "reason": reason})
    contract = {
        "schema": "compiler_edit_contract_v1",
        "authority": "host-frozen initial inventory; candidate declarations are descriptive only",
        "existing_symbols": existing,
        "helper_extensions": extensions,
        "protected_controls": [
            "frozen source models, inputs, weights and Phase 1 outcomes",
            "reference semantics, correctness checks and waivers",
            "measurement harness, warm/ROI contract and counter interpretation",
            "target facts, hardware pins and derived ISA constants",
            "evaluator, sandbox policy, launch commands and scoring",
            "edit contract itself; candidate manifest changes grant no permissions",
        ],
        "work_order_required_fields": [
            "surface_ids",
            "source_operation_ids",
            "current_plan_digest",
            "hypothesis",
            "expected_emitted_delta",
            "semantic_obligations",
            "cheap_validation",
            "stop_or_revert_condition",
        ],
        "enforcement": "requires host pre-execution submitted-change gate; catalog is not enforcement",
        "unmapped_mechanism": "record missing edit surface and request an explicit host-approved contract extension",
    }
    if new_python_components is not None:
        if not isinstance(new_python_components, Sequence) or isinstance(new_python_components, (str, bytes)):
            raise ValueError("new Python components require an explicit sequence of host declarations")
        components = []
        component_paths: list[PurePosixPath] = []
        source_paths = {row.path for row in inventory.symbols}
        for component in new_python_components:
            if not isinstance(component, Mapping) or set(component) != {"directory", "surface_ids", "reason"}:
                raise ValueError("new Python component requires directory, owning surfaces and reason")
            directory = _safe_relative(component["directory"], "new Python component directory")
            path = PurePosixPath(directory)
            if (
                directory != component["directory"]
                or directory == "."
                or directory in directories
                or directory in source_paths
                or (str(path.parent) != "." and str(path.parent) not in directories)
                or any(part in {".git", "out", "build", "__pycache__"} for part in path.parts)
                or any(path == other or path in other.parents or other in path.parents for other in component_paths)
            ):
                raise ValueError("new Python component must reserve one absent scoped compiler directory")
            owners = component["surface_ids"]
            reason = component["reason"]
            if (
                not isinstance(owners, Sequence)
                or isinstance(owners, (str, bytes))
                or not owners
                or any(not isinstance(owner, str) or owner not in ids for owner in owners)
            ):
                raise ValueError("new Python component requires frozen owning surface ids")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("new Python component requires a host-declared reason")
            component_paths.append(path)
            components.append({"directory": directory, "surface_ids": sorted(set(owners)), "reason": reason})
        contract["schema"] = "compiler_edit_contract_v2"
        contract["new_python_components"] = components
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def guidance_for_quantized_region_plan(
    plan: Mapping[str, Any], inventory: PackageOptimizationInventory
) -> dict[str, Any]:
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
    authorized = {(owner["surface_id"], owner["path"], owner["symbol"]) for owner in contract["existing_symbols"]}
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
        if (
            not isinstance(reason_codes, Sequence)
            or isinstance(reason_codes, (str, bytes))
            or any(not isinstance(reason, str) or not reason for reason in reason_codes)
        ):
            raise ValueError(f"quantized edit requirement {role!r} has malformed reasons")
        metadata = _QUANTIZED_REGION_EDIT_ROLES[role]
        axis_matches = [
            surface
            for surface in inventory.surfaces
            if metadata["cca_axes"].intersection(surface.cca_axes)
            and (surface.id, surface.path, surface.symbol) in authorized
        ]
        effect_matches = [
            surface
            for surface in inventory.surfaces
            if metadata["effects"].intersection(surface.effects)
            and (surface.id, surface.path, surface.symbol) in authorized
        ]
        matches = axis_matches or effect_matches
        rows.append(
            {
                "role": role,
                "status": "authorized" if matches else "missing_verified_surface",
                "purpose": metadata["purpose"],
                "reason_codes": sorted(set(reason_codes)),
                "required_effects": sorted(metadata["effects"]),
                "required_cca_axes": sorted(metadata["cca_axes"]),
                "mapping_basis": (
                    "exact CCA axis" if axis_matches else "broad effect compatibility" if effect_matches else None
                ),
                "authorized_edit_surfaces": [surface.to_dict() for surface in matches],
            }
        )
    protected = plan.get("protected_policy_blockers")
    if not isinstance(protected, Sequence) or isinstance(protected, (str, bytes)):
        raise ValueError("quantized region plan has malformed protected policy blockers")
    return {
        "schema": "agent_quantized_region_guidance_v1",
        "plan_source_sha256": plan.get("normalized_source_sha256"),
        "plan_status": plan.get("status"),
        "edit_contract_sha256": contract["sha256"],
        "requirements": rows,
        "protected_inputs": [
            {
                "name": "accuracy_and_corpus_policy",
                "editable_by_candidate": False,
                "reason_codes": sorted(set(str(reason) for reason in protected)),
                "required_action": (
                    "the experiment owner must provide a reviewed, source-bound policy and corpus; "
                    "compiler edits cannot self-authorize approximation"
                ),
            }
        ],
        "authority": (
            "only surfaces resolved from the host-frozen compiler inventory and named by the "
            "bound compiler_edit_contract_v1 are returned"
        ),
        "candidate_manifest_grants_authority": False,
        "compiler_edit_contract": contract,
    }


def guidance_for_report(report: WholeModelReport, inventory: PackageOptimizationInventory) -> dict[str, Any]:
    """Join ranked whole-model problems to declared, verified compiler edit surfaces."""
    actions: list[dict[str, Any]] = []
    for opportunity in report.opportunities:
        effects = _OPPORTUNITY_EFFECTS.get(opportunity.kind, frozenset())
        axes = _OPPORTUNITY_AXES.get(opportunity.kind, frozenset())
        axis_matches = [surface for surface in inventory.surfaces if axes.intersection(surface.cca_axes)]
        effect_matches = [surface for surface in inventory.surfaces if effects.intersection(surface.effects)]
        matches = axis_matches or effect_matches
        actions.append(
            {
                "opportunity": opportunity.to_dict(),
                "required_effects": sorted(effects),
                "required_cca_axes": sorted(axes),
                "edit_surfaces": [surface.to_dict() for surface in matches],
                "status": "actionable" if matches else "unmapped",
                "mapping_basis": (
                    "exact CCA axis" if axis_matches else "broad effect compatibility" if effect_matches else None
                ),
                "missing": (
                    None if matches else "no verified manifest surface declares an effect that addresses this gap"
                ),
            }
        )
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
            "full-size execution is optional post-freeze validation and is not required by Phase 2"
        ),
    }


def _host_dynamic_total(host_activity: Mapping[str, Any]) -> int:
    """Host-lane dynamic operations, summed over families."""
    families = host_activity.get("dynamic_operations")
    if not isinstance(families, Mapping):
        return 0
    return sum(value for value in families.values() if isinstance(value, int) and not isinstance(value, bool))


def _host_tasks_by_cost(host_activity: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    """Host tasks ranked by DYNAMIC OPERATIONS, with the family that dominates each.

    The pre-existing ranking is `top_tasks_by_scalar_memory_payload` -- ordered by BYTES. Bytes say
    how much data a task touched, never how much work it did: a task holding 4.0% of the host
    lane's operations appears in that list with 3,200 bytes while another with 12.5% carries 25,216,
    so the byte order and the cost order are different orders. An optimizer needs the cost one.
    """
    tasks = host_activity.get("tasks")
    if not isinstance(tasks, list):
        return []
    whole = _host_dynamic_total(host_activity)
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        families = {
            name: value
            for name, value in (task.get("dynamic_operations") or {}).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        total = sum(families.values())
        if not total:
            continue
        dominant = max(families.items(), key=lambda item: (item[1], item[0]))
        regions = [str(name) for name in (task.get("source_regions") or [])]
        rows.append(
            (
                total,
                {
                    "task_index": index,
                    "dynamic_operations": total,
                    "share_of_host_dynamic_operations": round(total / whole, 6) if whole else None,
                    "dominant_family": dominant[0],
                    "dominant_family_operations": dominant[1],
                    "source_regions": regions[:12],
                    "source_region_count": len(regions),
                },
            )
        )
    rows.sort(key=lambda item: -item[0])
    return [row for _, row in rows[:limit]]


def _dominant_signatures(host_activity: Mapping[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    """The top emitter shapes by dynamic operations, each with its share of the host lane.

    A signature is a block's operation-name sequence; blocks sharing one are the same emitter
    code instantiated at several sites. The share is against the WHOLE host lane so a reader can
    tell a shape worth 37% from one worth 0.3% without a second lookup.
    """
    rows = host_activity.get("block_signatures")
    families = host_activity.get("dynamic_operations")
    if not isinstance(rows, list) or not isinstance(families, Mapping):
        return []
    whole = sum(v for v in families.values() if isinstance(v, int) and not isinstance(v, bool))
    out = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        total = row.get("dynamic_total")
        out.append(
            {
                "signature": row.get("signature"),
                "blocks": row.get("blocks"),
                "trips": row.get("trips"),
                "operations_per_trip": row.get("operations_per_trip"),
                "dynamic_total": total,
                "share_of_host_dynamic_operations": (
                    round(total / whole, 6) if whole and isinstance(total, int) else None
                ),
                "dynamic_operations": row.get("dynamic_operations"),
            }
        )
    return out


def _host_hotspot_share(host_activity: Mapping[str, Any]) -> float | None:
    """Share of the host lane held by its single largest task, or ``None`` when unmeasurable."""
    ranked = _host_tasks_by_cost(host_activity, limit=1)
    if not ranked:
        return None
    share = ranked[0].get("share_of_host_dynamic_operations")
    return float(share) if isinstance(share, (int, float)) else None


def _resource_cost(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    declared_rates: Any,
    design: Any = None,
) -> dict[str, Any]:
    """Per-resource deltas for this candidate, with a composite only if every moved term is priced.

    ``declared_rates`` is whatever the analysis carries under ``resource_rates`` -- a mapping of
    term to ``{cycles_per_unit, status, provenance}``. It is normally absent, and that is the
    HONEST state today: no measured exchange rate between the host lane and the accelerator's
    accompanies a static emission analysis, so the composite is withheld and the per-term table --
    the half that cannot mislead -- is still reported.

    ``design`` is the DEVICE the analysis is about, as the pin registry names it. When the analysis
    resolves one, the measured ledger (:mod:`merlin.analysis.perf.contract.terms`) supplies the rates for it
    -- which is the half that was missing: this loop could NAME a trade and not score one, not
    because a model was absent but because the one measured rate the project had lived in a test
    file where nothing could cite it. A rate is taken ONLY for the named device (rates from another
    device are not weaker answers, they are wrong ones) and only if its closure was measured and
    holds; every entry the ledger refuses is reported under ``rates_unpriced`` with the reason, so a
    withheld composite says what is missing rather than merely that something is.

    A rate whose provenance is blank is REJECTED BY NAME rather than used or dropped. Using it
    would let an invented number decide which optimizations this loop pursues; dropping it silently
    would make an unusable declaration indistinguishable from an absent one.
    """
    from merlin.analysis.perf.contract.terms import UNPRICED, Rate, compare_terms

    rates: dict[str, Rate] = {}
    rejected: list[str] = []
    if isinstance(declared_rates, Mapping):
        for term, spec in declared_rates.items():
            if not isinstance(spec, Mapping):
                rejected.append(str(term))
                continue
            status = str(spec.get("status") or UNPRICED)
            if status == UNPRICED:
                continue
            value, provenance = spec.get("cycles_per_unit"), str(spec.get("provenance") or "")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not provenance.strip():
                rejected.append(str(term))
                continue
            rates[str(term)] = Rate(float(value), status, provenance)

    unpriced: dict[str, str] = {}
    ledger_priced: list[str] = []
    if isinstance(design, str) and design.strip():
        from merlin.analysis.perf.contract import terms as exchange_rates

        try:
            measured = exchange_rates.rates_for_design(design)
            unpriced = exchange_rates.unpriced_reasons(design)
        except exchange_rates.ExchangeRatesError as broken:
            # A ledger that cannot be read is a defect in the ledger, not a licence to price with
            # nothing: the terms stay unpriced and the reason travels with the withheld composite.
            unpriced = {"<ledger>": str(broken)}
            measured = {}
        for term, rate in measured.items():
            # A rate the ANALYSIS declared wins: it is about this measurement, the ledger's is about
            # the device in general. Nothing is overwritten silently.
            if term not in rates:
                rates[term] = rate
                ledger_priced.append(term)

    out = compare_terms(before, after, rates)
    out["rates_priced"] = sorted(rates)
    out["rates_rejected"] = sorted(rejected)
    out["rates_from_ledger"] = sorted(ledger_priced)
    out["rates_unpriced"] = dict(sorted(unpriced.items()))
    out["rates_design"] = design if isinstance(design, str) and design.strip() else None
    return out
