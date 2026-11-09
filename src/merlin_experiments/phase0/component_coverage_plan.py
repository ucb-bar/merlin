"""Immutable external reviewed component coverage declarations and source binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .component_generation import digest

PLAN_SCHEMA = "merlin.component_coverage_plan.v1"
BUDGETED_PLAN_SCHEMA = "merlin.component_coverage_plan.v2"


COHORTS = ("development", "functional_guard", "withheld_transfer")


_FORBIDDEN = {
    "name",
    "cat",
    "label",
    "source_role",
    "source_reference",
    "source",
    "pytorch_ref",
    "model",
    "capture",
    "capture_dir",
    "micro_model",
    "materialized_capture",
    "quant_recipe",
    "quant_scheme",
    "loader",
    "spec_ref",
    "capture_op",
    "capture_dtype",
    "application_signature_match",
    "global_objective",
    "component_coverage",
    "graph_variant",
    "_component_source_unavailable",
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_id(value):
    return isinstance(value, str) and value and all(c.isalnum() or c in "_-" for c in value)


@dataclass(frozen=True)
class ComponentObligation:
    """Immutable reviewed source declaration; mutable mappings are decoded copies."""

    id: str
    mandatory: bool
    cohort: str
    declaration_json: str

    def to_dict(self):
        return json.loads(self.declaration_json)


@dataclass(frozen=True)
class ComponentCoveragePlan:
    source_path: str
    source_sha256: str
    declaration_json: str
    obligations: tuple[ComponentObligation, ...]

    def to_dict(self):
        return json.loads(self.declaration_json)

    @classmethod
    def load(cls, path, *, evidence, semantic_basis=None):
        """Bind review, numerical policy, effects and selected external facts."""
        raw = Path(path).read_bytes()
        document = yaml.safe_load(raw)
        fields = {
            "schema",
            "status",
            "hardware",
            "software_spec_sha256",
            "numerical_semantics_sha256",
            "effects",
            "budget",
            "obligations",
        }
        if isinstance(document, dict) and document.get("schema") == BUDGETED_PLAN_SCHEMA:
            fields.add("execution_budget")
        if semantic_basis is not None:
            fields.add("semantic_basis_sha256")
        if not isinstance(document, dict) or set(document) != fields:
            raise ValueError("component coverage plan must declare its complete closed versioned schema")
        if semantic_basis is not None and document["semantic_basis_sha256"] != semantic_basis.source.sha256:
            raise ValueError("component coverage semantic basis differs from selected reviewed source bytes")
        if document["schema"] not in {PLAN_SCHEMA, BUDGETED_PLAN_SCHEMA} or document["status"] != "reviewed":
            raise ValueError("component coverage requires a reviewed versioned plan")
        if document["schema"] == BUDGETED_PLAN_SCHEMA:
            from .component_execution_budget import validate

            validate(document["execution_budget"])
        if evidence is None or evidence.software_spec.get("status") != "reviewed":
            raise ValueError("component coverage requires reviewed selected software and hardware evidence")
        hardware = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
        if any(value is None for value in hardware.values()) or document["hardware"] != hardware:
            raise ValueError("component coverage hardware identity differs from selected evidence")
        sources = [row for row in evidence.source_snapshots if row.role == "software-spec"]
        if len(sources) != 1 or document["software_spec_sha256"] != sources[0].sha256:
            raise ValueError("component coverage software identity differs from selected source bytes")
        if document["numerical_semantics_sha256"] != digest(evidence.software_spec["numerical_semantics"]):
            raise ValueError("component coverage numerical permissions differ from selected semantics")
        owners = {row["id"]: row for row in evidence.software_spec["operations"]}
        effects = document["effects"]
        if not isinstance(effects, list):
            raise ValueError("component effects must be an explicit reviewed declaration list")
        effect_owners = {}
        for row in effects:
            if (
                not isinstance(row, dict)
                or set(row) != {"id", "status", "kind", "basis"}
                or not _safe_id(row["id"])
                or row["id"] in effect_owners
                or row["status"] != "reviewed"
                or any(not isinstance(row[key], str) or not row[key].strip() for key in ("kind", "basis"))
            ):
                raise ValueError("each effect needs a unique id, reviewed status, kind and semantic basis")
            effect_owners[row["id"]] = row
        budget = document["budget"]
        if (
            not isinstance(budget, dict)
            or set(budget) != {"max_members", "max_interaction_cells"}
            or any(type(value) is not int or value < 1 for value in budget.values())
        ):
            raise ValueError("component coverage budget must explicitly bound members and interaction cells")
        declarations = document["obligations"]
        if not isinstance(declarations, list) or not declarations:
            raise ValueError("component coverage requires a nonempty obligation list")
        obligations, seen = [], set()
        for row in declarations:
            required = {
                "id",
                "mandatory",
                "cohort",
                "operations",
                "effects",
                "expectation",
                "frontend",
                "base",
                "axes",
                "interactions",
            }
            if semantic_basis is not None:
                required.add("semantic_basis")
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("component obligation must declare the complete closed v1 schema")
            if not _safe_id(row["id"]) or row["id"] in seen or type(row["mandatory"]) is not bool:
                raise ValueError("component obligation needs a unique safe id and explicit mandatory Boolean")
            seen.add(row["id"])
            if row["cohort"] not in COHORTS or row["expectation"] not in {"admitted_program", "unsupported_program"}:
                raise ValueError("component obligation has an unknown cohort or expectation")
            if row["frontend"] not in {"mlir", "pytorch"}:
                raise ValueError("component frontend must be normal generated MLIR or generated PyTorch")
            if row["cohort"] == "development" and row["expectation"] != "admitted_program":
                raise ValueError("development performance members must request admitted programs")
            for key, selected in (("operations", owners), ("effects", effect_owners)):
                values = row[key]
                if (
                    not isinstance(values, list)
                    or (key == "operations" and not values)
                    or len(set(values)) != len(values)
                    or any(
                        value not in selected or selected[value].get("status", "reviewed") != "reviewed"
                        for value in values
                    )
                ):
                    raise ValueError(f"obligation {row['id']}: {key} must name unique reviewed declarations")
            from .component_semantic_basis import validate_obligation_links

            validate_obligation_links(semantic_basis, row, effect_owners)
            base = row["base"]
            if not isinstance(base, dict) or set(base) & _FORBIDDEN or not isinstance(base.get("op"), str):
                raise ValueError("component base must name an independent operation without source/capture selectors")
            if base.get("kind", "isa") not in {"isa", "layer", "model_slice"}:
                raise ValueError("component coverage refuses whole models")
            if "graph_family" in base:
                from .component_graph_variants import validate, validate_owners

                validate(base["graph_family"])
                validate_owners([owners[name] for name in row["operations"]])
                if (
                    document["schema"] != BUDGETED_PLAN_SCHEMA
                    or base["op"] != "component_program"
                    or "program" in base
                    or row["frontend"] != "mlir"
                    or row["expectation"] != "admitted_program"
                ):
                    raise ValueError("graph family needs budgeted independent MLIR semantics and no authored program")
            if row["cohort"] == "development" and not isinstance(base.get("performance"), dict):
                raise ValueError("development coverage must use the normal performance declaration")
            if row["cohort"] != "development" and "performance" in base:
                raise ValueError(
                    "functional and withheld coverage have separate membership from development performance"
                )
            axes = row["axes"]
            if not isinstance(axes, dict) or any(not _safe_id(key) for key in axes):
                raise ValueError("component axes must be an explicit safe-name mapping")
            for name, axis in axes.items():
                if not isinstance(axis, dict) or axis.get("kind") not in {"extent", "choice", "resource"}:
                    raise ValueError(f"component axis {name}: unknown axis kind")
                if axis["kind"] == "resource":
                    if set(axis) != {"kind", "declaration"} or not isinstance(axis["declaration"], dict):
                        raise ValueError("resource axis requires one explicit boundary declaration")
                elif set(axis) != {"kind", "values"} or not isinstance(axis["values"], list) or not axis["values"]:
                    raise ValueError("component axis requires a nonempty explicit values list")
            groups = row["interactions"]
            if (
                not isinstance(groups, list)
                or any(
                    not isinstance(group, list)
                    or len(group) < 2
                    or len(set(group)) != len(group)
                    or any(name not in axes for name in group)
                    for group in groups
                )
                or len({tuple(sorted(group)) for group in groups}) != len(groups)
            ):
                raise ValueError("component interactions must name unique declared interacting axis sets")
            obligations.append(ComponentObligation(row["id"], row["mandatory"], row["cohort"], _json(row)))
        return cls(str(Path(path).resolve()), hashlib.sha256(raw).hexdigest(), _json(document), tuple(obligations))
