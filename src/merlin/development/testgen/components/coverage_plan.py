"""Immutable external reviewed component coverage declarations and source binding."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.digest import is_sha256
from merlin.development.targetgen.sources.frontend_trace import original_operation_semantics
from merlin.development.testgen.components.component_generation import digest

# ---- component_coverage_plan -------------------------------------------------------------------

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
            from merlin.development.testgen.components.component_execution_budget import validate

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

            validate_obligation_links(semantic_basis, row, effect_owners)
            base = row["base"]
            if not isinstance(base, dict) or set(base) & _FORBIDDEN or not isinstance(base.get("op"), str):
                raise ValueError("component base must name an independent operation without source/capture selectors")
            if base.get("kind", "isa") not in {"isa", "layer", "model_slice"}:
                raise ValueError("component coverage refuses whole models")
            if "graph_family" in base:
                from merlin.development.testgen.components.component_graph_variants import (
                    validate,
                    validate_owners,
                )

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


# ---- component_semantic_basis ------------------------------------------------------------------
# Typed preauthor selection of independent example graph semantics.
#
# Only explicit reviewed operation/effect semantics leave this owner. Original
# source bytes are ordinary selected inputs to the existing private freeze. Review
# status declares intent; protected source selection establishes review authority.

SCHEMA = "merlin.component_semantic_basis.v1"
PROVENANCE = {"role": "independent_training_example", "visibility": "public", "selection": "before_authoring"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(value):
    return isinstance(value, str) and value and all(c.isalnum() or c in "_-" for c in value)


def _operations(values):
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and value for value in values)
        and len(set(values)) == len(values)
    )


@dataclass(frozen=True)
class BasisSource:
    path: str
    sha256: str
    role: str

    def to_dict(self):
        return {"path": self.path, "sha256": self.sha256, "role": self.role}


def _read(pin, *, parent, routing, role):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not is_sha256(pin["sha256"]):
        raise ValueError("semantic basis source needs exactly an explicit path and SHA256")
    if not isinstance(pin["path"], str) or not pin["path"]:
        raise ValueError("semantic basis source needs a nonempty path")
    path = Path(pin["path"])
    path = (parent / path).absolute() if not path.is_absolute() else path.absolute()
    if routing:
        if str(path) not in routing:
            raise ValueError("semantic basis source is outside the frozen selected input inventory")
        path = Path(routing[str(path)])
    if path.is_symlink() or not path.is_file():
        raise ValueError("semantic basis source must be an ordinary selected file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("selected semantic basis source bytes changed")
    return raw, BasisSource(str(path.resolve()), pin["sha256"], role)


@dataclass(frozen=True)
class ComponentSemanticBasis:
    source: BasisSource
    declaration_json: str
    semantics_json: str
    graph_sources: tuple[BasisSource, ...]

    def semantics(self):
        """A decoded copy excludes paths, shapes, counts and objective weights."""
        return json.loads(self.semantics_json)

    def sources(self):
        return [source.to_dict() for source in (self.source, *self.graph_sources)]

    def reviewed_semantics(self):
        return {
            "schema": SCHEMA,
            "sha256": self.source.sha256,
            "semantics": self.semantics(),
        }

    def record(self):
        return {**self.reviewed_semantics(), "selected_sources": self.sources()}

    @classmethod
    def from_recipe(cls, recipe, *, routing=None):
        if recipe is None:
            return None
        if routing is None:
            from merlin.experiments.frozen_python import active_source_identity

            routing = (
                json.loads(os.environ.get("MERLIN_PHASE0_FROZEN_SOURCE_MAP", "{}")) if active_source_identity() else {}
            )
        recipe = Path(recipe).resolve()
        document = yaml.safe_load(recipe.read_bytes())
        if not isinstance(document, dict):
            raise ValueError("semantic basis recipe must be an explicit mapping")
        selection = document.get("semantic_basis")
        if selection is None:
            return None
        original = next((Path(old) for old, frozen in routing.items() if frozen == str(recipe)), recipe)
        raw, source = _read(selection, parent=original.parent, routing=routing, role="semantic-basis-roster")
        original_roster = Path(selection["path"])
        if not original_roster.is_absolute():
            original_roster = original.parent / original_roster
        return cls.load(raw, source=source, parent=original_roster.absolute().parent, routing=routing)

    @classmethod
    def load(cls, raw, *, source, parent, routing):
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError("selected semantic basis declaration bytes changed")
        document = yaml.safe_load(raw)
        if (
            not isinstance(document, dict)
            or set(document) != {"schema", "status", "provenance", "members"}
            or document["schema"] != SCHEMA
            or document["status"] != "reviewed"
            or document["provenance"] != PROVENANCE
        ):
            raise ValueError("semantic basis requires the closed reviewed public preauthor example schema")
        members = document["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("semantic basis requires a nonempty explicitly selected example roster")
        semantics, sources, seen, paths = [], [], set(), set()
        for member in members:
            fields = {"id", "kind", "path", "sha256", "schema", "operation_semantics", "effect_semantics"}
            if not isinstance(member, dict) or set(member) != fields or not _id(member["id"]):
                raise ValueError("semantic basis example must declare the closed source and semantics schema")
            if member["id"] in seen or member["kind"] != "model2mlir_frontend_trace":
                raise ValueError("semantic basis example id must be unique and source kind supported")
            seen.add(member["id"])
            graph_raw, graph_source = _read(
                {key: member[key] for key in ("path", "sha256")},
                parent=parent,
                routing=routing,
                role="semantic-basis-graph",
            )
            if graph_source.path in paths:
                raise ValueError("semantic basis graph source is selected more than once")
            paths.add(graph_source.path)
            trace = json.loads(graph_raw)
            if not isinstance(trace, dict):
                raise ValueError("semantic basis graph must be an explicit frontend trace mapping")
            provenance = trace.get("provenance")
            if provenance is not None and provenance != PROVENANCE:
                raise ValueError("semantic basis graph is marked with a different source provenance")
            if member["schema"] != trace.get("schema"):
                raise ValueError("semantic basis source schema disagrees with selected graph")
            graph_sha256, actual_operations = original_operation_semantics(trace)
            operations = member["operation_semantics"]
            if not _operations(operations) or set(operations) != set(actual_operations):
                raise ValueError("reviewed semantic operation roster disagrees with original graph call targets")
            effects, effect_ids = member["effect_semantics"], set()
            if not isinstance(effects, list):
                raise ValueError("semantic basis effects must be an explicit reviewed list")
            for effect in effects:
                if (
                    not isinstance(effect, dict)
                    or set(effect) != {"id", "kind", "basis", "operation_semantics"}
                    or not _id(effect["id"])
                    or effect["id"] in effect_ids
                    or any(not isinstance(effect[key], str) or not effect[key].strip() for key in ("kind", "basis"))
                    or not _operations(effect["operation_semantics"])
                    or not set(effect["operation_semantics"]) <= set(actual_operations)
                ):
                    raise ValueError("reviewed semantic effects must bind unique concrete graph operation semantics")
                effect_ids.add(effect["id"])
            semantics.append(
                {
                    "id": member["id"],
                    "graph_sha256": graph_sha256,
                    "operation_semantics": sorted(operations),
                    "effect_semantics": effects,
                }
            )
            sources.append(graph_source)
        return cls(source, _json(document), _json(semantics), tuple(sources))


def validate_obligation_links(basis, row, effect_owners):
    """Check explicit reviewed correspondences; never infer lowering rules."""
    links = row.get("semantic_basis")
    if basis is None:
        if links is not None:
            raise ValueError("obligation semantic links require a selected reviewed example basis")
        return
    if not isinstance(links, list) or not links:
        raise ValueError("selected semantic basis requires explicit obligation correspondences")
    examples = {example["id"]: example for example in basis.semantics()}
    seen = set()
    mapped_operations = set()
    for link in links:
        if not isinstance(link, dict) or set(link) != {"member", "operations", "effects"}:
            raise ValueError("obligation semantic basis link needs member, operations and effects")
        if link["member"] not in examples or link["member"] in seen:
            raise ValueError("obligation semantic basis member is unknown or duplicated")
        seen.add(link["member"])
        example = examples[link["member"]]
        effects = {effect["id"]: effect for effect in example["effect_semantics"]}
        for key, declarations, owners in (
            ("operations", set(example["operation_semantics"]), set(row["operations"])),
            ("effects", set(effects), set(row["effects"])),
        ):
            mappings = link[key]
            if not isinstance(mappings, list) or (key == "operations" and not mappings):
                raise ValueError("obligation semantic mappings must be explicit lists")
            pairs = set()
            for mapping in mappings:
                if (
                    not isinstance(mapping, dict)
                    or set(mapping) != {"source", "owner"}
                    or not isinstance(mapping["source"], str)
                    or not isinstance(mapping["owner"], str)
                    or mapping["source"] not in declarations
                    or mapping["owner"] not in owners
                    or (mapping["source"], mapping["owner"]) in pairs
                ):
                    raise ValueError("obligation semantic mapping differs from selected source or reviewed owner")
                pairs.add((mapping["source"], mapping["owner"]))
                if key == "operations":
                    mapped_operations.add(mapping["owner"])
                if key == "effects" and effects[mapping["source"]]["kind"] != effect_owners[mapping["owner"]]["kind"]:
                    raise ValueError("obligation semantic effect kind differs from its reviewed source")
    if mapped_operations != set(row["operations"]):
        raise ValueError("obligation has a reviewed operation owner without a selected source correspondence")
