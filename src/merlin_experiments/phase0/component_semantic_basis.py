"""Typed preauthor selection of independent example graph semantics.

Only explicit reviewed operation/effect semantics leave this owner. Original
source bytes are ordinary selected inputs to the existing private freeze. Review
status declares intent; protected source selection establishes review authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.digest import is_sha256
from merlin.targetgen.frontend_trace import original_operation_semantics

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
            from merlin_experiments.frozen_python import active_source_identity

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
