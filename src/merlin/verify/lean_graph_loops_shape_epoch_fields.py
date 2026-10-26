"""Data-only inventory of retained ShapeEpoch trees and their bound fields.

This review aid consumes existing JSON exports. It neither parses compiler IR,
executes a checker, loads Lean, nor establishes semantic acceptance. Matching
retained trees is consistency evidence, not independent projection review.
"""

from __future__ import annotations

import hashlib
import json

OP_FIELDS = frozenset(("name", "attributes", "properties", "location", "index_constant", "operands", "results", "regions"))
BLOCK_FIELDS = frozenset(("arguments", "operations"))


class FieldMapError(ValueError):
    """The retained representation is unsupported or internally inconsistent."""


def require(condition, message):
    if not condition:
        raise FieldMapError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def inventory_tree(tree):
    """Enumerate every raw field and check lexical uses in the retained tree.

    Global IDs come from the existing exporter. Region arguments and results
    must be fresh, and a nested result cannot leak into its surrounding scope.
    This checks the serialized syntax, not the compiler parser or operation
    typing/control semantics.
    """
    declarations = {}
    fields, operations, uses = [], [], []

    def declare(rows, path):
        require(isinstance(rows, list), path + " must be a list")
        names = []
        for index, row in enumerate(rows):
            location = f"{path}/{index}"
            require(isinstance(row, list) and len(row) == 2, location + " must contain ID and type")
            name, spelling = row
            require(type(name) is int and name >= 0, location + " has invalid SSA ID")
            require(isinstance(spelling, str) and spelling, location + " has no type spelling")
            require(name not in declarations, location + " duplicates an SSA definition")
            declarations[name] = dict(path=location, type=spelling)
            names.append(name)
        return names

    def properties(rows, path):
        require(isinstance(rows, list), path + " must be a list")
        names = set()
        for index, row in enumerate(rows):
            require(isinstance(row, list) and len(row) == 2, path + " has invalid metadata pair")
            name, value = row
            require(isinstance(name, str) and isinstance(value, str), path + " metadata must retain text")
            require(name not in names, path + " duplicates metadata name")
            names.add(name)
            fields.append(dict(path=f"{path}/{index}", value=row))

    def operation(node, path, scope):
        require(isinstance(node, dict) and set(node) == OP_FIELDS, path + " operation fields differ")
        require(isinstance(node["name"], str) and node["name"], path + " has no operation name")
        require(isinstance(node["location"], str), path + " has invalid location")
        require(node["index_constant"] is None or type(node["index_constant"]) is int, path + " constant is invalid")
        for field in sorted(OP_FIELDS):
            fields.append(dict(path=path + "/" + field, value_digest=digest(node[field])))
        properties(node["attributes"], path + "/attributes")
        properties(node["properties"], path + "/properties")
        require(isinstance(node["operands"], list), path + " operands must be a list")
        for index, name in enumerate(node["operands"]):
            require(type(name) is int and name in scope, path + " uses SSA outside lexical scope")
            uses.append(dict(path=f"{path}/operands/{index}", name=name, definition=declarations[name]["path"]))
        require(isinstance(node["regions"], list), path + " regions must be a list")
        for region_index, region in enumerate(node["regions"]):
            require(isinstance(region, list), path + " region must be a block list")
            for block_index, block in enumerate(region):
                block_path = f"{path}/regions/{region_index}/{block_index}"
                require(isinstance(block, dict) and set(block) == BLOCK_FIELDS, block_path + " fields differ")
                for field in sorted(BLOCK_FIELDS):
                    fields.append(dict(path=block_path + "/" + field, value_digest=digest(block[field])))
                local = scope | set(declare(block["arguments"], block_path + "/arguments"))
                require(isinstance(block["operations"], list), block_path + " operations must be a list")
                for child_index, child in enumerate(block["operations"]):
                    local |= set(operation(child, f"{block_path}/operations/{child_index}", local))
        results = declare(node["results"], path + "/results")
        operations.append(dict(path=path, name=node["name"], operation_digest=digest(node), results=results))
        return results

    operation(tree, "", set())
    return dict(
        tree_digest=digest(tree), fields=fields, operations=sorted(operations, key=lambda row: row["path"]),
        definitions=declarations, uses=uses, lexical_uses_checked=True,
        compiler_parser_or_typing_verified=False, semantic_acceptance=False,
    )


def bound_json(text, expected, name):
    require(isinstance(text, str), name + " binding is not serialized JSON")
    try:
        actual = json.loads(text)
    except (ValueError, TypeError) as error:
        raise FieldMapError(name + " binding is malformed JSON") from error
    require(actual == expected, name + " complete tree differs from its retained binding")
    return dict(name=name, tree_digest=digest(expected), retained_binding_matches=True)


def report(shape_capture, memory_capture, model):
    """Map retained complete trees to the exact existing query data.

    The report intentionally does not certify the final masked reuse/segment
    relation: that relation belongs to the recorded host matcher and Lean roots.
    It inventories the entire final tree and identifies that separate obligation.
    """
    require(shape_capture["producer_revision"] == memory_capture["producer_revision"], "producer revisions differ")
    require(shape_capture["after"] == memory_capture["before"], "submitted normalized buffer references differ")
    source = shape_capture["before_export"]["tree"]
    normalized = shape_capture["after_export"]["tree"]
    require(normalized == memory_capture["before_export"]["tree"], "normalized middle trees differ")
    reused = memory_capture["middle_export"]["tree"]
    segmented = memory_capture["after_export"]["tree"]
    trees = dict(source=source, normalized=normalized, reused=reused, segmented=segmented)
    inventories = {name: inventory_tree(tree) for name, tree in trees.items()}
    bindings = [
        bound_json(model["shape"]["source_binding"], source, "source"),
        bound_json(model["shape"]["target_binding"], normalized, "normalized"),
        bound_json(model["memory"]["middleBinding"], reused, "reused"),
    ]
    source_ops = {canonical(node): path for path, node in _nodes(source)}
    changed = []
    for index, site in enumerate(model["shape"]["changed_sites"]):
        operation = json.loads(site["source_operation"])
        key = canonical(operation)
        require(key in source_ops, "changed site is absent from complete source tree")
        require(site["kind"] in ("cast", "dimension"), "unknown shape rewrite kind")
        expected = "memref.cast" if site["kind"] == "cast" else "memref.dim"
        require(operation["name"] == expected, "changed-site kind disagrees with actual source operation")
        changed.append(dict(site=index, kind=site["kind"], source_path=source_ops[key], source_operation_digest=digest(operation)))
    require(len({row["source_path"] for row in changed}) == len(changed), "changed source site was repeated")
    plan = model["step_binding"]
    instructions = model["memory"]["reuse"]["instructions"]
    require(len(plan["steps"]) + 1 == len(instructions), "complete command/return count differs")
    for index, step in enumerate(plan["steps"]):
        require(step["position"] == index and step["instruction"] == instructions[index], "ordered command binding differs")
    require(plan["terminal"] == instructions[-1], "complete return binding differs")
    return dict(
        schema="merlin.graph.shape_epoch.retained_field_map.v1", producer_revision=shape_capture["producer_revision"],
        inventories=inventories, complete_tree_bindings=bindings, changed_sites=changed,
        ordered_commands=len(plan["steps"]), complete_instructions=len(instructions),
        final_masked_relation="Separate existing reuse/SCF matcher and audited Lean roots; not recomputed by this inventory",
        qualified=False, semantic_acceptance=False, independent_projection_review=False,
        raw_parser_trusted=True, no_lean_or_sdk_access=True,
    )


def _nodes(node, path=""):
    yield path, node
    for region_index, region in enumerate(node["regions"]):
        for block_index, block in enumerate(region):
            for index, child in enumerate(block["operations"]):
                yield from _nodes(child, f"{path}/regions/{region_index}/{block_index}/operations/{index}")
