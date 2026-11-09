"""Source-only proof for a typed captured program that directly returns inputs.

This is not host arithmetic admission, numerical validation, or a compiler pass.
Only an exact linked build may discharge the pending source support obligation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.runtime.modelinputs import file_sha256

PENDING = "source_structural_direct_return_pending_build"
LINKED = "source_structural_direct_return_linked"
SCOPE = "typed direct-return alias stage; no computation or numerical equivalence claim"
_TRACE_DTYPE = {
    "bool": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"pure direct-return source: {reason}")


def _mapping(value: Any) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), "malformed captured evidence")
    return value


def _read_bound(capture: Path, receipt: Mapping, name: str) -> tuple[Path, Any]:
    path = capture / name
    entry = _mapping(_mapping(receipt.get("artifacts")).get(name))
    _need(
        not path.is_symlink() and path.is_file() and entry.get("sha256") == file_sha256(path),
        f"{name} is absent or differs from the capture receipt",
    )
    return path, json.loads(path.read_bytes())


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _abi(value: Any) -> list[dict[str, Any]]:
    _need(isinstance(value, list) and bool(value), "missing complete tensor ABI")
    result = []
    for entry in value:
        _need(isinstance(entry, Mapping), "malformed tensor ABI")
        shape, dtype = entry.get("shape"), entry.get("dtype")
        _need(
            isinstance(shape, list)
            and all(type(n) is int and n >= 0 for n in shape)
            and isinstance(dtype, str)
            and bool(dtype),
            "unknown or dynamic tensor ABI",
        )
        result.append({"shape": shape, "dtype": dtype})
    return result


def _host_representation_types(selected: Mapping) -> set[str]:
    """Conservative source type screen from reviewed selected declarations, not host approval."""
    types: set[str] = set()
    _need(bool(selected), "no selected reviewed host capability profile")
    for profile in selected.values():
        profile = _mapping(profile)
        document = _mapping(profile.get("capability_spec"))
        _need(
            profile.get("status") == "reviewed" and document.get("status") == "reviewed",
            "host capability profile is not reviewed",
        )
        operations = document.get("operations")
        _need(isinstance(operations, list), "host capability has no operation roster")
        for operation in operations:
            operation = _mapping(operation)
            # An operation without its own status inherits the document's,
            # which was required to be reviewed above.
            if "status" in operation and operation["status"] != "reviewed":
                continue
            signature = _mapping(operation.get("signature"))
            for key in ("dtypes", "ordered_operand_dtypes", "ordered_result_dtypes"):
                values = signature.get(key, [])
                _need(
                    isinstance(values, list) and all(isinstance(v, str) and v for v in values),
                    "host capability has malformed dtype evidence",
                )
                types.update(values)
    _need(bool(types), "reviewed host capabilities declare no representation type")
    return types


def _tensor_abi(value: Any) -> dict[str, Any]:
    _need(str(value).startswith("tensor<"), "function ABI contains a non-tensor value")
    shape, dtype = mq.type_shape_dtype(value)
    return _abi([{"shape": shape, "dtype": dtype}])[0]


def _graph_mapping(graph: Any, input_abi: list[dict], output_abi: list[dict]) -> tuple[list[int], dict, list[str]]:
    graph = _mapping(graph)
    nodes, edges = graph.get("nodes"), graph.get("edges")
    _need(
        graph.get("status") == "complete" and isinstance(nodes, list) and isinstance(edges, list),
        "frontend graph is incomplete",
    )
    _need(not graph.get("subgraphs"), "frontend graph has an unaccounted nested computation")
    _need(
        len(nodes) >= len(input_abi) + 1 and all(isinstance(node, Mapping) for node in nodes),
        "frontend graph has missing or malformed nodes",
    )
    ids = [node.get("id") for node in nodes]
    _need(
        all(isinstance(node_id, str) and node_id for node_id in ids) and len(ids) == len(set(ids)),
        "frontend graph node identities are duplicated or absent",
    )
    roots: dict[str, int] = {}
    values: dict[str, dict] = {}
    input_names: list[str] = []
    expected_edges = []
    aliases = {}
    for index, node in enumerate(nodes):
        node_id = ids[index]
        op = node.get("op")
        if index < len(input_abi):
            _need(
                op == "placeholder"
                and node.get("classification") == "structural"
                and node.get("args") == []
                and node.get("kwargs") == {},
                "frontend input is not an ordered placeholder",
            )
            _need(isinstance(node.get("target"), str) and bool(node["target"]), "frontend input has no captured name")
            input_names.append(node["target"])
            expected = input_abi[index]
            roots[node_id] = index
        elif index == len(nodes) - 1:
            _need(
                op == "output" and node.get("classification") == "structural" and node.get("kwargs") == {},
                "frontend graph lacks a sole final output",
            )
            break
        else:
            _need(
                op == "call_function"
                and node.get("classification") == "aten"
                and node.get("target") == "aten.alias.default"
                and node.get("kwargs") == {},
                "frontend graph contains computation or an unproved alias",
            )
            args = node.get("args")
            _need(
                isinstance(args, list) and len(args) == 1 and isinstance(args[0], Mapping),
                "alias does not have one captured tensor source",
            )
            predecessor = args[0].get("node_id")
            _need(
                predecessor in roots and args[0].get("value_id") == f"{predecessor}:v0",
                "alias source is not an earlier typed tensor",
            )
            roots[node_id] = roots[predecessor]
            aliases[node_id] = predecessor
            expected = values[predecessor]
            expected_edges.append((predecessor, f"{predecessor}:v0", node_id, "args/0", expected))
        results = node.get("results")
        _need(
            isinstance(results, list) and len(results) == 1 and isinstance(results[0], Mapping),
            "frontend tensor result is missing or ambiguous",
        )
        result = results[0]
        observed = {"shape": result.get("shape"), "dtype": _TRACE_DTYPE.get(result.get("dtype"))}
        _need(
            result.get("id") == f"{node_id}:v0" and result.get("kind") == "tensor" and observed == expected,
            "frontend alias changes tensor type or shape",
        )
        values[node_id] = expected
    output = nodes[-1]
    args = output.get("args")
    _need(
        isinstance(args, list) and len(args) == 1 and isinstance(args[0], list) and len(args[0]) == len(output_abi),
        "frontend return cardinality differs from captured ABI",
    )
    return_to_input = []
    for index, ref in enumerate(args[0]):
        _need(isinstance(ref, Mapping), "frontend output is not a captured tensor reference")
        predecessor = ref.get("node_id")
        _need(
            predecessor in roots
            and ref.get("value_id") == f"{predecessor}:v0"
            and values[predecessor] == output_abi[index],
            "frontend return is not a typed input alias",
        )
        return_to_input.append(roots[predecessor])
        expected_edges.append((predecessor, f"{predecessor}:v0", ids[-1], f"args/0/{index}", values[predecessor]))
    observed_edges = []
    for edge in edges:
        edge = _mapping(edge)
        _need(edge.get("value_kind") == "tensor", "frontend graph has an untyped edge")
        dtype = _TRACE_DTYPE.get(edge.get("dtype"))
        _need(dtype is not None, "frontend graph has an unknown edge dtype")
        observed_edges.append(
            (
                edge.get("producer_node_id"),
                edge.get("producer_value_id"),
                edge.get("consumer_node_id"),
                edge.get("argument_path"),
                {"shape": edge.get("shape"), "dtype": dtype},
            )
        )
    _need(
        len(observed_edges) == len(expected_edges)
        and sorted(json.dumps(edge, sort_keys=True) for edge in observed_edges)
        == sorted(json.dumps(edge, sort_keys=True) for edge in expected_edges),
        "frontend graph edges omit or add a value dependency",
    )
    _need(
        type(graph.get("call_count")) is int
        and graph["call_count"] == len(aliases)
        and graph.get("by_target") == ({"aten.alias.default": len(aliases)} if aliases else {}),
        "frontend graph call inventory is incomplete",
    )
    _need(len(input_names) == len(set(input_names)), "frontend input names are duplicated")
    return return_to_input, aliases, input_names


def prove_direct_return_source(
    capture: Path,
    module: Any,
    *,
    host_capabilities: Mapping,
    raw_sha256: str,
    normalized_sha256: str,
    n_source_operations: int,
) -> dict:
    """Prove every captured output is an unmodified, typed input alias."""
    source = capture / "model.mlir"
    _need(file_sha256(source) == raw_sha256, "source bytes changed")
    try:
        module.verify()
    except Exception as exc:
        raise ValueError(f"pure direct-return source: malformed MLIR: {exc}") from exc
    parsed = tuple(mq.walk(module))
    _need(
        len(parsed) == n_source_operations == 3
        and [mq.op_name(op) for op in parsed] == ["builtin.module", "func.func", "func.return"],
        "source has an operation beyond a single direct return",
    )
    top_blocks = list(module.body.blocks)
    _need(
        len(top_blocks) == 1 and list(top_blocks[0].ops) == [parsed[1]],
        "source module has another declaration or operation",
    )
    function = parsed[1]
    blocks = list(function.body.blocks)
    _need(
        len(blocks) == 1 and list(blocks[0].ops) == [parsed[2]],
        "source function has effects, control flow, or an extra return",
    )
    arguments = list(blocks[0].args)
    returned = list(parsed[2].operands)
    _need(bool(arguments) and bool(returned), "source has no complete input/output ABI")
    source_input_abi = [_tensor_abi(arg.type) for arg in arguments]
    source_output_abi = [_tensor_abi(value.type) for value in returned]
    return_to_input = []
    for value in returned:
        matches = [index for index, arg in enumerate(arguments) if value is arg]
        _need(len(matches) == 1, "source return is not a direct block argument")
        return_to_input.append(matches[0])
    allowed = _host_representation_types(host_capabilities)
    _need(
        all(entry["dtype"] in allowed for entry in source_input_abi + source_output_abi),
        "source tensor representation lacks a reviewed selected host dtype",
    )
    receipt_path = capture / "capture_receipt.json"
    _need(not receipt_path.is_symlink() and receipt_path.is_file(), "capture receipt is absent or indirect")
    receipt = json.loads(receipt_path.read_bytes())
    receipt = _mapping(receipt)
    _need(
        receipt.get("schema") == "m2m.capture-receipt.v1"
        and _mapping(receipt.get("materialized_abi")).get("complete") is True
        and receipt["materialized_abi"].get("inputs") == len(arguments)
        and receipt["materialized_abi"].get("lifted_constants") == []
        and receipt.get("lifted_constants") == {},
        "capture has missing inputs or lifted state",
    )
    _, meta = _read_bound(capture, receipt, "meta.json")
    _, trace = _read_bound(capture, receipt, "frontend-trace.json")
    _, manifest = _read_bound(capture, receipt, "weights.safetensors.manifest.json")
    _, input_order = _read_bound(capture, receipt, "input_order.json")
    _need(
        _mapping(receipt.get("artifacts")).get("model.mlir", {}).get("sha256") == raw_sha256,
        "source is not bound to capture receipt",
    )
    _need(
        isinstance(meta, Mapping)
        and meta.get("ok") is True
        and type(meta.get("opaque")) is int
        and meta["opaque"] == 0
        and meta.get("func_name") == str(function.sym_name.data)
        and _abi(meta.get("input_abi")) == source_input_abi
        and _abi(meta.get("output_abi")) == source_output_abi,
        "capture metadata has opaque work or a different function ABI",
    )
    _need(
        _mapping(meta.get("frontend_trace")).get("status") == "complete"
        and meta["frontend_trace"].get("sha256") == file_sha256(capture / "frontend-trace.json"),
        "capture metadata does not bind complete frontend trace",
    )
    _need(
        isinstance(manifest, Mapping)
        and set(manifest) == {str(i) for i in range(len(arguments))}
        and all(isinstance(row, Mapping) and row.get("kind") == "input" for row in manifest.values()),
        "capture contains weight or non-input storage",
    )
    _need(
        isinstance(trace, Mapping)
        and trace.get("schema") == "m2m.frontend_trace.v1"
        and trace.get("status") == "complete"
        and trace.get("blockers") == [],
        "frontend trace is incomplete",
    )
    mlir = _mapping(trace.get("mlir"))
    _need(
        mlir.get("sha256") == raw_sha256
        and isinstance(mlir.get("operations"), list)
        and all(isinstance(row, Mapping) for row in mlir["operations"])
        and [(row.get("ordinal"), row.get("operation")) for row in mlir["operations"]]
        == [(0, "builtin.module"), (1, "func.func"), (2, "func.return")]
        and mlir["operations"][-1].get("operand_types") == [str(value.type) for value in returned],
        "frontend source correspondence differs from parsed return",
    )
    graphs = _mapping(trace.get("graphs"))
    _need(set(graphs) == {"original", "prepared", "quantized"}, "frontend graph phases are incomplete")
    staged = meta.get("fp32_staging")
    if staged is not None:
        staged = _mapping(staged)
        conversion = _mapping(meta.get("precision_conversion"))
        audit = _mapping(conversion.get("staged_precision_audit"))
        _need(
            staged.get("status") == "observed"
            and _abi(staged.get("staged_input_abi")) == source_input_abi
            and _abi(staged.get("staged_output_abi")) == source_output_abi
            and conversion.get("schema") == "m2m.frontend-precision-conversion.v1"
            and audit.get("status") == "complete"
            and type(audit.get("non_target_floating_values")) is int
            and audit["non_target_floating_values"] == 0,
            "precision staging does not bind the returned ABI",
        )
        original_input = _abi(staged.get("original_input_abi"))
        original_output = _abi(staged.get("original_output_abi"))
    else:
        original_input, original_output = source_input_abi, source_output_abi
    if staged is not None:
        _need(
            staged.get("output_cardinality") == len(source_output_abi)
            and isinstance(staged.get("output_metrics"), list)
            and len(staged["output_metrics"]) == len(source_output_abi),
            "original-to-staged output comparison is incomplete",
        )
    mappings = {}
    aliases = {}
    graph_inputs = {}
    for name in ("original", "prepared", "quantized"):
        input_abi = original_input if name == "original" else source_input_abi
        output_abi = original_output if name == "original" else source_output_abi
        mappings[name], aliases[name], graph_inputs[name] = _graph_mapping(graphs[name], input_abi, output_abi)
        _need(mappings[name] == return_to_input, "frontend graph returns another input than lowered source")
    _need(
        graph_inputs["original"] == graph_inputs["prepared"] == graph_inputs["quantized"]
        and isinstance(input_order, Mapping)
        and input_order == {name: index for index, name in enumerate(graph_inputs["prepared"])}
        and all(manifest[str(index)].get("name") == name for index, name in enumerate(graph_inputs["prepared"])),
        "selected input order differs from traced function arguments",
    )
    correspondence = mlir.get("source_correspondence")
    _need(
        isinstance(correspondence, list) and len(correspondence) == len(aliases["prepared"]),
        "frontend alias source join is incomplete",
    )
    observed_aliases = {}
    for entry in correspondence:
        entry = _mapping(entry)
        node_id = entry.get("node_id")
        _need(
            node_id in aliases["prepared"]
            and node_id not in observed_aliases
            and entry.get("status") == "alias"
            and entry.get("mlir_ordinals") == []
            and entry.get("alias_of_node_id") == aliases["prepared"][node_id]
            and entry.get("result_index") == 0,
            "frontend alias does not uniquely join the omitted MLIR identity",
        )
        observed_aliases[node_id] = entry
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "capture_receipt_sha256": file_sha256(capture / "capture_receipt.json"),
        "frontend_trace_sha256": file_sha256(capture / "frontend-trace.json"),
        "meta_sha256": file_sha256(capture / "meta.json"),
        "source_operations": len(parsed),
        "input_abi": source_input_abi,
        "output_abi": source_output_abi,
        "return_to_input": return_to_input,
        "alias_nodes": len(aliases["prepared"]),
    }


def prove_if_empty(
    capture: Path,
    module: Any,
    host_capabilities: Mapping,
    raw_sha256: str,
    normalized_sha256: str,
    inventory: Mapping,
    descriptors: list,
) -> dict | None:
    """Leave normal linalg admission unchanged; prove only truly empty rosters."""
    if descriptors:
        return None
    return prove_direct_return_source(
        capture,
        module,
        host_capabilities=host_capabilities,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
        n_source_operations=inventory.get("n_operations"),
    )


def link_direct_return(source: Mapping[str, Any], linked_build: Mapping[str, Any]) -> None:
    """Advance only a proven zero-linalg alias after the selected host image links."""
    regions = source.get("n_linalg_regions")
    _need(type(regions) is int and regions >= 0, "source has no complete linalg-region count")
    proof = source.get("direct_return_support")
    if regions:
        _need(proof is None, "computed stage masquerades as a direct return")
        return
    _need(
        isinstance(proof, dict)
        and proof.get("status") == PENDING
        and source.get("n_groups") == 0
        and source.get("eligible_groups") == [],
        "zero-linalg stage has no pending direct-return proof",
    )
    _need(
        all(_digest(linked_build.get(key)) for key in ("capture_tree_sha256", "elf_sha256", "candidate_tree_sha256")),
        "direct return has no exact linked image identity",
    )
    proof["status"] = LINKED
    proof["linked_build"] = dict(linked_build)


def linked_direct_return_complete(source: Mapping, entry: Mapping, candidate_sha256: str) -> bool:
    """Require the exact linked image before a source-only identity proof counts."""
    proof = source.get("direct_return_support")
    regions = source.get("n_linalg_regions")
    if (
        type(regions) is not int
        or regions < 0
        or type(source.get("n_groups")) is not int
        or not isinstance(source.get("eligible_groups"), list)
    ):
        return False
    if regions:
        return proof is None
    if not isinstance(proof, Mapping) or proof.get("status") != LINKED or proof.get("scope") != SCOPE:
        return False
    bound = proof.get("linked_build")
    return (
        isinstance(bound, Mapping)
        and _digest(candidate_sha256)
        and all(_digest(bound.get(key)) for key in ("capture_tree_sha256", "elf_sha256", "candidate_tree_sha256"))
        and all(
            _digest(proof.get(key))
            for key in (
                "raw_source_sha256",
                "normalized_source_sha256",
                "capture_receipt_sha256",
                "frontend_trace_sha256",
                "meta_sha256",
            )
        )
        and proof.get("normalized_source_sha256") == source.get("normalized_source_sha256")
        and isinstance(proof.get("input_abi"), list)
        and bool(proof["input_abi"])
        and isinstance(proof.get("output_abi"), list)
        and bool(proof["output_abi"])
        and isinstance(proof.get("return_to_input"), list)
        and len(proof["return_to_input"]) == len(proof["output_abi"])
        and all(type(index) is int and 0 <= index < len(proof["input_abi"]) for index in proof["return_to_input"])
        and type(proof.get("alias_nodes")) is int
        and proof["alias_nodes"] >= 0
        and proof.get("raw_source_sha256") == source.get("source_sha256") == entry.get("source_sha256")
        and proof.get("capture_receipt_sha256") == source.get("capture_receipt_sha256")
        and proof.get("source_operations") == source.get("n_source_operations") == 3
        and source["n_groups"] == 0
        and source["eligible_groups"] == []
        and bound.get("capture_tree_sha256") == entry.get("capture_tree_sha256")
        and bound.get("elf_sha256") == entry.get("elf_sha256")
        and bound.get("candidate_tree_sha256") == entry.get("candidate_tree_sha256") == candidate_sha256
        and type(entry.get("linked_device_groups")) is int
        and entry["linked_device_groups"] == 0
        and entry.get("sidecar_sha256") is None
        and entry.get("static_host_compute_audit") == []
    )
