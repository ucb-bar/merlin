"""Source-bound standalone contractions from an explicitly approved shape projection.

These operator inputs retain source occurrence identities, but use independent
stimuli. They establish neither captured-boundary equivalence nor model execution.
No target schedule, implementation, simulation or timing enters this projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def contractions(record: dict) -> list[dict]:
    """Group exact prepared int8 matrix products by their observed M/K/N shape."""
    if len(record.get("source_sha256", "")) != 64:
        raise ValueError("an original trace identity is required")
    graphs = [g for g in record["graphs"] if g["stage"] == "prepared"]
    if len(graphs) != 1:
        raise ValueError("select one complete prepared graph projection")
    graph = graphs[0]
    incoming = {}
    for edge in graph["edges"]:
        key = (edge["consumer_node_id"], edge.get("argument_path"))
        if key in incoming:
            raise ValueError("source argument identity is ambiguous")
        incoming[key] = edge
    groups = {}
    for node in graph["nodes"]:
        if node["target"] != "aten._int_mm.default":
            continue
        lhs = incoming[(node["id"], "args/0")]
        rhs = incoming[(node["id"], "args/1")]
        results = node["results"]
        a, b = lhs["shape"], rhs["shape"]
        if (lhs["dtype"] != "int8" or rhs["dtype"] != "int8"
                or len(a) != 2 or len(b) != 2 or a[1] != b[0]
                or any(type(x) is not int or x <= 0 for x in [*a, *b])
                or len(results) != 1 or results[0]["dtype"] != "int32"
                or results[0]["shape"] != [a[0], b[1]]):
            raise ValueError("selected source contraction has no exact integer matrix contract")
        shape = (a[0], a[1], b[1])
        row = groups.setdefault(shape, {
            "shape_mkn": list(shape), "model": record["model"], "program": record["program"],
            "trace_sha256": record["source_sha256"], "graph_sha256": graph.get("graph_sha256"),
            "operation": node["target"], "source_node_ids": [], "occurrences": [],
        })
        row["source_node_ids"].append(node["id"])
        row["occurrences"].append({"node": node["id"], "inputs": [lhs, rhs], "result": results[0]})
    return list(groups.values())


def write_sources(projection: Path, *, sha256: str, shapes: list[tuple[int, int, int]],
                  template: Path, destination: Path, name_prefix: str) -> list[Path]:
    """Render only requested, actually observed shapes through the native IR API."""
    import yaml
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

    raw = projection.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256 or projection.is_symlink():
        raise ValueError("approved source projection changed")
    groups = {tuple(row["shape_mkn"]): row for row in contractions(json.loads(raw))}
    if not shapes or len(shapes) != len(set(shapes)) or any(shape not in groups for shape in shapes):
        raise ValueError("requested shapes must be distinct observed source contractions")
    if destination.exists() or not name_prefix.replace("_", "").isalnum():
        raise ValueError("select a new source directory and an ordinary name prefix")
    cap0 = yaml.safe_load((template / "capsule.yaml").read_text())
    cb0 = parse_interface_mlir((template / cap0["interface_mlir"]).read_text())
    attrs = cap0["operation"]["attributes"]
    if (cap0["operation"]["op"] != "matmul" or attrs.get("epilogue") != []
            or attrs.get("output_dtype") != "i32" or cap0["numeric_policy"] != {"compare": "exact_int", "dtype": "i32"}):
        raise ValueError("template must be the independent raw int8-to-int32 matrix contract")
    destination.mkdir(parents=True)
    output = []
    for shape in shapes:
        m, k, n = shape
        cap, cb = copy.deepcopy(cap0), copy.deepcopy(cb0)
        name = f"{name_prefix}_m{m}k{k}n{n}_raw"
        binding = {**groups[shape], "projection_sha256": sha256,
                   "numerical_scope": "exact int32 product of independent deterministic int8 inputs",
                   "captured_boundary_values": False, "connected_cut": False, "full_model": False,
                   "execution_status": "not_executed", "development_engine": "spike"}
        cap.update(name=name, source_role="handauthored_compiler_test",
                   source_reference=(f"Approved {binding['model']}/{binding['program']} graph shape; "
                       f"trace SHA256 {binding['trace_sha256']}; all occurrences in source-shape-binding.json. "
                       "Independent integer contraction contract, not captured model-boundary equivalence."),
                   required_oracle_tiers=["L0", "L1", "L2"])
        for key in ("software_screen", "integer_partial_sum_bound", "extends", "max_oracle_tier", "performance"):
            cap.pop(key, None)
        for item in cap["inputs"]:
            if item["name"] not in {attrs["lhs"], attrs["weight"]} or item["dtype"] != "i8":
                raise ValueError("template input ABI is not the selected independent matrix contract")
            item["shape"] = [m, k] if item["name"] == attrs["lhs"] else [k, n]
            cb["tensors"][item["name"]]["shape"] = item["shape"]
        # The native interface often infers the result tensor from the command.
        if attrs["out"] in cb["tensors"]:
            cb["tensors"][attrs["out"]]["shape"] = [m, n]
        path = destination / name
        path.mkdir()
        (path / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
        (path / cap["interface_mlir"]).write_text(emit_interface_mlir(cb))
        (path / "source-shape-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        output.append(path)
    return output
