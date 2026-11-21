"""Join checked derived source arguments to immutable storage or runtime state.

The source trace supplies user order and argument roles. Existing tensor archive
checks validate every stored tensor, including buffers that remain runtime data.
No captured buffer snapshot is silently substituted for a live state argument.
"""

import json

from merlin.compiler.frontend.signature import _argument_bindings


def argument_bindings(function, signature, members):
    trace = json.loads(members["source/parent/frontend-trace.json"])
    graph = trace.get("graphs", {}).get("prepared", {})
    nodes = [node for node in graph.get("nodes", [])
             if node.get("op") == "placeholder" and node.get("graph_id") == graph.get("graph_id")]
    manifest = json.loads(members["weights.safetensors.manifest.json"])
    types = tuple(function.function_type.inputs)
    if (len(nodes) != len(types) or not isinstance(manifest, dict)
            or set(manifest) != {str(index) for index in range(len(types))}):
        raise ValueError("derived bindings require the complete original source argument roster")
    roles = {"input": "USER_INPUT", "param": "PARAMETER", "buffer": "BUFFER", "constant": "CONSTANT_TENSOR"}
    source_inputs, order, runtime_types, normalized = [], {}, [], {}
    for index, (node, typ) in enumerate(zip(nodes, types, strict=True)):
        row = manifest[str(index)]
        if not isinstance(row, dict) or node.get("input_kind") != roles.get(row.get("kind")):
            raise ValueError("derived argument role differs from the actual source placeholder")
        kind = row["kind"]
        if kind == "input":
            name = row.get("name")
            if not isinstance(name, str) or name in order or node.get("target") != name:
                raise ValueError("derived runtime input differs from the original source order/name")
            order[name] = len(source_inputs)
            source_inputs.append(signature["inputs"][index])
            runtime_types.append(typ)
            normalized[str(index)] = row
        else:
            if (kind != "constant" and node.get("input_target") != row.get("weight")):
                raise ValueError("derived initializer differs from the original source parameter/state target")
            # Constant metadata has already been source/byte checked by
            # checked_compiler_constants; retain it in the original member.
            normalized[str(index)] = ({key: row[key] for key in ("kind", "weight", "dtype", "shape")}
                                      if kind == "constant" else row)
    checked = _argument_bindings(function.function_type, source_inputs,
        {**members, "weights.safetensors.manifest.json": json.dumps(normalized).encode(),
         "input_order.json": json.dumps(order).encode()}, imported_runtime_types=tuple(runtime_types))
    runtime = 0
    for index, row in enumerate(checked):
        if row["kind"] == "buffer":
            node = nodes[index]
            row = {"kind": "input", "source_kind": "buffer", "state_binding": "explicit-runtime-argument",
                "name": node["input_target"], "mlir_type": str(types[index]), "abi_kind": "tensor",
                "shape": row["shape"], "dtype": row["dtype"]}
            checked[index] = row
        if row["kind"] == "input":
            row["input_index"] = runtime
            runtime += 1
    return tuple(checked)
