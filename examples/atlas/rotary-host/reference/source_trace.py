"""Reconstruct source FX from trace records without compiler evaluation."""
from types import SimpleNamespace
import operator


def rebuild_program(nodes, load_constant, *, selected_ids=None, output_ids=None,
                    transform_args=None, transform_metadata=None):
    """Build a complete trace or a selected dependency slice with source hooks."""
    import torch

    graph = torch.fx.Graph()
    built, specs, state, constants, users = {}, [], {}, {}, []

    def resolve(value):
        if isinstance(value, dict):
            if "node_id" in value:
                return built[value["node_id"]]
            if value.get("kind") in {"dtype", "layout"}:
                return getattr(torch, value["value"].removeprefix("torch."))
            if value.get("kind") == "device":
                return torch.device(value["value"])
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    for record in nodes:
        if selected_ids is not None and record["id"] not in selected_ids:
            continue
        if record["op"] == "placeholder":
            node = graph.placeholder(record["target"])
            kind, target = record["input_kind"], record.get("input_target")
            specs.append(SimpleNamespace(arg=SimpleNamespace(name=record["target"]),
                         kind=SimpleNamespace(name=kind), target=target))
            if kind == "USER_INPUT":
                users.append(record)
            elif kind in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
                tensor = load_constant(record)
                expected = record["results"][0]
                if (list(tensor.shape) != expected["shape"]
                        or str(tensor.dtype).removeprefix("torch.") != expected["dtype"]):
                    raise ValueError("source constant metadata differs: " + target)
                (constants if kind == "CONSTANT_TENSOR" else state)[target] = tensor
            else:
                raise ValueError("source trace contains unsupported input role")
        elif record["op"] == "call_function":
            if record["target"] == "<built-in function getitem>":
                target = operator.getitem
            else:
                target = torch.ops
                for part in record["target"].split("."):
                    target = getattr(target, part)
            args = resolve(record["args"])
            if transform_args is not None:
                args = transform_args(record, args)
            node = graph.call_function(target, tuple(args), resolve(record["kwargs"]))
        elif record["op"] == "output":
            graph.output(resolve(record["args"][0]))
            continue
        else:
            raise ValueError("source trace contains unsupported node role")
        results = record["results"]
        if transform_metadata is not None:
            results = transform_metadata(record, results)
        values = [torch.empty(result["shape"], dtype=getattr(torch, result["dtype"]), device="meta")
                  if result.get("kind") == "tensor" else None for result in results]
        node.meta["val"] = values[0] if len(values) == 1 else tuple(values)
        built[record["id"]] = node
    if output_ids is not None:
        graph.output(tuple(built[key] for key in output_ids))
    program = SimpleNamespace(graph=graph, graph_signature=SimpleNamespace(input_specs=specs),
                              state_dict=state, constants=constants)
    return program, users
