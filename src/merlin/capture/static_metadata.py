"""Specialize pure metadata queries under the static capture input ABI."""


def specialize_static_metadata(model, inputs):
    """Fold a metadata-only FX graph from input descriptors, never tensor values.

    Capture fixes shapes, strides and offsets. A call that reads only graph
    inputs is a descriptor query when it evaluates to the same int or bool on
    meta tensors with the same descriptors as on the inputs themselves: a meta
    tensor has no data, so a data-dependent call cannot evaluate. After folding, no call may read an
    input or draw seeded randomness. Keep every original input in the graph so
    normal boundary observations and alias checks still apply. Mixed
    data/metadata programs retain the ordinary frontend path.
    """
    import copy

    import torch
    from torch.fx.node import map_arg

    try:
        graph = torch.fx.symbolic_trace(copy.deepcopy(model))
    except (RuntimeError, TypeError, ValueError, torch.fx.proxy.TraceError):
        return model, None
    nodes = list(graph.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    if (
        any(node.op not in {"placeholder", "call_function", "output"} for node in nodes)
        or len(placeholders) != len(inputs)
        or not all(isinstance(value, torch.Tensor) for value in inputs)
    ):
        return model, None

    def descriptor(value):
        shape, stride, offset = value.shape, value.stride(), value.storage_offset()
        span = offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if value.numel() else 0)
        return torch.empty(span, dtype=value.dtype, device="meta").as_strided(shape, stride, offset)

    try:
        meta = {node: descriptor(value) for node, value in zip(placeholders, inputs, strict=True)}
    except RuntimeError:  # e.g. a quantized tensor has no strided meta form
        return model, None
    real = dict(zip(placeholders, inputs, strict=True))
    folded = []
    for node in [node for node in nodes if node.op == "call_function"]:
        if not any(source in meta for source in node.all_input_nodes):
            continue
        schema = getattr(node.target, "_schema", None)
        # A call that returns nothing or writes an argument is not a query, and never runs on the inputs.
        if (
            not set(node.all_input_nodes) <= meta.keys()
            or schema is None
            or not schema.returns
            or any(argument.alias_info is not None and argument.alias_info.is_write for argument in schema.arguments)
        ):
            return model, None
        try:
            value = node.target(*map_arg(node.args, meta.__getitem__), **map_arg(node.kwargs, meta.__getitem__))
            if type(value) not in {int, bool}:
                return model, None
            observed = node.target(*map_arg(node.args, real.__getitem__), **map_arg(node.kwargs, real.__getitem__))
        except (RuntimeError, NotImplementedError, TypeError, ValueError):
            return model, None
        # A descriptor query gives the same answer with and without data (e.g. aten._version does not).
        if type(observed) is not type(value) or observed != value:
            return model, None
        folded.append({"operator": str(node.target), "value": value})
        for user in list(node.users):
            user.args = map_arg(user.args, lambda arg: value if arg is node else arg)
            user.kwargs = map_arg(user.kwargs, lambda arg: value if arg is node else arg)
        graph.graph.erase_node(node)
    seeded = torch.Tag.nondeterministic_seeded
    if (
        not folded
        or any(node.users for node in placeholders)
        or any(seeded in getattr(node.target, "tags", ()) for node in graph.graph.nodes if node.op == "call_function")
    ):
        return model, None
    graph.graph.lint()
    graph.recompile()
    return graph, {
        "authority": "static_capture_input_descriptors",
        "inputs": [
            {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset()}
            for value in inputs
        ],
        "queries": folded,
    }
