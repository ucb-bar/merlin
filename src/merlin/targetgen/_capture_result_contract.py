"""Same-conversion exported tensor result roles and logical layout facts."""


def exported_contract(result, inputs):
    """Read roles/layout/alias facts from the SAME conversion, never an eager result."""
    ep = getattr(result, "exported_program", None)
    if ep is None:
        return None
    nodes = {node.name: node for node in ep.graph.nodes}
    placeholders = [node for node in ep.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(inputs):
        return None

    def metadata(value):
        def integer(v):
            try:
                return int(v)
            except (TypeError, ValueError, RuntimeError):
                return -1

        def stride(v):
            fixed = integer(v)
            if fixed >= 0:
                return fixed
            # Symbolic layout extents are expressions over this tensor's own shape.
            expression = getattr(getattr(v, "node", None), "expr", None)
            for axis, extent in enumerate(value.shape):
                if expression is not None and expression == getattr(getattr(extent, "node", None), "expr", extent):
                    return {"shape_axes": [axis]}
            return -1

        return dict(
            dtype=str(value.dtype).removeprefix("torch."),
            shape=[integer(v) for v in value.shape],
            stride=[stride(v) for v in value.stride()],
            storage_offset=integer(value.storage_offset()),
            requires_grad=bool(value.requires_grad),
        )

    def storage(value):
        return value.untyped_storage()._cdata

    source = [node.meta.get("val") for node in placeholders]
    if any(not hasattr(value, "untyped_storage") for value in source):
        return None
    mutation_sources = {}
    for spec in ep.graph_signature.output_specs:
        if spec.kind.name == "USER_INPUT_MUTATION":
            node = nodes.get(getattr(spec.arg, "name", None))
            if node is not None:
                mutation_sources[spec.target] = node.meta.get("val")
    records = []
    for spec in ep.graph_signature.output_specs:
        node = nodes.get(getattr(spec.arg, "name", None))
        value = node.meta.get("val") if node else None
        if not hasattr(value, "untyped_storage"):
            return None
        role = spec.kind.name.lower()
        if role not in ("user_output", "user_input_mutation"):
            return None
        entry = {
            **metadata(value),
            "role": role,
            "alias_inputs": [
                i
                for i, original in enumerate(source)
                if storage(original) == storage(value)
                or (
                    placeholders[i].name in mutation_sources
                    and storage(mutation_sources[placeholders[i].name]) == storage(value)
                )
            ],
        }
        if role == "user_input_mutation":
            target = next((i for i, node in enumerate(placeholders) if node.name == spec.target), None)
            if target is None:
                return None
            entry["input_index"] = target
        records.append(entry)
    return dict(
        schema_version=1,
        authority="same_conversion_exported_program",
        inputs=[metadata(value) for value in source],
        results=records,
    )


def float_reference(mdl, inputs, torch, input_abi, output_abi, to_native) -> dict:
    """The untransformed model's outputs on ``inputs``, in the golden's JSON shape.

    Every RNG this worker seeds is saved and restored around the forward, so a loader whose forward
    draws random numbers produces the same golden afterwards as it would have without this run.
    """
    import itertools
    import random

    import numpy as np

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    owned = [*input_abi(inputs)[0], *mdl.buffers()]
    snapshots = [
        (value, value.detach().clone(), value.size(), value.stride(), value.storage_offset()) for value in owned
    ]
    try:
        with torch.no_grad():
            y = mdl(*inputs)
        leaves, abi = output_abi(y)
        leaves = [leaf.detach().clone() for leaf in leaves]
        values = [to_native(x) for x in leaves]
    finally:
        with torch.no_grad():
            for value, snapshot, shape, stride, offset in snapshots:
                value.as_strided_(shape, stride, offset)
                if torch._debug_has_internal_overlap(value) == 0:
                    value.copy_(snapshot)
                else:
                    # The snapshot came from this original address map, so
                    # repeated addresses carry identical saved scalar bytes.
                    # Bulk copy refuses valid expanded/padded inputs.
                    for index in itertools.product(*(range(extent) for extent in shape)):
                        value[index].copy_(snapshot[index])
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    return {"outputs": values[0] if len(values) == 1 else values, "output_abi": abi, "leaves": leaves}


def raw_tensor_boundary(outputs, inputs):
    """Native eager storage bits, independent of compiler execution and Q/DQ."""
    import torch

    def raw(tensor):
        return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()

    return {
        "authority": "untransformed_native_cpu_eager",
        "output_bytes": [raw(tensor) for tensor in outputs],
        "post_input_bytes": [raw(tensor) for tensor in inputs],
    }
