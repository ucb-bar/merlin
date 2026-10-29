"""Public source-exact spatial pair from two existing ordinary source builders.

This describes an observable program, never a fused readout equivalence or a
hardware schedule. The independent reference retains each original rounding.
"""

from copy import deepcopy


def _composition_modes(*components):
    """Join source capabilities; movement-only describes the complete region.

    A load/reduction component's movement-only expectation cannot prohibit the
    contraction instructions required by another component. Other conflicting
    mode declarations refuse instead of being overwritten by source order.
    """
    modes = {}
    for component in components:
        for key, value in component.items():
            if key == "movement":
                continue
            if key in modes and modes[key] != value:
                raise ValueError("composed source has conflicting mode declarations: " + key)
            modes[key] = value
    if components and all(component.get("movement") is True for component in components):
        modes["movement"] = True
    return modes


def build(entry, binding):
    from merlin.llvmlower.connected_spatial import spatial_pair_interface
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    components = entry.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 2
        or [c.get("op") for c in components] != ["conv2d", "maxpool2d"]
    ):
        raise ValueError("spatial source pair requires exactly original convolution and pooling entries")
    producer, pool = (CS.build({**e, "source_reference": entry["source_reference"]}, binding)[0] for e in components)
    interface = spatial_pair_interface(
        {"schema": "connected_spatial_entry_v1", "components": components},
        lambda _: binding,
        source_reference=entry["source_reference"],
    )
    cb = parse_interface_mlir(interface)
    convolution = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
    maximum = cb["commands"][-1]
    internal, output = convolution["operands"]["dst"], maximum["operands"]["dst"]
    p, q = deepcopy(producer["operation"]), deepcopy(pool["operation"])
    for key in ("ifm", "weight", "bias", "out"):
        if key in p["attributes"]:
            p["attributes"][key] = "producer_" + p["attributes"][key]
    q["attributes"].update(src=internal, out=output)
    modes = _composition_modes(producer["expected"]["modes"], pool["expected"]["modes"])
    classes = list(dict.fromkeys(producer["expected"]["instruction_classes"] + pool["expected"]["instruction_classes"]))
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "label": entry.get("label", "public"),
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{**deepcopy(row), "name": "producer_" + row["name"]} for row in producer["inputs"]],
        "operation": {
            "op": "spatial_producer_pool",
            "attributes": {
                "producer": p,
                "pool": q,
                "intermediate_shape": cb["tensors"][internal]["shape"],
                "intermediate_dtype": cb["tensors"][internal]["dtype"],
                "out": output,
            },
        },
        "numeric_policy": deepcopy(producer["numeric_policy"]),
        "expected": {"instruction_classes": classes, "modes": modes},
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    return cap, interface
