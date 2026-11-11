"""Independent public-word/private-byte ABI and scratch custody checks.

This compiler-only reader imports no producer implementation. Declarations
grant no source equivalence, alias permission or target qualification.
"""

import copy

WIDTHS = {"i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}
SCHEMA = "t2m.typed-storage-transport.v1"


def storage_spec(trace):
    spec = copy.deepcopy(trace.get("spec"))
    profile = trace.get("typed_storage_transport")
    if profile is None:
        return spec
    if (
        not isinstance(profile, dict)
        or set(profile) != {"schema", "byte_order", "roots", "grants_semantics"}
        or profile["schema"] != SCHEMA
        or profile["byte_order"] not in ("little", "big")
        or profile["grants_semantics"] is not False
        or not isinstance(profile["roots"], list)
        or not profile["roots"]
        or trace.get("profile") != "shared-source-word-buffer"
    ):
        raise ValueError("invalid typed storage transport profile")
    roots = profile["roots"]
    selected = {root.get("name"): root for root in roots if isinstance(root, dict)}
    if len(selected) != len(roots):
        raise ValueError("typed storage roots must be unique named records")
    ordered = []
    for buffer in spec["buffers"]:
        if buffer["name"] not in selected:
            continue
        size = WIDTHS.get(buffer["dtype"])
        expected = {
            "name": buffer["name"],
            "logical_dtype": buffer["dtype"],
            "logical_shape": list(buffer["shape"]),
            "storage_dtype": "i8",
            "storage_shape": [*buffer["shape"], size],
            "element_bytes": size,
            "effect": buffer["effect"],
        }
        if size is None or selected[buffer["name"]] != expected:
            raise ValueError("typed storage root differs from its public ABI")
        ordered.append(expected)
        buffer.update(dtype="i8", shape=expected["storage_shape"])
    if ordered != roots:
        raise ValueError("typed storage roots differ from public buffer order")
    return spec


def result_guard_costs(trace, base):
    storage_spec(trace)
    sizes = {r["name"]: r["element_bytes"] for r in trace.get("typed_storage_transport", {}).get("roots", [])}
    return [base * sizes.get(b["name"], 1) for b in trace["spec"]["buffers"] if b["effect"] != "read"]


def checked_guard_costs(entry, trace, base):
    import json

    from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, StringAttr, i64

    costs = result_guard_costs(trace, base)
    contract = entry.attributes.get("prov.memory_guard_bytes_per_result_element")
    vector = entry.attributes.get("prov.memory_guard_bytes_by_result_element")
    transport = entry.attributes.get("prov.typed_storage_transport")
    profile = trace.get("typed_storage_transport")
    if (
        not isinstance(contract, IntegerAttr)
        or contract.value.data != max(costs, default=base)
        or profile is None
        and (vector is not None or transport is not None)
        or profile is not None
        and (
            not isinstance(transport, StringAttr)
            or json.loads(transport.data) != profile
            or not isinstance(vector, DenseArrayBase)
            or vector.elt_type != i64
            or list(vector.get_values()) != costs
        )
    ):
        raise ValueError("guarded kernel lacks its typed scratch storage contract")
    return costs
