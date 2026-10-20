"""Original selected public compilation resource policy, separate from correctness."""

from __future__ import annotations

import copy
import hashlib
import json
import math

from merlin.common.digest import is_sha256
from merlin.llvmlower.device_ir_scalability import DeviceIRBudget

SCHEMA = "merlin.public_codegen_resource_policy.v1"
FIELD = "public_codegen_resource_policy"


def validate_policy(value):
    """Normalize only an explicit original selection; no environment defaults."""
    if value is None:
        return None
    if type(value) is not dict or set(value) - {"schema", "required", "timeout_s", "capsules", "device_ir_budget"}:
        raise ValueError("public codegen resource policy requires closed original fields")
    if value.get("schema", SCHEMA) != SCHEMA or type(value.get("required")) is not bool:
        raise ValueError("public codegen resource policy has an invalid schema or required flag")
    deadline = value.get("timeout_s")
    if type(deadline) is not int or not 1 <= deadline <= 600:
        raise ValueError("public codegen resource deadline must be 1..600 seconds")
    members = value.get("capsules")
    if (type(members) is not list or not members
        or any(type(name) is not str or not name or name in {"all", ".", ".."}
               or any(not (char.isalnum() or char in "_.-") for char in name) for name in members)
        or len(set(members)) != len(members)):
        raise ValueError("public codegen resource members must be explicit unique public IDs")
    raw = value.get("device_ir_budget")
    if type(raw) is not dict:
        raise ValueError("public codegen resource policy requires original budget data")
    try:
        budget = DeviceIRBudget(**raw)
    except TypeError as error:
        raise ValueError("public codegen resource budget has unknown fields") from error
    if value["required"] and budget.mode != "enforce":
        raise ValueError("required public codegen resource policy must enforce its limits")
    return {"schema": SCHEMA, "required": value["required"], "timeout_s": deadline,
            "capsules": list(members), "device_ir_budget": budget.record()}


def policy_from_profile(profile):
    selected = profile.get(FIELD)
    policy = validate_policy(selected)
    if selected is not None and selected != policy:
        raise ValueError("selected public codegen policy is not canonical original data")
    return copy.deepcopy(policy)


def require_public_members(policy, names):
    if policy is not None and not set(policy["capsules"]) <= set(names):
        raise ValueError("selected resource members are absent from the full admitted public roster")


def _shape_ladder_complete(records):
    """Require the fixed producer domain, including actual larger tensor ABIs.

    Interface sources select hardware-boundary tails. The ordinary linalg
    producer selects only geometric extents; it supplies no hardware-tail claim.
    Unsupported exact-only programs establish neither of these scale domains.
    """
    derivations = [row["derivation"] for row in records]
    index_bits = derivations[0].get("host_index_bits")
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        return False
    if derivations[0].get("basis") == "retained_public_interface":
        roster = [(1, 0), (8, 0), (8, -1), (8, 1), (64, 0)]
        if (any(type(d.get("extent_tail")) is not int for d in derivations)
            or [(row["scale"], d.get("extent_tail")) for row, d in zip(records, derivations)] != roster):
            return False
        family = derivations[0].get("scale_family")
        edge = derivations[0].get("hardware_tail_edge")
        if (type(family) is not str or not family or family == "retained_only"
            or type(edge) is not int or edge <= 0):
            return False
        for row, d in zip(records, derivations):
            if (d.get("scale_family") != family or d.get("hardware_tail_edge") != edge
                or d.get("basis") != ("retained_public_interface" if row["scale"] == 1
                                      else "public_interface_shape_semantics")
                or d.get("variant") != f"scale{row['scale']}_tail{d['extent_tail']:+d}"):
                return False
    elif derivations[0].get("basis") == "retained_public_source":
        if [row["scale"] for row in records] != [1, 8, 64]:
            return False
        if any(d.get("basis") != ("retained_public_source" if row["scale"] == 1
                                 else "geometric_tensor_types")
               or d.get("variant", "original") != "original"
               for row, d in zip(records, derivations)):
            return False
    else:
        return False
    if (len({row["source_sha256"] for row in records}) != len(records)
        or any(type(d.get("extent_ratio")) is not int or d["extent_ratio"] != row["scale"]
               or d.get("host_index_bits") != index_bits or d.get("tensor_values_allocated") is not False
               or d.get("goldens_allocated") is not False for row, d in zip(records, derivations))):
        return False
    shapes = []
    for row in records:
        abi = row.get("original_abi")
        if type(abi) is not dict or set(abi) != {"inputs", "outputs"}:
            return False
        slots = []
        for role in ("inputs", "outputs"):
            tensors = abi[role]
            if type(tensors) is not list or not tensors:
                return False
            for tensor in tensors:
                if (type(tensor) is not dict or type(tensor.get("dtype")) is not str
                    or type(tensor.get("shape")) is not list or not tensor["shape"]
                    or any(type(size) is not int or not 0 < size < 1 << (index_bits - 1)
                           for size in tensor["shape"])):
                    return False
                slots.append((role, tensor["dtype"], tuple(tensor["shape"])))
        shapes.append(slots)
    baseline = shapes[0]
    for row, observed in zip(records[1:], shapes[1:]):
        if (len(observed) != len(baseline)
            or any(role != old_role or dtype != old_dtype or len(shape) != len(old_shape)
                   for (role, dtype, shape), (old_role, old_dtype, old_shape) in zip(observed, baseline))):
            return False
        for role in ("inputs", "outputs"):
            old = sum(math.prod(shape) for kind, _, shape in baseline if kind == role)
            new = sum(math.prod(shape) for kind, _, shape in observed if kind == role)
            if new <= old:
                return False
        tail = row["derivation"].get("extent_tail", 0)
        if not tail:
            if not any(size == original * row["scale"]
                       for (role, _, shape), (_, _, old_shape) in zip(observed, baseline) if role == "inputs"
                       for size, original in zip(shape, old_shape)):
                return False
        elif not any(size == ((original * 8 + edge - 1) // edge) * edge + tail
                     for (role, _, shape), (_, _, old_shape) in zip(observed, baseline) if role == "inputs"
                     for size, original in zip(shape, old_shape)):
            return False
    return True


def resources_complete(result, policy):
    """Require all actual selected ladder rows; issue no execution credit."""
    if type(result) is not dict or result.get("mode") != "scale_emission_only":
        return False
    manifest, rows = result.get("source_manifest"), result.get("cases")
    if (type(manifest) is not dict or manifest.get("unavailable") != []
        or type(manifest.get("cases")) is not list or not manifest["cases"]
        or type(rows) is not list or len(rows) != len(manifest["cases"])):
        return False
    if any(type(row) is not dict or type(row.get("derivation")) is not dict for row in manifest["cases"]):
        return False
    expected = [(row.get("member"), row.get("scale"), row["derivation"].get("variant", "original"),
                 row.get("source_sha256")) for row in manifest["cases"]]
    actual = [(row.get("member"), row.get("scale"), row.get("variant"), row.get("source_sha256"))
              for row in rows if type(row) is dict]
    if (len(expected) != len(rows) or actual != expected
        or any(type(member) is not str or type(scale) is not int or scale <= 0
               or type(variant) is not str or not variant or not is_sha256(digest)
               for member, scale, variant, digest in expected)
        or len(set(expected)) != len(expected)
        or {member for member, _, _, _ in expected} != set(policy["capsules"])):
        return False
    for member in policy["capsules"]:
        if not _shape_ladder_complete([row for row in manifest["cases"] if row["member"] == member]):
            return False
    for original, row in zip(manifest["cases"], rows):
        abi_sha256 = hashlib.sha256(json.dumps(original["original_abi"], sort_keys=True).encode()).hexdigest()
        if row.get("original_abi_sha256") != abi_sha256:
            return False
    for row in rows:
        growth, evaluated = row.get("code_growth"), row.get("resource_budget")
        if (row.get("lowering_status") != "emitted" or type(growth) is not dict
            or growth.get("lexical_complete") is not True or type(evaluated) is not dict
            or evaluated.get("status") != "accepted" or evaluated.get("policy") != policy["device_ir_budget"]
            or evaluated.get("exceeded") != []):
            return False
    return True
