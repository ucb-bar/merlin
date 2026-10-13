"""Original selected public compilation resource policy, separate from correctness."""

from __future__ import annotations

import copy

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
    for row in rows:
        growth, evaluated = row.get("code_growth"), row.get("resource_budget")
        if (row.get("lowering_status") != "emitted" or type(growth) is not dict
            or growth.get("lexical_complete") is not True or type(evaluated) is not dict
            or evaluated.get("status") != "accepted" or evaluated.get("policy") != policy["device_ir_budget"]
            or evaluated.get("exceeded") != []):
            return False
    return True
