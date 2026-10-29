"""Closed versioned data schema for conditional graph-chain receipts.

Parsing a receipt grants no authority. Actual original/snapshot binding and all
kernel work are reopened by the live receipt verifier. Boolean and integer
fields are disjoint; floats, duplicate keys and forged approval labels refuse.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

SCHEMA = "merlin.graph-state-chain.bound-receipt.v1"
MODE = "direct-and-independent-common-primitive"
KEYS = {
    "schema",
    "version",
    "mode",
    "producer_revision",
    "originals",
    "snapshots",
    "contract",
    "contract_snapshot",
    "original_contract_identity",
    "selected_tool",
    "selected_profiles",
    "sources",
    "proof_inventory",
    "dependency_inventory",
    "direct_report",
    "independent_report",
    "binding_report",
    "claims",
}


def closed(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} needs exactly its versioned fields")
    return value


def digest(value):
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("exact lowercase SHA256 string required")
    return value


def file_record(value):
    closed(value, {"path", "sha256", "bytes"}, "file identity")
    if type(value["path"]) is not str or not value["path"] or "\0" in value["path"]:
        raise ValueError("exact nonempty file path required")
    digest(value["sha256"])
    if type(value["bytes"]) is not int or value["bytes"] < 0:
        raise ValueError("exact nonnegative integer byte count required")
    return value


def inventory(value, *, nonempty=True):
    if type(value) is not list or nonempty and not value:
        raise ValueError("complete nonempty file inventory required")
    for row in value:
        file_record(row)
    if len({row["path"] for row in value}) != len(value):
        raise ValueError("duplicate file inventory paths refused")
    return value


def validate(value):
    closed(value, KEYS, "strict graph-chain receipt")
    if value["schema"] != SCHEMA or type(value["version"]) is not int or value["version"] != 1 or value["mode"] != MODE:
        raise ValueError("unsupported receipt schema, exact integer version or mode")
    if value["producer_revision"] != "4499eebb959c0f92cefd92cc6d80646ba66474ed":
        raise ValueError("producer revision differs from selected actual source")
    digest(value["original_contract_identity"])
    if type(value["originals"]) is not list:
        raise ValueError("complete ordered original stage endpoints required")
    for row in value["originals"]:
        file_record(row)
    inventory(value["snapshots"])
    if len(value["originals"]) != 4 or len(value["snapshots"]) != 4:
        raise ValueError("strict two-boundary receipt requires all four actual stage endpoints")
    for key in (
        "contract",
        "contract_snapshot",
        "selected_tool",
        "direct_report",
        "independent_report",
        "binding_report",
    ):
        file_record(value[key])
    for key in ("selected_profiles", "proof_inventory", "dependency_inventory"):
        inventory(value[key])
    if type(value["sources"]) is not list or not value["sources"]:
        raise ValueError("complete original/snapshot source closure required")
    for pair in value["sources"]:
        closed(pair, {"original", "snapshot"}, "source binding")
        file_record(pair["original"])
        file_record(pair["snapshot"])
    if len({row["original"]["path"] for row in value["sources"]}) != len(value["sources"]):
        raise ValueError("source inventory duplicates an original")
    claims = closed(
        value["claims"],
        {
            "status",
            "public_acceptance_issued",
            "source_contract_approved",
            "native_fidelity_proved",
            "numerical_gates_discharged",
            "full_model_claim",
        },
        "claim labels",
    )
    if claims["status"] != "conditional_model_checked":
        raise ValueError("strict mode records only its measured conditional model relation")
    if any(claims[key] is not False for key in claims if key != "status"):
        raise ValueError("approval/native/numerical/full-model flags must be exact false booleans")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field refused")
        result[key] = value
    return result


def _noninteger(value):
    raise ValueError("floating-point and nonfinite receipt fields refused")


def load_buffer(path):
    path = Path(path).resolve()
    before = path.stat()
    limit = 64 * 1024 * 1024
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("receipt must be a bounded regular file")
    with path.open("rb") as source:
        data = source.read(limit + 1)
    after = path.stat()
    if len(data) > limit:
        raise ValueError("receipt exceeds explicit byte budget")
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size) or len(data) != after.st_size:
        raise ValueError("receipt bytes changed while reading the parsed/hashed input buffer")
    value = validate(json.loads(data, object_pairs_hook=_pairs, parse_float=_noninteger, parse_constant=_noninteger))
    return value, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load(path):
    return load_buffer(path)[0]
