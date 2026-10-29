"""Pure stdlib scheduling receipt/journal schema; no file or SDK validation.

Shape acceptance is UNQUALIFIED metadata handling, never proof/runtime approval.
This leaf can be tested in isolation during the shared selected-runtime hold.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

if __package__:
    from .lean_graph_state_schedule_errors import ScheduleDomainRefusal, require
else:
    from lean_graph_state_schedule_errors import ScheduleDomainRefusal, require

SCHEMA = "merlin.graph-state-schedule.live-receipt.v1"
MODES = ("graph", "initialized-symbolic-provider-v1", "initialized-scalar-provider-v1")
CLAIMS = MappingProxyType({
    "relation": "complete return/error/terminal-world correspondence under recorded operation contracts",
    "all_region_operations_anchored": True,
    "partial_opaque_relative_order_preserved": True,
    "totality_from_empty_effects": False,
    "python_projection_proved": False,
    "qualified": False,
    "review_approved": False,
    "public_acceptance_issued": False,
    "native_or_poison_fidelity_proved": False,
    "primitive_or_model_approval": False,
    "source_success_for_all_models": False,
    "all_graph_passes_verified": False,
    "full_model_claim": False,
    "broker_integration": False,
    "old_failure_promoted": False,
})
KEYS = {
    "schema", "version", "root", "settings", "mode", "model", "originals", "orders", "journal",
    "snapshots", "order_snapshots", "journal_snapshot", "report", "capture", "profile", "query", "object",
    "sources", "files", "dependencies", "claims", "roots", "qualification_state",
}


def closed(value, keys, reason="closed scheduling record required"):
    require(type(value) is dict and set(value) == set(keys), reason)


def reference_row(value):
    closed(value, {"path", "sha256", "bytes"}, "closed three-field source/evidence reference required")
    require(type(value["path"]) is str and Path(value["path"]).is_absolute(), "exact live absolute reference path required")
    require(type(value["sha256"]) is str and len(value["sha256"]) == 64 and not set(value["sha256"]) - set("0123456789abcdef"),
            "lowercase SHA256 metadata required")
    require(type(value["bytes"]) is int and value["bytes"] >= 0, "nonnegative exact integer byte length required; bool refused")


def valid_claims(value):
    return type(value) is dict and set(value) == set(CLAIMS) and all(
        type(value[key]) is type(expected) and value[key] == expected for key, expected in CLAIMS.items()
    )


def settings(value):
    closed(value, {"factor", "tile", "index_bits", "count", "minimum_depth"})
    require(all(type(v) is int for v in value.values()), "configuration integers must not be bool/float")
    require(2 <= value["factor"] <= 4 and 2 <= value["tile"] <= 16 and value["index_bits"] in (32, 64)
            and 1 <= value["count"] <= 8 and value["minimum_depth"] >= 1, "unsupported schedule configuration")


def strict_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def noninteger(_):
        raise ScheduleDomainRefusal("floating/nonfinite JSON numeric metadata refused")

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_float=noninteger, parse_constant=noninteger)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ScheduleDomainRefusal("malformed receipt/journal JSON") from error


def journal_shape(value):
    require(type(value) is list and 1 <= len(value) <= 8, "bounded nonempty current-order journal required")
    for row in value:
        closed(row, {"block_index", "order", "witness"})
        require(type(row["block_index"]) is int and row["block_index"] > 0, "actual non-module block index required")
        order, witness = row["order"], row["witness"]
        closed(witness, {"block_index", "operation_sha256", "edges", "order", "unknown_effect_indices"})
        require(type(order) is list and all(type(i) is int for i in order), "exact ordered integer permutation required")
        require(type(witness["operation_sha256"]) is list and bool(witness["operation_sha256"]), "complete actual operation identity roster required")
        count = len(witness["operation_sha256"])
        require(count <= 2048 and sorted(order) == list(range(count)), "full current operation permutation required")
        require(type(witness["block_index"]) is int and witness["block_index"] == row["block_index"], "current request/witness block differs")
        require(type(witness["order"]) is list and all(type(i) is int for i in witness["order"])
                and witness["order"] == order, "current request/witness order differs")
        for digest in witness["operation_sha256"]:
            require(type(digest) is str and len(digest) == 64 and not set(digest) - set("0123456789abcdef"), "actual operation hash metadata malformed")
        edges = witness["edges"]
        require(type(edges) is list, "ordered dependency-pair list required")
        for edge in edges:
            require(type(edge) is list and len(edge) == 2 and all(type(i) is int for i in edge)
                    and 0 <= edge[0] < edge[1] < count, "malformed static source dependency")
        require(edges == [list(e) for e in sorted(set(tuple(e) for e in edges))], "dependency roster must be sorted/unique")
        unknown = witness["unknown_effect_indices"]
        require(type(unknown) is list and all(type(i) is int and 0 <= i < count for i in unknown)
                and unknown == sorted(set(unknown)), "unknown-effect roster malformed")
    return value


def shape(receipt):
    closed(receipt, KEYS)
    require(receipt["schema"] == SCHEMA and type(receipt["version"]) is int and receipt["version"] == 1,
            "unknown/legacy scheduling receipt version")
    require(type(receipt["root"]) is str and Path(receipt["root"]).is_absolute(), "live original root required; path edits are not portability qualification")
    settings(receipt["settings"])
    require(receipt["mode"] in MODES, "unknown scheduling proof/model mode")
    require(receipt["qualification_state"] == "UNQUALIFIED" and valid_claims(receipt["claims"]), "exact provisional claims required")
    for key in ("originals", "orders", "snapshots", "order_snapshots", "sources", "files", "dependencies"):
        require(type(receipt[key]) is list, "ordered complete reference roster required")
        for row in receipt[key]:
            reference_row(row)
    for key in ("journal", "journal_snapshot", "report", "capture", "profile", "query", "object"):
        reference_row(receipt[key])
    require(len(receipt["originals"]) in (2, 7) and len(receipt["snapshots"]) == len(receipt["originals"]),
            "complete original pair/seven-buffer roster required")
    require(2 <= len(receipt["orders"]) <= 9 and len(receipt["order_snapshots"]) == len(receipt["orders"]), "complete current-order snapshot roster required")
    require(len(receipt["orders"]) == receipt["settings"]["count"] + 1, "requested count/current-order trace differs")
    require(bool(receipt["sources"]) and bool(receipt["files"]) and bool(receipt["dependencies"]), "empty source/evidence/dependency closure refused")
    require(type(receipt["roots"]) is list and bool(receipt["roots"]) and all(type(r) is str and r.startswith("Merlin.") for r in receipt["roots"])
            and len(receipt["roots"]) == len(set(receipt["roots"])), "nonempty exact project theorem roster required")
    if receipt["mode"] == "graph":
        require(receipt["model"] is None, "graph mode cannot manufacture initialized witness metadata")
    else:
        require(type(receipt["model"]) is dict and receipt["model"].get("source_success_assumed") is False,
                "initialized model must explicitly derive source execution")
        for name in ("review_approved", "native_fidelity_proved", "poison_freedom_proved", "native_or_poison_fidelity_proved", "model_or_review_approval"):
            if name in receipt["model"]:
                require(type(receipt["model"][name]) is bool and receipt["model"][name] is False, "model cannot grant native/review approval")
    return receipt


def collect_refs(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256", "bytes"}:
            yield value
        for child in value.values():
            yield from collect_refs(child)
    elif type(value) is list:
        for child in value:
            yield from collect_refs(child)
