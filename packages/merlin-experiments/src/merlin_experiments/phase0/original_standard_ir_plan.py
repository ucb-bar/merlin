"""Exact upstream source selections and shared logical preallocation bounds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from merlin.targetgen.original_reference_values import format_record

from . import component_execution_budget as E
from . import original_reference_roster as R
from .command_intake import _git
from .original_call_sources import required_source_cohorts

SCHEMA = "merlin.original_standard_ir_selection.v1"
POINTWISE_SCHEMA = "merlin.original_standard_ir_selection.v2"
_BUDGET = {
    "max_members",
    "max_source_bytes",
    "max_total_source_bytes",
    "max_observation_bytes",
    "max_nesting",
    "max_integer_bits",
    "max_dense_elements",
    "max_dense_payload_bytes",
    "timeout_s",
}


def validate(selected, references):
    if (
        not isinstance(selected, dict)
        or set(selected)
        != {
            "schema",
            "reference_roster_sha256",
            "capture_checkout",
            "capture_commit",
            "mlir_opt",
            "budget",
            "execution_budget",
        }
        or selected["schema"]
        != (POINTWISE_SCHEMA if references.record_without_verification()["schema"] == R.POINTWISE_SCHEMA else SCHEMA)
        or selected["reference_roster_sha256"] != references.sha256
        or type(selected["capture_commit"]) is not str
        or len(selected["capture_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in selected["capture_commit"])
    ):
        raise ValueError("standard IR needs exact original references and a closed independent source selection")
    budget = selected["budget"]
    if type(budget) is not dict or set(budget) != _BUDGET or any(type(n) is not int or n < 1 for n in budget.values()):
        raise ValueError("standard IR needs complete positive construction/parser budgets")
    if budget["timeout_s"] > 600:
        raise ValueError("standard IR observation exceeds the bounded native execution domain")
    E.validate(selected["execution_budget"])
    R._plain(selected["mlir_opt"])
    checkout = Path(selected["capture_checkout"])
    if (
        not checkout.is_absolute()
        or not checkout.is_dir()
        or checkout.resolve() != checkout
        or any(item.is_symlink() for item in (checkout, *checkout.parents))
    ):
        raise ValueError("standard IR upstream checkout is absent or indirect")
    return selected


def capture_sources(selected):
    root, commit = Path(selected["capture_checkout"]), selected["capture_commit"]
    if _git(root, "rev-parse", "HEAD") != commit or _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("standard IR requires an unchanged clean selected upstream checkout")
    algorithm = _git(root, "rev-parse", "--show-object-format")
    if algorithm not in {"sha1", "sha256"}:
        raise ValueError("standard IR cannot independently verify the selected Git object format")
    rows = []
    for line in _git(root, "ls-tree", "-r", commit, "--", "m2m").splitlines():
        fields, relative = line.split("\t", 1)
        mode, kind, blob = fields.split()
        path = R._plain(root / relative)
        payload = path.read_bytes()
        observed = hashlib.new(algorithm, b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if mode not in {"100644", "100755"} or kind != "blob" or blob != observed:
            raise ValueError("standard IR selected upstream source differs from its exact tracked blob")
        rows.append(R._pin(path))
    if not rows or str(root / "m2m/__init__.py") not in {row["path"] for row in rows}:
        raise ValueError("standard IR lacks the selected ordinary upstream conversion entrypoint")
    return rows


def required_members(references):
    record = references.record()
    originals, seen, wanted = [], set(), []
    for member in record["members"]:
        key = (member["original_member_id"], member["graph_path"], member["node"], member["target"])
        if key not in seen:
            seen.add(key)
            originals.append(key)
        elif key != originals[-1]:
            raise ValueError("standard IR reference call membership is reordered or interleaved")
    for key in originals:
        wanted.extend((*key, cohort, extent) for cohort, extent in required_source_cohorts())
    actual = [
        tuple(member[field] for field in ("original_member_id", "graph_path", "node", "target", "cohort", "extent"))
        for member in record["members"]
    ]
    if actual != wanted or not wanted:
        raise ValueError("standard IR requires every original guard and private source slot in exact order")
    return record


def cost(metadata):
    slots = metadata["inputs"] + metadata["outputs"]
    count = sum(math.prod(row["shape"]) for row in slots)
    payload = sum(math.prod(row["shape"]) * (format_record(row["dtype"])["element_bits"] // 8) for row in slots)
    # Count complete framework execution, conversion examples, byte encodings
    # and parent comparison. Export/compiler metadata, heap and workspace are
    # explicitly not certified by these logical tensor counts.
    return {
        "reference_work": 4 * metadata["scalar_products"] + 8 * count,
        "materialized_elements": 8 * count,
        "tensor_payload_bytes": 8 * max(payload, 8 * count),
        "scalar_bits": max(64, *(format_record(row["dtype"])["element_bits"] for row in slots)),
    }


def preflight(record, selected):
    budget, policy = selected["budget"], selected["execution_budget"]
    # Reference/native allocations already admitted by the original owner are
    # included in this combined scope; no copied partial total substitutes.
    totals = dict(record["totals"]["execution"])
    members, decisions = [], []
    count = len(record["members"])
    reservation = (
        record["totals"]["source"]["source_bytes"]
        + 4 * count * budget["max_source_bytes"]
        + budget["max_observation_bytes"]
    )
    roster_denied = count > budget["max_members"] or reservation > budget["max_total_source_bytes"]
    for index, original in enumerate(record["members"]):
        row = {
            "index": index,
            "state": "unavailable",
            "reason": original.get("reason", "original reference unavailable"),
        }
        if original["state"] == "reference_checked":
            metadata = json.loads(Path(original["products"]["metadata"]["path"]).read_bytes())
            measured = cost(metadata)
            measured["materialized_elements"] += budget["max_dense_elements"]
            measured["tensor_payload_bytes"] += budget["max_dense_payload_bytes"]
            measured["reference_work"] += budget["max_dense_elements"]
            measured["scalar_bits"] = max(measured["scalar_bits"], budget["max_integer_bits"])
            limits = ["whole source/IR roster"] if roster_denied else E._exceeded(policy, measured, totals)
            if limits:
                row["reason"] = "combined source/reference preallocation budget exceeded: " + ", ".join(limits)
            else:
                for metric in E._METRICS:
                    totals[metric] += measured[metric]
                members.append(
                    {
                        "index": index,
                        **{key: original["products"][key]["path"] for key in ("source", "metadata", "inputs")},
                    }
                )
                row.update(
                    state="planned", reason="complete shared logical source/reference budgets checked", cost=measured
                )
        decisions.append(row)
    return members, decisions, {"execution": totals, "reserved_source_bytes": reservation}
