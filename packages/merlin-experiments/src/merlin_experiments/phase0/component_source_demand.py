"""Fixed private consumer of checked source contracts and logical demand.

The original generation and full references are replayed with their live source
selectors. Schedule selection is data, not an optimizer, hardware role or release
capability. No feature projection or author tool grant is made here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from merlin.common.jsonio import strict_json_equal
from merlin.perf import component_source_demand as D
from merlin.targetgen import component_program

from . import component_source_performance as P
from .component_generation import digest

SCHEMA = "merlin.component_source_demand_contracts.v1"
SELECTION_SCHEMA = "merlin.component_source_demand_selection.v1"


def _pin(path, *, maximum):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > maximum
    ):
        raise ValueError("source demand requires canonical bounded original source files")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(path, *, maximum):
    pin = _pin(path, maximum=maximum)
    data = json.loads(Path(pin["path"]).read_bytes())
    if _pin(path, maximum=maximum) != pin:
        raise ValueError("source demand selected source changed during reading")
    return data, pin


def prepare(
    *, root, coverage, hardware, software, source_contract, selection, max_document_bytes, max_members, max_total_nodes
):
    """Replay original membership and derive every selected schedule independently.

    Every source-checked development, guard and private member is required once;
    every denied original request remains in the output. Caller-owned explicit
    limits bound source metadata, arithmetic and selection files. This result is
    private source data; it cannot construct measured applicability coordinates.
    """
    if any(type(value) is not int or value < 1 for value in (max_document_bytes, max_members, max_total_nodes)):
        raise ValueError("source demand needs explicit positive document/member/node limits")
    selected, selected_pin = _read(selection, maximum=max_document_bytes)
    stored, contract_pin = _read(source_contract, maximum=max_document_bytes)
    if (
        type(selected) is not dict
        or set(selected) != {"schema", "generation_identity_sha256", "source_contract_sha256", "limits", "members"}
        or selected["schema"] != SELECTION_SCHEMA
        or type(selected["limits"]) is not dict
        or type(selected["members"]) is not list
    ):
        raise ValueError("source demand requires the closed explicit v1 source selection")
    try:
        limits = D.SourceDemandLimits(**selected["limits"])
        limits.verify()
    except (TypeError, ValueError) as error:
        raise ValueError("source demand metadata budget is unsupported") from error
    if len(selected["members"]) > max_members:
        raise ValueError("source demand complete member selection exceeds its metadata budget")
    schedules = {}
    for row in selected["members"]:
        if (
            type(row) is not dict
            or set(row) != {"member", "member_sha256", "source_sha256", "schedule"}
            or type(row["member"]) is not str
            or row["member"] in schedules
            or type(row["schedule"]) is not list
            or len(row["schedule"]) > limits.max_nodes
        ):
            raise ValueError("source demand omits or duplicates complete original schedule membership")
        schedules[row["member"]] = row
    if sum(len(row["schedule"]) for row in schedules.values()) > max_total_nodes:
        raise ValueError("source demand complete schedule roster exceeds its aggregate metadata budget")
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("source demand original generation owner is indirect")
    original = P.prepare_source_contracts(root=root, coverage=coverage, hardware=hardware, software=software)
    if (
        not strict_json_equal(stored, original)
        or selected["source_contract_sha256"] != contract_pin["sha256"]
        or selected["generation_identity_sha256"] != original["generation_identity_sha256"]
    ):
        raise ValueError("source demand differs from fixed complete source/reference generation replay")
    expected = {row["member"] for row in original["requested_members"] if row["state"] == "source_checked"}
    if set(schedules) != expected:
        raise ValueError("source demand selected another or incomplete development/guard/private roster")
    rows, source_pins = [], [selected_pin, contract_pin]
    for row in original["requested_members"]:
        if row["state"] != "source_checked":
            rows.append(
                {"member": row["member"], "cohort": row["cohort"], "state": "unavailable", "original_request": row}
            )
            continue
        member = row["original"]
        selected_member = schedules[row["member"]]
        if (
            selected_member["member_sha256"] != member["member_sha256"]
            or selected_member["source_sha256"] != member["source"]["sha256"]
        ):
            raise ValueError("source demand schedule belongs to another original member/source")
        path = root / row["member"] / "capsule.yaml"
        capsule_pin = _pin(path, maximum=max_document_bytes)
        capsule = yaml.safe_load(path.read_bytes())
        raw = capsule["operation"]["attributes"]["program"]
        storage = capsule["component_program"]["selected_storage"]
        features = D.derive_source_demand(
            program=raw,
            operand_dtype=storage["operand"],
            accumulator_dtype=storage["accumulator"],
            schedule=tuple(selected_member["schedule"]),
            limits=limits,
        )
        if (
            features["status"] == "observed"
            and features["typed_program_sha256"] != member["original_typed_program_sha256"]
        ):
            raise ValueError("source demand typed program differs from original ordered source contract")
        rows.append(
            {
                "member": row["member"],
                "cohort": row["cohort"],
                "state": features["status"],
                "original": member,
                "capsule": capsule_pin,
                "features": features,
            }
        )
        source_pins.extend(member["source_product_pins"])
    if (
        _pin(selection, maximum=max_document_bytes) != selected_pin
        or _pin(source_contract, maximum=max_document_bytes) != contract_pin
    ):
        raise ValueError("source demand original selections changed during complete derivation")
    for module in (D, P, component_program):
        source_pins.append(_pin(Path(module.__file__).resolve(), maximum=max_document_bytes))
    source_pins.append(_pin(Path(__file__).resolve(), maximum=max_document_bytes))
    unique = {pin["path"]: pin["sha256"] for pin in source_pins}
    for path, sha in unique.items():
        if _pin(path, maximum=max_document_bytes)["sha256"] != sha:
            raise ValueError("source demand complete source/reference membership changed")
    result = {
        "schema": SCHEMA,
        "scope": "private original logical source features only",
        "authority": "none",
        "generation_identity_sha256": original["generation_identity_sha256"],
        "coverage_sha256": original["coverage_sha256"],
        "source_contract": contract_pin,
        "selection": selected_pin,
        "limits": asdict(limits),
        "requested_members": rows,
        "aggregate_limits": {
            "max_document_bytes": max_document_bytes,
            "max_members": max_members,
            "max_total_nodes": max_total_nodes,
        },
        "source_product_pins": [{"path": path, "sha256": sha} for path, sha in sorted(unique.items())],
        "original_required_ids": original["original_required_ids"],
        "mandatory_missing_ids": original["mandatory_missing_ids"],
        "missing_producers": original["missing_producers"],
        "hardware_guard_link": "not_established",
        "candidate_verdict": "not_evaluated",
        "measured_baseline": "not_established",
        "release_authority": "not_issued",
    }
    result["sha256"] = digest(result)
    return result


def verify(record, **inputs):
    """Saved feature rows cannot replace fresh exact source derivation."""
    if not strict_json_equal(record, prepare(**inputs)):
        raise ValueError("source demand differs from complete original source derivation")
    return digest(record)
