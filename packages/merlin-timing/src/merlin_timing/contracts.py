"""Closed portable measurement contracts and exact provenance joins."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET = "merlin.spike_timing_dataset.v1"
MODEL = "merlin.spike_timing_screen.v1"
REPORT = "merlin.spike_timing_validation.v1"
CASES = "merlin.spike_timing_cases.v1"
NATIVE_CASES = "merlin.spike_timing_native_cases.v1"
PROFILES = {"phase1-public": ("public", "phase1"), "historical-private": ("private", "historical")}
SPLITS = {"train", "calibration", "test"}
DOMAIN_KEYS = {
    "device_sha256",
    "scope_sha256",
    "feature_engine_sha256",
    "feature_schema_sha256",
    "target_model_sha256",
    "memory_regime_sha256",
    "label_engine",
    "label_engine_sha256",
}
CASE_KEYS = {
    "case_id",
    "source_group_sha256",
    "workload_sha256",
    "mechanism_family",
    "split",
    "source",
    "elf_sha256",
    "input_sha256",
    "scope_sha256",
    "device_sha256",
}
RECORD_KEYS = {"record_id", "case_id", "features", "feature_provenance", "label", "event_report"}
MAX_FEATURES = 64
MAX_CASES = 10000
MAX_RECORDS = 50000
MAX_PAIR_COMPARISONS = 20000


class ContractError(ValueError):
    """An input cannot establish the declared measurement identity."""


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def closed(value: Any, keys: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{where} requires exactly {sorted(keys)}")
    return value


def sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{where} requires a lowercase SHA256 digest")
    return value


def text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ContractError(f"{where} requires bounded nonempty text")
    return value


def number(value: Any, where: str, *, positive: bool = False) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ContractError(f"{where} requires a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def read(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ContractError("JSON input exceeds the 16 MiB limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ContractError("JSON input must be an object")
    return value


def write(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def domain(value: dict) -> None:
    closed(value, DOMAIN_KEYS, "domain")
    for key in DOMAIN_KEYS - {"label_engine"}:
        sha(value[key], f"domain.{key}")
    text(value["label_engine"], "label_engine")
    if value["label_engine"] not in {"gsim", "verilator", "firesim"}:
        raise ContractError("label engine must name a measured cycle oracle")


def features(value: dict, names: list[str]) -> None:
    closed(value, set(names), "features")
    for name, item in value.items():
        if item is not None:
            number(item, f"feature {name}")


def execution_identity(case: dict) -> str:
    """The exact emitted ELF and input instance measured by both tools."""
    return digest(
        {key: case[key] for key in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")}
        | {"source_sha256": case["source"]["source_sha256"]}
    )


def workload(case: dict) -> str:
    """Same input/source/scope/device defines one within-workload comparison."""
    return digest(
        {key: case[key] for key in ("input_sha256", "scope_sha256", "device_sha256")}
        | {"source_sha256": case["source"]["source_sha256"]}
    )


def native_cases(document: dict) -> dict:
    """New inference inputs carry source and execution identity, never cycle labels."""
    closed(document, {"schema", "profile", "domain", "feature_names", "cases"}, "native prediction cases")
    if document["schema"] != NATIVE_CASES or document["profile"] not in PROFILES:
        raise ContractError("unsupported native prediction cases or profile")
    domain(document["domain"])
    names = document["feature_names"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= MAX_FEATURES
        or any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
    ):
        raise ContractError("declare 1..64 distinct native feature names")
    for name in names:
        text(name, "native feature name")
    if not isinstance(document["cases"], list) or len(document["cases"]) > MAX_CASES:
        raise ContractError("native prediction case count exceeds its limit")
    seen, executions, sources, families = set(), set(), {}, {}
    for row in document["cases"]:
        closed(row, {"case", "features", "event_report", "numeric"}, "native prediction case")
        case = closed(row["case"], CASE_KEYS - {"split"}, "native case identity")
        key = text(case["case_id"], "native case ID")
        if key in seen:
            raise ContractError("native prediction case IDs must be unique")
        seen.add(key)
        text(case["mechanism_family"], "native mechanism family")
        for field in CASE_KEYS - {"case_id", "mechanism_family", "split", "source"}:
            sha(case[field], field)
        source = closed(case["source"], {"visibility", "phase", "manifest_sha256", "source_sha256"}, "native source")
        visibility, phases = (
            ("public", {"phase1", "phase2"}) if document["profile"] == "phase1-public" else ("private", {"historical"})
        )
        if source["visibility"] != visibility or source["phase"] not in phases:
            raise ContractError("native inference source visibility/phase cannot enter this profile")
        sha(source["manifest_sha256"], "native source manifest")
        sha(source["source_sha256"], "native source")
        if any(case[key] != document["domain"][key] for key in ("device_sha256", "scope_sha256")):
            raise ContractError("native case device/scope differs from domain")
        if case["workload_sha256"] != workload(case):
            raise ContractError("native workload digest does not bind source/input/scope/device")
        if sources.setdefault(source["source_sha256"], case["source_group_sha256"]) != case["source_group_sha256"]:
            raise ContractError("one native source was assigned multiple leakage groups")
        if families.setdefault(case["workload_sha256"], case["mechanism_family"]) != case["mechanism_family"]:
            raise ContractError("one native workload cannot change its mechanism family")
        execution = (case["elf_sha256"], case["input_sha256"], case["scope_sha256"])
        if execution in executions:
            raise ContractError("one native execution cannot supply multiple prediction cases")
        executions.add(execution)
        features(row["features"], names)
        if row["numeric"] is not None:
            numeric = closed(
                row["numeric"],
                {"status", "receipt_sha256", "execution_sha256", "measured_now"},
                "native numeric receipt",
            )
            sha(numeric["receipt_sha256"], "native numerical receipt")
            if numeric["status"] not in {"pass", "fail", "unknown"} or type(numeric["measured_now"]) is not bool:
                raise ContractError("native numerical status and freshness must be explicit")
            if numeric["execution_sha256"] != execution_identity(case):
                raise ContractError("native numeric receipt differs from its exact execution")
    return document


def dataset(document: dict) -> dict:
    closed(document, {"schema", "profile", "domain", "feature_names", "cases", "records"}, "dataset")
    if document["schema"] != DATASET or document["profile"] not in PROFILES:
        raise ContractError("unsupported dataset schema or profile")
    domain(document["domain"])
    names = document["feature_names"]
    if (
        not isinstance(names, list)
        or not 1 <= len(names) <= MAX_FEATURES
        or any(not isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ContractError("declare 1..64 distinct feature names")
    for name in names:
        text(name, "feature name")
        if name == "fixed_cycles":
            raise ContractError("fixed_cycles is reserved for the explicit intercept")
    cases, records = document["cases"], document["records"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ContractError("declare 1..10000 cases")
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise ContractError("at most 50000 records may be imported")
    identities, groups, sources, workloads, executions = {}, {}, {}, {}, {}
    for case in cases:
        closed(case, CASE_KEYS, "case")
        key = text(case["case_id"], "case_id")
        if key in identities or case["split"] not in SPLITS:
            raise ContractError("case IDs must be unique and splits explicit")
        identities[key] = case
        text(case["mechanism_family"], "mechanism_family")
        for field in CASE_KEYS - {"case_id", "mechanism_family", "split", "source"}:
            sha(case[field], field)
        source = closed(case["source"], {"visibility", "phase", "manifest_sha256", "source_sha256"}, "source")
        if (source["visibility"], source["phase"]) != PROFILES[document["profile"]]:
            raise ContractError("source visibility/phase cannot enter this profile")
        sha(source["manifest_sha256"], "public/private source manifest")
        sha(source["source_sha256"], "source")
        if any(case[k] != document["domain"][k] for k in ("device_sha256", "scope_sha256")):
            raise ContractError("case device/scope differs from domain")
        if case["workload_sha256"] != workload(case):
            raise ContractError("workload digest does not bind source/input/scope/device")
        group, split = case["source_group_sha256"], case["split"]
        if groups.setdefault(group, split) != split:
            raise ContractError("a whole source leakage group crosses dataset splits")
        for index, identity in ((sources, source["source_sha256"]), (workloads, case["workload_sha256"])):
            if index.setdefault(identity, group) != group:
                raise ContractError("one source/workload was assigned multiple leakage groups")
        execution = (case["elf_sha256"], case["input_sha256"], case["scope_sha256"])
        if execution in executions:
            raise ContractError("repeat executions must remain records of one case")
        executions[execution] = key
    if {case["split"] for case in cases} != SPLITS:
        raise ContractError("explicit train, calibration and test source groups are required")
    seen = set()
    for record in records:
        closed(record, RECORD_KEYS, "record")
        key = text(record["record_id"], "record_id")
        if key in seen or record["case_id"] not in identities:
            raise ContractError("record IDs must be unique and join a declared case")
        seen.add(key)
        features(record["features"], names)
        proof = closed(
            record["feature_provenance"],
            {"engine_sha256", "model_sha256", "trace_sha256", "execution_sha256", "complete"},
            "feature provenance",
        )
        for key in ("engine_sha256", "model_sha256", "trace_sha256", "execution_sha256"):
            sha(proof[key], f"feature provenance {key}")
        if proof["execution_sha256"] != execution_identity(identities[record["case_id"]]):
            raise ContractError("feature report differs from its exact ELF/input/source/scope/device")
        if type(proof["complete"]) is not bool:
            raise ContractError("feature completeness must be explicit")
        if (
            proof["engine_sha256"] != document["domain"]["feature_engine_sha256"]
            or proof["model_sha256"] != document["domain"]["target_model_sha256"]
        ):
            raise ContractError("feature engine/model differs from exact domain")
        label = record["label"]
        if label is not None:
            closed(
                label,
                {
                    "engine",
                    "engine_sha256",
                    "receipt_sha256",
                    "execution_sha256",
                    "status",
                    "numeric",
                    "measured_now",
                    "cycles",
                },
                "cycle label",
            )
            sha(label["receipt_sha256"], "cycle measurement receipt")
            if label["execution_sha256"] != execution_identity(identities[record["case_id"]]):
                raise ContractError("cycle label differs from its exact ELF/input/source/scope/device")
            if (label["engine"], label["engine_sha256"]) != (
                document["domain"]["label_engine"],
                document["domain"]["label_engine_sha256"],
            ):
                raise ContractError("label engine differs from exact domain")
            if label["status"] not in {"pass", "fail", "unknown"} or label["numeric"] not in {
                "pass",
                "fail",
                "unknown",
            }:
                raise ContractError("label status and numerical status must be explicit")
            if type(label["measured_now"]) is not bool:
                raise ContractError("label measurement freshness must be explicit")
            if label["cycles"] is not None:
                number(label["cycles"], "measured cycles", positive=True)
        if record["event_report"] is not None and not isinstance(record["event_report"], dict):
            raise ContractError("event report must be an original report object or null")
    return document


def grouped_records(document: dict) -> dict[str, list[dict]]:
    rows = defaultdict(list)
    for record in document["records"]:
        rows[record["case_id"]].append(record)
    return dict(rows)


def case_point(case: dict, rows: list[dict], names: list[str]) -> tuple[dict | None, str | None]:
    """Aggregate repeats; failed/unknown repeats never become selected faster labels."""
    import statistics

    if not rows:
        return None, "no original feature/measurement records"
    for row in rows:
        label = row["label"]
        if (
            label is None
            or label["status"] != "pass"
            or label["numeric"] != "pass"
            or label["measured_now"] is not True
            or label["cycles"] is None
        ):
            return None, "missing, failed, stale or numerically unverified cycle label"
        if row["feature_provenance"]["complete"] is not True or any(row["features"][n] is None for n in names):
            return None, "incomplete feature observation"
    if any(row["features"] != rows[0]["features"] for row in rows):
        return None, "repeat feature observations disagree"
    return {
        "case": case,
        "features": rows[0]["features"],
        "cycles": statistics.median(row["label"]["cycles"] for row in rows),
        "records": rows,
    }, None
