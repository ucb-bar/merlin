"""Explicit compared-treatment bindings for paired evidence consumption.

Pure receipt checks issue no execution or timing authority. Callers still obtain
candidate handoffs from the original verifier with independently selected owners.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import compiler_library_state as LIBRARY
from . import compiler_pass_preparation as PREPARATION


def binding_arguments(handoff):
    """Reopen the original handoff owners before projecting expected data."""
    library = LIBRARY.handoff_selection(handoff)
    passes = PREPARATION.handoff_selection(handoff)
    return dict(
        compiler_library_binding=copy.deepcopy(handoff.compiler_library_binding) if library is not None else None,
        compiler_pass_preparation=copy.deepcopy(handoff.compiler_pass_preparation) if passes is not None else None,
    )


def validate_expected_treatments(manifest, expected, cells):
    """Require opt-in exact releases and actual prepared-arm identities."""
    library, prepared = expected.compiler_library_binding, expected.compiler_pass_preparation
    if library is not None:
        library = LIBRARY.validate_binding(library)
    if prepared is not None:
        if library is None:
            raise ValueError("prepared paired evidence lacks its independently selected library")
        prepared = PREPARATION.validate_pair(prepared, functional_sha256=expected.functional_submission_sha256)
        if prepared["selection"]["binding"]["library"] != library["snapshot"]["contract_sha256"]:
            raise ValueError("prepared paired evidence selected different library releases")
    plan = manifest.get("measurement_plan") or {}
    schema = (
        "paired_arm4_measurement_plan_v5"
        if prepared
        else ("paired_arm4_measurement_plan_v4" if library else "paired_arm4_measurement_plan_v3")
    )
    campaign = "paired_arm4_prepared_performance_campaign_v1" if prepared else "paired_arm4_performance_campaign_v2"
    if plan.get("schema") != schema or manifest.get("schema") != campaign:
        raise ValueError("unsupported paired measurement plan/campaign schema for the explicitly selected treatments")
    for name, value in (("additional_compiler_library", library), ("compiler_pass_preparation", prepared)):
        if (name in plan) != (value is not None) or plan.get(name) != value:
            raise ValueError("paired measurement changed its explicitly selected " + name)
    if library is None:
        if any(name in manifest for name in ("compiler_passes", "compiler_preparation", "comparison_baseline_sha256")):
            raise ValueError("legacy paired evidence cannot inherit prepared compiler treatments")
        return
    identities = manifest.get("identity_before") or {}
    if identities.get("compiler_library_binding_sha256") != library["sha256"]:
        raise ValueError("paired evidence does not retain its exact selected library identity")
    baseline = expected.functional_submission_sha256
    if prepared is not None:
        baseline = prepared["baseline"]["sha256"]
        if (
            manifest.get("comparison_baseline_sha256") != baseline
            or manifest.get("compiler_passes") != prepared["selection"]
            or manifest.get("compiler_preparation") != prepared["preparation"]
            or identities.get("compiler_pass_preparation_sha256") != prepared["sha256"]
        ):
            raise ValueError("paired evidence lost its original preparation or actual comparison baseline")
    elif any(name in manifest for name in ("compiler_passes", "compiler_preparation", "comparison_baseline_sha256")):
        raise ValueError("unprepared paired evidence cannot inherit a prepared comparison")
    if identities.get("baseline_sha256") != baseline or identities.get("candidate_sha256") != expected.candidate_sha256:
        raise ValueError("paired evidence changed its independently selected compared compiler bytes")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("selected-library paired evidence lacks its actual arm schedule")
    digests = {"baseline": baseline, "candidate": expected.candidate_sha256}
    for row in schedule:
        if (
            not isinstance(row, dict)
            or row.get("arm") not in digests
            or row.get("package_sha256") != digests[row["arm"]]
        ):
            raise ValueError("paired schedule differs from its actual independently selected compiler arm")
    _validate_raw_treatments(schedule, cells, library, prepared)


def _validate_raw_treatments(schedule, cells, library, prepared):
    """Reopen same-execution source/library bindings, not copied cycle labels."""
    fields = ("phase", "arm", "family", "capsule", "replicate")
    selected = {tuple(row.get(key) for key in fields): row for row in schedule}
    if len(selected) != len(schedule):
        raise ValueError("paired schedule repeats a selected compiler execution")
    reopened = {}
    consumed = set()
    for cell in cells:
        key = tuple(cell.get(name) for name in fields)
        path = Path(cell.get("raw_result_path") or "")
        digest = cell.get("raw_result_sha256")
        if (
            key not in selected
            or not path.is_absolute()
            or path.resolve() != path
            or not path.is_file()
            or path.stat().st_size > 32 << 20
        ):
            raise ValueError("selected-library paired evidence lacks a bounded original execution product")
        if (str(path), digest) not in reopened:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("selected-library paired raw execution bytes changed")
            reopened[str(path), digest] = json.loads(raw)
        raw = reopened[str(path), digest]
        if (
            raw.get("execution") != selected[key]
            or raw.get("compiler_library_binding_sha256") != library["sha256"]
            or ("compiler_pass_preparation_sha256" in raw) != (prepared is not None)
            or raw.get("compiler_pass_preparation_sha256") != (prepared["sha256"] if prepared else None)
        ):
            raise ValueError("paired raw execution changed its actual treatment or selected library")
        consumed.add(key)
    if consumed != set(selected):
        raise ValueError("paired raw evidence omits selected compiler executions")
