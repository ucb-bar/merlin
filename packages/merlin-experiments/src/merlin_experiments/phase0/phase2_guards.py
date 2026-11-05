"""Byte-bound functional guard inputs for the separate Phase 2 performance cohort.

Phase 2 performance capsules need real measurement claims. Negative-lane,
epilogue-fusion, and carried-configuration witnesses remain Phase 1 source
members and are referenced here as functional regression inputs, never recast as
performance capsules or evidence that a Phase 2 compiler executed them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from merlin.common.digest import sha256_bytes
from merlin.targetgen import conformance

from ..corpus.coverage import _selected_contract, merlin_supplied_host_axis
from ..corpus.phase_selection import validate_phase_selections
from ..phase1.source_inputs import fingerprint
from .coverage_commitment import INPUT_SCHEMA, _digest, _selected_program

SCHEMA = "merlin.phase0.phase2_functional_guards.v1"


def _selected_members(corpus: Path, inputs: dict, phase1_report: dict, phase2_report: dict):
    if inputs.get("schema") != INPUT_SCHEMA or not isinstance(inputs.get("conformance"), dict):
        raise ValueError("Phase 2 guard link requires selected Phase 0 coverage inputs")
    target = inputs.get("target")
    if not isinstance(target, str) or inputs["conformance"].get("target") != target:
        raise ValueError("Phase 2 guard link target differs from selected requirement")
    manifest_path = corpus / "MANIFEST.yaml"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Phase 2 guard link has no ordinary corpus manifest")
    manifest = yaml.safe_load(manifest_path.read_bytes()) or {}
    performance = ((manifest.get("performance_generation") or {}).get(target) or {}).get("phase") or {}
    category = performance.get("category")
    if not isinstance(category, str):
        raise ValueError("Phase 2 performance category is absent from generated provenance")
    selected = (manifest.get("phase_corpora") or {}).get(target)
    roles = validate_phase_selections(
        selected,
        performance_category=category,
        generated_members=set(manifest.get("generated") or []),
    )
    report_rows = {}
    for phase, report in (("phase1", phase1_report), ("phase2", phase2_report)):
        if report.get("inputs_sha256") != _digest(inputs):
            raise ValueError(f"{phase} coverage report selected-input identity changed")
        cohort = report.get("cohort") or {}
        rows = cohort.get("capsules")
        if not isinstance(rows, list) or cohort.get("sha256") != _digest(rows):
            raise ValueError(f"{phase} coverage report cohort identity changed")
        if len(rows) != len(roles[phase]):
            raise ValueError(f"{phase} selected member count differs from coverage report")
        by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
        if len(by_name) != len(rows):
            raise ValueError(f"{phase} coverage report has duplicate capsule names")
        documents = []
        for member in roles[phase]:
            directory = corpus / member
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"{phase} selected capsule is absent or indirect: {member}")
            descriptor_path = directory / "capsule.yaml"
            if descriptor_path.is_symlink() or not descriptor_path.is_file():
                raise ValueError(f"{phase} selected capsule descriptor is absent or indirect: {member}")
            descriptor = yaml.safe_load(descriptor_path.read_bytes()) or {}
            name = descriptor.get("name")
            row = by_name.get(name)
            if name != directory.name or row is None or fingerprint(directory) != row.get("sha256"):
                raise ValueError(f"{phase} selected capsule bytes changed: {member}")
            program = _selected_program(descriptor, directory)
            if program is None or sha256_bytes(program.read_bytes()) != row.get("program_sha256"):
                raise ValueError(f"{phase} selected capsule program bytes changed: {member}")
            documents.append((member, directory, descriptor, row))
        if {doc[2]["name"] for doc in documents} != set(by_name):
            raise ValueError(f"{phase} coverage report contains a foreign capsule")
        report_rows[phase] = documents
    return report_rows


def build_guard_link(corpus: Path, inputs: dict, phase1_report: dict, phase2_report: dict) -> dict:
    """Reference selected functional witnesses by exact bytes and requirement axis.

    This is Phase 0 source coverage, not proof of Phase 2 compiler behaviour. The
    guard inputs must be rerun after optimization to establish preservation.
    """
    corpus = Path(corpus)
    selected = _selected_members(corpus, inputs, phase1_report, phase2_report)
    spec = inputs["conformance"]
    contract = _selected_contract(spec, inputs)
    if contract is None:
        raise ValueError("Phase 2 guard link lacks the exact selected capability contract")
    negative_guards = [
        (member, directory, row)
        for member, directory, descriptor, row in selected["phase1"]
        if descriptor.get("source_role") == "derived_sweep"
        and "on_mesh" in ((descriptor.get("lanes") or {}).get("forbid") or ())
    ]
    # Host placement is Merlin-supplied from the frozen Phase 0 placement, so the host axes are no
    # public-capsule obligation; a retained forbid-the-mesh member is still byte-bound as a guard.
    host_lane = merlin_supplied_host_axis(spec, "host_lane")
    host_only = merlin_supplied_host_axis(spec, "host_only")
    functional_axes = {"epilogue": [], "carried_state": []}
    for member, directory, descriptor, row in selected["phase1"]:
        if descriptor.get("source_role") != "derived_sweep":
            continue
        axis = (descriptor.get("semantic") or {}).get("generalization_axis")
        if axis in functional_axes:
            functional_axes[axis].append((member, directory, row))
    epilogue_required = (spec.get("epilogue") or {}).get("required")
    carried_required = (spec.get("carried_state") or {}).get("required")
    epilogue = (
        conformance._epilogue_gap(
            epilogue_required, [directory for _, directory, _ in functional_axes["epilogue"]], labels={"public", "dev"}
        )
        if epilogue_required is not None
        else {"status": "not_applicable", "n_required": 0, "n_covered": 0, "uncovered": []}
    )
    carried_state = (
        conformance._carried_state_gap(
            carried_required,
            [directory for _, directory, _ in functional_axes["carried_state"]],
            labels={"public", "dev"},
        )
        if carried_required is not None
        else {"status": "not_applicable", "n_required": 0, "n_covered": 0, "uncovered": []}
    )
    coverage = {"host_lane": host_lane, "host_only": host_only, "epilogue": epilogue, "carried_state": carried_state}
    complete = all(
        axis.get("status") in {"ok", "not_applicable"}
        and not axis.get("uncovered")
        and not axis.get("unreadable_capsules")
        and axis.get("n_covered", 0) == axis.get("n_required", 0)
        for axis in coverage.values()
    )
    return {
        "schema": SCHEMA,
        "target": inputs["target"],
        "status": "axis_coverage_complete" if complete else "incomplete",
        "selected_inputs_sha256": _digest(inputs),
        "phase1_functional_cohort_sha256": phase1_report["cohort"]["sha256"],
        "phase2_performance_cohort_sha256": phase2_report["cohort"]["sha256"],
        "guards": [
            {"member": member, "name": row["name"], "sha256": row["sha256"], "program_sha256": row["program_sha256"]}
            for member, _, row in sorted(
                {
                    member: (member, directory, row)
                    for member, directory, row in (
                        negative_guards + functional_axes["epilogue"] + functional_axes["carried_state"]
                    )
                }.values()
            )
        ],
        "coverage": copy.deepcopy(coverage),
        "qualification": (
            "selected Phase 1 generated functional-regression inputs only; no Phase 2 compiler "
            "execution, numerical result, or performance claim is established"
        ),
    }


def verify_guard_link(saved: dict, corpus: Path, inputs: dict, phase1_report: dict, phase2_report: dict) -> None:
    """Fail closed if either cohort, a guard byte, or selected evidence changes."""
    if saved != build_guard_link(corpus, inputs, phase1_report, phase2_report):
        raise ValueError("Phase 2 functional guard link differs from selected source evidence")
