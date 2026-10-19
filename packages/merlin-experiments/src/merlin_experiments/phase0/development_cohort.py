"""Freeze bounded public source/reference development products, never admission.

Every requested member stays in the manifest, including unsupported numerical
forms and denied reference costs. No failing or expensive member is scored away.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from merlin.common.cpu_pool import shared_compute_pool

from . import capacity_families as C
from . import development_coverage as V
from . import development_envelopes as E
from . import development_reference as R
from . import development_sources as D

SCHEMA = "merlin.phase0.frozen_public_development_sources.v1"


def _bytes(document):
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def capacity_requests(sweep, *, target, facts):
    refused = []
    entries = C.capacity_family_entries(sweep, target=target, facts=facts, skipped=refused)
    requests = []
    for index, entry in enumerate(entries):
        row = {
            "basis": "selected_rtl_capacity_family",
            "capacity_request": entry,
            "capacity": entry["performance"][C.CAPACITY_BLOCK],
        }
        geometry, pattern = None, None
        if entry.get("op") == "matmul" and entry.get("epilogue", []) == []:
            pattern, geometry = "matmul", {key: entry[key] for key in ("M", "K", "N")}
        elif entry.get("op") == "conv2d" and entry.get("epilogue", []) == []:
            padding, stride = entry["padding"], entry["stride"]
            if padding[:2] == padding[2:]:
                pattern = "conv_nhwc"
                geometry = dict(
                    H=entry["Himg"],
                    W=entry["Wimg"],
                    CI=entry["ci"],
                    CO=entry["N"],
                    KH=entry["kh"],
                    KW=entry["kw"],
                    SH=stride[0],
                    SW=stride[1],
                    PH=padding[0],
                    PW=padding[1],
                )
        elif entry.get("op") == "resident_reuse":
            pattern, geometry = (
                "resident_rhs_reuse",
                {"M": [item["M"] for item in entry["matmuls"]], "K": entry["K"], "N": entry["N"]},
            )
        if pattern is None:
            row.update(
                status="source_refused",
                reason="original capacity numerical/epilogue form has no exact public source factory",
            )
        else:
            row["declaration"] = {**E.declaration(pattern, geometry, ordinal=index), "name": f"capacity_{index:04d}"}
        requests.append(row)
    return requests, refused


def freeze(requests, output_root, *, provenance, budgets, resource_basis=None, progress=None):
    """Create a readonly byte manifest after bounded independent references.

    The normal Phase 0 source exclusion, capture, software/hardware admission
    and performance objective still must be issued separately.
    """
    required = {
        "max_members",
        "max_source_bytes",
        "max_reference_elements",
        "max_reference_work",
        "max_total_reference_work",
        "max_total_reference_elements",
    }
    if (
        not isinstance(budgets, dict)
        or set(budgets) != required
        or any(type(value) is not int or value < 1 for value in budgets.values())
        or not isinstance(requests, list)
        or len(requests) > budgets["max_members"]
    ):
        raise ValueError("public development freezing needs explicit finite complete-roster budgets")
    root = Path(output_root).absolute()
    if root.exists() or not root.parent.is_dir():
        raise ValueError("public development output must be fresh with an existing parent")
    root.mkdir(mode=0o700)
    (root / "members").mkdir()
    manifest = {
        "schema": SCHEMA,
        "scope": "public_architecture_informed_development_not_validation",
        "provenance": provenance,
        "resource_basis": resource_basis,
        "budgets": budgets,
        "members": [],
        "reference_work": 0,
        "reference_elements": 0,
        "phase0_admitted": False,
        "performance_objective_frozen": False,
        "compiler_baseline": "UNKNOWN; no execution qualification supplied by source freezing",
        "required_remaining": [
            "operator_private_exclusion",
            "sealed_capture_attestation",
            "reviewed_software_correspondence",
            "required_accelerator_execution",
            "baseline_correctness",
            "qualified_timing",
            "frozen_objective",
        ],
    }
    seen = set()
    for ordinal, request in enumerate(requests):
        if progress is not None:
            progress(ordinal, len(requests))
        member = {
            "ordinal": ordinal,
            "request": request,
            "request_sha256": D._digest(request),
            "baseline": {"status": "UNKNOWN", "reason": "not compiled or executed by this source-only producer"},
        }
        manifest["members"].append(member)
        declaration = request.get("declaration")
        if declaration is None:
            member.update(
                status=request.get("status", "source_refused"),
                reason=request.get("reason", "public source unavailable"),
            )
            continue
        try:
            description = D.describe(declaration)
            name = declaration["name"]
            if name in seen:
                raise ValueError("duplicate complete public source identity")
            seen.add(name)
            source = D.render(declaration).encode()
            if len(source) > budgets["max_source_bytes"]:
                raise ValueError("complete public loader exceeds declared source-byte ceiling")
            # Reuse the original roster owner and writer entry, not a competing
            # capsule format. Each member has its complete independent closure.
            staged = D.stage([declaration], root / "members" / name)
            workload = staged["staging"]["performance_applications"][name]
            loader = Path(workload["workload_root"]) / "loader.py"
            member.update(
                description=description,
                loader=str(loader.relative_to(root)),
                loader_identity=_identity(loader),
                capsule_entry=D.capsule_entry(declaration, loader),
                numeric_policy=description.get("pending_numeric_policy")
                or {
                    "compare": "per_output",
                    "outputs": {
                        row["name"]: {"compare": "exact_int", "dtype": row["dtype"]} for row in description["outputs"]
                    },
                },
                performance_boundary={
                    "scope": "complete_source_call",
                    "includes": [
                        "packing",
                        "layout",
                        "readout",
                        "command_issue",
                        "accelerator_transfers",
                        "carried_state",
                    ],
                    "merlin_host_compute": "separate_and_currently_UNMEASURED",
                },
                expected_cost={
                    "scalar_products": description["scalar_products"],
                    "logical_elements": description["logical_elements"],
                    "target_cycles": None,
                },
                required_accelerator_work={
                    "operation": "signed_i8_i32_contraction",
                    "count": description["contractions"],
                    "executed": False,
                },
                resource_pressure=V.resources(description, resource_basis),
                pending_contract=description.get("pending_contract", []),
            )
            work, elements = description["scalar_products"], description["logical_elements"]
            if (
                work > budgets["max_reference_work"]
                or elements > budgets["max_reference_elements"]
                or work + manifest["reference_work"] > budgets["max_total_reference_work"]
                or elements + manifest["reference_elements"] > budgets["max_total_reference_elements"]
            ):
                member.update(
                    status="source_frozen_reference_budget_denied",
                    reason="complete independent reference cost exceeds the frozen per-member or aggregate budget",
                )
                continue
            with shared_compute_pool(timeout_s=120):
                inputs = R.inputs(declaration)
                outputs = R.evaluate(declaration, inputs)
            reference = root / "members" / name / "independent-reference.npz"
            np.savez(
                reference,
                **{f"input_{index}": value for index, value in enumerate(inputs)},
                **{f"output_{index}": value for index, value in enumerate(outputs)},
            )
            member.update(
                status="source_and_independent_reference_frozen",
                reference=str(reference.relative_to(root)),
                reference_identity=_identity(reference),
                source_reference_agreement="UNKNOWN until original fresh capture/eager correspondence is checked",
            )
            manifest["reference_work"] += work
            manifest["reference_elements"] += elements
        except (ValueError, KeyError, TypeError) as error:
            member.update(status="source_refused", reason=str(error))
    coverage = V.matrix(manifest["members"])
    (root / "coverage.json").write_bytes(_bytes(coverage))
    # Bind every emitted byte, including public declaration and loader copies.
    files = {str(path.relative_to(root)): _identity(path) for path in sorted(root.rglob("*")) if path.is_file()}
    manifest["files"] = files
    manifest["source_cohort_sha256"] = D._digest(
        {
            "requests": requests,
            "files": files,
            "budgets": budgets,
            "provenance": provenance,
            "resource_basis": resource_basis,
        }
    )
    (root / "manifest.json").write_bytes(_bytes(manifest))
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)
    return manifest


def verify(root, *, manifest_sha256):
    root = Path(root)
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("frozen public source owner contains an indirect path")
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("public development manifest differs from its independent host pin")
    manifest = json.loads(raw)
    actual = {
        str(path.relative_to(root)): _identity(path)
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    if actual != manifest["files"] or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("frozen public development source/reference bytes changed")
    return manifest
