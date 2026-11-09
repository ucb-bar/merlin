"""Read-only inventory of SW-contract evidence still missing from a frozen corpus.

This audits authored screening decisions. It does not observe target execution,
qualify a software spec, or promote any capsule to accelerator admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

_EVIDENCE = {
    "placement": (
        "Decode emitted target instructions and execution trace; prove the operation ran on the declared lane, "
        "with an independent numerical result."
    ),
    "operand_dtypes": (
        "Compare emitted operand ABI and transfer bytes with the native SW ABI; execute signed and boundary-value "
        "vectors on the target."
    ),
    "dtypes": (
        "Compare emitted operand/result ABI with native SW and RTL datapath widths; execute signed and boundary-value "
        "vectors on the target."
    ),
    "ordered_operand_dtypes": (
        "Record each ordered host operand's exact ABI dtype and execute a distinct-operand vector with the "
        "selected host compiler package."
    ),
    "ordered_result_dtypes": (
        "Record each ordered host result's exact ABI dtype and compare saved output bytes with an independent oracle."
    ),
    "compute_dtypes": (
        "Inspect selected host lowering for intermediate compute dtypes and execute rounding-sensitive vectors."
    ),
    "accumulator_dtype": (
        "Observe accumulator width and readout encoding in emitted commands and RTL trace; compare overflow-edge "
        "vectors against an independent oracle."
    ),
    "readout_dtype": (
        "Observe readout width and destination byte encoding in emitted commands and RTL trace; compare "
        "boundary-value vectors against an independent oracle."
    ),
    "ranks": (
        "Record emitted operand/result ranks for each claimed rank and execute representative batched and "
        "unbatched target cases."
    ),
    "layouts": (
        "Compare emitted strides and DMA address trace with native SW and RTL layout; execute a non-symmetric "
        "indexing vector."
    ),
    "tails": (
        "Execute non-tile-multiple extents with distinct padding canaries; inspect target transfer trace and "
        "valid-window readout."
    ),
    "broadcasting": (
        "Execute distinct per-batch operands and inspect source addresses; compare each batch with an "
        "independent reference."
    ),
    "aliasing": (
        "Record actual input/output address intervals at dispatch and execute the declared overlap or "
        "disjointness cases."
    ),
    "scale_granularity": (
        "Inspect scale encoding in emitted commands, native SW ABI and RTL; execute unequal per-channel values "
        "to distinguish tensor from channel scale."
    ),
    "epilogues": (
        "Observe the epilogue in the same target command/trace as its carrier, then compare sign-sensitive "
        "results with an independent oracle."
    ),
    "composed_with": (
        "Observe the carrier and composed stage in one target execution trace; rule out a standalone host fallback."
    ),
    "per_operation_inventory": (
        "Save every compiled operation's lane, dtype, shape, layout and transfer edges; validate each "
        "operation independently before whole-program admission."
    ),
    "host_capabilities": (
        "Bind the selected host compiler package and per-operation capability declaration by digest, then execute "
        "the operation on the host lane with an independent numerical comparison and no accelerator instructions."
    ),
    "contract_review": (
        "An independent reviewer must reconcile selected numerical and transfer declarations with pinned "
        "native SW, RTL and executed boundary-vector receipts; static declaration is not review evidence."
    ),
    "numerical_semantics": (
        "Execute target boundary vectors for internal overflow, readout scale rounding and narrowing; "
        "compare to an independent oracle and pinned native SW/RTL behavior."
    ),
    "transfer_contracts": (
        "Trace host-to-device load and device-to-host readout bytes, addresses, strides and valid windows "
        "against pinned native SW and RTL behavior."
    ),
    "missing_screen": (
        "Produce a per-operation authored SW-screen record for this frozen capsule; absence cannot be "
        "interpreted as admission."
    ),
    "unclassified_unknown": (
        "Inspect the frozen decision and record the missing typed observation or review receipt; unknown "
        "cannot be interpreted as admission."
    ),
}


def _evidence(axis: str) -> str:
    if axis.startswith("shape_bounds."):
        return (
            "Compare emitted dimensions with the declared bound and execute its lower, upper and "
            "out-of-boundary cases."
        )
    if axis in {"restriction", "semantics"} or axis.startswith(("restriction:", "semantics:")):
        return "Review the declared prose against pinned native SW, RTL and an executed discriminating vector."
    return _EVIDENCE.get(
        axis,
        "Identify a typed observation and independent execution or source receipt for this unresolved SW constraint.",
    )


def _decisions(rows: list, capsule_name: str):
    """Include host capability profiles, whose typed decisions are nested."""
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"software decision is malformed: {capsule_name}")
        yield row
        profiles = row.get("profiles") or []
        if not isinstance(profiles, list):
            raise ValueError(f"host capability profiles are malformed: {capsule_name}")
        for profile in profiles:
            if not isinstance(profile, dict) or not isinstance(profile.get("decisions") or [], list):
                raise ValueError(f"host capability profile is malformed: {capsule_name}")
            yield from _decisions(profile.get("decisions") or [], capsule_name)


def _performance_materialization(manifest: dict) -> dict:
    """Expose incomplete optional/required cohorts without interpreting them as a grade."""
    records = manifest.get("performance_generation") or {}
    if not isinstance(records, dict):
        raise ValueError("corpus performance-generation record is malformed")
    by_target = {}
    for target, record in sorted(records.items()):
        if not isinstance(record, dict):
            raise ValueError(f"performance-generation record is malformed: {target}")
        counts = record.get("counts") or {}
        families = counts.get("by_family") or {}
        if not isinstance(families, dict):
            raise ValueError(f"performance family counts are malformed: {target}")
        shortfalls = []
        for family, row in sorted(families.items()):
            if not isinstance(row, dict):
                raise ValueError(f"performance family count is malformed: {family}")
            admitted, written = row.get("admitted_members", 0), row.get("written_members", 0)
            if type(admitted) is not int or type(written) is not int or min(admitted, written) < 0:
                raise ValueError(f"performance family count is invalid: {family}")
            if written < admitted:
                shortfalls.append({"family": family, "admitted_members": admitted, "written_members": written})
        errors = record.get("errors") or []
        if not isinstance(errors, list):
            raise ValueError(f"performance errors are malformed: {target}")
        required_scope = {
            row.get("family")
            for row in record.get("families") or []
            if isinstance(row, dict)
            and (row.get("requirement_basis") or {}).get("axis") == "scope.performance.required"
        }
        declared = {
            row.get("family") for row in record.get("families") or [] if isinstance(row, dict)
        }
        unattributed = sorted(
            {str(row.get("family")) for row in errors if isinstance(row, dict) and row.get("family") not in declared}
        )
        by_target[target] = {
            "status": "incomplete" if errors or shortfalls else "materialized_not_verified",
            "errors": errors,
            "shortfalls": shortfalls,
            "required_scope_shortfalls": [row for row in shortfalls if row["family"] in required_scope],
            "unattributed_error_families": unattributed,
        }
    return by_target


def audit_corpus(corpus_root: Path) -> dict:
    """Return deterministic, non-certifying diagnostics for a frozen corpus root."""
    root = corpus_root.resolve(strict=True)
    manifest_bytes = (root / "MANIFEST.yaml").read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("generated"), list):
        raise ValueError("corpus MANIFEST.yaml has no generated capsule inventory")
    selected_spec = ((manifest.get("phase0_evidence") or {}).get("software_spec") or {})
    if not isinstance(selected_spec, dict):
        raise ValueError("corpus manifest software-spec identity is malformed")
    axes: dict[str, set[str]] = defaultdict(set)
    statuses: dict[str, int] = defaultdict(int)
    labels: dict[str, int] = defaultdict(int)
    refused: list[str] = []
    seen: set[str] = set()
    for name in manifest["generated"]:
        if not isinstance(name, str) or name in seen:
            raise ValueError("corpus manifest contains a duplicate or non-string capsule name")
        seen.add(name)
        capsule_path = (root / name / "capsule.yaml").resolve(strict=True)
        if not capsule_path.is_relative_to(root):
            raise ValueError(f"capsule path escapes frozen corpus: {name}")
        capsule = yaml.safe_load(capsule_path.read_bytes())
        if not isinstance(capsule, dict):
            raise ValueError(f"capsule descriptor is malformed: {name}")
        labels[str(capsule.get("label", "unknown"))] += 1
        screen = capsule.get("software_screen")
        if not isinstance(screen, dict):
            statuses["missing"] += 1
            axes["missing_screen"].add(name)
            continue
        status = str(screen.get("status", "unknown"))
        statuses[status] += 1
        if status == "unsupported":
            refused.append(name)
            continue
        if status != "unknown":
            # Authored admission is still not independent target execution.
            continue
        decisions = screen.get("decisions") or []
        if not isinstance(decisions, list):
            raise ValueError(f"software decisions are malformed: {name}")
        if not decisions:
            axes["per_operation_inventory"].add(name)
            continue
        before = sum(name in members for members in axes.values())
        for decision in _decisions(decisions, name):
            if decision.get("status") == "unsupported":
                continue
            if decision.get("role") == "host" and decision.get("status") == "unknown":
                axes["host_capabilities"].add(name)
            unresolved = decision.get("unresolved_constraints") or []
            if not isinstance(unresolved, list) or not all(isinstance(axis, str) for axis in unresolved):
                raise ValueError(f"unresolved SW constraints are malformed: {name}")
            for axis in unresolved:
                axes[axis].add(name)
            if decision.get("review_status") != "reviewed":
                axes["contract_review"].add(name)
        if sum(name in members for members in axes.values()) == before:
            axes["unclassified_unknown"].add(name)
    if selected_spec.get("status") != "reviewed":
        axes["contract_review"].update(seen)
    return {
        "schema": "merlin.phase0.software_contract_audit.v1",
        "verification_status": "not_established",
        "qualification": "static authored-screen inventory only; no target execution or independent review observed",
        "corpus": {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "generated_count": len(seen)},
        "selected_software_spec": selected_spec,
        "screen_status_counts": dict(sorted(statuses.items())),
        "label_counts": dict(sorted(labels.items())),
        "explicit_refusals": sorted(refused),
        "performance_materialization": _performance_materialization(manifest),
        "unresolved_axes": [
            {
                "axis": axis,
                "capsules": sorted(members),
                "count": len(members),
                "independent_evidence_needed": _evidence(axis),
            }
            for axis, members in sorted(axes.items())
        ],
        "global_contract_obligations": [
            {"axis": axis, "independent_evidence_needed": _evidence(axis)}
            for axis in ("numerical_semantics", "transfer_contracts")
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path, help="frozen Phase 0 capsules directory with MANIFEST.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(audit_corpus(args.corpus_root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
