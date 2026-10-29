"""Normative proof roster and live exact object recompilation for strict receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.common.owned_process import run_owned

from . import lean_graph_state as stable
from .lean_graph_state_chain_kernel import prepare
from .lean_graph_state_receipt_schema import file_record


def expected_roots(module):
    if module == "GraphStateInstance":
        return {
            "Merlin.StateEffects.preserveExchanges",
            "Merlin.GraphState.check_sound",
            "Merlin.GraphStateInstance.accepted",
            "Merlin.GraphStateInstance.actual_schedule",
        }
    if module == "GraphStateChainTransportInstance":
        return {
            "Merlin.GraphState.Chain.checkTransport_sound",
            "Merlin.GraphState.Chain.checkPrimitives_models",
            "Merlin.GraphStateChainTransportInstance.Actual.actual_transport",
        }
    if module == "GraphStateChainCompositionInstance":
        return {
            "Merlin.GraphStateChainCompositionInstance." + name
            for name in (
                "First.independently_preserved",
                "Second.independently_preserved",
                "independently_composed",
                "fixed_observations",
            )
        }
    if module == "GraphStateReceiptInstance":
        return {
            "Merlin.GraphState.Receipt.check_sound",
            "Merlin.GraphStateReceiptInstance.exactly_bound",
            "Merlin.GraphStateReceiptInstance.raw_stage_bytes",
            "Merlin.GraphStateReceiptInstance.direct_composite",
        }
    raise ValueError("unexpected proof module in strict receipt roster")


def checked_audit(report, module):
    expected_status = "graph_model_checked" if module == "GraphStateInstance" else "kernel_model_checked"
    if report.get("status") != expected_status:
        raise ValueError("retained proof report lacks its exact measured conditional status")
    for flag in (
        "public_acceptance_issued",
        "source_contract_approved",
        "native_fidelity_proved",
        "numerical_gates_discharged",
    ):
        if flag in report and report[flag] is not False:
            raise ValueError("forged approval/native/numerical flag in proof report")
    audit = report.get("audit", report.get("transitive_audit"))
    if type(audit) is not dict or audit.get("status") != "checked":
        raise ValueError("complete measured actual proof audit required")
    raw = audit.get("raw")
    if type(raw) is not dict or type(raw.get("theorems_audited")) is not int or raw["theorems_audited"] <= 0:
        raise ValueError("actual proof theorem count must be a positive exact integer")
    if raw.get("assurance") != "lean_kernel_std3_no_native_reflection":
        raise ValueError("actual raw audit assurance mode differs")
    if type(raw.get("violations")) is not list or raw["violations"]:
        raise ValueError("actual raw audit must have an explicit empty violation list")
    axioms = raw.get("axioms_used")
    if type(axioms) is not list or any(type(name) is not str for name in axioms) or len(set(axioms)) != len(axioms):
        raise ValueError("complete unique typed axiom inventory required")
    if not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("unapproved proof axiom in actual imported closure")
    statements = audit.get("statements")
    if (
        type(statements) is not dict
        or set(statements) != expected_roots(module)
        or any(type(value) is not str or not value for value in statements.values())
    ):
        raise ValueError("exact complete theorem statement roster required")
    if type(audit.get("imported_objects")) is not list or not audit["imported_objects"]:
        raise ValueError("actual imported object inventory required")
    for row in [audit["auditor"], *audit["imported_objects"], *audit["buffers"]]:
        file_record(row)
    module_inventory(audit)
    return audit


def module_inventory(audit):
    raw = audit["raw"]
    imports = raw.get("imports")
    if type(imports) is not list or not imports:
        raise ValueError("complete typed actual imported module roster required")
    by_path = {row["path"]: row for row in audit["imported_objects"]}
    if len(by_path) != len(audit["imported_objects"]):
        raise ValueError("duplicate imported object identity")
    result = {}
    for row in imports:
        if (
            type(row) is not dict
            or set(row) != {"module", "olean"}
            or any(type(value) is not str for value in row.values())
        ):
            raise ValueError("actual module/import fields need exact strings")
        if row["module"] in result or row["olean"] not in by_path:
            raise ValueError("actual imported object/module inventory is incomplete or duplicated")
        identity = by_path[row["olean"]]
        result[row["module"]] = {key: identity[key] for key in ("sha256", "bytes")}
    if {row["olean"] for row in imports} != set(by_path):
        raise ValueError("imported module/object inventories have different complete membership")
    return result


def query_roster(root, *, include_binding=True):
    root = Path(root)
    result = [(root / f"checks/direct/{name}", "GraphStateInstance") for name in ("stage-000", "stage-001", "complete")]
    result += [
        (root / f"checks/independent/transport-{index}/kernel", "GraphStateChainTransportInstance")
        for index in range(2)
    ]
    result.append((root / "checks/independent/kernel", "GraphStateChainCompositionInstance"))
    if include_binding:
        result.append((root / "checks/binding", "GraphStateReceiptInstance"))
    return result


def inventory(root):
    result = []
    for folder, module in query_roster(root):
        paths = [folder / f"{module}.lean", folder / f"{module}.olean", folder / "report.json"]
        paths += [
            folder / f"{name}.{suffix}" for name in ("kernel", "audit", "statements") for suffix in ("stdout", "stderr")
        ]
        result.extend(stable.file_ref(path) for path in paths)
    return sorted(result, key=lambda row: row["path"])


def dependencies(root):
    """Derive dependencies from the measured real imported objects and runtimes."""
    collected = {}

    def add(records):
        for record in records:
            file_record(record)
            if record["path"] in collected and collected[record["path"]] != record:
                raise ValueError("contradictory observed proof dependency identities")
            collected[record["path"]] = record

    for folder, module in query_roster(root):
        report = json.loads((folder / "report.json").read_text())
        audit = checked_audit(report, module)
        add(audit["imported_objects"])
        add([audit["auditor"], report.get("profile", report.get("proof_profile"))])
        if "source_closure" in report:
            for pair in report["source_closure"]:
                file_record(pair["original"])
                file_record(pair["retained"])
                if Path(pair["original"]["path"]).read_bytes() != Path(pair["retained"]["path"]).read_bytes():
                    raise ValueError("proof source original and retained source snapshot bytes differ")
            add([row["original"] for row in report["source_closure"]])
            add([row["retained"] for row in report["source_closure"]])
            runtime = report["host_runtime"]
            add([runtime["python"], *runtime["support_sources"]])
            external = report["producer_external_runtime"]
            add([external["python"], *external["loaded_sources"]])
    for index in range(2):
        transport = json.loads((Path(root) / f"checks/independent/transport-{index}/report.json").read_text())
        add(transport["dependencies"])
    for profile in (stable.prepare(), prepare()):
        add([profile["manifest"], *profile["files"]])
    add([stable.file_ref(stable.LEAN)])
    result = sorted(collected.values(), key=lambda row: row["path"])
    if not stable.fresh(result):
        raise ValueError("measured dependency closure changed")
    return result


def reproduce(root, destination):
    """Recompile each real retained query; no saved audit JSON grants authority.

    Original source paths and the current selected private profile reproduce the
    exact objects. This catches coherently rehashed object/report substitutions.
    Then fresh raw audits inspect the actual imported module in its retained root.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen = prepare()
    measured = []
    for index, (folder, module) in enumerate(query_roster(root)):
        folder = folder.resolve()
        target = destination / f"query-{index:03d}"
        target.mkdir()
        source = folder / f"{module}.lean"
        original_object = stable.file_ref(source.with_suffix(".olean"))
        output = target / f"{module}.olean"
        with (target / "compile.stdout").open("x") as out, (target / "compile.stderr").open("x") as err:
            checked = run_owned(
                [str(stable.LEAN), str(source), "-o", str(output)],
                cwd=folder,
                env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
                stdout=out,
                stderr=err,
                timeout=60,
                check=False,
            )
        stable._write(
            target / "compile.json",
            {
                "returncode": checked.returncode,
                "source": stable.file_ref(source),
                "selected_profile": frozen["manifest"],
                "selected_tool": stable.file_ref(stable.LEAN),
            },
        )
        if checked.returncode:
            raise RuntimeError(
                f"retained query recompilation unavailable: exit {checked.returncode}; buffers at {target}"
            )
        if stable.file_ref(output)["sha256"] != original_object["sha256"]:
            raise ValueError("retained proof object does not reproduce from its query under selected actual imports")
        report = json.loads((folder / "report.json").read_text())
        recorded = checked_audit(report, module)
        fresh = stable.audit_claim(target, frozen, module, list(recorded["statements"]))
        if (
            fresh["status"] != "checked"
            or fresh["statements"] != recorded["statements"]
            or module_inventory(fresh) != module_inventory(recorded)
            or any(
                fresh["raw"][key] != recorded["raw"][key] for key in ("theorems_audited", "axioms_used", "violations")
            )
        ):
            raise ValueError("retained raw proof audit/exact statements do not reproduce")
        measured.append({"original": original_object, "recompiled": stable.file_ref(output), "fresh_audit": fresh})
    stable._write(destination / "report.json", {"proofs": measured, "public_acceptance_issued": False})
    return measured
