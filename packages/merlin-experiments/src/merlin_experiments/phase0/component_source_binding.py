"""Reviewed source numerics without an already-authored target backend."""

import hashlib
from pathlib import Path

from merlin.targetgen import component_program
from merlin.targetgen.corpus_spec import CorpusBinding, dtype_info
from merlin.targetgen.semantic_families import from_op

from .software_intake import IndependentSoftwareIntake

SCHEMA = "merlin.phase0.source_semantics_admission.v1"


def require_live_selection(source_components, software, hardware, backend):
    """Check the explicit preauthor mode before any evidence discovery."""
    if source_components is not True or software is None or hardware is None or backend is not None:
        raise ValueError(
            "source-only component evidence requires live independent hardware/software and no backend contract"
        )


def backend_unknown(software_doc, *, contract):
    """Keep preauthor source semantics distinct from target backend support."""
    if contract or software_doc.get("capability_contract") is not None:
        raise ValueError("source-only components refuse inline backend capability declarations")
    return {
        "component": "backend-contract",
        "status": "unknown",
        "reason": "independent source generation precedes backend authoring; hardware support is unqualified",
    }


def identity(software, *, hardware):
    """Bind this explicit source-only mode to the original live selectors."""
    if type(software) is not IndependentSoftwareIntake or software.hardware is not hardware:
        raise ValueError("component source admission requires identical live independent software/hardware")
    software.verify()
    return {
        "schema": SCHEMA,
        "hardware_intake_sha256": hardware.sha256,
        "software_intake_sha256": software.sha256,
        "scope": "original source semantics only; target support, placement and physical effects UNKNOWN",
    }


def derive(software, *, hardware, datapath):
    """Bind only the independently issued integer reference declaration.

    Literal source extents need no hardware tile. Instruction classes, device
    support and physical resource roles remain unissued. Other numerical paths
    refuse before ordinary builders allocate tensors or reference outputs.
    """
    identity(software, hardware=hardware)
    numeric = software.public_facts()["numerical_semantics"]
    if (
        numeric["model"]["engine"] != "integer_reference"
        or not dtype_info(numeric["operand_dtype"])[3]
        or not dtype_info(numeric["accumulator_dtype"])[3]
    ):
        raise ValueError("component source binding has no independent reference for this numerical engine")
    try:
        readout = dtype_info(numeric["readout_dtype"])[0]
    except KeyError as error:
        raise ValueError("component source binding does not implement this distinct readout dtype") from error
    if readout != dtype_info(numeric["accumulator_dtype"])[0]:
        raise ValueError("component source binding does not implement a distinct readout dtype")
    allowed = {
        "operand_dtype",
        "accum_dtype",
        "subnormal_operand_flush",
        "numerical_semantics",
        "required_oracle_tiers",
        "compare",
    }
    if set(datapath) - allowed or datapath.get("compare", "exact_int") != "exact_int":
        raise ValueError("component source binding refuses undeclared target routing or numerical policies")
    for key, expected in (
        ("operand_dtype", numeric["operand_dtype"]),
        ("accum_dtype", numeric["accumulator_dtype"]),
        ("subnormal_operand_flush", numeric["subnormal_operand_flush"]),
        ("numerical_semantics", numeric),
    ):
        if datapath.get(key) != expected:
            raise ValueError("component source binding differs from reviewed numerical declarations")
    tiers = datapath.get("required_oracle_tiers", ["L0"])
    if tiers != ["L0"]:
        raise ValueError("component source binding cannot qualify physical or simulator oracle tiers")
    return CorpusBinding(
        target=hardware.target,
        tile_dim=None,
        operand_dtype=numeric["operand_dtype"],
        accum_dtype=numeric["accumulator_dtype"],
        integer=True,
        tiers=list(tiers),
        compare="exact_int",
        subnormal_operand_flush=numeric["subnormal_operand_flush"],
    )


def screen_written(capsule, directory, *, software, hardware):
    """Read only the generated standard source, retaining physical admission separately.

    Exact renderer correspondence fixes every source scalar body, use, and
    output. Reuse the source-only producer's conservative signature checker
    for every original semantic owner; unsupported constraints refuse. This
    is source preparation before compiler authoring, not compiled correctness.
    """
    from .component_compile_plan import _owner

    origin = identity(software, hardware=hardware)
    numeric = software.public_facts()["numerical_semantics"]
    if capsule.get("operation", {}).get("op") != "component_program":
        raise ValueError("source semantics admission supports only original explicit tensor DAGs")
    if (
        capsule.get("linalg_mlir") != "capsule.interface.mlir"
        or capsule.get("interface_mlir") != "capsule.interface.mlir"
    ):
        raise ValueError("source semantics admission requires the original ordinary standard source member")
    types = {
        "operand": dtype_info(numeric["operand_dtype"])[1],
        "accumulator": dtype_info(numeric["accumulator_dtype"])[1],
    }
    original = capsule["operation"]["attributes"]["program"]
    typed, expected = component_program.render(
        original, operand_dtype=types["operand"], accumulator_dtype=types["accumulator"]
    )
    if any(row["dtype"] not in set(types.values()) for row in typed["inputs"] + typed["nodes"]):
        raise ValueError("source semantics admission changes the independently selected integer source types")
    source = Path(directory) / "capsule.interface.mlir"
    raw = source.read_bytes() if not source.is_symlink() else None
    if (
        raw != expected.encode()
        or capsule.get("component_program") != typed
        or capsule.get("inputs") != typed["inputs"]
    ):
        raise ValueError("source semantics admission differs from the complete original typed DAG")
    owners, decisions = software.public_facts()["operations"], []
    values = {row["name"]: row for row in typed["inputs"] + typed["nodes"]}
    for node in typed["nodes"]:
        # Alias nodes vanish under this original logical SSA functionalization.
        # Require the corresponding identity/copy semantic owner nonetheless;
        # no physical alias signature is inferred from the source view.
        operation = {"alias": "copy", "update": "add"}.get(node["op"], node["op"])
        family = from_op(operation)
        selected = [
            owner
            for owner in owners
            if operation in owner.get("ops", []) or (not owner.get("ops") and family in owner.get("families", []))
        ]
        if len(selected) != 1:
            raise ValueError(
                "source semantics admission has missing or ambiguous independently reviewed operation owners"
            )
        if node["op"] == "alias" and "aliasing" in selected[0].get("signature", {}):
            raise ValueError("source semantics admission does not establish physical alias signature constraints")
        _owner(
            {"operation": operation, "operation_owner": selected[0]["id"]},
            software,
            types,
            original,
            typed_inputs=[values[name]["dtype"] for name in node["actual_inputs"]],
            result_dtype=node["dtype"],
        )
        decisions.append({"node": node["name"], "operation": operation, "owner": selected[0]["id"]})
    return {
        **origin,
        "status": "source_admitted",
        "program_sha256": hashlib.sha256(raw).hexdigest(),
        "operations": decisions,
        "output_roster": [row["name"] for row in typed["outputs"]],
    }


def verify_prepared_sources(root, report, *, software, hardware):
    """Reopen all retained originals using live selectors, without issuing HW grants."""
    import yaml

    from .component_generation import digest

    origin = identity(software, hardware=hardware)
    if digest({key: value for key, value in report.items() if key != "sha256"}) != report.get("sha256"):
        raise ValueError("prepared source report identity changed")
    saved = report["generation_identity"]
    if (
        saved.get("source_semantics_admission") != origin
        or saved.get("hardware_intake_sha256") != origin["hardware_intake_sha256"]
        or saved.get("software_intake_sha256") != origin["software_intake_sha256"]
        or report.get("hardware_intake_sha256") != origin["hardware_intake_sha256"]
        or report.get("software_intake_sha256") != origin["software_intake_sha256"]
        or "automatic_derivation" not in report
    ):
        raise ValueError("prepared sources differ from the exact live independent source mode")
    from .component_automatic import verify as verify_automatic

    root = Path(root)
    if root.is_symlink():
        raise ValueError("prepared source owner is indirect")
    # Replay the original required denominator before checking source bytes.
    # This source scope retains denied execution obligations; it cannot issue
    # the separate complete numerical-budget or physical coverage grant.
    verify_automatic(report["automatic_derivation"], report=report)
    checked = 0
    for obligation in report["obligations"]:
        for member in obligation["members"]:
            if member["state"] == "unavailable":
                continue
            if member["state"] != "source_generated":
                raise ValueError("source-only preparation cannot reinterpret physical or legacy member admission")
            relative = Path(member["member"])
            if relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
                raise ValueError("prepared source member escapes its original owner")
            directory = root / relative
            capsule_path = directory / "capsule.yaml"
            if (root / relative.parts[0]).is_symlink() or directory.is_symlink() or capsule_path.is_symlink():
                raise ValueError("prepared source member is indirect")
            capsule = yaml.safe_load(capsule_path.read_bytes())
            actual = screen_written(capsule, directory, software=software, hardware=hardware)
            if (
                actual != capsule.get("source_semantics_screen")
                or digest(actual) != member.get("source_semantics_screen_sha256")
                or actual["program_sha256"] != member["program_sha256"]
                or sorted(actual["output_roster"]) != member["output_roster"]
            ):
                raise ValueError("prepared source semantics or complete output roster changed")
            checked += 1
    return {
        "scope": origin["scope"],
        "prepared_members": checked,
        "missing_mandatory_obligations": [
            row["id"] for row in report["obligations"] if row["mandatory"] and row["state"] == "unavailable"
        ],
    }
