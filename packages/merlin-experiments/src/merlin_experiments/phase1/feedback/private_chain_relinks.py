"""Reopen actual per-input instruction and supplier observations after execution.

Transient ELFs may have been deliberately discarded. Their successful original
observations are joined by exact digest, never described as current ELF bytes.
The caller separately admits the native partition and numerical/readback proof.
"""

from pathlib import Path

from merlin.common import invocation_record as IR
from merlin.model_check.full_build_identity import verified_recipe
from merlin.model_check.full_output_audit import Files
from merlin_experiments.phase2.contracts import document_sha256


def _require(value, message):
    if not value:
        raise ValueError(message)


def _historical_issuer(files, path, *, report_path, report, elf, expected_check):
    """Only the original ELF input may be unavailable after intentional cleanup."""
    from merlin_experiments.phase2 import component_instruction_audit as audit

    call = files.json(path)
    _require(call.get("schema") == IR.SCHEMA and call.get("kind") == "python_call"
             and call.get("stage") == "whole_linked_instruction_policy"
             and call.get("callable") == audit.__name__ + "._scan"
             and call.get("status") == "completed" and call.get("outcome") == "returned"
             and all(call.get(k) is True for k in (
                 "inputs_unchanged", "dependencies_unchanged", "executable_unchanged")),
             "relink has no completed unchanged original instruction issuer")
    check = {"policy_sha256": report.get("policy_sha256"),
             "accessor_sha256": report.get("accessor_intake_sha256"),
             "selection": report.get("decoder_selection")}
    _require(document_sha256(check) == expected_check
             and call.get("arguments") == {
                 "policy_sha256": check["policy_sha256"], "accessor_intake_sha256": check["accessor_sha256"],
                 "decoder_selection": check["selection"]},
             "relink instruction issuer selected another original policy/accessor/decoder")
    for pin in (call["executable"], call["stdout"], call["stderr"], *call["dependencies"]):
        files.selected(pin)
    source = files.pin(Path(audit.__file__).resolve())
    _require(source in call["dependencies"], "relink instruction issuer lost its actual source producer")
    expected_elf = {"path": elf["path"], "sha256": elf["sha256"]}
    _require(call["inputs"].count(expected_elf) == 1
             and call["outputs"] == [files.pin(report_path)], "instruction issuer does not bind the actual relink ELF")
    for pin in call["inputs"]:
        if pin != expected_elf or Path(pin["path"]).is_file():
            files.selected(pin)


def verify_relinks(build, *, program, run_root, steps, selection_record):
    """Verify one native program's actual step policy/supplier receipts; return pins.

    ``run_root`` is the selected input directory containing ``program.0`` etc.
    The fixed native partition/producer is independently verified by the caller.
    """
    from merlin.llvmlower.link_supplier_trace import verify_link_suppliers

    files = Files()
    base = verified_recipe(build)
    expected_check = selection_record["instruction_selection"]["instruction_check_sha256"]
    _require(type(steps) is int and steps > 0, "relink verification needs the declared positive step count")
    native = build.get("native_qualification") or {}
    _require(native.get("program") == program
             and native.get("instruction_selection") == selection_record["instruction_selection"],
             "relink base build lost its original native instruction selection")
    files.selected(native["partition"])
    original_suppliers = base.get("link_suppliers")
    expected_suppliers = ({symbol: Path(item["selected_input"]["path"])
                           for symbol, item in original_suppliers["symbols"].items()}
                          if original_suppliers else None)
    rows = []
    for step in range(steps):
        directory = Path(run_root) / f"{program}.{step}"
        recipe = files.json(directory / "compilation_recipe.json")
        elf = recipe["executable"]
        _require(recipe.get("status") == "completed" and recipe.get("commands")
                 and all(row.get("status") == "returned" for row in recipe["commands"])
                 and recipe["commands"][-1].get("output") == elf
                 and elf["path"] == str(directory / "model.elf"),
                 "step has no completed original input-specific link")
        admission_path = directory / "native_linked_elf.json"
        admission = files.json(admission_path)
        report_path = Path(admission["report_path"])
        files.pin(report_path, admission["report_sha256"])
        report = files.json(report_path)
        _require(report.get("schema") == "merlin.independent_linked_instruction_audit.v1"
                 and report.get("status") == admission.get("status") == "accepted"
                 and report.get("elf_sha256") == admission.get("elf_sha256") == elf["sha256"]
                 and report.get("prohibited_hits") == [] and report.get("sections"),
                 "step instruction policy did not accept the actual input-specific ELF")
        invocations = list((report_path.parent / "invocations").glob("*/invocation.json"))
        _require(len(invocations) == 1, "relink instruction audit lacks one original issuer observation")
        _historical_issuer(files, invocations[0], report_path=report_path, report=report, elf=elf,
                           expected_check=expected_check)
        for pin in report["decoder_receipts"]:
            files.selected(pin)
        link = recipe["commands"][-1]
        if expected_suppliers:
            verify_link_suppliers(recipe.get("link_suppliers"), argv=link["argv"],
                                  input_identities=link["inputs"], expected_suppliers=expected_suppliers)
            for path in expected_suppliers.values():
                files.pin(path)
        execution = [files.json(path) for path in (directory / "invocations").glob("*/invocation.json")]
        execution = [row for row in execution if row.get("stage") == "selected_rocc_process"]
        _require(len(execution) == 1 and execution[0].get("status") == "completed"
                 and execution[0].get("returncode") == 0
                 and execution[0].get("inputs") == [{"path": elf["path"], "sha256": elf["sha256"]}],
                 "instruction-admitted ELF differs from the actually executed step")
        retained = Path(elf["path"]).is_file()
        if retained:
            files.selected(elf)
        rows.append({"step": step, "elf_sha256": elf["sha256"],
                     "transient_elf_current_reopen": "retained" if retained else "unavailable_deleted"})
    files.unchanged()
    return {"status": "verified", "program": program, "steps": len(rows), "rows": rows,
            "artifacts": [{"path": str(path), "sha256": digest} for path, digest in files.pins.items()],
            "scope": "original completed policy/supplier/execution correspondence; deleted ELF bytes unavailable"}
