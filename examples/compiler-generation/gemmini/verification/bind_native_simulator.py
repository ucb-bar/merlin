"""Bind a later Gemmini kernel execution to an existing exact-binary RTL rebuild.

This is a cheap byte-identity check. It does not rebuild Chipyard or re-execute
the kernel; those observations remain in their separate retained receipts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attest_native_simulator import ignored_output

from merlin.common.digest import sha256_file
from merlin.targetgen.rtl.build_provenance import bind_reproduced_binary


def _safe_member(name: object) -> bool:
    return (
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and "/" not in name
        and "\\" not in name
        and Path(name).name == name
    )


def checked_kernel_artifacts(receipt: Path, doc: dict) -> int:
    """Check the saved same-ELF and console bytes, without re-grading numerics."""
    recorded_root = Path(doc.get("evidence_roots", {}).get("output_root", "")).resolve()
    if recorded_root != receipt.parent or doc.get("summary", {}).get("status") != "passed":
        raise ValueError("kernel receipt does not describe a passing run at its own evidence root")
    cases = doc.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("kernel receipt has no cases")
    for name, case in cases.items():
        if not _safe_member(name) or not isinstance(case, dict):
            raise ValueError("kernel receipt has an unsafe case")
        rtl = case.get("verilator")
        if not isinstance(rtl, dict) or case.get("status") != "passed" or rtl.get("status") != "passed":
            raise ValueError(f"kernel case lacks a passing RTL observation: {name}")
        elf = receipt.parent / name / "spike/merlin_gemmini_c0.elf"
        spike_console = receipt.parent / name / "spike_console.log"
        rtl_console = receipt.parent / name / "verilator_console.log"
        elf_digest = sha256_file(elf)
        if elf_digest != case.get("elf_sha256") or elf_digest != rtl.get("same_elf_sha256"):
            raise ValueError(f"kernel case is not a retained same-ELF observation: {name}")
        if sha256_file(spike_console) != case.get("spike_console_sha256"):
            raise ValueError(f"retained Spike console changed: {name}")
        if sha256_file(rtl_console) != rtl.get("console_sha256"):
            raise ValueError(f"retained Verilator console changed: {name}")
    return len(cases)


def checked_headline_artifacts(receipt: Path, doc: dict, *, facts_evidence: Path | None, source_sha: str) -> int:
    """Check the bounded synthetic window's input/ELF/console bytes and fact ancestry."""
    if facts_evidence is None:
        raise ValueError("headline binding requires --facts-evidence")
    root = receipt.parent
    roots = doc.get("evidence_roots")
    limits = doc.get("diagnostic_limits")
    if (
        doc.get("status") != "spike_and_verilator_passed"
        or doc.get("source_model_equivalence_claim") != "none_synthetic_operands"
        or limits
        != {"synthetic_integer_operands": True, "headline_numerical_equivalence": False, "target_admission": False}
        or not isinstance(roots, dict)
        or Path(roots.get("output_root", "")).resolve() != root
    ):
        raise ValueError("headline receipt is not a passing bounded synthetic diagnostic at its own root")
    generation_path = root / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    source_window = generation.get("source_kernel_window")
    if (
        generation.get("schema") != "merlin.gemmini-headline-window-generation.v1"
        or generation.get("status") != "generated_not_executed"
        or not isinstance(source_window, dict)
        or source_window != doc.get("source_kernel_window")
        or source_window.get("model_equivalence_claim") != "none_synthetic_operands"
    ):
        raise ValueError("headline execution differs from its generated synthetic window")
    generated = doc.get("generated_evidence") or {}
    for path, expected in (
        (generation_path, generated.get("generation_sha256")),
        (root / "capsule.yaml", generated.get("capsule_sha256")),
        (root / "capsule.interface.mlir", generated.get("interface_sha256")),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"headline generated input changed: {path.name}")
    if generated.get("capsule_sha256") != generation.get("capsule_sha256") or generated.get(
        "interface_sha256"
    ) != generation.get("interface_sha256"):
        raise ValueError("headline receipt and generation disagree on emitted inputs")
    facts_path = facts_evidence / "facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    validation = json.loads((facts_evidence / "validation.json").read_text(encoding="utf-8"))
    inputs = facts.get("inputs")
    if (
        sha256_file(facts_path) != generation.get("facts_sha256")
        or not isinstance(inputs, dict)
        or inputs.get("source_bundle_sha256") != source_sha
        or validation.get("status") != "verified"
        or validation.get("facts_sha256") != generation.get("facts_sha256")
        or validation.get("source_selection_sha256") != source_sha
    ):
        raise ValueError("headline facts do not bind the selected source bytes")
    elf_sha = sha256_file(root / "native/package_kernel.elf")
    if elf_sha != doc.get("elf_sha256") or elf_sha != doc.get("same_elf_sha256"):
        raise ValueError("headline Spike and Verilator did not retain the same ELF bytes")
    if sha256_file(root / "spike_console.log") != doc.get("spike_console_sha256"):
        raise ValueError("retained headline Spike console changed")
    if sha256_file(root / "verilator_console.log") != doc.get("verilator_console_sha256"):
        raise ValueError("retained headline Verilator console changed")
    return 1


def bind(
    *,
    attestation: Path,
    attested_kernel_receipt: Path,
    current_kernel_receipt: Path,
    source_evidence: Path,
    simulator: Path,
    output_root: Path,
    facts_evidence: Path | None = None,
) -> dict:
    attestation = attestation.resolve(strict=True)
    attested_kernel_receipt = attested_kernel_receipt.resolve(strict=True)
    current_kernel_receipt = current_kernel_receipt.resolve(strict=True)
    source_evidence = source_evidence.resolve(strict=True)
    simulator = simulator.resolve(strict=True)
    facts_evidence = facts_evidence.resolve(strict=True) if facts_evidence is not None else None
    build_record = json.loads(attestation.read_text(encoding="utf-8"))
    if build_record.get("schema") != "merlin.gemmini-verilator-provenance.v1":
        raise ValueError("not a Gemmini Verilator build attestation")
    source_sha = sha256_file(source_evidence / "source-selection.json")
    checked_cases = {}
    for label, receipt in (("attested", attested_kernel_receipt), ("current", current_kernel_receipt)):
        doc = json.loads(receipt.read_text(encoding="utf-8"))
        if doc.get("schema") == "merlin.gemmini-kernel-numerical-qualification.v1":
            checked_cases[label] = checked_kernel_artifacts(receipt, doc)
        elif label == "current" and doc.get("schema") == "merlin.gemmini-headline-window-numerical.v1":
            checked_cases[label] = checked_headline_artifacts(
                receipt, doc, facts_evidence=facts_evidence, source_sha=source_sha
            )
        else:
            raise ValueError("not a supported Gemmini native execution receipt")
    original_rebuild = attestation.parent / "simulator-rebuild"
    if sha256_file(original_rebuild) != sha256_file(simulator):
        raise ValueError("retained rebuilt simulator no longer matches selected executable")
    core_hashes = build_record.get("core_rtl_sha256")
    if not isinstance(core_hashes, dict) or len(core_hashes) < 100:
        raise ValueError("attestation lacks its selected RTL core closure")
    for name, expected in core_hashes.items():
        if not _safe_member(name):
            raise ValueError("unsafe selected RTL filename in attestation")
        if sha256_file(attestation.parent / "firtool/gen-collateral" / name) != expected:
            raise ValueError(f"retained selected RTL changed: {name}")
    record = bind_reproduced_binary(
        attestation_path=attestation,
        attested_execution_path=attested_kernel_receipt,
        current_execution_path=current_kernel_receipt,
        source_selection_path=source_evidence / "source-selection.json",
        executable_path=simulator,
        attestation_prior_receipt_field=("kernel_receipt_sha256",),
        attestation_source_field=("source_selection_sha256",),
        attestation_tested_binary_field=("kernel_tested_binary_sha256",),
        attestation_rebuilt_binary_field=("rebuilt_binary_sha256",),
        execution_source_field=("selected_source", "selection", "sha256"),
        execution_binary_field=("toolchain", "verilator", "sha256"),
    )
    record["target"] = "gemmini"
    record["retained_rebuilt_binary_sha256"] = sha256_file(original_rebuild)
    record["retained_selected_core_rtl_files_checked"] = len(core_hashes)
    record["retained_kernel_cases_checked"] = checked_cases
    if facts_evidence is not None:
        record["current_facts_sha256"] = sha256_file(facts_evidence / "facts.json")
    record["limits"].append(
        "The retained Verilated C++ model was compared by the original attester, not rechecked here."
    )
    root = ignored_output(output_root)
    (root / "receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--attested-kernel-receipt", type=Path, required=True)
    parser.add_argument("--current-kernel-receipt", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--facts-evidence", type=Path, help="verified selected facts; required for headline runs")
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    record = bind(**vars(parser.parse_args()))
    print(json.dumps({"status": record["status"], "executable_sha256": record["executable_sha256"]}))


if __name__ == "__main__":
    main()
