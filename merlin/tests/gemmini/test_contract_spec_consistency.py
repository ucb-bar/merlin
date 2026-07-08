"""Anti-drift guard for the DOC-ONLY experiment-ABI contract specs.

Four contract specs are read by benchmark agents as prose, not loaded by Python:
scoring.yaml, target_dialect_contract.yaml, interface_dialect_contract.yaml, telemetry_schema.yaml.
Because no code loads them, they can silently drift from the code-enforced reality (the JSON schema
bundle, the mlir_oot_backend entrypoints, the command-buffer opcodes, the ABI VERSION). This test
pins the machine-checkable cross-references so drift fails CI instead of misleading an agent.
"""
from __future__ import annotations

import re

import yaml

from merlin.targetgen.contract.schemas import contract_dir

CONTRACT = contract_dir()
DOC_ONLY = ["scoring.yaml", "target_dialect_contract.yaml",
            "interface_dialect_contract.yaml", "telemetry_schema.yaml"]


def _load(name: str) -> dict:
    return yaml.safe_load((CONTRACT / name).read_text(encoding="utf-8"))


def test_specs_version_matches_abi_version():
    abi = (CONTRACT / "VERSION").read_text(encoding="utf-8").strip()
    for name in DOC_ONLY:
        v = str(_load(name).get("version", "")).strip()
        assert v == abi, f"{name}: version {v!r} != ABI VERSION {abi!r}"


def test_referenced_schema_files_exist():
    """Every schemas/*.schema.json a spec names must exist (catches deleted/renamed validators)."""
    for name in DOC_ONLY:
        text = (CONTRACT / name).read_text(encoding="utf-8")
        for ref in sorted(set(re.findall(r"schemas/[A-Za-z_]+\.schema\.json", text))):
            assert (CONTRACT / ref).is_file(), f"{name} references missing {ref} (schema drift)"


def test_interface_maps_to_valid_command_buffer_opcodes():
    """interface_dialect_contract required_ops[].maps_to must be real command-buffer opcodes."""
    valid = set(yaml.safe_load((CONTRACT / "command_buffer_abi.yaml").read_text())["opcodes"])
    ops = _load("interface_dialect_contract.yaml")["dialect"]["required_ops"]
    for op in ops:
        mt = op.get("maps_to")
        if not mt or mt.startswith("leaf"):   # tensor decl maps to a description, not an opcode
            continue
        assert mt in valid, f"interface op {op['name']} maps_to {mt!r} not in command-buffer opcodes {sorted(valid)}"


def test_required_outputs_reference_real_entrypoints():
    """target_dialect_contract produced_by names must be mlir_oot_backend_contract entrypoints."""
    entry = set(yaml.safe_load((CONTRACT / "mlir_oot_backend_contract.yaml").read_text())["entrypoints"])
    for out in _load("target_dialect_contract.yaml")["required_outputs"]:
        pb = out.get("produced_by")
        assert pb in entry, f"required_output {out['id']} produced_by {pb!r} not an entrypoint {sorted(entry)}"
