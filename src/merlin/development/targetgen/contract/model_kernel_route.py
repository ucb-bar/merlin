"""Observe selected OOT emission for exact integer model kernels.

This joins the source SSA bindings from ``model_kernel_outline`` to the command
buffer a selected compiler actually emits. It does not lower the host/device
boundary or claim that the complete model executes on the accelerator.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from subprocess import TimeoutExpired

from merlin.common.tree_hash import hash_tree
from merlin.development.targetgen import package_runtime
from merlin.development.targetgen.capsule_common import validate_interface_tensor_dtypes
from merlin.development.targetgen.contract import schemas
from merlin.development.targetgen.contract.model_kernel_outline import outline_integer_matmuls

SCHEMA = "merlin.model_kernel_route_probe.v1"


def _emit(package, source: Path, destination: Path, *, text: str, target: str, timeout: int) -> dict:
    destination.unlink(missing_ok=True)
    try:
        process = package_runtime.run_entrypoint(
            package,
            "emit_command_buffer",
            source,
            destination,
            timeout=timeout,
            write_bytecode=False,
        )
    except (OSError, TimeoutExpired) as exc:
        return {"status": "compiler_unavailable", "reason": str(exc)[:500]}
    if process.returncode != 0 or not destination.is_file():
        return {
            "status": "compiler_error",
            "returncode": process.returncode,
            "reason": process.stderr[-500:] if process.stderr else "no command buffer written",
        }
    raw = destination.read_bytes()
    try:
        command = json.loads(raw)
        if not isinstance(command, dict):
            raise ValueError("command buffer is not a mapping")
        if command.get("declined"):
            return {"status": "declined", "declined": command["declined"]}
        if not command.get("commands"):
            return {"status": "empty", "reason": "no target commands emitted"}
        schemas.validate_command_buffer(command)
        validate_interface_tensor_dtypes(command, text)
        if command.get("target") != target:
            raise ValueError("command buffer target differs from selected target")
        return {
            "status": "isolated_kernel_emitted",
            "command_buffer_sha256": hashlib.sha256(raw).hexdigest(),
            "commands": command["commands"],
        }
    except (ValueError, schemas.ContractViolation) as exc:
        return {"status": "invalid_command_buffer", "reason": str(exc)[:500]}


def probe_integer_model_kernels(
    model: bytes,
    *,
    target: str,
    software_spec: bytes,
    capability_contract: bytes,
    package_dir: str | Path,
    timeout: int = 30,
) -> dict:
    """Probe every exact candidate, deduplicating identical interface programs.

    An emitted command buffer is only an isolated kernel observation. The SSA
    binding and the unresolved composition obligations remain in the receipt.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    outline = outline_integer_matmuls(
        model, target=target, software_spec=software_spec, capability_contract=capability_contract
    )
    package = package_runtime.load_package(package_dir)
    if package.target != target:
        raise ValueError(f"selected OOT package target {package.target!r} differs from {target!r}")
    package_identity = hash_tree(package.directory)
    if not package_identity["present"] or not package_identity["sha256"]:
        raise ValueError("selected OOT package has no source identity")

    observed: dict[str, dict] = {}
    rows = []
    with tempfile.TemporaryDirectory(prefix="merlin_model_route_") as scratch:
        interface_path = Path(scratch) / "kernel.interface.mlir"
        command_path = Path(scratch) / "command_buffer.json"
        for candidate in outline["candidates"]:
            interface = candidate["interface_mlir"]
            digest = hashlib.sha256(interface.encode("utf-8")).hexdigest()
            if digest != candidate["interface_sha256"]:
                raise ValueError(f"interface digest changed for {candidate['operation_id']}")
            if digest not in observed:
                interface_path.write_text(interface, encoding="utf-8")
                observed[digest] = _emit(
                    package, interface_path, command_path, text=interface, target=target, timeout=timeout
                )
            rows.append(
                {
                    "operation_id": candidate["operation_id"],
                    "ordinal": candidate["ordinal"],
                    "operand_bindings": candidate["operand_bindings"],
                    "output_result_id": f"{candidate['operation_id']}:result:{candidate['output_result_index']}",
                    "output_type": candidate["output_type"],
                    "interface_sha256": digest,
                    "software_admission": candidate["software_admission"],
                    "emission": observed[digest],
                }
            )
        # This is an independent negative/positive boundary check. A package may
        # support isolated interfaces while declining the original linalg module.
        model_path = Path(scratch) / "complete_model.mlir"
        model_path.write_bytes(model)
        complete_model = _emit(
            package, model_path, command_path, text=model.decode("utf-8"), target=target, timeout=timeout
        )
        if complete_model["status"] == "isolated_kernel_emitted":
            complete_model["status"] = "commands_emitted_unverified"
    if hash_tree(package.directory) != package_identity:
        raise ValueError("selected OOT package changed during route probe")
    counts = Counter(row["emission"]["status"] for row in rows)
    return {
        "schema": SCHEMA,
        "target": target,
        "model_sha256": outline["model_sha256"],
        "software_spec_sha256": outline["software_spec_sha256"],
        "capability_contract_sha256": outline["capability_contract_sha256"],
        "package": {"package_id": package.package_id, **package_identity},
        "candidate_count": len(rows),
        "distinct_interfaces": len(observed),
        "refused": outline["refused"],
        "emission_counts": dict(sorted(counts.items())),
        "complete_model_direct_emission": complete_model,
        "candidates": rows,
        "stitching": {
            "status": outline["stitching"]["status"],
            "composition_status": outline["stitching"]["composition_plan"]["status"],
            "obligations": outline["stitching"]["composition_plan"]["obligations"],
        },
        "whole_model_offload_verified": False,
        "qualification": (
            "isolated OOT command emission only; no host dispatch, typed transfers, or model numerical proof"
        ),
    }
