"""Import an already observed whole-model build without recompiling its objects.

This is artifact transport, not native qualification. The protected caller must
first verify the native source, request, instruction and supplier gate and bind
its producer record in ``qualification``. Structural builds cannot manufacture
that qualification through this module.
"""

from __future__ import annotations

from pathlib import Path

from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.model_check import full_build as FB
from merlin.model_check.full_build_identity import build_material, verified_recipe


class NativeBuild(dict):
    """JSON-safe build fields plus the actual live, independently selected service."""

    def admit_relink(self, elf, directory):
        result = self.admission.evaluate(elf=elf, target=self["key"]["target"], evidence_root=directory)
        if result["status"] != "accepted":
            raise FB.FullBuildError("input-specific whole-model ELF failed selected instruction policy")
        import json

        (directory / "native_linked_elf.json").write_text(json.dumps(result, indent=2) + "\n")


def adopt(receipt, *, stage, target, label, submission, submission_sha256, settings, qualification,
          admission=None):
    """Reopen the original build's exact inputs, executable and compilation recipe."""
    stage, submission = Path(stage), Path(submission)
    inputs, output = receipt.get("inputs", {}), receipt.get("output", {})
    expected = {
        "capture": str(stage.resolve()), "target": target, "board": settings.board,
        "board_catalog_sha256": FB.sha256(settings.board_catalog),
        "dts_sha256": FB.sha256(settings.host_dts), "arena_mb": settings.arena_mb,
        "readback": "full", "host_math_policy": settings.host_math_policy,
        "respect_captured_quantization_scope": True, "host_llvm_bitcode": True,
        "capture_tree": strict_tree_sha256(stage),
        "package_tree": strict_tree_sha256(settings.host_package),
    }
    device = inputs.get("device")
    route = "device_requested_dispatch_unverified" if device is not None else "host_baseline"
    if (receipt.get("schema") != "merlin.baremetal-saved-model.v1"
            or receipt.get("status") != "compiled"
            or receipt.get("execution_route") != route
            or any(inputs.get(key) != value for key, value in expected.items())
            or (device is not None and device.get("package_tree") != strict_tree_sha256(submission))):
        raise FB.FullBuildError("native image does not correspond to the selected full-model build inputs")
    if not isinstance(qualification, dict) or not qualification:
        raise FB.FullBuildError("native image requires a protected qualification binding")
    elf = Path(output.get("elf", ""))
    if not elf.is_file() or elf.is_symlink() or FB.sha256(elf) != output.get("elf_sha256"):
        raise FB.FullBuildError("original native whole-model ELF is absent or changed")
    key = build_material(stage, target, label, submission, submission_sha256, settings, True)
    key["native_qualification"] = qualification
    build_dir = elf.parent
    weights = build_dir / "cgen/weights.bin"
    record = {
        "schema": FB.BUILD_SCHEMA, "key": key, "status": "linked",
        "elf": str(elf), "elf_sha256": output["elf_sha256"], "elf_bytes": elf.stat().st_size,
        "sections": FB.elf_sections(elf), "build_hash": output["build_hash"],
        "build_dir": str(build_dir), "compilation_recipe": output["compilation_recipe"],
        "runtime_abi_sha256": {name: FB.sha256(stage / name) for name in (
            "model.mlir", "weights.safetensors.manifest.json", "input_order.json",
        ) if (stage / name).is_file()},
        "weights_bin_sha256": FB.sha256(weights), "weights_bin_bytes": weights.stat().st_size,
        "device_routed": device is not None, "dedup_weights": bool(device and inputs.get("blob_weights")),
        "seconds": 0.0, "reused": True,
        "native_qualification": qualification,
    }
    verified_recipe(record)
    if admission is not None:
        from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

        if type(admission) is not LinkedElfAdmissionService:
            raise FB.FullBuildError("native image lost its original instruction admission service")
        admission.verify(target)
        record = NativeBuild(record)
        record.admission = admission
    return record
