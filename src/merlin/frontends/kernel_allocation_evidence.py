"""Compiler-only byte custody for allocation-state source capsules.

This independent reader imports no producer or evaluator. Retained source and
normalization declarations grant no source-equivalence or memory/target proof.
"""

from __future__ import annotations

import hashlib
import json

_BASE = frozenset(
    {
        "allocation-original.ttir",
        "allocation-input.mlir",
        "allocation-arguments.json",
        "allocation-source-stage.json",
        "allocation-source-stderr.log",
    }
)
_NORMALIZED = frozenset({"allocation-normalized.ttir", "allocation-intake.json", "allocation-inventory.json"})


def _record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def checked_allocation_sources(checked, provenance, bound_source):
    """Select only the explicit source roster and validate all retained byte links."""
    binding = provenance.get("source_argument_binding", {})
    capsule = binding.get("source_capsule")
    if capsule is None:
        if "descriptor_bindings" in binding:
            raise ValueError("descriptor allocation binding lacks portable source custody")
        return {}
    if (
        not isinstance(capsule, dict)
        or capsule.get("schema") != "t2m.source-allocation-capsule.v1"
        or type(capsule.get("normalized")) is not bool
        or capsule.get("grants_semantics") is not False
    ):
        raise ValueError("unknown or authority-granting allocation source capsule")
    names = _BASE | (_NORMALIZED if capsule["normalized"] else frozenset())
    if set(capsule.get("artifacts", {})) != names:
        raise ValueError("allocation source capsule has an incomplete or noncompiler roster")
    copied = {name: checked(name) for name in sorted(names)}
    if any(capsule["artifacts"][name] != _record(data) for name, data in copied.items()):
        raise ValueError("allocation source capsule artifact differs from its byte record")
    if (
        binding.get("schema") != "t2m.source-allocation-bindings.v1"
        or binding.get("grants_semantics") is not False
        or binding.get("allocation_entry") != provenance.get("spec", {}).get("entry")
        or binding.get("bound_source_sha256") != _record(bound_source)["sha256"]
        or binding.get("original_source_sha256") != _record(copied["allocation-original.ttir"])["sha256"]
    ):
        raise ValueError("allocation source capsule differs from original/bound source")
    arguments = json.loads(copied["allocation-arguments.json"])
    if (
        arguments.get("schema") not in ("t2m.source-allocation-arguments.v1", "t2m.source-allocation-arguments.v2")
        or arguments.get("source_entry") != binding.get("source_entry")
        or not isinstance(arguments.get("arguments"), list)
    ):
        raise ValueError("allocation source argument roster differs from its declared interface")
    descriptor = binding.get("descriptor_bindings")
    if descriptor is not None and (
        not capsule["normalized"]
        or arguments["schema"] != "t2m.source-allocation-arguments.v2"
        or descriptor.get("schema") != arguments["schema"]
        or descriptor.get("grants_semantics") is not False
        or descriptor.get("source_entry") != binding["source_entry"]
        or descriptor.get("allocation_entry") != binding["allocation_entry"]
    ):
        raise ValueError("descriptor allocation source interface differs from its custody record")
    stage = json.loads(copied["allocation-source-stage.json"])
    input_name = "allocation-normalized.ttir" if capsule["normalized"] else "allocation-original.ttir"
    stage_records = {
        "source.ttir": _record(copied[input_name]),
        "lowered.mlir": _record(copied["allocation-input.mlir"]),
        "stderr.log": _record(copied["allocation-source-stderr.log"]),
    }
    if (
        set(stage)
        != {
            "schema",
            "source_stage_schema",
            "stage",
            "stage_completed",
            "flags",
            "artifacts",
            "optimizer",
            "grants_semantics",
        }
        or stage.get("schema") != "t2m.source-stage-custody.v1"
        or stage.get("source_stage_schema") != "t2m.triton-shared-stage.v1"
        or stage.get("stage") != "source"
        or stage.get("stage_completed") is not True
        or stage.get("grants_semantics") is not False
        or stage.get("flags")
        != [
            "--mlir-disable-threading",
            "--verify-each",
            "--mlir-print-op-generic",
            "--mlir-print-debuginfo",
            "--mlir-print-local-scope",
        ]
        or stage.get("artifacts") != stage_records
        or set(stage.get("optimizer", {})) != {"sha256", "bytes"}
    ):
        raise ValueError("allocation source custody does not bind its native generic source stage")
    optimizer = stage["optimizer"]
    if (
        type(optimizer["bytes"]) is not int
        or optimizer["bytes"] <= 0
        or not isinstance(optimizer["sha256"], str)
        or len(optimizer["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in optimizer["sha256"])
    ):
        raise ValueError("allocation source stage lacks an actual optimizer fingerprint")
    if capsule["normalized"]:
        intake, inventory = (
            json.loads(copied[name]) for name in ("allocation-intake.json", "allocation-inventory.json")
        )
        allowed = {
            "schema",
            "triton_version",
            "pass",
            "pipeline",
            "target_backend",
            "normalized_opcodes",
            "source_sha256",
            "output_sha256",
            "dot_precision_implementation",
            "descriptor_implementation",
            "descriptor_profile",
            "descriptor_argument_expansion",
            "tf32_numeric_profile",
        }
        if (
            not set(intake) <= allowed
            or intake.get("schema") != "t2m.intake-pass.v1"
            or intake.get("target_backend") != "not-invoked"
            or intake.get("source_sha256") != _record(copied["allocation-original.ttir"])["sha256"]
            or intake.get("output_sha256") != _record(copied["allocation-normalized.ttir"])["sha256"]
            or inventory.get("normalization_passes") != intake.get("pipeline")
            or inventory.get("normalized_opcodes") != intake.get("normalized_opcodes")
        ):
            raise ValueError("allocation source custody does not bind its original/normalized intake")
    return copied
