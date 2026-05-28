"""Audit an agent's claimed classification against scanner evidence.

The agent-driven path (``targetgen_propose_modifications``) lets the agent
declare integration styles based on its own reading of the source. That
flexibility is the whole point — but it must be checked. This module
cross-references each claim against the ``SourceInventory`` the scanners
already produced and reports support / silence / contradiction per claim.

Audit findings are not pass/fail by themselves. Each finding has a
``severity`` (``info`` / ``warning`` / ``error``) and a ``conclusion``
the agent (or operator) can act on:

  supported    — at least one scanner kind in the expected set fired
  unsupported  — no expected scanner kind fired (silence; agent should
                 explain why or revise)
  contradicted — a different style's signature fired strongly and the
                 claim's signature did not (likely misclassification)

The mapping from claimed style → expected scanner kinds is intentionally
the *inverse* of ``intake/classifier.py``'s rules, so the audit catches
the same evidence the deterministic classifier would use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .model import SourceInventory

Severity = Literal["info", "warning", "error"]
Conclusion = Literal["supported", "unsupported", "contradicted"]


@dataclass(slots=True)
class AuditFinding:
    severity: Severity
    claim: str
    conclusion: Conclusion
    expected_evidence: list[str]
    actual_evidence: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class AuditReport:
    overall_status: Literal["pass", "warn", "fail"]
    findings: list[AuditFinding]
    unused_evidence: list[str] = field(default_factory=list)


# Scanner kinds that support each source-facing style. Mirrors
# ``intake/classifier.py``'s decision tree.
SOURCE_STYLE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "external_mlir_bridge": ("mlir_dialect", "mlir_op_definition", "mlir_dialect_cmake"),
    "external_toolchain_bridge": ("iree_external_hal_driver_registration",),
    "chipyard_generator": ("chipyard_project", "chipyard_config", "firesim_collateral"),
    "rocc_accelerator": ("chisel_attachment_rocc",),
    "mmio_accelerator": ("chisel_attachment_mmio", "rtl_mmio_register"),
    "rtl_or_systemc_model": ("verilog_module", "systemverilog_interface", "systemc_module"),
    "llvm_backend_extension": (
        "llvm_intrinsic_definition",
        "llvm_target_backend",
        "llvm_riscv_features",
    ),
    "gpu_codegen_stack": (
        # gpu_codegen_stack is itself a composite signal in the classifier.
        # Audit accepts any of the constituent doc kinds OR an explicit
        # GPU/SIMT documentation marker.
        "hal_driver_source",
        "hal_registration",
        "isa_documentation",
        "memory_model_documentation",
        "synchronization_documentation",
        "runtime_documentation",
        "driver_documentation",
    ),
}

# The four canonical TargetGen styles map to a *union* of source styles
# (since multiple source paths can lead to the same TargetGen style).
TARGETGEN_STYLE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "runtime_hal": (
        "chipyard_project",
        "chipyard_config",
        "chisel_attachment_mmio",
        "rtl_mmio_register",
        "verilog_module",
        "systemverilog_interface",
        "systemc_module",
        "hal_driver_source",
        "hal_registration",
        "iree_external_hal_driver_registration",
        "driver_documentation",
        "runtime_documentation",
    ),
    "structured_text_isa": (
        "isa_documentation",
        "synchronization_documentation",
        "memory_model_documentation",
    ),
    "post_global_plugin": (
        "mlir_dialect",
        "mlir_op_definition",
        "mlir_dialect_cmake",
        "chisel_attachment_rocc",
        "iree_compiler_plugin_registration",
    ),
    "llvm_ukernel": (
        "llvm_intrinsic_definition",
        "llvm_target_backend",
        "llvm_riscv_features",
        "llvm_inline_asm",
        "chisel_attachment_rocc",
    ),
}

# When a claim is unsupported, we also check for *contradictory* evidence:
# heavy presence of kinds that point to a different integration style.
CONTRADICTION_RULES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    # claim -> ((kinds suggesting a different style), conflicting style)
    "llvm_ukernel": ((("hal_driver_source", "iree_external_hal_driver_registration"), "runtime_hal"),),
    "structured_text_isa": ((("hal_driver_source", "verilog_module"), "runtime_hal"),),
    "external_mlir_bridge": ((("verilog_module", "systemc_module", "rtl_mmio_register"), "rtl_or_systemc_model"),),
}


def audit_claim(
    claimed_targetgen_styles: list[str],
    claimed_source_styles: list[str],
    inventory: SourceInventory,
) -> AuditReport:
    """Cross-check the agent's classification against scanner evidence."""
    detected = set(inventory.detected_source_kinds)
    findings: list[AuditFinding] = []
    used_kinds: set[str] = set()

    for style in claimed_source_styles:
        expected = SOURCE_STYLE_EVIDENCE.get(style)
        if expected is None:
            findings.append(
                AuditFinding(
                    severity="warning",
                    claim=f"source_style={style!r}",
                    conclusion="unsupported",
                    expected_evidence=[],
                    note=("Unknown source style. Known styles: " + ", ".join(sorted(SOURCE_STYLE_EVIDENCE.keys()))),
                )
            )
            continue
        present = sorted(set(expected) & detected)
        used_kinds.update(present)
        if present:
            findings.append(
                AuditFinding(
                    severity="info",
                    claim=f"source_style={style!r}",
                    conclusion="supported",
                    expected_evidence=list(expected),
                    actual_evidence=present,
                )
            )
        else:
            findings.append(
                AuditFinding(
                    severity="warning",
                    claim=f"source_style={style!r}",
                    conclusion="unsupported",
                    expected_evidence=list(expected),
                    note=(
                        f"None of the expected scanner kinds fired. Either "
                        f"the agent saw evidence the scanners missed (acceptable — "
                        f"explain in rationale) or {style!r} should be revised."
                    ),
                )
            )

    for style in claimed_targetgen_styles:
        expected = TARGETGEN_STYLE_EVIDENCE.get(style)
        if expected is None:
            findings.append(
                AuditFinding(
                    severity="error",
                    claim=f"targetgen_style={style!r}",
                    conclusion="unsupported",
                    expected_evidence=[],
                    note=(
                        "Unknown TargetGen style. Must be one of: " + ", ".join(sorted(TARGETGEN_STYLE_EVIDENCE.keys()))
                    ),
                )
            )
            continue
        present = sorted(set(expected) & detected)
        used_kinds.update(present)
        if present:
            findings.append(
                AuditFinding(
                    severity="info",
                    claim=f"targetgen_style={style!r}",
                    conclusion="supported",
                    expected_evidence=list(expected),
                    actual_evidence=present,
                )
            )
        else:
            contradiction = _check_contradiction(style, detected)
            if contradiction is not None:
                conflict_kinds, conflict_style = contradiction
                findings.append(
                    AuditFinding(
                        severity="error",
                        claim=f"targetgen_style={style!r}",
                        conclusion="contradicted",
                        expected_evidence=list(expected),
                        actual_evidence=sorted(conflict_kinds),
                        note=(
                            f"Scanner evidence ({', '.join(sorted(conflict_kinds))}) "
                            f"strongly suggests {conflict_style!r} instead of {style!r}."
                        ),
                    )
                )
            else:
                findings.append(
                    AuditFinding(
                        severity="warning",
                        claim=f"targetgen_style={style!r}",
                        conclusion="unsupported",
                        expected_evidence=list(expected),
                        note=(
                            "No expected scanner kind fired. Provide rationale or "
                            "additional source paths if this style is intentional."
                        ),
                    )
                )

    overall: Literal["pass", "warn", "fail"]
    if any(f.severity == "error" for f in findings):
        overall = "fail"
    elif any(f.severity == "warning" for f in findings):
        overall = "warn"
    else:
        overall = "pass"

    unused = sorted(detected - used_kinds)
    return AuditReport(overall_status=overall, findings=findings, unused_evidence=unused)


def _check_contradiction(claimed_style: str, detected: set[str]) -> tuple[set[str], str] | None:
    rules = CONTRADICTION_RULES.get(claimed_style, ())
    for kinds, conflict_style in rules:
        present = set(kinds) & detected
        # Require at least 2 contradiction signals to reduce false positives.
        if len(present) >= 2:
            return present, conflict_style
    return None
