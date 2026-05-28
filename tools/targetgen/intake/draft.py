"""Render a *loadable* capability draft from a Classification.

The earlier ``_render_capability_draft`` in ``tools/targetgen_cmd.py``
emits a sketch-level YAML that intentionally lacks the
``platform / execution_model / isa / runtime / ...`` blocks the
``loader.load_capability_spec`` validator requires. That sketch is
useful for human review but cannot be fed back into ``plan_target`` or
``get_modification_map``.

This module produces a *minimal but loadable* draft: every required
block is present with stub values driven by the classification, so the
downstream MCP tools can chain off the result without the agent
hand-writing a full spec first. The agent (or operator) is expected to
refine the stubs before promotion.

Mapping from classification → schema:

  targetgen_styles           → execution_model.kind, isa.exposure.kind
  source_styles              → execution_model.attachment, ISA features
  primary_integration        → submission_model, compiler_recovery_stage

Anything ambiguous or unknown is set to a conservative default that is
flagged in the draft's ``_targetgen_intake.unresolved`` list so the
agent knows what to revisit.
"""

from __future__ import annotations

from typing import Any

from ..model import Classification

# Mapping from (targetgen_styles tuple) to a sensible execution_model kind.
# The classifier returns at most a 4-element subset.
_EXECUTION_KIND_BY_STYLE_SET: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"runtime_hal", "structured_text_isa", "post_global_plugin"}), "simt_gpu"),
    (frozenset({"post_global_plugin", "llvm_ukernel"}), "rocc_accelerator"),
    (frozenset({"post_global_plugin", "structured_text_isa"}), "structured_npu"),
    (frozenset({"runtime_hal"}), "matrix_coprocessor"),
    (frozenset({"llvm_ukernel"}), "vector_cpu_extension"),
    (frozenset({"post_global_plugin"}), "matrix_coprocessor"),
)


def _execution_kind(targetgen_styles: list[str]) -> str:
    s = frozenset(targetgen_styles)
    for required, kind in _EXECUTION_KIND_BY_STYLE_SET:
        if required.issubset(s):
            return kind
    return "cpu_only"


def _attachment(source_styles: list[str]) -> str:
    if "rocc_accelerator" in source_styles:
        return "rocc"
    if "mmio_accelerator" in source_styles or "rtl_or_systemc_model" in source_styles:
        return "mmio"
    if "gpu_codegen_stack" in source_styles:
        return "device"
    if "external_mlir_bridge" in source_styles or "external_toolchain_bridge" in source_styles:
        return "external_toolchain"
    return "cpu_extension"


def _isa_exposure_kind(targetgen_styles: list[str]) -> str:
    if "llvm_ukernel" in targetgen_styles:
        return "llvm_intrinsics"
    if "structured_text_isa" in targetgen_styles:
        return "text_isa"
    if "runtime_hal" in targetgen_styles:
        return "hal_runtime"
    return "none"


def _compiler_recovery_stage(targetgen_styles: list[str]) -> str:
    if "post_global_plugin" in targetgen_styles:
        return "post_global_optimization"
    if "structured_text_isa" in targetgen_styles:
        return "text_isa_export"
    if "llvm_ukernel" in targetgen_styles:
        return "llvmcpu_codegen"
    return "runtime_only"


def _runtime_required(targetgen_styles: list[str]) -> bool:
    return "runtime_hal" in targetgen_styles


def render_loadable_draft(classification: Classification) -> dict[str, Any]:
    """Return a YAML-ready dict that ``load_capability_spec`` will accept.

    The values for blocks the classifier cannot infer (numeric tolerances,
    operation triples, memory layout, ...) are conservative stubs the
    operator must revisit. We list those under
    ``_targetgen_intake.unresolved`` so they are not silently shipped.
    """
    target = classification.target
    tg = list(classification.targetgen_styles)
    src = list(classification.source_styles)

    exec_kind = _execution_kind(tg)
    isa_kind = _isa_exposure_kind(tg)
    recovery = _compiler_recovery_stage(tg)
    needs_runtime = _runtime_required(tg)

    unresolved = [
        "operations.compute: list the native ops the target accelerates",
        "memory.spaces: enumerate scratchpad / DMA / global memory regions",
        "numeric.legal_type_triples: declare which datatypes the target supports",
        "verification.simulator: point at a Chipyard sim or external golden",
        "tiles.preferred_tiles: list canonical tile shapes for codegen",
    ]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "target": {
            "name": target,
            "display_name": target.replace("_", " ").title(),
            "vendor": "unknown",
            "maturity": "experimental",
        },
        "platform": {
            "host_isa": "riscv64",
            "operating_systems": ["bare-metal"],
            "environments": ["runtime-driver"] if needs_runtime else ["compile-only"],
        },
        "execution_model": {
            "kind": exec_kind,
            "attachment": _attachment(src),
            "submission_model": "queue_dispatch" if needs_runtime else "host_call",
            "compiler_recovery_stage": recovery,
        },
        "isa": {
            "base": "rv64gc",
            "features": [],
            "exposure": {
                "kind": isa_kind,
                "needs_llvm_backend_changes": "llvm_backend_extension" in src,
                "needs_new_intrinsics": "rocc_accelerator" in src or "llvm_backend_extension" in src,
                "needs_new_feature_bits": "llvm_backend_extension" in src,
            },
            "state_model": {
                "kind": "runtime_managed_device_state" if needs_runtime else "host_managed",
            },
            "register_constraints": {
                "accumulator_register_file": None,
                "even_alignment_required": False,
                "vl_dependent": False,
            },
        },
        "operations": {
            "compute": [
                {
                    "name": "placeholder_op",
                    "native": True,
                    "type_triples": ["generic"],
                }
            ],
            "movement": [],
            "synchronization": [],
        },
        "geometry": {
            "tiles": {
                "compute_array_kind": None,
                "vector_vlen_bits": None,
                "vector_dlen_bits": None,
                "native_tile_options": [],
                "preferred_tiles": [],
            }
        },
        "memory": {
            # The loader requires at least one memory space. We default to
            # a single placeholder DDR space that the agent must replace
            # with the real device's memory map.
            "spaces": [
                {
                    "name": "ddr",
                    "kind": "global",
                    "size_bytes": None,
                    "notes": "placeholder — replace with real device memory map",
                },
            ],
            "preferred_layouts": {},
            "packing": {},
        },
        "numeric": {
            "legal_type_triples": [],
            "quantization": {},
            "rounding_modes": [],
            "saturation": False,
        },
        "runtime": {
            "required": needs_runtime,
            "executable_format": "elf" if needs_runtime else "none",
            "driver_backend_id": None,
            "uri_schemes": [],
            "synchronization": {},
        },
        "verification": {
            # All four blocks need an `available: bool` so the loader accepts
            # them. Stubbed false; agent flips when an oracle is wired.
            "golden_model": {"available": False},
            "simulator": {"available": False},
            "rtl": {"available": False},
            "perf_counters": {"available": False},
            "tolerances": {},
        },
        "access": {
            # Use enum values the loader recognises. ``simulator`` is the
            # honest default for held-out targets that only run on a sim.
            "model": "simulator",
            "sdk_requirements": [],
            "credential_requirements": "none",
            "availability_class": "always_local",
            "verification_gates": [],
        },
        # The classifier's rationale is recoverable from
        # ``classify_target`` directly; we do not embed it in the YAML
        # because the loader rejects unknown top-level keys. The
        # ``unresolved`` list is left as a docstring at the top of the
        # rendered file (see ``render_loadable_draft_yaml``).
        "references": [],
    }
    payload.setdefault("_unresolved_intake_notes", unresolved)  # popped before dump
    return payload


def render_loadable_draft_yaml(classification: Classification) -> str:
    """Same as ``render_loadable_draft`` but returns YAML text with a
    leading comment block listing the unresolved fields the agent must
    revisit before promoting the draft."""
    import yaml

    payload = render_loadable_draft(classification)
    notes: list[str] = payload.pop("_unresolved_intake_notes", [])
    header_lines = [
        "# Capability draft auto-generated by `targetgen_create_capability_draft`.",
        "# This is a SKETCH — every TODO below must be addressed before the spec",
        "# is promoted to target_specs/examples/<target>/capability.yaml.",
        "#",
        f"# classifier primary_integration : {classification.primary_integration}",
        f"# classifier targetgen_styles    : {classification.targetgen_styles}",
        f"# classifier source_styles       : {classification.source_styles}",
        f"# classifier confidence          : {classification.confidence}",
        "#",
        "# UNRESOLVED — agent must replace each stub:",
    ]
    for note in notes:
        header_lines.append(f"#   - {note}")
    header_lines.append("")
    return "\n".join(header_lines) + "\n" + yaml.safe_dump(payload, sort_keys=False)
