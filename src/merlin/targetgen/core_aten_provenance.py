"""Source and selected hardware attribution for Core ATen verdicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.common import provenance
from merlin.common.paths import repo_root


def batch_provenance(
    directory: Path | None = None,
    *,
    target: str | None = None,
    package: Path | None = None,
    runner_options: dict[str, Any] | None = None,
    rtl_facts: Path | None = None,
    execution_provider: Path | None = None,
) -> dict[str, Any]:
    from merlin.integrations.model2mlir import root as m2m_root
    from merlin.runtime.backends.spike import spike_path
    from merlin.targetgen.core_aten_capture import _git_revision
    from merlin.targetgen.provenance import declared_pins
    from merlin.targetgen.target_registry import load_contract

    names = declared_pins(target)
    if target:
        try:
            contract = load_contract(target)
        except (FileNotFoundError, KeyError):
            contract = {}
        # Explicit descriptor dependencies select the hardware configuration.
        # Registry target membership can also include alternative bitstreams;
        # use that broader fallback only when the descriptor states no pins.
        if "hardware_pins" in contract:
            names = tuple(str(name) for name in contract["hardware_pins"])
    pins = {name: provenance.verify(name) for name in names}
    sources = [
        repo_root() / "src/merlin" / path
        for path in (
            "targetgen/core_aten_batch.py",
            "targetgen/core_aten_capture.py",
            "capture/static_metadata.py",
            "targetgen/core_aten_input_storage.py",
            "targetgen/capture_refusals.py",
            "targetgen/core_aten_numeric.py",
            "targetgen/core_aten_exact_dtype.py",
            "targetgen/core_aten_host_floor.py",
            "targetgen/core_aten_components.py",
            "llvmlower/integer_components.py",
            "llvmlower/integer_float_boundaries.py",
            "llvmlower/native_sign_components.py",
            "llvmlower/additive_regions.py",
            "llvmlower/fixed_point_ir.py",
            "llvmlower/floating_components.py",
            "llvmlower/boolean_components.py",
            "llvmlower/pointwise_primitives.py",
            "llvmlower/_ir.py",
            "targetgen/operation_numerics.py",
            "targetgen/core_aten_spec_int8.py",
            "targetgen/software_spec.py",
            "targetgen/operator_review.py",
            "targetgen/core_aten_readout.py",
            "targetgen/_recipe_quantizer.py",
            "targetgen/quant_recipe.py",
            "targetgen/core_aten_w8a8_reference.py",
            "targetgen/source_scalar.py",
            "targetgen/exact_arithmetic.py",
            "runtime/exact_readout.py",
            "targetgen/exact_quantized_reference.py",
            "targetgen/recipe_numerics.py",
            "targetgen/readout_facet.py",
            "xdsl_dialects/lowering/compute_groups.py",
            "xdsl_dialects/lowering/window_sum.py",
            "targetgen/_recipe_integer.py",
            "targetgen/_capture_integer_agreement.py",
            "llvmlower/contraction_quantization.py",
            "llvmlower/torchao_affine.py",
            "llvmlower/passes_xdsl.py",
            "llvmlower/exact_quantize.py",
            "targetgen/core_aten_semantics.py",
            "targetgen/_m2m_capture_worker.py",
            "targetgen/_capture_result_contract.py",
            "targetgen/capsule_source.py",
            "llvmlower/semantic_io.py",
            "runtime/semantic_readback.py",
            "targetgen/core_aten_batch_grade.py",
            "targetgen/core_aten_device.py",
            "targetgen/core_aten_provenance.py",
            "llvmlower/result_buffers.py",
            "llvmlower/result_layout.py",
            "llvmlower/input_layout.py",
            "llvmlower/pipeline.py",
            "llvmlower/c_runtime.py",
            "runtime/backends/spike_model.py",
        )
    ]
    sources.append(repo_root() / "merlin/runtime/baremetal/spike/model_main.c")
    from merlin.common.paths import contract_dir

    sources.extend(contract_dir() / name for name in ("quant_schema.json", "quant_schema_source.json"))
    if execution_provider is not None:
        sources.append(execution_provider)
    options = runner_options or {}
    artifacts = {}
    simulator = Path(options.get("spike_binary") or spike_path())
    if simulator.is_file():
        artifacts["spike"] = simulator
    if options.get("extlib"):
        artifacts["extension"] = Path(options["extlib"])
    if target:
        from merlin.targetgen.rtl.facts import ensure_facts

        facts = ensure_facts(target, explicit=rtl_facts)
        sources.append(facts)
        artifacts["rtl_facts"] = facts
    if directory:
        for name in (
            "model.mlir",
            "inputs.npz",
            "semantic_io.json",
            "spike-build/model.elf",
            "model.before_w8a8.mlir",
            "w8a8-request.json",
            "w8a8-reference.json",
            "w8a8-dispatch.json",
            "spec-int8-admission.json",
            "model.before_spec_int8.mlir",
            "spec-int8-request.json",
            "spec-int8-reference.json",
        ):
            path = directory / name
            if path.is_file():
                artifacts[name] = path
    if directory and (directory / "w8a8-dispatch.json").is_file():
        import json

        dispatch = json.loads((directory / "w8a8-dispatch.json").read_text())
        for i, site in enumerate(dispatch):
            sources.append(Path(site["software_spec_path"]))
            artifacts[f"w8a8_target_ir_{i}"] = Path(site["target_ir_path"])
            artifacts[f"w8a8_device_object_{i}"] = Path(site["object_path"])
    return provenance.record(
        pins=pins,
        sources=sources,
        artifacts=artifacts,
        extra={
            "target": target,
            "substrate": "spike_functional",
            "model2mlir_commit": _git_revision(m2m_root()),
            "oot_commit": _git_revision(package) if package else None,
        },
    )
