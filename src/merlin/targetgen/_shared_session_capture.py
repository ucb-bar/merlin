"""Explicit shared-model dynamic capture with one transform and sealed stage bindings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureServices:
    """Functions from the exact selected worker, independent of its module registration."""

    abi_version: int
    float_reference: Callable[..., dict]
    quant_for: Callable[..., Any]
    scalars: Callable[..., dict]
    input_abi: Callable[..., tuple]
    output_abi: Callable[..., tuple]


def exported_integer_mm_count(module) -> int:
    """Count proven i8×i8→i32 contractions in the emitted, unnormalized MLIR."""
    from xdsl.dialects.builtin import IntegerType, TensorType

    def width(value) -> int | None:
        typ = getattr(value, "type", None)
        if not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType):
            return None
        return int(typ.element_type.width.data)

    if module is None:
        return 0
    count = 0
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.operands) < 3 or len(op.results) != 1:
            continue
        prov = op.attributes.get("prov.op")
        if (
            str(getattr(prov, "data", "")) == "int_matmul"
            and [width(value) for value in op.operands[:2]] == [8, 8]
            and width(op.results[0]) == 32
        ):
            count += 1
    return count


def framework_catalog(torch) -> dict:
    """Observe the registry in this exact capture interpreter, never another build."""
    owner = Path(__file__).with_name("_aten_opset_worker.py")
    try:
        spec = importlib.util.spec_from_file_location("_capture_opset", owner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.core_opset()
    except Exception as exc:  # noqa: BLE001 -- unavailable is not an empty registry
        return {
            "schema": "merlin.pytorch_opset.v1",
            "status": "not_available",
            "torch": str(torch.__version__),
            "n_all_aten": None,
            "reason": f"capture framework catalog unavailable: {type(exc).__name__}: {exc}",
        }


def materialize(
    model, args, out: Path, *, determinism: dict, dependencies: list, torch, services: CaptureServices
) -> int:
    """Transform one explicitly shared source model once, retaining every stage binding."""
    if (
        args.dtype != "int8"
        or args.scheme != "int8_dyn_act_int8_weight"
        or not args.materialize_bundle
        or args.recipe
        or args.already_quantized
        or args.stage_fp32
        or args.quantize_activation_contractions
        or args.integer_nonlinear
    ):
        raise ValueError("shared dynamic capture requires only the explicit dynamic W8A8 scheme")
    import numpy as np
    import yaml
    from m2m.capture.bundle import _numpy_safe, write_quantized_multi_program_bundle
    from m2m.capture.bundle_integrity import write_bundle_integrity
    from m2m.capture.external_runtime import external_runtime_shared_model
    from m2m.capture.provenance import write_capture_receipt
    from m2m.capture.torchao_pipeline import apply_quantization
    from m2m.capture.trace import capture_frontend_snapshot
    from m2m.coverage import opaque_report

    shared = external_runtime_shared_model(model)
    original_session = shared.make_session(shared.model)
    originals, original_outputs = {}, {}
    with torch.no_grad():
        for program in original_session.programs:
            program.module.eval()
            originals[program.name] = capture_frontend_snapshot(program.module, program.inputs, stage="original")
            original_outputs[program.name] = services.float_reference(program.module, program.inputs, torch)
    if set(originals) != set(shared.shared_programs) or any(
        value.get("status") != "complete" for value in originals.values()
    ):
        raise ValueError("shared dynamic source snapshots must completely cover every program")
    quant = services.quant_for(args.dtype, args.scheme)
    policy = {
        "scheme": quant.scheme,
        "transform_api": "m2m.capture.torchao_pipeline.apply_quantization",
        "shared_transform_calls": 0,
        "quantization_preapplied_to_stages": True,
        "scope": "source-declared shared model; remaining floating operators are unchanged",
    }
    metadata = {
        "capture_abi_version": services.abi_version,
        "dtype": args.dtype,
        "scheme": quant.scheme,
        "torch_seed": int(args.seed),
        "determinism": determinism,
        "loader_dependency_sources": dependencies,
        "loader_provenance": services.scalars(dict(original_session.metadata.get("provenance") or {})),
        "loader_provenance_status": "declared",
        "loader_provenance_error": None,
        "loader_paper_ready": original_session.metadata.get("paper_ready"),
    }
    selected = {}

    def transform(module):
        if policy["shared_transform_calls"]:
            raise ValueError("shared quantization transform was invoked more than once")
        policy["shared_transform_calls"] += 1
        return apply_quantization(module, quant)

    def make_session(module):
        session = shared.make_session(module)
        selected["session"] = session
        with torch.no_grad():
            selected["abis"] = {}
            for program in session.programs:
                program.module.eval()
                _, input_abi = services.input_abi(program.inputs)
                _, output_abi = services.output_abi(program.module(*program.inputs))
                selected["abis"][program.name] = {"input_abi": input_abi, "output_abi": output_abi}
        return session

    summary = write_quantized_multi_program_bundle(
        shared.model,
        transform,
        make_session,
        out,
        shared_programs=shared.shared_programs,
        quantization=policy,
        quant=quant,
        capture_trace=True,
        source_path=Path(args.loader),
        metadata=metadata,
        original_frontend_by_program=originals,
    )
    if policy["shared_transform_calls"] != 1:
        raise ValueError("shared dynamic capture did not perform exactly one transform")
    catalog = framework_catalog(torch)
    catalog_bytes = (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode()
    stages = []
    session = selected["session"]
    # These are producer-owned metadata additions. Re-seal every modified stage,
    # then rebind the root to those exact receipts; never retain stale digests.
    for program in session.programs:
        stage = out / "stages" / program.name
        reference = original_outputs[program.name]
        reference_path = stage / "original-source-outputs.npz"
        np.savez(reference_path, **{f"out{i}": _numpy_safe(value) for i, value in enumerate(reference["leaves"])})
        (stage / "pytorch-opset.json").write_bytes(catalog_bytes)
        trace = json.loads((stage / "frontend-trace.json").read_bytes())
        opaque = opaque_report((stage / "model.mlir").read_text())
        from merlin.frontends.linalg_mlir import parse_mlir_file

        integer_roots = exported_integer_mm_count(parse_mlir_file(stage / "model.mlir"))
        precision = trace.get("precision") or {}
        projected = precision.get("status") == "projected" or bool(precision.get("projections"))
        meta_path = stage / "meta.json"
        meta = json.loads(meta_path.read_bytes())
        meta.update(selected["abis"][program.name])
        meta.update(
            ok=not opaque and not projected and trace.get("status") == "complete",
            opaque=sum(opaque.values()),
            opaque_detail=opaque,
            precision_realization=precision,
            captured_integer_mm_roots=integer_roots,
            original_source_outputs={
                "path": reference_path.name,
                "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "output_abi": reference["output_abi"],
                "role": "source_quality_only",
            },
            framework_catalog={
                "path": "pytorch-opset.json",
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "torch": catalog.get("torch"),
                "status": catalog.get("status", "unknown"),
            },
        )
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        receipt = write_capture_receipt(stage, source_path=Path(args.loader))
        stages.append(
            {
                "name": program.name,
                "ok": meta["ok"],
                "opaque": meta["opaque"],
                "integer_mm_roots": integer_roots,
                "trace_status": trace.get("status", "unknown"),
                "trace_blockers": trace.get("blockers", []),
                "receipt_sha256": hashlib.sha256((stage / "capture_receipt.json").read_bytes()).hexdigest(),
                "materialized_abi": receipt["materialized_abi"],
            }
        )
    binding_path = out / "quantized-session-binding.json"
    binding = json.loads(binding_path.read_bytes())
    binding["stage_receipts"] = {stage["name"]: stage["receipt_sha256"] for stage in stages}
    binding_path.write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n")
    contract_path = out / "session_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    contract["quantized_session_binding"]["sha256"] = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    report = {
        "schema": "merlin.shared_dynamic_session_capture.v1",
        "programs": stages,
        "determinism": determinism,
        "quantization": policy,
        "session_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "shared_binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "qualification": "capture only; original FP32 observations are quality references, not quantized goldens",
    }
    (out / "session-receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    write_bundle_integrity(out)
    ok = all(stage["ok"] for stage in stages) and any(stage["integer_mm_roots"] for stage in stages)
    print(
        "__M2M_CAPTURE__ "
        + json.dumps(
            {
                "ok": ok,
                "opaque": sum(stage["opaque"] for stage in stages),
                "session": summary["session_kind"],
                "programs": len(stages),
            }
        )
    )
    return 0 if ok else 3
