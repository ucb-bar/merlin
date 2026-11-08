"""Stage an exact saved-export conversion with immutable external parameters.

The capture's framework golden is capture data only. Compiler intake uses the
checked MLIR, argument signature, and parameter archive, never that golden.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ExternalCapsuleParameters:
    source: Any
    parameters: Any
    metadata: dict
    source_operations: tuple[str, ...]


def _raw_bytes(value) -> bytes:
    import torch

    if not isinstance(value, torch.Tensor):
        raise ValueError("saved export immutable arguments must be tensors")
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _immutable_arguments(program):
    mutated = set(program.graph_signature.buffers_to_mutate.values())
    if mutated:
        raise ValueError("external immutable parameters cannot replace mutated export buffers")
    values, readonly = {}, []
    for index, spec in enumerate(program.graph_signature.input_specs):
        kind = spec.kind.name
        if kind in {"PARAMETER", "BUFFER"}:
            value = program.state_dict[spec.target]
            if kind == "BUFFER":
                readonly.append(index)
        elif kind == "CONSTANT_TENSOR":
            value = program.constants[spec.target]
        elif kind == "USER_INPUT":
            continue
        else:
            raise ValueError(f"unsupported saved export argument kind: {kind}")
        values[index] = (spec.target, kind, value)
    return values, readonly


def prepare_external_capsule_parameters(
    program,
    conversion_result,
    inputs: tuple,
    *,
    source_path: str | Path,
    capture_dir: str | Path,
    staged_dir: str | Path,
    precision_policy: Mapping[str, Any],
    target_identity: str,
    source_operation_translators: Mapping[str, Callable],
    quantization=None,
) -> ExternalCapsuleParameters:
    """Return a byte-checked parameter source using existing capture/staging APIs.

    The first conversion must have used
    ``weights_path=capture_dir / 'weights.safetensors'``. Its exact MLIR and
    trace are reused; this helper never re-exports, quantizes, or casts weights.
    ``staged_dir`` must be fresh and separate from ``capture_dir``.
    """
    from xdsl.dialects.builtin import StringAttr
    from m2m.capture.bundle import write_bundle
    from merlin.compile.c_parameters import bind_application_parameters, parameter_source_identity
    from merlin.frontends.compile_inputs import (
        load_compile_inputs, load_lifted_argument_bytes, stage_compile_inputs,
    )
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

    capture_dir, staged_dir = Path(capture_dir).absolute(), Path(staged_dir).absolute()
    source_path = Path(source_path).absolute()
    result = conversion_result
    if (not result.ok or result.path_taken != "fx_importer" or result.module is None
            or result.exported_program is not program or result.capture_trace is None):
        raise ValueError("external parameters require the exact traced conversion of the supplied saved export")
    original_text = result.mlir_text
    original_hash = sha256(original_text.encode()).hexdigest()
    weight_reference = result.module.attributes.get("prov.weights_file")
    if (not isinstance(weight_reference, StringAttr)
            or Path(weight_reference.data).resolve() != (capture_dir / "weights.safetensors").resolve()):
        raise ValueError("first conversion must externalize weights into the exact capture directory")
    if (not source_path.is_file() or staged_dir.exists() or staged_dir.is_symlink()
            or staged_dir.is_relative_to(capture_dir) or capture_dir.is_relative_to(staged_dir)):
        raise ValueError("saved source must exist and capture/staged directories must be distinct and fresh")
    policy = dict(precision_policy)
    if policy.get("source_sha256") != original_hash:
        raise ValueError("external parameter precision policy differs from the exact captured MLIR")
    if result.capture_trace.get("mlir", {}).get("sha256") != original_hash:
        raise ValueError("external parameter conversion trace differs from its exact MLIR")
    values, readonly = _immutable_arguments(program)

    # The selected policy is a separately reviewed compiler input, outside the
    # capture directory as required by stage_compile_inputs.
    policy_path = staged_dir.with_name(staged_dir.name + ".precision-policy.json")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_path.open("x") as stream:
        stream.write(json.dumps(policy, sort_keys=True, indent=2) + "\n")
    model = program.module()
    model.eval = lambda: model
    model.train = lambda mode=True: model
    write_bundle(
        model, tuple(inputs), capture_dir, exported_program=program,
        conversion_result=result, capture_trace=True, capture_regions=False,
        source_path=source_path, quant=quantization,
        quantization_preapplied=quantization is not None,
        metadata={"capture_quantization": "already_materialized", "read_only_buffer_arguments": readonly},
    )
    if result.mlir_text != original_text or (capture_dir / "model.mlir").read_text() != original_text:
        raise ValueError("external parameter capture changed the selected source MLIR")
    stage_compile_inputs(capture_dir, staged_dir, precision_policy=policy_path)
    staged = load_compile_inputs(staged_dir)
    if staged.original_source_sha256 != original_hash or staged.precision_policy != policy:
        raise ValueError("external parameter stage changed source or selected precision policy")
    if any(row["kind"] == "buffer" for row in staged.argument_bindings):
        raise ValueError("external parameter stage retains a mutable buffer without a state interface")

    # This materializes bytes for validation only; parameter bytes never become
    # ConstantBindings in the semantic graph or literals in generated C.
    payloads = load_lifted_argument_bytes(staged_dir, staged)
    if set(payloads) != set(values):
        raise ValueError("staged immutable argument indices differ from the saved export")
    validated = []
    for index, (name, kind, value) in values.items():
        expected = _raw_bytes(value)
        if payloads[index] != expected:
            raise ValueError(f"staged immutable bytes differ from saved export argument {index}: {name}")
        row = staged.argument_bindings[index]
        if kind in {"PARAMETER", "BUFFER"} and row["kind"] != "param":
            raise ValueError("saved immutable model tensor was not preserved as an external parameter")
        validated.append({"argument_index": index, "source_target": name, "source_kind": kind,
                          "staged_kind": row["kind"], "shape": list(value.shape),
                          "dtype": str(value.dtype).removeprefix("torch."),
                          "bytes": len(expected), "sha256": sha256(expected).hexdigest()})
    constants = {index: data for index, data in payloads.items()
                 if staged.argument_bindings[index]["kind"] == "constant"}
    del payloads
    compiler_staged, compiler_view = staged, None
    if result.quantization_manifest is not None:
        from atlas_mlir_native.capture import compiler_input_view
        compiler_staged, compiler_view = compiler_input_view(staged, result.quantization_manifest)
        (capture_dir.parent / "compiler-input-view.json").write_text(json.dumps(compiler_view, indent=2) + "\n")
    translated = translate_linalg_text(
        compiler_staged.program, entry=staged.entry, target_identity=target_identity,
        lowering_policy="diagnostic", argument_constants=constants,
        source_operation_translators=source_operation_translators,
    )
    source = replace(translated.request, source_identity=parameter_source_identity(translated.request.source_identity, compiler_staged))
    parameters = bind_application_parameters(staged_dir, compiler_staged, source)
    if parameters is None:
        raise ValueError("external parameter mode requires immutable model parameter tensors")
    parameters.check(source)
    if any(source.node(key).effect != "input" for key in parameters.node_ids):
        raise ValueError("external parameter bytes became semantic constants")
    metadata = {
        "schema": "atlas.external_capsule_parameters.v1",
        "source_model_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "original_mlir_sha256": original_hash,
        "staged_program_sha256": staged.program_sha256,
        "compiler_program_sha256": compiler_staged.program_sha256,
        "compiler_input_view": compiler_view,
        "capture_receipt_sha256": sha256((capture_dir / "capture_receipt.json").read_bytes()).hexdigest(),
        "compile_inputs_sha256": sha256((staged_dir / "compile-inputs.json").read_bytes()).hexdigest(),
        "archive_sha256": parameters.archive_sha256,
        "payload_sha256": parameters.payload_sha256,
        "payload_bytes": parameters.payload_bytes,
        "source_request_digest": source.digest(),
        "source_identity": source.source_identity,
        "validated_arguments": validated,
        "parameter_node_ids": list(parameters.node_ids),
        "golden_role": "capture_data_only; excluded_from_compiler_intake",
        "weight_conversion": "none; exact saved-export tensor bytes",
    }
    return ExternalCapsuleParameters(source, parameters, metadata, tuple(translated.source_operations))
