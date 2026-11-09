"""Compile an unchanged graph capsule through model2MLIR and Merlin APIs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from safetensors.torch import load_file


def raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, default=Path("atlas-mlir"))
    parser.add_argument("--register-module", default="atlas_fx_ops")
    parser.add_argument("--translate-only", action="store_true")
    parser.add_argument("--allocation-timeout-ms", type=int, default=1000)
    parser.add_argument("--merge-adjacent-device-regions", action="store_true",
                        help="Opt in to Atlas's checked adjacent device-region alternatives")
    parser.add_argument("--io-memory-bytes", type=int, default=None)
    parser.add_argument("--application-memory-bytes", type=int, default=1 << 20)
    parser.add_argument("--external-parameters", action="store_true")
    parser.add_argument("--adapter-export", type=Path, help="authenticate saved native capsule through atlas_fp8 entry")
    parser.add_argument("--expected-contract-sha256", default="", help="contract SHA-256 from the frozen source receipt")
    parser.add_argument("--expected-policy-sha256", default="", help="policy SHA-256 from the frozen source receipt")
    parser.add_argument("--softmax-mask-policy", choices=("bf16-minfinite-zero-bounded32-nonempty-v1",
        "bf16-minfinite-zero-rowspan80-nonempty-v1"), default="bf16-minfinite-zero-bounded32-nonempty-v1")
    from precision_options import add_reduction_policy_argument

    add_reduction_policy_argument(parser)
    args = parser.parse_args()
    if args.allocation_timeout_ms <= 0:
        parser.error("allocation timeout must be positive")
    if args.adapter_export and not (args.expected_contract_sha256 and args.expected_policy_sha256):
        parser.error("--adapter-export requires both expected source SHA-256 values")
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

    from merlin.semantic_compiler.buffer_allocation import allocation_wall_budget

    with codegen_options(CodegenOptions(io_memory_bytes=args.io_memory_bytes)), allocation_wall_budget(args.allocation_timeout_ms):
        compile_capsule(args)


def compile_capsule(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from m2m import convert
    from atlas_mlir_native.capture import load_export, prepare_model, normalize_capture
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import AtlasNativeBinding, TARGET_IDENTITY
    from merlin.compile.semantic_application import compile_semantic_application
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path

    capsule, output, atlas = args.capsule.resolve(), args.output.resolve(), args.atlas_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tensors = load_file(str(capsule / "tensors.safetensors"))
    provenance = json.loads((capsule / "source.json").read_text())
    if sha256((capsule / "model.pt2").read_bytes()).hexdigest() != provenance["model_sha256"]:
        raise AssertionError("saved capsule model changed after source extraction")
    for key, metadata in provenance["tensors"].items():
        if sha256(raw_bytes(tensors[key])).hexdigest() != metadata["sha256"]:
            raise AssertionError(f"capsule source tensor changed: {key}")
    program = load_export(capsule / "model.pt2", ops_module=args.register_module)
    if "parameters" in provenance:
        if set(program.state_dict) != set(provenance["parameters"]):
            raise AssertionError("capsule lifted parameter inventory changed")
        for key, metadata in provenance["parameters"].items():
            tensor = program.state_dict[key]
            if (str(tensor.dtype) != metadata["dtype"] or list(tensor.shape) != metadata["shape"]
                    or sha256(raw_bytes(tensor)).hexdigest() != metadata["sha256"]):
                raise AssertionError(f"authored capsule parameter changed: {key}")
    # The already-quantized contract cannot materialize floating source weights
    # through the ordinary contraction policy's FP8 packing path.
    selected_host_bf16_policy = provenance["kind"] == "head"
    immutable_by_node = {}
    for spec in program.graph_signature.input_specs:
        if spec.kind.name in {"PARAMETER", "BUFFER"}:
            immutable_by_node[spec.arg.name] = (spec.target, program.state_dict[spec.target])
        elif spec.kind.name == "CONSTANT_TENSOR":
            immutable_by_node[spec.arg.name] = (spec.target, program.constants[spec.target])
    contractions = {"aten.linear.default": 1, "aten.mm.default": 1,
                    "aten.matmul.default": 1, "aten.addmm.default": 2}
    for node in program.graph.nodes:
        operand_index = contractions.get(str(node.target)) if node.op == "call_function" else None
        if operand_index is None or len(node.args) <= operand_index:
            continue
        operand = node.args[operand_index]
        source_weight = immutable_by_node.get(getattr(operand, "name", ""))
        if source_weight is not None and source_weight[1].is_floating_point() and not selected_host_bf16_policy:
            name, weight = source_weight
            refusal = {"status": "REFUSED_UNQUANTIZED_IMMUTABLE_CONTRACTION",
                       "source_model_sha256": provenance["model_sha256"],
                       "kind": provenance["kind"], "source_node": node.name,
                       "source_operation": str(node.target), "parameter": name,
                       "dtype": str(weight.dtype), "shape": list(weight.shape),
                       "sha256": sha256(raw_bytes(weight)).hexdigest(),
                       "reason": "ordinary contraction packs floating weights to FP8; source immutable weight conversion is forbidden",
                       "weight_conversion_performed": False}
            (output / "refusal.json").write_text(json.dumps(refusal, indent=2) + "\n")
            raise ValueError(refusal["reason"])
    if "input_keys" in provenance:
        input_tensors = tuple(tensors[key] for key in provenance["input_keys"])
    else:
        input_tensors = ((tensors["activation"],) if provenance["kind"] == "qlinear" else
                         tuple(tensors[key] for key in ("lhs_codes", "lhs_exp", "rhs_codes", "rhs_exp")))
    conversion_options = {}
    if args.external_parameters:
        (output / "capture").mkdir()
        conversion_options["weights_path"] = str(output / "capture/weights.safetensors")
    if getattr(args, "adapter_export", None) is not None:
        export = args.adapter_export.resolve(strict=True)
        program._atlas_saved_program_path = capsule / "model.pt2"
        program._atlas_saved_program_sha256 = provenance["model_sha256"]
        program, _ = prepare_model(program, input_tensors, export / "manifest.json", export / "policy.json", export_prefix="model.",
                                  expected_contract_sha256=args.expected_contract_sha256,
                                  expected_policy_sha256=args.expected_policy_sha256)
        manifest = program._m2m_external_quantization_manifest
        from m2m.capture.external_quantization import ExternalQuantizationConfig
        from m2m.capture.trace import snapshot_exported_program
        conversion_options.update(
            quantization=ExternalQuantizationConfig("atlas_fp8", export / "manifest.json", export / "policy.json",
                                                   args.expected_contract_sha256, args.expected_policy_sha256),
            quantization_preapplied=True,
            original_frontend_snapshot=snapshot_exported_program(program, stage="original"),
            normalize=normalize_capture,
        )
        (output / "adapter-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    result = convert(program, input_tensors, backend="fx_importer", capture_trace=True, decompose=False,
                     **conversion_options)
    if not result.ok or result.path_taken != "fx_importer":
        raise AssertionError(result.diagnostics)
    if "func.call" in result.mlir_text:
        raise AssertionError("opaque call in imported capsule")
    (output / "import.mlir").write_text(result.mlir_text)
    (output / "capture-trace.json").write_text(json.dumps(result.capture_trace, indent=2) + "\n")
    constants = {}
    for index, spec in enumerate(program.graph_signature.input_specs):
        if spec.kind.name in {"BUFFER", "PARAMETER"}:
            constants[index] = raw_bytes(program.state_dict[spec.target])
        elif spec.kind.name == "CONSTANT_TENSOR":
            constants[index] = raw_bytes(program.constants[spec.target])
    source_hash = sha256(result.mlir_text.encode()).hexdigest()
    from precision_options import build_precision_policy

    policy = build_precision_policy(provenance["kind"], source_hash,
                                    reduction_policy=args.reduction_policy,
                                    softmax_mask_policy=args.softmax_mask_policy)
    (output / "precision.json").write_text(json.dumps(policy, indent=2) + "\n")
    adapter = AtlasApplicationAdapter(policy, original_source_sha256=source_hash)
    if getattr(args, "merge_adjacent_device_regions", False):
        from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
        adapter = AdjacentDeviceAdapter(adapter)
    parameters, parameter_metadata = None, None
    if args.external_parameters:
        from external_capsule_parameters import prepare_external_capsule_parameters

        prepared = prepare_external_capsule_parameters(program, result, input_tensors,
            source_path=capsule / "model.pt2", capture_dir=output / "capture", staged_dir=output / "staged",
            precision_policy=policy, target_identity=TARGET_IDENTITY,
            source_operation_translators=adapter.source_operation_translators,
            quantization=conversion_options.get("quantization"))
        source, parameters, parameter_metadata = prepared.source, prepared.parameters, prepared.metadata
        source_operations = prepared.source_operations
    else:
        translated = translate_linalg_text(result.mlir_text, entry="forward", target_identity=TARGET_IDENTITY,
                                          lowering_policy="diagnostic", argument_constants=constants,
                                          source_operation_translators=adapter.source_operation_translators)
        source, source_operations = translated.request, translated.source_operations
    (output / "source-request.json").write_text(json.dumps(source.record(), indent=2) + "\n")
    bindings = {int(item.node_id.removeprefix("arg")): bytes.fromhex(item.data_hex)
                for item in source.constants if item.node_id.startswith("arg")}
    if not args.external_parameters and bindings != constants:
        raise AssertionError("immutable operand bytes changed during Merlin translation")
    print(json.dumps({"stage": "MERLIN_TRANSLATED", "nodes": len(source.nodes),
                      "constants": {key: sha256(value).hexdigest() for key, value in constants.items()}}), flush=True)
    if args.translate_only:
        plan = adapter.plan(source)
        (output / "selected-plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
        print(json.dumps({"stage": "PLANNED", "regions": len(plan.selected)}), flush=True)
        return
    snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=output / "snapshot",
                                    crate=packaged_egg_bridge_path(), cargo_target_dir=output.parent / "e2e-cargo",
                                    source_revision="unchanged-fx-option-b")
    limits = SearchLimits()
    manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot, target_source=atlas,
                                           destination=output / "application", limits=limits,
                                           name="capsule", capacity=args.application_memory_bytes, parameters=parameters,
                                           allow_host_only=selected_host_bf16_policy)
    interface = build_host_library(output / "application", manifest)
    data = {}
    for index, tensor in enumerate(input_tensors):
        if tensor.dtype == torch.bfloat16:
            value = tensor.contiguous().view(torch.uint16).numpy()
        else:
            value = tensor.contiguous().numpy()
        data[f"input_{index}"] = value
    for key, tensor in tensors.items():
        data[key] = (tensor.contiguous().view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16
                     else tensor.contiguous().numpy())
    np.savez(output / "tensors.npz", **data)
    record = {"status": "COMPILED", "kind": provenance["kind"], "capsule": str(capsule),
              "source_model_sha256": provenance["model_sha256"],
              "logical_source_tensors": provenance["tensors"],
              "source_request_digest": source.digest(), "source_operations": list(source_operations),
              "precision_policy_sha256": sha256((output / "precision.json").read_bytes()).hexdigest(),
              "constant_bytes_identical": True, "interface": interface,
              "io_memory_bytes": args.io_memory_bytes, "application_memory_bytes": args.application_memory_bytes,
              "allocation_timeout_ms": args.allocation_timeout_ms,
              "reduction_policy": policy.get("reduction"),
              "search_limits": asdict(limits),
              "merge_adjacent_device_regions": bool(getattr(args, "merge_adjacent_device_regions", False)),
              "segments": manifest["segments"], "regions": manifest["regions"],
              "external_parameters": parameter_metadata,
              "source_input_order": [node.id for node in source.nodes if node.effect == "input"]}
    (output / "compiled.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": "COMPILED", "segments": len(manifest["segments"]),
                      "regions": len(manifest["regions"])}), flush=True)


def build_host_library(application, manifest, *, package=None):
    """Build the example runtime shim for a standard capsule application.

    This packages emitted C only; it never captures or compiles source IR.
    The application must use the explicit ``capsule`` C namespace.
    """
    if manifest["application_name"] != "capsule":
        raise ValueError("example runtime packaging requires application-name capsule")
    package = application / "c-package" if package is None else package
    interface = manifest["application_interface"]
    header = interface["header"]
    shim = f'''#include "{header}"
static atlas_host host;
static capsule_context context;
int bind(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get) {{
  if(atlas_host_init(&host,(atlas_host_map){{0x20000u,0x40000u,131072u}},
      (atlas_host_mmio){{read,write,0}},(atlas_host_memory){{1,0,0}})) return -1;
  return capsule_bind(&context,&host,(atlas_tile_memory_io){{put,get,0}},20u);
}}
int run(const void *const *inputs,void *const *outputs) {{ return capsule_run(&context,inputs,outputs); }}
'''
    if interface.get("model_parameters"):
        shim += "int bind_parameters(const void *data,size_t bytes) { return capsule_bind_parameters(&context,data,bytes); }\n"
    if not interface.get("platform_binding"):
        parameter_field = "static capsule_parameters parameters;" if interface.get("model_parameters") else ""
        parameter_argument = "&parameters," if interface.get("model_parameters") else ""
        shim = f'''#include "{header}"
static capsule_driver_state state;
static _Alignas(64) unsigned char workspace[{interface["workspace_bytes"]}];
static void *contexts[{len(manifest["regions"])}];
static capsule_driver_contexts dispatch={{contexts,{len(manifest["regions"])}}};
{parameter_field}
int run(const void *const *inputs,void *const *outputs) {{
  return capsule_invoke(&state,workspace,sizeof(workspace),{parameter_argument}inputs,outputs,&dispatch);
}}
'''
        if interface.get("model_parameters"):
            shim += "int bind_parameters(const void *data,size_t bytes) { return capsule_parameters_bind(&parameters,data,bytes); }\n"
    (package / "shim.c").write_text(shim)
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-frounding-math", "-ffp-contract=off", "-shared", "-fPIC",
                    interface["source"], "shim.c", "-o", "capsule.so", "-lm"], cwd=package, check=True)
    return interface


if __name__ == "__main__":
    main()
