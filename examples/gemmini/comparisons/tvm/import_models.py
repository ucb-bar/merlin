#!/usr/bin/env python3
"""Inventory full-model ONNX/Relax intake without claiming compiled correctness.

Native stages supply subsequent stage inputs. Every stage/static signature is
exported through the public StageCompiler import-only path. No TVM optimization,
LLVM build, compiled inference, numerical comparison or hardware run is performed.
"""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from verify_resnet import file_identity, safe_path
from verify_session import (StageCompiler, array, describe_session, endpoint,
                            factory_export_options, load_module, load_protocol,
                            offline, outputs, parameter_digest, tensor_identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet50", "tinyllama", "smolvla"), required=True)
    parser.add_argument("--tvm-source", type=Path, required=True)
    parser.add_argument("--tvm-build", type=Path, required=True)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--factory-arguments", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bf16-fused-export", action="store_true")
    parser.add_argument("--opset", type=int, default=22)
    parser.add_argument("--save-programs", action="store_true", help="Save semantic Relax JSON per static stage for tuning/compilation")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = safe_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"model": args.model, "status": "running", "scope": "full_model_import_only",
              "numerical_comparison_performed": False, "compiled_execution": False,
              "device_execution": False, "paper_quality_qualified": False,
              "native_execution_role": "supply_actual_inputs_and_state_for_structural_import",
              "invocation": [sys.executable, *sys.argv], "stages": [], "native_calls": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        os.environ.update(TVM_LIBRARY_PATH=str(safe_path(args.tvm_build)), TVM_FFI="ctypes")
        sys.path.insert(0, str(safe_path(args.tvm_source) / "python"))
        import numpy as np
        import onnx
        import torch
        import tvm
        from tvm._ffi.base import _LIB
        if not safe_path(tvm.__file__).is_relative_to(safe_path(args.tvm_source) / "python") or safe_path(_LIB._name).parent != safe_path(args.tvm_build):
            raise ValueError("Loaded TVM differs from the explicitly selected source/build")
        torch.set_num_threads(1)
        torch.manual_seed(0)
        report.update(versions={"torch": torch.__version__, "onnx": onnx.__version__, "tvm": tvm.__version__, "numpy": np.__version__},
                      compiler_library=file_identity(_LIB._name, safe_path),
                      driver_source=file_identity(__file__, safe_path),
                      fused_export=args.bf16_fused_export)
        protocol = load_protocol(safe_path(args.model2mlir_root))
        with offline():
            if args.model == "resnet50":
                from torchvision.models import resnet50
                checkpoint = safe_path(args.checkpoint)
                if checkpoint.stat().st_size != 102540417:
                    raise ValueError("Expected the complete official ResNet50 V2 checkpoint")
                identity = file_identity(checkpoint, safe_path)
                if identity["sha256"] != "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca":
                    raise ValueError("ResNet checkpoint differs from the pinned official weights")
                model = resnet50(weights=None).eval().requires_grad_(False)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
                inputs = (torch.randn(1, 3, 224, 224),)
                report.update(checkpoint=identity, full_model=True, precision="original_FP32",
                              input_role="seeded_synthetic_structural_input_not_accuracy_data")
                session = protocol.external_runtime_session(model, inputs)
                export_options = None
            else:
                path = safe_path(Path(__file__).with_name(args.model + "_session.py"))
                factory = load_module("import_only_factory", path)
                arguments_path = safe_path(args.factory_arguments)
                arguments = json.loads(arguments_path.read_text())
                report.update(factory_source=file_identity(path, safe_path),
                              factory_arguments=file_identity(arguments_path, safe_path))
                model, inputs = factory.get_model_and_inputs(**arguments)
                selected = factory.get_verification_artifacts(model, inputs)
                report["selected_artifacts"] = [file_identity(p, safe_path) for p in selected]
                metadata_hook = getattr(factory, "get_session_spec", None)
                metadata = metadata_hook(model, inputs) if callable(metadata_hook) else None
                session = protocol.external_runtime_session(model, tuple(inputs), session=metadata)
                export_options = factory_export_options(factory, torch, model)
            report["session"] = describe_session(session, np, torch)[0]
            programs = {p.name: p for p in session.programs}
            state = {endpoint(b.target): b.initial.detach().clone() for b in session.input_bindings}
            original = {name: parameter_digest(p.module, torch) for name, p in programs.items()}
            compiler = StageCompiler(out, "baseline", np, torch, onnx, tvm, exporter="dynamo",
                                     opset=args.opset, bf16_fused_export=args.bf16_fused_export,
                                     export_options=export_options, save_programs=args.save_programs)
            seen = set()

            def call(invocation, observation):
                program = programs[invocation.program]
                if observation is not None:
                    for stream in session.streams:
                        if stream.target.program == program.name:
                            state[endpoint(stream.target)] = stream.values[observation].detach().clone()
                tensors = tuple(state[(program.name, i)] for i in range(len(program.inputs)))
                numpy_inputs = tuple(array(value, np, torch) for value in tensors)
                with torch.no_grad():
                    native = outputs(program.module(*tensors), np, torch)
                signature = (program.name, tuple((str(v.dtype), tuple(v.shape)) for v in numpy_inputs))
                if signature not in seen:
                    report["stage_in_progress"] = program.name
                    save()
                    compiler(program, numpy_inputs, len(native), import_only=True)
                    report["stages"].append(compiler.records[-1])
                    compiler.imported_modules.clear()
                    seen.add(signature)
                    save()
                    gc.collect()
                for route in session.routes:
                    if route.source.program == program.name:
                        from verify_session import torch_array
                        state[endpoint(route.target)] = torch_array(native[route.source.output_index], np, torch).clone()
                report["native_calls"].append({"program": program.name, "observation": observation,
                                                "output_count": len(native), "output_identities": [tensor_identity(v) for v in native]})
                save()

            for invocation in session.execution_schedule:
                if invocation.cadence == "once_before_observations":
                    for _ in range(invocation.repeats):
                        call(invocation, None)
            for observation in range(session.observations):
                for invocation in session.execution_schedule:
                    if invocation.cadence == "per_observation":
                        call(invocation, observation)
            for invocation in session.execution_schedule:
                if invocation.cadence == "once_after_observations":
                    for _ in range(invocation.repeats):
                        call(invocation, None)
            for name, program in programs.items():
                if parameter_digest(program.module, torch) != original[name]:
                    raise AssertionError("Native execution or export changed source parameters")
            counts = Counter()
            for record in report["stages"]:
                counts.update(record["onnx_inventory"]["operator_counts"])
            report.update(status="passed", all_scheduled_native_calls_executed=True,
                          source_parameters_preserved=True, unique_stage_signatures=len(seen),
                          onnx_operator_counts=dict(sorted(counts.items())),
                          numerical_differences="not_evaluated_by_import_only_run")
            report.pop("stage_in_progress", None)
            save()
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                      failure_classification="environment_export_or_import_failure_not_a_numerical_comparison")
        save()
        raise


if __name__ == "__main__":
    main()
