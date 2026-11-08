"""Execute a generated application and preserve raw outputs for independent comparison."""
from __future__ import annotations

import argparse
import ctypes as ct
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from model_runtime import ModelRuntime


def qualified_execution_options(native_qualification, memory_qualification, native_cache):
    """Authenticate both optional execution implementations before device I/O."""
    from npu_model.configs import numerics, rtl_math
    from npu_model.functional import state as original_state
    from qualified_native_dot import load_qualified_math
    from merlin.runtime.uniform_sparse_memory import UniformSparseByteMemory, verify_qualification

    native_qualification, memory_qualification = Path(native_qualification), Path(memory_qualification)
    proxy = load_qualified_math(numerics.NUMERICS["rtl"], rtl_math.product,
        {"numerics": numerics.__file__, "rtl_math": rtl_math.__file__}, native_qualification, native_cache)
    memory_receipt = json.loads(memory_qualification.read_text())
    memory_proof = verify_qualification(memory_receipt, reference_source=original_state.__file__,
        expected_reference_sha256="211081f8f356bccf397069de24f9fc328420b7589642c50f211565c0cfd78ca0")
    metadata = {"backend": "qualified_native",
        "arithmetic": proxy.execution_metadata(), "memory": memory_proof,
        "native_qualification_file_sha256": sha256(native_qualification.read_bytes()).hexdigest(),
        "memory_qualification_file_sha256": sha256(memory_qualification.read_bytes()).hexdigest(),
        "independent_reference_engine_used_for_execution": False}
    return dict(math_proxy=proxy, memory_factory=UniformSparseByteMemory, execution_metadata=metadata)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, default=Path("atlas-mlir"))
    parser.add_argument("--max-cycles", type=int, default=10000000)
    parser.add_argument("--max-instructions", type=int, default=1000000, help="bounded functional instruction count per launch")
    parser.add_argument("--backend", choices=("cycle", "functional", "qualified_native"), default="cycle")
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--memory-qualification", type=Path)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--compact-trace", action="store_true")
    parser.add_argument("--input-overrides", type=Path, help="generated predecessor inputs, keyed input_N")
    parser.add_argument("--execution-output", type=Path, help="separate mutable execution artifacts from compiled package")
    args = parser.parse_args()
    if args.max_instructions <= 0:
        parser.error("max-instructions must be positive")
    if args.backend == "qualified_native":
        if not args.native_qualification or not args.memory_qualification:
            parser.error("qualified_native requires separate arithmetic and memory qualifications")
    elif args.native_qualification or args.memory_qualification or args.native_cache:
        parser.error("qualification options require the qualified_native backend")
    root = args.compiled.resolve()
    record = json.loads((root / "compiled.json").read_text())
    arrays = dict(np.load(root / "tensors.npz", allow_pickle=False))
    for key, metadata in record["logical_source_tensors"].items():
        if sha256(arrays[key].tobytes()).hexdigest() != metadata["sha256"]:
            raise AssertionError("source tensor changed after compilation")
    inputs = [np.ascontiguousarray(arrays[f"input_{index}"])
              for index in range(len(record["interface"]["inputs"]))]
    input_overrides = {}
    if args.input_overrides:
        with np.load(args.input_overrides, allow_pickle=False) as overrides:
            for key in overrides.files:
                if not key.startswith("input_") or not key[6:].isdigit():
                    raise ValueError(f"invalid input override name: {key}")
                index = int(key[6:])
                if index >= len(inputs):
                    raise ValueError(f"input override index exceeds generated ABI: {key}")
                value = np.ascontiguousarray(overrides[key])
                if value.shape != inputs[index].shape or value.dtype != inputs[index].dtype:
                    raise ValueError(f"input override changes generated ABI: {key}")
                inputs[index] = value
                input_overrides[key] = sha256(value.tobytes()).hexdigest()
    initial = [value.tobytes() for value in inputs]
    execution_root = args.execution_output.resolve() if args.execution_output else root
    execution_root.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": np.uint16, "f32": np.float32, "i8": np.uint8, "i32": np.int32}
    outputs = [np.full(spec["shape"], 0xa5, dtype=dtype[spec["dtype"]])
               for spec in record["interface"]["output_types"]]
    images, plans = [], {}
    for segment in record["segments"]:
        directory = root / "application" / segment["path"]
        raw = (directory / "program.bin").read_bytes()
        if sha256(raw).hexdigest() != segment["binary_sha256"]:
            raise AssertionError("generated image changed")
        images.append(raw)
        plans[sha256(raw).hexdigest()] = json.loads((directory / "execution_plan.json").read_text())
    runtime_type, options = ModelRuntime, {"max_cycles": args.max_cycles}
    execution_qualification = None
    if args.backend in {"functional", "qualified_native"}:
        from functional_runtime import FunctionalModelRuntime

        runtime_type, options = FunctionalModelRuntime, {"trace_payloads": not args.compact_trace, "max_instructions": args.max_instructions}
        if args.backend == "qualified_native":
            selected = qualified_execution_options(args.native_qualification, args.memory_qualification,
                args.native_cache or execution_root.parent / "qualified-native-cache")
            options.update(selected)
            execution_qualification = selected["execution_metadata"]
    elif args.compact_trace:
        parser.error("compact traces currently require the functional backend")
    runtime = runtime_type(args.atlas_root.resolve(), args.model_root.resolve(), images, execution_root / "execution",
                           image_plans=plans, memory_bytes=max(262144, record.get("io_memory_bytes") or 0), **options)
    library = ct.CDLL(str(root / "application/c-package/capsule.so"))
    library.run.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    if record["interface"].get("platform_binding"):
        library.bind.argtypes = [type(callback) for callback in runtime.callbacks]
        if library.bind(*runtime.callbacks):
            raise AssertionError("generated application binding failed")
    parameters = None
    if record["interface"].get("model_parameters"):
        data = (root / "application/handoff/parameters.bin").read_bytes()
        if sha256(data).hexdigest() != record["interface"]["model_parameters"]["payload_sha256"]:
            raise AssertionError("external immutable parameter blob changed")
        parameters = ct.create_string_buffer(data)
        library.bind_parameters.argtypes = [ct.c_void_p, ct.c_size_t]
        if library.bind_parameters(parameters, len(data)):
            raise AssertionError("generated external parameter binding failed")
    in_pointers = (ct.c_void_p * len(inputs))(*(value.ctypes.data for value in inputs))
    out_pointers = (ct.c_void_p * len(outputs))(*(value.ctypes.data for value in outputs))
    status = library.run(in_pointers, out_pointers)
    execution = runtime.finish()
    if status:
        raise AssertionError(f"generated application returned {status}")
    if any(value.tobytes() != saved for value, saved in zip(inputs, initial, strict=True)):
        raise AssertionError("generated application changed immutable source inputs")
    if parameters is not None and ct.string_at(parameters, len(data)) != data:
        raise AssertionError("generated application changed immutable source parameters")
    np.savez(execution_root / "actual.npz", **{f"output_{index}": value for index, value in enumerate(outputs)})
    receipt = {"status": "EXECUTED_UNCOMPARED", "source_request_digest": record["source_request_digest"],
               "source_model_sha256": record["source_model_sha256"], "immutable_inputs": True,
               "functional_instruction_limit": args.max_instructions if args.backend in {"functional", "qualified_native"} else None,
               "input_overrides": input_overrides,
               "inputs": [{"sha256": sha256(value.tobytes()).hexdigest(), "shape": list(value.shape),
                           "dtype": value.dtype.str} for value in inputs],
               "device_launches": len(execution["launches"]), "backend": args.backend,
               **({"execution_qualification": execution["execution_qualification"]}
                  if execution_qualification is not None else {}),
               "placements": sorted({row["placement"] for row in record["regions"]}),
               "host_only": not bool(record["segments"]),
               **({"device_cycles": execution["total_cycles"]} if "total_cycles" in execution else
                  {"functional_instructions": execution["total_instructions"]}),
               "outputs": [{"dtype": value.dtype.str, "shape": list(value.shape),
                            "sha256": sha256(value.tobytes()).hexdigest()} for value in outputs],
               "independent_comparison": "PENDING", "hardware_qualification": "PENDING"}
    (execution_root / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
