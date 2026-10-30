"""Complete native model observations with caller-owned guarded descriptors.

Captured models get finite ordinary-versus-gated comparisons. The separately
defined MLP additionally gets an independent ordered numerical policy reference.
Neither observation grants compiler-proof or trained-model accuracy credit.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from time import perf_counter

MAX_SCENARIO_INPUT_BYTES = 2 << 30
MAX_SCENARIO_OUTPUT_BYTES = 64 << 20
# Whole owned shared-library build; ordinary codegen keeps its per-command budget.
HOST_SHARED_BUILD_TIMEOUT_SECONDS = 1200


def require_array_budget(contract):
    """Bound logical scenario arrays before generation or guarded copies.

    This counts static shapes and actual NumPy dtype widths without creating
    arrays. Compiler workspaces, guards and temporary copies are additional;
    metadata-only source ABI validation intentionally has no such allocation cap.
    """
    import numpy as np

    totals = {}
    for key, maximum in (("arguments", MAX_SCENARIO_INPUT_BYTES), ("outputs", MAX_SCENARIO_OUTPUT_BYTES)):
        total = 0
        for row in contract[key]:
            dtype = np.dtype(row["dtype"])
            shape = row["shape"]
            if dtype.kind not in "biuf" or dtype.itemsize < 1:
                raise ValueError("native scenario requires fixed-width numeric or boolean arrays")
            if type(shape) not in (list, tuple) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                raise ValueError("native scenario requires nonnegative static integer shapes")
            size = 0 if 0 in shape else dtype.itemsize
            if size:
                for dimension in shape:
                    if size > (maximum - total) // dimension:
                        raise ValueError(f"native scenario {key} array byte budget exceeded ({maximum} bytes)")
                    size *= dimension
            total += size
            if total > maximum:
                raise ValueError(f"native scenario {key} array byte budget exceeded ({maximum} bytes)")
        totals["input_bytes" if key == "arguments" else "output_bytes"] = total
    return totals


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def array_record(name, array):
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def generated_inputs(contract, *, seed):
    """Explicit finite scenarios, not recovered model weights or calibration data.

    Float parameters are strictly positive (including any quantization scales),
    i8 data are small signed values, zero points are zero, and integer gather
    indices are zero. Source-contract admissibility remains separately stated.
    """
    import numpy as np

    require_array_budget(contract)
    rng = np.random.default_rng(seed)
    result = {}
    for row in contract["arguments"]:
        dtype = np.dtype(row["dtype"])
        shape = row["shape"]
        if dtype.kind == "f":
            values = rng.uniform(0.0625, 0.125, size=shape).astype(dtype)
        elif dtype == np.dtype("int8"):
            values = rng.integers(-2, 3, size=shape).astype(dtype)
        elif dtype.kind == "b":
            values = np.ones(shape, dtype=dtype)
        elif dtype.kind in "iu":
            values = np.zeros(shape, dtype=dtype)
        else:
            raise ValueError("native demo has no input policy for " + dtype.name)
        result[row["canonical_input_name"]] = values
    return result


def validate_contract(source_path, contract):
    """Join ordered buffers to the actual pinned source, before any native call."""
    import numpy as np
    from xdsl.dialects import builtin, func

    from merlin.llvmlower.passes_xdsl import parse_mlir_text

    source = ref(source_path)
    raw = Path(source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("native model source changed while its ABI was read")
    module = parse_mlir_text(raw.decode())
    entries = [
        operation
        for operation in module.walk()
        if isinstance(operation, func.FuncOp) and operation.sym_name.data == contract["entry"]
    ]
    if len(entries) != 1 or not entries[0].body.blocks:
        raise ValueError("native contract does not select exactly one defined source entry")
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    signature = entries[0].function_type
    for key, types in (("arguments", signature.inputs), ("outputs", signature.outputs)):
        rows = contract[key]
        if len(rows) != len(types):
            raise ValueError("native contract omits actual source arguments or results")
        if len({row["name"] for row in rows}) != len(rows):
            raise ValueError("native contract buffer names must be unique")
        for row, actual in zip(rows, types, strict=True):
            if not isinstance(actual, builtin.TensorType):
                raise ValueError("native model demo requires static ranked tensor arguments/results")
            shape = list(actual.get_shape())
            if any(dimension <= 0 for dimension in shape) or shape != row["shape"]:
                raise ValueError("native metadata shape differs from actual original source")
            dtype = aliases.get(str(actual.element_type))
            if dtype is None or np.dtype(row["dtype"]) != np.dtype(dtype):
                raise ValueError("native metadata element type differs from actual original source")
    if ref(source_path) != source:
        raise ValueError("native source changed during interface validation")
    return source


def _runtime_library(llvm_path, directory):
    """Use the normal host builder, with one owned build group and exact records."""
    import os
    import shutil
    import sys

    from merlin.common import invocation_record
    from merlin.common.owned_process import run
    from merlin.common.paths import module_source_path
    from merlin.llvmlower import codegen, toolchain

    def executable(value):
        path = Path(str(value))
        return path if path.is_file() else Path(shutil.which(str(value)) or str(value))

    compiler = ref(executable(toolchain.clang()))
    names = (
        "merlin.llvmlower.codegen",
        "merlin.llvmlower.toolchain",
        "merlin.llvmlower.adjacent_exp",
        "merlin.llvmlower.guarded_zero_contraction",
        "merlin.common.proc",
        "merlin.common.invocation_record",
        "merlin.runtime.build_runtime",
    )
    paths = [module_source_path(name) for name in names] + [codegen.mlir_runtime_c()]
    runtime_sources = [ref(path) for path in paths]
    selected = [compiler]
    for value in (os.environ.get("CC", "").strip(), toolchain.host_llc()):
        if value:
            selected.append(ref(executable(value)))
    library = directory / "execution.so"
    script = (
        "from pathlib import Path; import sys; "
        "from merlin.llvmlower.codegen import build_host_shared; "
        "build_host_shared(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    command = [os.environ.get("MERLIN_COMPILER_PYTHON", sys.executable), "-c", script, str(llvm_path), str(library)]
    record = {
        "command": command,
        "timeout_seconds": HOST_SHARED_BUILD_TIMEOUT_SECONDS,
        "builder": "merlin.llvmlower.codegen.build_host_shared",
    }
    try:
        result = run(command, capture_output=True, text=True, timeout=HOST_SHARED_BUILD_TIMEOUT_SECONDS)
        record.update(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            owned_process_cleanup=getattr(result, "owned_process_cleanup", {}),
        )
        cleanup = record["owned_process_cleanup"]
        if result.returncode or cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True:
            raise RuntimeError("normal host shared build failed or did not confirm owned cleanup")
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        if "owned_process_cleanup" not in record:
            record["owned_process_cleanup"] = getattr(error, "owned_process_cleanup", {})
        raise
    finally:
        (directory / "BUILD-PROCESS.json").write_text(json.dumps(record, indent=2) + "\n")
    commands, observed_commands = [], []
    for path in sorted((directory / "invocations").glob("*/invocation.json")):
        observed = invocation_record.verify(path)
        if not any(
            observed["executable"]["path"] == choice["path"] and observed["executable"]["sha256"] == choice["sha256"]
            for choice in selected
        ):
            raise ValueError("host runtime invocation used an unselected executable")
        observed_commands.append(observed)
        commands.append(ref(path))

    def contains(pins, expected):
        return any(pin.get("path") == expected["path"] and pin.get("sha256") == expected["sha256"] for pin in pins)

    model_object = ref(directory / "execution.o")
    runtime_object = ref(directory / "mlir_runtime_host.o")
    runtime_source = ref(codegen.mlir_runtime_c())
    model_records = [value for value in observed_commands if value["stage"] == "object"]
    runtime_records = [value for value in observed_commands if value["stage"] == "runtime_object"]
    links = [value for value in observed_commands if value["stage"] == "link"]
    objects = [ref(path) for path in sorted(directory.glob("*.o"))]
    if (
        len(model_records) != 1
        or not contains(model_records[0]["inputs"], ref(llvm_path))
        or not contains(model_records[0]["outputs"], model_object)
        or not any(
            contains(value["inputs"], runtime_source) and contains(value["outputs"], runtime_object)
            for value in runtime_records
        )
        or len(links) != 1
        or not contains(links[0]["outputs"], ref(library))
        or "-shared" not in links[0]["argv"]
        or "-lm" not in links[0]["argv"]
        or {(value["path"], value["sha256"]) for value in links[0]["inputs"]}
        != {(value["path"], value["sha256"]) for value in objects}
    ):
        raise ValueError("normal host object/runtime/link records do not bind these exact products")
    for original in [*runtime_sources, *selected]:
        if ref(original["path"]) != original:
            raise ValueError("host runtime source or selected compiler changed during linking")
    link = {
        "builder": record["builder"],
        "compiler": compiler,
        "runtime_sources": runtime_sources,
        "selected_executables": selected,
        "commands": commands,
        "objects": objects,
        "build_process": ref(directory / "BUILD-PROCESS.json"),
        "owned_process_cleanup": record["owned_process_cleanup"],
        "cleanup_scope": "owned_host_shared_build_group",
        "scope": "ordinary Merlin runtime linkage, not a formal LLVM or numerical-policy proof",
    }
    return ctypes.CDLL(str(library)), link


def execute(llvm_path, destination, contract, inputs):
    import numpy as np
    from test_kernel_benchmark_formal_gates import _abi_arguments, _Guarded

    from merlin.llvmlower.abi import descriptor

    require_array_budget(contract)
    if contract["entry"] != "forward":
        raise ValueError("native demo requires the inspected forward out-parameter ABI")
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    llvm_path = Path(llvm_path).resolve()
    llvm_ref = ref(llvm_path)
    names = [row["canonical_input_name"] for row in contract["arguments"]]
    if list(inputs) != names:
        raise ValueError("provided model inputs must exactly match the complete ordered source ABI")
    for row in contract["arguments"]:
        value = inputs[row["canonical_input_name"]]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(row["dtype"])
            or list(value.shape) != row["shape"]
        ):
            raise ValueError("native input dtype or shape would require conversion/broadcasting")
    before = {name: array_record(name, array) for name, array in inputs.items()}
    abi = _abi_arguments(llvm_path.read_text(), len(contract["arguments"]) + len(contract["outputs"]))
    incoming = [
        _Guarded(np, row["shape"], row["dtype"], initial=inputs[row["canonical_input_name"]])
        for row in contract["arguments"]
    ]
    for row, buffer in zip(contract["arguments"], incoming, strict=True):
        if buffer.before != before[row["canonical_input_name"]]["sha256"]:
            raise AssertionError("guarded native upload differs from the recorded original input bytes")
    outgoing = [_Guarded(np, row["shape"], row["dtype"]) for row in contract["outputs"]]
    descriptors = [descriptor(buffer.array.ctypes.data, buffer.shape) for buffer in [*incoming, *outgoing]]
    descriptor_bytes = [bytes(value) for value in descriptors]
    library, runtime_link = _runtime_library(llvm_path, directory)
    library_ref = ref(directory / "execution.so")
    function = library._mlir_ciface_forward
    function.argtypes = [ctypes.c_void_p] * len(descriptors)
    function.restype = None
    start = perf_counter()
    function(*(ctypes.byref(value) for value in descriptors))
    elapsed = perf_counter() - start
    if any(bytes(value) != original for value, original in zip(descriptors, descriptor_bytes, strict=True)):
        raise AssertionError("model changed caller-owned descriptor metadata")
    for buffer in incoming:
        buffer.check(input_buffer=True)
    outputs, records = {}, []
    for row, buffer in zip(contract["outputs"], outgoing, strict=True):
        buffer.check(input_buffer=False)
        value = buffer.array.copy()
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise AssertionError("model scenario produced nonfinite public output: " + row["name"])
        outputs[row["name"]] = value
        output = directory / (row["name"] + ".npy")
        np.save(output, value, allow_pickle=False)
        records.append({**array_record(row["name"], value), "retained_array": ref(output)})
    if ref(llvm_path) != llvm_ref or ref(directory / "execution.so") != library_ref:
        raise AssertionError("native model product changed during its execution")
    if before != {name: array_record(name, array) for name, array in inputs.items()}:
        raise AssertionError("model execution changed original input arrays")
    record = {
        "llvm": llvm_ref,
        "library": library_ref,
        "runtime_link": runtime_link,
        "actual_abi_arguments": abi,
        "inputs": list(before.values()),
        "outputs": records,
        "all_input_and_output_guards_intact": True,
        "all_borrowed_descriptors_unchanged": True,
        "all_original_inputs_unchanged": True,
        "public_outputs_finite": True,
        "seconds_observed": elapsed,
        "timing_is_performance_evidence": False,
    }
    (directory / "NATIVE.json").write_text(json.dumps(record, indent=2) + "\n")
    return outputs, record


def compare(compiled, destination, contract, inputs, *, expected=None):
    import numpy as np

    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = validate_contract(Path(compiled) / "original-model.mlir", contract)
    baseline, baseline_record = execute(Path(compiled) / "baseline/model.ll", directory / "baseline", contract, inputs)
    checked, checked_record = execute(Path(compiled) / "checked/model.ll", directory / "checked", contract, inputs)
    required = [row["name"] for row in contract["outputs"]]
    if list(baseline) != required or list(checked) != required or (expected is not None and list(expected) != required):
        raise AssertionError("native execution dropped or reordered public model outputs")
    for name in required:
        if baseline[name].dtype != checked[name].dtype or baseline[name].shape != checked[name].shape:
            raise AssertionError("ordinary and checked model output ABI differs")
        if baseline[name].tobytes() != checked[name].tobytes():
            raise AssertionError("complete ordinary and checked model outputs differ: " + name)
        if expected is not None and (
            checked[name].dtype != expected[name].dtype
            or checked[name].shape != expected[name].shape
            or checked[name].tobytes() != expected[name].tobytes()
        ):
            np.save(directory / (name + "-independent-reference.npy"), expected[name], allow_pickle=False)
            raise AssertionError("complete model output differs from independent ordered policy: " + name)
    record = {
        "status": "PASS",
        "scope": "complete finite native model observations; no LLVM or model-accuracy proof",
        "baseline": baseline_record,
        "checked": checked_record,
        "actual_original_source_abi": source,
        "all_public_outputs_byte_equal": True,
        "independent_policy_reference": expected is not None,
        "trained_model_accuracy_checked": False,
        "whole_model_formal_proof": False,
    }
    if ref(source["path"]) != source:
        raise ValueError("native source changed after complete model execution")
    (directory / "COMPARISON.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
