#!/usr/bin/env python3
"""Export complete TVM session readback for Merlin's offline numerical checker.

This reads existing UART bytes; it does not execute, qualify timing, establish
the UART's hardware/ELF identity or compare against a reference. Retain the
independent execution receipt and reference manifest for those separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from merlin.perf.firesim_receipt import UartValidationPolicy, _verify_uart
from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details
from merlin.runtime.whole_model_readback import decode_outputs

from package_session import WIDTHS, tensor_abi


DTYPES = {"float32": "f32", "float64": "f64", "int8": "i8", "int16": "i16",
          "int32": "i32", "int64": "i64", "bool": "i1"}


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def decode_session(session, console, policy):
    if (session.get("schema") != "tvm_merlin_package_session_v1"
            or session.get("readback", {}).get("enabled") is not True):
        raise ValueError("Require a session built with complete readback enabled")
    plan = session["plan"]
    stages, calls, tensors = plan["stages"], plan["calls"], plan["outputs"]
    if (not isinstance(stages, list) or not stages or len(set(stages)) != len(stages)
            or any(not isinstance(name, str) or not name.isascii() or not name.isidentifier() for name in stages)
            or not calls or not tensors):
        raise ValueError("Require complete named stage and output inventories")
    indices = []
    for call in calls:
        if call["stage"] not in stages or not call["outputs"]:
            raise ValueError("Call references no stage or outputs")
        if any(type(index) is not int for index in call["outputs"]):
            raise ValueError("Output indices must be integers")
        indices.extend(call["outputs"])
    if indices != list(range(len(tensors))):
        raise ValueError("Every invocation output must occur exactly once in order")
    marker = f'TVM_SESSION_PASS calls={len(calls)} outputs={len(tensors)} resets=2'
    if policy.success_markers != (marker,):
        raise ValueError("UART policy does not bind the complete session counts")
    outputs, metrics, frames = parse_binary_console_details(console)
    diagnostics = binary_console_diagnostics(console).decode("utf-8", errors="strict")
    _verify_uart(diagnostics, policy.success_markers)
    if diagnostics.splitlines().count("TVM_SESSION_EXIT rc=0") != 1 or any(
            line.startswith("TVM_SESSION_EXIT") and line != "TVM_SESSION_EXIT rc=0"
            for line in diagnostics.splitlines()):
        raise ValueError("Session does not report exactly one successful exit")
    if set(outputs) != {f"out{index}" for index in range(len(tensors))}:
        raise ValueError("Readback added or omitted a declared output")
    arrays, encodings = [], []
    for index, tensor in enumerate(tensors):
        dtype, shape, size = tensor_abi(tensor)
        name = f"out{index}"
        frame = frames[name]
        cols = shape[-1] if shape else 1
        if (frame.rows, frame.cols, frame.word_bytes, frame.signed) != (
                size // WIDTHS[dtype] // cols, cols, WIDTHS[dtype], dtype.startswith("int")):
            raise ValueError("Readback frame differs from declared storage ABI: " + name)
        encoding = "native_numpy"
        if dtype in DTYPES:
            array = decode_outputs({"out0": outputs[name]}, {"out0": frame}, [(shape, DTYPES[dtype])])[0]
            if dtype == "bool":
                if not np.all((array == 0) | (array == 1)):
                    raise ValueError("Noncanonical Boolean output")
                array = array.astype(bool)
        elif dtype == "bfloat16":
            # NPZ cannot preserve NumPy's optional BF16 dtype identity. Keep its
            # exact words with the same explicit encoding as verify_session.py.
            array = np.asarray(outputs[name], dtype="<u2").reshape(shape)
            if np.any((array & 0x7f80) == 0x7f80):
                raise ValueError("Nonfinite BF16 output")
            encoding = "raw_uint16_bfloat16"
        else:
            array = np.asarray(outputs[name], dtype=np.dtype(dtype).newbyteorder("<")).reshape(shape)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("Nonfinite floating output")
        arrays.append(array)
        encodings.append(encoding)
    return arrays, encodings, metrics


def export_readback(session_path, console_path, policy_path, output_dir):
    bindings = [identity(path) for path in (session_path, console_path, policy_path)]
    session = json.loads(Path(session_path).read_text())
    policy = UartValidationPolicy.from_json(json.loads(Path(policy_path).read_text()))
    if policy.policy_id != "tvm-session-" + bindings[0]["sha256"]:
        raise ValueError("Validation policy belongs to another session manifest")
    arrays, encodings, metrics = decode_session(session, Path(console_path).read_bytes(), policy)
    for binding in bindings:
        if identity(binding["path"]) != binding:
            raise ValueError("Readback source changed during decoding")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    steps, counts = [], {}
    for ordinal, call in enumerate(session["plan"]["calls"]):
        program = call["stage"]
        step = counts.get(program, 0)
        counts[program] = step + 1
        path = out / f"call_{ordinal:04d}_{program}_{step:04d}.npz"
        np.savez(path, **{f"out{i}": arrays[index] for i, index in enumerate(call["outputs"])})
        steps.append({"program": program, "step": step, "actual": identity(path),
                      "outputs": [{"index": i, "session_output": index,
                                   "dtype": session["plan"]["outputs"][index]["dtype"],
                                   "shape": list(arrays[index].shape), "storage_encoding": encodings[index]}
                                  for i, index in enumerate(call["outputs"])]})
    result = {"schema": "tvm_merlin_session_readback_v1", "bindings": bindings,
              "producer": identity(__file__), "steps": steps,
              "observed_metrics": metrics, "timing_qualified": False,
              "execution_identity_verified": False, "numerical_comparison_performed": False,
              "scope": "Complete transport/ABI/warm-profile validation and actual-array export; requires separate execution and reference receipts"}
    (out / "readback.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session", "uart", "validation-policy", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    export_readback(args.session, args.uart, args.validation_policy, args.output_dir)
    print(args.output_dir / "readback.json")


if __name__ == "__main__":
    main()
