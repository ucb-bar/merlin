#!/usr/bin/env python3
"""Run the whole native C graph with strictly bound arithmetic-reference replies."""
import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, write_json, sha, run_command

import argparse
import subprocess
import os
import ctypes as c
import json
from pathlib import Path
import time
import numpy as np

try:
    from .atlas_path_guard import checked
except ImportError:
    from atlas_path_guard import checked


def read(path):
    return checked(path).read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("generated", "math", "reference", "out", "cc"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    for key in ("generated", "math", "reference", "out", "cc"):
        checked(getattr(args, key))
    out = args.out
    if out.exists():
        raise ValueError("fresh native gate output required")
    out.mkdir(parents=True)
    admission = Admission()
    inputs = admission.inputs
    admitted = admission.read

    script = Path(__file__).absolute()
    admitted(script)
    graph = json.loads(admitted(args.generated / "graph.json"))
    reference = json.loads(admitted(args.reference / "receipt.json"))
    for path, digest in graph["input_sha256"].items():
        admitted(Path(path), digest)
    for name, digest in graph["generated_sha256"].items():
        admitted(args.generated / name, digest)
    for name, digest in graph["params_sha256"].items():
        admitted(args.generated / name, digest)
    admitted(args.math)
    admitted(args.math.with_name("atlas_numeric.h"))
    admitted(args.math.with_name("atlas_host_layout.h"))
    admitted(args.cc)
    library = out / "atlas_resnet_graph.so"
    command = [str(args.cc), "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-fPIC", "-shared", str(args.math), str(args.generated / "graph.c"),
        str(args.generated / "constants.S"), "-o", str(library)]
    result = run_command(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ, TMPDIR=str(out)))
    (out / "native-build.log").write_text(result.stdout)
    native = {"command": command, "returncode": result.returncode, "inputs_sha256": dict(inputs),
        "source_bookends_equal": admission.recheck(),
        "compiler_version": subprocess.check_output([str(args.cc), "--version"], text=True),
        "output_sha256": sha(read(library)) if result.returncode == 0 else None}
    write_json(out / "native-build.json", native)
    if result.returncode or not native["source_bookends_equal"]:
        raise ValueError("native build failed/changed")
    admitted(out / "native-build.json")
    admitted(library, native["output_sha256"])
    for node in reference["nodes"]:
        if "payload" in node:
            admitted(Path(node["payload"]["path"]), node["payload"]["sha256"])
    for kernel in reference["kernels"]:
        for name in ("encoded_a_tiled", "expected_physical_output"):
            item = kernel[name]
            admitted(Path(item["path"]), item["sha256"])
    if len(reference["kernels"]) != 54 or len(graph["kernels"]) != 54:
        raise ValueError("incomplete graph")
    lib = c.CDLL(str(library))
    callback_type = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint, c.POINTER(c.c_uint8), c.c_size_t, c.c_int,
        c.POINTER(c.c_uint8), c.c_size_t)
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.atlas_resnet_graph_run.argtypes = [f32, f32, c.c_void_p, callback_type]
    lib.atlas_resnet_graph_node.argtypes = [c.c_uint, c.POINTER(c.c_size_t)]
    lib.atlas_resnet_graph_node.restype = c.POINTER(c.c_float)
    calls = []
    errors = []

    def callback(context, index, a, a_bytes, exponent, physical, output_bytes):
        try:
            if index != len(calls):
                raise ValueError("wrong ordered kernel index")
            expected = reference["kernels"][index]
            declared = graph["kernels"][index]
            if expected["site"] != declared["site"]:
                raise ValueError("wrong kernel site")
            actual_a = c.string_at(a, a_bytes)
            reference_a = read(Path(expected["encoded_a_tiled"]["path"]))
            if actual_a != reference_a or exponent != expected["a_scale_exponent"]:
                calls.append({"index": index, "site": expected["site"], "activation_bytes_exact": False,
                    "actual_a_sha256": sha(actual_a), "expected_a_sha256": sha(reference_a), "actual_scale": exponent,
                    "expected_scale": expected["a_scale_exponent"]})
                raise ValueError("dynamic C activation differs from qualified arithmetic reference")
            reply = read(Path(expected["expected_physical_output"]["path"]))
            if len(reply) != output_bytes:
                raise ValueError("wrong physical output size")
            c.memmove(physical, reply, len(reply))
            calls.append({"index": index, "site": expected["site"], "activation_bytes_exact": True,
                "a_scale_exponent": exponent, "physical_reply_sha256": sha(reply)})
            return 0
        except Exception as e:
            errors.append(str(e))
            return 1

    callback_object = callback_type(callback)
    entry_node = reference["nodes"][0]
    image = np.frombuffer(admitted(Path(entry_node["payload"]["path"]), entry_node["payload"]["sha256"]), dtype="<f4").copy()
    logits = np.zeros(1000, dtype=np.float32)
    start = time.monotonic()
    result = int(lib.atlas_resnet_graph_run(image, logits, None, callback_object))
    wall = time.monotonic() - start
    node_checks = []
    host_differences = []
    for index, node in enumerate(reference["nodes"]):
        if node["kind"] == "output":
            continue
        count = c.c_size_t()
        pointer = lib.atlas_resnet_graph_node(index, c.byref(count))
        if not pointer or count.value != int(np.prod(node["output"]["shape"])):
            raise ValueError("wrong native tensor extent")
        raw = c.string_at(pointer, count.value * 4)
        equal = sha(raw) == node["output"]["sha256"]
        check = {"index": index, "node": node["node"], "words": count.value, "reference_words_exact": equal}
        if not equal:
            check["actual_sha256"] = sha(raw)
            check["expected_sha256"] = node["output"]["sha256"]
            if "payload" in node:
                expected = np.frombuffer(read(Path(node["payload"]["path"])), dtype="<f4")
                actual = np.frombuffer(raw, dtype="<f4")
                check["differing_words"] = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                check["max_abs"] = float(np.max(np.abs(actual - expected)))
            host_differences.append(check)
        node_checks.append(check)
    expected_final = admitted(args.reference / "target-logits.f32.bin", reference["final_logits"]["sha256"])
    final_exact = logits.tobytes() == expected_final
    bookends = admission.recheck()
    all_calls = len(calls) == 54 and all(call["activation_bytes_exact"] for call in calls)
    status = "PASS" if result == 0 and not errors and all_calls and final_exact and bookends else "FAIL"
    report = {"status": status,
        "scope": (
        "Complete native C host graph with hash-bound arithmetic-reference device "
        "replies; no instruction-model/RTL execution in this gate"), "graph_returncode": result, "errors": errors,
        "kernels": calls, "node_checks": node_checks, "host_internal_fp32_differences": host_differences,
        "host_average_policy": graph["host_average_policy"], "all_54_device_inputs_and_scales_exact": all_calls,
        "final_1000_logit_words_exact": final_exact, "final_logits_sha256": sha(logits.tobytes()), "wall_seconds": wall,
        "source_bookends_equal": bookends, "inputs_sha256": inputs, "host_elf_executed": False, "full_soc_executed": False,
        "native_fp32_gate_policy": "Original failing FP32 gates unchanged; exact target input/output replay is separate"}
    write_json(out / "graph-replay.json", report)
    (out / "graph-logits.f32.bin").write_bytes(logits.tobytes())
    print(json.dumps({key: report[key] for key in ("status", "graph_returncode", "all_54_device_inputs_and_scales_exact",
        "final_1000_logit_words_exact", "host_internal_fp32_differences", "wall_seconds")}))
    raise SystemExit(status != "PASS")


if __name__ == "__main__":
    main()
