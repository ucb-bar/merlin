#!/usr/bin/env python3
"""Resource-admitted workload contraction model.

Inputs must already have explicit positive-zero tail padding.
"""
from atlas_host_layout import ATLAS_DRAM_BASE
import argparse
from datetime import datetime, timezone
import os
import io
import json
from pathlib import Path
import resource
import struct
import sys
import time
import traceback
import numpy as np
import kernel_library  # noqa: E402,F401
from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array
from atlas_program import assembly, dma_load_bytes, preflight_bundle
from atlas_path_guard import checked as safe_path
from atlas_tooling import sha256, write_json
from run_model_tile import (load_bundle, configure_model, encode_operands, execute_tile, input_sources, model_binding,
    metrics)


def resource_plan(m, k, n):
    """Bound each physical address/resource before encoding or model execution."""
    if any((type(x) is not int or x <= 0 or x % 32 for x in (m, k, n))) or k > 8192 or m * n > 1048576:
        raise ValueError("Require M/N/K positive multiples32,K<=8192,<=1048576 output words")
    input_bytes, output_bytes = (m * k + k * n, m * n * 2)
    total = input_bytes + output_bytes + 512
    if total > 64 * 1024 * 1024:
        raise ValueError("Contraction exceeds64MiB bounded DRAM window")
    resident_b = k * n <= 256 * 1024
    resident_a = not resident_b and m * k <= 256 * 1024
    return {"maximum_program_words": 8192, "maximum_dram_bytes": 64 * 1024 * 1024, "dram_bytes": total,
        "resident_b": resident_b, "resident_a": resident_a, "resident_bank1_bytes": k * n if resident_b else m * k
        if resident_a else 0,
        "resident_bank1_byte_range": [262144, 262144 + (k * n if resident_b else m * k if resident_a else 0)],
        "streamed_scratch_bank0_byte_ranges": [[8192, 9216], [9216, 10240], [10240, 12288]], "vmem_bank_bytes": 256 * 1024,
        "one_dma_low32_window": True}


def validate_workload_arrays(arrays, source_dtype):
    if set(arrays) != {"A", "B", "native_C"} or source_dtype != "float32":
        raise ValueError("Workload producer requires exact FP32 A,B,native_C source arrays")
    a, b, c = [arrays[x] for x in ("A", "B", "native_C")]
    if any((x.ndim != 2 for x in (a, b, c))):
        raise ValueError("Require rank2 contraction arrays")
    m, k = a.shape
    n = b.shape[1]
    resource_plan(m, k, n)
    if b.shape[0] != k or c.shape != (m, n):
        raise ValueError("Contraction dimensions disagree")
    if any((x.dtype != np.float32 or not np.isfinite(x).all() for x in (a, b, c))):
        raise ValueError("Require finite FP32 arrays without implicit conversion")
    return (a, b, c)


def physical_payload(bits):
    m, n = bits.shape
    return b"".join((bits[r : r + 32, c : c + 32].reshape(32, 2, 16).transpose(1, 0,
        2).astype("<u2").tobytes() for r in range(0, m, 32) for c in range(0, n, 32)))


def admit_program(words):
    if not words or len(words) > 8192 or words[-1] != 115 or (115 in words[:-1]) or (1048691 in words):
        raise ValueError("Require <=8192 instruction words and exactly one terminal ECALL")


def code_sources():
    names = ("run_workload_contraction.py", "atlas_program.py", "run_model_tile.py", "atlas_tile_arithmetic.py",
        "atlas_vector_arithmetic.py")
    # This script and run_model_tile.py are in tools/; the other three are atlas-kernels modules.
    paths = {name: safe_path(kernel_library.source_path(name)) for name in names}
    for name, expected in paths.items():
        module = sys.modules.get(Path(name).stem)
        filename = __file__ if name == "run_workload_contraction.py" else getattr(module, "__file__", None)
        if filename is None or safe_path(filename) != expected:
            raise ValueError("Workload helper imported outside bound producer source: " + name)
    return {name: sha256(path) for name, path in paths.items()}


def run(args, out, receipt):
    preflight_bundle(args)
    before = code_sources()
    root = safe_path(args.model_root)
    receipt["code_before"] = before
    snapshot = out / "producer-sources"
    snapshot.mkdir()
    for name, digest in before.items():
        (snapshot / name).write_bytes(safe_path(kernel_library.source_path(name)).read_bytes())
        if sha256(snapshot / name) != digest:
            raise ValueError("Producer changed while snapshotting")
    model, torch, assemble, pin = configure_model(root)
    model_before = model_binding(root)
    inputs_before = input_sources(args)
    arrays, manifest = load_bundle(args)
    a, b, native = validate_workload_arrays(arrays, manifest.get("source_dtype"))
    m, k = a.shape
    n = b.shape[1]
    plan = resource_plan(m, k, n)
    receipt["shape"] = [m, k, n]
    receipt["resource_admission"] = plan
    encoded, converted, conversions = encode_operands(arrays, "tensor-pow2")
    exponent = sum((x["scale_exponent"] for x in conversions.values()))
    expected = rescale_bf16_array(ordered_matmul(encoded["A"], encoded["B"]), exponent)
    if expected.shape != (m, n) or expected.dtype != np.uint16:
        raise ValueError("Vector oracle output contract differs")
    # Physical MXU outputs contain low16 columns, then high16 columns per tile.
    payload = physical_payload(expected)
    inputs = b"".join((encoded["A"][r : r + 32, q : q + 32].tobytes() for r in range(0, m, 32) for q in range(0, k, 32)))
    inputs += b"".join((encoded["B"][q : q + 32, c : c + 32].tobytes() for c in range(0, n, 32) for q in range(0, k, 32)))
    offset = len(inputs) + 256
    initial = inputs + b"\xa5" * 256 + b"\xcd" * len(payload) + b"Z" * 256
    # Each resident operand occupies bank 1; streamed 32x32 tiles and the
    # 2048-byte output use bank 0 scratch.
    resident_b, resident_a = (plan["resident_b"], plan["resident_a"])
    source = assembly(m, k, n, offset, exponent, resident_b=resident_b, pretranspose_resident_b=resident_b,
        resident_a=resident_a)
    program = assemble(io.StringIO(source))
    words = program.assemble()
    admit_program(words)
    plan["instruction_words"] = len(words)
    requested = {"load_bytes": dma_load_bytes(m, k, n, resident_a=resident_a, resident_b=resident_b),
        "store_bytes": len(payload)}
    for name, data in [("initial-dram.bin", initial), ("expected-bf16-tiles.bin", payload),
        ("program.bin", b"".join(struct.pack("<I", x) for x in words))]:
        (out / name).write_bytes(data)
    (out / "program.S").write_text(source)
    np.savez(out / "source.npz", **arrays)
    np.savez(out / "encoded.npz", **encoded)
    write_json(out / "source-manifest.json", manifest)
    receipt.update(producer="workload_contraction_v1", code_before=before, native_fidelity_claim=False,
        raw_image_execution_claim=False, model_input="new_assembler_decoded_instruction_objects",
        invocations="one_reset_core_per_repeat_all_tiles_in_one_instruction_program",
        recipe={"name": "tensor-pow2", "native_fidelity_claim": False}, shape=[m, k, n], conversion=conversions,
        source_inputs=inputs_before, model_binding_before=model_before, output_rescale_exponent=exponent,
        memory={"dram_base": ATLAS_DRAM_BASE, "input_bytes": len(inputs), "output_offset": offset,
        "output_bytes": len(payload), "guard_bytes_each": 256, "total_dram_bytes": len(initial)}, resource_admission=plan,
        requested_dma=requested,
        model={"revision": pin, "module_file": str(safe_path(model.__file__)), "torch_version": torch.__version__,
        "numerics": "rtl", "backend": "curve", "curve": "ee290sim_vcs_probe.json",
        "curve_sha256": model_before["curve_sha256"], "imported_source_sha256": model_before["imported_source_sha256"]},
        timing_boundary="one_fetch_to_ECALL_program_excluding_host_conversion_packing_and_loading", runs=[])
    for repeat in range(args.repeats):
        receipt["runs"].append(execute_tile(program, initial, payload, offset, requested, out, repeat, args.max_cycles,
            60 + repeat))
    actual_payload = (out / "actual-bf16-halves-0.bin").read_bytes()
    actual = np.empty_like(expected)
    index = 0
    for r in range(0, m, 32):
        for c in range(0, n, 32):
            halves = np.frombuffer(actual_payload[index : index + 2048], dtype="<u2").reshape(2, 32, 16)
            actual[r : r + 32, c : c + 32] = np.concatenate((halves[0], halves[1]), axis=1)
            index += 2048
    if not np.array_equal(actual, expected):
        raise ValueError("Actual complete output differs from independent target oracle")
    target = (actual.astype(np.uint32) << 16).view(np.float32)
    np.savez(out / "result.npz", target_C=target, actual_bits=actual, expected_bits=expected)
    model_after = model_binding(root)
    inputs_after = input_sources(args)
    after = code_sources()
    receipt.update(code_after=after,
        source_bookends={"model_after": model_after, "input_before": inputs_before, "input_after": inputs_after,
        "unchanged": before == after and model_before == model_after and (inputs_before == inputs_after)},
        diagnostic_errors={"target_vs_native_C": metrics(target, native)})
    if not receipt["source_bookends"]["unchanged"]:
        raise ValueError("Source/model/input changed during execution")
    receipt["status"] = "passed_model_only"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-cycles", type=int, default=20000000)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    args.smoke = False
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "status": "failed", "command": sys.argv, "cwd": os.getcwd(), "python": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    code = 1
    try:
        if not 1 <= args.repeats <= 3 or args.max_cycles <= 0:
            raise ValueError("Require1..3 repeats and positive cycle bound")
        run(args, out, receipt)
        code = 0
    except Exception as e:
        receipt["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    receipt.update(wall_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifacts={x.name: sha256(safe_path(x)) for x in out.iterdir() if safe_path(x).is_file()
        and x.name != "receipt.json"})
    write_json(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "error": receipt.get("error"),
        "cycles": [x["cycles"] for x in receipt.get("runs", [])]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
