#!/usr/bin/env python3
"""Run a source-bound 32xK by Kx32 tile with the pinned Atlas Python model."""

from atlas_host_layout import ATLAS_DRAM_BASE
# preflight_archive is intentionally re-exported for tools/rtl_core/probe.py's support loader.
import kernel_library  # noqa: E402,F401
from atlas_tooling import preflight_archive, run_command


import argparse
from atlas_path_guard import checked
from atlas_tooling import sha256, write_json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
import zipfile

from atlas_tile_arithmetic import (bf16_value, f32_to_bf16_bits, mxu0_dot, output_halves, pack_bf16, rescale_bf16,
    tensor_scale_exponent)

MODEL_PIN = "9e6313bddecb8ba936206eb347f5cd4e5335282c"
RECIPE = "atlas_mxu0_fp8_unit_bf16acc_v1"
RECIPES = {"unit": RECIPE, "tensor-pow2": "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1"}


class OutputMismatch(AssertionError):
    def __init__(self, differences, context):
        super().__init__(f"independent arithmetic mismatch: {len(differences)} bytes, first {differences[:8]}")
        self.context = context


safe_path = checked


def validate_arrays(arrays, source_dtype, *, matrix=False):
    import numpy as np

    required = {"A", "B", "native_C"}
    optional = {"A_bf16", "B_bf16", "native_C_bf16"}
    if not required <= arrays.keys() or arrays.keys() - required - optional:
        raise ValueError("NPZ requires A, B, native_C and only optional *_bf16 words")
    a, b, native = (arrays[key] for key in ("A", "B", "native_C"))
    if (a.ndim != 2 or b.ndim != 2 or not 32 <= a.shape[1] <= 4096 or a.shape[1] % 32 or b.shape[0] != a.shape[1]
        or a.shape[0] < 32 or b.shape[1] < 32 or a.shape[0] % 32 or b.shape[1] % 32
        or native.shape != (a.shape[0], b.shape[1]) or native.size > 1048576
        or (not matrix and (a.shape[0] != 32 or b.shape[1] != 32))):
        raise ValueError("require M/N multiples32, K=32..4096 multiple32, <=1048576 outputs; tile mode M=N=32")
    if source_dtype not in ("float32", "bfloat16"):
        raise ValueError("source_dtype must be float32 or bfloat16")
    for key in required:
        value = arrays[key]
        if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite float32")
        if key in ("A", "B") and source_dtype == "bfloat16":
            if np.any(value.view(np.uint32) & 65535):
                raise ValueError("declared BF16 operands are not exact BF16 float32 views")
        raw = arrays.get(key + "_bf16")
        if raw is not None:
            if (raw.dtype != np.dtype("uint16") or raw.shape != value.shape
                or not np.array_equal(raw.astype(np.uint32) << 16, value.view(np.uint32))):
                raise ValueError(f"{key}_bf16 does not authenticate its exact float32 view")
    return a, b, native


def validate_archive_members(members):
    required = {"A.npy", "B.npy", "native_C.npy"}
    optional = {"A_bf16.npy", "B_bf16.npy", "native_C_bf16.npy"}
    names = [member.filename for member in members]
    if len(names) != len(set(names)) or not required <= set(names) or set(names) - required - optional:
        raise ValueError("NPZ requires unique flat A/B/native_C.npy and only optional *_bf16.npy members")
    if (any(member.file_size < 0 for member in members) or sum(member.file_size for member in members) > 32 * 1024 * 1024):
        raise ValueError("NPZ exceeds bounded uncompressed size")


def load_bundle(args):
    import numpy as np

    if args.smoke:
        if args.input or args.manifest:
            raise ValueError("--smoke excludes --input/--manifest")
        row, k = np.indices((32, 64))
        a = (((row * 7 + k * 3) % 19 - 9) / 8).astype(np.float32)
        k, column = np.indices((64, 32))
        b = (((k * 5 + column * 11) % 23 - 11) / 16).astype(np.float32)
        arrays = {"A": a, "B": b, "native_C": (a.astype(np.float64) @ b).astype(np.float32)}
        manifest = {"schema_version": 1, "source_dtype": "float32",
            "source": {"kind": "synthetic_nonsymmetric_smoke_no_workload_claim"}, "site": {"M": 32, "N": 32, "K": 64}}
        return arrays, manifest
    if not args.input or not args.manifest:
        raise ValueError("require --input and --manifest, or --smoke")
    path, manifest_path = safe_path(args.input), safe_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1 or not isinstance(manifest.get("source"), dict) or not manifest["source"]
        or not isinstance(manifest.get("site"), dict) or not manifest["site"]):
        raise ValueError("manifest requires schema_version 1 and nonempty source/site objects")
    if manifest.get("npz_sha256") != sha256(path):
        raise ValueError("NPZ hash disagrees with manifest")
    with zipfile.ZipFile(path) as archive:
        validate_archive_members(archive.infolist())
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key].copy() for key in bundle.files}
    return arrays, manifest


def assembly(k_tiles, output_offset, rescale_exponent=None):
    # VMEM word bases: byte addresses 0x2000, 0x2400, 0x2800/0x2c00.
    lines = ["li x1, 0x800", "li x2, 0x900",
        "li x3, 0xa00", "li x5, 1024", "li x7, " + str(output_offset), "li x8, 2048", "li x30, 1", "dma.config x30, ch0"]
    for tile in range(k_tiles):
        lines += [f"li x4, {tile * 1024}", "dma.load x1, x4, x5, ch0", "dma.wait ch0", f"li x4, {(k_tiles + tile) * 1024}",
            "dma.load x2, x4, x5, ch0", "dma.wait ch0", "vload m0, 0(x1)", "delay 34", "vload m1, 0(x2)", "delay 34",
            "vtrpose.xlu m1, m1", "delay 66", "vmatpush.weight.mxu0 w0, m1", "delay 32",
            ("vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0") + " acc0, m0, w0", "delay 96"]
    lines += ["vmatpop.bf16.acc.mxu0 m2, acc0", "delay 34"]
    if rescale_exponent is not None:
        if not -126 <= rescale_exponent <= 127:
            raise ValueError("VPU rescale factor must be normal finite BF16")
        factor_bits = (rescale_exponent + 127) << 7
        lines += [f"vli.all m4, {factor_bits}", "delay 66", "vmul.bf16 m2, m2, m4", "delay 66"]
    lines += ["vstore m2, 0(x3)", "delay 34", "li x9, 0xb00", "vstore m3, 0(x9)", "delay 34", "dma.store x7, x3, x8, ch0",
        "dma.wait ch0", "nop", "ecall"]
    return "\n".join(lines) + "\n"


def metrics(actual, reference):
    import numpy as np

    delta = actual.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64)))
    error = float(np.linalg.norm(delta))
    return {"max_abs": float(np.max(np.abs(delta))), "l2": error, "relative_l2": error / norm if norm else None,
        "zero_reference": norm == 0, "exact_equal": bool(np.array_equal(actual, reference))}


def mutation_witnesses(encoded, expected):
    """Independent counterfactual outputs for omitted transpose/continuation."""
    k_size = encoded["A"].shape[1]
    witnesses = {}
    for row in range(32):
        for column in range(32):
            a = encoded["A"][row]
            if "omitted_B_transpose" not in witnesses:
                weights = [encoded["B"][(k // 32) * 32 + column, k % 32] for k in range(k_size)]
                changed = mxu0_dot(a, weights)
                if changed != expected[row][column]:
                    witnesses["omitted_B_transpose"] = {"cell": [row, column], "expected_bits": expected[row][column],
                        "mutated_bits": changed}
            if k_size > 32 and "reset_at_final_K_tile" not in witnesses:
                changed = mxu0_dot(a[-32:], encoded["B"][-32:, column])
                if changed != expected[row][column]:
                    witnesses["reset_at_final_K_tile"] = {"cell": [row, column], "expected_bits": expected[row][column],
                        "mutated_bits": changed}
            if len(witnesses) == (2 if k_size > 32 else 1):
                return witnesses
    return witnesses


def model_sources(model_root):
    sources = {}
    for name, module in list(sys.modules.items()):
        if name == "npu_model" or name.startswith("npu_model."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = safe_path(filename)
                if not path.is_relative_to(model_root):
                    raise RuntimeError(f"model module imported outside selected root: {name}")
                sources[str(path.relative_to(model_root))] = sha256(path)
    return sources


def input_sources(args):
    return {name: sha256(getattr(args, name)) for name in ("input", "manifest") if getattr(args, name, None)}


def check_model_pin(model_root):
    pin = subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip()
    if pin != MODEL_PIN:
        raise ValueError("model binding revision differs from selected pin")
    return pin


def model_binding(model_root):
    pin = check_model_pin(model_root)
    dirty = run_command(["git", "-C", str(model_root), "diff", "--quiet", "HEAD", "--", "npu_model"]).returncode
    if dirty:
        raise ValueError("model binding has tracked source changes or failed diff")
    return {"revision": pin, "tracked_source_diff_exit": dirty, "imported_source_sha256": model_sources(model_root),
        "curve_sha256": sha256(model_root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json")}


def check_bindings(args, model_root, receipt, model_before, input_before):
    model_after, input_after = model_binding(model_root), input_sources(args)
    receipt["model"]["imported_source_sha256"] = model_before["imported_source_sha256"]
    receipt["source_bookends"] = {"model_after": model_after, "input_before": input_before, "input_after": input_after,
        "unchanged": model_before == model_after and input_before == input_after}
    if not receipt["source_bookends"]["unchanged"]:
        raise AssertionError("source/model/input files changed during execution")


def execute_tile(program, initial, expected_bytes, output_offset, requested, out, repeat, max_cycles, seed=42):
    import torch
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.hardware import Core
    from npu_model.hardware.dma import DmaExecutionUnit
    from npu_model.logging.logger import Logger, LoggerConfig

    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=seed)
    logger = Logger(LoggerConfig(filename=str(out / f"perfetto-{repeat}.json")))
    core = None
    try:
        core = Core(config=config, logger=logger)
        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
        core.load_program(program)
        dma = next(exu for exu in core.exus if isinstance(exu, DmaExecutionUnit))
        events = Counter()
        with (out / f"cycles-{repeat}.jsonl").open("w") as trace:
            while not core.is_finished() and core.cycle_count < max_cycles:
                core.tick()
                event = dma.last_cycle
                a_fire = event.a
                if a_fire:
                    events["store_bytes" if a_fire[2] else "load_bytes"] += 32
                if event.d is not None:
                    events["response_beats"] += 1
                for name in ("vmem_read", "vmem_write"):
                    access = getattr(event, name)
                    if access and access[-1]:
                        events[name + "_bytes"] += 32
                if core.stall_reason:
                    events["stall_" + core.stall_reason] += 1
                if core.last_cycle["s1_fire"] or a_fire or event.d is not None or core.arch_state.halted:
                    trace.write(json.dumps({**core.last_cycle, "dma_a": a_fire, "dma_d": event.d}) + "\n")
        # Observe completion before any model stop/flush helper can modify state.
        state = core.arch_state
        if not state.halted or state.halt_reason != "ecall":
            raise AssertionError("timeout, implicit IMEM end or unexpected halt")
        if any(exu.has_in_flight for exu in core.exus) or any(state.check_flag(ch) for ch in range(8)):
            raise AssertionError("ECALL observed before engines/DMA drained")
        actual = bytes(state.read_dram(program.dram_base, len(initial)).tolist())
        actual_output = actual[output_offset : output_offset + len(expected_bytes)]
        (out / f"actual-bf16-halves-{repeat}.bin").write_bytes(actual_output)
        if (actual[:output_offset] != initial[:output_offset]
            or actual[output_offset + len(expected_bytes) :] != initial[output_offset + len(expected_bytes) :]):
            raise AssertionError("input or DRAM guard changed")
        if any(events[key] != size for key, size in requested.items()):
            raise AssertionError("observed DMA traffic differs from requested transfers")
        result = {"repeat": repeat, "initialization_seed": seed, "cycles": core.cycle_count,
            "issued_instructions": core.total_completed, "halt_reason": state.halt_reason, "halt_pc_word": state.execute_pc,
            "drained_before_halt_observation": True, "all_output_bytes_exact": True, "input_and_guards_preserved": True,
            "observed": dict(events), "engine_busy_cycles": {exu.name: exu.busy_cycles for exu in core.exus}}
        if actual_output != expected_bytes:
            differences = [index for index, (x, y) in enumerate(zip(actual_output, expected_bytes)) if x != y]
            result["all_output_bytes_exact"] = False
            raise OutputMismatch(differences, result)
    finally:
        logger.close()
        if core is not None:
            core.close()
    return result


def execute_negative_controls(source, initial, expected_bytes, output_offset, requested, out, max_cycles, assemble):
    if "vmatmul.acc.mxu0" not in source:
        raise ValueError("continuation control requires K>=64")
    last = source.rfind("vmatmul.acc.mxu0")
    mutations = {"omitted_B_transpose": source.replace("vtrpose.xlu m1, m1", "nop"),
        "reset_at_final_K_tile": source[:last] + source[last:].replace("vmatmul.acc.mxu0", "vmatmul.mxu0", 1),
        "wrong_oracle": source}
    results = {}
    for name, changed_source in mutations.items():
        destination = out / ("control-" + name)
        destination.mkdir()
        changed_expected = (bytes([expected_bytes[0] ^ 1]) + expected_bytes[1:] if name == "wrong_oracle"
            else expected_bytes)
        (destination / "program.S").write_text(changed_source)
        (destination / "expected.bin").write_bytes(changed_expected)
        changed = assemble(io.StringIO(changed_source))
        words = changed.assemble()
        (destination / "program.bin").write_bytes(b"".join(struct.pack("<I", word) for word in words))
        try:
            execute_tile(changed, initial, changed_expected, output_offset, requested, destination, 0, max_cycles)
        except OutputMismatch as error:
            actual = (destination / "actual-bf16-halves-0.bin").read_bytes()
            if name == "wrong_oracle" and actual != expected_bytes:
                raise AssertionError("wrong-oracle control changed correct execution") from error
            results[name] = {"status": "expected_output_mismatch", "message": str(error), "execution": error.context,
                "program_sha256": sha256(destination / "program.bin"),
                "expected_sha256": sha256(destination / "expected.bin"),
                "actual_sha256": sha256(destination / "actual-bf16-halves-0.bin")}
            write_json(destination / "receipt.json", results[name])
        else:
            raise AssertionError(f"negative control did not discriminate: {name}")
    return results


def encode_operands(arrays, recipe="unit"):
    import numpy as np
    import torch
    from npu_model.configs.numerics import RtlNumerics

    if recipe not in RECIPES:
        raise ValueError("unsupported target arithmetic recipe")
    encoded, converted = {}, {}
    conversion_stats = {}
    for key in ("A", "B"):
        value = arrays[key]
        bf16 = np.array([f32_to_bf16_bits(float(v)) for v in value.flat], dtype=np.uint16).reshape(value.shape)
        tensor = torch.from_numpy(bf16.copy()).view(torch.bfloat16)
        if not torch.equal(torch.from_numpy(value.copy()).to(torch.bfloat16).view(torch.uint16), tensor.view(torch.uint16)):
            raise AssertionError("independent BF16 cast differs from selected host cast")
        exponent = 0 if recipe == "unit" else tensor_scale_exponent(bf16.flat)
        scale_code = 127 + exponent
        independently_packed = np.array([pack_bf16(int(v), scale_code) for v in bf16.flat],
            dtype=np.uint8).reshape(value.shape)
        packed = RtlNumerics.to_fp8(tensor, scale_code).view(torch.uint8).numpy().copy()
        if not np.array_equal(independently_packed, packed):
            raise AssertionError("independent FP8 encoding disagrees with selected model")
        converted[key] = (bf16.astype(np.uint32) << 16).view(np.float32)
        encoded[key] = packed
        conversion_stats[key] = {"scale_exponent": exponent, "scale_code": scale_code,
            "bf16_cast": metrics(converted[key], value),
            "flushed_nonzero": int(np.sum((converted[key] != 0) & ((packed & 127) == 0))),
            "source_abs_gt448": int(np.sum(np.abs(converted[key]) > 448)),
            "scaled_abs_gt448": int(np.sum(np.abs(np.ldexp(converted[key].astype(np.float64), -exponent)) > 448)),
            "encoded_max_magnitude": int(np.sum((packed & 127) == 0x7E))}
    return encoded, converted, conversion_stats


def configure_model(model_root):
    pin = subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip()
    if pin != MODEL_PIN:
        raise ValueError("select the pinned model revision")
    if run_command(["git", "-C", str(model_root), "diff", "--quiet", "HEAD", "--", "npu_model"]).returncode:
        raise ValueError("selected model source has tracked changes")
    sys.path.insert(0, str(model_root))
    import npu_model
    import torch
    from npu_model.configs import isa_definition  # Register the new assembler ISA.
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.util.converter import input_to_program

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model_sources(model_root)
    return npu_model, torch, input_to_program, pin


def run(args, out, receipt):
    import numpy as np

    model_root = safe_path(args.model_root)
    npu_model, torch, input_to_program, pin = configure_model(model_root)
    model_before = model_binding(model_root)
    receipt["model_binding_before"] = model_before
    input_before = input_sources(args)
    arrays, manifest = load_bundle(args)
    a, b, native = validate_arrays(arrays, manifest.get("source_dtype"))
    np.savez(out / "source.npz", **arrays)
    write_json(out / "source-manifest.json", manifest)
    source_files = ({"npz": sha256(args.input), "manifest": sha256(args.manifest)} if not args.smoke
        else {"npz": sha256(out / "source.npz")})
    encoded, converted, conversion_stats = encode_operands(arrays, args.recipe)
    rescale_exponent = sum(item["scale_exponent"] for item in conversion_stats.values())
    np.savez(out / "encoded.npz", **encoded)
    raw_expected = [[mxu0_dot(encoded["A"][row], encoded["B"][:, col]) for col in range(32)] for row in range(32)]
    expected = ([[rescale_bf16(bits, rescale_exponent) for bits in row] for row in raw_expected]
        if args.recipe == "tensor-pow2" else raw_expected)
    expected_bytes = output_halves(expected)
    controls = mutation_witnesses(encoded, raw_expected)
    if args.smoke and len(controls) != 2:
        raise AssertionError("smoke must distinguish transpose and accumulator-reset errors")
    (out / "expected-bf16-halves.bin").write_bytes(expected_bytes)
    k_tiles = a.shape[1] // 32
    input_bytes = b"".join(encoded["A"][:, tile * 32 : (tile + 1) * 32].tobytes() for tile in range(k_tiles))
    input_bytes += b"".join(encoded["B"][tile * 32 : (tile + 1) * 32, :].tobytes() for tile in range(k_tiles))
    output_offset = len(input_bytes) + 32
    initial = input_bytes + b"\xa5" * 32 + b"\xcd" * 2048 + b"\x5a" * 32
    source = assembly(k_tiles, output_offset, rescale_exponent if args.recipe == "tensor-pow2" else None)
    (out / "program.S").write_text(source)
    program = input_to_program(io.StringIO(source))
    words = program.assemble()
    if len(words) > 32768 or words[-1] != 0x73:
        raise ValueError("program does not fit IMEM or lacks explicit ECALL")
    (out / "program.bin").write_bytes(b"".join(struct.pack("<I", word) for word in words))
    (out / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
    (out / "initial-dram.bin").write_bytes(initial)
    requested = {"load_bytes": 2 * k_tiles * 1024, "store_bytes": 2048}
    receipt.update({"recipe": RECIPES[args.recipe], "source_inputs": source_files, "shape": [32, a.shape[1], 32],
        "input_scale_code": 127 if args.recipe == "unit" else None,
        "input_scale_codes": {key: value["scale_code"] for key, value in conversion_stats.items()},
        "input_converter": "VPU_FP8Pack_" + args.recipe, "output_rescale_exponent": rescale_exponent,
        "output_rescale_execution": "VLI_ALL_and_VMUL_BF16" if args.recipe == "tensor-pow2" else "none",
        "native_fidelity_claim": False, "model_input": "new_assembler_decoded_instruction_objects",
        "raw_image_execution_claim": False, "instruction_words": len(words), "requested_dma": requested,
        "conversion": conversion_stats, "independent_mutation_witnesses": controls,
        "independent_mutation_witnesses_domain": "raw_MXU0_accumulator_before_rescale",
        "memory": {"dram_base": ATLAS_DRAM_BASE, "input_bytes": len(input_bytes), "output_offset": output_offset,
        "guard_bytes_each": 32, "total_dram_bytes": len(initial), "vmem_staging_bytes": 4096},
        "model": {"revision": pin, "module_file": str(safe_path(npu_model.__file__)), "torch_version": torch.__version__,
        "numerics": "rtl", "backend": "curve", "curve": "ee290sim_vcs_probe.json",
        "curve_sha256": sha256(model_root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json")}})
    runs = []
    for repeat in range(args.repeats):
        runs.append(execute_tile(program, initial, expected_bytes, output_offset, requested, out, repeat, args.max_cycles,
            42 + repeat))
        receipt["runs"] = runs
    if args.negative_controls or args.smoke:
        receipt["executed_negative_controls"] = execute_negative_controls(source, initial, expected_bytes, output_offset,
            requested, out, args.max_cycles, input_to_program)
    target = np.array([[float(bf16_value(bits)) for bits in row] for row in expected], dtype=np.float32)
    native_from_bf16 = converted["A"].astype(np.float64) @ converted["B"].astype(np.float64)
    receipt["diagnostic_errors"] = {"target_vs_native_C": metrics(target, native),
        "bf16_operands_float64_matmul_vs_native_C": metrics(native_from_bf16, native)}
    np.savez(out / "result.npz", target_C=target, expected_bits=np.array(expected, dtype=np.uint16),
        raw_accumulator_bits=np.array(raw_expected, dtype=np.uint16))
    check_bindings(args, model_root, receipt, model_before, input_before)
    receipt["status"] = "passed_model_only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--input")
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="unit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cycles", type=int, default=500000)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    out = safe_path(args.out)
    if out.exists():
        parser.error("output destination must be new")
    out.mkdir(parents=True)
    start = time.monotonic()
    receipt = {"status": "failed", "claim": "Atlas model only; no hardware or native-quality verdict", "command": sys.argv,
        "cwd": os.getcwd(), "python": sys.executable, "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS")}, "python_version": platform.python_version(), "recipe": RECIPES[args.recipe],
        "runner_sha256": sha256(__file__), "oracle_sha256": sha256(kernel_library.source_path("atlas_tile_arithmetic.py"))}
    try:
        run(args, out, receipt)
        receipt["code_after"] = {"runner_sha256": sha256(__file__),
            "oracle_sha256": sha256(kernel_library.source_path("atlas_tile_arithmetic.py"))}
        if any(receipt[key] != value for key, value in receipt["code_after"].items()):
            receipt["status"] = "failed"
            raise AssertionError("runner/oracle changed during execution")
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["artifacts"] = {path.name: sha256(path) for path in out.iterdir() if safe_path(path).is_file()
            and path.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "shape": receipt["shape"],
        "cycles": [run["cycles"] for run in receipt["runs"]], "out": str(out)}))


if __name__ == "__main__":
    main()
