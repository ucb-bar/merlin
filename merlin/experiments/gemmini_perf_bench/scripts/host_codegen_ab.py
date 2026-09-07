#!/usr/bin/env python3
"""Compare work-deletion features on one captured host model; never claim accelerator timing.

Builds the same prepared IR for host numerical replay and RVV object inspection. Variants are JSON
maps from labels to feature-name lists. All variants must preserve the baseline's output bytes.
Static counts cannot promote a target baseline without measured target timing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import _pbcommon  # noqa: F401
import numpy as np
from merlin.common.paths import artifacts_dir
from merlin.common.mlir_query import forward_signature
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower import qinner, toolchain, weight_prepack
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import build_host_shared, compile_ll
from merlin.llvmlower.passes_xdsl import preprocess_text_textual
from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE, lower_to_llvm_ir
from merlin.runtime.backends.zephyr_model import prepare_for_lowering
from merlin.runtime.dispatch_runtime import resolve_forward_args


def validate_buffers(signature: list, arrays: list) -> None:
    """Refuse ABI drift before calling native code (especially features adding trailing args)."""
    types = {"i1": "bool", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
             "f16": "float16", "bf16": "uint16", "f32": "float32", "f64": "float64"}
    if len(signature) != len(arrays):
        raise ValueError("prepared ABI and bound buffer counts differ")
    for index, ((shape, dtype), array) in enumerate(zip(signature, arrays)):
        if (dtype not in types or tuple(shape) != array.shape
                or np.dtype(types[dtype]) != array.dtype or not array.flags.c_contiguous):
            raise ValueError(f"buffer {index} disagrees with prepared ABI {shape}x{dtype}")


def probe(bundle: Path, root: Path, variants: dict, *, int8: bool, vlen: int) -> dict:
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a nonempty object of feature-name lists")
    if any(not isinstance(v, list) or any(not isinstance(f, str) for f in v)
           for v in variants.values()):
        raise ValueError("each variant must contain a list of feature names")
    root.resolve().relative_to(artifacts_dir().resolve())
    root.mkdir(parents=True, exist_ok=False)
    golden = np.load(bundle / "golden.npy")
    baseline = None
    report = {"schema": "merlin.host-codegen-ab.v1", "bundle": str(bundle),
              "int8_compute": int8, "vlen": vlen, "variants": {},
              "limits": "Host replay and static RVV objects only; no target speedup or quantization-accuracy claim."}
    for name, requested in variants.items():
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("variant labels must be safe path components")
        work = root / name
        work.mkdir()
        selected = weight_prepack.prepare_build_bundle(bundle, work, frozenset(requested))
        prepared, features = prepare_for_lowering(
            selected / "model.mlir", work, int8_compute=int8,
            features=frozenset(requested), vlen=vlen)
        counts = dict(collections.Counter(op.name for op in parse_mlir_file(prepared).walk()))
        upstream, _ = preprocess_text_textual(prepared.read_text())
        ll = work / "model.ll"
        ll.write_text(lower_to_llvm_ir(upstream, workdir=work, vectorize=True,
                                       transform_schedule=RVV_TRANSFORM_SCHEDULE, features=features))
        obj = compile_ll(ll, work / "model.rvv.o", extra_flags=(f"-march=rv64gcv_zvl{vlen}b",))
        dis = subprocess.run([str(toolchain.objdump()), "-d", "--no-show-raw-insn", str(obj)],
                             capture_output=True, text=True, check=True).stdout
        (work / "model.rvv.dis").write_text(dis)
        # Disassembly has an address field followed by mnemonic/operands. Labels have no hex
        # address field ending in ':'. Count instructions structurally, without opcode literals.
        mnemonics = collections.Counter()
        for line in dis.splitlines():
            address, sep, instruction = line.strip().partition(":")
            if sep and address and all(c in "0123456789abcdef" for c in address):
                tokens = instruction.split()
                if tokens:
                    mnemonics[tokens[0]] += 1
        so = build_host_shared(ll, work / "model_host.so")
        args = resolve_forward_args(selected)
        extra_args = qinner.plan_for_bundle(selected / "model.mlir")
        if extra_args:
            with np.load(selected / "extra.npz") as extra:
                args += qinner.resolve(extra, extra_args)
        output = np.zeros(golden.shape, dtype=golden.dtype)
        inputs, outputs = forward_signature(prepared)
        validate_buffers([*inputs, *outputs], args + [output])
        buffers = [(a.ctypes.data, list(a.shape)) for a in args + [output]]
        model = HostModel.load(str(so), n_args=len(buffers))
        model(buffers)
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{name}: nonfinite output")
        if baseline is None:
            baseline = output.copy()
        same = output.tobytes() == baseline.tobytes()
        np.save(work / "output.npy", output)
        row = {"requested_features": requested, "concrete_features": sorted(features),
               "prepared_op_counts": counts, "instructions": sum(mnemonics.values()),
               "vector_instructions": sum(n for m, n in mnemonics.items() if m.startswith("v")),
               "mnemonics": dict(mnemonics), "bit_identical_to_baseline": same,
               "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
               "llvm_sha256": hashlib.sha256(ll.read_bytes()).hexdigest(),
               "rvv_object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest()}
        report["variants"][name] = row
        (root / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"variant": name, **{k: row[k] for k in (
            "instructions", "vector_instructions", "bit_identical_to_baseline")}}), flush=True)
        if not same:
            raise ValueError(f"{name}: changed output bytes; variant is not promotable")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--vlen", type=int, required=True)
    parser.add_argument("--variants", type=json.loads, required=True)
    args = parser.parse_args()
    probe(args.bundle.resolve(), args.root.absolute(), args.variants, int8=args.int8, vlen=args.vlen)


if __name__ == "__main__":
    main()
