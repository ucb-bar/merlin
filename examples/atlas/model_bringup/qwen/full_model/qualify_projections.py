"""Run full-K projection panels on actual tensors from the pretrained session.

The same library generator and staging path as MODEL-BENCH are used. This
changes the data factory to captured operands, and compares against the saved
full-model graph outputs as well as the independent target-arithmetic golden.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--atlas-bin", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [
        str(args.library),
        str(args.library / "coverage"),
        str(args.model_root),
    ]
    import atlas_kernels_path  # noqa: F401
    import bench_model
    import kernel_bench
    import torch
    from atlas_execution import Executor

    torch.set_num_threads(1)
    rows = []
    for name in ("first_gate_projection", "first_down_projection"):
        source = args.capture / name
        region = json.loads((source / "region.json").read_text())
        inputs = load_file(str(source / "inputs.safetensors"))
        activation = next(r for r in region["inputs"] if r["parameter"] is None)
        weight = next(r for r in region["inputs"] if r["parameter"] is not None)
        a = inputs[activation["name"]].reshape(-1, inputs[activation["name"]].shape[-1])
        w = np.ascontiguousarray(inputs[weight["name"]][:128].T)
        expected = load_file(str(source / "golden.safetensors"))["0"].reshape(
            a.shape[0], -1
        )[:, :128]
        p = dict(
            m=a.shape[0],
            k=a.shape[1],
            n=128,
            rows=a.shape[0],
            logical_k=a.shape[1],
            logical_n=128,
        )
        entry = dict(family="projection_gemm", key=name, params=p)
        original = bench_model.synthetic_projection
        # Bind actual operands at the library's documented staging seam. The
        # generator, compiler comparison, arithmetic oracle and simulator stay
        # unchanged, and byte hashes identify the substituted operands.
        bench_model.synthetic_projection = lambda params, rng: (
            a,
            w,
            kernel_bench.quantize(a),
            kernel_bench.quantize(w),
        )
        try:
            built = bench_model.build(entry, args.out, args.model_root)
        finally:
            bench_model.synthetic_projection = original
        identity = bench_model.program_identity(
            built, args.atlas_bin, "stream-verified"
        )
        execution = Executor(args.model_root).execute(
            built.source,
            built.arena,
            built.output_offset,
            built.output_bytes,
            max_instructions=50_000_000,
        )
        golden = built.golden()
        if execution.output != golden:
            raise RuntimeError(f"Independent target-arithmetic golden differs: {name}")
        cycles, arena = kernel_bench.timed_cycles(
            args.model_root, built.source, built.arena, max_cycles=20_000_000
        )
        timed = arena[built.output_offset : built.output_offset + built.output_bytes]
        if timed != execution.output:
            raise RuntimeError(f"Timed/functional outputs differ: {name}")
        actual = built.decode(execution.output)
        row = dict(
            name=name,
            shape=[p["m"], p["k"], p["n"]],
            source_parameter=weight["parameter"],
            source_graph_sha256=region["source_graph_sha256"],
            input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
            model_cycles=cycles,
            independent_golden_exact=True,
            timed_functional_exact=True,
            **kernel_bench.fidelity(actual, expected),
            backend=identity["backend"],
            calibration=built.meta,
            source="pretrained model graph boundary tensors",
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.out / "qualification.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
