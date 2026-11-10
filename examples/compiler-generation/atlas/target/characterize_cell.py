"""Compare selected E4M3FMA source against independent exact-rational semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from merlin.development.testgen.cell_probe import characterize
from merlin.integrations.specir import importable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-source", required=True)
    parser.add_argument("--specir-root", required=True)
    parser.add_argument("--circt-opt", required=True)
    parser.add_argument("--verilator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", choices=("multiply", "accumulate"), default="accumulate")
    args = parser.parse_args()
    with importable(args.specir_root):
        from specir.oracle.refmodel import PortSpec, build_fp_fma

    reference = build_fp_fma(
        {
            "compute": "fma",
            "round_mode": "rne",
            "arith": {"subnormal": "flush_to_zero", "subnormal_operands": ["a", "b"]},
        },
        {},
        [PortSpec("io_a", "fp8_e4m3", "a"), PortSpec("io_b", "fp8_e4m3", "b"), PortSpec("io_addend16", "bf16", "c")],
        [PortSpec("io_out16", "bf16", "d")],
    )
    if reference is None:
        raise ValueError("selected independent model cannot implement the declared arithmetic")
    # Exponent 15/special values are not part of this finite-normal selection.
    operands = [sign | (exp << 3) | mant for sign in (0, 128) for exp in range(1, 15) for mant in range(8)]
    operands += [0, 128]
    cases = []
    for a in operands:
        for b in operands:
            inputs = {"io_a": a, "io_b": b, "io_addend16": 0}
            cases.append({"inputs": inputs, "expected": reference(inputs)})
    if args.domain == "accumulate":
        rng = random.Random(0)
        for _ in range(8192):
            inputs = {
                "io_a": rng.choice(operands),
                "io_b": rng.choice(operands),
                "io_addend16": rng.choice((0, 0x8000)) | (rng.randrange(50, 161) << 7) | rng.randrange(128),
            }
            cases.append({"inputs": inputs, "expected": reference(inputs)})
    reference_path = Path(args.specir_root).resolve() / "specir/oracle/refmodel.py"
    result = characterize(
        hw_source=args.hw_source,
        module="E4M3FMA",
        inputs={"io_a": 8, "io_b": 8, "io_addend16": 16},
        outputs={"io_out16": 16},
        cases=cases,
        circt_opt=args.circt_opt,
        verilator=args.verilator,
        output=args.output,
        reference={
            "engine": "specir.oracle.refmodel.build_fp_fma",
            "source": str(reference_path),
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(reference_path.parent.glob("*.py"))
            },
            "case_producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        domain={
            "operand_exponents": [1, 14],
            "signed_zero": True,
            "zero_addend": "all finite-normal operand pairs",
            "nonzero_addend_samples": 8192 if args.domain == "accumulate" else 0,
            "addend_exponent_range": [50, 160],
            "sample_seed": 0,
            "excluded": ["operand exponent 15", "BF16 addend exponent zero nonzero-mantissa", "NaN", "Inf"],
            "expected": "exact product plus exact addend, one BF16 RNE rounding",
        },
    )
    print(json.dumps({"status": result["status"], "vectors": result["vector_count"], "output": args.output}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
