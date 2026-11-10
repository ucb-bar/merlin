"""Characterize the selected integer cell, not an entire Gemmini operation."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    args = parser.parse_args()
    with importable(args.specir_root):
        from specir.oracle.refmodel import PortSpec, build_mac_dot_step

    reference = build_mac_dot_step(
        {"compute": "mac", "reduction": {"internal_precision": "int20"}},
        {"reference": {"accum": "int20", "combine": "wrap"}},
        [PortSpec("io_in_a", "int8", "a"), PortSpec("io_in_b", "int8", "b"), PortSpec("io_in_c", "int32", "c")],
        [PortSpec("io_out_d", "int20", "d")],
    )
    cases = []
    for a in range(256):
        for b in range(256):
            inputs = {"io_in_a": a, "io_in_b": b, "io_in_c": 0}
            cases.append({"inputs": inputs, "expected": reference(inputs)})
    boundaries = [0, 1, -1, (1 << 19) - 1, 1 << 19, -(1 << 19), -(1 << 19) - 1, (1 << 31) - 1, -(1 << 31)]
    for c in boundaries:
        for a in (0, 1, 127, 128, 255):
            for b in range(256):
                inputs = {"io_in_a": a, "io_in_b": b, "io_in_c": c & 0xFFFFFFFF}
                cases.append({"inputs": inputs, "expected": reference(inputs)})
    reference_path = Path(args.specir_root).resolve() / "specir/oracle/refmodel.py"
    result = characterize(
        hw_source=args.hw_source,
        module="MacUnit",
        inputs={"io_in_a": 8, "io_in_b": 8, "io_in_c": 32},
        outputs={"io_out_d": 20},
        cases=cases,
        circt_opt=args.circt_opt,
        verilator=args.verilator,
        output=args.output,
        reference={
            "engine": "specir.oracle.refmodel.build_mac_dot_step",
            "source": str(reference_path),
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(reference_path.parent.glob("*.py"))
            },
            "case_producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        domain={
            "zero_addend": "all 65536 operand raw-bit pairs",
            "accumulator_boundaries": boundaries,
            "boundary_operand_a": [0, 1, 127, 128, 255],
            "boundary_operand_b": "all 256 raw-bit values",
            "expected": "signed multiply-add wrapping to 20 bits",
        },
    )
    print(json.dumps({"status": result["status"], "vectors": result["vector_count"], "output": args.output}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
