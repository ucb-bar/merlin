"""Probe a frozen, selected AtlasCore 32x32 program with richer FP8 data.

This qualifies only the fixed instruction stream, dimensions, memory layout, and
selected native ARC model named by --bundle. It does not qualify tails, batches,
the compiler, a whole SoC, or arbitrary PyTorch model captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from functools import cache
from pathlib import Path

from merlin.integrations.specir import importable


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(sign: int, exponent: int, mantissa: int) -> int:
    if sign not in (0, 1) or not 1 <= exponent <= 14 or not 0 <= mantissa <= 7:
        raise ValueError("probe operands must be finite-normal E4M3")
    return (sign << 7) | (exponent << 3) | mantissa


def _cases() -> dict[str, tuple[bytes, bytes]]:
    n = 32
    # Non-symmetric indexing exposes row/column and stride mistakes. Both dense
    # cases span signs, every mantissa, and multiple exponents; the wider case
    # spans the full declared finite-normal exponent range.
    mixed_a = bytes(
        0 if (i + 3 * k) % 11 == 0 else _code((i + k) % 2, 5 + (i + 2 * k) % 5, (3 * i + k) % 8)
        for i in range(n)
        for k in range(n)
    )
    mixed_b = bytes(
        0 if (2 * k + j) % 13 == 0 else _code((k + 2 * j) % 2, 5 + (3 * k + j) % 5, (k + 5 * j) % 8)
        for k in range(n)
        for j in range(n)
    )
    wide_a = bytes(_code((i + 2 * k) % 2, 1 + (i + 3 * k) % 14, (i + k) % 8) for i in range(n) for k in range(n))
    wide_b = bytes(
        _code((2 * k + j + 1) % 2, 1 + (5 * k + j) % 14, (3 * k + j) % 8) for k in range(n) for j in range(n)
    )
    a, b = bytearray(n * n), bytearray(n * n)
    # BF16 ulp at 1 is 1/128. 1/256 is a tie: even 1 rounds down;
    # odd (1+1/128) rounds up. Also exercise the negative even tie.
    a[0], a[1], a[2] = _code(0, 7, 0), _code(0, 3, 0), _code(0, 3, 0)
    b[0 * n + 0], b[0 * n + 1], b[0 * n + 2] = _code(0, 7, 0), _code(0, 4, 0), _code(0, 3, 0)
    b[1 * n + 0], b[1 * n + 2] = _code(0, 7, 0), _code(0, 3, 0)
    b[2 * n + 0], b[2 * n + 2] = _code(1, 7, 0), _code(1, 3, 0)
    return {
        "signed_mantissas": (mixed_a, mixed_b),
        "wide_exponents": (wide_a, wide_b),
        "bf16_ties": (bytes(a), bytes(b)),
    }


def _golden(a: bytes, b: bytes, dtypes, fp_reduce) -> bytes:
    fmt, acc = dtypes.FP8_E4M3, dtypes.BF16

    @cache
    def product(x: int, y: int) -> int:
        return dtypes.round_to_format(dtypes.decode_float_exact(x, fmt) * dtypes.decode_float_exact(y, fmt), acc, "rne")

    @cache
    def step(x: int, y: int) -> int:
        return fp_reduce([x, y], acc, order="index_sequential", cadence="per_step", rm="rne")

    matrix = [[0] * 32 for _ in range(32)]
    for i in range(32):
        for j in range(32):
            # The selected SA weight buffer stores (output column, K) rows.
            # Its wrapper transposes to (K, output column) PE lanes. Thus raw
            # row-major DRAM weight bytes describe B_transposed, not B.
            terms = [product(a[i * 32 + k], b[j * 32 + k]) for k in range(32)]
            value = terms[0]
            for term in terms[1:]:
                value = step(value, term)
            matrix[i][j] = value
    # The two BF16 MREG/DRAM output halves are C[:, :16], C[:, 16:].
    return b"".join(
        matrix[i][j].to_bytes(2, "little") for col in (0, 16) for i in range(32) for j in range(col, col + 16)
    )


def _native(bundle: Path, words: list[int], a: bytes, b: bytes) -> tuple[bytes, dict]:
    support = bundle / "support"
    sys.path.insert(0, str(support))
    old_cwd = Path.cwd()
    try:
        os.chdir(support)  # ModeLIR's package bootstrap reads its frozen DIM cache.
        from mlc.backends.cosim_atlas import run_program

        result = run_program(
            bundle / "model.so",
            bundle / "state.json",
            words,
            preload=[(0, a), (1024, b)],
            halt_signal="scalar/halt_now",
            max_cycles=20_000,
        )
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(support))
    counts = {
        "halted": result.halted,
        "cycles": result.cycles,
        "dma_reads": result.reads,
        "dma_writes": result.writes,
        "halt_reason": result.halt_reason,
    }
    if not result.halted or result.reads <= 0 or result.writes <= 0:
        raise ValueError(f"native program did not halt with real DMA: {counts}")
    return result.slave.captured(2048, 2048), counts


def _operand_coverage(a: bytes, b: bytes) -> dict:
    values = a + b
    nonzero = [x for x in values if x & 0x7F]
    return {
        "nonzero_elements": len(nonzero),
        "signs": sorted({x >> 7 for x in nonzero}),
        "normal_exponents": sorted({(x >> 3) & 0xF for x in nonzero}),
        "mantissas": sorted({x & 7 for x in nonzero}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--specir-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle, specir, output = (
        args.bundle.resolve(strict=True),
        args.specir_root.resolve(strict=True),
        args.output.resolve(),
    )
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    receipt = runpy.run_path(str(bundle / "producer.py"))["verify"](bundle)
    program = json.loads((bundle / "case/program_bundle.json").read_text())
    if (
        program.get("output") != {"base": 2048, "shape": [64, 16], "dtype": "torch.bfloat16"}
        or len(program.get("words", [])) != 36
        or [x.get("base") for x in program.get("inputs", [])] != [0, 1024]
        or receipt["case"]["compared_elements"] != 1024
    ):
        raise ValueError("saved native program ABI differs from fixed 32x32 probe")
    with importable(specir):
        from specir.oracle import dtypes
        from specir.oracle.refmodel import fp_reduce

    sources = {
        "native_bundle_observation": bundle / "observation.json",
        "selected_firrtl": bundle / "raw.fir",
        "selected_core_hw": bundle / "core.hw.mlir",
        "native_model": bundle / "model.so",
        "native_state": bundle / "state.json",
        "native_program": bundle / "case/program_bundle.json",
        "modeLIR_cosim": bundle / "support/mlc/backends/cosim_atlas.py",
        "modeLIR_core": bundle / "support/mlc/backends/cosim_core.py",
        "modeLIR_tilelink": bundle / "support/mlc/backends/protocols.py",
        "specir_dtypes": specir / "specir/oracle/dtypes.py",
        "specir_refmodel": specir / "specir/oracle/refmodel.py",
        "probe": Path(__file__),
    }
    output.mkdir(parents=True)
    report = {
        "schema": "merlin.atlas_native_numerical_probe.v1",
        "scope": "frozen selected AtlasCore ARC, fixed 32x32 E4M3 matmul instructions and BF16 output halves",
        "operand_abi": "raw A[i,k] and raw W[j,k], computing C[i,j] = sum_k A[i,k]*W[j,k]",
        "output_abi": "BF16 little-endian C[:,0:16] followed by C[:,16:32]",
        "limitations": [
            "native build not reproduced here",
            "not compiler qualification",
            "not tails or batches",
            "not whole SoC",
            "not Python/model capture",
        ],
        "source_paths": {name: str(path.resolve(strict=True)) for name, path in sorted(sources.items())},
        "source_sha256": {name: _sha(path) for name, path in sorted(sources.items())},
        "cases": {},
    }
    overall = True
    for name, (a, b) in _cases().items():
        golden = _golden(a, b, dtypes, fp_reduce)
        if name == "bf16_ties":
            first_three = [int.from_bytes(golden[2 * i : 2 * i + 2], "little") for i in range(3)]
            if first_three != [0x3F82, 0x3F80, 0xBF80]:
                raise AssertionError(f"BF16 odd/even tie construction changed: {first_three!r}")
        observed, counts = _native(bundle, program["words"], a, b)
        paths = {
            "a": output / f"{name}.a.bin",
            "b": output / f"{name}.b.bin",
            "golden": output / f"{name}.golden.bin",
            "observed": output / f"{name}.observed.bin",
        }
        for field, data in (("a", a), ("b", b), ("golden", golden), ("observed", observed)):
            paths[field].write_bytes(data)
        mismatches = [
            {
                "index": i,
                "expected": f"0x{int.from_bytes(golden[2 * i : 2 * i + 2], 'little'):04x}",
                "observed": f"0x{int.from_bytes(observed[2 * i : 2 * i + 2], 'little'):04x}",
            }
            for i in range(1024)
            if golden[2 * i : 2 * i + 2] != observed[2 * i : 2 * i + 2]
        ]
        passed = not mismatches
        overall &= passed
        report["cases"][name] = {
            "status": "passed" if passed else "mismatch",
            "compared_elements": 1024,
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches[:16],
            "native_observations": counts,
            "operand_coverage": _operand_coverage(a, b),
            **({"tie_outputs_bf16_hex": ["0x3f82", "0x3f80", "0xbf80"]} if name == "bf16_ties" else {}),
            "sha256": {field: _sha(path) for field, path in paths.items()},
        }
    report["status"] = "passed_fixed_program_domain" if overall else "numerical_mismatch"
    (output / "result.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "mismatches": {name: c["mismatch_count"] for name, c in report["cases"].items()},
            },
            sort_keys=True,
        )
    )
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
