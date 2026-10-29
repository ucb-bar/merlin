"""Native upstream MLIR comparisons against separately authored valid SCF references."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path

from lean_overrank_cases import actual, fixture, reference

from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.differential import Tools, ValueType, execute_mlir

CASES = [
    ((2, 3), 4, 3),
    ((1, 2, 1), 3, 2),
    ((1, 1, 2, 1), 3, 1),
    ((1, 1), 1, 1),
    ((2, 1), 0, 3),
    ((0, 2), 3, 2),
    ((2, 1), 3, 0),
]


def bits(number, kind):
    return int.from_bytes(struct.pack("<f" if kind == "f32" else "<d", number), "little")


def compare(batch, k, n, kind, compiler_python, work):
    work.mkdir(parents=True, exist_ok=True)
    before, after, count = actual(fixture(batch, k, n, kind))
    assert count == 1
    (work / "before-invalid.mlir").write_text(before)
    (work / "after.mlir").write_text(after)
    (work / "reference.mlir").write_text(reference(batch, k, n, kind))
    shapes = [(*batch, k), (k, n), (*batch, n)]
    types = [ValueType(kind, shape) for shape in shapes]
    generator = random.Random(5301)
    samples = []
    choices = [-3.0, -0.0, 0.0, 0.125, 1.0, 2.0, 3.0]
    for _ in range(4):
        samples.append([[bits(generator.choice(choices), kind) for _ in range(math.prod(shape))] for shape in shapes])
    # Finite, signed zero, subnormal, infinity and NaN observations remain raw bit patterns.
    width = int(kind[1:])
    inf = 0x7F800000 if width == 32 else 0x7FF0000000000000
    special = [0, 1 << (width - 1), 1, inf, inf | (1 << (22 if width == 32 else 51)), inf | 17]
    samples.append([[special[i % len(special)] for i in range(math.prod(shape))] for shape in shapes])
    # Cancellation makes the sequential order observable. The old destination is nonzero.
    large = 2.0 ** (25 if kind == "f32" else 54)
    samples.append(
        [
            [
                bits(([large, 1.0, -large][i % 3] if arg == 0 else 1.0 if arg == 1 else 123.0), kind)
                for i in range(math.prod(shape))
            ]
            for arg, shape in enumerate(shapes)
        ]
    )
    tools = Tools(compiler_python, Path("unused"), Path("unused"))
    observations = {}
    for label in ("after", "reference"):
        observations[label] = execute_mlir(
            work / f"{label}.mlir", samples, work / label, tools, inputs=types, outputs=[types[-1]], function="subject"
        )
    left, right = [observations[label]["results"] for label in ("after", "reference")]
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]

    def nan(value):
        value = int(value)
        fraction = 23 if width == 32 else 52
        exponent = 8 if width == 32 else 11
        return (value >> fraction) & ((1 << exponent) - 1) == (1 << exponent) - 1 and value & ((1 << fraction) - 1) != 0

    material = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(left, right, strict=True))
        if any(x != y and not (nan(x) and nan(y)) for x, y in zip(a, b, strict=True))
    ]
    report = {
        "batch": batch,
        "k": k,
        "n": n,
        "format": kind,
        "samples": len(samples),
        "inputs": samples,
        "results": left,
        "reference_results": right,
        "bit_mismatches": mismatches,
        "non_nan_mismatches": material,
        "native_comparison_contract": "exact bits except NaN payload/sign; NaN classification only",
        "source_is_invalid_mlir": True,
        "intent_approved": False,
        "files": [_file_ref(work / name) for name in ("before-invalid.mlir", "after.mlir", "reference.mlir")],
        "native": {
            label: {key: value for key, value in data.items() if key != "results"}
            for label, data in observations.items()
        },
    }
    (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not material, report
    return report
