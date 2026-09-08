#!/usr/bin/env python3
"""Independent integrity and IMEM checks for the saved before/after inventory."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from test_compact_loops import _assert_control_targets


ROOT = Path(__file__).resolve().parent
IMEM_WORDS = 32768


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


before = json.loads((ROOT / "before_unique_shapes.json").read_text(encoding="utf-8"))
after_root = ROOT / "after_inventory"
after = json.loads((after_root / "unique_shapes.json").read_text(encoding="utf-8"))
summary = json.loads((after_root / "summary.json").read_text(encoding="utf-8"))
layers = [
    json.loads(line)
    for line in (after_root / "layers.jsonl").read_text(encoding="utf-8").splitlines()
]

before_by_id = {row["kernel_id"]: row for row in before}
after_by_id = {row["kernel_id"]: row for row in after}
assert before_by_id.keys() == after_by_id.keys()
assert len(after) == summary["unique_kernel_shapes"] == 28
assert len(layers) == summary["physical_contraction_layers"] == 391
assert all(row["status"] == "accepted" for row in after)
assert all(row["instruction_memory_fit"] is True for row in after)
assert max(row["instruction_words"] for row in after) == 32458 < IMEM_WORDS
assert Counter(row["instruction_memory_fit"] for row in layers) == {True: 391}

decoded_backward_edges = 0
for row in after:
    payload = gzip.decompress((after_root / row["assembly_gzip"]).read_bytes())
    assert len(payload) == row["assembly_file_bytes"]
    assert digest(payload) == row["assembly_sha256"]
    words = sum(line.lstrip().startswith(b".word") for line in payload.splitlines())
    assert words == row["instruction_words"]
    encoded_words = [
        int(line.split()[1], 16)
        for line in payload.decode("utf-8").splitlines()
        if line.lstrip().startswith(".word")
    ]
    decoded_backward_edges += _assert_control_targets(encoded_words)

recovered = [
    kernel_id for kernel_id, row in after_by_id.items()
    if before_by_id[kernel_id]["instruction_words"] > IMEM_WORDS
    and row["instruction_words"] <= IMEM_WORDS
]
verdict = {
    "ok": True,
    "claim": "compile/code-size only; no simulator or numeric-correctness claim",
    "imem_words": IMEM_WORDS,
    "before": {
        "fitting_unique_shapes": sum(row["instruction_words"] <= IMEM_WORDS for row in before),
        "fitting_physical_layers": sum(
            row["layer_occurrences"] for row in before
            if row["instruction_words"] <= IMEM_WORDS
        ),
        "maximum_words": max(row["instruction_words"] for row in before),
        "total_unique_program_words": sum(row["instruction_words"] for row in before),
    },
    "after": {
        "fitting_unique_shapes": len(after),
        "fitting_physical_layers": len(layers),
        "maximum_words": max(row["instruction_words"] for row in after),
        "minimum_imem_margin_words": IMEM_WORDS - max(row["instruction_words"] for row in after),
        "total_unique_program_words": sum(row["instruction_words"] for row in after),
    },
    "recovered_unique_shapes": len(recovered),
    "recovered_kernel_ids": sorted(recovered),
    "decoded_in_range_backward_edges": decoded_backward_edges,
}
(ROOT / "validation.json").write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(verdict, indent=2, sort_keys=True))
