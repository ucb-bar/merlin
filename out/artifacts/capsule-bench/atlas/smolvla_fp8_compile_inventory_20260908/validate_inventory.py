#!/usr/bin/env python3
"""Independent consistency checks for the saved compile inventory."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
unique = json.loads((ROOT / "unique_shapes.json").read_text(encoding="utf-8"))
layers = [json.loads(line) for line in (ROOT / "layers.jsonl").read_text(encoding="utf-8").splitlines()]
demands = [json.loads(line) for line in (ROOT / "demands.jsonl").read_text(encoding="utf-8").splitlines()]

assert len(demands) == summary["routing"]["total"] == 8570
assert Counter(row["bucket"] for row in demands) == Counter(summary["routing"]["by_bucket"])
assert len(layers) == summary["physical_contraction_layers"] == 391
assert sum(row["layer_occurrences"] for row in unique) == len(layers)
assert len(unique) == summary["unique_kernel_shapes"] == 28
assert all(row["status"] == "accepted" for row in unique)

by_id = {row["kernel_id"]: row for row in unique}
for layer in layers:
    emitted = by_id[layer["kernel_id"]]
    assert layer["compile_status"] == emitted["status"]
    assert layer["instruction_memory_fit"] == emitted["instruction_memory_fit"]

for row in unique:
    compressed = ROOT / row["assembly_gzip"]
    payload = gzip.decompress(compressed.read_bytes())
    assert len(payload) == row["assembly_file_bytes"]
    assert digest(payload) == row["assembly_sha256"]
    assert sum(line.lstrip().startswith(b".word") for line in payload.splitlines()) == row["instruction_words"]

verdict = {
    "ok": True,
    "demands": len(demands),
    "physical_layers": len(layers),
    "unique_shapes": len(unique),
    "emitter_accepted_unique_shapes": sum(row["status"] == "accepted" for row in unique),
    "imem_fitting_unique_shapes": sum(row["instruction_memory_fit"] is True for row in unique),
    "imem_fitting_physical_layers": sum(row["instruction_memory_fit"] is True for row in layers),
}
print(json.dumps(verdict, indent=2, sort_keys=True))
