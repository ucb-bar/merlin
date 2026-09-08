#!/usr/bin/env python3
"""Write the reproducible SmolVLA full-graph partition inventory."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import normalize_xdsl_parser_compat, parse_verified  # noqa: E402
from mlir_oot.full_graph import inventory_full_graph  # noqa: E402


source = CAPTURE.read_text(encoding="utf-8")
_, rewrites = normalize_xdsl_parser_compat(source)
report = inventory_full_graph(parse_verified(source))
report["capture"] = {
    "path": str(CAPTURE),
    "bytes": CAPTURE.stat().st_size,
    "sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
    "parser_compat_rewrites": rewrites,
}
(ROOT / "full_capture_partition_inventory.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "logical_regions": report["logical_regions"],
            "physical_contractions": report["physical_contractions"],
            "regions_by_partition_class": report["regions_by_partition_class"],
            "candidate_island_count": report["candidate_island_count"],
        },
        indent=2,
        sort_keys=True,
    )
)

