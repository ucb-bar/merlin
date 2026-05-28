#!/usr/bin/env python3
"""Materialize HTA island input captures from real CPU flow-runner captures.

`merlin-dispatch-flow-runner --capture-dispatch-io-dir` captures each CPU
dispatch's logical buffer-view arguments as raw arena files. HTA conv islands
consume a sliced tensor from one of those arenas, transformed from the CPU
wrapper's signed CHW layout to the HTA wrapper's unsigned HWC layout.

This tool performs only that deterministic layout conversion. It does not
generate data, synthesize missing captures, or profile anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np


def _iter_islands(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for row in manifest.get("dispatches", {}).values():
        cell = row.get("qnn_hta")
        if isinstance(cell, dict):
            yield cell


def _source_arena(capture_dir: pathlib.Path, source_dispatch: str) -> pathlib.Path:
    return capture_dir / source_dispatch / "input_0.bin"


def _materialize_one(
    *,
    island: dict[str, Any],
    capture_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> str:
    source_dispatch = str(island["source_dispatch"])
    source = _source_arena(capture_dir, source_dispatch)
    if not source.is_file():
        return "missing-source-capture"

    shape_hwc = tuple(int(v) for v in island["input_shape_hwc"])
    if len(shape_hwc) != 3:
        return "unsupported-rank"
    h, w, c = shape_hwc
    input_bytes = int(island["input_bytes"])
    if h * w * c != input_bytes:
        return "shape-byte-mismatch"

    offset = int(island.get("input_offset", 0))
    raw = np.frombuffer(source.read_bytes(), dtype=np.int8)
    if offset < 0 or offset + input_bytes > raw.size:
        return "source-too-small"

    chw = raw[offset : offset + input_bytes].reshape(c, h, w)
    hwc_u8 = (np.transpose(chw, (1, 2, 0)).astype(np.int16) + 128).astype(np.uint8)

    dst_dir = out_dir / source_dispatch
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "input_hwc_u8.bin").write_bytes(hwc_u8.tobytes())
    return "ok"


def _call_rows(manifest: dict[str, Any], call_graph: dict[str, Any]) -> list[dict[str, Any]]:
    by_canonical: dict[str, list[str]] = {}
    for call, row in call_graph.get("dispatch_graph", {}).items():
        canonical = str(row.get("canonical_dispatch") or call)
        by_canonical.setdefault(canonical, []).append(call)
    rows: list[dict[str, Any]] = []
    for island in _iter_islands(manifest):
        canonical = str(island["source_dispatch"])
        for call in by_canonical.get(canonical, []):
            call_island = dict(island)
            call_island["canonical_dispatch"] = canonical
            call_island["source_dispatch"] = call
            rows.append(call_island)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hta-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--cpu-capture-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--call-graph-json",
        type=pathlib.Path,
        help="Optional call-site graph. When set, materialize " "one HTA input per matching dispatch call.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.hta_manifest.read_text())
    call_graph = json.loads(args.call_graph_json.read_text()) if args.call_graph_json else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    rows = []
    islands = _call_rows(manifest, call_graph) if call_graph else list(_iter_islands(manifest))
    for island in islands:
        status = _materialize_one(
            island=island,
            capture_dir=args.cpu_capture_dir,
            out_dir=args.out_dir,
        )
        summary[status] = summary.get(status, 0) + 1
        rows.append(
            {
                "source_dispatch": island.get("source_dispatch"),
                "canonical_dispatch": island.get("canonical_dispatch"),
                "status": status,
                "input_offset": island.get("input_offset"),
                "input_bytes": island.get("input_bytes"),
            }
        )

    report = {"summary": summary, "captures": rows}
    report_path = args.out_dir / "materialized_hta_captures.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
