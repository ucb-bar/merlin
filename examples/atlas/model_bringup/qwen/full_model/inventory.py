"""Annotate the model export with Qwen session phases and checkpoint identity."""

import argparse
import importlib.util
import json
from pathlib import Path

from download_checkpoint import identify


def annotate(library, export, checkpoint, prefill, decode_steps):
    spec = importlib.util.spec_from_file_location(
        "atlas_export_inventory", library / "coverage/export_ops.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    records = json.loads((export / "nodes.json").read_text())
    step = -1
    for row in records:
        if row["op"] == "aten.embedding.default":
            step += 1
        row.update(step=max(step, 0), phase="prefill" if step <= 0 else "decode")
    if step != decode_steps:
        raise RuntimeError(f"Export has {step} decode steps; expected {decode_steps}")
    document = json.loads((export / "inventory.json").read_text())
    ordered = exporter.calls(records)
    actual = {
        name: sum(c["phase"] == name for c in ordered) for name in ("prefill", "decode")
    }
    expected = dict(prefill=28 * 7 + 1, decode=(28 * 7 + 1) * decode_steps)
    if actual != expected:
        raise RuntimeError(f"Incomplete model inventory: {actual}, expected {expected}")
    lengths = [
        r["outputs"][0][0][1] for r in records if r["op"] == "aten.embedding.default"
    ]
    if lengths != [prefill] + [1] * decode_steps:
        raise RuntimeError(f"Unexpected prefill/decode geometry: {lengths}")
    document.update(
        ops=exporter.aggregate(records),
        calls=ordered,
        attention=exporter.attention(records),
    )
    document["sources"]["checkpoint"] = identify(checkpoint)
    document["session"] = dict(
        prefill=prefill,
        decode_steps=decode_steps,
        captured_dtype="float32",
        attention="eager",
        real_checkpoint=True,
        projection_counts=actual,
    )
    (export / "nodes.json").write_text(json.dumps(records) + "\n")
    (export / "inventory.json").write_text(json.dumps(document, indent=2) + "\n")
    fixture = {
        k: v
        for k, v in document.items()
        if k not in ("argv", "export_seconds", "calls", "attention")
    }
    (export.parent / "qwen.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        f"Verified full Qwen inventory: {actual}; {len(document['attention'])} attention invocations",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=2)
    args = parser.parse_args()
    annotate(
        args.library, args.export, args.checkpoint, args.prefill, args.decode_steps
    )


if __name__ == "__main__":
    main()
