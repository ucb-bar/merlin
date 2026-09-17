#!/usr/bin/env python3
"""Run a whole lowered Voyager program on Voyager's own parameters and compare with Voyager's graph.

Reads a ``voyager_export.py`` output directory (``model.json``, ``tensor_files/``, and optionally
``io_extra.npz`` from ``--extra-inputs``). It lowers the program (``lower_model``) and executes it with
numpy (``execute_model``: accelerator schedules on the abstract executor, host ops on Voyager's
semantics), starting from the input and parameters Voyager itself dumped. It then compares every DRAM
tensor the program writes against the tensor Voyager's bufferized graph produced for the same input
(``compile(dump_tensors=True)``). Extra inputs compare the final output only.

Differences are expected only where the bridge's readout replaces Voyager's bf16 arithmetic
(concessions C1/C5): an int8 readout rounds acc * scale once, where Voyager dequantizes to bf16 and
re-quantizes through its tables. Anything else is a bug.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/whole_model_oracle.py --target <target> \
        --export out/runs/<target>/voyager-h2h/whole_model/<run>/voyager
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from merlin.baselines.voyager_ir import load_model, replay
from merlin.baselines.voyager_schedule import Geometry, HostOp, Schedule, execute_model, load_scalars, lower_model
from merlin.common.paths import runs_dir


def _load(tensor_dir: Path, name: str, shape: tuple, dtype: str) -> np.ndarray:
    values = np.fromfile(tensor_dir / f"{name}.bin", dtype="<f4").reshape(shape)
    return values if dtype in ("bfloat16", "float32") else values.astype(np.int64)


def compare(got: np.ndarray, want: np.ndarray) -> dict:
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    diff = np.abs(g - w)
    denom = float(np.sqrt((g * g).sum() * (w * w).sum()))
    return {
        "n": int(w.size),
        "exact_fraction": float((diff == 0).mean()),
        "within_one_fraction": float((diff <= 1).mean()),
        "cosine": float((g * w).sum() / denom) if denom else 1.0,
        "max_abs": float(diff.max()),
        "max_abs_ref": float(np.abs(w).max()),
        "max_rel": float(diff.max() / np.abs(w).max()) if np.abs(w).max() else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--export", required=True, type=Path, help="voyager_export.py --out directory")
    parser.add_argument(
        "--classes", type=int, default=None, help="logits counted for top-1 (the classifier's real width; default all)"
    )
    args = parser.parse_args(argv)

    export = args.export.resolve()
    manifest = json.loads((export / "manifest.json").read_text())
    config = manifest["config"]
    rows = config["pe_array_size"][0]
    geometry = Geometry(
        dim=rows,
        spad_rows=config["scratchpad_size"] // config["bank_width"],
        spad_row_bytes=config["bank_width"],
        acc_rows=config["accum_buffer_size"],
    )
    tensor_dir = export / "tensor_files"
    started = time.perf_counter()
    trace = replay(load_model(export / "model.json"))
    layers = lower_model(trace, geometry, load_scalars(tensor_dir))
    lowered_s = time.perf_counter() - started

    boxes = {box.node: box for box in list(trace.inputs) + list(trace.parameters)}
    base = {name: _load(tensor_dir, name, tuple(box.shape), box.dtype) for name, box in boxes.items()}
    (input_box,) = trace.inputs
    (output_box,) = trace.outputs
    started = time.perf_counter()
    tensors = execute_model(layers, dict(base), trace)
    executed_s = time.perf_counter() - started

    per_tensor = {}
    for layer in layers:
        for name in layer.writes:
            box = trace.allocations.get(name)
            if box is None or not (tensor_dir / f"{name}.bin").is_file() or name not in tensors:
                continue
            want = _load(tensor_dir, name, tuple(box.shape), box.dtype)
            kind = layer.kind if isinstance(layer.program, Schedule) else layer.program.target
            per_tensor[name] = {
                "layer": layer.name,
                "kind": kind,
                "dtype": box.dtype,
                **compare(np.asarray(tensors[name]), want),
            }

    # Local (teacher-forced) check: each Voyager layer runs alone on Voyager's own dumped inputs, so a
    # difference belongs to that layer rather than being inherited. The bridge's readout rounding
    # (C1/C5) may move an int8 output by one step; a larger step, or any difference in a host op that
    # follows Voyager's semantics, is a bug.
    groups: list[list] = []
    for layer in layers:
        if groups and groups[-1][0].name == layer.name:
            groups[-1].append(layer)
        else:
            groups.append([layer])
    local = {}
    for group in groups:
        produced = {name for layer in group for name in layer.writes}
        run = dict(base)
        for layer in group:
            for name in layer.reads:
                box = trace.allocations.get(name)
                if name in produced or name in run or box is None:
                    continue
                run[name] = _load(tensor_dir, name, tuple(box.shape), box.dtype)
        out = execute_model(group, run, trace)
        for layer in group:
            for name in layer.writes:
                box = trace.allocations.get(name)
                if box is None or not (tensor_dir / f"{name}.bin").is_file():
                    continue
                kind = layer.kind if isinstance(layer.program, Schedule) else layer.program.target
                local[name] = {
                    "layer": layer.name,
                    "kind": kind,
                    "dtype": box.dtype,
                    **compare(np.asarray(out[name]), _load(tensor_dir, name, tuple(box.shape), box.dtype)),
                }
    summary: dict[str, dict] = {}
    for row in local.values():
        key = f"{row['kind']}->{row['dtype']}"
        s = summary.setdefault(
            key,
            {
                "tensors": 0,
                "min_cosine": 1.0,
                "max_abs": 0.0,
                "min_exact_fraction": 1.0,
                "min_within_one_fraction": 1.0,
            },
        )
        s["tensors"] += 1
        s["min_cosine"] = min(s["min_cosine"], row["cosine"])
        s["max_abs"] = max(s["max_abs"], row["max_abs"])
        s["min_exact_fraction"] = min(s["min_exact_fraction"], row["exact_fraction"])
        s["min_within_one_fraction"] = min(s["min_within_one_fraction"], row["within_one_fraction"])

    classes = args.classes or int(np.prod(output_box.shape[1:]))

    def final(got: np.ndarray, want: np.ndarray) -> dict:
        # The graph's own output may already be sliced to the classifier's width, where the IR's
        # DRAM output keeps the padding; compare over the width both carry.
        width = min(got.size, want.size)
        got, want = got.reshape(-1)[:width], want.reshape(-1)[:width]
        g, w = got[:classes], want[:classes]
        return {
            **compare(got, want),
            "top1_merlin": int(np.argmax(g)),
            "top1_voyager": int(np.argmax(w)),
            "top1_agree": bool(np.argmax(g) == np.argmax(w)),
            "top5_overlap": len(set(np.argsort(-g)[:5]) & set(np.argsort(-w)[:5])),
        }

    want = _load(tensor_dir, output_box.node, tuple(output_box.shape), output_box.dtype)
    inputs = [{"input": "dumped", **final(np.asarray(tensors[output_box.node]), want)}]
    extra = export / "io_extra.npz"
    if extra.is_file():
        data = np.load(extra)
        k = 0
        while f"extra_lowered_inputs_{k}" in data:
            run = dict(base)
            run[input_box.node] = (
                data[f"extra_lowered_inputs_{k}"]
                .reshape(input_box.shape)
                .astype(np.int64 if input_box.dtype not in ("bfloat16", "float32") else np.float32)
            )
            out = execute_model(layers, run, trace)[output_box.node]
            inputs.append({"input": f"extra_{k}", **final(np.asarray(out), data[f"extra_lowered_outputs_{k}"])})
            k += 1

    host = sorted({layer.program.target for layer in layers if isinstance(layer.program, HostOp)})
    doc = {
        "export": str(export),
        "voyager_commit": manifest.get("voyager_commit"),
        "layers": len(layers),
        "schedules": sum(isinstance(l.program, Schedule) for l in layers),
        "host_ops": host,
        "lower_s": round(lowered_s, 1),
        "execute_s": round(executed_s, 1),
        "classes": classes,
        "final": inputs,
        "local_summary": summary,
        "local": local,
        "per_tensor": per_tensor,
    }
    out_dir = (
        runs_dir() / args.target / "voyager-h2h" / "whole_model_oracle" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("per_tensor", "local")}, indent=1))
    for name, row in sorted(local.items(), key=lambda kv: -kv[1]["max_abs"])[:6]:
        print(
            f"local {name:34s} {row['kind']:34s} {row['dtype']:9s} max_abs={row['max_abs']} "
            f"exact={row['exact_fraction']:.4f} within1={row['within_one_fraction']:.4f}"
        )
    worst = sorted(per_tensor.items(), key=lambda kv: kv[1]["cosine"])[:8]
    for name, row in worst:
        print(
            f"{name:40s} {row['kind']:34s} cos={row['cosine']:.6f} exact={row['exact_fraction']:.4f} "
            f"within1={row['within_one_fraction']:.4f} max_rel={row['max_rel']:.4f}"
        )
    print(f"results: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
