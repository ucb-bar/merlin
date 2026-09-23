#!/usr/bin/env python3
"""Measure every unique layer of a per-tensor model with the target's own library kernels, on GSIM.

This is the same-design, per-layer library reference the performance loop compares a schedule
against: one row per unique (op, shape) of the model, each measured as a standalone program on the
target's pinned cycle-accurate engine (warm, then measured), with the receipt cached under a
``LayerKey`` so an unchanged layer is never re-measured.

Input is the scale table ``extract_qdq_scales.py`` writes (it carries each conv's weight shape,
attributes, inferred input/output shapes and requant scale). Output is ``layer_library_table.json``
plus a Markdown table beside it.

    MERLIN_OUT_ROOT=/path/to/merlin-output \\
      python merlin/experiments/gemmini_perf_bench/scripts/layer_library_table.py \\
      --scales out/artifacts/perf-bench/gemmini/spec_model_scales_20260914/spec_scales.json \\
      --target gemmini --design-pin gemmini_gsim_model_serialclk --slots 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_layers(scales: dict) -> "OrderedDict[str, dict]":
    """One spec per unique (op, shape, relu) with its multiplicity and MAC count."""
    rows: "OrderedDict[str, dict]" = OrderedDict()
    for node in scales["nodes"]:
        if node["op"] == "Conv":
            co, ci, kh, kw = node["weight_shape"]
            n, _, h, w = node["in_shape"]
            _, _, oh, ow = node["out_shape"]
            if kh != kw or h != w or oh != ow:
                raise SystemExit(f"non-square conv not supported by this sweep: {node['name']}")
            stride, pad = node["attributes"]["strides"][0], node["attributes"]["pads"][0]
            spec = {
                "op": "conv2d",
                "batch": n,
                "in_dim": h,
                "in_channels": ci,
                "out_channels": co,
                "kernel": kh,
                "stride": stride,
                "padding": pad,
                "relu": bool(node["relu"]),
            }
            macs = n * oh * ow * co * ci * kh * kw
        elif node["op"] == "Gemm":
            j, k = node["weight_shape"]  # ONNX Gemm weight is [N, K] with transB=1
            spec = {"op": "matmul", "m": 1, "n": j, "k": k, "relu": bool(node["relu"])}
            macs = j * k
        else:
            continue
        sig = ":".join(f"{k}={spec[k]}" for k in sorted(spec))
        if sig not in rows:
            rows[sig] = {"spec": spec, "count": 0, "macs": macs, "scale": node["fused_requant_scale"], "names": []}
        rows[sig]["count"] += 1
        rows[sig]["names"].append(node["name"])
    return rows


def measure_one(
    sig: str,
    row: dict,
    *,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    contract_obj,
    emitter_digest: str,
    harness_version: str,
    max_cycles: int,
    timeout_s: float,
) -> dict:
    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import expected_digest, pack_operands
    from merlin.runtime.backends import base

    contract_digest = contract_obj.digest()

    backend = base.get_backend(target)
    label = "L" + _sha(sig.encode())[:12]
    spec = {**row["spec"], "label": label, "scale": row["scale"], "seed": 1, "protocol": "warm_then_measured"}
    blob, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
    source = backend.render_library_layer(spec, offsets=offsets)
    from merlin.targetgen import gsim_emulator

    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    key = LayerKey(
        target=target,
        design_pin=design_pin,
        engine_sha256=engine["binary_sha256"],
        group_signature=sig,
        contract_digest=contract_digest,
        schedule_digest=_sha(source.encode() + b"\0" + _sha(blob).encode()),
        emitter_digest=emitter_digest,
        harness_version=harness_version,
        protocol="warm_then_measured",
    )
    cached = cache.get(key)
    if cached is not None:
        return {"sig": sig, "cached": True, **cached}
    wd = workroot / key.digest()[:16]
    wd.mkdir(parents=True, exist_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    # Operands are in the image; the engine's load backdoor makes that free, so no size guard.
    built = build_program([wd / "layer.c"], wd, target=target, max_loaded_bytes=None)
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    if run.load_path != "backdoor":
        raise RuntimeError(
            "target declares no load backdoor; an embedded-operand program would spend "
            "its run loading. Measure it with a fill-on-device harness instead."
        )
    recs = [r for r in run.records if r.label == label]
    payload = {
        "spec": spec,
        "count": row["count"],
        "macs": row["macs"],
        "names": row["names"],
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 2),
        "engine_cycles": run.finish.cycles if run.finish else None,
        "cycles": recs[0].cycles if len(recs) == 1 else None,
        "digest": recs[0].fields.get("digest") if len(recs) == 1 else None,
        "stdout_sha256": run.stdout_sha256,
        "engine": engine,
    }
    # Numerics: the printed digest must equal the digest of the contract's expected output.
    payload["digest_expected"] = expected_digest(spec, contract_obj)
    payload["numerics"] = (
        "exact" if payload["digest"] is not None and payload["digest"] == payload["digest_expected"] else "mismatch"
    )
    if not run.completed or len(recs) != 1 or payload["numerics"] != "exact":
        payload["stderr_tail"] = run.stderr_tail
        return {"sig": sig, "cached": False, "key": key.to_dict(), **payload}
    return {"sig": sig, "cached": False, **cache.put(key, payload)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--design-pin", required=True, help="hardware_pins.yaml artifact name of the engine's modelled design"
    )
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--max-cycles", type=int, default=200_000_000)
    ap.add_argument("--timeout-s", type=float, default=7200)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--route-1x1",
        choices=("conv", "matmul"),
        default="conv",
        help="issue 1x1 stride-1 unpadded convs through tiled_conv_auto (conv) or as the NHWC "
        "matmul the library's own ResNet-50 uses (matmul)",
    )
    ap.add_argument("--only-rerouted", action="store_true", help="measure only the rows --route-1x1 changed")
    args = ap.parse_args(argv)

    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench import ReceiptCache
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract

    backend = base.get_backend(args.target)
    if not hasattr(backend, "render_library_layer"):
        print(f"target {args.target!r} declares no library-layer renderer", file=sys.stderr)
        return 2
    readout = backend.readout_facts()
    contract_obj = contract("per_tensor_readout_v1", readout)
    contract_digest = contract_obj.digest()
    emitter_digest = backend.library_layer_emitter_digest()
    harness_version = backend.LIBRARY_LAYER_HARNESS_VERSION

    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    rows = unique_layers(scales)
    # The route is set AFTER the signature is formed: it changes how the layer is issued (so the
    # rendered source, hence the schedule digest and cache key), never which layer it is.
    rerouted = set()
    for sig, row in rows.items():
        s = row["spec"]
        if (
            args.route_1x1 == "matmul"
            and s["op"] == "conv2d"
            and s["kernel"] == 1
            and s["stride"] == 1
            and s["padding"] == 0
        ):
            s["route"] = "matmul"
            rerouted.add(sig)
    if args.only_rerouted:
        rows = OrderedDict((sig, row) for sig, row in rows.items() if sig in rerouted)
    out_dir = args.out_dir or (
        artifacts_dir()
        / "perf-bench"
        / args.target
        / f"layer_library_table_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ReceiptCache(artifacts_dir() / "perf-bench" / args.target / "layer_cache")
    workroot = out_dir / "work"
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.slots)) as pool:
        futs = {
            pool.submit(
                measure_one,
                sig,
                row,
                target=args.target,
                design_pin=args.design_pin,
                workroot=workroot,
                cache=cache,
                contract_obj=contract_obj,
                emitter_digest=emitter_digest,
                harness_version=harness_version,
                max_cycles=args.max_cycles,
                timeout_s=args.timeout_s,
            ): sig
            for sig, row in rows.items()
        }
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one failed layer is a row, not a crash
                res = {"sig": futs[fut], "error": f"{type(exc).__name__}: {exc}"}
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("sig", "cycles", "wall_seconds", "cached", "error")}), flush=True)
    order = {sig: i for i, sig in enumerate(rows)}
    results.sort(key=lambda r: order.get(r["sig"], 1 << 30))
    peak = backend.mac_per_cycle_peak()
    lines = [
        "| layer shape | route | count | MACs | cycles | roofline | % of peak | total cycles | numerics |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    total = 0
    complete = True
    for r in results:
        cyc = r.get("cycles")
        row = rows[r["sig"]]
        route = row["spec"].get("route", "conv") if row["spec"]["op"] == "conv2d" else "matmul"
        if cyc is None or r.get("numerics") != "exact":
            complete = False
            why = r.get("error") or r.get("numerics") or "no record"
            lines.append(
                f"| `{r['sig']}` | {route} | {row['count']} | {row['macs']:,} | "
                f"{cyc if cyc is not None else 'FAILED'} | | | | {str(why)[:60]} |"
            )
            continue
        roof = -(-row["macs"] // peak)
        total += cyc * row["count"]
        lines.append(
            f"| `{r['sig']}` | {route} | {row['count']} | {row['macs']:,} | {cyc:,} | "
            f"{roof:,} | {100.0 * roof / cyc:.1f}% | {cyc * row['count']:,} | exact |"
        )
    summary = {
        "schema": "layer_library_table_v1",
        "target": args.target,
        "design_pin": args.design_pin,
        "scales": {"path": str(args.scales), "sha256": _sha(args.scales.read_bytes())},
        "contract_digest": contract_digest,
        "emitter_digest": emitter_digest,
        "harness_version": harness_version,
        "mac_per_cycle_peak": peak,
        "complete": complete,
        "sum_layer_cycles": total if complete else None,
        "rows": results,
    }
    (out_dir / "layer_library_table.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (out_dir / "layer_library_table.md").write_text(
        "\n".join(lines)
        + f"\n\nSum over layers (x multiplicity): {total:,} cycles"
        + ("" if complete else " -- INCOMPLETE, see FAILED rows")
        + "\n"
    )
    print(f"wrote {out_dir}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
