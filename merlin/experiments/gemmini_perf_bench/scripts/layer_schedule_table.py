#!/usr/bin/env python3
"""Measure the target's reference SCHEDULE of every matmul-shaped layer on GSIM, beside the library.

A one-shot, end-to-end validation of the ``merlin.sched`` path: kernel IR -> G0 static check -> C ->
the SAME layer harness the library programs use (operand blob, output placement, protocol, digest) ->
GSIM, digest-checked against the numerics contract. For every 1x1 stride-1 unpadded conv (issued as
its NHWC matmul) and every FC of a per-tensor model, the target's reference schedule is built, refused
if G0 reports anything, measured, and put beside the library reference row for the same layer.

    MERLIN_OUT_ROOT=/path/to/merlin-output \\
      python merlin/experiments/gemmini_perf_bench/scripts/layer_schedule_table.py \\
      --scales out/artifacts/perf-bench/gemmini/spec_model_scales_20260914/spec_scales.json \\
      --reference out/artifacts/perf-bench/gemmini/layer_library_reference_20260914/layer_library_reference.json \\
      --target gemmini --design-pin gemmini_gsim_model_serialclk --slots 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_library_table import unique_layers  # noqa: E402

SYMBOL = "mk_layer"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matmul_view(spec: dict) -> dict | None:
    """The layer as ``C[m][n] = A[m][k] @ B[k][n] + D[n]`` over the harness's own operand names."""
    if spec["op"] == "conv2d" and spec["kernel"] == 1 and spec["stride"] == 1 and spec["padding"] == 0:
        b, n, ci, co = spec["batch"], spec["in_dim"], spec["in_channels"], spec["out_channels"]
        return {
            "m": b * n * n,
            "n": co,
            "k": ci,
            "names": {"a": "input", "b": "weights", "d": "bias", "c": "output"},
            "shapes": {"a": (b, n, n, ci), "b": (1, 1, ci, co), "d": (co,), "c": (b, n, n, co)},
        }
    if spec["op"] == "matmul":
        m, n, k = spec["m"], spec["n"], spec["k"]
        return {
            "m": m,
            "n": n,
            "k": k,
            "names": {"a": "a", "b": "b", "d": "d", "c": "output"},
            "shapes": {"a": (m, k), "b": (k, n), "d": (n,), "c": (m, n)},
        }
    return None


def measure_one(
    sig: str,
    row: dict,
    *,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    contract_obj,
    max_cycles: int,
    timeout_s: float,
) -> dict:
    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import expected_digest, pack_operands
    from merlin.runtime.backends import base
    from merlin.sched.check.static import check_kernel
    from merlin.sched.codegen import EMITTER_VERSION, emit_c_function
    from merlin.sched.ir import TensorArg
    from merlin.targetgen import gsim_emulator

    backend = base.get_backend(target)
    iset = backend.sched_instruction_set()
    label = "S" + _sha(sig.encode())[:12]
    spec = {**row["spec"], "label": label, "scale": row["scale"], "seed": 1, "protocol": "warm_then_measured"}
    view = matmul_view(spec)
    elem, acc = iset.facts["elem_dtype"], iset.facts["acc_dtype"]
    ops = {
        r: TensorArg(view["names"][r], view["shapes"][r], acc if r == "d" else elem, "write" if r == "c" else "read")
        for r in ("a", "b", "d", "c")
    }
    kernel = backend.sched_matmul_reference(
        name="mm_" + label,
        m=view["m"],
        n=view["n"],
        k=view["k"],
        operands=ops,
        relu=bool(spec["relu"]),
        scale=float(spec["scale"]),
    )
    errors = check_kernel(kernel, iset)
    if errors:
        return {"sig": sig, "error": "G0: " + "; ".join(errors[:5]), "kernel_digest": kernel.digest()}
    blob, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
    kernel_c = emit_c_function(kernel, iset, symbol=SYMBOL)
    source = backend.render_schedule_layer(
        spec, offsets=offsets, kernel_c=kernel_c, symbol=SYMBOL, arg_names=[t.name for t in kernel.args]
    )
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    key = LayerKey(
        target=target,
        design_pin=design_pin,
        engine_sha256=engine["binary_sha256"],
        group_signature=sig,
        contract_digest=contract_obj.digest(),
        schedule_digest=_sha((kernel.digest() + _sha(source.encode()) + _sha(blob)).encode()),
        emitter_digest=_sha((iset.digest() + EMITTER_VERSION + backend.library_layer_emitter_digest()).encode()),
        harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION + "+sched",
        protocol="warm_then_measured",
    )
    cached = cache.get(key)
    if cached is not None:
        return {"sig": sig, "cached": True, **cached}
    wd = workroot / key.digest()[:16]
    wd.mkdir(parents=True, exist_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    (wd / "kernel.mk").write_text(kernel.text(), encoding="utf-8")
    built = build_program([wd / "layer.c"], wd, target=target, max_loaded_bytes=None)
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    recs = [r for r in run.records if r.label == label]
    payload = {
        "spec": spec,
        "count": row["count"],
        "macs": row["macs"],
        "kernel_digest": kernel.digest(),
        "tiles": dict(kernel.attrs).get("tiles"),
        "elf_sha256": built.elf_sha256,
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 2),
        "load_path": run.load_path,
        "cycles": recs[0].cycles if len(recs) == 1 else None,
        "digest": recs[0].fields.get("digest") if len(recs) == 1 else None,
        "digest_expected": expected_digest(spec, contract_obj),
        "engine": engine,
    }
    payload["numerics"] = (
        "exact" if payload["digest"] is not None and payload["digest"] == payload["digest_expected"] else "mismatch"
    )
    (wd / operands_name(backend)).unlink(missing_ok=True)
    if not run.completed or len(recs) != 1 or payload["numerics"] != "exact":
        payload["stderr_tail"] = run.stderr_tail
        payload["stdout_tail"] = run.stdout_tail
        return {"sig": sig, "cached": False, **payload}
    return {"sig": sig, "cached": False, **cache.put(key, payload)}


def operands_name(backend) -> str:
    return backend.LIBRARY_LAYER_OPERAND_BLOB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="layer_library_reference.json")
    ap.add_argument("--target", required=True)
    ap.add_argument("--design-pin", required=True)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--max-cycles", type=int, default=100_000_000)
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench import ReceiptCache
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract

    backend = base.get_backend(args.target)
    contract_obj = contract("per_tensor_readout_v1", backend.readout_facts())
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["contract_digest"] != contract_obj.digest():
        print("reference table was measured under a different contract", file=sys.stderr)
        return 2
    lib = {r["sig"]: r for r in reference["rows"]}
    rows = {
        sig: row
        for sig, row in unique_layers(json.loads(args.scales.read_text(encoding="utf-8"))).items()
        if matmul_view(row["spec"]) is not None
    }
    out_dir = args.out_dir or (
        artifacts_dir()
        / "perf-bench"
        / args.target
        / f"layer_schedule_table_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ReceiptCache(artifacts_dir() / "perf-bench" / args.target / "layer_cache")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.slots)) as pool:
        futs = {
            pool.submit(
                measure_one,
                sig,
                row,
                target=args.target,
                design_pin=args.design_pin,
                workroot=out_dir / "work",
                cache=cache,
                contract_obj=contract_obj,
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
            print(
                json.dumps({k: res.get(k) for k in ("sig", "tiles", "cycles", "numerics", "cached", "error")}),
                flush=True,
            )
    order = {sig: i for i, sig in enumerate(rows)}
    results.sort(key=lambda r: order.get(r["sig"], 1 << 30))
    lines = [
        "| layer | count | tiles (I,J,K) | schedule cycles | library cycles | sched / lib | numerics |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    tot_s = tot_l = 0
    complete = True
    for r in results:
        row, ref = rows[r["sig"]], lib.get(r["sig"])
        cyc = r.get("cycles")
        ok = cyc is not None and r.get("numerics") == "exact"
        complete &= ok
        lc = ref["cycles"] if ref else None
        if ok:
            tot_s += cyc * row["count"]
            tot_l += (lc or 0) * row["count"]
        lines.append(
            f"| `{r['sig']}` | {row['count']} | {r.get('tiles')} | "
            f"{f'{cyc:,}' if cyc is not None else 'FAILED'} | {f'{lc:,}' if lc else '-'} | "
            f"{f'{cyc / lc:.3f}' if ok and lc else '-'} | "
            f"{r.get('numerics') or str(r.get('error'))[:80]} |"
        )
    lines.append(
        f"\nSum over measured layers (x multiplicity): schedule {tot_s:,} vs library {tot_l:,}"
        + (f" = {tot_s / tot_l:.3f}x" if tot_l else "")
        + ("" if complete else " -- INCOMPLETE")
    )
    doc = {
        "schema": "layer_schedule_table_v1",
        "target": args.target,
        "design_pin": args.design_pin,
        "contract_digest": contract_obj.digest(),
        "inputs": {"scales": _sha(args.scales.read_bytes()), "reference": _sha(args.reference.read_bytes())},
        "complete": complete,
        "sum_schedule_cycles": tot_s,
        "sum_library_cycles": tot_l,
        "rows": results,
    }
    (out_dir / "layer_schedule_table.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "layer_schedule_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
