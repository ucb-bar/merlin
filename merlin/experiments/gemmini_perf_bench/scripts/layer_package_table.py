#!/usr/bin/env python3
"""Measure every unique layer of a per-tensor model as compiled by an mlir_oot PACKAGE, on GSIM.

The companion of ``layer_library_table.py``: same layers (from the same scale table), same operand
bytes (``merlin.perf.layer_bench.reference``), same engine, same digest check -- but the layer is
compiled by a Merlin package (e.g. the latest phase-1 arm-4 submission, or a phase-2 loop output)
through its own manifest argv, lowered to an object with the target's toolchain, and linked into the
target's package-layer program. Rows are directly comparable with the library table's.

Each layer is expressed in the package's own input grammar (``merlin_iface``): a ``conv2d`` with the
layer's per-tensor requant scale and ReLU applied on the store path (``epilogue=[acc_scale(,relu)]``,
i8 output). The interface conv carries no bias, so the reference is computed with zero bias. The FC
is a ``matmul`` + ``commit``.

    MERLIN_OUT_ROOT=<repo>/out PYTHONPATH=merlin/python .venv/bin/python \\
      merlin/experiments/gemmini_perf_bench/scripts/layer_package_table.py \\
      --package out/runs/gemmini/capsule-bench/merlin_assisted/merlincirct_g3arm97_cohort_20260909_r2/submission \\
      --label phase1_g3arm97 --scales out/artifacts/perf-bench/gemmini/spec_model_scales_20260914/spec_scales.json \\
      --target gemmini --design-pin gemmini_gsim_model_serialclk --slots 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_library_table import unique_layers  # noqa: E402  (same layer set as the library table)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_digest(package: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(package.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(package)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _f32(value: float) -> str:
    return repr(float(np.float32(value)))


#: How a layer's readout is DECLARED to the package. ``requant_i8`` is the model contract the library
#: and schedule arms also compute (narrow the i32 accumulator through the store path's scale, and the
#: activation, into i8). ``full_i32`` declares the accumulator's own width with no epilogue at all --
#: a different function, whose digest is therefore NOT comparable with an i8 row, and which exists
#: because a backend may lower the two through entirely different command families.
READOUTS = ("requant_i8", "full_i32")


def iface_module(spec: dict, scale: float, target: str, *, readout: str = "requant_i8") -> tuple[str, dict]:
    """The layer in merlin_iface form, plus the map from interface tensors to reference operands."""
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r} (known: {READOUTS})")
    head = (
        f'module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "{target}", '
        f'merlin_iface.abi_version = "0.1"}} {{\n'
    )
    if readout == "full_i32":
        if spec["relu"]:
            raise ValueError("a full-width accumulator readout carries no activation stage")
        epi, out_dtype, out_scale = "[]", "i32", 1.0
    else:
        epi = '["acc_scale", "relu"]' if spec["relu"] else '["acc_scale"]'
        out_dtype, out_scale = "i8", scale
    if spec["op"] == "conv2d":
        b, n, ci, co = spec["batch"], spec["in_dim"], spec["in_channels"], spec["out_channels"]
        k, st, p = spec["kernel"], spec["stride"], spec["padding"]
        out = (n + 2 * p - k) // st + 1
        kk = k * k * ci
        body = (
            f'  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : tensor<{b}x{n}x{n}x{ci}xi8>\n'
            f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{kk}x{co}xi8>\n'
            f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : (tensor<{kk}x{co}xi8>) -> !merlin_iface.resident\n'
            f"  %Y0 = merlin_iface.conv2d %IFM, %W_res {{kernel = [{k}, {k}, {ci}, {co}], stride = [{st}, {st}], "
            f'padding = [{p}, {p}, {p}, {p}], dilation = [1, 1], name = "Y0", epilogue = {epi}, '
            f'output_dtype = "{out_dtype}", acc_scale = {_f32(out_scale)} : f32, layout = "nhwc"}} : '
            f"(tensor<{b}x{n}x{n}x{ci}xi8>, !merlin_iface.resident) -> "
            f"tensor<{b * out * out}x{co}x{out_dtype}>\n"
            f"  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n"
        )
        return head + body + "}\n", {"IFM": "input", "W": "weights"}
    m, nn, kk = spec["m"], spec["n"], spec["k"]
    body = (
        f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{kk}x{nn}xi8>\n'
        f'  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{m}x{kk}xi8>\n'
        f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{kk}x{nn}xi8>) -> !merlin_iface.resident\n'
        f"  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{m}x{kk}xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n"
        f'  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = {epi}, output_dtype = "{out_dtype}", '
        f"acc_scale = {_f32(out_scale)} : f32}} : (!merlin_iface.acc<i32>) -> tensor<{m}x{nn}x{out_dtype}>\n"
        f"  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n"
    )
    return head + body + "}\n", {"A0": "a", "W": "b"}


def pack_for_abi(cb: dict, arrays: dict, binding: dict) -> tuple[bytes, dict]:
    """Blob of every READ kernel argument, each zero-padded to the shape the command buffer declares."""
    from merlin.perf.layer_bench.reference import OPERAND_ALIGN

    blob, offsets = bytearray(), {}
    for arg in cb["kernel_abi"]["args"]:
        if arg.get("access") != "read":
            continue
        name = arg["tensor"]
        if name not in binding:
            raise RuntimeError(f"package reads an argument the harness cannot supply: {name}")
        decl = cb["tensors"][name]
        if decl["dtype"] != "i8":
            raise RuntimeError(f"unsupported read dtype {decl['dtype']} for {name}")
        src = np.asarray(arrays[binding[name]], dtype=np.int8)
        src = src.reshape(-1, src.shape[-1]) if len(decl["shape"]) == 2 else src
        shape = tuple(int(d) for d in decl["shape"])
        if len(shape) != src.ndim or any(s < t for s, t in zip(shape, src.shape)):
            raise RuntimeError(f"{name}: declared {shape} cannot hold operand {src.shape}")
        padded = np.zeros(shape, dtype=np.int8)
        padded[tuple(slice(0, t) for t in src.shape)] = src
        blob += b"\0" * ((-len(blob)) % OPERAND_ALIGN)
        offsets[name] = len(blob)
        blob += padded.tobytes()
    return bytes(blob), offsets


def measure_one(
    sig,
    row,
    *,
    package: Path,
    pkg_digest: str,
    label: str,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    obj_cache: Path,
    contract_obj,
    harness_version: str,
    max_cycles: int,
    timeout_s: float,
) -> dict:
    import yaml

    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import DIGEST_MASK, expected_output, fnv1a64_words, operand_arrays
    from merlin.runtime.backends import base
    from merlin.targetgen import gsim_emulator
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

    backend = base.get_backend(target)
    spec = {**row["spec"], "scale": row["scale"], "seed": 1, "bias_span": 0}
    mod, binding = iface_module(spec, row["scale"], target)
    wd = workroot / _sha((sig + pkg_digest).encode())[:16]
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "layer.iface.mlir").write_text(mod)
    manifest = yaml.safe_load((package / "manifest.yaml").read_text())
    tool = package / manifest["entrypoints"]["tool"]
    argv = [
        a.replace("{tool}", str(tool))
        .replace("{input_mlir}", str(wd / "layer.iface.mlir"))
        .replace("{output_json}", str(wd / "cb.json"))
        for a in manifest["commands"]["emit_analysis_bundle"]["argv"]
    ]
    env = {**os.environ, "MERLIN_PYTHON": sys.executable}
    t0 = time.monotonic()
    comp = subprocess.run(
        [sys.executable, *argv] if tool.suffix == ".py" or not os.access(tool, os.X_OK) else argv,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
    compile_s = time.monotonic() - t0
    if comp.returncode != 0 or not (wd / "cb.json").is_file():
        return {"sig": sig, "error": f"package compile rc={comp.returncode}: {comp.stderr[-600:]}"}
    cb = json.loads((wd / "cb.json").read_text())
    if cb.get("declined") or not cb.get("commands"):
        return {
            "sig": sig,
            "error": f"package declined: {json.dumps(cb.get('declined'))[:400]}",
            "commands": len(cb.get("commands") or []),
        }
    lowered = comp.stdout
    (wd / "lowered.mlir").write_text(lowered)
    arrays = dict(operand_arrays(spec))
    blob, offsets = pack_for_abi(cb, arrays, binding)
    out_name = cb["kernel_abi"]["outputs"][0]
    source = backend.render_package_layer(cb, offsets=offsets, label="P" + _sha(sig.encode())[:12], output=out_name)
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    key = LayerKey(
        target=target,
        design_pin=design_pin,
        engine_sha256=engine["binary_sha256"],
        group_signature=sig,
        contract_digest=contract_obj.digest(),
        schedule_digest=_sha(lowered.encode() + b"\0" + _sha(blob).encode() + source.encode()),
        emitter_digest=pkg_digest,
        harness_version=harness_version,
        protocol="warm_then_measured",
    )
    cached = cache.get(key)
    if cached is not None:
        return {"sig": sig, "cached": True, **cached}
    # The object is a function of the lowered MLIR, the target and the clang that built it -- and it is
    # by far the slowest step (minutes to over an hour for a fully unrolled layer) -- so it is cached on
    # exactly those, independent of the harness around it.
    from merlin.llvmlower import toolchain

    okey = _sha(lowered.encode() + b"\0" + target.encode() + b"\0" + str(toolchain.clang()).encode())
    cached_obj = obj_cache / f"{okey}.o"
    t1 = time.monotonic()
    if cached_obj.is_file():
        obj = cached_obj
    else:
        built_obj = llvm_mlir_to_object(lowered, wd / "obj", target=target)
        obj_cache.mkdir(parents=True, exist_ok=True)
        tmp = obj_cache / f".{okey}.{os.getpid()}.tmp"
        tmp.write_bytes(built_obj.read_bytes())
        os.replace(tmp, cached_obj)
        obj = cached_obj
    lower_s = time.monotonic() - t1
    # A fully unrolled layer leaves up to ~1.5 GB of MLIR/LLVM text behind; the object is cached and
    # the lowered text is identified by its digest in the receipt, so the copies go.
    import shutil

    shutil.rmtree(wd / "obj", ignore_errors=True)
    lowered_sha = _sha(lowered.encode())
    (wd / "lowered.mlir").unlink(missing_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source)
    # support_first: a fully unrolled kernel object can exceed the crt's +-1 MiB jal reach if it is
    # linked between _start and _init (measured: 1.26 MB .text for the smallest ResNet layer).
    built = build_program([wd / "layer.c", obj], wd, target=target, max_loaded_bytes=None, support_first=True)
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    recs = [r for r in run.records]
    expected = expected_output(spec, contract_obj)
    exp_digest = fnv1a64_words(np.ascontiguousarray(expected).tobytes()) & DIGEST_MASK
    payload = {
        "spec": spec,
        "count": row["count"],
        "macs": row["macs"],
        "names": row["names"],
        "package": str(package),
        "commands": len(cb["commands"]),
        "opcodes": [c.get("opcode") for c in cb["commands"]],
        "compile_seconds": round(compile_s, 1),
        "lower_seconds": round(lower_s, 1),
        "lowered_sha256": lowered_sha,
        "lowered_bytes": len(lowered.encode()),
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 1),
        "engine_cycles": run.finish.cycles if run.finish else None,
        "cycles": recs[0].cycles if len(recs) == 1 else None,
        "digest": recs[0].fields.get("digest") if len(recs) == 1 else None,
        "digest_expected": exp_digest,
        "load_path": run.load_path,
        "engine": engine,
    }
    payload["numerics"] = "exact" if payload["digest"] == exp_digest else "mismatch"
    if not run.completed or len(recs) != 1 or payload["numerics"] != "exact":
        payload["stderr_tail"] = run.stderr_tail
        return {"sig": sig, "cached": False, "key": key.to_dict(), **payload}
    return {"sig": sig, "cached": False, **cache.put(key, payload)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--label", required=True, help="short name for the compiler, e.g. phase1_g3arm97")
    ap.add_argument("--scales", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--design-pin", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated substrings; measure matching layers only")
    ap.add_argument("--max-cycles", type=int, default=400_000_000)
    ap.add_argument("--timeout-s", type=float, default=14400)
    args = ap.parse_args(argv)

    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench import ReceiptCache
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract

    backend = base.get_backend(args.target)
    contract_obj = contract("per_tensor_readout_v1", backend.readout_facts())
    package = args.package.resolve()
    pkg_digest = package_digest(package)
    rows = unique_layers(json.loads(args.scales.read_text()))
    if args.only:
        wanted = [s for s in args.only.split(",") if s]
        rows = {k: v for k, v in rows.items() if any(w in k for w in wanted)}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = artifacts_dir() / "perf-bench" / args.target / f"layer_package_table_{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ReceiptCache(artifacts_dir() / "perf-bench" / args.target / "layer_cache")
    harness_version = backend.LIBRARY_LAYER_HARNESS_VERSION + "+package+support_first"
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.slots)) as pool:
        futs = {
            pool.submit(
                measure_one,
                sig,
                row,
                package=package,
                pkg_digest=pkg_digest,
                label=args.label,
                target=args.target,
                design_pin=args.design_pin,
                workroot=out_dir / "work",
                cache=cache,
                obj_cache=artifacts_dir() / "perf-bench" / args.target / "layer_obj_cache",
                contract_obj=contract_obj,
                harness_version=harness_version,
                max_cycles=args.max_cycles,
                timeout_s=args.timeout_s,
            ): sig
            for sig, row in rows.items()
        }
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- a failed layer is a row, not a crash
                res = {"sig": futs[fut], "error": f"{type(exc).__name__}: {exc}"}
            results.append(res)
            print(
                json.dumps(
                    {k: res.get(k) for k in ("sig", "cycles", "numerics", "wall_seconds", "lower_seconds", "error")}
                ),
                flush=True,
            )
    summary = {
        "schema": "layer_package_table_v1",
        "label": args.label,
        "package": str(package),
        "package_digest": pkg_digest,
        "target": args.target,
        "design_pin": args.design_pin,
        "contract_digest": contract_obj.digest(),
        "harness_version": harness_version,
        "rows": results,
    }
    (out_dir / "layer_package_table.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out_dir}")
    return 0 if all(r.get("numerics") == "exact" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
