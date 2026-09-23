#!/usr/bin/env python3
"""Measure a LIST OF GEMM SHAPES three ways -- reference schedule, vendor library, agent-generated
compiler package -- on GSIM.

``layer_library_table.py`` and ``layer_schedule_table.py`` answer the same question for the layers a
MODEL contains: they take a scale table and enumerate its unique layers. A shape from a paper is not a
layer of a model we hold, so neither script can be pointed at one. This script is those two scripts'
measurement bodies over an explicit ``MxNxK`` list instead of a model, so an externally-specified shape
can be measured on the same harness, under the same contract, with the same digest check.

All THREE arms are built from ONE spec, so they share the operand values, the output placement, the
requant scale, the protocol (warm, then measured) and the expected digest. Only the call differs:
the library arm issues ``tiled_matmul_auto``; the schedule arm issues the kernel
``sched_matmul_reference`` emits; the package arm issues the kernel an OUT-OF-TREE COMPILER PACKAGE
emitted for the same shape, expressed in that package's own declared input grammar and compiled by
its own manifest argv. That is what makes the ratios within-device, within-window ones that need no
cross-engine control -- the numbers come from programs that differ in exactly the thing compared.

A row is admitted only when its printed digest equals the digest of the contract's expected output,
computed off-device in exact integer arithmetic. A cycle count whose digest does not match is reported
as ``mismatch`` and is NOT put in the table as a result: it computed a different function.

THE BIAS, AND WHY ``--bias-span`` EXISTS. The library and schedule arms take a per-column i32 bias
operand; the interface GEMM a package is handed declares none. Left at its default the three arms
would compute two different functions and their digests would be incomparable by construction -- so
the package arm REFUSES unless the bias span is zero, which makes the shared bias operand all zeros
and the three arms one function again. Run the comparison with ``--bias-span 0``.

THE READOUT, AND WHY ``--package-readout`` EXISTS. ``requant_i8`` declares the model contract the
other two arms compute (narrow the i32 accumulator through the store path's scale into i8), and its
digest IS comparable with theirs. ``full_i32`` declares the accumulator's own width with no epilogue:
a different function, digest-checked against its own exact oracle and NOT comparable with an i8 row.
Both are reported because a backend may lower them through entirely different command families, and
which one a package was audited on decides what any earlier claim about it was actually about.

    MERLIN_OUT_ROOT=/path/to/merlin-output \\
      python merlin/experiments/gemmini_perf_bench/scripts/layer_shape_table.py \\
      --shape 512x512x512 --shape 12544x256x64 --bias-span 0 \\
      --arm schedule --arm library --arm package \\
      --package out/artifacts/targets/gemmini/<package>/compiler --package-readout requant_i8 \\
      --target gemmini --design-pin gemmini_gsim_model_serialclk --slots 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_package_table import READOUTS, iface_module, pack_for_abi, package_digest  # noqa: E402

SYMBOL = "mk_layer"

#: Arms this script can measure. ``package`` needs ``--package``; the other two are self-contained.
ARMS = ("schedule", "library", "package")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_shape(text: str) -> tuple[int, int, int]:
    """``"MxNxK"`` -> ``(M, N, K)``. Structural split; a malformed shape raises rather than defaulting."""
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise SystemExit(f"--shape {text!r} is not MxNxK")
    try:
        m, n, k = (int(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"--shape {text!r} has a non-integer extent") from exc
    if min(m, n, k) <= 0:
        raise SystemExit(f"--shape {text!r} has a non-positive extent")
    return m, n, k


def build_spec(shape: tuple[int, int, int], *, scale: float, relu: bool, bias_span: int) -> dict:
    m, n, k = shape
    return {"op": "matmul", "m": m, "n": n, "k": k, "relu": relu, "scale": scale, "bias_span": int(bias_span)}


#: Per-arm label prefix, so a record line names the arm that printed it.
_ARM_TAG = {"schedule": "S", "library": "L", "package": "P"}


def _package_kernel(
    spec: dict,
    *,
    package: Path,
    readout: str,
    target: str,
    workdir: Path,
    obj_cache: Path,
    contract_obj,
    timeout_s: float,
) -> dict:
    """Compile one GEMM with the PACKAGE and return everything the shared program needs to call it.

    The package is invoked only through its manifest argv -- it is never imported -- exactly as
    ``targetgen.oot_runner.load_package`` describes the command protocol. What comes back is the
    package's own command buffer, its lowered target artifact built into an object, the operand blob
    laid out for the ABI that buffer implies, and the digest its declared output must have.
    """
    import numpy as np
    import yaml

    from merlin.llvmlower import toolchain
    from merlin.perf.layer_bench.reference import (
        DIGEST_MASK,
        expected_digest,
        fnv1a64_words,
        matmul_accumulator,
        operand_arrays,
    )
    from merlin.runtime.backends import base
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

    backend = base.get_backend(target)
    if int(spec.get("bias_span", 0)) != 0:
        raise ValueError(
            "the interface GEMM handed to a package declares no bias operand, so with a non-zero bias "
            "span the package would compute a different function from the other two arms and the "
            "digests would be incomparable by construction. Rerun every arm with --bias-span 0."
        )
    mod, binding = iface_module(spec, float(spec["scale"]), target, readout=readout)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "layer.iface.mlir").write_text(mod, encoding="utf-8")
    manifest = yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8"))
    tool = package / manifest["entrypoints"]["tool"]
    # `emit_analysis_bundle` is the COMBINED entrypoint and is optional: a package predating it
    # carries the two separate commands instead, and oot_runner.py:449 already falls back that way.
    # Hardcoding the combined one made every such package unmeasurable with KeyError rather than
    # with a statement about the package -- which is how the FIRST phase-1 compiler went unmeasured.
    cmds = (
        ["emit_analysis_bundle"]
        if "emit_analysis_bundle" in manifest["commands"]
        else ["emit_command_buffer", "lower_target_to_llvm"]
    )
    missing = [c for c in cmds if c not in manifest["commands"]]
    if missing:
        raise RuntimeError(f"package manifest declares none of the lowering commands: missing {missing}")
    env = {**os.environ, "MERLIN_PYTHON": sys.executable}
    t0 = time.monotonic()
    comp = None
    for name in cmds:
        argv = [
            a.replace("{tool}", str(tool))
            .replace("{input_mlir}", str(workdir / "layer.iface.mlir"))
            .replace("{output_json}", str(workdir / "cb.json"))
            for a in manifest["commands"][name]["argv"]
        ]
        comp = subprocess.run(
            [sys.executable, *argv] if tool.suffix == ".py" or not os.access(tool, os.X_OK) else argv,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
        if comp.returncode != 0:
            raise RuntimeError(f"package {name} rc={comp.returncode}: {comp.stderr[-600:]}")
    compile_s = time.monotonic() - t0
    if not (workdir / "cb.json").is_file():
        raise RuntimeError(f"package produced no command buffer via {cmds}: {comp.stderr[-400:]}")
    cb = json.loads((workdir / "cb.json").read_text(encoding="utf-8"))
    if cb.get("declined") or not cb.get("commands"):
        raise RuntimeError(f"package declined: {json.dumps(cb.get('declined'))[:400]}")
    lowered = comp.stdout
    # A buffer may leave `kernel_abi` out; the pointer order is then a function of the commands, and
    # the backend resolves it from the SAME group parse its own emitter uses rather than this script
    # re-deriving the contract's order a second time.
    derive = getattr(backend, "kernel_abi_from_commands", None)
    if derive is None:
        raise RuntimeError(f"backend for {target!r} cannot resolve a command buffer's implied kernel ABI")
    cb["kernel_abi"] = derive(cb)
    blob, offsets = pack_for_abi(cb, dict(operand_arrays(spec)), binding)
    out_name = cb["kernel_abi"]["outputs"][0]
    out_dtype = cb["tensors"][out_name]["dtype"]
    if readout == "full_i32":
        if out_dtype != contract_obj.accumulator_dtype:
            raise RuntimeError(f"full-width readout declared {out_dtype}, not {contract_obj.accumulator_dtype}")
        ops = dict(operand_arrays(spec))
        acc = matmul_accumulator(ops["a"], ops["b"], ops["d"])
        width = int(contract_obj.accumulator_dtype[1:]) // 8
        wanted = acc.astype(f"<i{width}")
        if np.any(wanted.astype(np.int64) != acc):
            raise RuntimeError(f"accumulator does not fit {contract_obj.accumulator_dtype}")
        digest_expected = fnv1a64_words(np.ascontiguousarray(wanted).tobytes()) & DIGEST_MASK
    else:
        if out_dtype != contract_obj.output_dtype:
            raise RuntimeError(f"requant readout declared {out_dtype}, not {contract_obj.output_dtype}")
        digest_expected = expected_digest(spec, contract_obj)
    source = backend.render_package_layer(
        cb,
        offsets=offsets,
        label=str(spec["label"]),
        output=out_name,
        protocol=str(spec["protocol"]),
    )
    # The object is a function of the lowered artifact, the target and the clang that built it, and is
    # by far the slowest step, so it is cached on exactly those -- independent of the harness around it.
    okey = _sha(lowered.encode() + b"\0" + target.encode() + b"\0" + str(toolchain.clang()).encode())
    cached_obj = obj_cache / f"{okey}.o"
    t1 = time.monotonic()
    if not cached_obj.is_file():
        built_obj = llvm_mlir_to_object(lowered, workdir / "obj", target=target)
        obj_cache.mkdir(parents=True, exist_ok=True)
        tmp = obj_cache / f".{okey}.{os.getpid()}.tmp"
        tmp.write_bytes(built_obj.read_bytes())
        os.replace(tmp, cached_obj)
    lower_s = time.monotonic() - t1
    # A fully unrolled layer leaves hundreds of MB of MLIR/LLVM text behind; the object is cached and
    # the lowered text is identified by its digest in the receipt, so the copies go.
    shutil.rmtree(workdir / "obj", ignore_errors=True)
    return {
        "cb": cb,
        "source": source,
        "blob": blob,
        "object": cached_obj,
        "digest_expected": digest_expected,
        "lowered_sha256": _sha(lowered.encode()),
        "lowered_bytes": len(lowered.encode()),
        "commands": [c.get("opcode") for c in cb["commands"]],
        "abi_args": [a["tensor"] for a in cb["kernel_abi"]["args"]],
        "output_tensor": out_name,
        "output_dtype": out_dtype,
        "package_readout": readout,
        "compile_seconds": round(compile_s, 2),
        "lower_seconds": round(lower_s, 2),
    }


def measure_one(
    sig: str,
    spec_base: dict,
    *,
    arm: str,
    target: str,
    design_pin: str,
    workroot: Path,
    cache,
    contract_obj,
    max_cycles: int,
    timeout_s: float,
    protocol: str = "warm_then_measured",
    tiles: tuple[int, int, int] | None = None,
    package: Path | None = None,
    package_digest_value: str | None = None,
    package_readout: str = "requant_i8",
    obj_cache: Path | None = None,
) -> dict:
    """One (shape, arm) row: build the program, run it on GSIM, check the digest, cache the receipt."""
    from merlin.perf.layer_bench import LayerKey, build_program, run_on_gsim
    from merlin.perf.layer_bench.reference import expected_digest, pack_operands
    from merlin.runtime.backends import base
    from merlin.sched.check.static import check_kernel
    from merlin.sched.codegen import EMITTER_VERSION, emit_c_function
    from merlin.sched.ir import TensorArg
    from merlin.targetgen import gsim_emulator

    backend = base.get_backend(target)
    label = _ARM_TAG[arm] + _sha(f"{arm}:{sig}:{package_readout if arm == 'package' else ''}".encode())[:12]
    spec = {**spec_base, "label": label, "seed": 1, "protocol": protocol}
    # The SHARED operand blob and the SHARED oracle: every arm's operands are these bytes, and every
    # arm's cycle count is admitted only against this digest.
    blob, offsets = pack_operands(spec, accumulator_dtype=contract_obj.accumulator_dtype)
    digest_expected = expected_digest(spec, contract_obj)
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    extra: dict = {}
    objects: list[Path] = []
    support_first = False

    if arm == "library":
        source = backend.render_library_layer(spec, offsets=offsets)
        key = LayerKey(
            target=target,
            design_pin=design_pin,
            engine_sha256=engine["binary_sha256"],
            group_signature=f"library:{sig}",
            contract_digest=contract_obj.digest(),
            schedule_digest=_sha(source.encode() + b"\0" + _sha(blob).encode()),
            emitter_digest=backend.library_layer_emitter_digest(),
            harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION,
            protocol=protocol,
        )
        kernel = None
    elif arm == "schedule":
        iset = backend.sched_instruction_set()
        elem, acc = iset.facts["elem_dtype"], iset.facts["acc_dtype"]
        m, n, k = spec["m"], spec["n"], spec["k"]
        shapes = {"a": (m, k), "b": (k, n), "d": (n,), "c": (m, n)}
        names = {"a": "a", "b": "b", "d": "d", "c": "output"}
        ops = {
            r: TensorArg(names[r], shapes[r], acc if r == "d" else elem, "write" if r == "c" else "read")
            for r in ("a", "b", "d", "c")
        }
        kernel = backend.sched_matmul_reference(
            name="mm_" + label,
            m=m,
            n=n,
            k=k,
            operands=ops,
            relu=bool(spec["relu"]),
            scale=float(spec["scale"]),
            **({"tiles": tiles} if tiles else {}),
        )
        errors = check_kernel(kernel, iset)
        if errors:
            return {"sig": sig, "arm": arm, "error": "G0: " + "; ".join(errors[:5])}
        kernel_c = emit_c_function(kernel, iset, symbol=SYMBOL)
        source = backend.render_schedule_layer(
            spec, offsets=offsets, kernel_c=kernel_c, symbol=SYMBOL, arg_names=[t.name for t in kernel.args]
        )
        extra = {"kernel_digest": kernel.digest(), "tiles": dict(kernel.attrs).get("tiles")}
        key = LayerKey(
            target=target,
            design_pin=design_pin,
            engine_sha256=engine["binary_sha256"],
            group_signature=f"schedule:{sig}" + (f":tiles{tiles[0]}x{tiles[1]}x{tiles[2]}" if tiles else ""),
            contract_digest=contract_obj.digest(),
            schedule_digest=_sha((kernel.digest() + _sha(source.encode()) + _sha(blob)).encode()),
            emitter_digest=_sha((iset.digest() + EMITTER_VERSION + backend.library_layer_emitter_digest()).encode()),
            harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION + "+sched",
            protocol=protocol,
        )
    elif arm == "package":
        if package is None or package_digest_value is None or obj_cache is None:
            raise SystemExit("the package arm needs --package")
        built_pkg = _package_kernel(
            spec,
            package=package,
            readout=package_readout,
            target=target,
            workdir=workroot / ("pkg_" + _sha(f"{sig}:{package_readout}:{package_digest_value}".encode())[:16]),
            obj_cache=obj_cache,
            contract_obj=contract_obj,
            timeout_s=timeout_s,
        )
        source, blob = built_pkg["source"], built_pkg["blob"]
        digest_expected = built_pkg["digest_expected"]
        objects = [built_pkg["object"]]
        # A fully unrolled kernel object can exceed the crt's +-1 MiB jal reach if it is linked
        # between _start and _init, so the support objects go first.
        support_first = True
        extra = {k: v for k, v in built_pkg.items() if k not in ("cb", "source", "blob", "object", "digest_expected")}
        extra["package"] = str(package)
        extra["package_digest"] = package_digest_value
        extra["digest_comparable_with_other_arms"] = package_readout == "requant_i8"
        key = LayerKey(
            target=target,
            design_pin=design_pin,
            engine_sha256=engine["binary_sha256"],
            group_signature=f"package:{sig}:{package_readout}",
            contract_digest=contract_obj.digest(),
            schedule_digest=_sha((built_pkg["lowered_sha256"] + _sha(blob) + _sha(source.encode())).encode()),
            emitter_digest=package_digest_value,
            harness_version=backend.LIBRARY_LAYER_HARNESS_VERSION + "+package+support_first",
            protocol=protocol,
        )
        kernel = None
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    cached = cache.get(key)
    if cached is not None:
        return {"sig": sig, "arm": arm, "cached": True, **cached}

    wd = workroot / key.digest()[:16]
    wd.mkdir(parents=True, exist_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    if kernel is not None:
        (wd / "kernel.mk").write_text(kernel.text(), encoding="utf-8")
    built = build_program(
        [wd / "layer.c", *objects], wd, target=target, max_loaded_bytes=None, support_first=support_first
    )
    run = run_on_gsim(built.elf, target=target, max_cycles=max_cycles, timeout_s=timeout_s, backdoor=True)
    if run.load_path != "backdoor":
        raise RuntimeError("target declares no load backdoor; an embedded-operand program would spend its run loading")
    recs = [r for r in run.records if r.label == label]
    payload = {
        "spec": spec,
        "arm": arm,
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
        "completed": run.completed,
        "wall_seconds": round(run.wall_seconds, 2),
        "load_path": run.load_path,
        "engine_cycles": run.finish.cycles if run.finish else None,
        "cycles": recs[0].cycles if len(recs) == 1 else None,
        "digest": recs[0].fields.get("digest") if len(recs) == 1 else None,
        "digest_expected": digest_expected,
        "engine": engine,
        **extra,
    }
    payload["numerics"] = (
        "exact" if payload["digest"] is not None and payload["digest"] == payload["digest_expected"] else "mismatch"
    )
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).unlink(missing_ok=True)
    if not run.completed or len(recs) != 1 or payload["numerics"] != "exact":
        payload["stderr_tail"] = run.stderr_tail
        payload["stdout_tail"] = run.stdout_tail
        return {"sig": sig, "cached": False, **payload}
    return {"sig": sig, "cached": False, **cache.put(key, payload)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", action="append", required=True, help="MxNxK; repeatable")
    ap.add_argument("--target", required=True)
    ap.add_argument("--design-pin", required=True)
    ap.add_argument("--scale", type=float, default=0.00390625, help="requant scale both arms share")
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--arm", action="append", choices=ARMS, default=None)
    ap.add_argument("--package", type=Path, default=None, help="package dir holding manifest.yaml (package arm)")
    ap.add_argument("--package-readout", choices=READOUTS, default="requant_i8")
    ap.add_argument(
        "--bias-span",
        type=int,
        default=None,
        help="half-width of the shared bias operand; 0 makes it all zeros, which the package arm requires",
    )
    ap.add_argument("--protocol", choices=("warm_then_measured", "cold_single"), default="warm_then_measured")
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--max-cycles", type=int, default=400_000_000)
    ap.add_argument("--timeout-s", type=float, default=7200)
    ap.add_argument(
        "--tiles", default=None, help="I,J,K tile override for the schedule arm (default: the recipe chooses)"
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="receipt store to use; point it somewhere empty to FORCE a fresh "
        "measurement. A number nobody can re-measure is not verifiable.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    arms = tuple(args.arm) if args.arm else ("library", "schedule")
    tiles = tuple(int(x) for x in args.tiles.split(",")) if args.tiles else None
    if tiles is not None and len(tiles) != 3:
        raise SystemExit("--tiles wants exactly I,J,K")
    if "package" in arms and args.package is None:
        raise SystemExit("--arm package needs --package <dir holding manifest.yaml>")

    from merlin.common.paths import artifacts_dir
    from merlin.perf.layer_bench import ReceiptCache
    from merlin.perf.layer_bench.reference import DEFAULT_BIAS_SPAN
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract

    bias_span = DEFAULT_BIAS_SPAN if args.bias_span is None else args.bias_span
    if "package" in arms and bias_span != 0:
        raise SystemExit(
            "the interface GEMM a package is handed declares no bias operand, so a non-zero bias span "
            "would make the package arm compute a different function from the other two and their "
            "digests incomparable by construction. Run the comparison with --bias-span 0."
        )
    backend = base.get_backend(args.target)
    contract_obj = contract("per_tensor_readout_v1", backend.readout_facts())
    package = args.package.resolve() if args.package else None
    package_digest_value = package_digest(package) if package else None
    shapes = [parse_shape(s) for s in args.shape]
    specs = {
        f"{m}x{n}x{k}": build_spec((m, n, k), scale=args.scale, relu=args.relu, bias_span=bias_span)
        for m, n, k in shapes
    }

    out_dir = args.out_dir or (
        artifacts_dir()
        / "perf-bench"
        / args.target
        / f"layer_shape_table_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ReceiptCache(args.cache_dir or (artifacts_dir() / "perf-bench" / args.target / "layer_cache"))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.slots)) as pool:
        futs = {
            pool.submit(
                measure_one,
                sig,
                spec,
                arm=arm,
                target=args.target,
                design_pin=args.design_pin,
                workroot=out_dir / "work",
                cache=cache,
                contract_obj=contract_obj,
                max_cycles=args.max_cycles,
                timeout_s=args.timeout_s,
                protocol=args.protocol,
                tiles=tiles,
                package=package,
                package_digest_value=package_digest_value,
                package_readout=args.package_readout,
                obj_cache=artifacts_dir() / "perf-bench" / args.target / "layer_obj_cache",
            ): (sig, arm)
            for sig, spec in specs.items()
            for arm in arms
        }
        for fut in as_completed(futs):
            sig, arm = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one failed row is a row, not a crash
                res = {"sig": sig, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
            results.append(res)
            print(
                json.dumps({k: res.get(k) for k in ("sig", "arm", "cycles", "numerics", "cached", "error")}),
                flush=True,
            )

    by: dict[str, dict] = {}
    for r in results:
        by.setdefault(r["sig"], {})[r.get("arm", "?")] = r

    def cell(row: dict) -> str:
        cycles = row.get("cycles")
        # A cycle count without its correctness receipt is not a result here, so a row whose digest
        # did not match prints as a mismatch rather than as a number.
        if cycles and row.get("numerics") == "exact":
            return f"{cycles:,}"
        if cycles:
            return f"{cycles:,} (MISMATCH)"
        return "FAILED"

    shown = [a for a in ARMS if a in arms]
    header = "| shape (MxNxK) | " + " | ".join(f"{a} cycles" for a in shown) + " |"
    ratios = [(x, y) for x, y in (("library", "schedule"), ("package", "schedule")) if x in arms and y in arms]
    header += "".join(f" {x} / {y} |" for x, y in ratios) + " numerics |"
    lines = [header, "|---" + "|---:" * (len(shown) + len(ratios)) + "|---|"]
    for sig in specs:
        row = {a: by.get(sig, {}).get(a, {}) for a in shown}
        cells = [cell(row[a]) for a in shown]
        for x, y in ratios:
            xc, yc = row[x].get("cycles"), row[y].get("cycles")
            ok = xc and yc and row[x].get("numerics") == "exact" and row[y].get("numerics") == "exact"
            cells.append(f"{xc / yc:.3f}x" if ok else "-")
        note = ",".join(f"{a}={row[a].get('numerics') or str(row[a].get('error'))[:40]}" for a in shown)
        lines.append(f"| `{sig}` | " + " | ".join(cells) + f" | {note} |")
    table = "\n".join(lines)
    if "package" in arms and args.package_readout != "requant_i8":
        table += (
            f"\n\nThe package arm was measured on the `{args.package_readout}` readout, which is NOT the "
            "function the other arms compute; its digest is checked against its own exact oracle and its "
            "cycles are not a like-for-like row.\n"
        )
    print(table)

    doc = {
        "schema": "layer_shape_table_v2",
        "target": args.target,
        "design_pin": args.design_pin,
        "contract_digest": contract_obj.digest(),
        "scale": args.scale,
        "bias_span": bias_span,
        "protocol": args.protocol,
        "relu": bool(args.relu),
        "arms": list(arms),
        "package": str(package) if package else None,
        "package_digest": package_digest_value,
        "package_readout": args.package_readout if "package" in arms else None,
        "shapes": list(specs),
        "rows": results,
    }
    (out_dir / "layer_shape_table.json").write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "layer_shape_table.md").write_text(table + "\n", encoding="utf-8")
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
