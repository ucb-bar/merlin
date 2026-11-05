#!/usr/bin/env python3
"""Shared steps of the chip examples: capture a model, and test one operator end to end.

    chip_flow.py capture <name> [--loader loader.py]
        recaptures/<name>_int8_full + its W8A8 golden. With --loader, any PyTorch model whose loader
        defines get_model_and_inputs(); without it, a model in model2MLIR's workloads/ roster.

    chip_flow.py op <gemm|conv2d|encoder_block|path/to/loader.py> [dims ...] --board B [--harts N]
                 [--matrix-unit U --matrix-config C --matrix-support-target T] [--firesim-submit]
        Capture the operator (once per shape), build it for the board as plain RVV and -- with the
        matrix flags -- routed to the matrix unit, gate both on spike, and print one summary. When the
        support target declares a spike model of its unit (``runner.spike_extension``), a third leg,
        ``matrix_on_spike``, runs the SHIPPED matrix image on it and compares it bit for bit with the twin.
        For a board that names a FireSim `bitstream`, also plan one FireSim job per image (firesim_flow.py)
        and print its submission; --firesim-submit runs them and grades the result.

The board, the matrix unit and its configuration are arguments: a chip example passes its own. Nothing
here names a chip.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
WORKLOADS = REPO / "examples" / "workloads"

#: Built-in single-operator loaders, the environment variables their shape comes from, and defaults.
OPS = {
    "gemm": (("GEMM_M", "GEMM_N", "GEMM_K"), (64, 64, 64)),
    "conv2d": (("CONV_CIN", "CONV_COUT", "CONV_HW", "CONV_K"), (16, 64, 16, 3)),
    "encoder_block": ((), ()),
}


def _env(key: str) -> str:
    from merlin.common.paths import env

    return env(key) or ""


def _run(cmd: list[str], *, env: dict | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, text=True, capture_output=capture)


def bundle_dir(name: str) -> Path:
    from merlin.common.artifacts import recaptures_dir

    # `make_delivery --models <name>` and `merlin-compile --workload <name>` both read this directory.
    return recaptures_dir() / f"{name}_int8_full"


def capture(name: str, loader: Path | None, *, env: dict | None = None) -> Path:
    bundle = bundle_dir(name)
    m2m = _env("MERLIN_M2M_DIR")
    if not m2m:
        raise SystemExit("set MERLIN_M2M_DIR to a model2MLIR checkout (examples/kodiak/run.sh setup does it)")
    if loader is not None:
        m2m_py = _env("MERLIN_M2M_PYTHON")
        if not m2m_py:
            raise SystemExit("set MERLIN_M2M_PYTHON to the model2MLIR venv's python")
        cmd = [
            m2m_py,
            "src/merlin/targetgen/_m2m_capture_worker.py",
            "--m2m-dir",
            m2m,
            "--loader",
            loader,
            "--dtype",
            "int8",
            "--seed",
            "0",
            "--materialize-bundle",
            "--out",
            bundle,
        ]
    else:
        from merlin.compile.bundles import _capture_python

        cmd = [
            _capture_python(Path(m2m), name),
            Path(m2m) / "workloads" / "capture_consistent.py",
            name,
            "int8",
            bundle,
        ]
    if _run(cmd, env=env).returncode != 0 or not (bundle / "model.mlir").is_file():
        raise SystemExit(f"capture of {name} failed; see the output above")
    # Weight-only int8 capture; Merlin computes W8A8. This writes the reference a W8A8 image is graded
    # against (Merlin's own host datapath: it checks the DEVICE reproduces the host).
    if _run([PY, "build_tools/scripts/make_w8a8_golden.py", bundle]).returncode != 0:
        raise SystemExit(f"W8A8 golden for {name} failed; see the output above")
    return bundle


def _compile(name: str, args, matrix: bool) -> dict:
    cmd = [
        PY,
        "-m",
        "merlin.compile_cli",
        "--workload",
        name,
        "--dtype",
        "int8",
        "--target",
        "rvv",
        "--board",
        args.board,
        "--harts",
        str(args.harts),
        "--run",
        "spike",
        "--verify",
        "--json",
        "--no-capture",
    ]
    if matrix:
        cmd += [
            "--matrix-unit",
            args.matrix_unit,
            "--matrix-config",
            args.matrix_config,
            "--matrix-support-target",
            args.matrix_support_target,
        ]
    proc = _run(cmd, capture=True)
    text = "\n" + proc.stdout
    try:
        return json.loads(text[text.rindex("\n{\n") + 1 :])  # the result is the last top-level JSON object
    except ValueError:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        return {"status": "error", "error": "merlin-compile produced no JSON result"}


def _references(name: str) -> dict:
    """The tiers merlin-compile grades an int8 image against: the W8A8 golden and the fp32 one."""
    import numpy as np

    bundle = bundle_dir(name)
    refs = {"fp32": np.load(bundle / "golden.npy")}
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    return refs


def matrix_on_spike(name: str, args, matrix: dict) -> dict:
    """Third leg: the SHIPPED matrix image (the one with the unit's instructions) on spike, executed by the
    unit's own functional model, graded against the W8A8 golden and compared bit for bit with the twin.

    The model is whatever the support target DECLARES under ``runner.spike_extension`` (its name, where the
    operator put it, and its sha256), resolved by ``merlin.targetgen.spike_extension``: a missing or
    different ``.so`` raises rather than running something else. A target that declares none is skipped,
    and says so -- plain spike would take an illegal instruction on the first unit opcode.
    """
    from merlin.runtime.backends import spike as spike_backend
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.runtime.boards import board as board_desc
    from merlin.targetgen import spike_extension as SE

    target = args.matrix_support_target
    m = matrix.get("matrix") or {}
    ship, twin = matrix.get("binary"), matrix.get("twin_binary")
    if not ship or not twin or not m.get("ship_ram_bytes"):
        return {"status": "skipped", "reason": "no shipped matrix image to run"}
    try:
        if SE.declared_extension(target) is None:
            return {
                "status": "skipped",
                "reason": f"{target} declares no runner.spike_extension, so spike has no model of the unit",
            }
        # Declared: the defaults below are never used, resolve() either returns the declared bytes or raises.
        ext = SE.resolve(target, default_library_dir=spike_backend.spike_path().parent, default_extension_name="")
    except SE.SpikeExtensionError as exc:
        return {"status": "error", "error": str(exc)}

    brd = board_desc(args.board)
    run = dict(dram_base=brd.dram_base, harts=m["spike_harts"], vlen=brd.vlen, timeout=3600)
    res = zm.run_on_spike(ship, mem_bytes=m["ship_ram_bytes"], extension=ext, **run)
    out = {
        "binary": ship,
        "extension": {"name": ext.extension_name, "extlib": str(ext.extlib), "sha256": ext.sha256},
        "spike_cycles": res.get("metrics", {}).get("cycles"),
    }
    if res.get("prefix") is None or not len(res["prefix"]):
        return {**out, "status": "run_failed", "error": res.get("console", "")[-1500:]}
    g = zm._gate(res["prefix"], _references(name))
    # The twin is the same lowering with a scalar tile body, so on a correct model the two must agree to
    # the bit -- on the dumped prefix AND on the whole-tensor hash. Re-run here because merlin-compile
    # keeps only the twin's verdict, not its output.
    t = zm.run_on_spike(twin, mem_bytes=m["twin_ram_bytes"], **run)
    same_prefix = t.get("prefix") is not None and bool((t["prefix"] == res["prefix"]).all())
    hashes = (res.get("out_hash"), t.get("out_hash"))
    same_hash = None if None in hashes else hashes[0] == hashes[1]
    out["verify"] = {"gate_ok": bool(g.get("ok")), "of": "shipped image on the unit model", **g}
    out["bit_exact_vs_twin"] = same_prefix and same_hash is not False
    out["twin_compare"] = {"prefix_equal": same_prefix, "out_hash_equal": same_hash, "elements": len(res["prefix"])}
    out["status"] = "verified" if (g.get("ok") and out["bit_exact_vs_twin"]) else "run_mismatch"
    return out


def op(args) -> int:
    if args.op in OPS:
        keys, defaults = OPS[args.op]
        dims = [int(d) for d in args.dims] or list(defaults)
        if len(dims) != len(keys):
            raise SystemExit(f"{args.op} takes {len(keys)} dims ({' '.join(keys)}), got {dims}")
        loader = WORKLOADS / args.op / "loader.py"
        name = "op_" + args.op + ("_" + "x".join(map(str, dims)) if dims else "")
    else:
        loader, keys, dims = Path(args.op).resolve(), (), []
        if not loader.is_file():
            raise SystemExit(f"{args.op}: not one of {sorted(OPS)} and not a loader file")
        name = "op_" + loader.parent.name
    env = {**os.environ, **{k: str(v) for k, v in zip(keys, dims)}}
    if args.recapture or not (bundle_dir(name) / "golden_w8a8.npy").is_file():
        capture(name, loader, env=env)
    results = {"rvv": _compile(name, args, matrix=False)}
    if args.matrix_unit:
        results["matrix"] = _compile(name, args, matrix=True)
        results["matrix_on_spike"] = matrix_on_spike(name, args, results["matrix"])
    print(f"\n== {name} on {args.board}, {args.harts} hart(s)")
    for leg, res in results.items():
        verdict = res.get("status")
        ver = res.get("verify") or {}
        line = f"  {leg:15} {verdict:13} w8a8_cos={ver.get('w8a8_cos', 'n/a')}"
        if leg == "rvv":
            line += f"  cycles={res.get('cycles')}"
        elif leg == "matrix_on_spike":
            if res.get("reason"):
                line += f"  ({res['reason']})"
            else:
                line += f"  bit_exact_vs_twin={res.get('bit_exact_vs_twin')}  spike_cycles={res.get('spike_cycles')}"
        else:
            m = res.get("matrix") or {}
            line += (
                f"  routed={m.get('routed_contractions')} opu_mac_share={m.get('opu_mac_share')}"
                f"  unit_insns={m.get('unit_instruction_counts')}"
            )
        print(line)
        if res.get("error"):
            print(f"         error: {str(res['error'])[:400]}")
    m = results.get("matrix", {}).get("matrix") or {}
    if m:
        print(f"  not routed: {m.get('not_routed_by_reason') or 'nothing'}")
        print(f"  shipped ELF (has the unit's instructions): {results['matrix'].get('binary')}")
        print(f"  stand-in twin (scalar tile body):          {results['matrix'].get('twin_binary')}")
        print(f"  routing report: {m.get('routing_report')}")
        ext = (results.get("matrix_on_spike") or {}).get("extension")
        if ext:
            print(f"  the shipped ELF also ran on spike with the unit's functional model {ext['name']!r}")
            print(f"  ({ext['extlib']}, sha256 {ext['sha256'][:16]}...): its numbers, not its cycles.")
        else:
            print("  spike has no model of the unit here, so only the twin ran: it checks routing, pack, ABI")
            print("  and epilogue. Build and declare the unit's spike model (the chip example's README), or run")
            print("  the shipped ELF on the chip or FireSim for the unit's datapath and cycles.")
    print(f"  RVV ELF: {results['rvv'].get('binary')}")
    kept = _keep(name, args, results)
    print(f"  kept in: {kept}")
    # A leg skipped for lack of a declared model is reported, not failed; a leg that ran must verify.
    ok = all(r.get("status") in ("verified", "skipped") for r in results.values())
    if ok and _firesim_board(args.board):
        return _firesim(kept, name, args, results)
    if _firesim_board(args.board):
        print("  not planning FireSim jobs: an image that fails its spike gate is not worth an FPGA slot")
    return 0 if ok else 1


def _firesim_board(name: str) -> bool:
    from merlin.runtime.boards import board

    return bool(board(name).bitstream)


def _firesim(kept: Path, name: str, args, results: dict) -> int:
    """Plan one FireSim job per kept image: the OPU image is the DEVICE leg, the plain RVV image its
    CONTROL. Printed by default; --firesim-submit runs them through the queue and grades them."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import firesim_flow

    legs = {"control": {"elf": kept / "rvv.elf", "build_hash": results["rvv"].get("build_hash")}}
    if "matrix" in results:
        m = results["matrix"].get("matrix") or {}
        legs["device"] = {
            "elf": kept / "matrix.elf",
            "build_hash": results["matrix"].get("build_hash"),
            "unit_instruction_counts": m.get("unit_instruction_counts"),
        }
    firesim_flow.plan(kept, board=args.board, name=name, bundle=bundle_dir(name), legs=legs, harts=args.harts,
                      timeout_s=args.firesim_timeout)  # fmt: skip
    print()
    print(firesim_flow.describe(kept, firesim_flow.load(kept), firesim_flow.host_config()))
    if not args.firesim_submit:
        print("\n  (not submitted: pass --submit to run these on the FPGA)")
        return 0
    firesim_flow.submit(kept)
    result = firesim_flow.collect(kept)
    print(firesim_flow.summary(result))
    return 0 if result["all_verified"] else 1


def _keep(name: str, args, results: dict) -> Path:
    """Copy the ELFs, the routing report and the summary out of the compiler's scratch directories into
    ``out/artifacts/probes/<board>/<name>/``, which is what someone copies to a board."""
    import shutil

    from merlin.common.paths import artifacts_dir

    dest = artifacts_dir() / "probes" / args.board / f"{name}_h{args.harts}"
    dest.mkdir(parents=True, exist_ok=True)
    picks = {
        "rvv.elf": results["rvv"].get("binary"),
        "matrix.elf": results.get("matrix", {}).get("binary"),
        "matrix_twin.elf": results.get("matrix", {}).get("twin_binary"),
        "routing_report.json": (results.get("matrix", {}).get("matrix") or {}).get("routing_report"),
    }
    for target, source in picks.items():
        if source and Path(source).is_file():
            shutil.copy2(source, dest / target)
    (dest / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("name")
    cap.add_argument("--loader", type=Path)
    o = sub.add_parser("op")
    o.add_argument("op", help=f"one of {sorted(OPS)} or a path to a loader.py")
    o.add_argument("dims", nargs="*", help="shape for a built-in op, in the order its loader documents")
    o.add_argument("--board", required=True)
    o.add_argument("--harts", type=int, default=1)
    o.add_argument("--matrix-unit")
    o.add_argument("--matrix-config")
    o.add_argument("--matrix-support-target")
    o.add_argument("--recapture", action="store_true", help="capture again even if the bundle exists")
    o.add_argument("--firesim-submit", action="store_true", help="on a FireSim board, run the images on the FPGA")
    o.add_argument("--firesim-timeout", type=int, default=1800, help="FireSim RUNNING-phase limit per image, s")
    args = ap.parse_args(argv)
    if args.cmd == "capture":
        print(capture(args.name, args.loader.resolve() if args.loader else None))
        return 0
    matrix = (args.matrix_unit, args.matrix_config, args.matrix_support_target)
    if any(matrix) and not all(matrix):
        ap.error("--matrix-unit, --matrix-config and --matrix-support-target are given together")
    return op(args)


if __name__ == "__main__":
    raise SystemExit(main())
