#!/usr/bin/env python3
"""Build chipyard_riscv64 scalar Zephyr images for a set of model bundles and run each on
an external FireSim operator, collecting hash-bound observations and gating cos
against golden.npy. Resumable: results are appended to a JSONL ledger and already-passed
(bundle, dtype) pairs are skipped on re-run.

Usage:
    .venv/bin/python build_tools/scripts/firesim_sweep.py BUNDLE [BUNDLE ...]
    .venv/bin/python build_tools/scripts/firesim_sweep.py --ledger /tmp/fs_sweep.jsonl ...

Each BUNDLE is a directory name under out/artifacts/recaptures/ (e.g. rdt2_int8_consistent). The build runs
locally; prepare retains a portable workload. An external operator runs it and supplies
result.json plus UART. Collect applies the existing numerical gates; it never builds or launches.
"""

import argparse
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
from merlin.common.artifacts import recaptures_dir  # noqa: E402
from merlin.common.paths import artifacts_dir, build_dir  # noqa: E402
from merlin.compiler.artifact import Deployment  # noqa: E402
from merlin.runtime.backends import zephyr_model as zm  # noqa: E402


def already_done(ledger: Path, bundle: str) -> bool:
    if not ledger.is_file():
        return False
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("bundle") == bundle and r.get("ok"):
            return True
    return False


def cycle_report(ledger: Path) -> int:
    """Pair <model>_int8 vs <model>_fp32 cycle counts -> speedup table, PER BACKEND (scalar vs
    rvv are not comparable — the int8 win lives on the Saturn-OPU vector tile). Uses the most
    recent row per (model, dtype, backend)."""
    if not ledger.is_file():
        print(f"no ledger at {ledger}")
        return 1
    # (backend, model) -> {dtype: cycles}; last row wins (ledger is append-only / time-ordered)
    cyc: dict[tuple, dict[str, int]] = {}
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("ok") or r.get("cycles") is None:
            continue
        b = r.get("bundle", "")
        backend = r.get("backend", "scalar")
        parts = b.split("_")
        if "consistent" in parts:
            parts = parts[: parts.index("consistent")]
        dtype = parts[-1] if parts else ""
        model = "_".join(parts[:-1]) if len(parts) > 1 else b
        cyc.setdefault((backend, model), {})[dtype] = int(r["cycles"])
    print(f"{'backend':8s} {'model':18s} {'fp32_cycles':>14s} {'int8_cycles':>14s} {'speedup':>9s}")
    for backend, model in sorted(cyc):
        d = cyc[(backend, model)]
        fp, i8 = d.get("fp32"), d.get("int8")
        sp = f"{fp / i8:.2f}x" if (fp and i8) else "-"
        print(f"{backend:8s} {model:18s} {str(fp or '-'):>14s} {str(i8 or '-'):>14s} {sp:>9s}")
    return 0


def collect_study(prepared: Path, results: Path, ledger: Path) -> dict:
    """Collect an existing study package; no build and no infrastructure access."""
    request = json.loads((prepared / "request.json").read_text(encoding="utf-8"))
    files = {item["path"]: item for item in request["files"]}

    def frozen(name):
        data = (prepared / "inputs" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != files["inputs/" + name]["sha256"]:
            raise ValueError(f"prepared study input changed: {name}")
        return data

    rec = json.loads(frozen("study.json"))
    golden = np.load(io.BytesIO(frozen("golden.npy")), allow_pickle=False)
    refs = {"fp32": golden} if rec["int8_compute"] else None
    if refs is not None and "inputs/golden_w8a8.npy" in files:
        refs["w8a8"] = np.load(io.BytesIO(frozen("golden_w8a8.npy")), allow_pickle=False)
    r = zm.run_on_firesim(
        prepared, mode="collect", results=results, reference=golden if refs is None else None, references=refs
    )
    rec.update(
        status="collected",
        request_sha256=r["request_sha256"],
        observation=r["observation"],
        execution_authority="external_operator_report",
        hardware_qualification="not_established",
        cos=r.get("cos"),
        rel=r.get("rel"),
        ok=bool(r.get("ok")),
        cycles=r.get("metrics", {}).get("cycles"),
    )
    for key in ("w8a8_cos", "w8a8_rel", "fp32_cos", "fp32_argmax"):
        if key in r:
            rec[key] = r[key]
    # Preserve historical prefix and available aggregate gates; these do not
    # establish per-element correctness of an unobserved suffix.
    rec["verification_scope"] = "legacy_prefix_and_available_aggregates"
    gflat = golden.astype(np.float32).ravel()
    if gflat.size > 4096:
        checks = []
        if r.get("sum") is not None:
            g_sum = float(gflat.sum())
            sum_rel = abs(float(r["sum"]) - g_sum) / max(1e-6, abs(g_sum))
            rec["sum_rel"] = sum_rel
            # The SUM check is only meaningful when the true sum is a well-conditioned
            # aggregate. For mixed-sign outputs (logits) it can be a near-zero residual
            # of large-magnitude cancellation — bitvla's golden sums to ~1e-4 of its
            # total |mass|, so a rel-0.004 per-element diff blows sum_rel up to ~0.4
            # despite cos=0.99999 + 100% argmax. Gate sum only when |sum| is a real
            # fraction of the L2 magnitude; otherwise ARGMAX governs the full output.
            l2 = float(np.sqrt((gflat.astype(np.float64) ** 2).sum()))
            well_conditioned = abs(g_sum) >= 0.1 * l2
            rec["sum_conditioned"] = well_conditioned
            if well_conditioned:
                checks.append(sum_rel < 1e-3)
        if r.get("argmax") is not None:
            last = golden.shape[-1]
            g_arg = golden.astype(np.float32).reshape(-1, last).argmax(1)
            hw = np.asarray(r["argmax"])
            k = min(len(hw), len(g_arg))
            frac = float((hw[:k] == g_arg[:k]).mean()) if k else 0.0
            rec["argmax_match"] = frac
            checks.append(frac > 0.999)
        if checks:
            rec["ok"] = bool(rec["ok"] and all(checks))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rec) + "\n")
    print(f"COLLECTED {rec['bundle']}: ok={rec['ok']} cycles={rec['cycles']}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--ledger")
    ap.add_argument("--collect", type=Path, help="prepared study package; never builds")
    ap.add_argument("--results", type=Path, help="external result package for --collect")
    ap.add_argument("--board", help="explicit board descriptor for preparation")
    ap.add_argument("--isa", help="explicit image ISA for the external deployment")
    ap.add_argument("--abi", help="explicit image ABI for the external deployment")
    ap.add_argument("--harts", type=int, help="number of harts enabled in the image")
    ap.add_argument("--handoff-root", type=Path)
    ap.add_argument("--completion-metric-prefix", help="external operator completion marker")
    ap.add_argument(
        "--int8",
        action="store_true",
        help="run the W8A8 INTEGER-compute datapath (i8xi8->i32 + integer "
        "nonlinears) and gate multi-tier vs golden_w8a8.npy + golden.npy",
    )
    ap.add_argument(
        "--report", action="store_true", help="print the int8-vs-fp32 cycle/speedup table from the ledger and exit"
    )
    # The bitstream this sweep was written against is heterogeneous: its scalar host tile is hart 0 (a
    # Gemmini tile) and its vector tile is hart 1 (a Saturn-OPU tile) -- hence --rvv-hart's default.
    ap.add_argument(
        "--backend",
        default="scalar",
        choices=("scalar", "rvv"),
        help="scalar (the scalar host tile, hart 0) or rvv (the vector tile, see "
        "--rvv-hart). The int8 throughput win lives on rvv — i8 lanes pack ~4x an "
        "f32 lane.",
    )
    ap.add_argument(
        "--rvv-hart", type=int, default=1, help="hart the rvv model object runs on (the vector tile's hart)"
    )
    # Routing the contractions to a matrix unit is a THIRD thing, orthogonal to --backend: the host
    # core still runs everything the unit does not take, so the image is an rvv (or scalar) image
    # that additionally dispatches to the unit. Both names are required together and neither has a
    # default -- which unit a chip carries and in which geometry is a fact about that chip.
    ap.add_argument(
        "--matrix-unit",
        default=None,
        help="route contractions to this provider-declared matrix unit",
    )
    ap.add_argument(
        "--matrix-support-target",
        help="explicit OOT support provider target for matrix routing (distinct from the unit name)",
    )
    ap.add_argument(
        "--matrix-config",
        default=None,
        help="the unit's hardware configuration class, e.g. the one the bitstream was built "
        "from; decides the tile edge, so it is not optional",
    )
    # Build scratch goes on /scratch (3 TB), NOT /tmp (the small shared root disk) — a
    # multi-GB external-weights image in /tmp pressures root, where /home caches already
    # sit at ~94%. /scratch is the project's filesystem with terabytes free.
    ap.add_argument("--workroot", type=Path)
    ap.add_argument("--timeout", type=int, help="removed launch option; preparation never executes")
    ap.add_argument("--runner", help="removed launch option; execution belongs to the external operator")
    ap.add_argument("--force", action="store_true", help="re-run even if already passed")
    args = ap.parse_args()
    ledger = Path(args.ledger) if args.ledger else None
    routing_args = (args.matrix_unit, args.matrix_config, args.matrix_support_target)
    if any(routing_args) and not all(routing_args):
        ap.error(
            "--matrix-unit, --matrix-config and --matrix-support-target go together: the tile edge comes from the "
            "configuration, so a unit without one would be routed at a guessed geometry"
        )
    matrix = (
        zm.MatrixRouting(unit=args.matrix_unit, config=args.matrix_config, support_target=args.matrix_support_target)
        if args.matrix_unit
        else None
    )

    if args.runner is not None or args.timeout is not None:
        ap.error("runner/timeout launch options moved to the external operator; use prepare/collect")
    if args.report:
        return cycle_report(ledger or artifacts_dir() / "perf-studies" / "firesim.jsonl")
    if args.collect is not None:
        if args.results is None or args.bundles:
            ap.error("--collect requires --results and accepts no bundles to build")
        ledger = ledger or artifacts_dir() / "perf-studies" / "firesim.jsonl"
        try:
            return 0 if collect_study(args.collect, args.results, ledger)["ok"] else 1
        except Exception as exc:
            record = {
                "status": "collection-failed",
                "ok": False,
                "prepared": str(args.collect.resolve()),
                "results": str(args.results.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(record["error"], file=sys.stderr)
            return 1
    if args.results is not None:
        ap.error("--results requires --collect")
    if not args.bundles:
        return 0
    if not all((args.board, args.isa, args.abi, args.harts)):
        ap.error("preparation requires --board, --isa, --abi and --harts")
    from merlin.runtime.boards import board as board_descriptor

    descriptor = board_descriptor(args.board)
    if not 1 <= args.harts <= descriptor.harts:
        ap.error("--harts must fit the selected board's CPU count")
    if args.rvv_hart not in descriptor.hart_ids_for(args.backend):
        ap.error("--rvv-hart must select a board hart eligible for the requested backend")
    if args.rvv_hart >= args.harts:
        ap.error("--harts must include the selected worker hart ID; it is the image CPU count")
    ledger = ledger or artifacts_dir() / "perf-studies" / "firesim.jsonl"
    workroot = args.workroot or build_dir() / "firesim-sweep"
    handoff_root = args.handoff_root or artifacts_dir() / "handoff" / "firesim"
    completion_metric_prefix = args.completion_metric_prefix

    if matrix:
        from merlin.targetgen.plugins import load_declared

        provider = load_declared(args.matrix_support_target, "matrix_lowering")
        enc = provider.derive_encodings(provider.load_contract(args.matrix_unit)).encodings

    failed = False
    for bundle in args.bundles:
        if not args.force and already_done(ledger, bundle):
            print(f"SKIP {bundle} (already passed in ledger)", flush=True)
            continue
        mdir = recaptures_dir() / bundle
        rec = {
            "bundle": bundle,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "int8_compute": bool(args.int8),
            "backend": "matrix" if matrix else args.backend,
        }
        try:
            np.load(mdir / "golden.npy", allow_pickle=False)  # validate before a potentially expensive build
            # 1. build the chipyard image locally (int8: real W8A8 integer datapath). The rvv
            #    backend targets the Saturn-OPU vector tile (hart 1) where i8xi8->i32 packs ~4x
            #    the lanes of f32 — that is where the int8 throughput win actually shows up.
            extra = {}
            if matrix:
                # The feature is what makes the rewrite look for contractions at all; without it the
                # routing object is inert and the image would be an ordinary rvv image wearing the
                # matrix label.
                from merlin.llvmlower.impr_features import OPU_MATMUL_NAME, PEROP_BLOCK_NAME

                extra = {"matrix": matrix, "features": frozenset([PEROP_BLOCK_NAME, OPU_MATMUL_NAME])}
            b = zm.build_app(
                mdir,
                str(workroot / bundle),
                board=args.board,
                backend=args.backend,
                rvv_hart=args.rvv_hart,
                cpus=args.harts,
                int8_compute=args.int8,
                completion_metric_prefix=completion_metric_prefix,
                **extra,
            )
            rec["ram_mb"] = b["ram_bytes"] // (1024 * 1024)
            # The anti-cheat, in both directions, on the bytes that will boot. An image carrying none
            # of the unit's instructions computes correct answers on the host core and passes every
            # numerical gate below -- the most comfortable way for a matrix-unit run to be wrong. And
            # an unrouted image that DOES carry them is not the control it would be compared against.
            if args.matrix_unit or matrix:
                from merlin.kernels.decode import opu as unit_audit

                counts = {
                    k: int(v) for k, v in sorted((unit_audit.audit_object(b["elf"], enc).counts or {}).items()) if v
                }
                rec["unit_instruction_counts"] = counts
                if matrix and not counts:
                    raise RuntimeError(
                        f"the image for {bundle} contains NONE of "
                        f"{args.matrix_unit}'s instructions, so it would measure the "
                        "host core while claiming the unit"
                    )
            # The build and its reference inputs are retained; preparation is not a verdict.
            rec.update(
                status="prepared",
                build_hash=b.get("build_hash"),
                board=args.board,
                isa=args.isa,
                abi=args.abi,
                harts=args.harts,
                completion_metric_prefix=completion_metric_prefix,
            )
            study = workroot / bundle / "firesim-study.json"
            study.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            inputs = {"study.json": study, "golden.npy": mdir / "golden.npy"}
            w8a8 = mdir / "golden_w8a8.npy"
            if args.int8 and w8a8.is_file():
                inputs["golden_w8a8.npy"] = w8a8
            deployment = Deployment(
                args.board,
                "zephyr",
                args.isa,
                args.abi,
                args.harts,
                dram_base=descriptor.dram_base,
                memory_bytes=b["ram_bytes"],
            )
            r = zm.run_on_firesim(
                b["elf"], mode="prepare", output=handoff_root / bundle, deployment=deployment, inputs=inputs
            )
            rec.update(r)
            print(f"PREPARED {bundle}: {r['prepared']} (external operator required)", flush=True)
        except Exception as e:
            failed = True
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            rec["ok"] = False
            print(f"FSIM {bundle} ERR: {rec['error']}", flush=True)
            traceback.print_exc()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # Retain local build products and portable handoff for provenance/reproduction.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
