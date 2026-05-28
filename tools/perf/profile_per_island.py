"""Phase 4 — per-(island, target) profiling tool.

Given a partitioned MLIR module (output of `qnn_partition.partition`),
build a `.qnn-ctx` for each (island, target) pair on the board, run it
N times via `qnn-net-run`, and write a CSV of median + p99 latencies.

Output schema (one row per (island, target)):

    island_name,target,median_ms,p99_ms,iter_count,run_id

The CSV is consumed by `kernels/qnn/route.py::route_islands` to
make the per-island target decision (Phase 5).

Usage (from the merlin-dev env):

    ./merlin compile yolov8.onnx --target qrb5165_qnn --qnn-partition
    # writes build/.../qnn_partition.json

    conda run -n merlin-dev uv run python tools/profile_per_island.py \\
        --partition build/.../qnn_partition.json \\
        --mlir build/.../yolov8.mlir \\
        --on-board qdev \\
        --board-qairt-root /tmp/qnn_probe \\
        --output eval/qrb5165/heterogeneous/yolov8_per_island.csv

The tool is gated on board access; without `--on-board` (or with the
SDK / SSH config missing) it is a structural smoke test only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import statistics
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# kernels/ is at repo root (was tools/kernels/ before May 2026).
sys.path.insert(0, str(REPO_ROOT))


@dataclasses.dataclass
class ProfileRow:
    island_name: str
    target: str
    median_ms: float
    p99_ms: float
    iter_count: int
    run_id: str


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True, check=False)


def _measure_qnn_net_run(
    qnn_ctx: pathlib.Path,
    *,
    target: str,
    iters: int,
    ssh_host: str,
    board_qairt_root: str,
) -> tuple[float, float]:
    """Push a `.qnn-ctx` to the board, run `qnn-net-run` `iters` times,
    return (median_ms, p99_ms). Falls back to a no-op zero-latency
    sentinel when the board isn't reachable so this tool can be used
    as a structural smoke test offline."""
    backend_so_map = {
        "qnn-gpu": "libQnnGpu.so",
        "qnn-hta": "libQnnHta.so",
        "qnn-cpu": "libQnnCpu.so",
    }
    backend_so = backend_so_map.get(target)
    if backend_so is None:
        return (0.0, 0.0)

    remote_ctx = f"{board_qairt_root}/profiling_{qnn_ctx.name}"
    push_cmd = ["scp", "-q", str(qnn_ctx), f"{ssh_host}:{remote_ctx}"]
    res = subprocess.run(push_cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return (0.0, 0.0)

    samples: list[float] = []
    for _ in range(iters):
        cmd = (
            f"cd {board_qairt_root} && "
            f"export LD_LIBRARY_PATH=$PWD/lib && "
            f"./bin/qnn-net-run --retrieve_context {remote_ctx} "
            f"--backend ./lib/{backend_so} 2>&1 | tail -1"
        )
        t0 = time.time()
        res = _ssh(ssh_host, cmd)
        t1 = time.time()
        if res.returncode == 0:
            samples.append((t1 - t0) * 1000.0)
    if not samples:
        return (0.0, 0.0)
    samples.sort()
    median = statistics.median(samples)
    p99_idx = max(0, int(len(samples) * 0.99) - 1)
    p99 = samples[p99_idx]
    return (median, p99)


def profile_partition(
    partition_path: pathlib.Path,
    mlir_path: pathlib.Path,
    *,
    output_csv: pathlib.Path,
    iters: int,
    ssh_host: str | None,
    board_qairt_root: str,
    targets: list[str],
    cache_dir: pathlib.Path,
) -> list[ProfileRow]:
    """For each island in the partition file, build + run on each
    candidate target, accumulate ProfileRow records, write CSV."""
    payload = json.loads(partition_path.read_text())
    islands = payload["islands"]
    rows: list[ProfileRow] = []
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"profiling {len(islands)} islands × {len(targets)} targets " f"× {iters} iter; writing {output_csv}")

    for isl in islands:
        for target in targets:
            # In a full implementation, the partitioner emits a
            # per-island slice MLIR which the v2 emitter lowers to
            # `.qnn.cpp`, then `qnn_build.build_qnn_kernel_on_board`
            # produces the `.qnn-ctx`. For this initial Phase 4 tool we
            # only ship the wiring; the slice MLIR emission is Phase 5
            # territory (it's the partitioner's "emit per-island
            # manifest entry" extension).
            qnn_ctx = cache_dir / f"{isl['name']}_{target}.qnn-ctx"
            if not qnn_ctx.exists():
                # Sentinel: no ctxbin available means this island
                # hasn't been built yet. Record a zero-sample row so
                # the CSV reflects the inventory.
                rows.append(
                    ProfileRow(
                        island_name=isl["name"],
                        target=target,
                        median_ms=0.0,
                        p99_ms=0.0,
                        iter_count=0,
                        run_id=run_id,
                    )
                )
                continue
            if ssh_host is None:
                rows.append(
                    ProfileRow(
                        island_name=isl["name"],
                        target=target,
                        median_ms=0.0,
                        p99_ms=0.0,
                        iter_count=0,
                        run_id=run_id,
                    )
                )
                continue
            median_ms, p99_ms = _measure_qnn_net_run(
                qnn_ctx,
                target=target,
                iters=iters,
                ssh_host=ssh_host,
                board_qairt_root=board_qairt_root,
            )
            rows.append(
                ProfileRow(
                    island_name=isl["name"],
                    target=target,
                    median_ms=median_ms,
                    p99_ms=p99_ms,
                    iter_count=iters,
                    run_id=run_id,
                )
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "island_name",
                "target",
                "median_ms",
                "p99_ms",
                "iter_count",
                "run_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        type=pathlib.Path,
        required=True,
        help="qnn_partition.json from `./merlin compile --qnn-partition`",
    )
    parser.add_argument(
        "--mlir",
        type=pathlib.Path,
        required=True,
        help="Imported MLIR (the partitioner's input).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="CSV to write (typical: eval/qrb5165/heterogeneous/<model>_per_island.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "build" / "qnn_island_cache",
        help="Directory of per-island .qnn-ctx files (built upstream).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="qnn-net-run iters per (island, target). Default 30 — "
        "high enough that median is stable, low enough the run is "
        "<5min for 64 islands × 2 backends.",
    )
    parser.add_argument(
        "--on-board",
        default=None,
        help="SSH host for on-board profiling (typical: qdev). When "
        "omitted the tool emits zero-sample rows for inventory only.",
    )
    parser.add_argument(
        "--board-qairt-root",
        default="/tmp/qnn_probe",
        help="Board path with the QAIRT lib/ + bin/ staged.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["qnn-gpu", "qnn-hta"],
        help="Candidate targets to profile each island on.",
    )
    args = parser.parse_args()

    rows = profile_partition(
        args.partition,
        args.mlir,
        output_csv=args.output,
        iters=args.iters,
        ssh_host=args.on_board,
        board_qairt_root=args.board_qairt_root,
        targets=list(args.targets),
        cache_dir=args.cache_dir,
    )
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
