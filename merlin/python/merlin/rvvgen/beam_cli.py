"""``merlin-rvv-beam`` — the aet-instrumented, board-serialized entrypoint for the CCA beam (BB3).

Runs :func:`rvvgen.beam.run_beam` under an aet PARENT run (``out/runs/rvv/beam/<id>/``), then emits
one aet CHILD run per fork carrying a ``metrics/summary_metrics.json`` (speedup, cos-gate, cycles,
K1 wall, lever, depth, parent_run_id, CCA divergences closed) so ``aet runs`` / ``aet compare`` can
rank the forks. ``beam_tree.yaml`` (the full LLM-digestible per-step record) lands in the parent run.

Board serialization: the physical K1 has one deploy path + 8 cores, so concurrent forks would corrupt
each other and poison the cycle measurement. Two layers guard it — a host-wide ``k1.board_lock`` file
flock (serializes across ALL processes/sessions on this host) and, for a single beam, ``max_workers=1``
(the default here for a K1 target). The chia ``resources={"k1":1}`` gate (BB3, chia-venv) is the third
layer for a Ray-driven fan-out; this CLI runs the same beam without requiring the chia venv.

The expert CCA (the search target) is lifted deterministically from a decoded expert objdump fixture —
NO LLM authors it (the hard-principle: the CCA is tool-composed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..common.paths import repo_root
from ..common.yaml import write_yaml


def lift_expert_cca(objdump_path: str | Path, op: str):
    """Deterministically lift the expert CCA from a decoded objdump fixture (asm analyzer, no LLM)."""
    from ..kernels import cca
    from ..kernels.decode import rvv
    text = Path(objdump_path).read_text()
    return cca.lift_asm(rvv.decode_text(text), op=op, source="expert")


def _node_summary(node: dict) -> dict:
    """The ranking-visible headline metrics for one fork — the summary_metrics.json contract."""
    step = node.get("search_step") or {}
    return {
        "run_id": node.get("run_id"),
        "lever": node.get("lever"),
        "depth": node.get("depth"),
        "parent_run_id": node.get("parent_run_id"),
        "gate_ok": bool(node.get("gate_ok")),
        "speedup": node.get("speedup"),                 # REAL K1 speedup vs the frozen seed
        "k1_wall_ns": node.get("k1_wall_ns"),
        "cycles": node.get("cycles"),
        "structural_match": node.get("structural_match"),
        "cca_divergence_closed": bool(step.get("achieved")) if step else None,
        "cca_residual": step.get("residual") if step else None,
        "escalated": bool(node.get("escalations")),
    }


def run_instrumented_beam(
    *, seed_pkg: str | Path, model_dir: str | Path, expert_objdump: str | Path,
    op: str = "matmul", dtype: str = "f32", shape_regime: str = "square",
    targets: tuple[str, ...] = ("k1",), width: int = 3, depth: int = 2, top_k: int = 2,
    max_workers: int | None = None, curated_text: str | None = None,
    certify_fn=None,
) -> dict[str, Any]:
    """Open an aet parent run, run the CCA beam, emit a child aet run per fork, return the outcome."""
    from ..common.artifacts import finish_run, start_run
    from .beam import run_beam
    from .runner import certify_rvv
    certify_fn = certify_fn or certify_rvv

    op_key = {"op": op, "dtype": dtype, "shape_regime": shape_regime}
    expert_cca = lift_expert_cca(expert_objdump, op)
    curated = curated_text if curated_text is not None else Path(expert_objdump).read_text()
    # a K1 target defaults to serial (max_workers=1) — the board can't run concurrent forks.
    if max_workers is None and "k1" in targets:
        max_workers = 1

    suite = f"beam/{op}"
    parent = start_run(suite=suite, method="cca_beam", target="rvv",
                       extra={"op_key": op_key, "width": width, "depth": depth, "top_k": top_k,
                              "targets": list(targets), "workload": Path(model_dir).name,
                              "expert_objdump": str(expert_objdump), "role": "beam_parent"})
    status = "error"
    parent_summary: dict | None = None
    try:
        res = run_beam(seed_pkg=seed_pkg, model_dir=model_dir, curated_text=curated, op_key=op_key,
                       runs_root=parent.run_dir / "forks", out_root=str(parent.run_dir / "targets"),
                       width=width, depth=depth, top_k=top_k, target="rvv",
                       timestamp="beam", targets=targets, expert_cca=expert_cca,
                       max_workers=max_workers, certify_fn=certify_fn)
        # beam_tree.yaml (the full per-step record) into the parent run dir.
        tree_src = Path(res["tree_path"])
        if tree_src.is_file():
            (parent.run_dir / "beam_tree.yaml").write_text(tree_src.read_text(), encoding="utf-8")
        # one CHILD aet run per fork, carrying summary_metrics.json (aet compare reads these).
        for node in res.get("nodes", []):
            summ = _node_summary(node)
            child = start_run(suite=suite, method="fork", target="rvv",
                              extra={"parent_run_id": parent.run_id, "lever": summ["lever"],
                                     "depth": summ["depth"], "fork_run_id": summ["run_id"],
                                     "role": "beam_fork"})
            finish_run(child, status=("ok" if summ["gate_ok"] else "fail"), summary=summ)
        best = res.get("best")
        parent_summary = {"best_run_id": (best or {}).get("run_id"),
                          "best_speedup": (best or {}).get("speedup"),
                          "best_lever": (best or {}).get("lever"),
                          "n_forks": len(res.get("nodes", [])) - 1,
                          "n_deferred": len(res.get("deferred", []))}
        status = "ok"
        res["parent_run_dir"] = str(parent.run_dir)
        res["parent_run_id"] = parent.run_id
        return res
    finally:
        finish_run(parent, status=status, summary=parent_summary)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-rvv-beam",
                                 description="aet-instrumented, board-serialized CCA beam search (BB3).")
    ap.add_argument("--seed-pkg", default=None,
                    help="frozen baseline package (default: out/artifacts/targets/rvv/hand_v0)")
    ap.add_argument("--model-dir", required=True, help="workload bundle (model.mlir + inputs + golden)")
    ap.add_argument("--expert-objdump", required=True, help="decoded expert objdump fixture (CCA target)")
    ap.add_argument("--op", default="matmul")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--shape-regime", default="square")
    ap.add_argument("--targets", default="k1", help="comma-separated (e.g. 'k1' or 'spike,k1')")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=None,
                    help="sweep concurrency (default: 1 for a k1 target, board-safe)")
    args = ap.parse_args(argv)

    seed_pkg = args.seed_pkg or str(repo_root() / "out/artifacts/targets/rvv/hand_v0")
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    res = run_instrumented_beam(
        seed_pkg=seed_pkg, model_dir=args.model_dir, expert_objdump=args.expert_objdump,
        op=args.op, dtype=args.dtype, shape_regime=args.shape_regime, targets=targets,
        width=args.width, depth=args.depth, top_k=args.top_k, max_workers=args.max_workers)
    best = res.get("best") or {}
    print(f"parent_run={res.get('parent_run_dir')}")
    print(f"best: run_id={best.get('run_id')} lever={best.get('lever')} "
          f"speedup={best.get('speedup')} gate_ok={best.get('gate_ok')}")
    print(f"forks={len(res.get('nodes', [])) - 1} deferred={len(res.get('deferred', []))}")
    print(f"beam_tree: {res.get('parent_run_dir')}/beam_tree.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
