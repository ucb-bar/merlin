"""``merlin-rvv-beam`` — the aet-instrumented, board-serialized entrypoint for the CCA beam (BB3).

Runs :func:`mining.beam.run_beam` under an aet PARENT run (``out/runs/rvv/beam/<id>/``), then emits
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
        "speedup": node.get("speedup"),                 # REAL K1 speedup vs the seed
        "attainment_vs_expert": node.get("attainment_vs_expert"),  # vs XNNPACK (>=1 = matched/beat)
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
    certify_fn=None, sweep_fn=None, expert_wall_ns: float | None = None,
    validate_model_dir: str | Path | None = None, validate_fn=None,
    noise_margin: float | None = None, proposer=None, teachers: str | None = None,
) -> dict[str, Any]:
    """Open an aet parent run, run the CCA beam, emit a child aet run per fork, return the outcome.

    Two-phase objective: ``model_dir`` is the cheap EXPLORE bundle; when ``validate_model_dir`` is
    given, survivors are re-certified on that full/slow bundle (a ``validate_fn`` reusing the same
    ``certify_fn`` with the validate bundle is synthesized) before promotion. An explicit
    ``validate_fn`` wins if supplied (tests inject a mock).

    ``teachers`` widens the EXPERT side from the single ``--expert-objdump`` fixture to the family
    teacher set (``"all"``, or a comma-separated family list). The default single expert bounds what
    the search can find: an axis one expert cannot answer yields no divergence, routes to no action and
    is never forked, however much of the wall it owns. MEASURED on small_llama fp32 -- matmul teacher
    alone: 5 divergences / 4 mintable forks, with ``compute.activation_vectorization`` uncomparable
    while scalar `exp` is 16.48% of real model work. All teachers: 9 divergences / 6 mintable forks."""
    from ..common.artifacts import finish_run, start_run
    from .beam import run_beam
    from .runner import certify_rvv
    from .sweep import run_sweep
    certify_fn = certify_fn or certify_rvv
    sweep_fn = sweep_fn or run_sweep
    # Synthesize the validation seam from the validate bundle: same certify_fn, full whole-model dir.
    if validate_fn is None and validate_model_dir is not None:
        def validate_fn(*, model_dir=None, **job):
            return certify_fn(model_dir=str(validate_model_dir), **job)

    op_key = {"op": op, "dtype": dtype, "shape_regime": shape_regime}
    expert_cca = lift_expert_cca(expert_objdump, op)
    # The teacher-set expert side, when asked for. `teacher_audit` collects, per generation, which
    # teacher justified each axis and which axes NO teacher could answer -- the second is the honest
    # reading of "the search found nothing here" and lands in the parent run.
    compare_fn, teacher_audit = None, []
    if teachers:
        from .wholemodel_proposer import canonical_dtype, teacher_compare_fn
        fams = (None if teachers.strip().lower() == "all"
                else tuple(f.strip() for f in teachers.split(",") if f.strip()))
        if canonical_dtype(dtype) is None:
            raise SystemExit(
                f"--teachers needs a dtype with harvested fixtures; {dtype!r} normalises to nothing. "
                f"Run build_tools/scripts/harvest_xnnpack_fixtures.py, or drop --teachers to use the "
                f"single --expert-objdump expert.")
        compare_fn = teacher_compare_fn(dtype=dtype, families=fams, record=teacher_audit)
    curated = curated_text if curated_text is not None else Path(expert_objdump).read_text()
    # a K1 target defaults to serial (max_workers=1) — the board can't run concurrent forks.
    if max_workers is None and "k1" in targets:
        max_workers = 1

    suite = f"beam/{op}"
    parent = start_run(suite=suite, method="cca_beam", target="rvv",
                       extra={"op_key": op_key, "width": width, "depth": depth, "top_k": top_k,
                              "targets": list(targets), "workload": Path(model_dir).name,
                              "expert_objdump": str(expert_objdump), "role": "beam_parent",
                              "teachers": (teachers or "single-expert"),
                          "pass_slot_runs": sum(len(n.get("pass_slot") or ())
                                                for n in res.get("nodes", [])),
                          "pass_slot_accepted": sum(
                              1 for n in res.get("nodes", []) for r in (n.get("pass_slot") or ())
                              if r.get("accepted")),
                              "pass_slot_turns": pass_slot_turns})
    # The CODEGEN leaf. OFF unless asked for: a slot turn costs an agent and a build, so the ladder
    # never enters it implicitly. When on, every escalation the beam cannot fork is handed to the slot
    # -- which proposes a pass and gates it deterministically -- instead of only being recorded as a
    # work-item. Paths are inside the parent run, so a slot's builds and transcripts land with the
    # search that produced them.
    if pass_slot_fn is None and pass_slot_turns > 0:
        from .pass_slot_wiring import make_pass_slot_fn
        pass_slot_fn = make_pass_slot_fn(
            frozen_pkg=Path(seed_pkg), model_dir=Path(model_dir),
            runs_root=parent.run_dir / "pass_slot",
            targets_root=parent.run_dir / "targets" / "rvv",
            op=op, max_turns=pass_slot_turns)
    status = "error"
    parent_summary: dict | None = None
    try:
        res = run_beam(seed_pkg=seed_pkg, model_dir=model_dir, curated_text=curated, op_key=op_key,
                       runs_root=parent.run_dir / "forks", out_root=str(parent.run_dir / "targets"),
                       width=width, depth=depth, top_k=top_k, target="rvv",
                       timestamp="beam", targets=targets, expert_cca=expert_cca,
                       compare_fn=compare_fn, pass_slot_fn=pass_slot_fn,
                       max_workers=max_workers, certify_fn=certify_fn, sweep_fn=sweep_fn,
                       expert_wall_ns=expert_wall_ns, validate_fn=validate_fn,
                       noise_margin=noise_margin, proposer=proposer)
        # beam_tree.yaml (the full per-step record) into the parent run dir.
        tree_src = Path(res["tree_path"])
        if tree_src.is_file():
            (parent.run_dir / "beam_tree.yaml").write_text(tree_src.read_text(), encoding="utf-8")
        if teacher_audit:
            write_yaml(parent.run_dir / "teacher_audit.yaml",
                       {"teachers": teachers, "generations": teacher_audit})
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
                          "best_attainment_vs_expert": (best or {}).get("attainment_vs_expert"),
                          "best_lever": (best or {}).get("lever"),
                          "n_forks": len(res.get("nodes", [])) - 1,
                          "n_deferred": len(res.get("deferred", [])),
                          "teachers": (teachers or "single-expert"),
                          "teacher_unanswered_axes": sorted(
                              {a for g in teacher_audit for a in g.get("unanswered_axes", ())})}
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
                    help="seed package to FORK from (default: the frozen hand_v0 baseline, so a "
                         "from-scratch run rediscovers the levers honestly rather than inheriting a "
                         "pre-tuned config)")
    ap.add_argument("--expert-wall-ns", type=float, default=None,
                    help="XNNPACK measured wall (ns) for this workload — the REAL target; each fork "
                         "reports attainment_vs_expert = xnn_wall/fork_wall (>=1 = matched/beat XNNPACK)")
    ap.add_argument("--model-dir", required=True,
                    help="EXPLORE-phase workload bundle (model.mlir + inputs + golden). Use a fast "
                         "whole-model bundle here (e.g. out/artifacts/recaptures/bitvla_fp32_consistent) "
                         "so the cheap explore generations stay quick.")
    ap.add_argument("--validate-model-dir", default=None,
                    help="optional VALIDATE-phase whole-model bundle. When set, each generation's "
                         "survivor set is re-certified on this (full/slow) bundle and re-ranked before "
                         "promotion — the two-phase objective. Omit for single-phase (explore only).")
    ap.add_argument("--noise-margin", type=float, default=None,
                    help="win margin over the parent's speedup (default 0.02 >= the measured >=1.9%% "
                         "K1 noise floor; env MERLIN_BEAM_NOISE_MARGIN). Sub-margin deltas rank as ties.")
    ap.add_argument("--expert-objdump", required=True, help="decoded expert objdump fixture (CCA target)")
    ap.add_argument("--proposer", default="wholemodel", choices=("wholemodel", "cca"),
                    help="wholemodel: propose byte-traffic-ranked whole-model levers (transpose "
                         "fusion, per-matmul MR, reduction, ...) -- the right choice for a "
                         "whole-model objective. cca: legacy kernel-vs-expert divergence router.")
    ap.add_argument("--teachers", default=None,
                    help="widen the EXPERT side from --expert-objdump to the family teacher set: "
                         "'all', or a comma-separated family list (e.g. 'matmul,gelu,softmax'). One "
                         "expert cannot answer every axis, and an unanswerable axis raises no "
                         "divergence and is never forked -- measured on small_llama fp32, the matmul "
                         "teacher alone found 5 divergences to all-teachers' 9, missing the scalar "
                         "exp that is 16.48%% of that model's real work. Records teacher_audit.yaml.")
    ap.add_argument("--pass-slot-turns", type=int, default=0,
                    help="max agent turns per unforkable CODEGEN escalation (default 0 = OFF). The "
                         "ladder's leaf: when the router escalates to a rung no knob or feature "
                         "expresses, hand it to the pass slot, which proposes a compiler pass in a "
                         "sandbox and gates it deterministically (cheat scan, frozen baseline, "
                         "bit-exactness, inert check, promised facet). Costs an agent turn plus a "
                         "build per attempt, so it is never entered implicitly.")
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

    # Default seed = the FROZEN hand_v0 baseline. A from-scratch beam run rediscovers the winning
    # levers honestly from the naive control, instead of silently inheriting a pre-tuned config.
    default_seed = repo_root() / "out/artifacts/targets/rvv/hand_v0"
    seed_pkg = args.seed_pkg or str(default_seed)
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    proposer = None
    if args.proposer == "wholemodel":
        from .wholemodel_proposer import propose_wholemodel_levers
        proposer = propose_wholemodel_levers
    res = run_instrumented_beam(
        seed_pkg=seed_pkg, model_dir=args.model_dir, expert_objdump=args.expert_objdump,
        op=args.op, dtype=args.dtype, shape_regime=args.shape_regime, targets=targets,
        width=args.width, depth=args.depth, top_k=args.top_k, max_workers=args.max_workers,
        expert_wall_ns=args.expert_wall_ns, validate_model_dir=args.validate_model_dir,
        noise_margin=args.noise_margin, proposer=proposer, teachers=args.teachers,
        pass_slot_turns=args.pass_slot_turns)
    best = res.get("best") or {}
    print(f"parent_run={res.get('parent_run_dir')}")
    print(f"best: run_id={best.get('run_id')} lever={best.get('lever')} "
          f"speedup={best.get('speedup')} gate_ok={best.get('gate_ok')}")
    print(f"forks={len(res.get('nodes', [])) - 1} deferred={len(res.get('deferred', []))}")
    print(f"beam_tree: {res.get('parent_run_dir')}/beam_tree.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
