"""``merlin-rvv-mine`` — the deterministic mining driver that mints a versioned run.

Orchestrates the robust chain into ONE reproducible artifact, per the run discipline
(`artifacts/kernel-mining/<target>/mining_<target>_v{V}_{ts}/`):

  our codegen object (from a certified target package)  --decode.rvv-->  CCA(ours, asm)
  mined policies (mining artifact)                        --------------->  CCA(expert)
        cca_compare  -->  Divergences  -->  action_catalog  -->  typed CompilerActions

Writes the run folder (manifest + our/expert CCA + divergences + typed actions) so the "how we
mined → which abstractions → which compiler changes" chain is auditable and re-runnable, not
session-dependent. Target-agnostic: keyed by ``backend``; RVV is the first instantiation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from merlin.common.paths import repo_root

import yaml

from ..kernels import action_catalog as ac
from ..kernels import cca as ccamod
from ..kernels import cca_compare
from ..kernels.decode import rvv as rvv_decode
from ..common.paths import artifacts_dir

_REPO = repo_root()


# ---- abstraction -> expected codegen (expert CCA from mined policies) ----------------

# Which mined policy asserts each CCA facet axis (so each divergence cites the RIGHT kernels).
_AXIS_POLICY = {
    "compute.contraction_form": "fma_broadcast_policy",
    "compute.widening": "int8_widening_policy",
    "compute.epilogue": "requant_narrowing_policy",
    "compute.reduction_form": "vector_reduction_policy",
    "compute.accumulator_resident": "accumulator_commit_policy",
    "compute.nr_is_vsetvlmax": "vl_agnostic_loop_policy",
    "vector.lmul": "lmul_grouping_policy",
    "vector.vl_strategy": "vl_tail_policy",
}


def _expert_register_block(mined_dir: Path | None, op: str) -> tuple | None:
    """The experts' representative register block (MR, NR) read from the mined kernel index — the
    #1 GEMM data-movement decision the experts make and we were blind to. We surface the MAX MR the
    expert GEMM kernels ship for this op (XNNPACK ships MR=1..7 and SELECTS the high one; OpenBLAS
    16x8 -> MR=16), since that is the reuse ceiling the experts reach and we must learn to match.
    Returns None when no index / no register-blocked expert GEMM kernel is found."""
    if mined_dir is None:
        return None
    import json
    idx = mined_dir / "xnnpack_index.json"
    if not idx.is_file():
        return None
    try:
        recs = (json.loads(idx.read_text()) or {}).get("records", [])
    except Exception:  # noqa: BLE001
        return None
    op_aliases = {"gemm", "matmul", "igemm"} if op in ("gemm", "matmul") else {op}
    mrs = []
    for r in recs:
        if r.get("op") not in op_aliases or r.get("dtype") not in ("f32", "f16"):
            continue
        rb = (r.get("features", {}) or {}).get("rvv", {}).get("register_block", {}) or {}
        mr = rb.get("mr")
        if isinstance(mr, int) and mr > 0:
            mrs.append(mr)
    if not mrs:
        return None
    return (max(mrs), ("vsetvlmax", 4))   # NR = lmul-scaled vsetvlmax (the mined K1 LMUL=m4)


def expert_cca_from_policies(policies: list[dict], op: str,
                             backend: str = "rvv",
                             mined_dir: Path | None = None) -> tuple[ccamod.CCA, dict[str, list[str]]]:
    """Build the EXPERT CCA target from the promoted, evidence-backed mining policies + a per-axis
    evidence map (axis -> the kernels of the policy that asserts it), so each divergence/action
    cites the kernels that actually justify it (not a global aggregate)."""
    by_name = {p["policy"]: p for p in policies}
    names = set(by_name)
    compute = ccamod.ComputeFacet(op=op)
    vector = ccamod.VectorFacet()
    if "fma_broadcast_policy" in names:
        compute.contraction_form = "fused_fma"
        # The fma_broadcast_policy's third action is `register_block_rhs` — the experts reuse each
        # loaded RHS row across MR broadcast-FMA accumulators. Read the MR they actually ship from
        # the kernel index so register-blocking becomes a real, learned divergence (not null).
        compute.register_block = _expert_register_block(mined_dir, op)
    if "int8_widening_policy" in names:
        compute.widening = True
    if "requant_narrowing_policy" in names:
        compute.epilogue = "requant_narrow"
    if "vector_reduction_policy" in names:
        compute.reduction_form = "vredsum_tree"
    if "accumulator_commit_policy" in names:
        # the mined accumulator_commit_policy (keep_accumulator_resident / single_commit_store) =
        # the expert keeps the C accumulator resident across the reduction and commits once.
        compute.accumulator_resident = True
    if "vl_agnostic_loop_policy" in names:
        compute.nr_is_vsetvlmax = True          # NR tracks vsetvlmax (VL-adaptive expert)
    if "lmul_grouping_policy" in names:
        vector.lmul = 4.0                       # prefer_high_lmul -> m4 (mined target)
    if "vl_tail_policy" in names or "vl_agnostic_loop_policy" in names:
        vector.vl_strategy = "vsetvl_loop"
    ev_by_axis = {axis: list(by_name.get(pol, {}).get("evidence", []))[:8]
                  for axis, pol in _AXIS_POLICY.items() if pol in names}
    cca = ccamod.CCA(op=op, backend=[backend], compute=compute,
                     vector=vector if backend == "rvv" else None,
                     provenance={"level": "policy", "source": "mined_policies"})
    return cca, ev_by_axis


def _our_cca_from_run(runs_root: Path, run_glob: str, op: str) -> tuple[ccamod.CCA | None, str]:
    """Decode the most-recent matching certified object -> CCA(ours, asm-lifted)."""
    objs = sorted(runs_root.glob(f"{run_glob}/generated/model.o"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not objs:
        return None, ""
    obj = objs[0]
    return ccamod.lift_asm(rvv_decode.decode(obj), op=op, source="ours_baseline"), obj.parent.parent.name


def audit_achievement(action, obj_path: Path, op: str) -> dict:
    """CLOSE THE LOOP: did the fork that applied ``action`` actually ACHIEVE its machine-readable
    promise (``intended_facet``)? Lift the CCA from the fork's EMITTED asm, check the residual, and —
    when the action did NOT deliver — propose the ESCALATED action (next-stronger class for the unmet
    axis). This is what lets the mining loop detect "we routed `accumulator_resident` but emitted
    resident=False" and escalate to the CODEGEN microkernel, instead of silently accepting a structural
    miss. Returns {achieved, residual, escalations} (escalations is a list of CompilerAction)."""
    from ..kernels import action_catalog as ac
    from ..kernels.cca_compare import Divergence

    achieved_cca = ccamod.lift_asm(rvv_decode.decode(Path(obj_path)), op=op, source="fork_emitted")
    residual = ac.achieved_residual(action, achieved_cca)
    if not residual:
        return {"achieved": True, "residual": [], "escalations": []}
    escalations = []
    for axis in residual:
        want = (action.intended_facet or {}).get(axis)
        d = Divergence(axis=axis, expert=want, ours=ac._facet_value(achieved_cca, axis),
                       backend=action.backend, evidence=list(action.evidence))
        esc = ac.route_escalated(d, action.action_class)
        if esc is not None:
            escalations.append(esc)
    return {"achieved": False, "residual": residual, "escalations": escalations}


def _next_version(out_root: Path, target: str) -> int:
    existing = list(out_root.glob(f"mining_{target}_v*"))
    return len(existing) + 1


def mine_run(target: str, op: str, runs_root: Path, mined_dir: Path, out_root: Path,
             baseline_run_glob: str = "hand_v0_*") -> Path:
    """Mint a mining_<target>_vX_ts/ run from on-disk artifacts; return its path."""
    policies = yaml.safe_load((mined_dir / "policy_rules.yaml").read_text()) or []
    expert, ev_by_axis = expert_cca_from_policies(policies, op, backend=target, mined_dir=mined_dir)
    ours, baseline_run = _our_cca_from_run(runs_root, baseline_run_glob, op)
    divs = cca_compare.compare(expert, ours) if ours else []
    for d in divs:                       # attach the evidence of the policy that asserts this axis
        d.evidence = ev_by_axis.get(d.axis, [])
    actions, unrouted = ac.build_catalog(divs)

    v = _next_version(out_root, target)
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_dir = out_root / f"mining_{target}_v{v}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump({
        "run_id": run_dir.name, "target": target, "op": op, "version": v, "created": ts,
        "mined_from": str(mined_dir), "baseline_run": baseline_run,
        "policies_used": [p["policy"] for p in policies],
        "n_divergences": len(divs), "n_actions": len(actions), "n_unrouted": len(unrouted),
    }, sort_keys=False))
    (run_dir / "expert_cca.yaml").write_text(yaml.safe_dump(expert.to_dict(), sort_keys=False))
    if ours:
        (run_dir / "our_cca.yaml").write_text(yaml.safe_dump(ours.to_dict(), sort_keys=False))
    (run_dir / "divergences.yaml").write_text(yaml.safe_dump(
        [{"axis": d.axis, "expert": d.expert, "ours": d.ours, "backend": d.backend,
          "evidence": d.evidence} for d in divs], sort_keys=False))
    (run_dir / "actions.yaml").write_text(yaml.safe_dump(
        [{"axis": a.divergence_axis, "class": a.action_class, "target_seam": a.target_seam,
          "forkable_now": a.forkable_now, "change": a.change,
          "expected_effect": a.expected_effect, "evidence": a.evidence} for a in actions],
        sort_keys=False))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="rvv")
    ap.add_argument("--op", default="matmul")
    ap.add_argument("--runs-root", default="out/runs/rvv_experiment")
    ap.add_argument("--mined", required=True, help="out/artifacts/kernel-mining/<target>/<mining run> dir")
    ap.add_argument("--out-root", default=None, help="default artifacts/kernel-mining/<target>/")
    ap.add_argument("--baseline-glob", default="hand_v0_*")
    a = ap.parse_args(argv)
    out_root = Path(a.out_root) if a.out_root else artifacts_dir() / "kernel-mining" / a.target
    run_dir = mine_run(a.target, a.op, Path(a.runs_root), Path(a.mined), out_root,
                       baseline_run_glob=a.baseline_glob)
    man = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    print(f"minted {run_dir}")
    print(f"  divergences={man['n_divergences']} actions={man['n_actions']} "
          f"unrouted={man['n_unrouted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
