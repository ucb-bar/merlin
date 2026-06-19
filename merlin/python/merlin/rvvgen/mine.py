"""``merlin-rvv-mine`` — the deterministic mining driver that mints a versioned run.

Orchestrates the robust chain into ONE reproducible artifact, per the run discipline
(`mined_knowledge/<target>/mining_<target>_v{V}_{ts}/`):

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

import yaml

from ..kernels import action_catalog as ac
from ..kernels import cca as ccamod
from ..kernels import cca_compare
from ..kernels.decode import rvv as rvv_decode

_REPO = Path(__file__).resolve().parents[4]


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


def expert_cca_from_policies(policies: list[dict], op: str,
                             backend: str = "rvv") -> tuple[ccamod.CCA, dict[str, list[str]]]:
    """Build the EXPERT CCA target from the promoted, evidence-backed mining policies + a per-axis
    evidence map (axis -> the kernels of the policy that asserts it), so each divergence/action
    cites the kernels that actually justify it (not a global aggregate)."""
    by_name = {p["policy"]: p for p in policies}
    names = set(by_name)
    compute = ccamod.ComputeFacet(op=op)
    vector = ccamod.VectorFacet()
    if "fma_broadcast_policy" in names:
        compute.contraction_form = "fused_fma"
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


def _next_version(out_root: Path, target: str) -> int:
    existing = list(out_root.glob(f"mining_{target}_v*"))
    return len(existing) + 1


def mine_run(target: str, op: str, runs_root: Path, mined_dir: Path, out_root: Path,
             baseline_run_glob: str = "hand_v0_*") -> Path:
    """Mint a mining_<target>_vX_ts/ run from on-disk artifacts; return its path."""
    policies = yaml.safe_load((mined_dir / "policy_rules.yaml").read_text()) or []
    expert, ev_by_axis = expert_cca_from_policies(policies, op, backend=target)
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
    ap.add_argument("--runs-root", default="runs/rvv_experiment")
    ap.add_argument("--mined", required=True, help="mined_knowledge/<target>/<mining run> dir")
    ap.add_argument("--out-root", default=None, help="default mined_knowledge/<target>/")
    ap.add_argument("--baseline-glob", default="hand_v0_*")
    a = ap.parse_args(argv)
    out_root = Path(a.out_root) if a.out_root else _REPO / "mined_knowledge" / a.target
    run_dir = mine_run(a.target, a.op, Path(a.runs_root), Path(a.mined), out_root,
                       baseline_run_glob=a.baseline_glob)
    man = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    print(f"minted {run_dir}")
    print(f"  divergences={man['n_divergences']} actions={man['n_actions']} "
          f"unrouted={man['n_unrouted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
