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


def _expert_register_block(mined_dir: Path | None, op: str, backend: str) -> tuple | None:
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
        # Keyed by the BACKEND being mined, not by a fixed one. The index carries a features block per
        # ISA, and reading a fixed key returned an empty block -- so no register_block, so no
        # divergence -- for every other target, silently.
        rb = (r.get("features", {}) or {}).get(backend, {}).get("register_block", {}) or {}
        mr = rb.get("mr")
        if isinstance(mr, int) and mr > 0:
            mrs.append(mr)
    if not mrs:
        return None
    return (max(mrs), ("vsetvlmax", 4))   # NR = lmul-scaled vsetvlmax (the mined K1 LMUL=m4)


def _has_lane_engine(target: str) -> bool:
    """Does ``target`` declare a lane (vector) engine? Derived from its compute units.

    Fail-OPEN for a target with no resolvable contract, because the mining path's historical default
    is a lane target: a target we cannot classify must not silently LOSE its vector facet, which would
    delete every divergence on it.
    """
    try:
        from merlin.kernels import engines as _eng
        engines = _eng.engines_for(target)
    except Exception:  # noqa: BLE001
        return True
    return ("vector" in engines) if engines else True


def expert_cca_from_policies(policies: list[dict], op: str,
                             backend: str,
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
        compute.register_block = _expert_register_block(mined_dir, op, backend)
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
    # A VectorFacet belongs to a target that HAS a lane engine, which is a property of its declared
    # silicon rather than of its name. The name check here was invisible coupling: it hid behind a
    # module path that used to claim one ISA, and it silently dropped the facet for any other lane
    # target -- including a hybrid whose lane engine is exactly what the policies describe.
    cca = ccamod.CCA(op=op, backend=[backend], compute=compute,
                     vector=vector if _has_lane_engine(backend) else None,
                     provenance={"level": "policy", "source": "mined_policies"})
    return cca, ev_by_axis


def endpoint_for_target(target: str):
    """The compute endpoint to lift ``target`` through, or None for a plain vector target.

    None is the RVV path and is a real answer, not a failure: a lane-only target has no accelerator
    endpoint to decode and its CCA comes from the vector lifter.
    """
    try:
        from merlin.kernels import endpoints as _ep
        eps = [e for e in _ep.endpoints_for(target) if e.roles]
    except Exception:  # noqa: BLE001 — no endpoint data: the vector path, honestly
        return None
    if not eps:
        return None
    # Prefer the endpoint whose roles reach a contraction: that is the one a mined matmul is about.
    for e in eps:
        if "accumulate" in e.roles:
            return e
    return eps[0]


def cca_from_object(obj_path: Path, target: str, op: str, source: str,
                    endpoint=None) -> "ccamod.CCA | None":
    """Lift a CCA from a compiled object, through whichever decoder the TARGET's endpoint needs.

    This is the seam that was missing: the lifter, the role vocabulary and the per-target decoders all
    existed and composed in isolation, and the mining CLI still went through the vector lifter for
    every target — so the pipeline could not see an accelerator's instructions at all.

    Returns None when nothing could be decoded, rather than an empty CCA. An empty CCA compares equal
    to anything on the facets it did not fill, so it reports NO divergence for a kernel the decoder
    never read — a clean result for a stream nobody looked at.
    """
    endpoint = endpoint if endpoint is not None else endpoint_for_target(target)
    if endpoint is None:
        return ccamod.lift_asm(rvv_decode.decode(obj_path), op=op, source=source)

    geometry = _geometry_for(target)
    from merlin.kernels import decode as _dec

    # ONE dispatcher, shared with the audit path. This function used to decode with its own shorter
    # copy of that logic, and the two drifted: no ISA triple/mattr, no roles_of, no stream width, no
    # intrinsics table. Every difference produced the same symptom -- an expert kernel lifting to
    # None, reported as "decoded nothing of this endpoint", which reads as an empty corpus rather
    # than as a decoder that was never handed what it needed.
    settings = _dec.disasm_settings(target, endpoint)
    stream = rvv_decode.decode(obj_path, triple=settings["triple"], mattr=settings["mattr"])
    raws = [i.raw for i in getattr(stream, "insns", ())]
    spans = stream.loop_spans() if getattr(stream, "spans_reliable", lambda: False)() else None

    decoded = _dec.decode_for_endpoint(raws, target, endpoint)
    if not decoded:
        return None            # no object decoder for this endpoint's encoding (e.g. a text ISA)
    if not any(getattr(d, "roles", ()) for d in decoded):
        return None            # decoded nothing of this endpoint: say so instead of lifting a blank
    return ccamod.lift_asm_roles(decoded, endpoint, op=op, source=source,
                                 geometry=geometry, loop_spans=spans)


def _geometry_for(target: str) -> dict:
    """The target's derived hardware constants, for the IDENTITY fields of an engine facet.

    Read from the RTL facts, never inferred from the stream: a kernel using a 16-wide operand says
    nothing about how wide the array is.
    """
    try:
        from merlin.targetgen.rtl import facts as _F
        body = (_F.load_facts(target) or {}).get("facts") or {}
    except Exception:  # noqa: BLE001
        return {}
    mesh = next((a for a in (body.get("arrays") or ()) if a.get("name") == "mesh"), {})
    # Only a CORROBORATED array contributes geometry -- an uncorroborated one is a ranking, not a fact.
    if not mesh.get("corroborated"):
        return {}
    return {"pe_rows": mesh.get("rows"), "pe_cols": mesh.get("cols")}


def expert_cca_from_asm(expert_objs, target: str, op: str, endpoint=None):
    """expert CCA at ``level: asm`` — the thing every shipped mining run has lacked.

    ``expert_cca_from_policies`` summarizes what the mined POLICIES say, which is a reconstruction of
    the expert's intent rather than a reading of the expert's code; every shipped run therefore records
    ``expert_cca level: policy`` against ``our_cca level: asm`` and compares two different kinds of
    thing. This lifts the expert's own emitted stream, so both sides are the same kind of evidence.

    Returns ``(cca, source_path)``; ``(None, "")`` when nothing decoded, never a blank CCA.
    """
    for obj in expert_objs:
        cca = cca_from_object(Path(obj), target, op, source="expert_asm", endpoint=endpoint)
        if cca is not None:
            return cca, str(obj)
    return None, ""


def _our_cca_from_run(runs_root: Path, run_glob: str, op: str, *, target: str,
                      endpoint=None) -> tuple[ccamod.CCA | None, str]:
    """Decode the most-recent matching certified object -> CCA(ours, asm-lifted).

    Routed through the target's own endpoint decoder when it has one. Lifting an accelerator's stream
    with the vector lifter does not fail — it returns an empty vector facet — so the choice has to be
    made here rather than defaulted.
    """
    objs = sorted(runs_root.glob(f"{run_glob}/generated/model.o"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not objs:
        return None, ""
    obj = objs[0]
    return (cca_from_object(obj, target, op, source="ours_baseline", endpoint=endpoint),
            obj.parent.parent.name)


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
    endpoint = endpoint_for_target(target)

    # PREFER asm-vs-asm. The policy summary is a reconstruction of the expert's intent; the expert's
    # own emitted stream is the expert's decision. Every shipped run so far recorded
    # `expert_cca level: policy` against `our_cca level: asm` and compared two different kinds of
    # evidence, which is exactly the comparison this loop exists to avoid making.
    expert_objs = sorted((mined_dir / "expert_objects").glob("*.o")) if mined_dir else []
    expert, expert_src = expert_cca_from_asm(expert_objs, target, op, endpoint) if expert_objs \
        else (None, "")
    ev_by_axis: dict = {}
    if expert is None:
        # Fall back, and RECORD that we did: a run whose expert side is a policy summary must say so,
        # because the two levels are not interchangeable and a reader cannot tell them apart later.
        expert, ev_by_axis = expert_cca_from_policies(policies, op, backend=target,
                                                      mined_dir=mined_dir)
        expert_src = "mined_policies"

    ours, baseline_run = _our_cca_from_run(runs_root, baseline_run_glob, op,
                                           target=target, endpoint=endpoint)
    # A MISSING SIDE IS NOT AGREEMENT. With no `ours` this returned [], and the run reported
    # "divergences=0 actions=0" -- indistinguishable from a genuine match, for a comparison that was
    # never made. Measured: a run whose expert object failed to lift printed exactly that. The count
    # is only meaningful when BOTH sides lifted, so say which one did not.
    comparable = expert is not None and ours is not None
    divs = cca_compare.compare(expert, ours) if comparable else []
    missing = [n for n, c in (("expert", expert), ("ours", ours)) if c is None]
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
        # The LEVEL each side was lifted at. A run that compared a policy summary against emitted asm
        # is a weaker result than one that compared two streams, and the difference must be legible in
        # the artifact rather than inferred from which fields happen to be filled.
        # Which side(s) never lifted. Empty on a real comparison; a reader must never have to infer
        # "we compared nothing" from a zero.
        "sides_missing": missing,
        "comparison_made": comparable,
        "expert_level": (expert.provenance or {}).get("level") if expert is not None else None,
        "our_level": ((ours.provenance or {}).get("level") if ours else None),
        "expert_source": expert_src,
        "endpoint": getattr(endpoint, "name", None),
        "policies_used": [p["policy"] for p in policies],
        "n_divergences": len(divs), "n_actions": len(actions), "n_unrouted": len(unrouted),
    }, sort_keys=False))
    (run_dir / "expert_cca.yaml").write_text(yaml.safe_dump(expert.to_dict(), sort_keys=False))
    if ours:
        (run_dir / "our_cca.yaml").write_text(yaml.safe_dump(ours.to_dict(), sort_keys=False))
    (run_dir / "divergences.yaml").write_text(yaml.safe_dump(
        [{"axis": d.axis, "expert": d.expert, "ours": d.ours, "backend": d.backend,
          "evidence": d.evidence} for d in divs], sort_keys=False))
    # BOTH lowering traces, so a divergence can be attributed to a STEP rather than to a vendor.
    # Without the expert's side a divergence bottoms out at "their compiler is better", which cannot
    # be acted on: it does not distinguish their -O pipeline from their ISA flags from a source
    # decision. The expert's steps are stamped unmodifiable, which is the point rather than an
    # omission -- ours records the seam that can change each step and theirs has no such seam.
    _emit_traces(run_dir, target, expert, ours, expert_src, endpoint)

    # The corpus as a DSE search space. Emitted as a declared search-space document so the existing
    # explorer reads it like any other, rather than the corpus evidence living only in this run.
    _emit_design_space(run_dir, target, policies)

    (run_dir / "actions.yaml").write_text(yaml.safe_dump(
        [{"axis": a.divergence_axis, "class": a.action_class, "target_seam": a.target_seam,
          "forkable_now": a.forkable_now, "change": a.change,
          "expected_effect": a.expected_effect, "evidence": a.evidence} for a in actions],
        sort_keys=False))
    return run_dir


def _emit_traces(run_dir: Path, target: str, expert, ours, expert_src: str, endpoint) -> None:
    """Write both sides' lowering traces under ``traces/``. Never blocks a run: a trace we could not
    build is an absent file with a recorded reason, not a failed mining run."""
    from merlin.kernels import trace as _trace

    out = run_dir / "traces"
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if expert is not None:
        hand = (expert.provenance or {}).get("level") == "policy"
        try:
            t = _trace.LoweringTrace(
                kernel=expert.op, target=target, source=expert_src or "expert",
                steps=[], asm=None,
                provenance={**dict(expert.provenance or {}),
                            "expert_source": expert_src,
                            "no_lowering": ("expert side is a policy summary, not a build: there is "
                                            "no lowering to reconstruct from it") if hand else None})
            (out / "expert_trace.yaml").write_text(yaml.safe_dump(t.to_dict(), sort_keys=False))
            (out / "expert_trace.md").write_text(t.to_markdown())
        except Exception as exc:  # noqa: BLE001
            notes.append(f"expert trace unavailable: {type(exc).__name__}: {exc}")
    if ours is not None:
        try:
            t = _trace.LoweringTrace(
                kernel=ours.op, target=target, source="ours", steps=_trace.pipeline_steps(target),
                asm=None, provenance=dict(ours.provenance or {}))
            (out / "our_trace.yaml").write_text(yaml.safe_dump(t.to_dict(), sort_keys=False))
            (out / "our_trace.md").write_text(t.to_markdown())
            if expert is not None:
                rep = _trace.traces_agree(t, _trace.LoweringTrace(
                    kernel=expert.op, target=target, source=expert_src or "expert", steps=[]))
                (out / "trace_agreement.yaml").write_text(yaml.safe_dump(rep, sort_keys=False))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"our trace unavailable: {type(exc).__name__}: {exc}")
    if notes:
        (out / "NOTES.txt").write_text("\n".join(notes) + "\n")


def _emit_design_space(run_dir: Path, target: str, policies: list[dict]) -> None:
    """Write the mined corpus as a declared DSE search space, when the policies carry shape axes.

    Absent shapes, writes nothing rather than an empty grid: a search space with no axes is not a
    small space, it is a missing derivation, and emitting one invites a search that finds nothing and
    reports the space as explored.
    """
    from merlin.kernels import space as _space

    recs = [p.get("shape") for p in policies if isinstance(p.get("shape"), dict)]
    recs = [r for r in recs if r]
    if not recs:
        return
    axes = tuple(sorted({k for r in recs for k in r}))
    sp = _space.space_from_records(recs, axes=axes, target=target)
    if not sp.axes:
        return
    (run_dir / "design_space.yaml").write_text(
        yaml.safe_dump(_space.to_search_space_doc(sp), sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # REQUIRED, not defaulted. A default of one target silently mislabels every run for another
    # one -- the mined artifacts are written under <target>/ and the CCA is compared against that
    # target's expert corpus, so a mislabelled run compares the wrong things and says nothing about it.
    ap.add_argument("--target", required=True,
                    help="the target whose expert corpus is mined and whose endpoint is lifted")
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
    if not man.get("comparison_made", True):
        # Loudly, and with a non-zero exit: a zero here would read as "the expert and we agree".
        print(f"  NO COMPARISON MADE — {', '.join(man.get('sides_missing') or ['?'])} did not lift. "
              f"This is NOT agreement; nothing was compared.")
        return 2
    print(f"  divergences={man['n_divergences']} actions={man['n_actions']} "
          f"unrouted={man['n_unrouted']}  "
          f"[expert={man.get('expert_level')} ours={man.get('our_level')}]")
    if man.get("expert_level") != "asm":
        print("  NOTE: the expert side is a POLICY SUMMARY, not the expert's emitted code — a "
              "weaker result than an asm-vs-asm comparison. Supply <mined>/expert_objects/*.o.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
