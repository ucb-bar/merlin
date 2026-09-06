#!/usr/bin/env python3
"""Refuse a campaign launch whose tuning corpus the certificate cannot cover.

Campaign #11 was assembled, certified, and launched three times before anyone noticed that the
launch could not have measured anything: `discover_performance_corpus` admits only capsules listed
in MANIFEST.yaml's generated split, and `prepare_development_feedback` raises on any member outside
the GSIM certificate envelope.  Both failures surface deep inside the coordinator, after the harness
snapshot and the CHIA/Ray bring-up, as one opaque StageGateError.  Checking the same two properties
here costs a second and names the offending capsules.

Exit 0 = every discovered member is provenance-clean and inside the envelope.
"""
import sys, pathlib

def main() -> int:
    descriptor, certificate = sys.argv[1], sys.argv[2]
    # HONOUR THE SAME SELECTION THE CAMPAIGN WILL MEASURE. A member the reference simulator cannot
    # execute has no certified workload and is deliberately excluded from the run; checking the
    # whole corpus here would block a launch over a capsule the launch was never going to measure.
    selected = None
    if len(sys.argv) > 3 and sys.argv[3].strip() and sys.argv[3].strip() != "all":
        selected = {name.strip() for name in sys.argv[3].split(",") if name.strip()}
    import perf_agent_stage as S, perf_gsim_gate as G, run_paired_perf_bench as PAIR
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(descriptor)
    try:
        corpus = S.discover_performance_corpus(te)
    except Exception as exc:
        print(f"PREFLIGHT FAIL: the corpus does not discover: {exc}", file=sys.stderr)
        print("  the generated/hand_authored split in MANIFEST.yaml must list every capsule under\n"
              "  the performance phase; regenerate the corpus rather than editing it by hand.",
              file=sys.stderr)
        return 2

    cert = G.load_certificate(certificate)
    outside = []
    members = [m for m in corpus.capsules if selected is None or m.capsule in selected]
    if selected is not None:
        unknown = selected - {m.capsule for m in corpus.capsules}
        if unknown:
            print(f"PREFLIGHT FAIL: selection names {len(unknown)} capsule(s) not in the corpus: "
                  f"{' '.join(sorted(unknown))}", file=sys.stderr)
            return 4
    for member in sorted(members, key=lambda row: (row.family, row.capsule)):
        decision = G.plan_evaluation(cert, PAIR._gsim_workload(member),
                                     phase="development_correctness", gsim_available=True)
        if not (decision.admitted and decision.eligible
                and decision.selected_engine == "gsim" and decision.use_gsim):
            outside.append(f"{member.family}/{member.capsule}")

    # SHARED WORKLOAD IDENTITIES are legal and intended: several families measure the same workload
    # under a different lever, and the campaign measures one member per identity. Reported, never
    # refused -- a 1:1 assumption in `_verify_tuning_certificate` used to refuse the launch over
    # exactly this, and the three capsules below are each REQUIRED by their own family.
    import collections as _collections
    _by_identity = _collections.defaultdict(list)
    for member in members:
        _by_identity[G.workload_sha256(PAIR._gsim_workload(member))].append(
            f"{member.family}/{member.capsule}")
    for _ident, _names in sorted(_by_identity.items()):
        if len(_names) > 1:
            print(f"  shared workload {_ident[:16]}: {' '.join(sorted(_names))} "
                  f"(measured once, by design)")

    excluded = len(corpus.capsules) - len(members)
    print(f"preflight: {len(members)} member(s) to measure"
          + (f" ({excluded} deliberately excluded of {len(corpus.capsules)})" if excluded else "")
          + f", {len(cert.members)} certified workloads, {len(outside)} outside the envelope")
    if outside:
        print("PREFLIGHT FAIL: these capsules have no certified workload, so the stage would "
              "refuse the whole corpus:", file=sys.stderr)
        for name in outside:
            print(f"  - {name}", file=sys.stderr)
        print("  capture and certify them, or take them out of the performance phase.", file=sys.stderr)
        return 3
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
