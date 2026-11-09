"""Capsule evaluation API and supervised command-line entrypoint."""

from __future__ import annotations

import sys

# Explicit compatibility exports retain the established runner API.
from merlin.evaluation.capsule.adapters import _sim_engine_adapters as _sim_engine_adapters
from merlin.evaluation.capsule.bundle import _content_identity as _content_identity
from merlin.evaluation.capsule.bundle import _file_identity as _file_identity
from merlin.evaluation.capsule.bundle import _model_asset as _model_asset
from merlin.evaluation.capsule.bundle import _model_runtime_bundle as _model_runtime_bundle
from merlin.evaluation.capsule.bundle import _numpy_dtype as _numpy_dtype
from merlin.evaluation.capsule.bundle import _write_deterministic_npz as _write_deterministic_npz
from merlin.evaluation.capsule.operation import run_capsule as run_capsule
from merlin.evaluation.capsule.suite import run_suite as run_suite
from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.targetgen import oracle_policy as _oracle_policy
from merlin.targetgen.capsule_common import discover_capsules as discover_capsules
from merlin.targetgen.capsule_common import load_capsule as load_capsule
from merlin.targetgen.oracle_policy import _SIM_ORACLES as _SIM_ORACLES
from merlin.targetgen.oracle_policy import _bespoke_sim_via as _bespoke_sim_via
from merlin.targetgen.oracle_policy import _chipyard_available as _chipyard_available
from merlin.targetgen.oracle_policy import _endpoint_of as _endpoint_of
from merlin.targetgen.oracle_policy import _ensure_sim_oracles_discovered as _ensure_sim_oracles_discovered
from merlin.targetgen.oracle_policy import _screen_tiers_of as _screen_tiers_of
from merlin.targetgen.oracle_policy import _sim_oracle_lock as _sim_oracle_lock
from merlin.targetgen.oracle_policy import _SimOracle as _SimOracle
from merlin.targetgen.oracle_policy import chipyard_l3_selection as chipyard_l3_selection
from merlin.targetgen.oracle_policy import describe_l3_engine as describe_l3_engine
from merlin.targetgen.oracle_policy import register_sim_oracle as register_sim_oracle
from merlin.targetgen.oracle_policy import sim_oracle_caps as sim_oracle_caps
from merlin.targetgen.runner_config import CONVENTIONAL_RTL_TIERS as _RTL_TIERS  # noqa: F401 -- legacy override
from merlin.targetgen.runner_config import CONVENTIONAL_TIER_SIM as _TIER_SIM  # noqa: F401 -- legacy override
from merlin.targetgen.runner_config import _config_for_target as _config_for_target
from merlin.targetgen.runner_config import _default_config as _default_config


def main(argv: list[str] | None = None) -> int:
    import argparse

    # The child half of a BUDGETED whole-model grade (see _grade_model_capsule). Parsed before the
    # normal arguments because it shares none of them: the whole invocation is one spec file in and
    # one result file out, so the parent can kill the process group without losing what it knows.
    _av = list(sys.argv[1:] if argv is None else argv)
    if "--operator-grade" in _av:
        from merlin.experiments.phase1.feedback.operator_child import main as run_operator_grade

        return run_operator_grade(_av)
    if "--model-grade" in _av:
        from merlin.experiments.phase1.feedback.compiler_profile_child import run_model_grade

        return run_model_grade(_av)

    ap = argparse.ArgumentParser(description="capsule_bench_v0 runner")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsule", help="path to a single capsule dir")
    ap.add_argument("--capsules-root", help="run every capsule under this root")
    ap.add_argument("--labels", default="public,dev", help="comma-separated label filter")
    ap.add_argument("--runs-root", default="out/runs/capsule_bench")
    ap.add_argument("--contract", default="src/merlin/resources/contracts")
    ap.add_argument("--target", required=True, help="target to grade (its config/oracle are derived)")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)

    if not a.capsule and not a.capsules_root:
        ap.error("one of --capsule or --capsules-root is required")
    labels = None
    if a.capsule:
        caps = [load_capsule(a.capsule, contract=a.contract)]
    else:
        labels = set(a.labels.split(",")) if a.labels else None
        caps = discover_capsules(a.capsules_root, labels=labels, contract=a.contract)
    # AN EMPTY COHORT IS NOT A PASS. `npass == len(results)` is vacuously true at zero, so a root that
    # matched nothing printed "0/0 pass" and exited 0 -- byte-identical to a suite that ran and passed.
    # Measured: a capsules root built from symlinked capsule dirs discovers nothing, because
    # `discover_capsules` walks with `rglob`, which does not descend into symlinked directories. The run
    # reported success without grading a single capsule. Refuse the verdict instead, and say which of
    # the two reasons it was so the caller does not have to guess.
    if not caps:
        where = a.capsule or a.capsules_root
        why = f" at label(s) {sorted(labels)}" if labels else ""
        print(
            f"no capsule discovered under {str(where)!r}{why} -- refusing to report a verdict over "
            f"an empty cohort (capsule discovery does not descend into symlinked directories)",
            file=sys.stderr,
        )
        return 2
    results = run_suite(caps, a.package, runs_root=a.runs_root, contract=a.contract, timeout=a.timeout, target=a.target)
    npass = sum(1 for r in results if r["status"] == "pass")
    for r in results:
        print(f"  [{r['status']:10s}] {r['capsule']}")
    print(f"\n{npass}/{len(results)} pass")
    # Same fail-closed reasoning one level down: capsules were discovered but the suite returned no
    # row for any of them, so nothing was measured and there is no verdict to report.
    if not results:
        print(
            f"{len(caps)} capsule(s) discovered but the suite returned no result row -- nothing was measured",
            file=sys.stderr,
        )
        return 2
    return 0 if npass == len(results) else 1


# Late lookup is intentional: independent adapters may replace this established
# registry callback before selecting an oracle. The actual implementation lives
# in adapters; this API does not invent a second registry.
_oracle_policy.bind_sim_oracle_adapters("chipyard", lambda target: _sim_engine_adapters("chipyard", target))

if __name__ == "__main__":
    raise SystemExit(main())
