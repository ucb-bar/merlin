"""One operator-authorized gSIM measurement of a passing public scalability case (advisory only).

Reached only through ``agent_selfcheck.py --suite scalability --sim gsim`` in an evaluation child the
self-check broker started for a request id the idle measurer authorized in the run-side ledger
(:mod:`..joint_gsim_measurement`); every other scalability gSIM request is refused exactly as
:func:`.public_scalability.request_policy` refuses it. The case is graded by the ordinary grader with the
self-check's own gSIM adapter set (Spike functional tier, gSIM elaborated-RTL tier) on a private copy of
the frozen case, and the Spike timing model is attached to the same executable, so one response row
holds both the measured gSIM cycles and the modeled estimate of the same program (a calibration pair).

The response is the ordinary gSIM row shape (``sim: gsim``, ``barrier_tier: L3``, ``barrier_cycles``,
``execution_digest``) that :mod:`.performance_headroom`, :mod:`.perf_history`, the timing-model
calibration and the console already read. It is advisory performance evidence: no certificate, no
physical obligation, no promotion, no scalability-suite credit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA = "merlin.phase1.scalability_gsim_measurement.v1"


def _row(name, result, path, *, sim_tier):
    """The public gSIM row of one measured case (outcomes and cycles, never values or private paths)."""
    from merlin.targetgen.capsule_common import tier_status

    from ..timing import barrier_measurements
    from .public_scalability import _observed_result
    from .qa import _execution_digest_from_result

    observed = _observed_result(result)
    tiers = result.get("tiers") or {}
    statuses = {tier: tier_status(record) for tier, record in tiers.items()}
    performance = result.get("performance") if isinstance(result.get("performance"), dict) else None
    digest = _execution_digest_from_result(path) if path is not None and path.is_file() else None
    identity = "execution_digest"
    if digest is None and isinstance(performance, dict) and isinstance(performance.get("elf_sha256"), str):
        digest, identity = performance["elf_sha256"], "elf_sha256"
    numerics = statuses.get("L2") == "pass" and statuses.get(sim_tier) == "pass"
    row = {
        "capsule": name,
        "status": result.get("status"),
        "tiers": statuses,
        "pass": numerics,
        "numerical_execution_passed": numerics,
        "barrier_tier": sim_tier,
        "barrier_status": statuses.get(sim_tier),
        "execution_digest": digest,
        "execution_identity": identity,
        **barrier_measurements(tiers.get(sim_tier) or {}, cert_tier=True),
        **{key: observed[key] for key in ("failure",) if observed.get(key)},
    }
    if performance is not None:
        row["performance"] = performance
    return row


def measure(submission, *, root, record, context, output, timeout, contract=None, names, max_workers=0):
    """Grade ``names`` (frozen scalability cases) with the gSIM adapter set; return the public rows."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.capsule_common import load_capsule
    from merlin_experiments.corpus.preparation import copy_input

    from ...corpus import public_scalability as S
    from .joint_performance import attach as attach_performance
    from .joint_performance import observation_budget
    from .public_scalability import _readback, _sim_via
    from .selfcheck import _adapters

    names = sorted(set(names))
    if not names or not set(names) <= set(record["cases"]):
        raise ValueError("gSIM measurement names unknown or empty scalability cases")
    if S.verify(root, target=context.target) != record:
        raise ValueError("public scalability selected record changed")
    adapters, engine = _adapters("gsim", context.target, _sim_via(context), **_readback(context))
    tiers = sorted(adapters)
    if "L2" not in adapters or len(tiers) < 2:
        raise ValueError("selected target has no Spike plus gSIM adapter set")
    sim_tier = tiers[-1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = output / "subset"
    for name in names:
        copy_input(root / "cases" / name, cases / name, private=True)
    deadline = time.monotonic() + timeout
    results = {}

    def completed(result):
        name = result.get("capsule")
        generated = output / "runs" / "runs" / CR.suite_for(context.target) / str(name) / "generated"
        attach_performance(
            result,
            generated,
            result=result,
            context=context,
            compiler=Path(submission),
            capsule=load_capsule(root / "cases" / name),
            remaining_s=observation_budget(deadline),
            request_deadline=deadline,
        )
        results[name] = result

    from contextlib import redirect_stderr, redirect_stdout

    with (output / "grader.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        CG.grade(
            submission,
            capsules_root=cases,
            runs_root=output / "runs",
            target=context.target,
            labels={"public", "dev"},
            contract=contract or context.repo / "merlin/contract",
            oracle_adapters=adapters,
            timeout=timeout,
            max_workers=max_workers,
            on_capsule_complete=completed,
        )
    suite = output / "runs" / "runs" / CR.suite_for(context.target)
    return [
        _row(name, results[name], suite / name / "capsule_result.json", sim_tier=sim_tier)
        for name in names
        if name in results
    ], engine


def _engine_ceiling(context) -> int | None:
    """The selected gSIM process ceiling in seconds, as the backend states it; ``None`` when unknown."""
    try:
        from merlin.runtime.backends import base

        limit = getattr(base.get_backend(context.target), "execution_timeout_limit", None)
        value = limit("gsim") if callable(limit) else None
    except Exception:  # noqa: BLE001 -- an unknown ceiling leaves the lane's own timeout
        return None
    return int(value) if type(value) in (int, float) and value > 0 else None


def cli(args, *, context):
    """The measurement request's evaluation child (see the module docstring).

    Its gSIM runs may reach the simulated-cycle bound the lane derived from this job's wall budget
    (:mod:`..gsim_engine_bound`); how they went -- cycles, wall seconds, a stop at the cycle bound -- and
    the selected process ceiling are recorded in the lane's ledger for later jobs, even when the grade
    itself fails."""
    from .. import gsim_engine_bound as E
    from .. import joint_gsim_measurement as M

    rid = M.requested()
    if rid is None:
        raise ValueError("public scalability accepts only Spike functional requests")
    if not args.out:
        raise ValueError("public scalability requires a result output path")
    from . import public_scalability as P
    from . import scratch_retention
    from .compiler_profile_runtime import selfcheck_execution

    selection = P.selected(context)
    if selection is None:
        raise ValueError("no public scalability suite is selected")
    root, record = selection
    names = [v.strip() for v in str(args.capsules).split(",") if v.strip()]
    scratch_retention.raw_runs(Path(args.out).with_suffix(".profile"))
    scratch_retention.raw_runs(Path(args.out).with_suffix(".gsim"))
    job = M.read().get("jobs", {}).get(rid)
    job = job if isinstance(job, dict) else {}
    bound = {"max_cycles": job.get("engine_max_cycles"), "basis": job.get("engine_bound_basis")}
    ceiling = None
    try:
        with selfcheck_execution(
            args.compiler_bundle_manifest,
            context=context,
            submission=Path(args.submission),
            work_root=Path(args.out).with_suffix(".profile"),
        ), E.engine_cycle_budget(bound["max_cycles"]):
            ceiling = _engine_ceiling(context)
            rows, engine = measure(
                args.submission,
                root=root,
                record=record,
                context=context,
                output=Path(args.out).with_suffix(".gsim"),
                timeout=args.timeout,
                contract=args.contract,
                names=names,
                max_workers=args.workers,
            )
    finally:
        runs = E.engine_runs(Path(args.out).with_suffix(".gsim"))
        M.note_result(
            rid, {"engine_runs": runs, "engine_stop": E.engine_stop(runs), "engine_timeout_ceiling_s": ceiling}
        )
    stop = E.engine_stop(runs)
    measured = [row for row in rows if row.get("pass") and row.get("barrier_cycles")]
    for row in rows:
        M.note_result(
            rid,
            {
                "outcome": "measured" if row in measured else "cycle_cap" if stop else "not_measured",
                "barrier_cycles": row.get("barrier_cycles"),
                "numerics_passed": row.get("pass"),
                "execution_digest": row.get("execution_digest"),
                "elf_sha256": (row.get("performance") or {}).get("elf_sha256"),
            },
        )
    public = {
        "schema": SCHEMA,
        "suite": "scalability",
        "sim": engine,
        "measurement_lane": M.ORIGIN,
        "advisory_only": True,
        "certification_authority": False,
        "promotion_authority": False,
        "scalability_suite_credit": False,
        "all_pass": bool(rows) and len(measured) == len(rows),
        "n_capsules": len(rows),
        "n_measured": len(measured),
        "engine_cycle_bound": bound,
        **({"engine_stop": stop} if stop else {}),
        "per_capsule": rows,
        "note": "operator idle-time gSIM measurement of a passing scalability case: advisory performance "
        "evidence only; it certifies nothing and gives the scalability suite no credit",
    }
    from .performance_brief import attach as attach_brief

    attach_brief(public, rows, debuggable=False)
    text = json.dumps(public, indent=2)
    Path(args.out).write_text(text)
    print(text)
    return 0 if public["all_pass"] else 1


__all__ = ["SCHEMA", "cli", "measure"]
