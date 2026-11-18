"""The idle measurer's gSIM lane: measure passing scalability cases on gSIM.

Order (:func:`order`): first, per family, the smallest case with no timing evidence at all, so every
family gets cycles early; then largest headroom first. Within both, the representatives of the bundles of
passing cases (:func:`.family_rollup.preferred_cases`: a newly passing large case of a family and shape
class) go first, so evidence follows each case as it passes -- an ordering preference, never a skip. A
job long relative to the smaller pending work runs only in the long lane (:func:`long_lane`).

:class:`.idle_performance.IdleMeasurer` calls :meth:`GsimLane.tick` once its idle-capacity gates are open
(no author request waiting, free shared CPU slots beyond an admitting grade's reserve, disk room). The
lane ranks the passing performance-cohort cases whose current executable has no gSIM measurement and no
earlier attempt by headroom over their structural roofline bound:

* a case with a Spike timing-model estimate: its modeled cycles minus the bound, executable = the ELF
  its newest passing estimate describes;
* a case without one (the model could not account for every event, as for many of the largest
  shapes): cycles predicted from its structural floor (compute floor or ideal request service, from the
  target facts) and the inefficiency this run's measured gSIM rows show over their own floors
  (:func:`..joint_gsim_measurement.predicted_cycles`), executable = the ELF its newest verdict row's
  performance record names.

The job timeout is derived from those cycles (the upper prediction for a bound-only case) and this run's
recorded gSIM throughput; the lane reserves it in the campaign's measurement budget
(:mod:`..joint_gsim_measurement`) and submits one detached operator request (``suite: scalability``,
``sim: gsim``) at the advisory lease class. A case too large for the remaining budget or window is
skipped for a smaller one. The emulator's simulated-cycle bound is carried from the same timeout
(:mod:`..gsim_engine_bound`), so the harness's cycle cap does not end a run its wall budget allows; a case
whose expected runtime exceeds the selected gSIM process ceiling is skipped, and the log says why. After
the executable changes the case is eligible again, so the sample rotates with the author's edits. Nothing
here waits, retries in place or raises into the grader.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from merlin.evaluation.capsule import eligibility as capsule_eligibility

# ---- idle_gsim ---------------------------------------------------------------------------------

_STOP = (
    "no joint campaign",
    "gSIM measurement disabled by policy",
    "campaign window closed",
    "campaign clock not started",
    "measurement jobs in flight",
    "no free idle slot",
    "gSIM slots held by correctness or witness work",
)


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > (16 << 20):
            return None
        from merlin.development.backend.feedback.selfcheck.results import hydrate

        return hydrate(json.loads(path.read_text()), document_path=path)
    except (OSError, ValueError):
        return None


def measured_executables(channel: Path) -> set[tuple[str, str]]:
    """``(case, executable)`` pairs that already have a gSIM measurement row in this channel."""
    from merlin.development.backend.campaign.joint_gsim_measurement import REQUEST_PREFIX

    found: set[tuple[str, str]] = set()
    for path in Path(channel).glob(f"resp_{REQUEST_PREFIX}*.json"):
        document = _load(path)
        if not isinstance(document, dict) or document.get("sim") != "gsim":
            continue
        for row in document.get("per_capsule") or ():
            if not isinstance(row, dict) or not row.get("barrier_cycles"):
                continue
            for key in ((row.get("performance") or {}).get("elf_sha256"), row.get("execution_digest")):
                if isinstance(key, str) and key:
                    found.add((str(row.get("capsule")), key))
    return found


def _estimate(rows, digest):
    """The newest passing estimate of ``digest``, else the newest passing one: ``(cycles, performance)``."""
    passing = [row for row in rows or () if len(row) < 4 or row[3]]
    chosen = next((row for row in passing if row[1] == digest), passing[0] if passing else None)
    if chosen is None:
        return None
    performance = chosen[2] if isinstance(chosen[2], dict) else {}
    cycles = performance.get("estimated_whole_soc_cycles")
    return (cycles, performance) if type(cycles) in (int, float) and cycles > 0 else None


SOURCE_KEY = "_graded_source_sha256"


def verdict_performance(verdicts) -> dict:
    """``{case: performance record}`` of each case's newest verdict row that carries one (newest first)."""
    found = {}
    for verdict in verdicts:
        for row in verdict.get("per_capsule") or ():
            name = row.get("capsule") if isinstance(row, dict) else None
            if isinstance(name, str) and name not in found and isinstance(row.get("performance"), dict):
                graded = row.get("graded") if isinstance(row.get("graded"), dict) else {}
                source = graded.get("graded_source_sha256") or verdict.get("graded_source_sha256")
                found[name] = {**row["performance"], SOURCE_KEY: source if isinstance(source, str) else None}
    return found


def candidates(
    cohort,
    l2: dict,
    known: dict,
    digest,
    *,
    measured=frozenset(),
    attempted=frozenset(),
    records=None,
    model=None,
    families=None,
    preferred=frozenset(),
) -> list[dict]:
    """Passing cohort cases whose current executable lacks a gSIM measurement, in :func:`order`.

    ``records`` are the cases' verdict performance records and ``model`` the run's measured inefficiency
    (:func:`..joint_gsim_measurement.inefficiency`); together they rank and time the cases without a
    modeled estimate. ``cycles`` is what the timeout is derived from."""
    from merlin.development.backend.campaign.joint_gsim_measurement import predicted_cycles, structural_floor

    out = []
    for name in cohort:
        if l2.get(name) != "pass":
            continue
        found = _estimate(known.get(name), digest)
        if found is not None:
            cycles, performance = found
            floor, _ = structural_floor(performance.get("structural_roofline"))
            entry = {
                "modeled_cycles": cycles,
                "cycles": cycles,
                "basis": "Spike timing-model estimate",
                "headroom_cycles": cycles - floor if floor is not None and cycles >= floor else cycles,
            }
        else:
            performance = (records or {}).get(name) or {}
            floor, limiter = structural_floor(performance.get("structural_roofline"))
            predicted = predicted_cycles(model, floor, limiter)
            if predicted is None:
                continue
            entry = {
                "modeled_cycles": None,
                "structural_floor_cycles": round(floor, 1),
                "bound_by": limiter,
                "predicted_cycles": round(predicted["cycles"]),
                "cycles": predicted["upper_cycles"],
                "basis": predicted["basis"],
                "headroom_cycles": max(0.0, predicted["cycles"] - floor),
            }
        executable = performance.get("elf_sha256") if isinstance(performance.get("elf_sha256"), str) else None
        if executable is None and isinstance(performance.get(SOURCE_KEY), str):
            executable = "source:" + performance[SOURCE_KEY]  # no ELF named: the graded source stands in
        if executable is None or (name, executable) in measured or (name, executable) in attempted:
            continue
        measured_before = any(case == name for case, _ in measured)
        entry.update(
            capsule=name,
            executable=executable,
            headroom_cycles=round(entry["headroom_cycles"], 1),
            measured_before=measured_before,
            # No model estimate and never measured: the case has no timing evidence at all.
            timing_evidence=found is not None or measured_before,
            family=_family(name, performance, families),
            # The case's own declared work (useful MACs / logical bytes over the target's rates).
            work_cycles=floor if floor is not None else entry["cycles"],
            structural_roofline=performance.get("structural_roofline"),
            preferred=name in preferred,
        )
        out.append(entry)
    return order(out)


def _family(name, performance, families):
    """The case's family: a declared one when the caller supplies it, else its structural bound class."""
    from merlin.development.backend.campaign.joint_gsim_measurement import structural_floor

    if families and isinstance(families.get(name), str) and families[name]:
        return families[name]
    _, limiter = structural_floor(performance.get("structural_roofline"))
    return limiter or "unclassified"


def order(entries: list[dict]) -> list[dict]:
    """First evidence per family, smallest declared work first; then the headroom order.

    A family with no timing evidence gets its smallest such case measured before any long job, so
    every family has cycles within minutes. "Small" is relative: the case's declared work against the
    other untimed members of its family, never a constant. The rest keep the established order: never
    measured first, then largest headroom; a changed executable rotates in after them. A preferred entry
    (a bundle representative) goes ahead of the others in both groups."""
    firsts: dict[str, dict] = {}
    for entry in sorted(entries, key=lambda c: (c["work_cycles"], c["capsule"])):
        if not entry["timing_evidence"]:
            firsts.setdefault(entry["family"], entry)
    lead = sorted(firsts.values(), key=lambda c: (not c.get("preferred"), c["work_cycles"], c["capsule"]))
    chosen = {id(entry) for entry in lead}
    rest = sorted(
        (c for c in entries if id(c) not in chosen),
        key=lambda c: (not c.get("preferred"), c["measured_before"], -c["headroom_cycles"], c["capsule"]),
    )
    return lead + rest


def long_lane(entries: list[dict], expected: dict) -> set[str]:
    """Cases above the natural break in the pending cases' expected runtimes.

    The pending runtimes, sorted, split at their single largest ratio between neighbours; the cases
    above that break are long relative to the rest and run only in the long lane, so they never hold a
    normal slot the shorter cases need. Relative by construction: no break (a tie or one case) means no
    long case, and once the shorter cases are measured the remaining ones split again."""
    timed = sorted((expected[c["capsule"]], c["capsule"]) for c in entries if (expected.get(c["capsule"]) or 0) > 0)
    if len(timed) < 2:
        return set()
    ratios = [timed[i + 1][0] / timed[i][0] for i in range(len(timed) - 1)]
    largest = max(ratios)
    if largest <= 1 or ratios.count(largest) > 1:
        return set()
    cut = ratios.index(largest) + 1
    return {name for _, name in timed[cut:]}


def _spike_seconds(run_dir: Path) -> dict:
    """Each case's measured Spike-tier seconds in this run's runs roots (ordering history only)."""
    try:
        from merlin.development.backend.feedback.grading.loop_grade import _history_runs_roots
        from merlin.development.targetgen.grading.scheduling import measured_costs

        costs, _ = measured_costs(_history_runs_roots(run_dir), requested_tiers={"L2"}, engine="spike")
        return dict(costs)
    except Exception:  # noqa: BLE001 -- an unpriced case adds no Spike term; the margin still applies
        return {}


class GsimLane:
    """One per idle measurer; :meth:`tick` returns a status only when it submitted a request."""

    def __init__(self, workspace, run_dir):
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.last = None  # why the lane last stood aside (diagnostics)
        self._said: set[tuple] = set()  # (case, executable, reason) skips already logged

    def inflight(self, channel: Path, *, now: float | None = None) -> int:
        from merlin.development.backend.campaign.joint_gsim_measurement import REQUEST_PREFIX
        from merlin.development.backend.feedback.performance.idle_measurement import _deadline_s
        from merlin.development.backend.feedback.performance.idle_measurement import _load as load_request

        now = time.time() if now is None else now
        count = 0
        for request in Path(channel).glob(f"req_{REQUEST_PREFIX}*.json"):
            rid = request.name[len("req_") : -len(".json")]
            body = load_request(request)
            if (
                not (Path(channel) / f"resp_{rid}.json").exists()
                and isinstance(body, dict)
                and now <= _deadline_s(body, request)
            ):
                count += 1
        return count

    def tick(
        self,
        channel: Path,
        *,
        cohort,
        l2: dict,
        known: dict,
        digest,
        free=None,
        records=None,
        final_reserve=0,
        preferred=frozenset(),
    ) -> str | None:
        """One submission at most. ``free``: free shared slots beyond the grade reserve right now;
        ``final_reserve``: slots final grading keeps that ``free`` does not already exclude (a job still
        running in the final-model window starts only beyond them)."""
        try:
            return self._tick(Path(channel), cohort, l2, known, digest, free, records, final_reserve, preferred)
        except Exception as error:  # noqa: BLE001 -- the lane never reaches the grader
            from merlin.development.backend.feedback.grading import worker_resources as scratch_retention

            self.last = f"error: {type(error).__name__}"
            scratch_retention.say(f"[idle-gsim] skipped: {type(error).__name__}: {str(error)[:200]}")
            return None

    def _tick(
        self, channel: Path, cohort, l2, known, digest, free, records, final_reserve=0, preferred=frozenset()
    ) -> str | None:
        from merlin.development.backend.campaign import control as joint_control
        from merlin.development.backend.campaign import gsim_budget as E
        from merlin.development.backend.campaign import gsim_budget as W
        from merlin.development.backend.campaign import joint_gsim_measurement as M

        state = joint_control.selected()
        if state is None:
            self.last = "no joint campaign"
            return None
        knobs = M.settings(state.get("policy") or {})
        if not knobs["enabled"]:
            self.last = "disabled by policy"
            return None
        if free is not None and free < 1:
            self.last = "no free idle slot"
            return None
        if knobs["max_jobs"] and self.inflight(channel) >= knobs["max_jobs"]:
            self.last = "measurement jobs in flight"
            return None
        fit = M.throughput(channel, min_observations=knobs["min_observations"], quantile=knobs["timeout_quantile"])
        if fit is None:
            self.last = "no recorded gSIM throughput yet"
            return None
        model = M.inefficiency(channel, min_observations=knobs["min_observations"], quantile=knobs["timeout_quantile"])
        jobs = M.read().get("jobs", {})
        # An authorization that expired unclaimed never ran: its case stays eligible.
        attempted = {
            (j.get("capsule"), j.get("executable"))
            for j in jobs.values()
            if isinstance(j, dict) and j.get("status") != "expired"
        }
        retries = Counter()
        for job in jobs.values():
            if isinstance(job, dict) and job.get("outcome") in ("timeout", "lost", "cycle_cap"):
                retries[job["capsule"]] += 1
        ranked = candidates(
            cohort,
            l2,
            known,
            digest,
            measured=measured_executables(channel),
            attempted=attempted,
            records=records,
            model=model,
            preferred=preferred,
        )
        if not ranked:
            self.last = "every passing case's current executable is measured or attempted"
            return None
        spike = _spike_seconds(self.run_dir)
        for case in ranked:
            name = case["capsule"]
            if name not in spike and (name, "spike_timing_absent") not in self._said:
                from merlin.development.backend.feedback.grading import worker_resources as scratch_retention

                self._said.add((name, "spike_timing_absent"))
                scratch_retention.say(
                    f"[idle-gsim] {name}: Spike setup timing unavailable; scheduling uses the throughput model only"
                )
        try:
            remaining_drain, remaining_end = joint_control.remaining(drain=True), joint_control.remaining()
        except RuntimeError:
            self.last = "campaign clock not started"
            return None
        long_busy = sum(
            bool(j.get("long")) and j.get("status") in ("authorized", "running")
            for j in jobs.values()
            if isinstance(j, dict)
        )
        final_slots_free = free is None or free - max(0, int(final_reserve or 0)) >= 1
        expected = {
            c["capsule"]: M.expected_seconds(
                fit, c["modeled_cycles"] or c.get("predicted_cycles"), spike_seconds=spike.get(c["capsule"])
            )
            for c in ranked
        }
        long_cases = long_lane(ranked, expected)
        # What the selected engine allows, from the runs earlier measurement children recorded.
        rate, ceiling = E.engine_rate(jobs), E.engine_ceiling(jobs)
        from merlin.development.backend.campaign import gsim_budget as P

        wall_model = P.campaign_model(jobs, self.run_dir)
        for case in ranked:
            name = case["capsule"]
            wall_case = P.features(case["cycles"], case.get("structural_roofline"))
            decision = P.decide(wall_model, wall_case, cap_s=knobs["max_job_wall_s"])
            if not decision["run"]:
                self.last = decision["reason"]
                if (name, case["executable"], self.last) not in self._said:
                    from merlin.development.backend.feedback.grading import (
                        worker_resources as scratch_retention,
                    )

                    self._said.add((name, case["executable"], self.last))
                    scratch_retention.say(f"[idle-gsim] {name} not measured: {self.last}")
                continue
            plan, why = M.job_plan(
                fit,
                case["cycles"],
                case["modeled_cycles"] or case.get("predicted_cycles"),
                spike_seconds=spike.get(name),
                retries=retries[name],
                remaining_drain_s=remaining_drain,
                remaining_end_s=remaining_end,
                final_window_s=knobs["final_window_s"],
                long_allowed=long_busy < knobs["long_max_jobs"],
                final_slots_free=final_slots_free,
                long_lane=name in long_cases,
                engine_ceiling_s=ceiling,
                max_job_wall_s=knobs["max_job_wall_s"],
            )
            if plan is None:
                self.last = why
                if why.startswith((M.CEILING_SKIP, W.SKIP)) and (name, case["executable"], why) not in self._said:
                    from merlin.development.backend.feedback.grading import (
                        worker_resources as scratch_retention,
                    )

                    self._said.add((name, case["executable"], why))
                    scratch_retention.say(f"[idle-gsim] {name} not measured: {why}")
                continue
            # The wall prediction (work and this campaign's measured jobs) replaces the kernel-cycle
            # throughput timeout, which under-ran data-heavy cases; the window checks still apply.
            timeout = decision["timeout_s"] or plan["timeout_s"]
            bound = E.engine_cycle_bound(timeout, rate, margin=fit["margin"])
            now_ns = time.time_ns()
            identity = json.dumps([name, case["executable"], digest, now_ns])
            rid = M.REQUEST_PREFIX + f"{now_ns}_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            reason = M.authorize(
                rid,
                name,
                timeout_s=timeout,
                executable=case["executable"],
                modeled_cycles=case["modeled_cycles"],
                free_slots=free,
                long=plan["long"],
                engine_bound=bound,
                throughput=fit,
                wall={"features": wall_case, "predicted_s": decision["predicted_s"]},
            )
            if reason is None:
                from merlin.development.backend.feedback.grading import worker_resources as scratch_retention
                from merlin.development.backend.feedback.performance.idle_measurement import submit

                detail = {
                    key: case[key]
                    for key in (
                        "modeled_cycles",
                        "predicted_cycles",
                        "structural_floor_cycles",
                        "bound_by",
                        "headroom_cycles",
                        "basis",
                    )
                    if key in case
                }
                detail["spike_seconds"] = spike.get(name)
                submit(
                    channel,
                    [name],
                    digest,
                    origin=M.ORIGIN,
                    prefix=M.REQUEST_PREFIX,
                    timeout=timeout,
                    extra={
                        "sim": "gsim",
                        "measurement": {
                            **detail,
                            "long": plan["long"],
                            "expected_s": plan["expected_s"],
                            "throughput": fit,
                            "inefficiency": model,
                            "engine_bound": bound,
                            "wall_prediction": decision,
                        },
                    },
                    rid=rid,
                )
                self.last = "submitted"
                scratch_retention.say(
                    f"[idle-gsim] background gSIM measurement of {name} "
                    f"(timeout {timeout}s, {round(case['cycles'])} cycles: {case['basis']}; "
                    f"engine cycle bound {bound['max_cycles'] or 'selected'}: {bound['basis']}; "
                    f"predicted wall {decision['predicted_s']}s)"
                )
                return "submitted gSIM measurement"
            self.last = reason
            if reason in _STOP:
                return None
        return None


# ---- scalability_gsim --------------------------------------------------------------------------
# One operator-authorized gSIM measurement of a passing public scalability case (advisory only).
#
# Reached only through ``agent_selfcheck.py --suite scalability --sim gsim`` in an evaluation child the
# self-check broker started for a request id the idle measurer authorized in the run-side ledger
# (:mod:`..joint_gsim_measurement`); every other scalability gSIM request is refused exactly as
# :func:`.public_scalability.request_policy` refuses it. The case is graded by the ordinary grader with the
# self-check's own gSIM adapter set (Spike functional tier, gSIM elaborated-RTL tier) on a private copy of
# the frozen case, and the Spike timing model is attached to the same executable, so one response row
# holds both the measured gSIM cycles and the modeled estimate of the same program (a calibration pair).
#
# The response is the ordinary gSIM row shape (``sim: gsim``, ``barrier_tier: L3``, ``barrier_cycles``,
# ``execution_digest``) that :mod:`.performance_headroom`, :mod:`.perf_history`, the timing-model
# calibration and the console already read. It is advisory performance evidence: no certificate, no
# physical obligation, no promotion, no scalability-suite credit.

SCHEMA = "merlin.phase1.scalability_gsim_measurement.v1"


def _row(name, result, path, *, sim_tier):
    """The public gSIM row of one measured case (outcomes and cycles, never values or private paths)."""
    from merlin.development.backend.feedback.grading.qa import _execution_digest_from_result
    from merlin.development.backend.feedback.scalability.public_scalability import _observed_result
    from merlin.development.backend.session.inputs.oracle_timing import barrier_measurements
    from merlin.development.targetgen.corpus.capsule_io import tier_status

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
    from merlin.development.backend.feedback.campaign.timing import attach as attach_performance
    from merlin.development.backend.feedback.campaign.timing import observation_budget
    from merlin.development.backend.feedback.scalability.public_scalability import _readback, _sim_via
    from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import _adapters
    from merlin.development.targetgen.corpus.capsule_io import load_capsule
    from merlin.development.targetgen.grading import model_checks as CG
    from merlin.development.targetgen.grading import submission as _merlin_submission_1
    from merlin.experiments.corpus import public_scalability as S
    from merlin.experiments.corpus.source_staging import copy_input

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
        generated = output / "runs" / "runs" / capsule_eligibility.suite_for(context.target) / str(name) / "generated"
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
        _merlin_submission_1.grade(
            submission,
            capsules_root=cases,
            runs_root=output / "runs",
            target=context.target,
            labels={"public", "dev"},
            contract=contract or context.repo / "src/merlin/resources/contracts",
            oracle_adapters=adapters,
            timeout=timeout,
            max_workers=max_workers,
            on_capsule_complete=completed,
        )
    suite = output / "runs" / "runs" / capsule_eligibility.suite_for(context.target)
    return [
        _row(name, results[name], suite / name / "capsule_result.json", sim_tier=sim_tier)
        for name in names
        if name in results
    ], engine


def _engine_ceiling(context) -> int | None:
    """The selected gSIM process ceiling in seconds, as the backend states it; ``None`` when unknown."""
    try:
        from merlin.runtime.backends import registry as base

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
    from merlin.development.backend.campaign import gsim_budget as E
    from merlin.development.backend.campaign import joint_gsim_measurement as M

    rid = M.requested()
    if rid is None:
        raise ValueError("public scalability accepts only Spike functional requests")
    if not args.out:
        raise ValueError("public scalability requires a result output path")
    from merlin.development.backend.feedback.execution.profile_runtime import selfcheck_execution
    from merlin.development.backend.feedback.grading import worker_resources as scratch_retention
    from merlin.development.backend.feedback.scalability import public_scalability as P

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
        with (
            selfcheck_execution(
                args.compiler_bundle_manifest,
                context=context,
                submission=Path(args.submission),
                work_root=Path(args.out).with_suffix(".profile"),
            ),
            E.engine_cycle_budget(bound["max_cycles"]),
        ):
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
    from merlin.development.backend.feedback.performance.brief import attach as attach_brief

    attach_brief(public, rows, debuggable=False)
    text = json.dumps(public, indent=2)
    Path(args.out).write_text(text)
    print(text)
    return 0 if public["all_pass"] else 1


__all__ = [
    "GsimLane",
    "SCHEMA",
    "candidates",
    "cli",
    "measure",
    "measured_executables",
    "verdict_performance",
]
