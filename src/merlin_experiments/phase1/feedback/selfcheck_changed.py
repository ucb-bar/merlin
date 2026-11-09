"""``--capsules changed``: simulate only the public capsules an edit changed.

An author who cannot tell which capsules an edit affects either re-checks a few by hand or waits on
many: measured on one 300-minute run, 224 minutes were spent blocked on self-checks, most of them one
to three capsules at a time. The grade already decides "affected" exactly, at the one place where it
is expensive: an elaborated-RTL tier whose executable, request and engine selection are byte-identical
to a certified one carries that exact-byte simulator certificate instead of simulating
(:mod:`merlin.targetgen.simulation_certificate`). This selector asks that same question first.

It runs in two passes over one isolated copy of the submission, so both see the same bytes:

1. **Probe.** Every public capsule is graded on the grade's own adapters, in parallel over the shared
   CPU pool, with each elaborated-RTL engine refused just before dispatch
   (:func:`merlin.targetgen.simulation_certificate.probe_observer`). Emission, the ELF builds and the
   functional screen run as they always do -- no certificate carries the screen -- and the RTL tier is
   decided by the certificate lookup alone: a hit carries the certified verdict, a miss is refused
   unexecuted. Nothing is simulated at an RTL tier.
2. **Simulate.** The ``changed`` capsules then take the ordinary certifying ladder as a named
   request, bounded by ``MERLIN_SELFCHECK_MAX_CAPSULES`` and ordered longest first.

Categories, per capsule:

* ``unchanged-certified`` -- the probe produced a complete certified verdict: every mandatory tier
  passed, with each RTL verdict carried from a certificate for these exact bytes. Its row is the verdict
  an ordinary self-check reaches on the same bytes; no simulation was spent on it.
* ``unchanged-failing`` -- every mandatory RTL tier refused by the probe already FAILED for these exact
  bytes under the same key (a noted, never carried verdict: ``tier_cache.noted_failure``), and every
  other mandatory tier passed. It is not re-simulated and it is not a pass.
* ``failing`` -- the probe's own verdict already fails with no RTL simulation left to run: the build,
  emission or screen failed, or every RTL verdict was carried and the capsule still is not certified
  (a structural requirement). Simulating cannot change that verdict, so it is reported from the probe,
  not re-run; naming the capsule shows its full diagnostics.
* ``changed`` -- anything else: an RTL tier with no certificate and no noted failure for these bytes,
  or an unreadable or missing probe record. Only these are simulated. A provided-host capsule (the
  platform's own host image, ``owned_host_leaf_execution``) executes its RTL tier on a route with no
  exact-byte certificate, so it is not probed -- probing it would simulate it -- and is always
  ``changed``.

Pure selection. A capsule is certified only by an existing exact-byte certificate or by a fresh
certifying run in this request, so ``all_pass`` keeps its meaning: every public capsule certified. Only
public capsules (labels ``public``/``dev`` under the explicit public root) are probed, and only the
certificate store's own records for those names are read.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

SCHEMA = "merlin.selfcheck_changed_selection.v1"
UNCHANGED_CERTIFIED = "unchanged-certified"
UNCHANGED_FAILING = "unchanged-failing"
CHANGED = "changed"
FAILING = "failing"
CATEGORIES = (UNCHANGED_CERTIFIED, UNCHANGED_FAILING, FAILING, CHANGED)
LABELS = frozenset({"public", "dev"})
PROVIDED_HOST = "provided-host capsule: its RTL tier runs on a route with no exact-byte certificate"


def probe_engines(adapters: dict, target: str) -> tuple[frozenset[str], str]:
    """``(engines, refusal)``: the engines answering this selection's elaborated-RTL tiers.

    The probe refuses exactly these, so each must be a native simulator adapter (the only route that
    passes through the probe's pre-dispatch refusal). A tier answered any other way could not be held
    back without running it, and an engine the screen also uses would refuse the screen; both refuse
    the selector instead of simulating or guessing.
    """
    from merlin.targetgen.runner_config import _config_for_target

    rtl = _config_for_target(target, None, "").rtl_tiers  # The ladder's own resolution, fallback included.
    engines, screens = set(), set()
    for tier, adapter in sorted(adapters.items()):
        engine = getattr(adapter, "_merlin_simulator_engine", None)
        if tier not in rtl:
            screens.add(engine)
            continue
        if not isinstance(engine, str) or not engine:
            return frozenset(), f"tier {tier} is not answered by a native simulator adapter, so it cannot be probed"
        engines.add(engine)
    if not engines:
        return frozenset(), "this selection has no elaborated-RTL tier, so no certificate can be reused"
    if engines & screens:
        return frozenset(), f"engine(s) {sorted(engines & screens)} answer both the screen and an RTL tier"
    return frozenset(engines), ""


@contextlib.contextmanager
def probing(engines):
    """Refuse dispatch of ``engines`` for this process and every grader child it launches."""
    from merlin.targetgen.simulation_certificate import PROBE_ENV

    os.environ[PROBE_ENV] = ",".join(sorted(engines))
    try:
        yield
    finally:
        # Removed, not restored: the simulate pass and any later work must never inherit a probe.
        os.environ.pop(PROBE_ENV, None)


def classify(result, capsule, *, adapters: dict, barrier_tier: str, target: str) -> tuple[str, dict]:
    """``(category, evidence)`` for one probe result. Anything not provably unchanged is ``changed``."""
    from merlin.targetgen.simulation_certificate import PROBE_REASONS

    from . import qa as _qc
    from .selfcheck_results import _operator_feedback_state
    from .selfcheck_selection import capsule_barrier

    if not isinstance(result, dict) or not isinstance(result.get("tiers"), dict):
        return CHANGED, {"why": "no readable probe result"}
    if result.get("kind") == "model" or _qc._candidate_native_feedback(result) is not None:
        return CHANGED, {"why": "whole-program results are not classified by certificate"}
    tiers = {tier: record for tier, record in result["tiers"].items() if isinstance(record, dict)}
    statuses = {tier: record.get("status") for tier, record in tiers.items()}
    refused = {
        tier: PROBE_REASONS[record["reason"]]
        for tier, record in tiers.items()
        if record.get("mandatory") and record.get("status") == "unavailable" and record.get("reason") in PROBE_REASONS
    }
    others = [record.get("status") for tier, record in tiers.items() if record.get("mandatory") and tier not in refused]
    if refused and "fail" in others:
        return FAILING, _failure(result, statuses)  # A mandatory tier that ran already refutes it.
    if refused:
        if all(status == "pass" for status in others) and set(refused.values()) == {"failing"}:
            return UNCHANGED_FAILING, {"tiers": statuses, "failed_tiers": sorted(refused)}
        return CHANGED, {"tiers": statuses, "why": "no simulator certificate for these exact bytes"}
    bar_used = capsule_barrier(tiers, adapters, barrier_tier)
    bar = statuses.get(bar_used)
    certified, _, _, structural = _operator_feedback_state(
        result, capsule, target=target, barrier_status=None if bar == "unavailable" else bar
    )
    if not certified:
        return FAILING, {**_failure(result, statuses), "structural_qualification": structural}
    reused = sorted(
        tier
        for tier, record in tiers.items()
        if isinstance(record.get("carried"), dict) and record["carried"].get("carried")
    )
    return UNCHANGED_CERTIFIED, {"tiers": statuses, "reused_tiers": reused, "barrier_tier": bar_used}


def _failure(result: dict, statuses: dict) -> dict:
    """The probe verdict's own failure, golden-free: plane, category, tier and redacted detail."""
    from . import qa as _qc

    failure = result.get("failure") if isinstance(result.get("failure"), dict) else {}
    found = {"tiers": statuses, "status": result.get("status")}
    if failure:
        found["failure"] = {
            "plane": failure.get("plane"),
            "category": failure.get("category"),
            "tier": failure.get("tier"),
            "detail": _qc._redact_detail(failure.get("detail")),
        }
    if result.get("declined"):
        found["declined"] = result["declined"]
    return found


def longest_first(names, capsules: dict) -> list[str]:
    """``names`` in the grade's own longest-first order: measured cost, then cycles, then operand volume."""
    from merlin.targetgen import capsule_scheduling as S

    members = [capsules[name] for name in names]
    estimate = S.cost_estimates(members, *S.measured_costs(S.selected_history()))
    return sorted(names, key=lambda name: (-estimate[name], -S.operand_volume(capsules[name]), name))


def _probe_progress(path: Path, judge, names, *, started_ns: int, stop: threading.Event) -> None:
    """Publish the probe's classification as it lands, in the ordinary progress document's shape.

    ``judge(name)`` is the category of a capsule whose probe result is on disk, else ``None``. A landed
    result is final, so each capsule is judged once.
    """
    from .selfcheck import _atomic_json

    judged: dict[str, str] = {}

    def snapshot(status):
        for name in names:
            if name not in judged and (category := judge(name)) is not None:
                judged[name] = category
        rows = [{"capsule": name, "status": judged[name]} for name in names if name in judged]
        counts = {category: sum(row["status"] == category for row in rows) for category in CATEGORIES}
        return {
            "status": status,
            "phase": "changed_probe",
            "started_at_unix_ns": started_ns,
            "updated_at_unix_ns": time.time_ns(),
            "n_expected": len(names),
            "n_finished": len(rows),
            "n_remaining": len(names) - len(rows),
            "counts": {category: n for category, n in counts.items() if n},
            "per_capsule": rows,
            "note": (
                "Certificate probe of --capsules changed: each public capsule is built and screened and its "
                "RTL tier looked up by exact-byte certificate, without simulating it. Not a verdict; the "
                "changed capsules are simulated next."
            ),
        }

    last = None
    while not stop.wait(0.5):
        doc = snapshot("running")
        if doc["per_capsule"] != last:
            _atomic_json(path, doc)
            last = doc["per_capsule"]
    _atomic_json(path, snapshot("probe_complete"))


class ChangedSelection:
    """The probe's classification, and how it joins the ordinary verdict of the simulated subset."""

    def __init__(self, *, evidence, simulated, deferred, bound, not_covered, n_public, probe_s, n_probed):
        self.evidence = evidence  # name -> (category, evidence)
        self.simulated = list(simulated)
        self.deferred = list(deferred)
        self.bound = bound
        self.not_covered = list(not_covered)
        self.n_public = n_public
        self.probe_s = probe_s
        self.n_probed = n_probed

    def names(self, category):
        return sorted(name for name, (found, _) in self.evidence.items() if found == category)

    def _rows(self) -> list[dict]:
        rows = []
        for name, (category, found) in sorted(self.evidence.items()):
            if category == UNCHANGED_CERTIFIED:
                rows.append(
                    {
                        "capsule": name,
                        "pass": True,
                        "certified": True,
                        "selection": category,
                        "simulated": False,
                        **found,
                    }
                )
            elif category == UNCHANGED_FAILING:
                rows.append(
                    {
                        "capsule": name,
                        "pass": False,
                        "certified": False,
                        "selection": category,
                        "simulated": False,
                        **found,
                        "note": "not re-simulated: these exact bytes already failed "
                        + ", ".join(found["failed_tiers"])
                        + " under this engine selection. Name the capsule to see the failure again.",
                    }
                )
            elif category == FAILING:
                rows.append(
                    {
                        "capsule": name,
                        "pass": False,
                        "certified": False,
                        "selection": category,
                        "simulated": False,
                        **found,
                        "note": "fails on these bytes with no RTL simulation left to run (build, emission, screen "
                        "or a structural requirement); not re-run. Name the capsule for full diagnostics.",
                    }
                )
            elif name in self.deferred:
                rows.append(
                    {
                        "capsule": name,
                        "pass": False,
                        "certified": False,
                        "selection": category,
                        "simulated": False,
                        "note": "changed, but over this request's capsule bound: not simulated and not a pass. "
                        "Run --capsules changed again to continue.",
                    }
                )
        for name in self.not_covered:
            rows.append(
                {
                    "capsule": name,
                    "pass": False,
                    "certified": False,
                    "selection": "not_covered",
                    "simulated": False,
                    "note": "whole-model capsules are not selected by 'changed'; name them",
                }
            )
        return rows

    def report(self, out: dict) -> dict:
        """The verdict over EVERY public capsule: the simulated subset's own rows plus the probe's."""
        simulated = []
        for row in out.get("per_capsule") or []:
            if isinstance(row, dict):
                simulated.append({**row, "selection": CHANGED, "simulated": True})
        rows = simulated + self._rows()
        certified = sum(row.get("certified") is True for row in rows)
        complete = not out.get("error") and len({row.get("capsule") for row in rows}) == self.n_public
        all_pass = bool(complete and self.n_public and certified == self.n_public)
        counts = {category: len(self.names(category)) for category in CATEGORIES}
        unchecked = len(self.deferred) + len(self.not_covered)
        merged = dict(out)
        merged.update(
            selector="changed",
            scope="changed",
            suite_size=self.n_public,
            per_capsule=rows,
            n_capsules=len(rows),
            n_result_rows=len(rows) - unchecked,
            n_passed=sum(row.get("pass") is True for row in rows),
            n_certified=certified,
            n_unchecked=unchecked,
            n_unknown=max(0, self.n_public - len(rows)) + unchecked,
            n_simulated=len(simulated),
            n_reused=counts[UNCHANGED_CERTIFIED],
            all_pass=all_pass,
            certified_complete=all_pass,
            changed_selection={
                "schema": SCHEMA,
                "n_public": self.n_public,
                "counts": counts,
                "unchanged_certified": self.names(UNCHANGED_CERTIFIED),
                "unchanged_failing": self.names(UNCHANGED_FAILING),
                "failing": self.names(FAILING),
                "changed": self.names(CHANGED),
                "simulated": sorted(self.simulated),
                "not_simulated": sorted(self.deferred),
                "n_not_simulated": len(self.deferred),
                "bound": self.bound,
                "not_covered": sorted(self.not_covered),
                "n_probed": self.n_probed,
                "provided_host": sorted(
                    n for n, (_, found) in self.evidence.items() if found.get("why") == PROVIDED_HOST
                ),
                "probe_s": self.probe_s,
            },
            note=self.note(counts),
        )
        if out.get("note"):
            merged["simulated_note"] = out["note"]
        return merged

    def note(self, counts) -> str:
        bound = f" of {self.bound}" if self.bound is not None else ""
        return (
            f"CHANGED: {self.n_probed} of {self.n_public} public capsule(s) were built and screened and their RTL "
            f"tier looked up by exact-byte simulator certificate, without simulating it ({self.probe_s:.0f}s)"
            + (f"; {len(self.not_covered)} whole-model capsule(s) are not covered" if self.not_covered else "")
            + ". "
            f"{counts[UNCHANGED_CERTIFIED]} unchanged-certified (a certificate for these exact bytes and this "
            f"engine selection was reused; no simulation). {counts[UNCHANGED_FAILING]} unchanged-failing (these "
            f"exact bytes already failed an RTL tier; not re-simulated and not passes; name them to see why). "
            f"{counts[FAILING]} failing with no RTL simulation left to run (build, emission, screen, or a "
            f"structural requirement on carried RTL verdicts; see each row). "
            f"{counts[CHANGED]} changed: {len(self.simulated)} simulated now on the grade's own ladder"
            + (
                f", {len(self.deferred)} NOT simulated because they exceed this request's bound{bound} (run "
                "--capsules changed again to continue; they are not passes)"
                if self.deferred
                else ""
            )
            + ". `all_pass` and `certified_complete` cover every public capsule: each is certified only by a "
            "reused exact-byte certificate or by this request's own run."
        )


def _answer(a, out: dict) -> int:
    """Publish a verdict the selector reached without the ordinary simulate pass."""
    from .selfcheck import _atomic_json
    from .selfcheck_results import _log_telemetry

    print(json.dumps(out, indent=2))
    if a.out:
        _atomic_json(Path(a.out), out)
    _log_telemetry(out, a.capsules)
    return 0 if out.get("all_pass") else (2 if out.get("error") else 1)


def select_changed(a, *, sub, context, capsules_root, contract, adapters, combined, target, barrier_tier):
    """The ``changed`` capsules of this request; the probe's answer when nothing needs simulating."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import simulation_certificate as SC
    from merlin.targetgen.owned_host_leaf_execution import selected_host_leaf

    from ..brokers.selfcheck import _ordinary_capsule_limit
    from .compiler_profile_runtime import selfcheck_execution
    from .selfcheck_selection import tier_engines
    from .selfcheck_selectors import Selection

    def refuse(why):
        return Selection(
            None,
            answered=_answer(
                a, {"error": "--capsules changed: " + why, "all_pass": False, "sim": a.sim, "selector": "changed"}
            ),
        )

    if not combined:
        return refuse("it classifies against the grade's own tier selection; drop --sim/--tiers")
    if os.environ.get(SC.ENV) != "1" or SC.cache_root() is None:
        return refuse(
            f"exact-byte simulator certificates are disabled here ({SC.ENV}), so nothing can be "
            "shown unchanged; name the capsules to check instead"
        )
    engines, why = probe_engines(adapters, target)
    if why:
        return refuse(why)
    contract_root = contract if contract is not None else context.repo / "merlin/contract"
    capsules = {
        cap["name"]: cap for cap in CR.discover_capsules(capsules_root, labels=set(LABELS), contract=contract_root)
    }
    try:
        bound = _ordinary_capsule_limit(len(capsules))  # `auto` covers the public corpus; a number is exact
    except ValueError as error:
        return refuse(str(error))
    not_covered = sorted(name for name, cap in capsules.items() if cap.get("kind") == "model")
    provided = sorted(name for name, cap in capsules.items() if name not in not_covered and selected_host_leaf(cap))
    names = sorted(name for name in capsules if name not in not_covered and name not in provided)
    work = Path(tempfile.mkdtemp(prefix="changed-probe-", dir=Path(sub).parent))
    probe_root = Path(capsules_root)
    if not_covered or provided:
        probe_root = work / "capsules"
        for name in names:  # A corpus root is its set of capsules: a subset is a root of its own.
            shutil.copytree(capsules[name]["__dir__"], probe_root / name)
    runs = work / "runs"
    results = runs / "runs" / CR.suite_for(target)

    def judge(name):
        try:
            result = json.loads((results / name / "capsule_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return classify(result, capsules[name], adapters=adapters, barrier_tier=barrier_tier, target=target)[0]

    started, stop, publisher = time.monotonic(), threading.Event(), None
    if a.progress_out:
        publisher = threading.Thread(
            target=_probe_progress,
            args=(Path(a.progress_out), judge, names),
            kwargs={"started_ns": time.time_ns(), "stop": stop},
            daemon=True,
            name="selfcheck-changed-probe",
        )
        publisher.start()
    try:
        with (
            probing(engines),
            selfcheck_execution(
                a.compiler_bundle_manifest, context=context, submission=sub, work_root=work / "profile"
            ),
        ):
            CG.grade(
                str(sub),
                capsules_root=str(probe_root),
                runs_root=str(runs),
                labels=set(LABELS),
                contract=str(contract_root),
                oracle_adapters=adapters,
                timeout=a.timeout,
                max_workers=a.workers,
                target=target,
            )
    except Exception as error:  # noqa: BLE001 -- a failed probe classifies nothing as unchanged
        print(f"[selfcheck] certificate probe failed: {type(error).__name__}: {str(error)[:300]}", flush=True)
    finally:
        stop.set()
        if publisher is not None:
            publisher.join(timeout=5)
    probe_s = round(time.monotonic() - started, 1)
    evidence = {name: (CHANGED, {"why": PROVIDED_HOST}) for name in provided}
    for name in names:
        try:
            result = json.loads((results / name / "capsule_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None  # missing or unreadable: changed, never certified
        evidence[name] = classify(result, capsules[name], adapters=adapters, barrier_tier=barrier_tier, target=target)
    shutil.rmtree(work, ignore_errors=True)
    ordered = longest_first([name for name, (category, _) in evidence.items() if category == CHANGED], capsules)
    simulated = ordered if bound is None else ordered[:bound]
    selection = ChangedSelection(
        evidence=evidence,
        simulated=simulated,
        deferred=ordered[len(simulated) :],
        bound=bound,
        not_covered=not_covered,
        n_public=len(capsules),
        probe_s=probe_s,
        n_probed=len(names),
    )
    if simulated:
        return Selection(set(simulated), changed=selection)
    out = selection.report(
        {
            "sim": a.sim,
            "barrier_tier": barrier_tier,
            "oracle_tiers": sorted(adapters),
            "tier_engines": tier_engines(adapters),
            "per_capsule": [],
        }
    )
    return Selection(set(), answered=_answer(a, out))
