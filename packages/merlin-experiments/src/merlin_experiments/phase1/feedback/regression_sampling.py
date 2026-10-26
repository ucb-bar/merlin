"""Grade stable passing functional cases through a rotating covering sample instead of every grade.

Functional cases that pass and keep passing consumed most grading executions: on one joint run 1915
of 3110 executions re-ran the original correctness cohort, and after a case had passed twice in a row
no case of that run ever regressed. Every grade's bounded admission window was spent re-proving
those cases while failing and large cases waited. Here, inside the rolling correctness curriculum
(:mod:`.joint_curriculum`), a case is *carried* -- not re-run this grade -- when all of these hold:

* it belongs to a sampled cohort (policy ``cohorts``; default the original correctness cohort);
* its last :data:`stable grades <DEFAULTS>` completed results passed in a row (``stable_grades``; by
  default derived from this run: one more than the longest passing streak any regression in this run
  ended, never fewer than ``min_stable_grades``);
* its dependency closure is unchanged since that streak: the case's own source bytes, the sources of
  the selected cases it ``extends``, and the frozen grading context except the compiler under test
  (descriptor, runtime configuration, policy, performance selection);
* its newest completed result in this run's runs roots is a pass (a failing or missing result is
  never carried), it was actually graded within the last full-sweep period, and it awaits no
  certification decision (its certificate ledger entry, when it has one, holds a passing L3 record);
* it is not one of this grade's hinted cases (the cases the author named in
  ``submission/joint-test-scope.json``, else the representative cases of the declared families).

Of the carried-eligible cases each grade still runs a covering sample: the fewest cases whose declared
axes (:func:`merlin.targetgen.tier_policy.capsule_axes`) span every axis value the eligible cases
declare, ties broken toward the case graded least recently, so the sample rotates through the cohort.
The compiler under test is excluded from the closure on purpose: the sample is the regression detector
for compiler edits, and a regression it finds makes the next grade a full sweep. A full sweep -- no case
carried -- also runs every ``full_sweep_every`` grades (default derived: eligible cases / sample size, so
a sweep comes as often as the rotation could visit every case once), during the campaign's final window
(``final_window_s``, default the policy's ``drain_s``), and in every milestone or final grade, which never
uses the curriculum at all.

A carried case keeps its newest completed result, attributed to the grade and source that produced it;
the grade lists it as ``carried (sampled)`` (its row status is ``carried``), it stays pending for the
current source, and it never counts as a pass of this grade, so ``all_pass``, promotion and
certification are unchanged. Disable with ``regression_sampling: {"enabled": false}``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SECTION = "regression_sampling"
#: Policy knobs (``regression_sampling`` in the campaign policy) and their defaults; ``None`` is derived.
DEFAULTS = {
    "enabled": True,
    # Performance cohorts (joint_test_selection lanes) whose stable cases may be carried.
    "cohorts": ["correctness_regression"],
    # Consecutive passing grades before a case may be carried; None derives it from this run's regressions.
    "stable_grades": None,
    # Floor of the derived stable_grades.
    "min_stable_grades": 2,
    # Grades between two full sweeps; None derives eligible cases / covering-sample size.
    "full_sweep_every": None,
    # A regression observed in a grade makes the next grade a full sweep.
    "sweep_after_regression": True,
    # Seconds before the campaign's drain boundary in which every grade is a full sweep; None = drain_s.
    "final_window_s": None,
}
LANES = ("correctness_regression", "calibration_diagnostics", "other")
STATUS = "carried"
REASON = "sampled"
LABEL = "carried (sampled)"
#: Grade decisions kept in the curriculum ledger (newest last).
_KEEP_DECISIONS = 64


def validate(section) -> None:
    if section is None:
        return
    if not isinstance(section, dict) or set(section) - set(DEFAULTS):
        raise ValueError("regression sampling policy must be an object of known knobs")
    for key in ("enabled", "sweep_after_regression"):
        if key in section and type(section[key]) is not bool:
            raise ValueError(f"invalid regression sampling {key}")
    cohorts = section.get("cohorts", DEFAULTS["cohorts"])
    if not isinstance(cohorts, list) or not cohorts or any(c not in LANES for c in cohorts):
        raise ValueError("invalid regression sampling cohorts")
    for key, low, high in (
        ("stable_grades", 1, 10000),
        ("min_stable_grades", 1, 10000),
        ("full_sweep_every", 1, 10000),
        ("final_window_s", 0, 86400),
    ):
        value = section.get(key)
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError(f"invalid regression sampling {key}")


def settings(policy: dict) -> dict:
    """The effective knobs; the run-time derivations are reported per grade in the curriculum report."""
    section = {**DEFAULTS, **((policy or {}).get(SECTION) or {})}
    final = section["final_window_s"]
    if final is None:
        final = (policy or {}).get("drain_s") if type((policy or {}).get("drain_s")) is int else 0
    return {
        "enabled": bool(section["enabled"]),
        "cohorts": list(section["cohorts"]),
        "stable_grades": section["stable_grades"],
        "min_stable_grades": section["min_stable_grades"],
        "full_sweep_every": section["full_sweep_every"],
        "sweep_after_regression": bool(section["sweep_after_regression"]),
        "final_window_s": final,
        "basis": {
            "stable_grades": "policy"
            if section["stable_grades"]
            else "max(min_stable_grades, 1 + longest passing streak a regression in this run ended)",
            "full_sweep_every": "policy"
            if section["full_sweep_every"]
            else "ceil(carried-eligible cases / covering-sample size)",
            "final_window_s": "policy" if section["final_window_s"] is not None else "policy drain_s",
        },
        "certification_authority": False,
    }


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def closures(source_digests: dict, docs: dict, context: dict) -> dict:
    """``{case: digest}`` of each case's dependency closure (see the module docstring)."""
    from merlin.targetgen.capsule_scheduling import dependencies

    names = sorted(source_digests)
    try:
        deps = dependencies([{**(docs.get(n) or {}), "name": n} for n in names])
    except ValueError:
        deps = {n: set() for n in names}
    context = {k: v for k, v in (context or {}).items() if k != "compiler_sha256"}

    def ancestry(name, seen):
        for parent in sorted(deps.get(name, ())):
            if parent not in seen:
                seen.add(parent)
                ancestry(parent, seen)
        return seen

    return {
        n: _digest(
            {
                "context": context,
                "source": source_digests[n],
                "extends": {p: source_digests.get(p) for p in sorted(ancestry(n, set()))},
            }
        )
        for n in names
    }


def load_docs(paths: dict) -> dict:
    """``{case: capsule.yaml document}``; an unreadable declaration declares no axes."""
    import yaml

    docs = {}
    for name, path in paths.items():
        try:
            document = yaml.safe_load((Path(path) / "capsule.yaml").read_text())
        except (OSError, ValueError, yaml.YAMLError):
            document = None
        docs[name] = document if isinstance(document, dict) else {}
    return docs


def lanes(names, policy: dict) -> dict:
    """``{case: lane}`` by the same operator rosters :func:`.joint_test_scope.select` uses."""
    original = set((policy.get("correctness_curriculum") or {}).get("original_capsules") or ())
    witnesses = set((policy.get("gsim") or {}).get("witness_capsules") or ())
    return {
        n: "correctness_regression" if n in original else "calibration_diagnostics" if n in witnesses else "other"
        for n in names
    }


def _cert_settled(entry) -> bool:
    """No certification decision is waiting: no ledger entry, or a passing newest L3 record."""
    if not isinstance(entry, dict) or "L3" not in entry:
        return True
    record = entry.get("L3")
    return (record.get("status") if isinstance(record, dict) else record) in ("pass", "joined")


def rotating_cover(docs: dict, eligible, recency: dict) -> list[str]:
    """Greedy cover of the eligible cases' axis values, ties toward the least recently graded case."""
    from merlin.targetgen.tier_policy import capsule_axes

    axes = {n: capsule_axes(docs.get(n) or {}) for n in sorted(eligible)}
    uncovered = set().union(*axes.values()) if axes else set()
    order = sorted(axes, key=lambda n: (-recency.get(n, 0), n))
    chosen: list[str] = []
    while uncovered:
        best = max(order, key=lambda n: len(axes[n] & uncovered))  # first maximum: least recent wins ties
        gain = axes[best] & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
    if not chosen and order:
        chosen.append(order[0])  # cases declaring no axes still rotate one at a time
    return sorted(chosen)


def _state(ledger: dict) -> dict:
    state = ledger.setdefault("sampling", {})
    for key, default in (
        ("grade", 0),
        ("last_full_sweep_grade", 0),
        ("regression_grade", None),
        ("longest_streak_before_regression", 0),
        ("decisions", []),
    ):
        state.setdefault(key, default)
    return state


def _member(ledger: dict, name: str) -> dict:
    entry = ledger["members"].setdefault(name, {"attempts": 0, "passes": {}})
    return entry.setdefault(
        "sampling", {"streak": 0, "last_outcome": None, "closure": None, "last_completed_grade": None}
    )


def decide(
    ledger: dict,
    *,
    names,
    policy: dict,
    closures_now: dict,
    docs: dict,
    tier_state: dict | None,
    outcomes: dict | None,
    remaining_s,
    hint=(),
) -> tuple[dict, dict]:
    """Advance the grade counter and choose this grade's carried cases (``{case: info}``, report)."""
    knobs = settings(policy)
    state = _state(ledger)
    state["grade"] += 1
    grade = state["grade"]
    stable = knobs["stable_grades"] or max(
        knobs["min_stable_grades"], 1 + int(state["longest_streak_before_regression"])
    )
    lane = lanes(names, policy)
    tier_state = tier_state if isinstance(tier_state, dict) else {}
    outcomes = outcomes or {}
    hinted = set(hint or ())
    base, recency = [], {}
    for name in sorted(names):
        member = (ledger["members"].get(name) or {}).get("sampling") or {}
        newest = outcomes.get(name)
        if (
            lane[name] in knobs["cohorts"]
            and name not in hinted
            and member.get("last_outcome") == "pass"
            and int(member.get("streak") or 0) >= stable
            and member.get("closure") is not None
            and member.get("closure") == closures_now.get(name)
            and type(member.get("last_completed_grade")) is int
            and (newest is None or newest.get("passed") is True)
            and _cert_settled(tier_state.get(name))
        ):
            base.append(name)
            recency[name] = grade - member["last_completed_grade"]
    cover = rotating_cover(docs, base, recency) if base else []
    sweep_every = knobs["full_sweep_every"] or (math.ceil(len(base) / len(cover)) if cover else 1)
    # Stale: not actually graded within one sweep period (e.g. a sweep's admission window closed first).
    eligible = [n for n in base if recency[n] <= sweep_every]
    if len(eligible) != len(base):
        cover = rotating_cover(docs, eligible, recency) if eligible else []
    reasons = []
    if not knobs["enabled"]:
        reasons.append("disabled by policy")
    if grade - int(state["last_full_sweep_grade"] or 0) >= sweep_every:
        reasons.append(f"periodic full sweep (every {sweep_every} grades)")
    regression = state.get("regression_grade")
    if knobs["sweep_after_regression"] and type(regression) is int and regression >= grade - 1:
        reasons.append("a regression was observed in the previous grade")
    if type(remaining_s) in (int, float) and remaining_s <= knobs["final_window_s"]:
        reasons.append("final campaign window")
    if not eligible and not reasons:
        reasons.append("no case is stable yet")
    carried = (
        {}
        if reasons
        else {
            name: {
                "reason": REASON,
                "streak": ledger["members"][name]["sampling"]["streak"],
                "grades_since_graded": recency[name],
            }
            for name in eligible
            if name not in set(cover)
        }
    )
    if reasons:
        state["last_full_sweep_grade"] = grade
    decision = {
        "grade": grade,
        "eligible": len(eligible),
        "sample": len(cover) if not reasons else len(eligible),
        "carried": len(carried),
        "full_sweep": bool(reasons),
        "full_sweep_reasons": reasons,
    }
    state["decisions"] = (state["decisions"] + [decision])[-_KEEP_DECISIONS:]
    report = {
        "schema": "merlin.regression_sampling.v1",
        **decision,
        "stable_grades": stable,
        "full_sweep_every": sweep_every,
        "covering_sample": [] if reasons else cover,
        "cohorts": knobs["cohorts"],
        "status_label": LABEL,
        "scope": "carried cases keep their newest completed result from an earlier grade; they remain "
        "pending for the current source and never count as a pass of this grade",
        "certification_authority": False,
    }
    return carried, report


def record_completion(ledger: dict, name: str, *, passed: bool, closure, grade: int | None) -> None:
    """Advance one member's passing streak from an actually completed result (carried grades do not)."""
    member = _member(ledger, name)
    state = _state(ledger)
    if passed:
        member["streak"] = int(member.get("streak") or 0) + 1
        member["closure"] = closure
    else:
        if member.get("last_outcome") == "pass" and int(member.get("streak") or 0) > 0:
            state["longest_streak_before_regression"] = max(
                int(state["longest_streak_before_regression"]), int(member["streak"])
            )
            # The newest grade index, not the completing snapshot's: two snapshots may overlap and the
            # next decision must see the regression whichever of them reported it.
            state["regression_grade"] = state["grade"]
        member["streak"] = 0
        member["closure"] = None
    member["last_outcome"] = "pass" if passed else "fail"
    member["last_completed_grade"] = grade if type(grade) is int else state["grade"]


def carried_rows(selection) -> dict:
    """``{case: verdict row}`` for this grade's carried cases (no tiers: a carried row states no result)."""
    report = ((selection or {}).get("curriculum") or {}).get("sampling") or {}
    carried = ((selection or {}).get("curriculum") or {}).get("carried_sampled") or {}
    return {
        name: {
            "status": STATUS,
            "tiers": {},
            "carried": {
                "reason": REASON,
                "label": LABEL,
                "sampling_grade": report.get("grade"),
                **{k: v for k, v in (info or {}).items() if k != "reason"},
            },
        }
        for name, info in sorted(carried.items())
    }


__all__ = [
    "DEFAULTS",
    "LABEL",
    "SECTION",
    "STATUS",
    "carried_rows",
    "closures",
    "decide",
    "load_docs",
    "record_completion",
    "rotating_cover",
    "settings",
    "validate",
]
