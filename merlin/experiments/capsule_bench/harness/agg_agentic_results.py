"""Aggregate the 3-approach agentic A/B/C (baseline · merlin · merlin+CIRCT) into a tidy
agentic_results.json for plotting.

Arms are keyed by the run's `environment.yaml::bundle_id` (NOT the run dir — both merlin arms live under
runs/merlin_assisted/):
  raw_baseline_public_v0            -> baseline
  merlin_assisted_public_v0         -> merlin
  merlin_assisted_rtlchecks_public_v0 -> merlin_rtlchecks   (the CIRCT-checks arm)

Per run it surfaces BOTH dimensions:
  • authoring effort  — cost/tokens/tool-calls/wall + per-round n_passed (from cost_time_toolcalls.yaml +
                        qa_loop_summary.yaml)
  • dialect completeness — passed X/25, public/hidden split, first-failure planes (from the full-suite
                        audit reports/full_suite_audit.json::backends[run_id], if it has been run)
Everything is read from artifacts already on disk (no agent re-run). Honest about scale: each run is marked
valid/invalid and N is recorded. -> reports/agentic_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python

# The grader's OWN list of statuses that mean "this row produced no verdict" (`not_graded`, `gated`,
# `screened_only`, `budget_exhausted`, `infrastructure_fault`). IMPORTED, never restated here: a
# second copy would silently stop honouring a status the grader learned later, and this module's whole
# job is to be exact about what was and was not measured. Requires _common (above) on sys.path.
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES  # noqa: E402

EXP = C.EXP
REPORTS = C.REPORTS
# Every run-dir subtree an arm can land in. cpp_merlininfra was missing, and it is its OWN
# subtree (unlike arms 3/4, which share merlin_assisted/), so its runs were never scanned at all:
# the C++/infra rung silently contributed nothing to any aggregate or plot while looking like an
# arm nobody had run yet. Arm is still decided by bundle_id, never by the directory.
RUN_DIRS = ["raw_baseline", "cpp_merlininfra", "merlin_assisted"]
# Bundle id -> arm. Each arm ships several VARIANTS of its bundle (_public_v0, _realistic_v0,
# _hwbringup_v0, _hwbringup_nokernel_v0), and the variant is a condition, not a different arm — so the
# stem is matched structurally rather than enumerating every (arm x variant) pair. Order matters:
# "merlin_assisted_rtlchecks_*" also starts with "merlin_assisted_", so the CIRCT arm is tested first,
# otherwise every arm-4 run would be mis-filed as arm-3.
_ARM_STEMS = (
    ("merlin_assisted_rtlchecks", "merlin_rtlchecks"),
    ("cpp_merlininfra", "cpp_merlininfra"),
    ("merlin_assisted", "merlin"),
    ("raw_baseline", "baseline"),
)
ARM_ORDER = ["baseline", "cpp_merlininfra", "merlin", "merlin_rtlchecks"]


def arm_from_bundle_id(bundle_id: str | None) -> str | None:
    """The arm a bundle id belongs to, by stem. Unknown id -> None (fail closed, never guess)."""
    if not bundle_id:
        return None
    for stem, arm in _ARM_STEMS:
        if bundle_id.startswith(stem):
            return arm
    return None


def _arm_of(d: Path) -> str | None:
    env = d / "environment.yaml"
    if env.is_file():
        arm = arm_from_bundle_id((yaml.safe_load(env.read_text()) or {}).get("bundle_id"))
        if arm:
            return arm
    # fallback for older runs without environment.yaml bundle_id
    man = d / "input_bundle_manifest.yaml"
    if man.is_file():
        arm = arm_from_bundle_id((yaml.safe_load(man.read_text()) or {}).get("bundle_id"))
        if arm:
            return arm
    if (d / "TRACK_RTLCHECKS").exists():
        return "merlin_rtlchecks"
    if d.parent.name == "raw_baseline":
        return "baseline"
    if d.parent.name == "merlin_assisted":
        return "merlin"
    return None


def _completeness(run_id: str, audit: dict) -> dict | None:
    bk = (audit.get("backends") or {}).get(run_id)
    if not bk:
        return None

    def _frac(s):
        try:
            n, t = str(s).split("/")
            return {"passed": int(n), "total": int(t)}
        except Exception:
            return {"passed": None, "total": None}

    return {
        "all": _frac(bk.get("passed")),
        "public": _frac(bk.get("public_passed")),
        "hidden": _frac(bk.get("hidden_passed")),
        "first_failure_planes": bk.get("first_failure_planes"),
        "language": bk.get("language"),
    }


#: Which tier certifies. The existing evidence counters above read the same key.
_CERT_TIER_KEY = "L3"
#: Cert-tier statuses that mean "this tier did not run", as opposed to "it ran and did not pass".
_CERT_NOT_RUN = ("pass", "skipped", "not_run", "unavailable", "", None)

# A cert tier that RAN and hit its wall clock is not evidence of incorrectness -- it is evidence that
# certifying THIS program was unaffordable at the budget offered. The two readings must not share a
# bucket: measured on a 97-capsule gemmini batch, six abandoned certs (the six heaviest DRAM movers,
# ~20k movement operations against a median of 10) turned an arm's 93 into an 87 in the headline, and a
# lowering-COST problem was reported as six numeric defects.
#
# Detected STRUCTURALLY from the recorded reason text, never by regex (`check_no_regex.py` gates this
# repo, and a too-narrow pattern silently mis-buckets a differently-spelled reason -- exactly the defect
# this fix is about). Substring membership over a small vocabulary of budget-exhaustion phrasings: the
# spelling that a subprocess timeout produces (`timed out after 900 seconds`), the broker's own
# abandonment wording, and the redaction-safe forms -- a verdict's `failure_detail` has its digits
# replaced by `#`, so anything anchored on the number would miss every one of them.
_BUDGET_MARKERS = (
    "timed out",  # subprocess.TimeoutExpired: "Command '[...]' timed out after 900 seconds"
    "timeout",  # the adjectival/status spelling ("state": "timeout")
    "time budget",  # the broker's own wording: "<sim> exceeded its time budget"
    "wall clock",
    "budget exhausted",
    "out of budget",
    "killed by",  # a supervisor-killed job is abandonment, not a wrong answer
)


def _budget_abandoned(text) -> bool:
    """True when ``text`` records a cert that was ABANDONED on time/budget rather than answered wrongly.

    Case-folded substring matching only -- ``str`` operations, no regex. An empty/absent reason is False:
    an unexplained failure is NOT quietly reclassified as unaffordable, because that would move a real
    defect out of the failure count, which is the more expensive mistake of the two.
    """
    if not isinstance(text, str) or not text.strip():
        return False  # SILENCE IS NOT EVIDENCE OF UNAFFORDABILITY -- see the docstring
    low = text.lower()
    return any(m in low for m in _BUDGET_MARKERS)


def _cert_abandoned(c: dict) -> bool:
    """Was this capsule's CERT tier abandoned on time/budget rather than answered wrongly?

    Two sources, in order of authority:

    1. ``tiers_abandoned`` -- the grader's own STRUCTURED list of tiers it abandoned on budget (written
       by ``qa_check``). When present it is the answer: no text is being interpreted at all.
    2. otherwise the recorded reason text, so verdicts written before that field existed -- every
       verdict already on disk -- still classify instead of silently counting as defects.
    """
    abandoned = c.get("tiers_abandoned")
    if isinstance(abandoned, (list, tuple)) and _CERT_TIER_KEY in abandoned:
        return True
    return _budget_abandoned(_cert_reason(c))


def _cert_reason(c: dict) -> str:
    """Every place a capsule record can carry WHY its cert tier did not pass, joined for inspection.

    Two shapes exist on disk and both are read: the grader's tier record (``tiers.L3`` as a dict with a
    ``reason``, as in ``capsule_result.json``) and the verdict's flattened per-capsule entry (``tiers``
    as status strings plus ``failure_plane`` / ``failure_category`` / ``failure_detail``). Reading only
    one of them is how this classification would go silent the next time the verdict shape changes.
    """
    # RECORDED text only. The verdict's structured `tiers_abandoned` list is read by _cert_abandoned
    # instead of being turned into a synthetic reason sentence here: a classifier that matches text it
    # wrote itself cannot be told apart from one that reads nothing, and this function's callers want
    # the reason AS RECORDED.
    parts = []
    tier = (c.get("tiers") or {}).get(_CERT_TIER_KEY)
    if isinstance(tier, dict):
        for k in ("reason", "detail", "error"):
            v = tier.get(k)
            if isinstance(v, str):
                parts.append(v)
    for k in ("failure_detail", "failure_reason", "failure_category"):
        v = c.get(k)
        if isinstance(v, str):
            parts.append(v)
    return " | ".join(parts)


def _cert_status(c: dict) -> str:
    """The cert tier's status, from either record shape (dict tier record or flat status string)."""
    tier = (c.get("tiers") or {}).get(_CERT_TIER_KEY)
    if isinstance(tier, dict):
        tier = tier.get("status")
    return tier if isinstance(tier, str) else ("" if tier is None else str(tier))


def _l3_evidence(run_dir) -> dict:
    """RTL-backed evidence for a run: how many capsules PASS **and** clear L3, how many pass the cheap
    gate while L3 rejects them, and -- separately -- why each uncertified capsule is uncertified.

    Read from the newest verdict's ``per_capsule`` because that is the as-graded record; a capsule with
    ``status == pass`` and ``tiers.L3 != pass`` is an L2-only pass and must never be counted as evidence.
    Returns None values rather than zeros when no verdict exists, so "not measured" cannot read as
    "none found".

    THREE buckets, not two. ``rtl_clean`` keeps exactly its previous meaning (and so does ``l2_only``),
    and the capsules it excludes are now split by WHY:

    * ``not_certified_budget`` -- the cert tier ran and was abandoned on its wall clock/budget. Not a
      verdict on the program's correctness; a cost fact about certifying it.
    * ``not_certified_failed`` -- the cert tier ran and rejected the program. A real L3 failure.

    A caller printing ``rtl_clean`` therefore reports the same quantity it always did, with the
    breakdown available beside it instead of the two causes being indistinguishable.
    """
    j, vname = latest_verdict(run_dir)
    if j is not None:
        pc = j.get("per_capsule") or []
        clean = [c for c in pc if c.get("status") == "pass" and (c.get("tiers") or {}).get("L3") == "pass"]
        l2only = [c for c in pc if c.get("status") == "pass" and (c.get("tiers") or {}).get("L3") not in ("pass", None)]
        # Every capsule whose cert tier RAN and did not pass, split by whether it was abandoned or
        # rejected. Skipped/absent cert tiers are in neither: nothing ran, so there is nothing to
        # attribute (they are already visible as gate_passed minus rtl_clean).
        ran_not_passed = [c for c in pc if _cert_status(c) not in _CERT_NOT_RUN]
        budget = [c for c in ran_not_passed if _cert_abandoned(c)]
        failed = [c for c in ran_not_passed if not _cert_abandoned(c)]
        # WHAT THIS RUN NEVER MEASURED, BY NAME AND STATUS. `n_capsules` above is the MEASURED
        # denominator, so a deferred/gated/screened row is simply absent from it -- correct for this
        # run in isolation and invisible to anyone comparing it with another run. Carrying the names
        # here means a reader of one run's record can see the shrink, and `cohort_report` can
        # normalize a comparison without re-deriving anything.
        nm = not_measured(j)
        return {
            "rtl_clean": len(clean),
            "l2_only": len(l2only),
            "gate_passed": j.get("n_passed"),
            "n_capsules": j.get("n_capsules"),
            "l3_source": vname,
            "not_certified_budget": len(budget),
            "not_certified_failed": len(failed),
            "n_not_measured": len(nm),
            "not_measured": not_measured_labels(j),
            "not_measured_status": nm,
        }
    return {
        "rtl_clean": None,
        "l2_only": None,
        "gate_passed": None,
        "n_capsules": None,
        "l3_source": None,
        "not_certified_budget": None,
        "not_certified_failed": None,
        "n_not_measured": None,
        "not_measured": None,
        "not_measured_status": None,
    }


# ==================================================================================================
# COMMON-COHORT NORMALIZATION -- comparing runs whose own denominators differ
# ==================================================================================================
# `capsule_grade` sets `n_capsules` to the rows it MEASURED, excluding `NOT_MEASURED_STATUSES`. For a
# single run that is right: a deferred row is not a verdict, and counting it as a failure would put
# `all_pass` out of reach and disable an agent loop's only early exit.
#
# Across runs it is a lie, and it flatters the WEAKEST run. Measured on the g3arm gemmini batch: arms
# 1-3 reached an op pass fraction of ~0.96, so the whole-model gate OPENED, `M2_microvit_gemmini` and
# `SY_micro_model` RAN, FAILED, and stayed in a denominator of 97 -> 93/97 (95.9%). Arm 4 reached
# ~0.79, the gate stayed shut, those same two capsules were DEFERRED (`status: gated`) and left its
# denominator -> 75/95 (78.9%). Arm 4's percentage was computed over a cohort with the two hardest
# rows deleted while the others carried them as failures, so the reported gap UNDERSTATED the real
# one (on the common cohort it is 93/95 vs 75/95).
#
# The fix is NOT to re-add deferred rows as failures -- they were not measured, and
# "not run is not a pass" cuts both ways. It is to INTERSECT: score every run on the rows EVERY
# compared run measured, and name what each run did not measure beside its number.
# ==================================================================================================

#: Verdict stages that are NOT a comparable grade. `first_grade_loop_tier` is the deliberately cheap
#: L2-only snapshot written minutes into a run so the agent gets feedback before its first round
#: closes; scoring a run from it would compare somebody's warm-up against a finished run. Skipped only
#: where a COMPARISON is being built -- `_l3_evidence` keeps reading the newest verdict of any stage,
#: exactly as it always did.
_NOT_COMPARABLE_STAGES = ("first_grade_loop_tier",)


def _verdict_files(run_dir) -> list[Path]:
    """This run's archived verdicts, oldest first (by mtime, which is when the grade was written)."""
    qh = Path(run_dir) / "qa_history"
    if not qh.is_dir():
        return []
    return sorted(qh.glob("verdict_*.json"), key=lambda q: q.stat().st_mtime)


def latest_verdict(run_dir, *, skip_stages: tuple[str, ...] = ()) -> tuple[dict | None, str | None]:
    """The newest archived verdict that actually graded something, plus its file name.

    A verdict with no `per_capsule` rows or `n_capsules` 0 graded nothing and is skipped -- an empty
    grade must never read as a result. `skip_stages` additionally rejects stages that are not a
    comparable grade (see `_NOT_COMPARABLE_STAGES`); the default empty tuple keeps the historical
    behaviour for callers that want "the newest verdict, whatever it is".

    Returns `(None, None)` when the run has no usable verdict -- never a synthesized empty one.
    """
    for v in reversed(_verdict_files(run_dir)):
        try:
            j = json.loads(v.read_text())
        except Exception:  # a half-written verdict is not a verdict
            continue
        if not (j.get("per_capsule") or []) or not j.get("n_capsules"):
            continue
        if j.get("stage") in skip_stages:
            continue
        return j, v.name
    return None, None


def capsule_rows(verdict) -> dict[str, dict]:
    """`{capsule name: row}` from a verdict's `per_capsule`. Unnamed rows are dropped (there is
    nothing to intersect them ON); a repeated name keeps the LAST row, which is the current one."""
    rows: dict[str, dict] = {}
    for c in (verdict or {}).get("per_capsule") or []:
        if isinstance(c, dict) and isinstance(c.get("capsule"), str) and c["capsule"]:
            rows[c["capsule"]] = c
    return rows


def measured_names(verdict) -> set[str]:
    """The capsules this verdict actually MEASURED -- i.e. produced a verdict for.

    Exactly the complement of `NOT_MEASURED_STATUSES` over the rows present, which is the same rule
    `capsule_grade` uses to build `n_capsules`. A row absent from `per_capsule` altogether is not
    measured either, and is absent from this set for free.
    """
    return {n for n, c in capsule_rows(verdict).items() if c.get("status") not in NOT_MEASURED_STATUSES}


def not_measured(verdict) -> dict[str, str]:
    """`{capsule name: status}` for every row this verdict did NOT measure, sorted by name.

    Prefers the grader's own `not_measured_status` map when the verdict carries it (written by
    `capsule_grade.grade`, which knows the full result list), and otherwise derives it from
    `per_capsule` so every verdict already on disk classifies instead of reporting nothing.
    """
    v = verdict or {}
    supplied = v.get("not_measured_status")
    if isinstance(supplied, dict) and supplied:
        return {str(k): str(supplied[k]) for k in sorted(supplied)}
    return {
        n: str(c.get("status")) for n, c in sorted(capsule_rows(v).items()) if c.get("status") in NOT_MEASURED_STATUSES
    }


def not_measured_labels(verdict) -> list[str]:
    """`["M2_microvit_gemmini[gated]", ...]` -- the not-measured names WITH the status that excluded
    each one, so a dropped row is visible in a one-line report rather than silently absent."""
    return [f"{n}[{st}]" for n, st in not_measured(verdict).items()]


def common_cohort(verdicts: dict[str, dict]) -> set[str]:
    """The capsules EVERY one of `verdicts` measured -- the only rows a cross-run ratio may use.

    Intersection, never union: a run that never measured a row has no verdict for it, and inventing
    one (as a pass OR as a failure) is the thing this function exists to prevent. An empty mapping
    yields an empty cohort, which makes every ratio 0/0 -- fail closed, not a vacuous 100%.
    """
    sets = [measured_names(v) for v in verdicts.values()]
    return set.intersection(*sets) if sets else set()


def cohort_scores(verdict, cohort) -> dict:
    """This verdict's score restricted to `cohort`: gate passes and L3-clean passes over the same
    denominator every compared run gets. A cohort row missing from this verdict is not a pass."""
    rows = capsule_rows(verdict)
    passed = sorted(n for n in cohort if (rows.get(n) or {}).get("status") == "pass")
    l3 = sorted(n for n in passed if _cert_status(rows[n]) == "pass")
    return {
        "passed": len(passed),
        "l3_clean": len(l3),
        "of": len(cohort),
        "passed_ratio": f"{len(passed)}/{len(cohort)}",
        "l3_clean_ratio": f"{len(l3)}/{len(cohort)}",
    }


def cohort_report(runs: dict[str, dict]) -> dict:
    """Compare `runs` on the rows all of them measured.

    `runs` maps run id -> `{"arm": str, "verdict": dict, "verdict_file": str | None}`.

    Every run gets BOTH numbers: its own `n_passed/n_capsules` (a per-run fact, kept verbatim) and
    the cohort-normalized `passed/common` + `l3_clean/common`. `own_ratios_comparable` says whether
    the own ratios may be printed side by side at all: they may only when every run measured exactly
    the cohort. Equal own denominators are NOT sufficient -- two runs can both measure 96 of 97 rows
    and have measured DIFFERENT 96, in which case their own ratios are still not the same question.
    """
    verdicts = {rid: (meta.get("verdict") or {}) for rid, meta in runs.items()}
    cohort = common_cohort(verdicts)
    union: set[str] = set()
    for v in verdicts.values():
        union |= measured_names(v)
    per_run: dict[str, dict] = {}
    for rid, meta in sorted(runs.items()):
        v = verdicts[rid]
        mine = measured_names(v)
        per_run[rid] = {
            "arm": meta.get("arm"),
            "verdict_file": meta.get("verdict_file"),
            "own_n_passed": v.get("n_passed"),
            "own_n_capsules": v.get("n_capsules"),
            "own_ratio": f"{v.get('n_passed')}/{v.get('n_capsules')}",
            # True only when this run measured the whole cohort AND nothing outside it, i.e. its own
            # ratio asks the same question the cohort ratio does.
            "own_is_cohort": mine == cohort,
            "cohort": cohort_scores(v, cohort),
            "n_not_measured": len(not_measured(v)),
            "not_measured": not_measured_labels(v),
            "not_measured_status": not_measured(v),
            # Rows some OTHER compared run measured and this one did not -- the rows whose absence
            # is what shrinks this run's own denominator relative to its neighbours.
            "missing_vs_union": sorted(union - mine),
        }
    comparable = bool(per_run) and all(r["own_is_cohort"] for r in per_run.values())
    # NO row was measured by every selected run: the selection spans runs graded against different
    # capsule sets (different batches, or a suite that grew between them). Every cohort ratio is then
    # 0/0 -- arithmetically right, and worthless. Fail LOUD rather than emit a table of zeros.
    reason = None
    if per_run and not cohort:
        reason = (
            f"the {len(per_run)} selected run(s) share no measured capsule at all (union "
            f"{len(union)}), so every cohort ratio below is 0/0. Narrow the selection to runs "
            f"graded against the same capsule set."
        )
    return {
        "n_runs": len(per_run),
        "empty_cohort_reason": reason,
        "common_cohort_size": len(cohort),
        "common_cohort": sorted(cohort),
        "union_measured_size": len(union),
        "union_measured": sorted(union),
        "own_denominators": {rid: r["own_n_capsules"] for rid, r in per_run.items()},
        "own_ratios_comparable": comparable,
        "comparable_metric": "own" if comparable else "cohort",
        "runs": per_run,
        "note": (
            "Compare runs on `cohort` (passed/l3_clean over the "
            f"{len(cohort)} capsule(s) every one of the {len(per_run)} run(s) measured). "
            + (
                "Every run measured exactly that cohort here, so each run's own "
                "n_passed/n_capsules asks the same question and is equally comparable."
                if comparable
                else "The `own_*` figures are PER-RUN ONLY and must NOT be placed side by side: a "
                "run's own denominator excludes what it never measured, which shrinks it exactly "
                "where that run did worst. See each run's `not_measured` for what it dropped."
            )
        ),
    }


def format_cohort_table(report: dict) -> str:
    """The cohort report as text, leading with the comparable columns.

    When the own denominators are not comparable the own column is printed LAST and labelled as
    per-run-only, so no reader can line up two ratios with different denominators as if they
    answered the same question -- which is the specific lie this whole path exists to prevent.
    """
    runs = report.get("runs") or {}
    n = report.get("common_cohort_size", 0)
    comparable = bool(report.get("own_ratios_comparable"))
    lines = [
        f"common cohort: {n} capsule(s) measured by ALL {report.get('n_runs', 0)} run(s); "
        f"union {report.get('union_measured_size', 0)}"
    ]
    # An empty cohort makes every ratio 0/0. That is the correct arithmetic and a useless table, so
    # say WHY out loud instead of printing a wall of zeros a reader might take for a result.
    if report.get("empty_cohort_reason"):
        lines.append(f"NO COMPARISON POSSIBLE: {report['empty_cohort_reason']}")
    if not comparable:
        dens = sorted({d for d in (report.get("own_denominators") or {}).values() if d is not None})
        lines.append(
            f"NOT COMPARABLE side by side: own denominators/cohorts differ "
            f"({', '.join(str(d) for d in dens)}). Quote pass/cohort and L3/cohort; the "
            f"own column below is per-run only."
        )
    own_hdr = "own" if comparable else "own (per-run ONLY)"
    hdr = f"{'run':46s} {'arm':16s} {'pass/cohort':>12s} {'L3/cohort':>11s} {own_hdr:>19s}  not measured"
    lines += [hdr, "-" * len(hdr)]
    for rid, r in sorted(runs.items(), key=lambda kv: (str(kv[1].get("arm")), kv[0])):
        co = r.get("cohort") or {}
        nm = ", ".join(r.get("not_measured") or []) or "-"
        lines.append(
            f"{rid:46s} {str(r.get('arm')):16s} {co.get('passed_ratio', '?'):>12s} "
            f"{co.get('l3_clean_ratio', '?'):>11s} {r.get('own_ratio', '?'):>19s}  {nm}"
        )
    lines.append("")
    lines.append(report.get("note", ""))
    return "\n".join(lines)


def collect_comparable_runs(select: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Scan the arm subtrees for runs that have a comparable grade.

    `select` is a tuple of substrings; a run is considered when its id contains ANY of them (an empty
    tuple takes every run). Returns `(runs, skipped)` where `runs` feeds `cohort_report` and
    `skipped` maps run id -> why it is not in the comparison, so a run is never silently dropped.

    Deliberately independent of `load_run`: that function requires `cost_time_toolcalls.yaml`, which
    a run that has not been finalized yet does not have, and a run's GRADE is readable long before
    its cost accounting is written.
    """
    runs: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for sub in RUN_DIRS:
        base = C.RUNS / sub
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if select and not any(tok in d.name for tok in select):
                continue
            j, vname = latest_verdict(d, skip_stages=_NOT_COMPARABLE_STAGES)
            if j is None:
                skipped[d.name] = "no comparable verdict in qa_history/ (nothing graded yet)"
                continue
            runs[d.name] = {"arm": _arm_of(d) or d.parent.name, "verdict": j, "verdict_file": vname}
    return runs, skipped


def load_run(d: Path, audit: dict) -> dict | None:
    ct = d / "cost_time_toolcalls.yaml"
    if not ct.is_file():
        return None
    c = yaml.safe_load(ct.read_text()) or {}
    qa = d / "qa_loop_summary.yaml"
    q = yaml.safe_load(qa.read_text()) if qa.is_file() else {}
    rounds = [
        {
            "round": r.get("round"),
            "n_passed": r.get("n_passed"),
            "tool_calls": r.get("tool_calls"),
            "all_pass": r.get("all_pass"),
        }
        for r in (q.get("rounds") or [])
    ]
    man = {}
    mp = d / "run_manifest.yaml"
    if mp.is_file():
        man = yaml.safe_load(mp.read_text()) or {}
    converged = bool(q.get("converged"))
    valid = converged and c.get("available", True) and (c.get("wall_time_seconds", 0) > 60)
    return {
        "run_id": d.name,
        "valid": valid,
        "converged": converged,
        "wall_s": c.get("wall_time_seconds"),
        "cost_usd": c.get("estimated_cost_usd"),
        "tokens_total": c.get("tokens_total"),
        "tokens_input": c.get("tokens_input"),
        "tokens_cached": c.get("tokens_cached"),
        "tokens_output": c.get("tokens_output"),
        "tool_calls": c.get("tool_calls"),
        "thinking_blocks": c.get("thinking_blocks"),
        "n_rounds": q.get("n_rounds", len(rounds)),
        "rounds": rounds,
        "public_pass": man.get("public_dev_pass") or man.get("pass_public"),
        "hidden_pass": man.get("hidden_pass") or man.get("pass_hidden"),
        "fullsuite": _completeness(d.name, audit),
        # L3 IS THE METRIC. A bare pass count is an L2-GATE number: under bwrap the materializer caps
        # required_oracle_tiers at L2, so capsules can pass the gate and then be REJECTED by
        # cycle-accurate RTL. Measured across six gemmini runs, the gate over-stated by exactly six
        # every time (93/97 gate vs 87 RTL-clean), identically regardless of arm or seeding. Reporting
        # `passed` without this beside it over-claims systematically, so the loader now always carries
        # it and no consumer has to recompute it.
        **_l3_evidence(d),
    }


def cohort_main(select: tuple[str, ...]) -> int:
    """Print a cohort-normalized comparison of the selected runs. READ-ONLY: writes nothing, so it is
    safe to point at a batch while some of its runs are still live."""
    runs, skipped = collect_comparable_runs(select)
    if not runs:
        print(f"no run with a comparable verdict matched {list(select) or ['<all>']}")
        for rid, why in sorted(skipped.items()):
            print(f"  skipped {rid}: {why}")
        return 1
    report = cohort_report(runs)
    print(format_cohort_table(report))
    matched_skips = {rid: why for rid, why in skipped.items() if not select or any(tok in rid for tok in select)}
    if matched_skips:
        print("\nnot in the comparison (nothing graded to compare):")
        for rid, why in sorted(matched_skips.items()):
            print(f"  {rid}: {why}")
    return 0


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--cohort [SUBSTR,...]` compares runs on the rows all of them measured, instead of on each
    # run's own (differently sized) denominator. Substring selection so a batch is named the way it
    # already is on disk ("g3arm"); no argument compares every run that has a comparable grade.
    if argv and argv[0] == "--cohort":
        toks = tuple(t for t in ",".join(argv[1:]).split(",") if t)
        return cohort_main(toks)
    if argv:
        print(f"usage: {Path(__file__).name} [--cohort SUBSTR[,SUBSTR...]]", file=sys.stderr)
        return 2
    fa = REPORTS / "full_suite_audit.json"
    audit = json.loads(fa.read_text()) if fa.is_file() else {}
    out = {"arms": {a: [] for a in ARM_ORDER}, "n_valid": {}, "arm_order": ARM_ORDER}
    for sub in RUN_DIRS:
        base = C.RUNS / sub  # out/runs/<target>/capsule-bench/<arm>
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            arm = _arm_of(d)
            if arm is None:
                continue
            r = load_run(d, audit)
            if r:
                out["arms"][arm].append(r)
    for a in ARM_ORDER:
        out["n_valid"][a] = sum(1 for r in out["arms"][a] if r["valid"])
    if audit:
        out["coverage"] = {
            "class_coverage": audit.get("class_coverage"),
            "n_capsules": audit.get("n_capsules"),
            "backends": list((audit.get("backends") or {})),
        }
    out["caveat"] = (
        "3-arm A/B/C. valid converged runs: "
        + ", ".join(f"{a}={out['n_valid'][a]}" for a in ARM_ORDER)
        + ". full-suite completeness present where full_suite_audit has been run."
    )
    # EVERY run in this file carries its own MEASURED denominator, and those denominators are not
    # all the same number. Say so here rather than leaving a plotter to divide by whichever one it
    # happens to read: a ratio whose denominator differs from its neighbour's is not comparable to
    # it. `--cohort` prints the normalized comparison.
    _dens = sorted({r["n_capsules"] for a in ARM_ORDER for r in out["arms"][a] if r.get("n_capsules") is not None})
    if len(_dens) > 1:
        out["denominator_warning"] = {
            "own_denominators": _dens,
            "detail": (
                "These runs did NOT all measure the same number of capsules "
                f"(n_capsules in {_dens}): a run defers rows it could not reach (see each "
                "run's `not_measured`) and those rows leave its denominator. Do NOT compare "
                "n_passed/n_capsules across runs with different denominators -- run "
                "`agg_agentic_results.py --cohort <batch>` for the common-cohort figures."
            ),
        }
    p = REPORTS / "agentic_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for a in ARM_ORDER:
        vs = [r for r in out["arms"][a] if r["valid"]]
        line = ", ".join(
            f"{r['run_id']}(${(r['cost_usd'] or 0):.0f},{r['n_rounds']}rd,{r['tool_calls']}tc"
            + (f",{r['fullsuite']['all']['passed']}/{r['fullsuite']['all']['total']}" if r.get("fullsuite") else "")
            + ")"
            for r in vs
        )
        print(f"  {a}: {len(vs)} valid -> {line or '(none yet)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
