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
            n, t = str(s).split("/"); return {"passed": int(n), "total": int(t)}
        except Exception:
            return {"passed": None, "total": None}
    return {"all": _frac(bk.get("passed")), "public": _frac(bk.get("public_passed")),
            "hidden": _frac(bk.get("hidden_passed")),
            "first_failure_planes": bk.get("first_failure_planes"), "language": bk.get("language")}


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
    "timed out",          # subprocess.TimeoutExpired: "Command '[...]' timed out after 900 seconds"
    "timeout",            # the adjectival/status spelling ("state": "timeout")
    "time budget",        # the broker's own wording: "<sim> exceeded its time budget"
    "wall clock",
    "budget exhausted",
    "out of budget",
    "killed by",          # a supervisor-killed job is abandonment, not a wrong answer
)


def _budget_abandoned(text) -> bool:
    """True when ``text`` records a cert that was ABANDONED on time/budget rather than answered wrongly.

    Case-folded substring matching only -- ``str`` operations, no regex. An empty/absent reason is False:
    an unexplained failure is NOT quietly reclassified as unaffordable, because that would move a real
    defect out of the failure count, which is the more expensive mistake of the two.
    """
    if not isinstance(text, str) or not text.strip():
        return False          # SILENCE IS NOT EVIDENCE OF UNAFFORDABILITY -- see the docstring
    low = text.lower()
    return any(m in low for m in _BUDGET_MARKERS)


def _cert_reason(c: dict) -> str:
    """Every place a capsule record can carry WHY its cert tier did not pass, joined for inspection.

    Two shapes exist on disk and both are read: the grader's tier record (``tiers.L3`` as a dict with a
    ``reason``, as in ``capsule_result.json``) and the verdict's flattened per-capsule entry (``tiers``
    as status strings plus ``failure_plane`` / ``failure_category`` / ``failure_detail``). Reading only
    one of them is how this classification would go silent the next time the verdict shape changes.
    """
    parts = []
    # The verdict's own ABANDONMENT LIST, when qa_check wrote one. This is the only shape that carries
    # the distinction without the reason text (which holds absolute paths and engine command lines and
    # is therefore not on the redacted verdict), so it is checked FIRST and is authoritative.
    if _CERT_TIER_KEY in (c.get("tiers_abandoned") or []):
        parts.append("tier abandoned on time budget")
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
    import json as _json
    from pathlib import Path as _Path
    vs = sorted((_Path(run_dir) / "qa_history").glob("verdict_*.json"),
                key=lambda q: q.stat().st_mtime)
    for v in reversed(vs):
        try:
            j = _json.loads(v.read_text())
        except Exception:
            continue
        pc = j.get("per_capsule") or []
        if not pc or not j.get("n_capsules"):
            continue
        clean = [c for c in pc if c.get("status") == "pass"
                 and (c.get("tiers") or {}).get("L3") == "pass"]
        l2only = [c for c in pc if c.get("status") == "pass"
                  and (c.get("tiers") or {}).get("L3") not in ("pass", None)]
        # Every capsule whose cert tier RAN and did not pass, split by whether it was abandoned or
        # rejected. Skipped/absent cert tiers are in neither: nothing ran, so there is nothing to
        # attribute (they are already visible as gate_passed minus rtl_clean).
        ran_not_passed = [c for c in pc if _cert_status(c) not in _CERT_NOT_RUN]
        budget = [c for c in ran_not_passed if _budget_abandoned(_cert_reason(c))]
        failed = [c for c in ran_not_passed if not _budget_abandoned(_cert_reason(c))]
        return {"rtl_clean": len(clean), "l2_only": len(l2only),
                "gate_passed": j.get("n_passed"), "n_capsules": j.get("n_capsules"),
                "l3_source": v.name,
                "not_certified_budget": len(budget), "not_certified_failed": len(failed)}
    return {"rtl_clean": None, "l2_only": None, "gate_passed": None,
            "n_capsules": None, "l3_source": None,
            "not_certified_budget": None, "not_certified_failed": None}


def load_run(d: Path, audit: dict) -> dict | None:
    ct = d / "cost_time_toolcalls.yaml"
    if not ct.is_file():
        return None
    c = yaml.safe_load(ct.read_text()) or {}
    qa = d / "qa_loop_summary.yaml"
    q = yaml.safe_load(qa.read_text()) if qa.is_file() else {}
    rounds = [{"round": r.get("round"), "n_passed": r.get("n_passed"), "tool_calls": r.get("tool_calls"),
               "all_pass": r.get("all_pass")} for r in (q.get("rounds") or [])]
    man = {}
    mp = d / "run_manifest.yaml"
    if mp.is_file():
        man = yaml.safe_load(mp.read_text()) or {}
    converged = bool(q.get("converged"))
    valid = converged and c.get("available", True) and (c.get("wall_time_seconds", 0) > 60)
    return {
        "run_id": d.name, "valid": valid, "converged": converged,
        "wall_s": c.get("wall_time_seconds"), "cost_usd": c.get("estimated_cost_usd"),
        "tokens_total": c.get("tokens_total"), "tokens_input": c.get("tokens_input"),
        "tokens_cached": c.get("tokens_cached"), "tokens_output": c.get("tokens_output"),
        "tool_calls": c.get("tool_calls"), "thinking_blocks": c.get("thinking_blocks"),
        "n_rounds": q.get("n_rounds", len(rounds)), "rounds": rounds,
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


def main():
    fa = REPORTS / "full_suite_audit.json"
    audit = json.loads(fa.read_text()) if fa.is_file() else {}
    out = {"arms": {a: [] for a in ARM_ORDER}, "n_valid": {}, "arm_order": ARM_ORDER}
    for sub in RUN_DIRS:
        base = C.RUNS / sub        # out/runs/<target>/capsule-bench/<arm>
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
        out["coverage"] = {"class_coverage": audit.get("class_coverage"),
                           "n_capsules": audit.get("n_capsules"), "backends": list((audit.get("backends") or {}))}
    out["caveat"] = ("3-arm A/B/C. valid converged runs: " +
                     ", ".join(f"{a}={out['n_valid'][a]}" for a in ARM_ORDER) +
                     ". full-suite completeness present where full_suite_audit has been run.")
    p = REPORTS / "agentic_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    for a in ARM_ORDER:
        vs = [r for r in out["arms"][a] if r["valid"]]
        line = ", ".join(
            f"{r['run_id']}(${(r['cost_usd'] or 0):.0f},{r['n_rounds']}rd,{r['tool_calls']}tc" +
            (f",{r['fullsuite']['all']['passed']}/{r['fullsuite']['all']['total']}" if r.get('fullsuite') else "") + ")"
            for r in vs)
        print(f"  {a}: {len(vs)} valid -> {line or '(none yet)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
