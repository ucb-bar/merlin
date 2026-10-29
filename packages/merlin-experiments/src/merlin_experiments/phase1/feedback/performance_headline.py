"""A compact performance headline at the top of both verdict files and the round brief.

Once most test cases pass, the author kept doing correctness work: the ranked performance evidence
sat deep inside ``feedback_brief.performance_priority.development_headroom`` (tens of kilobytes into a
multi-megabyte file) and was never read, while the brief carried only the correctness counts. This
block repeats, in a few hundred bytes, what that evidence already says:

* the correctness state in one line, led by the cases passing numerics over EVERY test case (this
  grade's rows, and for a case it did not run -- admission closed, or carried by regression sampling --
  the newest completed result), by cohort; then L3 certificates including the ones being re-proven after
  an edit, and the originals' physical-correctness counts when the grade carries them;
* performance over the cases that hold the work: cycle-weighted array utilization, and the median over
  the cases that together hold :data:`WORK_SHARE` of the useful array work (a display cut derived from
  the data, not a target), each computed separately for measured (gSIM) and modeled (Spike timing model)
  evidence -- a ratio never mixes the two;
* the geometric-mean speedup of each case's current cycles over its first passing measurement of the
  same kind (retained gSIM certificates; self-check estimates of the same timing model);
* the largest-headroom cases with their bound, busiest unit, gap to the roofline and the catalogue
  mechanisms the headroom block already matched (for a case fenced after nearly every output tile, its
  fence reading and the fence mechanisms first), plus where the skills and idle-tuning results are;
  each case also carries ``next_step`` (:mod:`.next_steps`): one ranked, workload-scoped choice its
  compiler does not already use, with the command applying it, which the rendered line shows in place
  of the bare mechanism ids.

Information only: it imposes no threshold on the author and changes no grade, gate or ranking. Every
number is recomputed from evidence the grade already holds. No target name, no ``re``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import reference_baselines

SCHEMA = "merlin.performance_headline.v1"
KEY = "performance_headline"
#: Share of the useful array work whose cases the work-weighted median covers (display cut only).
WORK_SHARE = 0.9
#: Largest-headroom cases repeated in the headline; the full ranking stays in the headroom block.
TOP = 5
#: Display names of the joint test selection's performance lanes (``joint_test_selection.performance_cohorts``).
COHORT_LABELS = {
    "correctness_regression": "functional",
    "primary": "scalability",
    "calibration_diagnostics": "calibration",
}
#: Mechanism ids repeated per case.
MECHANISMS = 2
#: Author skills the headline points at, kept only when the skill ships with this harness.
SKILLS = ("roofline-and-utilization", "ab-library-pass", "pass-library", "fusion")
DETAIL = "feedback_brief.performance_priority.development_headroom"
_MAX_DOCUMENT_BYTES = 64 << 20


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _status(value):
    return value.get("status") if isinstance(value, dict) else value


def _kind(entry: dict) -> str:
    return "measured" if str(entry.get("evidence", "")).startswith("measured") else "modeled"


def _case_rows(workspace, verdict: dict) -> list[dict]:
    """Every test case's newest result: this verdict's rows, and for a case it did not run (gated,
    deferred, carried by regression sampling) the newest completed grade's row, as that file merges them."""
    from .completed_grade import FILENAME, executed

    rows = {
        row["capsule"]: row
        for row in verdict.get("per_capsule") or ()
        if isinstance(row, dict) and isinstance(row.get("capsule"), str)
    }
    # A grade lists only the cases it selected; the completed file holds every case graded so far.
    held = _read(Path(workspace) / "qa" / FILENAME) if workspace is not None else None
    for row in (held or {}).get("per_capsule") or ():
        name = row.get("capsule") if isinstance(row, dict) else None
        if isinstance(name, str) and executed(row) and not executed(rows.get(name)):
            rows[name] = row
    return list(rows.values())


def numerics(rows: list[dict], verdict: dict) -> dict:
    """``{passing, cases, by_cohort: {cohort: [passing, cases]}, carried_sampled}`` over ``rows``."""
    lanes = (verdict.get("joint_test_selection") or {}).get("performance_cohorts") or {}
    membership = {}
    for lane, names in lanes.items() if isinstance(lanes, dict) else ():
        for name in names if isinstance(names, list) else ():
            membership.setdefault(str(name), COHORT_LABELS.get(lane, str(lane)))
    by_cohort: dict[str, list[int]] = {}
    passing = 0
    for row in rows:
        ok = _status((row.get("tiers") or {}).get("L2")) == "pass"
        passing += ok
        counts = by_cohort.setdefault(membership.get(row["capsule"], "other") if membership else "all", [0, 0])
        counts[0] += ok
        counts[1] += 1
    carried = sum(row.get("status") == "carried" for row in verdict.get("per_capsule") or () if isinstance(row, dict))
    # A merged completed grade holds each carried case's earlier row; the selection still names them.
    curriculum = (verdict.get("joint_test_selection") or {}).get("curriculum") or {}
    carried = carried or len(curriculum.get("carried_sampled") or ())
    return {
        "passing": passing,
        "cases": len(rows),
        "by_cohort": dict(sorted(by_cohort.items())),
        "carried_sampled": carried,
    }


def _correctness(verdict: dict, tier_state: dict, workspace=None) -> tuple[str, bool, dict]:
    """One line of correctness state, whether the functional cohort is (nearly) all passing, and counts."""
    rows = _case_rows(workspace, verdict)
    counts = numerics(rows, verdict)
    numerics_passing = counts["passing"]
    failing = sum(_status((row.get("tiers") or {}).get("L2")) == "fail" or row.get("status") == "fail" for row in rows)
    cohorts = ", ".join(f"{name} {p}/{n}" for name, (p, n) in counts["by_cohort"].items())
    parts = [
        f"{numerics_passing}/{len(rows)} cases pass numerics (L2)"
        + (f" (by cohort: {cohorts})" if len(counts["by_cohort"]) > 1 else "")
    ]
    if failing:
        parts.append(f"{failing} fail")
    if counts["carried_sampled"]:
        parts.append(f"{counts['carried_sampled']} stable case(s) carried (sampled) with their newest result")
    certified = reproving = awaiting = refuted = 0
    for entry in tier_state.values():
        if not isinstance(entry, dict):
            continue
        current = _status(entry.get("L3"))
        slots = ((entry.get("<certs>") or {}).get("L3") or {}).values()
        earlier = any(_status(slot) in ("pass", "joined") for slot in slots)
        if current == "pass":
            certified += 1
        elif current == "fail":
            refuted += 1
        elif current is not None:
            reproving, awaiting = reproving + earlier, awaiting + (not earlier)
    if tier_state:
        line = f"L3 certified {certified}/{len(tier_state)} tracked"
        extra = [
            f"{reproving} being re-proven after an edit" if reproving else "",
            f"{awaiting} awaiting a first certificate" if awaiting else "",
            f"{refuted} failed" if refuted else "",
        ]
        extra = [text for text in extra if text]
        parts.append(line + (f" ({', '.join(extra)})" if extra else ""))
    physical = verdict.get("joint_physical_correctness")
    if isinstance(physical, dict) and isinstance(physical.get("required"), int):
        parts.append(
            f"originals' physical correctness {physical.get('passed')}/{physical['required']} "
            f"({physical.get('failed')} failed, {physical.get('pending')} pending)"
        )
    # "Nearly all": the failing count is within counting noise of the cohort size (sqrt n), a display
    # cut derived from the cohort itself. Pending certification is not failure.
    nearly = bool(rows) and numerics_passing > 0 and (len(rows) - numerics_passing) <= math.isqrt(len(rows))
    return "; ".join(parts) + ".", nearly, counts


def _array_work(entry: dict) -> bool:
    """Whether the entry states a utilization (its case has useful MACs) for positive cycles."""
    value = entry.get("array_utilization_pct")
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and _positive(entry.get("cycles"))


def _work_cut(entries: list[dict]) -> list[dict]:
    """The fewest cases, largest useful work first, that together hold WORK_SHARE of the work.

    Work is ``cycles * utilization`` = useful MACs / peak rate, the same unit for every case."""
    weighted = sorted(
        ((e["cycles"] * e["array_utilization_pct"] / 100.0, e) for e in entries if _array_work(e)),
        key=lambda pair: (-pair[0], str(pair[1].get("test_case"))),
    )
    total = sum(work for work, _ in weighted)
    chosen, held = [], 0.0
    for work, entry in weighted:
        if total <= 0 or held >= WORK_SHARE * total:
            break
        chosen.append(entry)
        held += work
    return chosen


def _median(values: list[float]):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _lane(entries: list[dict], speedups: dict) -> dict | None:
    if not entries:
        return None
    # A utilization that rounds to 0.0 is still array work: its cycles stay in the denominator.
    timed = [e for e in entries if _array_work(e)]
    cycles = sum(e["cycles"] for e in timed)
    lane = {"cases": len(entries)}
    if cycles > 0:
        lane["cycle_weighted_array_utilization_pct"] = round(
            sum(e["array_utilization_pct"] * e["cycles"] for e in timed) / cycles, 2
        )
        cut = _work_cut(timed)
        lane["work_cut_cases"] = len(cut)
        lane["work_cut_median_array_utilization_pct"] = (
            round(_median([e["array_utilization_pct"] for e in cut]), 2) if cut else None
        )
    ratios = [speedups[e["test_case"]] for e in entries if e.get("test_case") in speedups]
    if ratios:
        lane["geomean_speedup_vs_first"] = round(math.exp(sum(math.log(r) for r in ratios) / len(ratios)), 3)
        lane["speedup_cases"] = len(ratios)
    return lane


def first_measured(tier_state: dict, measured: dict, advisory: dict | None = None) -> dict:
    """``{case: cycles}`` of each case's earliest retained passing gSIM certificate, else (a case without a
    certificate tier) its earliest idle-time gSIM measurement."""
    found = {name: rows[-1]["cycles"] for name, rows in (advisory or {}).items() if rows}
    for name, entry in tier_state.items():
        slots = ((entry.get("<certs>") or {}).get("L3") or {}) if isinstance(entry, dict) else {}
        for slot in slots.values():
            observed = measured.get(slot.get("execution_digest")) if _status(slot) == "pass" else None
            if observed is not None and _positive(observed[0]):
                found[name] = observed[0]
                break
    return found


def first_modeled(estimates: dict) -> dict:
    """``{case: (cycles, model)}`` of each case's oldest passing self-check estimate."""
    found = {}
    for name, rows in estimates.items():
        passing = [row for row in rows if len(row) < 4 or row[3]]
        if passing:
            performance = passing[-1][2]
            found[name] = (performance.get("estimated_whole_soc_cycles"), performance.get("model_sha256"))
    return found


def _speedups(entries: list[dict], measured_first: dict, modeled_first: dict, models: dict) -> dict:
    out = {}
    for entry in entries:
        name, cycles = entry.get("test_case"), entry.get("cycles")
        if not _positive(cycles):
            continue
        if _kind(entry) == "measured":
            first = measured_first.get(name)
        else:
            recorded = modeled_first.get(name)
            if recorded is None:
                continue  # no earlier estimate: nothing to compare with
            first, model = recorded
            if model is not None and models.get(name) is not None and model != models.get(name):
                first = None  # a different timing model is a different instrument
        if _positive(first):
            out[name] = first / cycles
    return out


def _next_steps(workspace, verdict: dict):
    """A :class:`.next_steps.Context` for ranking each case's one next step, or None when unavailable."""
    if workspace is None:
        return None
    try:
        from .next_steps import Context

        return Context(workspace, verdict, verdict.get("per_capsule") or ())
    except Exception:  # noqa: BLE001 -- the headline keeps its catalogue ids without a ranked step
        return None


def _step(entry: dict, context) -> dict | None:
    if context is None:
        return None
    try:
        from .next_steps import choose

        found = choose(entry, context, suite="core")
    except Exception:  # noqa: BLE001 -- advisory only
        return None
    if found is None:
        return None
    return {key: found[key] for key in ("kind", "choice", "api", "choice_text", "apply", "read")}


def _top(block: dict, context=None, *, guidance: bool = True) -> list[dict]:
    out = []
    for entry in (block.get("largest_headroom") or ())[:TOP]:
        busiest = (entry.get("dominant_modeled_resource") or {}).get("name")
        item = {
            "test_case": entry.get("test_case"),
            "evidence": _kind(entry),
            "cycles": entry.get("cycles"),
            "headroom_cycles": entry.get("headroom_cycles"),
            "gap_to_roofline": entry.get("gap_ratio"),
            "bound_by": entry.get("bound_by"),
            "busiest_unit": busiest,
            "mechanisms": ([
                m.get("id") for m in entry.get("suggested_mechanisms") or () if isinstance(m, dict) and m.get("id")
            ][:MECHANISMS] if guidance else []),
        }
        fence = entry.get("fence_overlap") if isinstance(entry.get("fence_overlap"), dict) else {}
        if fence.get("fence_per_tile"):
            item["fences"] = fence.get("reading")
            if guidance:
                fenced = [m.get("id") for m in fence.get("suggested_mechanisms") or () if isinstance(m, dict)]
                item["mechanisms"] = list(dict.fromkeys([*fenced, *item["mechanisms"]]))[:MECHANISMS]
        if entry.get("measured_on_earlier_source"):
            item["measured_on_earlier_source"] = True
        if entry.get("reference_baselines"):
            item["vs_reference"] = reference_baselines.compact(entry["reference_baselines"])
        step = _step(entry, context) if guidance else None
        if step is not None:
            item["next_step"] = step
        out.append({key: value for key, value in item.items() if value not in (None, [])})
    return out


#: Bundle and family lines the headline repeats (the self-check brief carries every bundle).
FAMILY_LINES = 6


def _families(workspace, verdict: dict) -> dict | None:
    """Passing cases bundled by family and shape class, and each family's cases left (:mod:`.family_rollup`)."""
    if workspace is None:
        return None
    try:
        from . import family_rollup

        found = family_rollup.build(workspace, rows=verdict.get("per_capsule") or ())
    except Exception:  # noqa: BLE001 -- advisory only
        return None
    if not found:
        return None
    left = [r for r in found["families"] if r["correct"] < r["cases"]]
    left.sort(key=lambda r: (-r["left_large"], -(r["cases"] - r["correct"]), r["family"]))
    return {
        "line": found["family_line"],
        "ready_bundles": found["lines"][:FAMILY_LINES],
        "bundle_count": found["bundle_count"],
        "cases_left": [family_rollup.correctness_text(r) for r in left[:FAMILY_LINES]],
    }


def _skills() -> list[str]:
    root = Path(__file__).resolve().parent.parent / "author_skills"
    return [name for name in SKILLS if (root / name).is_dir()]


def _idle_tuning(workspace: Path, *, guidance: bool = True) -> str | None:
    from .idle_tuning import LEDGER

    path = Path(workspace) / LEDGER
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_DOCUMENT_BYTES:
            return None
        ledger = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    rows = [r for r in (ledger.get("experiments") or ()) if isinstance(r, dict)] if isinstance(ledger, dict) else []
    if not rows:
        return None
    faster = [
        r
        for r in rows
        if type(r.get("delta_pct")) in (int, float)
        and r["delta_pct"] < 0
        and str(r.get("numerics", "")).startswith("unchanged")
    ]
    text = f"{LEDGER.as_posix()}: {len(rows)} current result(s), {len(faster)} faster with numerics unchanged"
    if faster:
        best = min(faster, key=lambda r: r["delta_pct"])
        mechanism = f" {best.get('mechanism')}" if guidance else ""
        text += f"; best {best.get('test_case')}{mechanism} {best['delta_pct']}%"
    return text


def build(
    workspace,
    verdict: dict,
    *,
    ranked: list[dict],
    block: dict,
    tier_state: dict,
    measured: dict,
    estimates: dict,
    advisory: dict | None = None,
) -> dict:
    """The headline for ``verdict`` from the headroom evidence already collected for it."""
    from ..author_profiles import modern_advice

    guidance = modern_advice()
    correctness, nearly, counts = _correctness(verdict, tier_state if isinstance(tier_state, dict) else {}, workspace)
    models = {}
    for row in verdict.get("per_capsule") or ():
        if isinstance(row, dict) and isinstance(row.get("performance"), dict):
            models[row.get("capsule")] = row["performance"].get("model_sha256")
    speedups = _speedups(ranked, first_measured(tier_state or {}, measured, advisory), first_modeled(estimates), models)
    headline = {
        "schema": SCHEMA,
        "advisory_only": True,
        "grade": {
            "graded_source_sha256": str(verdict.get("graded_source_sha256") or "")[:16] or None,
            "graded_at": verdict.get("graded_at"),
        },
        "correctness": correctness,
        # Every test case's newest result (this grade's rows, the newest completed result for the rest).
        "numerics": counts,
    }
    if nearly:
        headline["objective"] = (
            "Correctness is essentially done; the objective now is fewer cycles on the cases listed in top_headroom."
        )
    lanes = {
        "measured": _lane([e for e in ranked if _kind(e) == "measured"], speedups),
        "modeled": _lane([e for e in ranked if _kind(e) == "modeled"], speedups),
    }
    headline["performance"] = {name: lane for name, lane in lanes.items() if lane is not None}
    headline["performance"]["work_cut_share_pct"] = round(100 * WORK_SHARE)
    without = (block.get("counts") or {}).get("without_evidence")
    if without:
        headline["performance"]["passing_without_timing"] = without
    compared = block.get("reference_baselines")
    if isinstance(compared, dict):
        headline["reference_baselines"] = {
            "line": reference_baselines.headline_line(compared),
            "by_label": compared.get("by_label"),
            "detail": f"{DETAIL}.reference_baselines and each entry's reference_baselines",
        }
    headline["top_headroom"] = _top(
        block, _next_steps(workspace, verdict) if guidance else None, guidance=guidance
    )
    families = _families(workspace, verdict)
    if families:
        headline["families"] = families
    headline["skills"] = _skills() if guidance else []
    tuning = _idle_tuning(Path(workspace), guidance=guidance)
    if tuning:
        headline["idle_tuning"] = tuning
    headline["detail"] = DETAIL
    return headline


def place(document: dict, headline: dict) -> dict:
    """``document`` with the headline as its first key (readers see it before anything else)."""
    return {KEY: headline, **{key: value for key, value in document.items() if key != KEY}}


def _number(value) -> str:
    if type(value) is float:
        return f"{value:.4g}"
    return str(value)


def render(headline) -> list[str]:
    """Markdown lines for the round brief and the one-line-per-case summary."""
    if not isinstance(headline, dict) or headline.get("schema") != SCHEMA:
        return []
    grade = headline.get("grade") or {}
    identity = ", ".join(
        f"{key} {grade[key]}" for key in ("token", "graded_source_sha256", "graded_at") if grade.get(key)
    )
    lines = [
        f"## Performance headline (grade {identity or 'unidentified'})",
        "",
        f"Correctness: {headline.get('correctness')}",
    ]
    if headline.get("objective"):
        lines.append(f"**{headline['objective']}**")
    performance = headline.get("performance") or {}
    labels = {"measured": "Measured (gSIM)", "modeled": "Modeled (Spike timing model)"}
    for name, label in labels.items():
        lane = performance.get(name)
        if not isinstance(lane, dict):
            continue
        text = f"{label}, {lane.get('cases')} cases:"
        if lane.get("cycle_weighted_array_utilization_pct") is not None:
            text += (
                f" cycle-weighted array utilization {_number(lane['cycle_weighted_array_utilization_pct'])}%;"
                f" median {_number(lane.get('work_cut_median_array_utilization_pct'))}% over the "
                f"{lane.get('work_cut_cases')} cases holding {performance.get('work_cut_share_pct')}% of "
                "useful array work;"
            )
        if lane.get("geomean_speedup_vs_first") is not None:
            first = "gSIM measurement" if name == "measured" else "estimate by the same model"
            text += (
                f" geomean speedup {_number(lane['geomean_speedup_vs_first'])}x vs each case's first passing "
                f"{first} ({lane.get('speedup_cases')} cases)"
            )
        lines.append("- " + text.rstrip(";"))
    if performance.get("passing_without_timing"):
        lines.append(f"- {performance['passing_without_timing']} passing cases have no timing evidence yet")
    compared = headline.get("reference_baselines")
    if isinstance(compared, dict) and compared.get("line"):
        lines.append(f"- {compared['line']} (advisory targets, not code to copy)")
    top = headline.get("top_headroom") or []
    if top:
        lines.append("Largest headroom (cycles above the roofline bound):")
    for index, entry in enumerate(top, 1):
        tail = f"; busiest {entry['busiest_unit']}" if entry.get("busiest_unit") else ""
        tail += f"; {entry['fences']}" if entry.get("fences") else ""
        step = entry.get("next_step") if isinstance(entry.get("next_step"), dict) else None
        if step and step.get("choice_text") and step.get("apply"):
            # One ranked, workload-scoped choice not already in the compiler, with the command applying it.
            tail += f"; next: {step['choice_text']} | apply: {step['apply']} | read: {step.get('read')}"
        elif entry.get("mechanisms"):
            tail += f"; try {', '.join(entry['mechanisms'])}"
        tail += "".join(
            f"; vs {label} {reference_baselines.times(value) if type(value) is float else value}"
            for label, value in (entry.get("vs_reference") or {}).items()
        )
        lines.append(
            f"{index}. `{entry.get('test_case')}` {entry.get('evidence')} {_number(entry.get('cycles'))} "
            f"cycles, {_number(entry.get('gap_to_roofline'))}x roofline ({entry.get('bound_by')}), "
            f"headroom {_number(entry.get('headroom_cycles'))}{tail}"
        )
    families = headline.get("families") if isinstance(headline.get("families"), dict) else {}
    if families.get("ready_bundles"):
        lines.append("Ready for performance now (passing cases bundled by family and shape class: optimize the first "
                     "case, check its siblings; no case waits for the rest of its family):")
        lines += [f"- {text}" for text in families["ready_bundles"]]
    if families.get("cases_left"):
        lines.append("Correctness left, beside the performance work (build on the named library route):")
        lines += [f"- {text}" for text in families["cases_left"]]
    if families.get("line"):
        lines.append(families["line"])
    pointers = []
    if headline.get("skills"):
        pointers.append("skills " + ", ".join(f"`{name}`" for name in headline["skills"]))
    if headline.get("idle_tuning"):
        pointers.append(f"idle tuning {headline['idle_tuning']}")
    pointers.append(f"full ranking `{headline.get('detail', DETAIL)}`")
    lines += ["See " + "; ".join(pointers) + ".", ""]
    return lines


def _read(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_DOCUMENT_BYTES:
            return None
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def load(workspace) -> dict | None:
    """The newest headline in ``qa/`` (latest completed grade, else the live verdict); never raises."""
    from .completed_grade import FILENAME, POINTER

    qa = Path(workspace) / "qa"
    found = []
    for order, name in enumerate((FILENAME, "verdict.json")):
        document = _read(qa / name)
        headline = document.get(KEY) if document else None
        if isinstance(headline, dict) and headline.get("schema") == SCHEMA:
            pointer = document.get(POINTER) if isinstance(document.get(POINTER), dict) else {}
            grade = dict(headline.get("grade") or {})
            if pointer.get("token"):
                grade["token"] = str(pointer["token"])
            found.append((str(grade.get("graded_at") or ""), -order, {**headline, "grade": grade}))
    return max(found, key=lambda item: item[:2])[2] if found else None
