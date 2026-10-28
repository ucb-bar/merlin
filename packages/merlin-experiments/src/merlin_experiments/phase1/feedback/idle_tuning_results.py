"""Candidate timing results that survive edits, and their ranked view (:mod:`.idle_tuning`).

A result compares one candidate's Spike timing model estimate of a passing case -- a ``variants/<name>/``
tree or a knob-override copy of ``submission/`` -- with an estimate of ``submission/`` on the same case
under the same model and timing epoch. Results come from every response that graded a candidate: the
idle tuner's own experiments, the author's variant and knob self-checks and each member of an author
``--sweep``. One response is parsed once (a published response never changes).

Which baseline. A variant's estimate depends only on its own bytes, so it is compared with the newest
comparable ``submission/`` estimate of the case. A knob copy is made of one ``submission/`` digest, so it
is compared with the estimate of that digest, else the newest comparable one (said in ``baseline.note``).

Edits do not drop a result. Each record is keyed by (candidate bytes, case, baseline digest, timing
model) and carries its baseline's digest, age and status:

* ``current`` -- the baseline is the current ``submission/`` digest, or the case's newest estimate of
  the current digest has the same emitted ELF;
* ``older_source`` -- ``submission/`` moved on, but no newer estimate shows the case's emitted ELF
  changed (the comparison still describes the case unless the edit touched its emission);
* ``superseded`` -- a newer estimate of the case shows a different emitted ELF (or the timing model
  changed): the comparison no longer describes the case. A variant is re-compared with the newer
  estimate without re-timing it; a knob copy is re-timed by the tuner.

A variant result of bytes the author has since edited stays listed with ``candidate_current: false``.
The ranked view puts the best candidate per case first and reports faster, slower and numerically
different candidates alike. Advisory only: nothing is applied, certified, gated or promoted.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from ..brokers import selfcheck_sources as sources
from . import idle_performance
from .performance_headroom import _positive

EXPERIMENT_SCHEMA = "merlin.idle_tuning_experiment.v1"
MODEL_KEYS = ("selection_sha256", "model_sha256", "timing_mode", "timing_epoch")
COMPARED_KEYS = ("estimated_whole_soc_cycles", *MODEL_KEYS, "scope_sha256", "comparison_domain_sha256",
                 "elf_sha256")
STATUSES = ("current", "older_source", "superseded")
#: Rows of each ranked list kept in the ledger (every record stays under ``experiments``).
RANK_ROWS = 40


def comparable(before: dict, after: dict) -> str | None:
    """Why two estimates are not comparable (another model or timing epoch), else ``None``."""
    for key in (*MODEL_KEYS, "scope_sha256", "comparison_domain_sha256"):
        if before.get(key) != after.get(key):
            return f"estimates differ in {key}"
    return None


def knob_key(overrides: dict) -> str:
    """``knob:name=value[,name=value]``: the identity of one set of knob values."""
    return "knob:" + ",".join(f"{name}={json.dumps(overrides[name])}" for name in sorted(overrides))


def model_key(performance: dict) -> str:
    return json.dumps([performance.get(key) for key in MODEL_KEYS], default=str)


def _elf(performance) -> str | None:
    value = performance.get("elf_sha256") if isinstance(performance, dict) else None
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------------------------
# One response
# ---------------------------------------------------------------------------------------------

def parse(path: Path) -> dict | None:
    """The candidate rows of one completed response, else ``None`` (not a candidate, or unreadable).

    Rows keep only the fields a comparison reads, so the parse cache stays small."""
    from .result_details import load_row

    document = idle_performance._load(path)
    if not isinstance(document, dict):
        return None
    variant = document.get("variant") if isinstance(document.get("variant"), dict) else None
    graded = variant.get("sha256") if variant else None
    if not isinstance(graded, str) or not graded:
        return None
    described = document.get("tuning_experiment")
    described = described if isinstance(described, dict) and described.get("schema") == EXPERIMENT_SCHEMA else {}
    label = document.get("selfcheck_submission") or sources.SUBMISSION
    if variant.get("kind") == "knob_overrides":
        overrides = variant.get("tuning_overrides")
        if label != sources.SUBMISSION or not isinstance(overrides, dict) or not overrides:
            return None  # a knob copy of a variant tree is not a submission/ comparison
        candidate = {"kind": "knob", "key": knob_key(overrides), "overrides": overrides,
                     "settings_file": variant.get("settings_file")}
        base = document.get("submission_sha256")
    else:
        name = variant.get("name")
        if not sources.valid_name(name):
            return None
        candidate = {"kind": "variant", "key": f"variant:{name}", "name": name}
        base = described.get("base_sha256")
    told = described.get("candidate") if isinstance(described.get("candidate"), dict) else {}
    for key in ("current", "settings_file"):
        if key in told and key not in candidate:
            candidate[key] = told[key]
    rows = []
    for compact in document.get("cases") or document.get("per_capsule") or ():
        row = load_row(compact, document_path=path)
        if isinstance(compact, dict) and compact.get("detail") and row is compact:
            continue
        name = row.get("capsule") if isinstance(row, dict) else None
        if not isinstance(name, str):
            continue
        performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
        rows.append({"test_case": name, "passed": idle_performance.passed(row),
                     "performance": {key: performance.get(key) for key in COMPARED_KEYS}})
    return {"candidate": candidate, "graded": graded, "base": base if isinstance(base, str) else None,
            "rows": rows, "request_id": document.get("selfcheck_request_id"),
            "graded_at": document.get("graded_at_unix_ns") or 0,
            "origin": described.get("origin") or ("idle_tuning" if described else "author"),
            "sweep": described.get("sweep"), "mechanism": described.get("mechanism"),
            "api": described.get("api"),
            **({"reference_experiment": described["reference_experiment"]}
               if isinstance(described.get("reference_experiment"), dict) else {})}


def scan(channel: Path, cache: dict) -> list[dict]:
    """Every parsed candidate response in ``channel``; ``cache`` maps file names to :func:`parse`."""
    found = []
    for path in sorted(Path(channel).glob("resp_*.json")):
        if path.name not in cache:  # the broker publishes a response atomically and never rewrites it
            cache[path.name] = parse(path)
        if cache[path.name] is not None:
            found.append(cache[path.name])
    return found


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------

def changed_since(rows, graded_ns: int) -> bool:
    """Whether the case's emitted ELF changed after ``graded_ns``: the newest estimate's ELF differs
    from the one current at that time (rows: :func:`.idle_performance.estimates`, newest first)."""
    if not rows:
        return False
    anchor = next((row for row in rows if row[0] <= graded_ns), rows[-1])
    newest, then = _elf(rows[0][2]), _elf(anchor[2])
    return bool(newest and then and newest != then)


def baseline_status(rows, chosen, digest: str) -> tuple[str, str]:
    """``(status, note)`` of a baseline row ``chosen`` among the case's estimates ``rows``."""
    newer = [row for row in rows if row[0] > chosen[0]]
    if any(comparable(chosen[2], row[2]) is not None for row in newer):
        return "superseded", "the timing model changed since this baseline"
    if chosen[1] == digest:
        return "current", "baseline is the current submission/"
    if changed_since(rows, chosen[0]):
        return "superseded", "a newer estimate shows this case's emitted ELF changed"
    elf = _elf(chosen[2])
    if elf and any(row[1] == digest and _elf(row[2]) == elf for row in newer):
        return "current", "the current submission/ emits the same ELF for this case"
    return "older_source", "submission/ moved on; no newer estimate shows this case's emission changed"


def choose_baseline(rows, performance: dict, *, base: str | None, kind: str):
    """``(row, note)``: the submission estimate a candidate row is compared with, else ``(None, why)``."""
    usable = [row for row in rows or () if comparable(row[2], performance) is None]
    if kind == "knob" and base:
        exact = next((row for row in usable if row[1] == base), None)
        if exact is not None:
            return exact, "the submission/ digest the knob copy was made from"
        if usable:
            return usable[0], f"newest comparable estimate; none of the knob copy's source {base[:12]}"
    if usable:
        return usable[0], "newest comparable submission/ estimate"
    return None, "no comparable submission/ estimate of this case yet"


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

def how_to_apply(candidate: dict) -> str:
    if candidate.get("kind") == "variant":
        return (f"compare variants/{candidate.get('name')}/ with submission/ and carry the change into "
                "submission/; only submission/ is graded officially")
    overrides = candidate.get("overrides") or {}
    settings = candidate.get("settings_file")
    return "; ".join(f"set {name} to {json.dumps(value)} in submission/{settings}" for name, value in overrides.items())


def label(candidate: dict) -> str:
    if candidate.get("kind") == "variant":
        return f"variants/{candidate.get('name')}"
    return candidate.get("key", "knob:?")[len("knob:"):]


def records(parsed: list[dict], known: dict, digest: str, *, variant_digests: dict, now_ns: int | None = None
            ) -> list[dict]:
    """One record per (candidate bytes, case): the newest timing, compared with its baseline."""
    now_ns = time.time_ns() if now_ns is None else now_ns
    newest: dict[tuple, tuple] = {}
    for item in sorted(parsed, key=lambda p: p["graded_at"]):
        for row in item["rows"]:
            identity = (item["candidate"]["key"], item["graded"], row["test_case"],
                        model_key(row["performance"]), row["performance"].get("scope_sha256"),
                        row["performance"].get("comparison_domain_sha256"))
            newest[identity] = (item, row)
    out = []
    for item, row in newest.values():
        candidate, name = item["candidate"], row["test_case"]
        performance = row["performance"]
        after = performance.get("estimated_whole_soc_cycles")
        after = after if _positive(after) else None
        rows = known.get(name) or []
        chosen, note = choose_baseline(rows, performance, base=item["base"], kind=candidate["kind"])
        if chosen is None and after is not None and not rows:
            continue  # nothing of submission/ to compare with yet; the tuner times the baseline
        current_candidate = (candidate["kind"] != "variant"
                             or variant_digests.get(candidate.get("name")) == item["graded"])
        record = {
            "test_case": name,
            "candidate": {**candidate, "label": label(candidate), "graded_sha256": item["graded"],
                          "elf_sha256": _elf(performance)},
            "candidate_current": current_candidate,
            "mechanism": item.get("mechanism") or label(candidate),
            "api": item.get("api"),
            "origin": item.get("origin"),
            "sweep": item.get("sweep"),
            "how_to_apply": how_to_apply(candidate),
            "request_id": item.get("request_id"),
            "graded_at_unix_ns": item.get("graded_at"),
            "candidate_age_s": _age(item.get("graded_at"), now_ns),
            "measured_cycles_after": after,
            "model": model_key(performance),
            "candidate_model": {key: performance.get(key) for key in (*MODEL_KEYS, "scope_sha256",
                                                                       "comparison_domain_sha256")},
        }
        if "reference_experiment" in item:
            record["reference_experiment"] = item["reference_experiment"]
        if chosen is None:
            record.update({"measured_cycles_before": None, "delta_pct": None, "ratio": None,
                           "baseline_status": "superseded", "baseline": {"sha256": None, "note": note},
                           "not_compared": note, "numerics": _numerics(row["passed"], None),
                           "source_sha256": None})
            record["identity"] = comparison_identity(record)
            out.append(record)
            continue
        before, before_performance = chosen[2]["estimated_whole_soc_cycles"], chosen[2]
        before_passed = chosen[3] if len(chosen) > 3 else True
        status, why = baseline_status(rows, chosen, digest)
        unusable = "no complete estimate of the candidate" if after is None else None
        record.update({
            "measured_cycles_before": before,
            "measured_cycles_after": after if unusable is None else None,
            "delta_pct": round(100.0 * (after - before) / before, 2) if unusable is None else None,
            "ratio": round(after / before, 4) if unusable is None else None,
            "numerics": _numerics(row["passed"], before_passed),
            "source_sha256": chosen[1],
            "candidate_timed_with_source": item["base"],
            "baseline_status": status,
            "baseline": {"sha256": chosen[1], "elf_sha256": _elf(before_performance),
                         "graded_at_unix_ns": chosen[0], "age_s": _age(chosen[0], now_ns),
                         "status": status, "note": f"{note}; {why}"},
            "baseline_model": {key: before_performance.get(key)
                               for key in (*MODEL_KEYS, "scope_sha256", "comparison_domain_sha256")},
            "evidence": (f"Spike timing model estimates of submission/ {chosen[1][:12]} and of the candidate, "
                         "same model and timing epoch"),
        })
        if unusable is not None:
            record["not_compared"] = unusable
        record["identity"] = comparison_identity(record)
        out.append(record)
    out.sort(key=sort_key)
    return out


def comparison_identity(record: dict) -> str:
    """Immutable candidate, baseline, case/scope and model identity, independent of publication time."""
    candidate, base = record.get("candidate") or {}, record.get("baseline") or {}
    key = [candidate.get("key"), candidate.get("graded_sha256"), candidate.get("elf_sha256"),
           record.get("test_case"), base.get("sha256"), base.get("elf_sha256"),
           record.get("candidate_model"), record.get("baseline_model")]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def retain(previous: list[dict], current: list[dict], known: dict, digest: str, variants: dict) -> list[dict]:
    """Keep prior comparisons when a newer baseline or model replaces their displayed comparison."""
    merged = {comparison_identity(row): row for row in previous if isinstance(row, dict)}
    merged.update({row["identity"]: row for row in current})
    for record in merged.values():
        candidate, base = record.get("candidate") or {}, record.get("baseline") or {}
        rows = known.get(record.get("test_case")) or []
        anchor = next((r for r in rows if r[1] == base.get("sha256")
                       and _elf(r[2]) == base.get("elf_sha256")
                       and comparable(r[2], record.get("baseline_model") or {}) is None), None)
        if anchor is not None:
            status, note = baseline_status(rows, anchor, digest)
            record["baseline_status"] = status
            record["baseline"] = {**base, "status": status, "note": note}
        if candidate.get("kind") == "variant":
            record["candidate_current"] = variants.get(candidate.get("name")) == candidate.get("graded_sha256")
    return sorted(merged.values(), key=sort_key)


def _age(graded_ns, now_ns) -> int | None:
    return max(0, round((now_ns - graded_ns) / 1e9)) if isinstance(graded_ns, int) and graded_ns > 0 else None


def _numerics(passed: bool, before_passed) -> str:
    if before_passed is None:
        return "unknown: no submission/ baseline to compare with"
    if passed and before_passed:
        return "unchanged: submission/ and the candidate both pass this case's numerical check"
    if before_passed:
        return "changed: the candidate fails this case's numerical check"
    return "baseline: submission/ did not pass this case at this digest"


def usable(record: dict) -> bool:
    """A comparison that still describes the case: comparable, numerics unchanged, baseline not superseded."""
    return (record.get("delta_pct") is not None and record.get("baseline_status") != "superseded"
            and str(record.get("numerics", "")).startswith("unchanged"))


def sort_key(record: dict):
    return (STATUSES.index(record.get("baseline_status")) if record.get("baseline_status") in STATUSES else 3,
            not record.get("candidate_current", True), record.get("delta_pct") is None,
            record.get("delta_pct") if record.get("delta_pct") is not None else 0.0,
            str(record.get("test_case")), str(record.get("request_id")))


# ---------------------------------------------------------------------------------------------
# Ranked view
# ---------------------------------------------------------------------------------------------

def compact(record: dict) -> dict:
    out = {key: record.get(key) for key in ("test_case", "mechanism", "measured_cycles_before",
                                             "measured_cycles_after", "delta_pct", "numerics", "how_to_apply",
                                             "baseline_status", "reference_experiment") if key in record}
    out["candidate"] = (record.get("candidate") or {}).get("label")
    baseline = record.get("baseline") or {}
    if baseline.get("sha256"):
        out["baseline"] = f"{baseline['sha256'][:12]} ({_minutes(baseline.get('age_s'))} old, {baseline.get('status')})"
    if record.get("candidate_current") is False:
        out["candidate_edited_since"] = True
    return out


def _minutes(seconds) -> str:
    return f"{round(seconds / 60)} min" if isinstance(seconds, (int, float)) else "?"


def by_case(found: list[dict], bundles: dict | None = None) -> list[dict]:
    """Per case: the best usable candidate first, with how many were faster, slower, unchanged or failing."""
    cases: dict[str, list] = {}
    for record in found:
        if record.get("candidate_current", True):
            cases.setdefault(record["test_case"], []).append(record)
    out = []
    for name, group in cases.items():
        good = sorted((r for r in group if usable(r)), key=lambda r: r["delta_pct"])
        row = {"test_case": name, "candidates": len(group),
               "faster": sum(r["delta_pct"] < 0 for r in good), "slower": sum(r["delta_pct"] > 0 for r in good),
               "same": sum(r["delta_pct"] == 0 for r in good),
               "numerics_changed": sum(str(r.get("numerics", "")).startswith("changed") for r in group),
               "superseded": sum(r.get("baseline_status") == "superseded" for r in group)}
        if bundles and name in bundles:
            row["bundle"] = bundles[name]
        if good:
            row["best"] = compact(good[0])
            if good[-1] is not good[0] and good[-1]["delta_pct"] > 0:
                row["worst"] = compact(good[-1])
        out.append(row)
    out.sort(key=lambda r: ((r.get("best") or {}).get("delta_pct") is None,
                            (r.get("best") or {}).get("delta_pct") or 0.0, r["test_case"]))
    return out[:RANK_ROWS]


def by_candidate(found: list[dict]) -> list[dict]:
    """Per candidate bytes: geometric-mean cycle ratio over its usable cases (lower is faster)."""
    groups: dict[tuple, list] = {}
    for record in found:
        candidate = record["candidate"]
        groups.setdefault((candidate.get("key"), candidate.get("graded_sha256")), []).append(record)
    out = []
    for (key, graded), group in groups.items():
        good = [r for r in group if usable(r) and r.get("ratio")]
        geomean = (round(math.exp(sum(math.log(r["ratio"]) for r in good) / len(good)), 4) if good else None)
        first = group[0]
        out.append({"candidate": first["candidate"].get("label"), "graded_sha256": graded,
                    "current": first.get("candidate_current", True), "mechanism": first.get("mechanism"),
                    "geomean_ratio": geomean, "cases_compared": len(good), "cases": len(group),
                    "faster": sum(r["ratio"] < 1 for r in good), "slower": sum(r["ratio"] > 1 for r in good),
                    "numerics_changed": sum(str(r.get("numerics", "")).startswith("changed") for r in group),
                    "how_to_apply": first.get("how_to_apply")})
    out.sort(key=lambda r: (not r["current"], r["numerics_changed"] > 0, r["geomean_ratio"] is None,
                            r["geomean_ratio"] or 0.0, str(r["candidate"])))
    return out[:RANK_ROWS]


__all__ = ["EXPERIMENT_SCHEMA", "MODEL_KEYS", "baseline_status", "by_candidate", "by_case", "changed_since",
           "choose_baseline", "comparable", "compact", "how_to_apply", "knob_key", "parse", "records", "scan",
           "usable"]
