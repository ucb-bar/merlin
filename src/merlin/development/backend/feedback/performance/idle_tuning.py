"""Use idle reserved CPUs to A/B-time candidate compiler changes on the bundles with the most headroom.

This uses complete cycle estimates for candidate comparisons. Counts and generic
resource diagnostics explain a result; they cannot substitute for elapsed time.
Public source roles restrict automatic exploration to representative performance
workloads. Small controls retain correctness and model-diagnostic obligations.

What it can change safely. The library mechanisms the brief suggests are compiler APIs that take the
author's own scheduling IR, problem description and target adapter; applying one means editing the
author's compiler at a place only the author knows, so the harness never writes a mechanism into
anyone's code. It times candidates that already exist as data:

* every ``variants/<name>/`` tree whose digest differs from ``submission/`` (the existing variant
  self-check; the catalogue mechanisms the variant newly imports, read with :mod:`ast`, name it);
* every single-knob change of the knobs the author's compiler declares in
  ``submission/TUNING_KNOBS.json`` (:mod:`..brokers.selfcheck_tuning`): the broker sets the value in
  its private copy of ``submission/``; the workspace is never written.

Where. On the performance bundles of :mod:`.family_rollup` (passing cases sharing an operation family
and shape class): the representative and its first siblings, in the roll-up's value order; a bundle's
representative must be a passing performance-cohort case. Without a roll-up, the passing cohort cases
with the largest modeled headroom form one bundle. A candidate the author already self-checked on a
case of a bundle, or that exercises a mechanism the brief suggests for its representative, goes first;
every other candidate-bundle pair follows (none is ever skipped for expected gain).

When. :meth:`.idle_performance.IdleMeasurer.tick` interleaves baseline timing requests and experiments
while free CPUs remain (:meth:`IdleTuner.plan` proposes; the measurer sizes each experiment to the
free CPUs and starts as many as fit). An experiment needs an estimate of ``submission/`` on the same
cases, so the measurer times a missing baseline beside it. Experiments do not wait until every case
has a current estimate. A mixed bundle is partitioned by its staged public suite;
each experiment is an operator Spike request with origin
``operator_idle_tuning`` and an ``op_idle_tune_`` id: the broker leases at the AUXILIARY class and
dispatches it after every author request, so author requests keep their priority through the lease
class, not by leaving CPUs idle.

Which ones are new. A variant is timed once per (bytes, case, timing model): its estimate depends on
its own bytes only. A knob copy is made of ``submission/``, so it is re-timed only when a newer estimate
shows the case's emitted ELF changed since it was timed (see :mod:`.idle_tuning_results`). Requests the
author made count: a variant the author checked on a case is not timed again there.

Optional caps, each an environment override (:data:`KNOBS`, 0 = none): cases per experiment,
experiments in flight, experiments per source digest, candidates considered, each request's wall-time
timeout. Disable with ``MERLIN_IDLE_TUNING=0``.

What the author sees. ``qa/idle_tuning.json`` lists every result with its baseline's digest, age and
status (current, older source or superseded) and ranks them: best candidate per case first and each
candidate's geometric-mean cycle ratio, faster and slower alike. Results from author variant checks and
``--sweep`` requests are included. Each self-check response's ``performance_brief`` carries the ones
whose baseline still describes the case. Strictly advisory: nothing is applied, certified or gated.
"""

from __future__ import annotations

import ast
import json
import os
import time
from collections import Counter
from pathlib import Path

from merlin.development.backend.brokers import selfcheck_sources as sources
from merlin.development.backend.brokers import selfcheck_tuning as tuning_knobs
from merlin.development.backend.feedback.grading import worker_resources as scratch_retention
from merlin.development.backend.feedback.guidance import mechanism_hints
from merlin.development.backend.feedback.performance import idle_measurement as idle_performance
from merlin.development.backend.feedback.performance import idle_tuning_results as R
from merlin.development.backend.feedback.performance.performance_headroom import (
    _bound,
    _dominant_resource,
    _positive,
)
from merlin.development.backend.tools.selfcheck import _case_groups

# ---- idle_tuning -------------------------------------------------------------------------------

_MODEL_KEYS = R.MODEL_KEYS

ENV = "MERLIN_IDLE_TUNING"

ORIGIN = idle_performance.TUNING_ORIGIN

REQUEST_PREFIX = idle_performance.REQUEST_PREFIX + "tune_"

SCHEMA = "merlin.idle_tuning_experiments.v1"

EXPERIMENT_SCHEMA = R.EXPERIMENT_SCHEMA

LEDGER = Path("qa") / "idle_tuning.json"

_EMPTY_RESULT_ATTEMPTS = 2

KNOBS = {
    "MERLIN_IDLE_TUNE_MAX_CASES": 3,  # cases per experiment: a representative and its first siblings
    "MERLIN_IDLE_TUNE_MAX_REQUESTS": 0,  # experiments in flight (0: as many as free CPUs fit)
    "MERLIN_IDLE_TUNE_MAX_PER_SOURCE": 0,  # experiments started for one submission digest (0: no cap)
    "MERLIN_IDLE_TUNE_MAX_CANDIDATES": 0,  # knob values and newest variants considered (0: every one)
    "MERLIN_IDLE_TUNE_TIMEOUT_S": 1800,  # wall-time timeout each experiment request declares
}

BRIEF_LIMIT = 12

LEDGER_BUNDLES = 24

LEDGER_SIBLINGS = 8

REWRITE_S = 300

_MAX_PY_BYTES = 2 << 20

_MAX_TREE_BYTES = 96 << 20

_IGNORED_PARTS = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})

LIBRARY_PASS_PENDING_PREFIX = ".library-pass-pending-"

OFFER = (
    "Idle reserved CPUs time candidate changes on the passing performance bundles with the most "
    "headroom: each variants/<name>/ that differs from submission/, and each value of a knob your "
    'compiler declares in submission/TUNING_KNOBS.json ({"schema": "merlin.tuning_knobs.v1", '
    '"settings_file": <JSON object file in submission/ your compiler reads>, "knobs": {name: '
    '{"values": [...], "mechanism": <catalogue id>}}}), set only in a private copy. Expose a '
    "mechanism choice as a knob or keep it in a variant to have it timed while you work. To time many "
    "candidates at once on chosen cases: python agent_selfcheck.py --sweep variants/a,variants/b "
    "--knob NAME=V1,V2 --bundle REPRESENTATIVE (or --capsules X,Y)."
)

NOTE = (
    "Advisory: Spike timing model estimates of submission/ and of a candidate on the same cases, "
    "same model and timing epoch. Each result names its baseline digest, age and status (current; "
    "older_source: submission/ moved on but no newer estimate shows the case's emission changed; "
    "superseded: it did). Nothing was applied to submission/; adopt a change by making it there. Not a "
    "grade or certificate."
)

_comparable = R.comparable

results = R.records


def knob(name: str, environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(name, "").strip()
    return int(raw) if raw.isascii() and raw.isdecimal() and int(raw) > 0 else KNOBS[name]


def enabled(environ=None) -> bool:
    """Idle advice is enabled unless the operator explicitly opts out."""
    raw = (os.environ if environ is None else environ).get(ENV)
    return raw is None or raw.strip() != "0"


def baseline(known: dict, name: str, digest: str):
    """``(cycles, performance, passed)`` of ``name``'s newest estimate of ``digest``, else ``None``."""
    for row in known.get(name) or ():
        if row[1] == digest:
            performance = row[2]
            return performance["estimated_whole_soc_cycles"], performance, (row[3] if len(row) > 3 else True)
    return None


def headroom_cases(cohort, l2: dict, known: dict, *, limit: int | None = None) -> list[dict]:
    """Passing cohort cases, largest modeled headroom (cycles - bound) of their newest estimate first;
    cases without an estimate follow in cohort order."""
    found, untimed = [], []
    for name in cohort:
        if l2.get(name) != "pass":
            continue
        rows = known.get(name) or ()
        if not rows:
            untimed.append({"test_case": name, "cycles": None, "headroom": 0.0, "bound_by": None, "dominant": None})
            continue
        performance = rows[0][2]
        cycles = performance["estimated_whole_soc_cycles"]
        bound, limiter = _bound(performance.get("structural_roofline") or {})
        found.append(
            {
                "test_case": name,
                "cycles": cycles,
                "bound_by": limiter,
                "headroom": cycles - bound if bound is not None and cycles >= bound else 0.0,
                "dominant": _dominant_resource(performance),
            }
        )
    found.sort(key=lambda case: (-case["headroom"], -case["cycles"], case["test_case"]))
    found += untimed
    return found if limit is None else found[:limit]


def ready_cases(cohort, l2: dict, known: dict, digest: str, *, limit: int) -> list[dict]:
    """Passing cohort cases with a current estimate, largest modeled headroom (cycles - bound) first."""
    found = []
    for name in cohort:
        current = baseline(known, name, digest) if l2.get(name) == "pass" else None
        if current is None or not current[2]:
            continue
        cycles, performance, _ = current
        if (
            not _positive(cycles)
            or performance.get("status") not in ("PROVISIONAL", "CALIBRATED")
            or performance.get("complete_event_accounting") is not True
        ):
            continue
        bound, limiter = _bound(performance.get("structural_roofline") or {})
        found.append(
            {
                "test_case": name,
                "cycles": cycles,
                "bound_by": limiter,
                "model": tuple((key, performance.get(key)) for key in _MODEL_KEYS),
                "headroom": cycles - bound if bound is not None and cycles >= bound else 0.0,
                "dominant": _dominant_resource(performance),
            }
        )
    found.sort(key=lambda case: (-case["headroom"], -case["cycles"], case["test_case"]))
    return found[:limit]


def bundles_of(rollup, cohort, l2: dict, known: dict) -> list[dict]:
    """``[{label, representative, siblings}]`` to explore, in value order.

    The roll-up's bundles whose representative is a passing cohort case (siblings: passing cases);
    without one, the passing cohort cases by headroom form a single bundle."""
    members = set(cohort)
    out = []
    for bundle in (rollup or {}).get("bundles") or ():
        if not isinstance(bundle, dict):
            continue
        rep = bundle.get("representative")
        if rep not in members or l2.get(rep) != "pass":
            continue
        siblings = [name for name in bundle.get("siblings") or () if l2.get(name) == "pass" and name != rep]
        out.append({"label": bundle.get("label") or rep, "representative": rep, "siblings": siblings})
    if out:
        return out
    ranked = [case["test_case"] for case in headroom_cases(cohort, l2, known)]
    if not ranked:
        return []
    return [{"label": "largest headroom", "representative": ranked[0], "siblings": ranked[1:]}]


def _catalogue(selection) -> list[dict]:
    rows = (selection or {}).get("all_mechanisms") or ()
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("api"), str) and row.get("id")]


def _python_files(root: Path) -> dict[str, Path]:
    found, total = {}, 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in rel.parts) or path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > _MAX_PY_BYTES or total > _MAX_TREE_BYTES:
            continue
        found[rel.as_posix()] = path
    return found


def _imports(data: bytes) -> set[tuple[str, str | None]]:
    try:
        tree = ast.parse(data)
    except (SyntaxError, ValueError):
        return set()
    names: set[tuple[str, str | None]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.update((node.module, alias.name) for alias in node.names)
            names.update((f"{node.module}.{alias.name}", None) for alias in node.names)  # a module import
            names.add((node.module, None))
        elif isinstance(node, ast.Import):
            names.update((alias.name, None) for alias in node.names)
    return names


def changed_mechanisms(submission: Path, variant: Path, selection) -> list[str]:
    """Catalogue mechanisms a variant imports in files that differ from ``submission/``'s, by id.

    Only files whose bytes differ are parsed, so a library copied unchanged into both trees names
    nothing; an import the same file of ``submission/`` already has is not new.
    """
    rows = _catalogue(selection)
    if not rows:
        return []
    official = _python_files(submission)
    added: set[tuple[str, str | None]] = set()
    for rel, path in _python_files(variant).items():
        data = path.read_bytes()
        other = official.get(rel)
        before = other.read_bytes() if other is not None else b""
        if data != before:
            added |= _imports(data) - _imports(before)
    hosted = Counter()
    for row in rows:
        hosted[row["api"]] += 1
    # A symbol import names its mechanism; a module import only when the module hosts just that one.
    return [
        row["id"]
        for row in rows
        if (row["api"], row.get("symbol")) in added or (hosted[row["api"]] == 1 and (row["api"], None) in added)
    ]


def _variant_names(workspace: Path, cap: int | None = None) -> list[str]:
    root = workspace / sources.VARIANTS
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if sources.valid_name(path.name) and not path.is_symlink() and path.is_dir():
            pending = root / (LIBRARY_PASS_PENDING_PREFIX + path.name)
            if pending.exists() or pending.is_symlink():
                continue
            try:
                found.append((path.stat().st_mtime, path.name))
            except OSError as error:
                scratch_retention.say(f"[idle-tune] variant discovery unavailable: {type(error).__name__}: {error}")
                continue
    names = [name for _, name in sorted(found, reverse=True)]
    return names[:cap] if cap else names


def offer() -> str:
    """The shared authored-variant interface, without a withheld catalogue requirement."""
    from merlin.development.backend.authoring.profiles import modern_advice

    return OFFER if modern_advice() else OFFER.replace("<catalogue id>", "<descriptive label>")


def construction_task(workspace: Path) -> dict:
    """Concrete missing integration work, without importing the author's backend.

    File presence is discovery only. Existing declared knobs/changed variants
    remain executable without this conventional library-pass adapter.
    """
    from merlin.development.backend.authoring.profiles import modern_advice

    if not modern_advice():
        return dict(
            kind="declare_alternatives",
            action="Keep alternative authored compiler implementations in immutable variants/ folders, "
            "or declare finite legal settings in submission/TUNING_KNOBS.json consumed by the "
            "same compiler. Compare alternatives on the same selected cases.",
            qualification="All alternatives retain the original numerical, instruction and timing checks; "
            "a candidate or setting alone is not qualification.",
        )
    from merlin.development.backend.feedback.guidance.library_pass import skill_root

    package = Path(workspace) / sources.SUBMISSION
    selected = idle_performance._load(package / "LIBRARY_PASS_SELECTION.json") or {}
    relative = selected.get("adapter") if isinstance(selected, dict) else ""
    path = Path(relative) if isinstance(relative, str) and relative else None
    # Discover the documented author template separately from a declared binding.
    if "adapter" not in selected and (package / "tools/library_pass_adapter.py").is_file():
        path = Path("tools/library_pass_adapter.py")
    present = (
        path is not None
        and not path.is_absolute()
        and ".." not in path.parts
        and (package / path).is_file()
        and not (package / path).is_symlink()
    )
    kind = "declare_alternatives" if present else "bind_source_effects"
    action = (
        "Expose finite legal settings in submission/TUNING_KNOBS.json, consumed by the SAME "
        "manifest emitter; registered actions must reach changed emitted products."
        if present
        else "Construct the authored source/effect adapter: prepare/source for apply_library_pass, "
        "or binding/identity/evaluate for the parallel recipe. Reuse one terminal target encoder "
        "and real completion; evaluation owns original numerics and selected complete ROI cycles."
    )
    return dict(
        kind=kind,
        action=action,
        adapter_discovery="file_present_mapping_unconfirmed" if present else "no_declared_library_pass_adapter_found",
        command=(
            f"python {skill_root(workspace)}/heavy-pass-recipes/scripts/parallel_schedule_recipe.py "
            "--adapter AUTHORED_OWNER.py --library submission/compiler_library --facts SELECTED_FACTS.json "
            "--shape M,K,N --max-in-flight 2 --output schedule-sweep-NEW"
        ),
        qualification="Replace placeholders with original source geometry and selected facts. The target "
        "adapter uses ordinary shared leases/broker; neutral compilation is not numerics or timing.",
    )


def _api(selection, mechanism_ids) -> str | None:
    rows = {row["id"]: row for row in _catalogue(selection)}
    apis = [f"{rows[m]['api']}:{rows[m].get('symbol')}" for m in mechanism_ids if m in rows]
    return ", ".join(apis) or None


def identity(candidate: dict, digest: str, model) -> tuple[str, str, str]:
    """When two experiments are the same one: a knob copy is made of one ``submission/`` digest, while a
    variant's estimate depends only on its own bytes and the timing model."""
    source = candidate.get("sha256") if candidate["kind"] == "variant" else digest
    return (str(candidate["key"]), str(source), f"model:{model}")


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.next")
    staged.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def _request_cases(body: dict) -> list[str]:
    raw = body.get("capsules")
    return [name.strip() for name in raw.split(",") if name.strip()] if isinstance(raw, str) else []


def _without_ages(document: dict) -> dict:
    def strip(records):
        rows = []
        for record in records:
            record = {k: v for k, v in record.items() if k != "candidate_age_s"}
            if isinstance(record.get("baseline"), dict):
                record["baseline"] = {k: v for k, v in record["baseline"].items() if k != "age_s"}
            rows.append(record)
        return rows

    return {
        **document,
        "experiments": strip(document["experiments"]),
        "comparison_history": strip(document.get("comparison_history") or []),
        "ranked_cases": None,
    }


def matching_progress(ledger, document: dict) -> dict | None:
    """Do not show another source/model epoch's queue as this reply's work."""
    if (
        not isinstance(ledger, dict)
        or not isinstance(document.get("submission_sha256"), str)
        or ledger.get("source_sha256") != document["submission_sha256"]
    ):
        return None
    progress = ledger.get("progress")
    if not isinstance(progress, dict) or progress.get("source_sha256") != document["submission_sha256"]:
        return None
    domains = progress.get("model_domains") or ()
    for row in document.get("cases") or document.get("per_capsule") or ():
        performance = row.get("performance") if isinstance(row, dict) else None
        if (
            isinstance(performance, dict)
            and domains
            and not any(all(performance.get(key) == model.get(key) for key in _MODEL_KEYS) for model in domains)
        ):
            return None
    return progress


def attach(document: dict, workspace) -> dict:
    """Add the results whose baseline still describes the case to a ``submission/`` response's
    ``performance_brief``; never fails it. A variant or knob-copy response gets nothing."""
    try:
        brief = document.get("performance_brief")
        if (
            not enabled()
            or not isinstance(brief, dict)
            or document.get("variant")
            or document.get("selfcheck_submission") not in (None, sources.SUBMISSION)
        ):
            return document
        ledger = idle_performance._load(Path(workspace) / LEDGER)
        digest = document.get("submission_sha256")
        experiments = []
        if isinstance(ledger, dict):
            experiments = [
                r
                for r in ledger.get("experiments") or ()
                if isinstance(r, dict)
                and r.get("baseline_status") in {"current", "older_source"}
                and r.get("candidate_current") is True
            ]
        observed = {
            row.get("capsule"): row["performance"]
            for row in document.get("cases") or document.get("per_capsule") or ()
            if isinstance(row, dict) and isinstance(row.get("performance"), dict)
        }
        experiments = [
            row
            for row in experiments
            if not observed.get(row.get("test_case"))
            or R.comparable(row.get("baseline_model") or {}, observed[row["test_case"]]) is None
        ]
        entries = [e for e in brief.get("largest_headroom") or () if isinstance(e, dict)]
        named = {e.get("test_case") for e in entries}
        for entry in entries:
            rows = [R.compact(r) for r in experiments if r.get("test_case") == entry.get("test_case")]
            rows.sort(key=lambda r: (r.get("delta_pct") is None, r.get("delta_pct") or 0.0))
            if rows:
                entry["idle_tuning_experiments"] = rows[:4]
        shown = sorted(
            experiments,
            key=lambda r: (
                r.get("test_case") not in named,
                r.get("delta_pct") is None,
                r.get("delta_pct") or 0.0,
                str(r.get("test_case")),
            ),
        )
        ranked = (
            [
                r
                for r in (ledger or {}).get("ranked_candidates") or ()
                if isinstance(r, dict) and r.get("current") is True
            ][:5]
            if isinstance(ledger, dict)
            else []
        )
        brief["idle_tuning"] = {
            "schema": SCHEMA,
            "advisory_only": True,
            "promotion_authority": False,
            "source_sha256": digest,
            "count": len(experiments),
            "experiments": [R.compact(r) for r in shown[:BRIEF_LIMIT]],
            "ranked_candidates": [
                {
                    k: r.get(k)
                    for k in ("candidate", "geomean_ratio", "cases_compared", "faster", "slower", "numerics_changed")
                }
                for r in ranked
            ],
            "all": LEDGER.as_posix(),
            "note": NOTE,
            "offer": offer(),
        }
        progress = matching_progress(ledger, document)
        if progress is not None:
            brief["idle_tuning"]["progress"] = progress
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return document
    return document
