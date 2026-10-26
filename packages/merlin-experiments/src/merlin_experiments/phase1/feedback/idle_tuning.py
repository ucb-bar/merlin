"""Use idle reserved CPUs to A/B-time candidate compiler changes on the cases with the most headroom.

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

When. Only from :meth:`.idle_performance.IdleMeasurer.tick`, after all its gates opened (no author
request waiting, free shared slots beyond the reserve of an admitting grade, a multi-lane broker, disk
room) and when every passing cohort case already has an estimate of the current source; the request
shares its in-flight budget and submission interval. One experiment is an operator ``suite: scalability``
Spike request with origin ``operator_idle_tuning`` and an ``op_idle_tune_`` id: the broker leases at the
AUXILIARY class and dispatches it after every author request.

Bounds, each an environment override (:data:`KNOBS`): cases per experiment (``min(cap, free slots)``),
experiments in flight (``min(cap, free // cases)``), experiments per source digest, candidates
considered and each request's wall-time timeout. Disable with ``MERLIN_IDLE_TUNING=0``.

What the author sees. Each result compares the Spike timing model estimate of a passing performance
case on ``submission/`` with the candidate's estimate under the same model and timing epoch, and states
whether the candidate still passes the case's numerical check. Each result is bound to the
``submission/`` digest it compares against and is dropped as soon as that digest moves on. A knob copy
is made of one submission digest, so a new digest needs a new experiment; a variant's estimate depends
only on its own graded bytes, so it is compared with the current submission estimate of the same case
and re-timed only when those bytes or the timing model change. Current results are written to
``qa/idle_tuning.json`` and attached to each self-check response's ``performance_brief`` for that same
digest. Strictly advisory: nothing is applied, certified, gated or ranked; slower and numerically
different candidates are reported like faster ones.
"""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

from ..brokers import selfcheck_sources as sources
from ..brokers import selfcheck_tuning as tuning_knobs
from . import idle_performance, mechanism_hints, scratch_retention
from .performance_headroom import _bound, _dominant_resource, _positive

ENV = "MERLIN_IDLE_TUNING"
ORIGIN = idle_performance.TUNING_ORIGIN
REQUEST_PREFIX = idle_performance.REQUEST_PREFIX + "tune_"
SCHEMA = "merlin.idle_tuning_experiments.v1"
EXPERIMENT_SCHEMA = "merlin.idle_tuning_experiment.v1"
LEDGER = Path("qa") / "idle_tuning.json"
#: Knobs, each an environment override (name: default). The ceilings are safety caps; work is sized
#: from the free capacity the idle measurer saw.
KNOBS = {
    "MERLIN_IDLE_TUNE_MAX_CASES": 3,         # ceiling on cases per experiment (actual: min(this, free slots))
    "MERLIN_IDLE_TUNE_MAX_REQUESTS": 2,      # ceiling on experiments in flight (actual: min(this, free // cases))
    "MERLIN_IDLE_TUNE_MAX_PER_SOURCE": 16,   # ceiling on experiments started for one submission digest
    "MERLIN_IDLE_TUNE_MAX_CANDIDATES": 8,    # knob values and newest variants considered each round
    "MERLIN_IDLE_TUNE_TIMEOUT_S": 1800,      # wall-time timeout each experiment request declares
}
#: Experiments shown in one brief; the ledger keeps every current one.
BRIEF_LIMIT = 12
_MAX_PY_BYTES = 2 << 20
_MAX_TREE_BYTES = 96 << 20
_IGNORED_PARTS = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})

OFFER = ("Idle reserved CPUs time candidate changes on the passing performance cases with the most "
         "headroom: each variants/<name>/ that differs from submission/, and each value of a knob your "
         "compiler declares in submission/TUNING_KNOBS.json ({\"schema\": \"merlin.tuning_knobs.v1\", "
         "\"settings_file\": <JSON object file in submission/ your compiler reads>, \"knobs\": {name: "
         "{\"values\": [...], \"mechanism\": <catalogue id>}}}), set only in a private copy. Expose a "
         "mechanism choice as a knob or keep it in a variant to have it timed while you work.")
NOTE = ("Advisory: Spike timing model estimates of submission/ and of a candidate on the same cases, "
        "same model and timing epoch, bound to this submission digest. Nothing was applied to "
        "submission/; adopt a change by making it there. Not a grade, certificate or ranking.")


def knob(name: str, environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(name, "").strip()
    return int(raw) if raw.isascii() and raw.isdecimal() and int(raw) > 0 else KNOBS[name]


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "1").strip() != "0"


# ---------------------------------------------------------------------------------------------
# Which cases, which candidates
# ---------------------------------------------------------------------------------------------

def baseline(known: dict, name: str, digest: str):
    """``(cycles, performance, passed)`` of ``name``'s newest estimate of ``digest``, else ``None``."""
    for row in known.get(name) or ():
        if row[1] == digest:
            performance = row[2]
            return performance["estimated_whole_soc_cycles"], performance, (row[3] if len(row) > 3 else True)
    return None


def headroom_cases(cohort, l2: dict, known: dict, digest: str, *, limit: int) -> list[dict]:
    """Passing cohort cases with a current estimate, largest modeled headroom (cycles - bound) first."""
    found = []
    for name in cohort:
        current = baseline(known, name, digest) if l2.get(name) == "pass" else None
        if current is None or not current[2]:
            continue
        cycles, performance, _ = current
        bound, limiter = _bound(performance.get("structural_roofline") or {})
        found.append({"test_case": name, "cycles": cycles, "bound_by": limiter,
                      "model": tuple((key, performance.get(key)) for key in _MODEL_KEYS),
                      "headroom": cycles - bound if bound is not None and cycles >= bound else 0.0,
                      "dominant": _dominant_resource(performance)})
    found.sort(key=lambda case: (-case["headroom"], -case["cycles"], case["test_case"]))
    return found[:limit]


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
    hosted: dict[str, int] = {}
    for row in rows:
        hosted[row["api"]] = hosted.get(row["api"], 0) + 1
    # A symbol import names its mechanism; a module import only when the module hosts just that one.
    return [row["id"] for row in rows if (row["api"], row.get("symbol")) in added
            or (hosted[row["api"]] == 1 and (row["api"], None) in added)]


def _variant_names(workspace: Path, cap: int) -> list[str]:
    root = workspace / sources.VARIANTS
    if root.is_symlink() or not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if sources.valid_name(path.name) and not path.is_symlink() and path.is_dir():
            try:
                found.append((path.stat().st_mtime, path.name))
            except OSError:
                continue
    return [name for _, name in sorted(found, reverse=True)[:cap]]


def _api(selection, mechanism_ids) -> str | None:
    rows = {row["id"]: row for row in _catalogue(selection)}
    apis = [f"{rows[m]['api']}:{rows[m].get('symbol')}" for m in mechanism_ids if m in rows]
    return ", ".join(apis) or None


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

def _comparable(before: dict, after: dict) -> str | None:
    """Why two estimates are not comparable (another model or timing epoch), else ``None``."""
    for key in (*_MODEL_KEYS, "scope_sha256", "comparison_domain_sha256"):
        if before.get(key) != after.get(key):
            return f"estimates differ in {key}"
    return None


def _how_to_apply(candidate: dict) -> str:
    if candidate.get("kind") == "variant":
        return (f"compare variants/{candidate.get('name')}/ with submission/ and carry the change into "
                "submission/; only submission/ is graded officially")
    overrides = candidate.get("overrides") or {}
    settings = candidate.get("settings_file")
    return "; ".join(f"set {name} to {json.dumps(value)} in submission/{settings}" for name, value in overrides.items())


_MODEL_KEYS = ("selection_sha256", "model_sha256", "timing_mode", "timing_epoch")
_COMPARED_KEYS = ("estimated_whole_soc_cycles", *_MODEL_KEYS, "scope_sha256", "comparison_domain_sha256")


def experiment(path: Path) -> dict | None:
    """The parts of one completed experiment response that results are built from, else ``None``."""
    document = idle_performance._load(path)
    described = document.get("tuning_experiment") if isinstance(document, dict) else None
    if not isinstance(described, dict) or described.get("schema") != EXPERIMENT_SCHEMA:
        return None
    candidate = described.get("candidate") if isinstance(described.get("candidate"), dict) else {}
    graded = (document.get("variant") or {}).get("sha256") if isinstance(document.get("variant"), dict) else None
    # A knob copy is made of the submission/ bytes the broker actually graded. A variant's estimate
    # depends on its own bytes only, so it is compared with whichever submission/ digest is current.
    base = document.get("submission_sha256") if candidate.get("kind") == "knob" else described.get("base_sha256")
    if not isinstance(base, str) or not isinstance(graded, str):
        return None
    cases = set(described.get("test_cases") or ())
    rows = []
    for row in document.get("cases") or document.get("per_capsule") or ():
        name = row.get("capsule") if isinstance(row, dict) else None
        if name in cases:
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            rows.append({"test_case": name, "passed": idle_performance.passed(row),
                         "performance": {key: performance.get(key) for key in _COMPARED_KEYS}})
    return {"experiment": described, "candidate": candidate, "graded": graded, "base": base, "rows": rows,
            "request_id": document.get("selfcheck_request_id"), "graded_at": document.get("graded_at_unix_ns")}


def results(channel: Path, known: dict, digest: str, *, cache: dict | None = None) -> list[dict]:
    """Every completed experiment row with a ``submission/`` estimate to compare it with.

    A knob copy is compared with the estimate of the very submission/ bytes it was made from; a variant
    with the estimate of ``digest`` (the current submission/), since its own bytes are all that its
    estimate depends on. ``cache`` maps response file names to :func:`experiment` (a published
    response never changes).
    """
    cache = {} if cache is None else cache
    found = []
    for path in sorted(channel.glob(f"resp_{REQUEST_PREFIX}*.json")):
        if path.name not in cache:
            cache[path.name] = experiment(path)
        parsed = cache[path.name]
        if parsed is None:
            continue
        described, candidate = parsed["experiment"], parsed["candidate"]
        base = digest if candidate.get("kind") == "variant" else parsed["base"]
        document = {"selfcheck_request_id": parsed["request_id"], "graded_at_unix_ns": parsed["graded_at"]}
        graded = parsed["graded"]
        for row in parsed["rows"]:
            name = row["test_case"]
            current = baseline(known, str(name), base)
            if current is None:
                continue
            before, before_performance, before_passed = current
            performance = row["performance"]
            after = performance.get("estimated_whole_soc_cycles")
            after = after if _positive(after) else None
            passes = row["passed"]
            unusable = ("no complete estimate of the candidate" if after is None
                        else _comparable(before_performance, performance))
            record = {
                "test_case": name,
                "mechanism": described.get("mechanism"),
                "api": described.get("api"),
                "candidate": {**candidate, "graded_sha256": graded},
                "measured_cycles_before": before,
                "measured_cycles_after": after if unusable is None else None,
                "delta_pct": round(100.0 * (after - before) / before, 2) if unusable is None else None,
                "numerics": ("unchanged: submission/ and the candidate both pass this case's numerical check"
                             if passes and before_passed else
                             "changed: the candidate fails this case's numerical check" if before_passed else
                             "baseline: submission/ did not pass this case at this digest"),
                "evidence": (f"operator idle tuning experiment on source {base}: Spike timing model estimates"
                             + (", same model and timing epoch" if unusable is None else "")),
                "how_to_apply": _how_to_apply(candidate),
                "source_sha256": base,
                "candidate_timed_with_source": parsed["base"],
                "baseline_model": {key: before_performance.get(key)
                                   for key in (*_MODEL_KEYS, "scope_sha256", "comparison_domain_sha256")},
                "request_id": document.get("selfcheck_request_id"),
                "graded_at_unix_ns": document.get("graded_at_unix_ns"),
            }
            if unusable is not None:
                record["not_compared"] = unusable
            found.append(record)
    return found


def current_results(found: list[dict], digest: str, variant_digests: dict) -> list[dict]:
    """Results still describing the current bytes: this submission digest and the variant's own bytes.

    One record per candidate bytes and case: the newest timing (e.g. after a timing-model change).
    """
    newest: dict[tuple, dict] = {}
    for record in sorted(found, key=lambda r: r.get("graded_at_unix_ns") or 0):
        candidate = record["candidate"]
        newest[(candidate.get("key"), candidate.get("graded_sha256"), record["test_case"])] = record
    keep = []
    for record in newest.values():
        candidate = record["candidate"]
        if record["source_sha256"] != digest:
            continue
        if candidate.get("kind") == "variant" and variant_digests.get(candidate.get("name")) != candidate.get(
                "graded_sha256"):
            continue
        keep.append(record)
    keep.sort(key=lambda r: (r["delta_pct"] is None, r["delta_pct"] if r["delta_pct"] is not None else 0.0,
                             str(r["test_case"]), str(r["request_id"])))
    return keep


def identity(candidate: dict, digest: str, model) -> tuple[str, str, str]:
    """When two experiments are the same one: a knob copy is made of one ``submission/`` digest, while a
    variant's estimate depends only on its own bytes and the timing model.
    Both identities include the selection, model and epoch; budgets reset with them."""
    source = candidate.get("sha256") if candidate["kind"] == "variant" else digest
    return (str(candidate["key"]), str(source), f"model:{model}")


def _write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.next")
    staged.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    staged.replace(path)


# ---------------------------------------------------------------------------------------------
# The tuner
# ---------------------------------------------------------------------------------------------

class IdleTuner:
    """Owned by an :class:`.idle_performance.IdleMeasurer`, which calls it only with every gate open."""

    def __init__(self, workspace, *, digest=None):
        self.workspace = Path(workspace)
        self._digest = digest
        self.requested: set[tuple[tuple[str, str, str], str]] = set()  # (identity, base digest)
        self._variant_memo: dict[str, tuple[tuple, str]] = {}
        self._mechanism_memo: dict[tuple[str, str, str], list[str]] = {}
        self._published: str | None = None
        self._responses: dict[str, dict | None] = {}

    def _tree_digest(self, label: str) -> str:
        if self._digest is not None:
            return self._digest(label)
        from merlin.common.cpu_pool import AUXILIARY, cpu_priority

        from ..brokers.selfcheck import _submission_digest

        with cpu_priority(AUXILIARY):  # holds no slot while it waits for one
            return sources.source_digest(self.workspace, label, _submission_digest)

    def variant_digest(self, name: str) -> str:
        """The digest of ``variants/<name>/``, re-hashed only when a file's size or mtime changed."""
        root = self.workspace / sources.VARIANTS / name
        fingerprint = tuple(sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                                   for p in root.rglob("*") if p.is_file() and not p.is_symlink()))
        memo = self._variant_memo.get(name)
        if memo is not None and memo[0] == fingerprint:
            return memo[1]
        sha = self._tree_digest(f"{sources.VARIANTS}/{name}")
        self._variant_memo[name] = (fingerprint, sha)
        return sha

    def refresh(self, channel: Path, digest: str, known: dict) -> list[dict]:
        """Publish the results that describe the current bytes; drop every other one."""
        if not enabled():
            return []
        found = results(channel, known, digest, cache=self._responses)
        names = {r["candidate"].get("name") for r in found
                 if r["source_sha256"] == digest and r["candidate"].get("kind") == "variant"}
        variant_digests = {}
        for name in sorted(n for n in names if sources.valid_name(n)):
            try:
                variant_digests[name] = self.variant_digest(name)
            except (OSError, ValueError):
                continue  # a removed variant's results are dropped
        current = current_results(found, digest, variant_digests)
        document = {"schema": SCHEMA, "advisory_only": True, "promotion_authority": False,
                    "source_sha256": digest, "note": NOTE, "offer": OFFER, "experiments": current}
        text = json.dumps(document, sort_keys=True)
        if text != self._published:
            _write_atomic(self.workspace / LEDGER, document | {"updated_at_unix_ns": time.time_ns()})
            self._published = text
        return current

    def candidates(self, digest: str, selection) -> list[dict]:
        """Knob values, then the newest variants whose bytes differ from ``submission/``."""
        cap = knob("MERLIN_IDLE_TUNE_MAX_CANDIDATES")
        self._mechanism_memo = {key: value for key, value in self._mechanism_memo.items() if key[0] == digest}
        found = []
        for change in tuning_knobs.alternatives(self.workspace / sources.SUBMISSION):
            mechanisms = [change["mechanism"]] if change["mechanism"] else []
            found.append({"kind": "knob", "key": f"knob:{change['knob']}={json.dumps(change['value'])}",
                          "overrides": {change["knob"]: change["value"]}, "settings_file": change["settings_file"],
                          "current": change["current"], "mechanisms": mechanisms,
                          "label": change["mechanism"] or f"knob {change['knob']}"})
        for name in _variant_names(self.workspace, cap):
            try:
                sha = self.variant_digest(name)
            except (OSError, ValueError):
                continue
            if sha == digest:
                continue  # identical bytes time identically
            if (digest, name, sha) not in self._mechanism_memo:
                try:
                    self._mechanism_memo[(digest, name, sha)] = changed_mechanisms(
                        self.workspace / sources.SUBMISSION, self.workspace / sources.VARIANTS / name, selection)
                except OSError:
                    self._mechanism_memo[(digest, name, sha)] = []
            mechanisms = self._mechanism_memo[(digest, name, sha)]
            found.append({"kind": "variant", "key": f"variant:{name}", "name": name, "sha256": sha,
                          "mechanisms": mechanisms,
                          "label": ", ".join(mechanisms) or f"author variant {name} (no newly imported "
                                                              "catalogue mechanism)"})
        return found[:cap]

    def _in_flight(self, channel: Path) -> tuple[int, set]:
        """Live tuning requests and every ``(identity, base digest)`` started so far."""
        now, live, started = time.time(), 0, set()
        for request in channel.glob(f"req_{REQUEST_PREFIX}*.json"):
            body = idle_performance._load(request)
            experiment = body.get("tuning_experiment") if isinstance(body, dict) else None
            if isinstance(experiment, dict) and isinstance(experiment.get("identity"), list):
                started.add((tuple(str(part) for part in experiment["identity"]), str(experiment.get("base_sha256"))))
            rid = request.name[len("req_"):-len(".json")]
            if (isinstance(body, dict) and not (channel / f"resp_{rid}.json").exists()
                    and now <= idle_performance._deadline_s(body, request)):
                live += 1
        return live, started

    def tick(self, channel: Path, digest: str, cohort, l2: dict, known: dict, *, free: int) -> str:
        """Start at most one experiment with the capacity the measurer found free; returns why."""
        if not enabled():
            return "estimates current; tuning disabled"
        cases = min(knob("MERLIN_IDLE_TUNE_MAX_CASES"), free)
        live, started = self._in_flight(channel)
        if live >= min(knob("MERLIN_IDLE_TUNE_MAX_REQUESTS"), max(1, free // max(cases, 1))):
            return "tuning experiments in flight"
        started |= self.requested
        asked = {same for same, _ in started}
        chosen = headroom_cases(cohort, l2, known, digest, limit=cases)
        if not chosen:
            return "estimates current; no passing case to tune"
        model = chosen[0]["model"]
        model_key = f"model:{model}"
        if sum(base == digest and same[-1:] == (model_key,) for same, base in started) >= knob(
                "MERLIN_IDLE_TUNE_MAX_PER_SOURCE"):
            return "tuning budget for this source used"
        selection = mechanism_hints.load_selection(self.workspace)
        suggested = set()
        for case in chosen:
            for hint in mechanism_hints.hints(selection, dominant=case["dominant"], bound_by=case["bound_by"]):
                suggested.add(hint.get("id"))
        pending = [c for c in self.candidates(digest, selection) if identity(c, digest, model) not in asked]
        if not pending:
            return "estimates current; no untried tuning candidate"
        # Candidates exercising a mechanism the brief suggests for these cases go first; otherwise the
        # declared order (knobs, then the newest variants). No candidate is ever ranked by expected gain.
        pending.sort(key=lambda c: not suggested.intersection(c["mechanisms"]))
        candidate = pending[0]
        names = [case["test_case"] for case in chosen]
        self._submit(channel, names, digest, candidate, selection, suggested, identity(candidate, digest, model))
        # Kept only for this digest: older entries are covered by the request files on disk.
        self.requested = {entry for entry in self.requested if entry[1] == digest}
        self.requested.add((identity(candidate, digest, model), digest))
        scratch_retention.say(f"[idle-tune] {candidate['key']} on {len(names)} case(s): {', '.join(names)}")
        return "tuning submitted"

    def _submit(self, channel: Path, names: list[str], digest: str, candidate: dict, selection, suggested,
                same: tuple) -> str:
        public = {key: candidate[key] for key in ("kind", "key", "name", "sha256", "overrides", "settings_file",
                                                   "current") if key in candidate}
        experiment = {"schema": EXPERIMENT_SCHEMA, "base_sha256": digest, "candidate": public,
                      "test_cases": names, "mechanism": candidate["label"],
                      "mechanism_ids": candidate["mechanisms"], "api": _api(selection, candidate["mechanisms"]),
                      "suggested_for_these_cases": sorted(str(s) for s in suggested if s),
                      "identity": list(same)}
        extra = {"tuning_experiment": experiment}
        target = digest
        if candidate["kind"] == "variant":
            extra["submission"] = f"{sources.VARIANTS}/{candidate['name']}"
            target = candidate["sha256"]
        else:
            extra[tuning_knobs.REQUEST_KEY] = candidate["overrides"]
        return idle_performance.submit(channel, names, target, origin=ORIGIN, prefix=REQUEST_PREFIX,
                                       timeout=knob("MERLIN_IDLE_TUNE_TIMEOUT_S"), extra=extra)


# ---------------------------------------------------------------------------------------------
# Publication in the self-check brief
# ---------------------------------------------------------------------------------------------

def _compact(record: dict) -> dict:
    return {key: record.get(key) for key in ("mechanism", "measured_cycles_before", "measured_cycles_after",
                                             "delta_pct", "numerics", "how_to_apply") if key in record}


def attach(document: dict, workspace) -> dict:
    """Add the current experiments to a ``submission/`` response's ``performance_brief``; never fails it.

    Only results bound to the response's own ``submission_sha256`` are shown; a variant or knob-copy
    response gets nothing, since its brief is about other bytes.
    """
    try:
        brief = document.get("performance_brief")
        if (not enabled() or not isinstance(brief, dict) or document.get("variant")
                or document.get("selfcheck_submission") not in (None, sources.SUBMISSION)):
            return document
        ledger = idle_performance._load(Path(workspace) / LEDGER)
        digest = document.get("submission_sha256")
        experiments = []
        if isinstance(ledger, dict) and isinstance(digest, str) and ledger.get("source_sha256") == digest:
            experiments = [r for r in ledger.get("experiments") or () if isinstance(r, dict)
                           and r.get("source_sha256") == digest]
        observed = {row.get("capsule"): row["performance"]
                    for row in document.get("cases") or document.get("per_capsule") or ()
                    if isinstance(row, dict) and isinstance(row.get("performance"), dict)}
        experiments = [row for row in experiments if not observed.get(row.get("test_case"))
                       or _comparable(row.get("baseline_model") or {}, observed[row["test_case"]]) is None]
        entries = [e for e in brief.get("largest_headroom") or () if isinstance(e, dict)]
        named = {e.get("test_case") for e in entries}
        for entry in entries:
            rows = [_compact(r) for r in experiments if r.get("test_case") == entry.get("test_case")]
            if rows:
                entry["idle_tuning_experiments"] = rows[:4]
        shown = sorted(experiments, key=lambda r: (r.get("test_case") not in named, r.get("delta_pct") is None,
                                                   r.get("delta_pct") or 0.0, str(r.get("test_case"))))
        brief["idle_tuning"] = {"schema": SCHEMA, "advisory_only": True, "promotion_authority": False,
                                "source_sha256": digest, "count": len(experiments),
                                "experiments": shown[:BRIEF_LIMIT], "all": LEDGER.as_posix(),
                                "note": NOTE, "offer": OFFER}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return document
    return document
