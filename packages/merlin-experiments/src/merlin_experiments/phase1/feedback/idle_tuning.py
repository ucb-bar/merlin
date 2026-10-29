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
from pathlib import Path

from ..brokers import selfcheck_sources as sources
from ..brokers import selfcheck_tuning as tuning_knobs
from ..tools.selfcheck import _case_groups
from . import idle_performance, mechanism_hints, scratch_retention
from . import idle_tuning_results as R
from .performance_headroom import _bound, _dominant_resource, _positive

_MODEL_KEYS = R.MODEL_KEYS

ENV = "MERLIN_IDLE_TUNING"
ORIGIN = idle_performance.TUNING_ORIGIN
REQUEST_PREFIX = idle_performance.REQUEST_PREFIX + "tune_"
SCHEMA = "merlin.idle_tuning_experiments.v1"
EXPERIMENT_SCHEMA = R.EXPERIMENT_SCHEMA
LEDGER = Path("qa") / "idle_tuning.json"
# No-row failures are not coverage, but a broken candidate must not consume an
# unbounded stream of retries. Source/model/suite identity changes reopen work.
_EMPTY_RESULT_ATTEMPTS = 2
#: Optional caps, each an environment override (name: default; 0 = no cap). Work is sized from the free
#: CPUs the idle measurer saw.
KNOBS = {
    "MERLIN_IDLE_TUNE_MAX_CASES": 3,         # cases per experiment: a representative and its first siblings
    "MERLIN_IDLE_TUNE_MAX_REQUESTS": 0,      # experiments in flight (0: as many as free CPUs fit)
    "MERLIN_IDLE_TUNE_MAX_PER_SOURCE": 0,    # experiments started for one submission digest (0: no cap)
    "MERLIN_IDLE_TUNE_MAX_CANDIDATES": 0,    # knob values and newest variants considered (0: every one)
    "MERLIN_IDLE_TUNE_TIMEOUT_S": 1800,      # wall-time timeout each experiment request declares
}
#: Experiments shown in one brief; the ledger keeps every one.
BRIEF_LIMIT = 12
#: Bundles published in the ledger (``--sweep --bundle`` reads them) and siblings kept per bundle.
LEDGER_BUNDLES = 24
LEDGER_SIBLINGS = 8
#: Seconds after which an unchanged ledger is rewritten anyway, so the ages it states stay near.
REWRITE_S = 300
_MAX_PY_BYTES = 2 << 20
_MAX_TREE_BYTES = 96 << 20
_IGNORED_PARTS = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})
# Author apply CLI creates this sibling before exposing any variant files, and
# removes it only after actual emission validation and the success receipt.
LIBRARY_PASS_PENDING_PREFIX = ".library-pass-pending-"

OFFER = ("Idle reserved CPUs time candidate changes on the passing performance bundles with the most "
         "headroom: each variants/<name>/ that differs from submission/, and each value of a knob your "
         "compiler declares in submission/TUNING_KNOBS.json ({\"schema\": \"merlin.tuning_knobs.v1\", "
         "\"settings_file\": <JSON object file in submission/ your compiler reads>, \"knobs\": {name: "
         "{\"values\": [...], \"mechanism\": <catalogue id>}}}), set only in a private copy. Expose a "
         "mechanism choice as a knob or keep it in a variant to have it timed while you work. To time many "
         "candidates at once on chosen cases: python agent_selfcheck.py --sweep variants/a,variants/b "
         "--knob NAME=V1,V2 --bundle REPRESENTATIVE (or --capsules X,Y).")
NOTE = ("Advisory: Spike timing model estimates of submission/ and of a candidate on the same cases, "
        "same model and timing epoch. Each result names its baseline digest, age and status (current; "
        "older_source: submission/ moved on but no newer estimate shows the case's emission changed; "
        "superseded: it did). Nothing was applied to submission/; adopt a change by making it there. Not a "
        "grade or certificate.")

# Names other modules and tests read.
_comparable = R.comparable
results = R.records


def knob(name: str, environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(name, "").strip()
    return int(raw) if raw.isascii() and raw.isdecimal() and int(raw) > 0 else KNOBS[name]


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "1").strip() != "0"


# ---------------------------------------------------------------------------------------------
# Which cases
# ---------------------------------------------------------------------------------------------

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
            untimed.append({"test_case": name, "cycles": None, "headroom": 0.0, "bound_by": None,
                            "dominant": None})
            continue
        performance = rows[0][2]
        cycles = performance["estimated_whole_soc_cycles"]
        bound, limiter = _bound(performance.get("structural_roofline") or {})
        found.append({"test_case": name, "cycles": cycles, "bound_by": limiter,
                      "headroom": cycles - bound if bound is not None and cycles >= bound else 0.0,
                      "dominant": _dominant_resource(performance)})
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
        if (not _positive(cycles) or performance.get("status") not in ("PROVISIONAL", "CALIBRATED")
                or performance.get("complete_event_accounting") is not True):
            continue
        bound, limiter = _bound(performance.get("structural_roofline") or {})
        found.append({"test_case": name, "cycles": cycles, "bound_by": limiter,
                      "model": tuple((key, performance.get(key)) for key in _MODEL_KEYS),
                      "headroom": cycles - bound if bound is not None and cycles >= bound else 0.0,
                      "dominant": _dominant_resource(performance)})
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


# ---------------------------------------------------------------------------------------------
# Which candidates
# ---------------------------------------------------------------------------------------------

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
            except OSError:
                continue
    names = [name for _, name in sorted(found, reverse=True)]
    return names[:cap] if cap else names


def offer() -> str:
    """The shared authored-variant interface, without a withheld catalogue requirement."""
    from ..author_profiles import modern_advice

    return OFFER if modern_advice() else OFFER.replace("<catalogue id>", "<descriptive label>")


def construction_task(workspace: Path) -> dict:
    """Concrete missing integration work, without importing the author's backend.

    File presence is discovery only. Existing declared knobs/changed variants
    remain executable without this conventional library-pass adapter.
    """
    from ..author_profiles import modern_advice

    if not modern_advice():
        return dict(
            kind="declare_alternatives",
            action="Keep alternative authored compiler implementations in immutable variants/ folders, "
                   "or declare finite legal settings in submission/TUNING_KNOBS.json consumed by the "
                   "same compiler. Compare alternatives on the same selected cases.",
            qualification="All alternatives retain the original numerical, instruction and timing checks; "
                          "a candidate or setting alone is not qualification.",
        )
    from .library_pass import skill_root

    package = Path(workspace) / sources.SUBMISSION
    selected = idle_performance._load(package / "LIBRARY_PASS_SELECTION.json") or {}
    relative = selected.get("adapter", "tools/library_pass_adapter.py") if isinstance(selected, dict) else ""
    path = Path(relative) if isinstance(relative, str) and relative else None
    present = (path is not None and not path.is_absolute() and ".." not in path.parts
               and (package / path).is_file() and not (package / path).is_symlink())
    kind = "declare_alternatives" if present else "bind_source_effects"
    action = ("Expose finite legal settings in submission/TUNING_KNOBS.json, consumed by the SAME "
              "manifest emitter; registered actions must reach changed emitted products."
              if present else
              "Construct the authored source/effect adapter: prepare/source for apply_library_pass, "
              "or binding/identity/evaluate for the parallel recipe. Reuse one terminal target encoder "
              "and real completion; evaluation owns original numerics and selected complete ROI cycles.")
    return dict(kind=kind, action=action, adapter_discovery="file_present_mapping_unconfirmed" if present
                else "no_declared_library_pass_adapter_found", command=(
        f"python {skill_root(workspace)}/heavy-pass-recipes/scripts/parallel_schedule_recipe.py "
        "--adapter AUTHORED_OWNER.py --library submission/compiler_library --facts SELECTED_FACTS.json "
        "--shape M,K,N --max-in-flight 2 --output schedule-sweep-NEW"),
        qualification="Replace placeholders with original source geometry and selected facts. The target "
        "adapter uses ordinary shared leases/broker; neutral compilation is not numerics or timing.")


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


# ---------------------------------------------------------------------------------------------
# The tuner
# ---------------------------------------------------------------------------------------------

class IdleTuner:
    """Owned by an :class:`.idle_performance.IdleMeasurer`, which sizes and starts what it proposes."""

    def __init__(self, workspace, *, digest=None):
        self.workspace = Path(workspace)
        self._digest = digest
        self._variant_memo: dict[str, tuple[tuple, str]] = {}
        self._mechanism_memo: dict[tuple[str, str, str], list[str]] = {}
        self._published: tuple[str, float] | None = None
        self._responses: dict[str, dict | None] = {}
        self._bodies: dict[str, dict | None] = {}
        self._rollup: tuple | None = None
        self.bundles: list[dict] = []
        saved = idle_performance._load(self.workspace / LEDGER)
        self._history = ((saved.get("comparison_history") or saved.get("experiments") or [])
                         if isinstance(saved, dict) else [])

    # -- inputs -----------------------------------------------------------------------------------

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
        if root.is_symlink() or not root.is_dir():
            raise OSError(f"no variant {name}")
        fingerprint = tuple(sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                                   for p in root.rglob("*") if p.is_file() and not p.is_symlink()))
        memo = self._variant_memo.get(name)
        if memo is not None and memo[0] == fingerprint:
            return memo[1]
        sha = self._tree_digest(f"{sources.VARIANTS}/{name}")
        self._variant_memo[name] = (fingerprint, sha)
        return sha

    def _rollup_bundles(self, cohort, l2: dict, known: dict) -> list[dict]:
        """:func:`bundles_of` refreshed after source or completed numerical evidence changes."""
        from .family_rollup import build, selfcheck_identity

        path = self.workspace / "qa" / "latest_completed_grade.json"
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        key = (key, selfcheck_identity(self.workspace))
        if self._rollup is None or self._rollup[0] != key:
            rollup = None
            try:
                rollup = build(self.workspace)
            except Exception:  # noqa: BLE001 -- bundles are an ordering; the fallback still explores
                rollup = None
            self._rollup = (key, rollup)
        self.bundles = bundles_of(self._rollup[1], cohort, l2, known)
        return self.bundles

    def _requests(self, channel: Path) -> list[dict]:
        """Every candidate request on the channel (operator and author), with whether it is live."""
        now, out = time.time(), []
        for path in channel.glob("req_*.json"):
            if path.name not in self._bodies:
                body = idle_performance._load(path)
                self._bodies[path.name] = None if not isinstance(body, dict) else {
                    "label": body.get("submission") or sources.SUBMISSION, "sha": body.get("submission_sha256"),
                    "suite": body.get("suite", "core"),
                    "cases": _request_cases(body), "overrides": body.get(tuning_knobs.REQUEST_KEY),
                    "timed": (body.get("suite") == "scalability" or
                              body.get("suite", "core") in {"core", "programs"}
                              and body.get("sim") == "spike"),
                    "requested_at": body.get("requested_at_unix_ns") or 0,
                    "deadline": idle_performance._deadline_s(body, path),
                    "operator": path.name.startswith("req_" + REQUEST_PREFIX),
                    "experiment": body.get("tuning_experiment") if isinstance(
                        body.get("tuning_experiment"), dict) else None}
            body = self._bodies[path.name]
            if body is None:
                continue
            rid = path.name[len("req_"):-len(".json")]
            answered = (channel / f"resp_{rid}.json").exists()
            out.append({**body, "rid": rid, "answered": answered, "live": not answered and now <= body["deadline"]})
        return out

    def candidates(self, digest: str, selection) -> list[dict]:
        """Knob values, then every variant (newest first) whose bytes differ from ``submission/``."""
        cap = knob("MERLIN_IDLE_TUNE_MAX_CANDIDATES") or None
        self._mechanism_memo = {key: value for key, value in self._mechanism_memo.items() if key[0] == digest}
        found, guided_overrides = self._reference_candidates(digest)
        for change in tuning_knobs.alternatives(self.workspace / sources.SUBMISSION):
            if json.dumps({change["knob"]: change["value"]}, sort_keys=True) in guided_overrides:
                continue
            mechanisms = [change["mechanism"]] if change["mechanism"] else []
            found.append({"kind": "knob", "key": R.knob_key({change["knob"]: change["value"]}),
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
        return found[:cap] if cap else found

    def _reference_candidates(self, digest: str) -> tuple[list[dict], set[str]]:
        """Use selected requests through ordinary private knob copies and budgets."""
        from .reference_experiments import pending_requests

        declared = tuning_knobs.declaration(self.workspace / sources.SUBMISSION)
        if declared is None:
            return [], set()
        current = tuning_knobs.settings(self.workspace / sources.SUBMISSION, declared)
        if current is None:
            return [], set()
        found, seen, covered = [], set(), set()
        for request in pending_requests(digest, include_observed=True):
            if request.get("role") != "variant" or request.get("split") != "training":
                continue
            payload = request.get("broker_payload") or {}
            if payload.get("suite") != "scalability" or request.get("sim") != "spike" or request.get("tiers"):
                continue  # Other protocols remain author requests; this lane only owns Spike scalability.
            try:
                overrides = tuning_knobs._overrides(
                    {tuning_knobs.REQUEST_KEY: request.get("tuning_overrides")}, declared)
            except ValueError:
                continue
            if (request.get("settings") != {**current, **overrides}
                    or all(name in current and tuning_knobs.same_value(current[name], value)
                           for name, value in overrides.items())):
                continue
            capsule = request.get("capsule")
            if not isinstance(capsule, str) or not capsule:
                continue
            key = json.dumps(overrides, sort_keys=True)
            covered.add(key)
            if request.get("owner_observed") or request.get("status") != "ready":
                continue
            identity = (key, capsule)
            if identity in seen:
                continue
            seen.add(identity)
            found.append({
                "kind": "knob", "key": R.knob_key(overrides),
                "overrides": overrides, "settings_file": declared["settings_file"],
                "current": current, "mechanisms": [], "test_cases": [capsule],
                "label": "controlled reference experiment " + request["task_id"],
                "reference_experiment": {name: request[name] for name in
                    ("task_id", "request_id", "packet_identity", "context_identity")},
            })
        return found, covered

    # -- what was tried ---------------------------------------------------------------------------

    def _tried(self, channel: Path, known: dict, selection) -> tuple[set, set, dict]:
        """``(variant (sha, case), knob (key, case), author cases per variant name)`` already covered.

        A request in flight covers its cases. An answered one covers them unless its result is of
        another timing model; a knob copy's also stops covering a case once a newer estimate shows the
        case's emitted ELF changed since it was requested."""
        parsed = {item["request_id"]: item for item in R.scan(channel, self._responses)}
        variants, knobs, authored = set(), set(), {}
        for request in self._requests(channel):
            if not (request["live"] or request["answered"]):
                continue  # expired unanswered: it never ran
            label, cases = request["label"], request["cases"]
            result = parsed.get(request["rid"])
            rows = {row["test_case"]: row for row in (result or {}).get("rows") or ()}
            # A refused/malformed request with no completed case never tested its
            # candidate. Retain it as evidence, but allow the correctly routed retry.
            cases = [case for case in cases if request["live"] or case in rows]

            def other_model(case):
                row = rows.get(case)
                return (selection is not None and row is not None
                        and row["performance"].get("selection_sha256") != selection)

            if label.startswith(sources.VARIANTS + "/") and not request["operator"]:
                authored.setdefault(label.partition("/")[2], set()).update(cases)
            if not request["timed"]:
                continue  # a correctness check returns no cycle estimate
            if label.startswith(sources.VARIANTS + "/"):
                shas = {request["sha"], (result or {}).get("graded")} - {None}
                variants.update((sha, case) for sha in shas for case in cases if not other_model(case))
            elif isinstance(request["overrides"], dict) and request["overrides"]:
                key = R.knob_key(request["overrides"])
                for case in cases:
                    if other_model(case):
                        continue
                    if request["live"] or not R.changed_since(known.get(case) or [], request["requested_at"]):
                        knobs.add((key, case))
        return variants, knobs, authored

    def _empty_attempts(self, channel: Path, digest: str, selection, *, candidates) -> dict:
        """Count missing case results only for the corrected, exact source/model route."""
        parsed = {item["request_id"]: item for item in R.scan(channel, self._responses)}
        identities = {identity(candidate, digest, selection) for candidate in candidates}
        counts = {}
        for request in self._requests(channel):
            experiment = request["experiment"] or {}
            if (not request["answered"] or experiment.get("base_sha256") != digest
                    or experiment.get("public_suite") != request["suite"]):
                continue  # Legacy wrong-suite refusals do not exhaust a corrected route.
            candidate = experiment.get("candidate")
            candidate = candidate if isinstance(candidate, dict) else {}
            expected = list(identity(candidate, digest, selection)) if candidate.get("kind") in {
                "knob", "variant"} and candidate.get("key") else None
            if (experiment.get("identity") != expected or expected is None
                    or tuple(expected) not in identities):
                continue
            returned = {row["test_case"] for row in (parsed.get(request["rid"]) or {}).get("rows") or ()}
            for case in request["cases"]:
                if case not in returned:
                    key = (candidate["key"], case, request["suite"])
                    counts[key] = counts.get(key, 0) + 1
        return counts

    # -- proposals --------------------------------------------------------------------------------

    def plan(self, channel: Path, digest: str, cohort, l2: dict, known: dict, *, selection=None,
             limit: int | None = None) -> list[dict]:
        """Untried candidate-bundle pairs in value order: ``{candidate, cases, bundle, ...}``."""
        if not enabled():
            return []
        inflight = knob("MERLIN_IDLE_TUNE_MAX_REQUESTS")
        if inflight and self.in_flight(channel) >= inflight:
            return []
        bundles = self._rollup_bundles(cohort, l2, known)
        if not bundles:
            return []
        cap = knob("MERLIN_IDLE_TUNE_MAX_PER_SOURCE")
        if cap and sum(1 for r in self._requests(channel) if r["operator"] and (r["experiment"] or {}).get(
                "base_sha256") == digest) >= cap:
            return []
        catalogue = mechanism_hints.load_selection(self.workspace)
        candidates = self.candidates(digest, catalogue)
        if not candidates:
            return []
        variants, knobs, authored = self._tried(channel, known, selection)
        empty = self._empty_attempts(channel, digest, selection, candidates=candidates)
        per_case = knob("MERLIN_IDLE_TUNE_MAX_CASES")
        pairs = []
        ready = {row["test_case"]: row for row in ready_cases(cohort, l2, known, digest, limit=len(cohort))}
        for order, candidate in enumerate(candidates):
            if "reference_experiment" not in candidate:
                continue
            names = candidate["test_cases"]
            if (len(names) > per_case or any(name not in ready for name in names)
                    or any((candidate["key"], name) in knobs for name in names)
                    or any(empty.get((candidate["key"], name, "scalability"), 0) >= _EMPTY_RESULT_ATTEMPTS
                           for name in names)
                    or len({ready[name]["model"] for name in names}) != 1):
                continue
            pairs.append((-3, -1, order, {"candidate": candidate, "cases": names,
                          "bundle": "selected reference experiment", "suggested": [],
                          "relevance": 3, "model": selection, "suite": "scalability"}))
        for index, bundle in enumerate(bundles):
            cases = [bundle["representative"], *bundle["siblings"]][:per_case]
            groups = _case_groups(self.workspace, cases)
            suggested = self._suggested(catalogue, bundle["representative"], known)
            for order, candidate in enumerate(candidates):
                if "reference_experiment" in candidate:
                    continue  # Its exact source-bound cases were handled above, independently of siblings.
                if candidate["kind"] == "variant":
                    untried = [c for c in cases if (candidate["sha256"], c) not in variants]
                    relevant = 2 if authored.get(candidate["name"], set()) & set(cases) else 0
                else:
                    untried = [c for c in cases if (candidate["key"], c) not in knobs]
                    relevant = 0
                if suggested & set(candidate["mechanisms"]):
                    relevant = max(relevant, 1)
                for suite, members in groups:
                    remaining = [case for case in members if case in untried
                                 and empty.get((candidate["key"], case, suite), 0) < _EMPTY_RESULT_ATTEMPTS]
                    if not remaining:
                        continue
                    pairs.append((-relevant, index, order, {
                        "candidate": candidate, "cases": remaining, "bundle": bundle["label"],
                        "suggested": sorted(str(s) for s in suggested if s), "relevance": relevant,
                        "model": selection, "suite": suite}))
        pairs.sort(key=lambda item: item[:3])
        return [job for *_, job in pairs[:limit]]

    def _suggested(self, catalogue, name: str, known: dict) -> set:
        rows = known.get(name) or ()
        if not rows or not catalogue:
            return set()
        performance = rows[0][2]
        _, limiter = _bound(performance.get("structural_roofline") or {})
        case_selection = mechanism_hints.for_case(catalogue, name, self.workspace)
        return {hint.get("id") for hint in mechanism_hints.hints(
            case_selection, dominant=_dominant_resource(performance), bound_by=limiter)}

    def in_flight(self, channel: Path) -> int:
        return sum(1 for r in self._requests(channel) if r["operator"] and r["live"])

    def submit(self, channel: Path, job: dict, names: list[str], digest: str, *, timeout: int | None = None) -> str:
        """Write one experiment request for ``job`` on ``names`` (its first cases, sized by the measurer)."""
        candidate = job["candidate"]
        if "reference_experiment" in candidate and names != candidate["test_cases"]:
            raise ValueError("reference experiments require their exact selected cases")
        groups = _case_groups(self.workspace, names)
        if len(groups) != 1 or groups[0][0] != job.get("suite", "scalability"):
            raise ValueError("idle tuning request must use its declared public case suite")
        suite = groups[0][0]
        catalogue = mechanism_hints.load_selection(self.workspace)
        public = {key: candidate[key] for key in ("kind", "key", "name", "sha256", "overrides", "settings_file",
                                                   "current") if key in candidate}
        experiment = {"schema": EXPERIMENT_SCHEMA, "base_sha256": digest, "candidate": public,
                      "test_cases": names, "mechanism": candidate["label"], "origin": "idle_tuning",
                      "bundle": job.get("bundle"), "mechanism_ids": candidate["mechanisms"],
                      "api": _api(catalogue, candidate["mechanisms"]),
                      "public_suite": suite,
                      "suggested_for_these_cases": job.get("suggested") or [],
                      "identity": list(identity(candidate, digest, job.get("model")))}
        if "reference_experiment" in candidate:
            experiment["reference_experiment"] = candidate["reference_experiment"]
        extra = {"tuning_experiment": experiment, "suite": suite}
        if suite == "core":
            extra["tiers"] = "L0,L1,L2"  # Spike development evidence; no RTL/certification request.
        target = digest
        if candidate["kind"] == "variant":
            extra["submission"] = f"{sources.VARIANTS}/{candidate['name']}"
            target = candidate["sha256"]
        else:
            extra[tuning_knobs.REQUEST_KEY] = candidate["overrides"]
        rid = idle_performance.submit(channel, names, target, origin=ORIGIN, prefix=REQUEST_PREFIX,
                                      timeout=timeout or knob("MERLIN_IDLE_TUNE_TIMEOUT_S"), extra=extra)
        scratch_retention.say(f"[idle-tune] {candidate['key']} on {job.get('bundle')}: {', '.join(names)}")
        return rid

    # -- publication ------------------------------------------------------------------------------

    def progress(self, channel: Path, digest: str, known: dict, *, cohort=None, l2=None,
                 candidates=None) -> dict:
        """Current source/model readiness, independent from retained historical rankings."""
        if cohort is None:
            from ..workload_roles import PRIMARY, workspace_roles
            cohort = [n for n, role in (workspace_roles(self.workspace) or {}).items() if role in PRIMARY]
        if l2 is None:
            l2 = {n: "pass" for n in cohort}
        ready = ready_cases(cohort, l2, known, digest, limit=len(cohort))
        if candidates is None:
            candidates = self.candidates(digest, mechanism_hints.load_selection(self.workspace))
        coverage = {}
        remaining = 0
        for case in ready:
            selection = dict(case["model"]).get("selection_sha256")
            if selection not in coverage:
                coverage[selection] = self._tried(channel, known, selection)
            variants, knobs, _ = coverage[selection]
            for candidate in candidates:
                key = candidate["sha256"] if candidate["kind"] == "variant" else candidate["key"]
                covered = variants if candidate["kind"] == "variant" else knobs
                remaining += (key, case["test_case"]) not in covered
        live = self.in_flight(channel)
        empty = self._empty_attempts(channel, digest, None, candidates=candidates)
        for selection in coverage:
            empty.update(self._empty_attempts(channel, digest, selection, candidates=candidates))
        deferred = sum(count >= _EMPTY_RESULT_ATTEMPTS and name in {case["test_case"] for case in ready}
                       for (_candidate, name, _suite), count in empty.items())
        models = [dict(model) for model in dict.fromkeys(c["model"] for c in ready)]
        return dict(source_sha256=digest, ready_baselines=len(ready), declared_candidates=len(candidates),
                    outstanding_candidate_cases=remaining, requests_in_flight=live,
                    model_domains=models[:4], retry_deferred_candidate_cases=deferred,
                    retry_note=("Missing case results remain unknown; retries resume after source/model changes."
                                if deferred else None),
                    status=("running" if live else "waiting_baseline" if not ready
                        else "retry_deferred" if remaining and deferred >= remaining
                        else "ready" if remaining else "complete_declared_work"
                        if candidates else "construction_required"),
                    **({"construction_task": construction_task(self.workspace)} if not candidates else {}))

    def refresh(self, channel: Path, digest: str, known: dict, *, cohort=None, l2=None,
                progress_known=None) -> list[dict]:
        """Publish every result with its baseline label, and the ranked view."""
        if not enabled():
            return []
        parsed = R.scan(channel, self._responses)
        variant_digests = {}
        for name in sorted({p["candidate"].get("name") for p in parsed if p["candidate"]["kind"] == "variant"}):
            try:
                variant_digests[name] = self.variant_digest(name)
            except (OSError, ValueError):
                continue  # a removed variant's results stay, marked as no longer current
        found = R.records(parsed, known, digest, variant_digests=variant_digests)
        self._history = R.retain(self._history, found, known, digest, variant_digests)
        member = {}
        for bundle in self.bundles:
            for name in [bundle["representative"], *bundle["siblings"]]:
                member.setdefault(name, bundle["label"])
        counts = {status: sum(r.get("baseline_status") == status for r in found) for status in R.STATUSES}
        document = {"schema": SCHEMA, "advisory_only": True, "promotion_authority": False,
                    "source_sha256": digest, "note": NOTE, "offer": offer(), "counts": counts,
                    "ranked_candidates": R.by_candidate(found), "ranked_cases": R.by_case(found, member),
                    "bundles": [{"label": b["label"], "representative": b["representative"],
                                 "siblings": b["siblings"][:LEDGER_SIBLINGS]} for b in self.bundles[:LEDGER_BUNDLES]],
                    "experiments": found, "comparison_history": self._history,
                    "progress": self.progress(channel, digest, known if progress_known is None else progress_known,
                                              cohort=cohort, l2=l2)}
        text = json.dumps(document, sort_keys=True, default=str)
        stable = json.dumps(_without_ages(document), sort_keys=True, default=str)
        now = time.monotonic()
        if self._published is None or self._published[0] != stable or now - self._published[1] > REWRITE_S:
            _write_atomic(self.workspace / LEDGER, json.loads(text) | {"updated_at_unix_ns": time.time_ns()})
            self._published = (stable, now)
        return found


def _without_ages(document: dict) -> dict:
    def strip(records):
        rows = []
        for record in records:
            record = {k: v for k, v in record.items() if k != "candidate_age_s"}
            if isinstance(record.get("baseline"), dict):
                record["baseline"] = {k: v for k, v in record["baseline"].items() if k != "age_s"}
            rows.append(record)
        return rows

    return {**document, "experiments": strip(document["experiments"]),
            "comparison_history": strip(document.get("comparison_history") or []), "ranked_cases": None}


# ---------------------------------------------------------------------------------------------
# Publication in the self-check brief
# ---------------------------------------------------------------------------------------------

def matching_progress(ledger, document: dict) -> dict | None:
    """Do not show another source/model epoch's queue as this reply's work."""
    if (not isinstance(ledger, dict) or not isinstance(document.get("submission_sha256"), str)
            or ledger.get("source_sha256") != document["submission_sha256"]):
        return None
    progress = ledger.get("progress")
    if not isinstance(progress, dict) or progress.get("source_sha256") != document["submission_sha256"]:
        return None
    domains = progress.get("model_domains") or ()
    for row in document.get("cases") or document.get("per_capsule") or ():
        performance = row.get("performance") if isinstance(row, dict) else None
        if isinstance(performance, dict) and domains and not any(
                all(performance.get(key) == model.get(key) for key in _MODEL_KEYS) for model in domains):
            return None
    return progress


def attach(document: dict, workspace) -> dict:
    """Add the results whose baseline still describes the case to a ``submission/`` response's
    ``performance_brief``; never fails it. A variant or knob-copy response gets nothing."""
    try:
        brief = document.get("performance_brief")
        if (not enabled() or not isinstance(brief, dict) or document.get("variant")
                or document.get("selfcheck_submission") not in (None, sources.SUBMISSION)):
            return document
        ledger = idle_performance._load(Path(workspace) / LEDGER)
        digest = document.get("submission_sha256")
        experiments = []
        if isinstance(ledger, dict):
            experiments = [r for r in ledger.get("experiments") or () if isinstance(r, dict)
                           and r.get("baseline_status", "current") != "superseded"
                           and r.get("candidate_current", True) is not False]
        observed = {row.get("capsule"): row["performance"]
                    for row in document.get("cases") or document.get("per_capsule") or ()
                    if isinstance(row, dict) and isinstance(row.get("performance"), dict)}
        experiments = [row for row in experiments if not observed.get(row.get("test_case"))
                       or R.comparable(row.get("baseline_model") or {}, observed[row["test_case"]]) is None]
        entries = [e for e in brief.get("largest_headroom") or () if isinstance(e, dict)]
        named = {e.get("test_case") for e in entries}
        for entry in entries:
            rows = [R.compact(r) for r in experiments if r.get("test_case") == entry.get("test_case")]
            rows.sort(key=lambda r: (r.get("delta_pct") is None, r.get("delta_pct") or 0.0))
            if rows:
                entry["idle_tuning_experiments"] = rows[:4]
        shown = sorted(experiments, key=lambda r: (r.get("test_case") not in named, r.get("delta_pct") is None,
                                                   r.get("delta_pct") or 0.0, str(r.get("test_case"))))
        ranked = [r for r in (ledger or {}).get("ranked_candidates") or () if isinstance(r, dict)
                  and r.get("current", True)][:5] if isinstance(ledger, dict) else []
        brief["idle_tuning"] = {"schema": SCHEMA, "advisory_only": True, "promotion_authority": False,
                                "source_sha256": digest, "count": len(experiments),
                                "experiments": [R.compact(r) for r in shown[:BRIEF_LIMIT]],
                                "ranked_candidates": [{k: r.get(k) for k in (
                                    "candidate", "geomean_ratio", "cases_compared", "faster", "slower",
                                    "numerics_changed")} for r in ranked],
                                "all": LEDGER.as_posix(), "note": NOTE, "offer": offer()}
        progress = matching_progress(ledger, document)
        if progress is not None:
            brief["idle_tuning"]["progress"] = progress
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return document
    return document
