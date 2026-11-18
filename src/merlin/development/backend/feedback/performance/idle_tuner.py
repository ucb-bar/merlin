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
from merlin.development.backend.feedback.performance.idle_tuning import (
    _EMPTY_RESULT_ATTEMPTS,
    EXPERIMENT_SCHEMA,
    LEDGER,
    LEDGER_BUNDLES,
    LEDGER_SIBLINGS,
    NOTE,
    ORIGIN,
    REQUEST_PREFIX,
    REWRITE_S,
    SCHEMA,
    _api,
    _request_cases,
    _variant_names,
    _without_ages,
    _write_atomic,
    bundles_of,
    changed_mechanisms,
    construction_task,
    enabled,
    identity,
    knob,
    offer,
    ready_cases,
)
from merlin.development.backend.feedback.performance.performance_headroom import (
    _bound,
    _dominant_resource,
    _positive,
)
from merlin.development.backend.tools.selfcheck import _case_groups

# ---- idle_tuning -------------------------------------------------------------------------------


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
        self._history = (
            (saved.get("comparison_history") or saved.get("experiments") or []) if isinstance(saved, dict) else []
        )

    # -- inputs -----------------------------------------------------------------------------------

    def _tree_digest(self, label: str) -> str:
        if self._digest is not None:
            return self._digest(label)
        from merlin.common.process.cpu_pool import AUXILIARY, cpu_priority
        from merlin.development.backend.brokers.selfcheck_dispatch import _submission_digest

        with cpu_priority(AUXILIARY):  # holds no slot while it waits for one
            return sources.source_digest(self.workspace, label, _submission_digest)

    def variant_digest(self, name: str) -> str:
        """The digest of ``variants/<name>/``, re-hashed only when a file's size or mtime changed."""
        root = self.workspace / sources.VARIANTS / name
        if root.is_symlink() or not root.is_dir():
            raise OSError(f"no variant {name}")
        fingerprint = tuple(
            sorted(
                (p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                for p in root.rglob("*")
                if p.is_file() and not p.is_symlink()
            )
        )
        memo = self._variant_memo.get(name)
        if memo is not None and memo[0] == fingerprint:
            return memo[1]
        sha = self._tree_digest(f"{sources.VARIANTS}/{name}")
        self._variant_memo[name] = (fingerprint, sha)
        return sha

    def _rollup_bundles(self, cohort, l2: dict, known: dict) -> list[dict]:
        """:func:`bundles_of` refreshed after source or completed numerical evidence changes."""
        from merlin.development.backend.feedback.performance.family_rollup import build, selfcheck_identity

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
                suite_declared = isinstance(body, dict) and "suite" in body
                if isinstance(body, dict) and "suite" not in body:
                    # The public broker protocol admits omitted suite as core.
                    # Keep its declaration state distinct from an explicit core request.
                    body = {**body, "suite": "core", "suite_declared": False}
                self._bodies[path.name] = (
                    None
                    if not isinstance(body, dict)
                    else {
                        "label": body.get("submission") or sources.SUBMISSION,
                        "sha": body.get("submission_sha256"),
                        "suite": body["suite"],
                        "suite_declared": suite_declared,
                        "cases": _request_cases(body),
                        "overrides": body.get(tuning_knobs.REQUEST_KEY),
                        "timed": (
                            body.get("suite") == "scalability"
                            or body["suite"] in {"core", "programs"}
                            and body.get("sim") == "spike"
                        ),
                        "requested_at": body.get("requested_at_unix_ns") or 0,
                        "deadline": idle_performance._deadline_s(body, path),
                        "operator": path.name.startswith("req_" + REQUEST_PREFIX),
                        "experiment": body.get("tuning_experiment")
                        if isinstance(body.get("tuning_experiment"), dict)
                        else None,
                    }
                )
            body = self._bodies[path.name]
            if body is None:
                continue
            rid = path.name[len("req_") : -len(".json")]
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
            found.append(
                {
                    "kind": "knob",
                    "key": R.knob_key({change["knob"]: change["value"]}),
                    "overrides": {change["knob"]: change["value"]},
                    "settings_file": change["settings_file"],
                    "current": change["current"],
                    "mechanisms": mechanisms,
                    "label": change["mechanism"] or f"knob {change['knob']}",
                }
            )
        for name in _variant_names(self.workspace, cap):
            try:
                sha = self.variant_digest(name)
            except (OSError, ValueError) as error:
                scratch_retention.say(f"[idle-tune] candidate digest unavailable: {type(error).__name__}: {error}")
                continue
            if sha == digest:
                continue  # identical bytes time identically
            if (digest, name, sha) not in self._mechanism_memo:
                try:
                    self._mechanism_memo[(digest, name, sha)] = changed_mechanisms(
                        self.workspace / sources.SUBMISSION, self.workspace / sources.VARIANTS / name, selection
                    )
                except OSError:
                    self._mechanism_memo[(digest, name, sha)] = []
            mechanisms = self._mechanism_memo[(digest, name, sha)]
            found.append(
                {
                    "kind": "variant",
                    "key": f"variant:{name}",
                    "name": name,
                    "sha256": sha,
                    "mechanisms": mechanisms,
                    "label": ", ".join(mechanisms) or f"author variant {name} (no newly imported catalogue mechanism)",
                }
            )
        return found[:cap] if cap else found

    def _reference_candidates(self, digest: str) -> tuple[list[dict], set[str]]:
        """Use selected requests through ordinary private knob copies and budgets."""
        from merlin.development.backend.feedback.references.experiment_guidance import pending_requests

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
                    {tuning_knobs.REQUEST_KEY: request.get("tuning_overrides")}, declared
                )
            except ValueError as error:
                scratch_retention.say(f"[idle-tune] reference candidate refused: {error}")
                continue
            if request.get("settings") != {**current, **overrides} or all(
                name in current and tuning_knobs.same_value(current[name], value) for name, value in overrides.items()
            ):
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
            found.append(
                {
                    "kind": "knob",
                    "key": R.knob_key(overrides),
                    "overrides": overrides,
                    "settings_file": declared["settings_file"],
                    "current": current,
                    "mechanisms": [],
                    "test_cases": [capsule],
                    "label": "controlled reference experiment " + request["task_id"],
                    "reference_experiment": {
                        name: request[name] for name in ("task_id", "request_id", "packet_identity", "context_identity")
                    },
                }
            )
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
                return (
                    selection is not None
                    and row is not None
                    and row["performance"].get("selection_sha256") != selection
                )

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
        counts = Counter()
        for request in self._requests(channel):
            experiment = request["experiment"] or {}
            if (
                not request["answered"]
                or experiment.get("base_sha256") != digest
                or experiment.get("public_suite") != request["suite"]
            ):
                continue  # Legacy wrong-suite refusals do not exhaust a corrected route.
            candidate = experiment.get("candidate")
            candidate = candidate if isinstance(candidate, dict) else {}
            expected = (
                list(identity(candidate, digest, selection))
                if candidate.get("kind") in {"knob", "variant"} and candidate.get("key")
                else None
            )
            if experiment.get("identity") != expected or expected is None or tuple(expected) not in identities:
                continue
            returned = {row["test_case"] for row in (parsed.get(request["rid"]) or {}).get("rows") or ()}
            for case in request["cases"]:
                if case not in returned:
                    key = (candidate["key"], case, request["suite"])
                    counts[key] += 1
        return counts

    # -- proposals --------------------------------------------------------------------------------

    def plan(
        self, channel: Path, digest: str, cohort, l2: dict, known: dict, *, selection=None, limit: int | None = None
    ) -> list[dict]:
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
        if (
            cap
            and sum(
                1
                for r in self._requests(channel)
                if r["operator"] and (r["experiment"] or {}).get("base_sha256") == digest
            )
            >= cap
        ):
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
            if (
                len(names) > per_case
                or any(name not in ready for name in names)
                or any((candidate["key"], name) in knobs for name in names)
                or any(empty[candidate["key"], name, "scalability"] >= _EMPTY_RESULT_ATTEMPTS for name in names)
                or len({ready[name]["model"] for name in names}) != 1
            ):
                continue
            pairs.append(
                (
                    -3,
                    -1,
                    order,
                    {
                        "candidate": candidate,
                        "cases": names,
                        "bundle": "selected reference experiment",
                        "suggested": [],
                        "relevance": 3,
                        "model": selection,
                        "suite": "scalability",
                    },
                )
            )
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
                    remaining = [
                        case
                        for case in members
                        if case in untried and empty[candidate["key"], case, suite] < _EMPTY_RESULT_ATTEMPTS
                    ]
                    if not remaining:
                        continue
                    pairs.append(
                        (
                            -relevant,
                            index,
                            order,
                            {
                                "candidate": candidate,
                                "cases": remaining,
                                "bundle": bundle["label"],
                                "suggested": sorted(str(s) for s in suggested if s),
                                "relevance": relevant,
                                "model": selection,
                                "suite": suite,
                            },
                        )
                    )
        pairs.sort(key=lambda item: item[:3])
        return [job for *_, job in pairs[:limit]]

    def _suggested(self, catalogue, name: str, known: dict) -> set:
        rows = known.get(name) or ()
        if not rows or not catalogue:
            return set()
        performance = rows[0][2]
        _, limiter = _bound(performance.get("structural_roofline") or {})
        case_selection = mechanism_hints.for_case(catalogue, name, self.workspace)
        return {
            hint.get("id")
            for hint in mechanism_hints.hints(
                case_selection, dominant=_dominant_resource(performance), bound_by=limiter
            )
        }

    def in_flight(self, channel: Path) -> int:
        return sum(1 for r in self._requests(channel) if r["operator"] and r["live"])

    def submit(self, channel: Path, job: dict, names: list[str], digest: str, *, timeout: int | None = None) -> str:
        """Write one experiment request for ``job`` on ``names`` (its first cases, sized by the measurer)."""
        candidate = job["candidate"]
        if "reference_experiment" in candidate and names != candidate["test_cases"]:
            raise ValueError("reference experiments require their exact selected cases")
        groups = _case_groups(self.workspace, names)
        if len(groups) != 1 or groups[0][0] != job["suite"]:
            raise ValueError("idle tuning request must use its declared public case suite")
        suite = groups[0][0]
        catalogue = mechanism_hints.load_selection(self.workspace)
        public = {
            key: candidate[key]
            for key in ("kind", "key", "name", "sha256", "overrides", "settings_file", "current")
            if key in candidate
        }
        experiment = {
            "schema": EXPERIMENT_SCHEMA,
            "base_sha256": digest,
            "candidate": public,
            "test_cases": names,
            "mechanism": candidate["label"],
            "origin": "idle_tuning",
            "bundle": job.get("bundle"),
            "mechanism_ids": candidate["mechanisms"],
            "api": _api(catalogue, candidate["mechanisms"]),
            "public_suite": suite,
            "suggested_for_these_cases": job.get("suggested") or [],
            "identity": list(identity(candidate, digest, job.get("model"))),
        }
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
        rid = idle_performance.submit(
            channel,
            names,
            target,
            origin=ORIGIN,
            prefix=REQUEST_PREFIX,
            timeout=timeout or knob("MERLIN_IDLE_TUNE_TIMEOUT_S"),
            extra=extra,
        )
        scratch_retention.say(f"[idle-tune] {candidate['key']} on {job.get('bundle')}: {', '.join(names)}")
        return rid

    # -- publication ------------------------------------------------------------------------------

    def progress(self, channel: Path, digest: str, known: dict, *, cohort=None, l2=None, candidates=None) -> dict:
        """Current source/model readiness, independent from retained historical rankings."""
        if cohort is None:
            from merlin.development.backend.session.inputs.capability_inventory import (
                PRIMARY,
                workspace_roles,
            )

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
        deferred = sum(
            count >= _EMPTY_RESULT_ATTEMPTS and name in {case["test_case"] for case in ready}
            for (_candidate, name, _suite), count in empty.items()
        )
        models = [dict(model) for model in dict.fromkeys(c["model"] for c in ready)]
        return dict(
            source_sha256=digest,
            ready_baselines=len(ready),
            declared_candidates=len(candidates),
            outstanding_candidate_cases=remaining,
            requests_in_flight=live,
            model_domains=models[:4],
            retry_deferred_candidate_cases=deferred,
            retry_note=(
                "Missing case results remain unknown; retries resume after source/model changes." if deferred else None
            ),
            status=(
                "running"
                if live
                else "waiting_baseline"
                if not ready
                else "retry_deferred"
                if remaining and deferred >= remaining
                else "ready"
                if remaining
                else "complete_declared_work"
                if candidates
                else "construction_required"
            ),
            **({"construction_task": construction_task(self.workspace)} if not candidates else {}),
        )

    def refresh(
        self, channel: Path, digest: str, known: dict, *, cohort=None, l2=None, progress_known=None
    ) -> list[dict]:
        """Publish every result with its baseline label, and the ranked view."""
        if not enabled():
            return []
        parsed = R.scan(channel, self._responses)
        variant_digests = {}
        for name in sorted({p["candidate"].get("name") for p in parsed if p["candidate"]["kind"] == "variant"}):
            try:
                variant_digests[name] = self.variant_digest(name)
            except (OSError, ValueError) as error:
                scratch_retention.say(f"[idle-tune] variant freshness unavailable: {type(error).__name__}: {error}")
                continue  # a removed variant's results stay, marked as no longer current
        found = R.records(parsed, known, digest, variant_digests=variant_digests)
        self._history = R.retain(self._history, found, known, digest, variant_digests)
        member = {}
        for bundle in self.bundles:
            for name in [bundle["representative"], *bundle["siblings"]]:
                member.setdefault(name, bundle["label"])
        counts = {status: sum(r.get("baseline_status") == status for r in found) for status in R.STATUSES}
        document = {
            "schema": SCHEMA,
            "advisory_only": True,
            "promotion_authority": False,
            "source_sha256": digest,
            "note": NOTE,
            "offer": offer(),
            "counts": counts,
            "ranked_candidates": R.by_candidate(found),
            "ranked_cases": R.by_case(found, member),
            "bundles": [
                {
                    "label": b["label"],
                    "representative": b["representative"],
                    "siblings": b["siblings"][:LEDGER_SIBLINGS],
                }
                for b in self.bundles[:LEDGER_BUNDLES]
            ],
            "experiments": found,
            "comparison_history": self._history,
            "progress": self.progress(
                channel, digest, known if progress_known is None else progress_known, cohort=cohort, l2=l2
            ),
        }
        text = json.dumps(document, sort_keys=True, default=str)
        stable = json.dumps(_without_ages(document), sort_keys=True, default=str)
        now = time.monotonic()
        if self._published is None or self._published[0] != stable or now - self._published[1] > REWRITE_S:
            _write_atomic(self.workspace / LEDGER, json.loads(text) | {"updated_at_unix_ns": time.time_ns()})
            self._published = (stable, now)
        return found
