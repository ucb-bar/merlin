"""The whole-model objective a phase-2 ``whole_model_measured`` loop optimises.

TWO MACHINES, TWO JOBS, NEVER ONE NUMBER.  The SCREEN (a board: a whole model in minutes) measures
EVERY candidate the loop produces, so an agent sees the consequence of an edit while it still
remembers making it.  The CERTIFIER (the elaborated-RTL emulator: hours per model) runs only when a
candidate becomes the new best on the screen, and a best that fails there is RETRACTED.  Each
machine has its own reference arm -- the vendor library measured on THAT machine -- and every
comparison is made against the reference on the same device.  A cycle count from one machine is
never compared with one from the other.

A best must also hold up SOLO: a promoted best is re-measured alone (``repeats_on_best``), and a best
whose solo repeat does not beat the previous best's solo result by more than the NOISE MARGIN is
demoted to a tie.  The noise margin is the median batched-vs-solo repeat spread measured in THIS
store, floored at :data:`NOISE_FLOOR` -- a candidate crowned on a margin narrower than the noise the
board itself produces on a rerun is not a measured improvement, it is the spread.

``measure`` is the loop's per-candidate hook.  It never blocks on a simulator: it requests a
measurement of the exact bytes and returns what is known about them now -- a result, or ``PENDING``
-- together with the feedback the agent acts on.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V
from merlin.perf.whole_model_capsules import resolve_capsules

from . import batch as BATCH
from . import feedback as F
from . import gates as G
from . import jobs as J
from . import transfer as TC
from .batch import PROMOTED_REPEAT_PRIORITY
from .identity import epoch_of, package_digest, write_json_atomic
from .jobs import ROLE_CANDIDATE, TIMING_PENDING
from .service import MeasurementService

SCHEMA = "merlin_whole_model_objective_v1"

CERT_PENDING, CERTIFIED, RETRACTED, UNCERTIFIED = "pending", "certified", "retracted", "uncertified"
#: A best whose certification would not finish in time: projected from its own board cycles and the
#: certifier reference's measured rate, it is never started, and the projection is recorded.
DEFERRED_INFEASIBLE = "deferred_infeasible"
#: The longest certification this loop starts (the certifier's own timeout caps it further).
CERT_MAX_PROJECTED_SECONDS = 6 * 3600


def _load(path: Path | None) -> dict[str, Any] | None:
    """A reference result, or None while it has not landed (a slow machine's reference may lag)."""

    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prohibited_text(result: Mapping[str, Any] | None) -> str | None:
    """The lead for a candidate refused because its linked program emits a prohibited instruction:
    the refusal line, then where each instruction sits -- a group's own kernel, or the program's code
    (the library calls for the groups routed to it, and host code). Data only: where, never how."""
    report = (result or {}).get("isa_prohibited")
    if not isinstance(report, Mapping):
        return None
    lines = [str((result or {}).get("refusal") or "isa_prohibited")]
    for where, count in sorted((report.get("summary") or {}).items()):
        lines.append(f"  {where}: {count} instruction(s)")
    roles = ", ".join(str(r) for r in report.get("roles") or ())
    lines.append(
        f"The rule: the linked program may issue no instruction with role {roles or '?'}, in any group -- the "
        "library's groups and host code included. Checked on the whole program before any run; nothing was run."
    )
    return "\n".join(lines)


def _screen_basis(screen: Any) -> dict[str, Any]:
    """HOW the screen times a candidate: a whole-program run, or a cell's sum of its own groups."""
    machine = (getattr(screen, "machine", None) or {}) if screen is not None else {}
    if machine.get("kind") == "cell":
        return {"basis": "cell_sum", "machine": (machine.get("timing") or {}).get("registry_machine")}
    timing = machine.get("timing") or machine
    return {
        "basis": "whole_window",
        "machine": timing.get("registry_name") or timing.get("hw_config") or timing.get("kind"),
    }


def _board_status(screen: Any) -> dict[str, Any] | None:
    """The screen store's board outage in progress, if any: since when, and when it is tried again."""
    root = getattr(screen, "root", None)
    outage = BATCH.board_outage(root) if root is not None else None
    if not outage:
        return None
    return {
        "state": "unavailable",
        "since": outage.get("opened_at"),
        "failures": len(outage.get("failures") or []),
        "next_try_epoch": outage.get("retry_after_epoch"),
        "note": "board measurements are deferred, never refused; the functional-model grade keeps running",
    }


class WholeModelObjective:
    """Orchestrates the screen and the certifier over one run's candidates."""

    def __init__(
        self,
        *,
        screen: MeasurementService,
        screen_reference: Path | None,
        certifier: MeasurementService | None = None,
        certifier_reference: Path | None = None,
        repeats_on_best: int = 2,
        primary_name: str = "",
        held_out: Mapping[str, tuple[MeasurementService, Path | None]] | None = None,
    ) -> None:
        if repeats_on_best < 1:
            raise ValueError("a best is measured at least once")
        self.screen = screen
        self.certifier = certifier
        self.screen_reference_path = Path(screen_reference) if screen_reference else None
        self.certifier_reference_path = Path(certifier_reference) if certifier_reference else None
        self.screen_reference = _load(self.screen_reference_path)
        self.certifier_reference = _load(self.certifier_reference_path)
        self.repeats_on_best = int(repeats_on_best)
        # THE TRANSFER CHECK'S OWN INPUTS -- each held-out model's own screen service (it measures the
        # SAME candidate bytes against a DIFFERENT model capsule) and its own reference. Both optional:
        # a launch that configures none of this gets no transfer check and nothing else changes, which
        # is also how every existing config (with no `held_out` section) keeps working unmodified.
        self.primary_name = str(primary_name)
        self.held_out: dict[str, tuple[MeasurementService, Any | None]] = dict(held_out or {})
        self.held_out_references = {name: _load(reference) for name, (_svc, reference) in self.held_out.items()}
        self._lock = threading.Lock()
        #: The launch's objective config.  Set by :func:`.config.from_config`.
        self.config: dict[str, Any] | None = None
        #: Labels every summary carries (for example an adopted freeze's freeze_kind / hidden_grade), so a
        #: report copied out of this run still says what its starting candidate was and was not.
        self.labels: dict[str, Any] = {}
        self._promoted: dict[str, dict[str, Any]] = {}
        #: A best whose solo repeat did not beat the previous best's solo result: digest -> that previous best.
        self._demoted: dict[str, str] = {}
        #: When the current authoring session began (set by the session loop), for the stagnation signal.
        self.session_started_epoch: float | None = None
        #: The run's OOT history (:class:`.ledger.OotLedger`): a commit per candidate, a tag per measurement.
        self.ledger: Any = None
        #: The launcher's heartbeat (:class:`.liveness.Heartbeat`), set by the process that runs the sessions;
        #: a reader that opens the objective (``status``) leaves it None and never writes one.
        self.heartbeat: Any = None
        # THE SCREEN ARMS ITS OWN COVERAGE GATE AT REQUEST TIME: a gate armed only on poll left the first
        # request after every relaunch without one, and a candidate that handed work back reached the board.
        if hasattr(screen, "coverage_gate_provider"):
            screen.coverage_gate_provider = self.coverage_gate_document

    def mark_session(self, epoch: float | None = None) -> None:
        """Record that an authoring session begins now: the stagnation signal compares against it."""
        import time

        self.session_started_epoch = float(epoch if epoch is not None else time.time())
        if self.heartbeat is not None:
            self.heartbeat.tick("session started", force=True)

    def stagnation(self, *, top: int = 5) -> dict[str, Any] | None:
        """Whether this session's correct measurements moved the best's top gap-holding groups (a
        diagnostic: which gaps moved, never how).  ``None`` before a session is marked or a best exists."""
        if self.session_started_epoch is None:
            return None
        best = self._screen_best_unretracted()
        if not best or not best.get("verdict"):
            return None
        holders = [str(h["group"]) for h in (self.feedback(best).get("gap_holders") or [])[:top]]
        if not holders:
            return None
        history = []
        for job in self.screen.jobs():
            if int(job.get("replicate") or 0):
                continue
            found = self.screen.result_by_key(job.get("job_key") or job["package_sha256"]) or {}
            if found.get("timing_status") != V.TIMING_MEASURED or not self.eligible(found):
                continue
            epoch = _epoch(found.get("finished_at"))
            if epoch is None:
                continue
            counts = {
                str(r["group"]): r.get("cycles")
                for r in V.group_table(found.get("verdict") or {})
                if r.get("correct", True)
            }
            history.append((epoch, counts))
        return F.stagnation(history, holders, since_epoch=self.session_started_epoch, noise=self.noise_margin())

    # ---- what the agent reads about one result
    def feedback(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """:func:`.feedback.compare` of ``result`` against the screen's reference, WITH the model's
        fact-derived rooflines by default (:meth:`rooflines`): a whole-model result carries no diagnostics
        of its own, and an agent that sees only the vendor's cycles optimizes where the vendor is weak.
        When the rooflines cannot be derived the feedback says why, never nothing."""
        diagnosed = dict(result)
        # A result that carries its own diagnostics (a cell's) is read as it is; only the others need the model's.
        rooflines = self.rooflines() if not diagnosed.get("diagnostics") else {}
        if not diagnosed.get("diagnostics") and rooflines.get("per_group"):
            diagnosed["diagnostics"] = {
                "schema": "merlin_whole_model_diagnostics_v1",
                "per_group": {g: {"roofline": r} for g, r in rooflines["per_group"].items()},
                "source": rooflines.get("source"),
            }
        document = F.compare(diagnosed, self.screen_reference)
        if not diagnosed.get("diagnostics") and rooflines.get("why"):
            document["roofline_unavailable"] = rooflines["why"]
        return document

    def rooflines(self) -> dict[str, Any]:
        """Each group's derived roofline for the screen's model (:mod:`.roofline`), computed once per model
        and machine facts and kept in the store (``rooflines.json``): ``{"per_group", "source"}``, or
        ``{"why"}`` when it cannot be derived (no model capsule, no facts, an opt-out by the config's
        ``roofline_feedback: false``)."""
        cached = getattr(self, "_rooflines", None)
        if cached is not None:
            return cached
        self._rooflines = self._derive_rooflines()
        return self._rooflines

    def _derive_rooflines(self) -> dict[str, Any]:
        import hashlib

        from . import roofline as ROOF

        config = self.config or {}
        if config.get("roofline_feedback") is False:
            return {"why": "the objective config turned roofline feedback off (roofline_feedback: false)"}
        capsule = ((config.get("screen") or {}).get("build_options") or {}).get("model_capsule")
        target = str(getattr(self.screen, "target", "") or "")
        if not capsule or not Path(str(capsule)).is_dir() or not target:
            return {"why": "the screen names no model capsule to derive each group's roofline from"}
        try:
            from merlin.perf.whole_model_capsule import load_model_capsule

            interface = Path(load_model_capsule(capsule).interface)
            machine = ROOF.roofline_machine(target)
            key = hashlib.sha256(
                json.dumps(
                    {
                        "interface": hashlib.sha256(interface.read_bytes()).hexdigest(),
                        "target": target,
                        "machine": machine,
                    },
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            root = getattr(self.screen, "root", None)
            kept = Path(root) / "rooflines.json" if root else None
            stored = _load(kept) if kept is not None else None
            if stored and stored.get("key") == key:
                return stored
            shapes = ROOF.group_shapes(capsule, target=target)
            document = {
                "key": key,
                "source": {"model_capsule": str(capsule), "machine": machine.get("provenance"), "schema": ROOF.SCHEMA},
                "per_group": {g: ROOF.group_roofline(shape, machine) for g, shape in shapes.items()},
            }
            if kept is not None:
                with contextlib.suppress(OSError):
                    write_json_atomic(kept, document)
            return document
        except Exception as exc:  # noqa: BLE001 -- said in the feedback, never a silent absence
            return {"why": f"the rooflines could not be derived: {type(exc).__name__}: {str(exc)[:300]}"}

    # ---- the loop's hook
    def measure(
        self,
        candidate: Path,
        sentinel: Any = None,
        *,
        label: str = "",
        seed: bool = False,
        attribution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request the exact bytes on the screen and return what is known about them now.

        ``seed`` marks the session's starting bytes: they are run even when the capsule screen fails,
        so the first per-group table exists to attribute from.  ``attribution`` records who asked
        (:func:`.jobs.merge_attribution`), in the store and in the run's history."""
        job = self._request(candidate, label=label or "candidate", seed=seed, attribution=attribution)
        self.poll()
        return self.document(job["package_sha256"])

    def _request(
        self, candidate: Path, *, label: str, seed: bool = False, attribution: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """THE ONE WAY bytes enter the store: requested with who asked, and committed to the run's history.
        A second path that created the job without its attribution made the agent's bytes read as the
        harness's -- attributable, so a refused round's bytes could have become the best."""
        job = self.screen.request(
            Path(candidate),
            label=label,
            role=ROLE_CANDIDATE,
            screen_exempt=seed,
            attribution=attribution,
        )
        if self.ledger is not None:
            # The HARNESS commits the exact bytes requested (their own digest, even when aliased).
            self.ledger.record(
                Path(candidate),
                str(job.get("aliased_from") or job["package_sha256"]),
                label=label,
                attribution=self.screen.attribution(job["package_sha256"]) if attribution is not None else None,
            )
        return job

    def attribute(self, digest: str, change: Mapping[str, Any]) -> dict[str, Any]:
        """Record who earned ``digest``'s bytes, in the store and in the run's history."""
        record = self.screen.attribute(digest, change)
        if self.ledger is not None:
            self.ledger.attribute(digest, record)
        return record

    def latest_measured(self, *, exclude: str | None = None) -> dict[str, Any] | None:
        """The newest finished measurement (replicate 0) with a verdict: its digest, status and failures."""
        failing_summary = J.failing_summary
        newest: tuple[str, dict[str, Any]] | None = None
        for job in self.screen.jobs():
            if int(job.get("replicate") or 0) or job.get("package_sha256") == exclude:
                continue
            result = self.screen.result_by_key(job.get("job_key") or job["package_sha256"])
            if not result or not (result.get("verdict") or result.get("screen_failed") or result.get("isa_prohibited")):
                continue
            stamp = str(result.get("finished_at") or "")
            if newest is None or stamp > newest[0]:
                newest = (stamp, result)
        if newest is None:
            return None
        result = newest[1]
        verdict = result.get("verdict") or {}
        feedback = self.feedback(result) if result.get("verdict") else None
        return {
            "package_sha256": result.get("package_sha256"),
            "finished_at": result.get("finished_at"),
            "timing_status": result.get("timing_status"),
            "screen_failed": bool(result.get("screen_failed")),
            "whole_window_cycles": verdict.get("whole_window_cycles"),
            "failing": failing_summary(result),
            "feedback": feedback,
            "feedback_text": F.render(feedback) if feedback else result.get("refusal"),
        }

    def document(self, digest: str) -> dict[str, Any]:
        screen = self.screen.measurement_for(digest)
        feedback = self.feedback(screen) if screen.get("verdict") else None
        coverage = self.coverage(screen)
        if feedback is not None and coverage is not None and coverage.get("coverage_regression"):
            feedback["coverage_regression"] = coverage["coverage_regression"]
        if feedback is not None and self.held_out:
            feedback["transfer_check"] = self.transfer_check(digest, screen)
        status = screen.get("timing_status")
        elf = (screen.get("build") or {}).get("elf_sha256")
        if feedback is not None and elf:
            # AN EDIT THAT CHANGED NO PROGRAM. Different package bytes can link the very same ELF (a
            # docs change, an unreached path); the agent should learn that from the measurement rather
            # than read a repeated cycle count as a lever with no effect.
            feedback["same_program_as"] = sorted(
                job["package_sha256"]
                for job in self.screen.jobs()
                if job["package_sha256"] != digest
                and not job.get("replicate")
                and ((self.screen.result(job["package_sha256"]) or {}).get("build") or {}).get("elf_sha256") == elf
            )
        early = screen.get("correctness_first")
        latest = self.latest_measured()
        feedback_text = F.render(feedback) if feedback else F.render_early(early)
        stagnant = self.stagnation()
        if stagnant is not None:
            if feedback is not None:
                feedback["stagnation"] = stagnant
            line = F.render_stagnation(stagnant)
            if line:
                feedback_text = (feedback_text + "\n" if feedback_text else "") + line
        ineligible = self.ineligible_text(screen)
        if ineligible:
            feedback_text = ineligible + ("\n" + feedback_text if feedback_text else "")
        prohibited = prohibited_text(screen)
        if prohibited:
            feedback_text = prohibited + ("\n" + feedback_text if feedback_text else "")
        if screen.get("notice") and status == TIMING_PENDING:
            # A measurement the HOST lost and re-queued: said as such, never as a refusal of the bytes.
            feedback_text = str(screen["notice"]) + ("\n" + feedback_text if feedback_text else "")
        from_digest = None
        if feedback is None and latest is not None and latest.get("package_sha256") != digest:
            # THE CURRENT BYTES HAVE NO RESULT YET, so the agent reads the nearest measured one's table,
            # labelled with whose it is -- never nothing (an agent that saw no table for an hour
            # concluded the inverse of what the table said).
            from_digest = latest.get("package_sha256")
            feedback_text = "\n".join(
                part
                for part in (
                    feedback_text,
                    f"(latest measured result, from digest {str(from_digest)[:12]}; your current bytes are not "
                    "measured yet)",
                    latest.get("feedback_text"),
                )
                if part
            )
        return {
            "schema": SCHEMA,
            "from_digest": from_digest,
            "latest_measured": latest,
            "package_sha256": digest,
            "timing_status": status if status != TIMING_PENDING else TIMING_PENDING,
            "objective_cycles": screen.get("objective_cycles"),
            "screen": _compact(screen),
            "feedback": feedback,
            # BEFORE THE TIMING LANDS the agent still reads correctness: the capsule screen (seconds) and
            # the functional model's whole-model grade (minutes), failing groups first.
            "feedback_text": feedback_text,
            "correctness_first": early,
            "certification": copy.deepcopy(self._promoted.get(digest)),
            "objective": self.summary(),
        }

    # ---- correctness, fast
    def correctness(
        self, candidate: Path, *, label: str = "correctness check", attribution: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Request the exact bytes (the same job a measurement uses) and return only what is known about
        their CORRECTNESS now -- the capsule screen, then the functional model's per-group grade."""
        job = self._request(Path(candidate), label=label, attribution=attribution)
        self.poll()
        digest = job["package_sha256"]
        screen = self.screen.measurement_for(digest)
        early = screen.get("correctness_first") or self.screen.early_correctness(digest)
        final = screen.get("verdict") if screen.get("timing_status") != TIMING_PENDING else None
        text = F.render_early(early)
        if final:
            feedback = self.feedback(screen)
            text = F.render(feedback)
        text = prohibited_text(screen) or text
        document = self.compact(digest)
        document["schema"] = "merlin_whole_model_correctness_v1"
        regression = document.get("regression")
        if regression:
            text = regression + "\n" + (text or "")
        document["text"] = _clip(
            text
            or "no correctness result yet: the build and the functional-model run take a few minutes; "
            "call wait-for-result",
            COMPACT_TEXT,
        )
        return document

    def state_of(self, digest: str) -> tuple[Any, ...]:
        """What a waiter watches: the job's state, its timing status, and which correctness files exist."""
        screen = self.screen.measurement_for(digest)
        early = screen.get("correctness_first") or {}
        local = early.get("whole_model_functional") or {}
        return (
            screen.get("job_state"),
            screen.get("timing_status"),
            bool(early.get("capsule_screen")) or bool(screen.get("screen")),
            local.get("status"),
            bool(local.get("attribution")),
        )

    def wait(self, candidate: Path | None, *, digest: str | None = None, timeout_s: float = 540) -> dict[str, Any]:
        """Block until what is known about these bytes CHANGES (or ``timeout_s``), then return it compact.

        Replaces the agent's own sleep loops, which were a quarter of its time and a third of its cost.
        ``candidate`` names the current bytes; ``digest`` names any requested digest instead."""
        import time as _time

        if digest is None:
            digest = package_digest(Path(candidate))
        before = self.state_of(digest)
        deadline = _time.monotonic() + max(1.0, min(float(timeout_s), WAIT_MAX_SECONDS))
        changed = False
        while _time.monotonic() < deadline:
            self.poll()
            if self.state_of(digest) != before:
                changed = True
                break
            _time.sleep(WAIT_POLL_SECONDS)
        document = self.compact(digest)
        document["changed"] = changed
        return document

    def compact(self, digest: str) -> dict[str, Any]:
        """Everything an agent needs about one digest in a few kilobytes: its state, its failing groups
        (or failing capsules), the text table, the bar, the best, and the latest measured result."""
        failing_summary = J.failing_summary
        screen = self.screen.measurement_for(digest)
        document = self.document(digest)
        early = screen.get("correctness_first") or {}
        local = early.get("whole_model_functional") or {}
        failing = failing_summary(screen)
        if not failing and local.get("groups"):
            failing = failing_summary(
                {"verdict": {"groups": local.get("groups")}, "build": {"groups": local.get("routes")}}
            )
        summary = document.get("objective") or {}
        latest = document.get("latest_measured") or {}
        best = summary.get("best") or {}
        regression = None
        if best.get("package_sha256") and failing and best.get("package_sha256") != digest:
            # A CORRECT BEST EXISTS, so any failing group is a regression against it: said first, by group.
            groups = [f"g{row['group']}" for row in failing if row.get("group")]
            regression = (
                f"REGRESSION: {', '.join(groups[:20])} now fail(s); the best ({str(best['package_sha256'])[:12]}) is "
                "correct on every group. These bytes get no board time until they are correct again."
            )
        return {
            "package_sha256": digest,
            "regression": regression,
            "same_program_as": screen.get("same_program_as"),
            "job_state": screen.get("job_state") or ("done" if screen.get("verdict") else None),
            "timing_status": screen.get("timing_status"),
            "objective_cycles": screen.get("objective_cycles"),
            "capsule_screen": (screen.get("screen") or {}).get("summary")
            and {
                k: (screen.get("screen") or {}).get("summary", {}).get(k) for k in ("n_passed", "n_capsules", "failing")
            },
            "functional_model": local.get("status"),
            "package_authored": {
                k: (self.coverage(screen) or {}).get(k)
                for k in ("groups_answered", "groups_total", "priced_share", "eligible", "coverage_regression")
            },
            "by_kind": [
                {k: row.get(k) for k in ("kind", "groups", "cycles", "gather_cycles", "reference_cycles", "ratio")}
                for row in ((document.get("feedback") or {}).get("by_kind") or [])
            ],
            "failing": failing[:COMPACT_FAILING],
            "failing_count": len(failing),
            "text": _clip(
                ((regression + "\n") if regression else "") + (document.get("feedback_text") or ""), COMPACT_TEXT
            ),
            "bar_cycles": (summary.get("bar") or {}).get("screen_whole_window_cycles"),
            "best": {
                k: (summary.get("best") or {}).get(k)
                for k in ("package_sha256", "screen_whole_window_cycles", "package_authored")
            }
            if summary.get("best")
            else None,
            "latest_measured": {
                "package_sha256": latest.get("package_sha256"),
                "timing_status": latest.get("timing_status"),
                "whole_window_cycles": latest.get("whole_window_cycles"),
                "failing": (latest.get("failing") or [])[:8],
            }
            if latest
            else None,
            "from_digest": document.get("from_digest"),
        }

    def check_capsules(
        self, candidate: Path, names: str, *, out: Path, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Grade ``candidate`` on named capsules, or on the capsules of a failing group's op form."""
        spec = dict(self.screen.pre_measure_check or {})
        if not spec.get("argv"):
            raise ValueError("this run declares no capsule check")
        if timeout_s is not None:
            spec["timeout_seconds"] = min(float(spec.get("timeout_seconds") or timeout_s), float(timeout_s))
        catalog = Path(str(spec["catalog"])) if spec.get("catalog") else None
        chosen, mapping = resolve_capsules(names, self._model_routes(), catalog, self._group_forms())
        if not chosen:
            raise ValueError(f"no capsule matched {names!r} ({mapping})")
        out.parent.mkdir(parents=True, exist_ok=True)
        check = G.run_capsule_check(spec, Path(candidate), out, cwd=out.parent, capsules=",".join(chosen))
        report = check.get("report")
        document = F._read(report) if report else None
        return {
            "schema": "merlin_whole_model_capsule_check_v1",
            "capsules": chosen,
            "resolved_from": mapping,
            "passed": check.get("passed"),
            "summary": check.get("summary"),
            "wall_seconds": check.get("wall_seconds"),
            "text": "\n".join(F.render_capsule_report(document))
            if isinstance(document, Mapping)
            else check.get("output_tail"),
        }

    def _group_forms(self) -> dict[str, dict[str, Any]]:
        """Each group's op-form, as the launch declared it (``group_forms`` in the objective config,
        derived once from the one grouping); empty (op-only resolution, said so in the mapping)."""
        forms = (self.config or {}).get("group_forms") or {}
        return {str(k): dict(v) for k, v in forms.items() if isinstance(v, Mapping)}

    def _model_routes(self) -> dict[str, Mapping[str, Any]]:
        """Each group's op, from any build in this store (a group's op is the model's, not the package's)."""
        for job in reversed(self.screen.jobs()):
            key = job.get("job_key") or job.get("package_sha256")
            for name in ("result.json", "local_verdict.json"):
                document = _load(self.screen.root / str(key) / name)
                rows = ((document or {}).get("build") or {}).get("groups") or (document or {}).get("routes")
                if rows:
                    return {str(r.get("group")): r for r in rows if isinstance(r, Mapping)}
        return {}

    # ---- promotion
    def poll(self) -> None:
        """Advance both machines; promote a new screen best to repeats and to the certifier."""
        with self._lock:
            if self.screen_reference is None:
                self.screen_reference = _load(self.screen_reference_path)
            if self.certifier_reference is None:
                self.certifier_reference = _load(self.certifier_reference_path)
            self._arm_coverage_gate()
            self.screen.poll()
            if self.certifier is not None:
                self.certifier.poll()
            best = self._screen_best_unretracted()
            if best is not None:
                digest = best["package_sha256"]
                record = self._promoted.get(digest)
                if record is None or (
                    record.get("state") == UNCERTIFIED and "superseded" in str(record.get("reason") or "")
                ):
                    self._promote(digest)  # a best restored after a demotion is certified again
                if self._confirmed(digest):
                    # Only a best that has held up solo takes the certifier from the one before it.
                    self.govern_certifications(digest)
            for digest, record in self._promoted.items():
                self._refresh(digest, record)
            self._record_staleness_on_disk()
            if self.ledger is not None:
                self.ledger.sync(self)
            if self.heartbeat is not None:
                self.heartbeat.tick("poll")

    def _record_staleness_on_disk(self) -> None:
        """Every certification verdict gets its staleness record, whether or not THIS process promoted
        it: a relaunch keeps the store but not the in-memory promotions, and a verdict that lands after
        one would otherwise never say how far behind its candidate was."""
        if self.certifier is None:
            return
        for job in self.certifier.jobs():
            digest = job.get("package_sha256")
            if job.get("state") != "done" or job.get("replicate") or job.get("role") == "reference":
                continue
            if (self.certifier.root / str(digest) / "staleness.json").is_file():
                continue
            result = self.certifier.result(digest)
            if result is not None and not result.get("superseded"):
                self._staleness(digest, result)

    def certification_projection(self, digest: str) -> dict[str, Any]:
        """How long certifying ``digest`` would take, and whether that fits.

        Projected, never guessed: the candidate's own board cycles times the certifier reference's
        measured rate (its certification wall time over the SAME reference's board cycles). The limit
        is the smaller of :data:`CERT_MAX_PROJECTED_SECONDS` (or the config's
        ``cert_max_projected_seconds``) and the certifier's timeout. A projection that cannot be made
        is NOT feasible: a certification nobody can bound is not started on hope."""
        cycles = (self.screen.result(digest) or {}).get("objective_cycles")
        reference_cycles = ((self.screen_reference or {}).get("verdict") or {}).get("whole_window_cycles")
        reference_seconds = self._expected_certification_seconds()
        limits = [float((self.config or {}).get("cert_max_projected_seconds") or CERT_MAX_PROJECTED_SECONDS)]
        if self.certifier is not None and getattr(self.certifier, "timeout_seconds", None):
            limits.append(float(self.certifier.timeout_seconds))
        limit = min(limits)
        document: dict[str, Any] = {
            "board_cycles": cycles,
            "reference_board_cycles": reference_cycles,
            "reference_certification_seconds": reference_seconds,
            "limit_seconds": limit,
        }
        if not (isinstance(cycles, int) and reference_cycles and reference_seconds):
            document.update(projected_seconds=None, feasible=False, reason="UNKNOWN: no measured rate to project from")
            return document
        projected = float(cycles) * float(reference_seconds) / float(reference_cycles)
        document.update(projected_seconds=round(projected), feasible=projected <= limit)
        document["reason"] = (
            f"projected {projected / 3600:.1f} h (board cycles {cycles} at the reference's "
            f"{reference_seconds:.0f} s per {reference_cycles} cycles) against a limit of {limit / 3600:.1f} h"
        )
        return document

    def _promote(self, digest: str) -> None:
        snapshot = self.screen.root / digest / "package"
        record: dict[str, Any] = {"package_sha256": digest, "state": CERT_PENDING, "repeats": []}
        for index in range(1, self.repeats_on_best):
            job = self.screen.request(
                snapshot,
                label=f"repeat {index} of promoted best (solo)",
                replicate=index,
                solo=True,
                priority=PROMOTED_REPEAT_PRIORITY,
            )
            record["repeats"].append(job.get("job_key"))
        projection = self.certification_projection(digest) if self.certifier is not None else None
        if projection is not None and not projection["feasible"]:
            record.update(state=DEFERRED_INFEASIBLE, reason=projection["reason"], projection=projection)
        elif self.certifier is not None:
            job = self.certifier.request(snapshot, label="certify promoted best")
            record["certifier_job"] = job.get("package_sha256")
        else:
            record["state"] = UNCERTIFIED
            record["reason"] = "no certifying machine is configured"
        self._promoted[digest] = record

    def govern_certifications(self, best_digest: str, *, now: float | None = None) -> list[dict[str, Any]]:
        """Keep the certifier on the CURRENT best. A best is superseded every quarter hour at times and each
        certification takes hours, so without this the emulators pile up one per superseded best:

        * a QUEUED certification of a superseded best is dropped, recorded ``superseded``;
        * SLOT A: the EARLIEST-started running certification always runs to completion, superseded or
          not -- a best is superseded faster than a certification finishes, so without this none ever
          finishes. When it finishes, the next-earliest becomes slot A;
        * any other RUNNING one (slot B) continues once past CERT_KEEP_FRACTION of its expected run (the
          certifier reference's own wall time on the same machine), else it is stopped by exact PID and
          recorded; with no expected time known it continues;
        * the current best's certification is never touched; it runs as soon as a slot is free.

        Concurrency itself is the certifier's ``slots`` (the launch config sets 2)."""
        import time as _time

        if self.certifier is None:
            return []
        now = _time.time() if now is None else now
        expected = self._expected_certification_seconds()
        actions = []
        running = sorted(
            (j for j in self.certifier.jobs() if j.get("state") == "running" and j.get("role") != "reference"),
            key=lambda j: _epoch(j.get("dispatched_at") or j.get("started_at")) or now,
        )
        for job in list(running):
            if self.eligible(self.screen.result(job.get("package_sha256"))):
                continue
            done = self.certifier.supersede(
                _key_of(job),
                reason="ineligible_coverage_regression: its package hands work back to the library, so its "
                "cycles are not the compiler's",
                stop_running=True,
            )
            running.remove(job)
            if done is not None:
                actions.append(
                    {
                        "package_sha256": job.get("package_sha256"),
                        "state": done.get("state"),
                        "stopped": done.get("stopped_pids"),
                        "reason": "ineligible_coverage_regression",
                    }
                )
                if job.get("package_sha256") in self._promoted:
                    self._promoted[job["package_sha256"]].update(
                        state=UNCERTIFIED, reason="ineligible_coverage_regression"
                    )
        slot_a = _key_of(running[0]) if running else None
        for job in self.certifier.jobs():
            digest = job.get("package_sha256")
            if digest == best_digest or job.get("state") not in ("pending", "running") or job.get("replicate"):
                continue
            if job.get("role") == "reference":
                continue
            reason = f"a newer best ({best_digest[:12]}) superseded {str(digest)[:12]} before its certification"
            if job.get("state") == "pending":
                done = self.certifier.supersede(_key_of(job), reason=reason + " started", stop_running=False)
            elif _key_of(job) == slot_a:
                continue  # slot A: the earliest-started certification always finishes
            else:
                started = _epoch(job.get("dispatched_at") or job.get("started_at"))
                fraction = (now - started) / expected if expected and started else None
                if fraction is not None and fraction >= CERT_KEEP_FRACTION:
                    continue  # past half its expected run: let it finish
                if fraction is None:
                    continue  # nothing to judge progress by: keep it
                done = self.certifier.supersede(
                    _key_of(job),
                    reason=f"{reason} was {fraction:.0%} through its expected {expected:.0f} s",
                    stop_running=True,
                )
            if done is not None:
                actions.append(
                    {"package_sha256": digest, "state": done.get("state"), "stopped": done.get("stopped_pids")}
                )
                if digest in self._promoted:
                    self._promoted[digest].update(state=UNCERTIFIED, reason="certification superseded by a newer best")
        return actions

    def _staleness(self, digest: str, result: Mapping[str, Any]) -> dict[str, Any]:
        """How far the certified candidate had fallen behind when its verdict landed: its own and the
        then-current best's digest and board cycles. Written beside the verdict, never into it."""
        best = self._screen_best_unretracted() or {}
        mine = self.screen.result(digest) or {}
        document = {
            "schema": "merlin_whole_model_certification_staleness_v1",
            "certified": {"package_sha256": digest, "screen_objective_cycles": mine.get("objective_cycles")},
            "best_at_finish": {
                "package_sha256": best.get("package_sha256"),
                "screen_objective_cycles": best.get("objective_cycles"),
            },
            "certifier_finished_at": result.get("finished_at"),
            "certified_is_current_best": best.get("package_sha256") == digest,
        }
        with contextlib.suppress(OSError):
            write_json_atomic(self.certifier.root / digest / "staleness.json", document)
        return document

    def _expected_certification_seconds(self) -> float | None:
        """The certifier reference's own wall time: how long one whole-model certification takes there."""
        if self.certifier_reference_path is None:
            return None
        job = _load(self.certifier_reference_path.parent / "job.json") or {}
        start, end = _epoch(job.get("dispatched_at") or job.get("started_at")), _epoch(job.get("finished_at"))
        return (end - start) if start and end and end > start else None

    def _refresh(self, digest: str, record: dict[str, Any]) -> None:
        repeats = [self.screen.result_by_key(key) for key in record.get("repeats") or []]
        record["repeat_statuses"] = [(r or {}).get("timing_status") or TIMING_PENDING for r in repeats]
        repeat_failed = any(status in (V.TIMING_MEASURED_INVALID,) for status in record["repeat_statuses"])
        if record.get("state") == DEFERRED_INFEASIBLE:
            if repeat_failed:
                record["state"], record["reason"] = RETRACTED, "a repeat on the screening machine was wrong"
            return  # never started, so there is no certifying run to read
        if self.certifier is not None:
            result = self.certifier.result(digest)
            record["certifier_status"] = (result or {}).get("timing_status") or TIMING_PENDING
            if result is not None and "staleness" not in record and not result.get("superseded"):
                record["staleness"] = self._staleness(digest, result)
            if result is not None:
                record["certifier_cycles"] = (result.get("verdict") or {}).get("whole_window_cycles")
                record["certifier_feedback"] = F.compare(result, self.certifier_reference)
        if repeat_failed:
            record["state"], record["reason"] = RETRACTED, "a repeat on the screening machine was wrong"
        elif record.get("certifier_status") == V.TIMING_MEASURED_INVALID:
            record["state"], record["reason"] = RETRACTED, "the certifying machine found it wrong"
        elif record.get("certifier_status") == V.TIMING_REFUSED:
            record["state"], record["reason"] = UNCERTIFIED, "the certifying run produced no admissible reading"
        elif record.get("certifier_status") == V.TIMING_MEASURED and all(
            status == V.TIMING_MEASURED for status in record["repeat_statuses"]
        ):
            record["state"] = CERTIFIED

    # ---- coverage: the objective is what the PACKAGE compiles
    def transfer_check(self, digest: str, primary_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Whether this candidate's gains on the primary model carry to every configured held-out
        model, on the same op kind (:mod:`merlin.perf.transfer_check`). ``None`` when no held-out
        model is configured for this run (the default) or the primary result carries no verdict yet
        -- never a guess from a partial measurement.
        """
        if not primary_result or not self.held_out:
            return None
        held_out_results: dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]] = {}
        for name, (service, _reference_path) in self.held_out.items():
            result = service.measurement_for(digest)
            if result.get("verdict"):
                held_out_results[name] = (result, self.held_out_references.get(name))
        if not held_out_results:
            return {
                "schema": TC.SCHEMA,
                "primary": self.primary_name,
                "why": "no held-out model has measured this candidate yet",
            }
        return TC.build_transfer_report(
            primary_name=self.primary_name or "primary",
            primary_result=primary_result,
            primary_reference=self.screen_reference,
            held_out_results=held_out_results,
        )

    def coverage(self, result: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """The candidate's package-authored share and whether it keeps the seed's coverage. A candidate
        that hands work back to the library measures the library, not the compiler: it is measured and
        shown, and it is never the best nor certified."""
        if not result or not (result.get("build") or {}).get("groups"):
            return None
        mine = F.package_authored(result, self.screen_reference)
        floor = self.coverage_floor()
        document = {**{k: mine[k] for k in ("groups_answered", "groups_total", "priced_share")}, "eligible": True}
        if floor is not None and floor["package_sha256"] != result.get("package_sha256"):
            declined = sorted(set(floor["groups"]) - set(mine["groups"]), key=V._order)
            below = (
                mine["priced_share"] is not None
                and floor["priced_share"] is not None
                and (mine["priced_share"] + 1e-9 < floor["priced_share"])
            )
            document["declined_vs_seed"] = declined
            if below and declined:  # nothing handed back: not a regression, whatever the pricing says
                document["eligible"] = False
                document["coverage_regression"] = (
                    f"coverage_regression: {len(declined)} group(s) declined to the library "
                    f"(package-authored work {100 * mine['priced_share']:.1f}% < the seed's "
                    f"{100 * floor['priced_share']:.1f}%)"
                )
        return document

    def coverage_floor(self) -> dict[str, Any] | None:
        """The seed's package-authored coverage: the store's earliest-requested candidate with a build."""
        if getattr(self, "_coverage_floor", None) is not None:
            return self._coverage_floor
        jobs = sorted(
            (
                j
                for j in self.screen.jobs()
                if not j.get("replicate") and j.get("role", ROLE_CANDIDATE) == ROLE_CANDIDATE
            ),
            key=lambda j: float(j.get("requested_epoch") or 0),
        )
        for job in jobs:
            result = self.screen.result_by_key(job.get("job_key") or job["package_sha256"])
            if result and (result.get("build") or {}).get("groups") and self.screen_reference is not None:
                mine = F.package_authored(result, self.screen_reference)
                self._coverage_floor = {"package_sha256": job["package_sha256"], **mine}
                return self._coverage_floor
        return None

    def coverage_gate_document(self) -> dict[str, Any] | None:
        """The coverage gate a candidate request carries: the seed's package-authored groups and share,
        and the per-group prices (the same-machine reference's own cycles).  None until both exist."""
        if self.screen_reference is None:
            self.screen_reference = _load(self.screen_reference_path)
        floor = self.coverage_floor()
        if floor is None or self.screen_reference is None or floor.get("priced_share") is None:
            return None
        price = {
            str(row["group"]): int(row.get("cycles") or 0)
            for row in V.group_table((self.screen_reference or {}).get("verdict") or {})
        }
        return {
            "floor_package_sha256": floor["package_sha256"],
            "floor_share": floor["priced_share"],
            "floor_groups": list(floor["groups"]),
            "price": price,
        }

    def _arm_coverage_gate(self) -> None:
        gate = self.coverage_gate_document()
        if gate is not None:
            self.screen.coverage_gate = gate

    def ineligible_text(self, result: Mapping[str, Any] | None) -> str | None:
        """The lead line for a candidate below the coverage floor: which groups it declined, and what
        each costs when the package lowers it -- from the last eligible measurement's own per-group rows
        against the same-machine reference. Data only: the gap to reduce, never how."""
        mismatch = self.exactness_mismatch(result)
        if mismatch:
            return f"INELIGIBLE -- {mismatch}"
        coverage = self.coverage(result)
        if coverage is None or coverage.get("eligible"):
            return None
        declined = list(coverage.get("declined_vs_seed") or [])
        best = self._screen_best_unretracted() or {}
        ours = {str(r["group"]): r.get("cycles") for r in V.group_table(best.get("verdict") or {})}
        ref = {
            str(r["group"]): r.get("cycles") for r in V.group_table((self.screen_reference or {}).get("verdict") or {})
        }
        parts = [
            f"g{g} {ours.get(g) if ours.get(g) is not None else '?'} vs {ref.get(g) if ref.get(g) is not None else '?'}"
            for g in declined[:20]
        ]
        return (
            f"INELIGIBLE -- declines groups {', '.join('g' + g for g in declined[:20])}; delegating to the library "
            f"never counts. Those groups currently cost (ours vs the reference, cycles, from the last eligible "
            f"measurement {str(best.get('package_sha256'))[:12]}) when the package lowers them: {'; '.join(parts)}. "
            "That per-group gap is what to reduce."
        )

    def eligible(self, result: Mapping[str, Any] | None) -> bool:
        coverage = self.coverage(result)
        return (coverage is None or bool(coverage.get("eligible"))) and self.exactness_mismatch(result) is None

    def exactness_mismatch(self, result: Mapping[str, Any] | None) -> str | None:
        """Why ``result`` was graded under a different exactness contract than this run holds, or None.  A
        result graded under another contract -- looser or stricter -- is shown and never the best: its
        correctness answered a different question (re-measuring grades it under this one)."""
        from merlin.perf import exactness as EX

        verdict = (result or {}).get("verdict")
        if not verdict:
            return None
        carried = (getattr(self.screen, "exactness", None) or {}).get("contract")
        try:
            current = EX.Contract.from_value(carried, target=str(getattr(self.screen, "target", "") or ""))
        except EX.ExactnessError as exc:
            return f"this run's exactness contract cannot be read: {exc}"
        # A cell's result records its contract beside its verdict; a whole-model verdict carries its own.
        graded = EX.graded_semantics({"exactness": (result or {}).get("exactness") or verdict.get("exactness")})
        if graded == current.semantics_sha256:
            return None
        return (
            f"graded under exactness contract {graded[:12]}, while this run holds {current.semantics_sha256[:12]}; "
            "re-measure these bytes to grade them under this run's contract"
        )

    def noise_margin(self) -> float:
        """What counts as an improvement ON THIS MACHINE: the largest of NOISE_FLOOR, the median
        batched-vs-solo repeat spread measured in this store, and the machine's own same-day solo spread
        of an identical program (:func:`.noise.margin`). A fixed 0.1% floor undercounted one board's own
        noise (repeats of one identical package landing up to 0.4% apart) and another board moved 2.4%
        between two solo runs of one ELF: a candidate crowned on a margin narrower than the noise the
        machine itself produces on a rerun is not a measured improvement, it is the spread. NOISE_FLOOR
        remains the bound for a store too young to have a repeat -- never zero, or a two-candidate store
        would crown on a single tied board run."""
        return float(self.noise()["margin"])

    def machine_noise(self) -> dict[str, Any]:
        """The screen machine's noise (:func:`.noise.machine_noise`), from every solo reading this store
        and its reference hold; recomputed only when the store holds a different set of results."""
        from . import noise as NOISE

        root = getattr(self.screen, "root", None)
        roots = [Path(root)] if root else []
        reference = self.screen_reference_path
        extra = [reference] if reference is not None and reference.is_file() else []
        paths = [p for r in roots for p in NOISE.result_paths(r)]
        key = (len(paths), tuple(str(p) for p in extra))
        cached = getattr(self, "_machine_noise_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        readings = NOISE.solo_readings(roots, extra=extra)
        device = ((self.screen_reference or {}).get("device") or {}).get("binary_sha256")
        if not device and readings:
            device = readings[-1]["device"]
        document = NOISE.machine_noise(readings, device=device, controls=NOISE.control_readings(roots))
        self._machine_noise_cache = (key, document)
        return document

    def noise(self) -> dict[str, Any]:
        """The margin, which measurement set it, and the machine's noise behind it (flagged when fewer
        than two same-day solo repeats exist)."""
        from . import noise as NOISE

        machine = self.machine_noise()
        return {**NOISE.margin(machine, floor=NOISE_FLOOR, batched_vs_solo=self.repeat_spread()), "machine": machine}

    def repeat_spread(self) -> float | None:
        """The median relative difference between a batched result and its solo repeat -- IS the noise
        margin (see :meth:`noise_margin`), and is also surfaced to the agent as context."""
        spreads = []
        for job in self.screen.jobs():
            if int(job.get("replicate") or 0) != 1:
                continue
            first = self.screen.result(job["package_sha256"]) or {}
            again = self.screen.result_by_key(job.get("job_key")) or {}
            a, b = first.get("objective_cycles"), again.get("objective_cycles")
            if a and b:
                spreads.append(abs(int(b) - int(a)) / int(a))
        spreads.sort()
        return spreads[len(spreads) // 2] if spreads else None

    def _screen_best_unretracted(self) -> dict[str, Any] | None:
        contenders: list[tuple[float, dict[str, Any]]] = []
        for job in self.screen.jobs():
            if job.get("replicate"):
                continue
            digest = job["package_sha256"]
            if (self._promoted.get(digest) or {}).get("state") == RETRACTED:
                continue
            if not self.screen.attributable(digest):
                # Bytes requested mid-round by a round that was not (or not yet) authored: measured and
                # shown, never the best, so never promoted, certified, tagged or exported.
                continue
            result = self.screen.result(digest) or {}
            reference_elf = ((self.screen_reference or {}).get("build") or {}).get("elf_sha256")
            if reference_elf and (result.get("build") or {}).get("elf_sha256") == reference_elf:
                # The reference program under a candidate's digest (a package that answered no group)
                # is the bar measured again, never an achievement.
                continue
            cycles = V.objective_cycles(result.get("verdict"))
            if cycles is None or result.get("timing_status") != V.TIMING_MEASURED:
                continue
            if not self.eligible(result):
                continue  # the library's cycles under the package's digest are never an achievement
            contenders.append((float(job.get("requested_epoch") or 0), result))
        return self._best_of(contenders)

    def _best_of(self, contenders: list[tuple[float, dict[str, Any]]]) -> dict[str, Any] | None:
        """The best of ``contenders`` ((requested epoch, result) pairs).

        A TIE IS NOT AN IMPROVEMENT: within NOISE_FLOOR of the lowest, the EARLIEST stays the best. And a
        new best must hold up SOLO: when its solo repeat has landed and is not better than the previous
        best's solo result by more than the floor, it is demoted to a tie with that previous best, which
        keeps the title and its certification. Without a solo result on either side, nothing is demoted."""
        if not contenders:
            return None
        lowest = min(int(r["objective_cycles"]) for _e, r in contenders)
        margin = self.noise_margin()
        within = [(e, r) for e, r in contenders if int(r["objective_cycles"]) <= lowest * (1 + margin)]
        epoch, best = min(within, key=lambda item: item[0])
        previous = self._best_of([(e, r) for e, r in contenders if e < epoch])
        if previous is not None:
            mine, theirs = self._solo_cycles(best["package_sha256"]), self._solo_cycles(previous["package_sha256"])
            if mine is not None and theirs is not None and not mine < theirs * (1 - margin):
                self._demoted[best["package_sha256"]] = previous["package_sha256"]
                return self._best_of([(e, r) for e, r in contenders if r is not best])
        self._demoted.pop(best["package_sha256"], None)
        return best

    def _confirmed(self, digest: str) -> bool:
        """Whether a best has held up: no repeat is asked for, or its solo repeat has finished."""
        if self.repeats_on_best <= 1:
            return True
        return any(
            job.get("state") in ("done", "failed", "superseded", "screen_failed")
            for job in self.screen.jobs()
            if job.get("package_sha256") == digest and int(job.get("replicate") or 0) == 1
        )

    def _solo_cycles(self, digest: str) -> int | None:
        """The solo repeat's cycles of a promoted best (replicate 1), when it landed MEASURED."""
        repeat = self.screen.result_by_key(f"{digest}.r1") or {}
        if repeat.get("timing_status") != V.TIMING_MEASURED:
            return None
        return V.objective_cycles(repeat.get("verdict")) or repeat.get("objective_cycles")

    def _history_with_coverage(self) -> list[dict[str, Any]]:
        rows = self.screen.history()
        for row in rows:
            result = self.screen.result_by_key(
                row["package_sha256"] if not row.get("replicate") else f"{row['package_sha256']}.r{row['replicate']}"
            )
            coverage = self.coverage(result)
            if coverage is not None:
                row["package_groups"] = coverage["groups_answered"]
                row["package_priced_share"] = coverage["priced_share"]
                row["eligible"] = coverage["eligible"]
                if coverage.get("coverage_regression"):
                    row["coverage_regression"] = coverage["coverage_regression"]
        best = self._screen_best_unretracted()
        if best is not None:
            margin, bar = self.noise_margin(), int(best["objective_cycles"])
            for row in rows:
                cycles = row.get("objective_cycles")
                if (
                    cycles
                    and not row.get("replicate")
                    and row.get("package_sha256") != best["package_sha256"]
                    and row.get("eligible", True)
                    and abs(int(cycles) - bar) <= bar * margin
                ):
                    row["tie_with"] = best["package_sha256"]
        for row in rows:
            if not row.get("replicate") and row.get("package_sha256") in self._demoted:
                row["tie_with"] = self._demoted[row["package_sha256"]]
                row["tie_reason"] = (
                    f"its solo repeat did not beat the previous best's solo result by the noise margin "
                    f"({self.noise_margin():.4%})"
                )
        return rows

    def rule(self) -> dict[str, Any] | None:
        """The instruction rule this run's programs are held to, or None: the prohibited ROLES (the
        screen's build option, the same one the builder routes by and the service checks) and the
        instructions the target's facts give them. The reference is measured WITHOUT the rule, so under
        one it is orientation, and the number to beat is this run's own best."""
        roles = list((getattr(self.screen, "build_options", None) or {}).get(G.PROHIBITED_ROLES) or ())
        if not roles:
            return None
        try:
            from merlin.perf.isa_prohibition import prohibited_instructions

            names = sorted(prohibited_instructions(str(self.screen.target), roles).values())
        except Exception as exc:  # noqa: BLE001 -- stated as unknown, never dropped
            names = [f"UNKNOWN: {type(exc).__name__}: {exc}"]
        return {
            "prohibited_roles": roles,
            "prohibited_instructions": names,
            "library_groups": list((self.config or {}).get("rule_library_groups") or []),
            "reference_is_orientation": True,
        }

    def _noise_brief(self) -> dict[str, Any]:
        try:
            noise = self.noise()
        except Exception as exc:  # noqa: BLE001 -- an unreadable store states its noise as unknown, never zero
            return {"margin": NOISE_FLOOR, "basis": "floor", "established": False, "flag": f"UNKNOWN: {exc}"}
        machine = noise.get("machine") or {}
        return {
            "margin": noise["margin"],
            "basis": noise["basis"],
            "established": noise["established"],
            "flag": noise.get("flag"),
            "device": machine.get("device"),
            "same_day": machine.get("same_day"),
            "cross_day": machine.get("cross_day"),
        }

    # ---- what the agent and the report read
    def summary(self) -> dict[str, Any]:
        bar = (self.screen_reference or {}).get("verdict") or {}
        cert_bar = (self.certifier_reference or {}).get("verdict") or {}
        best = self._screen_best_unretracted()
        best_digest = best["package_sha256"] if best else None
        certification = copy.deepcopy(self._promoted.get(best_digest)) if best_digest else None
        document: dict[str, Any] = {
            "schema": SCHEMA,
            "screen_machine": (self.screen_reference or {}).get("device", {}).get("artifact")
            or self.screen.machine.get("kind"),
            # HOW the screen times a candidate: a whole-program run, or (board unavailable) the sum of its
            # groups each timed alone on the emulator -- a different quantity, never compared with the other.
            "screen_basis": _screen_basis(self.screen),
            "certifier_machine": (self.certifier_reference or {}).get("device", {}).get("artifact")
            if self.certifier is not None
            else None,
            "bar": {
                "screen_whole_window_cycles": bar.get("whole_window_cycles"),
                "screen_reference_status": (self.screen_reference or {}).get("timing_status"),
                "screen_reference_vendor_also_fails_count": bar.get("vendor_also_fails_count"),
                "certifier_whole_window_cycles": cert_bar.get("whole_window_cycles"),
                "certifier_reference_status": (self.certifier_reference or {}).get("timing_status"),
                "note": "each machine is compared only with its own reference; the two are different devices",
                # THE VENDOR'S CYCLES ARE CONTEXT: another implementation's number, never the target. The
                # machine's own fact-derived roofline is what a group's headroom is measured against.
                "role": "context_only",
            },
            "best": None,
            "candidate_provenance": dict(self.labels) or None,
            # Declared by the launch, orientation only: figures from other designs or analytical bounds.
            "orientation": list((self.config or {}).get("orientation") or []),
            "rule": self.rule(),
            # Corrections to the HARNESS the launch declares (a builder defect found and fixed): what the
            # agent's earlier results were measuring, stated so they are not read as verdicts on its code.
            "harness_notices": [str(n) for n in (self.config or {}).get("harness_notices") or ()],
            "history": self._history_with_coverage(),
            "coverage_floor": {
                k: (self.coverage_floor() or {}).get(k)
                for k in ("package_sha256", "groups_answered", "groups_total", "priced_share")
            },
            "certifier_history": self.certifier.history() if self.certifier is not None else [],
            "board": _board_status(self.screen),
            # WHAT COUNTS AS AN IMPROVEMENT ON THIS MACHINE, and how well its noise is known.
            "noise": self._noise_brief(),
        }
        if best is not None:
            cycles = int(best["objective_cycles"])
            screen_bar = bar.get("whole_window_cycles")
            coverage = self.coverage(best) or {}
            document["best"] = {
                "package_sha256": best_digest,
                "screen_whole_window_cycles": cycles,
                # WHAT THE PACKAGE AUTHORED of those cycles, beside them -- never a bare count.
                "package_authored": {k: coverage.get(k) for k in ("groups_answered", "groups_total", "priced_share")},
                "screen_vendor_also_fails_count": (best.get("verdict") or {}).get("vendor_also_fails_count"),
                "screen_ratio_to_bar": round(cycles / int(screen_bar), 4) if screen_bar else None,
                "screen_gap_cycles": cycles - int(screen_bar) if screen_bar else None,
                "certification": certification,
            }
        return document


DONE_STATE = "done"
#: The smallest relative difference ever treated as an improvement on the board.
NOISE_FLOOR = 0.001
#: A running certification of a superseded best continues once past this fraction of its expected run.
CERT_KEEP_FRACTION = 0.5


def _key_of(job: Mapping[str, Any]) -> str:
    return str(job.get("job_key") or job["package_sha256"])


def _epoch(stamp: Any) -> float | None:
    return epoch_of(stamp)


#: A compact agent response stays under about 4 KB: the audit measured 161 s of an agent's time spent
#: parsing 30-37 KB responses.
COMPACT_TEXT = 2200
COMPACT_FAILING = 12
WAIT_MAX_SECONDS = 540.0
WAIT_POLL_SECONDS = 10.0


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 40] + f"\n... [{len(text) - limit + 40} chars clipped]"


def _compact(result: Mapping[str, Any]) -> dict[str, Any]:
    verdict = result.get("verdict") or {}
    return {
        "timing_status": result.get("timing_status"),
        "job_state": result.get("job_state"),
        "whole_window_cycles": verdict.get("whole_window_cycles"),
        "objective_cycles": result.get("objective_cycles"),
        "correctness": verdict.get("correctness"),
        "vendor_also_fails_count": verdict.get("vendor_also_fails_count"),
        "refusal": result.get("refusal") or verdict.get("refusal"),
        "invalid_reason": verdict.get("invalid_reason"),
        "pre_measure_check": {
            key: (result.get("pre_measure_check") or {}).get(key) for key in ("label", "required", "passed", "summary")
        }
        if result.get("pre_measure_check")
        else None,
        "device": (result.get("device") or {}).get("artifact"),
        "job_id": (result.get("run") or {}).get("job_id"),
    }


__all__ = [
    "CERTIFIED",
    "CERT_PENDING",
    "DEFERRED_INFEASIBLE",
    "NOISE_FLOOR",
    "RETRACTED",
    "SCHEMA",
    "UNCERTIFIED",
    "WholeModelObjective",
    "resolve_capsules",
]
