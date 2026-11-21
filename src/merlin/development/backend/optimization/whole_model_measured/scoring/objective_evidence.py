"""Evidence derivation and stable ranking of measured whole-model candidates.

Functions receive the existing feedback/objective explicitly; its lifetime and
independent admission state remain owned by the public class.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.analysis.whole_model.grading import verdict as V
from merlin.development.backend.optimization.whole_model_measured.measurement import board_batches as BATCH
from merlin.development.backend.optimization.whole_model_measured.measurement.identity import (
    write_json_atomic,
)
from merlin.development.backend.optimization.whole_model_measured.measurement.jobs import (
    TIMING_PENDING,
)
from merlin.development.backend.optimization.whole_model_measured.scoring import feedback as F
from merlin.development.backend.optimization.whole_model_measured.scoring.governance import NOISE_FLOOR, _epoch


def _derive_rooflines(self) -> dict[str, Any]:
    import hashlib

    from merlin.development.backend.optimization.whole_model_measured.scoring import roofline as ROOF

    config = self.config or {}
    if config.get("roofline_feedback") is False:
        return {"why": "the objective config turned roofline feedback off (roofline_feedback: false)"}
    capsule = ((config.get("screen") or {}).get("build_options") or {}).get("model_capsule")
    target = str(getattr(self.screen, "target", "") or "")
    if not capsule or not Path(str(capsule)).is_dir() or not target:
        return {"why": "the screen names no model capsule to derive each group's roofline from"}
    try:
        from merlin.analysis.whole_model.build.inputs import load_model_capsule

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


def _staleness(self, digest: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """How far the certified candidate had fallen behind when its verdict landed: its own and the
    then-current best's digest and board cycles. Written beside the verdict, never into it."""
    best = (
        _screen_best_unretracted(
            self,
        )
        or {}
    )
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
    from merlin.development.backend.optimization.whole_model_measured.scoring.objective import (
        CERTIFIED,
        DEFERRED_INFEASIBLE,
        RETRACTED,
        UNCERTIFIED,
    )

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
            record["staleness"] = _staleness(self, digest, result)
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


def _arm_coverage_gate(self) -> None:
    gate = self.coverage_gate_document()
    if gate is not None:
        self.screen.coverage_gate = gate


def _screen_best_unretracted(self) -> dict[str, Any] | None:
    from merlin.development.backend.optimization.whole_model_measured.scoring.objective import RETRACTED

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
    return _best_of(self, contenders)


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
    previous = _best_of(self, [(e, r) for e, r in contenders if e < epoch])
    if previous is not None:
        mine, theirs = _solo_cycles(self, best["package_sha256"]), _solo_cycles(self, previous["package_sha256"])
        if mine is not None and theirs is not None and not mine < theirs * (1 - margin):
            self._demoted[best["package_sha256"]] = previous["package_sha256"]
            return _best_of(self, [(e, r) for e, r in contenders if r is not best])
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
    best = _screen_best_unretracted(
        self,
    )
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
