"""The run's OOT history: one harness commit per candidate, a tag per measurement, ``best`` on a win.

A measured run's ``oot/`` is a git repository cloned from Phase 1's ``frozen`` tag
(:func:`merlin.common.oot_repo.init_from`) and written ONLY by the harness.  For every candidate the
loop offers, :meth:`OotLedger.record` commits the exact bytes measured and verifies that the commit's
tree digest IS the store's package digest; the commit record -- never a copy of the package -- is the
iteration row (``iterations.jsonl``).  When a candidate's measurement lands it is tagged
``measured/<n>`` (immutable); ``best`` moves only on a CONFIRMED win: the objective's best, held up by
its solo repeat and, where the run has a certifier, certified.  :func:`champion_records` assembles the
evidence :func:`merlin.targetgen.champions.export_champion` requires, from the store's own results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common import oot_repo
from merlin.perf import whole_model_verdict as V

from . import jobs as J

ITERATIONS = "iterations.jsonl"
LANDED = (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID, V.TIMING_REFUSED)


class LedgerError(RuntimeError):
    """The OOT history disagrees with the measurement store."""


class OotLedger:
    """The harness's writer of one run's ``oot/`` repository."""

    def __init__(
        self,
        repo: Path,
        *,
        run_id: str,
        records: Path,
        sandbox_roots: Sequence[Path] = (),
        clock: Any = None,
        oot: Any = oot_repo,
    ) -> None:
        self.repo = Path(repo)
        self.run_id = str(run_id)
        self.records = Path(records)
        self.sandbox_roots = tuple(Path(p) for p in sandbox_roots)
        self.oot = oot
        if clock is None:
            from merlin.common.artifacts import utc_stamp

            clock = utc_stamp
        self.clock = clock
        self.oot.check_outside_sandbox(self.repo, self.sandbox_roots)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.records.is_file():
            return []
        return [json.loads(line) for line in self.records.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append(self, row: Mapping[str, Any]) -> None:
        self.records.parent.mkdir(parents=True, exist_ok=True)
        with self.records.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")

    def candidates(self) -> list[dict[str, Any]]:
        return [row for row in self._rows() if row.get("kind") == "candidate"]

    def record(
        self, package: Path, digest: str, *, label: str, attribution: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Commit ``package`` (the bytes measured as ``digest``) once, and return its iteration row.
        ``attribution`` is the store's record of who earned them, when an authoring round asked."""
        for row in self.candidates():
            if row["package_sha256"] == digest:
                if attribution is not None:
                    self.attribute(digest, attribution)
                return row
        n = len(self.candidates()) + 1
        commit = self.oot.commit_candidate(
            self.repo,
            Path(package),
            label=f"candidate {n}",
            when=self.clock(),
            run_id=self.run_id,
            metadata={"label": label},
            sandbox_roots=self.sandbox_roots,
        )
        try:
            self.oot.verify(self.repo, commit.commit, digest)
        except Exception as exc:  # noqa: BLE001 -- restated as the ledger's own refusal
            raise LedgerError(
                f"candidate {n}'s commit is not the bytes the store measured ({digest[:12]}): {exc}"
            ) from exc
        row = {"kind": "candidate", "n": n, "package_sha256": digest, "label": label, "commit": commit.as_record()}
        if attribution is not None:
            row["attribution"] = (attribution.get("history") or [{}])[-1] | {"state": attribution.get("state")}
        self._append(row)
        return row

    def attribute(self, digest: str, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append the store's attribution ``record`` for ``digest`` to the history, as it now stands."""
        row = {
            "kind": "attribution",
            "package_sha256": digest,
            "state": record.get("state"),
            "change": (record.get("history") or [None])[-1],
            "at": self.clock(),
        }
        self._append(row)
        return row

    def sync(self, objective: Any) -> list[dict[str, Any]]:
        """Tag each candidate whose measurement landed, and move ``best`` to a confirmed win."""
        events = []
        tags = self.oot.tags(self.repo)
        rows = self.candidates()
        by_digest = {row["package_sha256"]: row for row in rows}
        for row in rows:
            name = f"{self.oot.MEASURED_PREFIX}{row['n']}"
            if name in tags:
                continue
            found = objective.screen.measurement_for(row["package_sha256"])
            if found.get("timing_status") in LANDED:
                self.oot.tag(self.repo, name, row["commit"]["commit"])
                event = {
                    "kind": "measured",
                    "n": row["n"],
                    "tag": name,
                    "timing_status": found.get("timing_status"),
                    "objective_cycles": found.get("objective_cycles"),
                    "at": self.clock(),
                }
                self._append(event)
                events.append(event)
        winner = confirmed_best(objective)
        if winner is not None and winner in by_digest:
            commit = by_digest[winner]["commit"]["commit"]
            if tags.get(self.oot.BEST_TAG) != commit:
                self.oot.tag(self.repo, self.oot.BEST_TAG, commit, move=True)
                event = {"kind": "best", "n": by_digest[winner]["n"], "package_sha256": winner, "at": self.clock()}
                self._append(event)
                events.append(event)
                from merlin.targetgen import target_index

                target_index.refresh_for_run(self.repo.parent, 2)  # the index lists the run's new best
        return events


def confirmed_best(objective: Any) -> str | None:
    """The objective's best when it is a CONFIRMED win: its solo repeat held, and -- where the run has
    a certifier -- the certifier certified it.  None otherwise."""
    summary = objective.summary() or {}
    best = summary.get("best") or {}
    digest = best.get("package_sha256")
    if not digest or not objective._confirmed(digest) or not objective.screen.attributable(digest):
        return None
    if getattr(objective, "certifier", None) is not None:
        if (best.get("certification") or {}).get("state") != "certified":
            return None
    return str(digest)


def champion_records(
    objective: Any,
    digest: str,
    *,
    phase1_run: str,
    frozen_commit: str,
    corpus_seal_digest: str,
    phase0_evidence_digest: str,
    roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """The four records :func:`merlin.targetgen.champions.export_champion` requires, read from the
    store's own results for ``digest``: the board measurement (batched, with the in-batch control), the
    certifier's verdict, and the whole-ELF instruction scan.  A field the store cannot support is left
    out, so the export refuses it rather than defaulting it."""
    screen = objective.screen.result(digest) or {}
    batch = screen.get("batch") or {}
    control = batch.get("control") or {}
    firesim: dict[str, Any] = {
        "standing_attempt": screen.get("from_attempt"),
        "cycles": screen.get("objective_cycles"),
        "machine": (screen.get("device") or {}).get("artifact"),
        "header": (screen.get("build") or {}).get("parameter_header_sha256"),
        "control": {"in_batch": bool(control.get("ok")) and int(batch.get("size") or 0) > 1, **dict(control)},
        "timing_status": screen.get("timing_status"),
        "batch": batch.get("batch"),
    }
    certifier = getattr(objective, "certifier", None)
    certified = certifier.result(digest) if certifier is not None else None
    gsim = {"verdict": "pass" if (certified or {}).get("timing_status") == V.TIMING_MEASURED else "not_certified"}
    if certified:
        gsim.update(
            timing_status=certified.get("timing_status"),
            device=(certified.get("device") or {}).get("artifact"),
            cycles=(certified.get("verdict") or {}).get("whole_window_cycles"),
        )
    census = (screen.get("build") or {}).get("isa_census")
    # The scan's own record of what it held the program to (the instruction gate keeps it on a clean
    # build); a clean verdict without one is not evidence, so it is never reported clean here.
    isa = (screen.get("build") or {}).get("isa_prohibition") or {}
    prohibited = {str(k): str(v) for k, v in (isa.get("prohibited") or {}).items()}
    scanned = (
        bool(roles)
        and census is not None
        and not screen.get("isa_prohibited")
        and isa.get("verdict") == "clean"
        and bool(prohibited)
    )
    return {
        "provenance": {
            "phase1": {"run": phase1_run, "frozen_commit": frozen_commit},
            "corpus_seal_digest": corpus_seal_digest,
            "phase0_evidence_digest": phase0_evidence_digest,
        },
        "measurements": {"package_digest": digest, "firesim": firesim, "exactness": exactness_record(screen)},
        "certification": {"gsim": gsim},
        "isa_prohibition": {
            "scope": "whole_elf",
            "verdict": "clean" if scanned else "not_scanned",
            "prohibited_roles": [str(r) for r in roles],
            "prohibited_instructions": prohibited,
            **({"sealed_source": isa["sealed_source"]} if isa.get("sealed_source") else {}),
        },
    }


def exactness_record(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The exactness contract a measurement was graded under, as a champion records it: the contract's
    identity, the label (``exact`` only when every group was held exact) and each group's contract.
    None when the measurement recorded none -- the export then refuses it rather than assume ``exact``."""
    from merlin.perf import exactness as EX

    applied = (result.get("verdict") or {}).get("exactness") or result.get("exactness")
    if not isinstance(applied, Mapping) or not applied.get("contract"):
        return None
    contract = applied["contract"]
    return {
        "contract_sha256": contract.get("sha256"),
        "semantics_sha256": contract.get("semantics_sha256"),
        "contract_path": contract.get("path"),
        "label": EX.label_summary(applied),
        "per_group": dict(applied.get("per_group") or {}),
        "bounded_forms": [form for form in contract.get("forms") or () if form.get("mode") != EX.EXACT],
    }


def export_best(
    objective: Any, ledger: OotLedger, *, target: str, package_id: str, roles: Sequence[str], **provenance: str
) -> Path:
    """Export the run's ``best`` as a champion, with its records assembled from the store."""
    from merlin.targetgen import champions

    best = ledger.oot.tags(ledger.repo).get(ledger.oot.BEST_TAG)
    row = next((r for r in ledger.candidates() if r["commit"]["commit"] == best), None)
    if row is None:
        raise LedgerError("the run has no confirmed best to export")
    if not objective.screen.attributable(row["package_sha256"]):
        state = J.attribution_state(objective.screen.attribution(row["package_sha256"]))
        raise LedgerError(f"the best's bytes are {state}: no authored round earned them, so they are no champion")
    records = champion_records(objective, row["package_sha256"], roles=roles, **provenance)
    return champions.export_champion(target, ledger.repo, package_id=package_id, phase=champions.PHASE2, **records)


__all__ = [
    "ITERATIONS",
    "LedgerError",
    "OotLedger",
    "champion_records",
    "confirmed_best",
    "exactness_record",
    "export_best",
]
