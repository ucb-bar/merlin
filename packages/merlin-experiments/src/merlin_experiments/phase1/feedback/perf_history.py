"""Keep a per-test-case performance ledger: one row whenever a completed grade sees a case's numbers change.

The operator console's per-case view (first passing cycles, best, latest, per evidence kind) had to be
derived from every self-check and simulation response and every archived verdict of the run: on a
six-hour run that was about 200 MB of JSON parsed at each console start, and the simulation responses
carry no timestamp of their own, so their times were file modification times, which a copy of the run
does not keep. The grade already computes each passing case's current evidence for the headroom
ranking (:mod:`.performance_headroom`); this module records it as it is, once per change.

``qa/perf_history.jsonl``, schema ``merlin.phase1.perf_history.v1``, one JSON object per line::

    {"schema", "t" (unix seconds the evidence was observed, else recorded), "recorded_at",
     "test_case", "kind": "measured"|"modeled"|"status"|"pair", "grade_token", "graded_source_sha256",
     measured/modeled: "cycles", "evidence", "execution_digest" | "source_sha256",
                       "roofline_bound_cycles", "bound_by", "array_utilization_pct", "useful_macs",
                       "structural_peak_macs_per_cycle", "dominant_modeled_resource",
     status: "l2", "l3",
     pair: "execution_digest", "modeled", "measured"}

* ``measured``: the cycles of the case's current gSIM L3 certificate (the simulator row whose execution
  digest ``qa/tier_state.json`` certified), or for a case without a certificate tier its operator
  idle-time gSIM measurement (:mod:`.idle_gsim`, evidence ``measured: gSIM measurement (advisory)``);
* ``modeled``: the Spike timing model estimate the grade row carries, else the newest ``submission/``
  self-check estimate (:func:`.idle_performance.estimates`), as the headroom ranking chooses them;
* ``status``: the loop-tier result and the newest L3 ledger status;
* ``pair``: the newest same-executable (modeled, measured) pair (:func:`.model_calibration.pairs`).

A row is appended only when it differs from the case's previous row of the same kind, so the file grows
with change, not with the number of grades. It is written under the completed-grade lock, beside the
document it is derived from, adds nothing the author's own records do not hold, and never affects a
grade: any failure is logged and dropped. The console tails it when present and derives otherwise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA = "merlin.phase1.perf_history.v1"
FILENAME = "perf_history.jsonl"
#: Past this size the ledger stops growing (a note says so); readers keep working on what it holds.
MAX_BYTES = 64 << 20
_MODELED = ("PROVISIONAL", "QUALIFIED")


def _evidence_fields(cycles, roofline: dict, performance: dict) -> dict:
    from .performance_headroom import _bound, _dominant_resource, _positive

    roofline = roofline if isinstance(roofline, dict) else {}
    bound, limiter = _bound(roofline)
    useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
    fields = {
        "cycles": cycles,
        "roofline_bound_cycles": bound,
        "bound_by": limiter,
        "useful_macs": useful if _positive(useful) else None,
        "structural_peak_macs_per_cycle": peak if _positive(peak) else None,
        "array_utilization_pct": round(100.0 * useful / (cycles * peak), 2)
        if _positive(useful) and _positive(peak) and _positive(cycles)
        else None,
    }
    dominant = _dominant_resource(performance if isinstance(performance, dict) else {})
    if dominant is not None:
        fields["dominant_modeled_resource"] = dominant
    if isinstance(performance, dict) and performance:
        fields["performance_identity"] = {key: performance.get(key) for key in (
            "complete_event_accounting", "selection_sha256", "model_sha256", "facts_sha256",
            "comparison_domain_sha256", "scope_sha256", "capsule_sha256")}
    return fields


def _complete(performance) -> bool:
    """A timing-model estimate with complete event accounting (the only kind recorded as modeled)."""
    from .performance_headroom import _positive

    return (
        isinstance(performance, dict)
        and performance.get("status") in _MODELED
        and performance.get("complete_event_accounting") is True
        and _positive(performance.get("estimated_whole_soc_cycles"))
    )


def _tier(state: dict, tier: str):
    value = state.get(tier)
    return value.get("status") if isinstance(value, dict) else value


class Recorder:
    """Collects one completed grade's per-case evidence (:meth:`observe`) and appends what changed."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.error: str | None = None
        self.current: dict[str, tuple] = {}  # case -> (current L2 digest, certified L3 digest)

    def observe(self, passing: list, tier_state: dict, measured: dict, estimates: dict, advisory=None) -> None:
        """Hook of :func:`.performance_headroom.build`: never raises (the ranking must not depend on it)."""
        try:
            self._observe(
                passing,
                tier_state if isinstance(tier_state, dict) else {},
                measured or {},
                estimates or {},
                advisory or {},
            )
        except Exception as error:  # noqa: BLE001 -- an advisory record never fails the grade
            self.error = f"{type(error).__name__}: {error}"

    def _observe(self, passing: list, tier_state: dict, measured: dict, estimates: dict, advisory=None) -> None:
        from .performance_headroom import ADVISORY, advisory_current

        for row in passing:
            name = row.get("capsule", row.get("name"))
            if not isinstance(name, str):
                continue
            state = tier_state.get(name) if isinstance(tier_state.get(name), dict) else {}
            self.current[name] = tuple(
                (state.get(t) or {}).get("execution_digest") if isinstance(state.get(t), dict) else None
                for t in ("L2", "L3")
            )
            tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
            l2 = tiers.get("L2")
            self.rows.append(
                {
                    "test_case": name,
                    "kind": "status",
                    "l2": l2.get("status") if isinstance(l2, dict) else l2,
                    "l3": _tier(state, "L3"),
                }
            )
            certificate = state.get("L3") if isinstance(state.get("L3"), dict) else {}
            hit = measured.get(certificate.get("execution_digest")) if certificate.get("status") == "pass" else None
            if hit is not None:
                cycles, roofline = hit
                self.rows.append(
                    {
                        "test_case": name,
                        "kind": "measured",
                        "evidence": "measured: latest gSIM certificate",
                        "execution_digest": certificate.get("execution_digest"),
                        **_evidence_fields(cycles, roofline, {}),
                    }
                )
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            lane, _ = advisory_current((advisory or {}).get(name), performance) if hit is None else (None, False)
            if lane is not None:  # an idle-time gSIM measurement of a case without a certificate tier
                self.rows.append(
                    {
                        "test_case": name,
                        "kind": "measured",
                        "evidence": ADVISORY,
                        "execution_digest": lane["execution_digest"],
                        "t": lane["t"] / 1e9 if type(lane["t"]) is int and lane["t"] > 0 else None,
                        **_evidence_fields(lane["cycles"], lane["roofline"], {}),
                    }
                )
            timed = [e for e in estimates.get(name) or () if _complete(e[2])]
            if _complete(performance):
                graded = row.get("graded") if isinstance(row.get("graded"), dict) else {}
                self.rows.append(
                    {
                        "test_case": name,
                        "kind": "modeled",
                        "evidence": f"modeled: Spike timing model ({performance['status'].lower()})",
                        "source_sha256": graded.get("graded_source_sha256"),
                        **_evidence_fields(
                            performance["estimated_whole_soc_cycles"],
                            performance.get("structural_roofline"),
                            performance,
                        ),
                    }
                )
            elif timed:
                graded_ns, digest, performance = timed[0]
                self.rows.append(
                    {
                        "test_case": name,
                        "kind": "modeled",
                        "evidence": "modeled: Spike timing model (self-check)",
                        "source_sha256": digest,
                        "t": graded_ns / 1e9 if type(graded_ns) is int and graded_ns > 0 else None,
                        **_evidence_fields(
                            performance["estimated_whole_soc_cycles"],
                            performance.get("structural_roofline"),
                            performance,
                        ),
                    }
                )

    def append(self, workspace, origin: dict) -> int:
        """Append the rows that changed since the ledger's last row of their case and kind; never raises."""
        try:
            return self._append(Path(workspace), origin if isinstance(origin, dict) else {})
        except (OSError, ValueError, TypeError) as error:
            print(f"[perf-history] not recorded: {type(error).__name__}: {error}", flush=True)
            return 0

    def _pairs(self, workspace: Path) -> list[dict]:
        from . import model_calibration

        newest: dict[str, dict] = {}
        for pair in model_calibration.pairs(workspace):
            name = pair["test_case"]
            # The pair of the current loop-tier executable, else of the certified one, else any.
            rank = {d: i for i, d in enumerate(self.current.get(name, ())) if d}.get(pair["execution_digest"], 9)
            if name in self.current and (name not in newest or rank < newest[name][0]):
                newest[name] = (rank, pair)
        newest = {name: pair for name, (_, pair) in newest.items()}
        return [
            {
                "test_case": name,
                "kind": "pair",
                "execution_digest": p["execution_digest"],
                "modeled": p["modeled"],
                "measured": p["measured"],
            }
            for name, p in sorted(newest.items())
        ]

    def _append(self, workspace: Path, origin: dict) -> int:
        if self.error:
            print(f"[perf-history] not recorded: {self.error}", flush=True)
            return 0
        path = workspace / "qa" / FILENAME
        size = path.stat().st_size if path.is_file() else 0
        if size > MAX_BYTES:
            return 0
        last = _last_rows(path) if size else {}
        now = time.time()
        rows = self.rows + (self._pairs(workspace) if self.rows else [])
        fresh = []
        for row in rows:
            key = (row["test_case"], row["kind"])
            if _identity(row) == _identity(last.get(key)):
                continue
            last[key] = row
            fresh.append(
                {
                    "schema": SCHEMA,
                    "t": row.get("t") or now,
                    "recorded_at": now,
                    "grade_token": origin.get("token"),
                    "graded_source_sha256": origin.get("graded_source_sha256"),
                    **{k: v for k, v in row.items() if k != "t"},
                }
            )
        if fresh:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fresh))
        return len(fresh)


def _identity(row: dict | None):
    """What makes a row new: its numbers, and for a measurement the certified executable.

    An estimate is keyed by its cycles alone: the source digest moves with every edit, also edits that
    leave the case's program unchanged, and a ledger row per edit would record nothing new.
    """
    if not isinstance(row, dict):
        return None
    kind = row.get("kind")
    if kind == "status":
        return (row.get("l2"), row.get("l3"))
    if kind == "pair":
        return (row.get("execution_digest"), row.get("modeled"), row.get("measured"))
    if kind == "measured":
        return (row.get("cycles"), row.get("execution_digest"))
    return (row.get("cycles"), row.get("evidence"))


def _last_rows(path: Path) -> dict:
    """``{(test case, kind): newest row}`` of an existing ledger (lines that do not parse are skipped)."""
    last: dict = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == SCHEMA and isinstance(row.get("test_case"), str):
                last[(row["test_case"], row.get("kind"))] = row
    return last


__all__ = ["FILENAME", "MAX_BYTES", "Recorder", "SCHEMA"]
