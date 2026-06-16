"""Instrumentation adapter: ingest *measured* CPU/runtime coupling from the ``aet`` harness.

The DSE guidance must never substitute invented constants for measurements. Measured CPU
coupling (host submit, command encode, sync wait) comes from one of two real sources:

  * an ``aet`` run directory (``/scratch/agustin/projects/aet`` style layout), reading the
    canonical collected-metrics file ``metrics/summary_metrics.json`` (flat, dotted keys); or
  * an explicit ``cpu_coupling`` YAML in the schema of the same name (e.g. exported from aet).

If neither is supplied, :func:`ingest` returns ``None`` and callers report
:data:`UNAVAILABLE_MESSAGE` verbatim — no fabricated numbers.

aet ``summary_metrics.json`` key convention (dotted), per dispatch regime::

    cpu.op_level_dispatch.host_submit_ns
    cpu.op_level_dispatch.command_encode_ns
    cpu.op_level_dispatch.sync_wait_ns
    cpu.batched_command_buffer.host_submit_ns
    cpu.batched_command_buffer.command_encode_ns
    cpu.batched_command_buffer.sync_wait_ns

aet runs are real executions, so their evidence tag defaults to ``measured``; a regime may
downgrade itself to ``trace_derived`` via a ``cpu.<regime>.source`` key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import schemas
from merlin.common.yaml import load_yaml

UNAVAILABLE_MESSAGE = "CPU coupling result unavailable: no measured overhead file provided."

OP_LEVEL = "op_level_dispatch"
BATCHED = "batched_command_buffer"
_NS_PER_MS = 1_000_000.0


@dataclass
class Regime:
    name: str
    host_submit_ns: float = 0.0
    command_encode_ns: float = 0.0
    sync_wait_ns: float = 0.0
    source: str = "measured"

    @property
    def dispatch_ns(self) -> float:
        """Per-dispatch host overhead (submit + encode); sync is accounted separately."""
        return self.host_submit_ns + self.command_encode_ns


@dataclass
class CpuCoupling:
    workload: str
    regimes: dict[str, Regime]
    provenance: str = "cpu_coupling_yaml"
    warnings: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        """Weakest evidence tag across the regimes (defaults to ``measured``)."""
        from merlin.dse_guidance.evidence import weakest_evidence
        tags = [r.source for r in self.regimes.values()] or ["measured"]
        return weakest_evidence(tags)

    def per_replan(self, dispatches_per_step: int, K: int,
                   num_regions: int = 1) -> dict:
        """Measured cpu_dispatch_ms / sync_ms per replan, op-level vs batched.

        Op-level issues ``dispatches_per_step * K`` submits per replan; a batched command
        buffer collapses them to ``num_regions`` (typically 1). The saved time is the measured
        benefit available to the ``command_batching`` / ``autonomous_K_loop`` axes.
        """
        op = self.regimes.get(OP_LEVEL)
        ba = self.regimes.get(BATCHED)
        n_op = max(int(dispatches_per_step) * max(int(K), 1), 0)
        n_ba = max(int(num_regions), 1)

        def regime_cost(regime: Regime | None, n: int) -> dict | None:
            if regime is None:
                return None
            return {
                "num_dispatches": n,
                "cpu_dispatch_ms": n * regime.dispatch_ns / _NS_PER_MS,
                "sync_ms": n * regime.sync_wait_ns / _NS_PER_MS,
                "source": regime.source,
            }

        out: dict = {
            "op_level": regime_cost(op, n_op),
            "batched": regime_cost(ba, n_ba),
        }
        if out["op_level"] and out["batched"]:
            out["cpu_dispatch_ms_saved"] = (
                out["op_level"]["cpu_dispatch_ms"] - out["batched"]["cpu_dispatch_ms"])
            out["sync_ms_saved"] = (
                out["op_level"]["sync_ms"] - out["batched"]["sync_ms"])
            out["source"] = self.source
        return out


def _regime_from_mapping(name: str, m: dict) -> Regime:
    return Regime(
        name=name,
        host_submit_ns=float(m.get("host_submit_ns", 0.0) or 0.0),
        command_encode_ns=float(m.get("command_encode_ns", 0.0) or 0.0),
        sync_wait_ns=float(m.get("sync_wait_ns", 0.0) or 0.0),
        source=str(m.get("source", "measured")),
    )


def from_cpu_coupling_doc(doc: dict) -> CpuCoupling:
    """Parse a ``cpu_coupling`` YAML mapping into a :class:`CpuCoupling`."""
    schemas.validate_or_raise(doc, "cpu_coupling")
    measurements = doc.get("measurements") or {}
    regimes = {name: _regime_from_mapping(name, m)
               for name, m in measurements.items() if isinstance(m, dict)}
    return CpuCoupling(workload=str(doc["workload"]), regimes=regimes,
                       provenance="cpu_coupling_yaml")


def from_aet_run(run_dir: str | Path, workload: str | None = None) -> CpuCoupling | None:
    """Read measured coupling from an ``aet`` run's ``metrics/summary_metrics.json``.

    Returns ``None`` if the run has no summary metrics or no ``cpu.*`` coupling keys.
    """
    run = Path(run_dir)
    summary = run / "metrics" / "summary_metrics.json"
    if not summary.is_file():
        return None
    try:
        metrics = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(metrics, dict):
        return None

    regimes: dict[str, Regime] = {}
    for regime in (OP_LEVEL, BATCHED):
        prefix = f"cpu.{regime}."
        keys = {k[len(prefix):]: v for k, v in metrics.items()
                if isinstance(k, str) and k.startswith(prefix)}
        if not keys:
            continue
        regimes[regime] = _regime_from_mapping(regime, keys)
    if not regimes:
        return None

    wl = workload or metrics.get("workload") or metrics.get("method") or run.name
    return CpuCoupling(workload=str(wl), regimes=regimes, provenance=f"aet_run:{run}")


def ingest(cpu_coupling_path: str | Path | None = None,
           aet_run: str | Path | None = None,
           workload: str | None = None) -> CpuCoupling | None:
    """Best-effort measured coupling from an explicit YAML or an aet run; ``None`` if absent."""
    if cpu_coupling_path:
        return from_cpu_coupling_doc(load_yaml(cpu_coupling_path))
    if aet_run:
        return from_aet_run(aet_run, workload=workload)
    return None
