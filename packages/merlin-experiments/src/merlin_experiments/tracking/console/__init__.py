"""The experiment command console: discover, track, inspect and diagnose any experiment from its records.

Three layers keep the console honest and testable:

* :mod:`.sources` -- bounded, read-only readers, one per concern (agents, tokens, stages, capsules,
  performance, passes, jobs, resources, events). Each returns plain JSON-compatible data; a missing
  record is ``None`` plus a notice, never a zero.
* :mod:`.model` -- assembles one experiment's snapshot on a fast and a slow cadence, and
  :mod:`.diagnose` derives bottleneck findings from it without granting any verdict.
* Views -- :mod:`.serve` streams snapshots as JSON lines to the Bubble Tea frontend
  (``packages/merlin-experiments/console``), and :mod:`.plain` prints the same snapshot as text
  for ``--once``, pipes and hosts without the built frontend.

Nothing here writes into a run, a workspace or a measurement store, launches a job, or touches hardware.
"""

from __future__ import annotations

SNAPSHOT_SCHEMA = "merlin.console.snapshot.v1"
EXPERIMENTS_SCHEMA = "merlin.console.experiments.v1"

__all__ = ["EXPERIMENTS_SCHEMA", "SNAPSHOT_SCHEMA"]
