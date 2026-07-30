"""``BenchTargetSpec`` — the per-target parameters a shared bench driver needs.

The gemmini and muon experiment harnesses cloned the *structure* of a perf bench, a redacted
self-check, and an agentic QA loop, differing only in (a) which capsule-runner family they drive
(``capsule_runner`` vs ``muon_capsule_runner`` — both expose ``discover_capsules`` + ``run_capsule``
with the same ``{status, tiers, failure, numeric}`` result shape) and (b) how they read a perf number
off the timing tier + name the peak. A ``BenchTargetSpec`` captures exactly those differences so the
driver (`benchharness.selfcheck`, `benchharness.perf`) is written once and a NEW target is a spec +
corpus, not a forked directory.

Deliberately NOT captured here: genuinely bespoke, single-instance machinery (gemmini's 8-approach
cross-backend matrix + bare-metal golden-C arm, its verilator/simjob broker). Those stay in the
gemmini experiment as callbacks/extra logic — folding them in would inject target-conditionals into
the shared driver (the overfit this repo forbids).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchTargetSpec:
    """Everything a shared bench driver needs to run one target.

    ``runner`` is any module exposing ``discover_capsules(root, *, labels=, contract=)`` and
    ``run_capsule(cap, package_dir, *, runs_root, run_id, contract=, timeout)`` returning the common
    ``{status, tiers, failure, numeric}`` dict (``merlin.targetgen.capsule_runner`` /
    ``muon_capsule_runner`` both qualify)."""

    name: str
    runner: Any
    corpus_root: Path
    # The RESOLVABLE target name (lowercase, as target_registry/manifest loaders expect) — distinct from
    # ``name``, which is a display label. Threaded into ``runner.run_capsule(target=...)`` so the shared
    # driver never relies on a runner-side default target. None only for stub/test runners that ignore it.
    target: str | None = None
    labels: set[str] | None = None
    # None -> the runner's default contract dir (absolute DEFAULT_CONTRACT_DIR / $MERLIN_CONTRACT_DIR);
    # pass an ABSOLUTE path to override. Avoid CWD-relative strings (break when not run from repo root).
    contract: str | None = None
    perf_tier: str = "L2"                       # the cycle-accurate timing tier
    # extract the target's perf headline (e.g. {"gflops":.., "pct_fp_peak":..} or {"util_pct":..})
    # from that tier's result dict; keys flow straight into the perf row + table.
    perf_fields: Callable[[dict], dict] = field(default=lambda tier: {})
    peak_note: str = ""                         # one-line description of the peak for report headers

    def discover(self) -> list[dict]:
        return self.runner.discover_capsules(str(self.corpus_root), labels=self.labels,
                                             contract=self.contract)
