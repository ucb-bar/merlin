"""Ordered invocation of Merlin's in-process xDSL rewrites.

These existing rewrites mutate a module and report a count; they are not xDSL
ModulePass registrations. Native xDSL pass pipelines should continue to use its
pass manager. This adapter only preserves rewrite order, counters and observation
points for callers which already own their policy and selected transforms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp


@dataclass(frozen=True)
class Rewrite:
    """One selected rewrite and its existing reporting names.

    A missing stage intentionally leaves the rewrite inside the next observed
    stage, as the optional numerical preprocessing currently requires.
    """

    statistic: str
    stage: str | None
    apply: Callable[[ModuleOp], int]


def run_rewrites(
    module: ModuleOp,
    rewrites: Iterable[Rewrite],
    *,
    observe: Callable[[str, ModuleOp], None] | None = None,
    initial_stage: str | None = None,
) -> dict[str, int]:
    """Run each occurrence, observing only successfully completed rewrites.

    The iterable may resolve later transforms after earlier ones execute.
    Repeated transforms remain repeated. Counter names retain the existing
    last-write behavior; this is reporting, not pass-name deduplication.
    Exceptions propagate before subsequent rewrites or their observations.
    """
    if observe is not None and initial_stage is not None:
        observe(initial_stage, module)
    statistics = {}
    for rewrite in rewrites:
        statistics[rewrite.statistic] = rewrite.apply(module)
        if observe is not None and rewrite.stage is not None:
            observe(rewrite.stage, module)
    return statistics
