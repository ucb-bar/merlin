"""Carry one assigned evaluation CPU budget into every local child process.

This configures a dedicated worker only. It acquires no parent CPU lease:
bounded build and simulator steps retain the existing shared-slot owner.
"""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.cpu_pool import grading_cpus, shared_compute_enabled

from .contracts import StageGateError


def environment(cpus: tuple[int, ...], *, temporary: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Narrow cooperative resource selection without replacing a shared namespace."""
    if (
        type(cpus) is not tuple
        or not cpus
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        or len(set(cpus)) != len(cpus)
        or not set(cpus) <= os.sched_getaffinity(0)
        or not temporary.is_absolute()
        or temporary.resolve() != temporary
        or any(path.is_symlink() for path in (temporary, *temporary.parents))
    ):
        raise StageGateError("worker resources require assigned CPUs and ordinary private scratch")
    result = dict(inherited)
    if shared_compute_enabled(result):
        if not set(cpus) <= set(grading_cpus(result)):
            raise StageGateError("worker CPU assignment exceeds the inherited shared compute grant")
    else:
        # With no outer shared selection, disjoint queue grants still share one
        # namespace among this job's build, preparation and simulator children.
        result["MERLIN_COMPUTE_CPU_SLOTS"] = str(temporary / "cpu-slots")
    roster = ",".join(map(str, cpus))
    result.update(
        TMPDIR=str(temporary),
        MERLIN_COMPUTE_CPUS=roster,
        MERLIN_PREPARATION_CPUS=roster,
        MERLIN_GRADE_CPUS=roster,
        CMAKE_BUILD_PARALLEL_LEVEL=str(len(cpus)),
    )
    # Parallelism belongs to admitted work items. Avoid multiplying it by
    # implicit numerical-library thread pools in each subprocess.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        result[key] = "1"
    return result


def configure(cpus: tuple[int, ...], *, temporary: Path) -> None:
    """Dedicated child entry: affinity and environment propagate to descendants."""
    selected = environment(cpus, temporary=temporary, inherited=dict(os.environ))
    temporary.mkdir(mode=0o700)
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != set(cpus):
        raise StageGateError("worker CPU affinity differs from its assigned grant")
    os.environ.update(selected)
