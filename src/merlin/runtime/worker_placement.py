"""Explicit OS-CPU placement over caller-declared physical-hart capabilities.

This promotes the runtime's existing worker-list and session-controller policy.
It grants no vector/tile context safety, reentrancy, DMA support or execution
qualification. Target bootstrap must independently establish the supplied map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkerPlacement:
    cpu_hart_ids: tuple[int, ...]
    worker_harts: tuple[int, ...]
    worker_cpus: tuple[int, ...]
    controller_hart: int | None
    controller_cpu: int | None
    mapping_basis: str

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "worker_count": len(self.worker_harts),
            "os_cpu_count": len(self.cpu_hart_ids),
            "tile_context_safety_established": False,
            "concurrent_call_reentrancy_established": False,
        }


def plan_placement(
    brd,
    *,
    cpus: int,
    workers: int,
    backend: str = "rvv",
    worker_harts: tuple[int, ...] | None = None,
    master: int = 0,
    matrix: bool = False,
    controller_hart: int | None = None,
    cpu_hart_ids: tuple[int, ...] | None = None,
) -> WorkerPlacement:
    """Validate worker capabilities and translate physical IDs to OS pin indices.

    Absent an explicit map, preserve the existing identity-map convention.
    A controller is optional, scalar-only by role, and excluded from workers.
    This is a placement plan, not proof of the target's CPU enumeration.
    """
    if type(cpus) is not int or not 1 <= cpus <= brd.harts or type(workers) is not int or not 1 <= workers <= cpus:
        raise ValueError("CPU/worker count must fit the selected board")
    mapping = tuple(range(cpus)) if cpu_hart_ids is None else cpu_hart_ids
    if (
        not isinstance(mapping, tuple)
        or len(mapping) != cpus
        or any(type(h) is not int or not 0 <= h < brd.harts for h in mapping)
        or len(set(mapping)) != cpus
    ):
        raise ValueError("CPU-to-hart map must contain distinct declared physical harts")
    allowed = tuple(brd.matrix_harts if matrix else brd.hart_ids_for(backend))
    selected = tuple([master, *(h for h in allowed if h != master)][:workers]) if worker_harts is None else worker_harts
    if (
        not isinstance(selected, tuple)
        or len(selected) != workers
        or any(type(h) is not int or h not in allowed or h not in mapping for h in selected)
        or len(set(selected)) != workers
    ):
        raise ValueError("Worker placement/count is incompatible with CPUs and selected unit")
    if controller_hart is not None and (
        type(controller_hart) is not int or controller_hart not in mapping or controller_hart in selected
    ):
        raise ValueError("Controller must be an online physical hart outside the compute workers")
    return WorkerPlacement(
        mapping,
        selected,
        tuple(mapping.index(h) for h in selected),
        controller_hart,
        mapping.index(controller_hart) if controller_hart is not None else None,
        "caller_declared" if cpu_hart_ids is not None else "legacy_identity",
    )


def worker_placement(
    brd,
    *,
    cpus: int,
    workers: int,
    backend: str = "rvv",
    worker_harts: tuple[int, ...] | None = None,
    master: int = 0,
    matrix: bool = False,
) -> tuple[int, ...]:
    """Compatibility API for the original explicit worker-list validator."""
    return plan_placement(
        brd, cpus=cpus, workers=workers, backend=backend, worker_harts=worker_harts, master=master, matrix=matrix
    ).worker_harts
