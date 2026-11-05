"""Derive a whole-model build's ``DeviceRouting`` from a placement, rather than declaring it.

Split out of :mod:`merlin.llvmlower.device_build` (which re-exports both names) to keep that module
under the repository's module-size limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    from .device_build import DeviceRouting


def routing_for_placement(
    placement,
    device: str,
    package_dir: str | Path,
    *,
    numeric_policy=None,
    granularity: str = "contraction",
    capture: str | Path | None = None,
    model: str = "",
    linked_elf_admission: LinkedElfAdmissionService | None = None,
) -> DeviceRouting:
    """The ``DeviceRouting`` a whole-model build needs, derived from a placement rather than declared.

    This is the step that made the fused single-ELF path unreachable in production. Every piece of it
    exists -- the rewrite, the shim, the per-extent kernel build, the link -- and all of it is inert
    until a caller supplies ``select``, which nothing did. So a "compiled" whole model ran its
    contractions through the Python interpreter while the artifact that would have run them on the
    device was never asked for.

    The operand and accumulate formats are read off the placement, NOT defaulted: they are what the
    router matched the contraction against, and a build that assumed them would emit kernels in a
    precision the placement never chose. Two consequences, both deliberate:

    * a placement that put nothing on this device raises, because a routing with no work is a caller
      error rather than an empty build;
    * device placements that disagree about the datapath raise too. One ELF carries one device
      datapath; picking the first and dropping the rest is how half a model gets computed in a
      precision nobody selected.

    The accumulate format has a SECOND derivation because a unit legitimately has no accumulate rule to
    match: a contract that declares its formats without an accumulate matrix routes fine and reports
    ``acc=None`` (measured: the reference systolic mesh does exactly this). That is a gap in the
    contract, not in the hardware, so the format is then read off the device's own RTL datapath facts
    for this operand pair -- the same source :mod:`merlin.system.offload` uses -- and it is still an
    error when those facts name none, or name more than one.
    """
    from merlin.system.place import device_selector

    # At call time: that module re-exports this function, so it imports this one.
    from .device_build import DeviceRouting

    on_dev = [p for p in placement.placed if p.on_device and p.device == device]
    if not on_dev:
        raise ValueError(f"this placement puts no work on {device!r}; there is nothing to build")
    operands = {getattr(p.demand, "in_fmt", None) for p in on_dev}
    weights = {getattr(p.demand, "weight_fmt", None) or getattr(p.demand, "in_fmt", None) for p in on_dev}
    accums = {p.acc for p in on_dev}
    if len(operands) != 1 or len(weights) != 1 or len(accums) != 1:
        raise ValueError(
            f"{device!r} placements disagree about the datapath (operands={sorted(map(str, operands))}, "
            f"accumulate={sorted(map(str, accums))}); one image carries one device datapath, so the "
            f"placement has to be split before it can be built"
        )
    operand, weight, accum = operands.pop(), weights.pop(), accums.pop()
    if not operand:
        raise ValueError(
            f"{device!r} placements carry no operand format; the kernel precision is "
            f"underivable and assuming one emits the wrong datapath"
        )
    accum = accum or _accum_from_facts(device, operand, weight)
    return DeviceRouting(
        device=device,
        package_dir=package_dir,
        operand_dtype=str(operand),
        accum_dtype=str(accum),
        select=device_selector(placement),
        numeric_policy=numeric_policy,
        granularity=str(granularity),
        capture=capture,
        model=str(model),
        linked_elf_admission=linked_elf_admission,
    )


def _accum_from_facts(device: str, operand: str, weight: str) -> str:
    """The accumulate format this device's RTL datapath declares for ``operand`` x ``weight``.

    Fails closed in both directions. No matching triple means the device does not declare what it
    accumulates this pair into, and more than one means it declares several -- and a build that picked
    among them would be choosing the model's arithmetic on the strength of dictionary order.
    """
    from merlin.system.offload import device_dtype_triples
    from merlin.targetgen.routing import _fmt_ok  # noqa: PLC2701 -- one format-equality predicate

    found = {a for i, w, a in device_dtype_triples(device) if _fmt_ok(operand, (i,)) and _fmt_ok(weight, (w,))}
    if len(found) != 1:
        raise ValueError(
            f"{device!r} declares {len(found)} accumulate format(s) for {operand} x {weight} "
            f"({sorted(found) or 'none'}); the unit matched no accumulate rule either, so the kernel "
            f"precision is underivable and assuming one emits the wrong datapath"
        )
    return found.pop()
