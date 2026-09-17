"""The Voyager compiler as an external baseline: its machine model, derived from a target's facts.

Voyager (github.com/jeffreyyu0602/voyager-compiler) sizes every tile against an ``AcceleratorConfig``
-- PE array, per-edge L1 buffers, a banked L2 scratchpad. To compile for a target it was not built
for, that config has to describe the TARGET, and a hand-typed one silently decides the comparison: a
halved scratchpad or an accumulator stated in bytes where Voyager reads elements hands it a different
machine. So nothing here is typed in. Every field is read from the target's own derived address space
(:func:`merlin.targetgen.address_space.derive_address_space`, itself read from the RTL facts), each
value carries the fact it came from, and a field that cannot be derived is refused -- never defaulted.

Where Voyager's machine model has a structure the target does not (a per-PE L1 input or weight buffer),
the mapping is a DERIVATION with a stated rule, not a tuning knob:

* ``input_buffer_size`` = scratchpad rows. The array streams activations straight from the scratchpad
  on every compute, so the scratchpad IS the input-side L1.
* ``weight_buffer_size`` = scratchpad rows (``weight_residency="scratchpad"``, the default). Weights
  also reach the array from the scratchpad (a preload), so by the same rule the scratchpad is the
  weight-side L1. The one array-sized block the PEs hold is Voyager's interstellar LEVEL 0 (the PE
  partition), which its model already carries -- counting it again as L1 was the first reading here,
  and it made every 3x3 convolution unmappable (nine resident blocks against one). That reading is
  kept as ``weight_residency="pe_block"``, a named sensitivity variant, never the default: the
  comparison must not let a machine-model choice of ours decide that Voyager fails.
* ``accum_buffer_size`` = accumulator rows. One accumulator row holds one array-edge of partial sums.

Voyager's own scheduling choices (double-buffering the L2, its runtime tolerance) are left at its
defaults: they are the compiler under test, not facts about the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DerivedConfig", "VoyagerConfigError", "accelerator_config_for", "geometry_for"]


class VoyagerConfigError(ValueError):
    """A field of Voyager's machine model could not be derived from the target's facts."""


@dataclass(frozen=True)
class DerivedConfig:
    """``AcceleratorConfig`` keyword arguments plus, per field, the fact it was derived from."""

    target: str
    fields: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    not_modelled: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """The ``--config`` document ``voyager_export.py`` consumes (AcceleratorConfig fields)."""
        return dict(self.fields)


def _need(value: Any, what: str, target: str) -> Any:
    if value is None:
        raise VoyagerConfigError(
            f"{target}: {what} is not derivable from the target's facts; refusing to hand Voyager a guessed machine"
        )
    return value


WEIGHT_RESIDENCY = ("scratchpad", "pe_block")


def accelerator_config_for(
    target: str, *, facts: dict[str, Any] | None = None, weight_residency: str = "scratchpad"
) -> DerivedConfig:
    """Derive Voyager's ``AcceleratorConfig`` for ``target`` from its address space.

    ``facts`` overrides the facts artifact (for tests); otherwise it is read through
    ``load_facts(target)``. ``weight_residency`` selects the reading of the weight-side L1 (see the
    module docstring); only the default is a headline configuration. Raises
    :class:`VoyagerConfigError` if any required field is UNKNOWN.
    """
    from ..targetgen.address_space import derive_address_space

    if weight_residency not in WEIGHT_RESIDENCY:
        raise VoyagerConfigError(f"weight_residency {weight_residency!r} not in {WEIGHT_RESIDENCY}")

    space = derive_address_space(target, facts=facts)
    rows = _need(getattr(space, "array_rows", None), "the array row count", target)
    cols = _need(getattr(space, "array_cols", None), "the array column count", target)

    def store(name: str):
        try:
            found = space.store(name)
        except (KeyError, LookupError, ValueError) as exc:
            raise VoyagerConfigError(f"{target}: no {name!r} store in the derived address space ({exc})") from exc
        if found is None:
            raise VoyagerConfigError(f"{target}: no {name!r} store in the derived address space")
        return found

    spad, acc = store("scratchpad"), store("accumulator")
    fields = {
        "pe_array_size": [int(rows), int(cols)],
        "scratchpad_size": int(_need(spad.nbytes, "scratchpad bytes", target)),
        "num_banks": int(_need(spad.banks, "scratchpad bank count", target)),
        "bank_width": int(_need(spad.row_bytes, "scratchpad row width", target)),
        "input_buffer_size": int(_need(spad.total_rows, "scratchpad row count", target)),
        "weight_buffer_size": (int(spad.total_rows) if weight_residency == "scratchpad" else int(rows)),
        "accum_buffer_size": int(_need(acc.total_rows, "accumulator row count", target)),
    }
    sources = {
        "pe_array_size": f"address space array {getattr(space, 'array_name', None)!r} ({rows}x{cols})",
        "scratchpad_size": f"store 'scratchpad'.nbytes ({spad.sources.get('bytes_depth', '?')})",
        "num_banks": "store 'scratchpad'.banks = total_rows / per-bank depth",
        "bank_width": f"store 'scratchpad'.row_bytes ({spad.sources.get('row_bytes', '?')})",
        "input_buffer_size": "store 'scratchpad'.total_rows: activations stream from the scratchpad",
        "weight_buffer_size": (
            "store 'scratchpad'.total_rows: weights are preloaded from the scratchpad (weight_residency=scratchpad)"
            if weight_residency == "scratchpad"
            else "array rows: only the PE-resident block counts as L1 (weight_residency=pe_block, sensitivity variant)"
        ),
        "accum_buffer_size": "store 'accumulator'.total_rows: one row = one array edge of partials",
    }
    not_modelled = {
        "dram_bandwidth": "Voyager default (64 GB/s): the target's DMA width is not in its derived "
        "address space, so Voyager's DRAM term prices a machine we did not state",
        "dram_access_latency": "Voyager default (100 ns)",
        "double_buffered_l2": "Voyager default (True): a scheduling choice of the compiler under test",
    }
    return DerivedConfig(target=target, fields=fields, sources=sources, not_modelled=not_modelled)


def geometry_for(target: str, *, facts: dict[str, Any] | None = None):
    """The :class:`merlin.baselines.voyager_schedule.Geometry` a lowered Voyager schedule is laid out
    on, from the same derived address space as :func:`accelerator_config_for` -- so the machine
    Voyager planned for and the rows the bridge addresses are one set of facts."""
    from .voyager_schedule import Geometry

    derived = accelerator_config_for(target, facts=facts)
    f = derived.fields
    rows, cols = f["pe_array_size"]
    if rows != cols:
        raise VoyagerConfigError(f"{target}: a {rows}x{cols} array has no single block edge")
    return Geometry(
        dim=int(rows),
        spad_rows=int(f["input_buffer_size"]),
        spad_row_bytes=int(f["bank_width"]),
        acc_rows=int(f["accum_buffer_size"]),
    )
