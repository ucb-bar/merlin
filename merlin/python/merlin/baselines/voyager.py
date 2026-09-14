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

* ``weight_buffer_size`` = array rows. A weight-stationary array holds exactly one array-sized weight
  block in its PEs, so the resident-weight capacity per output column is one element per PE row. A
  mapping that needs more (a 3x3 convolution keeps nine blocks) has no equivalent on the target, and
  Voyager's own tiler refusing it is the honest result.
* ``input_buffer_size`` = scratchpad rows. The array streams activations straight from the scratchpad
  on every compute, so the scratchpad IS the input-side L1.
* ``accum_buffer_size`` = accumulator rows. One accumulator row holds one array-edge of partial sums.

Voyager's own scheduling choices (double-buffering the L2, its runtime tolerance) are left at its
defaults: they are the compiler under test, not facts about the hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DerivedConfig", "VoyagerConfigError", "accelerator_config_for"]


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
        raise VoyagerConfigError(f"{target}: {what} is not derivable from the target's facts; "
                                 "refusing to hand Voyager a guessed machine")
    return value


def accelerator_config_for(target: str, *, facts: dict[str, Any] | None = None) -> DerivedConfig:
    """Derive Voyager's ``AcceleratorConfig`` for ``target`` from its address space.

    ``facts`` overrides the facts artifact (for tests); otherwise it is read through
    ``load_facts(target)``. Raises :class:`VoyagerConfigError` if any required field is UNKNOWN.
    """
    from ..targetgen.address_space import derive_address_space

    space = derive_address_space(target, facts=facts)
    rows = _need(getattr(space, "array_rows", None), "the array row count", target)
    cols = _need(getattr(space, "array_cols", None), "the array column count", target)
    def store(name: str):
        try:
            found = space.store(name)
        except (KeyError, LookupError, ValueError) as exc:
            raise VoyagerConfigError(f"{target}: no {name!r} store in the derived address space "
                                     f"({exc})") from exc
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
        "weight_buffer_size": int(rows),
        "accum_buffer_size": int(_need(acc.total_rows, "accumulator row count", target)),
    }
    sources = {
        "pe_array_size": f"address space array {getattr(space, 'array_name', None)!r} "
                         f"({rows}x{cols})",
        "scratchpad_size": f"store 'scratchpad'.nbytes ({spad.sources.get('bytes_depth', '?')})",
        "num_banks": "store 'scratchpad'.banks = total_rows / per-bank depth",
        "bank_width": f"store 'scratchpad'.row_bytes ({spad.sources.get('row_bytes', '?')})",
        "input_buffer_size": "store 'scratchpad'.total_rows: activations stream from the scratchpad",
        "weight_buffer_size": "array rows: the PE array holds one array-sized weight block",
        "accum_buffer_size": "store 'accumulator'.total_rows: one row = one array edge of partials",
    }
    not_modelled = {
        "dram_bandwidth": "Voyager default (64 GB/s): the target's DMA width is not in its derived "
                          "address space, so Voyager's DRAM term prices a machine we did not state",
        "dram_access_latency": "Voyager default (100 ns)",
        "double_buffered_l2": "Voyager default (True): a scheduling choice of the compiler under test",
    }
    return DerivedConfig(target=target, fields=fields, sources=sources, not_modelled=not_modelled)
