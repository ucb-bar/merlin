#!/usr/bin/env python3
"""Activation range observers a recipe may name beyond TorchAO's own -- runs INSIDE the capture venv.

A static recipe names its activation observer (``recipe["activation"]["observer"]``). TorchAO builds
``minmax`` and ``histogram``. This module adds the percentile observer, spelled ``p<percent>``
(``p99.999``): the symmetric range is that percentile of ``|x|`` over the whole calibration stream.
A min/max range is set by the single largest value ever seen; a percentile range ignores the rarest
values and gives every other value a finer step. The percentile is read from a histogram of
``log2 |x|`` with :data:`PER_OCTAVE` bins per octave between ``2**LOW`` and ``2**HIGH``, and the range
is the upper edge of the bin the percentile falls in, never above the largest value seen.

Nothing here imports merlin, names a model or a target; torch and TorchAO are imported lazily, so the
recipe side can validate a name without them.
"""

from __future__ import annotations

from typing import Any

#: The histogram of ``log2 |x|``: octaves ``LOW .. HIGH``, :data:`PER_OCTAVE` bins each (2.2% wide).
LOW, HIGH, PER_OCTAVE = -40, 20, 32


def percentile_of(name: Any) -> float | None:
    """The percentile a ``p<percent>`` observer name selects; ``None`` when ``name`` is not one."""
    if not isinstance(name, str) or not name.startswith("p"):
        return None
    try:
        value = float(name[1:])
    except ValueError:
        return None
    return value if 0.0 < value <= 100.0 else None


_CLASS: list[Any] = []


def percentile_observer() -> Any:
    """The PT2E observer class (built once): ``.with_args(percentile=..., eps=...)`` like any other."""
    if _CLASS:
        return _CLASS[0]
    import torch
    from torchao.quantization.pt2e.observer import UniformQuantizationObserverBase

    bins = (HIGH - LOW) * PER_OCTAVE

    class PercentileObserver(UniformQuantizationObserverBase):
        """A symmetric per-tensor range at a percentile of ``|x|`` over every observed call."""

        def __init__(
            self,
            percentile: float = 100.0,
            dtype: Any = torch.int8,
            qscheme: Any = torch.per_tensor_symmetric,
            quant_min: int | None = None,
            quant_max: int | None = None,
            eps: float = torch.finfo(torch.float32).eps,
            **kwargs: Any,
        ) -> None:
            if qscheme != torch.per_tensor_symmetric:
                raise ValueError("a percentile range is a symmetric per-tensor range")
            super().__init__(dtype=dtype, qscheme=qscheme, quant_min=quant_min, quant_max=quant_max, eps=eps, **kwargs)
            self.percentile = float(percentile)
            self.register_buffer("histogram", torch.zeros(bins, dtype=torch.float64))
            self.register_buffer("absmax", torch.tensor(0.0))

        def forward(self, x: Any) -> Any:
            if x.numel() == 0:
                return x
            magnitude = x.detach().to(torch.float32).abs().reshape(-1)
            self.absmax = torch.maximum(self.absmax, magnitude.max())
            octaves = torch.log2(magnitude.clamp_min(2.0**LOW))
            self.histogram += torch.histc(octaves, bins=bins, min=LOW, max=HIGH).to(torch.float64)
            return x

        def clip(self) -> float:
            total = float(self.histogram.sum())
            if total == 0.0:
                return float(self.absmax)
            cumulative = torch.cumsum(self.histogram, 0) / total
            index = int(torch.searchsorted(cumulative, torch.tensor(self.percentile / 100.0, dtype=torch.float64)))
            edge = 2.0 ** (LOW + (min(index, bins - 1) + 1) / PER_OCTAVE)
            return min(edge, float(self.absmax))

        @torch.jit.export
        def calculate_qparams(self) -> Any:
            bound = torch.tensor(self.clip(), dtype=torch.float32)
            return self._calculate_qparams(-bound, bound)

        def extra_repr(self) -> str:
            return f"percentile={self.percentile}, absmax={float(self.absmax)}"

    _CLASS.append(PercentileObserver)
    return PercentileObserver
