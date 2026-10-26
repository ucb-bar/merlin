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


_GROUPED: list[Any] = []
BLOCK_SCALE_REDUCTIONS = ("max_abs", "mean_abs")


def grouped_channel_observer() -> Any:
    """Per-output-channel weight scales shared by each run of ``channels_per_scale`` channels.

    A readout that applies one scale to a group of output columns can hold a per-channel scale only
    when it is uniform within each group. This observer takes each channel's symmetric range as the
    largest range in its group by default (``max_abs``). The explicit ``mean_abs`` policy averages
    channel ranges and can clip outlier channels. The last group may be shorter.
    """
    if _GROUPED:
        return _GROUPED[0]
    import torch
    from torchao.quantization.pt2e import PerChannelMinMaxObserver

    class GroupedChannelObserver(PerChannelMinMaxObserver):
        def __init__(self, channels_per_scale: int = 1, block_scale_reduction: str = "max_abs", **kwargs: Any) -> None:
            if type(channels_per_scale) is not int or channels_per_scale < 1:
                raise ValueError("channels_per_scale must be a positive integer")
            if block_scale_reduction not in BLOCK_SCALE_REDUCTIONS:
                raise ValueError(f"block_scale_reduction must be one of {BLOCK_SCALE_REDUCTIONS}")
            super().__init__(**kwargs)
            if self.qscheme not in (torch.per_channel_symmetric,):
                raise ValueError("grouped channel scales are symmetric per-channel scales")
            self.channels_per_scale = channels_per_scale
            self.block_scale_reduction = block_scale_reduction

        @torch.jit.export
        def calculate_qparams(self) -> Any:
            low, high = self.min_val, self.max_val
            magnitude = torch.maximum(low.abs(), high.abs())
            n = self.channels_per_scale
            shared = magnitude.clone()
            for start in range(0, magnitude.numel(), n):
                block = magnitude[start : start + n]
                shared[start : start + n] = block.max() if self.block_scale_reduction == "max_abs" else block.mean()
            return self._calculate_qparams(-shared, shared)

        def extra_repr(self) -> str:
            return f"channels_per_scale={self.channels_per_scale}, block_scale_reduction={self.block_scale_reduction}"

    _GROUPED.append(GroupedChannelObserver)
    return GroupedChannelObserver


class RepeatedObservationSkip:
    """Skip a min/max observer's call when its input is the exact tensor it observed last.

    A stored weight reaches its observer as the same parameter on every calibration sample, and
    TorchAO's ``MinMaxObserver`` and ``PerChannelMinMaxObserver`` keep only a running min and max:
    a second observation of the same values leaves both buffers bit-identical. Those two exact
    classes and the known grouped observer with its inherited min/max forward are covered; arbitrary
    subclasses, moving averages and histograms are not. The skip requires the
    same tensor object, still alive, with the same version counter, storage, shape, strides and
    dtype, so an in-place update or a fresh activation is always observed. The instance-level
    ``forward`` is removed again when calibration ends.
    """

    def __init__(self, prepared: Any) -> None:
        from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver

        grouped = grouped_channel_observer()
        known = (MinMaxObserver, PerChannelMinMaxObserver)
        if grouped.forward is PerChannelMinMaxObserver.forward:
            known += (grouped,)  # only qparam grouping differs; the observed min/max update is identical
        self._modules = [module for module in prepared.modules() if type(module) in known]

    @staticmethod
    def _key(x: Any) -> tuple:
        return (x._version, x.data_ptr(), tuple(x.shape), x.stride(), x.dtype, x.device)

    def __enter__(self) -> RepeatedObservationSkip:
        import weakref

        for module in self._modules:
            observe = type(module).forward.__get__(module)
            last: list[Any] = [None]

            def forward(x: Any, _observe: Any = observe, _last: list[Any] = last) -> Any:
                seen = _last[0]
                if seen is not None and seen[0]() is x and seen[1] == self._key(x):
                    return x
                result = _observe(x)
                _last[0] = (weakref.ref(x), self._key(x))
                return result

            module.forward = forward
        return self

    def __exit__(self, *exc: Any) -> None:
        for module in self._modules:
            module.__dict__.pop("forward", None)
