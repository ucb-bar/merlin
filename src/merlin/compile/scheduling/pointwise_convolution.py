"""Exact contiguous NHWC pointwise views for the existing dense scheduler.

This changes only logical indexing: it supplies no target adapter, memory
assignment, arithmetic/readout permission or completion law. The source owner
retains original dtypes, reduction order, bias and every output coefficient.
Strided or padded spatial gathers keep their existing convolution route.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import prod

from .window_transfer import Window2D


def _identity(window, columns):
    return hashlib.sha256(json.dumps((asdict(window), columns), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PointwiseDenseView:
    """Source-bound no-copy row-major views, independent of primitive geometry.

    Input offset ``((b*H+y)*W+x)*CI+c`` equals matrix offset
    ``m*CI+c`` for ``m=(b*H+y)*W+x``. RHS retains CI-major/CO-minor
    order and output uses the same flattened spatial row. No payload is read,
    allocated or packed. Runtime ownership and external nonalias remain caller
    obligations, as for the original window source.
    """
    window: Window2D
    output_channels: int
    _source_identity: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_source_identity', _identity(self.window, self.output_channels))

    def _validate(self):
        if type(self.window) is not Window2D:
            raise ValueError('pointwise view requires the original typed NHWC window')
        self.window.__post_init__()
        if (self.window.kernel != (1, 1) or self.window.stride != (1, 1)
                or self.window.padding != (0, 0, 0, 0)):
            raise ValueError('pointwise dense view requires unpadded unit-stride 1x1 source')
        if type(self.output_channels) is not int or self.output_channels <= 0:
            raise ValueError('pointwise view requires positive original output channels')
        if prod(self.window.input_shape[:3])*self.output_channels >= 2**63:
            raise ValueError('pointwise output exceeds signed 64-bit element indexing')

    def revalidate(self):
        self._validate()
        if _identity(self.window, self.output_channels) != self._source_identity:
            raise ValueError('original pointwise source changed; derive a new view')

    @property
    def dimensions(self):
        """M,K,N for DenseProblem; primitive/resource facts are still separate."""
        self.revalidate()
        return prod(self.window.input_shape[:3]), self.window.input_shape[-1], self.output_channels

    @property
    def matrix_shapes(self):
        m, k, n = self.dimensions
        return (m, k), (k, n), (m, n)

    def record(self):
        self.revalidate()
        return dict(schema='merlin.pointwise_dense_view.v1', source_identity=self._source_identity,
                    window=asdict(self.window), dimensions=self.dimensions,
                    matrix_shapes=self.matrix_shapes, payload_copies=0,
                    reduction_order='original input-channel order',
                    qualification='logical indexing only; original arithmetic, readout and target obligations remain')


def pointwise_dense_view(window, output_channels, *, output_shape):
    """Return a checked exact matrix view, or None for a spatial-gather source.

    ``output_shape`` must be the current source's flattened or NHWC result,
    not a requested new shape. Bad source geometry refuses. Unsupported valid
    kernel/stride/padding returns None for the original convolution fallback.
    Dilation has no effect when the kernel is 1x1 and needs no special case.
    """
    if type(window) is not Window2D:
        raise ValueError('pointwise view requires the original typed NHWC window')
    window.__post_init__()
    if type(output_channels) is not int or output_channels <= 0:
        raise ValueError('pointwise view requires positive original output channels')
    oh, ow = window.output_spatial
    batch = window.input_shape[0]
    if (type(output_shape) is not tuple or any(type(d) is not int or d <= 0 for d in output_shape)
            or output_shape not in (
            (batch*oh*ow, output_channels), (batch, oh, ow, output_channels))):
        raise ValueError('pointwise result shape differs from the original convolution')
    if window.kernel != (1, 1) or window.stride != (1, 1) or window.padding != (0, 0, 0, 0):
        return None
    return PointwiseDenseView(window, output_channels)
