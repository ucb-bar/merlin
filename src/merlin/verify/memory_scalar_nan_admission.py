"""Scoped NaN/domain admission with bounded dense-literal parsing.

No producer, numeric meaning, generic parser default or exporter is changed.
Only literal metadata is inspected; runtime values are outside this guard.
"""

from __future__ import annotations

import math

from xdsl.dialects import builtin

from .gate_errors import GateUnavailable
from .lean_graph_input_admission import InputAdmissionError, _attributes, _fields

MAX_DENSE_FLOAT_BYTES = 256 * 1024
MAX_LOOP_DENSE_BYTES = 4 * 1024 * 1024


def _parse_bounded(raw, *, maximum_bytes, memory):
    """Limit actual dense literal storage before the parent repeats a splat.

    The local class is part of this function's frozen code constants. Ordinary
    tensor signatures are never inspected by these dense-attribute-only hooks.
    """
    from xdsl.context import Context
    from xdsl.dialects import arith, func, memref, scf

    from merlin.common.dense_float_ir import DenseFloatParser

    class BoundedParser(DenseFloatParser):
        def __init__(self, *args, **kwargs):
            self.remaining_dense_bytes = maximum_bytes
            super().__init__(*args, **kwargs)

        def _charge_dense_type(self, shaped):
            element = shaped.get_element_type()
            if (
                memory
                and isinstance(element, builtin.ComplexType)
                and isinstance(element.element_type, builtin.AnyFloat)
            ):
                raise InputAdmissionError("complex-floating-dense", "parsed-dense-literal-type")
            width = element.compile_time_size
            if type(width) is not int or width < 1:
                raise GateUnavailable("dense literal element storage width is unavailable")
            shape = list(shaped.get_shape())
            if any(type(extent) is not int or extent < 0 for extent in shape):
                raise GateUnavailable("dense literal needs a static bounded storage shape")
            count = 0 if 0 in shape else 1
            if count:
                for extent in shape:
                    if extent > self.remaining_dense_bytes // width // count:
                        raise GateUnavailable("dense literal decoded-storage resource budget exceeded")
                    count *= extent
            required = count * width
            if required > self.remaining_dense_bytes:
                raise GateUnavailable("dense literal decoded-storage resource budget exceeded")
            self.remaining_dense_bytes -= required

        def _parse_dense_literal_type(self):
            shaped = super()._parse_dense_literal_type()
            self._charge_dense_type(shaped)
            return shaped

        def parse_dense_int_or_fp_elements_attr(self, type):
            if type is not None:
                self._charge_dense_type(type)
            return super().parse_dense_int_or_fp_elements_attr(type)

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    module = BoundedParser(context, raw).parse_module()
    module.verify()
    return module


def parse_memory_module(raw):
    """Original memory profile: 256KiB aggregate decoded dense literal bytes."""
    return _parse_bounded(raw, maximum_bytes=MAX_DENSE_FLOAT_BYTES, memory=True)


def parse_loop_module(raw):
    """Scalar-loop profile: 4MiB aggregate dense bytes; exact dense is retained."""
    return _parse_bounded(raw, maximum_bytes=MAX_LOOP_DENSE_BYTES, memory=False)


def _admit_nan_attributes(module, *, reject_dense_nan):
    """Reuse the structural visitor; inspect stored bytes without expansion."""
    for index, operation in enumerate(module.walk()):
        for value, field in _fields(operation):
            for attribute, path in _attributes(value, f"operation[{index}].{field}"):
                if isinstance(attribute, builtin.FloatData) and math.isnan(attribute.data):
                    raise InputAdmissionError("scalar-nan", path)
                if not reject_dense_nan:
                    continue
                if isinstance(attribute, builtin.DenseIntOrFPElementsAttr):
                    element = attribute.get_element_type()
                    if isinstance(element, builtin.ComplexType) and isinstance(element.element_type, builtin.AnyFloat):
                        raise InputAdmissionError("complex-floating-dense", path)
                elif isinstance(attribute, builtin.DenseArrayBase):
                    element = attribute.elt_type
                else:
                    continue
                if not isinstance(element, builtin.AnyFloat):
                    continue
                if len(attribute.data.data) > MAX_DENSE_FLOAT_BYTES:
                    raise GateUnavailable("dense float stored-payload resource budget exceeded")
                try:
                    # Both iter_values implementations unpack actual raw bytes;
                    # get_values may use a shaped count and is never called.
                    if any(math.isnan(value) for value in attribute.iter_values()):
                        raise InputAdmissionError("dense-nan", path)
                except (NotImplementedError, ValueError, OverflowError) as error:
                    if isinstance(error, InputAdmissionError):
                        raise
                    raise InputAdmissionError("dense-float-encoding", path) from error


def admit_memory_scalar_attributes(module):
    """Memory metadata currently loses scalar and dense NaN payload bits."""
    _admit_nan_attributes(module, reject_dense_nan=True)


def admit_loop_scalar_attributes(module):
    """Reject scalar parsing loss; retain the loop profile's exact dense bytes."""
    _admit_nan_attributes(module, reject_dense_nan=False)
