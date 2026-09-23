"""Read-only xDSL inspection serialization; never executable compiler input."""

from __future__ import annotations

import json
from io import StringIO


def reconstruct_tensor(directory, descriptor, *, tensor_type):
    """Reconstruct dense storage in an explicitly supplied static, unencoded tensor.

    Existing inspection descriptors do not identify the original shaped container
    or encoding. This therefore reconstructs only the requested supported tensor,
    not an original operation/module or an executable inspection view. Raw storage
    is xDSL's representation, not a portable numeric interchange format.
    """
    from math import prod

    from xdsl.dialects.builtin import (
        BFloat16Type,
        BytesAttr,
        DenseIntOrFPElementsAttr,
        Float16Type,
        Float32Type,
        Float64Type,
        IntegerType,
        NoneAttr,
        TensorType,
    )

    from merlin.common.ir_audit import read_tensor_payload

    payload = read_tensor_payload(directory, descriptor)
    if not isinstance(tensor_type, TensorType) or not isinstance(tensor_type.encoding, NoneAttr):
        raise ValueError("reconstruction requires a static unencoded TensorType")
    shape = tensor_type.get_shape()
    element = tensor_type.get_element_type()
    if any(dim < 0 for dim in shape):
        raise ValueError("reconstruction requires a static unencoded TensorType")
    if isinstance(element, IntegerType):
        if element.width.data not in (8, 16, 32, 64):
            raise ValueError("unsupported inspection tensor element type")
    elif not isinstance(element, (Float16Type, BFloat16Type, Float32Type, Float64Type)):
        raise ValueError("unsupported inspection tensor element type")
    if list(shape) != descriptor["shape"] or str(element) != descriptor["element_type"]:
        raise ValueError("requested tensor type differs from inspection descriptor")
    if len(payload) != prod(shape) * element.compile_time_size:
        raise ValueError("inspection tensor storage does not match its static type")
    return DenseIntOrFPElementsAttr(tensor_type, BytesAttr(payload))


def compact_text(module, *, elements_limit: int = 64, tensor_sink=None) -> str:
    """Generic operation syntax with large dense tensor attributes elided structurally.

    Custom operation printers may bypass attribute dispatch, so inspection deliberately
    uses generic syntax. This is a diagnostic view, not a round-trippable module or
    executable tensor externalization. An optional sink preserves original dense storage
    bytes for inspection alongside the view. Neither attributes nor operations are modified.
    """
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    from xdsl.printer import Printer

    if elements_limit < 0:
        raise ValueError("inspection element limit must be nonnegative")

    class InspectionPrinter(Printer):
        def print_attribute(self, attribute):
            if isinstance(attribute, DenseIntOrFPElementsAttr) and len(attribute) > elements_limit:
                if tensor_sink is None:
                    self.print_string("dense<...> : ")
                else:
                    descriptor = tensor_sink(
                        attribute.data.data,
                        element_type=str(attribute.get_element_type()),
                        shape=tuple(attribute.get_shape()),
                    )
                    self.print_string(
                        f"inspection_tensor<file={json.dumps(descriptor['file'])}, "
                        f"sha256={json.dumps(descriptor['sha256'])}> : "
                    )
                self.print_attribute(attribute.get_type())
            else:
                super().print_attribute(attribute)

    output = StringIO()
    storage = "referenced as raw payloads" if tensor_sink is not None else "elided"
    output.write(f"// xDSL generic inspection; dense tensors above {elements_limit} elements are {storage}.\n")
    InspectionPrinter(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def record_stage(audit, name: str, module) -> None:
    """Serialize only when requested, using the shared audit's exact-byte attribution."""
    if audit.directory is None:
        return
    from ._common import text

    tensors = []

    def tensor_sink(payload, *, element_type, shape):
        descriptor = audit.tensor(payload, element_type=element_type, shape=shape)
        tensors.append(descriptor)
        return descriptor

    inspection = compact_text(module, tensor_sink=tensor_sink) if audit.mode in {"compact", "both"} else None
    audit.stage(
        name,
        text(module),
        inspection=inspection,
        inspection_tensors=tensors if inspection is not None else None,
    )
