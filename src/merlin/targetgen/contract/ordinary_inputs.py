"""Integer stimulus for an ordinary program's logical harness, from the runner's own inputs.

Split from ``compile`` (which re-exports it as ``_ordinary_integer_harness_inputs``): the program's
existing integer inputs are materialized, never a model fallback, and payloads above the inline
size travel through the exact-byte sidecar transport when the caller streams inputs.
"""

from __future__ import annotations


def ordinary_integer_harness_inputs(cb, *, max_storage_bytes=16 << 20, stream_inputs=False):
    """Materialize the runner's existing integer stimulus, never a model fallback."""
    from merlin.common.quant_formats import get
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.runtime.harness_render import _roster, _tensor

    if cb.get("kernel_abi") is not None:
        raise ValueError("whole-program harness requires explicit semantic inputs")
    normalized, slots = _roster(cb)
    total = 0
    for spec in normalized["tensors"].values():
        total += _tensor(spec)[2]
        if get(spec["dtype"]).kind != "int_affine":
            raise ValueError("non-integer harness requires explicit semantic inputs")
    if type(max_storage_bytes) is not int or not 0 < max_storage_bytes <= 256 << 20:
        raise ValueError("ordinary integer harness needs a bounded selected storage budget")
    if total > max_storage_bytes:
        raise ValueError("neutral harness exceeds its complete storage budget before input rendering")
    leaves = materialize_inputs(cb)
    # Preserve existing small-program harness bytes. Previously refused large
    # programs use the existing exact-byte sidecar transport, never megabytes
    # of C scalar literals or a changed stimulus distribution.
    if total > 16 << 20 and stream_inputs:
        import numpy as np

        from merlin.llvmlower.weight_payload import WeightPayload
        from merlin.runtime.tensor_payload import TensorPayload

        payloads = {}
        for name, access in slots.items():
            if access == "write":
                continue
            spec = normalized["tensors"][name]
            shape, dtype, extent = _tensor(spec)
            width = dtype.element_bits // 8
            signed = dtype.signed
            minimum = -(1 << (width * 8 - 1)) if signed else 0
            maximum = (1 << (width * 8 - int(signed))) - 1
            values = leaves[name].data
            if any(type(value) is not int or not minimum <= value <= maximum for value in values):
                raise ValueError("ordinary integer payload differs from its exact scalar storage")
            array = np.asarray(values, dtype=f"<{'i' if signed else 'u'}{width}")
            array.flags.writeable = False
            payload = WeightPayload.array(extent, lambda array=array: array)
            payloads[name] = TensorPayload(shape, spec["dtype"], payload)
        return payloads
    return {name: leaves[name].to_list() for name, access in slots.items() if access != "write"}
