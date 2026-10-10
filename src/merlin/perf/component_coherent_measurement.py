"""Bounded original harness bytes and raw samples, without measurement roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from merlin.common.quant_formats import get
from merlin.runtime.backends.base import decode_float_readback
from merlin.runtime.direct_kernel_harness import _layout, _raw
from merlin.targetgen.contract.prepared_process_readback import PreparedProcessReadbackPlan


@dataclass(frozen=True)
class CoherentMeasurementPlan:
    """Explicit ordered, unframed object bytes from the original typed plans.

    The selected reader/process still owns ELF resolution and readback. This
    fixed grammar only splits the declared bytes; it does not authenticate the
    observer or establish actual device completion, clocks or elapsed costs.
    """

    readback: PreparedProcessReadbackPlan
    max_console_bytes: int

    def record(self):
        if (
            type(self) is not CoherentMeasurementPlan
            or type(self.readback) is not PreparedProcessReadbackPlan
            or self.readback.counter_plan is None
            or self.readback.phase_plan is None
            or self.readback.frame_bytes != 0
            or type(self.max_console_bytes) is not int
            or not 1 <= self.max_console_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("coherent measurement requires complete typed plans and bounded unframed bytes")
        readback = self.readback.record()
        if self.readback.invocation_plan.count * len(self.readback.invocation_plan.original_abi.outputs) > 1024:
            raise ValueError("coherent measurement per-call metadata exceeds its complete original member budget")
        return {
            "schema": "merlin.coherent_measurement_plan.v1",
            "readback": readback,
            "max_console_bytes": self.max_console_bytes,
            "grammar": "original ordered objects concatenated without framing",
            "scope": "raw harness observations only; no timing, cost or qualification",
        }

    def decode(self, payload, *, cb, inputs):
        """Recheck the full roster before expanding original scalar values."""
        self.record()
        selected = self.readback
        roster = selected.bind(cb)
        if type(payload) is not bytes or len(payload) != sum(extent for _, extent in roster):
            raise ValueError("coherent measurement omitted complete original object bytes")
        objects, offset = {}, 0
        for name, extent in roster:
            objects[name] = payload[offset : offset + extent]
            offset += extent
        outputs = {}
        for index, argument in enumerate(cb["kernel_abi"]["args"]):
            name = argument["tensor"]
            spec = cb["tensors"][name]
            raw = objects["tensor_" + str(index)]
            if argument["access"] == "read":
                count, width, dtype = _layout(spec)
                expected = _raw(
                    spec, inputs.get(name), count=count, width=width, dtype=dtype, byte_order=selected.abi.byte_order
                )
                if raw != expected:
                    raise ValueError("coherent measurement changed original immutable input bytes")
            else:
                outputs[name] = tensor_values(raw, spec=spec, byte_order=selected.abi.byte_order)
        tensor_symbols = {"tensor_" + str(index) for index in range(len(cb["kernel_abi"]["args"]))}
        phases = selected.phase_plan.decode(
            {name: raw for name, raw in objects.items() if name not in tensor_symbols},
            cb=cb,
            abi=selected.abi,
            invocation_plan=selected.invocation_plan,
        )
        histories = selected.invocation_plan.bind(
            cb, entry_symbol=selected.abi.entry_symbol, completion_symbol=selected.abi.completion_symbol
        )
        by_name = {row.original_name: row for row in histories}
        calls = [{} for _ in range(phases.invocations.observed_count)]
        for name, snapshots in phases.invocations.output_bytes:
            history = by_name[name]
            for call, raw in zip(calls, snapshots, strict=True):
                call[history.emitted_tensor] = tensor_values(
                    raw, spec={"dtype": history.dtype, "shape": list(history.shape)}, byte_order=selected.abi.byte_order
                )
        return {
            "schema": "merlin.coherent_measurement_observation.v1",
            "plan": self.record(),
            "objects": [{"symbol": name, "bytes": extent, "hex": objects[name].hex()} for name, extent in roster],
            "object_bytes": len(payload),
            "outputs": outputs,
            "call_outputs": calls,
            "completed_count": phases.invocations.observed_count,
            "counter_samples": asdict(phases.counters),
            "harness_phases": dict(phases.phases),
            "cold": None,
            "warm": None,
            "unknown": list(phases.unknown),
            "scope": phases.scope,
        }


def scalar_values(raw, *, dtype, byte_order):
    """Decode only byte-aligned, unscaled original scalar storage."""
    fmt = get(dtype)
    if fmt.element_bits not in (8, 16, 32, 64) or fmt.scale.kind != "none" or byte_order not in ("little", "big"):
        raise ValueError("coherent measurement has unsupported original scalar storage")
    width = fmt.element_bits // 8
    if type(raw) is not bytes or not raw or len(raw) % width:
        raise ValueError("coherent measurement has incomplete original scalar bytes")
    values = [
        int.from_bytes(raw[index : index + width], byte_order, signed=fmt.signed and not fmt.is_float)
        for index in range(0, len(raw), width)
    ]
    return decode_float_readback({"value": values}, {"value": dtype})["value"]


def tensor_values(raw, *, spec, byte_order):
    values = scalar_values(raw, dtype=spec["dtype"], byte_order=byte_order)
    width = spec["shape"][-1]
    return [values[index : index + width] for index in range(0, len(values), width)]
