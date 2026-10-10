"""Raw samples around fixed ordinary harness code, without complete-cost roles.

The renderer supplies the boundaries. No caller label can move work into them.
Accessors remain explicitly selected support; raw samples are not elapsed costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from merlin.common.quant_formats import get

from .direct_kernel_counter import DirectKernelCounterObservation, DirectKernelCounterPlan
from .direct_kernel_invocation import DirectKernelInvocationObservation, DirectKernelInvocationPlan

FIELDS = ("start", "end", "state_before", "state_after")
PHASES = ("main_body", "console_setup", "calibration_loop", "history_copy", "done_publication")
UNKNOWN = (
    "static_initialization_allocation_and_loader_startup",
    "return_exit_and_parent_readback",
    "outer_sample_storage_and_printing",
    "counter_units_ordering_and_integrity",
    "completion_and_device_synchronization_semantics",
    "cold_warm_and_complete_eleven_stage_costs",
    "observer_resource_and_physical_execution_domain",
    "runtime_stage_measurement_and_held_qualification",
)


@dataclass(frozen=True)
class DirectKernelPhaseObservation:
    """Complete selected samples and histories; not a feature or qualification."""

    phases: tuple[tuple[str, tuple[tuple[int, int, int, int], ...]], ...]
    counters: DirectKernelCounterObservation
    invocations: DirectKernelInvocationObservation
    unknown: tuple[str, ...] = UNKNOWN
    scope: str = "raw fixed harness boundaries only; no elapsed costs, cold/warm or qualification"


@dataclass(frozen=True)
class DirectKernelPhasePlan:
    """Opt-in complete object capture for an existing counter and repeat plan.

    The inclusive main bracket contains console initialization, calibration,
    calls/completion, their counter storage, output histories and DONE plus
    inner instrumentation. It ends before storing its own four samples and
    the final phase count, before exit/return and before external readback.
    Those excluded operations are UNKNOWN, never assumed free. History brackets
    include the original completed-call count stores, but not their own samples.
    Publication measures the existing DONE call, not the parent memory reader.
    """

    counter_plan: DirectKernelCounterPlan
    storage_prefix: str
    max_observation_bytes: int

    def record(self):
        from .direct_kernel_harness import _identifier

        if type(self) is not DirectKernelPhasePlan or type(self.counter_plan) is not DirectKernelCounterPlan:
            raise ValueError("harness phases require the exact selected counter plan")
        _identifier(self.storage_prefix)
        if type(self.max_observation_bytes) is not int or not 1 <= self.max_observation_bytes <= 4 * 1024 * 1024:
            raise ValueError("harness phases require an explicit bounded complete observation budget")
        return {
            "schema": "merlin.direct_kernel_phases.v1",
            "counter_plan": self.counter_plan.record(),
            "storage_prefix": self.storage_prefix,
            "max_observation_bytes": self.max_observation_bytes,
            "phases": list(PHASES),
            "boundary": "fixed generated harness code; inclusive main ends before outer sample storage and exit",
            "unknown": list(UNKNOWN),
        }

    def bind(self, cb, *, abi, invocation_plan):
        """Bound combined buffers/history/counter metadata before rendering bytes."""
        self.record()
        if type(invocation_plan) is not DirectKernelInvocationPlan:
            raise ValueError("harness phases require the original explicit repeated-call plan")
        counters = self.counter_plan.bind(cb, abi=abi, invocation_plan=invocation_plan)
        histories = invocation_plan.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
        roster = {
            self.storage_prefix + "_" + phase + "_" + field: (invocation_plan.count if phase == "history_copy" else 1)
            * 8
            for phase in PHASES
            for field in FIELDS
        }
        roster[self.storage_prefix + "_completed"] = 8
        total = sum(roster.values()) + sum(counters.values()) + 8 + sum(row.byte_extent for row in histories)
        for argument in cb["kernel_abi"]["args"]:
            spec = cb["tensors"][argument["tensor"]]
            dtype = get(spec["dtype"])
            if dtype.element_bits not in (8, 16, 32, 64) or dtype.is_block_scaled:
                raise ValueError("harness phases require byte-aligned scalar original buffers")
            total += math.prod(spec["shape"]) * (dtype.element_bits // 8)
        if total > self.max_observation_bytes:
            raise ValueError("harness phases exceed the complete buffer/history/counter observation budget")
        locals_ = {"phase_byte", *("phase_" + prefix + field for prefix in ("main_", "value_") for field in FIELDS)}
        reserved = {
            abi.entry_symbol,
            abi.completion_symbol,
            "main",
            "console_init",
            "htif_puts",
            "htif_exit",
            "invocation",
            "invocation_completed",
            "byte",
            "counter_index",
            *("counter_" + field for field in FIELDS),
            *("tensor_" + str(index) for index in range(len(cb["kernel_abi"]["args"]))),
            *counters,
            *(row.symbol for row in histories),
            invocation_plan.count_symbol,
        }
        selected = {self.counter_plan.counter_symbol, self.counter_plan.state_symbol}
        if set(roster) & (reserved | locals_ | selected) or locals_ & (reserved | selected):
            raise ValueError("harness phase symbols overlap original storage, accessors or generated locals")
        return roster

    def decode(self, objects, *, cb, abi, invocation_plan):
        """Retain order/wrap/control changes with exact original counts/histories."""
        roster = self.bind(cb, abi=abi, invocation_plan=invocation_plan)
        counters = self.counter_plan.bind(cb, abi=abi, invocation_plan=invocation_plan)
        histories = invocation_plan.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
        history_roster = {invocation_plan.count_symbol: 8, **{row.symbol: row.byte_extent for row in histories}}
        complete = roster | counters | history_roster
        if type(objects) is not dict or set(objects) != set(complete):
            raise ValueError("harness phase readback changed its complete original object roster")
        if any(type(objects[name]) is not bytes or len(objects[name]) != extent for name, extent in complete.items()):
            raise ValueError("harness phase readback omitted complete original object bytes")
        if int.from_bytes(objects[self.storage_prefix + "_completed"], abi.byte_order) != invocation_plan.count:
            raise ValueError("harness phase count differs from the selected original calls")
        phases = []
        for phase in PHASES:
            arrays = [objects[self.storage_prefix + "_" + phase + "_" + field] for field in FIELDS]
            phases.append(
                (
                    phase,
                    tuple(
                        tuple(int.from_bytes(raw[index : index + 8], abi.byte_order) for raw in arrays)
                        for index in range(0, len(arrays[0]), 8)
                    ),
                )
            )
        return DirectKernelPhaseObservation(
            tuple(phases),
            self.counter_plan.decode(
                {name: objects[name] for name in counters}, cb=cb, abi=abi, invocation_plan=invocation_plan
            ),
            invocation_plan.decode(
                {name: objects[name] for name in history_roster},
                cb=cb,
                entry_symbol=abi.entry_symbol,
                completion_symbol=abi.completion_symbol,
                byte_order=abi.byte_order,
            ),
        )

    def declarations(self, roster):
        return [
            *(f"volatile unsigned char {name}[{extent}]={{0}};" for name, extent in roster.items()),
        ]

    def begin(self, phase, indent):
        prefix = "phase_main_" if phase == "main_body" else "phase_value_"
        return [
            indent + f"{prefix}state_before={self.counter_plan.state_symbol}();",
            indent + f"{prefix}start={self.counter_plan.counter_symbol}();",
        ]

    def end(self, phase, indent):
        prefix = "phase_main_" if phase == "main_body" else "phase_value_"
        return [
            indent + f"{prefix}end={self.counter_plan.counter_symbol}();",
            indent + f"{prefix}state_after={self.counter_plan.state_symbol}();",
        ]

    def store(self, phase, *, index, byte_order, indent):
        prefix = "phase_main_" if phase == "main_body" else "phase_value_"
        offset = "phase_byte" if byte_order == "little" else "(7-phase_byte)"
        return [
            indent + "for(unsigned phase_byte=0;phase_byte<8;phase_byte++){",
            *(
                indent + f"  {self.storage_prefix}_{phase}_{field}[({index})*8+{offset}]="
                f"(unsigned char)({prefix}{field}>>(phase_byte*8));"
                for field in FIELDS
            ),
            indent + "}",
        ]

    def count(self, value, *, byte_order, indent):
        offset = "phase_byte" if byte_order == "little" else "(7-phase_byte)"
        return [
            indent + "for(unsigned phase_byte=0;phase_byte<8;phase_byte++)",
            indent + f"  {self.storage_prefix}_completed[{offset}]="
            f"(unsigned char)((uint64_t)({value})>>(phase_byte*8));",
        ]
