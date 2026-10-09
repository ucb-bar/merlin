"""Complete raw counter samples at explicitly selected pointer-call boundaries.

Counter and control-state functions belong to the explicitly selected runtime.
This owner assigns no units, clock, monotonicity, completion or timer integrity.
Raw observations are not cycle measurements or complete-stage cost evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectKernelCounterObservation:
    completed_count: int
    call_samples: tuple[tuple[int, int, int, int], ...]
    calibration_samples: tuple[tuple[int, int, int, int], ...]
    scope: str = "raw counter/control-state samples only; no units, integrity or complete-stage costs"


@dataclass(frozen=True)
class DirectKernelCounterPlan:
    """Explicit opaque uint64 accessors and bounded complete observation storage.

    Each sample retains counter start/end and control-state before/after. Empty
    brackets execute before the first selected call without invoking the entry
    or completion. No overhead distribution or subtraction is inferred. Only
    the entry and optional completion lie inside the call bracket;
    allocation, preparation and output/history publication remain outside it.
    """

    counter_symbol: str
    state_symbol: str
    storage_prefix: str
    calibration_count: int
    max_observation_bytes: int

    def record(self):
        from .direct_kernel_harness import _identifier

        for value in (self.counter_symbol, self.state_symbol, self.storage_prefix):
            _identifier(value)
        if self.counter_symbol == self.state_symbol:
            raise ValueError("counter and control-state accessors must be distinct")
        if (
            type(self.calibration_count) is not int
            or not 1 <= self.calibration_count < 1 << 64
            or type(self.max_observation_bytes) is not int
            or not 1 <= self.max_observation_bytes < 1 << 64
        ):
            raise ValueError("counter capture requires explicit positive uint64 calibration/storage bounds")
        return {
            "schema": "merlin.direct_kernel_counter_plan.v1",
            "counter_symbol": self.counter_symbol,
            "state_symbol": self.state_symbol,
            "storage_prefix": self.storage_prefix,
            "calibration_count": self.calibration_count,
            "max_observation_bytes": self.max_observation_bytes,
            "boundary": "entry through explicitly selected completion; excludes harness history/publication",
            "scope": "raw uint64 observations only; no counter units, integrity, cold/warm or stage-cost authority",
        }

    def bind(self, cb, *, abi, invocation_plan=None):
        from .direct_kernel_harness import DirectKernelAbi
        from .direct_kernel_invocation import DirectKernelInvocationPlan

        self.record()
        if type(abi) is not DirectKernelAbi:
            raise ValueError("counter capture requires the exact explicit pointer ABI")
        abi.verify()
        if type(cb) is not dict or type(cb.get("kernel_abi")) is not dict:
            raise ValueError("counter capture needs a complete pointer-call argument roster")
        args = cb["kernel_abi"].get("args")
        if type(args) is not list or not args:
            raise ValueError("counter capture needs the original nonempty pointer roster")
        count, histories = 1, ()
        if invocation_plan is not None:
            if type(invocation_plan) is not DirectKernelInvocationPlan:
                raise ValueError("counter capture requires an explicit typed repeated-call plan")
            histories = invocation_plan.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
            count = invocation_plan.count
        fields = ("start", "end", "state_before", "state_after")
        roster = {self.storage_prefix + "_call_" + field: count * 8 for field in fields}
        roster.update({self.storage_prefix + "_calibration_" + field: self.calibration_count * 8 for field in fields})
        roster[self.storage_prefix + "_completed"] = 8
        if sum(roster.values()) > self.max_observation_bytes:
            raise ValueError("counter capture exceeds its explicit complete observation storage budget")
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
            "counter_start",
            "counter_end",
            "counter_state_before",
            "counter_state_after",
            *("tensor_" + str(index) for index in range(len(args))),
            *(row.symbol for row in histories),
        }
        if invocation_plan is not None:
            reserved.add(invocation_plan.count_symbol)
        selected = (self.counter_symbol, self.state_symbol, *roster)
        if len(set(selected)) != len(selected) or any(name in reserved for name in selected):
            raise ValueError("counter capture symbols overlap original storage, entrypoints or harness locals")
        return roster

    def decode(self, objects, *, cb, abi, invocation_plan=None):
        """Retain complete raw samples, including wrapping or changing controls.

        The caller independently establishes actual ELF object membership and
        execution. This decoder computes no elapsed time and grants no timer
        semantics from supplied values, matching counts or unchanged controls.
        """
        roster = self.bind(cb, abi=abi, invocation_plan=invocation_plan)
        if type(objects) is not dict or set(objects) != set(roster):
            raise ValueError("counter capture omitted or changed its complete original object roster")
        for name, extent in roster.items():
            if type(objects[name]) is not bytes or len(objects[name]) != extent:
                raise ValueError("counter capture omitted complete original sample bytes")
        expected = invocation_plan.count if invocation_plan is not None else 1
        completed = int.from_bytes(objects[self.storage_prefix + "_completed"], abi.byte_order)
        if completed != expected:
            raise ValueError("counter capture completed count differs from the selected original calls")

        def samples(kind, count):
            arrays = [
                objects[self.storage_prefix + "_" + kind + "_" + name]
                for name in ("start", "end", "state_before", "state_after")
            ]
            return tuple(
                tuple(int.from_bytes(raw[index * 8 : (index + 1) * 8], abi.byte_order) for raw in arrays)
                for index in range(count)
            )

        return DirectKernelCounterObservation(
            completed, samples("call", expected), samples("calibration", self.calibration_count)
        )

    def declarations(self, roster):
        return [
            f"extern uint64_t {self.counter_symbol}(void);",
            f"extern uint64_t {self.state_symbol}(void);",
            *(f"volatile unsigned char {name}[{extent}]={{0}};" for name, extent in roster.items()),
        ]

    def begin(self, indent):
        return [
            indent + f"counter_state_before={self.state_symbol}();",
            indent + f"counter_start={self.counter_symbol}();",
        ]

    def end(self, indent):
        return [
            indent + f"counter_end={self.counter_symbol}();",
            indent + f"counter_state_after={self.state_symbol}();",
        ]

    def store(self, *, kind, index, byte_order, indent):
        offset = "byte" if byte_order == "little" else "(7-byte)"
        return [
            indent + "for(unsigned byte=0;byte<8;byte++){",
            *(
                indent + f"  {self.storage_prefix}_{kind}_{name}[({index})*8+{offset}]="
                f"(unsigned char)(counter_{name}>>(byte*8));"
                for name in ("start", "end", "state_before", "state_after")
            ),
            indent + "}",
        ]

    def count(self, value, *, byte_order, indent):
        offset = "byte" if byte_order == "little" else "(7-byte)"
        return [
            indent + "for(unsigned byte=0;byte<8;byte++)",
            indent + f"  {self.storage_prefix}_completed[{offset}]=(unsigned char)((uint64_t)({value})>>(byte*8));",
        ]
