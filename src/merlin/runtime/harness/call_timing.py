"""Complete raw counter samples at explicitly selected pointer-call boundaries.

Counter and control-state functions belong to the explicitly selected runtime.
This owner assigns no units, clock, monotonicity, completion or timer integrity.
Raw observations are not cycle measurements or complete-stage cost evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from merlin.common.target_data import get
from merlin.runtime.harness.direct_kernel import DirectKernelInvocationObservation, DirectKernelInvocationPlan

# ---- direct_kernel_counter ---------------------------------------------------------------------


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
        from merlin.runtime.harness.direct_kernel import _identifier

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
        from merlin.runtime.harness.direct_kernel import DirectKernelAbi, DirectKernelInvocationPlan

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


# ---- direct_kernel_phases ----------------------------------------------------------------------
# Raw samples around fixed ordinary harness code, without complete-cost roles.
#
# The renderer supplies the boundaries. No caller label can move work into them.
# Accessors remain explicitly selected support; raw samples are not elapsed costs.

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
        from merlin.runtime.harness.direct_kernel import _identifier

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


# ---- complete_call_timing ----------------------------------------------------------------------
# Explicit normal caller timing from reviewed counter/clock source selection.
#
# No default counter instruction, unit or clock exists here. This is a fixed C
# boundary for the original caller, not a new runtime qualification protocol.

COMPLETE_CALL_SCHEMA = "merlin.complete_call_timing.v1"
COMPLETE_CALL_BOUNDARY = "main preparation through completed whole-program outputs and selected output publication"
EXCLUDED = (
    "static immutable input staging and loader/startup",
    "metric formatting and DONE/exit",
    "external parent transport/readback/decoding",
)


_NUMERICS_ONLY = ContextVar("original_numerics_optional_timing", default=False)


@contextmanager
def optional_timing_for_numerics():
    """Keep reviewed caller bytes when available; grant no timing authority."""
    token = _NUMERICS_ONLY.set(True)
    try:
        yield
    finally:
        _NUMERICS_ONLY.reset(token)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _pin(value):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("complete-call clock requires exact original source file pins")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("complete-call clock source is unavailable or indirect")
    if path.stat().st_size > 16 << 20 or hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise ValueError("complete-call clock source changed or exceeds its source bound")
    return dict(value)


@dataclass(frozen=True)
class CompleteCallTiming:
    counter_asm: str
    counter_record: str
    selection_sha256: str
    max_delta_cycles: int

    def verify(self):
        if type(self) is not CompleteCallTiming:
            raise ValueError("complete-call timing requires the exact typed normal selection")
        record = json.loads(self.counter_record)
        _validate(record, self.counter_asm, self.max_delta_cycles)
        if (
            type(self.selection_sha256) is not str
            or len(self.selection_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.selection_sha256)
        ):
            raise ValueError("complete-call selection has no exact identity")
        for pin in record["source_pins"]:
            _pin(pin)
        return self.record()

    def record(self):
        return {
            "schema": COMPLETE_CALL_SCHEMA,
            "selection_sha256": self.selection_sha256,
            "counter_semantics": json.loads(self.counter_record),
            "counter_asm": self.counter_asm,
            "max_delta_cycles": self.max_delta_cycles,
            "boundary": COMPLETE_CALL_BOUNDARY,
            "excluded": list(EXCLUDED),
            "source_role": "reviewed normal counter/clock selection; original engine certificate remains required",
            "cold_warm": "UNKNOWN",
        }

    def declarations(self, *, console="htif"):
        self.verify()
        if console not in ("htif", "stdio"):
            raise ValueError("complete-call metric requires an original fixed console")
        return [
            "static uint64_t merlin_complete_call_counter(void){",
            f'  uint64_t value; __asm__ __volatile__("{self.counter_asm}" : "=r"(value) :: "memory"); return value;',
            "}",
            "static void merlin_complete_call_metric(uint64_t value){",
            "  char digits[21]; unsigned i=20; digits[i]=0;",
            "  do { digits[--i]=(char)('0'+value%10); value/=10; } while(value);",
            (
                '  htif_puts("METRIC cycles "); htif_puts(digits+i); htif_puts("\\n");'
                if console == "htif"
                else '  printf("METRIC cycles %s\\n", digits+i);'
            ),
            "}",
        ]

    def begin(self):
        return ["  uint64_t merlin_complete_call_start=merlin_complete_call_counter();"]

    def end(self, *, console="htif"):
        if console not in ("htif", "stdio"):
            raise ValueError("complete-call metric requires an original fixed console")
        failure = "htif_exit(4)" if console == "htif" else "return 4"
        return [
            "  uint64_t merlin_complete_call_end=merlin_complete_call_counter();",
            "  uint64_t merlin_complete_call_cycles=merlin_complete_call_end-merlin_complete_call_start;",
            (
                "  if(!merlin_complete_call_cycles || "
                f"merlin_complete_call_cycles>UINT64_C({self.max_delta_cycles})) {failure};"
            ),
            "  merlin_complete_call_metric(merlin_complete_call_cycles);",
        ]


def _validate(record, assembly, maximum):
    if (
        type(record) is not dict
        or not {"unit", "width_bits", "wrap", "clock", "source_pins"} <= set(record)
        or record["unit"] != "cycles"
        or type(record["width_bits"]) is not int
        or record["width_bits"] != 64
        or record["wrap"] != "modulo"
        or type(record["clock"]) is not str
        or not record["clock"]
    ):
        raise ValueError("complete-call timing has no original cycle-unit/width/wrap/clock semantics")
    if type(record["source_pins"]) is not list or not record["source_pins"] or len(record["source_pins"]) > 64:
        raise ValueError("complete-call timing lacks bounded original accessor/ISA/clock source pins")
    if (
        type(assembly) is not str
        or not assembly
        or len(assembly) > 4096
        or any(c in assembly for c in ('"', "\\", "\n", ";", "\r", "\x00"))
    ):
        raise ValueError("complete-call timing lacks the selected plain counter instruction")
    if type(maximum) is not int or not 0 < maximum < 1 << 63:
        raise ValueError("complete-call timing requires an explicit unambiguous wrap/delta bound")


def complete_call_plan(contract):
    """Select only an explicitly reviewed normal counter/ISA/clock declaration."""
    try:
        return _from_contract(contract)
    except (OSError, ValueError):
        if not _NUMERICS_ONLY.get():
            raise
        return None


def _from_contract(contract):
    logical = contract.get("logical_harness") or {}
    selection = logical.get("complete_call_timing")
    if selection is None:
        return None
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "counter_semantics", "max_delta_cycles"}
        or selection["schema"] != COMPLETE_CALL_SCHEMA
    ):
        raise ValueError("complete-call timing requires its explicit closed normal selection")
    record = (contract.get("counter_semantics") or {}).get(selection["counter_semantics"])
    assembly = logical.get("cycle_counter_asm")
    maximum = selection["max_delta_cycles"]
    _validate(record, assembly, maximum)
    for pin in record["source_pins"]:
        _pin(pin)
    return CompleteCallTiming(
        assembly,
        json.dumps(record, sort_keys=True),
        _digest(
            {
                "logical_harness": logical,
                "counter_semantics": record,
                "harness_abi": contract.get("harness_abi"),
                "execution_capabilities": contract.get("execution_capabilities"),
                "readout_semantics": contract.get("readout_semantics"),
            }
        ),
        maximum,
    )


# ---- kernel_call_timing ------------------------------------------------------------------------
# Opt-in measured cycles for one pointer kernel entry through completion.

KERNEL_CALL_SCHEMA = "merlin.kernel_call_timing.v1"
KERNEL_CALL_BOUNDARY = "immediately before kernel entry through explicitly selected completion"


@dataclass(frozen=True)
class KernelCallTiming:
    counter_asm: str
    counter_record: str
    selection_sha256: str
    max_delta_cycles: int

    def verify(self) -> dict:
        if type(self) is not KernelCallTiming:
            raise ValueError("kernel-call timing requires the exact typed selection")
        record = json.loads(self.counter_record)
        _validate(record, self.counter_asm, self.max_delta_cycles)
        if (
            type(self.selection_sha256) is not str
            or len(self.selection_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.selection_sha256)
        ):
            raise ValueError("kernel-call timing lacks a selection identity")
        for pin in record["source_pins"]:
            _pin(pin)
        return self.record()

    def record(self) -> dict:
        return {
            "schema": KERNEL_CALL_SCHEMA,
            "selection_sha256": self.selection_sha256,
            "counter_semantics": json.loads(self.counter_record),
            "counter_asm": self.counter_asm,
            "max_delta_cycles": self.max_delta_cycles,
            "boundary": KERNEL_CALL_BOUNDARY,
            "excluded": ["input staging", "output publication", "metric formatting", "parent readback"],
            "cold_warm": "UNKNOWN",
        }

    def source_marker(self) -> str:
        """Carry the reviewed scope and pinned semantics with the generated caller."""
        self.verify()
        encoded = base64.b64encode(json.dumps(self.record(), sort_keys=True).encode()).decode("ascii")
        return (
            f"/* merlin.kernel_call_timing.v1 selection_sha256={self.selection_sha256} "
            f"boundary=entry-through-completion record_b64={encoded} */"
        )

    def declarations(self) -> list[str]:
        self.verify()
        return [
            "static uint64_t merlin_kernel_call_counter(void){",
            f'  uint64_t value; __asm__ __volatile__("{self.counter_asm}" : "=r"(value) :: "memory"); return value;',
            "}",
            "static void merlin_kernel_call_metric(uint64_t value){",
            "  char digits[21]; unsigned i=20; digits[i]=0;",
            "  do { digits[--i]=(char)('0'+value%10); value/=10; } while(value);",
            '  htif_puts("METRIC cycles "); htif_puts(digits+i); htif_puts("\\n");',
            "}",
        ]

    def begin(self) -> list[str]:
        return ["  uint64_t merlin_kernel_call_start=merlin_kernel_call_counter();"]

    def end(self) -> list[str]:
        return [
            "  uint64_t merlin_kernel_call_end=merlin_kernel_call_counter();",
            "  uint64_t merlin_kernel_call_cycles=merlin_kernel_call_end-merlin_kernel_call_start;",
            "  if(!merlin_kernel_call_cycles || "
            f"merlin_kernel_call_cycles>UINT64_C({self.max_delta_cycles})) htif_exit(4);",
            "  merlin_kernel_call_metric(merlin_kernel_call_cycles);",
        ]


def kernel_call_plan(contract) -> KernelCallTiming | None:
    """Require a reviewed counter/clock selection; never infer one from the ISA."""
    logical = contract.get("logical_harness") or {}
    selection = logical.get("kernel_call_timing")
    if selection is None:
        return None
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "counter_semantics", "max_delta_cycles"}
        or selection["schema"] != KERNEL_CALL_SCHEMA
    ):
        raise ValueError("kernel-call timing requires its explicit closed selection")
    record = (contract.get("counter_semantics") or {}).get(selection["counter_semantics"])
    assembly = logical.get("cycle_counter_asm")
    maximum = selection["max_delta_cycles"]
    _validate(record, assembly, maximum)
    for pin in record["source_pins"]:
        _pin(pin)
    return KernelCallTiming(
        assembly,
        json.dumps(record, sort_keys=True),
        _digest({"logical_harness": logical, "counter_semantics": record, "harness_abi": contract.get("harness_abi")}),
        maximum,
    )
