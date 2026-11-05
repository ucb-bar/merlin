"""Explicit normal caller timing from reviewed counter/clock source selection.

No default counter instruction, unit or clock exists here. This is a fixed C
boundary for the original caller, not a new runtime qualification protocol.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "merlin.complete_call_timing.v1"
BOUNDARY = "main preparation through completed whole-program outputs and selected output publication"
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
            "schema": SCHEMA,
            "selection_sha256": self.selection_sha256,
            "counter_semantics": json.loads(self.counter_record),
            "counter_asm": self.counter_asm,
            "max_delta_cycles": self.max_delta_cycles,
            "boundary": BOUNDARY,
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


def from_contract(contract):
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
        or selection["schema"] != SCHEMA
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
