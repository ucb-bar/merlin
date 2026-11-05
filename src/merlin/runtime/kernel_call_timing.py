"""Opt-in measured cycles for one pointer kernel entry through completion."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from .complete_call_timing import _digest, _pin, _validate

SCHEMA = "merlin.kernel_call_timing.v1"
BOUNDARY = "immediately before kernel entry through explicitly selected completion"


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
            "schema": SCHEMA,
            "selection_sha256": self.selection_sha256,
            "counter_semantics": json.loads(self.counter_record),
            "counter_asm": self.counter_asm,
            "max_delta_cycles": self.max_delta_cycles,
            "boundary": BOUNDARY,
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


def from_contract(contract) -> KernelCallTiming | None:
    """Require a reviewed counter/clock selection; never infer one from the ISA."""
    logical = contract.get("logical_harness") or {}
    selection = logical.get("kernel_call_timing")
    if selection is None:
        return None
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "counter_semantics", "max_delta_cycles"}
        or selection["schema"] != SCHEMA
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
