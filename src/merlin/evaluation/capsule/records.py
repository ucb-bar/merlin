"""Observed operation artifacts and numerical reference values passed between grader stages.

These records preserve partial evidence when a later stage refuses or raises.
They carry observations; they do not select target semantics or acceptance policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgramEvidence:
    command_buffer: dict | None = None
    lowered_text: str | None = None
    compiler_source: dict | None = None
    owned_source_compilation: dict = field(default_factory=dict)
    whole_program: bool = False
    tiers: dict = field(default_factory=dict)
    trace_check: dict = field(default_factory=lambda: {"status": "skipped", "violations": []})
    decoded_trace: dict | None = None
    parsed_lowered_module: Any = None
    numeric: dict = field(default_factory=lambda: {"status": "skipped"})
    execution_tiers: set[str] = field(default_factory=set)
    executability: dict = field(default_factory=dict)
    epilogue_applicability: dict = field(default_factory=dict)
    completion_order: dict | None = None
    order_check: dict | None = None
    status: str = "pass"
    failure: dict | None = None
    declined: dict | None = None


@dataclass
class ReferenceData:
    independent_float: bool = False
    direct_oracle_golden: bool = False
    golden_source: str | None = None
    gold: dict | None = None
    reference: dict | None = None
    simulation: dict | None = None
    output_contract: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OracleAttempt:
    completed: bool = False
    result: dict | None = None
    elapsed_s: float | None = None
    carried_built: bool = False


@dataclass(frozen=True)
class OracleMeasurement:
    passed: bool
    mismatch_reason: str
    simulator: str
    oracle: Any
