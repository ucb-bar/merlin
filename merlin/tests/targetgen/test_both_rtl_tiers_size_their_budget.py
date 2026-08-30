"""Both RTL tiers must size their cycle budget from the workload, not one from a fixed default.

The regression: the arc adapter passed `max_cycles=derive_cycle_budget(cb)` while the Verilator
adapter passed nothing, so every capsule ran against a 20000 default there. A correct kernel needing
more raised `ProgramDidNotHalt`, and the runner attributes that to the submission -- so a harness
limit was reported as the agent's bug. Two tiers of one target had different budgets for no reason.
"""
from __future__ import annotations

import inspect

from merlin.targetgen import program_oracle


def _adapter_source(factory_name: str) -> str:
    return inspect.getsource(getattr(program_oracle, factory_name))


def test_the_arc_tier_sizes_its_budget_from_the_workload() -> None:
    assert "derive_cycle_budget(cb)" in _adapter_source("program_oracle_adapter")


def test_the_verilator_tier_sizes_its_budget_the_same_way() -> None:
    """The point of the fix: neither tier may silently fall back to run_program's default."""
    assert "derive_cycle_budget(cb)" in _adapter_source("program_verilator_adapter")


def test_the_budget_scales_with_the_workload() -> None:
    """A bigger command buffer must buy more cycles, or sizing it is decorative."""
    small = program_oracle.derive_cycle_budget({"tensors": {"a": {"shape": [32, 32]}}})
    large = program_oracle.derive_cycle_budget({"tensors": {"a": {"shape": [512, 512]}}})
    assert large > small, f"budget did not grow with the workload: {small} -> {large}"


def test_the_budget_has_a_floor() -> None:
    """An empty or unparseable command buffer must not yield a budget of zero cycles."""
    assert program_oracle.derive_cycle_budget({}) >= 20000
