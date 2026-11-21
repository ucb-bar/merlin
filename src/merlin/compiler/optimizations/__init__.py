"""Selectable shared optimization strategies and their actual source commitment.

Pass invocation lives in compiler.passes; IR algorithms and command scheduling
retain their own contracts. Importing this package registers no strategy.
"""

from pathlib import Path

from merlin.common.paths import module_source_path


def implementation_paths() -> tuple[Path, ...]:
    """Reviewed implementation sources, without importing or registering strategies."""
    modules = (
        "merlin.compiler.optimizations.registry",
        "merlin.compiler.optimizations.register_blocking",
        "merlin.compiler.optimizations.contraction_loops",
        "merlin.compiler.optimizations.registry",
        "merlin.compiler.optimizations.accumulator_residency",
        "merlin.compiler.optimizations.contraction_loops",
        "merlin.compiler.optimizations.vectorization",
        "merlin.compiler.optimizations.register_blocking",
        "merlin.compiler.optimizations.vectorization",
        "merlin.compiler.optimizations.accumulator_residency",
        "merlin.compiler.optimizations.contraction_loops",
    )
    return (Path(__file__), *(module_source_path(module) for module in modules))
