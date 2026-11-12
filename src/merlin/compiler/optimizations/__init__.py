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
        "merlin.compiler.optimizations.blocks",
        "merlin.compiler.optimizations.contraction",
        "merlin.compiler.optimizations.discovery",
        "merlin.compiler.optimizations.hoisting",
        "merlin.compiler.optimizations.loops",
        "merlin.compiler.optimizations.passes",
        "merlin.compiler.optimizations.perop",
        "merlin.compiler.optimizations.pointwise",
        "merlin.compiler.optimizations.resident",
        "merlin.compiler.optimizations.variants",
    )
    return (Path(__file__), *(module_source_path(module) for module in modules))
