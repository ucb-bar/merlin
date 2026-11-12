"""Modules reached only by computed names or ``python -m`` must exist.

The experiment console imports each panel source by name and renders an import
failure as an empty panel, and the managed-execution guardian and supervisor are
started only by a service manager, so neither shows up as a static import. A
deletion of either family is silent everywhere except here.
"""

import importlib

from merlin.experiments.tracking.console import model


def test_console_sources_and_managed_execution_modules_import():
    names = {name for name, _tier in model.SOURCES} | set(model.INSIGHTS) | set(model.OPERATOR_ONLY)
    for name in sorted(names):
        importlib.import_module(f"merlin.experiments.tracking.console.sources.{name}")
    for module in ("_protocol", "chia_native", "native_guardian", "native_supervisor"):
        importlib.import_module(f"merlin.experiments.execution.{module}")
