"""Build a sandbox's POLICY on a host that lacks the simulator checkout its sim family names.

The chipyard sim family derives its bind paths from ``MERLIN_EXT_CHIPYARD`` and refuses to resolve
when that key is unset -- deliberately, so a real launch never silently loses its RTL-sim tools
(``packages/merlin-experiments/tests/test_sandbox_optional_config.py`` holds that refusal). A test of
the mount table, the probe list or the exported environment is not a launch: binds are already
filtered by existence, so on a host without the checkout an empty stand-in root yields exactly the
table that host would build. A host that HAS the checkout keeps it, and the same tests then cover the
real binds.
"""

from __future__ import annotations

from pathlib import Path

import external_sources
import pytest


def stand_in_absent_chipyard(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point an absent ``MERLIN_EXT_CHIPYARD`` at the empty ``root`` and re-resolve the family per test.

    The family registry memoizes its first resolution for the whole process, so it is reset to the
    lazy resolver here and restored afterwards: a stand-in resolved for one test must not become the
    family every later test in the worker sees.
    """
    from merlin.targetgen.sandbox import toolchain

    if external_sources.missing("chipyard"):
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("MERLIN_EXT_CHIPYARD", str(root))
    monkeypatch.setitem(toolchain.SIM_TOOLCHAINS.data, "chipyard", toolchain._chipyard)
