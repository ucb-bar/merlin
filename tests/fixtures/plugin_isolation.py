"""Load a test-selected support provider's plugins, then forget them as a fresh process would.

Plugin ownership is process-immutable by design (:mod:`merlin.runtime.backends.base`): once a backend,
dialect or sim-oracle module has loaded from the provider selected on ``MERLIN_TARGET_PATH``, changing
or removing that selection makes EVERY later registry query raise ``PluginOwnershipError`` and ask for
a fresh process. A test that selects a provider with ``monkeypatch.setenv`` and loads its backend
in-process therefore poisons the rest of its worker the moment ``monkeypatch`` restores the
environment: every later test that so much as lists backends fails, naming a target it never touched.

:func:`fresh_plugin_state` is that fresh process, scoped to a block. On entry it settles discovery for
the AMBIENT selection (so a provider selected for the whole session is loaded before the snapshot and
is never unloaded under a later test) and records the loader's state; on exit it restores that state
and drops every plugin module imported inside the block. The refusal itself is untouched: inside the
block a changed selection is refused exactly as it is in production.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator

#: The loader state each plugin seam keeps: ``(module, attribute)``. A registry dict is put back as the
#: SAME object with its entry-time contents, so a caller holding a reference sees the restored registry.
_STATE = (
    ("merlin.runtime.backends.base", "_REGISTRY"),
    ("merlin.runtime.backends.base", "_LOAD_FAILURES"),
    ("merlin.runtime.backends.base", "_LOADED_PLUGIN_OWNERS"),
    ("merlin.runtime.backends.base", "_oot_env_seen"),
    ("merlin.xdsl_dialects.lowering.target_lowering", "_PLUGIN_SPECS"),
    ("merlin.xdsl_dialects.lowering.target_lowering", "_PLUGIN_OPCODES"),
    ("merlin.xdsl_dialects.lowering.target_lowering", "_dialect_env_seen"),
    ("merlin.targetgen.oracle_policy", "_SIM_ORACLES"),
    ("merlin.targetgen.oracle_policy", "_SIM_ADAPTER_FACTORIES"),
    ("merlin.targetgen.oracle_policy", "_sim_oracle_env_seen"),
    ("merlin.targetgen.oracle_policy", "_sim_metadata_env_seen"),
)


@contextlib.contextmanager
def fresh_plugin_state() -> Iterator[None]:
    """Run the block as if in a fresh process with respect to OOT plugins, then restore the caller's."""
    import importlib

    from merlin.runtime.backends import base

    base._ensure_discovered()  # the ambient selection's plugins (and the in-tree ones) are not the block's
    namespaces = tuple(f"{namespace}." for namespace in base._PLUGIN_KEYS)
    modules_before = {name for name in sys.modules if name.startswith(namespaces)}
    saved = []
    for module_name, attribute in _STATE:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        saved.append((module, attribute, value, dict(value) if isinstance(value, dict) else None))
    try:
        yield
    finally:
        with base._oot_lock:
            for name in [name for name in sys.modules if name.startswith(namespaces)]:
                if name not in modules_before:
                    del sys.modules[name]
            for module, attribute, value, contents in saved:
                if contents is not None:
                    value.clear()
                    value.update(contents)
                setattr(module, attribute, value)
