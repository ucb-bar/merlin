"""Load a target's probe backend from the file its timing space names (the target's data directory).

The backend is target data with code: it lives beside the timing space, outside every library root,
and is selected by that document -- never by a target name in this package. It must expose
``bind(space=, facts=, contract=)`` returning an object with ``SHAPES``, ``realize(request, space)``,
``render(realized, timing, probe_id=)`` and ``identity()``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from .space import SpaceError, TimingSpace


def backend_path(space: TimingSpace) -> Path:
    declared = space.document.get("backend")
    if not isinstance(declared, str) or not declared:
        raise SpaceError("the timing space names no probe backend")
    path = (space.data_dir / declared).resolve()
    if space.data_dir.resolve() not in path.parents:
        raise SpaceError("the probe backend must live in the timing space's own directory")
    return path


def load_backend(space: TimingSpace, *, facts: dict, contract: dict):
    path = backend_path(space)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"merlin_timing_probe_backend_{digest}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise SpaceError(f"probe backend {path} does not load")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.bind(space=space, facts=facts, contract=contract)
