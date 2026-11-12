"""Compatibility recipe entry for package and isolated direct-file capture.

Editable implementations live in compiler/quantization/torchao. The isolated
workflow loads only the exact sibling source roster through its portable owner;
it does not import an installed Merlin or change the interpreter search path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __package__:
    from merlin.compiler.quantization.torchao import application as _owner

    sys.modules[__name__] = _owner
else:
    _path = Path(__file__).resolve().parents[2] / "compiler/quantization/torchao/portable.py"
    if _path.is_symlink() or not _path.is_file():
        raise ImportError("selected recipe source lacks its portable quantization owner")
    _spec = importlib.util.spec_from_file_location("_merlin_recipe_portable", _path)
    _portable = sys.modules.get(_spec.name)
    if _portable is None:
        _portable = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _portable
        _spec.loader.exec_module(_portable)
    elif Path(_portable.__file__).resolve() != _path:
        raise ImportError("preloaded portable quantization owner belongs to another source tree")
    _owner = _portable.populate(globals())
    if _owner is not None:
        sys.modules[__name__] = _owner
        globals().update(vars(_owner))
