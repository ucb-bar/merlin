"""Compatibility shim: ``merlin.rvvgen`` is now :mod:`merlin.mining`.

The package was named for the one backend it could handle. It mines expert kernels, lifts them into
the CCA and routes divergences to compiler actions, and none of that is vector-specific any more — the
lifter keys on instruction ROLES, so a systolic mesh, an outer-product tile, a SIMT cluster and a lane
engine all enter the same loop. A name that claims otherwise is a claim about scope, and this one had
become wrong.

Kept so out-of-tree callers, saved run configs and shipped artifacts that reference the old dotted path
keep working.

⚠️ It ALIASES rather than re-imports. Sharing ``__path__`` would let ``import merlin.rvvgen.beam``
execute ``mining/beam.py`` a SECOND time under a second name, giving two module objects with two copies
of any module-level state — and this package's neighbours hold registries (routes, seams, backends), so
a duplicate would register everything twice and each half would see only its own. The finder below
resolves the new module and installs it under the old name, so both names are the same object.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import MetaPathFinder

_OLD = "merlin.rvvgen"
_NEW = "merlin.mining"


class _AliasLoader:
    """A loader that hands back an ALREADY-IMPORTED module instead of executing anything.

    Returning the target's own spec instead would make the import machinery build a fresh module from
    it and run the file again — the duplicate-state outcome this shim exists to avoid.
    """

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        return None                                    # already executed under its real name


class _AliasFinder(MetaPathFinder):
    """Resolve ``merlin.rvvgen.X`` to the already-imported ``merlin.mining.X``."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD):]
        try:
            module = sys.modules.get(new_name) or importlib.import_module(new_name)
        except ImportError:
            return None                                # genuinely absent: let the error be the truth
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str):
    module = importlib.import_module(_NEW)
    try:
        return getattr(module, name)
    except AttributeError:
        return importlib.import_module(f"{_NEW}.{name}")


def __dir__():
    return dir(importlib.import_module(_NEW))
