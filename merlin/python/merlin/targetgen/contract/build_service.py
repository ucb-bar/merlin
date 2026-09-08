"""Pure build services: no oracle registry, execution engine or reference imports.

Only trusted host adapters construct these objects. A serialized candidate
request is not a renderer, recipe, or filesystem authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_build_package(package_init: Path) -> ModuleType:
    """Load a host-selected pure target package WITHOUT discovering any backend.

    Paths come from the installed target adapter, never an agent action. The
    complete package's Python file set/bytes are checked on every reuse. A
    changed host package requires a fresh process; old class identities are not
    silently mixed with new source bytes.
    """
    path = Path(package_init).absolute()
    if path.is_symlink() or path.name != "__init__.py" or path.resolve() != path or not path.is_file():
        raise ValueError("pure build package needs a real absolute initializer")
    paths = sorted(path.parent.rglob("*.py"))
    if any(item.is_symlink() or item.resolve() != item for item in paths):
        raise ValueError("pure build source must not escape through symlinks")
    pins = {str(item): file_digest(item) for item in paths}
    namespace = "merlin._pure_build_packages"
    name = namespace + ".p_" + hashlib.sha256(str(path).encode()).hexdigest()
    if name in sys.modules:
        module = sys.modules[name]
        if getattr(module, "_merlin_build_source_pins", None) != pins:
            raise ValueError("loaded pure build package changed; start a fresh host process")
        return module
    if namespace not in sys.modules:
        parent = ModuleType(namespace)
        parent.__path__ = []
        sys.modules[namespace] = parent
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise ValueError("pure build package has no Python source loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if any(file_digest(Path(item)) != pin for item, pin in pins.items()):
            raise ValueError("pure build package changed while loading")
        module._merlin_build_source_pins = pins
        return module
    except BaseException:
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(name + "."):
                del sys.modules[loaded]
        raise


@dataclass(frozen=True)
class BuildOnlyService:
    """Typed host capability used only by the optional cache-free compile branch.

    The renderer consumes explicit inputs and returns C text. It does not run a
    reference or emulator. Source pins are obligations supplied by the trusted
    adapter; they do not grant paths to a sandbox or qualify emitted numerics.
    """
    target: str
    recipe: object
    renderer: Callable
    source_pins: tuple[tuple[str, str], ...]

    def verify(self, target: str) -> None:
        from .build_recipe import HarnessBuildRecipe
        if (type(self) is not BuildOnlyService or self.target != target
                or type(self.recipe) is not HarnessBuildRecipe or not callable(self.renderer)
                or not self.source_pins or len(dict(self.source_pins)) != len(self.source_pins)):
            raise ValueError("build-only service must be a typed target-bound host capability")
        for path, expected in self.source_pins:
            item = Path(path)
            if (not item.is_absolute() or item.resolve() != item or not item.is_file()
                    or file_digest(item) != expected):
                raise ValueError("build-only source/tool pin changed: " + str(path))

    def render(self, cb, *, target, inputs, warm_profile=None):
        self.verify(target)
        if inputs is None or not isinstance(inputs, dict) or not inputs:
            raise ValueError("build-only service requires explicit logical inputs")
        kwargs = {"inputs": inputs}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        result = self.renderer(cb, **kwargs)
        self.verify(target)
        if not isinstance(result, str):
            raise ValueError("build-only renderer did not return source text")
        return result
