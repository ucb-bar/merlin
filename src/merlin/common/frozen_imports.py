"""Process-local, source-only imports from an already verified frozen source receipt.

Load this file by absolute filename in a stdlib-only bootstrap, before importing Merlin or
enabling site initialization. The caller owns receipt verification and descendant launches.
This is an attribution boundary, not a sandbox against Python that removes its import guard.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.resources.abc
import importlib.util
import io
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

_OWNERS = frozenset({"merlin", "merlin_experiments", "merlin_analysis", "merlin_dse", "merlin_mining"})


class _Resource(importlib.resources.abc.Traversable):
    def __init__(self, guard, paths):
        self.guard = guard
        self.paths = tuple(paths)

    @property
    def name(self):
        return self.paths[0].name

    def __fspath__(self):
        """Physical consumers require an immutable host snapshot after this verification.

        Validate the complete requested subtree before exposing its filesystem path. A merged
        namespace has no single physical location; its callers must use resources.as_file().
        """
        if len(self.paths) != 1:
            raise TypeError("merged frozen resources require importlib.resources.as_file")
        path = self.paths[0]
        self.guard.verify_resource_path(path)
        return str(path)

    def __str__(self):
        # Existing data_path consumers use Path(str(resource)), not os.fspath(resource).
        return self.__fspath__()

    def is_dir(self):
        return all(path.is_dir() and self.guard.contained(path) for path in self.paths)

    def is_file(self):
        return len(self.paths) == 1 and self.paths[0].is_file() and self.guard.contained(self.paths[0])

    def iterdir(self):
        if not self.is_dir():
            raise NotADirectoryError(self.name)
        names = sorted({child.name for path in self.paths for child in path.iterdir()})
        for name in names:
            resource = self.joinpath(name)
            if all(self.guard.contained(path) for path in resource.paths):
                yield resource

    def joinpath(self, *descendants):
        resource = self
        for descendant in descendants:
            path = PurePosixPath(descendant)
            if path.is_absolute() or ".." in path.parts or "\\" in descendant:
                raise ValueError("frozen resource path must remain inside its package")
            for part in path.parts:
                candidates = [base / part for base in resource.paths if (base / part).exists()]
                if not candidates:
                    candidates = [resource.paths[0] / part]
                # Namespace directories merge; a file/directory conflict uses the first owner,
                # matching importlib's MultiplexedPath without depending on its private internals.
                if not all(candidate.is_dir() for candidate in candidates):
                    candidates = candidates[:1]
                resource = _Resource(self.guard, candidates)
        return resource

    def open(self, mode="r", *args, **kwargs):
        if mode not in ("r", "rb"):
            raise ValueError("frozen resources are read-only")
        stream = io.BytesIO(self.guard.source_bytes(self.paths[0]))
        if mode == "rb":
            return stream
        return io.TextIOWrapper(stream, *args, **kwargs)


class _Resources(importlib.resources.abc.TraversableResources):
    def __init__(self, guard, paths):
        self.guard = guard
        self.paths = paths

    def files(self):
        return _Resource(self.guard, map(Path, self.paths))


class _Source(importlib.abc.InspectLoader):
    def __init__(self, guard, filename: Path, package: bool):
        self.guard = guard
        self.filename = filename
        self.package = package

    def get_filename(self, fullname):
        return str(self.filename)

    def is_package(self, fullname):
        return self.package

    def get_source(self, fullname):
        return importlib.util.decode_source(self.guard.source_bytes(self.filename))

    def get_code(self, fullname):
        return compile(self.guard.source_bytes(self.filename), str(self.filename), "exec", dont_inherit=True)

    def get_resource_reader(self, fullname):
        if self.package:
            # Match ordinary package semantics: resources belong to this initializer's owner.
            return _Resources(self.guard, [str(self.filename.parent)])
        return None

    def exec_module(self, module):
        exec(self.get_code(module.__name__), module.__dict__)
        if self.package:
            # pkgutil.extend_path may discover editable owners, even after an isolated startup.
            module.__path__ = self.guard.package_paths(module.__path__)
            module.__spec__.submodule_search_locations = module.__path__


class _Namespace(importlib.abc.Loader):
    def __init__(self, guard, paths):
        self.guard = guard
        self.paths = paths

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        pass

    def get_resource_reader(self, fullname):
        return _Resources(self.guard, self.paths)


class _FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(
        self, root: Path, roots: tuple[Path, ...], sources: dict[str, str], names: frozenset[str], auxiliary_roots=()
    ):
        self.root = root
        self.roots = roots
        self.source_roots = (*roots, *auxiliary_roots)
        self.sources = sources
        self.names = names
        self.directories = {parent for name in sources for parent in PurePosixPath(name).parents}

    def protected(self, name: str) -> bool:
        return name.split(".", 1)[0] in _OWNERS or name in self.names

    def contained(self, path: Path) -> bool:
        # Symlinks, including internal aliases, are never executable import roots or sources.
        return (
            path.is_relative_to(self.root)
            and path.resolve() == path
            and any(path.is_relative_to(root) for root in self.source_roots)
        )

    def package_paths(self, paths):
        result = []
        for entry in paths:
            path = Path(entry).absolute()
            if (
                self.contained(path)
                and path.is_dir()
                and PurePosixPath(path.relative_to(self.root).as_posix()) in self.directories
                and str(path) not in result
            ):
                result.append(str(path))
        return result

    def source_bytes(self, path: Path) -> bytes:
        if not self.contained(path):
            raise ImportError(f"frozen source escapes its declared roots: {path}")
        expected = self.sources.get(path.relative_to(self.root).as_posix())
        if expected is None:
            raise ImportError(f"frozen source is absent from the verified receipt: {path}")
        if not stat.S_ISREG(path.stat().st_mode):
            raise ImportError(f"frozen source is not a regular file: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ImportError(f"frozen source hash mismatch: {path}")
        return payload

    def verify_resource_path(self, path: Path) -> None:
        if not path.is_dir():
            self.source_bytes(path)
            return
        prefix = path.relative_to(self.root).as_posix() + "/"
        expected = {name for name in self.sources if name.startswith(prefix)}
        found = set()
        pending = [path]
        while pending:
            current = pending.pop()
            if not self.contained(current):
                raise ImportError(f"frozen resource escapes its declared roots: {current}")
            if current.is_dir():
                pending.extend(current.iterdir())
            else:
                self.source_bytes(current)
                found.add(current.relative_to(self.root).as_posix())
        if found != expected:
            raise ImportError(f"frozen resource subtree membership mismatch: {path}")

    def find_spec(self, fullname, path=None, target=None):
        if not self.protected(fullname):
            return None
        search = self.roots if path is None else map(Path, self.package_paths(path))
        leaf = fullname.rsplit(".", 1)[-1]
        namespaces = []
        for base in search:
            directory = base / leaf
            for filename, package in ((directory / "__init__.py", True), (base / f"{leaf}.py", False)):
                if filename.is_file():
                    if package and fullname in self.names:
                        raise ImportError(f"legacy frozen names authorize bare modules, not packages: {fullname}")
                    # Validate before returning a spec as well as before executing its bytes.
                    self.source_bytes(filename)
                    return importlib.util.spec_from_file_location(
                        fullname,
                        filename,
                        loader=_Source(self, filename, package),
                        submodule_search_locations=[str(directory)] if package else None,
                    )
            namespaces.extend(self.package_paths([str(directory)]))
        if namespaces and fullname not in self.names:
            spec = importlib.machinery.ModuleSpec(fullname, _Namespace(self, namespaces), is_package=True)
            spec.submodule_search_locations = namespaces
            return spec
        # Raising, rather than returning None, prevents editable finders and sys.path fallback.
        raise ModuleNotFoundError(f"module unavailable in frozen source receipt: {fullname}", name=fullname)


def activate(
    *,
    snapshot_root: str | Path,
    import_roots: Sequence[str | Path],
    sources: Mapping[str, str],
    legacy_names: Sequence[str] = (),
    auxiliary_roots: Sequence[str | Path] = (),
) -> None:
    """Install one process's import boundary using caller-verified relative-path SHA256 pins.

    All five Merlin owner namespaces are protected. Additional legacy names are exact bare
    modules (not prefixes). Roots must be explicit, ordered, ordinary directories within the
    snapshot; compatibility symlinks are not roots. No bytecode or native extension satisfies
    a protected import. Unprotected third-party dependencies retain normal import behavior.
    Auxiliary roots admit explicitly mounted protected package namespaces/resources,
    but never participate in top-level lookup or change sys.path.
    """
    root = Path(snapshot_root).absolute()
    roots = tuple(Path(entry).absolute() for entry in import_roots)
    auxiliary = tuple(Path(entry).absolute() for entry in auxiliary_roots)
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("frozen snapshot root must be an ordinary directory without symlink components")
    if not roots or any(
        not entry.is_dir() or entry.resolve() != entry or not entry.is_relative_to(root)
        for entry in (*roots, *auxiliary)
    ):
        raise ValueError("frozen import roots must be ordinary directories within the snapshot")
    names = frozenset(legacy_names)
    if any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise ValueError("legacy import names must be exact bare Python module names")
    pins = dict(sources)
    for name, digest in pins.items():
        if (
            not isinstance(name, str)
            or not name
            or name == "."
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"invalid frozen source receipt entry: {name!r}")
    guard = _FrozenImports(root, roots, pins, names, auxiliary)
    loaded = sorted(name for name in sys.modules if guard.protected(name))
    if loaded:
        raise ImportError(f"protected owners imported before frozen source isolation: {', '.join(loaded)}")
    sys.meta_path.insert(0, guard)
    root_paths = list(dict.fromkeys(str(entry) for entry in roots))
    sys.path[:] = root_paths + [entry for entry in sys.path if entry not in root_paths]
