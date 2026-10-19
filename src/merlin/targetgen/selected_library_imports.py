"""Mechanical startup for an already host-selected compiler source snapshot.

This stdlib-only file may be copied into a compiler package. It grants no paths,
imports or review authority: the original profile/sandbox and final library
handoff still own those decisions. In particular its JSON input is source data,
never an executable qualification receipt. Run before compiler imports.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path, PurePosixPath


def _sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("selected compiler source is absent or linked")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guest_root(contract_sha256):
    return Path("/perf-compiler-library") / contract_sha256 / "python"


def _record(selection, *, relocated=False):
    if type(selection) is not dict or set(selection) != {"root", "contract", "contract_sha256"}:
        raise ValueError("compiler startup requires an exact original selected source record")
    contract = selection["contract"]
    if type(contract) is not dict or set(contract) != {"schema", "review_id", "public_modules", "members"}:
        raise ValueError("selected compiler source contract is not explicit")
    if contract["schema"] != "merlin.compiler_library.v1":
        raise ValueError("unsupported selected compiler source contract")
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != selection["contract_sha256"]:
        raise ValueError("selected compiler contract bytes changed")
    root = Path(selection["root"])
    if (
        not root.is_absolute()
        or any(parent.is_symlink() for parent in (root, *root.parents))
        or root.resolve(strict=not relocated) != root
    ):
        raise ValueError("selected compiler source root is not canonical")
    if relocated and not root.exists():
        root = _guest_root(selection["contract_sha256"])
    if (
        not root.is_dir()
        or any(parent.is_symlink() for parent in (root, *root.parents))
        or root.resolve(strict=True) != root
    ):
        raise ValueError("selected compiler source root is absent, linked or not canonical")
    members = contract["members"]
    if type(members) is not list or not 1 <= len(members) <= 512:
        raise ValueError("selected compiler source membership exceeds its bounded startup")
    modules, paths = {}, set()
    for member in members:
        if type(member) is not dict or set(member) != {"module", "path", "sha256"}:
            raise ValueError("selected compiler source member is not closed")
        relative, name = PurePosixPath(member["path"]), member["module"]
        if (
            relative.is_absolute()
            or str(relative) != member["path"]
            or ".." in relative.parts
            or "\\" in member["path"]
            or member["path"] in paths
        ):
            raise ValueError("selected compiler source membership is unsafe or duplicate")
        paths.add(member["path"])
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)) or _sha(path) != member["sha256"]:
            raise ValueError("selected compiler source member changed")
        if name is not None:
            if (
                type(name) is not str
                or not all(part.isidentifier() for part in name.split("."))
                or name in modules
                or str(relative) not in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py")
            ):
                raise ValueError("selected compiler import membership is invalid")
            modules[name] = path
    if not set(contract["public_modules"]) <= modules.keys():
        raise ValueError("selected compiler public imports are absent")
    return root, modules


def _activate(selection, *, original_host_roots, relocated):
    root, modules = _record(selection, relocated=relocated)
    host_roots = tuple(Path(path) for path in original_host_roots)
    if any(
        not path.is_absolute()
        or any(parent.is_symlink() for parent in (path, *path.parents))
        or path.resolve(strict=not relocated) != path
        or relocated and path.exists() and not path.is_dir()
        for path in host_roots
    ):
        raise ValueError("original host dependency roots are not canonical")
    # A missing host directory exposes no dependencies. V2 must not require its
    # original host alias to be mounted into a separately admitted sandbox.
    host_roots = tuple(path for path in host_roots if path.is_dir())
    return _load(selection, root, modules, host_roots, relocated=relocated)


def activate_selected_sources(selection, *, original_host_roots=()):
    """Select exact source imports at fresh compiler startup, without a grant.

    Host roots are already admitted dependency locations, appended only after
    selected package leaves. They do not change the original filesystem sandbox
    or its private masks. Previously loaded foreign leaves refuse instead of
    replacing live class objects. Only byte-identical namespace packages may
    already exist; every selected module is then preloaded from its snapshot.
    """
    return _activate(selection, original_host_roots=original_host_roots, relocated=False)


def activate_selected_sources_v2(selection, *, original_host_roots=()):
    """Reopen the same release at its original or already admitted guest path.

    Only an absent original root permits the contract-derived sandbox location.
    An existing invalid source refuses without fallback. All selected bytes and
    imports retain their original checks. Absent host dependency roots add no
    imports; exact public dependencies may already exist in the guest tree.
    This creates no filesystem grant and must be selected before source freeze.
    """
    return _activate(selection, original_host_roots=original_host_roots, relocated=True)


def _load(selection, root, modules, host_roots, *, relocated):
    for name, path in modules.items():
        loaded = sys.modules.get(name)
        if loaded is not None:
            loaded_path = Path(getattr(loaded, "__file__", ""))
            if loaded_path != path and (path.name != "__init__.py" or _sha(loaded_path) != _sha(path)):
                raise ValueError("foreign compiler leaf was imported before source selection: " + name)
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    packages = sorted(
        (name for name, path in modules.items() if path.name == "__init__.py"), key=lambda name: (name.count("."), name)
    )
    for name in packages:
        module = importlib.import_module(name)
        path = modules[name]
        if _sha(Path(module.__file__)) != _sha(path):
            raise ValueError("selected compiler namespace differs from its snapshot")
        selected_directory = str(path.parent)
        dependencies = [
            str(base.joinpath(*name.split("."))) for base in host_roots if base.joinpath(*name.split(".")).is_dir()
        ]
        # Existing paths were already available to the original host process;
        # selected source precedence is explicit and independent of sys.path.
        module.__path__ = list(dict.fromkeys([selected_directory, *dependencies, *module.__path__]))
    for name in sorted(modules, key=lambda name: (name.count("."), name)):
        importlib.import_module(name)
    root_after, modules_after = _record(selection, relocated=relocated)
    if root_after != root or modules_after != modules:
        raise ValueError("selected source membership changed during import")
    for name, path in modules.items():
        loaded = Path(sys.modules[name].__file__)
        if _sha(loaded) != _sha(path):
            raise ValueError("compiler startup did not load the exact selected source: " + name)
    return {
        "schema": (
            "merlin.selected_compiler_source_startup.v2" if relocated else "merlin.selected_compiler_source_startup.v1"
        ),
        "contract_sha256": selection["contract_sha256"],
        "root": str(root),
        "loaded_modules": len(modules),
        "authority": "source custody only; original sandbox/profile and independent checks remain required",
    }
