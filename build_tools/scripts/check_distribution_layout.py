"""Check release archives for duplicate ownership, source drift and private corpora."""

from __future__ import annotations

import argparse
import collections
import configparser
import fnmatch
import os
import tomllib
import zipfile
from pathlib import Path


def audit(wheels: list[Path]) -> list[str]:
    files = collections.defaultdict(list)
    commands = collections.defaultdict(list)
    errors = []
    for wheel in wheels:
        distribution = wheel.name.split("-", 1)[0]
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                if ".dist-info/" not in name:
                    files[name].append(distribution)
                if name.startswith(("merlin/_data/contract/capsules/", "merlin/contract/capsules/",
                                    "merlin_experiments/_data/contract/capsules/")):
                    errors.append(f"historical capsule corpus shipped in {distribution}: {name}")
                if name.endswith(".dist-info/entry_points.txt"):
                    parser = configparser.ConfigParser()
                    parser.read_string(archive.read(name).decode())
                    for command in parser["console_scripts"] if parser.has_section("console_scripts") else ():
                        commands[command].append(distribution)
                if distribution != "merlin" and name in {
                    "merlin/__init__.py",
                    "merlin/mining/__init__.py",
                    "merlin/baselines/__init__.py",
                    "merlin/targetgen/__init__.py",
                    "merlin/targetgen/rtl/__init__.py",
                    "merlin/verify/__init__.py",
                    "merlin/perf/__init__.py",
                    "merlin/kernels/__init__.py",
                    "merlin/kernels/ceiling_drivers/__init__.py",
                }:
                    errors.append(f"extension overwrites core namespace initializer: {distribution}:{name}")
    for label, entries in (("file", files), ("command", commands)):
        errors.extend(
            f"duplicate {label} owner: {name}: {owners}" for name, owners in entries.items() if len(owners) > 1
        )
    return errors


def _python_sources(project: Path, metadata: dict) -> dict[str, Path]:
    """Inventory the repository's setuptools src packages without executing build hooks.

    Match package discovery, not recursive Python globbing: omitted namespace parents
    contribute no initializer, includes select whole packages, and exclusions need not
    exclude descendants. The inspected build hooks only stage non-Python resources.
    Refuse other layouts so future packaging changes cannot silently weaken this audit.
    """
    config = metadata.get("tool", {}).get("setuptools", {})
    if not isinstance(config, dict):
        raise ValueError("setuptools configuration must be a mapping")
    packages = config.get("packages", {"find": {"where": ["src"]}})
    if not isinstance(packages, dict) or set(packages) != {"find"}:
        raise ValueError("requires setuptools packages.find")
    discovery = packages["find"]
    if not isinstance(discovery, dict):
        raise ValueError("packages.find must be a mapping")
    for key in ("where", "include", "exclude"):
        if key in discovery and (
            not isinstance(discovery[key], list) or not all(isinstance(item, str) for item in discovery[key])
        ):
            raise ValueError(f"packages.find.{key} must be a list of strings")
    if "namespaces" in discovery and not isinstance(discovery["namespaces"], bool):
        raise ValueError("packages.find.namespaces must be a boolean")
    if (
        set(discovery) - {"where", "include", "exclude", "namespaces"}
        or discovery.get("where", ["src"]) != ["src"]
        or config.get("package-dir", {"": "src"}) != {"": "src"}
        or config.get("py-modules")
    ):
        raise ValueError("unsupported setuptools source layout")
    include = discovery.get("include", ["*"])
    exclude = ["ez_setup", "*__pycache__", *discovery.get("exclude", [])]
    namespaces = discovery.get("namespaces", True)
    source = project.parent / "src"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source root must be an existing unlinked directory")

    def unreadable(error: OSError) -> None:
        raise error

    result = {}
    for directory, children, _ in os.walk(source, onerror=unreadable):
        candidates = children[:]
        children[:] = []
        for child in candidates:
            path = Path(directory) / child
            if path.is_symlink():
                raise ValueError(f"linked source directory: {path.relative_to(source)}")
            package = ".".join(path.relative_to(source).parts)
            if "." in child or (not namespaces and not (path / "__init__.py").is_file()):
                continue
            if any(fnmatch.fnmatchcase(package, pattern) for pattern in include) and not any(
                fnmatch.fnmatchcase(package, pattern) for pattern in exclude
            ):
                for module in sorted(p for p in path.iterdir() if p.suffix == ".py"):
                    if module.is_symlink():
                        raise ValueError(f"linked Python source: {module.relative_to(source)}")
                    if module.is_file():
                        result[module.relative_to(source).as_posix()] = module
            if f"{package}*" not in exclude and f"{package}.*" not in exclude:
                children.append(child)
    return result


def audit_sources(wheels: list[Path], root: Path) -> list[str]:
    """Compare shipped Python against its canonical owner, not a stale build staging tree."""
    owners: dict[str, Path] = {}
    inventories: dict[str, dict[str, Path]] = {}
    errors = []
    projects = [
        root / "pyproject.toml",
        *sorted((root / "packages").glob("*/pyproject.toml")),
        *sorted((root / "examples").glob("*/pyproject.toml")),
    ]
    for project in projects:
        metadata = tomllib.loads(project.read_text())
        name = metadata["project"]["name"].replace("-", "_").lower()
        if name in owners:
            errors.append(f"duplicate source distribution: {name}")
        owners[name] = project.parent / "src"
        try:
            inventories[name] = _python_sources(project, metadata)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot inventory source distribution: {name}: {exc}")
    for wheel in wheels:
        name = wheel.name.split("-", 1)[0].lower()
        source = owners.get(name)
        if source is None:
            errors.append(f"unknown source distribution: {name}")
            continue
        with zipfile.ZipFile(wheel) as archive:
            expected = inventories.get(name, {})
            for missing in sorted(expected.keys() - set(archive.namelist())):
                errors.append(f"required Python member missing: {name}:{missing}")
            for member in archive.namelist():
                if not member.endswith(".py") or ".dist-info/" in member:
                    continue
                relative = Path(member)
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"escaping Python member: {name}:{member}")
                    continue
                canonical = source / relative
                if not canonical.is_file():
                    errors.append(f"Python member has no canonical source: {name}:{member}")
                elif name in inventories and member not in expected:
                    errors.append(f"Python member excluded by package configuration: {name}:{member}")
                elif canonical.read_bytes() != archive.read(member):
                    errors.append(f"Python member differs from canonical source: {name}:{member}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path, nargs="+")
    parser.add_argument("--source-root", type=Path, help="also compare shipped Python bytes to this source tree")
    args = parser.parse_args()
    errors = audit(args.wheels)
    if args.source_root is not None:
        errors.extend(audit_sources(args.wheels, args.source_root))
    for error in errors:
        print(error)
    if not errors:
        print(f"{len(args.wheels)} wheels: unique file/CLI ownership; source parity and corpus exclusions checked")
        if args.source_root is not None:
            print("packaged Python matches canonical source bytes")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
