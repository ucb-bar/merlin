"""Check release archives for duplicate ownership, source drift and private corpora."""

from __future__ import annotations

import argparse
import collections
import configparser
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
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
                if name.startswith(
                    (
                        "merlin/_data/contract/capsules/",
                        "merlin/contract/capsules/",
                        "merlin_experiments/_data/contract/capsules/",
                        "experiments/corpora/",
                        "merlin/resources/contracts/capsules/",
                        "merlin/resources/corpora/",
                    )
                ):
                    errors.append(f"historical capsule corpus shipped in {distribution}: {name}")
                if name.endswith(".dist-info/entry_points.txt"):
                    parser = configparser.ConfigParser()
                    parser.read_string(archive.read(name).decode())
                    for command in parser["console_scripts"] if parser.has_section("console_scripts") else ():
                        commands[command].append(distribution)
                if distribution != "merlin" and name in {
                    "merlin/__init__.py",
                    "merlin/mining/__init__.py",
                    "merlin/targetgen/__init__.py",
                    "merlin/targetgen/rtl/__init__.py",
                    "merlin/perf/__init__.py",
                    "merlin/kernels/__init__.py",
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
        *sorted((root / "tools").glob("timing/pyproject.toml")),
        *sorted((root / "examples").glob("*/pyproject.toml")),
    ]
    for project in projects:
        metadata = tomllib.loads(project.read_text(encoding="utf-8"))
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
            if name == "merlin":
                errors.extend(audit_resources(archive, root))
                errors.extend(audit_generated_skill(archive, root))
    return errors


def audit_resources(archive: zipfile.ZipFile, root: Path) -> list[str]:
    """Compare the actual wheel's complete public data payload to its reviewed authority."""
    manifest = root / "build_tools/package_resources.json"
    if not manifest.is_file():
        return []  # Historical and synthetic source layouts have no curated data manifest.
    document = json.loads(manifest.read_text(encoding="utf-8"))
    names = document.get("files")
    if document.get("version") != 2 or not isinstance(names, list) or not names:
        return ["public resource manifest is unsupported or empty"]
    expected = {}
    for name in names:
        if not isinstance(name, str):
            return ["public resource manifest contains a non-path member"]
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or not name.startswith(("src/merlin/resources/", "src/merlin/runtime/native/"))
        ):
            return [f"public resource escapes its canonical owner: {name}"]
        member = name.removeprefix("src/")
        if member in expected:
            return [f"duplicate public resource: {name}"]
        expected[member] = root / path
    actual = {
        name
        for name in archive.namelist()
        if name.startswith(("merlin/resources/", "merlin/runtime/native/", "merlin/_data/"))
        and not name.endswith(("/", ".py", ".pyc"))
    }
    errors = [f"required public resource missing: {name}" for name in sorted(expected.keys() - actual)]
    errors.extend(f"unreviewed public resource shipped: {name}" for name in sorted(actual - expected.keys()))
    for name in sorted(expected.keys() & actual):
        path = expected[name]
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            errors.append(f"public resource source is absent, linked or noncanonical: {name}")
        elif path.read_bytes() != archive.read(name):
            errors.append(f"public resource differs from canonical source: {name}")
    return errors


def audit_generated_skill(archive: zipfile.ZipFile, root: Path) -> list[str]:
    """Admit the closed generated pass-library payload against its actual producer."""
    prefix = "merlin/development/backend/author_skills/pass-library/"
    catalog = root / "src/merlin/resources/optimizations/authoring.json"
    generator = root / "build_tools/packaging/gen_author_pass_library.py"
    try:
        document = json.loads(catalog.read_text(encoding="utf-8"))
        facets = {row["facet"] for row in document["mechanisms"]}
        if any(not isinstance(facet, str) or Path(facet).name != facet or facet in {".", ".."} for facet in facets):
            raise ValueError("invalid pass-library facet path")
        names = {"SKILL.md", *(f"reference/{facet}.md" for facet in facets)}
        with tempfile.TemporaryDirectory(prefix="merlin-pass-library-audit-") as temporary:
            if generator.is_file():
                output = Path(temporary) / "pass-library"
                completed = subprocess.run(
                    [sys.executable, str(generator.resolve()), "--output", str(output)],
                    cwd=root,
                    env=dict(os.environ, PYTHONPATH=str((root / "src").resolve())),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if completed.returncode:
                    raise ValueError("pass-library renderer failed: " + completed.stderr[-1000:])
            else:
                # Source releases retain the original build payload; their
                # pruned tests/registries cannot reproduce the checkout render.
                output = root / "src" / prefix.rstrip("/")
            expected = {}
            for path in output.rglob("*"):
                if path.is_symlink():
                    raise ValueError("pass-library source payload contains a link")
                if path.is_file():
                    expected[path.relative_to(output).as_posix()] = path.read_bytes()
            if expected.keys() != names:
                raise ValueError("pass-library source payload is absent or has changed membership")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        return [f"generated pass-library expectation unavailable: {exc}"]
    actual = {
        name.removeprefix(prefix) for name in archive.namelist() if name.startswith(prefix) and not name.endswith("/")
    }
    errors = [f"required generated skill member missing: {prefix}{name}" for name in sorted(names - actual)]
    errors.extend(f"unexpected generated skill member: {prefix}{name}" for name in sorted(actual - names))
    for name in sorted(names & actual):
        if archive.read(prefix + name) != expected[name]:
            errors.append(f"generated skill member differs from its source: {prefix}{name}")
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
            print("packaged Python and reviewed public resources match canonical source bytes")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
