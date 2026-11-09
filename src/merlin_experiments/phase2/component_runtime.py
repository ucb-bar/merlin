"""Inventory explicit reviewed system runtime files for component launch inputs.

This preparation command establishes file membership and byte identities only.
It does not establish namespace support, compiler correctness or model readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import RuntimeGrant

from .component_runtime_authority import (
    IndependentComponentRuntime as IndependentComponentRuntime,
)
from .component_runtime_authority import (
    IndependentRuntimeServices as IndependentRuntimeServices,
)
from .component_runtime_authority import (
    admit_independent_component_runtime as admit_independent_component_runtime,
)
from .component_runtime_authority import (
    require_independent_runtime as require_independent_runtime,
)


def _dependency_relocations(rows):
    """Validate explicit coordinate projections, without granting a source tree."""
    if not isinstance(rows, tuple):
        raise C.StageGateError("runtime dependency relocations require a closed immutable selection")
    selected = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2 or not isinstance(row[0], Path):
            raise C.StageGateError("runtime dependency relocation requires an explicit source/destination pair")
        source, destination = row
        try:
            canonical = source.resolve(strict=True)
        except OSError as exc:
            raise C.StageGateError("runtime dependency relocation source is unavailable") from exc
        if (
            not source.is_absolute()
            or canonical != source
            or not source.is_dir()
            or any(path.is_symlink() for path in (source, *source.parents))
        ):
            raise C.StageGateError("runtime dependency relocation source is indirect or unavailable")
        RuntimeGrant.verify_destination(destination)
        target = PurePosixPath(destination)
        if any(
            source.is_relative_to(other_source)
            or other_source.is_relative_to(source)
            or target.is_relative_to(other_target)
            or other_target.is_relative_to(target)
            for other_source, other_target in selected
        ):
            raise C.StageGateError("runtime dependency relocation prefixes overlap or collide")
        selected.append((source, target))
    return tuple(selected)


def inventory_runtime(
    *,
    files: tuple[tuple[Path, str], ...],
    trees: tuple[tuple[Path, str], ...],
    executables: tuple[tuple[Path, str], ...],
    dependency_relocations: tuple[tuple[Path, str], ...] = (),
) -> tuple[RuntimeGrant, ...]:
    """Pin declared files and the actual dynamic-loader dependencies of trusted tools.

    Callers review the input executable owners before running ``ldd``. Extra
    resources (Python modules, certificates, simulator data) must be declared
    explicitly. Optional reviewed prefix projections apply only to dependencies
    actually reported by those trusted executables; they never inventory a SDK
    tree or establish independence. Relative dependency names are preserved so
    the explicitly relocated tool layout can still resolve its original RPATH.
    The launch's actual readiness probes detect incomplete closure.
    """
    relocations = _dependency_relocations(dependency_relocations)
    index: dict[str, tuple[Path, str]] = {}

    def add(source, destination):
        source = Path(source).resolve(strict=True)
        RuntimeGrant.verify_destination(destination)
        identity = source, C.sha256_file(source)
        previous = index.get(destination)
        if previous is not None and previous != identity:
            raise C.StageGateError("runtime inventory has conflicting destination identities")
        index[destination] = identity

    for source, destination in files:
        add(source, destination)
    for root, destination in trees:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise C.StageGateError("runtime tree is not an explicit reviewed directory")
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, str(Path(destination) / path.relative_to(root)))
    for executable, destination in executables:
        add(executable, destination)
        result = subprocess.run(
            ["ldd", str(Path(executable).resolve(strict=True))],
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        # ldd's ordinary diagnostic for a trusted statically linked tool is
        # valid. Any unresolved dynamic dependency remains an inventory error.
        if "not found" in result.stdout:
            raise C.StageGateError("trusted runtime executable has an unresolved dynamic dependency")
        if result.returncode and not any(
            text in result.stderr + result.stdout for text in ("not a dynamic executable", "statically linked")
        ):
            raise C.StageGateError("trusted runtime executable dependency inspection failed")
        for line in result.stdout.splitlines():
            # ldd names dependency paths as whitespace-delimited absolute
            # tokens; no target/tool names or pattern parsing are needed.
            for token in line.split():
                if token.startswith("/") and Path(token).is_file():
                    # Resolve parent coordinates while keeping the requested
                    # alias basename, which the dynamic loader still needs.
                    selected_path = Path(token).parent.resolve(strict=True) / Path(token).name
                    lexical_path = Path(os.path.normpath(token))
                    destination = str(lexical_path)
                    for source_root, destination_root in relocations:
                        if selected_path.is_relative_to(source_root) or lexical_path.is_relative_to(source_root):
                            if not selected_path.is_relative_to(source_root) or not selected_path.resolve(
                                strict=True
                            ).is_relative_to(source_root):
                                raise C.StageGateError(
                                    "projected runtime dependency escaped its selected source prefix"
                                )
                            destination = str(destination_root / selected_path.relative_to(source_root))
                    add(Path(token), destination)
    # No grant is constructed until the entire selected inventory has been
    # checked for identity and file/directory destination collisions.
    destinations = sorted(index)
    for destination in destinations:
        if any(str(parent) in index for parent in PurePosixPath(destination).parents):
            raise C.StageGateError("runtime inventory destinations collide as file and directory")
    grants = tuple(RuntimeGrant(index[name][0], name, index[name][1]) for name in destinations)
    for grant in grants:
        grant.verify()
    return grants


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("file", "tree", "executable", "dependency-relocation"):
        parser.add_argument("--" + flag, action="append", default=[], metavar="SOURCE=DESTINATION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def values(name):
        result = []
        for value in getattr(args, name):
            source, separator, destination = value.partition("=")
            if not separator or not Path(source).is_absolute() or not Path(destination).is_absolute():
                parser.error("runtime membership requires absolute SOURCE=DESTINATION")
            result.append((Path(source), destination))
        return tuple(result)

    if args.output.exists() or args.output.is_symlink():
        parser.error("runtime inventory destination must be fresh")
    grants = inventory_runtime(
        files=values("file"),
        trees=values("tree"),
        executables=values("executable"),
        dependency_relocations=values("dependency_relocation"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [{"source": str(row.source), "destination": row.destination, "sha256": row.sha256} for row in grants],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
