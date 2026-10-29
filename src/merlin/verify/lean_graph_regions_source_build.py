"""Private Core/Sem build using explicitly selected existing package objects only.

This route is separate from the unavailable Lean-MLIR selection. It never invokes
Lake, downloads packages, or writes an existing numerical build/capsule.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as prior


def source_dependencies(lean: Path, path: Path, environment: dict, log: Path) -> list[Path]:
    """Use Lean's real header parser; never strip comments or guess import lines."""
    before = prior.file_ref(path)
    command = [str(lean), "-j1", "--src-deps", str(path)]
    process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    log.write_text(
        json.dumps(
            dict(
                command=command,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                source=before,
            ),
            indent=2,
        )
        + "\n"
    )
    process.check_returncode()
    prior.fresh_ref(before)
    return list(dict.fromkeys(Path(line).resolve(strict=True) for line in process.stdout.splitlines() if line))


def build(destination: Path, roots: list[str]):
    destination.mkdir(parents=True, exist_ok=False)
    root = repo_root()
    lean = Path.home() / ".elan/toolchains/leanprover--lean4---v4.35.0/bin/lean"
    if prior.file_ref(lean)["sha256"] != prior.LEAN_SHA:
        raise ValueError("selected Lean executable differs")
    manifest_path = root / "merlin/lean/lake-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    packages = root / "out/build/lean/packages"
    external, package_sources = [], []
    selection = []
    for package in manifest["packages"]:
        directory = packages / package["name"]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if revision != package["rev"]:
            raise ValueError("installed package differs from the explicitly selected manifest")
        library = directory / ".lake/build/lib/lean"
        available = library.is_dir()
        if available:
            external.append(str(library))
        package_sources.append(directory.resolve())
        # Unused manifest packages are recorded explicitly. An actually imported
        # missing module still fails Lean's build; no empty success is issued.
        selection.append(
            dict(name=package["name"], revision=revision, library=str(library), objects_available=available)
        )
    output = destination / "compiled"
    output.mkdir()
    environment = os.environ.copy()
    old_root, new_root = root / "merlin/lean", root / "src/merlin/verify/lean"
    tool_root = lean.parent.parent
    standard_sources = tool_root / "src/lean"
    environment.update(
        LEAN_PATH=os.pathsep.join([str(output), *external]),
        LEAN_SRC_PATH=os.pathsep.join(map(str, [old_root, new_root, *package_sources])),
        PYTHONDONTWRITEBYTECODE="1",
    )
    origins = [
        (old_root, output, True),
        (new_root, output, True),
        (standard_sources / "lake", tool_root / "lib/lean", False),
        (standard_sources, tool_root / "lib/lean", False),
    ]
    origins += [(path, path / ".lake/build/lib/lean", False) for path in package_sources]
    origins.sort(key=lambda row: len(str(row[0])), reverse=True)
    order, seen, sources, closure = [], set(), {}, {}
    (destination / "import-discovery").mkdir()

    def visit(path):
        path = path.resolve(strict=True)
        if path in seen:
            return
        seen.add(path)
        origin = next((row for row in origins if path.is_relative_to(row[0])), None)
        if origin is None:
            raise ValueError(f"import source is outside selected roots: {path}")
        source_root, library, project = origin
        relative = path.relative_to(source_root).with_suffix("")
        module = ".".join(relative.parts)
        before = prior.file_ref(path)
        discovery_log = destination / "import-discovery" / f"{len(seen):05d}.json"
        dependencies = source_dependencies(lean, path, environment, discovery_log)
        closure[str(path)] = dict(
            source=before,
            module=module,
            project=project,
            dependencies=[str(dependency) for dependency in dependencies],
            discovery=prior.file_ref(discovery_log),
        )
        if not project:
            binary = library / relative.with_suffix(".olean")
            # Every actually imported external object is required. Unused Cli
            # package availability cannot conceal a missing consumed module.
            objects = {".olean": prior.file_ref(binary)}
            for suffix in (".olean.private", ".olean.server", ".ilean"):
                candidate = library / relative.with_suffix(suffix)
                if candidate.is_file():
                    objects[suffix] = prior.file_ref(candidate)
            closure[str(path)]["objects"] = objects
        for dependency in dependencies:
            visit(dependency)
        prior.fresh_ref(before)
        if project:
            if module in sources:
                raise ValueError("duplicate project module ownership")
            sources[module] = (source_root, path)
            order.append(module)

    for module in roots:
        source_root = old_root if module.startswith("MerlinVerify.") else new_root
        visit(source_root / (module.replace(".", "/") + ".lean"))
    identities = {}
    for module in order:
        source_root, path = sources[module]
        before = prior.file_ref(path)
        target = output / (module.replace(".", "/") + ".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [str(lean), "-j1", f"--root={source_root}", "-o", str(target), str(path)]
        process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=120)
        log = destination / (module + ".json")
        log.write_text(
            json.dumps(
                dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
                indent=2,
            )
            + "\n"
        )
        process.check_returncode()
        prior.fresh_ref(before)
        identities[module] = dict(source=before, object=prior.file_ref(target))
    identity = dict(
        lean=prior.file_ref(lean),
        lean_arguments=["-j1"],
        project=identities,
        selected_manifest=prior.file_ref(manifest_path),
        packages=selection,
        builder=prior.file_ref(Path(__file__)),
        LeanMLIR_route="unavailable and not used",
        import_discovery="pinned Lean --src-deps structural header parser; transitive source/object closure",
        import_closure=closure,
    )
    (destination / "IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return lean, environment, identity
