"""Optional EL6 author interface; common transformations invoke the checker privately."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .lean_contract import LeanToolchain, coverage_policy_record, file_digest, imported_product


def prepare_toolchain(
    *,
    executable: Path,
    library_roots: tuple[Path, ...],
    runtime_files: tuple[Path, ...],
    output: Path,
    work_root: Path,
    dependency_revision: str,
    timeout_s: int,
    memory_mb: int,
    mathematical_version: str = "3",
    allow_native_reflection: bool = False,
    sat_solver: Path | None = None,
) -> Path:
    """Compile fixed packaged mathematical modules against an explicit installation.

    This operator tool installs nothing and selects no hardware or compiler strategy.
    Platform qualification and native-loader closure remain the operator's responsibility;
    extra runtime files are pinned explicitly, rather than discovered through PATH.
    """
    from .lean_checker import _run

    executable, output, work_root = executable.resolve(), output.absolute(), work_root.resolve()
    runtime_files = tuple(path.absolute() for path in runtime_files)
    if sat_solver is not None:
        sat_solver = sat_solver.absolute()
        runtime_files = (*runtime_files, sat_solver) if sat_solver not in runtime_files else runtime_files
    roots = tuple(root.resolve() for root in library_roots)
    if not roots or len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        raise ValueError("select distinct existing Lean import roots explicitly")
    if not dependency_revision:
        raise ValueError("select a dependency revision explicitly")
    if output.exists():
        raise ValueError("toolchain preparation output already exists")
    modules = {"1": ("Foundation",), "2": ("Foundation",), "3": ("Foundation", "ResourceGeometry", "StateEffects")}
    if mathematical_version not in modules:
        raise ValueError("select a supported original mathematical module version")
    sources = tuple(Path(__file__).parent / "lean/Merlin" / (name + ".lean") for name in modules[mathematical_version])
    if any(path.is_symlink() or path.absolute() != path.resolve() for path in runtime_files):
        raise ValueError("stage runtime aliases as ordinary files preserving their loader names before preparation")
    paths = {*(source.resolve() for source in sources), *(path.resolve() for path in runtime_files)}
    paths.update(
        path.resolve() for root in roots for path in root.rglob("*") if path.is_file() and imported_product(path)
    )
    files = tuple((path, file_digest(path)) for path in sorted(paths))
    from .lean_sat_runtime import prepared_sat_solver

    solver = prepared_sat_solver(executable, runtime_files, files, explicit=sat_solver)
    selected = LeanToolchain(
        executable,
        file_digest(executable),
        roots,
        files,
        work_root,
        timeout_s,
        memory_mb,
        dependency_revision,
        solver,
        allow_native_reflection,
    )
    selected.verify()
    output.mkdir(parents=True)
    product_root = output / "lib"
    (product_root / "Merlin").mkdir(parents=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lean-prepare-", dir=work_root) as temporary:
        for source in sources:
            product = product_root / "Merlin" / (source.stem + ".olean")
            rc, stdout, stderr = _run(
                [
                    str(executable),
                    "--json",
                    "-j1",
                    f"-M{memory_mb}",
                    f"--root={source.parent.parent}",
                    "-o",
                    str(product),
                    str(source),
                ],
                selected,
                Path(temporary) / source.stem,
                deadline=started + timeout_s,
            )
            messages = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            selected.verify()
            if rc or not product.is_file() or any(row.get("severity") in {"error", "warning"} for row in messages):
                raise ValueError("packaged Lean module did not compile: " + source.stem + ": " + stderr[-1000:])
    products = tuple(
        (path, file_digest(path))
        for path in sorted(product_root.rglob("*"))
        if path.is_file() and imported_product(path)
    )
    completed = LeanToolchain(
        executable,
        selected.executable_sha256,
        (product_root, *roots),
        (*files, *products),
        work_root,
        timeout_s,
        memory_mb,
        dependency_revision,
        solver,
        allow_native_reflection,
    )
    completed.verify()
    manifest = output / "lean-toolchain.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "merlin.lean_toolchain.v1",
                "executable": {"path": str(executable), "sha256": completed.executable_sha256},
                "library_roots": [str(root) for root in completed.library_roots],
                "files": [{"path": str(path), "sha256": sha} for path, sha in completed.files],
                "dependency_revision": dependency_revision,
                "timeout_s": timeout_s,
                "memory_mb": memory_mb,
                **({"sat_solver": {"path": str(solver[0]), "sha256": solver[1]}} if solver is not None else {}),
                "allow_native_reflection": allow_native_reflection,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="report the fixed formal domain and acceptance policy")
    prepare = commands.add_parser(
        "prepare-toolchain",
        help="compile the shared foundation and pin explicitly installed dependencies; no downloads",
    )
    prepare.add_argument("--executable", type=Path, required=True)
    prepare.add_argument("--library-root", type=Path, action="append", required=True)
    prepare.add_argument(
        "--runtime-file",
        type=Path,
        action="append",
        default=[],
        help="Pin native runtime files; explicitly selected adjacent cadical enables bounded bit-vector SAT proofs",
    )
    prepare.add_argument(
        "--allow-native-reflection",
        action="store_true",
        help="Original operator admission of Lean's native LRAT reflection and compiler TCB; not kernel-only proof",
    )
    prepare.add_argument(
        "--sat-solver",
        type=Path,
        help="Explicit original SAT executable; auto-inventoried, no PATH or sibling fallback",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--work-root", type=Path, required=True)
    prepare.add_argument("--dependency-revision", required=True)
    prepare.add_argument("--timeout-s", type=int, default=120)
    prepare.add_argument("--memory-mb", type=int, default=1024)
    prepare.add_argument("--mathematical-version", choices=("1", "2", "3"), default="3")
    check = commands.add_parser("check", help="check actual before/after integer SSA under admitted semantics")
    check.add_argument("--before", type=Path, required=True)
    check.add_argument("--after", type=Path, required=True)
    check.add_argument("--entry", required=True)
    check.add_argument("--toolchain", type=Path, required=True)
    check.add_argument("--work-root", type=Path, required=True)
    check.add_argument("--isa-model", type=Path)
    check.add_argument("--tactic", choices=("simp", "rfl", "bv_decide"), default="simp")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(coverage_policy_record(), sort_keys=True))
        return 0
    if args.command == "prepare-toolchain":
        try:
            manifest = prepare_toolchain(
                executable=args.executable,
                library_roots=tuple(args.library_root),
                runtime_files=tuple(args.runtime_file),
                allow_native_reflection=args.allow_native_reflection,
                sat_solver=args.sat_solver,
                output=args.output,
                work_root=args.work_root,
                dependency_revision=args.dependency_revision,
                timeout_s=args.timeout_s,
                memory_mb=args.memory_mb,
                mathematical_version=args.mathematical_version,
            )
        except (OSError, ValueError, TypeError, TimeoutError) as exc:
            print(json.dumps({"status": "unavailable", "reason": str(exc)}))
            return 2
        print(json.dumps({"status": "prepared_unqualified", "manifest": str(manifest)}))
        return 0
    from .lean_checker import verify_scalar_transform

    registry = None
    if args.isa_model:
        from merlin.targetgen.isa_spec import resolved_isa_from_record

        model = resolved_isa_from_record(json.loads(args.isa_model.read_text()))
        registry = {
            f"{model.namespace}.{instruction.name}": {
                "effects": json.loads(instruction.effects_json),
                "semantics": json.loads(instruction.semantics_json),
                "operand_names": [arg.name for arg in instruction.operands],
                "operand_types": [
                    {"kind": arg.type.kind, "bits": arg.type.bits, "signed": arg.type.signed}
                    for arg in instruction.operands
                ],
                "result_types": [
                    {"kind": arg.type.kind, "bits": arg.type.bits, "signed": arg.type.signed}
                    for arg in instruction.results
                ],
                "immediates": [arg.name for arg in instruction.immediates],
            }
            for instruction in model.instructions
        }
    try:
        toolchain = LeanToolchain.from_manifest(args.toolchain, work_root=args.work_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}))
        return 2
    receipt = verify_scalar_transform(
        args.before.read_text(),
        args.after.read_text(),
        entry=args.entry,
        registry=registry,
        toolchain=toolchain,
        tactic=args.tactic,
    )
    print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0 if receipt.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
