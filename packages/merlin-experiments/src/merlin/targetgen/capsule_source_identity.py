"""Retain exact ordinary capsule compiler source and bind emitted artifacts."""

from pathlib import Path


def _capture_compiler_source(package_dir, generated):
    """Retain the actual compiler before lowering, without generated build state."""
    import shutil

    from merlin_experiments.phase2.retained_revisions import source_inventory

    from merlin.common.tree_hash import hash_tree

    from . import executed_program

    # A reused build directory still holds an earlier invocation's executed programs; the receipt
    # finished below names only what this invocation's tiers execute.
    executed_program.clear(generated)
    source = Path(package_dir).resolve()
    if not source.is_dir():
        # Observation cannot replace the ordinary package admission below.
        # An unavailable package supplies no timing source authority.
        return None
    # Bound and reject linked/special source before the ordinary canonical hash.
    source_inventory(source, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
    before = hash_tree(source)
    destination = Path(generated) / "compiler-source" / before["sha256"]
    if destination.resolve().is_relative_to(source) or destination.is_symlink():
        raise ValueError("compiler source evidence cannot be linked or candidate-owned")
    if not destination.exists():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    if hash_tree(source) != before or hash_tree(destination) != before:
        raise ValueError("compiler source changed while capturing capsule input")
    return {
        "package": str(source),
        "candidate_sha256": before["sha256"],
        "snapshot": "compiler-source/" + before["sha256"],
    }


def _finish_compiler_source(captured, generated):
    """Bind a stable retained compiler to this invocation's emitted artifacts.

    ``executed_programs`` lists, per simulator run, the exact ELF, caller and kernel object that engine
    executed (:mod:`merlin.targetgen.executed_program`); ``artifacts`` names only the bytes left in place
    afterwards, which are the last tier's."""
    from merlin.common.digest import sha256_file
    from merlin.common.tree_hash import hash_tree

    from . import executed_program

    if captured is None:
        return None
    generated = Path(generated)
    executed = executed_program.collect(generated)
    expected = captured["candidate_sha256"]
    if (
        hash_tree(Path(captured["package"]))["sha256"] != expected
        or hash_tree(generated / captured["snapshot"])["sha256"] != expected
    ):
        return {
            "schema": "merlin.capsule_compiler_source.v1",
            "status": "source_changed",
            "executed_programs": executed,
        }
    artifacts = {}
    for name in ("package_kernel.elf", "harness.c", "input.interface.mlir", "command_buffer.json"):
        path = generated / name
        if path.is_file() and not path.is_symlink():
            artifacts[name] = sha256_file(path)
    return {
        "schema": "merlin.capsule_compiler_source.v1",
        "status": "stable",
        **captured,
        "artifacts": artifacts,
        "executed_programs": executed,
    }
