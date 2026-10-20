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


def _pin_model_capsule(capsule: dict, destination: Path) -> dict:
    """Give one grade its own byte-frozen, GC-independent capsule directory."""
    raw = capsule.get("__dir__")
    if not raw:
        return capsule
    from merlin.common import content_store

    lexical_root = Path(raw)
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError(f"model capsule source directory is missing or a symlink: {raw}")
    source = lexical_root.resolve(strict=True)

    def require_local_source(lexical: Path, canonical: Path, _destination: Path) -> None:
        # The generic store dereferences aliases. A model grade must not turn an
        # asset that the runtime bundle rejects into an apparently local file.
        if lexical.is_symlink() or not canonical.is_relative_to(source):
            raise ValueError(f"model capsule snapshot source is a symlink or escapes its directory: {lexical}")

    content_store.place_tree(source, destination, content_store.store_root(), observe=require_local_source)
    # We own these directory entries, not the shared file inodes. Owner-only
    # directories keep withheld assets private and allow cleanup to unlink
    # read-only assets without chmodding a shared store object.
    for directory in [destination, *destination.rglob("*")]:
        if directory.is_dir():
            directory.chmod(0o700)
    if not (destination / "capsule.yaml").is_file():
        raise ValueError(f"model capsule snapshot has no capsule.yaml: {raw}")
    # CAS placement may share source inodes. These two files are subsequently
    # consumed by the compiler/context and reopened as original source roles;
    # give them private inodes BEFORE either consumer receives this capsule.
    import shutil
    import tempfile

    from .native_model_execution import _digest

    interface = Path(capsule.get("interface_mlir", "capsule.interface.mlir"))
    if interface.is_absolute() or ".." in interface.parts:
        raise ValueError("model interface must be local to its capsule")
    transport = {}
    inputs = [("capsule_declaration", Path("capsule.yaml"))]
    if "interface_mlir" in capsule or (source / interface).exists():
        inputs.append(("source_interface", interface))
    for role, relative in inputs:
        original, consumed = source / relative, destination / relative
        if (original.is_symlink() or original.resolve(strict=True) != original
                or not original.is_file() or original.stat().st_size > 64 << 20):
            raise ValueError("model source input is indirect, missing or exceeds byte limit")
        before = original.stat()
        pin = _digest(original)
        with tempfile.NamedTemporaryFile(dir=consumed.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(consumed, temporary)
            temporary.chmod(0o400)
            copied = _digest(temporary)
            after = original.stat()
            stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (copied["sha256"] != pin["sha256"] or _digest(original) != pin
                    or any(getattr(before, key) != getattr(after, key) for key in stable)):
                raise ValueError("model source changed during private input transport")
            temporary.replace(consumed)
        finally:
            temporary.unlink(missing_ok=True)
        transport[role] = {"original": pin, "consumed": _digest(consumed)}
    return {**capsule, "__dir__": str(destination), "__model_source_transport": {
        "schema": "merlin.model_source_input_transport.v1", "inputs": transport}}
