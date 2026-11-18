"""Retain exact ordinary capsule compiler source and bind emitted artifacts."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any


# ---- source ------------------------------------------------------------------------------------


class SharedCompilerSource:
    """One retained compiler tree per live evaluator, guarded across workers.

    The guard only reuses a full content check while its watches remain intact.
    Any event, unavailable watcher, or mount change reopens both trees. This
    object is host-owned and never selected by candidate metadata.
    """

    def __init__(self, source, results, *, expected_source_sha256=None):
        import os
        import shutil
        import threading

        from merlin.common.digest import hash_tree
        from merlin.common.storage.bundle_snapshots import _RosterGuard
        from merlin.development.backend.optimization.retained_revisions import source_inventory

        self.source, self.results = Path(source), Path(results)
        self.pid, self.closed = os.getpid(), False
        self.lock = threading.RLock()
        self.guards = []
        if (
            self.results.resolve() != self.results
            or self.results.is_symlink()
            or self.results.is_relative_to(self.source)
        ):
            raise ValueError("retained compiler evidence must be outside its source")
        identity = source_inventory(self.source, max_bytes=128 << 20, max_files=16384, exclude_generated=True)
        if expected_source_sha256 is not None and identity["sha256"] != expected_source_sha256:
            raise ValueError("compiler source differs from evaluator admission")
        self.guards.append(_RosterGuard(self.source, _source_members(self.source), scanner=_source_members))
        self.digest = hash_tree(self.source)
        self.destination = self.results / "compiler-source" / self.digest["sha256"]
        self.destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.destination.resolve() != self.destination or self.destination.is_symlink():
            self.close()
            raise ValueError("retained compiler evidence is indirect")
        try:
            if not self.destination.exists():
                shutil.copytree(
                    self.source, self.destination, ignore=shutil.ignore_patterns("build", "__pycache__", ".git")
                )
            self.guards.append(
                _RosterGuard(self.destination, _source_members(self.destination), scanner=_source_members)
            )
            self._verify_full()
        except BaseException:
            self.close()
            raise

    def _verify_full(self):
        from merlin.common.digest import hash_tree
        from merlin.development.backend.optimization.retained_revisions import source_inventory

        for root in (self.source, self.destination):
            source_inventory(root, max_bytes=128 << 20, max_files=16384, exclude_generated=True)
            if hash_tree(root) != self.digest:
                raise ValueError("compiler source changed during capsule evaluation")

    def check(self):
        import os

        from merlin.common.storage.bundle_snapshots import _RosterGuard

        # Forked workers may inherit a lock held by a vanished thread.
        if self.closed or os.getpid() != self.pid:
            raise ValueError("retained compiler evidence owner is not live")
        with self.lock:
            if self.closed:
                raise ValueError("retained compiler evidence owner is not live")
            if all(guard.unchanged() for guard in self.guards):
                return
                # Rearm BEFORE full verification: writes during the read invalidate
                # the new guard and cannot license reuse at the next boundary.
            for guard in self.guards:
                guard.close()
            try:
                self.guards = []
                for root in (self.source, self.destination):
                    self.guards.append(_RosterGuard(root, _source_members(root), scanner=_source_members))
                self._verify_full()
            except BaseException:
                self.close()
                raise

    def capture(self):
        self.check()
        return {
            "schema": "merlin.capsule_compiler_source.v2",
            "package": str(self.source),
            "candidate_sha256": self.digest["sha256"],
            "snapshot": str(self.destination),
        }

    def close(self):
        with self.lock:
            for guard in self.guards:
                guard.close()
            self.closed = True


def _source_members(root):
    """No-follow source roster, with the same generated-directory exclusions."""
    import os
    import stat

    if root.resolve() != root or root.is_symlink():
        raise ValueError("compiler source is indirect")
    members, pending = [], [(root, ())]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in {"build", "__pycache__", ".git"}:
                    continue
                parts = (*prefix, entry.name)
                info = entry.stat(follow_symlinks=False)
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    kind, size = "directory", None
                    pending.append((path, parts))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    kind, size = "file", info.st_size
                else:
                    raise ValueError("compiler source contains a linked or special member")
                members.append((parts, path, kind, size))
    return sorted(members, key=lambda member: member[0])


def _shared_source(package=None):
    from merlin.development.backend.feedback.snapshot_execution import active_snapshot

    owner = active_snapshot()
    evidence = getattr(owner, "source_evidence", None)
    if evidence is not None and (package is None or Path(package).resolve() == evidence.source):
        return evidence
    return None


def _capture_compiler_source(package_dir, generated):
    """Retain the actual compiler before lowering, without generated build state."""
    from merlin.common.digest import hash_tree
    from merlin.development.backend.optimization.retained_revisions import source_inventory
    from merlin.development.targetgen import executed_program

    # A reused build directory still holds an earlier invocation's executed programs; the receipt
    # finished below names only what this invocation's tiers execute.
    executed_program.clear(generated)
    shared = _shared_source(package_dir)
    if shared is not None:
        return shared.capture()
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
        _place_source(source, destination)
    if hash_tree(source) != before or hash_tree(destination) != before:
        raise ValueError("compiler source changed while capturing capsule input")
    return {
        "package": str(source),
        "candidate_sha256": before["sha256"],
        "snapshot": "compiler-source/" + before["sha256"],
    }


def _place_source(source: Path, destination: Path) -> None:
    """Freeze the compiler as read-only links to one shared copy of each distinct file.

    Every capsule of a grade, and every self-check, retained its own copy of the same compiler: 67
    identical 16 MB trees in one grade and 14 GB of copies across one run's self-checks. The
    content store keeps one copy of each distinct file and links it here, so an unchanged file costs
    nothing in later grades either. The retained bytes are still a copy the candidate cannot reach:
    an in-place edit of the source changes its own inode, never the store's. Placement is staged
    and renamed, so an interrupted one (a full disk) never leaves a partial tree under the final
    name for the next capsule to reuse.
    """
    import os
    import shutil

    from merlin.common.storage import content_store

    try:
        # Its own namespace: these links must not change the link counts of the run's frozen inputs.
        root = content_store.store_root("compiler-source")
    except OSError:
        root = None  # an unwritable store degrades to a private read-only copy
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        # Full digests, no read leases: a lease recheck waits for a CPU slot this grade already holds.
        with content_store.without_read_leases():
            content_store.place_tree(
                source, staging, root, ignore=shutil.ignore_patterns("build", "__pycache__", ".git")
            )
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise  # not a concurrent capture of the same bytes winning the rename
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _finish_compiler_source(captured, generated):
    """Bind a stable retained compiler to this invocation's emitted artifacts.

    ``executed_programs`` lists, per simulator run, the exact ELF, caller and kernel object that engine
    executed (:mod:`merlin.development.targetgen.executed_program`); ``artifacts`` names only the bytes left in place
    afterwards, which are the last tier's."""
    from merlin.common.digest import hash_tree, sha256_file
    from merlin.development.targetgen import executed_program

    if captured is None:
        return None
    generated = Path(generated)
    executed = executed_program.collect(generated)
    expected = captured["candidate_sha256"]
    schema = captured.get("schema", "merlin.capsule_compiler_source.v1")
    if schema == "merlin.capsule_compiler_source.v2":
        shared = _shared_source(captured["package"])
        try:
            stable = shared is not None and shared.capture() == captured
        except (OSError, ValueError, RuntimeError):
            stable = False
    else:
        stable = (
            hash_tree(Path(captured["package"]))["sha256"] == expected
            and hash_tree(generated / captured["snapshot"])["sha256"] == expected
        )
    if not stable:
        return {
            "schema": schema,
            "status": "source_changed",
            "executed_programs": executed,
        }
    artifacts = {}
    for name in ("package_kernel.elf", "harness.c", "input.interface.mlir", "command_buffer.json"):
        path = generated / name
        if path.is_file() and not path.is_symlink():
            artifacts[name] = sha256_file(path)
    return {
        "schema": schema,
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
    from merlin.common.storage import content_store

    lexical_root = Path(raw)
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError(f"model capsule source directory is missing or a symlink: {raw}")
    source = lexical_root.resolve(strict=True)

    def require_local_source(lexical: Path, canonical: Path, _destination: Path) -> None:
        # The generic store dereferences aliases. A model grade must not turn an
        # asset that the runtime bundle rejects into an apparently local file.
        if lexical.is_symlink() or not canonical.is_relative_to(source):
            raise ValueError(f"model capsule snapshot source is a symlink or escapes its directory: {lexical}")

            # Its own store namespace: per-grade pins must not change the link counts of the run's frozen inputs.

    store = content_store.store_root("model-capsule")
    content_store.place_tree(source, destination, store, observe=require_local_source)
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

    from merlin.development.targetgen.native_model_execution import _digest

    interface = Path(capsule.get("interface_mlir", "capsule.interface.mlir"))
    if interface.is_absolute() or ".." in interface.parts:
        raise ValueError("model interface must be local to its capsule")
    transport = {}
    inputs = [("capsule_declaration", Path("capsule.yaml"))]
    if "interface_mlir" in capsule or (source / interface).exists():
        inputs.append(("source_interface", interface))
    for role, relative in inputs:
        original, consumed = source / relative, destination / relative
        if (
            original.is_symlink()
            or original.resolve(strict=True) != original
            or not original.is_file()
            or original.stat().st_size > 64 << 20
        ):
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
            if (
                copied["sha256"] != pin["sha256"]
                or _digest(original) != pin
                or any(getattr(before, key) != getattr(after, key) for key in stable)
            ):
                raise ValueError("model source changed during private input transport")
            temporary.replace(consumed)
        finally:
            temporary.unlink(missing_ok=True)
        transport[role] = {"original": pin, "consumed": _digest(consumed)}
    return {
        **capsule,
        "__dir__": str(destination),
        "__model_source_transport": {"schema": "merlin.model_source_input_transport.v1", "inputs": transport},
    }


# ---- identity ----------------------------------------------------------------------------------
# Capsule evaluation: identity.

CONTRACT_VERSION = "0.1"


_MODEL_NUMERIC_UNSET = object()


_SUITE_MODEL_PIN = object()


def concurrency_stamp(workers: int | None = None) -> dict:
    """What else the host was doing at the moment of measurement.

    Taken at measurement time, not at suite start: load average moves, and a stamp that describes the
    machine an hour ago describes nothing. ``workers`` is the fan-out this suite was told to use;
    ``None`` means the caller did not state one (a direct :func:`run_capsule` call), and it stays
    ``None`` rather than being guessed at 1 -- other processes share this host.

    Why this exists at all: cycle counts are concurrency-invariant and wall times are not, by a
    measured factor of 6.3x. Every wall number in a ``timing`` block is uninterpretable without it,
    and 22,845 blocks already on disk have to be read as "concurrency unrecorded" because it did not
    exist when they were written. They are NOT retro-labelled: the concurrency they ran at is gone.
    """
    try:
        load = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):  # no load average on this platform
        load = None
    return {
        "workers": None if workers is None else int(workers),
        "nproc": os.cpu_count(),
        "load_avg": load,
        "serial": None if workers is None else bool(int(workers) <= 1),
        "sampled": "at the adapter's return, so the load average covers the measurement itself",
    }


def _graded_program(res: Any, package_dir: str | Path) -> str | None:
    """WHICH PROGRAM produced this tier's cycle count, or ``None`` when it cannot be established.

    Only an adapter that BUILT AND RAN AN ELF from the package under grade may claim the submission,
    and ``res["elf"]`` is that evidence. A tier graded on a harness fixture -- a block-scaled capsule
    runs the harness's own reference kernel, not the submission -- returns ``None`` and stays
    UNATTRIBUTED. Defaulting the other way is what let a fixture's pass travel as a submission's, in
    the recorded run that scored 40/40 with nine of the forty measuring the fixture.

    The path is RESOLVED to match :func:`submission_identity`, which records the resolved package on
    every tier record. Comparing an unresolved string against a resolved one makes
    ``comparand.attribute`` return OTHER_PROGRAM -- a FALSE claim about a different program, strictly
    worse than the UNATTRIBUTED it replaces.
    """
    if not isinstance(res, dict) or not res.get("elf"):
        return None
    pkg = Path(package_dir)
    return str(pkg.resolve()) if pkg.exists() else str(pkg)


def submission_identity(package_dir: str | Path, *, run_id: str | None = None) -> dict:
    """WHICH SUBMISSION a measurement belongs to, stated explicitly.

    The harvest layer has had to infer this from where a file sits (the parent of the nearest
    enclosing ``runs`` directory), and a path heuristic mis-keys as soon as a layout changes -- which
    pools incomparable programs into one series. The package directory is the submission and is known
    here for certain, so it is recorded; the campaign labels (run id, arm, round) are supplied by
    whichever harness is driving, through the environment, and are ``None`` when nothing supplied
    them rather than being reconstructed from a guess.
    """
    pkg = Path(package_dir)
    env = os.environ
    ident = {
        "package": str(pkg.resolve()) if pkg.exists() else str(pkg),
        "run_id": run_id or env.get("MERLIN_SUBMISSION_RUN_ID") or None,
        "arm": env.get("MERLIN_SUBMISSION_ARM") or None,
        "round": env.get("MERLIN_SUBMISSION_ROUND") or None,
    }
    ident["stated"] = sorted(k for k, v in ident.items() if k != "package" and v)
    return ident
