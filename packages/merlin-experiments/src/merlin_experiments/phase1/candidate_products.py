"""Box one evaluated compiler; retain data through existing invocation records.

Only a caller-selected frozen package sandbox is accepted. Candidate Python is
never imported here. Products and observations establish compilation identity,
not numerical, target, runtime or qualification authority. Retained workspaces
are private operator artifacts and must outlive invocation reopening.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import string
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.execution_deadline import ExecutionDeadline
from merlin.common.source_membership import file_members
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin_experiments.phase2 import campaign, qualification_policy


def _plain(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("compiler product owner must be an ordinary canonical path")
    return path


def _bytes(path: Path, maximum: int, deadline: ExecutionDeadline, *, unique=False) -> bytes:
    path = _plain(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum or unique and observed.st_nlink != 1:
            raise ValueError("compiler file is nonregular, aliased or exceeds its bound")
        chunks, size = [], 0
        while True:
            deadline.remaining()
            data = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not data:
                break
            chunks.append(data)
            size += len(data)
            if size > maximum:
                raise ValueError("compiler file exceeds its bound")
        fields = ("st_dev", "st_ino", "st_size", "st_mode", "st_nlink", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(current, field) != getattr(observed, field)
            for current in (os.fstat(descriptor), path.stat())
            for field in fields
        ):
            raise ValueError("compiler file changed during observation")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _descriptor(path: Path, data: bytes) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _tree(root: Path, deadline: ExecutionDeadline, maximum: int, count: int) -> dict:
    members = file_members(_plain(root))
    if not members or len(members) > count:
        raise ValueError("compiler source membership is empty or exceeds its bound")
    rows, size = {}, 0
    for name, path in members.items():
        data = _bytes(path, maximum - size, deadline)
        size += len(data)
        rows[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "mode": stat.S_IMODE(path.stat().st_mode) & ~0o222,
        }
    directories = {
        p.relative_to(root).as_posix(): stat.S_IMODE(p.stat().st_mode) & ~0o222
        for p in (root, *root.rglob("*"))
        if p.is_dir() and "__pycache__" not in p.relative_to(root).parts
    }
    if len(directories) + len(rows) > count:
        raise ValueError("compiler source membership exceeds its bound")
    digest = hashlib.sha256(json.dumps([rows, directories], sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "members": rows, "directories": directories}


@dataclass(frozen=True)
class ProductSpec:
    """One host-selected relative product and its maximum retained byte count."""

    relative_path: str
    max_bytes: int

    def __post_init__(self):
        if type(self.relative_path) is not str:
            raise ValueError("compiler product roster requires a relative file name")
        relative = Path(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or "\\" in self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.relative_path
            or type(self.max_bytes) is not int
            or not 0 < self.max_bytes <= 256 * 1024 * 1024
        ):
            raise ValueError("compiler product roster requires bounded canonical relative files")


def _capture(command, *, cwd, environment, deadline, limit):
    """Bound stdio and terminate the bwrap process group, including timeout paths.

    The selected policy's PID namespace closes namespace descendants when its
    init dies. This is not a managed-service/ECHILD or OS hard-deadline receipt.
    """
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        with selectors.DefaultSelector() as selector:
            for name in streams:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline.remaining("compiler invocation budget expired")
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    else:
                        streams[key.data].extend(data)
                        if sum(map(len, streams.values())) > limit:
                            raise ValueError("compiler invocation output exceeds its bound")
            code = process.wait(timeout=deadline.remaining())
        return subprocess.CompletedProcess(command, code, bytes(streams["stdout"]), bytes(streams["stderr"]))
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()
        process.stderr.close()


class CandidateProductTransport:
    """Native-owned frozen policy; no candidate selector or live runtime fallback."""

    def __init__(
        self,
        target_experiment,
        *,
        inputs: campaign.FrozenPackageSandboxInputs,
        bwrap_binary: Path,
        private_paths: tuple[Path, ...],
        workspace_root: Path,
    ):
        if type(inputs) is not campaign.FrozenPackageSandboxInputs:
            raise ValueError("candidate products require an existing frozen runtime selection")
        if type(private_paths) is not tuple or not private_paths:
            raise ValueError("candidate products require explicit private input denials")
        self.target_experiment = target_experiment
        self.inputs = inputs
        self._record = json.dumps(inputs.record, sort_keys=True).encode()
        self.bwrap_binary = _plain(bwrap_binary)
        self.workspace_root = _plain(workspace_root)
        self.private_paths = tuple(_plain(path) for path in private_paths)
        if not self.workspace_root.is_dir() or any(not path.exists() for path in self.private_paths):
            raise ValueError("candidate product workspace or private denial is absent")
        if any(
            self.workspace_root == p or self.workspace_root.is_relative_to(p) or p.is_relative_to(self.workspace_root)
            for p in self.private_paths
        ):
            raise ValueError("candidate compiler workspace overlaps private answers")
        self._bwrap = _descriptor(self.bwrap_binary, self.bwrap_binary.read_bytes())
        self._frozen_inputs()

    def _frozen_inputs(self):
        if json.dumps(self.inputs.record, sort_keys=True).encode() != self._record:
            raise ValueError("candidate compiler runtime selection changed")
        restored = qualification_policy.restore(self.inputs.root, json.loads(self._record))
        attributes = ("root", "repo", "argv", "probes", "env_prefix", "surfaces")
        if (
            any(getattr(restored, key) != getattr(self.inputs, key) for key in attributes)
            or _descriptor(self.bwrap_binary, self.bwrap_binary.read_bytes()) != self._bwrap
        ):
            raise ValueError("candidate compiler runtime or executable changed")
        return restored

    def _tool_substitutions(self, substitutions, frozen):
        if not isinstance(substitutions, dict):
            raise ValueError("candidate compiler tool substitutions require an explicit mapping")
        for key, value in substitutions.items():
            if type(key) is not str or not key.isascii() or not key.isidentifier() or type(value) is not str:
                raise ValueError("candidate compiler tool substitution is malformed")
            path = Path(value)
            if not path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("candidate compiler tool substitution has an indirect path")
            matches = [m for m in frozen.record["mounts"] if path.is_relative_to(Path(m["destination"]))]
            if not matches:
                raise ValueError("candidate compiler tool has no frozen runtime owner")
            mount = max(matches, key=lambda m: len(Path(m["destination"]).parts))
            actual = BW.bundle_snapshot_root(frozen.root / frozen.record["workspace"]) / mount["source"]
            actual = actual / path.relative_to(Path(mount["destination"]))
            if not _plain(actual).exists():
                raise ValueError("candidate compiler selected tool is absent in its frozen runtime")
        return substitutions

    def tool_substitutions(self) -> dict[str, str]:
        """Return selected interpreter/LLVM destinations after reopening their frozen owner."""
        frozen = self._frozen_inputs()
        paths = frozen.record["inputs"]["paths"]
        return self._tool_substitutions(
            {"python": str(Path(paths["venv"]) / "bin/python"), "llvm_bin": str(Path(paths["llvm"]) / "bin")},
            frozen,
        )

    def compile(
        self,
        package: Path,
        prepared: Path,
        *,
        argv: tuple[str, ...],
        products: tuple[ProductSpec, ...],
        destination: Path,
        timeout_s: float,
        substitutions: dict[str, str] | None = None,
        private_paths: tuple[Path, ...] = (),
        max_candidate_bytes: int = 256 * 1024 * 1024,
        max_candidate_files: int = 20_000,
        max_output_bytes: int = 2 * 1024 * 1024,
    ) -> dict:
        deadline = ExecutionDeadline.start(timeout_s)
        if (
            type(argv) is not tuple
            or not argv
            or len(argv) > 256
            or any(type(v) is not str or not v or "\0" in v for v in argv)
            or sum(map(len, argv)) > 131_072
            or type(products) is not tuple
            or not 0 < len(products) <= 64
            or any(type(p) is not ProductSpec for p in products)
            or len({p.relative_path for p in products}) != len(products)
            or sum(p.max_bytes for p in products) > 256 * 1024 * 1024
            or any(type(v) is not int or v <= 0 for v in (max_candidate_bytes, max_candidate_files, max_output_bytes))
            or type(private_paths) is not tuple
        ):
            raise ValueError("candidate compiler request or closed product roster is malformed")
        if any(
            Path(left.relative_path).is_relative_to(Path(right.relative_path))
            for left in products
            for right in products
            if left is not right
        ):
            raise ValueError("candidate compiler product files overlap")
        package, prepared, destination = _plain(package), _plain(prepared), _plain(destination)
        private = (*self.private_paths, *tuple(_plain(path) for path in private_paths))
        if any(not path.exists() for path in private):
            raise ValueError("candidate compiler invocation-private denial is absent")
        if any(
            self.workspace_root == path
            or self.workspace_root.is_relative_to(path)
            or path.is_relative_to(self.workspace_root)
            for path in private
        ):
            raise ValueError("candidate compiler workspace overlaps private invocation inputs")
        if (
            destination.exists()
            or any(destination == p or p.is_relative_to(destination) for p in (package, prepared))
            or destination.is_relative_to(package)
            or self.workspace_root.is_relative_to(package)
        ):
            raise ValueError("candidate products need a fresh owner outside compiler/source inputs")
        frozen = self._frozen_inputs()
        for surface in frozen.surfaces:
            owner = surface.path
            if package == owner or owner.is_relative_to(package) or package.is_relative_to(owner):
                if not any(package.is_relative_to(owner / sub) for sub in surface.grantable):
                    raise ValueError("candidate compiler intake overlaps a withheld source owner")
        tools = self._tool_substitutions(substitutions or {}, frozen)
        BW._validate_host_sources([("candidate", package)], [], private)
        before = _tree(package, deadline, max_candidate_bytes, max_candidate_files)
        source = _bytes(prepared, 64 * 1024 * 1024, deadline, unique=True)
        workspace = self.workspace_root / ("compiler-products-" + uuid.uuid4().hex)
        workspace.mkdir(mode=0o700)
        clone, source_path, output = workspace / "compiler", workspace / "input.mlir", workspace / "products"
        clone.mkdir()
        output.mkdir()
        for name in before["directories"]:
            (clone / name).mkdir(parents=True, exist_ok=True)
        for name, pin in before["members"].items():
            path = clone / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = _bytes(package / name, pin["bytes"], deadline)
            if hashlib.sha256(data).hexdigest() != pin["sha256"]:
                raise ValueError("candidate source changed while creating its immutable compiler view")
            path.write_bytes(data)
            path.chmod(pin["mode"])
        for name, mode in sorted(before["directories"].items(), key=lambda row: len(Path(row[0]).parts), reverse=True):
            (clone / name).chmod(mode)
        source_path.write_bytes(source)
        source_path.chmod(0o400)
        if _tree(clone, deadline, max_candidate_bytes, max_candidate_files) != before:
            raise ValueError("candidate compiler view differs from its selected source")
        policy = campaign.package_sandbox_policy(self.target_experiment, workspace, clone, inputs=frozen)
        prefix = [*policy.argv, "--ro-bind", str(source_path), str(source_path), "--new-session", "--cap-drop", "ALL"]
        prefix[0] = str(self.bwrap_binary)
        denials = [
            AnswerSurface("private compiler input", p, "dir" if p.is_dir() else "file", "private") for p in private
        ]
        # Shared projection includes lexical/physical runtime and private aliases.
        surfaces = [*frozen.surfaces, *denials]
        prefix = BW.apply_answer_masks(prefix, surfaces)
        if BW.coverage_gap(prefix, surfaces):
            raise ValueError("candidate compiler policy exposes private answer surfaces")
        values = {"package": str(clone), "input_mlir": str(source_path), "output_dir": str(output), **tools}
        if set(tools) & {"package", "input_mlir", "output_dir"}:
            raise ValueError("candidate compiler tool substitutions replace input/output owners")
        for token in argv:
            if any(
                field and (field not in values or conversion or format_spec)
                for _, field, format_spec, conversion in string.Formatter().parse(token)
            ):
                raise ValueError("candidate compiler argv has an unsupported placeholder")
        command = [
            *prefix,
            "--chdir",
            str(clone),
            "bash",
            "-c",
            policy.env_prefix + 'exec "$@"',
            "compiler-products",
            *(token.format_map(values) for token in argv),
        ]
        destination.mkdir(parents=True, mode=0o700)
        dependencies = [Path(__file__), self.bwrap_binary, *file_members(clone).values()]
        snapshot = BW.bundle_snapshot_root(frozen.root / frozen.record["workspace"])
        dependencies.extend(file_members(snapshot).values())
        # Host lowering may subsequently rewrite its prepared source. The
        # recorded compiler input is the immutable copy admitted below.
        dependencies.extend(file_members(package).values())
        retained = {p.relative_path: destination / p.relative_path for p in products}
        with invocation_record.observe(
            destination,
            stage="source_bound_compiler_products",
            argv=command,
            cwd=clone,
            env={},
            inputs=(source_path,),
            outputs=(*retained.values(), *(output / p.relative_path for p in products)),
            dependencies=dependencies,
        ) as observation:
            process = _capture(command, cwd=clone, environment={}, deadline=deadline, limit=max_output_bytes)
            if process.returncode:
                observation.complete(process)
                raise ValueError("isolated candidate compiler command failed")
            members = file_members(output)
            directories = {
                str(parent) for p in products for parent in Path(p.relative_path).parents if str(parent) != "."
            }
            if (
                set(members) != set(retained)
                or any("__pycache__" in p.relative_to(output).parts for p in output.rglob("*"))
                or {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_dir()} != directories
            ):
                raise ValueError("candidate compiler product membership differs from its closed roster")
            records = {}
            for product in products:
                data = _bytes(members[product.relative_path], product.max_bytes, deadline, unique=True)
                target = retained[product.relative_path]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o400)
                records[product.relative_path] = _descriptor(target, data)
            if (
                _tree(package, deadline, max_candidate_bytes, max_candidate_files) != before
                or _tree(clone, deadline, max_candidate_bytes, max_candidate_files) != before
                or _bytes(prepared, len(source), deadline, unique=True) != source
                or _bytes(source_path, len(source), deadline, unique=True) != source
            ):
                raise ValueError("candidate compiler or prepared source changed during compilation")
            self._frozen_inputs()
            deadline.remaining()
            observation.complete(process)
            invocation_record.verify(observation.path)
        for path in workspace.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~0o222)
        return {
            "source": _descriptor(prepared, source),
            "package": {"path": str(package), **before},
            "products": records,
            "invocation_record": {
                "path": str(observation.path),
                "sha256": hashlib.sha256(observation.path.read_bytes()).hexdigest(),
            },
            "workspace": str(workspace),
            "frozen_runtime_sha256": frozen.record["sha256"],
            "private_paths": [str(path) for path in private],
        }
