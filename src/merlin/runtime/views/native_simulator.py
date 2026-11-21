"""Live, shared readonly filesystem views for the original native simulator.

This closes filesystem lookup and attributes actual native streams. It does not
establish an unspecified engine's reset/seed or replay determinism. Such normal
selections continue executing, including identical compiled artifacts.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import hmac
import inspect
import io
import json
import locale
import os
import resource
import secrets
import selectors
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from merlin.common.process import invocation_records as I
from merlin.common.storage import retention as storage

# ---- native_runtime ----------------------------------------------------------------------------

"""Live, shared readonly filesystem views for the original native simulator.

This closes filesystem lookup and attributes actual native streams. It does not
establish an unspecified engine's reset/seed or replay determinism. Such normal
selections continue executing, including identical compiled artifacts.
"""


pass
pass

_ACTIVE = contextvars.ContextVar("original_native_runtime_view", default=None)
_LAST = contextvars.ContextVar("original_native_runtime_observation", default=None)
_ISSUED = {}
_MISSING_STATE = "original native engine reset/seed and non-file input closure remain UNKNOWN"


def _backend_data(backend):
    from merlin.runtime.backends.rocc import chipyard_backend as neutral
    from merlin.runtime.backends.rocc.bound_backend import BoundRoCCBackend

    if type(backend) is BoundRoCCBackend:
        block = backend._engine("gsim")
        argv = backend.execution_argv_template("gsim")
        selection = backend._selection_json
        if argv != tuple(block["argv"]):
            selection = document({"backend_selection": selection, "effective_argv": argv})
        return {
            "engine": block["binary"]["path"],
            "cwd": block["cwd"],
            "environment": dict(block["environment"]),
            "argv": argv,
            "selection": selection,
            "unlimited_stack": False,
            "text_mode": "utf8",
            "source": Path(inspect.getsourcefile(backend.run_elf)).resolve(),
            "code": backend.run_elf.__func__.__code__,
        }
    if backend is neutral:
        return {
            "engine": str(backend.gsim_path()),
            "cwd": str(Path.cwd()),
            "environment": dict(os.environ),
            "argv": ("{elf}", f"+max-cycles={backend.gsim_max_cycles()}", "+loadmem={elf}"),
            "selection": backend._target(),
            "unlimited_stack": True,
            "text_mode": locale.getencoding(),
            "source": Path(inspect.getsourcefile(backend.run_elf)).resolve(),
            "code": backend.run_elf.__code__,
        }
    raise ValueError("native view requires the original normal Chipyard process owner")


def _guest_member(image, absolute):
    """Resolve guest links against guest '/', without borrowing host targets."""
    parts = list(Path(absolute).parts[1:])
    resolved, hops = [], 0
    while parts:
        part = parts.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if resolved:
                resolved.pop()
            continue
        candidate = image.joinpath(*resolved, part)
        if candidate.is_symlink():
            hops += 1
            if hops > 40:
                raise ValueError("native runtime guest lookup contains a link cycle")
            target = os.readlink(candidate)
            if target.startswith("/"):
                resolved = []
            parts = list(Path(target).parts[1:] if target.startswith("/") else Path(target).parts) + parts
        else:
            resolved.append(part)
    return image.joinpath(*resolved)


class NativeRuntimeView:
    """Only the live factory-issued object can adopt or reopen this runtime."""

    def __init__(self, *, root, backend, engine, output_root, bwrap, max_files, max_bytes, max_console_bytes, issuer):
        self.root, self.image = root, root / "image"
        self.engine, self.output_root, self.bwrap = engine, output_root, bwrap
        self.max_files, self.max_bytes, self.max_console_bytes = max_files, max_bytes, max_console_bytes
        self._issuer, self._secret, self._parent_pid = issuer, secrets.token_bytes(32), os.getpid()
        selected = _backend_data(backend)
        self._selection, self._code = selected["selection"], selected["code"]
        self._source = pin(selected["source"], max_bytes=max_bytes)
        self._engine = pin(engine, max_bytes=max_bytes)
        self._bwrap = pin(bwrap, max_bytes=max_bytes)
        self._source_pins = (pin(Path(__file__).resolve(), max_bytes=max_bytes),)
        self._tree = inventory(self.image, max_files=max_files, max_bytes=max_bytes)
        self._stamps = stamps(self.image, max_files=max_files)
        self._image_engine = pin(_guest_member(self.image, str(engine)), max_bytes=max_bytes)
        if self._image_engine["sha256"] != self._engine["sha256"]:
            raise ValueError("native view engine differs from its original selected binary")
        self._pin_token = None
        self._closed = False

    def verify(self, backend=None):
        if _ISSUED.get(self._issuer) is not self or self._closed:
            raise ValueError("native runtime requires the exact live issued owner")
        if stamps(self.image, max_files=self.max_files) != self._stamps:
            raise ValueError("native runtime snapshot membership or bytes changed")
        for pin in (self._source, self._bwrap, *self._source_pins):
            if pin(Path(pin["path"]), max_bytes=self.max_bytes) != pin:
                raise ValueError("native original process or snapshot owner changed")
        if backend is not None:
            selected = _backend_data(backend)
            if (
                selected["selection"] != self._selection
                or selected["code"] is not self._code
                or Path(selected["engine"]) != self.engine
                or pin(self.engine, max_bytes=self.max_bytes) != self._engine
            ):
                raise ValueError("native original backend or selected engine changed")
        return {
            "runtime_view_sha256": self._tree["sha256"],
            "engine": dict(self._engine),
            "process_owner": dict(self._source),
            "selection_sha256": hashlib.sha256(self._selection.encode()).hexdigest(),
            "scope": "shared readonly native filesystem view; reset/seed and replay UNKNOWN",
        }

    def selection(self, backend, elf, *, argv, environment, cwd, capture_bytes, unlimited_stack):
        identity = self.verify(backend)
        elf = canonical(elf)
        if not elf.is_relative_to(self.output_root) or type(capture_bytes) is not bool:
            raise ValueError("native invocation escaped its original output owner")
        selected = _backend_data(backend)
        if (
            dict(environment) != selected["environment"]
            or str(cwd) != selected["cwd"]
            or unlimited_stack != selected["unlimited_stack"]
        ):
            raise ValueError("native invocation differs from its original effective process selection")
        expected = [
            str(self.engine),
            *(str(elf) if token == "{elf}" else token.replace("{elf}", str(elf)) for token in selected["argv"]),
        ]
        if list(argv) != expected:
            raise ValueError("native argv differs from its original selected command")
        guest = [
            str(self.engine),
            *(
                "/run/merlin/input.elf" if token == "{elf}" else token.replace("{elf}", "/run/merlin/input.elf")
                for token in selected["argv"]
            ),
        ]
        guest_cwd = _guest_member(self.image, str(cwd))
        if not guest_cwd.is_dir() or not guest_cwd.is_relative_to(self.image):
            raise ValueError("native cwd has no frozen original lookup view")
        return {
            "runtime": identity,
            "guest_argv": guest,
            "environment": dict(environment),
            "cwd": str(cwd),
            "capture_bytes": capture_bytes,
            "unlimited_stack": unlimited_stack,
            "text_mode": selected["text_mode"],
            "reset_seed_closure": {"status": "UNKNOWN", "reason": _MISSING_STATE},
        }

    def observation_sources(self):
        return tuple(Path(row["path"]) for row in (self._bwrap, self._image_engine, *self._source_pins))

    def execution_identity(self, backend, elf, *, capture_bytes=False):
        selected = _backend_data(backend)
        argv = [
            selected["engine"],
            *(str(elf) if token == "{elf}" else token.replace("{elf}", str(elf)) for token in selected["argv"]),
        ]
        chosen = self.selection(
            backend,
            elf,
            argv=argv,
            environment=selected["environment"],
            cwd=selected["cwd"],
            capture_bytes=capture_bytes,
            unlimited_stack=selected["unlimited_stack"],
        )
        return {**chosen, "environment": I.environment_identity(chosen["environment"])}

    def run(self, backend, elf, *, argv, environment, cwd, timeout, capture_bytes, unlimited_stack, raw_result=False):
        _LAST.set(None)
        selection = self.selection(
            backend,
            elf,
            argv=argv,
            environment=environment,
            cwd=cwd,
            capture_bytes=capture_bytes,
            unlimited_stack=unlimited_stack,
        )
        directory = canonical(Path(elf).parent)
        result, record, adoption = run(
            self, selection, Path(elf), timeout=timeout, directory=directory, unlimited_stack=unlimited_stack
        )
        raw_stdout, raw_stderr = result.stdout, result.stderr
        console = raw_stdout if capture_bytes else _decode(raw_stdout, selection["text_mode"])
        if not capture_bytes and not raw_result:
            result.stdout = console
            result.stderr = _decode(raw_stderr, selection["text_mode"])
        body = {
            "selection": {**selection, "environment": I.environment_identity(environment)},
            "record": pin(record, max_bytes=1024 * 1024),
            "adoption": adoption,
            "elf": pin(Path(elf), max_bytes=self.max_bytes),
            "console_sha256": hashlib.sha256(console if type(console) is bytes else console.encode()).hexdigest(),
            "capture_bytes": capture_bytes,
            "returncode": result.returncode,
        }
        _LAST.set((self, body, self._sign(body)))
        return result

    def _sign(self, body):
        return hmac.new(self._secret, document(body).encode(), hashlib.sha256).hexdigest()

    def completed_observation(self, *, elf, console):
        last = _LAST.get()
        if last is None or last[0] is not self:
            raise ValueError("native owner has no actual completed observation for this call")
        body, signature = last[1:]
        self.reopen_observation(body, signature, elf=elf, console=console)
        return {"body": body, "signature": signature}

    def reopen_observation(self, body, signature, *, elf=None, console=None):
        self.verify()
        if not hmac.compare_digest(signature, self._sign(body)) or body["returncode"] != 0:
            raise ValueError("native observation was not issued by this actual completed process owner")
        pin = body["record"]
        record = Path(pin["path"])
        if not record.is_relative_to(self.output_root) or pin(record, max_bytes=1024 * 1024) != pin:
            raise ValueError("native original process record changed or escaped")
        observed = I.verify(record)
        if observed["kind"] != "subprocess" or observed["stage"] != "native_runtime_process":
            raise ValueError("native observation does not reopen its actual native process")
        if body["elf"] not in observed["inputs"]:
            raise ValueError("native observed process consumed another requested artifact")
        if elf is not None and pin(Path(elf), max_bytes=self.max_bytes)["sha256"] != body["elf"]["sha256"]:
            raise ValueError("native reused observation belongs to different ELF bytes")
        data = Path(observed["stdout"]["path"]).read_bytes()
        value = data if body["capture_bytes"] else _decode(data, body["selection"]["text_mode"])
        wire = value if type(value) is bytes else value.encode()
        supplied = console if type(console) is bytes else console.encode() if type(console) is str else None
        if hashlib.sha256(wire).hexdigest() != body["console_sha256"] or console is not None and supplied != wire:
            raise ValueError("native returned console differs from its actual original raw stream")
        return value

    def close(self):
        """Creating host calls after its original evaluation owners prove cleanup."""
        self.verify()
        if os.getpid() != self._parent_pid:
            raise ValueError("native runtime release belongs to its creating host parent")
        if self._pin_token is not None:
            storage.unpin(self.root, self._pin_token)
        self._closed = True


def prepare_native_runtime_view(
    root, *, backend, engine, readonly_roots, output_root, bwrap, max_files, max_bytes, max_console_bytes
):
    """Freeze one original runtime view, with no per-job filesystem copy."""
    if (
        type(readonly_roots) is not tuple
        or not readonly_roots
        or any(type(value) is not int or value <= 0 for value in (max_files, max_bytes, max_console_bytes))
    ):
        raise ValueError("native runtime snapshot requires explicit positive file/byte budgets")
    selected = _backend_data(backend)
    engine, bwrap = canonical(engine), canonical(bwrap)
    if Path(selected["engine"]) != engine or not os.access(engine, os.X_OK) or not os.access(bwrap, os.X_OK):
        raise ValueError("native snapshot did not select its original executable engine")
    root, output_root = canonical(root, exists=False), canonical(output_root)
    if root.exists() or not output_root.is_dir():
        raise ValueError("native snapshot needs a fresh root and original existing output owner")
    mounts = tuple(
        (Path(value[0]), Path(value[1])) if type(value) is tuple and len(value) == 2 else (Path(value), Path(value))
        for value in readonly_roots
    )
    roots = tuple(source for source, _ in mounts)
    if len(roots) > 32 or any(not path.is_absolute() or path == Path("/") or ".." in path.parts for path in roots):
        raise ValueError("native snapshot cannot include an arbitrary host filesystem")
    if any(root.is_relative_to(path) or output_root.is_relative_to(path) for path in roots):
        raise ValueError("native snapshot overlaps its original writable outputs")
    targets = tuple(target for _, target in mounts)
    if any(not path.is_absolute() or path == Path("/") or ".." in path.parts for path in targets):
        raise ValueError("native runtime guest roots must retain closed absolute lookup paths")
    if any(path.parts[1:2] and path.parts[1] in {"proc", "dev", "sys", "run", "tmp"} for path in targets):
        raise ValueError("native runtime roots cannot borrow mutable kernel or temporary state")
    root.mkdir(mode=0o700)
    image = root / "image"
    image.mkdir(mode=0o700)
    budget = [max_files - 1, max_bytes]
    for path, target in sorted(mounts, key=lambda pair: len(pair[1].parts)):
        destination = _guest_member(image, str(target))
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_member(path, destination, budget=budget)
    image_engine = _guest_member(image, str(engine))
    if not image_engine.exists():
        image_engine.parent.mkdir(parents=True, exist_ok=True)
        copy_member(engine, image_engine, budget=budget)
    for name in ("run/merlin", "run/work", "tmp"):
        path = image / name
        if path.is_symlink():
            raise ValueError("native runtime reserved mount point is linked")
        path.mkdir(parents=True, exist_ok=True)
    (image / "run/merlin/input.elf").touch(exist_ok=False)
    (image / "run/merlin/input.elf").chmod(0o444)
    for tool in ("/bin/bash", "/usr/bin/env"):
        member = _guest_member(image, tool)
        if not member.is_file() or not stat.S_ISREG(member.stat().st_mode) or not os.access(member, os.X_OK):
            raise ValueError("native original runtime omits its fixed bootstrap executable")
    seal_directories(image, max_files=max_files)
    issuer = object()
    owner = NativeRuntimeView(
        root=root,
        backend=backend,
        engine=engine,
        output_root=output_root,
        bwrap=bwrap,
        max_files=max_files,
        max_bytes=max_bytes,
        max_console_bytes=max_console_bytes,
        issuer=issuer,
    )
    _ISSUED[issuer] = owner
    owner.verify(backend)
    owner._pin_token = storage.pin(root, reason="shared original native runtime filesystem view")
    return owner


def _decode(data, mode):
    return data.decode("utf-8") if mode == "utf8" else io.TextIOWrapper(io.BytesIO(data), encoding=mode).read()


def run_in_native_view(
    backend, elf, *, argv, environment, cwd, timeout, capture_bytes, unlimited_stack, raw_result=False
):
    """Unsupported original process choices preserve the fresh default launch."""
    owner = active_native_runtime()
    forget_native_observation()
    if owner is None:
        return None
    if type(owner) is not NativeRuntimeView:
        raise ValueError("native launch lost its exact original runtime owner")
    try:
        owner.selection(
            backend,
            elf,
            argv=argv,
            environment=environment,
            cwd=cwd,
            capture_bytes=capture_bytes,
            unlimited_stack=unlimited_stack,
        )
    except (ValueError, OSError, KeyError, TypeError):
        return None
    return owner.run(
        backend,
        elf,
        argv=argv,
        environment=environment,
        cwd=cwd,
        timeout=timeout,
        capture_bytes=capture_bytes,
        unlimited_stack=unlimited_stack,
        raw_result=raw_result,
    )


def active_native_runtime():
    return _ACTIVE.get()


def forget_native_observation():
    _LAST.set(None)


@contextlib.contextmanager
def executing_native_runtime(owner=None):
    if owner is not None:
        if type(owner) is not NativeRuntimeView:
            raise ValueError("native runtime selection requires an exact live owner")
        owner.verify()
    token = _ACTIVE.set(owner)
    try:
        yield owner
    finally:
        _LAST.set(None)
        _ACTIVE.reset(token)


# ---- _native_files -----------------------------------------------------------------------------
# Bounded file snapshots for an original native runtime view.


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def document(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical(path, *, exists=True):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("native owner path must be canonical and unlinked")
    if exists and not path.exists():
        raise ValueError("native owner path is unavailable")
    return path


def pin(path, *, max_bytes):
    path = canonical(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError("native observation member is special or exceeds its byte bound")
    result = {"path": str(path), "sha256": digest(path)}
    if identity(before) != identity(path.stat()):
        raise ValueError("native observation member changed during its read")
    return result


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


def children(path, *, limit):
    names = []
    with os.scandir(path) as entries:
        for entry in entries:
            if len(names) >= limit:
                raise ValueError("native runtime directory exceeds its file bound")
            names.append(entry.name)
    return sorted(names)


def members(root, *, max_files):
    remaining = max_files

    def visit(path):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("native runtime view exceeds its file bound")
        yield path
        if path.is_dir() and not path.is_symlink():
            for name in children(path, limit=remaining):
                yield from visit(path / name)

    yield from visit(root)


def stamps(root, *, max_files):
    return tuple(
        (path.relative_to(root).as_posix(), identity(path.lstat())) for path in members(root, max_files=max_files)
    )


def inventory(root, *, max_files, max_bytes):
    """Include lookup membership, modes and links; bound reads before hashing."""
    root = canonical(root)
    rows, size = [], 0
    for path in members(root, max_files=max_files):
        before = path.lstat()
        name = path.relative_to(root).as_posix()
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(path)
            resolved = root / target.lstrip("/") if target.startswith("/") else path.parent / target
            # Guest absolute links resolve inside the guest root, never the host.
            if ".." in Path(target).parts and not resolved.absolute().is_relative_to(root):
                raise ValueError("native runtime link escapes its closed view")
            row = {"name": name, "kind": "symlink", "target": target}
        elif stat.S_ISDIR(before.st_mode):
            row = {"name": name, "kind": "directory", "mode": stat.S_IMODE(before.st_mode)}
        elif stat.S_ISREG(before.st_mode):
            size += before.st_size
            if size > max_bytes:
                raise ValueError("native runtime view exceeds its byte bound")
            row = {
                "name": name,
                "kind": "file",
                "mode": stat.S_IMODE(before.st_mode),
                "bytes": before.st_size,
                "sha256": digest(path),
            }
        else:
            raise ValueError("native runtime view contains a special file")
        if identity(before) != identity(path.lstat()):
            raise ValueError("native runtime view changed during inventory")
        rows.append(row)
    return {"sha256": hashlib.sha256(document(rows).encode()).hexdigest(), "files": len(rows), "bytes": size}


def copy_member(source, destination, *, budget):
    """Copy bytes, never mutable hard links; preserve guest lookup names."""
    source, destination = Path(source), Path(destination)
    before = source.lstat()
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("native snapshot exceeds its file bound")
    if stat.S_ISLNK(before.st_mode):
        destination.symlink_to(os.readlink(source))
    elif stat.S_ISDIR(before.st_mode):
        destination.mkdir()
        names = children(source, limit=budget[0])
        for name in names:
            copy_member(source / name, destination / name, budget=budget)
        if names != children(source, limit=budget[0] + len(names)):
            raise ValueError("native source directory changed during snapshot")
    elif stat.S_ISREG(before.st_mode):
        budget[1] -= before.st_size
        if budget[1] < 0:
            raise ValueError("native snapshot exceeds its byte bound")
        source_hash = hashlib.sha256()
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as incoming, destination.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                outgoing.write(chunk)
                source_hash.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            if identity(before) != identity(os.fstat(incoming.fileno())):
                raise ValueError("native source file changed during snapshot")
        if source_hash.hexdigest() != digest(destination):
            raise ValueError("native copied runtime differs from its selected bytes")
        destination.chmod(0o444 | (stat.S_IMODE(before.st_mode) & 0o111))
    else:
        raise ValueError("native snapshot source is special")
    if identity(before) != identity(source.lstat()):
        raise ValueError("native snapshot source changed during copy")


def seal_directories(root, *, max_files):
    for path in sorted(members(root, max_files=max_files), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o555)
    root.chmod(0o555)


# ---- _native_launch ----------------------------------------------------------------------------
# Fixed closed-view native launch and kernel-derived adoption observation.

"""Fixed closed-view native launch and kernel-derived adoption observation."""


def _stack():
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (OSError, ValueError):
        pass  # Preserve the original backend's stack policy.


def _mount_path(value):
    for encoded, decoded in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def _same_inode(left, right):
    a, b = os.stat(left), os.stat(right)
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def verify_adoption(pid, leader, image, work, temporary, elf):
    """Inspect the blocked real child; a planned bind list cannot satisfy this."""
    return verify_process_adoption(pid, leader, image, work, temporary, ((elf, "/run/merlin/input.elf"),))


def verify_process_adoption(pid, leader, image, work, temporary, readonly_mounts):
    """Reopen each exact readonly input bind and the two private writable binds."""
    child = Path("/proc") / str(pid)
    _closed_status((child / "status").read_text())
    ancestor = pid
    for _ in range(32):
        if ancestor == leader:
            break
        data = (Path("/proc") / str(ancestor) / "status").read_text()
        ancestor = next(int(row.partition(":")[2]) for row in data.splitlines() if row.startswith("PPid:"))
    else:
        raise ValueError("native READY sender is not the original launched child")
    if os.stat(child / "ns/mnt").st_ino == os.stat("/proc/self/ns/mnt").st_ino:
        raise ValueError("native process did not adopt a private runtime mount namespace")
    root = child / "root"
    expected = {"/run/work": work, "/tmp": temporary}
    readonly = {str(target): source for source, target in readonly_mounts}
    if len(readonly) != len(readonly_mounts) or not _same_inode(root, image):
        raise ValueError("native process did not adopt its exact runtime and readonly input roster")
    for target, source in readonly.items():
        if not _same_inode(root / target.lstrip("/"), source):
            raise ValueError("native process did not adopt an exact readonly input")
    seen, inventory = set(), []
    for row in (child / "mountinfo").read_text().splitlines():
        left, separator, right = row.partition(" - ")
        fields, backing = left.split(), right.split()
        if not separator or len(fields) < 6 or len(backing) < 3:
            raise ValueError("native runtime mount observation is unsupported")
        target, options = _mount_path(fields[4]), fields[5].split(",")
        if target in expected:
            if "rw" not in options or not _same_inode(root / target.lstrip("/"), expected[target]):
                raise ValueError("native writable backing differs from its fresh private scratch")
        elif target in {"/", *readonly}:
            if "ro" not in options:
                raise ValueError("native runtime or input mount is writable")
        else:
            raise ValueError("native runtime exposed an unrelated mount")
        seen.add(target)
        inventory.append({"target": target, "options": options, "device": fields[2], "filesystem": backing[0]})
    if seen != {"/", *readonly, *expected}:
        raise ValueError("native process omitted an original closed-view mount")
    return {
        "pid": pid,
        "mount_namespace_inode": os.stat(child / "ns/mnt").st_ino,
        "mounts": sorted(inventory, key=lambda row: row["target"]),
    }


def _closed_status(text):
    observed = dict(row.split(":", 1) for row in text.splitlines() if ":" in row)
    if (
        int(observed["CapEff"].strip(), 16)
        or int(observed["CapPrm"].strip(), 16)
        or observed["NoNewPrivs"].strip() != "1"
        or tuple(map(int, observed["NSpid"].split()))[-1] != 1
    ):
        raise ValueError("native runtime lacks an unprivileged private namespace leader")


def _ready(control, process, *, timeout):
    control.settimeout(timeout)
    packet, ancillary, flags, _ = control.recvmsg(32, socket.CMSG_SPACE(struct.calcsize("3i")))
    credentials = [
        struct.unpack("3i", data)
        for level, kind, data in ancillary
        if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
    ]
    if packet != b"READY" or flags or len(credentials) != 1 or credentials[0][1] != os.getuid():
        raise ValueError("native runtime has no kernel-authenticated READY process")
    pid = credentials[0][0]
    if pid <= 0 or process.poll() is not None:
        raise ValueError("native runtime exited before adoption verification")
    return pid


def _streams(process, *, deadline, max_bytes, writable_budget=None):
    data = {"stdout": bytearray(), "stderr": bytearray()}
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ, "stdout")
        poll.register(process.stderr, selectors.EVENT_READ, "stderr")
        while poll.get_map():
            if writable_budget is not None:
                _measure_writable(*writable_budget)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    process.args, 0, output=bytes(data["stdout"]), stderr=bytes(data["stderr"])
                )
            for key, _ in poll.select(min(0.1, max(0, deadline - time.monotonic()))):
                total = sum(map(len, data.values()))
                chunk = os.read(key.fileobj.fileno(), min(65536, max_bytes + 1 - total))
                if not chunk:
                    poll.unregister(key.fileobj)
                else:
                    data[key.data].extend(chunk)
                    if sum(map(len, data.values())) > max_bytes:
                        raise ValueError("native captured streams exceed their original byte budget")
    process.wait(timeout=max(0.001, deadline - time.monotonic()))
    return subprocess.CompletedProcess(process.args, process.returncode, bytes(data["stdout"]), bytes(data["stderr"]))


def _measure_writable(roots, max_files, max_bytes):
    """Bound the observed private products; this is not an aggregate disk quota."""
    pass

    files, size = 0, 0
    for root in roots:
        for path in members(root, max_files=max_files):
            files += 1
            if files > max_files or path.is_symlink():
                raise ValueError("native private products exceed their closed file budget")
            if path.is_file():
                size += path.stat().st_size
                if size > max_bytes:
                    raise ValueError("native private products exceed their byte budget")


@contextlib.contextmanager
def _retained_scratch(directory):
    owner = SimpleNamespace(path=Path(tempfile.mkdtemp(prefix="native-call-", dir=directory)), closed=False)
    try:
        yield owner
    finally:
        if owner.closed:
            shutil.rmtree(owner.path)
        # Unknown process closure retains the original writable backing. The
        # enclosing original guardian still owns and reaps the invocation.


def run(owner, selection, elf, *, timeout, directory, unlimited_stack):
    """No caller callbacks or shell payloads: the bootstrap only gates and execs."""
    return run_process(
        owner,
        selection,
        readonly_mounts=((elf, "/run/merlin/input.elf"),),
        inputs=(elf,),
        timeout=timeout,
        directory=directory,
        unlimited_stack=unlimited_stack,
    )


def run_process(
    owner, selection, *, readonly_mounts, inputs, timeout, directory, unlimited_stack=False, outputs=(), build=False
):
    """One shared kernel-verified namespace launch for native and build owners."""
    stage = "native_runtime_process"
    if build:
        # Reopen the live owner's complete current request before preserving
        # the original compiler step in the actual process record. Saved stage
        # claims cannot select this boundary independently of that request.
        selection = owner.reopen_selection(selection)
        stage = selection["identity"]["stage"]
        selection = {
            "guest_argv": selection["identity"]["guest_argv"],
            "cwd": selection["identity"]["guest_cwd"],
            "environment": selection["child_environment"],
            "storage_directories": selection["identity"]["storage_directories"],
            **selection,
        }
    with _retained_scratch(directory) as retained:
        scratch = retained.path
        work, temporary = scratch / "work", scratch / "tmp"
        work.mkdir()
        temporary.mkdir()
        if build:
            for selected_path in selection.get("storage_directories", ()):
                relative = Path(selected_path).relative_to("/tmp")
                if len(relative.parts) != 1 or relative.name in {".", ".."}:
                    raise ValueError("build storage selection escaped private temporary backing")
                (temporary / relative).mkdir()
        # Bash read consumes bytes; SEQPACKET would discard the remainder of
        # a GO packet after its first byte and hang before engine exec.
        control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        # Source snapshot leases can retain thousands of descriptors. Only this
        # socket is inherited via Popen(pass_fds); bubblewrap forwards inherited
        # descriptors and has no --preserve-fds option. Never pass a whole range.
        descriptor = child.fileno()
        if not 3 <= descriptor <= 65535:
            control.close()
            child.close()
            raise ValueError("native private channel exceeds its fixed descriptor bound")
        script = (
            f"printf READY >&{descriptor}; read -r native_go <&{descriptor}; "
            f'[[ $native_go == GO ]] || exit 126; exec {descriptor}>&-; exec /usr/bin/env -- "$@"'
        )
        argv = [
            str(owner.bwrap),
            "--unshare-all",
            "--unshare-user",
            "--as-pid-1",
            "--cap-drop",
            "ALL",
            "--disable-userns",
            "--assert-userns-disabled",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--ro-bind",
            str(owner.image),
            "/",
            *(token for source, target in readonly_mounts for token in ("--ro-bind", str(source), str(target))),
            "--bind",
            str(work),
            "/run/work",
            "--bind",
            str(temporary),
            "/tmp",
            "--chdir",
            selection["cwd"],
            "--",
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
            "native-runtime",
            *(f"{key}={value}" for key, value in sorted(selection["environment"].items())),
            *selection["guest_argv"],
        ]
        process, child_fd, adoption = None, None, None
        deadline = time.monotonic() + timeout
        try:
            with I.observe(
                directory,
                stage=stage,
                argv=argv,
                cwd=directory,
                env={},
                inputs=inputs,
                outputs=outputs,
                dependencies=owner.observation_sources(),
            ) as observed:
                process = subprocess.Popen(
                    argv,
                    cwd=directory,
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(descriptor,),
                    start_new_session=True,
                    preexec_fn=_stack if unlimited_stack else None,
                )
                child.close()
                pid = _ready(control, process, timeout=min(10, max(0.001, deadline - time.monotonic())))
                child_fd = os.pidfd_open(pid)
                adoption = (
                    verify_process_adoption(pid, process.pid, owner.image, work, temporary, readonly_mounts)
                    if build
                    else verify_adoption(pid, process.pid, owner.image, work, temporary, inputs[0])
                )
                owner.verify()
                if build:
                    resource.prlimit(pid, resource.RLIMIT_FSIZE, (owner.max_output_bytes, owner.max_output_bytes))
                control.send(b"GO\n")
                result = _streams(
                    process,
                    deadline=deadline,
                    max_bytes=owner.max_console_bytes,
                    **(
                        {"writable_budget": ((work, temporary), owner.max_output_files, owner.max_output_bytes)}
                        if build
                        else {}
                    ),
                )
                # The child is PID 1 of its private PID namespace. Its actual
                # exit closes that namespace's descendants; the wrapper is waited.
                with selectors.DefaultSelector() as exited:
                    exited.register(child_fd, selectors.EVENT_READ)
                    if not exited.select(0):
                        raise ValueError("native namespace leader has not actually exited")
                retained.closed = True
                if build and result.returncode == 0:
                    _measure_writable((work, temporary), owner.max_output_files, owner.max_output_bytes)
                    owner.collect_products(selection, work)
                observed.complete(result)
                owner.verify()
                return result, observed.path, adoption
        finally:
            child.close()
            control.close()
            if child_fd is not None:
                try:
                    signal.pidfd_send_signal(child_fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            if child_fd is not None:
                with selectors.DefaultSelector() as exited:
                    exited.register(child_fd, selectors.EVENT_READ)
                    retained.closed = bool(exited.select(5)) and process.poll() is not None
                os.close(child_fd)
