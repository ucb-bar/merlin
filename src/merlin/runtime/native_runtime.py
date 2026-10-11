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
import locale
import os
import secrets
import stat
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import storage_lifecycle as storage

from . import _native_files as F
from . import _native_launch as L

_ACTIVE = contextvars.ContextVar("original_native_runtime_view", default=None)
_LAST = contextvars.ContextVar("original_native_runtime_observation", default=None)
_ISSUED = {}
_MISSING_STATE = "original native engine reset/seed and non-file input closure remain UNKNOWN"


def _backend_data(backend):
    from .backends import chipyard_rocc as neutral
    from .backends.chipyard_rocc_bound import BoundRoCCBackend

    if type(backend) is BoundRoCCBackend:
        block = backend._engine("gsim")
        return {
            "engine": block["binary"]["path"],
            "cwd": block["cwd"],
            "environment": dict(block["environment"]),
            "argv": tuple(block["argv"]),
            "selection": backend._selection_json,
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
        self._source = F.pin(selected["source"], max_bytes=max_bytes)
        self._engine = F.pin(engine, max_bytes=max_bytes)
        self._bwrap = F.pin(bwrap, max_bytes=max_bytes)
        self._source_pins = tuple(F.pin(Path(module.__file__).resolve(), max_bytes=max_bytes) for module in (F, L)) + (
            F.pin(Path(__file__).resolve(), max_bytes=max_bytes),
        )
        self._tree = F.inventory(self.image, max_files=max_files, max_bytes=max_bytes)
        self._stamps = F.stamps(self.image, max_files=max_files)
        self._image_engine = F.pin(_guest_member(self.image, str(engine)), max_bytes=max_bytes)
        if self._image_engine["sha256"] != self._engine["sha256"]:
            raise ValueError("native view engine differs from its original selected binary")
        self._pin_token = None
        self._closed = False

    def verify(self, backend=None):
        if _ISSUED.get(self._issuer) is not self or self._closed:
            raise ValueError("native runtime requires the exact live issued owner")
        if F.stamps(self.image, max_files=self.max_files) != self._stamps:
            raise ValueError("native runtime snapshot membership or bytes changed")
        for pin in (self._source, self._bwrap, *self._source_pins):
            if F.pin(Path(pin["path"]), max_bytes=self.max_bytes) != pin:
                raise ValueError("native original process or snapshot owner changed")
        if backend is not None:
            selected = _backend_data(backend)
            if (
                selected["selection"] != self._selection
                or selected["code"] is not self._code
                or Path(selected["engine"]) != self.engine
                or F.pin(self.engine, max_bytes=self.max_bytes) != self._engine
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
        elf = F.canonical(elf)
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
        directory = F.canonical(Path(elf).parent)
        result, record, adoption = L.run(
            self, selection, Path(elf), timeout=timeout, directory=directory, unlimited_stack=unlimited_stack
        )
        raw_stdout, raw_stderr = result.stdout, result.stderr
        console = raw_stdout if capture_bytes else _decode(raw_stdout, selection["text_mode"])
        if not capture_bytes and not raw_result:
            result.stdout = console
            result.stderr = _decode(raw_stderr, selection["text_mode"])
        body = {
            "selection": {**selection, "environment": I.environment_identity(environment)},
            "record": F.pin(record, max_bytes=1024 * 1024),
            "adoption": adoption,
            "elf": F.pin(Path(elf), max_bytes=self.max_bytes),
            "console_sha256": hashlib.sha256(console if type(console) is bytes else console.encode()).hexdigest(),
            "capture_bytes": capture_bytes,
            "returncode": result.returncode,
        }
        _LAST.set((self, body, self._sign(body)))
        return result

    def _sign(self, body):
        return hmac.new(self._secret, F.document(body).encode(), hashlib.sha256).hexdigest()

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
        if not record.is_relative_to(self.output_root) or F.pin(record, max_bytes=1024 * 1024) != pin:
            raise ValueError("native original process record changed or escaped")
        observed = I.verify(record)
        if observed["kind"] != "subprocess" or observed["stage"] != "native_runtime_process":
            raise ValueError("native observation does not reopen its actual native process")
        if body["elf"] not in observed["inputs"]:
            raise ValueError("native observed process consumed another requested artifact")
        if elf is not None and F.pin(Path(elf), max_bytes=self.max_bytes)["sha256"] != body["elf"]["sha256"]:
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
    engine, bwrap = F.canonical(engine), F.canonical(bwrap)
    if Path(selected["engine"]) != engine or not os.access(engine, os.X_OK) or not os.access(bwrap, os.X_OK):
        raise ValueError("native snapshot did not select its original executable engine")
    root, output_root = F.canonical(root, exists=False), F.canonical(output_root)
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
        F.copy_member(path, destination, budget=budget)
    image_engine = _guest_member(image, str(engine))
    if not image_engine.exists():
        image_engine.parent.mkdir(parents=True, exist_ok=True)
        F.copy_member(engine, image_engine, budget=budget)
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
    F.seal_directories(image, max_files=max_files)
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
