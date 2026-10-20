"""Shared immutable lookup views adopted by original build subprocesses.

Only actual completed, kernel-verified calls issue observations. Product reuse
is a separate caller selection; this owner grants no runtime, numerical or
compiler determinism qualification. Uncovered process choices remain fresh.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import hmac
import io
import locale
import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import storage_lifecycle as storage
from merlin.common.paths import module_source_path

from . import _native_files as F
from . import _native_launch as L
from .native_runtime import _guest_member

_ACTIVE = contextvars.ContextVar("original_build_runtime", default=None)
_BUILD = contextvars.ContextVar("original_build_observation", default=None)
_STORAGE = contextvars.ContextVar("original_build_storage", default=())
_ISSUED = {}


def _hash(value):
    return hashlib.sha256(F.document(value).encode()).hexdigest()


class UnsupportedBuild(ValueError):
    """A process outside this selected view must use its original fresh route."""


def _path(value, cwd):
    path = Path(value)
    return F.canonical(path if path.is_absolute() else Path(cwd) / path)


def _rewrite(token, paths):
    if token in paths:
        return paths[token]
    head, separator, tail = token.partition("=")
    if separator and tail in paths:
        return head + "=" + paths[tail]
    for prefix in ("-I", "-L", "-T", "-o"):
        if token.startswith(prefix) and token[len(prefix) :] in paths:
            return prefix + paths[token[len(prefix) :]]
    return token


def _closed_arguments(argv, paths, image):
    """Recognize original compiler/translator forms with closed lookups.

    Extra flags stay on the original fresh route until their extra inputs and
    products have an explicit argument owner. Target and ABI values are data.
    """
    switches = {
        "-c",
        "--mlir-to-llvmir",
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Os",
        "-Oz",
        "-g",
        "-g0",
        "-ffreestanding",
        "-fno-builtin",
        "-fno-common",
        "-fno-pie",
        "-fno-pic",
        "-fPIC",
        "-fpic",
        "-fPIE",
        "-fno-stack-protector",
        "-fno-asynchronous-unwind-tables",
        "-fno-unwind-tables",
        "-fstack-usage",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-tree-vectorize",
        "-funroll-loops",
        "-static",
        "-nostdlib",
        "-nostartfiles",
        "-nodefaultlibs",
        "-shared",
        "-rdynamic",
        "-no-pie",
        "-pie",
        "-pthread",
        "-w",
        "-Wall",
        "-Wextra",
        "-Werror",
    }
    values = ("--target=", "-march=", "-mabi=", "-mcmodel=", "-mtune=", "-std=", "-D", "-U", "-l")
    link_switches = {
        "--gc-sections",
        "--no-relax",
        "-Bstatic",
        "-Bdynamic",
        "--start-group",
        "--end-group",
        "--as-needed",
        "--no-as-needed",
    }

    def lookup(token):
        return token in paths or (Path(token).is_absolute() and _guest_member(image, token).exists())

    tokens = iter(argv[1:])
    for token in tokens:
        if token in switches or token.startswith("-Wno-"):
            continue
        if token in ("-o", "-I", "-L", "-T"):
            operand = next(tokens, None)
            if operand is None or not lookup(operand):
                return False
        elif any(token.startswith(prefix) and len(token) > len(prefix) for prefix in ("-o", "-I", "-L", "-T")):
            if not lookup(token[2:]):
                return False
        elif any(token.startswith(prefix) and len(token) > len(prefix) for prefix in values):
            continue
        elif token.startswith("-Wl,"):
            if not set(token[4:].split(",")) <= link_switches:
                return False
        elif not lookup(token):
            return False
    return True


class BuildRuntimeView:
    """Exact live shared snapshot, not an issuer reconstructed from a receipt."""

    def __init__(self, *, root, tools, output_root, bwrap, budgets, issuer):
        self.root, self.image, self.output_root, self.bwrap = root, root / "image", output_root, bwrap
        self.max_files, self.max_bytes, self.max_output_files, self.max_output_bytes, self.max_console_bytes = budgets
        self._issuer, self._secret, self._parent_pid = issuer, secrets.token_bytes(32), os.getpid()
        self._tools = tuple(F.pin(path, max_bytes=self.max_bytes) for path in tools)
        self._sources = (
            tuple(F.pin(Path(module.__file__).resolve(), max_bytes=self.max_bytes) for module in (F, L))
            + (
                F.pin(Path(__file__).resolve(), max_bytes=self.max_bytes),
                F.pin(bwrap, max_bytes=self.max_bytes),
            )
            + tuple(
                F.pin(module_source_path(name).resolve(), max_bytes=self.max_bytes)
                for name in (
                    "merlin.runtime.native_runtime",
                    "merlin.common.invocation_record",
                    "merlin.common.paths",
                    "merlin.common.proc",
                    "merlin.common.cpu_pool",
                    "merlin.common.execution_deadline",
                    "merlin.llvmlower.codegen",
                    "merlin.llvmlower.toolchain",
                    "merlin.targetgen.contract.compile",
                )
            )
        )
        self._tree = F.inventory(self.image, max_files=self.max_files, max_bytes=self.max_bytes)
        self._stamps = F.stamps(self.image, max_files=self.max_files)
        self._pin_token, self._closed = None, False

    def verify(self):
        if _ISSUED.get(self._issuer) is not self or self._closed:
            raise ValueError("build view requires the exact live issued owner")
        if F.stamps(self.image, max_files=self.max_files) != self._stamps:
            raise ValueError("build lookup snapshot membership or bytes changed")
        for pin in (*self._tools, *self._sources):
            if F.pin(Path(pin["path"]), max_bytes=self.max_bytes) != pin:
                raise ValueError("original build tool or lookup owner changed")
        return {
            "view_sha256": self._tree["sha256"],
            "tools": list(self._tools),
            "owner_source_pins": list(self._sources),
            "scope": "immutable filesystem lookup only; non-file inputs and determinism UNKNOWN",
        }

    def observation_sources(self):
        return tuple(Path(pin["path"]) for pin in (*self._sources, *self._tools))

    def selection(
        self,
        argv,
        *,
        inputs,
        outputs,
        cwd,
        environment,
        directory,
        stage,
        capture_output=True,
        text=False,
        encoding=None,
        errors=None,
        storage_roots=(),
        _existing_products=False,
    ):
        identity = self.verify()
        try:
            if (
                not argv
                or not inputs
                or not outputs
                or capture_output is not True
                or type(text) is not bool
                or errors not in (None, "strict")
            ):
                raise UnsupportedBuild("build stream or complete input/output selection is unsupported")
            cwd, directory = F.canonical(cwd), F.canonical(directory)
            if not directory.is_relative_to(self.output_root) or not directory.is_dir():
                raise UnsupportedBuild("build directory escaped its original private output owner")
            original = tuple(str(token) for token in argv)
            tool = Path(original[0]).resolve(strict=True)
            if not Path(original[0]).is_absolute() or not any(pin["path"] == str(tool) for pin in self._tools):
                raise UnsupportedBuild("build tool is outside the explicit original tool roster")
            image_tool = _guest_member(self.image, original[0])
            pin = F.pin(tool, max_bytes=self.max_bytes)
            if not image_tool.is_file() or F.digest(image_tool) != pin["sha256"]:
                raise UnsupportedBuild("build tool lookup is absent from the shared immutable view")
            source = tuple(_path(value, cwd) for value in inputs)
            products = tuple(
                F.canonical(Path(value) if Path(value).is_absolute() else cwd / value, exists=False)
                for value in outputs
            )
            if (
                len(source) != len(set(source))
                or len(products) != len(set(products))
                or set(source) & set(products)
                or any(
                    p.parent != directory or (p.exists() and not _existing_products) or (p.exists() and not p.is_file())
                    for p in products
                )
            ):
                raise UnsupportedBuild("build needs fresh disjoint direct products in its owned directory")
            if len(source) > self.max_files or len(products) + 2 > self.max_output_files:
                raise UnsupportedBuild("build input/output roster exceeds its declared file budget")
            parents = list(dict.fromkeys(p.parent for p in source))
            mapped = {str(p): f"/run/merlin/inputs/p{parents.index(p.parent)}/{p.name}" for p in source}
            mapped.update({str(p): "/run/work/" + p.name for p in products})
            mapped.update({str(p): f"/run/merlin/inputs/p{i}" for i, p in enumerate(parents)})
            storage_roots = tuple(F.canonical(p) for p in storage_roots)
            if len(storage_roots) != len(set(storage_roots)) or any(
                not p.is_dir() or not p.is_relative_to(self.output_root) or p == directory or p in parents
                for p in storage_roots
            ):
                raise UnsupportedBuild("build storage relocation must select separate owned directories")
            storage_mapping = {str(p): f"/tmp/storage-{i}" for i, p in enumerate(storage_roots)}
            mapped.update(storage_mapping)
            for original_path, path in zip(inputs, source, strict=True):
                mapped[str(original_path)] = mapped[str(path)]
            for original_path, path in zip(outputs, products, strict=True):
                mapped[str(original_path)] = mapped[str(path)]
            # Relative names are interpreted against the original cwd only;
            # arbitrary suffix/string replacements never select a file.
            for path in (*source, *products):
                if path.is_relative_to(cwd):
                    mapped[str(path.relative_to(cwd))] = mapped[str(path)]
            if str(directory) not in mapped:
                mapped[str(directory)] = "/run/work"
            guest = tuple(_rewrite(token, mapped) for token in original)
            if not _closed_arguments(original, mapped, self.image):
                raise UnsupportedBuild("build arguments have no closed original input/product grammar")
            side_effects = (
                "-M",
                "-MM",
                "-MD",
                "-MMD",
                "-MF",
                "-MJ",
                "-MT",
                "-MQ",
                "-save-temps",
                "--save-temps",
                "-ftime-trace",
                "-fsave-optimization-record",
                "-serialize-diagnostics",
                "--dependency-file",
            )
            if any(token == flag or token.startswith(flag + "=") for token in original for flag in side_effects):
                raise UnsupportedBuild("build has an undeclared product producer")
            external_lookups = ("-fplugin", "-load", "-plugin", "--load", "-fprofile", "-fauto-profile", "-Xclang")
            if any(token.startswith(prefix) for token in original for prefix in external_lookups):
                raise UnsupportedBuild("build plugin or profile lookup has no original argument owner")
            for token in original[1:]:
                for prefix in ("-I", "-L", "-T"):
                    if token.startswith(prefix) and token != prefix:
                        path = token[len(prefix) :]
                        if path not in mapped and (
                            not Path(path).is_absolute() or not _guest_member(self.image, path).exists()
                        ):
                            raise UnsupportedBuild("build includes an unclosed relative or external lookup")
            if any(token.startswith("@") for token in original) or any(
                str(p) not in original and p.name not in original for p in source
            ):
                # Header dependencies are permitted when their include parent
                # is selected; response files require another argument owner.
                hidden = [
                    p
                    for p in source
                    if str(p) not in original
                    and p.name not in original
                    and not any(str(p.parent) in token for token in original)
                ]
                if hidden or any(token.startswith("@") for token in original):
                    raise UnsupportedBuild("build includes unclosed implicit or response-file inputs")
            if not any(token == "-o" or token.startswith("-o") for token in original):
                raise UnsupportedBuild("build command has no explicit output operand")
            environment = dict(environment)
            if any(
                type(k) is not str or type(v) is not str or "=" in k or "\x00" in k + v for k, v in environment.items()
            ):
                raise UnsupportedBuild("build environment is not a closed string mapping")
            child_environment = {key: mapped.get(value, value) for key, value in environment.items()}
            child_environment.update(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
            effective_text = text or encoding is not None or errors is not None
            if not effective_text:
                raise UnsupportedBuild("original build view requires captured text streams")
            effective_encoding = encoding or locale.getencoding()
            effective_errors = errors or "strict"
            # TextIOWrapper owns the same newline/encoding behavior as subprocess.
            io.TextIOWrapper(io.BytesIO(), encoding=effective_encoding, errors=effective_errors).close()
            input_pins = tuple(F.pin(p, max_bytes=self.max_bytes) for p in source)
            if sum(p.stat().st_size for p in source) > self.max_bytes:
                raise UnsupportedBuild("complete build inputs exceed selected byte budget")
            body = {
                **identity,
                "tool": {**pin, "requested_path": original[0]},
                "guest_argv": list(guest),
                "guest_cwd": mapped.get(str(cwd), "/run/work"),
                "environment": I.environment_identity(child_environment),
                "storage_directories": list(storage_mapping.values()),
                "streams": {
                    "text": effective_text,
                    "encoding": effective_encoding,
                    "errors": effective_errors,
                    "newline": None,
                },
                "inputs": [
                    {"guest": mapped[str(p)], "sha256": member["sha256"], "bytes": p.stat().st_size}
                    for p, member in zip(source, input_pins, strict=True)
                ],
                "outputs": [{"guest": mapped[str(p)], "name": p.name} for p in products],
                "stage": str(stage),
            }
            return {
                "key": _hash(body),
                "identity": body,
                "request": {
                    "argv": list(original),
                    "inputs": list(map(str, source)),
                    "outputs": list(map(str, products)),
                    "cwd": str(cwd),
                    "environment": environment,
                    "directory": str(directory),
                    "stage": str(stage),
                    "capture_output": capture_output,
                    "text": effective_text,
                    "encoding": effective_encoding,
                    "errors": effective_errors,
                    "storage_roots": list(map(str, storage_roots)),
                },
                "input_pins": list(input_pins),
                "child_environment": child_environment,
                "original_environment": I.environment_identity(environment),
            }
        except (OSError, TypeError, LookupError, ValueError):
            return None

    def reopen_selection(self, selection):
        expected = self.selection(**selection["request"], _existing_products=True)
        if expected is None or expected != selection:
            raise ValueError("original build selection or fresh input bytes changed")
        return expected

    def collect_products(self, selection, work):
        """Collect only the exact declared artifacts after real namespace exit."""
        actual = F.inventory(work, max_files=self.max_output_files, max_bytes=self.max_output_bytes)
        expected = {row["name"] for row in selection["identity"]["outputs"]}
        if {p.name for p in work.iterdir()} != expected or actual["files"] != len(expected) + 1:
            raise ValueError("closed build produced an undeclared or missing product")
        budget = [self.max_output_files, self.max_output_bytes]
        products = []
        for row, path in zip(selection["identity"]["outputs"], selection["request"]["outputs"], strict=True):
            source, target = work / row["name"], F.canonical(Path(path), exists=False)
            if not target.parent.is_relative_to(self.output_root):
                raise ValueError("build product escaped its private output owner")
            produced = source.lstat()
            if not stat.S_ISREG(produced.st_mode) or produced.st_nlink != 1:
                raise ValueError("build product must be an unlinked regular file")
            previous = target.lstat() if target.exists() else None
            if previous is not None and (not stat.S_ISREG(previous.st_mode) or previous.st_nlink != 1):
                raise ValueError("build destination must be an unlinked regular file")
            products.append((source, target, produced, previous))
        for source, target, produced, previous in products:
            self._promote_product(source, target, produced, previous, budget)

    @staticmethod
    def _promote_product(source, target, produced, previous, budget):
        """Copy first, then replace only the original private regular destination."""
        directory = target.parent
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        staged_root = None
        try:
            parent = os.fstat(descriptor)
            staged_root = Path(tempfile.mkdtemp(prefix="build-product-", dir=directory))
            staged = staged_root / "product"
            F.copy_member(source, staged, budget=budget)
            staged.chmod(stat.S_IMODE(produced.st_mode) & 0o777)
            if F.identity(source.lstat()) != F.identity(produced) or F.digest(staged) != F.digest(source):
                raise ValueError("build product bytes or mode changed during collection")
            staged_identity = F.identity(staged.lstat())
            source_digest = F.digest(staged)
            current_parent = directory.stat()
            if (parent.st_dev, parent.st_ino) != (current_parent.st_dev, current_parent.st_ino):
                raise ValueError("build product output directory changed")
            try:
                current = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if (previous is None) != (current is None) or (
                current is not None
                and (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or F.identity(current) != F.identity(previous)
                )
            ):
                raise ValueError("build product destination changed during collection")
            os.replace(staged, target.name, dst_dir_fd=descriptor)
            completed = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
            if (
                (completed.st_dev, completed.st_ino) != staged_identity[:2]
                or completed.st_nlink != 1
                or stat.S_IMODE(completed.st_mode) != stat.S_IMODE(produced.st_mode) & 0o777
                or F.digest(target) != source_digest
            ):
                raise ValueError("promoted build product differs from its original bytes or permissions")
        finally:
            os.close(descriptor)
            if staged_root is not None:
                shutil.rmtree(staged_root)

    def run(self, selection, *, timeout):
        selection = self.reopen_selection(selection)
        directory = Path(selection["request"]["directory"])
        packed = Path(tempfile.mkdtemp(prefix="build-inputs-", dir=directory))
        try:
            budget = [self.max_files, self.max_bytes]
            for row, path in zip(selection["identity"]["inputs"], selection["request"]["inputs"], strict=True):
                target = packed / Path(row["guest"]).relative_to("/run/merlin/inputs")
                target.parent.mkdir(parents=True, exist_ok=True)
                F.copy_member(Path(path), target, budget=budget)
                if F.digest(target) != row["sha256"]:
                    raise ValueError("copied build input differs from its freshly selected bytes")
            F.seal_directories(packed, max_files=self.max_files)
            result, record, adoption = L.run_process(
                self,
                selection,
                readonly_mounts=((packed, "/run/merlin/inputs"),),
                inputs=tuple(map(Path, selection["request"]["inputs"])),
                outputs=tuple(map(Path, selection["request"]["outputs"])),
                timeout=timeout,
                directory=directory,
                build=True,
            )
            if result.returncode != 0:
                scope = _BUILD.get()
                if scope is not None and scope["owner"] is self:
                    scope["eligible"] = False
                result.args = selection["request"]["argv"]
                self._decode(result, selection)
                return result, None
            observed = I.verify(record)
            self.reopen_selection(selection)
            if observed["stage"] != selection["identity"]["stage"]:
                raise ValueError("actual build stage differs from the original selected step")
            if observed["inputs"] != sorted(selection["input_pins"], key=lambda pin: pin["path"]):
                raise ValueError("actual build inputs differ from the original selected byte roster")
            output_pins = {pin["path"]: pin for pin in observed["outputs"]}
            if set(output_pins) != set(selection["request"]["outputs"]):
                raise ValueError("actual build products differ from the complete selected roster")
            body = {
                "selection_key": selection["key"],
                "identity": selection["identity"],
                "original_request_sha256": _hash(selection["request"]),
                "record": F.pin(record, max_bytes=self.max_console_bytes + 1024 * 1024),
                "inputs": list(observed["inputs"]),
                "outputs": [output_pins[p] for p in selection["request"]["outputs"]],
                "adoption": adoption,
                "returncode": result.returncode,
                "stdout": observed["stdout"],
                "stderr": observed["stderr"],
            }
            observation = {
                "body": body,
                "signature": hmac.new(self._secret, F.document(body).encode(), hashlib.sha256).hexdigest(),
            }
            result.args = selection["request"]["argv"]
            self._decode(result, selection)
            scope = _BUILD.get()
            if scope is not None and scope["owner"] is self:
                scope["observations"].append(observation)
            return result, observation
        finally:
            # Readonly copies can be removed by their creating host after the
            # shared launcher has confirmed namespace exit. Unknown closure is
            # retained under the existing job/guardian owner instead.
            if "adoption" in locals():
                for p in F.members(packed, max_files=self.max_files):
                    if p.is_dir() and not p.is_symlink():
                        p.chmod(0o700)
                shutil.rmtree(packed)

    @staticmethod
    def _decode(result, selection):
        request = selection["request"]
        if request["text"]:
            for field in ("stdout", "stderr"):
                with io.TextIOWrapper(
                    io.BytesIO(getattr(result, field)),
                    encoding=request["encoding"],
                    errors=request["errors"],
                    newline=None,
                ) as stream:
                    setattr(result, field, stream.read())

    def reopen_observation(self, observation, *, selection=None, outputs=None):
        self.verify()
        if selection is not None:
            selection = self.reopen_selection(selection)
        body = observation["body"]
        expected = hmac.new(self._secret, F.document(body).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, observation["signature"]) or body["returncode"] != 0:
            raise ValueError("build observation is not an original completed successful process")
        if selection is not None and (
            body["selection_key"] != selection["key"] or body["identity"] != selection["identity"]
        ):
            raise ValueError("build observation belongs to another complete selection")
        record = body["record"]
        if (
            not Path(record["path"]).is_relative_to(self.output_root)
            or F.pin(Path(record["path"]), max_bytes=self.max_console_bytes + 1024 * 1024) != record
        ):
            raise ValueError("build original process record changed")
        observed = I.verify(Path(record["path"]))
        if (
            observed["stage"] != body["identity"]["stage"]
            or observed["kind"] != "subprocess"
            or not body["adoption"]
            or observed["inputs"] != body["inputs"]
            or observed["outputs"] != sorted(body["outputs"], key=lambda pin: pin["path"])
            or observed["stdout"] != body["stdout"]
            or observed["stderr"] != body["stderr"]
        ):
            raise ValueError("build observation differs from its actual closed process")
        if outputs is not None and list(map(str, outputs)) != [p["path"] for p in body["outputs"]]:
            raise ValueError("build observation has a different complete product roster")
        return body

    def begin_build(self):
        self.verify()
        if _BUILD.get() is not None:
            raise ValueError("original build observation scopes cannot overlap")
        scope = {"owner": self, "observations": [], "eligible": True}
        scope["token"] = _BUILD.set(scope)
        return scope

    def end_build(self, scope):
        if _BUILD.get() is not scope or scope["owner"] is not self:
            raise ValueError("build observation lost its original active owner")
        _BUILD.reset(scope["token"])
        return {
            "eligible": scope["eligible"] and bool(scope["observations"]),
            "observations": tuple(scope["observations"]),
        }

    def verify_build_observation(self, observation):
        if observation.get("eligible") is not True or not observation.get("observations"):
            raise ValueError("build has unsupported or incomplete process closure")
        return tuple(self.reopen_observation(row) for row in observation["observations"])

    def close(self):
        self.verify()
        if os.getpid() != self._parent_pid:
            raise ValueError("build view release belongs to its creating host parent")
        if self._pin_token is not None:
            storage.unpin(self.root, self._pin_token)
        self._closed = True


def prepare_build_runtime_view(
    root,
    *,
    tools,
    readonly_roots,
    output_root,
    bwrap,
    max_files,
    max_bytes,
    max_output_files,
    max_output_bytes,
    max_console_bytes,
):
    budgets = (max_files, max_bytes, max_output_files, max_output_bytes, max_console_bytes)
    if (
        type(tools) is not tuple
        or not tools
        or type(readonly_roots) is not tuple
        or not readonly_roots
        or any(type(value) is not int or value <= 0 for value in budgets)
    ):
        raise ValueError("build snapshot requires explicit tool/lookup roots and positive byte/file budgets")
    root, output_root, bwrap = F.canonical(root, exists=False), F.canonical(output_root), F.canonical(bwrap)
    tools = tuple(F.canonical(Path(tool).resolve(strict=True)) for tool in tools)
    if root.exists() or not output_root.is_dir() or any(not os.access(tool, os.X_OK) for tool in (*tools, bwrap)):
        raise ValueError("build view requires a fresh snapshot and executable original tools")
    root.mkdir(mode=0o700)
    image = root / "image"
    image.mkdir()
    budget = [max_files - 1, max_bytes]
    for value in readonly_roots:
        source, target = (
            (Path(value[0]), Path(value[1])) if type(value) is tuple and len(value) == 2 else (Path(value), Path(value))
        )
        source = F.canonical(source)
        if (
            not target.is_absolute()
            or ".." in target.parts
            or target == Path("/")
            or target.parts[1] in {"proc", "dev", "sys", "run", "tmp"}
            or root.is_relative_to(source)
            or output_root.is_relative_to(source)
        ):
            raise ValueError("build immutable roots overlap mutable outputs or kernel state")
        destination = _guest_member(image, str(target))
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            F.copy_member(source, destination, budget=budget)
    for tool in tools:
        destination = _guest_member(image, str(tool))
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            F.copy_member(tool, destination, budget=budget)
    for name in ("run/merlin/inputs", "run/work", "tmp"):
        path = image / name
        if path.is_symlink():
            raise ValueError("build reserved mount point is linked")
        path.mkdir(parents=True, exist_ok=True)
    for tool in ("/bin/bash", "/usr/bin/env"):
        path = _guest_member(image, tool)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or not os.access(path, os.X_OK):
            raise ValueError("build view lacks its shared fixed bootstrap executable")
    F.seal_directories(image, max_files=max_files)
    issuer = object()
    owner = BuildRuntimeView(
        root=root, tools=tools, output_root=output_root, bwrap=bwrap, budgets=budgets, issuer=issuer
    )
    _ISSUED[issuer] = owner
    owner.verify()
    owner._pin_token = storage.pin(root, reason="shared original build filesystem view")
    return owner


def active_build_runtime():
    return _ACTIVE.get()


@contextlib.contextmanager
def executing_build_runtime(owner=None, *, storage_roots=()):
    if owner is not None:
        if type(owner) is not BuildRuntimeView:
            raise ValueError("build runtime requires an exact live view")
        owner.verify()
    token = _ACTIVE.set(owner)
    storage_token = _STORAGE.set(tuple(storage_roots))
    try:
        yield owner
    finally:
        _ACTIVE.reset(token)
        _STORAGE.reset(storage_token)


def run_in_build_view(
    argv,
    *,
    inputs,
    outputs,
    directory,
    stage,
    cwd=None,
    env=None,
    timeout=None,
    capture_output=True,
    text=False,
    encoding=None,
    errors=None,
    **kwargs,
):
    owner = active_build_runtime()
    if owner is None:
        return None
    if type(owner) is not BuildRuntimeView:
        raise ValueError("original build lost its live lookup owner")
    selection = (
        None
        if kwargs
        or timeout is None
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout < float("inf")
        else owner.selection(
            argv,
            inputs=inputs,
            outputs=outputs,
            cwd=Path.cwd() if cwd is None else cwd,
            environment=dict(os.environ) if env is None else env,
            directory=directory,
            stage=stage,
            capture_output=capture_output,
            text=text,
            encoding=encoding,
            errors=errors,
            storage_roots=_STORAGE.get(),
        )
    )
    if selection is None:
        scope = _BUILD.get()
        if scope is not None and scope["owner"] is owner:
            scope["eligible"] = False
        return None
    try:
        from merlin.targetgen.contract import build_products
    except ImportError as exc:
        if getattr(exc, "name", None) not in ("merlin.targetgen.contract.build_products", "merlin.targetgen.contract"):
            raise
        build_products = None
    cached = (
        None
        if build_products is None
        else build_products.lookup_build_products(owner, selection, inputs, outputs, directory, stage)
    )
    if cached is not None:
        scope = _BUILD.get()
        if scope is not None and scope["owner"] is owner:
            # A reuse receipt is not a new subprocess observation. A caller
            # asking for whole-build fresh observations cannot omit that call.
            scope["eligible"] = False
        return cached
    try:
        result, observation = owner.run(selection, timeout=timeout)
    except BaseException:
        scope = _BUILD.get()
        if scope is not None and scope["owner"] is owner:
            scope["eligible"] = False
        raise
    if result.returncode == 0 and build_products is not None:
        build_products.stage_build_products(owner, selection, inputs, outputs, observation)
    return result
