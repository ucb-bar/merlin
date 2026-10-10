"""Retain exact inputs and products at an actually invoked process boundary.

These observations confer no correctness or stage applicability. Callers own
which inputs/dependencies are complete and the verifier owns semantic lift.
Records are written outside invoked packages and never accepted from stdout.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

SCHEMA = "merlin.invocation_record.v1"


def _pin(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except OSError as error:
        # An observation must not replace the process's own missing-tool/input
        # failure. A verifier cannot admit an unavailable pin.
        return {"path": str(path), "sha256": None, "unavailable": type(error).__name__}


def _pins(paths) -> list[dict]:
    return [_pin(path) for path in sorted({Path(path).resolve() for path in paths})]


def environment_identity(environment) -> dict:
    """Bind actual process environment bytes without retaining their values.

    POSIX environment keys and values use the filesystem encoding, as Python's
    process launcher does. The complete sorted mapping is hashed with lengths
    and a domain separator. This is an observation, not a replay environment,
    dependency closure, secret-management mechanism or sandbox qualification.
    """
    members = {}
    for key, value in environment.items():
        encoded_key, encoded_value = os.fsencode(key), os.fsencode(value)
        if (
            not encoded_key
            or b"=" in encoded_key
            or b"\0" in encoded_key
            or b"\0" in encoded_value
            or encoded_key in members
        ):
            raise ValueError("environment has an invalid or ambiguous process mapping")
        members[encoded_key] = encoded_value
    digest = hashlib.sha256(b"merlin.process_environment.v1\0")
    for key, value in sorted(members.items()):
        for part in (key, value):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return {
        "schema": "merlin.process_environment.v1",
        "sha256": digest.hexdigest(),
        "keys": [os.fsdecode(key) for key in sorted(members)],
        "scope": "complete effective mapping; values withheld; transitive dependencies unproved",
    }


class Invocation:
    def __init__(
        self,
        directory: Path,
        *,
        stage: str,
        argv,
        cwd=None,
        env=None,
        inputs=(),
        outputs=(),
        dependencies=(),
    ):
        if not stage or not argv:
            raise ValueError("invocation observation requires a stage and command")
        self.directory = Path(directory).resolve() / "invocations" / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.outputs = tuple(Path(path).resolve() for path in outputs)
        self.inputs = tuple(Path(path).resolve() for path in inputs)
        self.dependencies = tuple(Path(path).resolve() for path in dependencies)
        selected_env = dict(os.environ if env is None else env)
        command = [str(token) for token in argv]
        work = Path.cwd() if cwd is None else Path(cwd).resolve()
        executable = Path(command[0])
        if not executable.is_absolute():
            if os.path.dirname(command[0]):
                executable = work / executable
            else:
                # Relative search directories belong to the child's cwd.
                search = [str(work / entry) for entry in os.get_exec_path(selected_env)]
                found = shutil.which(command[0], path=os.pathsep.join(search))
                executable = Path(found) if found else work / executable
        self.document = {
            "schema": SCHEMA,
            "stage": stage,
            "argv": command,
            "cwd": str(work),
            "kind": "subprocess",
            "executable": _pin(executable),
            "inputs": _pins(self.inputs),
            "dependencies": _pins(self.dependencies),
            "status": "running",
            "started_ns": time.time_ns(),
            "environment": environment_identity(selected_env),
        }
        self.path = self.directory / "invocation.json"
        self._write()

    def _write(self):
        self.path.write_text(json.dumps(self.document, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def complete(self, result: subprocess.CompletedProcess):
        for name in ("stdout", "stderr"):
            value = getattr(result, name, None)
            data = value if isinstance(value, bytes) else (value or "").encode("utf-8")
            path = self.directory / (name + ".bin")
            path.write_bytes(data)
            self.document[name] = _pin(path)
        self.document.update(
            status="completed" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            finished_ns=time.time_ns(),
            outputs=_pins(path for path in self.outputs if path.is_file()),
            inputs_unchanged=_pins(self.inputs) == self.document["inputs"],
            dependencies_unchanged=_pins(self.dependencies) == self.document["dependencies"],
            executable_unchanged=_pin(Path(self.document["executable"]["path"])) == self.document["executable"],
        )
        self._write()

    def failed(self, error: BaseException):
        if self.document["status"] == "running":
            self.document.update(status="interrupted", error=type(error).__name__, finished_ns=time.time_ns())
            self._write()


class CallInvocation(Invocation):
    """An observed Python dispatch, distinct from a subprocess command claim."""

    def __init__(
        self, directory: Path, *, stage: str, function, arguments: dict, inputs=(), outputs=(), dependencies=()
    ):
        try:
            origin = inspect.getsourcefile(function)
        except TypeError:
            origin = None
        # An uninspectable callable remains an unavailable observation; it does
        # not turn optional lineage into a new functional refusal.
        source = origin or "/unavailable-callable-source"
        super().__init__(
            directory,
            stage=stage,
            argv=(source,),
            inputs=inputs,
            outputs=outputs,
            dependencies=(*dependencies, Path(source)),
        )
        name = (
            f"{getattr(function, '__module__', type(function).__module__)}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}"
        )
        self.document.update(
            kind="python_call", callable=name, callable_source_available=origin is not None, arguments=arguments
        )
        self.document.pop("argv")
        self.document.pop("environment")
        self._write()

    def returned(self, *, stdout=b"", stderr=b""):
        self.complete(subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr))
        self.document.pop("returncode")
        self.document["outcome"] = "returned"
        self._write()


@contextlib.contextmanager
def observe(directory: Path, **kwargs):
    """Observe only a caller's actual invocation, including failed boundaries."""
    invocation = Invocation(directory, **kwargs)
    try:
        yield invocation
    except BaseException as error:
        invocation.failed(error)
        raise
    finally:
        if invocation.document["status"] == "running":
            invocation.failed(RuntimeError("invocation did not report completion"))


def run(argv, *, directory: Path, stage: str, inputs=(), outputs=(), dependencies=(), **kwargs):
    # Observe and execute the same immutable snapshot, even if a caller's
    # mutable mapping or ambient environment changes at the observation point.
    kwargs["env"] = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
    with observe(
        directory,
        stage=stage,
        argv=argv,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        cwd=kwargs.get("cwd"),
        env=kwargs.get("env"),
    ) as record:
        result = subprocess.run(argv, **kwargs)
        record.complete(result)
        return result


@contextlib.contextmanager
def observe_call(directory: Path, **kwargs):
    record = CallInvocation(directory, **kwargs)
    try:
        yield record
    except BaseException as error:
        record.failed(error)
        raise
    finally:
        if record.document["status"] == "running":
            record.failed(RuntimeError("call did not report its return"))


def verify(path: Path, *, pin_replay=None) -> dict:
    """Reopen exact observation bytes; completeness/applicability remain caller-owned."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    returned = (
        document.get("returncode") == 0
        if document.get("kind") == "subprocess"
        else document.get("outcome") == "returned"
    )
    unchanged = ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
    if (
        document.get("schema") != SCHEMA
        or document.get("status") != "completed"
        or not returned
        or any(document.get(name) is not True for name in unchanged)
    ):
        raise ValueError("invocation did not complete with unchanged observed inputs")
    roles = [(document["executable"], True), (document["stdout"], False), (document["stderr"], False)]
    roles.extend((pin, False) for pin in (*document["inputs"], *document["outputs"]))
    roles.extend((pin, True) for pin in document["dependencies"])
    for pin, selected_role in roles:
        actual = None
        if selected_role and pin_replay is not None:
            from .selected_pin_replay import replayed_pin

            actual = replayed_pin(pin_replay, Path(pin["path"]))
        if actual is None:
            actual = _pin(Path(pin["path"]))
        if not pin.get("sha256") or actual != pin:
            raise ValueError("invocation input, dependency or product changed")
    return document


def require_environment(path: Path, *, environment, pin_replay=None) -> dict:
    """Reopen an actual successful process and compare its protected selection.

    Older observations without this binding cannot establish environment
    identity. A matching saved record alone confers no live execution authority.
    """
    document = verify(path, pin_replay=pin_replay)
    if document.get("kind") != "subprocess" or document.get("environment") != environment_identity(environment):
        raise ValueError("invocation has no matching effective process environment binding")
    return document
