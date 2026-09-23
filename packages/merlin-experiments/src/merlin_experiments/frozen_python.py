"""Guarded host Python transport for an existing caller-verified source snapshot.

This is trusted launch instrumentation, not a second grading receipt or a sandbox. Source
authority remains the snapshot seal admitted by the explicitly supplied verifier. The current
source_snapshot verifier requires v4; historical verifiers retain their own policies. Existing supervisors own
process lifetime. Descendants must explicitly use ``inherited_python_command``; merely inheriting
PYTHONPATH does not propagate a process-local import guard. Site initialization is never enabled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import stat
import subprocess
import sys
import types
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

CONTEXT = "MERLIN_FROZEN_PYTHON_CONTEXT"
_OWNER = "merlin_experiments.frozen_python"
_LEGACY_OWNER = "perf_frozen_python"
_ACTIVE_CONTEXT: dict | None = None


def active_source_identity() -> dict[str, str] | None:
    """Existing seal verified at this process's guarded startup, never ambient env.

    This is an attribution reference, not a new seal or a sandbox against trusted
    host code modifying interpreter state. Ordinary imports return no identity.
    """
    return dict(_ACTIVE_CONTEXT["seal"]) if _ACTIVE_CONTEXT is not None else None


def _read(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError("frozen Python instrumentation must be an ordinary file")
        return stream.read()


def _pin(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(_read(path)).hexdigest()}


def _checked(pin: dict[str, str]) -> Path:
    path = Path(pin["path"])
    if path.resolve() != path or _pin(path) != pin:
        raise RuntimeError("frozen Python transport instrumentation or seal changed")
    return path


def _load(path: Path, name: str, digest: str | None = None):
    # Compile checked bytes directly: stale bytecode must not supersede a source pin.
    module = types.ModuleType(name)
    module.__file__ = str(path)
    payload = _read(path)
    if digest is not None and hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError("frozen Python instrumentation changed before execution")
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


@lru_cache(maxsize=16)
def _interpreter_dependencies(executable: str) -> tuple[str, ...]:
    program = (
        "import json,sys,sysconfig; from pathlib import Path; "
        "prefix=Path(sys.argv[1]).absolute().parent.parent; "
        "variables={'base':str(prefix),'platbase':str(prefix)} if (prefix/'pyvenv.cfg').is_file() else {}; "
        "print(json.dumps([sysconfig.get_path(k,vars=variables) for k in ('purelib','platlib')]))"
    )
    checked = subprocess.run(
        [executable, "-I", "-S", "-c", program, executable], capture_output=True, text=True, timeout=15, check=True
    )
    return tuple(json.loads(checked.stdout))


def _dependency_roots(executable: str, excluded: Sequence[str]) -> list[str]:
    # Explicit ordinary directories only: never site.main(), .pth execution or editable finders.
    # The chosen interpreter may be a distinct CHIA venv; preserve its unresolved executable path.
    roots = [str(Path(value).resolve()) for value in _interpreter_dependencies(executable) if Path(value).is_dir()]
    roots += [
        str(Path(value).resolve())
        for value in sys.path
        if value
        and Path(value).is_dir()
        and not any(part in {"site-packages", "dist-packages"} for part in Path(value).parts)
    ]
    owners = ("merlin", "merlin_experiments", "merlin_analysis", "merlin_dse", "merlin_mining")
    return list(
        dict.fromkeys(
            value
            for value in roots
            if not any(Path(value).is_relative_to(Path(item)) for item in excluded)
            and (
                Path(value).name in {"site-packages", "dist-packages"}
                or not any((Path(value) / name).is_dir() for name in owners)
            )
        )
    )


def _sealed_transport(receipt: dict, layout: dict, owner: str) -> bool:
    if owner == _OWNER:
        suffix, roots = "merlin_experiments/frozen_python.py", layout["python_roots"]
    elif owner == _LEGACY_OWNER:
        suffix, roots = "perf_frozen_python.py", layout["legacy_roots"]
    else:
        raise RuntimeError("unsupported frozen Python transport owner")
    return any(f"{value}/{suffix}" in receipt["files"] for value in roots)


def _context(snapshot: Path, verifier_source: Path | None) -> dict:
    inherited = os.environ.get(CONTEXT)
    if inherited:
        instruments = json.loads(inherited)
        verifier = _checked(instruments["verifier"])
    else:
        if verifier_source is None:
            raise ValueError("a root frozen Python launch requires an explicit trusted verifier_source")
        from merlin.common.paths import module_source_path

        verifier = Path(verifier_source)
        instruments = {
            "bootstrap": _pin(Path(__file__)),
            "verifier": _pin(verifier),
            "resolver": _pin(module_source_path("merlin.common.frozen_imports")),
        }
    snap = _load(verifier, "_perf_snapshot_transport", instruments["verifier"]["sha256"])
    snapshot = snapshot.resolve()
    receipt = snap.verify(snapshot)
    seal, _ = snap.load_seal(snapshot, "snapshot")
    layout = snap.import_layout(snapshot, receipt)
    # Reject historical external selection before building a launch command.
    if hasattr(snap, "provider_environment"):
        snap.provider_environment(snapshot, receipt)
    elif os.environ.get("MERLIN_TARGET_PATH"):
        raise RuntimeError(
            "historical snapshot has no external provider ownership; create a newly frozen selected-provider run"
        )
    owner = _OWNER if _sealed_transport(receipt, layout, _OWNER) else _LEGACY_OWNER
    has_transport = _sealed_transport(receipt, layout, owner)
    return {
        "snapshot": str(snapshot),
        "seal": _pin(seal),
        "transport_module": "sealed-source" if has_transport else "current-pinned-instrumentation",
        "transport_owner": owner,
        "excluded_dependency_roots": [str(Path(receipt["source_root"]) / value) for value in receipt["source_roots"]]
        + [str(Path(instruments["bootstrap"]["path"]).parent)]
        + ([receipt["selected_provider"]["source"]] if receipt.get("selected_provider") else []),
        **{key: instruments[key] for key in ("bootstrap", "verifier", "resolver")},
    }


def _command(context: dict, native_argv: Sequence[str]) -> list[str]:
    argv = list(native_argv)
    if len(argv) < 2:
        raise ValueError("frozen Python requires an explicit script, module or inline program")
    bootstrap = _checked(context["bootstrap"])
    document = {
        "context": context,
        "native_argv": argv,
        "dependency_roots": _dependency_roots(argv[0], context["excluded_dependency_roots"]),
    }
    flags = []
    for item in argv[1:]:
        if item not in {"-B", "-u", "-s", "-E", "-I", "-S", "-P"}:
            break
        if item == "-u":
            flags.append(item)
    return [argv[0], "-I", "-S", "-B", *flags, str(bootstrap), "--execute", json.dumps(document, separators=(",", ":"))]


def python_command(snapshot: Path, native_argv: Sequence[str], *, verifier_source: Path | None = None) -> list[str]:
    """Build host transport; retain native argv separately in the existing receipt.

    Root launches must explicitly name trusted verifier source. Descendants always
    retain their already-pinned verifier, including historical callers using the
    former two-argument API; no neighboring live native source is discovered.
    """
    return _command(_context(snapshot, verifier_source), native_argv)


def inherited_python_command(native_argv: Sequence[str]) -> list[str]:
    """Guard an audited host child, or preserve ordinary execution outside a frozen launch."""
    if _ACTIVE_CONTEXT is not None:
        return _command(_ACTIVE_CONTEXT, native_argv)
    value = os.environ.get(CONTEXT)
    return _command(json.loads(value), native_argv) if value else list(native_argv)


def _execute(document: dict) -> None:
    context = document["context"]
    _checked(context["bootstrap"])
    seal = _checked(context["seal"])
    verifier = _load(_checked(context["verifier"]), "_perf_snapshot_transport", context["verifier"]["sha256"])
    root = Path(context["snapshot"])
    receipt = verifier.verify(root)
    actual_seal, _ = verifier.load_seal(root, "snapshot")
    if actual_seal != seal:
        raise RuntimeError("frozen source seal identity changed")
    layout = verifier.import_layout(root, receipt)
    if hasattr(verifier, "provider_environment"):
        os.environ.update(verifier.provider_environment(root, receipt))
    elif os.environ.get("MERLIN_TARGET_PATH"):
        raise RuntimeError(
            "historical snapshot has no external provider ownership; create a newly frozen selected-provider run"
        )
    if not layout["python_roots"]:
        raise RuntimeError("historical snapshot has no sealed Python owner roots; explicit requalification required")
    resolver = _load(_checked(context["resolver"]), "_merlin_frozen_imports_transport", context["resolver"]["sha256"])
    provider = (
        receipt.get("selected_provider")
        if receipt["schema"] in {"merlin.performance-source-snapshot.v3", "merlin.performance-source-snapshot.v4"}
        else None
    )
    provider_roots = {"auxiliary_roots": [root / provider["root"]]} if provider else {}
    resolver.activate(
        snapshot_root=root,
        import_roots=[root / value for value in [*layout["python_roots"], *layout["legacy_roots"]]],
        sources=receipt["files"],
        legacy_names=layout["legacy_names"],
        **provider_roots,
    )
    owner = context.get("transport_owner", _LEGACY_OWNER)
    has_transport = _sealed_transport(receipt, layout, owner)
    expected_transport = "sealed-source" if has_transport else "current-pinned-instrumentation"
    if context["transport_module"] != expected_transport:
        raise RuntimeError("frozen transport module attribution changed")
    if not has_transport:
        if owner != _LEGACY_OWNER:
            raise RuntimeError("canonical frozen transport is absent from its source receipt")
        # A narrowly named, separately pinned instrumentation dependency, not a newly
        # attributed historical grader source. All other native names remain guarded.
        sys.modules["perf_frozen_python"] = _load(
            _checked(context["bootstrap"]), "perf_frozen_python", context["bootstrap"]["sha256"]
        )
    # Dependency paths are admitted only after all Merlin namespaces are guarded.
    sys.path.extend(value for value in document["dependency_roots"] if value not in sys.path)
    os.environ[CONTEXT] = json.dumps(context, separators=(",", ":"))
    os.environ["MERLIN_REPO_ROOT"] = str(root)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # The verified canonical module has a distinct identity from this bootstrap
    # executed as __main__. Publish only into the actual selected transport owner;
    # historical layouts must never acquire an unsealed canonical owner.
    transport = importlib.import_module(owner)
    transport._ACTIVE_CONTEXT = json.loads(json.dumps(context))
    argv = list(document["native_argv"])[1:]
    while argv and argv[0] in {"-B", "-u", "-s", "-E", "-I", "-S", "-P"}:
        argv.pop(0)
    if not argv:
        raise ValueError("missing frozen Python entrypoint")
    if argv[0] == "-c":
        if len(argv) < 2:
            raise ValueError("missing inline Python program")
        sys.argv = ["-c", *argv[2:]]
        main = types.ModuleType("__main__")
        sys.modules["__main__"] = main
        exec(compile(argv[1], "<string>", "exec"), main.__dict__)
    elif argv[0] == "-m":
        if len(argv) < 2:
            raise ValueError("missing frozen module")
        spec = importlib.util.find_spec(argv[1])
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if (
            origin is None
            or not origin.is_relative_to(root)
            or origin.relative_to(root).as_posix() not in receipt["files"]
        ):
            raise RuntimeError("Python entrypoint module is not sealed source")
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        if argv[0].startswith("-"):
            raise ValueError("unsupported frozen Python interpreter option")
        script = Path(argv[0]).resolve()
        relative = script.relative_to(root).as_posix()
        if relative not in receipt["files"]:
            raise RuntimeError("Python entrypoint script is not sealed source")
        payload = script.read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["files"][relative]:
            raise RuntimeError("Python entrypoint source changed")
        sys.argv = argv
        main = types.ModuleType("__main__")
        main.__file__ = str(script)
        main.__package__ = None
        sys.modules["__main__"] = main
        exec(compile(payload, str(script), "exec"), main.__dict__)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--execute":
        raise SystemExit("internal frozen Python transport: use python_command()")
    _execute(json.loads(sys.argv[2]))
