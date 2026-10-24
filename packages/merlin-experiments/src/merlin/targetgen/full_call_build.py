"""Build a full-call image from submitted code in an isolated process.

A Core ATen execution provider binds device calls by importing modules from the SUBMITTED package.
That code must never share a process with the grader: it could patch the readback grader or read the
private runtime bundle. The grader therefore decides the route from trusted inputs
(:func:`core_aten_device.source_route`) and runs only the provider/candidate build here, under a
deny-by-default bubblewrap sandbox that sees the selected interpreter, Merlin, the support provider,
the toolchains, the submission (read-only) and a fresh work directory holding copies of the public
program inputs. The grader alone executes the returned image and grades its readback.

    python -m merlin.targetgen.full_call_build <request.json>     (inside the sandbox)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Program inputs a build reads; nothing else of the private runtime bundle is copied out.
PROGRAM_INPUTS = (
    "model.mlir",
    "inputs.npz",
    "input_order.json",
    "semantic_io.json",
    "weights.safetensors.manifest.json",
    "weights.safetensors",
)
ADMISSION = "spec-int8-admission.json"


def _clean(text, limit: int) -> str:
    lines = str(text).strip().splitlines()
    words = lines[0].split() if lines else []
    message = " ".join("<path>" if "/" in word or "\\" in word else word for word in words)
    return "".join(ch for ch in message if ch.isprintable())[:limit]


def sanitized(text, *, limit: int = 120) -> str:
    """What sandboxed text may reach feedback: one path-free line of at most ``limit`` characters, keeping an
    identifier-shaped ``Category:`` prefix (an exception name).

    Anything the sandbox reports is candidate-influenced (its code shares the process with the support
    provider and can read it), so nothing longer, multi-line or path-shaped passes.
    """
    head, separator, rest = str(text).strip().partition(":")
    if separator and head.isidentifier() and len(head) <= 40:
        return f"{head}: {_clean(rest, limit)}".rstrip(": ")
    return _clean(text, limit)


def sanitized_record(value, *, depth: int = 0):
    """A JSON value from the sandbox with every string :func:`sanitized` and no deep or huge structure."""
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, str):
        return sanitized(value, limit=80)
    if depth < 3 and isinstance(value, list):
        return [sanitized_record(item, depth=depth + 1) for item in value[:64]]
    if depth < 3 and isinstance(value, dict):
        return {sanitized(k, limit=40): sanitized_record(v, depth=depth + 1) for k, v in list(value.items())[:32]}
    return None


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    """Child: route and build one image, then report the build and its routing record; or, with
    ``--run-elf``, run one image on a backend engine and leave its console beside the image."""
    argv = list(argv or sys.argv[1:])
    if argv[0] == "--run-elf":
        from merlin.runtime.backends import base

        target, engine, elf, timeout = argv[1:5]
        console = base.get_backend(target).run_elf(Path(elf), simulator=engine, timeout=int(timeout))
        (Path(elf).parent / "console.txt").write_text(console)
        return 0
    request = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(request["result"])
    try:
        from merlin.llvmlower.device_build import load_region_offload
        from merlin.llvmlower.device_offload import load_sidecar
        from merlin.runtime.backends import spike_model
        from merlin.targetgen.core_aten_device import bind_source_route, load_execution_provider, selected_facts

        target, bundle, package = request["target"], Path(request["bundle"]), Path(request["package"])
        build_dir = Path(request["build"])
        provider = load_execution_provider(target, Path(request["provider"]))
        routing, reason = None, request["reason"]
        with selected_facts(target, Path(request["facts"])) as facts:
            prepare = getattr(provider, "prepare_bundle", None)
            if request["prepare"] and prepare is not None:
                prepare(bundle, target=target, facts=facts)
            if request["validate"]:
                validate = getattr(provider, "validate_package", None)
                if not callable(validate):
                    raise ValueError("optional-device fallback requires independent provider package validation")
                if validate(bundle, target=target, package=package, facts=facts) is not None:
                    raise ValueError("provider package validation must return None or raise")
            if request["route"] is not None:
                routing, reason = bind_source_route(request["route"], reason, bundle, target, package, provider, facts)
            if routing is None and request["require_device"]:
                raise ValueError("submitted backend did not route this capsule: " + reason)
            build = spike_model.build(
                bundle,
                build_dir,
                inputs_npz=bundle / "inputs.npz",
                arena_mb=request["arena_mb"],
                dump_all_outputs=True,
                backend=request["backend"],
                device=routing,
                **request["layout"],
            )
        # A bound route reports its bindings (possibly none, which the grader refuses); no route reports None.
        routed = (load_region_offload(build_dir) or load_sidecar(build_dir)).get("routed") or [] if routing else None
        keys = ("elf", "mem_bytes", "build_hash", "vlen")
        _write(out, {"build": {k: build.get(k) for k in keys}, "routed": routed, "reason": reason, "error": None})
    except Exception as exc:  # noqa: BLE001 -- a refusal of the submitted build is a verdict input
        _write(out, {"error": f"{type(exc).__name__}: {exc}"[:2000]})
    return 0


def _covering(paths, *, excluded) -> list[Path]:
    """The given existing paths, minus any containing an ``excluded`` path or inside another one."""
    roots: list[Path] = []
    candidates = {Path(p).resolve() for p in paths if p and Path(p).exists()}
    for path in sorted(candidates, key=lambda p: len(p.parts)):
        if any(path == e or e.is_relative_to(path) for e in excluded):
            continue
        if not any(path.is_relative_to(root) for root in roots):
            roots.append(path)
    return roots


#: Operator-declared read-only roots a simulator jail may see (os.pathsep list), e.g. the RTL simulator
#: binary and its database, and the frozen toolchain. Nothing else of the host filesystem is visible.
ORACLE_ROOTS_ENV = "MERLIN_ORACLE_JAIL_ROOTS"
#: Library search path inside a simulator jail (the frozen toolchain's libraries), so a binary whose
#: RPATH names a host toolchain resolves the selected copy instead.
ORACLE_LIBRARY_ENV = "MERLIN_ORACLE_LD_LIBRARY_PATH"
#: Tool directories on ``PATH`` inside a simulator jail (e.g. the frozen ``dtc`` Spike runs).
ORACLE_PATH_ENV = "MERLIN_ORACLE_PATH"


def _merlin_roots(target: str) -> list:
    """The interpreter, Merlin and the selected support provider: what a jailed Merlin child imports."""
    from merlin.targetgen.plugins import resolve_support

    return [
        sys.prefix,
        sys.base_prefix,
        *(entry for entry in sys.path if entry and Path(entry).is_dir()),
        resolve_support(target).base,
        os.environ.get("MERLIN_REPO_ROOT"),
    ]


def _libraries(executables, library_path: str) -> list[Path]:
    """Directories outside the system holding the shared libraries ``executables`` load (``ldd``).

    A library that ``library_path`` also provides is left to it: an executable's DT_RPATH (searched before
    ``LD_LIBRARY_PATH``) may name a live install, which the jail then does not mount, so the loader falls
    through to the pinned copy instead of binding the install.
    """
    directories = []
    provided = [Path(entry) for entry in library_path.split(os.pathsep) if entry]
    environment = {**os.environ, "LD_LIBRARY_PATH": library_path}
    for executable in executables:
        listing = subprocess.run(["ldd", str(executable)], capture_output=True, text=True, env=environment).stdout
        for line in listing.splitlines():
            parts = line.split()
            if "=>" in parts and parts.index("=>") + 1 < len(parts) and parts[parts.index("=>") + 1].startswith("/"):
                library = Path(parts[parts.index("=>") + 1]).resolve().parent
                if any((entry / parts[0]).is_file() for entry in provided):
                    continue
                if not library.is_relative_to("/usr") and not library.is_relative_to("/lib"):
                    directories.append(library)
    return directories


def _jail(work: Path, roots, environment: dict, private, *, network: bool = False, procfs: bool = True) -> list[str]:
    """Deny-by-default bubblewrap prefix: system read-only, /scratch* hidden, ``roots`` read-only, ``work``
    writable. Refused if any ``private`` path would be reachable. A variable naming a path passes only
    when that path is visible; ``MERLIN_*`` selections and licence servers pass; everything else is cleared."""
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    excluded = [Path(p).resolve() for p in (work, *private)]
    argv = BW.base_argv(work, {}, include_claude_home=False, inherit_environment=False)
    # No terminal to inject into; no network unless the caller runs a licensed simulator and no candidate code.
    argv += ["--new-session", *([] if network else ["--unshare-net"])]
    if not procfs:  # /proc/self/fd and /dev/fd would hand a guest the simulator's own trace descriptor
        argv += ["--tmpfs", "/proc"]
    bound = _covering(roots, excluded=excluded)
    for root in bound:
        argv += ["--ro-bind", str(root), str(root)]
    visible = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", "/etc", work, *bound)]
    for key, value in os.environ.items():
        if key in environment or key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
            continue
        if not value.startswith("/"):  # a selection or licence server, never a path
            if key.startswith("MERLIN_") or key.endswith("LICENSE_FILE"):
                environment[key] = value
        elif any(Path(value).is_relative_to(root) for root in visible):  # names something visible
            environment[key] = value
        elif any(Path(value).resolve().is_relative_to(root) for root in bound):  # through a host link
            environment[key] = str(Path(value).resolve())
    for key in ("PATH", "LD_LIBRARY_PATH"):
        entries = environment.get(key, os.environ.get(key, "/usr/bin:/bin" if key == "PATH" else ""))
        environment[key] = os.pathsep.join(p for p in entries.split(os.pathsep) if p and BW.is_exposed(argv, Path(p)))
    for key, value in sorted(environment.items()):
        argv += ["--setenv", key, value]
    surfaces = [
        AnswerSurface("full-call private input", Path(p).resolve(), "dir" if Path(p).is_dir() else "file", "private")
        for p in private
    ]
    if BW.coverage_gap(argv, surfaces):
        raise RuntimeError("full-call sandbox would expose private grading inputs")
    return argv


def build_in_sandbox(
    request: dict, *, bundle: Path, work: Path, package: Path, provider: Path, facts: Path, denied, timeout
) -> dict:
    """Parent: copy the public program inputs into ``work``, run :func:`main` sandboxed, return its report.

    ``denied`` are paths the sandbox must not expose (the private runtime bundle, the capsule corpus with
    its answers); the command is refused if any of them is reachable. ``bundle`` is never mounted.
    """
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends.spike import gcc_path
    from merlin.targetgen.plugins import resolve_support

    if work.exists():
        shutil.rmtree(work)
    inputs, contract = work / "in", work / "contract"
    for directory in (inputs, contract, work / "tmp", work / "out"):
        directory.mkdir(parents=True)
    for name in PROGRAM_INPUTS:
        if (bundle / name).is_file():
            shutil.copyfile(bundle / name, inputs / name)
    if (bundle / ADMISSION).is_file():  # only the selected numeric policy, never its references
        selection = json.loads((bundle / ADMISSION).read_text(encoding="utf-8"))["selection"]
        _write(inputs / ADMISSION, {"selection": {"spec": selection["spec"]}})
    shutil.copyfile(facts, work / "facts.json")
    environment = {"MERLIN_RTL_FACTS": str(work / "facts.json")}
    if os.environ.get("MERLIN_TARGET_CONTRACT"):
        shutil.copyfile(os.environ["MERLIN_TARGET_CONTRACT"], contract / "target_contract.yaml")
        environment["MERLIN_TARGET_CONTRACT"] = str(contract / "target_contract.yaml")
    request = {
        **request,
        "bundle": str(inputs),
        "build": str(work / "out" / "build"),
        "package": str(package),
        "provider": str(provider),
        "facts": str(work / "facts.json"),
        "result": str(work / "out" / "result.json"),
    }
    _write(work / "request.json", request)
    compiler = os.environ.get("MERLIN_COMPILER_PYTHON")
    private = [bundle, *denied]
    required = [Path(package).resolve(), Path(resolve_support(request["target"]).base).resolve()]
    if any(Path(p).resolve() == r or Path(p).resolve().is_relative_to(r) for r in required for p in [work, *private]):
        raise RuntimeError("full-call sandbox would expose private grading inputs")
    roots = [
        *_merlin_roots(request["target"]),
        package,
        toolchain.llvm_install(),
        gcc_path().parents[2],  # the environment the cross toolchain and its shared libraries live in
        Path(compiler).parents[1] if compiler else None,
    ]
    environment.update(
        MERLIN_RISCV_GCC=str(gcc_path()),  # the bound cross compiler, not a toolchain root left outside
        HOME=str(work),
        TMPDIR=str(work / "tmp"),
        MERLIN_OUT_ROOT=str(work / "merlin-out"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.environ.get("PYTHONPATH", ""),
    )
    argv = _jail(work, roots, environment, private)
    command = [*argv, sys.executable, "-P", "-m", "merlin.targetgen.full_call_build", str(work / "request.json")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    result = work / "out" / "result.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("isolated full-call build produced no report: " + sanitized(done.stderr or done.stdout))
    return json.loads(result.read_text(encoding="utf-8"))


def oracle_jail(
    elf: Path, work: Path, *, private, executables=(), roots=(), path=(), network: bool = False, procfs: bool = False
) -> tuple[list[str], Path]:
    """``(bwrap prefix, jailed image)`` for running a simulator on a copy of ``elf`` in a fresh ``work``.

    A guest can reach the host filesystem through the simulator's HTIF/fesvr syscall proxy, so the
    simulator sees only ``work`` (the image copy; writable), the operator's :data:`ORACLE_ROOTS_ENV`,
    ``roots``, the libraries ``executables`` load, and ``path`` and :data:`ORACLE_PATH_ENV` on
    ``PATH``. The trusted caller reads the
    console back. A trace the simulator streams to its own file descriptor never enters the jail, and the
    guest cannot reopen that descriptor: fesvr maps guest fds 1 and 2 to host stdout, and without ``procfs``
    neither ``/proc/self/fd`` nor ``/dev/fd`` exists.
    """
    if work.exists():
        shutil.rmtree(work)
    (work / "tmp").mkdir(parents=True)
    jailed = work / Path(elf).name
    shutil.copyfile(elf, jailed)
    jailed.chmod(0o500)
    library_path = os.environ.get(ORACLE_LIBRARY_ENV, "")
    declared = [p for p in os.environ.get(ORACLE_ROOTS_ENV, "").split(os.pathsep) if p]
    executables = [*executables, *(p for p in declared if Path(p).is_file() and os.access(p, os.X_OK))]
    libraries = _libraries(executables, library_path)
    path = [*path, *(Path(p) for p in os.environ.get(ORACLE_PATH_ENV, "").split(os.pathsep) if p)]
    environment = {
        "HOME": str(work),
        "TMPDIR": str(work / "tmp"),
        "PATH": os.pathsep.join([*map(str, path), "/usr/bin", "/bin"]),
        "LD_LIBRARY_PATH": os.pathsep.join([library_path, *map(str, libraries)]).strip(os.pathsep),
    }
    argv = _jail(
        work,
        [*declared, *roots, *libraries, *path, *(Path(e).parent for e in executables)],
        environment,
        private,
        network=network,
        procfs=procfs,
    )
    return argv, jailed


def run_elf_jailed(target: str, engine: str, elf: Path, work: Path, *, private, timeout) -> str:
    """Run ``elf`` on ``target``'s ``engine`` through its backend, inside :func:`oracle_jail`; the console."""
    # The RTL engine checks out a network licence; the jail runs only Merlin, the backend and the guest image,
    # whose HTIF proxy offers no sockets.
    argv, jailed = oracle_jail(elf, work, private=private, roots=_merlin_roots(target), network=True, procfs=True)
    environment = [
        "--setenv",
        "PYTHONPATH",
        os.environ.get("PYTHONPATH", ""),
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
    ]
    command = [*argv, *environment, sys.executable, "-P", "-m", "merlin.targetgen.full_call_build", "--run-elf"]
    command += [target, engine, str(jailed), str(int(timeout))]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 120, stdin=subprocess.DEVNULL)
    console = work / "console.txt"
    if done.returncode or not console.is_file() or console.is_symlink():
        lines = [line for line in (done.stderr or done.stdout).splitlines() if line.strip()]
        raise RuntimeError(f"{engine} run failed in its jail: " + sanitized(lines[-1] if lines else "no output"))
    return console.read_text(errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
