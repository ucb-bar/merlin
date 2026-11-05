"""The functional engine's accelerator command stream without a commit log.

A commit log prints every retired instruction (one ~60-byte line each) so that a reader can replay the
integer registers and recover the operand values of the few accelerator commands. On large programs
that is the whole cost: one 86.6 M-instruction executable runs in 0.95 s without the log and 228-301 s
with it. :file:`spike_command_recorder.cc` is an engine extension library that records, as they
retire, only the instructions a command-stream reader yields (the accelerator extension's own
instructions and host fences) with the values their source registers held, plus the retired count.

:func:`observe` runs the declared commit-log invocation in that form and returns the recorded
``(bits, rs1, rs2)`` stream, or ``None`` whenever the record is not a complete equivalent of the log
(no recorder selected, a different engine binary, more than one hart, a program that touches the
retirement counter, a retired count near the reader's bound, a run that did not finish): the caller
then reads the commit log as before. Only an operator-built recorder is used, selected by a pinned
manifest (:data:`SELECTION_ENV`) that names the exact engine binary it was compiled against.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

#: JSON ``{"path": <manifest>, "sha256": <its digest>}`` selecting a built recorder (see :func:`build`).
SELECTION_ENV = "MERLIN_COMMAND_RECORDER"
OUT_ENV = "MERLIN_COMMAND_RECORDER_OUT"
INNER_ENV = "MERLIN_COMMAND_RECORDER_INNER"
LIMIT_ENV = "MERLIN_COMMAND_RECORDER_LIMIT"
EXTENSION = "command_recorder"
SOURCE = Path(__file__).with_name("spike_command_recorder.cc")
SCHEMA = "merlin.command_recorder.v1"
_MAGIC = b"MCREC001"
_RECORD = struct.Struct("<IIQQ")
_COMMIT_FLAG = "--log-commits"
DEFAULT_MAX_COMMANDS = 4_000_000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected() -> dict[str, Any] | None:
    """The selected recorder manifest with its pin, or ``None`` (none selected, or its bytes changed)."""
    raw = os.environ.get(SELECTION_ENV, "").strip()
    if not raw:
        return None
    try:
        pin = json.loads(raw)
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            return None
        path = Path(pin["path"])
        if not path.is_absolute():
            return None
        if _sha(path) != pin["sha256"]:
            return None
        manifest = json.loads(path.read_text(encoding="utf-8"))
        library = manifest["library"]
        if (
            manifest.get("schema") != SCHEMA
            or _sha(Path(library["path"])) != library["sha256"]
            or _sha(SOURCE) != manifest.get("source_sha256")
            or not Path(library["path"]).is_absolute()
            or manifest.get("byte_order") != "little"
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {**manifest, "pin": {"path": str(path), "sha256": pin["sha256"]}}


def invocation(argv, env, out: Path, manifest) -> tuple[list[str], dict[str, str]] | None:
    """The recorder form of a declared commit-log ``argv``, or ``None`` when it has no exact equivalent.

    The declared command must log commits only (no execution lines, no log file), run one hart, name
    one accelerator extension and run the very engine binary the recorder was built against.
    """
    argv = [str(a) for a in argv]
    extensions = [i for i, a in enumerate(argv) if a.startswith("--extension=")]
    if (
        not argv
        or argv.count(_COMMIT_FLAG) != 1
        or "-l" in argv
        or any(a.startswith("--log=") or a == "--log" for a in argv)
        or len(extensions) != 1
        or any(a.startswith("-p") and a != "-p1" for a in argv[1:])
        or any(a.startswith("--nprocs") or a.startswith("--hartids") for a in argv[1:])
        or any(a in ("-d", "--debug-cmd", "--interactive") or a.startswith("--debug-cmd=") for a in argv[1:])
        or argv[-1].startswith("-")
    ):
        return None
    try:
        import shutil

        engine = shutil.which(argv[0], path=env.get("PATH"))
        if engine is None or _sha(Path(engine).resolve(strict=True)) != manifest["engine"]["sha256"]:
            return None
    except (OSError, KeyError, TypeError, IndexError):
        return None
    at = extensions[0]
    inner = argv[at].partition("=")[2]
    if not inner or inner == EXTENSION:
        return None
    command = [a for a in argv[: at + 1] if a != _COMMIT_FLAG]
    command += ["--extlib=" + manifest["library"]["path"], "--extension=" + EXTENSION]
    command += [a for a in argv[at + 1 :] if a != _COMMIT_FLAG]
    return command, {**env, OUT_ENV: str(out), INNER_ENV: inner}


def read(path: Path, *, max_retired: int, max_commands: int = DEFAULT_MAX_COMMANDS):
    """The recorded ``(bits, rs1, rs2)`` stream, or ``None`` unless it equals the commit log's.

    ``max_retired`` is the commit-log reader's bound. The recorder counts retirements the way the
    engine does; the record is used only well inside the bound (half of it), where the two counts
    cannot fall on different sides, and only when nothing touched the counter.
    """
    if any(type(v) is not int or v < 1 for v in (max_retired, max_commands)):
        return None
    size = _RECORD.size
    # Bound both the on-disk output and Python memory before reading any bytes. Decoding stays lazy.
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(len(_MAGIC) + size * (max_commands + 1) + 1)
    except OSError:
        return None
    if len(data) > len(_MAGIC) + size * (max_commands + 1):
        return None
    if not data.startswith(_MAGIC) or (len(data) - len(_MAGIC)) % size or len(data) < len(_MAGIC) + size:
        return None
    bits, flags, retired, count = _RECORD.unpack_from(data, len(data) - size)
    body = memoryview(data)[len(_MAGIC) : len(data) - size]
    if bits != 0 or flags != 0 or count != len(body) // size or retired > max_retired // 2:
        return None
    if count > max_commands or retired < count:
        return None
    for word, hart, rs1, rs2 in _RECORD.iter_unpack(body):
        if hart != 0 or word == 0:
            return None
    return ((word, rs1, rs2) for word, _hart, rs1, rs2 in _RECORD.iter_unpack(body))


def observe(argv, env, *, scratch: Path, timeout_s: float, max_retired: int, max_commands=DEFAULT_MAX_COMMANDS):
    """``(records, manifest)`` from one recorder run of the declared ``argv``, or ``None`` (see module)."""
    manifest = selected()
    if manifest is None:
        return None
    if timeout_s <= 0 or type(max_commands) is not int or max_commands < 1:
        return None
    out = Path(scratch) / "commands.rec"
    plan = invocation(argv, env, out, manifest)
    if plan is None:
        return None
    command, environment = plan
    environment[LIMIT_ENV] = str(max_commands)
    try:
        out.unlink(missing_ok=True)
        finished = subprocess.run(  # noqa: S603 - the declared engine invocation plus the recorder
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if finished.returncode != 0 or not out.is_file():
        return None
    # Reopen the actual selected products after execution as well; changed bytes revoke this path.
    if selected() != manifest or invocation(argv, env, out, manifest) is None:
        return None
    records = read(out, max_retired=max_retired, max_commands=max_commands)
    return None if records is None else (records, manifest)


def build(include_root: Path, engine: Path, out_dir: Path, *, compiler: str = "g++") -> dict[str, str]:
    """Compile the recorder against ``engine``'s own installed headers; returns the selection pin.

    ``include_root`` is the include directory the engine binary was built with (its ``riscv/`` headers
    define the hart and instruction-table layout the recorder reads), ``engine`` that binary.
    """
    import sys

    if sys.byteorder != "little":
        raise ValueError("the recorder binary protocol requires a little-endian build host")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / "libcommand_recorder.so"
    command = [compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-I", str(include_root), "-o", str(library), str(SOURCE)]
    subprocess.run(command, check=True)  # noqa: S603 - the operator's compiler on this module's source
    manifest = {
        "schema": SCHEMA,
        "byte_order": sys.byteorder,
        "library": {"path": str(library.resolve()), "sha256": _sha(library)},
        "engine": {"path": str(Path(engine).resolve()), "sha256": _sha(Path(engine).resolve())},
        "source_sha256": _sha(SOURCE),
        "include_root": str(Path(include_root).resolve()),
        "headers_sha256": hashlib.sha256(json.dumps(
            {str(p.relative_to(include_root)): _sha(p) for p in sorted(Path(include_root).rglob("*.h"))},
            sort_keys=True,
        ).encode()).hexdigest(),
        "compiler": command,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the functional engine's command recorder.")
    parser.add_argument("--include", required=True, type=Path, help="the engine build's include directory")
    parser.add_argument("--engine", required=True, type=Path, help="the engine binary it will be loaded into")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args(argv)
    pin = build(args.include, args.engine, args.out, compiler=args.compiler)
    print(f"export {SELECTION_ENV}='{json.dumps(pin)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
