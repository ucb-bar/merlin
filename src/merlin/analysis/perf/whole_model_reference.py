"""The REFERENCE ARM an open model's end result is judged against: its own host code, exact devices.

WHY NOT THE ORACLE. A model whose host code quantizes its activations dynamically amplifies a last-bit
difference: a rounding step flips, and the flip propagates. Measured on SmolVLA (int8 linears,
per-row dynamic activation scales), no two faithful computations of the model agree within its
capsule's numeric policy -- the numpy oracle against the capture's golden 1280/1600 elements, an x86
build of the very host code against the oracle 1244/1600, the RISC-V program against the oracle
1250/1600 -- while two packages whose every dispatch is exact print bit-identical outputs. Scaling the
input by 1 + 1e-7 moves the output by 0.109. An element-wise check against the oracle therefore fails
every correct package, and loosening it until it passes would pass incorrect ones too.

WHAT IS JUDGED INSTEAD. Every device group is exact by the program's own local check (the primary
correctness grade, unchanged). The END RESULT is then the model's own host code, compiled and linked
the same way, run on the same ISA, toolchain, machine and simulator, with every device group computed
by the target's own library -- the reference arm. Every dispatch is integer-exact, so a correct
package's output is BIT-IDENTICAL to it, and so is every dispatch's result digest (``GM_WORDS``), which
replaces the chained activation check as a gate (that check stays a reported diagnostic).

THE CACHE KEY is everything the reference's bytes depend on: the capsule (interface, weights, leaf
arguments), the host code (the cut module, the model's objects, the runtime and harness sources, the
flags), the toolchain (compilers, assembler, linker, C libraries), the machine and its ABI header, and
the simulator (binary and extension). A part nobody can state is ``UNKNOWN`` and the judgement FAILS
CLOSED -- a reference whose provenance cannot be named is not one.

WHERE BIT-IDENTITY CANNOT HOLD (a group whose declared epilogue is float, which a package may round
differently), the model's own declaration may state the fallback: ``end_result: {criterion:
declared_bound, cosine_min: ..., max_abs: ...}``, judged against the SAME reference arm, never the
oracle. The default is ``bit_identical``, and the result always says which criterion it applied. No
bound is written here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "whole_model_reference_v1"
BIT_IDENTICAL = "bit_identical"
DECLARED_BOUND = "declared_bound"
CRITERIA = (BIT_IDENTICAL, DECLARED_BOUND)
UNKNOWN = "UNKNOWN"
#: What a reference's bytes depend on; each is a digest or UNKNOWN.
KEY_PARTS = ("capsule", "host_code", "toolchain", "machine", "simulator")


class ReferenceUnavailable(RuntimeError):
    """No reference arm can be named or produced for this build; the end result fails closed."""


# ------------------------------------------------------------------------------------ identity


def file_digest(path: str | Path | None) -> str:
    """sha256 of a file's bytes, or ``UNKNOWN`` when there is no such file."""
    if path is None or not Path(path).is_file():
        return UNKNOWN
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(parts: Mapping[str, Any]) -> str:
    """One digest over a mapping of named parts; ``UNKNOWN`` when any part (at any depth) is."""

    def known(value: Any) -> bool:
        if isinstance(value, Mapping):
            return all(known(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(known(v) for v in value)
        return value is not None and value != UNKNOWN

    if not parts or not known(parts):
        return UNKNOWN
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


BUILD_ROOT = "<build>"


def files_identity(
    files: Mapping[str, str | Path | None], flags: Sequence[str] = (), *, root: str | Path | None = None
) -> dict[str, Any]:
    """``{"files": {name: sha}, "flags": [...], "digest": ...}`` -- the digest is UNKNOWN if any file is.

    ``root`` is the build's own directory: a flag that names a path inside it (an include directory the
    build copied its sources into) is recorded relative to it, because the same program built in two
    directories is the same program -- the bytes those directories hold are digested in ``files``."""
    prefix = str(Path(root)) if root is not None else None
    named = {str(name): file_digest(path) for name, path in sorted(files.items())}
    body = {"files": named, "flags": [_build_relative(str(f), prefix) for f in flags]}
    return {**body, "digest": identity_digest(body)}


def _build_relative(flag: str, prefix: str | None) -> str:
    if not prefix:
        return flag
    head, sep, tail = flag.partition(prefix)
    if not sep or (tail and not tail.startswith(os.sep)):
        return flag
    return head + BUILD_ROOT + tail


def toolchain_identity(compiler: str | Path, clang: str | Path, flags: Sequence[str] = ()) -> dict[str, Any]:
    """The toolchain a program was built with, by content: both compilers, the GCC driver's own passes and
    the C libraries it links for ``flags`` (asked of the driver, never assumed)."""

    def asked(option: str) -> str | None:
        try:
            answer = subprocess.run(
                [str(compiler), *map(str, flags), option], capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        # A driver that does not find a file echoes the bare name back.
        return answer if answer and os.sep in answer else None

    files = {
        "compiler": compiler,
        "clang": clang,
        "cc1": asked("-print-prog-name=cc1"),
        "as": asked("-print-prog-name=as"),
        "ld": asked("-print-prog-name=ld"),
        "libc.a": asked("-print-file-name=libc.a"),
        "libm.a": asked("-print-file-name=libm.a"),
        "libgcc.a": asked("-print-libgcc-file-name"),
    }
    return files_identity(files, flags)


def key_parts(record: Mapping[str, Any], simulator: Mapping[str, Any] | None) -> dict[str, str]:
    """The cache key's parts from an open build record (``reference_identity``) and a simulator identity."""
    identity = record.get("reference_identity") or {}
    return {
        "capsule": str((identity.get("capsule") or {}).get("digest") or UNKNOWN),
        "host_code": str((identity.get("host_code") or {}).get("digest") or UNKNOWN),
        "toolchain": str((identity.get("toolchain") or {}).get("digest") or UNKNOWN),
        "machine": identity_digest(identity.get("machine") or {}),
        "simulator": str((simulator or {}).get("digest") or UNKNOWN),
    }


def reference_key(parts: Mapping[str, str]) -> str:
    """The cache key, or :class:`ReferenceUnavailable` naming every part nobody could state."""
    unknown = [name for name in KEY_PARTS if parts.get(name) in (None, "", UNKNOWN)]
    if unknown:
        raise ReferenceUnavailable(
            f"the reference arm's key has UNKNOWN part(s) {unknown}: a reference whose provenance cannot be "
            "named is not one, so the end result fails closed"
        )
    return identity_digest({name: parts[name] for name in KEY_PARTS})


def simulator_identity(target: str, simulator: str) -> dict[str, Any] | None:
    """The target backend's own statement of which simulator build runs a program (``None``: it states none)."""
    from merlin.runtime.backends import base as backends

    stated = getattr(backends.get_backend(target), "simulator_identity", None)
    return stated(simulator) if callable(stated) else None


# --------------------------------------------------------------------------------------- cache


def cache_root(target: str) -> Path:
    from merlin.common.artifacts import cache_dir

    return Path(cache_dir("whole-model-reference")) / str(target)


def lookup(key: str, *, target: str, root: Path | None = None) -> dict[str, Any] | None:
    path = (root or cache_root(target)) / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim(key: str, *, target: str, root: Path, wait_s: int) -> dict[str, Any] | None:
    """The cached entry under ``key``, or ``None`` once this process holds its producer lock.

    One producer per key: an exclusive lock file; a waiter polls for the entry, and a lock whose holder
    died is taken over."""
    lock = root / f"{key}.lock"
    deadline = time.monotonic() + wait_s
    while True:
        existing = lookup(key, target=target, root=root)
        if existing is not None:
            return existing
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            return None
        except FileExistsError:
            holder = lock.read_text(encoding="utf-8").strip() if lock.is_file() else ""
            if holder.isdigit() and not _alive(int(holder)):
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise ReferenceUnavailable(f"another process has held the reference lock {lock} past {wait_s} s")
            time.sleep(30)


def _run_and_record(
    built: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    key: str,
    parts: Mapping[str, str],
    simulator_stated: Mapping[str, Any] | None,
    timeout: int,
    root: Path,
    build_record: Path,
) -> dict[str, Any]:
    """Run a built reference arm on ``simulator`` and cache what it printed under ``key``."""
    from merlin.analysis.perf import whole_model_verdict as V
    from merlin.runtime.backends import base as backends

    started = time.monotonic()
    console = backends.get_backend(target).run_elf(built["elf"], simulator=simulator, timeout=timeout)
    wall = round(time.monotonic() - started, 1)
    (root / f"{key}.console.txt").write_text(console, encoding="utf-8")
    driver = backends.whole_model_driver(target)
    parsed = V.parse_log(console, getattr(driver.program, "UART", None))
    inexact = sorted(g for g, (mismatches, _of) in parsed.local.items() if mismatches != 0)
    if inexact or not parsed.local:
        raise ReferenceUnavailable(f"the reference arm's own dispatches are not exact: {inexact[:8]}")
    if parsed.output_digest is None:
        raise ReferenceUnavailable("the reference arm printed no output digest")
    entry = {
        "schema": SCHEMA,
        "key": key,
        "parts": dict(parts),
        "target": target,
        "simulator": {"name": simulator, **dict(simulator_stated or {})},
        "output_digest": {"bytes": parsed.output_digest[0], "digest": parsed.output_digest[1]},
        "output_bits": output_bits(console),
        "words": {g: list(v) for g, v in parsed.words.items()},
        "dispatches": len(parsed.local),
        "elf_sha256": built["elf_sha256"],
        "build_record": str(build_record),
        "console": str(root / f"{key}.console.txt"),
        "wall_s": wall,
        "produced_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "provenance": built.get("provenance"),
    }
    (root / f"{key}.json").write_text(json.dumps(entry, indent=1, default=str) + "\n", encoding="utf-8")
    # The image is regenerable; its digest, record and console are what the entry cites.
    Path(built["elf"]).unlink(missing_ok=True)
    return entry


def produce(
    record: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    key: str,
    parts: Mapping[str, str],
    simulator_stated: Mapping[str, Any] | None,
    timeout: int,
    jobs: int | None = None,
    root: Path | None = None,
    wait_s: int = 6 * 3600,
) -> dict[str, Any]:
    """Build and run the reference arm of ``record``'s model and cache what it printed.

    The reference is the SAME capsule, machine and header built with no package -- the target's
    library answers every group -- and its host code must digest equal to the candidate's, or it is not
    this model's reference and is refused."""
    from merlin.analysis.perf import whole_model_open as WO

    root = root or cache_root(target)
    root.mkdir(parents=True, exist_ok=True)
    existing = _claim(key, target=target, root=root, wait_s=wait_s)
    if existing is not None:
        return existing
    try:
        work = root / key
        header = (record.get("program") or {}).get("abi_header") or {}
        extra = ((record.get("arguments") or {}).get("extra") or {}).get("path")
        # The candidate's host code is the reference's only if it is cut into the same chunks: a chunked
        # candidate's main.mlir never digests equal to an unchunked reference's, and the unchunked
        # reference is the 2-hour compile the chunking exists to avoid.
        chunk_ops = (record.get("forward_chunks") or {}).get("chunk_ops")
        # ...and run on the same harts: a candidate whose host code runs on a vector hart is compiled
        # for that hart's ISA, so a one-hart reference would never digest equal to it.
        host_hart = ((record.get("program") or {}).get("harts") or {}).get("host")
        built = WO.build(
            None,
            str((record.get("capsule") or {}).get("directory")),
            target=target,
            machine=str(record.get("machine")),
            header=str(header.get("header")),
            header_sha256=None if header.get("status") == "registry_declared" else header.get("sha256"),
            extra=extra,
            out=work / "build",
            verify="local",
            jobs=jobs,
            **({"chunk_ops": int(chunk_ops)} if chunk_ops is not None else {}),
            **({"host_hart": int(host_hart)} if host_hart is not None else {}),
        )
        mine = (record.get("reference_identity") or {}).get("host_code") or {}
        theirs = (built.get("reference_identity") or {}).get("host_code") or {}
        if mine.get("digest") in (None, UNKNOWN) or mine.get("digest") != theirs.get("digest"):
            differ = sorted(
                name
                for name in set(mine.get("files") or {}) | set(theirs.get("files") or {})
                if (mine.get("files") or {}).get(name) != (theirs.get("files") or {}).get(name)
            )
            flags = sorted(set(mine.get("flags") or ()) ^ set(theirs.get("flags") or ()))
            raise ReferenceUnavailable(
                f"the reference arm's host code does not digest equal to the candidate's (differs in "
                f"{differ[:8]}, flags {flags[:4]}); "
                "it would not be this program's reference"
            )
        return _run_and_record(
            built,
            target=target,
            simulator=simulator,
            key=key,
            parts=parts,
            simulator_stated=simulator_stated,
            timeout=timeout,
            root=root,
            build_record=work / "build" / "whole_model_open_build.json",
        )
    finally:
        (root / f"{key}.lock").unlink(missing_ok=True)


def prepare(
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    simulator: str = "spike",
    timeout: int = 7200,
    jobs: int | None = None,
    root: Path | None = None,
    wait_s: int = 6 * 3600,
    chunk_ops: int | None = None,
) -> dict[str, Any]:
    """Produce a model's reference arm AHEAD of any candidate, so a gate finds it cached.

    ``chunk_ops`` must be the size the candidates are built with (see :func:`produce`); ``None`` is the
    unchunked program.

    The reference's key is a property of the model, the machine, the toolchain and the simulator, never
    of a package, so it can be built and run while a candidate is still building: its own build with no
    package states the same host code a candidate's does (``produce`` refuses one that does not). The
    build is kept under ``<cache>/prepared/``; a key some other process already cached or is producing
    is returned or waited for, never produced twice."""
    from merlin.analysis.perf import whole_model_open as WO

    root = root or cache_root(target)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = root / "prepared" / f"{Path(model_capsule).name}_{machine}_{stamp}_{os.getpid()}"
    try:
        built = WO.build(
            None,
            str(model_capsule),
            target=target,
            machine=machine,
            header=str(header),
            header_sha256=header_sha256,
            out=work,
            verify="local",
            jobs=jobs,
            **({"chunk_ops": chunk_ops} if chunk_ops is not None else {}),
            # The harts a candidate built through the service runs on (its default: the machine's
            # vector hart, when it declares one), so the prepared reference is the candidate's.
            host_hart=WO.vector_host_hart(machine),
        )
    except BaseException:
        # A refused or failed build leaves nothing an entry could cite.
        shutil.rmtree(work, ignore_errors=True)
        raise
    stated = simulator_identity(target, simulator)
    parts = key_parts(built, stated)
    key = reference_key(parts)
    existing = _claim(key, target=target, root=root, wait_s=wait_s)
    if existing is not None:
        return existing
    try:
        return _run_and_record(
            built,
            target=target,
            simulator=simulator,
            key=key,
            parts=parts,
            simulator_stated=stated,
            timeout=timeout,
            root=root,
            build_record=work / "whole_model_open_build.json",
        )
    finally:
        (root / f"{key}.lock").unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------- judge


def output_bits(console: str) -> list[int] | None:
    """The ``OUT <n> <bits...>`` values a program printed (its first ``n`` output elements)."""
    for line in console.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == "OUT" and parts[1].isdigit():
            values = parts[2 : 2 + int(parts[1])]
            return [int(v) for v in values] if all(v.isdigit() for v in values) else None
    return None


def _floats(bits: Sequence[int]):
    import numpy as np

    return np.asarray([int(b) & 0xFFFFFFFF for b in bits], np.uint32).view(np.float32).astype(np.float64)


def judge(
    console: str,
    reference: Mapping[str, Any],
    *,
    declaration: Mapping[str, Any] | None = None,
    templates: Mapping[str, str] | None = None,
    elements: int | None = None,
) -> dict[str, Any]:
    """A candidate's console against a reference entry, under the model's declared criterion."""
    import numpy as np

    from merlin.analysis.perf import whole_model_verdict as V

    declaration = dict(declaration or {})
    criterion = str(declaration.get("criterion") or BIT_IDENTICAL)
    parsed = V.parse_log(console, templates)
    mine = parsed.output_digest
    theirs = reference.get("output_digest") or {}
    result: dict[str, Any] = {
        "basis": "reference arm",
        "end_result_criterion": criterion,
        "reference": {
            "key": reference.get("key"),
            "output_digest": theirs,
            "elf_sha256": reference.get("elf_sha256"),
            "simulator": reference.get("simulator"),
            "provenance": reference.get("provenance"),
        },
        "output_digest": None if mine is None else {"bytes": mine[0], "digest": mine[1]},
    }
    ref_words = {str(g): tuple(v) for g, v in (reference.get("words") or {}).items()}
    differ = sorted(
        (g for g in set(ref_words) | set(parsed.words) if ref_words.get(g) != parsed.words.get(g)),
        key=lambda g: int(g) if g.isdigit() else 10**9,
    )
    result["words"] = {"dispatches": len(ref_words), "agree": len(ref_words) - len(differ), "differ": differ[:24]}
    identical = mine is not None and [mine[0], mine[1]] == [theirs.get("bytes"), theirs.get("digest")]
    result["bit_identical"] = identical
    if criterion == BIT_IDENTICAL:
        result["passed"] = identical and not differ
    elif criterion == DECLARED_BOUND:
        cosine_min, max_abs = declaration.get("cosine_min"), declaration.get("max_abs")
        got, want = output_bits(console), reference.get("output_bits")
        if cosine_min is None or max_abs is None:
            result.update(passed=False, note="the declared bound states no cosine_min and max_abs")
        elif not got or not want or len(got) != len(want) or (elements is not None and len(got) != int(elements)):
            result.update(
                passed=False,
                note="a declared bound is judged over the whole output; the program printed only part of it",
            )
        else:
            a, b = _floats(got), _floats(want)
            cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
            worst = float(np.abs(a - b).max())
            result["bound"] = {"cosine": cosine, "max_abs": worst, "cosine_min": cosine_min, "max_abs_bound": max_abs}
            result["passed"] = cosine >= float(cosine_min) and worst <= float(max_abs)
    else:
        result.update(passed=False, note=f"unknown end-result criterion {criterion!r}; declared one of {CRITERIA}")
    return result


def end_result(
    console: str,
    record: Mapping[str, Any],
    *,
    target: str,
    simulator: str,
    declaration: Mapping[str, Any] | None = None,
    templates: Mapping[str, str] | None = None,
    timeout: int = 7200,
    jobs: int | None = None,
    root: Path | None = None,
    produce_missing: bool = True,
    elements: int | None = None,
) -> dict[str, Any]:
    """The end result of an open model's run against its reference arm; never raises -- a reference that
    cannot be named or produced is a failed end result with the reason."""
    try:
        stated = simulator_identity(target, simulator)
        parts = key_parts(record, stated)
        key = reference_key(parts)
        reference = lookup(key, target=target, root=root)
        if reference is None:
            if not produce_missing:
                raise ReferenceUnavailable(f"no reference arm is cached under {key[:16]} and none was produced")
            reference = produce(
                record,
                target=target,
                simulator=simulator,
                key=key,
                parts=parts,
                simulator_stated=stated,
                timeout=timeout,
                jobs=jobs,
                root=root,
            )
    except ReferenceUnavailable as why:
        return {
            "passed": False,
            "basis": "reference arm",
            "end_result_criterion": str((declaration or {}).get("criterion") or BIT_IDENTICAL),
            "note": str(why),
        }
    return judge(console, reference, declaration=declaration, templates=templates, elements=elements)
