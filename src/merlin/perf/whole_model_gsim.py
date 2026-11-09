"""Run a whole-model program on the GSIM emulator and grade its outputs from a host-side memory dump.

    verdict = run_gsim_whole_model(elf, memory_map, oracle, target=target)

WHY A DUMP. A whole-model program built with ``verify="host_dump"`` (:func:`merlin.perf.whole_model_build.build`)
computes no digest on the core: on an elaborated-RTL simulator a byte-wise on-target digest costs
about 12 cycles per byte, 3-13M cycles per group and over 250M for the model, which made the whole
model uncertifiable there. Instead the emulator, when the program exits, writes every buffer the
build's ``memory_map.json`` names to a file, and :func:`merlin.perf.whole_model_build.grade_memory`
grades the file against the oracle.

WHERE THE BYTES COME FROM. Not from the simulator's DRAM backing store alone: whatever the last-level
cache still holds dirty has not been written back, and those bytes are stale there (measured: 142 of
66,560 bytes on a 16x16x16 matmul program). The emulator's default ``hybrid`` dump takes the backing
store and then re-reads, through the SoC's own coherent host port, every line the inclusive L2's
directory has an entry for; ``coherent`` reads every byte through that port (exact and ~40x slower).
``backdoor_diagnostic=True`` also dumps the bare backing store and reports how many output bytes it
would have got wrong -- the claim is measured on every such run, not assumed.

WHICH BYTES. Every buffer record anywhere in a group's row of the map -- a mapping with an
``address`` and a ``bytes`` -- is dumped, whatever field holds it: the output, a tolerance-declared
group's operands, and any inputs a later map version adds for local grading. A record that lies wholly
in a READ-ONLY loadable segment of the ELF (weights, constants) is served from the ELF's own bytes
instead of the dump, since the program cannot have legitimately changed it; the verdict counts those
bytes. Anything else the grader asks for raises: an unread byte is never zero.

WHAT IS REFUSED, NEVER PASSED. A run that did not exit, a dump that is truncated, malformed or
missing a region, a console that does not print exactly one timing line per group of the map in its
order, a map recorded for a different ELF: each makes the verdict ``status="refused"`` with the
reason, and a refused verdict is never quotable.

WHAT THE CYCLES ARE. The program's own ``rdcycle`` brackets: the whole measured window and one count
per group. They come off an elaborated-RTL simulator, which the measurement ladder
(``src/merlin/resources/contracts/measurement_ladder.yaml``) admits for ranking and output equivalence, not as a
cycle authority; the verdict carries that adjudication beside the numbers.

The UART spellings are the target's own (its ``whole_model_driver`` declares them); nothing here
names a target.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import struct
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from . import emulator_capture as EC

__all__ = [
    "DUMP_ENGINE",
    "SCHEMA",
    "DumpError",
    "GsimWholeModelError",
    "buffer_records",
    "dump_regions",
    "measure",
    "parse_console",
    "read_dump",
    "regrade",
    "run_gsim_whole_model",
]

SCHEMA = "gsim_whole_model_verdict_v1"

#: The engine home (``<build>/rtl_engines/<target>/<engine>/``) of the emulator that can dump memory.
DUMP_ENGINE = "gsim-hostdump"

_MAGIC, _TRAILER = b"GSIMDMP1", b"GSIMEND1"
_SOURCES = {0: "coherent", 1: "backing_store", 2: "hybrid"}
_EXACT_SOURCES = ("coherent", "hybrid")
_LOADMEM_ENV = "MERLIN_GSIM_LOADMEM"  # the harness's own switch: load the ELF by backdoor, not over TSI


class GsimWholeModelError(RuntimeError):
    """The run cannot even be attempted (inputs disagree, no emulator); nothing was simulated."""


class DumpError(ValueError):
    """A dump file that is not a complete dump of the regions asked for, or a read it cannot serve."""


# ------------------------------------------------------------------------------------ what to read


def buffer_records(row: Any) -> Iterator[Mapping[str, Any]]:
    """Every buffer record in a map row: any mapping carrying an ``address`` and a ``bytes``, at any depth."""
    if isinstance(row, Mapping):
        if "address" in row and "bytes" in row:
            yield row
        for value in row.values():
            if isinstance(value, (Mapping, list, tuple)):
                yield from buffer_records(value)
    elif isinstance(row, (list, tuple)):
        for value in row:
            yield from buffer_records(value)


def _constant_segments(elf: Path) -> list[tuple[int, int, int]]:
    """``(address, size, file offset)`` of each loadable segment the ELF maps without write permission."""
    data = elf.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise GsimWholeModelError(f"{elf} is not a little-endian ELF64")
    (phoff,) = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    segments = []
    for index in range(phnum):
        kind, flags, offset, vaddr, _paddr, filesz, _memsz, _align = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phentsize
        )
        if kind == 1 and not flags & 0x2 and filesz:  # PT_LOAD, not PF_W: bytes the program cannot own
            segments.append((vaddr, filesz, offset))
    return segments


def dump_regions(layout: Mapping[str, Any], constant: list[tuple[int, int, int]] = ()) -> list[tuple[int, int]]:
    """Every ``(address, nbytes)`` the dump must hold, in first-seen order, each once.

    Every buffer record of every group row (:func:`buffer_records`), except one lying wholly inside a
    read-only segment in ``constant`` -- that one is read from the ELF.
    """
    seen: dict[tuple[int, int], None] = {}
    for row in layout.get("groups") or ():
        for place in buffer_records(row):
            address, nbytes = int(place["address"]), int(place["bytes"])
            if any(base <= address and address + nbytes <= base + size for base, size, _ in constant):
                continue
            seen.setdefault((address, nbytes), None)
    if not seen:
        raise GsimWholeModelError("the memory map names no writable buffer")
    return list(seen)


# -------------------------------------------------------------------------------------------- dump


def read_dump(path: str | Path, expected: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    """Parse a dump file; with ``expected``, require exactly those regions, in that order.

    Returns ``{"source": "coherent" | "hybrid" | "backing_store", "regions": {(address, nbytes): bytes}}``.
    Raises :class:`DumpError` on anything short of a complete, well-formed dump.
    """
    data = Path(path).read_bytes()
    if len(data) < 24 or data[:8] != _MAGIC:
        raise DumpError(f"{path} is not a GSIM dump (bad or missing header)")
    source, count = struct.unpack_from("<QQ", data, 8)
    if source not in _SOURCES:
        raise DumpError(f"{path} declares an unknown source {source}")
    offset, total = 24, 0
    regions: dict[tuple[int, int], bytes] = {}
    order: list[tuple[int, int]] = []
    for index in range(count):
        if offset + 16 > len(data):
            raise DumpError(f"{path} is truncated in the header of region {index} of {count}")
        address, nbytes = struct.unpack_from("<QQ", data, offset)
        offset += 16
        if offset + nbytes > len(data):
            raise DumpError(f"{path} is truncated inside region {index} ([{address:#x}, +{nbytes}))")
        regions[(address, nbytes)] = data[offset : offset + nbytes]
        order.append((address, nbytes))
        offset += nbytes
        total += nbytes
    if data[offset : offset + 8] != _TRAILER or len(data) != offset + 24:
        raise DumpError(f"{path} has no complete trailer after {count} regions (truncated or overlong)")
    tail_count, tail_total = struct.unpack_from("<QQ", data, offset + 8)
    if (tail_count, tail_total) != (count, total):
        raise DumpError(f"{path}'s trailer says {tail_count} regions / {tail_total} bytes, the body {count} / {total}")
    if expected is not None and order != list(expected):
        missing = [r for r in expected if r not in regions]
        raise DumpError(
            f"{path} does not hold the regions asked for: {len(missing)} missing"
            + (f", first [{missing[0][0]:#x}, +{missing[0][1]})" if missing else ", or not in the order asked")
        )
    return {"source": _SOURCES[source], "regions": regions}


def _reader(
    regions: Mapping[tuple[int, int], bytes], elf_bytes: bytes = b"", constant: list[tuple[int, int, int]] = ()
) -> tuple[Callable[[int, int], bytes], dict[str, int]]:
    """``read(address, size)`` over a dump, then the ELF's read-only segments; anything else raises."""
    served = {"from_dump": 0, "from_elf": 0}

    def read(address: int, size: int) -> bytes:
        for (base, nbytes), blob in regions.items():
            if base <= address and address + size <= base + nbytes:
                served["from_dump"] += size
                return blob[address - base : address - base + size]
        for base, nbytes, offset in constant:
            if base <= address and address + size <= base + nbytes:
                served["from_elf"] += size
                start = offset + address - base
                return elf_bytes[start : start + size]
        raise DumpError(f"neither the dump nor a read-only ELF segment holds [{address:#x}, +{size})")

    return read, served


# ----------------------------------------------------------------------------------------- console


def _match(template: str, line: str) -> dict[str, str] | None:
    """Fields of ``line`` if it has ``template``'s shape (whitespace tokens; ``{x}`` and ``k={x}``)."""
    want, got = template.split(), line.split()
    if len(want) != len(got):
        return None
    fields: dict[str, str] = {}
    for pattern, token in zip(want, got, strict=True):
        head, brace, rest = pattern.partition("{")
        if not brace:
            if pattern != token:
                return None
            continue
        name, _, tail = rest.partition("}")
        if not token.startswith(head) or not token.endswith(tail) or len(token) < len(head) + len(tail):
            return None
        fields[name] = token[len(head) : len(token) - len(tail)]
    return fields


def _int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def parse_console(text: str, protocol: Mapping[str, str]) -> dict[str, Any]:
    """The run's own timing lines, read with the target's declared spellings (``protocol``).

    ``groups`` is every group line in order (a group printed twice is kept twice, so the count check
    sees it); ``full_model``/``metric`` the window and bracket totals; ``argmax`` the core's own
    comparison. An absent line is ``None``, never zero.
    """
    groups: list[dict[str, Any]] = []
    found: dict[str, Any] = {"full_model": None, "metric": None, "argmax": None}
    for line in text.splitlines():
        fields = _match(protocol["group"], line)
        if fields is not None and _int(fields["group"]) is not None and _int(fields["cycles"]) is not None:
            groups.append({"group": int(fields["group"]), "kind": fields["kind"], "cycles": int(fields["cycles"])})
            continue
        for key in ("full_model", "metric"):
            fields = _match(protocol[key], line) if key in protocol else None
            if fields is not None and _int(fields["cycles"]) is not None:
                found[key] = int(fields["cycles"])
        fields = _match(protocol["argmax"], line) if "argmax" in protocol else None
        if fields is not None:
            found["argmax"] = {k: _int(v) for k, v in fields.items()}
    return {"groups": groups, **found}


# ------------------------------------------------------------------------------------------ runner


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else json.loads(Path(value).read_text(encoding="utf-8"))


def _unlimited_stack() -> None:  # pragma: no cover - runs in the child
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


def _engine(target: str, emulator: str | Path | None) -> dict[str, Any]:
    """The dump-capable emulator and what its receipt says it is; an unreceipted binary is refused."""
    from merlin.targetgen.gsim_emulator import engine_home

    binary = Path(emulator) if emulator is not None else engine_home(target, DUMP_ENGINE) / "emulator"
    if not binary.is_file():
        raise GsimWholeModelError(f"no dump-capable GSIM emulator at {binary}")
    receipt_path = binary.parent / "build_receipt.json"
    if not receipt_path.is_file():
        raise GsimWholeModelError(f"{binary} has no build_receipt.json beside it; which model it is is unrecorded")
    digest = _sha256(binary)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("binary_sha256") != digest:
        raise GsimWholeModelError(
            f"{binary} hashes to {digest[:12]}, its receipt describes {str(receipt.get('binary_sha256'))[:12]}"
        )
    engine = {
        "path": str(binary),
        "sha256": digest,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "firrtl_sha256": receipt.get("firrtl_sha256"),
    }
    equivalence = binary.parent / "equivalence.json"
    if equivalence.is_file():
        engine["equivalence"] = json.loads(equivalence.read_text(encoding="utf-8")).get("summary")
    return engine


def _refused(verdict: dict[str, Any], why: str) -> dict[str, Any]:
    verdict.update({"status": "refused", "refusal": why, "quotable": False})
    return verdict


def run_gsim_whole_model(
    elf: str | Path,
    memory_map: str | Path | Mapping[str, Any],
    oracle: str | Path | Mapping[str, Any],
    *,
    target: str,
    out: str | Path | None = None,
    emulator: str | Path | None = None,
    max_cycles: int = 400_000_000,
    timeout_s: float | None = None,
    dump_mode: str = "hybrid",
    backdoor_diagnostic: bool = False,
    local: Any = None,
) -> dict[str, Any]:
    """Run ``elf`` on the dump-capable GSIM emulator and grade every group from the dump.

    Inputs: the ELF a ``verify="host_dump"`` build produced, that build's ``memory_map.json`` and
    ``oracle.json`` (paths or loaded dicts), and ``target``. Optional: ``out`` (where the console, the
    dump and ``verdict.json`` go; default a fresh directory under
    ``out/artifacts/perf-bench/<target>/whole-model-gsim/``), ``emulator`` (overrides the engine
    home's binary), ``max_cycles`` (bounds program AND dump), ``timeout_s`` (wall clock),
    ``dump_mode`` (``hybrid`` or ``coherent``, see the module doc), ``backdoor_diagnostic``, and
    ``local`` -- the :class:`~merlin.perf.whole_model_build.LocalReference` (or the model capsule
    directory to build one from) that grades each exact group LOCALLY, on its own dumped inputs;
    without it an exact group is ``unverified`` and the verdict is not quotable.

    Returns the verdict (also written to ``<out>/verdict.json``):
      * ``status`` -- ``graded`` or ``refused`` (with ``refusal``); ``quotable`` is true only when
        :func:`grade_memory` finds every group correct and the argmax read from the dump equals the
        oracle's and the core's own;
      * ``whole_window_cycles`` / ``bracket_sum_cycles`` -- the program's own window and bracket total;
      * ``per_group`` -- ``{group, kind, cycles, correct}`` in program order;
      * ``grade`` -- :func:`grade_memory`'s verdict, unchanged;
      * ``argmax`` -- from the dumped logits, from the core's own line, the oracle's and the golden's;
      * ``cycles_adjudication`` -- what the measurement ladder says a GSIM cycle count may claim;
      * ``wall_seconds``, ``emulator`` (path, digest, receipt), ``dump``, ``provenance``.
    Raises :class:`GsimWholeModelError` only when the run cannot be attempted at all.
    """
    from merlin.common import provenance as PROV
    from merlin.perf import whole_model_build as W
    from merlin.runtime.backends import base as backends

    if dump_mode not in _EXACT_SOURCES:
        raise GsimWholeModelError(f"dump_mode is one of {_EXACT_SOURCES}, not {dump_mode!r}")
    elf = Path(elf).resolve()
    layout, expected = _load(memory_map), _load(oracle)
    if layout.get("schema") != W.MEMORY_MAP_SCHEMA:
        raise GsimWholeModelError(f"memory map schema is {layout.get('schema')!r}, not {W.MEMORY_MAP_SCHEMA!r}")
    elf_sha = _sha256(elf)
    if layout.get("elf_sha256") != elf_sha:
        raise GsimWholeModelError(
            f"the memory map was recorded for ELF {str(layout.get('elf_sha256'))[:12]}, not {elf_sha[:12]}"
        )
    engine = _engine(target, emulator)
    protocol = dict(backends.whole_model_driver(target).program.UART)
    constant = _constant_segments(elf)
    regions = dump_regions(layout, constant)

    if out is None:
        from merlin.common.artifacts import git_sha7, utc_stamp
        from merlin.common.paths import artifacts_dir

        out = artifacts_dir() / "perf-bench" / target / "whole-model-gsim" / f"{utc_stamp()}_{git_sha7()}_{elf_sha[:8]}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regions.txt").write_text("".join(f"{a:#x} {n}\n" for a, n in regions), encoding="utf-8")
    dump, backdoor = out / "memory.dump", out / "memory.backing_store.dump"
    for stale in (dump, backdoor):
        stale.unlink(missing_ok=True)
    argv = [
        engine["path"],
        str(elf),
        f"+max-cycles={int(max_cycles)}",
        f"+loadmem={elf}",
        f"+dump-regions={out / 'regions.txt'}",
        f"+dump-out={dump}",
        f"+dump-mode={dump_mode}",
        *([f"+dump-backdoor-out={backdoor}"] if backdoor_diagnostic else []),
    ]
    verdict: dict[str, Any] = {
        "schema": SCHEMA,
        "target": target,
        "elf": str(elf),
        "elf_sha256": elf_sha,
        "argv": argv,
        "emulator": engine,
        "regions": {"count": len(regions), "bytes": sum(n for _, n in regions)},
    }
    started = time.monotonic()
    with open(out / "console.txt", "wb") as console:
        # stderr is bounded (head + tail kept, the middle counted) and the run ends at its first
        # hardware assertion: see merlin.perf.emulator_capture for the incident that set both.
        capture = EC.run_capped(
            argv,
            stdout=console,
            stderr_path=out / "emulator.stderr.txt",
            env={**os.environ, _LOADMEM_ENV: "1"},
            timeout_s=timeout_s,
            preexec_fn=_unlimited_stack,
        )
    returncode: int | None = None if capture.timed_out else capture.returncode
    verdict["wall_seconds"] = round(time.monotonic() - started, 1)
    verdict["returncode"] = returncode
    verdict["stderr_capture"] = {k: v for k, v in capture.to_dict().items() if k not in ("returncode", "timed_out")}
    text = (out / "console.txt").read_text(encoding="utf-8", errors="replace")
    stderr = (out / "emulator.stderr.txt").read_text(encoding="utf-8", errors="replace")
    verdict["simulated"] = _finished(stderr)
    verdict["machine"] = _machine(engine.get("firrtl_sha256"))
    verdict["provenance"] = PROV.record(
        pins=W._pins_for(target),
        artifacts={"elf": elf, "emulator": engine["path"], "emulator_receipt": engine["receipt"]},
        extra={
            "engine": DUMP_ENGINE,
            "memory_map_elf_sha256": layout.get("elf_sha256"),
            "machine": verdict["machine"],
        },
    )
    verdict["cycles_adjudication"] = _adjudication(target)
    local = _local(local, target)
    try:
        if capture.assertion is not None:
            # The design reached a state its RTL declares impossible: that is the verdict, whatever
            # the console or a dump might say (the run was stopped there, so there is no dump).
            where = f" at {capture.assertion['location']}" if capture.assertion.get("location") else ""
            return _refused(verdict, f"hardware_assertion: {capture.assertion.get('message') or 'unnamed'}{where}")
        return _grade(
            verdict, text, layout, expected, regions, protocol, dump, backdoor, returncode, elf, constant, local
        )
    finally:
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n", encoding="utf-8")


def _local(local: Any, target: str) -> Any:
    """A LocalReference as given, or built from a model capsule directory; None stays None."""
    if local is None or not isinstance(local, (str, Path)):
        return local
    from merlin.perf import whole_model_build as W

    return W.LocalReference.from_capsule(local, target=target)


def regrade(
    run_dir: str | Path,
    memory_map: str | Path | Mapping[str, Any],
    oracle: str | Path | Mapping[str, Any],
    *,
    target: str,
    local: Any = None,
    write: str | None = "verdict.regraded.json",
) -> dict[str, Any]:
    """Grade an EXISTING run directory again -- no simulation: its console, stderr and dump as they are.

    The dump must still hold every byte the (possibly newer) ``memory_map`` asks for; a read it
    cannot serve refuses the verdict, never zeros. The first verdict's run facts (ELF, emulator,
    exit status, wall time, provenance) are carried over; only the grading is redone.
    """
    from merlin.runtime.backends import base as backends

    run_dir = Path(run_dir)
    first = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))
    layout, expected = _load(memory_map), _load(oracle)
    elf = Path(first["elf"])
    if _sha256(elf) != first["elf_sha256"] or layout.get("elf_sha256") != first["elf_sha256"]:
        raise GsimWholeModelError("the ELF, the run and the memory map do not all name the same bytes")
    asked = []
    for line in (run_dir / "regions.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            address, nbytes = line.split()
            asked.append((int(address, 0), int(nbytes)))
    keep = ("schema", "target", "elf", "elf_sha256", "argv", "emulator", "regions", "wall_seconds", "returncode",
            "simulated", "machine", "provenance", "cycles_adjudication")  # fmt: skip
    verdict = {k: first[k] for k in keep if k in first}
    verdict["regraded_from"] = str(run_dir / "verdict.json")
    text = (run_dir / "console.txt").read_text(encoding="utf-8", errors="replace")
    protocol = dict(backends.whole_model_driver(target).program.UART)
    result = _grade(
        verdict, text, layout, expected, asked, protocol, run_dir / "memory.dump",
        run_dir / "memory.backing_store.dump", first.get("returncode"), elf, _constant_segments(elf),
        _local(local, target),
    )  # fmt: skip
    if write:
        (run_dir / write).write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    return result


def _finished(stderr: str) -> dict[str, Any]:
    """The emulator's own FINISHED line and dump span, read structurally."""
    found: dict[str, Any] = {}
    for line in stderr.splitlines():
        fields = dict(p.split("=", 1) for p in line.split() if "=" in p)
        if line.startswith("[gsim-emu] FINISHED:"):
            found["emulator"] = fields
        elif line.startswith("[gsim-dump] begin"):
            found["dump_begin"] = fields
        elif line.startswith("[gsim-dump] complete"):
            found["dump_complete"] = fields
    return found


def _adjudication(target: str) -> Any:
    try:
        from merlin.perf import measurement_ladder as ML

        return ML.adjudicate(target, "gsim").to_dict()
    except Exception as error:  # noqa: BLE001 -- recorded as UNKNOWN, never dropped
        return {"state": "UNKNOWN", "reason": str(error)}


def _grade(verdict, text, layout, expected, regions, protocol, dump, backdoor, returncode, elf, constant, local):
    import numpy as np

    from merlin.perf import whole_model_build as W

    if returncode is None:
        return _refused(verdict, "the emulator hit its wall-clock timeout")
    console = parse_console(text, protocol)
    wanted = [(int(r["group"]), str(r["kind"])) for r in layout["groups"]]
    printed = [(g["group"], g["kind"]) for g in console["groups"]]
    verdict["whole_window_cycles"] = console["full_model"]
    verdict["bracket_sum_cycles"] = console["metric"]
    if printed != wanted:
        return _refused(
            verdict,
            f"the console printed {len(printed)} group line(s) and the map has {len(wanted)} groups "
            f"(or they differ in order or kind); the program did not complete its window",
        )
    if console["full_model"] is None:
        return _refused(verdict, "the console has no whole-window cycle line")
    if returncode != 0:
        return _refused(verdict, f"the emulator exited {returncode} (a program failure or an incomplete dump)")
    try:
        parsed = read_dump(dump, regions)
    except (OSError, DumpError) as error:
        return _refused(verdict, f"the dump is not complete: {error}")
    if parsed["source"] not in _EXACT_SOURCES:
        return _refused(verdict, f"the dump was read from the {parsed['source']}, which is not coherent")
    verdict["dump"] = {"path": str(dump), "sha256": _sha256(dump), "source": parsed["source"]}
    done = verdict["simulated"].get("dump_complete") or {}
    for key in ("cycles", "coherent_lines", "stale_bytes"):
        if _int(str(done.get(key))) is not None:
            verdict["dump"][key] = int(done[key])
    read, served = _reader(parsed["regions"], elf.read_bytes(), constant)
    try:
        grade = W.grade_memory(read, layout, expected, local=local)
        final = layout["groups"][-1]
        raw = read(int(final["address"]), int(final["bytes"]))
    except DumpError as error:
        return _refused(verdict, str(error))
    verdict["dump"]["bytes_served"] = dict(served)
    logits = np.frombuffer(raw, dtype=f"<i{int(final['element_bytes'])}")
    host_argmax = int(np.argmax(logits))  # the first maximum, as the core's own loop picks it
    agree = set(grade.get("agree") or ())
    cycles = {g["group"]: g["cycles"] for g in console["groups"]}
    verdict["per_group"] = [{"group": g, "kind": k, "cycles": cycles[g], "correct": str(g) in agree} for g, k in wanted]
    verdict["grade"] = grade
    core = console["argmax"] or {}
    verdict["argmax"] = {
        "from_dump": host_argmax,
        "from_core": core.get("got"),
        "oracle": expected.get("argmax"),
        "golden": expected.get("golden_argmax"),
        "agrees_with_oracle": host_argmax == expected.get("argmax"),
        "core_agrees_with_dump": core.get("got") == host_argmax,
    }
    if backdoor.is_file():
        verdict["backing_store_diagnostic"] = _stale(read_dump(backdoor, regions)["regions"], parsed["regions"], layout)
    verdict["status"] = "graded"
    verdict["quotable"] = bool(
        grade.get("quotable") and verdict["argmax"]["agrees_with_oracle"] and verdict["argmax"]["core_agrees_with_dump"]
    )
    return verdict


def _registry_machine(firrtl_sha256: str | None) -> str | None:
    """The hardware-registry entry whose declared digest IS the emulator's FIRRTL, or None."""
    import yaml

    from merlin.common import provenance as PROV

    registry = yaml.safe_load(PROV.pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(registry.get("pins") or {}), **(registry.get("artifacts") or {})}
    named = sorted(n for n, e in entries.items() if firrtl_sha256 and (e or {}).get("digest") == firrtl_sha256)
    return named[0] if len(named) == 1 else None


def _machine(firrtl_sha256: str | None) -> dict[str, Any]:
    """Which registered hardware the emulator models: the entry declaring its FIRRTL, and that check."""
    from merlin.common import provenance as PROV

    entry = _registry_machine(firrtl_sha256)
    found: dict[str, Any] = {"registry_entry": entry, "firrtl_sha256": firrtl_sha256}
    if entry is None:
        found["gap"] = "no hardware-registry entry declares this FIRRTL digest; the machine is UNKNOWN"
        return found
    try:
        found["artifact_check"] = PROV.verify_artifact(entry).to_dict()
    except Exception as error:  # noqa: BLE001 -- recorded, never dropped
        found["artifact_check"] = {"UNKNOWN": str(error)}
    return found


def _flat_image(verdict: Mapping[str, Any], layout: Mapping[str, Any], out: Path) -> dict[str, Any] | None:
    """One raw image from ``base`` covering every buffer the map names, for a reader that slices.

    Filled from the dump, then from the ELF's read-only segments; any byte neither covers is counted
    in ``holes`` (it cannot be a byte the map names, since every named one is in one or the other).
    """
    dump = verdict.get("dump") or {}
    if verdict.get("status") != "graded" or not dump.get("path"):
        return None
    elf = Path(verdict["elf"])
    constant = _constant_segments(elf)
    regions = read_dump(dump["path"])["regions"]
    places = [(int(p["address"]), int(p["bytes"])) for row in layout["groups"] for p in buffer_records(row)]
    base = min(a for a, _ in places)
    image = bytearray(max(a + n for a, n in places) - base)
    covered = bytearray(len(image))
    elf_bytes = elf.read_bytes()
    for start, size, offset in constant:
        lo, hi = max(start, base), min(start + size, base + len(image))
        if lo < hi:
            image[lo - base : hi - base] = elf_bytes[offset + lo - start : offset + hi - start]
            covered[lo - base : hi - base] = b"\x01" * (hi - lo)
    for (address, nbytes), blob in regions.items():
        image[address - base : address - base + nbytes] = blob
        covered[address - base : address - base + nbytes] = b"\x01" * nbytes
    path = out / "memory.image"
    path.write_bytes(bytes(image))
    return {"path": str(path), "base": base, "bytes": len(image), "holes": covered.count(0)}


def measure(
    *,
    elf: str | Path,
    elf_sha256: str,
    machine: str,
    target: str,
    build_record: Mapping[str, Any],
    out: str | Path,
) -> dict[str, Any]:
    """The pipeline's MEASURER CONTRACT over :func:`run_gsim_whole_model` (``--measurer
    merlin.perf.whole_model_gsim:measure``).

    Returns ``elf_sha256`` and ``machine`` as actually RUN -- the machine is the registry entry whose
    FIRRTL digest the emulator's receipt names, derived, never echoed from the request, with the
    requested one beside it -- ``engine``, ``cycles`` (the program's whole-window count) with
    ``cycles_source``, and CORRECTNESS SEPARATELY (``correctness``: the grade, the argmax, and whether
    it is quotable), so a consumer that may not quote an elaborated-RTL cycle count can still take the
    correctness verdict. The output is ``memory_dump`` (``{"path", "base"}``, a raw image covering
    every buffer the map names) when the run was graded, else ``uart`` (the console) so a reader of it
    finds the groups absent rather than passed.
    """
    elf = Path(elf)
    actual = _sha256(elf)
    if actual != elf_sha256:
        raise GsimWholeModelError(f"asked to measure ELF {elf_sha256[:12]} but {elf} is {actual[:12]}")
    memory_map = build_record.get("memory_map")
    oracle = (build_record.get("oracle") or {}).get("path")
    if not memory_map or not oracle:
        raise GsimWholeModelError("the build record carries no memory map or oracle; build it with verify='host_dump'")
    out = Path(out)
    capsule = (build_record.get("capsule") or {}).get("directory")
    verdict = run_gsim_whole_model(elf, memory_map, oracle, target=target, out=out, local=capsule)
    ran_on = (verdict.get("machine") or {}).get("registry_entry")
    result: dict[str, Any] = {
        "elf_sha256": verdict["elf_sha256"],
        "machine": ran_on or f"UNKNOWN: no registry entry declares FIRRTL {verdict['emulator'].get('firrtl_sha256')}",
        "machine_requested": machine,
        "engine": "gsim",
        "cycles": verdict.get("whole_window_cycles"),
        "cycles_source": "the program's own rdcycle window (whole model), on elaborated-RTL GSIM",
        "per_group_cycles": {str(r["group"]): r["cycles"] for r in verdict.get("per_group") or ()},
        "correctness": {
            "status": verdict.get("status"),
            "refusal": verdict.get("refusal"),
            "quotable": verdict.get("quotable"),
            "grade": verdict.get("grade"),
            "argmax": verdict.get("argmax"),
        },
        "cycles_adjudication": verdict.get("cycles_adjudication"),
        "verdict": str(out / "verdict.json"),
        "wall_seconds": verdict.get("wall_seconds"),
    }
    image = _flat_image(verdict, _load(memory_map), out)
    if image is not None:
        result["memory_dump"] = image
    else:
        result["uart"] = str(out / "console.txt")
    return result


def main(argv: list[str] | None = None) -> int:
    """``merlin-whole-model-gsim --build <whole-model build dir>``: run it on GSIM, grade it from the dump."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="merlin-whole-model-gsim",
        description="Run a verify=host_dump whole-model build on the dump-capable GSIM emulator and grade "
        "every group from an end-of-run memory dump.",
    )
    parser.add_argument("--build", required=True, type=Path, help="a merlin-whole-model-build output directory")
    parser.add_argument("--out", type=Path, help="where the console, dump and verdict go")
    parser.add_argument("--emulator", type=Path, help="override the engine home's emulator")
    parser.add_argument("--max-cycles", type=int, default=400_000_000)
    parser.add_argument("--timeout", type=float, help="wall-clock seconds")
    parser.add_argument("--dump-mode", choices=_EXACT_SOURCES, default="hybrid")
    parser.add_argument("--backdoor-diagnostic", action="store_true", help="also measure the bare backing store")
    args = parser.parse_args(argv)
    record = json.loads((args.build / "whole_model_build.json").read_text(encoding="utf-8"))
    if record.get("verify") != "host_dump":
        parser.error(f"{args.build} was built with verify={record.get('verify')!r}; the dump needs host_dump")
    verdict = run_gsim_whole_model(
        record["elf"],
        record["memory_map"],
        record["oracle"]["path"],
        target=record["target"],
        out=args.out,
        emulator=args.emulator,
        max_cycles=args.max_cycles,
        timeout_s=args.timeout,
        dump_mode=args.dump_mode,
        backdoor_diagnostic=args.backdoor_diagnostic,
        local=(record.get("capsule") or {}).get("directory"),
    )
    grade = verdict.get("grade") or {}
    print(
        json.dumps(
            {
                "status": verdict.get("status"),
                "refusal": verdict.get("refusal"),
                "quotable": verdict.get("quotable"),
                "whole_window_cycles": verdict.get("whole_window_cycles"),
                "groups_correct": len(grade.get("agree") or ()),
                "groups_wrong": [d.get("group") for d in grade.get("disagree") or ()],
                "argmax": verdict.get("argmax"),
                "wall_seconds": verdict.get("wall_seconds"),
            },
            indent=1,
        )
    )
    if verdict.get("status") != "graded":
        return 2
    return 0 if verdict.get("quotable") else 1


def _stale(backing: Mapping, coherent: Mapping, layout: Mapping[str, Any]) -> dict[str, Any]:
    """How many bytes of each group's output the DRAM backing store alone would have got wrong."""
    stale_bytes, stale_groups = 0, []
    for row in layout["groups"]:
        key = (int(row["address"]), int(row["bytes"]))
        if key not in coherent:
            continue
        differ = sum(1 for x, y in zip(backing[key], coherent[key], strict=True) if x != y)
        if differ:
            stale_groups.append({"group": int(row["group"]), "bytes_differing": differ})
            stale_bytes += differ
    return {
        "what": "bytes of each group's output where the DRAM backing store disagrees with the exact dump",
        "bytes_differing": stale_bytes,
        "groups_affected": stale_groups,
    }
