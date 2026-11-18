"""Audit a produced ELF against a board's memory map, BEFORE anyone loads it onto that board.

Every "will it fit" decision in this repo is *predictive arithmetic* (``_ram_for_weights``
and the selected board's external-memory layout); nothing has ever looked at the artifact to confirm the prediction
came true. That is tolerable when the person running the binary can attach a debugger. It is not
tolerable when the binary is mailed to someone else's bench, because the failure modes are silent:

* a segment outside DRAM, or a region larger than the chip has → the boot dies before ``main`` with **no
  console output**, which is indistinguishable from the model hanging;
* the entry point not where the loader resumes → nothing runs at all;
* a missing ``.htif`` section → the loader cannot find ``tohost``/``fromhost``, so it never sees output
  or the exit request, and reports a timeout;
* an image with no vector instructions → a "successful" run that silently measured the scalar fallback.

Upload time also belongs here. A UART loader transmits each segment's **MemSiz** (not its file size), so
a large ``.bss`` or an embedded weights blob costs minutes of wall clock per attempt — the board's own
demos warn about exactly this. Reporting the estimate turns "why is flashing taking 20 minutes" into a
number we printed up front.

Reads section/segment headers with the toolchain's ``readelf``/``objdump`` (no pyelftools dependency),
and reuses :mod:`merlin_rvv.src.rvv.runtime.coverage` for the instruction mix — including its hard-won preference
for ``llvm-objdump``, because GNU objdump silently mis-decodes rv64gcv in bulk and fabricates ~0 %
vector coverage. Fails CLOSED: an unreadable ELF is an error, never a pass.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.process import invocation_records as _proc

# ---- elf_audit ---------------------------------------------------------------------------------


def _bytes_per_s(brd) -> float:
    return float(brd.loader_bytes_per_s)


def upload_bytes(brd, segments: list[Segment], sections: dict[str, tuple[int, int, str]]) -> tuple[int, str]:
    """Bytes this board's loader actually puts on the wire, and a one-line statement of why.

    The two loaders in use disagree by an order of magnitude on the same ELF:

    * ``uart_tsi`` writes every PT_LOAD's **MemSiz**, zero-filling past ``filesz`` — so an arena that
      claims the rest of DRAM is transmitted, as zeros, before the program starts;
    * ``pyuartsi`` walks the SECTION table and writes only ``SHT_PROGBITS`` with ``sh_addr > 0``,
      skipping ``SHT_NOBITS`` entirely.

    Guessing one model for both is what put a "4 min" figure next to an image that takes an hour.
    """
    from merlin.runtime.boards.catalog import LOADER_PYUARTSI, LOADER_UART_TSI

    if brd.loader == LOADER_PYUARTSI:
        total = sum(size for (addr, size, kind) in sections.values() if kind == "PROGBITS" and addr > 0)
        return total, "PROGBITS sections with addr>0 (pyuartsi skips NOBITS)"
    if brd.loader != LOADER_UART_TSI:
        raise ElfAuditError(f"{brd.name}: no qualified upload loader")
    return sum(s.memsz for s in segments), "PT_LOAD MemSiz incl. zero-fill (uart_tsi/fesvr)"


class ElfAuditError(RuntimeError):
    pass


@dataclass
class Segment:
    kind: str
    vaddr: int
    filesz: int
    memsz: int
    flags: str
    paddr: int | None = None

    @property
    def end(self) -> int:
        return self.vaddr + self.memsz


@dataclass
class AuditReport:
    elf: str
    board: str
    entry: int
    segments: list[Segment] = field(default_factory=list)
    #: name -> (addr, size, kind); `kind` is the ELF section type (PROGBITS/NOBITS/...)
    sections: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {
            "elf": self.elf,
            "board": self.board,
            "entry": hex(self.entry),
            "ok": self.ok,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "facts": dict(self.facts),
            "segments": [
                {"kind": s.kind, "vaddr": hex(s.vaddr), "filesz": s.filesz, "memsz": s.memsz, "flags": s.flags}
                for s in self.segments
            ],
            "sections": {k: {"addr": hex(a), "size": n, "kind": t} for k, (a, n, t) in self.sections.items()},
        }

    def render(self) -> str:
        lines = [f"ELF audit: {Path(self.elf).name}  board={self.board}  {'OK' if self.ok else 'FAIL'}"]
        for k, v in self.facts.items():
            lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for p in self.problems:
            lines.append(f"  FAIL  {p}")
        return "\n".join(lines)


def _tool(name: str) -> str | None:
    """A RISC-V binutil from the chipyard toolchain, or the plain name if it is on PATH."""
    from merlin.runtime.backends.baremetal import spike_launcher as _spike

    try:
        cand = Path(_spike.gcc_path()).with_name(f"riscv64-unknown-elf-{name}")
        if cand.is_file():
            return str(cand)
    except Exception:  # noqa: BLE001
        pass
    from shutil import which

    return which(f"riscv64-unknown-elf-{name}") or which(name)


def _run(cmd: list[str]) -> str:
    return _proc.run_checked(cmd, error=ElfAuditError, timeout=300, wrap_timeout=False, tail=500).stdout


def read_elf(elf: str | Path, *, strict: bool = False) -> tuple[int, list[Segment], dict[str, tuple[int, int, str]]]:
    """``(entry, segments, sections)`` from the ELF headers.

    Parsed from ``readelf -lSh`` text with ``str.split`` — no regex (repo rule) and no extra dependency.
    """
    readelf = _tool("readelf")
    if readelf is None:
        raise ElfAuditError("no readelf available (need the chipyard riscv toolchain on PATH)")
    out = _run([readelf, "-h", "-l", "-S", "-W", str(elf)])
    entry = 0
    has_entry = False
    segments: list[Segment] = []
    sections: dict[str, tuple[int, int, str]] = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Entry point address:"):
            entry = int(s.split(":", 1)[1].strip(), 16)
            has_entry = True
        elif s.startswith("LOAD"):
            # LOAD  offset vaddr paddr filesz memsz flags align
            parts = s.split()
            try:
                segment = Segment(
                    "LOAD",
                    int(parts[2], 16),
                    int(parts[4], 16),
                    int(parts[5], 16),
                    "".join(parts[6:-1]),
                    paddr=int(parts[3], 16),
                )
                if strict:
                    segment.flags = segment.flags.replace("E", "X")
                if strict and (
                    not segment.flags
                    or set(segment.flags) - set("RWX")
                    or not 0 <= segment.filesz <= segment.memsz
                    or segment.memsz <= 0
                ):
                    raise ElfAuditError(f"invalid LOAD segment in {elf}: {s}")
                segments.append(segment)
            except (IndexError, ValueError) as exc:
                if strict:
                    raise ElfAuditError(f"malformed LOAD segment in {elf}: {s}") from exc
                continue
        elif s.startswith("[") and "]" in s:
            # section table row: [ N] name type addr off size ...
            rest = s.split("]", 1)[1].split()
            if len(rest) >= 5:
                name = rest[0]
                try:
                    # (addr, size, kind). The kind matters: a section-loading loader skips NOBITS,
                    # so what it actually transmits is not what a segment view suggests.
                    sections[name] = (int(rest[2], 16), int(rest[4], 16), rest[1])
                except ValueError:
                    continue
    if not segments:
        raise ElfAuditError(f"no LOAD segments in {elf} — not a linked executable?")
    if strict and not has_entry:
        raise ElfAuditError(f"missing entry point in {elf}")
    return entry, segments, sections


def audit(
    elf: str | Path, brd, *, expect_htif: bool | None = None, require_vector: bool = True, ram_bytes: int | None = None
) -> AuditReport:
    """Check ``elf`` against board ``brd`` (a :class:`runtime.boards.Board`).

    ``expect_htif`` defaults to the board's console being HTIF. ``ram_bytes`` is the region the image
    was linked for, when the caller knows it (``build_app`` returns it) — the audit then also checks
    that the region itself fits the chip, not just that the segments do.
    """
    elf = Path(elf)
    if not elf.is_file():
        raise ElfAuditError(f"no such ELF: {elf}")
    entry, segments, sections = read_elf(elf)
    rep = AuditReport(elf=str(elf), board=brd.name, entry=entry, segments=segments, sections=sections)

    lo, hi = brd.dram_base, brd.dram_base + brd.dram_bytes
    for seg in segments:
        # A segment outside DRAM boots into nothing and prints nothing. Scratchpads/high weight regions
        # are legitimate targets, so only flag a segment that is neither inside DRAM nor obviously
        # another declared region: below DRAM base is a scratchpad/flash address, above the end is what
        # the external-weights mode does on purpose (checked separately).
        if seg.vaddr >= lo and seg.end > hi:
            rep.problems.append(
                f"LOAD segment at {hex(seg.vaddr)} ends at {hex(seg.end)}, past the board's DRAM end "
                f"{hex(hi)} ({brd.dram_bytes / 2**20:.0f} MB) — the image would not boot"
            )

    total_mem = sum(s.memsz for s in segments)
    total_file = sum(s.filesz for s in segments)
    rep.facts["load_segments"] = len(segments)
    rep.facts["image_memsz_mb"] = round(total_mem / 2**20, 2)
    rep.facts["image_filesz_mb"] = round(total_file / 2**20, 2)
    up_bytes, up_note = upload_bytes(brd, segments, sections)
    rep.facts["upload_bytes"] = up_bytes
    rep.facts["upload_loader"] = brd.loader
    rep.facts["upload_baud"] = brd.loader_baud
    rep.facts["upload_basis"] = up_note
    rep.facts["upload_estimate_s"] = round(up_bytes / _bytes_per_s(brd), 1)
    rep.facts["dram_used_pct"] = round(100.0 * total_mem / max(1, brd.dram_bytes), 1)
    if total_mem > brd.dram_bytes:
        rep.problems.append(
            f"image needs {total_mem / 2**20:.0f} MB of memory but the board has {brd.dram_bytes / 2**20:.0f} MB"
        )
    if ram_bytes is not None:
        rep.facts["linked_region_mb"] = round(ram_bytes / 2**20, 1)
        if ram_bytes > brd.dram_bytes:
            rep.problems.append(
                f"linked for a {ram_bytes / 2**20:.0f} MB region but the board has {brd.dram_bytes / 2**20:.0f} MB"
            )
    if rep.facts["upload_estimate_s"] > 120:
        rep.warnings.append(
            f"upload takes ~{rep.facts['upload_estimate_s'] / 60:.0f} min: "
            f"{up_bytes / 2**20:.1f} MB via {rep.facts['upload_loader']} at "
            f"{rep.facts['upload_baud']} baud ({up_note}) — say so in the delivery README, with the "
            "baud, so the reader can rescale if they run a faster link"
        )

    # Entry point: the loader resumes at a fixed address; an entry elsewhere runs nothing.
    if not (lo <= entry < hi):
        rep.problems.append(
            f"entry point {hex(entry)} is outside DRAM [{hex(lo)}, {hex(hi)}) — the loader would resume into nothing"
        )

    want_htif = brd.console == "htif" if expect_htif is None else expect_htif
    if want_htif and ".htif" not in sections:
        rep.problems.append(
            "no .htif section: the loader locates tohost/fromhost by scanning for it, so it would see "
            "neither console output nor the exit request and report a timeout"
        )
    if not want_htif and ".htif" in sections and getattr(brd, "flow", "") == "baremetal":
        # The inverse check, and the one that matters for a board nobody here can attach to. HTIF is
        # host-ASSISTED: the image writes tohost and waits for a host to clear it, so on silicon with
        # no host it hangs on its second character -- inside the first print, before any model work,
        # looking exactly like a core that never booted. That is not hypothetical; it is what the first
        # binaries sent to such a board did.
        #
        # Scoped to the BARE-METAL flow, where this harness owns the console and `.htif` can only have
        # come from linking the HTIF backend. Under an RTOS the same section name is also allocated by
        # the SoC's reboot support (a NOBITS block nothing writes once the HTIF console is configured
        # out), so flagging it there would be a false alarm on a correctly-built image.
        rep.problems.append(
            f"a .htif section is present but this board's console is '{brd.console}': HTIF needs a host "
            "servicing tohost, and with none the image hangs in its first print before running the "
            "model -- indistinguishable from a core that never booted"
        )
    rep.facts["console"] = brd.console
    rep.facts["has_htif_section"] = ".htif" in sections

    # Instruction mix. A shipped image with no vector ops is a run that silently measured the scalar
    # fallback -- worse than a failure, because it looks like a result.
    try:
        from merlin_rvv.src.rvv.runtime.coverage import audit_binary

        mix = audit_binary(elf)
        rep.facts["vector_instructions"] = mix.vector
        rep.facts["scalar_compute_instructions"] = mix.scalar_compute
        rep.facts["vector_coverage"] = round(mix.coverage_overall, 4)
        if require_vector and mix.vector == 0:
            rep.problems.append(
                "no vector instructions in the image — this would measure the scalar fallback while "
                "looking like an RVV result"
            )
        fallbacks = mix.scalar_fallback_symbols()
        if fallbacks:
            rep.facts["scalar_fallback_symbols"] = len(fallbacks)
    except Exception as exc:  # noqa: BLE001
        # Fail closed on the audit we could not do, rather than reporting a pass we did not earn.
        rep.warnings.append(f"instruction-mix audit unavailable ({type(exc).__name__}: {exc})")
    return rep


def audit_json(elf: str | Path, brd, out: str | Path | None = None, **kw) -> dict:
    """Audit and (optionally) write the report as JSON for a delivery package."""
    rep = audit(elf, brd, **kw)
    d = rep.to_dict()
    if out is not None:
        Path(out).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return d


def main(argv: list[str] | None = None) -> int:
    import argparse

    from merlin.runtime.boards.catalog import board as _board

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--board", default="spike_riscv64")
    ap.add_argument(
        "--dram-mb", type=int, default=None, help="override the board's DRAM (state what the chip really has)"
    )
    ap.add_argument("--no-require-vector", action="store_true")
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    kw = {"dram_bytes": a.dram_mb * 1024 * 1024} if a.dram_mb else {}
    brd = _board(a.board, **kw)
    if brd.dram_base is None or brd.dram_bytes is None:
        ap.error(f"board {brd.name} ({brd.flow}) has no physical DRAM window to audit an image against")
    rep = audit(a.elf, brd, require_vector=not a.no_require_vector)
    print(rep.render())
    if a.json:
        Path(a.json).write_text(json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ---- execution_memory --------------------------------------------------------------------------
# Admit loaded and absolute runtime intervals against a selected decoded map.
#
# The execution provider owns the map and its derivation. A simulator backing
# store size is not a decoded-map contract. This validator supports identity
# mapped executables and disjoint, simultaneously live runtime reservations;
# aliasing and virtual translation need separate ownership/translation proofs.

"""Admit loaded and absolute runtime intervals against a selected decoded map.

The execution provider owns the map and its derivation. A simulator backing
store size is not a decoded-map contract. This validator supports identity
mapped executables and disjoint, simultaneously live runtime reservations;
aliasing and virtual translation need separate ownership/translation proofs.
"""


class ExecutionMemoryError(ValueError):
    """Selected memory evidence or deployment intervals do not close."""


def _pin(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    except OSError as exc:
        raise ExecutionMemoryError(f"memory evidence unavailable: {path}") from exc


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _extent(name: str, begin: object, size: object, bits: int, alignment: object = 1) -> tuple[int, int]:
    if type(begin) is not int or type(size) is not int or not 0 <= begin < 1 << bits or size <= 0:
        raise ExecutionMemoryError(f"{name}: unknown or invalid memory extent")
    if begin + size > 1 << bits:
        raise ExecutionMemoryError(f"{name}: address extent overflows")
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1) or begin % alignment:
        raise ExecutionMemoryError(f"{name}: invalid or unsatisfied alignment")
    return begin, begin + size


def _permissions(value: object, name: str) -> frozenset[str]:
    if not isinstance(value, str) or not value or set(value) - set("RWX") or len(set(value)) != len(value):
        raise ExecutionMemoryError(f"{name}: unknown memory permissions")
    return frozenset(value)


@dataclass(frozen=True)
class MemoryMapBinding:
    """Pin to the selected provider's complete decoded-map JSON artifact."""

    path: Path
    sha256: str
    execution_identity: str

    def validate(self) -> dict:
        """Close map bytes and all declared source/elaboration evidence."""
        if not _digest(self.sha256) or _pin(self.path)["sha256"] != self.sha256:
            raise ExecutionMemoryError("selected memory map changed or has an unknown identity")
        try:
            record = json.loads(Path(self.path).read_text())
        except (OSError, ValueError) as exc:
            raise ExecutionMemoryError("unreadable selected memory map") from exc
        if not isinstance(record, dict) or record.get("schema") != "merlin.execution-memory-map.v1":
            raise ExecutionMemoryError("unknown memory map schema")
        if record.get("status") != "complete" or record.get("identity_mapped") is not True:
            raise ExecutionMemoryError("memory map is incomplete or not identity mapped")
        if not isinstance(record.get("execution_identity"), str) or not record["execution_identity"]:
            raise ExecutionMemoryError("memory map lacks the selected execution identity")
        if (
            not isinstance(self.execution_identity, str)
            or not self.execution_identity
            or record["execution_identity"] != self.execution_identity
        ):
            raise ExecutionMemoryError("memory map does not match the selected execution identity")
        bits = record.get("address_bits")
        if type(bits) is not int or bits not in (32, 64):
            raise ExecutionMemoryError("memory map has unknown address width")
        evidence = record.get("file_pins")
        if not isinstance(evidence, list) or not evidence:
            raise ExecutionMemoryError("memory map lacks source/elaboration evidence")
        for declared in evidence:
            if (
                not isinstance(declared, dict)
                or not isinstance(declared.get("path"), str)
                or not _digest(declared.get("sha256"))
            ):
                raise ExecutionMemoryError("memory map has malformed evidence")
            actual = _pin(declared["path"])
            if actual["sha256"] != declared["sha256"]:
                raise ExecutionMemoryError(f"memory evidence changed: {declared['path']}")
            if "bytes" in declared and (type(declared["bytes"]) is not int or declared["bytes"] != actual["bytes"]):
                raise ExecutionMemoryError(f"memory evidence extent changed: {declared['path']}")
        regions = record.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ExecutionMemoryError("memory map lacks decoded regions")
        intervals, names = [], set()
        for region in regions:
            if (
                not isinstance(region, dict)
                or not isinstance(region.get("name"), str)
                or not region["name"]
                or region["name"] in names
            ):
                raise ExecutionMemoryError("memory map has missing or duplicate region names")
            names.add(region["name"])
            begin, end = _extent(region["name"], region.get("begin"), region.get("bytes"), bits)
            _permissions(region.get("permissions"), region["name"])
            intervals.append((begin, end))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ExecutionMemoryError("decoded memory regions overlap")
        if _pin(self.path)["sha256"] != self.sha256:
            raise ExecutionMemoryError("selected memory map changed during validation")
        return record


@dataclass(frozen=True)
class MemoryReservation:
    """One absolute runtime allocation, live for the complete execution.

    It must be disjoint from other reservations and loaded segments. Storage
    inside ELF allocations is already covered by the segment, and is not an
    additional reservation. This API does not grant aliasing or phase reuse.
    """

    name: str
    begin: int
    bytes: int
    permissions: str = "RW"
    alignment: int = 1


def _covered(begin: int, end: int, permissions: frozenset[str], regions: list[dict]) -> bool:
    cursor = begin
    for region in sorted(regions, key=lambda item: item["begin"]):
        if not permissions <= frozenset(region["permissions"]):
            continue
        lo, hi = region["begin"], region["begin"] + region["bytes"]
        if hi <= cursor:
            continue
        if lo > cursor:
            return False
        cursor = hi
        if cursor >= end:
            return True
    return False


def admit_execution_memory(
    elf: str | Path, binding: MemoryMapBinding, reservations: tuple[MemoryReservation, ...]
) -> dict:
    """Refuse unknown maps, gaps, overflow, missing permissions and collisions.

    The caller/provider remains responsible for complete demands and for tying
    map derivation to the actually selected execution engine. Passing this gate
    proves interval admission, not a numerical or performance result.
    """
    if not isinstance(binding, MemoryMapBinding):
        raise ExecutionMemoryError("an explicit selected memory map is required")
    record = binding.validate()
    elf = Path(elf)
    original_pin = _pin(elf)
    with elf.open("rb") as stream:
        ident = stream.read(20)
    if len(ident) != 20 or ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2) or ident[6] != 1:
        raise ExecutionMemoryError("unknown ELF address width")
    if int.from_bytes(ident[16:18], "little" if ident[5] == 1 else "big") != 2:
        raise ExecutionMemoryError("memory admission requires an absolute executable ELF")
    if (32 if ident[4] == 1 else 64) != record["address_bits"]:
        raise ExecutionMemoryError("ELF and selected memory map address widths disagree")
    entry, segments, _ = read_elf(elf, strict=True)
    intervals, loads = [], []
    for index, segment in enumerate(segments):
        name = f"LOAD[{index}]"
        begin, end = _extent(name, segment.vaddr, segment.memsz, record["address_bits"])
        permissions = _permissions(segment.flags, name)
        if segment.paddr != segment.vaddr:
            raise ExecutionMemoryError(f"{name}: identity physical mapping is unproved")
        if not _covered(begin, end, permissions, record["regions"]):
            raise ExecutionMemoryError(f"{name}: outside decoded map or required permissions")
        intervals.append((begin, end, name))
        loads.append({"name": name, "begin": begin, "bytes": segment.memsz, "permissions": segment.flags})
    if not any(segment.vaddr <= entry < segment.end and "X" in segment.flags for segment in segments):
        raise ExecutionMemoryError("entry is outside executable loaded memory")
    runtime, names = [], set()
    if not isinstance(reservations, tuple):
        raise ExecutionMemoryError("runtime reservations must be an immutable tuple")
    for use in reservations:
        if not isinstance(use, MemoryReservation) or not isinstance(use.name, str) or not use.name or use.name in names:
            raise ExecutionMemoryError("runtime reservation lacks a distinct name")
        names.add(use.name)
        begin, end = _extent(use.name, use.begin, use.bytes, record["address_bits"], use.alignment)
        permissions = _permissions(use.permissions, use.name)
        if not _covered(begin, end, permissions, record["regions"]):
            raise ExecutionMemoryError(f"{use.name}: outside decoded map or required permissions")
        intervals.append((begin, end, use.name))
        runtime.append(
            {
                "name": use.name,
                "begin": begin,
                "bytes": use.bytes,
                "permissions": use.permissions,
                "alignment": use.alignment,
            }
        )
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ExecutionMemoryError(f"{left[2]} and {right[2]} overlap; storage ownership is unproved")
    if _pin(elf) != original_pin:
        raise ExecutionMemoryError("ELF changed during memory admission")
    binding.validate()
    return {
        "schema": "merlin.execution-memory-admission.v1",
        "status": "pass",
        "elf": original_pin,
        "memory_map": _pin(binding.path),
        "execution_identity": record["execution_identity"],
        "address_bits": record["address_bits"],
        "file_pins": record["file_pins"],
        "loaded_intervals": loads,
        "runtime_reservations": runtime,
        "scope": (
            "Identity-mapped decoded intervals and complete-execution disjoint reservations only; "
            "no numerical or timing certification."
        ),
    }
