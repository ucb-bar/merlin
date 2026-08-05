"""Audit a produced ELF against a board's memory map, BEFORE anyone loads it onto that board.

Every "will it fit" decision in this repo is *predictive arithmetic* (``_ram_for_weights``,
``LINK_LIMIT``, ``EXT_MAX_WEIGHTS``); nothing has ever looked at the artifact to confirm the prediction
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
and reuses :mod:`merlin.baselines.rvv_audit` for the instruction mix — including its hard-won preference
for ``llvm-objdump``, because GNU objdump silently mis-decodes rv64gcv in bulk and fabricates ~0 %
vector coverage. Fails CLOSED: an unreadable ELF is an error, never a pass.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Observed UART-TSI throughput, bytes/second. 921600 baud 8N1 ≈ 92 KB/s of payload; the loader sends
#: MemSiz, so this converts an image's memory footprint into minutes on someone's bench.
UART_BYTES_PER_S = 92_000


class ElfAuditError(RuntimeError):
    pass


@dataclass
class Segment:
    kind: str
    vaddr: int
    filesz: int
    memsz: int
    flags: str

    @property
    def end(self) -> int:
        return self.vaddr + self.memsz


@dataclass
class AuditReport:
    elf: str
    board: str
    entry: int
    segments: list[Segment] = field(default_factory=list)
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)   # name -> (addr, size)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {"elf": self.elf, "board": self.board, "entry": hex(self.entry), "ok": self.ok,
                "problems": list(self.problems), "warnings": list(self.warnings),
                "facts": dict(self.facts),
                "segments": [{"kind": s.kind, "vaddr": hex(s.vaddr), "filesz": s.filesz,
                              "memsz": s.memsz, "flags": s.flags} for s in self.segments],
                "sections": {k: {"addr": hex(a), "size": n} for k, (a, n) in self.sections.items()}}

    def render(self) -> str:
        lines = [f"ELF audit: {Path(self.elf).name}  board={self.board}  "
                 f"{'OK' if self.ok else 'FAIL'}"]
        for k, v in self.facts.items():
            lines.append(f"  {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for p in self.problems:
            lines.append(f"  FAIL  {p}")
        return "\n".join(lines)


def _tool(name: str) -> str | None:
    """A RISC-V binutil from the chipyard toolchain, or the plain name if it is on PATH."""
    from ..runtime.backends import spike as _spike
    try:
        cand = Path(_spike.gcc_path()).with_name(f"riscv64-unknown-elf-{name}")
        if cand.is_file():
            return str(cand)
    except Exception:                                            # noqa: BLE001
        pass
    from shutil import which
    return which(f"riscv64-unknown-elf-{name}") or which(name)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ElfAuditError(f"{cmd[0]} failed: {proc.stderr[-500:]}")
    return proc.stdout


def read_elf(elf: str | Path) -> tuple[int, list[Segment], dict[str, tuple[int, int]]]:
    """``(entry, segments, sections)`` from the ELF headers.

    Parsed from ``readelf -lSh`` text with ``str.split`` — no regex (repo rule) and no extra dependency.
    """
    readelf = _tool("readelf")
    if readelf is None:
        raise ElfAuditError("no readelf available (need the chipyard riscv toolchain on PATH)")
    out = _run([readelf, "-h", "-l", "-S", "-W", str(elf)])
    entry = 0
    segments: list[Segment] = []
    sections: dict[str, tuple[int, int]] = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Entry point address:"):
            entry = int(s.split(":", 1)[1].strip(), 16)
        elif s.startswith("LOAD"):
            # LOAD  offset vaddr paddr filesz memsz flags align
            parts = s.split()
            try:
                segments.append(Segment("LOAD", int(parts[2], 16), int(parts[4], 16),
                                        int(parts[5], 16), "".join(parts[6:-1])))
            except (IndexError, ValueError):
                continue
        elif s.startswith("[") and "]" in s:
            # section table row: [ N] name type addr off size ...
            rest = s.split("]", 1)[1].split()
            if len(rest) >= 5:
                name = rest[0]
                try:
                    sections[name] = (int(rest[2], 16), int(rest[4], 16))
                except ValueError:
                    continue
    if not segments:
        raise ElfAuditError(f"no LOAD segments in {elf} — not a linked executable?")
    return entry, segments, sections


def audit(elf: str | Path, brd, *, expect_htif: bool | None = None,
          require_vector: bool = True, ram_bytes: int | None = None) -> AuditReport:
    """Check ``elf`` against board ``brd`` (a :class:`runtime.boards.Board`).

    ``expect_htif`` defaults to the board's console being HTIF. ``ram_bytes`` is the region the image
    was linked for, when the caller knows it (``build_app`` returns it) — the audit then also checks
    that the region itself fits the chip, not just that the segments do.
    """
    elf = Path(elf)
    if not elf.is_file():
        raise ElfAuditError(f"no such ELF: {elf}")
    entry, segments, sections = read_elf(elf)
    rep = AuditReport(elf=str(elf), board=brd.name, entry=entry,
                      segments=segments, sections=sections)

    lo, hi = brd.dram_base, brd.dram_base + brd.dram_bytes
    for seg in segments:
        # A segment outside DRAM boots into nothing and prints nothing. Scratchpads/high weight regions
        # are legitimate targets, so only flag a segment that is neither inside DRAM nor obviously
        # another declared region: below DRAM base is a scratchpad/flash address, above the end is what
        # the external-weights mode does on purpose (checked separately).
        if seg.vaddr >= lo and seg.end > hi:
            rep.problems.append(
                f"LOAD segment at {hex(seg.vaddr)} ends at {hex(seg.end)}, past the board's DRAM end "
                f"{hex(hi)} ({brd.dram_bytes / 2**20:.0f} MB) — the image would not boot")

    total_mem = sum(s.memsz for s in segments)
    total_file = sum(s.filesz for s in segments)
    rep.facts["load_segments"] = len(segments)
    rep.facts["image_memsz_mb"] = round(total_mem / 2**20, 2)
    rep.facts["image_filesz_mb"] = round(total_file / 2**20, 2)
    rep.facts["upload_estimate_s"] = round(total_mem / UART_BYTES_PER_S, 1)
    rep.facts["dram_used_pct"] = round(100.0 * total_mem / max(1, brd.dram_bytes), 1)
    if total_mem > brd.dram_bytes:
        rep.problems.append(
            f"image needs {total_mem / 2**20:.0f} MB of memory but the board has "
            f"{brd.dram_bytes / 2**20:.0f} MB")
    if ram_bytes is not None:
        rep.facts["linked_region_mb"] = round(ram_bytes / 2**20, 1)
        if ram_bytes > brd.dram_bytes:
            rep.problems.append(
                f"linked for a {ram_bytes / 2**20:.0f} MB region but the board has "
                f"{brd.dram_bytes / 2**20:.0f} MB")
    if rep.facts["upload_estimate_s"] > 120:
        rep.warnings.append(
            f"upload takes ~{rep.facts['upload_estimate_s'] / 60:.0f} min over a UART loader "
            f"(it transmits MemSiz, not file size) — say so in the delivery README")

    # Entry point: the loader resumes at a fixed address; an entry elsewhere runs nothing.
    if not (lo <= entry < hi):
        rep.problems.append(f"entry point {hex(entry)} is outside DRAM "
                            f"[{hex(lo)}, {hex(hi)}) — the loader would resume into nothing")

    want_htif = brd.console == "htif" if expect_htif is None else expect_htif
    if want_htif and ".htif" not in sections:
        rep.problems.append(
            "no .htif section: the loader locates tohost/fromhost by scanning for it, so it would see "
            "neither console output nor the exit request and report a timeout")
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
            "model -- indistinguishable from a core that never booted")
    rep.facts["console"] = brd.console
    rep.facts["has_htif_section"] = ".htif" in sections

    # Instruction mix. A shipped image with no vector ops is a run that silently measured the scalar
    # fallback -- worse than a failure, because it looks like a result.
    try:
        from ..baselines.rvv_audit import audit_binary
        mix = audit_binary(elf)
        rep.facts["vector_instructions"] = mix.vector
        rep.facts["scalar_compute_instructions"] = mix.scalar_compute
        rep.facts["vector_coverage"] = round(mix.coverage_overall, 4)
        if require_vector and mix.vector == 0:
            rep.problems.append(
                "no vector instructions in the image — this would measure the scalar fallback while "
                "looking like an RVV result")
        fallbacks = mix.scalar_fallback_symbols()
        if fallbacks:
            rep.facts["scalar_fallback_symbols"] = len(fallbacks)
    except Exception as exc:                                     # noqa: BLE001
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

    from .boards import board as _board

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("elf")
    ap.add_argument("--board", default="spike_riscv64")
    ap.add_argument("--dram-mb", type=int, default=None,
                    help="override the board's DRAM (state what the chip really has)")
    ap.add_argument("--no-require-vector", action="store_true")
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    kw = {"dram_bytes": a.dram_mb * 1024 * 1024} if a.dram_mb else {}
    brd = _board(a.board, **kw)
    rep = audit(a.elf, brd, require_vector=not a.no_require_vector)
    print(rep.render())
    if a.json:
        Path(a.json).write_text(json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
