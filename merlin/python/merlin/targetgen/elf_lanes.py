"""Prove a capsule's NEGATIVE lane contract from the one artifact the operator path already produces.

A capsule that declares ``lanes.forbid: [<accelerator lane>]`` asserts that the compiler left this
family on the host. Until now only the WHOLE-MODEL path could answer a lane question at all, because
only it owns a routing plan and a dynamic dispatch ledger; an operator/model-slice capsule that
declared the assertion had it silently ignored (``LANE_CONTRACT_NOT_EVALUATED``), so the capsule
written to catch a compiler that accelerates an inadmissible family could never catch one.

**Why an ELF scan is sound here, and only here.** The assertion is a NEGATIVE, and absence is exactly
what a complete instruction stream can establish. The linked ELF is that stream: every instruction the
program can execute is in it, whatever spelling the backend used to emit it. That closes the hole the
IR-level decoder leaves open — :func:`merlin.targetgen.capsule_runner.accelerator_lane_violated` reads
the emitted ``llvm.inline_asm`` ops, so an accelerator instruction emitted as a raw ``.word``/``.insn``
datum decodes as silence, and silence read as "the host carried it" is a free pass. Bytes have no
spelling. This is therefore a STRENGTHENING of that gate, not a way around it: it can only see MORE
instructions than the IR decoder, never fewer.

**What it still cannot prove, and does not claim.** Presence of an instruction in the binary is not
proof it EXECUTED, so this evidence can never credit a REQUIRED lane — a required lane judged only from
an ELF stays unmeasured and its capsule stays ``incomplete``. The rung is admissible for the negative
direction only, which is why it is named separately from
:data:`merlin.targetgen.capsule_runner.EXECUTED_LANE_EVIDENCE` (whose members mean "something ran")
rather than being folded into it.

**Scope of the scan: sections the linker marked executable.** Instructions live in ``SHF_EXECINSTR``
sections; data does not. Widening the scan to every allocated section was measured on the ten host-lane
capsules and manufactured 1-6 phantom hits on EVERY one of them — constant pools and string data whose
bytes happen to carry the custom opcode in their low 7 bits. A phantom hit fails a conformant
submission, so the scan stays inside the sections that hold code. An ELF with no readable section
headers is UNMEASURED, never clean.

**No baked encoding.** The custom major opcode is DERIVED per target from that target's own RTL facts
(``funct_decode_table.custom_opcode``), falling back to its capability manifest's encoding block. A
target with neither yields ``UNKNOWN`` and the lane stays unmeasured — a self-hosted-ISA or
command-buffer target does not reach its accelerator through a host custom opcode at all, and guessing
one would decode another device's ISA and report a clean, wrong result.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

#: The linked executable the operator path emits, by the name :mod:`merlin.targetgen.oot_runner` gives it.
PACKAGE_ELF_NAME = "package_kernel.elf"

#: The evidence rung this module produces. NOT a member of ``EXECUTED_LANE_EVIDENCE``: nothing ran. It
#: is stronger than ``routing_plan`` (which records intent) and admissible for a FORBIDDEN lane only.
LINKED_ELF_EVIDENCE = "linked_elf_scan"

#: The rung for "nobody could tell" -- an unreadable ELF, or a target whose accelerator opcode is not
#: derivable. Reported so an unmeasured lane is visible rather than absent.
NO_EVIDENCE = "unmeasured"

#: Fail-closed sentinel for a fact that could not be derived, spelled as the rest of the repo spells it.
UNKNOWN = "UNKNOWN"

# --- ELF constants. These are the ELF format's own, from the gABI -- not a target's ISA facts. -------
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32, _ELFCLASS64 = 1, 2
_ELFDATA2LSB, _ELFDATA2MSB = 1, 2
_SHT_PROGBITS = 1
_SHF_EXECINSTR = 0x4
_EM_RISCV = 243


class ElfUnreadable(Exception):
    """The bytes handed over are not an ELF this module can walk. Always fail closed on it."""


def negative_lane_evidence() -> tuple[str, ...]:
    """Rungs on which a FORBIDDEN lane may be judged: anything that proves execution, plus this
    module's static scan of the complete instruction stream. Derived from the runner's exported
    vocabulary rather than restating it, so the two cannot drift."""
    from .capsule_runner import EXECUTED_LANE_EVIDENCE
    return (*EXECUTED_LANE_EVIDENCE, LINKED_ELF_EVIDENCE)


# --- opcode derivation ------------------------------------------------------------------------
def accelerator_opcode(target: str) -> tuple[int | None, str]:
    """``(major opcode, where it came from)`` for ``target``'s host-issued accelerator instructions.

    Derived, never written down: the RTL-extracted ``funct_decode_table`` first (the same interface the
    trace decoder reads, so the two cannot disagree about whose ISA they are reading), then the
    capability manifest's encoding block for a target whose facts carry no decode table. ``(None, ...)``
    when neither grounds one -- the caller must then report UNKNOWN, never a default.
    """
    try:
        from .rtl.facts import load_facts
        facts = (load_facts(target) or {}).get("facts") or {}
        for itf in facts.get("interfaces") or []:
            if itf.get("name") == "funct_decode_table" and itf.get("custom_opcode") is not None:
                return int(itf["custom_opcode"]), "rtl_facts.funct_decode_table.custom_opcode"
    except Exception:                       # noqa: BLE001 -- an unavailable facts tree derives nothing
        pass
    try:
        from .target_experiment import load_capability_manifest
        enc = (load_capability_manifest(target).encoding or {})
        if enc.get("custom_opcode") is not None:
            return int(enc["custom_opcode"]), "capability_manifest.encoding.custom_opcode"
    except Exception:                       # noqa: BLE001 -- ditto for an unavailable manifest
        pass
    return None, "UNDERIVABLE: no funct_decode_table.custom_opcode in RTL facts and none in the " \
                 "capability manifest's encoding block"


# --- structural ELF walk (no regex, no external tool, no ISA literals) -------------------------
def _endian(blob: bytes) -> str:
    if blob[5] == _ELFDATA2LSB:
        return "<"
    if blob[5] == _ELFDATA2MSB:
        return ">"
    raise ElfUnreadable(f"unknown ELF data encoding {blob[5]}")


def executable_sections(blob: bytes) -> list[tuple[str, int, int, int]]:
    """``(name, file offset, size, virtual address)`` for every PROGBITS section marked executable.

    Parsed from the section-header table structurally (``struct.unpack``), so it holds for any linker
    script and any section naming. Raises :class:`ElfUnreadable` when the header table is absent or
    malformed -- the caller reports UNMEASURED rather than an empty, clean-looking scan.
    """
    if len(blob) < 64 or blob[:4] != _ELF_MAGIC:
        raise ElfUnreadable("not an ELF file (bad magic)")
    end = _endian(blob)
    cls = blob[4]
    if cls == _ELFCLASS64:
        shoff, = struct.unpack_from(end + "Q", blob, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", blob, 0x3A)
        fmt, name_i, type_i, flags_i, addr_i, off_i, size_i = end + "IIQQQQIIQQ", 0, 1, 2, 3, 4, 5
    elif cls == _ELFCLASS32:
        shoff, = struct.unpack_from(end + "I", blob, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", blob, 0x2E)
        fmt, name_i, type_i, flags_i, addr_i, off_i, size_i = end + "IIIIIIIIII", 0, 1, 2, 3, 4, 5
    else:
        raise ElfUnreadable(f"unknown ELF class {cls}")
    if not shoff or not shnum or shstrndx >= shnum:
        raise ElfUnreadable("ELF carries no usable section-header table (stripped?)")
    rows = []
    for i in range(shnum):
        base = shoff + i * shentsize
        if base + shentsize > len(blob):
            raise ElfUnreadable("section-header table runs past end of file")
        rows.append(struct.unpack_from(fmt, blob, base))
    strtab = rows[shstrndx]
    strs = blob[strtab[off_i]:strtab[off_i] + strtab[size_i]]

    def _name(idx: int) -> str:
        stop = strs.find(b"\0", idx)
        return strs[idx:stop if stop >= 0 else len(strs)].decode("utf-8", "replace")

    out: list[tuple[str, int, int, int]] = []
    for r in rows:
        if r[type_i] != _SHT_PROGBITS or not (r[flags_i] & _SHF_EXECINSTR):
            continue
        off, size = r[off_i], r[size_i]
        if off + size > len(blob):
            raise ElfUnreadable("executable section runs past end of file")
        out.append((_name(r[name_i]), off, size, r[addr_i]))
    return out


def _insn_length(word: int) -> int:
    """Bytes occupied by the instruction starting with ``word``, per RISC-V's own length encoding.

    This is the ISA's *variable-length* convention (the low bits say how long the instruction is), not
    an assumption about any target's instructions: it is what makes walking a stream that mixes 16- and
    32-bit instructions possible at all, and it is the same rule an assembler applies.
    """
    if (word & 0x3) != 0x3:
        return 2
    if (word & 0x1F) != 0x1F:
        return 4
    if (word & 0x3F) == 0x1F:
        return 6
    if (word & 0x7F) == 0x3F:
        return 8
    return 10 + 2 * ((word >> 12) & 0x7)


def instruction_words(data: bytes, base_addr: int = 0):
    """Yield ``(address, 32-bit word)`` for every 32-bit-wide instruction in ``data``.

    Walks the stream by the ISA's length encoding rather than striding 4 bytes, because a compressed
    (16-bit) instruction shifts everything after it off 4-byte alignment -- measured: the accelerator
    instructions in a real linked kernel sit at 2-mod-4 addresses, and a 4-byte stride sees none of them.
    """
    i, n = 0, len(data)
    while i + 1 < n:
        half = struct.unpack_from("<H", data, i)[0]
        if (half & 0x3) != 0x3:
            i += 2
            continue
        if i + 4 > n:
            break
        word = struct.unpack_from("<I", data, i)[0]
        step = _insn_length(word)
        if step == 4:
            yield base_addr + i, word
        i += step


@dataclass(frozen=True)
class ElfScan:
    """What a scan of one linked ELF for one target's accelerator opcode found."""

    status: str                          # "measured" | "unmeasured"
    opcode: int | str                    # the derived major opcode, or UNKNOWN
    opcode_source: str
    detail: str
    elf: str | None = None
    sections: tuple[str, ...] = ()
    n_instruction_words: int = 0
    hits: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    def to_dict(self) -> dict:
        d = {"status": self.status, "opcode": self.opcode, "opcode_source": self.opcode_source,
             "detail": self.detail, "elf": self.elf, "sections": list(self.sections),
             "n_instruction_words": self.n_instruction_words, "n_hits": self.n_hits}
        if self.hits:
            # A handful of witnesses, so a violation names WHERE it was found and is checkable by hand.
            d["hits"] = [dict(h) for h in self.hits[:8]]
        return d


def scan_elf_for_accelerator(elf_path, target: str, *, max_hits: int = 64) -> ElfScan:
    """Scan a linked ELF's executable sections for ``target``'s accelerator major opcode.

    Every way this can fail to measure -- an underivable opcode, a missing/unreadable/stripped ELF, a
    non-RISC-V object -- returns ``status="unmeasured"`` with the reason, never a clean zero.
    """
    opcode, source = accelerator_opcode(target)
    path = Path(elf_path)
    if opcode is None:
        return ElfScan(status="unmeasured", opcode=UNKNOWN, opcode_source=source, elf=str(path),
                       detail=("this target's accelerator major opcode is not derivable, so an "
                               "instruction stream cannot be judged against it"))
    if not path.is_file():
        return ElfScan(status="unmeasured", opcode=opcode, opcode_source=source, elf=str(path),
                       detail=f"no linked executable at {path}")
    try:
        blob = path.read_bytes()
        if _endian(blob) != "<":
            # The instruction walk reads little-endian halfwords, which is what the RISC-V instruction
            # stream is. Refusing a big-endian object is the fail-closed answer; decoding it with the
            # wrong byte order would report a confident, meaningless count.
            raise ElfUnreadable("big-endian ELF: this instruction walk reads little-endian words only")
        machine, = struct.unpack_from("<H", blob, 0x12)
        if machine != _EM_RISCV:
            raise ElfUnreadable(f"ELF e_machine {machine} is not the RISC-V host this scan walks")
        sections = executable_sections(blob)
    except (ElfUnreadable, OSError, struct.error, IndexError) as exc:
        return ElfScan(status="unmeasured", opcode=opcode, opcode_source=source, elf=str(path),
                       detail=f"linked executable could not be decoded ({type(exc).__name__}: {exc})")
    if not sections:
        return ElfScan(status="unmeasured", opcode=opcode, opcode_source=source, elf=str(path),
                       detail="the linked executable declares no executable sections to scan")
    hits: list[dict] = []
    words = 0
    for name, off, size, addr in sections:
        for at, word in instruction_words(blob[off:off + size], addr):
            words += 1
            if (word & 0x7F) == opcode and len(hits) < max_hits:
                hits.append({"section": name, "addr": at, "word": word})
    return ElfScan(status="measured", opcode=opcode, opcode_source=source, elf=str(path),
                   sections=tuple(s[0] for s in sections), n_instruction_words=words,
                   hits=tuple(hits),
                   detail=(f"walked {words} instruction word(s) in {len(sections)} executable "
                           f"section(s); {len(hits)} carry major opcode 0x{opcode:02x}"))


# --- the lane report -------------------------------------------------------------------------
def lane_report_from_elf(capsule: dict, elf_path, *, target: str) -> dict | None:
    """A ``lane_report``-shaped verdict for the operator path, judged from the linked ELF.

    Returns ``None`` when the capsule declares no lanes -- same contract as
    :func:`merlin.targetgen.capsule_runner.lane_report`, whose keys and per-lane ``evidence`` mapping
    this mirrors so both ends of the grade read one shape.

    Only the accelerator lane can be judged here, and only negatively. Every other declared lane -- and
    the accelerator lane itself when the scan could not measure -- is reported at rung
    :data:`NO_EVIDENCE`, which :func:`unjudged_lanes` turns into an ``incomplete`` capsule.
    """
    from .capsule_runner import _ACCELERATOR_LANE

    decl = capsule.get("lanes") or {}
    req = [str(x) for x in (decl.get("require") or [])]
    forbid = [str(x) for x in (decl.get("forbid") or [])]
    if not req and not forbid:
        return None
    both = sorted(set(req) & set(forbid))
    if both:
        raise ValueError(f"capsule lanes {both} are both required and forbidden; one of the two "
                         f"assertions can never hold")

    scan = scan_elf_for_accelerator(elf_path, target) if _ACCELERATOR_LANE in forbid else None
    evidence = {ln: NO_EVIDENCE for ln in (*req, *forbid)}
    violated: list[str] = []
    if scan is not None and scan.status == "measured":
        evidence[_ACCELERATOR_LANE] = LINKED_ELF_EVIDENCE
        if scan.n_hits:
            violated.append(_ACCELERATOR_LANE)

    admissible = negative_lane_evidence()
    out: dict = {
        "required": req,
        # Nothing RAN on this path, so no lane is credited as having carried work -- a required lane is
        # unexercised as far as this evidence goes, which is what keeps it from passing.
        "observed": [],
        "unexercised": list(req),
        "evidence": evidence,
        "host_contractions_ran": None,
        "judged_by": LINKED_ELF_EVIDENCE,
    }
    if scan is not None:
        out["elf_scan"] = scan.to_dict()
    if forbid:
        out["forbidden"] = forbid
        out["violated"] = violated
        unmeasured = [ln for ln in forbid if evidence.get(ln) not in admissible]
        if unmeasured:
            out["unmeasured_forbidden"] = unmeasured
    if req:
        out["caveat"] = (
            f"lanes {req} are REQUIRED, and a linked-ELF scan cannot credit them: an instruction present "
            f"in the binary is not one that executed. They stay unmeasured on this path.")
        out["unmeasured_required"] = list(req)
    return out


def unjudged_lanes(report: dict | None, lanes_decl: dict | None) -> list[str]:
    """Declared lanes that ``report`` does not actually settle -- the capsule must not pass while any remain.

    A required lane needs evidence that something RAN (``EXECUTED_LANE_EVIDENCE``); a forbidden lane may
    additionally rest on a linked-ELF scan, which for a negative assertion reads the complete instruction
    stream. Anything else -- a routing plan, an unreadable ELF, an underivable opcode, no report at all --
    is unmeasured, and an unmeasured assertion is not a satisfied one.
    """
    from .capsule_runner import EXECUTED_LANE_EVIDENCE

    decl = lanes_decl or {}
    req = [str(x) for x in (decl.get("require") or [])]
    forbid = [str(x) for x in (decl.get("forbid") or [])]
    if not req and not forbid:
        return []
    if not isinstance(report, dict):
        return sorted({*req, *forbid})
    ev = report.get("evidence")
    if not isinstance(ev, dict):
        return sorted({*req, *forbid})
    carried = set(report.get("observed") or [])
    admissible = negative_lane_evidence()
    out = {ln for ln in req if ev.get(ln) not in EXECUTED_LANE_EVIDENCE or ln not in carried}
    out |= {ln for ln in forbid if ev.get(ln) not in admissible}
    return sorted(out)
