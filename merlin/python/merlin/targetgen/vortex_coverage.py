"""SIMT coverage gate for Vortex: does the compiled kernel actually use the machine?

Numerics alone cannot answer that. A backend that lowers every capsule to a single-threaded scalar loop
can be bit-exact on all 38 capsules while using 1/256th of the hardware, and the whole point of this
benchmark is how well a compiler maps work onto the machine. This module decodes the emitted code and
checks it against the capsule's ``expected_simt_coverage.yaml``.

Two things make it more than a grep.

**It must decode raw instruction words.** `llvm-objdump` prints `<unknown>` for CUSTOM0: stock LLVM's
assembler understands `.insn`, but its disassembler has no CUSTOM0 decode table. So mnemonics are not
available and the words are decoded here.

**It must scan the AGENT's code only.** The runner-owned startup performs KMU dispatch, so it contains
SIMT ops of its own. Measured on the staged `vx_start_min.o`: it alone supplies **TMC**, `WSYNC` and a
CTA CSR read (`cta_entry`). TMC is the anti-scalar-collapse class — "the kernel never enables threads" —
so scanning the linked ELF would satisfy it from the harness and a genuinely scalar kernel would pass
that half of the gate for free. (`WSPAWN` happens to be absent from the *minimal* startup because the
KMU path does not spawn, but the full `vx_start.S` emits it on its non-KMU branches, so relying on that
would be fragile even where it holds.) The unit of measurement is therefore the object file compiled
from the agent's own LLVM IR, before linking.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# --- CUSTOM0 -------------------------------------------------------------------------------------
# funct7 picks the family, funct3 the op within it, so funct3 alone is ambiguous: funct3=0 is `tmc`
# under funct7=0 but `vote_all` under funct7=1. Keying on funct3 only would miscount every cooperative
# op as a warp-control op.
CUSTOM0_OPCODE = 0x0B

_WARP_CTRL = {0: "TMC", 1: "WSPAWN", 2: "SPLIT", 3: "JOIN", 4: "BARRIER", 5: "PRED", 7: "WSYNC"}
_COOPERATIVE = {0: "VOTE_ALL", 1: "VOTE_ANY", 2: "VOTE_UNI", 3: "VOTE_BALLOT",
                4: "SHFL_UP", 5: "SHFL_DOWN", 6: "SHFL_BFLY", 7: "SHFL_IDX"}
CUSTOM0_CLASSES: dict[int, dict[int, str]] = {0: _WARP_CTRL, 1: _COOPERATIVE}

# --- base RISC-V opcodes we classify -------------------------------------------------------------
_LOAD, _STORE, _LOAD_FP, _STORE_FP, _SYSTEM = 0x03, 0x23, 0x07, 0x27, 0x73
# The four R4-type fused multiply-add opcodes (fmadd/fmsub/fnmsub/fnmadd).
_FMA_OPCODES = {0x43, 0x47, 0x4B, 0x4F}

# CTA identity/geometry CSRs (spec sheet §4). Reading one of these is how a kernel learns which
# coordinate it is, so their presence is evidence the kernel is coordinate-aware rather than scalar.
CTA_CSRS = {
    0xCD0: "cta_id", 0xCD1: "cta_rank", 0xCD2: "cta_size",
    0xCD3: "thread_id.x", 0xCD4: "thread_id.y", 0xCD5: "thread_id.z",
    0xCD6: "block_id.x", 0xCD7: "block_id.y", 0xCD8: "block_id.z",
    0xCD9: "block_dim.x", 0xCDA: "block_dim.y", 0xCDB: "block_dim.z",
    0xCDC: "grid_dim.x", 0xCDD: "grid_dim.y", 0xCDE: "grid_dim.z",
    0xCDF: "local_mem_base", 0xCE0: "cluster_size", 0xCE1: "cta_entry",
    0xCC0: "thread_id", 0xCC1: "warp_id", 0xCC2: "core_id",
    0xFC0: "num_threads", 0xFC1: "num_warps", 0xFC2: "num_cores", 0xFC4: "num_barriers",
}

# Classes no static decoder can establish, and which therefore must never gate. A shared-memory access
# uses the SAME load/store instructions as a global one — only the address differs, and the scratchpad
# base arrives at run time in CSR 0xCDF. Distinguishing them needs value tracking the gate does not do,
# so their absence from a report is "not determined", never "not present".
UNDECIDABLE_CLASSES = {"SMEM_LD", "SMEM_ST", "FENCE_SMEM"}


class CoverageError(RuntimeError):
    """The artifact could not be decoded at all (missing/!ELF/no code) — never a silent pass."""


# --- ELF reading -----------------------------------------------------------------------------------

def _elf_sections(blob: bytes) -> list[tuple[str, int, bytes]]:
    """-> [(name, sh_flags, data)] for an ELF32/64 little-endian object or executable.

    A deliberately small reader rather than a pyelftools dependency: only the section headers and their
    contents are needed, and the gate must run wherever the runner does.
    """
    if blob[:4] != b"\x7fELF":
        raise CoverageError("not an ELF file")
    is64 = blob[4] == 2
    if blob[5] != 1:
        raise CoverageError("only little-endian ELF is supported")
    if is64:
        e_shoff, e_shentsize, e_shnum, e_shstrndx = (
            struct.unpack_from("<Q", blob, 0x28)[0], struct.unpack_from("<H", blob, 0x3A)[0],
            struct.unpack_from("<H", blob, 0x3C)[0], struct.unpack_from("<H", blob, 0x3E)[0])
        fmt, name_off, flags_off, off_off, size_off = "<I", 0, 8, 24, 32
    else:
        e_shoff, e_shentsize, e_shnum, e_shstrndx = (
            struct.unpack_from("<I", blob, 0x20)[0], struct.unpack_from("<H", blob, 0x2E)[0],
            struct.unpack_from("<H", blob, 0x30)[0], struct.unpack_from("<H", blob, 0x32)[0])
        fmt, name_off, flags_off, off_off, size_off = "<I", 0, 8, 16, 20

    def hdr(i: int) -> tuple[int, int, int, int]:
        base = e_shoff + i * e_shentsize
        name = struct.unpack_from(fmt, blob, base + name_off)[0]
        if is64:
            flags = struct.unpack_from("<Q", blob, base + flags_off)[0]
            off = struct.unpack_from("<Q", blob, base + off_off)[0]
            size = struct.unpack_from("<Q", blob, base + size_off)[0]
        else:
            flags = struct.unpack_from("<I", blob, base + flags_off)[0]
            off = struct.unpack_from("<I", blob, base + off_off)[0]
            size = struct.unpack_from("<I", blob, base + size_off)[0]
        return name, flags, off, size

    _, _, str_off, str_size = hdr(e_shstrndx)
    strtab = blob[str_off:str_off + str_size]

    out = []
    for i in range(e_shnum):
        name_idx, flags, off, size = hdr(i)
        end = strtab.find(b"\x00", name_idx)
        name = strtab[name_idx:end if end >= 0 else None].decode("utf-8", "replace")
        out.append((name, flags, blob[off:off + size]))
    return out


_SHF_EXECINSTR = 0x4


def executable_bytes(path: str | Path) -> bytes:
    """Concatenated contents of every executable section — the code to decode.

    Section-based rather than symbol-based on purpose: the agent's backend may emit helper functions
    beside its entry, and with `-ffunction-sections` those land in their own `.text.*` sections. Taking
    every executable section of the AGENT's object covers all of them without having to guess names.
    """
    blob = Path(path).read_bytes()
    code = b"".join(data for _, flags, data in _elf_sections(blob) if flags & _SHF_EXECINSTR and data)
    if not code:
        raise CoverageError(f"{path}: no executable section content to decode")
    return code


# --- decoding --------------------------------------------------------------------------------------

def iter_words(code: bytes):
    """Yield (offset, 32-bit word) for each 4-byte instruction, skipping 16-bit compressed ones.

    RISC-V marks a 16-bit instruction by `word & 0b11 != 0b11`. The frozen ABI is rv64imafd with no `C`
    extension, so compressed instructions should not appear — but stepping over them keeps the decoder
    from losing alignment and silently misreading the rest of a section if one ever does.
    """
    i, n = 0, len(code)
    while i + 2 <= n:
        half = int.from_bytes(code[i:i + 2], "little")
        if half & 0b11 != 0b11:
            i += 2
            continue
        if i + 4 > n:
            break
        yield i, int.from_bytes(code[i:i + 4], "little")
        i += 4


def decode(code: bytes) -> dict[str, Any]:
    """Classify every instruction in `code`.

    -> {"classes": {CLASS: count}, "csrs": {name: count}, "illegal_custom0": [...], "n_insns": int}
    """
    classes: dict[str, int] = {}
    csrs: dict[str, int] = {}
    illegal: list[dict[str, Any]] = []
    total = 0

    def bump(d: dict, k) -> None:
        d[k] = d.get(k, 0) + 1

    for off, w in iter_words(code):
        total += 1
        opcode = w & 0x7F
        if opcode == CUSTOM0_OPCODE:
            funct3 = (w >> 12) & 0x7
            funct7 = (w >> 25) & 0x7F
            family = CUSTOM0_CLASSES.get(funct7)
            name = family.get(funct3) if family else None
            if name is None:
                # An encoding the hardware's decoder does not accept. Reported, not silently dropped:
                # it means the backend emitted something the machine will treat as illegal.
                illegal.append({"offset": off, "word": f"0x{w:08x}",
                                "funct3": funct3, "funct7": funct7})
            else:
                bump(classes, name)
        elif opcode in (_LOAD, _LOAD_FP):
            bump(classes, "GMEM_LD")
        elif opcode in (_STORE, _STORE_FP):
            bump(classes, "GMEM_ST")
        elif opcode in _FMA_OPCODES:
            bump(classes, "FMA")
        elif opcode == _SYSTEM:
            funct3 = (w >> 12) & 0x7
            if funct3 in (0b001, 0b010, 0b011, 0b101, 0b110, 0b111):      # a CSR access, not ECALL
                csr = (w >> 20) & 0xFFF
                if csr in CTA_CSRS:
                    bump(classes, "CTA_CSR")
                    bump(csrs, CTA_CSRS[csr])
    return {"classes": classes, "csrs": csrs, "illegal_custom0": illegal, "n_insns": total}


# --- the gate --------------------------------------------------------------------------------------

def check(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Grade `observed` (from :func:`decode`) against one capsule's expected-coverage document.

    Honors the three forms the corpus emits:

    * ``simt_classes`` — conjunctive: every class must appear.
    * ``simt_classes_any_of`` — a list of groups; at least ONE group must appear in full. Used where the
      ISA offers several legal mechanisms (divergence via split/join OR predication) and requiring a
      specific one would gate on a mapping decision instead of on correctness.
    * ``simt_classes_advisory`` — never gates. Reported so a missed optimisation is visible.

    -> {"status": pass|fail, "violations": [...], "advisory_missing": [...], "observed": {...}}
    """
    present = {c for c, n in observed["classes"].items() if n > 0}
    violations: list[str] = []

    for cls in expected.get("simt_classes", []):
        if cls in UNDECIDABLE_CLASSES:
            violations.append(
                f"{cls} is listed as REQUIRED but cannot be established statically "
                f"(shared and global accesses use the same instructions); it must be advisory")
        elif cls not in present:
            violations.append(f"missing required class {cls}")

    groups = expected.get("simt_classes_any_of") or []
    if groups:
        satisfied = [g for g in groups if set(g) <= present]
        if not satisfied:
            violations.append(
                "no any_of group satisfied: need all of one of "
                + " OR ".join("{" + ", ".join(g) + "}" for g in groups)
                + f"; present: {sorted(present) or 'none'}")

    if observed["illegal_custom0"]:
        bad = observed["illegal_custom0"][0]
        violations.append(
            f"{len(observed['illegal_custom0'])} illegal CUSTOM0 encoding(s), first at offset "
            f"{bad['offset']} ({bad['word']}, funct3={bad['funct3']} funct7={bad['funct7']}) — "
            f"the hardware decoder does not accept this")

    advisory = [c for c in expected.get("simt_classes_advisory", [])
                if c not in present and c not in UNDECIDABLE_CLASSES]
    undecidable = [c for c in expected.get("simt_classes_advisory", []) if c in UNDECIDABLE_CLASSES]

    return {"status": "fail" if violations else "pass",
            "violations": violations,
            "advisory_missing": advisory,
            "advisory_undecidable": undecidable,
            "observed": {"classes": dict(sorted(observed["classes"].items())),
                         "cta_csrs": dict(sorted(observed["csrs"].items())),
                         "n_insns": observed["n_insns"]}}


def check_object(expected: dict[str, Any], object_path: str | Path) -> dict[str, Any]:
    """:func:`check` against the agent's compiled object. See the module docstring on why not the ELF."""
    return check(expected, decode(executable_bytes(object_path)))


def expected_for(capsule: dict[str, Any]) -> dict[str, Any] | None:
    """The capsule's `expected_simt_coverage.yaml`, or None if it declares none.

    None means "this capsule is not coverage-gated", which is how a target or corpus without coverage
    documents stays unaffected by the gate being enabled for its compute-unit kind.
    """
    import yaml
    d = capsule.get("__dir__")
    if not d:
        return None
    f = Path(d) / "expected_simt_coverage.yaml"
    if not f.is_file():
        return None
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def gate(capsule: dict[str, Any], llvm_text: str, workdir: str | Path,
         *, timeout: int = 600) -> dict[str, Any] | None:
    """Compile the agent's module and grade its SIMT coverage. -> the report, or None if not gated.

    Deliberately compiles rather than reusing a later artifact, so the gate can run BEFORE any
    simulation: a scalar-collapsed kernel should be rejected without first paying for rtlsim. The
    compile goes through :func:`vortex_oracle.compile_object`, the same path the graded image uses.
    """
    expected = expected_for(capsule)
    if expected is None:
        return None
    from .vortex_oracle import compile_object
    obj = compile_object(llvm_text, Path(workdir), timeout=timeout)
    report = check_object(expected, obj)
    report["capsule"] = capsule.get("name")
    return report
