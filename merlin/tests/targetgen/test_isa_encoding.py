"""The mlc-derived instruction-ENCODING consumption path: a fixed-format IsaModel built from an
``isa_encoding`` fact (field bit-ranges + opcode table recovered from a target's RTL decoder) decodes a
word stream by field extraction. The hermetic tests use a tiny synthetic fact; the corpus test decodes a
real Muon reference ELF end-to-end (gated on the radiance-kernels corpus + LLVM tools being present)."""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen.isa_model import isa_model_from_encoding
from merlin.targetgen import isa_disasm, isa_asm


# a minimal fixed-format fact: opcode[2:0] selects, rd[7:4] + rs1[11:8] operands, 16-bit words
SYNTH_FACT = {
    "inst_width": 16,
    "fields": {"opcode": [2, 0], "rd": [7, 4], "rs1": [11, 8]},
    "opcodes": {"add": 1, "load": 2},
    "provenance": {"field.opcode": "rtl:src"},
}


def test_isa_model_from_encoding_is_fixed_format():
    m = isa_model_from_encoding("synth", SYNTH_FACT)
    assert m.is_empty()          # by_mnemonic is empty (this is not the probe path) ...
    assert m.is_fixed_format()   # ... but it IS a fixed-format model (field layout + opcode table)
    assert m.inst_width == 16
    assert m.field_layout["opcode"] == (2, 0)
    assert m.opcode_table == {"add": 1, "load": 2}


def test_fixed_format_decode_extracts_fields_and_flags_illegal():
    m = isa_model_from_encoding("synth", SYNTH_FACT)
    # word A: opcode=1(add), rd=x3, rs1=x5  -> 0b0101_0011_0001 = 0x531
    # word B: opcode=2(load), rd=x1, rs1=x0 -> 0b0000_0001_0010 = 0x012
    # word C: opcode=7 (not in table)       -> 0x007  => illegal
    recs = isa_disasm.disassemble(m, [0x531, 0x012, 0x007])
    assert recs[0]["mnemonic"] == "add" and recs[0]["operands"] == {"rd": 3, "rs1": 5}
    assert recs[1]["mnemonic"] == "load" and recs[1]["operands"] == {"rd": 1, "rs1": 0}
    assert recs[2].get("illegal") is True and recs[2]["mnemonic"] is None
    assert recs[0]["word"] == "0x0531"   # hex width follows inst_width (16b -> 4 nibbles)


def test_empty_fact_yields_empty_model():
    m = isa_model_from_encoding("synth", {"fields": {}, "opcodes": {}})
    assert m.is_empty() and not m.is_fixed_format()


def test_carved_opcode_groups_extension_variants():
    # opcode field is 3 bits but the table has a 4-bit value (an extension variant sharing the low 3 bits):
    # base=0b011 (val 3), variant=0b1011 (val 11) both decode from opcode[2:0]==3 -> ambiguous, not illegal.
    fact = {"inst_width": 8, "fields": {"opcode": [2, 0]}, "opcodes": {"base": 3, "variant": 11}}
    m = isa_model_from_encoding("synth", fact)
    rec = isa_disasm.disassemble(m, [0x3])[0]
    assert rec["mnemonic"] in ("base", "variant")
    assert set(rec.get("ambiguous", [])) == {"base", "variant"}


def test_fixed_format_encode_decode_round_trip():
    m = isa_model_from_encoding("synth", SYNTH_FACT)
    w = isa_asm.assemble_fixed(m, "add", {"rd": 3, "rs1": 5})
    assert w == 0x531
    rec = isa_disasm.disassemble(m, [w])[0]
    assert rec["mnemonic"] == "add" and rec["operands"] == {"rd": 3, "rs1": 5}
    # opcode-table value placed at the opcode low bit fills opcode + any carved extension above it
    fact = {"inst_width": 8, "fields": {"opcode": [2, 0], "ext2": [4, 3]}, "opcodes": {"nu": 0b11011}}
    mm = isa_model_from_encoding("synth", fact)
    assert isa_asm.assemble_fixed(mm, "nu") == 0b11011   # opcode[2:0]=0b011, ext2[4:3]=0b11


def test_assemble_fixed_refuses_bad_input():
    m = isa_model_from_encoding("synth", SYNTH_FACT)
    with pytest.raises(isa_asm.AssembleError):
        isa_asm.assemble_fixed(m, "nope", {})            # unknown opcode
    with pytest.raises(isa_asm.AssembleError):
        isa_asm.assemble_fixed(m, "add", {"rd": 16})     # rd is 4-bit, 16 does not fit
    with pytest.raises(isa_asm.AssembleError):
        isa_asm.assemble_fixed(m, "add", {"zzz": 1})     # unknown field


def test_to_data_lines_width():
    assert isa_asm.to_data_lines([0x531], 16).strip() == ".word 0x00000531"
    assert isa_asm.to_data_lines([0xdeadbeefcafe], 64).strip() == ".quad 0x0000deadbeefcafe"


# a fixed-format fact with an address-space selector: opcode[2:0], ext2[4:3], rd[8:5]; 16-bit
MEM_FACT = {
    "inst_width": 16,
    "fields": {"opcode": [2, 0], "ext2": [4, 3], "rd": [8, 5]},
    "opcodes": {"LOAD": 2, "STORE": 1},
    "address_spaces": {"global": 0, "shared": 1},
    "address_space_field": "ext2",
}


def test_encode_mem_op_sets_the_space_selector():
    m = isa_model_from_encoding("synth", MEM_FACT)
    assert m.address_spaces == {"global": 0, "shared": 1} and m.address_space_field == "ext2"
    g = isa_asm.encode_mem_op(m, "LOAD", "global", {"rd": 3})
    s = isa_asm.encode_mem_op(m, "LOAD", "shared", {"rd": 3})
    assert isa_disasm.disassemble(m, [g])[0]["operands"]["ext2"] == 0
    assert isa_disasm.disassemble(m, [s])[0]["operands"]["ext2"] == 1
    # a shared op differs from the global op only in the derived ext2 bits
    assert (s ^ g) == (1 << 3)
    with pytest.raises(isa_asm.AssembleError):
        isa_asm.encode_mem_op(m, "LOAD", "nowhere", {})          # unknown space
    with pytest.raises(isa_asm.AssembleError):
        isa_asm.encode_mem_op(m, "LOAD", "global", {"ext2": 1})  # operand conflicts with the space


def test_lint_flags_undefined_address_space():
    from merlin.targetgen import isa_lint
    m = isa_model_from_encoding("synth", MEM_FACT)
    good = isa_asm.encode_mem_op(m, "LOAD", "shared", {"rd": 1})
    bad = isa_asm.assemble_fixed(m, "LOAD", {"ext2": 2, "rd": 1})   # ext2=2 is not a defined space (0/1)
    findings = isa_lint.lint(m, [good, bad])
    rules = {(f["rule"], f.get("index")) for f in findings}
    assert ("undefined_address_space", 1) in rules
    assert not any(f["rule"] == "undefined_address_space" and f.get("index") == 0 for f in findings)


def test_lint_fixed_flags_illegal_opcode():
    from merlin.targetgen import isa_lint
    m = isa_model_from_encoding("synth", MEM_FACT)
    findings = isa_lint.lint(m, [0b111])   # opcode 7 not in {LOAD=2, STORE=1}
    assert any(f["rule"] == "illegal_opcode" for f in findings)


# a 64-bit fixed-format fact (Muon-shaped: opcode[6:0], ext2[8:7], rd[16:9], rs1[27:20]) for the full
# agent-facing chain: encode -> .quad -> stock llvm-mc -> read back -> disasm -> lint.
WIDE_FACT = {
    "inst_width": 64,
    "fields": {"opcode": [6, 0], "ext2": [8, 7], "rd": [16, 9], "rs1": [27, 20]},
    "opcodes": {"LOAD": 3, "STORE": 0x23, "OP": 0x33},
    "address_spaces": {"global": 0, "shared": 1},
    "address_space_field": "ext2",
}


def test_agent_chain_encode_assemble_disasm_lint(tmp_path):
    """The exact chain the ISA-tools broker runs for a fixed-format target: assemble the emitted words with
    stock llvm-mc grouped at the target width, then disassemble + lint. Proves a 64-bit kernel round-trips
    through the real assembler (not just in-memory)."""
    from merlin.targetgen.contract.toolchain import mlir_bin
    from merlin.targetgen.program_oracle import _assemble_kernel_words
    from merlin.targetgen import isa_lint
    if not (mlir_bin("llvm-mc").is_file() and mlir_bin("llvm-objcopy").is_file()):
        pytest.skip("prebuilt stock LLVM (llvm-mc/llvm-objcopy) unavailable")
    m = isa_model_from_encoding("synth", WIDE_FACT)
    words = [isa_asm.encode_mem_op(m, "LOAD", "shared", {"rd": 5, "rs1": 6}),
             isa_asm.assemble_fixed(m, "OP", {"rd": 1, "rs1": 2}),
             isa_asm.encode_mem_op(m, "STORE", "global", {"rd": 0, "rs1": 3})]
    ks = tmp_path / "kernel.S"
    ks.write_text(".text\n" + isa_asm.to_data_lines(words, 64))
    got = _assemble_kernel_words(ks, tmp_path, inst_width=64)
    assert got == words, f"assemble round-trip failed: {[hex(w) for w in got]} vs {[hex(w) for w in words]}"
    recs = isa_disasm.disassemble(m, got)
    assert [r["mnemonic"] for r in recs] == ["LOAD", "OP", "STORE"]
    assert recs[0]["operands"]["ext2"] == 1 and recs[2]["operands"]["ext2"] == 0
    assert not isa_lint.lint(m, got)   # all valid -> clean


# --- corpus integration: the real RTL-derived Muon fact decodes a real reference ELF -----------------

def _llvm(tool: str) -> str | None:
    from merlin.common.paths import repo_root
    p = repo_root() / "third_party" / "llvm-install" / "bin" / tool
    return str(p) if p.is_file() else None


def _text_words(elf: str, objcopy: str, objdump: str, tmp: Path) -> list[int]:
    """Extract the executable (TEXT) sections of a Muon ELF as little-endian 64-bit words."""
    hdr = subprocess.run([objdump, "-h", elf], capture_output=True, text=True).stdout
    secs = [p[1] for p in (l.split() for l in hdr.splitlines())
            if len(p) >= 5 and p[0].isdigit() and p[-1] == "TEXT"]
    words: list[int] = []
    for sec in secs:
        b = tmp / f"{sec.strip('.')}.bin"
        subprocess.run([objcopy, "-O", "binary", f"--only-section={sec}", elf, str(b)],
                       capture_output=True)
        raw = b.read_bytes() if b.exists() else b""
        words += [struct.unpack_from("<Q", raw, i * 8)[0] for i in range(len(raw) // 8)]
    return words


def test_muon_rtl_fact_decodes_reference_elf(tmp_path):
    """The mlc isa_encoding fact for the SIMT target, consumed through merlin, decodes a real Muon ELF with
    zero illegal instructions — the end-to-end 'RTL-derived encoding reads real HW code' regression."""
    from merlin.targetgen.rtl import mlc_bridge
    fact = mlc_bridge.isa_encoding_for("radiance")
    if not fact:
        pytest.skip("mlc isa_encoding fact (muon_isa.json) not present")
    objcopy, objdump = _llvm("llvm-objcopy"), _llvm("llvm-objdump")
    if not (objcopy and objdump):
        pytest.skip("prebuilt LLVM (objcopy/objdump) unavailable")
    from merlin.common.paths import env as _env
    corpus = _env("MERLIN_RADIANCE_KERNELS")
    elf = None
    for cand in (f"{corpus}/kernels/_compgen_post_refactor/kernel.radiance.elf" if corpus else "",):
        if cand and Path(cand).is_file():
            elf = cand
    if not elf:
        pytest.skip("radiance-kernels reference ELF not present")

    m = isa_model_from_encoding("radiance", fact)
    assert m.is_fixed_format() and m.inst_width == 64
    # the derived address spaces (global/shared) came through with their selector field
    assert m.address_spaces and m.address_space_field in m.field_layout
    words = _text_words(elf, objcopy, objdump, tmp_path)
    assert words, "no code words extracted"
    recs = isa_disasm.disassemble(m, words)
    illegal = [r for r in recs if r.get("illegal")]
    assert not illegal, f"{len(illegal)}/{len(recs)} words did not decode: {illegal[:4]}"
    # every register field is in the 8-bit file the RTL declares
    for r in recs:
        for f in ("rd", "rs1", "rs2", "rs3"):
            assert r["operands"].get(f, 0) <= 255

    # ENCODE round-trip: re-encoding the decoded physical fields reproduces exactly those bits of every
    # real instruction (the imm/alias overlaps are excluded; opcode is set by the mnemonic).
    phys = [f for f in m.field_layout if f not in ("opcode", "csrimm", "imm24")]
    covered = 0
    for name in ["opcode"] + phys:
        hi, lo = m.field_layout[name]
        covered |= ((1 << (hi - lo + 1)) - 1) << lo
    for w, r in zip(words, recs):
        mnem = r["mnemonic"]
        ops = {f: r["operands"][f] for f in phys}
        enc = isa_asm.assemble_fixed(m, mnem, ops)
        assert enc == (w & covered), f"re-encode of {mnem} differs on modeled bits: {enc:#018x} vs {w & covered:#018x}"


def test_muon_fact_decodes_whole_corpus(tmp_path):
    """The RTL-derived layout decodes EVERY built reference ELF in the radiance-kernels corpus with zero
    illegal instructions — the broad regression that the encoding fact generalizes across the real kernel
    set, not just one example."""
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.common.paths import env as _env
    fact = mlc_bridge.isa_encoding_for("radiance")
    objcopy, objdump = _llvm("llvm-objcopy"), _llvm("llvm-objdump")
    corpus = _env("MERLIN_RADIANCE_KERNELS")
    if not (fact and objcopy and objdump and corpus):
        pytest.skip("mlc fact / LLVM tools / radiance-kernels corpus not all present")
    elfs = sorted(Path(f"{corpus}/kernels").glob("*/kernel.radiance.elf"))
    if not elfs:
        pytest.skip("no built reference ELFs in the corpus")
    m = isa_model_from_encoding("radiance", fact)
    n_words, n_elfs, dirty = 0, 0, []
    for elf in elfs:
        words = _text_words(str(elf), objcopy, objdump, tmp_path)
        if not words:
            continue
        n_elfs += 1
        n_words += len(words)
        recs = isa_disasm.disassemble(m, words)
        bad = [r for r in recs if r.get("illegal")]
        if bad:
            dirty.append((elf.parent.name, len(bad), len(recs)))
    assert n_elfs >= 5, f"expected many built ELFs, saw {n_elfs}"
    assert not dirty, f"{len(dirty)} ELFs had undecodable words: {dirty[:6]}"
    print(f"\n[corpus] decoded {n_words} instructions across {n_elfs} reference ELFs, 0 illegal")
