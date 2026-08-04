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
from merlin.targetgen import isa_disasm


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
    words = _text_words(elf, objcopy, objdump, tmp_path)
    assert words, "no code words extracted"
    recs = isa_disasm.disassemble(m, words)
    illegal = [r for r in recs if r.get("illegal")]
    assert not illegal, f"{len(illegal)}/{len(recs)} words did not decode: {illegal[:4]}"
    # every register field is in the 8-bit file the RTL declares
    for r in recs:
        for f in ("rd", "rs1", "rs2", "rs3"):
            assert r["operands"].get(f, 0) <= 255
