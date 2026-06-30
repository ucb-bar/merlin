"""R2: the structured asm decoder recovers real vtype/structure (no regex guessing).

Core tests run with no toolchain (synthetic objdump text). An optional integration test decodes a
real built object when present.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

from pathlib import Path

import pytest

from merlin.kernels.decode import objdump, rvv

# A canonical llvm-objdump -d --no-aliases snippet (riscv64 RVV), mixed vtypes + a loop back-edge.
_SNIPPET = """\

model.o:\tfile format elf64-littleriscv

Disassembly of section .text:

0000000000000000 <kernel>:
       0: cd147057     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 020d6407     \tvle32.v\tv8, (s10)
       8: cd70f057     \tvsetivli\tzero, 0x1, e32, mf2, ta, ma
       c: 5e042457     \tvfmul.vv\tv8, v8, v9
      10: 02b50557     \tvfadd.vv\tv10, v10, v11
      14: 00d70063     \tbne\ta4, a3, 0x0
"""


def test_parse_vtype_reads_explicit_operands():
    assert rvv._parse_vtype(["zero", "0x8", "e32", "m2", "ta", "ma"]) == rvv.VType(32, 2.0, "ta", "ma")
    assert rvv._parse_vtype(["zero", "0x1", "e32", "mf2", "ta", "ma"]).lmul == 0.5
    assert rvv._parse_vtype(["zero", "0x4", "e64", "m4", "tu", "mu"]) == rvv.VType(64, 4.0, "tu", "mu")


def test_tokenize_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: _SNIPPET)
    raws = objdump.tokenize("ignored.o")
    mnem = [r.mnemonic for r in raws]
    assert mnem == ["vsetivli", "vle32.v", "vsetivli", "vfmul.vv", "vfadd.vv", "bne"]
    assert raws[0].operands[2:4] == ["e32", "m2"]   # comma-split operands, not regex
    assert raws[-1].addr == 0x14 and raws[-1].operands[-1] == "0x0"


def test_decode_tracks_effective_vtype(tmp_path, monkeypatch):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: _SNIPPET)
    s = rvv.decode("ignored.o")
    # vfmul ran under the mf2 vtype that was set just before it (state tracking).
    vfmul = next(i for i in s.insns if i.raw.mnemonic == "vfmul.vv")
    assert vfmul.vtype.sew == 32 and vfmul.vtype.lmul == 0.5
    # vle32 ran under the earlier m2 vtype.
    vle = next(i for i in s.insns if i.raw.mnemonic == "vle32.v")
    assert vle.vtype.lmul == 2.0
    assert s.has_loop()                       # bne to 0x0 < 0x14 = back-edge
    assert s.count("vfmacc") == 0 and s.count("vfmul") == 1   # the fused-MAC gap, structurally


@pytest.mark.skipif(
    not (repo_root()
         / "runs/rvv_experiment/hand_v0_matmul_f32_64x64x64/generated/model.o").is_file(),
    reason="built matmul object not present")
def test_decode_real_object():
    obj = (repo_root()
           / "runs/rvv_experiment/hand_v0_matmul_f32_64x64x64/generated/model.o")
    s = rvv.decode(obj)
    assert sum(1 for i in s.insns if i.is_vector) > 0
    vt = s.vtype_histogram()
    assert any(k.startswith("e32") for k in vt)   # recovered real vtype, not guessed
