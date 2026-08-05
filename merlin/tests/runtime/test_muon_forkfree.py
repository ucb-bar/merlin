"""The productionized FORK-FREE build driver (muon.compile_kernel_forkfree): stock clang -> derived
transcode -> stock llvm-mc -> fork-free link. The full end-to-end (kernel + vendored BSP -> cyclotron
correct result) is proven byte-identical to the fork baseline in the session record; here we lock the
hermetic fail-closed contracts that do not need the vendored BSP or the simulator."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from merlin.runtime.backends import muon


def _stock_clang() -> bool:
    from merlin.targetgen.contract.toolchain import mlir_bin
    return mlir_bin("clang").is_file() and mlir_bin("llvm-objcopy").is_file()


def test_forkfree_fails_closed_on_a_kernel_with_relocations(tmp_path):
    """A kernel that is NOT self-contained (a global table -> a relocation) cannot be a pure field re-map;
    the fork-free driver must refuse rather than emit a mis-encoded kernel."""
    if not _stock_clang():
        pytest.skip("stock LLVM (clang/objcopy) unavailable")
    from merlin.targetgen.rtl import mlc_bridge
    if not mlc_bridge.isa_encoding_for("radiance"):
        pytest.skip("derived ISA encoding fact not present")
    # a global lookup table forces a PC-relative relocation into the kernel .text
    src = "static const int T[4]={1,2,3,4};\nvolatile int s;\nint main(void){s=T[s&3];return 0;}"
    with pytest.raises(muon.MuonError, match="relocation"):
        muon.compile_kernel_forkfree(src, tmp_path, bsp_objs=[], target="radiance")


def test_forkfree_needs_the_derived_fact(tmp_path, monkeypatch):
    """Without a derived ISA encoding fact for the target, the driver fails closed (never guesses)."""
    if not _stock_clang():
        pytest.skip("stock LLVM unavailable")
    from merlin.targetgen.rtl import mlc_bridge
    monkeypatch.setattr(mlc_bridge, "isa_encoding_for", lambda t: None)
    with pytest.raises(muon.MuonUnavailable):
        muon.compile_kernel_forkfree("int main(void){return 0;}", tmp_path, bsp_objs=[], target="radiance")


def test_forkfree_e2e_when_bsp_present(tmp_path):
    """If the vendored BSP objects + stock tools are present, the driver builds a valid Muon kernel whose
    every instruction decodes cleanly (the compile+transcode+assemble+link chain is coherent). The cyclotron
    functional check is exercised in the session proof (byte-identical to the fork baseline)."""
    import os
    if not _stock_clang():
        pytest.skip("stock LLVM unavailable")
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen.isa_model import isa_model_from_encoding
    from merlin.targetgen import isa_disasm
    from merlin.targetgen.contract.toolchain import mlir_bin
    fact = mlc_bridge.isa_encoding_for("radiance")
    bsp_root = os.environ.get("MERLIN_MUON_BSP_OBJS")   # dir holding mu_start.rebuilt.o / murt.o / tohost.o
    if not (fact and bsp_root):
        pytest.skip("no derived fact / MERLIN_MUON_BSP_OBJS not set (vendored BSP objects)")
    bsp = [Path(bsp_root) / n for n in ("mu_start.rebuilt.o", "murt.o", "tohost.o")]
    if not all(b.is_file() for b in bsp):
        pytest.skip("vendored BSP objects not all present")
    kc = ("static inline void pc(char c){*(volatile char*)0xFF080000u=c;}"
          "int main(void){volatile int a=3,b=4;int c=a+b;pc('0'+c);return 0;}")
    elf = muon.compile_kernel_forkfree(kc, tmp_path, bsp, target="radiance")
    assert elf.is_file()
    # the kernel section decodes clean under the derived model
    m = isa_model_from_encoding("radiance", fact)
    binf = tmp_path / "t.bin"
    subprocess.run([str(mlir_bin("llvm-objcopy")), "-O", "binary", "--only-section=.text",
                    str(elf), str(binf)], capture_output=True)
    import struct
    words = [w for (w,) in struct.iter_unpack("<Q", binf.read_bytes())]
    illegal = [r for r in isa_disasm.disassemble(m, words) if r.get("illegal")]
    assert not illegal, f"{len(illegal)} undecodable words in the fork-free ELF"
