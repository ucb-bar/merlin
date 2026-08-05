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


# a self-contained uniform kernel: derives its inputs in-code (no rodata/relocation), computes
# C[i]=A[i]+B[i] with A[i]=i+1, B[i]=10(i+1) (=> 11,22,...,88), and prints via inline MMIO to the
# cyclotron console, guarded to hart 0 so the byte stream is clean. No mu_schedule, no vendor intrinsics.
_VECADD_KERNEL = r"""
#include <stdint.h>
#define N 8
static inline uint32_t hid(void){uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}
static inline void pc(char c){*(volatile char*)0xFF080000u=c;}
static inline void ph(uint32_t v){for(int i=7;i>=0;--i){uint32_t n=(v>>(i*4))&0xF;pc(n<10?(char)('0'+n):(char)('a'+n-10));}}
int main(void){
  volatile uint32_t A[N],B[N],C[N];
  for(int i=0;i<N;i++){A[i]=(uint32_t)(i+1);B[i]=(uint32_t)(10*(i+1));}
  for(int i=0;i<N;i++)C[i]=A[i]+B[i];
  if(hid()==0){pc('O');pc('U');pc('T');pc(' ');for(int i=0;i<N;i++){ph(C[i]);pc(' ');}pc('\n');
               pc('D');pc('O');pc('N');pc('E');pc('\n');}
  return 0;
}
"""


def test_forkfree_e2e_regenerated_from_source_is_correct_on_cyclotron(tmp_path):
    """L2 of the offline validation ladder, fully REPRODUCIBLE: build a kernel with the fork-free driver,
    REGENERATING the entire BSP (boot + shims) from the shipped sources — no transient/committed binaries —
    then run it on cyclotron and assert the correct computed result. This exercises the exact infra a live
    run uses, so a live run never debugs the pipeline. Gated on the stock toolchain + the derived fact +
    cyclotron being available."""
    from merlin.targetgen.rtl import mlc_bridge
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    elf = muon.compile_kernel_forkfree(_VECADD_KERNEL, tmp_path, target="radiance")   # BSP regenerated
    assert elf.is_file()
    console, cycles, _ = muon.run_elf(str(elf), simulator="cyclotron", timeout=180)
    expected = "0000000b 00000016 00000021 0000002c 00000037 00000042 0000004d 00000058"
    assert expected in console, f"fork-free kernel produced wrong output:\n{console[-400:]}"
    assert "DONE" in console


# a self-contained, relocation-free, single-warp kernel emitting the OUT/DONE grading protocol in decimal.
_PROTOCOL_KERNEL = r"""
#include <stdint.h>
static inline uint32_t hid(void){uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}
static inline void pc(char c){*(volatile char*)0xFF080000u=c;}
static void pd(uint32_t v){char b[10];int n=0;if(!v){pc('0');return;}while(v){b[n++]=(char)('0'+v%10);v/=10;}while(n)pc(b[--n]);}
int main(void){
  volatile uint32_t A[8],B[8],C[8];
  for(int i=0;i<8;i++){A[i]=(uint32_t)(i+1);B[i]=(uint32_t)(10*(i+1));}
  for(int i=0;i<8;i++)C[i]=A[i]+B[i];
  if(hid()==0){pc('O');pc('U');pc('T');pc(' ');pc('Y');pc(' ');pc('1');pc(' ');pc('8');
    for(int i=0;i<8;i++){pc(' ');pd(C[i]);}pc('\n');pc('D');pc('O');pc('N');pc('E');pc('\n');}
  return 0;
}
"""


def test_grading_oracle_routes_forkfree_and_stamps_the_toolchain(tmp_path):
    """P3 live-wiring: the capsule grading oracle (the SAME adapter a real run calls) compiles the emitted
    kernel via the fork-free thesis path FIRST, records ``toolchain`` so the experiment measures fork-free
    coverage (never a hidden fork fallback), runs cyclotron, and parses the graded outputs. Proves the infra
    the live run uses works offline. Gated on the stock toolchain + the derived fact + cyclotron."""
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen import muon_oracles as MO
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    res = MO.cyclotron_adapter()({"target": "radiance"}, _PROTOCOL_KERNEL, tmp_path, 180)
    assert res["toolchain"] == "fork-free", f"expected the thesis path, got {res['toolchain']!r}"
    assert res["outputs"] == {"Y": [[11, 22, 33, 44, 55, 66, 77, 88]]}, res["outputs"]
    assert res["cycles"] and res["cycles"] > 0
