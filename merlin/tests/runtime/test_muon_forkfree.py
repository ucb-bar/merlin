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


def test_forkfree_builds_a_kernel_with_a_relocation_via_the_object_transcode(tmp_path):
    """A kernel that is NOT self-contained (a global table -> a PC-relative relocation) is NOT a flat .text
    re-map, but the driver no longer refuses it: it transcodes the whole OBJECT (preserving the reloc
    records) and links via the fork-free linker, which resolves each relocation at the DERIVED field
    positions. Hermetic (build only): asserts a real relocation survived and an ELF was produced."""
    if not _stock_clang():
        pytest.skip("stock LLVM (clang/objcopy) unavailable")
    from merlin.targetgen.rtl import mlc_bridge
    if not mlc_bridge.isa_encoding_for("radiance"):
        pytest.skip("derived ISA encoding fact not present")
    # a volatile global table forces a PC-relative relocation that survives -O2 into the kernel object
    src = ("static volatile int T[4]={1,2,3,4};\nvolatile int s;\n"
           "int main(void){s=T[s&3];return 0;}")
    elf = muon.compile_kernel_forkfree(src, tmp_path, target="radiance")
    assert _reloc_kernel_object_has_relocations(tmp_path), "no relocation survived; test would be vacuous"
    assert elf.is_file()


def test_multiwarp_graded_on_the_rtl_arc_oracle(tmp_path):
    """The generalizable, vendor-sim-free multi-warp grade: emit the SIMT scaffold ENTIRELY from the derived
    runtime_abi (render_simt_runtime — wspawn/tmc/CSRs all from facts), build it fork-free, run it on the
    target's RTL-derived arc model (mlc cosim_muon, compiled from the RTL via CIRCT-arc), and read the output
    buffer back from memory (no console print, which races across lanes). Proves the emitted multi-warp kernel
    is bit-exact correct against the oracle a real new target would have (its RTL), NOT a vendor sim. Gated on
    the stock toolchain + the derived fact + the compiled arc model."""
    import struct
    from merlin.targetgen.rtl import mlc_bridge
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.arc_oracle_available("radiance")):
        pytest.skip("stock LLVM / derived fact / RTL-arc model not all available")
    model = muon._model_for("radiance")
    body = "for(uint32_t i=wid;i<8;i+=MU_NUM_WARPS)C[i]=(i+1)+10*(i+1);"
    prog = muon.render_simt_runtime(model, num_warps=4, worker_body=body, globals="volatile uint32_t C[8];")
    elf = muon.compile_kernel_forkfree(prog, tmp_path, target="radiance", num_warps=4)
    # C's address from the linked ELF (never hardcoded); read its 8 words back from the arc model's memory.
    from merlin.targetgen.contract.toolchain import mlir_bin
    st = subprocess.run([str(mlir_bin("llvm-objdump")), "-t", str(elf)], capture_output=True, text=True).stdout
    c_addr = next(int(l.split()[0], 16) for l in st.splitlines() if l.rstrip().endswith(" C") or l.rstrip().endswith("\tC"))
    data = muon.run_elf_arc(elf, target="radiance", base=c_addr, length=32)
    assert list(struct.unpack("<8I", data)) == [11, 22, 33, 44, 55, 66, 77, 88]


def test_arc_readback_adapter_fails_closed_without_the_model(tmp_path, monkeypatch):
    """The RTL-arc readback grading adapter (the sim-independent multi-warp oracle) must fail closed with
    MuonUnavailable when the compiled arc model is absent — never fabricate a verdict. Hermetic."""
    from merlin.targetgen import muon_oracles as MO
    monkeypatch.setattr(muon, "arc_oracle_available", lambda target="radiance": False)
    with pytest.raises(muon.MuonUnavailable):
        MO.arc_readback_adapter()({"target": "radiance", "tensors": {}}, "int main(void){return 0;}",
                                  tmp_path, 60)


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


# the SAME vecadd, but reading its addends from a VOLATILE cross-section global. The volatile load cannot be
# constant-folded, so stock clang emits real R_RISCV_PCREL_HI20/LO12 relocations (an auipc+lw pair) into the
# object -- exactly the shape a constant pool or a mu_schedule warp-callback pointer produces. The flat .text
# re-map fails closed on that auipc; the build must transcode the whole OBJECT (preserving the reloc records)
# and let the fork-free linker resolve them at the DERIVED field positions.
_RELOC_KERNEL = r"""
#include <stdint.h>
static volatile uint32_t B[8]={10,20,30,40,50,60,70,80};
static inline uint32_t hid(void){uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}
static inline void pc(char c){*(volatile char*)0xFF080000u=c;}
static inline void ph(uint32_t v){for(int i=7;i>=0;--i){uint32_t n=(v>>(i*4))&0xF;pc(n<10?(char)('0'+n):(char)('a'+n-10));}}
int main(void){
  volatile uint32_t C[8];
  for(int i=0;i<8;i++)C[i]=(uint32_t)(i+1)+B[i];
  if(hid()==0){pc('O');pc('U');pc('T');pc(' ');for(int i=0;i<8;i++){ph(C[i]);pc(' ');}pc('\n');
               pc('D');pc('O');pc('N');pc('E');pc('\n');}
  return 0;
}
"""


def _reloc_kernel_object_has_relocations(work) -> bool:
    """objdump the intermediate rv32 object the driver left behind: assert it really carried an R_RISCV
    relocation (guards against a compiler build that folds the volatile load away and reverts to the fast
    path, which would make the test silently vacuous)."""
    from merlin.targetgen.contract.toolchain import mlir_bin
    import subprocess
    rr = subprocess.run([str(mlir_bin("llvm-objdump")), "-r", str(work / "kernel.o")],
                        capture_output=True, text=True).stdout
    return any("R_RISCV" in ln for ln in rr.splitlines())


def test_forkfree_kernel_with_a_relocation_takes_the_reloc_preserving_path(tmp_path):
    """A kernel whose data reference survives as an R_RISCV relocation (the auipc+lw of a constant pool /
    a mu_schedule warp-callback pointer) cannot be a flat .text re-map. The build must transcode the whole
    OBJECT (preserving the reloc records) and let the fork-free linker resolve every relocation at the
    DERIVED field positions. Proves it is functionally correct on cyclotron -- the P2c relocation-preserving
    path. Gated on stock LLVM + the derived fact + cyclotron."""
    from merlin.targetgen.rtl import mlc_bridge
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    elf = muon.compile_kernel_forkfree(_RELOC_KERNEL, tmp_path, target="radiance")
    assert _reloc_kernel_object_has_relocations(tmp_path), "no relocation survived; test would be vacuous"
    assert elf.is_file()
    console, _cycles, _ = muon.run_elf(str(elf), simulator="cyclotron", timeout=180)
    expected = "0000000b 00000016 00000021 0000002c 00000037 00000042 0000004d 00000058"
    assert expected in console, f"reloc-path kernel produced wrong output:\n{console[-400:]}"
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


# a self-contained HARDWARE-FLOAT kernel: it materializes float operands from integer bit patterns (no
# constant pool -> no relocations), computes in FP, and prints fixed decimals. Exercises the derived-march
# (zfinx) FP path end to end. y = A*2 + 1 for A in {1.5, 3.0} -> 4.000, 7.000.
_FLOAT_KERNEL = r"""
#include <stdint.h>
static inline uint32_t hid(void){uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}
static inline void pc(char c){*(volatile char*)0xFF080000u=c;}
static void pu(uint32_t v){char b[12];int n=0;if(!v){pc('0');return;}while(v){b[n++]=(char)('0'+v%10);v/=10;}while(n)pc(b[--n]);}
static float u2f(uint32_t b){union{uint32_t u;float f;}x; x.u=b; return x.f;}
static void pf(float x){uint32_t ip=(uint32_t)x;float fr=x-(float)ip;float k=u2f(0x447a0000u),h=u2f(0x3f000000u);
  uint32_t fp=(uint32_t)(fr*k+h);pu(ip);pc('.');if(fp<100)pc('0');if(fp<10)pc('0');pu(fp);}
int main(void){
  if(hid()==0){uint32_t a[2];a[0]=0x3fc00000u;a[1]=0x40400000u;float two=u2f(0x40000000u),one=u2f(0x3f800000u);
    for(int i=0;i<2;i++){float y=u2f(a[i])*two+one;pc('V');pc(' ');pf(y);pc('\n');}
    pc('D');pc('O');pc('N');pc('E');pc('\n');}
  return 0;
}
"""


def test_forkfree_float_path_uses_derived_zfinx_march_and_is_correct(tmp_path):
    """The fork-free build DERIVES its -march (incl. the FP mode) from the target's opcode table, so a
    hardware-float kernel compiles to the target's actual FP encoding (Muon = zfinx) with no hand-set flag.
    Proves the float codegen path end to end on cyclotron. Gated on stock LLVM + derived fact + cyclotron."""
    from merlin.targetgen.rtl import mlc_bridge
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    elf = muon.compile_kernel_forkfree(_FLOAT_KERNEL, tmp_path, target="radiance")   # march DERIVED
    console, _cycles, _ = muon.run_elf(str(elf), simulator="cyclotron", timeout=180)
    vlines = [l for l in console.splitlines() if l.startswith("V ")]
    assert vlines == ["V 4.000", "V 7.000"], f"wrong float result: {vlines}\n{console[-300:]}"
    assert "DONE" in console


def test_real_fp32_capsule_grades_forkfree_against_its_golden(tmp_path):
    """The capstone: a REAL corpus capsule (R0_gemm_fp32, a 16x16 fp32 GEMM) grades FORK-FREE end to end.
    The runner-owned harness embeds the capsule's canonical operands, runs a whole-computation kernel
    function through the fork-free driver (stock LLVM + derived zfinx transcode + fork-free link), and the
    cyclotron output matches the independent golden within the capsule's float tolerance -- closing the gap
    where live fork-free coverage was ~0. Gated on stock LLVM + the derived fact + cyclotron."""
    import yaml
    import numpy as np
    from merlin.common.paths import repo_root
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.runtime.backends import muon_harness as MH
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    gfile = repo_root() / "merlin/contract/capsules/radiance/isa/R0_gemm_fp32/golden.yaml"
    if not gfile.is_file():
        pytest.skip("R0_gemm_fp32 capsule not present")
    g = yaml.safe_load(gfile.read_text())
    ins = g["oracle_provenance"]["inputs"]
    y_gold = np.array(g["outputs"]["Y0"], dtype=np.float32)
    # a whole-computation reference kernel FUNCTION (the shape the emit contract asks the agent for):
    # Y0 = A0 @ W, row-major 16x16. arg order = [weight] ++ [lhs] ++ [out].
    kfn = ("void radiance_kernel(float* W, float* A0, float* Y0){"
           "for(int i=0;i<16;i++)for(int j=0;j<16;j++){float a=0.0f;"
           "for(int k=0;k<16;k++)a+=A0[i*16+k]*W[k*16+j];Y0[i*16+j]=a;}}")
    prog = MH.build_program(
        kfn,
        [MH.TensorArg("W", 16, 16, ins["W"]["decoded"], "f32"),
         MH.TensorArg("A0", 16, 16, ins["A0"]["decoded"], "f32")],
        [MH.TensorArg("Y0", 16, 16, [0.0] * 256, "f32")],
        kernel_symbol="radiance_kernel", model=muon._model_for("radiance"))
    elf = muon.compile_kernel_forkfree(prog, tmp_path, target="radiance")
    console, cycles, _ = muon.run_elf(str(elf), simulator="cyclotron", timeout=300)
    outputs, _ = muon.parse_output(console, cycles)
    y = np.array(outputs["Y0"], dtype=np.float32)
    assert y.shape == (16, 16)
    max_err = float(np.max(np.abs(y - y_gold)))
    assert max_err <= 0.03125, f"fork-free GEMM diverges from the golden: max abs err {max_err}"


@pytest.mark.parametrize("capsule", ["R0_gemm_fp32", "R2_gemm_bf16"])
def test_grading_adapter_wraps_a_kernel_function_and_grades_forkfree(tmp_path, capsule):
    """Full LIVE wiring: the cyclotron grading adapter (the SAME entrypoint a real run calls) is handed a
    whole-computation kernel FUNCTION + a cb carrying the canonical operands, wraps it in the runner-owned
    harness, grades fork-free, and matches the capsule's golden -- so live fork-free coverage is real, not
    ~0. Covers both an fp32 (R0) and a bf16 (R2) GEMM: the f32-materialize + fp32-accumulate reference
    matches the pre-rounded bf16 golden within tolerance."""
    import yaml
    import numpy as np
    from merlin.common.paths import repo_root
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen import muon_oracles as MO
    if not (_stock_clang() and mlc_bridge.isa_encoding_for("radiance") and muon.available("cyclotron")):
        pytest.skip("stock LLVM / derived fact / cyclotron not all available")
    cdir = repo_root() / "merlin/contract/capsules/radiance/isa" / capsule
    if not (cdir / "golden.yaml").is_file():
        pytest.skip(f"{capsule} capsule not present")
    g = yaml.safe_load((cdir / "golden.yaml").read_text())
    atol = float(yaml.safe_load((cdir / "capsule.yaml").read_text())["numeric_policy"]["atol"])
    ins = g["oracle_provenance"]["inputs"]
    y_gold = np.array(g["outputs"]["Y0"], dtype=np.float32)
    cb = {"target": "radiance",
          "tensors": {"W": {"shape": [16, 16], "role": "weight"}, "A0": {"shape": [16, 16], "role": "input"},
                      "Y0": {"shape": [16, 16], "role": "output"}},
          "commands": [{"opcode": "MATMUL", "operands": {"weight": "W", "lhs": "A0", "out": "Y0"}}],
          "canonical_inputs": {"W": {"shape": [16, 16], "values": ins["W"]["decoded"]},
                               "A0": {"shape": [16, 16], "values": ins["A0"]["decoded"]}}}
    kfn = ("void radiance_kernel(float* W, float* A0, float* Y0){"
           "for(int i=0;i<16;i++)for(int j=0;j<16;j++){float a=0.0f;"
           "for(int k=0;k<16;k++)a+=A0[i*16+k]*W[k*16+j];Y0[i*16+j]=a;}}")
    res = MO.cyclotron_adapter()(cb, kfn, tmp_path, 300)
    assert res["toolchain"] == "fork-free"
    y = np.array(res["outputs"]["Y0"], dtype=np.float32)
    assert float(np.max(np.abs(y - y_gold))) <= atol
