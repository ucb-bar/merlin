"""Hermetic tests for the runner-owned self-contained-C harness generator (muon_harness.build_program).

No toolchain / simulator needed — these assert the generated C has the relocation-free shape the fork-free
driver requires (element-wise volatile-stack operand fills, no aggregate initializers, an inlined kernel,
and the OUT/DONE protocol). The end-to-end cyclotron proof on a real capsule lives in the gated
test_muon_forkfree suite."""
from __future__ import annotations

from merlin.runtime.backends.muon_harness import build_program, TensorArg


_KFN = ("void radiance_kernel(float* W, float* A0, float* Y0){"
        "for(int i=0;i<2;i++)for(int j=0;j<2;j++){float a=0.0f;"
        "for(int k=0;k<2;k++)a+=A0[i*2+k]*W[k*2+j];Y0[i*2+j]=a;}}")


def _prog():
    return build_program(
        _KFN,
        [TensorArg("W", 2, 2, [1.0, 2.0, 3.0, 4.0], "f32"),
         TensorArg("A0", 2, 2, [0.5, 0.25, 0.125, 1.0], "f32")],
        [TensorArg("Y0", 2, 2, [0.0] * 4, "f32")],
        kernel_symbol="radiance_kernel")


def test_program_embeds_operands_element_wise_relocation_free():
    p = _prog()
    # each operand element is an individual immediate store into a VOLATILE STACK array (SP-relative,
    # un-coalesceable) — never an aggregate initializer (which would emit a PC-relative .rodata memcpy).
    assert "volatile uint32_t _in_W[4];" in p
    assert "_in_W[0]=0x3f800000u;" in p and "_in_W[3]=0x40800000u;" in p   # 1.0, 4.0 as IEEE bits
    assert "_in_A0[0]=0x3f000000u;" in p                                   # 0.5
    assert "static uint32_t" not in p                                      # nothing in .bss/.rodata
    assert "={" not in p                                                   # no aggregate array initializer


def test_program_inlines_the_kernel_and_prints_the_protocol():
    p = _prog()
    assert "always_inline" in p                       # kernel inlined -> no R_RISCV_CALL relocation
    assert "radiance_kernel((float*)_in_W, (float*)_in_A0, (float*)_out_Y0);" in p
    assert '_ps("OUT Y0 2 2");' in p                   # OUT <name> <rows> <cols> ...
    assert '_ps("DONE\\n");' in p
    assert "_pf(" in p                                 # float outputs printed as fixed decimals
    assert "if(_hid()!=0)return 0;" in p               # single-thread (hart-0) guard


def test_integer_output_uses_base10_not_float_print():
    p = build_program("void k(int32_t* X, int32_t* Y){Y[0]=X[0];}",
                      [TensorArg("X", 1, 1, [7], "i32")],
                      [TensorArg("Y", 1, 1, [0], "i32")], kernel_symbol="k")
    assert "_in_X[0]=0x00000007u;" in p
    assert "k((int32_t*)_in_X, (int32_t*)_out_Y);" in p
    assert "_pu(_out_Y[i]);" in p and "_pf(" not in p.split("int main")[1]   # integer path prints base-10
