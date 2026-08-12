"""Hermetic tests for the runner-owned self-contained-C harness generator (muon_harness.build_program).

No toolchain / simulator needed — these assert the generated C has the relocation-free shape the fork-free
driver requires (element-wise volatile-stack operand fills, no aggregate initializers, an inlined kernel,
and the OUT/DONE protocol). The end-to-end cyclotron proof on a real capsule lives in the gated
test_muon_forkfree suite."""
from __future__ import annotations

from merlin.runtime.backends.base import get_backend
_mh = get_backend("muon").muon_harness          # evicted SIMT backend, resolved via plugin discovery
build_program, program_from_cb, TensorArg = _mh.build_program, _mh.program_from_cb, _mh.TensorArg
from merlin.targetgen.isa_model import IsaModel

# a minimal derived-ABI model: the harness reads ONLY the hart-id CSR and the console-MMIO aperture from it
# (both from runtime_abi) — no hardcoded number. A synthetic fact suffices for these hermetic shape tests.
_MODEL = IsaModel(target="synth", runtime_abi={"special_csrs": {"mhartid": 0xF14},
                                               "apertures": {"console_mmio": 0xFF080000}})


_KFN = ("void radiance_kernel(float* W, float* A0, float* Y0){"
        "for(int i=0;i<2;i++)for(int j=0;j<2;j++){float a=0.0f;"
        "for(int k=0;k<2;k++)a+=A0[i*2+k]*W[k*2+j];Y0[i*2+j]=a;}}")


def _prog():
    return build_program(
        _KFN,
        [TensorArg("W", 2, 2, [1.0, 2.0, 3.0, 4.0], "f32"),
         TensorArg("A0", 2, 2, [0.5, 0.25, 0.125, 1.0], "f32")],
        [TensorArg("Y0", 2, 2, [0.0] * 4, "f32")],
        kernel_symbol="radiance_kernel", model=_MODEL)


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
                      [TensorArg("Y", 1, 1, [0], "i32")], kernel_symbol="k", model=_MODEL)
    assert "_in_X[0]=0x00000007u;" in p
    assert "k((int32_t*)_in_X, (int32_t*)_out_Y);" in p
    assert "_pu(_out_Y[i]);" in p and "_pf(" not in p.split("int main")[1]   # integer path prints base-10


# a command buffer the runner produces (RES_PACK + MATMUL_RESIDENT + COMMIT) + the operand values it
# attaches at grade time (canonical_inputs = the independent golden's decoded operands).
_CB = {
    "tensors": {"W": {"shape": [2, 2], "role": "weight"}, "A0": {"shape": [2, 2], "role": "input"}},
    "commands": [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
                 {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
                 {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}}],
    "canonical_inputs": {"W": {"shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]},
                         "A0": {"shape": [2, 2], "values": [0.5, 0.25, 0.125, 1.0]}},
}
_FN = "void radiance_kernel(float* W, float* A0, float* Y0){Y0[0]=A0[0]*W[0];}"


def test_program_from_cb_derives_abi_order_and_embeds_operands():
    p = program_from_cb(_CB, _FN, _MODEL)
    assert p is not None
    # ABI order [weight] ++ [lhs] ++ [output] -> radiance_kernel(W, A0, Y0); output name = COMMIT dst (Y0)
    assert "radiance_kernel((float*)_in_W, (float*)_in_A0, (float*)_out_Y0);" in p
    assert "_in_W[0]=0x3f800000u;" in p and "_in_A0[0]=0x3f000000u;" in p   # canonical 1.0, 0.5
    assert '_ps("OUT Y0 2 2");' in p


def test_program_from_cb_passes_through_a_full_program():
    # an artifact that is already a full program (has main) is graded directly, not re-wrapped.
    assert program_from_cb(_CB, "int main(void){return 0;}", _MODEL) is None


def test_program_from_cb_materializes_deterministically_without_canonical():
    # No canonical operands -> the harness embeds the SAME deterministic materialization the golden uses when
    # a capsule ships no recorded raws (never a false pass: a float capsule that DID ship raws has them
    # attached, and a wrong-operand run fails the golden rather than passing it).
    cb = {k: v for k, v in _CB.items() if k != "canonical_inputs"}
    p = program_from_cb(cb, _FN, _MODEL)
    assert p is not None
    assert "radiance_kernel((float*)_in_W, (float*)_in_A0, (float*)_out_Y0);" in p
