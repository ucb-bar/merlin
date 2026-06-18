"""tree-sitter structural extractor — loop nest/order, prepack idiom, AST-accurate op counts.

Skips if the kernels-ast extra (tree-sitter) is not installed; the extractor is a graceful no-op
in that case so the mining pipeline degrades to the regex/intrinsic layer.
"""
import pytest

from merlin.kernels.types import NormalizedKernel
from merlin.kernels.features.ast_struct import extract_ast_struct, available

pytestmark = pytest.mark.skipif(not available(), reason="tree-sitter (kernels-ast extra) not installed")

_GEMM = """
void gemm(const float* a, const float* w, float* c, size_t nc, size_t kc) {
  const size_t nr = __riscv_vsetvlmax_e32m4();
  do {
    size_t vl = __riscv_vsetvl_e32m4(nc);
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;
    size_t k = kc;
    do {
      const float va0 = *a++;
      vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    __riscv_vse32_v_f32m4(c, vacc0, vl);
    nc -= vl;
  } while (nc != 0);
}
"""


def _nk(text, target="rvv"):
    return NormalizedKernel(source="x", target=target, path="k.c", op="gemm", dtype="f32",
                            raw_text=text)


def test_loop_nest_and_order():
    s = extract_ast_struct(_nk(_GEMM), {})["struct"]
    assert s["loop_nest_depth"] == 2
    assert s["loop_order"][:2] == ["nc", "k"]      # N-tile loop outside the K reduction


def test_pointer_advance_prepack_detected():
    s = extract_ast_struct(_nk(_GEMM), {})["struct"]
    assert s["pointer_advance_prepack"] is True    # `w = w + nr` weight streaming


def test_ast_op_counts():
    s = extract_ast_struct(_nk(_GEMM), {})["struct"]
    assert s["n_fma_calls"] == 1
    assert s["n_vector_loads"] == 2 and s["n_vector_stores"] == 1


def test_non_rvv_is_noop():
    assert extract_ast_struct(_nk("int main(){return 0;}", target="avx2"), {}) == {}
