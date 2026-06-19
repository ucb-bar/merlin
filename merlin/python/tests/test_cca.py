"""R3: Common Compute Abstraction lift + cross-level agreement (target-agnostic)."""
from __future__ import annotations

from merlin.kernels import cca
from merlin.kernels.decode import objdump, rvv

# matmul-ish stream: mul_add (vfmul+vfadd), no vfmacc, fixed vsetivli, e32m2.
_MUL_ADD = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmul.vv\tv8, v8, v9
       c: 00     \tvfadd.vv\tv10, v10, v8
"""
# fused stream: vfmacc present, e32m4.
_FUSED = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m4, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmacc.vv\tv12, v8, v9
"""


def _cca(monkeypatch, snippet, source):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: snippet)
    return cca.lift_asm(rvv.decode("x.o"), op="matmul", source=source)


def test_lift_asm_mul_add(monkeypatch):
    c = _cca(monkeypatch, _MUL_ADD, "ours")
    assert c.backend == ["rvv"]
    assert c.compute.contraction_form == "mul_add"   # no vfmacc -> mul_add
    assert c.vector.sew == 32 and c.vector.lmul == 2.0
    assert c.vector.vl_strategy == "vsetivli_fixed"


def test_lift_asm_fused(monkeypatch):
    c = _cca(monkeypatch, _FUSED, "expert")
    assert c.compute.contraction_form == "fused_fma"
    assert c.vector.lmul == 4.0


def test_agreement_flags_divergences(monkeypatch):
    ours = _cca(monkeypatch, _MUL_ADD, "ours")
    expert = _cca(monkeypatch, _FUSED, "expert")
    rep = cca.cca_agree(expert, ours)
    assert not rep.agree
    axes = {d.split(":")[0] for d in rep.disagreements}
    assert "compute.contraction_form" in axes   # the vfmacc gap
    assert "vector.lmul" in axes                 # 4 vs 2
    # self-agreement is the validity baseline
    assert cca.cca_agree(ours, ours).agree


def test_composite_backend_supported():
    # a heterogeneous region (NPU+RVV) is just a backend list — not a special case
    c = cca.CCA(op="attention", backend=["npu", "rvv"])
    assert c.backend == ["npu", "rvv"]
    assert c.spatial is None and c.dataflow is None   # facets populated only when relevant
