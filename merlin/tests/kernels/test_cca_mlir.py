"""WS-C C2: the CCA expressed IN MLIR (cca dialect) + deterministic-analyzer round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.xdsl_dialects._common import HAS_XDSL

pytestmark = pytest.mark.skipif(not HAS_XDSL, reason="xDSL not installed")

_ASM_DIR = merlin_dir() / "tests" / "data" / "cca_asm"


def _compose_from_fixture(name: str):
    """Compose a CCA the deterministic way — the asm analyzer (lift_asm) over a real disassembly."""
    import unittest.mock as m

    from merlin.kernels import cca
    from merlin.kernels.decode import objdump, rvv
    with m.patch.object(objdump, "disassemble_text", lambda *a, **k: (_ASM_DIR / name).read_text()):
        return cca.lift_asm(rvv.decode("k.o"), op="matmul", source=name.split("_")[0])


def test_to_mlir_emits_cca_dialect_ops():
    from merlin.kernels import cca_mlir
    c = _compose_from_fixture("openblas_sgemm_rvv.objdump")
    text = cca_mlir.to_mlir(c)
    # a proper MLIR module carrying the cca dialect ops (composed by the analyzer, not an LLM)
    assert "builtin.module" in text
    assert "cca.kernel" in text and "cca.compute" in text and "cca.vector" in text
    assert 'contraction_form = "fused_fma"' in text


@pytest.mark.parametrize("fixture", ["openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump",
                                     "ours_baseline_matmul.objdump"])
def test_cca_mlir_roundtrip_preserves_scalar_facets(fixture):
    from merlin.kernels import cca_mlir
    c = _compose_from_fixture(fixture)
    c2 = cca_mlir.from_mlir(cca_mlir.to_mlir(c))
    for f in ("contraction_form", "accumulator_dtype", "widening", "reduction_form", "epilogue",
              "accumulator_resident", "nr_is_vsetvlmax", "activation_vectorization"):
        assert getattr(c.compute, f) == getattr(c2.compute, f), f
    # register_block restored as (mr, None) — MR (the field the compare uses) preserved
    mr = c.compute.register_block[0] if c.compute.register_block else None
    mr2 = c2.compute.register_block[0] if c2.compute.register_block else None
    assert mr == mr2
    for f in ("sew", "lmul", "vl_strategy", "tail"):
        assert getattr(c.vector, f) == getattr(c2.vector, f), f
    assert c.backend == c2.backend


def test_roundtrip_hand_built_cca():
    from merlin.kernels import cca, cca_mlir
    c = cca.CCA(op="gelu", backend=["rvv"],
                compute=cca.ComputeFacet(op="gelu", activation_vectorization="scalar_libm_call",
                                         accumulator_dtype="f32"),
                vector=cca.VectorFacet(sew=32, lmul=1.0, vl_strategy="vsetvl_loop", tail="tu"),
                provenance={"source": "hand", "level": "asm"})
    c2 = cca_mlir.from_mlir(cca_mlir.to_mlir(c))
    assert c2.op == "gelu" and c2.backend == ["rvv"]
    assert c2.compute.activation_vectorization == "scalar_libm_call"
    assert c2.vector.vl_strategy == "vsetvl_loop" and c2.vector.tail == "tu"
    assert c2.provenance["source"] == "hand"
