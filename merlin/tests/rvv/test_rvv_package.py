"""RVV target-package loader + non-perturbation regression.

The whole RVV experiment machinery must NOT change the shipping codegen. The baseline package
artifacts/targets/rvv/hand_v0/ is a VERBATIM capture of pipeline.RVV_TRANSFORM_SCHEDULE +
RVV_CFLAGS; this test asserts the capture is exact (so it cannot silently rot) and that the
loader + cflags-allowlist integrity behave.
"""
import os

import pytest

from merlin.llvmlower import pipeline
from merlin.runtime.backends import zephyr_model as zm
from merlin.mining import load_rvv_package

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "out/artifacts/targets", "rvv", "hand_v0")


def test_hand_v0_schedule_is_verbatim_capture():
    pkg = load_rvv_package(HAND_V0)
    assert pkg.schedule_text == pipeline.RVV_TRANSFORM_SCHEDULE


def test_hand_v0_cflags_match_shipping_rvv_cflags():
    pkg = load_rvv_package(HAND_V0)
    # The package owns the RVV-specific flags; the applier appends _CFLAGS_COMMON. Their union
    # must equal the shipping RVV_CFLAGS so apply(hand_v0) == build_app(backend="rvv").
    assert pkg.cflags + zm._CFLAGS_COMMON == zm.RVV_CFLAGS


def test_hand_v0_package_identity():
    pkg = load_rvv_package(HAND_V0)
    assert pkg.name == "rvv"
    assert pkg.run_id == "hand_v0"
    assert pkg.dtype_strategy == "fp32"
    assert pkg.is_int8 is False
    assert {m["op"] for m in pkg.op_match} == {"linalg.matmul", "linalg.batch_matmul"}
    # MEASURED fp32-GEMM instruction set of the baseline schedule (vfmul.vv+vfadd.vv, NOT a fused
    # vfmacc — the known codegen gap the S4 comparison + gap-router target).
    assert "vfmul.vv" in pkg.expected_instructions
    assert "vfmacc" not in pkg.expected_instructions


def test_hand_v0_lowering_is_byte_identical_to_shipping_rvv(tmp_path):
    """The strong non-perturbation proof: lowering a model through the DEFAULT shipping RVV path
    (transform_schedule=None) and through the hand_v0 package (its verbatim schedule) yields
    BYTE-IDENTICAL LLVM IR. This corroborates that apply_rvv_package(hand_v0) IS the RVV codegen
    we ship/run — not merely a string match. Skips if the m2m lowering toolchain is unavailable."""
    import numpy as np
    from merlin.mining import workloads
    try:
        from merlin.llvmlower.lower import lower_model_file
        from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
        from merlin.llvmlower.toolchain import m2m_python
        if not os.path.exists(str(m2m_python())):
            import pytest; pytest.skip("m2m lowering toolchain unavailable")
    except Exception:
        import pytest; pytest.skip("m2m lowering toolchain unavailable")
    bundle = workloads.gen_matmul_f32(tmp_path, M=8, N=8, K=8, seed=0)
    pkg = load_rvv_package(HAND_V0)
    outs = []
    for tag, sched in (("default", None), ("package", pkg.schedule_text)):
        w = tmp_path / tag
        w.mkdir(parents=True, exist_ok=True)
        prep = _prepare_model_mlir(bundle / "model.mlir", w, int8_compute=False)
        lower_model_file(prep, w / "lower", targets=(), textual=True, vectorize=True,
                         transform_schedule=sched)
        outs.append((w / "lower" / "model.ll").read_text())
    assert outs[0] == outs[1], "package lowering diverged from shipping RVV path"


def test_cflags_allowlist_rejects_arbitrary_flags(tmp_path):
    # A package smuggling a non-compiler flag must fail load (data-not-code integrity).
    (tmp_path / "manifest.yaml").write_text(
        "target: rvv\nrun_id: evil\nfamily: vector_schedule\n"
        "authoring: {mode: hand_curated, author: human}\n"
        "outputs: {schedule: schedule.mlir, knobs: knobs.yaml}\n")
    (tmp_path / "schedule.mlir").write_text("module {}\n")
    (tmp_path / "knobs.yaml").write_text(
        "schedule_file: schedule.mlir\ndtype_strategy: fp32\n"
        "cflags: ['-march=rv64gcv', '; rm -rf /']\n")
    with pytest.raises(ValueError, match="allowlist"):
        load_rvv_package(str(tmp_path))
