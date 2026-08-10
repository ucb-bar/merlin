"""The parametric target-dialect factory builds the reference dialects from their dialect_plan.

Guards WS-C: `xdsl_dialects.targets.factory.build_dialect` synthesizes the tensor-resident op/type
classes from data, and the built toy_npu/saturn dialects reproduce the hand-written ones (same op
names, verify, and lowering output).
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_build_dialect_shape():
    from merlin.xdsl_dialects.targets.factory import build_dialect

    vec = {"vector_map", "vector_reduce"}
    for target, dname, ops in (
            ("toy_npu", "toynpu", {"res_pack", "matmul", "commit", "evict"} | vec),
            ("saturn", "saturn", {"pack", "matmul", "commit", "release"} | vec)):
        b = build_dialect(target, matmul_rhs_typed=(target == "toy_npu"),
                          matmul_vl_policy=(target == "saturn"))
        assert b.dialect.name == dname
        assert {op.name.split(".", 1)[1] for op in b.dialect.operations} == ops
        assert {t.name.split(".", 1)[1] for t in b.dialect.attributes} == {b.resident_type.name.split(".", 1)[1],
                                                                           "accumulator"}
        # the TargetSpec exposes the four op-class handles the lowering loop consumes
        assert b.spec.pack_op is b.pack_op and b.spec.matmul_op is b.matmul_op
        assert b.spec.commit_op is b.commit_op and b.spec.evict_op is b.evict_op


def test_reference_modules_use_the_factory():
    """The in-tree reference modules expose the factory-built classes under their stable names."""
    from merlin.xdsl_dialects.targets import saturn, toynpu

    assert toynpu.get_dialect().name == "toynpu"
    assert {o.name for o in toynpu.get_dialect().operations} == {
        "toynpu.res_pack", "toynpu.matmul", "toynpu.commit", "toynpu.evict",
        "toynpu.vector_map", "toynpu.vector_reduce"}
    assert saturn.get_dialect().name == "saturn"
    assert {o.name for o in saturn.get_dialect().operations} == {
        "saturn.pack", "saturn.matmul", "saturn.commit", "saturn.release",
        "saturn.vector_map", "saturn.vector_reduce"}


def test_factory_lowering_matches_per_target():
    """End-to-end lowering through the factory-built dialects emits the expected target ops."""
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    for target, prefix, evict in (("toy_npu", "toynpu", "evict"), ("saturn", "saturn", "release")):
        txt = _common.text(lower_repeated_rhs_matmul(target=target).target_module)
        assert f"{prefix}.matmul" in txt and f"{prefix}.commit" in txt and f"{prefix}.{evict}" in txt
        lower_repeated_rhs_matmul(target=target).target_module.verify()  # verifies clean
