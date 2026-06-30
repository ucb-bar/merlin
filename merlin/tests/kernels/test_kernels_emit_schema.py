"""Emitted records/candidates/rules validate against the shared schemas."""
import os
from merlin.common.paths import merlin_dir

from merlin.common import schemas
from merlin.kernels.emit.abstraction_candidate import emit_abstraction_candidate
from merlin.kernels.emit.kernel_record import emit_kernel_record
from merlin.kernels.emit.policy_rule import emit_policy_rule
from merlin.kernels.ingest.generic import ingest_generic

DATA = str(merlin_dir() / "tests" / "data" / "kernels")


def test_kernel_record_valid():
    nk = list(ingest_generic(os.path.join(DATA, "xnnpack_qs8_gemm_rvv.c"),
                             source="xnnpack", target="rvv", op="gemm", dtype="i8"))[0]
    rec = emit_kernel_record(nk)  # validates internally
    assert schemas.validate(rec, "kernel_record") == []
    assert set(rec) >= {"source", "target", "path", "op", "dtype", "features", "evidence"}


def test_abstraction_candidate_valid():
    c = emit_abstraction_candidate(
        name="resident_packed_tensor", kind="memory_state", motivation="reuse",
        evidence=["xnnpack_rvv_gemm", "autocomp_gemmini_matmul"],
        interface_features=["resident_pack"])
    assert schemas.validate(c, "abstraction_candidate") == []


def test_policy_rule_valid():
    r = emit_policy_rule(policy="packed_rhs_policy", evidence=["xnnpack_rvv_gemm"],
                         when={"rhs_reuse_count": ">= 2"}, actions=["hoist_pack"])
    assert schemas.validate(r, "policy_rule") == []


def test_validate_rejects_missing_field():
    problems = schemas.validate({"policy": "x"}, "policy_rule")
    assert any("evidence" in p for p in problems)
