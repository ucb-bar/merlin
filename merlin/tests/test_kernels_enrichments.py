"""Enrichment coverage: per-tensor roles + measured reuse, op_sequence, dispatch metrics,
interface candidates (4 variants), runtime candidates, Stage-D validation, Exo schedule mining."""
import os

from merlin.kernels import policy, validate
from merlin.kernels.emit.kernel_record import emit_kernel_record
from merlin.kernels.ingest.generic import ingest_generic

DATA = os.path.join(os.path.dirname(__file__), "data", "kernels")


def _rec(name, source, target, op, dtype):
    nk = list(ingest_generic(os.path.join(DATA, name), source=source, target=target,
                             op=op, dtype=dtype))[0]
    return emit_kernel_record(nk)


def test_memory_behavior_and_measured_reuse():
    r = _rec("xnnpack_qs8_gemm_rvv.c", "xnnpack", "rvv", "gemm", "i8")
    mb = r["features"]["memory_behavior"]
    assert mb["rhs"]["role"] == "reusable_weight"
    assert mb["rhs"]["reuse_count"] == 4           # measured MR (register blocking)
    assert mb["rhs"]["packed_once"] is True
    assert mb["acc"]["materialized_before_epilogue"] is False
    assert r["features"]["op_sequence"][0] == "matmul"
    assert "reused_packed_rhs" in r["evidence"]["motifs"]


def test_dispatch_metrics_present_for_gemmini():
    r = _rec("autocomp_gemmini_matmul.c", "autocomp", "gemmini", "matmul", "i8")
    dm = r["features"]["dispatch_metrics"]
    assert dm["n_dispatches"] > 0
    assert dm["small_dispatch_fraction"] > 0.5


def test_promote_emits_interfaces_and_runtime():
    def st(srcs, n):
        s = policy.MotifStat(kernel_count=n); s.sources = set(srcs)
        s.evidence_ids = {x + "_t_op" for x in srcs}; return s
    stats = {"packed_rhs": st(["xnnpack", "autocomp"], 900),
             "accumulator_commit": st(["xnnpack", "autocomp"], 400),
             "many_small_dispatches": st(["autocomp"], 600)}
    res = policy.promote(stats, min_kernels=10,
                         records=[{"features": {"dispatch_metrics": {"n_dispatches": 44, "small_dispatch_fraction": 0.9}}}] * 25)
    inames = {i["name"] for i in res.interfaces}
    assert {"resident_packed_tensor", "accumulator_commit"} <= inames
    iface = next(i for i in res.interfaces if i["name"] == "resident_packed_tensor")
    assert iface["lowering_variants"] == ["baseline", "software_visible", "hardware_managed", "oracle"]
    assert iface["compiler_must_prove"] and iface["hardware_must_provide"]
    assert any(rc["name"] == "command_buffer_batching" for rc in res.runtime_candidates)


def test_stage_d_validation():
    rules = policy.promote({"packed_rhs": _stat(), "accumulator_commit": _stat()}).rules
    v = validate.validate_policies(rules)
    assert v["packed_rhs_policy"]["workloads"]["repeated_rhs_matmul"] == "holds"
    assert v["packed_rhs_policy"]["workloads"]["no_reuse_matmul"] == "correctly_silent"
    # capacity sweep: footprint 131072 overflows the 65536 budget but fits the larger ones
    sweep = {row["resident_store_bytes"]: row["fits"] for row in v["packed_rhs_policy"]["capacity_sweep"]}
    assert sweep[65536] is False and sweep[131072] is True


def _stat():
    s = policy.MotifStat(kernel_count=900); s.sources = {"xnnpack", "autocomp"}
    s.evidence_ids = {"xnnpack_rvv_gemm", "autocomp_gemmini_matmul"}; return s


def test_triton_ingest_and_motifs():
    from merlin.kernels.ingest.triton import ingest_triton
    ks = list(ingest_triton(DATA))
    assert ks, "no triton kernels found in fixture dir"
    mm = next(k for k in ks if k.op == "matmul")
    assert mm.source == "triton" and mm.dtype == "f32"
    r = emit_kernel_record(mm)
    motifs = r["evidence"]["motifs"]
    assert {"packed_rhs", "accumulator_lifetime", "epilogue_before_commit",
            "tiling_blocking"} <= set(motifs)


def test_llm_summary_is_real_not_stub():
    from merlin.common.llm import summarize
    table = {"packed_rhs": {"kernels": 900, "sources": ["xnnpack", "autocomp", "exo"]},
             "weight_stationary_dataflow": {"kernels": 800, "sources": ["autocomp"]}}
    out = summarize(table, ["packed_rhs_policy"])
    assert isinstance(out, str) and len(out) > 40
    assert "packed_rhs" in out  # references the strongest cross-source motif


def test_exo_schedule_markers_fire():
    # A synthetic Exo schedule snippet should fire schedule-level motifs.
    from merlin.kernels.markers import fired_markers
    sched = ("gemmini = set_memory(gemmini, 'res', GEMM_ACCUM)\n"
             "gemmini = set_memory(gemmini, 'b', GEMM_SCRATCH)\n"
             "gemmini = tile_outer_loops(gemmini)\n"
             "gemmini = replace_gemmini_calls(gemmini)\n")
    fired = fired_markers(sched, "exo_schedule")
    assert "accumulator_lifetime" in fired   # GEMM_ACCUM
    assert "packed_rhs" in fired             # GEMM_SCRATCH staging
    assert "weight_stationary_dataflow" in fired
