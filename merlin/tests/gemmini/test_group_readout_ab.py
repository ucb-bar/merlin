"""The grouped-versus-split program is built from a model's groups, and only from the closed ones."""

from __future__ import annotations

import importlib.util

from merlin.common.paths import merlin_dir


def _script():
    path = merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts" / "group_readout_ab.py"
    spec = importlib.util.spec_from_file_location("group_readout_ab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(name, op, stages, *, count=1, raw_of=None, **extents):
    entry = {"op": op, "epilogue": stages, **extents}
    if "acc_scale" in stages:
        entry["acc_scale"] = 0.0125
    row = {"name": name, "count": count, "entry": entry}
    if raw_of:
        row["raw_of"] = raw_of
    return row


_REPORT = {
    "model": "toy",
    "entries": [
        _row("closed", "matmul", ["bias_add", "acc_scale", "relu"], count=5, M=196, K=1024, N=256),
        _row("closed_raw", "matmul", [], raw_of="closed", M=196, K=1024, N=256),
        _row("open", "matmul", ["bias_add"], M=196, K=256, N=1024),
        _row("window", "conv2d", ["bias_add", "acc_scale", "relu"], ci=64, N=64, Himg=56, Wimg=56, kh=3, kw=3),
    ],
}


def test_only_closed_contraction_groups_are_measured_and_their_count_is_kept() -> None:
    (layer,) = _script().layers(_REPORT)
    assert (layer["name"], layer["count"], layer["M"], layer["K"], layer["N"]) == ("closed", 5, 196, 1024, 256)
    assert layer["bias"] and layer["relu"] and layer["scale"] == 0.0125


def test_the_program_runs_both_arms_on_one_stimulus_and_voids_a_row_that_disagrees() -> None:
    module = _script()
    source = module.render(module.layers(_REPORT))
    assert '{"closed", 196, 1024, 256, 0.0125f, 1, 1, 5}' in source
    # One device call with the readout stages, against a full-width readout and the library's own
    # per-element function on the host; the two outputs are compared byte for byte.
    assert source.count("tiled_matmul_auto(") == 2 and "scale_and_sat(" in source
    assert "OUT_GROUPED[i] != OUT_SPLIT[i]" in source and "AB_MISMATCHES" in source
    # The measured pass replays the warm-up's stimulus, so the two passes do the same work.
    assert source.count("lcg_state = 12345u") == 2


def _sum_row(name, *, lhs, rhs, bound, count=1, raw_of=None, relu=True):
    entry = {"op": "residual_add", "M": 14336, "N": 14, "lhs_scale": lhs, "rhs_scale": rhs, "bound_lsb": bound}
    entry["epilogue"] = ["relu"] if relu else []
    row = {"name": name, "count": count, "entry": entry}
    if raw_of:
        row["raw_of"] = raw_of
    return row


def test_integer_sums_are_measured_against_their_declared_bound_not_byte_equality() -> None:
    module = _script()
    report = {
        "entries": [
            *_REPORT["entries"],
            _sum_row("sum_b1", lhs=0.5, rhs=0.75, bound=1, count=4),
            _sum_row("sum_b2", lhs=1.5, rhs=0.75, bound=2),
            _sum_row("sum_b1_raw", lhs=0.5, rhs=0.75, bound=1, raw_of="sum_b1", relu=False),
        ]
    }
    rows = module.sums(report)
    assert [(r["name"], r["elements"], r["bound"], r["count"]) for r in rows] == [
        ("sum_b1", 14336 * 14, 1, 4),
        ("sum_b2", 14336 * 14, 2, 1),
    ]
    source = module.render(module.layers(report), rows)
    assert '{"sum_b2", 14336, 14, 1.5f, 0.75f, 1, 2, 1}' in source
    # The unit's own residual add, with a multiplier above one divided out of the loads and given
    # to the readout; the host arm rounds ONCE; the comparison is the declared bound.
    assert "tiled_resadd_auto(S->rows, S->cols, S->lhs / factor, S->rhs / factor, factor," in source
    assert "ROUND_NEAR_EVEN(SUM_A[i] * S->lhs + SUM_B[i] * S->rhs)" in source
    assert "beyond += d > S->bound;" in source and "AB_SUM_BEYOND_BOUND" in source
    assert source.count("static int next_value(int span)") == 1 and "run_sums(measured);" in source
    # A report with no sums renders the program it always did.
    plain = module.render(module.layers(_REPORT))
    assert "run_sums" not in plain and "SUM_A" not in plain
