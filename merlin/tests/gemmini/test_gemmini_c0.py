"""Gemmini C0 RTL-certification path.

C0 scope: i8 x i8 -> i32, matmul only, empty epilogue, i32 passthrough output, a single
DIM(16)x16x16 tile. The command buffer reuses the target-independent ABI
(RES_PACK / MATMUL_RESIDENT / COMMIT / EVICT); Gemmini is just another `target`.

Tests climb the oracle ladder:
  L0  reference_outputs(cb) == simulate(cb)            (always; no toolchain)
      pipeline descends to the gemmini dialect + simulates
      codegen smoke (added with A4)
  L1  spike + gemmini extension  (bootstrap; derived_from_rtl: false)   (added with A5/A6)
  L2  Gemmini Verilator RTL      (certification; derived_from_rtl: true)(added with A5/A6)

Diagnostic routing when a level fails:
  L0 fails                      -> merlin cb semantics / reference / simulator / lowering
  L0 passes, spike fails        -> Gemmini codegen or spike invocation
  L0+spike pass, Verilator fails-> runtime/kernel plane (config order, fences, layout,
                                   stationary transpose, accumulator addressing) — NOT the
                                   spec/dialect.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

from pathlib import Path

import pytest

from merlin.runtime import outputs_match, reference_outputs, simulate
from merlin.runtime.commandbuffer import materialize_inputs

DIM = 16


def _matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Independent integer matmul (full precision) for the C0 correctness check."""
    m, k, n = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]


def c0_command_buffer(m: int = DIM, k: int = DIM, n: int = DIM) -> dict:
    """The C0 workload as a command buffer: pack W, matmul A0 @ W -> acc, commit i32, evict.

    Shapes default to a single 16x16x16 tile (DIM-aligned). Leaf tensor values are
    materialized deterministically by the runtime, so the same bytes feed the reference,
    the simulator, and (later) the generated Gemmini kernel.
    """
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "backend": "verilator",
        "tensors": {
            "W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
            "A0": {"shape": [m, k], "dtype": "i8", "role": "input"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
            {"opcode": "EVICT", "operands": {"handle": "W_res"}},
        ],
    }


def test_c0_command_buffer_reference_matches_simulator():
    """L0: the independent reference and the command-buffer simulator agree on C0."""
    cb = c0_command_buffer()
    ref = reference_outputs(cb)
    sim = simulate(cb)["outputs"]
    assert outputs_match(sim, ref), f"sim={sim}\nref={ref}"
    # Independent full-precision recompute from the same materialized leaves confirms both
    # the arithmetic and the i32 passthrough (no i8 clamp on the empty-epilogue commit).
    leaves = materialize_inputs(cb)
    a0 = leaves["A0"].to_list()
    w = leaves["W"].to_list()
    assert ref["Y0"] == _matmul(a0, w), f"Y0={ref['Y0']}\nexpected={_matmul(a0, w)}"


# --- xDSL-gated: the lowering pipeline descends to the gemmini dialect ---
try:
    from merlin.xdsl_dialects import _common
    _HAS_XDSL = _common.HAS_XDSL
except Exception:  # pragma: no cover
    _HAS_XDSL = False


_REPO = repo_root()
_GEMMINI_PKG = _REPO / "artifacts/targets/gemmini/hand_v0"


@pytest.mark.skipif(not _HAS_XDSL, reason="xDSL not installed")
@pytest.mark.skipif(not _GEMMINI_PKG.is_dir(), reason="gemmini target package not present")
def test_gemmini_pipeline_descends_and_simulates():
    """A3: interface -> gemmini -> command buffer via the ISOLATED, dynamically-loaded target
    package (gemmini is NOT in the core tree) — verified at every stage, simulates correctly."""
    from merlin.xdsl_dialects.lowering import execute, lower_repeated_rhs_matmul
    from merlin.targetgen.registry import load_target

    pkg = load_target(_GEMMINI_PKG)
    lowered = lower_repeated_rhs_matmul(reuse=2, m=DIM, k=DIM, n=DIM, target_package=pkg)
    for mod in lowered.modules():
        mod.verify()

    target_ops = {op.name for op in lowered.target_module.walk()} - {
        "builtin.module", "func.func", "func.return"}
    assert target_ops == {"gemmini.pack", "gemmini.matmul", "gemmini.commit", "gemmini.release"}

    cb = lowered.command_buffer
    assert cb["target"] == "gemmini"
    opcodes = {c["opcode"] for c in cb["commands"]}
    assert opcodes <= {"RES_PACK", "MATMUL_RESIDENT", "MATMUL", "COMMIT", "EVICT"}

    assert execute(lowered)["correct"] is True


# --- codegen smoke (no toolchain) ---
def test_gemmini_codegen_emits_real_driver():
    """A4: the C0 driver uses the explicit low-level Gemmini intrinsic sequence."""
    from merlin.runtime.backends.gemmini_codegen import generate_driver

    src = generate_driver(c0_command_buffer())
    for needle in ("gemmini_config_ex", "gemmini_mvin", "gemmini_preload",
                   "gemmini_compute_preloaded", "gemmini_mvout", "read_cycles",
                   "OUT Y0", "DONE"):
        assert needle in src, f"generated driver missing {needle!r}"
    assert "tiled_matmul_auto" not in src  # must be the explicit sequence


# --- oracle-gated: compile + run on real Gemmini, three-way bit-exact, per conformance rung ---
from merlin.runtime.backends import gemmini  # noqa: E402  (import-safe without toolchain)
from merlin.eval.gemmini_conformance import RUNGS, build  # noqa: E402

RUNG_IDS = sorted(RUNGS)


def _assert_three_way(res: dict, cb: dict, derived_from_rtl: bool):
    assert res["oracle"]["derived_from_rtl"] is derived_from_rtl
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb)
    assert res["outputs"] == simulate(cb)["outputs"]


@pytest.mark.parametrize("rung", RUNG_IDS)
def test_rung_reference_matches_simulator(rung):
    """L0 over every rung: independent reference == command-buffer simulator (no toolchain)."""
    cb = build(rung)
    assert outputs_match(simulate(cb)["outputs"], reference_outputs(cb))


@pytest.mark.skipif(not gemmini.available("spike"), reason="spike-gemmini unavailable")
@pytest.mark.parametrize("rung", RUNG_IDS)
def test_rung_spike_bootstrap(rung, tmp_path):
    """L1 bootstrap (functional model, NOT certification): three-way bit-exact per rung."""
    cb = build(rung)
    res = gemmini.run_command_buffer(cb, workdir=tmp_path, simulator="spike", timeout=180)
    _assert_three_way(res, cb, derived_from_rtl=False)


@pytest.mark.skipif(not gemmini.available("verilator"), reason="Gemmini Verilator sim unavailable")
@pytest.mark.parametrize("rung", RUNG_IDS)
def test_rung_verilator_certification(rung, tmp_path):
    """L2 RTL certification: three-way bit-exact + measured cycle anchor per rung."""
    cb = build(rung)
    res = gemmini.run_command_buffer(cb, workdir=tmp_path, simulator="verilator", timeout=900)
    _assert_three_way(res, cb, derived_from_rtl=True)
    m = res["metrics"]
    assert m["cycle_window"] == "gemmini_region"
    assert m["cycle_source"] in ("rdcycle", "unknown")
    if m["cycle_source"] == "rdcycle":
        assert m["cycles"] > 0
