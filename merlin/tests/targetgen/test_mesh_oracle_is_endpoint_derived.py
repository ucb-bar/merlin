"""The mesh oracle a target is graded on must follow that target's DERIVED endpoint, and a report must
name the executor that ACTUALLY ran.

Every defect pinned here presented as "no reachable oracle" for a target whose oracle was reachable:

* ``_mesh_verify`` certified every tile through the RoCC/OOT path regardless of endpoint kind, so a
  self-hosted-ISA target reported "package declares target X with no registered backend" on all 15 tiles
  while the very same target's whole-model run executed fine through the endpoint-aware dispatcher;
* the whole-model report recorded the simulator it was ASKED for, not the one that ran — two of the three
  dispatch paths ignore it entirely;
* the program oracle capped every program at a constant 20000 cycles, so a model-sized layer was stopped
  mid-flight and the resulting ``ProgramDidNotHalt`` — a verdict about the PROGRAM — was caught as its
  ``OracleUnavailable`` parent and collapsed into a skip;
* the scalar/RVV package was chosen from the dtype token naming how the model was QUANTIZED rather than
  from what the compiled IR carries.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common


def _binding(*, accum_dtype: str, operand_dtype: str = "fp8_e4m3", flush: bool = False):
    """A minimal CorpusBinding for the numeric-reference tests. Built through the real dataclass (not a
    stub) so a field added to the datapath contract shows up here rather than being quietly absent."""
    from merlin.targetgen.corpus_spec import CorpusBinding
    return CorpusBinding(target="t", tile_dim=16, operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                         integer=False, tiers=[], compare="tolerance_float",
                         subnormal_operand_flush=flush)

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


# --------------------------------------------------------------------------- cycle budget
def test_the_cycle_budget_grows_with_the_declared_work():
    """A constant cap turns a big-but-healthy program into a fake unavailability."""
    from merlin.targetgen.program_oracle import derive_cycle_budget

    tile = {"tensors": {"A0": {"shape": [32, 32]}, "W": {"shape": [32, 32]},
                        "Y0": {"shape": [32, 32]}}}
    layer = {"tensors": {"A0": {"shape": [32, 352]}, "W": {"shape": [352, 128]},
                         "Y0": {"shape": [256, 16]}}}
    assert derive_cycle_budget(layer) > derive_cycle_budget(tile), \
        "a larger program must get a larger halt budget"
    # and the layer that actually hung at the old constant must now clear it by a wide margin
    assert derive_cycle_budget(layer) > 20000 * 10


def test_the_budget_never_drops_below_the_original_floor():
    """Small programs keep the behavior they had; the floor is a floor, not a target."""
    from merlin.targetgen.program_oracle import derive_cycle_budget

    assert derive_cycle_budget({}) == 20000
    assert derive_cycle_budget({"tensors": {}}) == 20000


def test_a_hung_program_is_a_verdict_not_an_absent_oracle():
    """``ProgramDidNotHalt`` subclasses ``OracleUnavailable``; catching the parent first reported a program
    that RAN and hung as a missing oracle, which the caller turned into a skip."""
    from merlin.targetgen import program_oracle as PO

    assert issubclass(PO.ProgramDidNotHalt, PO.OracleUnavailable)
    src = __import__("inspect").getsource(
        __import__("merlin.targetgen.mesh_program_run", fromlist=["x"]).matmul_on_program_oracle)
    assert "except PO.ProgramDidNotHalt" in src, \
        "the hung-program verdict must be handled BEFORE the generic unavailability"
    assert src.index("except PO.ProgramDidNotHalt") < src.index("except PO.OracleUnavailable"), \
        "a subclass handler placed after its parent never runs"


# --------------------------------------------------------------------------- scalar-lane dtype
def test_the_scalar_datapath_follows_the_ir_not_the_bundle_name(tmp_path):
    """A weight-only capture is mostly f32 by COUNT but still needs its narrow datapath, so selection is
    by presence of the narrowest type — a majority vote silently drops int8."""
    from merlin.compile_cli import ir_scalar_dtype

    weight_only_int8 = tmp_path / "int8_bundle"
    weight_only_int8.mkdir()
    (weight_only_int8 / "model.mlir").write_text(
        "".join(["tensor<4x4xf32>\n"] * 1190 + ["tensor<4x4xi8>\n"] * 46), encoding="utf-8")
    assert ir_scalar_dtype(weight_only_int8) == "int8", \
        "46 i8 tensors against 1190 f32 ones still require the int8 datapath"

    fake_quant_fp8 = tmp_path / "fp8_bundle"
    fake_quant_fp8.mkdir()
    (fake_quant_fp8 / "model.mlir").write_text("tensor<4x4xf32>\n" * 1303, encoding="utf-8")
    assert ir_scalar_dtype(fake_quant_fp8) == "fp32", \
        "a dequantized fp8 capture carries no fp8 in its IR: its datapath is f32"


def test_an_unreadable_bundle_yields_no_dtype_rather_than_a_guess(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    from merlin.compile_cli import ir_scalar_dtype
    assert ir_scalar_dtype(empty) is None, "a missing model.mlir must fail closed, not default"


def test_a_dtype_with_no_scalar_datapath_is_refused_not_substituted():
    """The map must not name a strategy no package can declare — that error reads as a missing artifact
    and sends the reader off to mint a package that cannot exist."""
    from merlin.compile_cli import _DTYPE_STRATEGY
    from merlin.rvvgen.tuning_agent import _DTYPE_STRATEGIES

    unsatisfiable = {d: s for d, s in _DTYPE_STRATEGY.items() if s not in _DTYPE_STRATEGIES}
    assert not unsatisfiable, \
        f"these dtypes map to strategies no package may declare: {unsatisfiable}"


def test_every_ir_spelling_maps_to_a_real_strategy():
    from merlin.compile_cli import _DTYPE_STRATEGY, _IR_ELEMENT_ORDER, _IR_ELEMENT_SPELLING

    assert set(_IR_ELEMENT_ORDER) == set(_IR_ELEMENT_SPELLING)
    assert set(_IR_ELEMENT_SPELLING) <= set(_DTYPE_STRATEGY), \
        "an IR spelling that selects a dtype with no strategy cannot be acted on"


# --------------------------------------------------------------------------- endpoint dispatch
def test_mesh_verify_picks_the_oracle_from_the_endpoint_not_a_fixed_path():
    """The OOT-cert path is one endpoint's oracle, not every target's."""
    import inspect

    from merlin import compile_cli

    src = inspect.getsource(compile_cli._mesh_verify)
    assert "_endpoint_of" in src, "the tile oracle must be chosen from the DERIVED endpoint"
    assert "_certify_tile_via_executor" in src, \
        "a non-OOT endpoint needs the shared endpoint-aware executor, not oot_runner.certify"


@pytest.mark.parametrize("target,expect_oot", [("gemmini", True), ("atlas", False)])
def test_each_targets_endpoint_selects_its_own_certifier(target, expect_oot):
    """Derived per target, never a target-name branch: a RoCC endpoint certifies through the OOT package,
    a self-hosted-ISA endpoint through its own program oracle."""
    from merlin.targetgen.capsule_runner import _bespoke_sim_via, _endpoint_of, _SIM_ORACLES

    so = _SIM_ORACLES.get(_bespoke_sim_via(target))
    endpoint, _ = _endpoint_of(target)
    via_oot = not (so is not None and so.exclusive) and endpoint in (
        None, "inline_asm_insn", "upstream_target")
    assert via_oot is expect_oot, f"{target}: endpoint {endpoint!r} routed to the wrong certifier"


# --------------------------------------------------------------------------- numeric gate
def test_the_tile_gate_tolerance_is_derived_from_the_accumulator_format():
    """An integer accumulator does not round and gates bit-exact; a narrow float one cannot, and picking
    a tolerance by hand is wrong in both directions."""
    from merlin.compile_cli import _accum_rel_tolerance

    assert _accum_rel_tolerance("i32", 352) == 0.0, "integer accumulation is exact"
    assert _accum_rel_tolerance("int8", 64) == 0.0
    deep, shallow = _accum_rel_tolerance("bf16", 352), _accum_rel_tolerance("bf16", 32)
    assert deep > shallow > 0.0, "a deeper reduction admits more accumulated rounding"
    assert _accum_rel_tolerance("fp32", 352) < _accum_rel_tolerance("bf16", 352), \
        "a wider accumulator must gate tighter than a narrower one"


def test_an_unresolvable_accumulator_refuses_to_pick_a_tolerance():
    """Fail closed: too tight condemns a good mesh, too loose passes a broken one."""
    from merlin.compile_cli import _accum_rel_tolerance

    assert _accum_rel_tolerance("not_a_format", 8) is None


def test_the_float_tolerance_is_tight_enough_to_reject_a_broken_mesh():
    """A derived bound still has to be a GATE. An earlier version returned 1.17 (117%) for a bf16
    accumulator at k=352, which accepts essentially any output; the measured max relative error on that
    exact layer was 0.067, so the bound must sit above the measurement and far below 1."""
    from merlin.compile_cli import _accum_rel_tolerance

    bound = _accum_rel_tolerance("bf16", 352)
    assert bound > 0.067, "the bound must cover the error real bf16 accumulation produces"
    assert bound < 0.25, f"a {bound:.0%} tolerance is not a gate — it accepts a broken mesh"


# --------------------------------------------------------------------------- accumulator reference
def test_the_reference_accumulates_in_the_targets_declared_format():
    """Gating a narrow-float mesh against an f32 product condemns a correct device.

    Measured on a 32x352x128 fp8xbf16 layer: 796 of 4096 elements differ from the f32 product by up to 22
    — and the device reproduces a sequential bf16 accumulation BIT-FOR-BIT on every element. The mesh was
    exact; the reference was wrong."""
    import numpy as np

    from merlin.compile_cli import _reference_on_datapath

    rng = np.random.default_rng(0xA71A5)
    A = np.rint(rng.standard_normal((32, 352)) * 3).clip(-8, 7).astype(np.float32)
    W = np.rint(rng.standard_normal((352, 128)) * 3).clip(-8, 7).astype(np.float32)

    bf16_ref = _reference_on_datapath(A, W, _binding(accum_dtype="bf16"))
    assert bf16_ref is not None, "bf16 is a declared accumulator format and must be modelled"
    n_diff = int((bf16_ref != A @ W).sum())
    assert n_diff > 0, "a bf16 accumulator cannot equal the f32 product at this depth"
    # every difference must be rounding-sized, not structural
    assert float(np.abs(bf16_ref - A @ W).max()) < 64.0


def test_an_exact_accumulator_needs_no_model():
    """An integer accumulator does not round, and f32 IS the product — modelling either would only add a
    way to be wrong. Both fall through to the plain reference."""
    import numpy as np

    from merlin.compile_cli import _reference_on_datapath

    A = np.ones((4, 4), dtype=np.float32)
    W = np.ones((4, 4), dtype=np.float32)
    assert _reference_on_datapath(A, W, _binding(accum_dtype="i32")) is None
    assert _reference_on_datapath(A, W, _binding(accum_dtype="fp32")) is None
    assert _reference_on_datapath(A, W, _binding(accum_dtype="not_a_format")) is None


def test_the_reference_reads_operands_the_way_the_datapath_does():
    """A datapath that flushes subnormal operands must be MODELLED as flushing them.

    Without this the reference reads an operand the hardware never saw, and grades correct silicon as
    broken -- which is the same defect in the mirror as feeding the hardware an operand it cannot hold.
    The fact is declared per target (``subnormal_operand_flush``) and now arrives here on the binding.
    """
    import numpy as np

    from merlin.compile_cli import _reference_on_datapath
    from merlin.runtime.fp8_formats import normal_range

    min_normal, _mx = normal_range("fp8_e4m3")
    sub = min_normal / 2.0                                # a subnormal the format still represents
    A = np.full((4, 4), sub, dtype=np.float32)
    W = np.ones((4, 4), dtype=np.float32)

    flushing = _reference_on_datapath(A, W, _binding(accum_dtype="bf16", flush=True))
    admitting = _reference_on_datapath(A, W, _binding(accum_dtype="bf16", flush=False))
    assert flushing is not None and admitting is not None
    assert float(np.abs(flushing).max()) == 0.0, "a flushing datapath multiplies by zero"
    assert float(np.abs(admitting).max()) > 0.0, "a datapath that admits subnormals must not zero them"


def test_the_strongest_gate_is_tried_first():
    """Bit-exact against the declared accumulator beats a tolerance, and the record must say which gate
    carried the verdict — otherwise a tolerance pass is indistinguishable from an exact one."""
    import inspect

    from merlin import compile_cli

    src = inspect.getsource(compile_cli._certify_tile_via_executor)
    assert src.index("_reference_on_datapath") < src.index("_accum_rel_tolerance"), \
        "the exact gate must be attempted before falling back to a tolerance"
    assert "bit-exact vs" in src and "tolerance" in src, \
        "the record must name which gate produced the verdict"


# --------------------------------------------------------------------------- sub-tile extents
def test_a_sub_tile_extent_is_padded_to_the_mesh_edge():
    """A real model layer does not get to choose its M.

    Every matmul layer of an 8-token sequence has M=8, against a tile edge of 32 (atlas) or 16 (gemmini).
    The generated package is entitled to reject a sub-tile extent, so the mesh refused all 15 layers and
    the dispatch runtime fell back to the host kernel for every one — a whole model reporting "on mesh"
    while running entirely on the CPU. The boundary now rounds up to the tile edge and slices back;
    zero-padding is exact for a contraction."""
    import inspect

    from merlin import compile_cli

    src = inspect.getsource(compile_cli.run_matmul_on_mesh)
    assert "_padded" in src and "_unpad" in src, "the mesh boundary must pad sub-tile extents"
    # and the result must come back at the extent the caller asked for, not the padded one
    assert "_m_true" in src and "_n_true" in src


def test_padding_a_contraction_does_not_change_it():
    """The arithmetic claim the padding rests on, checked directly."""
    import numpy as np

    rng = np.random.default_rng(7)
    M, K, N, D = 8, 130, 30, 32
    A = np.rint(rng.standard_normal((M, K)) * 2).astype(np.float32)
    W = np.rint(rng.standard_normal((K, N)) * 2).astype(np.float32)

    def up(x):
        return ((x + D - 1) // D) * D

    Ap = np.zeros((up(M), up(K)), dtype=np.float32)
    Ap[:M, :K] = A
    Wp = np.zeros((up(K), up(N)), dtype=np.float32)
    Wp[:K, :N] = W
    assert np.array_equal((Ap @ Wp)[:M, :N], A @ W), \
        "zero padding must leave every retained output element untouched"
