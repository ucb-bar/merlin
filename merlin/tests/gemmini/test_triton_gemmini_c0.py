"""The go/no-go: a Triton kernel reaches a generated accelerator and is certified on its RTL.

Everything before this could be true of a frontend that produces well-formed IR nobody can execute.
Here the kernel descends the whole staged pipeline — contract, schedule, interface, the dynamically
loaded Gemmini target dialect, runtime, command buffer — and is then run on the Gemmini Verilator
RTL, with the result compared against an independent integer matmul computed in Python.

What makes it evidence for the architecture, rather than for one kernel, is what did NOT happen:
no Triton-specific code exists anywhere on this path. The Gemmini package is loaded from disk, the
routing decision is read from its dialect plan, and the frontend never learns the target's name.
The zero-diff guards for that live in `merlin/tests/infra/test_triton_target_independence.py`;
this file establishes that the path works at all.

The kernel is two activations against one shared weight, not a single tile. That is not incidental:
a weight-stationary systolic array earns its keep by keeping the weight resident, and its driver is
built around a packed right-hand side, so a lone matmul never produces a RES_PACK for it to consume.
The shared operand is what makes residency inferrable.
"""
from __future__ import annotations

import numpy as np
import pytest
import triton_kernels as K

from merlin.common.paths import repo_root
from merlin.runtime import reference_outputs, simulate
# The gemmini reference backend was EVICTED from runtime/backends/ into its own target package
# (contract `plugin.backend`), so it is reached through the registry rather than by module path.
from merlin.runtime.backends.base import get_backend

gemmini = get_backend("gemmini")
from merlin.triton import source
from merlin.triton.bridge import to_linalg

GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def _package():
    from merlin.targetgen.registry import load_target

    if not GEMMINI_PACKAGE.is_dir():
        pytest.skip("gemmini target package not present")
    return load_target(GEMMINI_PACKAGE)


@pytest.fixture(scope="module")
def descent():
    """Triton source all the way to a Gemmini command buffer, computed once."""
    from merlin import compile_core

    spec = K.repeated_rhs_matmul_spec()
    ttir = source.make_ttir(spec)
    bridged = to_linalg(ttir, spec)
    result = compile_core.compile_core_mlir(bridged.module, target_package=_package())
    return {"spec": spec, "ttir": ttir, "bridged": bridged, "result": result,
            "lowered": result.staged}


def test_the_payload_routes_to_the_staged_pipeline(descent):
    """Read off the target's own dialect plan — never off its name."""
    route = descent["result"].route
    assert route.kind == "staged", route.reason
    assert route.payload == ("matmul",)
    assert "matmul" in route.materializable


def test_the_ttir_carries_a_dot_and_the_core_mlir_carries_a_matmul(descent):
    """The two ends of the bridge, checked at both ends."""
    assert descent["ttir"].has_op("tt.dot")
    text = descent["bridged"].text
    assert "linalg.quantized_matmul" in text
    assert "tt." not in text and "ttg." not in text, "a Triton op survived into core MLIR"


def test_all_six_stage_modules_verify(descent):
    modules = list(descent["lowered"].modules())
    assert len(modules) == 6, [m.name for m in modules]
    for module in modules:
        module.verify()


def test_it_reaches_the_gemmini_dialect_and_makes_the_weight_resident(descent):
    """Residency is the whole point of a weight-stationary array, and it is inferred, not asked for."""
    ops = {op.name for op in descent["lowered"].target_module.walk()}
    ops -= {"builtin.module", "func.func", "func.return"}
    assert ops == {"gemmini.pack", "gemmini.matmul", "gemmini.commit", "gemmini.release"}


def test_the_command_buffer_is_the_certified_c0_shape(descent):
    cb = descent["lowered"].command_buffer
    assert cb["target"] == "gemmini"
    assert [c["opcode"] for c in cb["commands"]] == [
        "RES_PACK", "MATMUL_RESIDENT", "COMMIT", "MATMUL_RESIDENT", "COMMIT", "EVICT"]
    dtypes = {t["dtype"] for t in cb["tensors"].values()}
    assert dtypes == {"i8"}, dtypes


def test_the_command_buffer_simulates_against_an_independent_reference(descent):
    """L0: no toolchain needed, so this runs everywhere and localizes a failure to semantics."""
    from merlin.runtime.commandbuffer import materialize_inputs

    cb = descent["lowered"].command_buffer
    outputs = simulate(cb)["outputs"]
    assert outputs == reference_outputs(cb)

    # And against a matmul computed here, so the reference is not merely Merlin agreeing with itself.
    # Operand names are read out of the command buffer rather than assumed, so this keeps checking
    # the right tensors if naming changes.
    tensors = materialize_inputs(cb)
    packed = {c["operands"]["dst"]: c["operands"]["src"]
              for c in cb["commands"] if c["opcode"] == "RES_PACK"}
    lhs_of = {c["operands"]["dst"]: (c["operands"]["lhs"], c["operands"]["rhs"])
              for c in cb["commands"] if c["opcode"].startswith("MATMUL")}
    commits = [c for c in cb["commands"] if c["opcode"] == "COMMIT"]
    assert commits, "no COMMIT to check"
    for commit in commits:
        lhs, rhs = lhs_of[commit["operands"]["src"]]
        activation = np.array(tensors[lhs].to_list(), dtype=np.int64)
        weight = np.array(tensors[packed.get(rhs, rhs)].to_list(), dtype=np.int64)
        got = np.array(outputs[commit["operands"]["dst"]], dtype=np.int64)
        assert np.array_equal(got, activation @ weight), commit["operands"]["dst"]


@pytest.mark.slow
@pytest.mark.skipif(not gemmini.available("spike"), reason="spike-gemmini unavailable")
def test_l1_spike_bootstrap(descent, tmp_path):
    """A functional model, explicitly NOT certification — the flag says so and is asserted."""
    cb = descent["lowered"].command_buffer
    res = gemmini.run_command_buffer(cb, workdir=tmp_path, simulator="spike", timeout=300)
    assert res["oracle"]["derived_from_rtl"] is False
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]


@pytest.mark.slow
@pytest.mark.skipif(not gemmini.available("verilator"), reason="Gemmini Verilator sim unavailable")
def test_l2_verilator_certification(descent, tmp_path):
    """The gate: the real RTL computes what the Triton kernel said, bit for bit."""
    cb = descent["lowered"].command_buffer
    res = gemmini.run_command_buffer(cb, workdir=tmp_path, simulator="verilator", timeout=900)
    assert res["oracle"]["derived_from_rtl"] is True
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]
    metrics = res["metrics"]
    assert metrics["cycle_window"] == "gemmini_region"
    if metrics["cycle_source"] == "rdcycle":
        assert metrics["cycles"] > 0
