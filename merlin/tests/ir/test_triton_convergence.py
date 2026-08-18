"""Triton and hand-written linalg must converge on ONE compiler, not two that agree.

Every other test in this series shows the Triton path works. None of them would catch the failure
that actually matters here: a second lowering stack growing quietly beside the first, producing
right answers by a different route. Correctness cannot detect that — only identity can.

So the same workload is built twice — `build_input_module(reuse=2)`, which predates this work, and
the Triton kernel — and the two descents are compared stage by stage. What is compared is
everything after the frontends meet: interface, target dialect, runtime, and the command buffer.
Anything Triton-specific surviving into those would show up as a difference.

The measured result is stronger than the plan asked for. No canonicalization is needed: the printed
modules are already byte-identical, SSA numbering included. That only holds because the bridge emits
`linalg.quantized_matmul` with zero points — Merlin's own idiom — instead of an equivalent spelling
of its own.
"""
from __future__ import annotations

import json

import pytest
import triton_kernels as K

from merlin.common.paths import repo_root
from merlin.triton import source
from merlin.triton.bridge import to_linalg
from merlin.xdsl_dialects._common import text

GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


@pytest.fixture(scope="module")
def both_paths():
    """The same workload, lowered from a hand-authored module and from Triton."""
    from merlin import compile_core
    from merlin.targetgen.registry import load_target
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    if not GEMMINI_PACKAGE.is_dir():
        pytest.skip("gemmini target package not present")
    package = load_target(GEMMINI_PACKAGE)

    spec = K.repeated_rhs_matmul_spec()
    bridged = to_linalg(source.make_ttir(spec), spec)
    triton_path = compile_core.compile_core_mlir(bridged.module, target_package=package).staged
    # Same shapes, same reuse — and the kernel is deliberately named after this workload, so even
    # the function symbol matches and the comparison needs no renaming step.
    hand_path = lower_repeated_rhs_matmul(reuse=2, m=K.TILE_M, k=K.TILE_K, n=K.TILE_N,
                                          target_package=package)
    return hand_path, triton_path


@pytest.mark.parametrize("stage", ["interface_module", "target_module", "runtime_module"])
def test_the_two_frontends_produce_identical_stage_modules(both_paths, stage):
    hand, triton = both_paths
    assert text(getattr(triton, stage)) == text(getattr(hand, stage))


def test_the_two_frontends_produce_identical_command_buffers(both_paths):
    """The command buffer is what the hardware executes, so identity here is the real claim."""
    hand, triton = both_paths
    assert json.dumps(triton.command_buffer, sort_keys=True) == json.dumps(
        hand.command_buffer, sort_keys=True)


def test_the_input_modules_differ_only_in_accumulator_initialization(both_paths):
    """The convergence above is a result, not a tautology — the two inputs are NOT the same module.

    The bridge zeroes the accumulator with an explicit `linalg.fill`; the hand-authored workload
    passes a bare `tensor.empty`, whose contents are undefined. Both are dropped as support during
    interface materialization, which is precisely why the descents converge despite starting apart.
    """
    hand, triton = both_paths
    hand_text, triton_text = text(hand.input_module), text(triton.input_module)
    assert hand_text != triton_text
    assert "linalg.fill" in triton_text and "linalg.fill" not in hand_text
    assert triton_text.count("linalg.quantized_matmul") == hand_text.count("linalg.quantized_matmul")


def test_convergence_holds_for_the_generic_target_too(both_paths):
    """Not a Gemmini coincidence: the same identity must hold on a structurally different target."""
    from merlin import compile_core
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    spec = K.repeated_rhs_matmul_spec()
    bridged = to_linalg(source.make_ttir(spec), spec)
    triton_path = compile_core.compile_core_mlir(bridged.module, target="saturn").staged
    hand_path = lower_repeated_rhs_matmul(reuse=2, m=K.TILE_M, k=K.TILE_K, n=K.TILE_N,
                                          target="saturn")
    assert text(triton_path.target_module) == text(hand_path.target_module)
    assert json.dumps(triton_path.command_buffer, sort_keys=True) == json.dumps(
        hand_path.command_buffer, sort_keys=True)
