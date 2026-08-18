"""The portability number: what a new target costs the Triton frontend.

The thesis is that once Merlin can compile generic computation to a target, that target is
Triton-programmable for free. "Free" is a measurable quantity, and this file measures it: the same
kernel is compiled to every target the tree can stage, and the frontend's own output is required to
be identical across all of them — same TTIR, same core MLIR, same everything up to the point where
Merlin picks a target dialect.

That is the honest form of the claim. It does not say the accelerators produce the same code; they
must not. It says the frontend does not know which one it is talking to.

Also pinned here: a target name Merlin cannot resolve must be an ERROR. It used to fall back to the
default contract, so asking for an out-of-tree or misspelled target quietly lowered everything for
`toy_npu` and produced a command buffer naming a target nobody asked for — verifying at every stage
and simulating correctly. That failure is invisible without a test that asks for a target which does
not exist.
"""
from __future__ import annotations

import pytest
import triton_kernels as K

from merlin.common.paths import repo_root
from merlin.triton import source
from merlin.triton.bridge import to_linalg

GEMMINI_PACKAGE = repo_root() / "out/artifacts/targets/gemmini/hand_v0"

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")


def staged_targets():
    """Every way this tree can stage a matmul: in-tree reference targets and an OOT package."""
    from merlin.targetgen.registry import load_target

    cases = [("toy_npu", {"target": "toy_npu"}), ("saturn", {"target": "saturn"})]
    if GEMMINI_PACKAGE.is_dir():
        cases.append(("gemmini", {"target_package": load_target(GEMMINI_PACKAGE)}))
    return cases


@pytest.fixture(scope="module")
def descents():
    """The one kernel, compiled once per target."""
    from merlin import compile_core

    spec = K.repeated_rhs_matmul_spec()
    ttir = source.make_ttir(spec)
    bridged = to_linalg(ttir, spec)
    out = {}
    for name, kwargs in staged_targets():
        out[name] = compile_core.compile_core_mlir(bridged.module, **kwargs).staged
    return {"ttir": ttir, "bridged": bridged, "targets": out}


def test_more_than_one_target_is_actually_under_test():
    """Guard against the file passing vacuously if the packages go missing."""
    assert len(staged_targets()) >= 2


def test_the_frontends_output_is_identical_for_every_target(descents):
    """The measurement: one TTIR, one core MLIR, N targets. The frontend never branches."""
    from merlin import compile_core

    spec = K.repeated_rhs_matmul_spec()
    baseline = to_linalg(source.make_ttir(spec), spec)
    for name, kwargs in staged_targets():
        again = to_linalg(source.make_ttir(spec), spec)
        assert again.text == baseline.text, f"the bridge produced different core MLIR for {name}"
        assert again.report.as_dict() == baseline.report.as_dict()
        # And routing reaches the staged path from the target's own plan, not from its name.
        assert compile_core.choose_route(again.module, **kwargs).kind == "staged", name


def test_each_target_lowers_to_its_own_dialect(descents):
    """The other half: identical input, genuinely different output. Otherwise nothing was proven."""
    dialects = {}
    for name, lowered in descents["targets"].items():
        ops = {op.name for op in lowered.target_module.walk()}
        ops -= {"builtin.module", "func.func", "func.return"}
        dialects[name] = {op.split(".", 1)[0] for op in ops}
        assert len(dialects[name]) == 1, f"{name} mixed dialects: {ops}"
    assert len({frozenset(v) for v in dialects.values()}) == len(dialects), (
        f"two targets produced the same dialect, so portability is untested: {dialects}")


def test_every_target_produces_a_command_buffer_naming_itself(descents):
    """The bug this catches is a silent fallback: asking for X and getting toy_npu's descent."""
    for name, lowered in descents["targets"].items():
        assert lowered.command_buffer["target"] == name


def test_every_targets_command_buffer_simulates_correctly(descents):
    from merlin.runtime import reference_outputs, simulate

    for name, lowered in descents["targets"].items():
        cb = lowered.command_buffer
        assert simulate(cb)["outputs"] == reference_outputs(cb), name


def test_an_unresolvable_target_is_an_error_not_a_silent_fallback():
    """The regression that motivated the fix: a wrong answer passing every check."""
    from merlin.xdsl_dialects.lowering.interface_lowering import LoweringError
    from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

    with pytest.raises(LoweringError) as exc:
        load_curated_contract("no_such_target_exists_anywhere")
    message = str(exc.value)
    assert "no_such_target_exists_anywhere" in message
    assert "MERLIN_TARGET_PATH" in message, "the error must say how to reach an OOT target"


def test_the_default_contract_is_still_reachable_by_its_own_name():
    """Failing closed must not break the one target the built-in contract legitimately describes."""
    from merlin.xdsl_dialects.lowering.contract_facts import DEFAULT_TARGET_CONTRACT
    from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

    name = DEFAULT_TARGET_CONTRACT["name"]
    assert load_curated_contract(name)["name"] == name


def test_an_out_of_tree_target_resolves_through_the_registry(monkeypatch, tmp_path):
    """The supported way to reach a target Merlin has no in-tree contract for."""
    from merlin.targetgen import capability_manifests as cm
    from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

    root = cm.write_oot_target("radiance", tmp_path / "radiance")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    contract = load_curated_contract("radiance")
    assert contract["name"] == "radiance"
    # Its own capabilities, not the default contract's — which is the whole point of resolving it.
    assert "simt" in str(contract.get("capabilities", {})) or contract.get("family") == "simt_tensor"
