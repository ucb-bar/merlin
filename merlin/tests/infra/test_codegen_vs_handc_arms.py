"""The 2x2 that isolates our EMITTED matmul from everything else.

This driver answers the question the whole RVV effort turns on -- how far our codegen is from
hand-written RVV -- by routing every f32 matmul to a hand-C shim in two of its four arms, so the
emitted matmul finally has a call boundary to be measured against. Its recorded result: on rdt2 the
emitted matmul was 14.04 s against the shim's 3.33 s, 4.21x, which was 106.5% of the whole-model gap.

The feature under test used to be a module constant, so the driver could only ever weigh ONE
historical lever and its number went stale the moment a better one landed. These tests pin the
parameterisation and the arm shape, without needing a board.
"""
import importlib.util

from merlin.common.paths import repo_root

_spec = importlib.util.spec_from_file_location(
    "_cvh", repo_root() / "build_tools" / "scripts" / "k1_codegen_vs_handc.py")


def _mod():
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)
    return m


def test_the_arms_form_a_complete_2x2_plus_a_control():
    """Both factors must be crossed, or the subtraction that isolates the matmul is not available."""
    m = _mod()
    arms = m.build_arms("some_feature")
    lowering = {t: bool(f) for t, (f, _kb) in arms.items()}
    matmul = {t: (kb == "ours") for t, (_f, kb) in arms.items()}
    cells = {(lowering[t], matmul[t]) for t in ("A_base", "B_vf", "C_base_shim", "D_vf_shim")}
    assert cells == {(False, False), (True, False), (False, True), (True, True)}, \
        "the 2x2 is incomplete, so codegen_matmul cannot be isolated"
    assert arms["ctrl"] == arms["C_base_shim"], \
        "the control must be C's exact configuration -- it is the board noise floor"


def test_the_feature_under_test_is_a_parameter():
    """Pinning it to one constant is what made the recorded 4.21x go stale."""
    m = _mod()
    for feat in ("accumulator_resident_wholemodel_vf_mrpad", "perop_register_block"):
        arms = m.build_arms(feat)
        assert arms["B_vf"][0] == [feat]
        assert arms["D_vf_shim"][0] == [feat]
        # the baseline arms must stay featureless, or there is no baseline
        assert arms["A_base"][0] == [] and arms["C_base_shim"][0] == []


def test_the_default_reproduces_the_recorded_measurement():
    """An unparameterised re-run must still be a like-for-like read of the recorded cell."""
    m = _mod()
    assert m.DEFAULT_FEATURE == "accumulator_resident_wholemodel_vf"
    assert m.build_arms() == m.build_arms(m.DEFAULT_FEATURE)


def test_every_feature_the_driver_can_be_pointed_at_is_registered():
    """A typo'd feature name would die deep inside a board build, after the arms had been built and
    the toolchain invoked. The registry is the authority and it is cheap to consult."""
    from merlin.llvmlower import impr_features as F
    m = _mod()
    assert m.DEFAULT_FEATURE in F._REGISTRY
    for feat in ("accumulator_resident_wholemodel_vf_mrpad", "perop_register_block"):
        assert feat in F._REGISTRY, f"{feat} is not a registered impr feature"
