"""The scalar-residue vectorization lever is explicit and default-off."""
from merlin.llvmlower import impr_features as features
from merlin.llvmlower.residual_autovec import FEATURE, ensure_registered


def test_default_flags_are_byte_for_byte_unchanged():
    flags = ["-march=rv64gcv", "-O2", "-fno-vectorize", "-fno-slp-vectorize"]
    assert features.apply_cflags(flags, frozenset()) == flags


def test_feature_removes_only_global_vectorizer_disables():
    flags = [
        "-march=rv64gcv", "-O2", "-fno-vectorize", "-fno-slp-vectorize",
        "-fno-unroll-loops",
    ]
    enabled = features.normalize([ensure_registered()])
    assert FEATURE in enabled
    assert features.apply_cflags(flags, enabled) == [
        "-march=rv64gcv", "-O2", "-fno-unroll-loops",
    ]


def test_feature_is_idempotent_when_package_does_not_disable_vectorization():
    flags = ["-march=rv64gcv", "-O2"]
    enabled = features.normalize([ensure_registered()])
    assert features.apply_cflags(flags, enabled) == flags
