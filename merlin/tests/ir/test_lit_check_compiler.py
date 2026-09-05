"""The derived per-target pass-test generator.

The property that matters is not how many checks come out — it is that **every declared obligation is
either checked or has a recorded reason it is not**. A generator that quietly emits nothing for four
of six targets is indistinguishable from one that is broken, which is why coverage is a ledger rather
than a count.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

TARGETS = ("gemmini", "atlas", "mx_gemmini", "radiance", "saturn_opu", "saturn_opu_rvv")


# Marked slow: deriving RTL facts shells out to the introspection bridge per target, ~15 s cold.
# The fast CI job runs `-m "not slow"`; the cheap single-target properties below stay in it.
@pytest.mark.slow
@pytest.mark.parametrize("target", TARGETS)
def test_every_declared_obligation_is_checked_or_explained(target):
    from merlin.targetgen.lit_check_compiler import compile_checks

    c = compile_checks(target)
    cov = c.coverage
    assert cov["emitted"] + cov["omitted"] == cov["obligations_declared"]
    for o in cov["omission_reasons"]:
        assert len(o["reason"]) > 30, (
            f"{target}/{o['obligation']}: an omission reason must say WHY, not just that it was "
            f"omitted (got {o['reason']!r})")


@pytest.mark.slow
@pytest.mark.parametrize("target", TARGETS)
def test_no_check_is_emitted_from_an_underived_fact(target):
    """A check grounded in a fact the target did not actually yield would be a fabricated constant."""
    from merlin.targetgen.lit_check_compiler import compile_checks

    for check in compile_checks(target).checks:
        if check.derived:
            assert "derived" in check.grounded_by or "manifest" in check.grounded_by, (
                f"{target}/{check.obligation} claims derived=True but names no source")


def test_a_target_without_a_manifest_fails_closed():
    """No manifest must mean 'nothing derivable, here is why', never a silent empty suite."""
    from merlin.targetgen.lit_check_compiler import compile_checks

    c = compile_checks("definitely_not_a_real_target_xyz")
    assert c.checks == []
    assert c.omissions and "no capability manifest" in c.omissions[0].reason


def test_shape_provenance_is_reported():
    """A shape whose provenance is unknown is not evidence.

    Regression: the derived mesh edge can COINCIDE with the fallback shape, so provenance must be
    tracked explicitly rather than inferred by comparing the shape to the default.
    """
    from merlin.targetgen.lit_suite import emit

    cov = emit("gemmini")
    assert "derived" in cov["shape_source"], cov["shape_source"]


def test_generated_test_uses_one_prefix_per_obligation():
    """FileCheck reads one prefix as an ordered sequence, so obligations must not share one.

    Regression: with a shared CHECK prefix the residency block and the commit block were read as a
    single ordered sequence and failed spuriously.
    """
    from merlin.targetgen.lit_check_compiler import compile_checks
    from merlin.targetgen.lit_suite import _shape_for, render_test

    c = compile_checks("gemmini")
    if len(c.checks) < 2:
        pytest.skip("gemmini emits fewer than two checks; nothing to disambiguate")
    shape, _ = _shape_for(c)
    text = render_test(c, shape)
    prefixes = {line.split("--check-prefix=")[1].strip()
                for line in text.splitlines() if "--check-prefix=" in line}
    assert len(prefixes) == len(c.checks), f"expected one prefix per obligation, got {prefixes}"
