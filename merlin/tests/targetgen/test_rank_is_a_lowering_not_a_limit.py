"""A rank the target does not declare is a COMPILER question, not a hardware limit.

`_split_ineligible` withholds a capsule only on a hard structural fact — one no arrangement of the
program can rescue. Operand DTYPE is such a fact: if no datapath holds the format, nothing puts it on
the device. RANK is not, and treating it as one made the suite ask the question a library of kernels
asks ("is this op already shaped like the mesh?") instead of the one a compiler answers ("can I get it
there?").

Measured: `RP14_patch_embed_bf16_pt` was withheld as "rank 4 not in contraction legal ranks [2, 3]" and
never graded, while `linalg_lower.convolution_im2col_matmul` — which derives the conv geometry from the
operand shapes and emits an (m,k,n) matmul — sat in the tree unused. The withhold reported a hardware
limit that did not exist.
"""
from __future__ import annotations

import copy

from merlin.common.paths import merlin_dir
from merlin.targetgen import coverage_report as CR
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.capsule_runner import _split_ineligible

CAPS = merlin_dir() / "contract/capsules"


def _radiance_caps():
    roots = [str(CAPS / "radiance/isa"), str(CAPS / "radiance/model_slices")]
    return list(discover_capsules(roots, labels={"public", "dev"}))


def test_a_rank_the_target_does_not_declare_is_still_graded():
    caps = _radiance_caps()
    if not caps:
        import pytest
        pytest.skip("radiance corpus not present in this checkout")
    keep, withheld = _split_ineligible(caps, "radiance")
    names = {c.get("name") for c in keep}
    rank4 = [c for c in caps
             if (getattr(CR._capsule_region(c), "rank", None) or 0) >= 4]
    assert rank4, "no rank-4 capsule in the corpus; this test would be vacuous"
    for c in rank4:
        assert c.get("name") in names, (
            f"{c.get('name')} withheld for its rank; the compiler lowers rank-4 conv via im2col")


def test_the_compiler_actually_carries_the_lowering_this_relies_on():
    """The claim above is only sound while the conv->im2col->matmul lowering exists. If it is ever
    removed, grading rank-4 capsules stops being justified and this test says so."""
    import inspect

    from merlin.targetgen import linalg_lower as LL
    src = inspect.getsource(LL)
    assert "convolution_im2col_matmul" in src
    assert "linalg.matmul" in src


def test_an_operand_dtype_with_no_datapath_is_STILL_withheld():
    """The hard fact must survive. No lowering makes a mesh hold a format it has no datapath for, so
    this capsule can never pass and grading it would only burn oracle time and mislead the agent."""
    caps = _radiance_caps()
    if not caps:
        import pytest
        pytest.skip("radiance corpus not present in this checkout")
    alien = copy.deepcopy(caps[0])
    alien["name"] = "SYNTH_alien_dtype"
    for key in ("operand_dtype", "dtype"):
        if key in alien:
            alien[key] = "fp64"
    for t in (alien.get("inputs") or []):
        t["dtype"] = "fp64"
    if isinstance(alien.get("operation"), dict):
        alien["operation"].setdefault("attributes", {})["dtype"] = "fp64"

    keep, withheld = _split_ineligible([alien], "radiance")
    assert not keep and len(withheld) == 1, (keep, withheld)
    detail = withheld[0]["failure"]["detail"]
    assert "dtype" in detail and "fp64" in detail, detail
    # and it must not blame the rank, which is no longer a withholding reason at all
    assert "rank" not in detail.split("(eligibility also reports")[0], detail
