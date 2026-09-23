"""The conv capsules that require a quantized epilogue to ride the mesh store path.

WHY THEY EXIST, measured. On the ResNet-50 capture all 53 convolutions DO route on-mesh -- the
contraction was never the problem. The EPILOGUE is: 50 quantize regions, 49 minmax regions and 16
elementwise regions stay on the host, 91.68% of all host IR work, and 116 of the 119 host refusals
are contradicted by the target's own derived capability. Nothing caught it because ZERO capsules
required a conv's requant/bias/relu to ride the store path: the conv capsules that existed declared
no epilogue at all, so a backend that commits the raw accumulator and lets the host finish the job
matched every one of them exactly.

WHAT THESE TESTS PIN, and what they deliberately do not. They pin the CONTRACT the four capsules
assert -- the three stages, in the ABI's order, at the narrow commit dtype, with the bias declared as
a real operand and the mesh lane required -- because each of those is a property a regeneration or a
profile edit can silently drop, and every one of them is the difference between a capsule that tests
the gap and one that reads as if it did. They do not assert that the capsules PASS: they are expected
to fail against the reference backend in this tree today (its ``InterfaceToGemmini`` conv lowering
carries ``epilogue``, ``output_dtype`` and ``acc_scale`` onto the commit and carries no ``bias`` at
all), and a corpus addition whose point is to be failable must not be pinned green.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir

#: The stages, in the order the command-buffer ABI applies them. Order is load-bearing: bias lands on
#: the accumulator BEFORE the readout multiply, and a backend that scales first computes a different
#: number from the golden while emitting the same three stage names.
EPILOGUE = ["bias_add", "acc_scale", "relu"]

NAMES = (
    "GQ0_conv2d_requant_relu_i8",
    "GQ1_conv2d_1x1_requant_relu_i8",
    "GQ2_conv2d_pad_requant_relu_i8",
    "GQ3_conv2d_stride2_requant_relu_i8",
)


def _profile() -> dict:
    path = merlin_dir() / "contract" / "capsules" / "profiles" / "gemmini.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entries() -> dict:
    return {e["name"]: e for e in _profile()["capsules"] if isinstance(e, dict) and e.get("name") in NAMES}


def _generated(name: str) -> dict:
    path = merlin_dir() / "contract" / "capsules" / "layers" / name / "capsule.yaml"
    assert path.is_file(), f"{name} has not been generated; run generate_corpus.py --target gemmini"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_family_exists_and_is_a_family() -> None:
    """Four capsules, not one. A single shape would be the defect GC7/GC8 were added to fix, restated:
    every conv capsule in this corpus held padding, stride and window at one identical default."""
    assert set(_entries()) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_declares_the_three_stage_quantized_epilogue_on_the_conv_itself(name) -> None:
    entry = _entries()[name]
    assert entry["op"] == "conv2d" and entry["cat"] == "layers"
    assert entry["epilogue"] == EPILOGUE, "the ABI order is the contract, not a set of names"
    assert isinstance(entry.get("acc_scale"), float), "an acc_scale stage with no multiplier is read as 1.0"


@pytest.mark.parametrize("name", NAMES)
def test_each_requires_the_mesh_lane(name) -> None:
    """The half that makes this a claim about PLACEMENT. Without it a backend that computes the whole
    epilogue on the host still matches the golden bit for bit and passes."""
    assert _entries()[name]["lanes"] == {"require": ["on_mesh"]}


def test_every_member_carries_a_different_multiplier() -> None:
    """A backend that reads the scale from one capsule and hardcodes it must fail the other three."""
    scales = [_entries()[n]["acc_scale"] for n in NAMES]
    assert len(set(scales)) == len(scales), scales


def test_the_geometry_axes_are_actually_varied() -> None:
    """Padding, stride and window, each exercised by at least one member and not by all of them --
    otherwise the family is one shape with four names."""
    entries = _entries()
    assert any(e.get("padding", [0, 0, 0, 0]) != [0, 0, 0, 0] for e in entries.values())
    assert any(e.get("stride", [1, 1]) != [1, 1] for e in entries.values())
    windows = {(e.get("kh", 3), e.get("kw", 3)) for e in entries.values()}
    assert len(windows) > 1, f"every member uses the same window {windows}"


@pytest.mark.parametrize("name", NAMES)
def test_the_generated_capsule_commits_narrow_and_declares_the_bias_as_an_operand(name) -> None:
    """Both halves are derived rather than typed in the profile, and both are what a backend that
    skips the stage gets wrong.

    The commit dtype comes from the target's own ``requant_output_dtype``: a backend that drops the
    narrowing commits the wrong WIDTH, not merely the wrong values. The bias is a declared tensor with
    the ACCUMULATOR's dtype, because that is the domain the addition happens in -- declaring it at the
    operand width would describe a different computation from the one the golden performs.
    """
    capsule = _generated(name)
    attrs = capsule["operation"]["attributes"]
    assert capsule["operation"]["op"] == "conv2d"
    assert attrs["epilogue"] == EPILOGUE
    assert attrs["output_dtype"] == "i8"
    bias = next((i for i in capsule["inputs"] if i.get("role") == "bias"), None)
    assert bias is not None, "a bias_add stage with no bias operand reaches for a tensor nobody declared"
    assert bias["dtype"] == "i32" and attrs["bias"] == bias["name"]


@pytest.mark.parametrize("name", NAMES)
def test_the_generated_capsule_keeps_the_lane_and_the_acceleration_demand(name) -> None:
    """``lanes.require`` survived generation AND ``must_accelerate`` is on, so an eligible region that
    falls back to the host is a hard failure rather than a tolerated route."""
    capsule = _generated(name)
    assert capsule["lanes"] == {"require": ["on_mesh"]}
    assert capsule["semantic"]["must_accelerate"] is True
    assert "elementwise_map" in capsule["semantic"]["composed_families"], (
        "the epilogue is what credits the fused-only family; without it these are ordinary convs"
    )


#: The tightest per-engine certification ceiling this target has measured, in written output elements
#: (`conformance._cert_affordability`, verilator, 300 s budget, 2026-09-21). Quoted rather than
#: recomputed because recomputing it walks the whole corpus; it is a BOUND here, so a ceiling that has
#: since moved up only makes this test more permissive and never wrongly red.
_MEASURED_CERT_CEILING_ELEMENTS = 923


@pytest.mark.parametrize("name", NAMES)
def test_each_member_is_small_enough_that_someone_can_certify_it(name) -> None:
    """A capsule nobody can run cycle-accurately demands certification and never gets it, which is the
    same as not testing the gap while reporting that it must be tested. Every member commits well
    inside the tightest per-engine ceiling this target has measured."""
    text = (merlin_dir() / "contract" / "capsules" / "layers" / name / "capsule.interface.mlir").read_text()
    # The committed extent is the result type of the conv line: `... -> tensor<RxCxi8>`. Read
    # structurally off the last arrow rather than pattern-matched.
    result = text.rsplit("-> tensor<", 1)[1].split(">", 1)[0]
    rows, cols, _dtype = result.split("x", 2)
    written = int(rows) * int(cols)
    assert 0 < written <= _MEASURED_CERT_CEILING_ELEMENTS, f"{name} commits {written} elements"


@pytest.mark.parametrize("name", NAMES)
def test_no_member_is_pinned_as_passing(name) -> None:
    """The point of the family is that it can fail. A capsule added with an expectation block that
    already describes today's behaviour would test nothing -- which is exactly how the corpus came to
    have four convolutions and no epilogue coverage."""
    capsule = _generated(name)
    assert "expected" in capsule and capsule["expected"]["modes"]["acc_scale"] is True
    assert capsule["expected"]["modes"]["relu"] is True
