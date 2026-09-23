"""A fallback must be counted and justified, or the build does not ship.

Measured on a whole-model ResNet-50: 119 operations ran as host regions -- 50 quantize, 49 min/max
clamp, 17 residual adds, 1 max-pool, 1 reduction -- every one of which the vendor library performs
on the accelerator, and the emitted program said nothing about any of them. On a language model the
same compiler placed 5,558 regions on the host and only declined the program because one of them
exceeded a straight-line budget; a larger budget would have shipped it silently.

The policy encoded here is deliberately NOT "zero host placements". Some are forced -- an ISA with
no vector-map opcode cannot host a standalone element-wise map. A rule demanding zero would be
unsatisfiable and therefore ignored. What it demands is that every fallback NAME its reason, so
"the hardware cannot express this" is distinguishable from "the compiler did not try".
"""

from __future__ import annotations

from merlin.perf.lowering_coverage import ACCELERATOR, HOST, Placement, census, coverage_gate, unjustified

FORCED = "target ISA has no standalone vector-map opcode; expressible only as a fused stage"


def test_coverage_is_the_accelerator_share() -> None:
    rows = [Placement("a", ACCELERATOR), Placement("b", ACCELERATOR), Placement("c", HOST, reason=FORCED)]
    out = census(rows)
    assert out["on_accelerator"] == 2 and out["on_host"] == 1
    assert out["coverage"] == round(2 / 3, 6)


def test_an_unexplained_host_placement_is_flagged() -> None:
    rows = [Placement("conv", ACCELERATOR), Placement("relu", HOST)]
    assert [r.operation for r in unjustified(rows)] == ["relu"]
    assert census(rows)["unjustified_host_operations"] == ["relu"]


def test_a_blank_reason_is_not_a_justification() -> None:
    """'' and '   ' are how an unexplained fallback survives review."""
    rows = [Placement("a", HOST, reason=""), Placement("b", HOST, reason="   "), Placement("c", HOST, reason=FORCED)]
    assert [r.operation for r in unjustified(rows)] == ["a", "b"]


def test_the_no_fallback_policy_refuses_an_unjustified_build() -> None:
    rows = [Placement("conv", ACCELERATOR)] + [Placement(f"q_{i}", HOST) for i in range(50)]
    out = coverage_gate(rows, allow_fallback=False)
    assert out["admitted"] is False
    assert len(out["blocking_operations"]) == 50


def test_the_no_fallback_policy_admits_justified_fallbacks() -> None:
    """A forced fallback is allowed -- it just has to say why."""
    rows = [Placement("conv", ACCELERATOR), Placement("map", HOST, reason=FORCED)]
    out = coverage_gate(rows, allow_fallback=False)
    assert out["admitted"] is True and out["blocking_operations"] == []
    assert out["host_reasons"] == [{"reason": FORCED, "operations": 1}]


def test_permitting_fallback_admits_the_same_build_that_strict_refuses() -> None:
    rows = [Placement("conv", ACCELERATOR), Placement("relu", HOST)]
    assert coverage_gate(rows, allow_fallback=True)["admitted"] is True
    assert coverage_gate(rows, allow_fallback=False)["admitted"] is False


def test_reasons_are_censused_by_frequency() -> None:
    rows = [Placement(f"a{i}", HOST, reason="no vector opcode") for i in range(9)] + [
        Placement("b", HOST, reason="data-dependent index")
    ]
    assert census(rows)["host_reasons"] == [
        {"reason": "no vector opcode", "operations": 9},
        {"reason": "data-dependent index", "operations": 1},
    ]


def test_an_unrecognised_placement_is_surfaced_not_bucketed() -> None:
    """Guessing a lane would make coverage look better than it is."""
    out = census([Placement("a", ACCELERATOR), Placement("mystery", "somewhere_else")])
    assert out["unclassified"] == ["mystery"]
    assert out["on_accelerator"] == 1 and out["on_host"] == 0


def test_by_family_separates_what_reached_the_accelerator() -> None:
    rows = [
        Placement("c1", ACCELERATOR, "contraction"),
        Placement("c2", ACCELERATOR, "contraction"),
        Placement("m1", HOST, "elementwise_map", reason=FORCED),
    ]
    fam = census(rows)["by_family"]
    assert fam["contraction"] == {ACCELERATOR: 2, HOST: 0}
    assert fam["elementwise_map"] == {ACCELERATOR: 0, HOST: 1}


def test_the_measured_resnet50_shape_reports_its_real_coverage() -> None:
    """53 contractions on the array against 116 unexplained element-wise fallbacks."""
    rows = [Placement(f"conv_{i}", ACCELERATOR, "contraction", "i8") for i in range(53)]
    rows += [Placement(f"quantize_{i}", HOST, "elementwise_map", "i8") for i in range(50)]
    rows += [Placement(f"minmax_{i}", HOST, "elementwise_map", "i8") for i in range(49)]
    rows += [Placement(f"add_{i}", HOST, "elementwise_map", "i8") for i in range(17)]
    out = coverage_gate(rows, allow_fallback=False)
    assert out["operations"] == 169 and out["on_accelerator"] == 53
    assert out["coverage"] == round(53 / 169, 6)
    assert len(out["blocking_operations"]) == 116
    assert out["admitted"] is False
