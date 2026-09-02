"""The generalization claim: how much of Core ATen the captured models contain, and how much routes.

The claim has to be bounded and checkable, and the ways it can quietly become a slogan are specific:
a denominator that is not actually Core ATen, a numerator that swallows regions nobody could classify,
and a precision gap reported as a routing gap. Each is pinned here.
"""

from __future__ import annotations

import pathlib

import pytest

from merlin.common.paths import artifacts_dir
from merlin.targetgen import aten_coverage as AC


def _captures() -> dict:
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    return {d.name.replace("_fp32_consistent", "").replace("_consistent", ""): d / "model.mlir"
            for d in sorted(root.iterdir()) if (d / "model.mlir").is_file()}


@pytest.fixture(scope="module")
def opset():
    try:
        return AC.core_opset()
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"Core ATen opset needs the m2m venv: {exc}")


def test_the_denominator_is_the_core_opset_not_the_decomposition_table(opset):
    """`torch._decomp.core_aten_decompositions()` LOOKS like the Core ATen opset and is its rough
    complement -- the ops decomposed AWAY on the way to Core ATen. Measured: ~1004 entries against 188
    core-tagged overloads. Using it would report coverage of about a fifth of the true figure against
    a set that is not the one meant."""
    assert opset["n_core"] == len(opset["ops"])
    assert 100 < opset["n_core"] < 400, (
        f"{opset['n_core']} core ops is outside the plausible band; if this is ~1000 the decomposition "
        f"table has been used as the denominator")
    assert all(o.startswith("aten.") for o in opset["ops"])
    assert opset["torch"], "the opset is a property of a torch VERSION and must record which"


def test_the_census_separates_core_from_non_core(opset):
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    cen = AC.census(caps, opset=opset)
    assert cen["n_observed_core"] <= cen["n_core"]
    core = set(opset["ops"])
    assert not (set(cen["non_core_observed"]) & core), (
        "an op cannot be both core and non-core")
    assert set(cen["observed_core"]) <= core
    # A non-core op a model contains is REAL work the compiler must handle. Folding it into the core
    # count would inflate the numerator against a denominator that never contained it.
    for name, d in cen["per_model"].items():
        if d.get("status") == "ok":
            assert not (set(d["non_core"]) & core), name


def test_unclassified_regions_are_never_folded_into_either_bucket(opset):
    """A region whose family could not be determined is evidence neither of coverage nor of a gap.
    The routing denominator is routed + fallback; unclassified sits beside it."""
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    rt = rep["routing"]
    assert rt["denominator"] == rt["routed"] + rt["fallback"]
    assert rt["unclassified"] >= 0
    assert rt["unclassified"] not in (rt["routed"], rt["denominator"]) or rt["unclassified"] == 0


def test_a_precision_gap_is_not_reported_as_a_routing_gap(opset):
    """Precision is judged only over the family-admitted subset. A region whose family is admitted but
    whose dtype is not is a PRECISION gap; reporting it as a routing gap misattributes it -- and on an
    int8-only target against f32 captures that is the majority of regions."""
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    assert "precision" in rep and "routing" in rep
    prec = rep["precision"]
    assert prec["dtype_ok"] + prec["dtype_blocked"] <= rep["routing"]["routed"], (
        "precision is judged over the family-admitted subset, so it cannot exceed it")


def test_the_claim_sentence_carries_only_what_the_numbers_support(opset):
    caps = _captures()
    if not caps:
        pytest.skip("no captured models")
    rep = AC.coverage(caps, "gemmini", opset=opset)
    sentence = AC.claim_sentence(rep)
    assert str(rep["opset"]["n_core"]) in sentence
    assert str(rep["observed"]["n_core_observed"]) in sentence
    assert "unclassified" in sentence, "the honest caveat must survive into the claim"


def test_an_unreadable_capture_is_reported_not_counted_as_zero(opset, tmp_path):
    bad = tmp_path / "missing.mlir"
    rep = AC.coverage({"ghost": bad}, "gemmini", opset=opset)
    assert rep["per_model"]["ghost"]["status"] == "unreadable"
    assert rep["routing"]["denominator"] == 0, (
        "a model we could not read contributes no regions, rather than contributing zeros")


def test_each_half_of_the_claim_names_its_own_model_set(opset, tmp_path):
    """Counting operators and measuring routing can run over DIFFERENT models, and the sentence says so.

    The census reads ``prov.aten`` tags out of the module text; routing has to PARSE the module. So a
    capture that yields its op tags and fails to parse contributes to one half and not the other.
    Measured on the real corpus: smolvla's capture gives 46 ops and raises ParseError in
    ``model_coverage.load_module``, so the operator count came from four models and the routing
    fraction from three -- and one sentence carrying both read as though it were four throughout.
    """
    good = tmp_path / "good.mlir"
    good.write_text(
        'builtin.module {\n  func.func @forward(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {\n'
        '    func.return %a : tensor<4x4xf32>\n  }\n}\n', encoding="utf-8")
    # Tagged like a capture, but not parseable IR: the census can read it, the parser cannot.
    tagless = tmp_path / "unparseable.mlir"
    tagless.write_text('this is not MLIR { prov.aten = "aten.mm.default" }\n', encoding="utf-8")

    rep = AC.coverage({"parses": good, "does_not": tagless}, "gemmini", opset=opset)
    ms = rep["models"]
    assert "does_not" in ms["counted_for_operators"], (
        "a module whose op tags are readable must count toward the operator half")
    assert "does_not" in ms["unparsed_for_routing"]
    assert "does_not" not in ms["measured_for_routing"]
    assert ms["agree"] is False

    sentence = AC.claim_sentence(rep)
    assert f"{len(ms['counted_for_operators'])} captured model(s)" in sentence
    assert f"the {len(ms['measured_for_routing'])} of those" in sentence, (
        "the routing half must state its own denominator, not reuse the operator half's")
    assert "does_not did not parse" in sentence


def test_when_both_halves_share_a_model_set_the_sentence_does_not_caveat(opset, tmp_path):
    """The over-correction mirror: a report whose sets agree must not carry a did-not-parse clause."""
    good = tmp_path / "good.mlir"
    good.write_text(
        'builtin.module {\n  func.func @forward(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {\n'
        '    func.return %a : tensor<4x4xf32>\n  }\n}\n', encoding="utf-8")
    rep = AC.coverage({"parses": good}, "gemmini", opset=opset)
    assert rep["models"]["agree"] is True
    assert "did not parse" not in AC.claim_sentence(rep)


def test_a_frontend_composite_tag_is_classified_not_dismissed(opset, tmp_path):
    """A non-core `prov.aten` tag does NOT mean the capture is at the wrong IR level.

    `prov.aten` records the FRONTEND op -- which is what a provenance tag is for. Measured on
    `resnet50_v1_5`: every one of its eight tags is a composite (`aten.conv2d.default`,
    `aten.linear.default`, `aten.batch_norm.default`, ...), and its linalg is nevertheless fully
    decomposed -- 688 contraction regions, 54 `linalg.matmul` reached through im2col, batch-norm
    lowered to `linalg.generic`. An earlier reading of this same data concluded the capture was at the
    torch level and needed re-capturing; it did not.

    So the census classifies instead: a composite torch's decomposition table can take apart is a
    different fact from an op nothing can name. The in-place and aliasing variants (`relu_`, `add_`,
    `flatten.using_ints`) land in the second bucket deliberately -- guessing that `relu_` is `relu` is
    the name-matching this repo exists to avoid.
    """
    core = set(opset["ops"])
    decomposed = set(opset.get("decomposed") or ())
    if not decomposed:
        pytest.skip("this torch build exposes no decomposition table")

    composites = ["aten.conv2d.default", "aten.linear.default", "aten.max_pool2d.default",
                  "aten.adaptive_avg_pool2d.default"]
    variants = ["aten.relu_.default", "aten.add_.Tensor", "aten.flatten.using_ints"]
    for op in composites + variants:
        assert op not in core, f"{op} is not a core op; the opset must not contain it"
    for op in composites:
        assert op in decomposed, f"{op} should be a known frontend composite"

    mod = tmp_path / "frontend_level.mlir"
    mod.write_text("".join(f'  %x = "op"() {{prov.aten = "{op}"}} : () -> ()\n'
                           for op in composites + variants), encoding="utf-8")
    cen = AC.census({"frontend": mod}, opset=opset)
    assert cen["n_observed_core"] == 0, "none of these tags is core"
    assert sorted(cen["composite_observed"]) == sorted(composites), (
        "a composite whose lowering is core must be reported as such, not merely as 'not core'")
    assert sorted(cen["unclassified_observed"]) == sorted(variants), (
        "an in-place or aliasing variant is neither core nor decomposable, and must not be folded "
        "into the composite bucket")
    assert not (set(cen["composite_observed"]) & set(cen["unclassified_observed"]))


def test_a_parse_failure_names_the_construct_not_just_the_exception(opset, tmp_path):
    """`ParseError: <path>:17083:5` and "the capture is corrupt" license opposite actions.

    One says re-capture the model; the other says teach the parser a form it does not read. Measured on
    `smolvla_fp32_consistent`: valid MLIR that xDSL 0.68.0 refuses, because its `linalg.generic`
    assembly accepts only the single-result `-> tensor<...>` form while a fused argmin yields two
    results. One construct in a 4.3 MB module cost that model's entire routing evidence, and the report
    said only "ParseError".

    The reported line is the failure POINT and the line before it, because a parser that rejects an op
    reports where it gave up -- which is the start of the next statement, not the offending one.
    """
    bad = tmp_path / "unreadable.mlir"
    bad.write_text(
        'builtin.module {\n'
        '  func.func @forward(%a: tensor<4xi64>) -> tensor<i64> {\n'
        '    %e = tensor.empty() : tensor<i64>\n'
        '    %f = tensor.empty() : tensor<i64>\n'
        '    %0, %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, '
        'affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} '
        'ins(%a : tensor<4xi64>) outs(%e, %f : tensor<i64>, tensor<i64>) {\n'
        '    ^bb0(%x: i64, %y: i64, %z: i64):\n'
        '      linalg.yield %x, %y : i64, i64\n'
        '    } -> (tensor<i64>, tensor<i64>)\n'
        '    func.return %0 : tensor<i64>\n'
        '  }\n'
        '}\n', encoding="utf-8")

    rep = AC.coverage({"multi_result": bad}, "gemmini", opset=opset)
    d = rep["per_model"]["multi_result"]
    assert d["status"] == "unreadable"
    assert isinstance(d.get("line"), int) and d["line"] > 0, "the location must be structural"
    # The construct the parser refused ends on the line BEFORE where it gave up.
    assert "linalg.generic" in d.get("after", "") or "-> (tensor" in d.get("after", ""), (
        f"the report must show the construct, got after={d.get('after')!r}")
    assert rep["routing"]["denominator"] == 0, "an unreadable model contributes no regions"
