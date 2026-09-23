"""Every region of a model is on a unit, on the host for a stated reason, or a finding."""

from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.perf import gate_phase as GP
from merlin.perf import lowering_coverage as LC
from merlin.perf import placement_census as PC
from merlin.targetgen import eligibility as E

_TARGET = "toy_npu"  # in-tree, no hardware, no oracle


def _admitted_dtype(family: str) -> str:
    return E.capability_map_for_target(_TARGET)[family].dtypes[0]


def _published_ratios(report: dict) -> dict:
    """Every offload ratio a census publishes, read defensively.

    Written to survive a report from EITHER broken era so the mutation tests below fail on the NUMBER
    rather than on a field that did not exist yet. A test that fails with `KeyError` proves only that
    the fix added a key; these have to fail because the old answer was wrong.
    """
    out = {key: value for key, value in report.get("macs", {}).items() if key.startswith("offload_of")}
    out.update({key: value for key, value in (report.get("offload") or {}).items() if key.startswith("offload_of")})
    return out


def _refused_dtype(family: str) -> str:
    """A registry format this target's ``family`` capability REFUSES -- derived, never named here.

    Asking the eligibility oracle itself rather than reading the contract keeps the test honest when
    the contract changes, and keeps a format spelling out of a test that is not about one. An empty
    search is UNKNOWN and fails the test rather than silently selecting nothing.
    """
    cap_map = E.capability_map_for_target(_TARGET)
    for name in qf.names():
        probe = E.RegionDescriptor(op="matmul", family=family, in_dtype=name, m=1, k=1, n=1)
        verdict = E.is_eligible(probe, cap_map)
        if not verdict.eligible and verdict.refusal == "input_dtype":
            return name
    raise AssertionError(f"UNKNOWN: no registry format is refused by {_TARGET}'s {family} capability")


def _regions() -> list[E.RegionDescriptor]:
    dtype = _admitted_dtype("contraction")
    return [
        E.RegionDescriptor(op="matmul", family="contraction", in_dtype=dtype, m=4, k=8, n=2),
        E.RegionDescriptor(op="generic", family="reduction", in_dtype=dtype),
        E.RegionDescriptor(op="generic", family="contraction", in_dtype="fp32", m=2, k=2, n=2),
        E.RegionDescriptor(op="generic", family="elementwise_map", in_dtype=None),
        E.RegionDescriptor(op="mystery", family=None, in_dtype=dtype),
    ]


def _nothing_routes(demands):
    return {"mesh": [], "fallback": [], "scalar_rvv": [], "results": []}


def _routes_everything(demands):
    """A router that takes every demand, however the eligibility oracle judges it.

    The two are separate oracles -- `routing._legal_on` and `eligibility.is_eligible` -- and a caller
    may pass its own placement. When they part company the census must report the disagreement rather
    than divide one population by the other.
    """
    from merlin.targetgen import routing as R

    results = [R.RouteResult(demand=demand, unit="unit", acc=None, gap=None) for demand in demands]
    return {"mesh": list(demands), "fallback": [], "scalar_rvv": [], "results": results}


def test_a_host_placement_names_its_refusal_and_its_owner() -> None:
    report = PC.census(_regions(), _TARGET)
    rows = report["regions"]
    assert rows[0]["placement"] == "accelerator" and rows[0]["unit"]
    assert [row["refusal"] for row in rows[1:]] == [
        "undeclared_family",
        "input_dtype",
        "unexpressed_dtype",
        "unrecognized_family",
    ]
    assert [row["gap_class"] for row in rows[1:]] == ["OG1", "OG7", "OG7", "OG7"]
    assert report["silent_fallbacks"] == [] and report["admitted"]
    # Offload is judged on work, against what was eligible: the fp32 contraction is work the target
    # could not take, so it is in the total and not in the denominator.
    assert report["macs"] == {
        "regions_with_known_extents": 2,
        "regions_with_unknown_extents": 3,
        "total": 72,
        "eligible": 64,
        "on_unit": 64,
        "on_unit_and_eligible": 64,
        "on_unit_not_eligible": 0,
        "eligible_not_on_unit": 0,
        "offload_of_eligible": 1.0,
        "offload_of_total": 0.888889,
    }
    assert report["lane_placement"][0] == {"lane": rows[0]["unit"], "family": "contraction"}
    assert report["lane_placement"][1] == {"lane": "host", "family": "reduction"}


def test_an_unexpressed_precision_is_not_a_fallback_the_compiler_chose() -> None:
    row = PC.census(_regions(), _TARGET)["regions"][3]
    assert not row["eligible"] and "no operand precision" in row["reason"]


def test_eligible_work_nobody_took_is_silent_and_refused() -> None:
    report = PC.census(_regions(), _TARGET, router=_nothing_routes)
    assert report["silent_fallbacks"] == ["0:matmul"] and not report["admitted"]
    assert report["macs"]["offload_of_eligible"] == 0.0
    with pytest.raises(PC.ZeroOffloadError, match="64 eligible"):
        PC.require_offload(report)


def test_a_model_with_no_eligible_work_is_not_a_zero_offload_defect() -> None:
    report = PC.census(_regions()[1:], _TARGET, router=_nothing_routes)
    assert report["macs"]["eligible"] == 0 and report["macs"]["offload_of_eligible"] is None
    PC.require_offload(report)


def test_every_refusal_has_an_owner_and_an_unknown_one_fails_the_gate(monkeypatch) -> None:
    assert set(E.REFUSALS) <= set(PC.GAP_CLASS_OF_REFUSAL)
    monkeypatch.delitem(PC.GAP_CLASS_OF_REFUSAL, "undeclared_family")
    report = PC.census(_regions(), _TARGET)
    assert report["unclassified_refusals"] == [1] and not report["admitted"]


def test_a_model_compile_that_offloads_none_of_its_eligible_work_does_not_succeed(monkeypatch) -> None:
    from merlin import compile_cli

    monkeypatch.setattr(compile_cli, "compile_rvv", lambda *a, **k: {"status": "compiled"})
    report = {"target": "toy_npu", "silent_fallbacks": ["0:matmul"], "macs": {"eligible": 64, "on_unit": 0}}
    monkeypatch.setattr(PC, "census_of_module", lambda module, target, **_: report)
    module = "func.func @f() {\n  return\n}\n"

    out = compile_cli.compile_model(
        "w",
        "int8",
        target="toy_npu",
        run="none",
        verify=False,
        package=None,
        auto_capture=False,
        timeout=5,
        linalg_mlir=module,
    )
    assert out["status"] == "zero_offload" and "64 eligible" in out["reason"]

    # The same compile with the work placed is an ordinary success.
    report["macs"]["on_unit"] = 64
    out = compile_cli.compile_model(
        "w",
        "int8",
        target="toy_npu",
        run="none",
        verify=False,
        package=None,
        auto_capture=False,
        timeout=5,
        linalg_mlir=module,
    )
    assert out["status"] == "compiled" and "offload_refusal" not in out


# --- the census must judge the program the compiler is BUILDING ---------------------------------
#
# A capture is not the program. A model quantized at run time carries `f32` element types on every
# region while the compiler lowers all of it to the target's integer datapath, and this census used
# to read the element types: on a real ResNet-50 it refused all 53 of the model's matmuls with
# `input_dtype 'fp32' not in contraction formats ['int8']` while the router -- asked at the declared
# datapath, as `capsule_source.model_op_demands` has always asked it -- placed 54/54. Coverage read
# 0.004, and `offload_of_eligible` read 1.0 over a denominator of one region.
#
# Graded, that number fails every correct submission. Both mutations below are therefore required:
# the declared datapath must move the coverage, AND a region the target genuinely cannot take must
# stay uncovered for its own reason rather than being swept along.

_PRE_INTEGER_PREPARATION = """builtin.module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<8x2xf32>, %o: tensor<4x2xf32>,
                     %r: tensor<4xf32>) -> tensor<4xf32> {
    %mm = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x2xf32>)
          outs(%o : tensor<4x2xf32>) -> tensor<4x2xf32>
    %red = linalg.reduce ins(%mm : tensor<4x2xf32>) outs(%r : tensor<4xf32>) dimensions = [1]
      (%x: f32, %acc: f32) {
        %s = arith.addf %x, %acc : f32
        linalg.yield %s : f32
      }
    func.return %red : tensor<4xf32>
  }
}"""


def _parse(text: str):
    from merlin.common import mlir_query as mq

    return mq.parse(text)


def _module_census(target: str, **kwargs):
    return PC.census_of_module(_parse(_PRE_INTEGER_PREPARATION), target, **kwargs)


# --- THE MUTATION TESTS: BOTH DIRECTIONS ---------------------------------------------------------
#
# This instrument has been confidently wrong in BOTH directions, and a test pinning the number it
# happened to print would have passed in either broken era. So there are two, and each asserts the
# shape of one failure rather than a value:
#
#   era 1 (pessimistic)  a fully placed model read `coverage: 0.000` with `admitted: True` beside it,
#                        because the census judged a pre-preparation capture at its own element types.
#   era 2 (flattering)   the same census at the declared datapath read `offload_of_eligible: 1.0`
#                        while 99.95% of a real ResNet-50's multiply-accumulates ran on the host --
#                        the 53 convolutions had left the DENOMINATOR by the same verdict that kept
#                        them off the unit, so the ratio measured whether the two oracles agreed and
#                        nothing else. It is 1.0 whenever they agree, however little of the model that
#                        is, which is why it is no longer the headline.


@pytest.mark.parametrize("target", [_TARGET])
def test_a_module_whose_contractions_are_placed_never_reads_as_zero_coverage(target: str) -> None:
    """DIRECTION 1. Fails if the census reports a decided zero for a model the router places.

    The capture spells `f32` because integer preparation has not run; the compile declares the format
    it will lower to. At that format the contraction is placed, and nothing this census publishes may
    read as a measured zero. Asked WITHOUT that declaration it places nothing -- and must then say it
    could not decide, rather than print the zero that falls out.
    """
    # The module as captured, with nothing declared: every region is judged at a format the backend
    # will not place from, so this census has decided NOTHING about the program being compiled.
    undeclared = _module_census(target)
    assert not any(row["placement"] == "accelerator" for row in undeclared["regions"])

    # THE ERA-1 ASSERTION, AND IT FAILS BY VALUE. This field read 0.000 -- a decided zero, with
    # `admitted: True` beside it -- for a model whose router places every contraction it is given.
    assert undeclared["coverage"]["coverage"] is None
    assert not (undeclared["coverage"]["coverage"] == 0.0 and undeclared["admitted"])

    assert [row["refusal"] for row in undeclared["regions"]] == ["input_dtype", "undeclared_family"]
    assert {row["dtype_authority"] for row in undeclared["regions"]} == {PC.FROM_CAPTURE}
    assert undeclared["status"] == GP.STATUS_INCOMPLETE
    assert not undeclared["offload"]["admitted"] and not undeclared["offload"]["blocking"]
    assert undeclared["offload"]["offload_of_addressable"] is None
    # The bound spans everything nobody decided, and nothing else. The contraction is undecided (its
    # refusal rests on a format the backend will not place from), so the ceiling reaches it; the
    # reduce is decidedly unacceleratable -- this target declares no reduction unit at ANY format --
    # so it is ruled out rather than left open, and the ceiling stops short of it.
    assert undeclared["coverage"]["coverage_undecided_regions"] == 1
    assert undeclared["coverage"]["coverage_floor"] == 0.0
    assert undeclared["coverage"]["coverage_ceiling"] == 0.5
    assert undeclared["regions"][1]["off_accelerator_cause"] == LC.NO_UNIT

    # The SAME module at the format the compile declares it will lower to: the contraction is placed,
    # and now every number this census publishes is a measurement rather than a refusal to make one.
    placed = _module_census(target, datapath=_admitted_dtype("contraction"))
    contractions = [row for row in placed["regions"] if row["family"] == "contraction"]
    assert contractions and all(row["placement"] == "accelerator" for row in contractions)
    assert placed["coverage"]["coverage_floor"] > 0.0
    assert placed["coverage"]["coverage"] != 0.0
    assert placed["macs"]["offload_of_total"] > 0.0
    assert placed["offload"]["offload_of_addressable"] > 0.0


def test_a_model_whose_work_a_unit_refused_never_reads_as_full_offload() -> None:
    """DIRECTION 2. Fails if the census reports full coverage for a model with unplaced work.

    One tiny contraction the unit takes and one enormous one it refuses on a property. Every ratio
    whose denominator is the census's OWN verdict reads 1.0 here -- correctly, and uselessly, since
    the refusal removed the refused work from both halves at once. At least one published ratio must
    still see the model: a unit refusing an instance is something only a unit that EXISTS can do, so
    that work is addressable and stays in the denominator.
    """
    taken, refused = _admitted_dtype("contraction"), _refused_dtype("contraction")
    regions = [
        E.RegionDescriptor(op="matmul", family="contraction", in_dtype=taken, m=2, k=2, n=2),
        E.RegionDescriptor(op="matmul", family="contraction", in_dtype=refused, m=100, k=100, n=100),
    ]
    report = PC.census(regions, _TARGET)
    rows = report["regions"]
    assert rows[0]["placement"] == "accelerator"
    assert rows[1]["placement"] == "host" and rows[1]["refusal"] == "input_dtype"

    host_share = 1 - report["macs"]["on_unit"] / report["macs"]["total"]
    assert host_share > 0.99, "the fixture must leave almost all of the model's work on the host"

    # THE ERA-2 ASSERTION, AND IT FAILS BY VALUE. Era 2 published exactly one ratio, it read 1.0, and
    # so NO published number saw the 99.99% of this model that never reached a unit.
    published = _published_ratios(report)
    assert any(value is not None and value < 0.01 for value in published.values()), (
        f"every ratio this census publishes reads as offload the compiler did not achieve: {published}"
    )
    assert report["offload"]["offload_of_addressable"] < 0.01
    assert report["macs"]["offload_of_total"] < 0.01
    # And the agreement ratio is still 1.0 -- kept, and no longer mistakable for coverage.
    assert report["macs"]["offload_of_eligible"] == 1.0
    # Refusing an instance is something only a unit that EXISTS can do, which is why the refused loaf
    # stays addressable instead of becoming an exemption. This is the line the ResNet-50 census had
    # in the wrong place.
    assert rows[1]["off_accelerator_cause"] == LC.REFUSED_ON_PROPERTY


def test_the_headline_number_tells_a_placed_model_from_a_refused_one() -> None:
    """The anti-tautology. One population, two routes, and the number must move between them.

    A ratio that reads the same whatever the compiler did is not an instrument. `offload_of_eligible`
    reads 1.0 for both of these, which is the whole defect; the addressable ratio separates them.
    """
    taken, refused = _admitted_dtype("contraction"), _refused_dtype("contraction")
    small = {"op": "matmul", "family": "contraction", "m": 2, "k": 2, "n": 2}
    large = {"op": "matmul", "family": "contraction", "m": 100, "k": 100, "n": 100}
    # Two models over the same shapes. One the target takes whole; one where it takes the crumb and
    # refuses the loaf on a property -- which is the real ResNet-50 route, in miniature.
    everything = PC.census(
        [E.RegionDescriptor(in_dtype=taken, **small), E.RegionDescriptor(in_dtype=taken, **large)], _TARGET
    )
    crumb_only = PC.census(
        [E.RegionDescriptor(in_dtype=taken, **small), E.RegionDescriptor(in_dtype=refused, **large)], _TARGET
    )

    # THE BY-VALUE ASSERTION. Era 2 published `offload_of_eligible: 1.0` for BOTH of these, because
    # the refusal removed the loaf from the denominator and the numerator at once. A census that
    # cannot tell them apart is not measuring the compiler.
    assert _published_ratios(everything) != _published_ratios(crumb_only), (
        "no number this census publishes tells a model the target ran from one it refused: "
        f"{_published_ratios(everything)}"
    )

    assert everything["offload"]["offload_of_addressable"] == 1.0
    assert crumb_only["offload"]["offload_of_addressable"] < 0.01
    assert everything["status"] == LC.STATUS_OFFLOADED and crumb_only["status"] == LC.STATUS_OFFLOADED
    # The agreement ratio is 1.0 for both, unchanged and still true -- and that is exactly why it was
    # the wrong number to put at the top of the report.
    assert everything["macs"]["offload_of_eligible"] == crumb_only["macs"]["offload_of_eligible"] == 1.0


def test_a_region_on_a_unit_the_oracle_refused_is_a_disagreement_not_a_bonus() -> None:
    """The ratio's numerator can no longer escape its denominator.

    `on_unit` was the router's population and `eligible` was the oracle's; dividing one by the other
    published ratios above 1.0 whenever they parted company. Each disagreement is now reported as
    itself.
    """
    refused = _refused_dtype("contraction")
    regions = [
        E.RegionDescriptor(op="matmul", family="contraction", in_dtype=refused, m=4, k=4, n=4),
        E.RegionDescriptor(op="matmul", family="contraction", in_dtype=_admitted_dtype("contraction"), m=1, k=1, n=1),
    ]
    took_everything = PC.census(regions, _TARGET, router=_routes_everything)
    macs = took_everything["macs"]
    # BY VALUE: this published 65.0 -- 65 multiply-accumulates on a unit over the 1 the oracle
    # admitted. A share of a population cannot exceed it, and a report that prints 6500% offload has
    # divided two different populations.
    assert 0.0 <= macs["offload_of_eligible"] <= 1.0
    assert macs["offload_of_eligible"] == 1.0  # over the ONE region the oracle admitted, and no more
    assert macs["on_unit_not_eligible"] == 64 and macs["on_unit_and_eligible"] == 1
    assert macs["offload_of_total"] == 1.0


@pytest.mark.parametrize("target", [_TARGET])
def test_a_weights_manifest_is_not_ranked_into_declaring_the_datapath(target: str) -> None:
    """`regions_from_module` joins a manifest and falls back to the element type in one expression, so
    no region can afterwards say which set its format. An authority that cannot be attributed per
    region is not one this census claims -- it stays undecided rather than guessing."""
    report = PC.census_of_module(_parse(_PRE_INTEGER_PREPARATION), target, precisions={"nothing.matches": "int8"})
    assert report["coverage"]["coverage"] is None
    assert report["status"] == GP.STATUS_INCOMPLETE


@pytest.mark.parametrize("target", [_TARGET])
def test_the_declared_datapath_moves_the_coverage_and_says_it_did(target: str) -> None:
    datapath = _admitted_dtype("contraction")
    report = _module_census(target, datapath=datapath)
    assert report["datapath"] == datapath
    assert report["coverage"]["coverage"] > 0.0
    contraction = report["regions"][0]
    assert contraction["placement"] == "accelerator" and contraction["refusal"] is None
    # The capture's own spelling survives beside the one the region was judged at.
    assert contraction["dtype"] == datapath and contraction["captured_dtype"] == "fp32"
    assert contraction["dtype_authority"] == PC.FROM_DATAPATH


@pytest.mark.parametrize("target", [_TARGET])
def test_a_region_the_target_cannot_take_stays_uncovered_for_its_own_reason(target: str) -> None:
    """The half that keeps the fix from being a rubber stamp.

    `toy_npu` declares no reduction capability at all, so the reduce is unplaceable at ANY operand
    format. Re-describing it at the datapath must not move it onto a unit, and its refusal must stay
    the one that is true (`undeclared_family`, OG1) rather than becoming a dtype story.
    """
    report = _module_census(target, datapath=_admitted_dtype("contraction"))
    reduction = report["regions"][1]
    assert reduction["placement"] == "host" and not reduction["eligible"]
    assert reduction["refusal"] == "undeclared_family" and reduction["gap_class"] == "OG1"
    assert report["coverage"]["on_host"] == 1


@pytest.mark.parametrize("target", [_TARGET])
def test_an_empty_silent_fallback_list_says_whether_it_was_in_a_position_to_be_complete(target: str) -> None:
    """The field on its way to being graded, and the zero it produces by not having decided.

    A silent fallback is a region the oracle ADMITTED that nothing took. Judged at a format the
    backend will not place from, every region is refused instead, so none can qualify -- and the list
    reads `[]` for a model nobody measured, which is indistinguishable from a clean one. It must say
    which it is.
    """
    undecided = _module_census(target)
    assert undecided["silent_fallbacks"] == [] and undecided["silent_fallbacks_status"] == GP.STATUS_INCOMPLETE

    decided = _module_census(target, datapath=_admitted_dtype("contraction"))
    assert decided["silent_fallbacks"] == [] and decided["silent_fallbacks_status"] == LC.STATUS_OFFLOADED


def test_two_authorities_for_one_operand_format_are_refused_rather_than_ranked() -> None:
    from merlin.common import mlir_query as mq

    with pytest.raises(PC.TwoDtypeAuthorities, match="pass one"):
        PC.census_of_module(mq.parse(_PRE_INTEGER_PREPARATION), _TARGET, precisions={"a.b": "int8"}, datapath="int8")
