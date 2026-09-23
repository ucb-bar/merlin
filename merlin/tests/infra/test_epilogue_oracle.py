"""The gap between what a command buffer asked the store path for and what the target admits.

Every test here is a MUTATION: the rule is exercised by changing exactly one thing and requiring the
verdict to move. A rule that cannot fire measures nothing, and a rule nobody can satisfy gets routed
around -- so both directions are pinned, and so is the third outcome that is neither.
"""

from __future__ import annotations

import json

import fake_quant_layer as FQ
import pytest

from merlin.common import mlir_query as mq
from merlin.common.paths import merlin_dir
from merlin.perf import epilogue_oracle as EO

#: What a store path that applies bias, requantization and an activation licenses, in the ABI's
#: words. Stated by the test rather than read from a target: the oracle takes it as a parameter, so
#: a second target supplies its own and edits nothing shared (see the generalization test below).
_LICENSED = ("bias_add", "acc_scale", "relu", "maxpool")


def _module():
    return mq.parse(FQ.module(weight_dequantize="per_tensor"))


#: The fixture's stored tensor, by model-argument index: ``@forward(%x, %w, %ws, %wz, %b, %skip)``.
_WEIGHT_ARGS = (1,)


def _admitted(oracle=None):
    got = EO.admissions(_module(), "synthetic", oracle=oracle or FQ.Oracle(), weight_args=_WEIGHT_ARGS)
    assert not got["unstatable"], got["unstatable"]
    return got["sites"]


def _buffer(stages, *, declined=None, opcode="COMMIT"):
    command = {"opcode": opcode, "operands": {"src": "acc", "dst": "out"}, "attributes": {"epilogue": list(stages)}}
    return {"abi_version": "0.1", "target": "synthetic", "commands": [command], "declined": declined}


# --- the admissible set is the target's, not the test's -----------------------------------------
def test_the_admission_is_what_the_target_closed_the_group_around() -> None:
    sites = _admitted()
    assert len(sites) == 1
    assert sites[0].stages == ("bias_add", "acc_scale", "relu")


def test_a_target_that_refuses_the_scale_admits_less_and_the_denominator_follows() -> None:
    """Same model, one capability changed: the readout no longer holds the group's granularity."""
    narrow = _admitted(FQ.Oracle(holds=("channel",)))
    assert narrow[0].stages == ("bias_add", "relu")  # growth stopped before the requantize
    report = EO.compare(EO.asks(_buffer([])["commands"]), narrow, licensed=_LICENSED)
    assert report["stages"]["admissible"] == 2


# --- the two mutations the rule exists for ------------------------------------------------------
def test_a_buffer_that_omits_a_licensed_stage_is_reported_as_a_gap() -> None:
    report = EO.compare(EO.asks(_buffer(["bias_add"])["commands"]), _admitted(), licensed=_LICENSED)
    assert report["verdict"] == EO.VERDICT_GAP
    assert report["stages"] == {"admissible": 3, "asked": 1, "omitted": 2}
    assert report["epilogue_share_on_store_path"] == pytest.approx(1 / 3)
    assert report["sites"][0]["omitted"] == ["acc_scale", "relu"]
    assert report["omitted_sites"] == [0] and report["silent_fallbacks"] == [0]


def test_a_buffer_that_carries_every_licensed_stage_is_clean() -> None:
    report = EO.compare(
        EO.asks(_buffer(["bias_add", "acc_scale", "relu"])["commands"]), _admitted(), licensed=_LICENSED
    )
    assert report["verdict"] == EO.VERDICT_CLEAN
    assert report["epilogue_share_on_store_path"] == 1.0
    assert report["omitted_sites"] == [] and report["silent_fallbacks"] == []


def test_an_omission_the_buffer_declared_is_not_a_silent_fallback() -> None:
    buffer = _buffer(["bias_add"], declined={"reason": "no readout multiplier was derivable"})
    report = EO.compare(EO.asks(buffer["commands"]), _admitted(), licensed=_LICENSED, declined=buffer["declined"])
    assert report["verdict"] == EO.VERDICT_GAP and report["omitted_sites"] == [0]
    assert report["silent_fallbacks"] == []


# --- incomplete is a status, never a pass -------------------------------------------------------
def test_a_target_whose_readout_licenses_nothing_cannot_be_scored() -> None:
    report = EO.compare(EO.asks(_buffer([])["commands"]), _admitted(), licensed=None)
    assert report["verdict"] == EO.VERDICT_INCOMPLETE
    # NOT 0.0. A share nobody could compute must not read as the worst possible score, or a missing
    # toolchain fails every correct submission.
    assert report["epilogue_share_on_store_path"] is None
    assert report["stages"] == {"admissible": None, "asked": None, "omitted": None}


def test_a_capture_that_admits_no_licensed_stage_is_incomplete_and_not_clean() -> None:
    """The failure this module exists to prevent: nothing asked, nothing licensed, same number.

    The site pairs cleanly and the buffer asks for nothing; every stage the group carries is one
    this target does not license, so the denominator is empty. ``clean`` here would say the compiler
    did everything available to it, on evidence that says nothing at all.
    """
    admitted = [EO.Admission(index=0, op="matmul", stages=("maxpool",), extents=None)]
    report = EO.compare(EO.asks(_buffer([])["commands"]), admitted, licensed=("relu",))
    assert report["verdict"] == EO.VERDICT_INCOMPLETE and "admits any licensed stage" in report["why"]
    assert report["epilogue_share_on_store_path"] is None


def test_an_alignment_the_extents_contradict_is_refused_rather_than_paired() -> None:
    sites = _admitted()
    mismatched = [EO.Admission(index=0, op="matmul", stages=sites[0].stages, extents=(999, 999))]
    buffer = {"commands": [{"opcode": "CONV2D", "attributes": {"epilogue": [], "kernel": [1, 1, 8, 16]}}]}
    report = EO.compare(EO.asks(buffer["commands"]), mismatched, licensed=_LICENSED)
    assert report["verdict"] == EO.VERDICT_INCOMPLETE and "not the same program" in report["why"]


def test_a_buffer_site_with_no_admission_is_undecided_rather_than_clean() -> None:
    buffer = {"commands": [{"opcode": "COMMIT", "attributes": {"epilogue": []}}] * 2}
    report = EO.compare(EO.asks(buffer["commands"]), _admitted(), licensed=_LICENSED)
    assert report["verdict"] == EO.VERDICT_INCOMPLETE and "no admission" in report["why"]


def test_admitted_work_the_buffer_has_no_command_for_is_a_gap_at_an_absent_site() -> None:
    report = EO.compare(EO.asks({"commands": []}["commands"]), _admitted(), licensed=_LICENSED)
    assert report["verdict"] == EO.VERDICT_GAP
    assert report["alignment"]["absent_sites"] == 1
    assert report["omitted_sites"] == ["group_0"] and report["sites"][0]["site"] is None


def test_a_stage_the_abi_has_no_name_for_is_refused_rather_than_counted() -> None:
    with pytest.raises(EO.OracleError, match="epilogue vocabulary"):
        EO.asks([{"opcode": "COMMIT", "attributes": {"epilogue": ["fused_softmax"]}}])


# --- what the agent is allowed to see -----------------------------------------------------------
def test_the_agent_view_hands_over_the_gap_and_never_the_grouping() -> None:
    report = EO.compare(EO.asks(_buffer(["bias_add"])["commands"]), _admitted(), licensed=_LICENSED)
    view = EO.agent_view(report)
    assert view["verdict"] == EO.VERDICT_GAP
    assert view["sites"] == [{"site": 0, "opcode": "COMMIT", "omitted_stage_count": 2, "has_store_path_command": True}]
    # Nothing that would answer the question the experiment asks: which operations close with which
    # contraction, at what geometry, with what multiplier. The aggregate counts ARE the score and
    # are disclosed; what is withheld is every per-site statement of the grouping.
    site_keys = {key for row in view["sites"] for key in row}
    assert site_keys.isdisjoint({"group", "op", "admissible", "asked", "extents", "omitted"})
    assert "acc_scale" not in json.dumps(view["sites"])
    assert "acc_scale" in json.dumps(EO.agent_view(report, detail=True)["sites"]), "detail=True names them"


# --- zero core lines: a second target plugs in through the parameters ---------------------------
def test_a_second_target_is_scored_with_no_edit_to_shared_code() -> None:
    """A target that holds no requantization at all, and whose store path applies only an activation.

    It supplies an oracle and a licensed vocabulary -- both already parameters -- and nothing in
    ``merlin/python/merlin`` changes. Its admissible set, its denominator and its verdict all track
    its own contract.
    """
    other = FQ.Oracle(holds=("channel",))  # its readout does not hold this group's per-tensor scale
    sites = _admitted(other)
    licensed = ("relu",)  # ...and its store path applies an activation and nothing else
    gapped = EO.compare(EO.asks(_buffer([])["commands"]), sites, licensed=licensed)
    assert gapped["verdict"] == EO.VERDICT_GAP and gapped["stages"] == {"admissible": 1, "asked": 0, "omitted": 1}
    clean = EO.compare(EO.asks(_buffer(["relu"])["commands"]), sites, licensed=licensed)
    assert clean["verdict"] == EO.VERDICT_CLEAN and clean["epilogue_share_on_store_path"] == 1.0
    # A stage this target does not license is out of BOTH sides of the ratio, not a free pass.
    assert "bias_add" not in clean["sites"][0]["admissible"]


# --- the recorded emission -----------------------------------------------------------------------
def test_the_deployable_emissions_own_capture_admits_nothing_and_is_not_scored() -> None:
    """The regression pin, on tracked data only.

    The ResNet-50 package whose 109-command buffer this repo records was compiled from a capture
    whose activation scale is computed at RUN TIME. There is no compile-time multiplier to place on
    a store path, so the grouper admits no site -- and the oracle must say so rather than score the
    compiler for it. This is the case a two-valued gate gets wrong in the expensive direction.
    """
    capsule = merlin_dir() / "contract/capsules/model/SY_model_resnet50/capsule.interface.mlir"
    sites = json.loads((merlin_dir() / "tests/data/recorded_conv_sites_resnet50.json").read_text())["sites"]
    assert len(sites) == 53 and all("epilogue" not in site["attributes"] for site in sites)
    report = EO.gap(sites, mq.parse(capsule.read_text()), "gemmini")
    assert report["verdict"] == EO.VERDICT_INCOMPLETE
    assert report["epilogue_share_on_store_path"] is None
