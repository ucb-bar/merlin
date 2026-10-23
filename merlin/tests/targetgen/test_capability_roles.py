"""A standalone capability survives a prohibited-roles policy only through evidence that avoids those roles."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from fake_quant_layer import Oracle as _Oracle
from fake_quant_layer import residual_module as _residual

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root
from merlin.targetgen import capability_roles as CR
from merlin.targetgen import compute_units as CU
from merlin.targetgen.rtl import accumulate_fact as AF
from merlin.targetgen.rtl import firrtl_struct as F
from merlin.xdsl_dialects.lowering import compute_groups as CG

NO_FSM = ("loop_descriptor",)
NON_FSM_PATH = {"fact": "accumulate_on_load", "roles": ["operand_load", "config", "readout", "commit"]}
LOOP_ONLY_PATH = {"fact": "loop_resadd", "roles": ["loop_descriptor", "readout"]}


def _cap(*paths, composed=()):
    raw = {"family": "elementwise_map", "dtypes": ["int8"], "standalone_evidence": list(paths)}
    if composed:
        raw = {"family": "elementwise_map", "dtypes": ["int8"], "composed_with": list(composed)}
    return CU._sem_cap(raw, "unit0")


def test_non_fsm_evidence_survives_a_loop_descriptor_prohibition():
    verdict = CR.standalone_admission(_cap(NON_FSM_PATH), prohibited_roles=NO_FSM, review_status="reviewed")
    assert verdict["standalone"] is True and verdict["basis"] == "declared_unchecked"
    assert verdict["paths"][0]["prohibited_roles_used"] == []


def test_loop_fsm_evidence_alone_is_refused_under_the_same_policy():
    verdict = CR.standalone_admission(_cap(LOOP_ONLY_PATH), prohibited_roles=NO_FSM)
    assert verdict["standalone"] is False and "loop_descriptor" in verdict["reason"]
    # The same evidence with no policy in force is admitted; and the non-FSM path rescues it.
    assert CR.standalone_admission(_cap(LOOP_ONLY_PATH), review_status="reviewed")["standalone"] is True
    assert CR.standalone_admission(
        _cap(LOOP_ONLY_PATH, NON_FSM_PATH), prohibited_roles=NO_FSM, review_status="reviewed"
    )["standalone"]


def test_a_path_whose_fact_is_not_derived_is_refused():
    unknown = {"facts": {"accumulate_on_load": {"status": "UNKNOWN"}}}
    derived = {"facts": {"accumulate_on_load": {"status": "derived"}}}
    assert CR.standalone_admission(_cap(NON_FSM_PATH), semantic_facts=unknown)["standalone"] is False
    assert CR.standalone_admission(_cap(NON_FSM_PATH), semantic_facts={"facts": {}})["standalone"] is False
    assert CR.standalone_admission(_cap(NON_FSM_PATH), semantic_facts=derived)["standalone"] is True


def test_composed_and_undeclared_evidence_and_bad_declarations():
    assert CR.standalone_admission(_cap(composed=["contraction"]))["standalone"] is False
    plain = CU._sem_cap({"family": "elementwise_map", "dtypes": ["int8"]}, "unit0")
    assert CR.standalone_admission(plain, prohibited_roles=NO_FSM)["basis"] == "declared_unchecked"
    with pytest.raises(ValueError, match="unknown role"):
        _cap({"fact": "x", "roles": ["teleport"]})
    with pytest.raises(ValueError, match="at once"):
        CU._sem_cap(
            {"family": "elementwise_map", "composed_with": ["contraction"], "standalone_evidence": [NON_FSM_PATH]},
            "u",
        )


def _reviewed_contract() -> dict:
    return yaml.safe_load((repo_root() / "examples/gemmini/target/contracts/target_contract.yaml").read_text())


@pytest.mark.target("gemmini")
def test_the_reviewed_contract_admits_a_standalone_residual_add_without_the_loop_fsm():
    verdicts = CR.contract_admission(_reviewed_contract(), prohibited_roles=NO_FSM, review_status="reviewed")
    add = verdicts["elementwise_map"]
    assert add["standalone"] is True
    assert all("loop_descriptor" not in path["roles"] for path in add["paths"])
    assert "accumulate_on_load" in {p["fact"] for p in add["paths"]}
    assert verdicts["reduction"]["standalone"] is True  # declared sum path; pooling stays composed-only


@pytest.mark.target("gemmini")
def test_the_reviewed_contract_with_only_loop_evidence_would_be_refused():
    contract = copy.deepcopy(_reviewed_contract())
    for unit in contract["compute_units"]:
        for cap in unit.get("semantic_capabilities") or ():
            if cap.get("family") == "elementwise_map":
                cap["standalone_evidence"] = [LOOP_ONLY_PATH]
    assert CR.contract_admission(contract, prohibited_roles=NO_FSM)["elementwise_map"]["standalone"] is False


class _Policy(_Oracle):
    """The fake oracle with a declared elementwise_map capability judged under a policy."""

    def __init__(self, cap, prohibited, **kwargs):
        super().__init__(**kwargs)
        self._unit = CU.ComputeUnit(name="unit0", kind="systolic", semantic_capabilities=(cap,))
        self._prohibited = prohibited

    def standalone_admission(self, family):
        return CR.family_admission([self._unit], family, prohibited_roles=self._prohibited, review_status="reviewed")


def _summing_with(cap, prohibited):
    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",),
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    return _Policy(cap, prohibited, readout=RF.TargetReadout((facet,)))


def test_grouping_places_the_residual_add_only_on_surviving_evidence():
    text = _residual(lhs_scale=1.0, rhs_scale=1.0, out_scale=1.0)
    (placed,) = CG.form_groups(mq.parse(text), "synthetic", oracle=_summing_with(_cap(NON_FSM_PATH), NO_FSM))
    assert placed.placement == "unit0" and placed.operand_sum is not None
    (refused,) = CG.form_groups(mq.parse(text), "synthetic", oracle=_summing_with(_cap(LOOP_ONLY_PATH), NO_FSM))
    assert refused.placement == CG.HOST and refused.operand_sum is None
    assert refused.refusal == CG.OPERAND_SUM and "loop_descriptor" in refused.reason


# --- the RTL fact the non-FSM path names ------------------------------------------------------------

_STORE = """  module Acc :
    input io : { write : { bits : { data : SInt<32>, acc : UInt<1>}}, flip adder : { sum : SInt<32>}}
    reg pipelined : { bits : { data : SInt<32>, acc : UInt<1>}}[2], clock
    inst mem of Mem
    node _T = mux(pipelined[1].bits.acc, io.adder.sum, pipelined[1].bits.data)
    connect mem.io.wdata, _T
"""
_LOAD = """  module Pad :
    input io : { dma : { read : { req : { bits : { laddr : { accumulate : UInt<1>}}}}}}
    wire bank : { write : { bits : { acc : UInt<1>}}}[2]
    inst writer of W
    inst repeater of R
    connect writer.io.req.bits.laddr.accumulate, io.dma.read.req.bits.laddr.accumulate
    node _sel = mux(from_mvin, repeater.io.resp.bits.tag.accumulate, writer.io.resp.bits.laddr.accumulate)
    connect bank[0].write.bits.acc, _sel
"""
_SPEC = {
    "store": {"module": "Acc", "write_data": "mem.io.wdata", "select": "pipelined[*].bits.acc", "sum": "io.adder.sum"},
    "load": {
        "module": "Pad",
        "write_select": "bank[*].write.bits.acc",
        "carried_by": ["repeater.io.resp.bits.tag.accumulate"],
        "request": "io.dma.read.req.bits.laddr.accumulate",
    },
    "roles": ["operand_load"],
}


def _modules(store=_STORE, load=_LOAD):
    def parse(text):
        lines = list(enumerate(text.splitlines(), 1))
        return F.parse_module(lines[0][1].split()[1], lines)

    return {"Acc": parse(store), "Pad": parse(load)}


def test_the_accumulate_on_load_fact_is_read_from_the_store_and_the_load_path():
    value = AF.derive(_SPEC, _modules(), {"LOAD_CMD": ["operand_load"], "LOOP_WS": ["loop_descriptor"]})
    assert value["accumulates_on_load"] is True and value["roles"] == ["operand_load"]
    assert value["instructions"] == {"operand_load": ["LOAD_CMD"]}


@pytest.mark.parametrize(
    "store, load, why",
    [
        (_STORE.replace("io.adder.sum", "io.write.bits.data"), _LOAD, "selects"),  # the store never adds
        (_STORE, _LOAD.replace("repeater.io.resp.bits.tag.accumulate", "from_mvin"), "reads any"),
        (
            _STORE,
            _LOAD.replace("connect writer.io.req.bits.laddr.accumulate, io.dma", "connect writer.x, io.dmx"),
            "never",
        ),
    ],
)
def test_without_the_rtl_structure_the_fact_is_unreadable(store, load, why):
    with pytest.raises(AF.Unreadable, match=why):
        AF.derive(_SPEC, _modules(store, load), {"LOAD_CMD": ["operand_load"]})


def test_roles_the_target_does_not_carry_are_refused():
    with pytest.raises(AF.Unreadable, match="carries role"):
        AF.derive({**_SPEC, "roles": ["weight_load"]}, _modules(), {"LOAD_CMD": ["operand_load"]})


@pytest.mark.target("gemmini")
def test_the_selected_elaboration_derives_accumulate_on_load_without_a_loop_role():
    from merlin.targetgen.rtl import semantic_facts as S

    if S.load_probes("gemmini") is None:
        pytest.skip("no gemmini support provider with semantic probes is selected")
    fact = S.derive("gemmini", only=("accumulate_on_load",))["facts"]["accumulate_on_load"]
    if fact["status"] != S.DERIVED and "unavailable" in str(fact.get("unknown_reason")):
        pytest.skip(f"elaboration unavailable here: {fact.get('unknown_reason')}")
    assert fact["status"] == S.DERIVED, fact
    assert fact["value"]["roles"] == ["operand_load"] and "loop_descriptor" not in fact["value"]["instructions"]
    assert Path(fact["value"]["store"]["cite"]["scala_file"]).name == "AccumulatorMem.scala"
