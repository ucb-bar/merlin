"""The per-form exactness contract: exact by default, a bounded form only as declared, enforced exactly as
declared by every grader, and recorded as the contract that was applied."""

from __future__ import annotations

import numpy as np
import pytest

from merlin.perf import exactness as EX
from merlin.perf import whole_model_verdict as V

BOUNDED_CONV = {
    "name": "device_requant_conv",
    "match": {"op": "conv2d"},
    "exactness": {"mode": "bounded", "bound": {"max_abs_lsb": 1}, "reason": "the device rounds half to even"},
}


def _contract(*forms, **top) -> EX.Contract:
    return EX.Contract(
        {"schema": EX.SCHEMA, "target": "toy", "default": {"mode": "exact"}, "forms": list(forms), **top}
    )


# ------------------------------------------------------------------------------------------ the contract


@pytest.mark.parametrize(
    "document, refused",
    [
        ({"default": {"mode": "bounded"}}, "default is exact"),
        ({"forms": [{**BOUNDED_CONV, "exactness": {"mode": "bounded", "bound": {"max_abs_lsb": 1}}}]}, "reason"),
        (
            {"forms": [{**BOUNDED_CONV, "exactness": {**BOUNDED_CONV["exactness"], "bound": {"max_abs_lsb": 0}}}]},
            "at least 1",
        ),
        ({"forms": [{**BOUNDED_CONV, "exactness": {"mode": "exact", "bound": {"max_abs_lsb": 1}}}]}, "states no bound"),
        (
            {
                "forms": [
                    {
                        **BOUNDED_CONV,
                        "exactness": {**BOUNDED_CONV["exactness"], "bound": {"max_abs_lsb": 1, "max_fraction": 2}},
                    }
                ]
            },
            "fraction",
        ),
        ({"forms": [BOUNDED_CONV, BOUNDED_CONV]}, "own non-empty name"),
        ({"forms": [{**BOUNDED_CONV, "match": {}}]}, "at least one form field"),
        ({"forms": [], "loosen_everything": True}, "unknown top-level"),
        ({"forms": None}, "forms"),
    ],
)
def test_a_contract_that_would_loosen_implicitly_or_vaguely_is_refused(document, refused):
    base = {"schema": EX.SCHEMA, "target": "toy", "default": {"mode": "exact"}, "forms": []}
    with pytest.raises((EX.ExactnessError, ValueError), match=refused):
        EX.Contract({**base, **document})


def test_a_form_no_entry_names_is_exact_and_an_unseeable_match_never_loosens():
    contract = _contract(BOUNDED_CONV)
    assert contract.resolve({"op": "conv2d"}).label() == "bounded(<=1 LSB)"
    assert contract.resolve({"op": "matmul"}) == EX.EXACTLY
    assert _contract({**BOUNDED_CONV, "match": {"op": "conv2d", "geometry": "strided"}}).resolve({"op": "conv2d"}) == (
        EX.EXACTLY
    )
    assert contract.resolve(None).label() == "exact"


def test_an_ops_own_tolerance_applies_unless_the_contract_names_its_form_either_way():
    residual = {"op": "residual_add"}
    assert EX.Contract.default().resolve(residual, op_bound_lsb=1).declared_by == EX.DECLARED_OP
    tightened = _contract({"name": "exact_residual", "match": {"op": "residual_add"}, "exactness": {"mode": "exact"}})
    assert tightened.resolve(residual, op_bound_lsb=1).label() == "exact"


def test_two_entries_that_disagree_are_ambiguous_not_a_choice():
    other = {**BOUNDED_CONV, "name": "wider", "exactness": {**BOUNDED_CONV["exactness"], "bound": {"max_abs_lsb": 3}}}
    with pytest.raises(EX.ExactnessError, match="ambiguous"):
        _contract(BOUNDED_CONV, other).resolve({"op": "conv2d"})


def test_the_contract_round_trips_by_value_and_detects_an_edit():
    contract = _contract(BOUNDED_CONV)
    carried = contract.to_document()
    assert EX.Contract.from_value(carried).semantics_sha256 == contract.semantics_sha256
    carried["document"]["forms"][0]["exactness"]["bound"]["max_abs_lsb"] = 5
    with pytest.raises(EX.ExactnessError, match="does not hash"):
        EX.Contract.from_value(carried)
    renamed = _contract({**BOUNDED_CONV, "name": "another_name"})
    assert renamed.semantics_sha256 == contract.semantics_sha256 and renamed.sha256 != contract.sha256


# ------------------------------------------------------------------------------------------ grading


def test_exact_rejects_a_one_lsb_difference():
    want = np.arange(64, dtype=np.int64)
    got = want.copy()
    got[17] += 1
    assert EX.judge_arrays(want, want, EX.EXACTLY)["passed"] is True
    graded = EX.judge_arrays(got, want, EX.EXACTLY)
    assert graded["passed"] is False and graded["evidence"] == {"max_abs": 1, "mismatches": 1, "elements": 64}


def test_bounded_accepts_within_its_bound_and_rejects_beyond_it():
    bounded = _contract(BOUNDED_CONV).resolve({"op": "conv2d"})
    want = np.zeros(100, dtype=np.int64)
    within = want.copy()
    within[:30] = 1
    beyond = within.copy()
    beyond[5] = 2
    assert EX.judge_arrays(within, want, bounded)["passed"] is True
    rejected = EX.judge_arrays(beyond, want, bounded)
    assert rejected["passed"] is False and "exceeds bounded(<=1 LSB)" in rejected["why"]


def test_a_fraction_bound_counts_the_elements_that_differ():
    fraction = EX.Exactness(mode=EX.BOUNDED, max_abs_lsb=1, max_fraction=0.05, reason="r", declared_by="form:f")
    want = np.zeros(100, dtype=np.int64)
    few, many = want.copy(), want.copy()
    few[:5], many[:6] = 1, 1
    assert EX.judge_arrays(few, want, fraction)["passed"] is True
    assert EX.judge_arrays(many, want, fraction)["passed"] is False
    assert fraction.label() == "bounded(<=1 LSB on <=0.05 of elements)"


def test_evidence_that_cannot_show_a_bound_held_fails_closed():
    bounded = EX.Exactness(mode=EX.BOUNDED, max_abs_lsb=1, reason="r")
    count_only = EX.judge(bounded, mismatches=3, elements=100)
    assert count_only["passed"] is False and count_only["verifiable"] is False
    differs = EX.judge(bounded, equal=False)
    assert differs["passed"] is False and differs["verifiable"] is False
    assert EX.judge(bounded, equal=True)["passed"] and EX.judge(EX.EXACTLY, mismatches=0, elements=4)["passed"]
    assert EX.judge(EX.EXACTLY)["verifiable"] is False


# ------------------------------------------------------------------------------- a verdict, re-judged


def _verdict(*rows, status=V.TIMING_MEASURED):
    failed = [r["group"] for r in rows if r["state"] == "failed"]
    return {
        "schema": V.SCHEMA,
        "timing_status": status,
        "whole_window_cycles": 1000,
        "objective_cycles": 1000 if status == V.TIMING_MEASURED else None,
        "correctness": {
            "status": "pass" if not failed else "fail",
            "groups_failed": failed,
            "argmax": {"agrees_with_oracle": True},
        },
        "groups": [dict(r) for r in rows],
    }


EXACT_ROW = {"group": "1", "compare": "exact", "basis": "local", "state": "correct", "mismatches": 0, "elements": 8}
BOUND_ROW = {
    "group": "2",
    "compare": "bounded_int",
    "basis": "device_bound_check",
    "state": "correct",
    "max_abs": 1,
    "over": 0,
}


def test_the_default_contract_changes_no_verdict_and_records_itself():
    contract = EX.Contract.default()
    out = EX.apply_to_verdict(
        _verdict(EXACT_ROW, BOUND_ROW), EX.resolver(contract, op_bounds={"2": 1}), contract=contract
    )
    assert out["timing_status"] == V.TIMING_MEASURED and out["exactness"]["regraded_groups"] == []
    assert out["exactness"]["per_group"] == {"1": "exact", "2": "bounded(<=1 LSB)"}
    assert EX.label_summary(out["exactness"]) == "bounded(<=1 LSB) x1, exact x1"


def test_a_form_declared_exact_tightens_an_ops_own_tolerance():
    contract = _contract({"name": "exact_add", "match": {"op": "add"}, "exactness": {"mode": "exact"}})
    resolve = EX.resolver(contract, routes=[{"group": "2", "op": "add"}], op_bounds={"2": 1})
    out = EX.apply_to_verdict(_verdict(EXACT_ROW, BOUND_ROW), resolve, contract=contract)
    assert out["timing_status"] == V.TIMING_MEASURED_INVALID and out["objective_cycles"] is None
    assert out["correctness"]["groups_failed"] == ["2"] and out["exactness"]["per_group"]["2"] == "exact"


def test_a_bounded_form_loosens_only_where_the_size_of_the_difference_is_known():
    contract = _contract(BOUNDED_CONV)
    routes = [{"group": "1", "op": "conv2d"}, {"group": "2", "op": "conv2d"}]
    counted = {**EXACT_ROW, "state": "failed", "mismatches": 3}  # a count, not a size: stays failed
    sized = {**BOUND_ROW, "state": "failed", "max_abs": 1, "over": 2}  # the op said 0, the form says 1
    out = EX.apply_to_verdict(
        _verdict(counted, sized, status=V.TIMING_MEASURED_INVALID),
        EX.resolver(contract, routes=routes, op_bounds={"2": 0}),
        contract=contract,
    )
    rows = {r["group"]: r for r in out["groups"]}
    assert rows["1"]["state"] == "failed" and rows["1"]["exactness"]["verifiable"] is False
    assert rows["2"]["state"] == "correct" and out["exactness"]["regraded_groups"] == ["2"]
    assert out["timing_status"] == V.TIMING_MEASURED_INVALID  # group 1 still fails


def test_a_board_race_is_never_a_tolerance():
    contract = _contract(BOUNDED_CONV)
    raced = {**BOUND_ROW, "state": "failed", "max_abs": 1, "board_bytes_equal_functional_model": False}
    out = EX.apply_to_verdict(
        _verdict(raced, status=V.TIMING_MEASURED_INVALID),
        EX.resolver(contract, routes=[{"group": "2", "op": "conv2d"}], op_bounds={"2": 0}),
        contract=contract,
    )
    assert out["groups"][0]["state"] == "failed" and out["timing_status"] == V.TIMING_MEASURED_INVALID


# ------------------------------------------------------------------------- a memory dump, graded


class _Local:
    def __init__(self, want):
        self.want = want

    def expected(self, group, inputs):
        return self.want


def _dump(got):
    raw = np.asarray(got, dtype="<i4").tobytes()
    return (lambda address, size: raw[address : address + size]), len(raw)


def _layout(size, *, exactness=None):
    row = {"group": 3, "kind": "conv2d", "compare": "exact", "symbol": "out", "address": 0, "bytes": size}
    row.update(elements=size // 4, element_bytes=4, inputs=[])
    if exactness is not None:
        row["exactness"] = exactness.to_dict()
    return {"groups": [row]}


def test_a_dump_is_graded_under_the_contract_its_row_carries():
    from merlin.perf.whole_model_memory import grade_memory

    want = np.zeros(16, dtype=np.int64)
    off_by_one = want.copy()
    off_by_one[4] = 1
    read, size = _dump(off_by_one)
    plain = grade_memory(read, _layout(size), {}, local=_Local(want))
    assert plain["disagree"][0]["mismatches"] == 1 and plain["evidence"]["3"]["max_abs"] == 1
    assert plain["contracts"] == {"3": "exact"}
    bounded = _contract(BOUNDED_CONV).resolve({"op": "conv2d"})
    held = grade_memory(read, _layout(size, exactness=bounded), {}, local=_Local(want))
    assert held["agree"] == ["3"] and held["contracts"] == {"3": "bounded(<=1 LSB)"}
    assert held["evidence"]["3"] == {"max_abs": 1, "mismatches": 1, "elements": 16}
    off_by_two = want.copy()
    off_by_two[4] = 2
    read, size = _dump(off_by_two)
    beyond = grade_memory(read, _layout(size, exactness=bounded), {}, local=_Local(want))
    assert beyond["agree"] == [] and beyond["disagree"][0]["group"] == "3"
    assert beyond["contracts"] == {"3": "bounded(<=1 LSB)"} and beyond["evidence"]["3"]["max_abs"] == 2


@pytest.mark.target("gemmini")
def test_the_gemmini_example_contract_is_valid_and_holds_every_form_exact():
    from merlin.common.paths import repo_root

    contract = EX.load(repo_root() / "examples" / "gemmini" / "phase2" / "exactness.yaml")
    assert contract.forms == [] and contract.semantics_sha256 == EX.DEFAULT_SEMANTICS_SHA256


# ------------------------------------------------------------------------- the whole-model gate


def _screen(*rows):
    return {"status": "screened", "groups": [dict(r) for r in rows]}


def test_the_whole_model_gate_holds_each_group_to_its_contract_and_records_it():
    from merlin.perf.whole_model_gate import grade_exactness

    exact_ok = {"group": 1, "kind": "conv2d", "local": "correct", "check": {"mismatches": 0, "of": 8}}
    exact_wrong = {"group": 2, "kind": "conv2d", "local": "wrong", "check": {"mismatches": 2, "of": 8}}
    add_off_by_one = {"group": 3, "kind": "add", "local": "correct", "check": {"max_abs": 1, "over": 0, "bound": 1}}
    expectations = {
        "groups": {"1": {"compare": "exact"}, "2": {"compare": "exact"}, "3": {"compare": "bounded", "bound_lsb": 1}}
    }
    routes = [{"group": 1, "op": "conv2d"}, {"group": 2, "op": "conv2d"}, {"group": 3, "op": "add"}]
    default = EX.Contract.default()
    wrong, record = grade_exactness(
        _screen(exact_ok, exact_wrong, add_off_by_one), expectations, default, routes=routes, forms=None
    )
    assert [w["group"] for w in wrong] == ["2"] and record["label"] == "bounded(<=1 LSB) x1, exact x2"
    assert record["contract"]["semantics_sha256"] == EX.DEFAULT_SEMANTICS_SHA256
    # The SAME screen under a contract that declares the add exact and the conv bounded: the add's one-LSB
    # difference now fails, and the conv's mismatch count cannot show a bound, so it still fails.
    strict_add = {"name": "exact_add", "match": {"op": "add"}, "exactness": {"mode": "exact"}}
    contract = _contract(BOUNDED_CONV, strict_add)
    wrong, record = grade_exactness(
        _screen(exact_ok, exact_wrong, add_off_by_one), expectations, contract, routes=routes, forms=None
    )
    assert sorted(w["group"] for w in wrong) == ["2", "3"]
    assert record["per_group"]["2"]["verifiable"] is False and record["per_group"]["3"]["contract"] == "exact"


def test_the_structure_screen_keeps_every_groups_check_numbers():
    from merlin.perf.whole_model_screen import screen_console

    console = "\n".join(
        [
            "MERLIN_INVOCATIONS warmup=1 measured=1",
            "GM_GROUP 1 conv2d 100 sum=1 fnv1a=2",
            "GM_LOCAL 1 mismatches=0 of=8 first=-1",
            "FM full model cycles: 100",
            "GM_ARGMAX got=1 want=1 agrees=1",
            "MERLIN_WINDOW end label=model",
        ]
    )
    rows = screen_console(console, {"1": "exact"})["groups"]
    assert rows[0]["local"] == "correct" and rows[0]["check"] == {"mismatches": 0, "of": 8}
