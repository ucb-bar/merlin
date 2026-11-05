"""Interprocedural entry contexts for accelerator regions, from the generated call graph."""

from __future__ import annotations

from types import SimpleNamespace

from merlin.mining.reference_database_analysis import MAX_ENTRY_CONTEXTS, interprocedural_contexts


def _regions(callers):
    regions = {"art:kernel": (dict(region_id="kernel", accelerator_sites=3), ["kernel-body"]),
               "art:alloc": (dict(region_id="alloc", accelerator_sites=0), ["alloc-body"])}
    for name in callers:
        regions["art:" + name] = (dict(region_id=name, accelerator_sites=0), [name])
    return regions


def _edge(caller, pc, callee="art:kernel", kind="call", target=0x1000):
    return dict(caller_id=caller, call_pc=pc, target=target, kind=kind, callee_id=callee, target_offset=0)


def _machine(states, *, summaries=None):
    """A lifter fake returning ``call_sites`` per caller body and memo-countable summaries."""
    calls = []

    def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
        calls.append((rows[0], call_summaries))
        sites = [dict(pc=pc, target_pc=0x1000, argument_state=state) for pc, state in states.get(rows[0], {}).items()]
        return {}, dict(status="analyzed", call_sites=sites)

    def summarize_function(rows, *, entry_state=None, call_summaries=None):
        summaries.append(rows[0])
        return dict(returns=dict(a0="heap_pointer", a1=None), status="proved")

    machine = SimpleNamespace(register_facts=register_facts)
    if summaries is not None:
        machine.summarize_function = summarize_function
    return machine, calls


def test_all_callers_agree_sets_the_entry_state():
    state = dict(a0=64, a1=None, a2=7)
    machine, _ = _machine({"f": {8: state}, "g": {12: dict(state)}})
    edges = [_edge("art:f", 8), _edge("art:g", 12)]
    context = interprocedural_contexts(_regions(["f", "g"]), edges, machine=machine)["kernel"]
    assert context["entry_state"] == {"a0": 64, "a2": 7}
    record = context["entry_context"]
    assert record["status"] == "all_callers_agree" and record["callers"] == 2
    assert record["registers"] == {"a0": 64, "a2": 7}
    assert "contexts" not in context


def test_distinct_tuples_become_bounded_contexts_within_one_region():
    machine, _ = _machine({"f": {8: dict(a0=1, a1=5)}, "g": {12: dict(a0=2, a1=5)}})
    contexts = interprocedural_contexts(_regions(["f", "g"]), [_edge("art:f", 8), _edge("art:g", 12)],
                                        machine=machine)
    assert set(contexts) == {"kernel"}  # one region, not new kernels
    context = contexts["kernel"]
    assert context["entry_context"]["status"] == "per_caller_contexts"
    assert context["entry_state"] == {"a1": 5}
    assert [row["context_id"] for row in context["contexts"]] == ["ctx-0", "ctx-1"]
    assert [row["registers"] for row in context["contexts"]] == [dict(a0=1, a1=5), dict(a0=2, a1=5)]


def test_more_than_the_context_bound_is_unknown():
    names = ["c" + str(index) for index in range(MAX_ENTRY_CONTEXTS + 1)]
    machine, _ = _machine({name: {4: dict(a0=index)} for index, name in enumerate(names)})
    edges = [_edge("art:" + name, 4) for name in names]
    record = interprocedural_contexts(_regions(names), edges, machine=machine)["kernel"]["entry_context"]
    assert record["status"] == "unknown" and "exceed" in record["reason"]


def test_todays_lifter_without_call_sites_stays_unknown():
    def register_facts(rows, *, max_steps=2_000_000):
        return {}, dict(status="analyzed")

    machine = SimpleNamespace(register_facts=register_facts)
    record = interprocedural_contexts(_regions(["f"]), [_edge("art:f", 8)], machine=machine)["kernel"]
    assert record["entry_context"]["status"] == "unknown"
    assert "no call-site argument states" in record["entry_context"]["reason"]
    assert record.get("entry_state") is None and record["call_summaries"] is None


def test_uncalled_region_reports_no_callers():
    machine, _ = _machine({})
    record = interprocedural_contexts(_regions([]), [], machine=machine)["kernel"]["entry_context"]
    assert record["status"] == "unknown" and record["callers"] == 0


def test_mid_span_transfers_are_not_entry_callers():
    machine, _ = _machine({"f": {8: dict(a0=1)}})
    edge = dict(_edge("art:f", 8), target_offset=16)
    record = interprocedural_contexts(_regions(["f"]), [edge], machine=machine)["kernel"]["entry_context"]
    assert record["callers"] == 0


def test_callee_summaries_are_memoized_and_passed():
    summaries = []
    machine, calls = _machine({"f": {8: dict(a0=9)}}, summaries=summaries)
    edges = [_edge("art:f", 4, callee="art:alloc", target=0x2000), _edge("art:f", 8),
             _edge("art:kernel", 20, callee="art:alloc", target=0x2000)]
    context = interprocedural_contexts(_regions(["f"]), edges, machine=machine)["kernel"]
    assert sorted(summaries) == ["alloc-body", "kernel-body"]  # each callee summarized once
    assert set(context["call_summaries"]) == {0x2000}
    assert set(calls[0][1]) == {0x1000, 0x2000}
    assert context["entry_context"]["status"] == "all_callers_agree"


def _stateful_machine(sites_of):
    """A lifter fake whose call-site arguments depend on the entry state it is lifted under."""
    lifts = []

    def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
        lifts.append((rows[0], dict(entry_state or {})))
        sites = [dict(pc=pc, target_pc=0x1000, argument_state=state)
                 for pc, state in sites_of(rows[0], entry_state or {}).items()]
        return {}, dict(status="analyzed", call_sites=sites)

    return SimpleNamespace(register_facts=register_facts), lifts


def test_caller_contexts_carry_constants_from_two_calls_up():
    # r calls f twice with a3 = 16 and 32; f forwards its a3 to the kernel in a1.
    def sites_of(name, entry):
        if name == "r":
            return {4: dict(a3=16), 6: dict(a3=32)}
        if name == "f":
            return {8: dict(a1=entry.get("a3"), a2=5)}
        return {}

    machine, lifts = _stateful_machine(sites_of)
    edges = [_edge("art:r", 4, callee="art:f"), _edge("art:r", 6, callee="art:f"), _edge("art:f", 8)]
    context = interprocedural_contexts(_regions(["r", "f"]), edges, machine=machine)["kernel"]
    record = context["entry_context"]
    assert record["status"] == "per_caller_contexts" and record["caller_context_lifts"] == 2
    assert context["entry_state"] == {"a2": 5}
    assert [row["registers"] for row in context["contexts"]] == [dict(a1=16, a2=5), dict(a1=32, a2=5)]
    assert [row["call_sites"][0]["caller_context"] for row in context["contexts"]] == ["ctx-0", "ctx-1"]
    assert ("f", {"a3": 16}) in lifts and ("f", {"a3": 32}) in lifts


def test_without_caller_contexts_the_forwarded_argument_stays_unknown():
    def sites_of(name, entry):
        return {8: dict(a1=entry.get("a3"))} if name == "f" else {}

    machine, _ = _stateful_machine(sites_of)
    record = interprocedural_contexts(_regions(["f"]), [_edge("art:f", 8)], machine=machine)["kernel"]
    assert record["entry_context"]["status"] == "unknown" and record.get("entry_state") is None
    assert "caller_context_lifts" not in record["entry_context"]


def test_recursive_callers_terminate_with_a_stateless_lift():
    def sites_of(name, entry):
        return {4: dict(a0=entry.get("a0", 1))} if name in ("f", "g") else {}

    machine, _ = _stateful_machine(sites_of)
    edges = [_edge("art:f", 4, callee="art:g"), _edge("art:g", 4, callee="art:f"), _edge("art:g", 4)]
    context = interprocedural_contexts(_regions(["f", "g"]), edges, machine=machine)["kernel"]
    assert context["entry_context"]["callers"] == 1
    assert context["entry_state"] == {"a0": 1}


def test_stepped_calls_without_a_known_return_are_named_with_the_callee_reason():
    from merlin.mining.reference_database_analysis import _note_unproved_returns, unproved_call_returns

    counted = dict(summarized_calls={0x30: 1, 0x40: 2}, assumed_returning_calls={0x50: 2})
    summaries = {0x30: dict(status="proved", returns=dict(a0=None, a1=None), reason=None),
                 0x40: dict(status="proved", returns=dict(a0=7, a1=None)),
                 0x50: dict(status="unknown", returns=dict(a0=None, a1=None), reason="callee_control_gaps:x")}
    unproved = unproved_call_returns(counted, summaries)
    assert set(unproved) == {"0x30", "0x50"}  # a proved return value is applied, so it is not listed
    assert unproved["0x30"]["stepping"] == "proved_summary" and "memory written before" in unproved["0x30"]["reason"]
    assert unproved["0x50"] == dict(calls=2, stepping="assumed_to_return", reason="callee_control_gaps:x")
    effects = dict(facts=dict(
        operand_buffering=dict(status="observed", limits=["3 loads unresolved: load local address unresolved"]),
        prefetch_distance=dict(status="unknown", unresolved={"load local address unresolved": 3}),
        issue_overlap=dict(status="observed", limits=[])))
    _note_unproved_returns(effects, dict(unproved_call_returns=unproved), dict(entry_context=dict(callers=0)))
    for name in ("operand_buffering", "prefetch_distance"):
        assert "0x30, 0x50" in effects["facts"][name]["limits"][-1]
        assert "static callers of this region: 0" in effects["facts"][name]["limits"][-1]
    assert effects["facts"]["issue_overlap"]["limits"] == []
    assert unproved_call_returns(dict(summarized_calls={0x40: 1}), summaries) == {}
