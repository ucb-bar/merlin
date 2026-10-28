from merlin_experiments.corpus.coverage import disclose_withheld_scope, source_capsule_name, withheld_from_generation

A, B, C = "movement -> contraction -> elementwise_map", "reduction -> reduction", "generic -> generic"
REQ = [
    {"signature": A, "observed_in": ["m1", "m2"]},
    {"signature": B, "observed_in": ["m3"]},
    {"signature": C, "observed_in": ["m4"]},
]


def scope(uncovered):
    return {"status": "ok", "n_required": 3, "n_covered": 3 - len(uncovered), "uncovered": list(uncovered)}


def test_all_planned_witnesses_withheld_is_disclosed_never_covered():
    s = scope([A, B])
    disclose_withheld_scope(s, REQ, {"SY_source_m1": "r1", "SY_source_m2": "r2", "SY_source_m3": "r3"})
    assert s["uncovered"] == [A, B] and s["n_covered"] == 1 and s["status"] == "ok"
    assert s["withheld_unplaceable"][A]["planned_witnesses"] == ["SY_source_m1", "SY_source_m2"]
    assert s["withheld_unplaceable"][A]["withheld"] == {"SY_source_m1": "r1", "SY_source_m2": "r2"}
    assert s["n_withheld_unplaceable"] == 2


def test_partially_withheld_or_reasonless_chain_stays_a_plain_gap():
    s = scope([A, B])
    disclose_withheld_scope(s, REQ, {"SY_source_m1": "r1", "SY_source_m3": ""})
    assert "withheld_unplaceable" not in s and s["uncovered"] == [A, B]


def test_unplanned_chain_without_observed_in_is_not_disclosed():
    s = scope([C])
    disclose_withheld_scope(s, [{"signature": C}], {"SY_source_m4": "r"})
    assert "withheld_unplaceable" not in s


def test_covered_scope_or_no_withheld_is_untouched():
    for s, w in ((scope([]), {"SY_source_m1": "r"}), (scope([A]), None), (scope([A]), {})):
        before = dict(s)
        disclose_withheld_scope(s, REQ, w)
        assert s == before
    disclose_withheld_scope({"status": "not_measured"}, REQ, {"x": "y"})


def test_withheld_from_generation_reads_reasons_and_ignores_blank(tmp_path):
    import json

    bundle = tmp_path / "b"
    (bundle / "coverage").mkdir(parents=True)
    (bundle / "coverage" / "generation.json").write_text(json.dumps({
        "host_placement_not_published": [{"capsule": "SY_source_m1", "reason": "why"}, {"capsule": "X", "reason": " "}],
        "unprovable_forbids": [{"capsule": "Y", "reason": "z"}]}))
    assert withheld_from_generation({"phase0_evidence_bundle": str(bundle)}) == {"SY_source_m1": "why", "Y": "z"}
    assert withheld_from_generation({}) == {} and withheld_from_generation({"phase0_evidence_bundle": str(tmp_path)}) == {}
    assert source_capsule_name("m1") == "SY_source_m1"
