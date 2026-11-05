"""Complete operand localization survives bounded displays and refused populations."""

from merlin.kernels.instruction_comparison import OperandSummary, compare_operand_summaries


def summary(values, **limits):
    result = OperandSummary(**limits)
    for value in values:
        result.observe("OPAQUE_PACKET", dict(extent=value, mode="selected"))
    return result.report()


def test_rare_late_operand_change_is_localized_outside_displayed_profile():
    before, after = summary(range(40)), summary([*range(39), 99])
    assert before["field_profiles"] == after["field_profiles"]
    comparison = compare_operand_summaries(before, after)
    assert comparison["same_recovered_operands"] is False
    assert comparison["same_recovered_operand_population"] is False
    assert [row["field"] for row in comparison["operand_field_differences"]] == ["OPAQUE_PACKET.extent"]
    assert "recovered_values_changed" in comparison["operand_field_differences"][0]["reasons"]
    assert comparison["semantic_alignment"] == comparison["execution_multiplicity"] == "unknown"


def test_identical_reordered_and_count_only_controls_remain_distinct():
    before = summary(range(40))
    identical = compare_operand_summaries(before, summary(range(40)))
    reordered = compare_operand_summaries(before, summary(reversed(range(40))))
    count_only = compare_operand_summaries(summary([5, 5]), summary([5, 5, 5]))
    assert identical["same_recovered_operands"] is True and not identical["operand_field_differences"]
    assert reordered["order_change"] == "reordered_same_recovered_operands"
    assert not reordered["operand_field_differences"]
    for difference in count_only["operand_field_differences"]:
        assert "site_count_changed" in difference["reasons"]
        assert "recovered_values_changed" not in difference["reasons"]


def test_value_changes_remain_visible_when_count_changes_exceed_reply_bound():
    before, after = OperandSummary(), OperandSummary()
    fields = {f"field{index}": 3 for index in range(12)}
    before.observe("PACKET", fields)
    after.observe("PACKET", fields)
    after.observe("PACKET", {**fields, "field9": 8})
    comparison = compare_operand_summaries(before.report(), after.report())
    assert comparison["operand_field_differences"][0]["field"] == "PACKET.field9"
    assert comparison["omitted_field_differences"] == 4


def test_typed_scalars_do_not_conflate_bool_integer_and_float():
    result = summary([True, 1, 1.0])
    field = result["field_fingerprints"][0]
    assert field["distinct_values"] == 3
    assert len(result["field_profiles"]["OPAQUE_PACKET.extent"]) == 3
    assert compare_operand_summaries(summary([True]), summary([1]))["same_recovered_operands"] is False


def test_missing_and_unresolved_fields_are_not_silent_or_proved_values():
    before, after = OperandSummary(), OperandSummary()
    for values in (dict(extent=None), {}, dict(extent=8)):
        before.observe("PACKET", values)
    for values in (dict(extent=8), dict(extent=8), dict(extent=8)):
        after.observe("PACKET", values)
    result = before.report()
    field = result["field_fingerprints"][0]
    assert field["missing_sites"] == field["unresolved_sites"] == 1
    difference = compare_operand_summaries(result, after.report())["operand_field_differences"][0]
    assert "missingness_changed" in difference["reasons"]
    assert "unresolved_sites_changed" in difference["reasons"]
    compound = OperandSummary()
    compound.observe("PACKET", dict(extent=dict(low=8, high=None)))
    assert compound.report()["field_fingerprints"][0]["unresolved_sites"] == 1


def test_population_budget_exhaustion_preserves_sequence_localization_and_unknown():
    before = summary(range(40), max_population_entries=3)
    after = summary([*range(39), 99], max_population_entries=3)
    comparison = compare_operand_summaries(before, after)
    assert before["operand_population_status"] == "unknown_budget_exhausted"
    assert comparison["same_recovered_operand_population"] is None
    assert comparison["same_recovered_operands"] is False
    difference = comparison["operand_field_differences"][0]
    assert difference["field"] == "OPAQUE_PACKET.extent"
    assert "recovered_sequence_changed_population_unknown" in difference["reasons"]


def test_field_budget_and_legacy_fingerprint_versions_do_not_claim_full_comparison():
    before, after = summary([4], max_fields=1), summary([5], max_fields=1)
    comparison = compare_operand_summaries(before, after)
    assert before["omitted_field_sites"] == 1
    assert comparison["field_localization_status"] == "bounded_or_unavailable"
    legacy = {key: value for key, value in before.items() if key != "operand_fingerprint_schema"}
    comparison = compare_operand_summaries(legacy, before)
    assert comparison["same_recovered_operands"] is None
    assert comparison["same_recovered_operand_population"] is None
    assert comparison["field_localization_status"] == "unknown_fingerprint_schema_mismatch"
    classes = OperandSummary(max_fields=1)
    classes.observe("FIRST", {})
    classes.observe("UNTRACKED", {})
    assert classes.report()["field_fingerprint_status"] == "unknown_field_budget_exhausted"


def test_machine_mechanism_group_budget_preserves_witnesses_and_reports_incomplete_census():
    from merlin.kernels.machine_evidence import _MechanismObservations

    mechanisms = _MechanismObservations({"PACKET": ["weight_load"]}, max_groups=2, max_witnesses=1)
    operands = OperandSummary()
    for index, value in enumerate([1, 2, 3, 1]):
        fields = dict(extent=value)
        mechanisms.observe(index, dict(instruction="PACKET", fields=fields, pc=index * 4))
        operands.observe("PACKET", fields)
    report = mechanisms.report(operands.report())
    assert len(mechanisms.groups) == 2 and report["omitted_group_sites"] == 1
    assert report["repetition_census"] == "unknown_group_budget_exhausted"
    assert report["repeated_operand_groups"][0]["sites"] == 2
    assert report["candidates"][0]["optimization_applicability"] == "unknown"
