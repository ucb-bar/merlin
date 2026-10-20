"""Source-bound data choices cannot widen original pass permissions or limits."""

import hashlib
from dataclasses import replace

import pytest

from merlin.llvmlower.optional_passes import Selection, applied, switched
from merlin.llvmlower.source_optimization_request import (
    GATHER_PASS,
    PACKING_PASS,
    SCHEMA,
    SourceOptimizationRequest,
    request_from_record,
)

SOURCE = "module {}"
PERMITTED = (PACKING_PASS, GATHER_PASS)


def request(source=SOURCE, *, packing=True, gather=True, maximum=4096):
    return SourceOptimizationRequest(hashlib.sha256(source.encode()).hexdigest(), packing, gather, maximum)


@pytest.mark.parametrize("packing,gather", [(False, False), (True, False), (False, True), (True, True)])
def test_exact_roundtrip_and_both_states_override_environment(packing, gather, monkeypatch):
    original = request(packing=packing, gather=gather)
    record = original.record()
    assert record["schema"] == SCHEMA
    decoded = request_from_record(record)
    selection, maximum = decoded.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=4096)
    assert maximum == 4096
    assert selection.state(PACKING_PASS) is packing and selection.state(GATHER_PASS) is gather
    assert selection.enable | selection.disable == frozenset(PERMITTED)
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_PERMUTATIONS", "0" if packing else "1")
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_GATHERS", "0" if gather else "1")
    with applied(selection):
        assert switched(PACKING_PASS, not packing) is packing
        assert switched(GATHER_PASS, not gather) is gather
    record["maximum_added_operations"] = 0
    assert original.maximum_added_operations == 4096


@pytest.mark.parametrize("changed", [SOURCE + "\n", "module { // changed\n}"])
def test_actual_source_changes_refuse_instead_of_retargeting(changed):
    original = request()
    original.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=4096)
    with pytest.raises(ValueError, match="current source"):
        original.bind(changed, permitted_passes=PERMITTED, maximum_added_operations=4096)


@pytest.mark.parametrize("permissions", [(), (PACKING_PASS,), (GATHER_PASS,)])
def test_request_cannot_supply_original_permission(permissions):
    with pytest.raises(ValueError, match="permission"):
        request().bind(SOURCE, permitted_passes=permissions, maximum_added_operations=4096)
    baseline, _ = request(packing=False, gather=False).bind(
        SOURCE, permitted_passes=permissions, maximum_added_operations=4096
    )
    assert baseline.disable == frozenset(PERMITTED)


def test_permission_revocation_is_checked_again_on_the_same_request():
    original = request()
    original.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=4096)
    with pytest.raises(ValueError, match="permission"):
        original.bind(SOURCE, permitted_passes=(PACKING_PASS,), maximum_added_operations=4096)


@pytest.mark.parametrize("permissions", [None, list(PERMITTED), (PACKING_PASS,) * 2, ("fusion",), (True,)])
def test_permission_shape_is_closed_and_not_inferred(permissions):
    with pytest.raises(ValueError, match="permission tuple"):
        request().bind(SOURCE, permitted_passes=permissions, maximum_added_operations=4096)


@pytest.mark.parametrize("maximum", [True, -1, 1 << 21, None, 1.5])
def test_candidate_and_original_budgets_are_exact_bounded_integers(maximum):
    with pytest.raises(ValueError, match="budget"):
        request(maximum=maximum)
    with pytest.raises(ValueError, match="budget"):
        request().bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=maximum)


def test_budget_cannot_inflate_original_ceiling_or_survive_a_later_reduction():
    original = request(maximum=35)
    assert original.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=35)[1] == 35
    with pytest.raises(ValueError, match="ceiling"):
        original.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=34)
    assert request(maximum=0).bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=0)[1] == 0


@pytest.mark.parametrize("field", ["callback", "provider", "source_file", "checker", "grants", "numerical_policy"])
def test_record_cannot_select_authority_or_semantics(field):
    record = request().record()
    record[field] = "candidate supplied"
    with pytest.raises(ValueError, match="unknown or missing"):
        request_from_record(record)


@pytest.mark.parametrize("field", list(request().record()))
def test_missing_fields_do_not_acquire_defaults_at_the_data_boundary(field):
    record = request().record()
    del record[field]
    with pytest.raises(ValueError, match="unknown or missing"):
        request_from_record(record)


def test_existing_host_v1_v2_request_schemas_remain_disjoint():
    from merlin.llvmlower.host_optimization_request import (
        REQUEST_SCHEMA,
        REQUEST_SCHEMA_V2,
        HostOptimizationRequest,
    )
    from merlin.llvmlower.host_optimization_request import request_from_record as legacy_from_record

    for schema in (REQUEST_SCHEMA, REQUEST_SCHEMA_V2):
        legacy = HostOptimizationRequest(request().source_sha256, schema=schema).record()
        assert legacy_from_record(legacy).record() == legacy
        with pytest.raises(ValueError, match="unknown or missing"):
            request_from_record(legacy)
        with pytest.raises(ValueError, match="unknown schema"):
            legacy_from_record(request().record())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "merlin.host_optimization_request.v2"),
        ("source_sha256", "A" * 64),
        ("producer_packing", 1),
        ("padding_gather", None),
    ],
)
def test_wrong_schema_and_scalar_types_refuse(field, value):
    with pytest.raises(ValueError):
        request_from_record({**request().record(), field: value})


def test_companion_mutation_and_unrelated_optional_passes_do_not_gain_permission():
    original = request()
    object.__setattr__(original, "padding_gather", "enabled")
    with pytest.raises(ValueError, match="booleans"):
        original.record()
    with pytest.raises(ValueError, match="booleans"):
        original.bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=4096)
    selection, _ = request().bind(SOURCE, permitted_passes=PERMITTED, maximum_added_operations=4096)
    assert selection.state("sink-deallocs") is None
    assert replace(request(), producer_packing=False).record()["producer_packing"] is False


@pytest.mark.parametrize("kind", ["packing", "gather"])
def test_bound_choices_drive_existing_checked_passes_and_zero_budget_falls_back(kind):
    from test_im2col_padding import source as gather_source
    from test_packed_producer_destinations import program as packing_source

    from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers

    source = packing_source() if kind == "packing" else gather_source()

    def prepare(maximum):
        if kind == "packing":
            return prepare_automatic_destinations(
                source, serial=True, permuted_outputs=True, max_added_operations=maximum
            )
        return prepare_automatic_gathers(source, serial=True, max_added_operations=maximum)

    for enabled, maximum in ((False, 4096), (True, 0), (True, 4096)):
        chosen = request(
            source, packing=enabled and kind == "packing", gather=enabled and kind == "gather", maximum=maximum
        )
        selection, budget = chosen.bind(source, permitted_passes=PERMITTED, maximum_added_operations=4096)
        with applied(selection):
            actual, report = prepare(budget)
        assert report["rewrites"] == int(enabled and maximum > 0)
        assert (actual != source) is (enabled and maximum > 0)


def test_request_selection_is_not_installed_and_ambient_state_is_not_a_permission():
    from merlin.llvmlower.optional_passes import active

    with applied(Selection.parse("sink-deallocs")):
        before = active()
        selection, _ = request(packing=False, gather=False).bind(
            SOURCE, permitted_passes=(), maximum_added_operations=4096
        )
        assert active() == before
        assert selection.state("sink-deallocs") is None
