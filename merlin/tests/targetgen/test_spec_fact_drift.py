"""The authored software spec against fact-derived capability: every classification, both directions."""

import copy

import pytest

from merlin.targetgen import spec_fact_drift as D

_CONTRACT = {
    "name": "fixture",
    "compute_units": [
        {
            "name": "mesh",
            "kind": "systolic",
            "dtypes": ["int8"],
            "ops": ["matmul"],
            # A readout requant: the unit's own intent to fuse elementwise work onto a contraction.
            "requant": {"ref": "fixture.readout_scale"},
            "scaling": "per_tensor",
            "semantic_capabilities": [
                {"family": "contraction", "dtypes": ["int8"], "ranks": [2]},
                {
                    "family": "elementwise_map",
                    "dtypes": ["int8"],
                    "standalone_evidence": [{"fact": "accumulate_on_load", "roles": ["operand_load", "readout"]}],
                },
                {"family": "movement", "dtypes": ["int8"]},
                {"family": "reduction", "dtypes": ["int8"], "composed_with": ["contraction"]},
            ],
        }
    ],
}
_RAW_FACTS = {
    "facts": {
        # A grid's shape alone no longer proves it computes MACs. Model the
        # corroborated extractor output so this fixture tests a licensed
        # contraction rather than relying on the former shape-only inference.
        "arrays": [
            {
                "name": "mesh",
                "rows": 4,
                "cols": 4,
                "corroborated": True,
                "mac_idiom": {"muls": 1, "adds": 1},
            }
        ],
        "datapaths": [{"name": "input", "dtype": "i8"}],
        "interfaces": [{"name": "mesh_dma"}],
        "storage_datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
    }
}
_FACETS = [
    {
        "unit": "mesh",
        "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu", "maxpool"], "evidence": "observed"}],
        "unknown": {},
        "scale": {"granularities": ["tensor"], "granularities_complete": True, "carriers": []},
    }
]
_QUANT = [
    {"status": "derived", "unit": "mesh", "format": "int8", "recipe": {"families": ["contraction", "operand_sum"]}}
]


def _facts(**overrides):
    views = {
        "target": "fixture",
        "contract": _CONTRACT,
        "raw_facts": _RAW_FACTS,
        "readout_facets": _FACETS,
        "quantization_candidates": _QUANT,
        "taxonomy": {},
        "semantic_facts": {"facts": {"accumulate_on_load": {"status": "derived"}}},
        **overrides,
    }
    return D.fact_capabilities(**views)


def _spec(**changes):
    """A spec that states exactly what the fixture facts establish."""
    operations = {
        "contraction": {
            "families": ["contraction"],
            "placement": "accelerator",
            "signature": {"operand_dtypes": ["int8"], "ranks": [2]},
        },
        "elementwise_map": {
            "families": ["elementwise_map", "operand_sum"],
            "placement": "accelerator",
            "signature": {"dtypes": ["int8"], "epilogues": ["acc_scale", "relu"], "scale_granularity": ["tensor"]},
        },
        "movement": {"families": ["movement"], "placement": "accelerator", "signature": {"dtypes": ["int8", "i32"]}},
        "reduction": {
            "families": ["reduction"],
            "placement": "fused_accelerator",
            "signature": {"composed_with": ["contraction"]},
        },
    }
    for name, change in changes.items():
        operations[name] = {**operations[name], **change}
    return {
        "target": "fixture",
        "operations": [{"id": name, **row} for name, row in operations.items()],
        "quantization": {
            "formats": [
                {
                    "id": "int8__int32",
                    "operand_dtype": "int8",
                    "accumulator_dtype": "int32",
                    "eligible_operations": ["contraction", "elementwise_map"],
                }
            ]
        },
    }


def _only(report, family, field):
    rows = [row for row in report["findings"] if row["family"] == family and row["field"] == field]
    assert rows, f"no finding for {family}.{field}"
    return rows


def test_facts_are_derived_from_the_ladder_the_evidence_path_and_the_readout():
    facts = _facts()["families"]
    assert facts["contraction"]["placement"].values == ("accelerator",)
    # Standalone only through the contract's evidence path, never as a bare declaration.
    assert facts["elementwise_map"]["placement"].values == ("accelerator",)
    assert "accumulate_on_load" in facts["elementwise_map"]["placement"].evidence[0]
    # Pooling on the readout is fused reduction, so it is not an elementwise epilogue.
    assert facts["reduction"]["placement"].values == ("fused_accelerator",)
    assert facts["reduction"]["placement"].complete
    assert facts["elementwise_map"]["epilogues"].values == ("acc_scale", "relu")
    assert facts["movement"]["dtypes"].values == ("int32", "int8")


def test_a_spec_equal_to_the_facts_is_consistent():
    report = D.spec_fact_drift(_spec(), _facts())
    assert report["status"] == "consistent", [row for row in report["findings"] if row["classification"] != "ok"]
    assert report["n_blocking"] == 0 and D.blockers(report) == []
    assert {row["classification"] for row in report["findings"]} == {D.OK}


def test_narrower_without_a_reason_forbids_what_the_facts_establish():
    spec = _spec(elementwise_map={"placement": "fused_accelerator", "signature": {"composed_with": ["contraction"]}})
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "elementwise_map", "placement")
    assert row["classification"] == D.FORBIDS and row["values"] == ["accelerator"]
    assert [blocker["family"] for blocker in D.blockers(report)] == ["elementwise_map"]


def test_narrower_with_a_recorded_reason_is_reported_but_not_blocking():
    spec = _spec(elementwise_map={"placement": "fused_accelerator", "signature": {"composed_with": ["contraction"]}})
    spec["restrictions"] = [
        {
            "family": "elementwise_map",
            "field": "placement",
            "values": ["accelerator"],
            "reason": "no compiler route lowers a standalone map yet",
        }
    ]
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "elementwise_map", "placement")
    assert row["classification"] == D.RESTRICTED and row["restrictions"] == spec["restrictions"]
    assert report["n_blocking"] == 0


def test_a_restriction_covers_only_the_values_it_names():
    spec = _spec(movement={"signature": {"dtypes": ["int8"]}})
    spec["restrictions"] = [
        {"family": "movement", "field": "dtypes", "values": ["fp32"], "reason": "a reason about another format"}
    ]
    [row] = _only(D.spec_fact_drift(spec, _facts()), "movement", "dtypes")
    assert row["classification"] == D.FORBIDS and row["values"] == ["int32"]


def test_wider_than_complete_facts_blocks():
    spec = _spec(contraction={"signature": {"operand_dtypes": ["int8", "fp32"], "ranks": [2]}})
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "contraction", "dtypes")
    assert row["classification"] == D.EXCEEDS and row["values"] == ["fp32"]
    assert report["status"] == "drift"


def test_wider_than_an_axis_no_rung_decides_is_unconfirmed_not_refused():
    spec = _spec(contraction={"signature": {"operand_dtypes": ["int8"], "ranks": [2, 4]}})
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "contraction", "ranks")
    assert row["classification"] == D.UNCONFIRMED and row["values"] == [4]
    assert report["n_blocking"] == 0


def test_a_host_only_spec_for_a_fused_family_forbids_it():
    spec = _spec(reduction={"placement": "host", "signature": {"dtypes": ["f32"]}})
    [row] = _only(D.spec_fact_drift(spec, _facts()), "reduction", "placement")
    assert row["classification"] == D.FORBIDS and row["values"] == ["fused_accelerator"]


def test_an_unevidenced_composition_cannot_hold_the_spec_to_it():
    contract = copy.deepcopy(_CONTRACT)
    facets = [{**_FACETS[0], "readouts": [{"selector": "i8", "applies": ["acc_scale"], "evidence": "observed"}]}]
    spec = _spec(reduction={"placement": "host", "signature": {"dtypes": ["f32"]}})
    report = D.spec_fact_drift(spec, _facts(contract=contract, readout_facets=facets))
    [row] = _only(report, "reduction", "placement")
    assert row["classification"] == D.UNDETERMINED


def test_an_unknown_authored_placement_is_an_open_decision_not_a_refusal():
    spec = _spec(elementwise_map={"placement": "unknown"})
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "elementwise_map", "placement")
    assert row["classification"] == D.UNDETERMINED and "unknown" in row["reason"]


def test_quantization_eligibility_is_compared_with_the_derived_recipe_families():
    spec = _spec()
    spec["quantization"]["formats"][0]["eligible_operations"] = ["contraction"]
    report = D.spec_fact_drift(spec, _facts())
    [row] = _only(report, "operand_sum", "quantization")
    assert row["classification"] == D.FORBIDS and row["values"] == ["int8__int32"]
    spec["restrictions"] = [
        {"family": "operand_sum", "field": "quantization", "values": ["int8__int32"], "reason": "kept in f32 by choice"}
    ]
    [row] = _only(D.spec_fact_drift(spec, _facts()), "operand_sum", "quantization")
    assert row["classification"] == D.RESTRICTED


def test_quantizing_a_family_no_recipe_quantizes_blocks():
    quant = [{**_QUANT[0], "recipe": {"families": ["contraction"]}}]
    [row] = _only(D.spec_fact_drift(_spec(), _facts(quantization_candidates=quant)), "operand_sum", "quantization")
    assert row["classification"] == D.EXCEEDS


@pytest.mark.parametrize(
    "restriction, message",
    [
        ({"family": "movement", "field": "dtypes", "values": ["i32"], "reason": "no"}, "stated reason"),
        ({"family": "movement", "field": "layout", "values": ["x"], "reason": "a stated reason here"}, "not one of"),
        ({"family": "movement", "field": "dtypes", "values": [], "reason": "a stated reason here"}, "declined values"),
        ({"family": "movement", "field": "dtypes", "values": ["i32"]}, "exactly family"),
    ],
)
def test_a_restriction_must_name_its_values_and_reason(restriction, message):
    with pytest.raises(ValueError, match=message):
        D.validate_restrictions({"restrictions": [restriction]})


def test_an_absent_report_is_a_readiness_blocker():
    assert D.blockers(None) == [{"component": "software_spec_drift", "reason": "spec-vs-facts drift check is absent"}]


def test_coverage_commitment_blocks_on_drift():
    from merlin_experiments.phase0 import coverage_commitment as C

    spec = _spec(elementwise_map={"placement": "fused_accelerator", "signature": {"composed_with": ["contraction"]}})
    inputs = {
        "schema": C.INPUT_SCHEMA,
        "software_spec": {"status": "reviewed"},
        "spec_fact_drift": D.spec_fact_drift(spec, _facts()),
    }
    report = C.build_commitment(inputs, [])
    drift = [row for row in report["blockers"] if row["component"] == "software_spec_drift"]
    assert [(row["family"], row["field"]) for row in drift] == [("elementwise_map", "placement")]
    consistent = C.build_commitment({**inputs, "spec_fact_drift": D.spec_fact_drift(_spec(), _facts())}, [])
    assert not [row for row in consistent["blockers"] if row["component"] == "software_spec_drift"]


# --- filling hardware-shaped fields from the facts ------------------------------------------------


def _derived_spec(**extra):
    spec = {
        "schema": "merlin.software_spec.v1",
        "target": "fixture",
        "status": "reviewed",
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
        },
        "operations": {
            "contraction": {"placement": "accelerator", "operand_dtypes": ["int8"], "ranks": [2]},
            "elementwise_map": {"hardware": "fused"},
            "residual_add": {"families": ["elementwise_map", "operand_sum"], "hardware": "standalone"},
            "pooling": {"families": ["reduction"], "hardware": "fused"},
            "movement": {"hardware": "standalone", "layouts": ["row_major_contiguous"]},
        },
        "quantization": {
            "formats": [
                {
                    "id": "int8__int32",
                    "operand_dtype": "int8",
                    "accumulator_dtype": "int32",
                    "eligible_operations": "from_facts",
                }
            ]
        },
        **extra,
    }
    from merlin.targetgen.software_spec import validate_software_spec

    return validate_software_spec(spec, target="fixture")


def _rows(spec):
    return {row["id"]: row for row in spec["operations"]}


def test_each_hardware_form_is_filled_from_its_own_facts():
    resolved, record = D.resolve_spec(_derived_spec(), _facts())
    rows = _rows(resolved)
    assert record["status"] == "resolved" and record["unresolved"] == []
    assert rows["residual_add"]["placement"] == "accelerator"
    assert rows["residual_add"]["signature"] == {"dtypes": ["int8"], "scale_granularity": ["tensor"]}
    assert rows["elementwise_map"]["placement"] == "fused_accelerator"
    assert rows["elementwise_map"]["signature"]["composed_with"] == ["contraction"]
    assert rows["elementwise_map"]["signature"]["epilogues"] == ["acc_scale", "relu"]
    assert rows["pooling"]["placement"] == "fused_accelerator"
    assert rows["pooling"]["signature"]["composed_with"] == ["contraction"]
    assert rows["movement"]["signature"] == {"dtypes": ["int32", "int8"], "layouts": ["row_major_contiguous"]}
    # Authored software fields survive; the row records which form the facts filled.
    assert rows["movement"]["derived_from_facts"] == {"form": "standalone", "family": "movement"}
    assert "hardware" not in rows["movement"]
    assert resolved["quantization"]["formats"][0]["eligible_operations"] == ["contraction", "residual_add"]
    # Every derived value carries the evidence it came from.
    by_id = {row["id"]: row for row in record["declarations"]}
    assert "accumulate_on_load" in by_id["residual_add"]["evidence"]["placement"][0]


def test_operand_sum_readout_composition_needs_the_same_facet() -> None:
    from merlin.targetgen.software_spec import admit_operation

    spec = _derived_spec()
    spec["operations"].append(
        {"id": "sum_readout", "families": ["elementwise_map"], "hardware": "fused_operand_sum", "signature": {}}
    )
    absent, record = D.resolve_spec(spec, _facts())
    assert _rows(absent)["sum_readout"]["placement"] == "unknown"
    assert record["status"] == "unresolved"

    facets = copy.deepcopy(_FACETS)
    facets[0]["operand_sum"] = {"operands": 2, "operand_dtype": "i8"}
    resolved, record = D.resolve_spec(spec, _facts(readout_facets=facets))
    assert record["status"] == "resolved"
    row = _rows(resolved)["sum_readout"]
    assert row["placement"] == "fused_accelerator"
    assert row["signature"] == {
        "composed_with": ["residual_add"],
        "dtypes": ["int8"],
        "epilogues": ["acc_scale", "relu"],
        "scale_granularity": ["tensor"],
    }
    signature = {
        "family": "elementwise_map",
        "operand_dtype": "int8",
        "composed_with": ["residual_add"],
        "epilogues": ["relu"],
        "scale_granularity": "tensor",
    }
    assert admit_operation(resolved, "relu", signature, "fused_accelerator")["status"] == "admitted"
    contraction = {**signature, "composed_with": ["contraction"]}
    assert admit_operation(resolved, "relu", contraction, "fused_accelerator")["status"] == "admitted"

    alternate = copy.deepcopy(facets[0])
    alternate["unit"] = "second_mesh"
    alternate["readouts"][0]["applies"] = ["acc_scale"]
    mixed, _ = D.resolve_spec(spec, _facts(readout_facets=[*facets, alternate]))
    assert _rows(mixed)["sum_readout"]["signature"]["epilogues"] == ["acc_scale"]
    assert admit_operation(mixed, "relu", signature, "fused_accelerator")["status"] == "unsupported"


def test_a_resolved_spec_has_no_drift_against_the_same_facts():
    resolved, _ = D.resolve_spec(_derived_spec(), _facts())
    report = D.spec_fact_drift(resolved, _facts())
    assert report["n_blocking"] == 0, [row for row in report["findings"] if row["classification"] != "ok"]


def test_a_form_the_facts_do_not_establish_resolves_to_unknown():
    contract = copy.deepcopy(_CONTRACT)
    del contract["compute_units"][0]["semantic_capabilities"][1]["standalone_evidence"]
    contract["compute_units"][0]["semantic_capabilities"][1]["composed_with"] = ["contraction"]
    resolved, record = D.resolve_spec(_derived_spec(), _facts(contract=contract))
    assert _rows(resolved)["residual_add"]["placement"] == "unknown"
    assert record["status"] == "unresolved"
    assert [row["id"] for row in record["unresolved"] if "id" in row] == ["residual_add"]


def test_a_restriction_narrows_a_derived_value_and_cannot_empty_it():
    restriction = {"family": "movement", "field": "dtypes", "values": ["i32"], "reason": "kept on the host by choice"}
    resolved, _ = D.resolve_spec(_derived_spec(restrictions=[restriction]), _facts())
    assert _rows(resolved)["movement"]["signature"]["dtypes"] == ["int8"]
    everything = {**restriction, "values": ["i32", "int8"]}
    with pytest.raises(ValueError, match="decline every"):
        D.resolve_spec(_derived_spec(restrictions=[everything]), _facts())
    placement = {"family": "movement", "field": "placement", "values": ["accelerator"], "reason": "declined by choice"}
    with pytest.raises(ValueError, match="own restriction declines"):
        D.resolve_spec(_derived_spec(restrictions=[placement]), _facts())


def test_a_declaration_restriction_narrows_only_its_derived_form():
    spec = _derived_spec()
    spec["operations"].append(
        {"id": "sum_readout", "families": ["elementwise_map"], "hardware": "fused_operand_sum", "signature": {}}
    )
    spec["restrictions"] = [
        {
            "declaration": "sum_readout",
            "family": "elementwise_map",
            "field": "epilogues",
            "values": ["bias_add"],
            "reason": "this composition has no reviewed bias witness",
        }
    ]
    facets = copy.deepcopy(_FACETS)
    facets[0]["readouts"][0]["applies"].append("bias_add")
    facets[0]["operand_sum"] = {"operands": 2, "operand_dtype": "i8"}
    resolved, record = D.resolve_spec(spec, _facts(readout_facets=facets))
    rows = _rows(resolved)
    assert rows["sum_readout"]["signature"]["epilogues"] == ["acc_scale", "relu"]
    assert rows["elementwise_map"]["signature"]["epilogues"] == ["acc_scale", "bias_add", "relu"]
    assert record["declarations"][-1]["restrictions"]["epilogues"] == spec["restrictions"]
    assert D.spec_fact_drift(resolved, _facts(readout_facets=facets))["n_blocking"] == 0
    invalid = {**spec, "restrictions": [{**spec["restrictions"][0], "declaration": "elementwise_map_standalone"}]}
    with pytest.raises(ValueError, match="fact-derived declaration"):
        D.validate_restrictions(invalid)


def test_contraction_seed_route_derives_fused_bias_without_operand_sum_bias():
    from merlin.targetgen.software_spec import admit_operation

    spec = _derived_spec()
    spec["operations"].append(
        {"id": "sum_readout", "families": ["elementwise_map"], "hardware": "fused_operand_sum", "signature": {}}
    )
    facets = copy.deepcopy(_FACETS)
    facets[0]["operand_sum"] = {"operands": 2, "operand_dtype": "i8"}
    facets[0]["stage_routes"] = [
        {
            "stages": ["bias_add", "bias"],
            "site": "accumulator_seed",
            "composed_with": "contraction",
            "readouts": ["i8", "i32"],
            "evidence": "bias preloads the accumulator before contraction",
        }
    ]
    facts = _facts(readout_facets=facets)
    fused = facts["forms"]["elementwise_map"]["fused"]
    after_sum = facts["forms"]["elementwise_map"]["fused_operand_sum"]
    assert "bias_add" in fused["epilogues"].values
    assert "bias_add" not in after_sum["epilogues"].values
    resolved, _ = D.resolve_spec(spec, facts)
    rows = _rows(resolved)
    assert "bias_add" in rows["elementwise_map"]["signature"]["epilogues"]
    assert "bias_add" not in rows["sum_readout"]["signature"]["epilogues"]
    request = {
        "family": "elementwise_map",
        "operand_dtype": "int8",
        "epilogues": ["bias_add"],
        "scale_granularity": "tensor",
    }
    assert (
        admit_operation(resolved, "bias_add", {**request, "composed_with": ["contraction"]}, "fused_accelerator")[
            "status"
        ]
        == "admitted"
    )
    assert (
        admit_operation(resolved, "bias_add", {**request, "composed_with": ["residual_add"]}, "fused_accelerator")[
            "status"
        ]
        == "unsupported"
    )
    assert admit_operation(resolved, "bias_add", request, "accelerator")["status"] == "unsupported"


def test_a_quantization_restriction_removes_the_family_from_derived_eligibility():
    restriction = {"family": "operand_sum", "field": "quantization", "values": ["int8__int32"], "reason": "kept in f32"}
    resolved, _ = D.resolve_spec(_derived_spec(restrictions=[restriction]), _facts())
    assert resolved["quantization"]["formats"][0]["eligible_operations"] == ["contraction"]


@pytest.mark.parametrize(
    "row, message",
    [
        ({"hardware": "standalone", "dtypes": ["int8"]}, "derives"),
        ({"hardware": "standalone", "placement": "accelerator"}, "derives"),
        ({"hardware": "sideways"}, "must be one of"),
        ({"families": ["elementwise_map", "reduction"], "hardware": "fused"}, "exactly one semantic family"),
    ],
)
def test_a_hardware_declaration_cannot_also_author_what_it_derives(row, message):
    spec = _derived_spec()
    spec["operations"] = [{"id": "movement", **row}]
    spec.pop("quantization")
    from merlin.targetgen.software_spec import validate_software_spec

    with pytest.raises(ValueError, match=message):
        validate_software_spec(spec, target="fixture")


def test_admission_refuses_an_unresolved_declaration():
    from merlin.targetgen.software_spec import admit_operation

    spec = _derived_spec()
    with pytest.raises(ValueError, match="resolved spec"):
        admit_operation(spec, "aten.add.Tensor", {"family": "elementwise_map", "operand_dtype": "int8"}, "accelerator")
    # An authored declaration is still screened from the authored bytes.
    decision = admit_operation(
        spec, "aten.mm.default", {"family": "contraction", "operand_dtype": "int8", "rank": 2}, "accelerator"
    )
    assert decision["declaration"] == "contraction"
    resolved, _ = D.resolve_spec(spec, _facts())
    decision = admit_operation(
        resolved, "aten.add.Tensor", {"family": "elementwise_map", "operand_dtype": "int8"}, "accelerator"
    )
    assert decision["declaration"] == "residual_add"


def test_the_gemmini_spec_derives_every_hardware_shaped_field_but_the_contraction():
    from merlin.common.paths import repo_root
    from merlin.targetgen.software_spec import load_software_spec

    spec = load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml", target="gemmini")
    rows = _rows(spec)
    assert {name for name, row in rows.items() if "hardware" in row} == {
        "elementwise_map",
        "elementwise_map_standalone",
        "elementwise_map_after_operand_sum",
        "pooling_on_readout",
        "reduction_standalone",
        "movement",
    }
    assert rows["elementwise_map_after_operand_sum"]["status"] == "reviewed"
    assert rows["elementwise_map_after_operand_sum"]["numerical_contract"] == "operand_sum_exhaustive_i8_v1"
    assert spec["quantization"]["formats"][0]["eligible_operations"] == "from_facts"


def test_a_fused_form_reads_the_accumulator_its_readout_drains():
    facets = [{**_FACETS[0], "accumulator_dtype": "i32"}]
    resolved, _ = D.resolve_spec(_derived_spec(), _facts(readout_facets=facets))
    rows = _rows(resolved)
    # The pooling stage on the i8 readout reads the i32 accumulator, as the SW screen observes it.
    assert rows["pooling"]["signature"]["dtypes"] == ["int32", "int8"]
    assert rows["residual_add"]["signature"]["dtypes"] == ["int8"]
    assert D.spec_fact_drift(resolved, _facts(readout_facets=facets))["n_blocking"] == 0


def test_unchecked_semantic_paths_are_never_fact_derived():
    facts = _facts(semantic_facts=None)
    assert "standalone" not in facts["forms"]["elementwise_map"]
    resolved, resolution = D.resolve_spec(_derived_spec(), facts)
    row = _rows(resolved)["residual_add"]
    assert row["placement"] == "unknown"
    assert "derived_from_facts" not in row
    assert resolution["unresolved"]


@pytest.mark.parametrize("review", [None, "unreviewed", "reviewed"])
def test_unchecked_form_admission_requires_an_explicitly_reviewed_spec_row(review):
    spec = _derived_spec()
    original = _rows(spec)["residual_add"]
    if review is not None:
        original["status"] = review
    resolved, record = D.resolve_spec(spec, _facts(semantic_facts=None))
    row = _rows(resolved)["residual_add"]
    assert "derived_from_facts" not in row
    if review == "reviewed":
        assert row["placement"] == "accelerator"
        assert row["admission_basis"] == "declared_unchecked"
        declared = next(r for r in record["declarations"] if r["id"] == "residual_add")
        assert declared["basis"] == "declared_unchecked" and declared["evidence"] == {}
    else:
        assert row["placement"] == "unknown"
