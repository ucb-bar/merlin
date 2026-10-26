"""Composition coverage is finite, deterministic, source-bound and shape-generic."""

from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0.composition_generation import bind_capsule, prepare_entries

from merlin.common.paths import repo_root
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.composition_corpus import arbitrary_scale_grid, entries, held_out_shapes


@pytest.fixture
def software():
    return yaml.safe_load((repo_root() / "examples/gemmini/target/software-spec.yaml").read_text())


@pytest.fixture
def facts():
    return dict(
        tile_dim=4,
        columns_per_scale=3,
        dtype="int8",
        weight_panel_width=3,
        max_k_extension=256,
        output_path=dict(
            schema="output_path_v1",
            accumulator=dict(dtype="i32"),
            accumulator_stages=[dict(kind="preload", dtype="i32")],
            readout=[
                dict(kind="activation", function="relu", domain="accumulator"),
                dict(kind="convert", to="f32", rounding="half_even"),
                dict(kind="scale", dtype="f32", rounding="half_even", granularity="per_command", columns_per_scale=3),
                dict(kind="requant", to="i8", rounding="half_even", clamp=[-128, 127]),
                dict(kind="pool", reductions=["max"], padding="unknown"),
            ],
            exits=dict(accumulator=True, requant=True),
        ),
        dataflows=["WS", "OS"],
        producer_placement={p: ["accelerator"] for p in ["movement", "operand_sum", "contraction"]},
        operand_signed_bits=8,
        mesh_signed_bits=12,
        pool_window_within_register_fields=dict(size_bits=4, stride_bits=4),
        max_requires_store_path_pooling=True,
        k_extension_within=dict(max_k=256, partial_sum_bits=32),
        shared_input_resident=dict(capacity_bytes=4096),
    )


@pytest.fixture
def binding():
    return CS.CorpusBinding(
        target="synthetic",
        tile_dim=4,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L0", "L1"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )


def _selected(chain):
    selection = chain["quantization_schemes"]
    return [*selection["primary"], *(row["scheme"] for row in selection.get("cross_checks", []))]


def _declared_pairs(software):
    return {(c["id"], scheme) for c in software["compositions"]["chains"] for scheme in _selected(c)}


def test_deterministic_and_every_declared_pair(software, facts):
    first = entries(software, facts)
    assert first == entries(copy.deepcopy(software), copy.deepcopy(facts))
    pairs = Counter(
        (e["generalization"]["composition_chain"]["id"], e["generalization"]["quantization_scheme"]["id"])
        for e in first
    )
    assert set(pairs) == _declared_pairs(software)
    assert len({e["name"] for e in first}) == len(first)
    assert 40 <= len(first) <= 60
    for entry in first:
        assert (entry["M"], entry["K"], entry["N"]) not in held_out_shapes()


def test_collision_and_unknown_geometry_fail_closed(software, facts):
    with pytest.raises(ValueError, match="held-out"):
        entries(software, facts, excluded_shapes={(4, 4, 4)})
    del facts["tile_dim"]
    with pytest.raises(ValueError, match="tile_dim"):
        entries(software, facts)


def test_budget_overflow_raises_without_truncation(software, facts):
    from merlin.targetgen.corpus_synth import SynthesisError, synthesize

    document = dict(target="synthetic", cells=[], software_spec=software, composition_facts=facts)
    count = len(entries(software, facts))
    with pytest.raises(SynthesisError, match="budget"):
        synthesize(document, budget=count - 1, capability_contract={})
    assert len(synthesize(document, budget=count, capability_contract={})["capsules"]) == count


@pytest.mark.parametrize("selection", [{}, {"synthesize_compositions": True}])
def test_composition_synthesis_default_and_true_are_unchanged(software, facts, selection):
    from merlin.targetgen.corpus_synth import synthesize

    document = dict(target="synthetic", cells=[], software_spec=software, composition_facts=facts)
    expected = synthesize(document, capability_contract={})
    assert expected["capsules"] == entries(software, facts)
    assert synthesize(document, workload_spec=selection, capability_contract={}) == expected


def test_composition_optout_preserves_budget_and_reports_scope(software, facts):
    from merlin_experiments.phase0.composition_generation import coverage

    from merlin.targetgen.corpus_synth import SynthesisError, synthesize

    document = dict(
        target="synthetic",
        cells=[dict(cell="movement/i8", family="movement", dtype="i8", alignment="aligned")],
        boundaries=dict(extent_probes=[dict(boundary="tile_edge", edge=facts["tile_dim"])]),
        software_spec=software,
        composition_facts=facts,
    )
    workload = dict(max_synthesized_capsules=1, synthesize_compositions=False)
    result = synthesize(document, workload_spec=workload, capability_contract={})
    assert len(result["capsules"]) == result["provenance"]["budget"] == 1
    assert all(entry["op"] != "composition_program" for entry in result["capsules"])
    excluded = result["provenance"]["composition_coverage"]
    assert excluded == coverage([], excluded_spec=software)
    assert {(row["composition"], row["scheme"]) for row in excluded} == {
        (chain["id"], scheme["id"])
        for chain in software["compositions"]["chains"]
        for scheme in software["quantization"]["schemes"]["schemes"]
    }
    assert all(row["count"] == 0 and row["status"] == "out_of_scope" for row in excluded)
    assert all("workload_spec.synthesize_compositions" in row["reason"] for row in excluded)
    workload["synthesize_compositions"] = True
    with pytest.raises(SynthesisError, match="budget"):
        synthesize(document, workload_spec=workload, capability_contract={})


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_composition_optout_requires_a_boolean_in_synthesis_and_descriptor(tmp_path, value):
    from merlin.targetgen.corpus_synth import synthesize
    from merlin.targetgen.target_experiment import load_target_experiment

    workload = {"synthesize_compositions": value}
    with pytest.raises(ValueError, match="workload_spec.synthesize_compositions must be a bool"):
        synthesize(dict(target="synthetic", cells=[]), workload_spec=workload, capability_contract={})
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump(dict(target="synthetic", workload_spec=workload)))
    with pytest.raises(ValueError, match="workload_spec.synthesize_compositions must be a bool"):
        load_target_experiment(descriptor)


@pytest.mark.parametrize("has_schemes", [True, False])
def test_optout_needs_no_composition_facts_or_resolved_schemes(software, has_schemes):
    from merlin.targetgen.corpus_synth import synthesize

    if not has_schemes:
        del software["quantization"]
    result = synthesize(
        dict(target="synthetic", cells=[]),
        software_spec=software,
        workload_spec={"synthesize_compositions": False},
        capability_contract={},
    )
    assert result["capsules"] == []
    excluded = result["provenance"]["composition_coverage"]
    assert {row["composition"] for row in excluded} == {chain["id"] for chain in software["compositions"]["chains"]}
    assert all(row["status"] == "out_of_scope" for row in excluded)
    if not has_schemes:
        assert all(row["scheme"] is None for row in excluded)


@pytest.mark.parametrize("selection", [{}, {"synthesize_compositions": True}, {"synthesize_compositions": False}])
def test_descriptor_accepts_composition_synthesis_selection(tmp_path, selection):
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump(dict(target="synthetic", workload_spec=selection)))
    assert load_target_experiment(descriptor).workload_spec == selection


def test_property_searched_scales_and_predecessors():
    from merlin.runtime.exact_readout import f32_readout

    disagreements, predecessors = arbitrary_scale_grid()
    for row in disagreements:
        a, multiplier = row["witness"], row["multiplier"]
        assert round(a * Fraction(multiplier)) != int(f32_readout(a, multiplier, out_dtype="i32"))
    row = predecessors[0]
    products = np.asarray(row["vectors"], dtype=np.float32) * np.float32(row["multiplier"])
    assert any(np.any(products == np.nextafter(np.float32(2**e), np.float32(0))) for e in range(7))


def test_all_sources_parse_verify_and_compute_goldens(software, facts, binding):
    from merlin.common import mlir_query as mq
    from merlin.targetgen.capsule_golden import golden

    for entry in prepare_entries(entries(software, facts)):
        cap, source = CS.build(entry, binding)
        mq.parse(source).verify()
        bind_capsule(cap, source)
        from merlin.targetgen.contract.schemas import validate_capsule

        validate_capsule(cap)
        outputs = golden(cap)
        assert outputs and set(outputs) == set(cap["operation"]["attributes"]["outputs"])
        assert cap["expected"]["composition"]["source_op_indices"]
        assert cap["semantic"]["grading"] == "placement_and_instruction_audit"


def test_os_constants_and_runtime_twin_have_distinct_truth_and_source(software, facts, binding):
    corpus = prepare_entries(entries(software, facts))
    constant = next(e for e in corpus if e["composition_instance"]["shape_class"] == "os_constant")
    runtime = next(e for e in corpus if e["composition_instance"]["shape_class"] == "os_runtime")
    outside = next(e for e in corpus if e["composition_instance"]["shape_class"] == "os_bound_outside")
    assert constant["generalization"]["composition_truth"]["outcome"] == "pass"
    assert runtime["outcome"] == outside["outcome"] == "refuse"
    cap, source = CS.build(constant, binding)
    assert not any(row["role"] == "weight" for row in cap["inputs"])
    assert "%W0 = arith.constant dense<" in source
    cap, source = CS.build(runtime, binding)
    assert any(row["role"] == "weight" for row in cap["inputs"])
    assert "%W0 = arith.constant dense<" not in source


def test_every_selected_scheme_keeps_an_aligned_and_a_tail_case(software, facts):
    shapes = {}
    for entry in entries(software, facts):
        semantic = entry["generalization"]
        key = (semantic["composition_chain"]["id"], semantic["quantization_scheme"]["id"])
        shapes.setdefault(key, set()).add(entry["composition_instance"]["shape_class"])
    assert set(shapes) == _declared_pairs(software)
    for classes in shapes.values():
        assert "aligned" in classes
        assert classes & {"partial", "m_tail", "n_tail", "k_tail"}


def test_other_shape_classes_appear_once_per_chain(software, facts):
    corpus = entries(software, facts)
    per_chain = Counter(
        (e["generalization"]["composition_chain"]["id"], e["composition_instance"]["shape_class"]) for e in corpus
    )
    assert all(count == 1 for (_, label), count in per_chain.items() if label != "aligned")
    for twin in ("os_runtime", "os_bound_outside", "standalone_pool_unfenced", "padded_pool_unfenced"):
        assert sum(e["composition_instance"]["shape_class"] == twin for e in corpus) == 1
    regressions = [
        e
        for e in corpus
        if e["composition_instance"]["shape_class"] in {"regression_vectors", "regression_second", "pred_pow2"}
    ]
    assert len(regressions) == 3
    assert all(e["generalization"]["quantization_scheme"]["scale_value"] == "arbitrary_f32" for e in regressions)


def test_selection_is_declared_data_and_fails_closed(software, facts):
    chain = software["compositions"]["chains"][0]
    chain["quantization_schemes"]["primary"] = ["q_undeclared"]
    with pytest.raises(ValueError, match="undeclared schemes"):
        entries(software, facts)
    del chain["quantization_schemes"]
    with pytest.raises(ValueError, match="declares no quantization scheme selection"):
        entries(software, facts)


def test_duplicate_scheme_selection_is_rejected(software):
    from merlin.targetgen.software_spec import validate_compositions

    chain = software["compositions"]["chains"][0]
    chain["quantization_schemes"]["cross_checks"] = [
        dict(scheme=chain["quantization_schemes"]["primary"][0], justification="duplicate")
    ]
    with pytest.raises(ValueError, match="more than once"):
        validate_compositions(software["compositions"])


def test_every_cross_check_is_justified_and_pow2_stays_covered(software):
    chains = software["compositions"]["chains"]
    for chain in chains:
        for row in chain["quantization_schemes"].get("cross_checks", []):
            assert row["justification"].strip()
    schemes = software["quantization"]["schemes"]["schemes"]
    pow2 = {s["id"] for s in schemes if s["scale_value"] == "power_of_two"}
    assert sum(bool(pow2 & set(_selected(chain))) for chain in chains) == 1
    assert {s["id"] for s in schemes} <= {scheme for chain in chains for scheme in _selected(chain)}


def test_release_union_raises_on_overflow_and_preserves_the_roster(software, facts):
    from merlin_experiments.phase0.composition_generation import append_declared, coverage

    from merlin.targetgen.corpus_synth import SynthesisError

    corpus = entries(software, facts)
    with pytest.raises(SynthesisError, match="never truncate"):
        append_declared([], software, facts, budget=len(corpus) - 1)
    release = append_declared(corpus, software, facts, budget=len(corpus))
    assert len(release) == len(corpus)
    ledger = coverage(release)
    assert {(row["composition"], row["scheme"]) for row in ledger} == _declared_pairs(software)
    for row in ledger:
        assert row["role"] in {"primary", "cross_check"}
        assert (row["role"] == "cross_check") == bool(row.get("justification"))


def test_placement_rejects_partial_or_host_chain_members(software, facts, binding):
    from merlin_experiments.phase0.composition_generation import require_placement

    entry = next(e for e in prepare_entries(entries(software, facts)) if e.get("outcome") != "refuse")
    cap, source = CS.build(entry, binding)
    bind_capsule(cap, source)
    members = cap["expected"]["composition"]["source_op_indices"]
    screen = dict(decisions=[dict(status="admitted", placement="accelerator", source_op_indices=[i]) for i in members])
    require_placement(cap, screen)
    screen["decisions"].pop()
    with pytest.raises(ValueError, match="every original chain member"):
        require_placement(cap, screen)


def test_movement_pooling_has_fenced_positive_and_refusal_twins(software, facts, binding):
    facts["output_path"]["readout"][-1]["padding"] = "zero"
    corpus = prepare_entries(entries(software, facts))
    movement = [e for e in corpus if e["composition_instance"]["producer"] == "movement"]
    for label in ("standalone_pool", "padded_pool"):
        positive = [e for e in movement if e["composition_instance"]["shape_class"] == label]
        assert positive and all(e.get("outcome") != "refuse" for e in positive)
        twins = [e for e in movement if e["composition_instance"]["shape_class"] == label + "_unfenced"]
        assert len(twins) == len(positive) and all(e.get("outcome") == "refuse" for e in twins)
        for entry in positive:
            cap, source = CS.build(entry, binding)
            bind_capsule(cap, source)
            assert cap["expected"]["modes"]["scratchpad_source_pooling_fenced"]
            assert cap["expected"]["mode_parameters"]["pooled_store"]["fields"]["upad"] == int(label == "padded_pool")
            from merlin.common import mlir_query as mq

            mq.parse(source).verify()


def test_generated_bias_has_an_explicit_source_operand(software, facts, binding):
    entry = next(
        e for e in prepare_entries(entries(software, facts)) if e["composition_instance"]["shape_class"] == "bias_relu"
    )
    cap, source = CS.build(entry, binding)
    bind_capsule(cap, source)
    seed = cap["expected"]["mode_parameters"]["accumulator_preload"]["input"]["arg_index"]
    assert cap["inputs"][seed]["role"] == "bias"
    assert cap["input_palette"]["inputs"][-1]["name"] == cap["inputs"][seed]["name"]


def test_runtime_scale_cross_check_is_bounded_and_uses_its_input(software, facts, binding):
    from merlin_experiments.phase0.composition_generation import coverage

    from merlin.common import mlir_query as mq
    from merlin.targetgen.capsule_golden import golden, materialize_capsule_leaves

    corpus = prepare_entries(entries(software, facts))
    assert len(corpus) == 60
    runtime = [e for e in corpus if e["generalization"]["quantization_scheme"].get("scale_source") == "runtime_input"]
    assert runtime and {e["generalization"]["composition_chain_id"] for e in runtime} == {"shared_input_contractions"}
    ledger = [row for row in coverage(corpus) if row.get("axes") == {"scale_source": "runtime_input"}]
    assert len(ledger) == 1 and ledger[0]["role"] == "cross_check" and ledger[0]["justification"]
    cap, source = CS.build(runtime[0], binding)
    mq.parse(source).verify()
    bind_capsule(cap, source)
    name = cap["operation"]["attributes"]["case"]["runtime_scale_input"]
    index = next(i for i, leaf in enumerate(cap["inputs"]) if leaf["name"] == name)
    assert cap["inputs"][index] == dict(name=name, shape=[1], dtype="f32", role="input")
    assert f"%{name}: tensor<1xf32>" in source and "%readout0_gains = arith.constant" not in source
    assert cap["expected"]["mode_parameters"]["readout_multipliers"]["input"]["arg_index"] == index
    pattern = next(row for row in cap["input_palette"]["inputs"] if row["name"] == name)
    outputs = []
    for scale in (0.0, 0.25):
        pattern["values"] = [scale]
        values = {key: np.asarray(value.to_list()) for key, value in materialize_capsule_leaves(cap).items()}
        expected = np.rint(np.maximum(values["A"] @ values["W0"], 0).astype(np.float32) * np.float32(scale))
        result = golden(cap)
        np.testing.assert_array_equal(result["readout0"], np.clip(expected, -128, 127))
        outputs.append(result)
    assert outputs[0]["readout0"] != outputs[1]["readout0"]
    assert outputs[0]["mac1"] == outputs[1]["mac1"]
    # Keep the source's synthetic constant metadata unchanged: the reference reads the operand.
    assert cap["operation"]["attributes"]["case"]["multipliers"] != [0.25]


def test_runtime_axis_refuses_invalid_selection_and_missing_input(software, facts):
    from merlin.targetgen.functional_truth_compositions import expectation

    chain = next(c for c in software["compositions"]["chains"] if c["id"] == "shared_input_contractions")
    chain["quantization_schemes"]["cross_checks"][0]["axes"]["scale_source"] = "unknown"
    with pytest.raises(ValueError):
        entries(software, facts)
    chain["quantization_schemes"]["cross_checks"][0]["axes"]["scale_source"] = "runtime_input"
    semantic = next(
        e["generalization"]
        for e in entries(software, facts)
        if e["generalization"]["quantization_scheme"].get("scale_source") == "runtime_input"
    )
    case = copy.deepcopy(semantic["composition_instance"])
    del case["runtime_scale_input"]
    truth = expectation(chain, semantic["quantization_scheme"], case, facts=facts)
    assert truth["outcome"] == "refuse" and any("scale input" in reason for reason in truth["reasons"])


@pytest.fixture
def admitted_pool():
    rows, cases = [], []
    for index in range(25):
        identity = f"candidate-{index:02d}"
        rows.append(
            dict(
                candidate_id=identity,
                state="candidate",
                required_case_ids=[
                    f"{identity}-{label}" for label in ("aligned", "partial", "rne_saturation", "conversion_boundary")
                ],
                admitted=True,
                outcome="admitted",
                reasons=[],
                semantic=dict(
                    structure="single",
                    nodes=[
                        dict(
                            form="contraction",
                            scheme=f"scheme-{index}",
                            readout=dict(exit="requant", stages=[dict(kind="requant")]),
                        )
                    ],
                ),
            )
        )
        cases.extend(
            dict(candidate_id=identity, case_id=f"{identity}-{label}", shape_class=label)
            for label in ("aligned", "partial", "rne_saturation", "conversion_boundary")
        )
    return dict(rows=rows), dict(rows=cases)


def test_validated_sample_exact_shared_budget_and_minimum(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    result = sample_validated(validation, suite, software_spec={}, workload_count=110)
    assert result["mandatory_count"] == result["minimum_cover"] == 25
    assert result["composition_count"] == 50
    assert result["composition_count"] + result["workload_count"] == result["budget"] == 160
    assert result["unsampled_regimes"] == []
    limited = sample_validated(validation, suite, software_spec={}, workload_count=111)
    assert limited["composition_count"] == 49 and limited["underfill"] == 1
    # Full mandatory cover survives both the composition ceiling and shared budget.
    for index in range(40):
        row = copy.deepcopy(validation["rows"][0])
        row["candidate_id"] = f"extra-{index}"
        row["semantic"]["nodes"][0]["scheme"] = f"extra-scheme-{index}"
        validation["rows"].append(row)
        suite["rows"].append(dict(candidate_id=row["candidate_id"], case_id=row["candidate_id"], shape_class="partial"))
    with pytest.warns(UserWarning, match="emitting full cover"):
        result = sample_validated(validation, suite, software_spec={}, budget=60)
    assert result["mandatory_count"] == result["composition_count"] == 65
    assert result["over_limit"] and not result["unsampled_regimes"]


def test_sample_targets_fifty_and_ties_use_candidate_id(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:10]
    identities = {row["candidate_id"] for row in validation["rows"]}
    suite["rows"] = [row for row in suite["rows"] if row["candidate_id"] in identities]
    for index in range(20):
        suite["rows"].append(dict(candidate_id="candidate-00", case_id=f"extra-{index:02d}", shape_class="edge"))
    first = sample_validated(validation, suite, software_spec={})
    assert first["composition_count"] == 50 and first["cross_checks"] == 40
    assert first == sample_validated(
        dict(rows=list(reversed(validation["rows"]))), dict(rows=list(reversed(suite["rows"]))), software_spec={}
    )
    # Identical regimes choose the lexically first candidate, independent of row order.
    twin = copy.deepcopy(validation["rows"][0])
    twin["candidate_id"] = "candidate-00-earlier-twin"
    validation["rows"].insert(0, twin)
    suite["rows"].append(dict(candidate_id=twin["candidate_id"], case_id="twin", shape_class="aligned"))
    selected = sample_validated(validation, suite, software_spec={})["selected"]
    assert all(
        row["candidate_id"] == "candidate-00"
        for row in selected
        if row["role"] == "sampled_regime" and row["regime"] == selected[0]["regime"]
    )


def test_bound_synthesis_defers_aliases_to_phase0(software, facts):
    from merlin.targetgen.corpus_synth import synthesize

    declaration = dict(path="validation.json", sha256="a" * 64)
    document = dict(target="synthetic", cells=[], software_spec=software, composition_facts=facts)
    first = synthesize(document, workload_spec=dict(fusion_validation=declaration), capability_contract={})
    assert first["capsules"] == []
    software["compositions"]["chains"] = []
    assert synthesize(document, workload_spec=dict(fusion_validation=declaration), capability_contract={}) == first


def test_spec_can_bind_validation_without_authored_chains():
    from merlin.targetgen.composition_corpus import validation_binding
    from merlin.targetgen.software_spec import validate_compositions

    binding = dict(path="record.json", sha256="a" * 64)
    block = dict(
        schema="merlin.software_spec.compositions.v1",
        status="reviewed",
        grading="placement_and_instruction_audit",
        validation=binding,
    )
    assert validate_compositions(block) == block
    assert validation_binding(dict(compositions=block)) == binding
    with pytest.raises(ValueError, match="conflicting"):
        validation_binding(dict(compositions=block), dict(fusion_validation={**binding, "sha256": "b" * 64}))
    with pytest.raises(ValueError, match="sha256"):
        validation_binding({}, dict(fusion_validation={**binding, "sha256": "bad"}))


def test_tail_only_regime_needs_one_capsule(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:1]
    suite["rows"] = [
        row for row in suite["rows"] if row["candidate_id"] == "candidate-00" and row["shape_class"] != "aligned"
    ]
    result = sample_validated(validation, suite, software_spec={})
    assert result["minimum_cover"] == 1
    assert [row["shape_class"] for row in result["selected"] if row["role"] == "sampled_regime"] == ["partial"]


@pytest.mark.parametrize("admissions", [True, False])
def test_sampler_requires_the_selected_spec_and_nonzero_admissions(admitted_pool, admissions):
    from merlin.targetgen.composition_corpus import sample_validated
    from merlin.targetgen.corpus_synth import SynthesisError

    validation, suite = admitted_pool
    for row in validation["rows"]:
        row["admitted"] = admissions
    with pytest.raises(SynthesisError, match="selected software spec"):
        sample_validated(validation, suite)
    if not admissions:
        with pytest.raises(SynthesisError, match="at least one admitted"):
            sample_validated(validation, suite, software_spec={})


@pytest.mark.parametrize("missing", ["not_admitted", "absent", "wrong_producer", "no_case"])
def test_required_scheme_cannot_disappear_from_admitted_rows(software, admitted_pool, missing):
    from merlin.targetgen.composition_corpus import sample_validated
    from merlin.targetgen.corpus_synth import SynthesisError

    software["compositions"]["chains"] = [
        row for row in software["compositions"]["chains"] if row["producer"] == "operand_sum"
    ]
    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:2]
    for row, scheme in zip(validation["rows"], ("q_f32_tensor", "q_f32_per_operand_sum"), strict=True):
        row["semantic"]["nodes"][0].update(form="operand_sum", scheme=scheme)
    good = sample_validated(validation, suite, software_spec=software)
    assert good["composition_count"] == 8 and good["unsampled_regimes"] == []
    required = validation["rows"][1]
    if missing == "not_admitted":
        required["admitted"] = False
    elif missing == "absent":
        validation["rows"].pop()
    elif missing == "wrong_producer":
        required["semantic"]["nodes"][0]["form"] = "contraction"
    else:
        suite["rows"] = [row for row in suite["rows"] if row["candidate_id"] != required["candidate_id"]]
    with pytest.raises(SynthesisError, match="required fusion regimes|no validated case"):
        sample_validated(validation, suite, software_spec=software)


@pytest.mark.parametrize("mismatch", ["axis", "structure", "extension", "exit"])
def test_required_chain_and_explicit_axes_need_matching_representatives(software, admitted_pool, mismatch):
    from merlin.targetgen.composition_corpus import sample_validated
    from merlin.targetgen.corpus_synth import SynthesisError

    software["compositions"]["chains"] = [row for row in software["compositions"]["chains"] if row.get("multiplicity")]
    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:2]
    for row, scheme in zip(validation["rows"], ("q_f32_packed_panel", "q_f32_tensor"), strict=True):
        node = row["semantic"]["nodes"][0]
        node.update(scheme=scheme, regime=dict(scale_source="runtime_input"))
        row["semantic"].update(structure="shared_input", nodes=[node, copy.deepcopy(node)])
    assert sample_validated(validation, suite, software_spec=software)["unsampled_regimes"] == []
    required = validation["rows"][1]["semantic"]
    for node in required["nodes"]:
        if mismatch == "axis":
            node["regime"]["scale_source"] = "constant"
        elif mismatch == "extension":
            node["residual"] = dict(kind="integer", coefficient=1)
        elif mismatch == "exit":
            node["readout"]["exit"] = "UNKNOWN"
    if mismatch == "structure":
        required["structure"] = "single"
    with pytest.raises(SynthesisError, match="required fusion regimes"):
        sample_validated(validation, suite, software_spec=software)


def test_cover_uses_semantic_structure_and_tail_first(admitted_pool):
    from merlin.targetgen.composition_corpus import admitted_regime, fusion_structure_class, sample_validated

    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:1]
    base = validation["rows"][0]
    twin = copy.deepcopy(base)
    twin["candidate_id"] = "earlier-aligned"
    node = twin["semantic"]["nodes"][0]
    node.update(realization="conv_as_matmul", seed=True, regime=dict(scale_source="runtime_input"))
    node["readout"]["stages"].insert(0, dict(kind="activation"))
    assert admitted_regime(base["semantic"]) == admitted_regime(twin["semantic"])
    validation["rows"].append(twin)
    suite["rows"].append(dict(candidate_id=twin["candidate_id"], case_id="earlier", shape_class="aligned"))
    sample = sample_validated(validation, suite, software_spec={}, target_count=1)
    assert sample["mandatory_count"] == sample["composition_count"] == 1
    assert sample["selected"][0]["shape_class"] == "partial"
    assert fusion_structure_class(twin["semantic"]) == "contraction_epilogue"
    node["readout"]["stages"].append(dict(kind="pool"))
    node["readout"]["pool_padding_class"] = "padded"
    assert fusion_structure_class(twin["semantic"]) == "fused_pool_padded"
    node["readout"]["pool_padding_class"] = "unpadded"
    assert fusion_structure_class(twin["semantic"]) == "fused_pool_unpadded"
    for update, expected in [
        (dict(form="window_movement", realization="pool"), "standalone_pool"),
        (dict(realization="mean"), "mean"),
        (dict(form="operand_sum"), "scaled_loads"),
        (dict(residual=dict(kind="integer")), "residual_k_extension"),
    ]:
        node.update(update)
        assert fusion_structure_class(twin["semantic"]) == expected
    twin["semantic"]["structure"] = "shared_input"
    assert fusion_structure_class(twin["semantic"]) == "shared_input"


def test_declared_sample_target_controls_fill_and_underfill(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    spec = dict(compositions=dict(target_count=55))
    result = sample_validated(validation, suite, software_spec=spec)
    assert result["target_count"] == result["composition_count"] == 55
    assert result["underfill"] == 0
    limited = sample_validated(validation, suite, software_spec=spec, budget=30)
    assert limited["underfill"] == 25


def test_required_axes_remain_sampled_inside_one_operator_cell(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:2]
    for row, source in zip(validation["rows"], ("constant", "runtime_input"), strict=True):
        row["semantic"]["nodes"][0].update(scheme="one-scheme", regime=dict(scale_source=source))
    spec = dict(
        quantization=dict(
            schemes=dict(
                axes=dict(scale_source=["constant", "runtime_input"]),
                schemes=[
                    dict(
                        id="one-scheme",
                        scale_value="arbitrary_f32",
                        weight_scale_granularity="tensor",
                        readout_strategy="f32_product_rne",
                    )
                ],
            )
        ),
        compositions=dict(
            chains=[
                dict(
                    id="semantic-selection",
                    producer="contraction",
                    exits=["requant"],
                    quantization_schemes=dict(
                        primary=[],
                        cross_checks=[
                            dict(
                                scheme="one-scheme",
                                axes=dict(scale_source="runtime_input"),
                                justification="input scale",
                            )
                        ],
                    ),
                )
            ]
        ),
    )
    result = sample_validated(validation, suite, software_spec=spec, target_count=1)
    assert result["mandatory_count"] == 1 and result["composition_count"] == 2
    assert {row["candidate_id"] for row in result["selected"]} == {"candidate-00", "candidate-01"}


@pytest.mark.parametrize(
    "complete,reasons",
    [(False, []), (None, []), (True, ["pending_execution"]), (True, ["pending_execution: no positive required cases"])],
)
def test_bound_reader_refuses_incomplete_or_pending_execution(tmp_path, monkeypatch, complete, reasons):
    import hashlib
    import json
    from types import SimpleNamespace

    from merlin_experiments.phase0 import fusion_validation
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    record = tmp_path / "validation.json"
    record.write_text(json.dumps(dict(execution_complete=complete, rows=[dict(reasons=reasons)])))
    monkeypatch.setattr(fusion_validation, "verify_seal", lambda ledger: None)
    binding = dict.fromkeys(("candidates", "cases", "facts", "spec", "backend", "provider", "extension"), "unused")
    binding.update(path=str(record), sha256=hashlib.sha256(record.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="execution is incomplete"):
        read_bound_validation(binding, base=tmp_path, evidence=SimpleNamespace())


@pytest.mark.parametrize(
    "invalid", ["missing", "extra", "shape", "float_for_integer", "unsigned", "i8_overflow", "i32_overflow"]
)
def test_golden_validates_supplied_input_contract(software, facts, binding, invalid):
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate

    entry = next(e for e in entries(software, facts) if "preload" in e["composition_instance"]["accumulator_stages"])
    cap, _ = CS.build(entry, binding)
    values = {name: np.asarray(tensor.to_list()) for name, tensor in materialize_capsule_leaves(cap).items()}
    evaluate(cap, inputs=values)
    if invalid == "missing":
        del values["A"]
    elif invalid == "extra":
        values["extra"] = np.array([1])
    elif invalid == "shape":
        values["A"] = values["A"].reshape(-1)
    elif invalid == "float_for_integer":
        values["A"] = values["A"].astype(np.float32)
    elif invalid == "unsigned":
        values["A"] = np.zeros(values["A"].shape, dtype=np.uint8)
    elif invalid == "i8_overflow":
        values["A"].flat[0] = 128
    elif invalid == "i32_overflow":
        values["Bias"].flat[0] = 2**31
    with pytest.raises(ValueError, match="composition input"):
        evaluate(cap, inputs=values)


def test_mean_case_enumeration_refuses_unsafe_geometry(software, facts, binding):
    from merlin.targetgen.composition_capsules import build_composition

    entry = next(e for e in entries(software, facts) if e["composition_instance"].get("reduction_kind") == "sum")
    entry["K"] = 14
    with pytest.raises(ValueError, match="division and reciprocal product disagree"):
        build_composition(entry, binding, excluded_shapes=set())


@pytest.mark.parametrize("second_exit", ["accumulator", "requant"])
def test_validated_shared_input_builds_binds_and_computes_both_goldens(software, facts, binding, second_exit):
    import hashlib

    from merlin_experiments.phase0.composition_generation import _validated_entry
    from merlin_experiments.phase0.fusion_cases import _merge_sources

    from merlin.common import mlir_query as mq
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate
    from merlin.targetgen.contract.schemas import validate_capsule

    base = next(
        e
        for e in entries(software, facts)
        if e["generalization"]["quantization_scheme_id"] == "q_f32_tensor"
        and e["composition_instance"]["shape_class"] == "aligned"
        and e["composition_instance"]["producer"] == "contraction"
        and not e["composition_instance"].get("siblings")
        and "coefficient" not in e["composition_instance"]
    )
    nodes, semantic_nodes, values, references = [], [], {}, []
    for index, exit_kind in enumerate(("requant", second_exit)):
        entry = copy.deepcopy(base)
        case = entry["composition_instance"]
        case.update(exit=exit_kind, accumulator_stages=[])
        if exit_kind == "accumulator":
            case["readout_stages"] = []
        cap, source = CS.build(entry, binding)
        nodes.append((cap, source))
        env = {name: np.asarray(tensor.to_list()) for name, tensor in materialize_capsule_leaves(cap).items()}
        # Different weights ensure both siblings are independently evaluated.
        env["W0"] = np.full(env["W0"].shape, index + 1, dtype=np.int64)
        for name, value in env.items():
            values["A" if name == "A" else f"node{index}_{name}"] = value.tolist()
        references.extend(
            dict(name=f"node{index}_{name}", shape=list(np.shape(value)), values=value)
            for name, value in evaluate(cap, inputs=env).items()
        )
        semantic_nodes.append(
            dict(
                form="contraction",
                realization="matmul",
                scheme="q_f32_tensor",
                seed=False,
                residual=None,
                readout=dict(exit=exit_kind, stages=[dict(kind=kind) for kind in case["readout_stages"]]),
            )
        )
    source, inputs = _merge_sources(nodes, shared=True)
    payload = dict(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inputs=inputs,
        input_values=values,
        references=references,
        nodes=[cap for cap, _ in nodes],
        producer_views=[],
    )
    candidate = dict(
        candidate_id="derived_shared", semantic=dict(structure="shared_input", dataflow="WS", nodes=semantic_nodes)
    )
    case = dict(case_id="partial_shared", shape_class="partial")
    entry = _validated_entry(candidate, case, payload, facts, "a" * 64, "sampled_regime")
    cap, actual_source = CS.build(prepare_entries([entry])[0], binding)
    assert actual_source == source
    mq.parse(actual_source).verify()
    bind_capsule(cap, actual_source)
    validate_capsule(cap)
    assert evaluate(cap) == {row["name"]: row["values"] for row in references}
    assert [leaf["name"] for leaf in cap["inputs"]].count("A") == 1
    assert all(modes["shared_input"] for modes in cap["expected"]["stage_scoped_modes"].values())
    for name, parameters in cap["expected"]["stage_scoped_mode_parameters"].items():
        assert parameters["shared_input"]["input"]["arg_index"] == 0
        assert parameters[
            "readout_multipliers" if name.startswith("node0_") or second_exit == "requant" else "i32_exit"
        ]["outputs"]


def test_narrow_bypass_still_binds_an_identity_readout(software, facts, binding):
    entry = next(
        e
        for e in entries(software, facts)
        if e["composition_instance"]["producer"] == "contraction"
        and e["composition_instance"]["exit"] == "requant"
        and not e["composition_instance"].get("siblings")
    )
    entry["composition_instance"]["readout_stages"] = ["requant"]
    entry["composition_instance"]["multipliers"] = [1.0] * len(entry["composition_instance"]["multipliers"])
    entry["generalization"]["composition_instance"] = copy.deepcopy(entry["composition_instance"])
    cap, source = CS.build(prepare_entries([entry])[0], binding)
    bind_capsule(cap, source)
    assert cap["expected"]["modes"]["readout_multipliers"]
    assert set(cap["expected"]["mode_parameters"]["readout_multipliers"]["acc_scale_bits"]) == {0x3F800000}


@pytest.mark.parametrize(
    "bad_scale", [np.array([1], dtype=np.int32), np.array([np.nan]), np.array([np.inf]), np.array([1e100])]
)
def test_golden_refuses_invalid_runtime_input_dtypes_and_values(software, facts, binding, bad_scale):
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate

    entry = next(
        e
        for e in entries(software, facts)
        if e["generalization"]["quantization_scheme"].get("scale_source") == "runtime_input"
    )
    cap, _ = CS.build(entry, binding)
    values = {name: np.asarray(tensor.to_list()) for name, tensor in materialize_capsule_leaves(cap).items()}
    values[cap["operation"]["attributes"]["case"]["runtime_scale_input"]] = bad_scale
    with pytest.raises(ValueError, match="requires finite f32"):
        evaluate(cap, inputs=values)


def test_golden_uses_the_frozen_literal_seed(software, facts, binding):
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate

    entry = next(e for e in entries(software, facts) if "preload" in e["composition_instance"]["accumulator_stages"])
    cap, _ = CS.build(entry, binding)
    env = {name: np.asarray(tensor.to_list()) for name, tensor in materialize_capsule_leaves(cap).items()}
    cap["operation"]["attributes"]["case"]["constant_seed"] = env["Bias"].tolist()
    expected = evaluate(cap, inputs=env)
    cap["inputs"] = [leaf for leaf in cap["inputs"] if leaf["name"] != "Bias"]
    del env["Bias"]
    assert evaluate(cap, inputs=env) == expected


def test_mixed_sibling_exits_add_no_extra_operator_obligation(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    validation["rows"] = validation["rows"][:3]
    for row, exits in zip(
        validation["rows"],
        [("accumulator", "accumulator"), ("requant", "requant"), ("accumulator", "requant")],
        strict=True,
    ):
        nodes = [copy.deepcopy(row["semantic"]["nodes"][0]) for _ in exits]
        for node, exit_kind in zip(nodes, exits, strict=True):
            node["scheme"] = "one-scheme"
            node["readout"]["exit"] = exit_kind
        row["semantic"].update(structure="shared_input", nodes=nodes)
    result = sample_validated(validation, suite, software_spec={}, target_count=1)
    assert len(result["admitted_regimes"]) == result["mandatory_count"] == result["composition_count"] == 2
    assert not result["unsampled_regimes"]


@pytest.mark.parametrize(
    "state", ["deferred_by_stratification", "pending_facts", "incompatible", "not_selected_for_execution", "absent"]
)
def test_required_regime_blocks_even_without_any_execution(admitted_pool, state, tmp_path):
    import json

    from merlin.targetgen.composition_corpus import sample_validated
    from merlin.targetgen.corpus_synth import SynthesisError

    validation, suite = admitted_pool
    spec = dict(quantization=dict(schemes=dict(schemes=[dict(id="scheme-0")])))
    validation["rows"] = validation["rows"][:1]
    validation["scope"] = dict(campaign=dict(per_stratum=2))
    if state == "absent":
        validation["rows"].pop(0)
    else:
        validation["rows"][0].update(state=state, admitted=False, required_case_ids=[])
    path = tmp_path / "selection.json"
    with pytest.raises(SynthesisError, match="required fusion regimes") as error:
        sample_validated(validation, suite, software_spec=spec, selection_path=path)
    selection = json.loads(path.read_bytes())
    assert selection == error.value.selection
    assert selection["status"] == "blocked" and selection["selected"] == []
    (blocker,) = selection["seal_blockers"]
    assert blocker["regime"]["scheme"] == "scheme-0"
    assert blocker["counts_by_state"] == ({} if state == "absent" else {state: 1})
    assert blocker["reason"] == ("not enumerated" if state == "absent" else "no executed, admitted representative")


def test_workload_target_cannot_override_spec_target(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    spec = dict(compositions=dict(target_count=55))
    with pytest.raises(ValueError, match="conflicts"):
        sample_validated(validation, suite, software_spec=spec, target_count=50)
    assert sample_validated(validation, suite, software_spec=spec, target_count=55)["target_count"] == 55


def test_over_limit_has_structured_excess(admitted_pool):
    from merlin.targetgen.composition_corpus import sample_validated

    validation, suite = admitted_pool
    with pytest.warns(UserWarning):
        selection = sample_validated(validation, suite, software_spec={}, budget=30, workload_count=10)
    assert selection["over_limit"]
    assert selection["limit_excess"] == dict(composition=0, budget=5)


@pytest.mark.parametrize("reason, waived", [("", False), ("reviewed exception", True)])
def test_only_explicit_reasoned_per_regime_waiver_is_honored(software, admitted_pool, reason, waived):
    from merlin.targetgen.composition_corpus import sample_validated
    from merlin.targetgen.corpus_synth import SynthesisError

    validation, suite = admitted_pool
    chain = next(row for row in software["compositions"]["chains"] if row["id"] == "scaled_operand_sum")
    software["compositions"]["chains"] = [chain]
    from merlin.targetgen.composition_corpus import _required_regimes

    chain["regime_waivers"] = [
        dict(scheme=scheme, axes=axes, reason=reason) for _, scheme, axes in _required_regimes(software)
    ]
    if waived:
        assert sample_validated(validation, suite, software_spec=software)["seal_blockers"] == []
    else:
        with pytest.raises(SynthesisError, match="required fusion regimes"):
            sample_validated(validation, suite, software_spec=software)


def test_regime_waivers_are_validated_spec_fields(software):
    from merlin.targetgen.contract.schemas import validate

    chain = software["compositions"]["chains"][0]
    chain["regime_waivers"] = [dict(scheme=chain["quantization_schemes"]["primary"][0], reason="reviewed exception")]
    validate(software["compositions"], "software_compositions")
    chain["regime_waivers"][0].pop("reason")
    with pytest.raises(ValueError, match="reason"):
        validate(software["compositions"], "software_compositions")


def test_single_rounding_scaled_operands_carry_source_order_witnesses(software, facts):
    from merlin.targetgen.composition_corpus import operand_sum_rounding_witnesses

    chain = next(row for row in software["compositions"]["chains"] if row["producer"] == "operand_sum")
    chain["quantization_schemes"] = dict(primary=["q_f32_tensor"])
    software["compositions"]["chains"] = [chain]
    scaled = [
        entry
        for entry in entries(software, facts)
        if entry["generalization"]["shape_class"] == "scaled_operands"
        and entry["generalization"]["quantization_scheme"]["operand_sum_rounding"] == "single_rounding"
    ]
    assert scaled
    for entry in scaled:
        instance = entry["generalization"]["composition_instance"]
        witnesses = set(operand_sum_rounding_witnesses(tuple(instance["operand_multipliers"]), 8))
        palette = {row["name"]: row["values"] for row in entry["input_palette"]["inputs"]}
        assert witnesses and set(zip(palette["Left"], palette["Right"], strict=True)) <= witnesses
