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


def test_deterministic_and_every_declared_pair(software, facts):
    first = entries(software, facts)
    assert first == entries(copy.deepcopy(software), copy.deepcopy(facts))
    pairs = Counter(
        (e["generalization"]["composition_chain"]["id"], e["generalization"]["quantization_scheme"]["id"])
        for e in first
    )
    assert set(pairs) == {
        (c["id"], s["id"])
        for c in software["compositions"]["chains"]
        for s in software["quantization"]["schemes"]["schemes"]
    }
    assert len({e["name"] for e in first}) == len(first)
    assert len(first) <= 160
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


def test_pairwise_scheme_probes_share_an_extra_shape(software, facts):
    from itertools import combinations

    corpus = entries(software, facts)
    for chain in software["compositions"]["chains"]:
        shapes = {}
        for entry in corpus:
            semantic = entry["generalization"]
            if semantic["composition_chain"]["id"] == chain["id"]:
                shapes.setdefault(semantic["quantization_scheme"]["id"], set()).add(
                    entry["composition_instance"]["shape_class"]
                )
        for left, right in combinations(shapes.values(), 2):
            assert (left & right) - {"aligned"}


def test_release_union_raises_on_overflow_and_preserves_the_roster(software, facts):
    from merlin_experiments.phase0.composition_generation import append_declared, coverage

    from merlin.targetgen.corpus_synth import SynthesisError

    corpus = entries(software, facts)
    with pytest.raises(SynthesisError, match="never truncate"):
        append_declared([], software, facts, budget=len(corpus) - 1)
    release = append_declared(corpus, software, facts, budget=len(corpus))
    assert len(release) == len(corpus)
    assert len(coverage(release)) == len(software["compositions"]["chains"]) * len(
        software["quantization"]["schemes"]["schemes"]
    )


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
