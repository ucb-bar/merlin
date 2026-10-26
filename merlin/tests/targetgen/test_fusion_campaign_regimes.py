"""The fusion campaign reserves executed representatives for every required regime."""

import copy
import random

import pytest

from merlin.common.jsonio import canonical_json
from merlin.common.paths import repo_root
from merlin.compile.fusion_enumeration import campaign_stratum, stratify_candidates, verify_stratification
from merlin.targetgen.required_regimes import regime_identity, represents_required_regime, required_regimes

RUNTIME = dict(composition="shared_input_contractions", scheme="q_f32_tensor", axes=dict(scale_source="runtime_input"))


@pytest.fixture(scope="module")
def spec():
    from merlin.targetgen.software_spec import load_software_spec

    return load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml", target="gemmini")


def node(scheme, exit, *, realization="matmul", scale_source="constant", relu=False, form="contraction"):
    stages = [] if exit == "accumulator" else [dict(kind="convert"), dict(kind="scale"), dict(kind="requant")]
    if relu:
        stages.insert(0, dict(kind="activation"))
    return dict(
        form=form,
        realization=realization,
        readout=dict(exit=exit, stage_indices=list(range(len(stages))), stages=stages),
        seed=False,
        residual=None,
        scheme=scheme,
        regime=dict(scale_source=scale_source),
    )


def row(candidate_id, *nodes, dataflow="WS"):
    structure = "shared_input" if len(nodes) > 1 else "single"
    semantic = dict(structure=structure, nodes=sorted(nodes, key=canonical_json), dataflow=dataflow)
    return dict(
        candidate_id=candidate_id,
        semantic=semantic,
        state="candidate",
        reasons=[],
        pruning_scope="row",
        negative_control=False,
    )


def witness(candidate_id, *, relu, constant_realization="matmul"):
    """A shared WS graph: a constant raw sibling and a runtime-scaled narrow sibling."""
    return row(
        candidate_id,
        node("q_f32_tensor", "accumulator", realization=constant_realization),
        node("q_f32_tensor", "requant", scale_source="runtime_input", relu=relu),
    )


@pytest.fixture
def manifest():
    rows = [
        # conv_as_matmul siblings sort before the matmul witnesses, as in the real enumeration.
        *(witness(f"0{index}", relu=index % 2 == 1, constant_realization="conv_as_matmul") for index in range(4)),
        witness("f0-no-relu", relu=False),
        witness("f1-relu", relu=True),
        # Multi-regime rows: a packed-panel sibling beside a runtime tensor sibling.
        *(
            row(
                f"a{index}",
                node("q_f32_packed_panel", "accumulator"),
                node("q_f32_tensor", "requant", scale_source="runtime_input"),
            )
            for index in range(5)
        ),
        *(
            row(f"b{index}", node("q_f32_packed_panel", "requant"), node("q_f32_packed_panel", "accumulator"))
            for index in range(3)
        ),
        *(row(f"c{index}", node("q_f32_column_block", "requant")) for index in range(3)),
        *(row(f"d{index}", node("q_pow2_tensor", "accumulator")) for index in range(3)),
        *(row(f"e{index}", node("q_f32_per_operand_sum", "requant", form="operand_sum")) for index in range(3)),
        # No required regime: shared siblings under a scheme no shared-input chain selects.
        *(
            row(f"9{index}", node("q_f32_column_block", "requant"), node("q_f32_column_block", "accumulator"))
            for index in range(4)
        ),
        dict(row("77-refused", node("q_f32_tensor", "requant")), state="incompatible", reasons=[dict(reason="x")]),
    ]
    return dict(schema="merlin.fusion_candidates.v1", rows=sorted(rows, key=lambda r: r["candidate_id"]))


def executed(result):
    return {r["candidate_id"] for r in result["rows"] if r["state"] == "candidate"}


def test_regime_free_practice_key_alone_defers_the_runtime_witnesses(manifest):
    # The pre-regime campaign: two members per campaign_stratum in candidate_id order.
    seen = {}
    legacy = set()
    for candidate in manifest["rows"]:
        if candidate["state"] == "candidate":
            key = canonical_json(campaign_stratum(candidate["semantic"]))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= 2:
                legacy.add(candidate["candidate_id"])
    assert not {"f0-no-relu", "f1-relu"} & legacy


def test_every_required_regime_executes_min_n_eligible_representatives(manifest, spec):
    for limit in (1, 2, 3):
        result = stratify_candidates(
            manifest, dict(spec, compositions=dict(spec["compositions"], campaign_per_stratum=limit))
        )
        chosen = executed(result)
        coverage = {canonical_json(entry["regime"]): entry for entry in result["campaign"]["regime_coverage"]}
        assert len(coverage) == len(required_regimes(spec))
        for chain, scheme, axes in required_regimes(spec):
            eligible = [
                r["candidate_id"]
                for r in manifest["rows"]
                if r["state"] == "candidate" and represents_required_regime(r["semantic"], chain, scheme, axes)
            ]
            count = len(set(eligible) & chosen)
            assert count >= min(limit, len(eligible))
            if eligible:
                assert count >= 1
            entry = coverage[canonical_json(regime_identity(chain, scheme, axes))]
            assert (entry["eligible"], entry["executed"]) == (len(eligible), count)


def test_runtime_input_witnesses_are_executed(manifest, spec):
    result = stratify_candidates(manifest, spec)
    assert result["campaign"]["per_stratum"] == 2
    states = {r["candidate_id"]: r for r in result["rows"]}
    for identity in ("f0-no-relu", "f1-relu"):
        assert states[identity]["state"] == "candidate"
        assert states[identity]["stratum_key"]["required_regimes"] == [RUNTIME]
        assert states[identity]["stratum_key"]["realizations"] == ["matmul"]
    # The conv_as_matmul siblings form their own stratum inside the same regime.
    assert [states[f"0{i}"]["state"] for i in range(4)] == ["candidate"] * 2 + ["deferred_by_stratification"] * 2


def test_multi_regime_rows_join_every_matching_regime_stratum(manifest, spec):
    result = stratify_candidates(manifest, spec)
    states = {r["candidate_id"]: r for r in result["rows"]}
    assert states["a0"]["stratum_key"]["required_regimes"] == [
        dict(composition="shared_input_contractions", scheme="q_f32_packed_panel", axes={}),
        RUNTIME,
    ]
    assert [states[f"a{i}"]["state"] for i in range(5)] == ["candidate"] * 2 + ["deferred_by_stratification"] * 3


def test_rows_matching_no_regime_keep_the_practice_key(manifest, spec):
    result = stratify_candidates(manifest, spec)
    plain = [r for r in result["rows"] if r["candidate_id"].startswith("9")]
    for candidate in plain:
        assert candidate["stratum_key"] == campaign_stratum(candidate["semantic"])
    assert [r["state"] for r in plain] == ["candidate"] * 2 + ["deferred_by_stratification"] * 2
    refused = next(r for r in result["rows"] if r["candidate_id"] == "77-refused")
    assert refused["state"] == "incompatible" and "stratum_key" not in refused


def test_stratification_is_deterministic(manifest, spec):
    expected = canonical_json(stratify_candidates(manifest, spec))
    for seed in range(3):
        shuffled = copy.deepcopy(manifest)
        random.Random(seed).shuffle(shuffled["rows"])
        result = stratify_candidates(shuffled, spec)
        result["rows"].sort(key=lambda r: r["candidate_id"])
        assert canonical_json(result) == expected
    assert canonical_json(stratify_candidates(manifest, spec)) == expected


@pytest.mark.parametrize(
    "edit",
    ["defer_witness", "execute_deferred", "coverage", "drop_regime", "stratum_key", "definition"],
)
def test_reader_refuses_a_hand_edited_executed_set(manifest, spec, edit):
    result = stratify_candidates(manifest, spec)
    verify_stratification(result, spec, required=True)
    verify_stratification(result, required=True)
    changed = copy.deepcopy(result)
    rows = {r["candidate_id"]: r for r in changed["rows"]}
    if edit == "defer_witness":
        rows["f1-relu"]["state"] = "deferred_by_stratification"
    elif edit == "execute_deferred":
        rows["02"]["state"] = "candidate"
    elif edit == "coverage":
        changed["campaign"]["regime_coverage"][-1]["executed"] += 1
    elif edit == "drop_regime":
        # A consistently rewritten campaign without the runtime cross-check.
        without = copy.deepcopy(spec)
        without["compositions"]["chains"][-1]["quantization_schemes"]["cross_checks"] = []
        changed = stratify_candidates(manifest, without)
        verify_stratification(changed, without)
    elif edit == "stratum_key":
        rows["f0-no-relu"]["stratum_key"].pop("required_regimes")
    elif edit == "definition":
        changed["campaign"]["stratification"] = "merlin.fusion_campaign.practice_strata.v0"
    with pytest.raises(ValueError, match="stratification"):
        verify_stratification(changed, spec, required=True)
    if edit != "drop_regime":
        with pytest.raises(ValueError, match="stratification"):
            verify_stratification(changed, required=True)


def test_stratification_definition_is_in_the_generator_identity():
    from merlin.compile.fusion_enumeration import _generator_inventory

    assert "merlin.targetgen.required_regimes" in _generator_inventory()
    assert "enumerator" in _generator_inventory()


def test_lost_per_regime_guarantee_is_a_value_error(manifest, spec, monkeypatch):
    from merlin.compile import fusion_enumeration as FE

    # Collapse every stratum onto one key, so later regimes lose their representatives.
    monkeypatch.setattr(FE, "canonical_json", lambda value: "one stratum")
    with pytest.raises(ValueError, match="lost its per-regime guarantee"):
        stratify_candidates(manifest, dict(spec, compositions=dict(spec["compositions"], campaign_per_stratum=1)))
