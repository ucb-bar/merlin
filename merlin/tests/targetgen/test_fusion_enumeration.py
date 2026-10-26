"""The minimal fusion denominator is complete, deterministic and evidence-bound."""

import random

import pytest
import yaml

from merlin.common.jsonio import canonical_json, canonical_sha256
from merlin.common.paths import repo_root
from merlin.compile.fusion_enumeration import enumerate_candidates, write_manifest
from merlin.targetgen.contract.schemas import validate


def evidence(value):
    return {"status": "derived", "value": value, "evidence": ["synthetic#/proof", "synthetic#/source"]}


@pytest.fixture
def snapshot():
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [{"kind": "preload", "dtype": "i32"}],
        "readout": [
            {"kind": "activation", "function": "relu", "domain": "accumulator"},
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
        "exits": {"accumulator": True, "requant": True},
    }
    fields = {
        key: evidence(True)
        for key in (
            "bypass.0",
            "bypass.1",
            "bias_preload",
            "accumulate",
            "operand_sum",
            "pool_window",
            "mean_realization",
            "shared_input",
            "os_prefix_bound",
            "k_extension",
        )
    }
    fields["operand_bits"] = evidence(3)
    fields["mesh_signed_bits"] = evidence(16)
    fields["k_extension_within"] = evidence({"max_k": 127, "partial_sum_bits": 16})
    fields["weight_panel_width"] = evidence(4)
    fields["shared_input_resident"] = evidence({"capacity_bytes": 4096})
    fields["producer_placement"] = evidence(
        {"contraction": ["accelerator"], "operand_sum": ["accelerator"], "movement": ["accelerator"]}
    )
    fields["accumulator_bits"] = evidence(16)
    fields["dataflows"] = evidence(["weight_stationary", "output_stationary"])
    fields["shared_input_representations"] = evidence(
        {
            scheme: {
                "dtype": "i8",
                "zero_point": 0,
                "layout": "row_major_contiguous",
                "code_scale": "input_scale_0",
                "immutable": True,
            }
            for scheme in ("exact", "float")
        }
    )
    spec = {
        "operations": [{"families": ["contraction"], "placement": "accelerator", "ranks": [2, 4]}],
        "quantization": {
            "schemes": {
                "schema": "merlin.quantization_schemes.v1",
                "status": "reviewed",
                "axes": {
                    "scale_value": ["power_of_two", "arbitrary_f32"],
                    "weight_scale_granularity": ["tensor"],
                    "activation_scale_granularity": ["tensor"],
                    "readout_exit": ["narrow_saturating", "accumulator_full"],
                    "operand_sum_rounding": ["single_rounding"],
                    "residual_coefficient": ["none", "integer_constant"],
                },
                "constraints": [],
                "schemes": [
                    {
                        "id": "exact",
                        "scale_value": "power_of_two",
                        "weight_scale_granularity": "tensor",
                        "operand_sum_rounding": "single_rounding",
                        "readout_strategy": "exact_rational_host",
                    },
                    {
                        "id": "float",
                        "scale_value": "arbitrary_f32",
                        "weight_scale_granularity": "tensor",
                        "operand_sum_rounding": "single_rounding",
                        "readout_strategy": "f32_product_rne",
                    },
                ],
            }
        },
    }
    return {
        "target": "synthetic",
        "facts": {"fusion_enumeration": fields},
        "contract": {"output_epilogue": {"output_path": path}},
        "spec": spec,
    }


def test_complete_bounded_grammar_and_schema(snapshot):
    manifest = enumerate_candidates(**snapshot)
    assert manifest["candidate_count"] == 2164
    assert manifest["summary"]["forms"] == {"contraction": 288, "shared_input": 1872, "serial": 4}
    assert len({row["candidate_id"] for row in manifest["rows"]}) == manifest["candidate_count"]
    for row in manifest["rows"]:
        assert row["candidate_id"] == canonical_sha256(row["semantic"])
        assert row["semantic"]["profile"] == manifest["profile"]["schema"]
        if row["state"] != "candidate":
            assert row["reasons"] and all(r["reason"] and r["evidence"] for r in row["reasons"])
    validate(manifest, "fusion_candidates")


def shuffle_inputs(value, randomizer, key=""):
    if isinstance(value, dict):
        items = list(value.items())
        randomizer.shuffle(items)
        return {k: shuffle_inputs(v, randomizer, k) for k, v in items}
    if isinstance(value, list):
        result = [shuffle_inputs(v, randomizer) for v in value]
        if key not in {"readout", "clamp"}:
            randomizer.shuffle(result)
        return result
    return value


def test_byte_identical_under_shuffled_inputs(snapshot, tmp_path):
    expected = canonical_json(enumerate_candidates(**snapshot))
    for seed in range(3):
        assert canonical_json(enumerate_candidates(**shuffle_inputs(snapshot, random.Random(seed)))) == expected
    path = write_manifest(enumerate_candidates(**snapshot), destination=tmp_path / "fusion-candidates.json")
    assert path.read_bytes() == expected + b"\n"


def test_composition_catalog_is_not_an_input(snapshot):
    first = canonical_json(enumerate_candidates(**snapshot))
    snapshot["spec"]["compositions"] = {"chains": [{"id": "unrelated", "producer": "bogus"}]}
    assert canonical_json(enumerate_candidates(**snapshot)) == first
    snapshot["spec"]["compositions"]["chains"][0]["id"] = "renamed"
    assert canonical_json(enumerate_candidates(**snapshot)) == first


def test_seed_source_classes_and_structural_owners(snapshot):
    from merlin.common.mlir_query import parse
    from merlin.compile.fusion_enumeration import seed_source

    nodes = [n for row in enumerate_candidates(**snapshot)["rows"] for n in row["semantic"].get("nodes", [])]
    assert {n["seed_source"] for n in nodes if n["seed"]} == {"constant", "runtime_argument"}
    module = parse("""builtin.module { func.func @f(%constant: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %runtime = arith.constant dense<7> : tensor<2x2xi32>
      %scalar = arith.constant 3 : i32
      %splat = tensor.splat %scalar : tensor<2x2xi32>
      %unknown = tensor.empty() : tensor<2x2xi32>
      func.return %runtime : tensor<2x2xi32>
    }}""")
    block = next(op for op in module.walk() if op.name == "func.func").body.block
    assert seed_source(block.args[0]) == "runtime_argument"
    assert [seed_source(op.results[0]) for op in block.ops if op.results] == [
        "constant",
        "constant",
        "constant",
        "UNKNOWN",
    ]


def test_real_spec_chain_list_does_not_affect_bytes(snapshot):
    snapshot["spec"] = yaml.safe_load((repo_root() / "examples/gemmini/target/software-spec.yaml").read_text())
    # The synthetic overrides must agree with the selected spec's arithmetic premises.
    mesh = snapshot["spec"]["numerical_semantics"]["native_unit_contract"]["mesh_mac"]
    snapshot["facts"]["fusion_enumeration"]["mesh_signed_bits"] = evidence(mesh["signed_bits"])
    snapshot["facts"]["fusion_enumeration"]["dataflows"] = evidence(mesh["dataflows"])
    # Use only one scheme to keep this input-authority check inexpensive.
    snapshot["spec"]["quantization"]["schemes"]["schemes"] = snapshot["spec"]["quantization"]["schemes"]["schemes"][:1]
    first = canonical_json(enumerate_candidates(**snapshot))
    for row in snapshot["spec"]["compositions"]["chains"]:
        row["id"] = "renamed_" + row["id"]
    assert canonical_json(enumerate_candidates(**snapshot)) == first
    del snapshot["spec"]["compositions"]
    assert canonical_json(enumerate_candidates(**snapshot)) == first


def test_removing_fact_preserves_ids_and_explains_pending(snapshot):
    before = {r["candidate_id"]: r for r in enumerate_candidates(**snapshot)["rows"]}
    del snapshot["facts"]["fusion_enumeration"]["shared_input_resident"]
    after = {r["candidate_id"]: r for r in enumerate_candidates(**snapshot)["rows"]}
    assert before.keys() == after.keys()
    changed = 0
    for identity, row in after.items():
        if before[identity]["state"] == "candidate" and row["semantic"]["structure"] == "shared_input":
            assert row["state"] == "pending_facts"
            assert any(reason["evidence"] == "fusion_facts#/fields/shared_input_resident" for reason in row["reasons"])
            changed += 1
        else:
            assert row == before[identity]
    assert changed > 0


def test_mixed_scheme_siblings_have_independent_readouts_and_seeds(snapshot):
    shared = [r for r in enumerate_candidates(**snapshot)["rows"] if r["semantic"]["structure"] == "shared_input"]
    assert any(
        {n["scheme"] for n in r["semantic"]["nodes"]} == {"exact", "float"}
        and {n["readout"]["exit"] for n in r["semantic"]["nodes"]} == {"accumulator", "requant"}
        and {n["seed"] for n in r["semantic"]["nodes"]} == {True, False}
        and r["state"] == "candidate"
        for r in shared
    )
    pairs = [canonical_json(r["semantic"]) for r in shared]
    assert len(pairs) == len(set(pairs))


def test_all_integer_coefficients_and_fractional_negative_controls(snapshot):
    manifest = enumerate_candidates(**snapshot)
    bounds = manifest["extension_bounds"]
    assert bounds["T_max"] == 30
    assert bounds["coefficient_bound"] == 90
    assert {(p["sign"], p["tiles"]) for p in bounds["partitions"]} == {
        (sign, tile) for sign in (-1, 1) for tile in range(1, 31)
    }
    for proof in bounds["boundary_proofs"]:
        assert sum(proof["tile_digits"]) == proof["coefficient"]
        assert len(proof["tile_digits"]) <= 30
        assert all(abs(digit) <= 3 for digit in proof["tile_digits"])
    controls = [
        row
        for row in manifest["rows"]
        if any((node["residual"] or {}).get("kind") == "fractional" for node in row["semantic"].get("nodes", []))
    ]
    assert controls and all(row["state"] == "incompatible" and row["negative_control"] for row in controls)
    assert all(any("exact integer" in reason["reason"] for reason in row["reasons"]) for row in controls)
    assert not any(
        (n["residual"] or {}).get("coefficient") == 0
        for row in manifest["rows"]
        for n in row["semantic"].get("nodes", [])
    )


def test_missing_controls_and_conflicting_evidence_fail_closed(snapshot):
    del snapshot["facts"]["fusion_enumeration"]["bypass.0"]
    rows = enumerate_candidates(**snapshot)["rows"]
    assert any(
        r["state"] == "pending_facts" and any("bypass.0" in issue["evidence"] for issue in r["reasons"]) for r in rows
    )
    snapshot["contract"]["readout_semantics"] = {"fusion_enumeration": {"shared_input": evidence(False)}}
    rows = enumerate_candidates(**snapshot)["rows"]
    shared = [r for r in rows if r["semantic"]["structure"] == "shared_input"]
    assert shared and all(r["state"] == "incompatible" for r in shared)


def test_zero_tap_max_pool_requires_nonnegative_prefix_for_padding(snapshot):
    path = snapshot["contract"]["output_epilogue"]["output_path"]
    path["readout"].append({"kind": "pool", "reductions": ["max"], "padding": "unknown"})
    snapshot["facts"]["fusion_enumeration"]["bypass.3"] = evidence(True)
    missing = enumerate_candidates(**snapshot)
    assert any(
        any(r.get("missing_fact", "").endswith("/padding") for r in row["reasons"])
        for row in missing["rows"]
        if row["state"] == "pending_facts"
    )
    path["readout"][-1]["padding"] = "zero_tap"
    known = enumerate_candidates(**snapshot)
    pooled = [
        row
        for row in known["rows"]
        if any(s["kind"] == "pool" for node in row["semantic"].get("nodes", []) for s in node["readout"]["stages"])
    ]
    admitted = [
        node
        for row in pooled
        if row["state"] == "candidate"
        for node in row["semantic"]["nodes"]
        if any(s["kind"] == "pool" for s in node["readout"]["stages"])
    ]
    assert {n["readout"]["pool_padding"] for n in admitted} == {"unpadded_only", "zero_or_padded"}
    assert {n["readout"]["pool_padding_class"] for n in admitted} == {"unpadded", "padded"}
    for node in admitted:
        activation = any(s.get("function") == "relu" for s in node["readout"]["stages"])
        assert (node["readout"]["pool_padding"] == "zero_or_padded") == activation
        if node["readout"]["pool_padding_class"] == "padded":
            assert activation


def test_mean_realization_is_bound_to_its_single_f32_scheme(snapshot):
    snapshot["spec"]["operations"][0]["families"].append("window_mean")
    rows = enumerate_candidates(**snapshot)["rows"]
    means = [row for row in rows if any(n["realization"] == "mean" for n in row["semantic"].get("nodes", []))]
    exact = [row for row in means if row["semantic"]["nodes"][0]["scheme"] == "exact"]
    assert exact and all(row["state"] == "incompatible" for row in exact)
    assert any(any("single f32 boundary" in r["reason"] for r in row["reasons"]) for row in exact)


def test_safety_limit_fails_without_truncation(snapshot, monkeypatch):
    from merlin.compile import fusion_enumeration

    monkeypatch.setitem(fusion_enumeration.PROFILE, "max_rows", 20)
    with pytest.raises(ValueError, match="safety limit"):
        enumerate_candidates(**snapshot)


def test_all_four_producer_forms_have_a_complete_denominator(snapshot):
    snapshot["spec"]["operations"][0]["families"] += ["operand_sum", "movement", "reduction", "window_mean"]
    manifest = enumerate_candidates(**snapshot)
    assert manifest["candidate_count"] < 2500
    forms = {
        row["semantic"]["structure"]
        if row["semantic"]["structure"] != "single"
        else row["semantic"]["nodes"][0]["form"]
        for row in manifest["rows"]
    }
    assert forms == {"contraction", "operand_sum", "window_movement", "shared_input", "serial"}


def test_pure_enumeration_does_not_discover_or_regenerate_facts(snapshot, monkeypatch):
    from merlin.targetgen import capability_manifests
    from merlin.targetgen.rtl import facts, semantic_facts

    def forbidden(*args, **kwargs):
        pytest.fail("pure enumeration attempted external derivation")

    monkeypatch.setattr(facts, "load_facts", forbidden)
    monkeypatch.setattr(semantic_facts, "derive", forbidden)
    monkeypatch.setattr(capability_manifests, "_derived_output_epilogue", forbidden)
    enumerate_candidates(**snapshot)
    del snapshot["contract"]["output_epilogue"]
    manifest = enumerate_candidates(**snapshot)
    assert all(row["state"] != "candidate" for row in manifest["rows"])
    assert any(any("ordered output path" in issue["reason"] for issue in row["reasons"]) for row in manifest["rows"])


def test_regime_bytes_are_part_of_candidate_identity(snapshot):
    before = {
        row["candidate_id"]
        for row in enumerate_candidates(**snapshot)["rows"]
        if any(n["scheme"] == "float" for n in row["semantic"].get("nodes", []))
    }
    scheme = snapshot["spec"]["quantization"]["schemes"]["schemes"][1]
    scheme.update(scale_value="power_of_two", readout_strategy="exact_rational_host")
    after = {
        row["candidate_id"]
        for row in enumerate_candidates(**snapshot)["rows"]
        if any(n["scheme"] == "float" for n in row["semantic"].get("nodes", []))
    }
    assert before.isdisjoint(after)


def test_shared_input_quantization_mismatch_is_incompatible(snapshot):
    inputs = snapshot["facts"]["fusion_enumeration"]["shared_input_representations"]["value"]
    inputs["float"]["zero_point"] = 1
    rows = enumerate_candidates(**snapshot)["rows"]
    mixed = [
        r
        for r in rows
        if r["semantic"]["structure"] == "shared_input" and len({n["scheme"] for n in r["semantic"]["nodes"]}) == 2
    ]
    assert not mixed  # Only identical operand representations enter scheme tuples.


@pytest.mark.parametrize("authored", [None, "matching", "width_conflict", "flow_conflict"])
def test_composition_derivation_is_the_width_and_dataflow_authority(snapshot, monkeypatch, authored):
    from merlin.compile.fusion_enumeration import normalize_fusion_facts
    from merlin.targetgen import composition_facts

    calls = []

    def derived(**kwargs):
        calls.append(kwargs)
        return {
            "facts": {"operand_signed_bits": 7, "dataflows": ["WS", "OS"]},
            "evidence": {"operand_signed_bits": "shared#/width", "dataflows": "shared#/flow"},
            "unknown": {},
        }

    monkeypatch.setattr(composition_facts, "derive", derived)
    fields = snapshot["facts"]["fusion_enumeration"]
    if authored is None:
        del fields["operand_bits"]
        del fields["dataflows"]
    else:
        fields["operand_bits"] = evidence(3 if authored == "width_conflict" else 7)
        fields["dataflows"] = evidence(["bogus"] if authored == "flow_conflict" else ["OS", "WS"])
    if authored in {"width_conflict", "flow_conflict"}:
        with pytest.raises(ValueError, match="conflicts with derived value"):
            normalize_fusion_facts(**snapshot)
        return
    normalized = normalize_fusion_facts(**snapshot)
    assert len(calls) == 1
    assert normalized["fields"]["operand_bits"] == {"status": "derived", "value": 7, "evidence": ["shared#/width"]}
    assert normalized["fields"]["dataflows"]["value"] == ["OS", "WS"]


def test_extension_reaches_hundreds_and_checks_the_first_refused_value(snapshot):
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    fields = snapshot["facts"]["fusion_enumeration"]
    fields["operand_bits"] = evidence(8)
    fields["weight_panel_width"] = evidence(16)
    fields["k_extension_within"] = evidence({"max_k": 8191, "partial_sum_bits": 32})
    bounds = enumerate_candidates(**snapshot)["extension_bounds"]
    assert bounds["T_max"] == 510
    assert bounds["coefficient_bound"] == 64770
    assert any(p["abs_min"] <= 610 <= p["abs_max"] for p in bounds["partitions"])
    assert lower_k_extension(610, **bounds["representative"])["tile_digits"] == [127] * 4 + [102]
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="capacity"):
            lower_k_extension(sign * (bounds["coefficient_bound"] + 1), **bounds["representative"])


def test_authored_width_cannot_override_the_actual_datapath(snapshot):
    from merlin.compile.fusion_enumeration import normalize_fusion_facts

    snapshot["facts"]["datapaths"] = [
        dict(name="input", dtype="i8"),
        dict(name="accumulator", dtype="i32"),
    ]
    with pytest.raises(ValueError, match="operand_bits conflicts with derived value 8"):
        normalize_fusion_facts(**snapshot)


def test_prefix_bound_can_limit_extension_before_k_capacity(snapshot):
    from merlin.xdsl_dialects.lowering.k_extension import lower_k_extension

    snapshot["facts"]["fusion_enumeration"]["k_extension_within"] = evidence({"max_k": 10000, "partial_sum_bits": 8})
    bounds = enumerate_candidates(**snapshot)["extension_bounds"]
    assert bounds["coefficient_bound"] == 15 and bounds["T_max"] == 5
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="overflow"):
            lower_k_extension(sign * 16, **bounds["representative"])


def test_missing_extension_bound_is_named_and_aggregated(snapshot):
    del snapshot["facts"]["fusion_enumeration"]["k_extension_within"]
    manifest = enumerate_candidates(**snapshot)
    missing = "fusion_facts#/fields/k_extension_within"
    rows = [
        r
        for r in manifest["rows"]
        if r["state"] == "pending_facts" and any(reason.get("missing_fact") == missing for reason in r["reasons"])
    ]
    assert rows
    assert manifest["summary"]["pending_facts_by_missing_fact"][missing] == len(rows)
    for row in manifest["rows"]:
        if row["state"] == "pending_facts":
            assert any(reason.get("missing_fact") for reason in row["reasons"])


def test_shared_classes_are_unordered_and_cover_readout_relations(snapshot):
    manifest = enumerate_candidates(**snapshot)
    shared = [r for r in manifest["rows"] if r["semantic"]["structure"] == "shared_input"]
    relations = set()
    for row in shared:
        nodes = row["semantic"]["nodes"]
        assert nodes == sorted(nodes, key=canonical_json)
        left, right = [node["readout"] for node in nodes]
        relations.add((left["exit"] == right["exit"], left["stage_indices"] == right["stage_indices"]))
    assert relations == {(True, True), (True, False), (False, False)}
    assert len(shared) < 2000


def test_external_completeness_lists_cannot_change_derivation_or_sample(snapshot, tmp_path, monkeypatch):
    from merlin_experiments.phase0.fusion_validation import record

    from merlin.targetgen.composition_corpus import sample_validated

    external = tmp_path / "external-regimes.json"
    monkeypatch.setenv("MERLIN_FUSION_EXTERNAL_REGIMES", str(external))
    candidate_manifest = enumerate_candidates(**snapshot)
    # A small fixed campaign over enumerated rows, independent of the external file.
    candidates = dict(
        candidate_manifest, rows=[row for row in candidate_manifest["rows"] if row["state"] == "candidate"][:2]
    )
    cases = [
        dict(candidate_id=row["candidate_id"], case_id=f"{index}-{label}", shape_class=label)
        for index, row in enumerate(candidates["rows"])
        for label in ("aligned", "partial")
    ]
    suite = dict(identity={}, case_count=len(cases), rows=cases)
    outcomes = {
        row["case_id"]: dict(
            case_id=row["case_id"], outcome="pass", numeric=True, placement=True, trace=True, evidence_members=[]
        )
        for row in cases
    }
    identity = dict(facts_sha256="facts", contract_sha256="contract")
    admitted = record(candidates, suite, identity, outcomes)
    sample = sample_validated(admitted, suite, software_spec={})
    for regimes in (
        [],
        [{"source": "Dragon", "regime": "invented"}],
        [{"source": "Exo", "regime": "different"}, {"source": "Dragon", "regime": "added"}],
    ):
        external.write_bytes(canonical_json(regimes))
        assert canonical_json(enumerate_candidates(**snapshot)) == canonical_json(candidate_manifest)
        assert record(candidates, suite, identity, outcomes) == admitted
        assert sample_validated(admitted, suite, software_spec={}) == sample


@pytest.mark.parametrize("stratified", [False, True])
@pytest.mark.parametrize("mutation", [None, "rows", "generator", "summary", "normalized_facts"])
def test_snapshot_reproduces_the_whole_candidate_manifest(snapshot, monkeypatch, tmp_path, mutation, stratified):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from merlin_experiments.phase0.fusion_validation import verify_snapshot

    from merlin.targetgen import core_aten_device, software_spec, spec_fact_drift, target_registry

    placement = {"producer_placement": snapshot["facts"]["fusion_enumeration"]["producer_placement"]["value"]}
    snapshot["spec"]["composition_resolution"] = {"chains": [placement]}
    candidates = enumerate_candidates(**snapshot)
    if stratified:
        from merlin.compile.fusion_enumeration import stratify_candidates

        candidates = stratify_candidates(candidates, snapshot["spec"])

    @contextmanager
    def selected(*args, **kwargs):
        yield snapshot["facts"]

    monkeypatch.setattr(core_aten_device, "selected_facts", selected)
    monkeypatch.setattr(target_registry, "load_contract", lambda *a: snapshot["contract"])
    monkeypatch.setattr(software_spec, "load_software_spec", lambda *a, **k: snapshot["spec"])
    monkeypatch.setattr(spec_fact_drift, "fact_capabilities", lambda **k: {})
    monkeypatch.setattr(spec_fact_drift, "resolve_spec", lambda spec, facts: (spec, {}))
    monkeypatch.setattr(spec_fact_drift, "_resolve_chain", lambda *a: placement)
    args = SimpleNamespace(target=snapshot["target"], facts=tmp_path / "facts", spec=tmp_path / "spec", snapshot=None)
    verify_snapshot(args, candidates)
    if stratified:
        import copy

        changed = copy.deepcopy(candidates)
        next(row for row in changed["rows"] if row["state"] == "deferred_by_stratification")["state"] = "candidate"
        with pytest.raises(ValueError, match="deterministic enumeration"):
            verify_snapshot(args, changed)
        snapshot["spec"]["compositions"] = dict(campaign_per_stratum=1)
        with pytest.raises(ValueError, match="deterministic enumeration"):
            verify_snapshot(args, candidates)
        snapshot["spec"].pop("compositions")
    if mutation == "rows":
        candidates["rows"] = candidates["rows"][:1]
        candidates["candidate_count"] = 1
    elif mutation == "generator":
        candidates["identity"]["generator_sha256"] = "0" * 64
    elif mutation == "summary":
        candidates["summary"] = {}
    elif mutation == "normalized_facts":
        candidates["normalized_facts"]["fields"] = {}
    if mutation:
        with pytest.raises(ValueError, match="deterministic enumeration"):
            verify_snapshot(args, candidates)


def test_mean_enumeration_keeps_the_per_case_exhaustive_guard(snapshot):
    snapshot["spec"]["operations"][0]["families"].append("window_mean")
    manifest = enumerate_candidates(**snapshot)
    means = [
        row
        for row in manifest["rows"]
        if any(node["realization"] == "mean" for node in row["semantic"].get("nodes", []))
    ]
    assert means
    assert all("per_case_mean_division_matches_product" in row["semantic"]["guards"] for row in means)


def campaign_strata(row):
    """Each stratum a stratified row belongs to: one per required regime, else its practice key."""
    key = dict(row["stratum_key"])
    regimes = key.pop("required_regimes", None)
    if regimes is None:
        return [canonical_json(key)]
    return [canonical_json(dict(regime=regime, **key)) for regime in regimes]


def assert_first_members_of_some_stratum_execute(manifest, limit):
    groups = {}
    for row in sorted(manifest["rows"], key=lambda row: row["candidate_id"]):
        if row["state"] in ("candidate", "deferred_by_stratification"):
            for key in campaign_strata(row):
                groups.setdefault(key, []).append(row["candidate_id"])
    first = {key: set(ids[:limit]) for key, ids in groups.items()}
    for row in manifest["rows"]:
        if row["state"] in ("candidate", "deferred_by_stratification"):
            executed = any(row["candidate_id"] in first[key] for key in campaign_strata(row))
            assert row["state"] == ("candidate" if executed else "deferred_by_stratification")
    return groups


def test_campaign_stratification_retains_denominator_and_reproduces_scope(snapshot):
    import copy

    from merlin.compile.fusion_enumeration import stratify_candidates, verify_stratification

    enumeration = enumerate_candidates(**snapshot)
    first = stratify_candidates(enumeration, snapshot["spec"])
    shuffled = copy.deepcopy(enumeration)
    shuffled["rows"].reverse()
    second = stratify_candidates(shuffled, snapshot["spec"])
    assert first["candidate_count"] == enumeration["candidate_count"]
    assert first["summary"] == enumeration["summary"]
    assert {row["candidate_id"]: row for row in first["rows"]} == {row["candidate_id"]: row for row in second["rows"]}
    for original, row in zip(enumeration["rows"], first["rows"], strict=True):
        assert row["semantic"] == original["semantic"]
    # Without chains, every declared scheme is a required regime, so rows carry regime strata.
    assert first["campaign"]["required_regimes"] and any(
        "required_regimes" in row.get("stratum_key", {}) for row in first["rows"]
    )
    assert_first_members_of_some_stratum_execute(first, 2)
    validate(first, "fusion_candidates")
    verify_stratification(first)
    changed = copy.deepcopy(first)
    next(row for row in changed["rows"] if row["state"] == "deferred_by_stratification")["state"] = "candidate"
    with pytest.raises(ValueError, match="stratification"):
        verify_stratification(changed)
    # A spec without schemes declares no regime: one plain practice stratum per row, as before.
    one = stratify_candidates(enumeration, dict(compositions=dict(campaign_per_stratum=1)))
    assert one["campaign"]["required_regimes"] == []
    assert not any("required_regimes" in row.get("stratum_key", {}) for row in one["rows"])
    groups = assert_first_members_of_some_stratum_execute(one, 1)
    assert sum(row["state"] == "candidate" for row in one["rows"]) == len(groups)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            stratify_candidates(enumeration, dict(compositions=dict(campaign_per_stratum=value)))


def test_campaign_key_tracks_semantic_pool_axes(snapshot):
    import copy

    from merlin.compile.fusion_enumeration import campaign_stratum

    semantic = next(row["semantic"] for row in enumerate_candidates(**snapshot)["rows"] if row["state"] == "candidate")
    semantic = copy.deepcopy(semantic)
    semantic["nodes"][0]["readout"]["stages"].append(dict(kind="pool", reductions=["max"], padding="zero_tap"))
    baseline = campaign_stratum(semantic)
    for field, value in (("reductions", ["sum"]), ("padding", "negative_infinity")):
        changed = copy.deepcopy(semantic)
        changed["nodes"][0]["readout"]["stages"][-1][field] = value
        assert campaign_stratum(changed) != baseline
    changed = copy.deepcopy(semantic)
    changed["nodes"][0]["readout"]["pool_padding_class"] = "padded"
    assert campaign_stratum(changed) != baseline


@pytest.mark.parametrize("missing", [False, True])
def test_selected_spec_prevents_campaign_self_certification(snapshot, missing):
    from merlin.compile.fusion_enumeration import stratify_candidates, verify_stratification

    spec = dict(compositions=dict(campaign_per_stratum=1))
    manifest = stratify_candidates(enumerate_candidates(**snapshot), spec)
    verify_stratification(manifest, spec)
    if missing:
        manifest.pop("campaign")
    else:
        manifest = stratify_candidates(
            enumerate_candidates(**snapshot), dict(compositions=dict(campaign_per_stratum=2))
        )
    with pytest.raises(ValueError, match="campaign|stratification"):
        verify_stratification(manifest, spec)


@pytest.mark.parametrize("declared", [False, True])
def test_runtime_scale_axis_is_enumerated_and_blocked_until_evidenced(snapshot, declared):
    if declared:
        snapshot["spec"]["quantization"]["schemes"]["axes"]["scale_source"] = ["constant", "runtime_input"]
    manifest = enumerate_candidates(**snapshot)
    runtime = [
        node
        for row in manifest["rows"]
        for node in row["semantic"].get("nodes", [])
        if node["regime"].get("scale_source") == "runtime_input"
    ]
    rows = [
        row
        for row in manifest["rows"]
        if any(node["regime"].get("scale_source") == "runtime_input" for node in row["semantic"].get("nodes", []))
    ]
    assert bool(runtime) == declared
    # A raw exit consumes no scale, so no runtime operand is manufactured for it.
    assert {node["readout"]["exit"] for node in runtime} <= {"requant"}
    for row in rows:
        assert row["state"] != "candidate"
        assert "fusion_facts#/fields/runtime_readout_scale" in {reason.get("missing_fact") for reason in row["reasons"]}


def _single_rows(manifest, form, realization):
    return [
        row
        for row in manifest["rows"]
        if row["semantic"]["structure"] == "single"
        and (row["semantic"]["nodes"][0]["form"], row["semantic"]["nodes"][0]["realization"]) == (form, realization)
    ]


def test_dataflow_variants_follow_mesh_product_semantics(snapshot):
    from merlin.compile.fusion_enumeration import campaign_stratum, stratify_candidates

    snapshot["spec"]["operations"][0]["families"] += ["operand_sum", "movement", "window_mean"]
    manifest = enumerate_candidates(**snapshot)
    validate(manifest, "fusion_candidates")
    expected = {
        ("operand_sum", "scaled_loads"): ["not_applicable"],
        ("window_movement", "pool"): ["not_applicable"],
        ("window_movement", "mean"): ["OS", "WS"],
        ("contraction", "matmul"): ["OS", "WS"],
        ("contraction", "conv_as_matmul"): ["OS", "WS"],
    }
    for (form, realization), flows in expected.items():
        rows = _single_rows(manifest, form, realization)
        assert rows
        variants = {}
        for row in rows:
            core = canonical_json({k: v for k, v in row["semantic"].items() if k not in {"dataflow", "guards"}})
            variants.setdefault(core, []).append(row["semantic"]["dataflow"])
        assert all(sorted(found) == flows for found in variants.values())
    # A graph without a mesh product carries no dataflow guard or dataflow requirement.
    for row in _single_rows(manifest, "operand_sum", "scaled_loads"):
        assert not {"constant_weights", "all_prefix_sums_fit_mesh_signed_bits"} & set(row["semantic"]["guards"])
        assert not any("dataflow" in reason["evidence"] for reason in row["reasons"])
    assert {
        row["semantic"]["dataflow"] for row in manifest["rows"] if row["semantic"]["structure"] == "shared_input"
    } == {
        "OS",
        "WS",
    }
    stratified = stratify_candidates(manifest, snapshot["spec"])
    keys = [
        row["stratum_key"]
        for row in stratified["rows"]
        if "stratum_key" in row and row["semantic"]["nodes"][0]["form"] == "operand_sum"
    ]
    assert keys and {key["dataflow"] for key in keys} == {"not_applicable"}
    semantic = _single_rows(manifest, "operand_sum", "scaled_loads")[0]["semantic"]
    assert campaign_stratum(semantic)["dataflow"] == "not_applicable"


def test_unknown_producer_form_has_no_assumed_dataflow():
    from merlin.compile.fusion_enumeration import dataflow_variants

    assert dataflow_variants([dict(form="operand_sum", realization="scaled_loads")]) == ("not_applicable",)
    assert dataflow_variants([dict(form="contraction", realization="matmul")]) == ("WS", "OS")
    with pytest.raises(ValueError, match="UNKNOWN"):
        dataflow_variants([dict(form="contraction", realization="matmul"), dict(form="gather", realization="dma")])
