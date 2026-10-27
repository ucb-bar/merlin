"""Fixed fusion inputs, exclusions, complete admission and byte bindings."""

import copy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import fusion_cases as C
from merlin_experiments.phase0 import fusion_validation as V
from merlin_experiments.phase0.fusion_execution import prove_placement, source_members
from merlin_experiments.phase0.fusion_reference import boundary


def test_release_pool_constraints_remain_unique():
    from merlin_experiments.phase0.composition_generation import _release_chain

    from merlin.targetgen.software_spec import validate_compositions

    node = dict(
        form="contraction", seed=False, residual=None, realization="matmul", scheme="q",
        readout=dict(exit="requant", stages=[dict(kind="pool")]),
    )
    chain = _release_chain(node, "WS", dict(producer="contraction", readout_stages=["pool"]))
    assert len(chain["constraints"]) == 4
    validate_compositions(dict(
        schema="merlin.software_spec.compositions.v1", status="reviewed",
        grading="placement_and_instruction_audit", chains=[chain],
    ))


@pytest.mark.parametrize("strategy", ["f32_product_rne", "exact_rational_host"])
@pytest.mark.parametrize("activation", [False, True])
def test_residual_references_follow_each_source_f32_operation(strategy, activation):
    import numpy as np
    from merlin_experiments.phase0.fusion_reference import exact_reference, unfused_reference

    gain = float(np.float32(1 / 6))
    case = dict(
        producer="contraction",
        exit="requant",
        coefficient=1,
        multipliers=[gain],
        accumulator_stages=[],
        readout_stages=["activation"] if activation else [],
    )
    scheme = dict(readout_strategy=strategy, weight_scale_granularity="tensor")
    env = dict(A=np.array([[-7, -4], [7, 4]]), W0=np.array([[117], [1]]), Residual=np.array([[64], [-64]]))
    cap = dict(
        operation=dict(attributes=dict(M=2, K=2, N=1, outputs=["Y"], case=case, quantization_scheme=scheme)),
        inputs=[dict(name=name, dtype="i8", shape=list(value.shape)) for name, value in env.items()],
    )
    wanted = {"Y": [[0 if activation else -127], [127]]}
    assert exact_reference(cap, env)[0] == wanted
    assert unfused_reference(cap, env) == wanted
    assert boundary(-823 + 64, gain, "f32_product_rne", False) == -126


@pytest.mark.parametrize(
    "dataflow, seed_source", [("WS", None), ("OS", None), ("WS", "constant"), ("WS", "runtime_argument")]
)
@pytest.mark.parametrize("activation", [False, True])
def test_residual_counterexample_is_a_required_frozen_stimulus(profile, tmp_path, dataflow, seed_source, activation):
    import numpy as np

    candidate, schemes, _, _ = profile
    node = candidate["semantic"]["nodes"][0]
    node["residual"] = dict(coefficient_bound=127)
    candidate["semantic"]["dataflow"] = dataflow
    if seed_source:
        node.update(seed=True, seed_source=seed_source)
    if activation:
        node["readout"]["stages"].append(dict(kind="activation"))
    schemes[node["scheme"]].update(scale_value="arbitrary_f32", readout_strategy="f32_product_rne")
    assert ("residual_rounding", 1) in C.parameters(candidate)
    frozen = payload(profile, tmp_path, "residual_rounding")
    values = frozen["input_values"]
    case = frozen["nodes"][0]["operation"]["attributes"]["case"]
    weights = case.get("constant_weights", values.get("node0_W0"))
    accumulator = np.array(values["node0_A"]) @ np.array(weights)
    assert accumulator[0, 0] == -823 and values["node0_Residual"][0][0] == 64
    assert np.min(values["node0_A"]) >= -128 and np.max(values["node0_A"]) <= 127
    assert case["coefficient"] == 1 and case["multipliers"] == [float(np.float32(1 / 6))]
    assert frozen["references"][0]["values"][0][0] == (0 if activation else -127)
    assert frozen["references"][0]["values"][1][0] == 127
    assert frozen["witnesses"][0]["residual_rounding_difference"]
    assert dict(node=0, event="residual_rounding_difference") in frozen["required_witnesses"]


def test_shared_residual_counterexample_preserves_the_witness_input(profile, tmp_path):
    candidate, schemes, _, _ = profile
    node = candidate["semantic"]["nodes"][0]
    # Update the scheme first so each sibling's effective regime agrees with it.
    schemes[node["scheme"]].update(scale_value="arbitrary_f32", readout_strategy="f32_product_rne")
    residual = copy.deepcopy(node)
    residual["residual"] = dict(coefficient_bound=127)
    candidate["semantic"].update(structure="shared_input", nodes=[node, residual])
    frozen = payload(profile, tmp_path, "residual_rounding")
    assert frozen["references"][1]["values"][0][0] == -127
    assert frozen["witnesses"][1]["residual_rounding_difference"]


@pytest.mark.parametrize("k", [4, 14, 26, 28, 30, 52, 56, 60, 82, 94, 102])
def test_mean_references_divide_over_the_entire_byte_window_sum_domain(k):
    import numpy as np
    from merlin_experiments.phase0.fusion_reference import exact_reference, unfused_reference

    totals = np.arange(-128 * k, 127 * k + 1, dtype=np.int64)
    quotient, remainder = np.divmod(totals, k)
    image = np.repeat(quotient[:, None, None], k, axis=2)
    image[:, 0, :] += np.arange(k)[None, :] < remainder[:, None]
    case = dict(
        producer="contraction",
        exit="requant",
        window_mean=True,
        multipliers=[1 / k],
        constant_weights=[[1] for _ in range(k)],
        accumulator_stages=[],
        readout_stages=[],
    )
    scheme = dict(readout_strategy="f32_product_rne", weight_scale_granularity="tensor")
    cap = dict(
        operation=dict(attributes=dict(M=len(totals), K=k, N=1, outputs=["Y"], case=case, quantization_scheme=scheme)),
        inputs=[dict(name="Image", dtype="i8", shape=list(image.shape))],
    )
    wanted = {"Y": np.rint(totals.astype(np.float32) / np.float32(k)).astype(np.int64)[:, None].tolist()}
    multiplied = np.rint(totals.astype(np.float32) * np.float32(1 / k)).astype(np.int64)
    if np.array_equal(multiplied, np.array(wanted["Y"])[:, 0]):
        assert exact_reference(cap, dict(Image=image))[0] == wanted
        assert unfused_reference(cap, dict(Image=image)) == wanted
        assert "arith.divf" in C._mean_source(cap)
        return
    # Enumeration does not plan such windows; the oracle and source still divide,
    # and the unfused reference, which multiplies by the reciprocal, refuses.
    if k == 14:
        assert np.count_nonzero(multiplied != np.array(wanted["Y"])[:, 0]) == 78
    assert exact_reference(cap, dict(Image=image))[0] == wanted
    assert "arith.divf" in C._mean_source(cap)
    with pytest.raises(ValueError, match="disagree after RNE"):
        unfused_reference(cap, dict(Image=image))


@pytest.fixture
def profile(tmp_path):
    path = dict(
        schema="output_path_v1",
        accumulator=dict(dtype="i32"),
        accumulator_stages=[],
        readout=[
            dict(kind="convert", to="f32", rounding="half_even"),
            dict(kind="scale", dtype="f32", rounding="half_even", granularity="per_command", columns_per_scale=4),
            dict(kind="requant", to="i8", rounding="half_even", clamp=[-128, 127]),
        ],
        exits=dict(accumulator=True, requant=True),
    )
    scheme = dict(
        id="test_scale",
        status="resolved",
        graded=True,
        unknown=[],
        scale_value="power_of_two",
        operand_sum_rounding="single_rounding",
        readout_strategy="exact_rational_host",
        weight_scale_granularity="tensor",
        columns_per_scale=4,
    )
    node = dict(
        form="contraction",
        realization="matmul",
        scheme=scheme["id"],
        regime=scheme,
        seed=False,
        residual=None,
        readout=dict(exit="requant", stages=path["readout"]),
    )
    candidate = dict(
        candidate_id="candidate",
        state="candidate",
        reasons=[],
        semantic=dict(structure="single", dataflow="WS", nodes=[node]),
    )
    facts = dict(
        tile_dim=4,
        columns_per_scale=4,
        dataflows=["WS"],
        operand_signed_bits=8,
        mesh_signed_bits=20,
        output_path=path,
        k_extension_within=dict(max_k=1024, partial_sum_bits=32),
    )
    context = SimpleNamespace(source_root=tmp_path, held_out_inputs=[], workload_spec={})
    return candidate, {scheme["id"]: scheme}, facts, context


def payload(profile, tmp_path, label="aligned", excluded=frozenset()):
    candidate, schemes, facts, context = profile
    return C.build_payload(candidate, schemes, facts, label, None, excluded, context, tmp_path / "screen")


def test_case_and_reference_determinism(profile, tmp_path):
    first = payload(profile, tmp_path)
    second = payload(copy.deepcopy(profile), tmp_path)
    assert C.canonical(first) == C.canonical(second)
    assert first["source"] and first["references"]
    assert payload(profile, tmp_path, "partial")["inputs"][0]["shape"] == [5, 5]
    assert C.digest(first) != C.digest(payload(profile, tmp_path, "partial"))


def test_every_shape_passes_private_geometry_and_publication(profile, tmp_path):
    blocked = frozenset({(4, 4, 4)})
    result = payload(profile, tmp_path, excluded=blocked)
    assert result["inputs"][0]["shape"] == [8, 8]
    with pytest.raises(ValueError, match="pending_shape_exclusion"):
        payload(profile, tmp_path, excluded=frozenset((v, v, v) for v in (4, 8, 12, 16, 20)))
    hidden = tmp_path / "private" / "source"
    hidden.mkdir(parents=True)
    (hidden / "capsule.yaml").write_text(
        yaml.safe_dump(
            dict(
                operation=dict(op="matmul"),
                inputs=[dict(name="a", role="input", shape=[4, 4]), dict(name="w", role="weight", shape=[4, 4])],
            )
        )
    )
    profile[3].held_out_inputs = ["private"]
    assert payload(profile, tmp_path)["inputs"][0]["shape"] == [8, 8]


def test_each_sibling_and_physical_extension_is_screened(profile, tmp_path):
    candidate, _, _, _ = profile
    candidate["semantic"]["nodes"][0]["residual"] = dict(coefficient_bound=127)
    result = payload(profile, tmp_path, excluded=frozenset({(4, 8, 4)}))
    assert result["inputs"][0]["shape"] == [8, 8]
    candidate["semantic"].update(structure="shared_input", nodes=candidate["semantic"]["nodes"] * 2)
    result = payload(profile, tmp_path, excluded=frozenset({(4, 8, 4)}))
    assert len(result["producer_views"]) == 4
    assert [row["name"] for row in result["inputs"]].count("A") == 1


@pytest.mark.parametrize("failed", ["host_fallback", "split_kernels", "numeric_mismatch", "pending_execution"])
def test_one_failure_refuses_the_candidate(profile, failed):
    candidate = profile[0]
    passing = dict(outcome="pass", numeric=True, placement=True, trace=True)
    assert V.candidate_decision(candidate, ["one", "two"], dict(one=passing, two=passing))["admitted"]
    failing = {**passing, "outcome": "not_admitted", "reason": failed}
    failing["numeric" if failed == "numeric_mismatch" else "placement"] = False
    observations = dict(one=passing, two=failing)
    assert not V.candidate_decision(candidate, ["one", "two"], observations)["admitted"]
    assert not V.candidate_decision(candidate, ["one", "missing"], observations)["admitted"]
    assert not V.candidate_decision(candidate, [], {})["admitted"]


def test_smoke_covers_seeded_and_scaled_loads_at_both_exits(profile):
    rows = []
    for form, seed in (("contraction", False), ("contraction", True), ("operand_sum", False)):
        for exit_name in ("accumulator", "requant"):
            row = copy.deepcopy(profile[0])
            row["candidate_id"] = f"{form}_{seed}_{exit_name}"
            node = row["semantic"]["nodes"][0]
            node.update(form=form, realization="scaled_loads" if form == "operand_sum" else "matmul", seed=seed)
            node["readout"]["exit"] = exit_name
            rows.append(row)
    assert set(V.smoke_ids(dict(rows=rows))) == {row["candidate_id"] for row in rows}


def test_partial_producer_admission_and_split_kernels_earn_nothing(profile, tmp_path):
    source = payload(profile, tmp_path)["source"]
    _, operations, stages, ordinals = source_members(source)
    members = [ordinals[operations[i]] for values in stages.values() for i in values]
    proof = dict(bindings=[dict(symbol="kernel", replaced_ordinals=members[:1])])
    assert "host_fallback" in prove_placement(source, proof)[1]
    proof["bindings"].append(dict(symbol="other", replaced_ordinals=members[1:]))
    assert "split_kernels" in prove_placement(source, proof)[1]
    proof["bindings"] = [dict(symbol="kernel", replaced_ordinals=members)]
    assert prove_placement(source, proof)[0]["symbol"] == "kernel"


def test_smoke_keeps_constant_seeds_and_pooling_separate(profile):
    rows = []
    for realization in ("matmul", "conv_as_matmul"):
        row = copy.deepcopy(profile[0])
        row["candidate_id"] = realization
        node = row["semantic"]["nodes"][0]
        node.update(seed=True, seed_source="constant", realization=realization)
        node["readout"]["stages"].insert(0, dict(kind="activation"))
        rows.append(row)
    for structure, padding in (
        ("single", "unpadded"),
        ("single", "padded"),
        ("shared_input", None),
        ("shared_input", "unpadded"),
    ):
        row = copy.deepcopy(profile[0])
        row["candidate_id"] = f"{structure}_{padding}"
        row["semantic"]["structure"] = structure
        if padding:
            readout = row["semantic"]["nodes"][0]["readout"]
            readout.update(pool_padding="zero_or_padded", pool_padding_class=padding)
            readout["stages"].append(dict(kind="pool"))
        rows.append(row)
    assert set(V.smoke_ids(dict(rows=rows))) == {row["candidate_id"] for row in rows}


def test_reference_ties_saturation_conversion_and_widening():
    assert [boundary(v, 0.5, "f32_product_rne", False) for v in (-5, -3, -1, 1, 3, 5)] == [-2, -2, 0, 0, 2, 2]
    assert boundary(2**24 + 1, 2**-24, "f32_product_rne", False) == 1
    assert boundary(-10000, 0.5, "f32_product_rne", False) == -128
    assert boundary(10000, 0.5, "f32_product_rne", False) == 127
    assert boundary(127, 1.0, "f32_product_rne", False) + boundary(127, 1.0, "f32_product_rne", False) == 254


def test_record_binds_denominator_cases_identity_and_evidence(profile, tmp_path):
    from merlin.compile.fusion_enumeration import stratify_candidates

    candidate = profile[0]
    candidates = stratify_candidates(
        dict(target="synthetic", profile={}, rows=[candidate], enumeration_complete=True), {}
    )
    suite = dict(identity={}, case_count=1, rows=[dict(case_id="case", candidate_id="candidate")])
    identity = dict(facts_sha256="facts", contract_sha256="contract")
    ledger = V.record(candidates, suite, identity, {})
    path = tmp_path / "validation.json"
    path.write_bytes(C.canonical(ledger))
    assert not V.read_validation(path, candidates, suite, identity, tmp_path)["admitted_ids"]
    with pytest.raises(ValueError, match="binding/decision"):
        V.read_validation(path, candidates, suite, {**identity, "facts_sha256": "other"}, tmp_path)
    changed = copy.deepcopy(suite)
    changed["rows"][0]["case_id"] = "other"
    with pytest.raises(ValueError, match="binding/decision"):
        V.read_validation(path, candidates, changed, identity, tmp_path)
    ledger["rows"][0]["admitted"] = True
    path.write_bytes(C.canonical(ledger))
    with pytest.raises(ValueError, match="seal"):
        V.read_validation(path, candidates, suite, identity, tmp_path)
    member = tmp_path / "trace"
    member.write_bytes(b"original")
    pins = [dict(path="trace", bytes=member.stat().st_size, sha256=C.file_digest(member))]
    V.verify_members(pins, tmp_path)
    member.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="evidence"):
        V.verify_members(pins, tmp_path)


def test_frozen_writes_refuse_changed_bytes(tmp_path):
    path = tmp_path / "frozen"
    C.write_frozen(path, b"same")
    C.write_frozen(path, b"same")
    with pytest.raises(ValueError, match="frozen member changed"):
        C.write_frozen(path, b"changed")


@pytest.mark.parametrize(
    "scope,error",
    [
        (["selected"], None),
        (None, "omits a candidate"),
        (["selected", "other"], "deterministic execution scope"),
        (["unknown"], "scope is invalid"),
        (["selected", "selected"], "scope is invalid"),
        ([], "scope is invalid"),
        ("selected", "scope is invalid"),
        ([{}], "scope is invalid"),
    ],
)
def test_read_suite_checks_the_sealed_candidate_scope(profile, tmp_path, monkeypatch, scope, error):
    candidate, schemes, facts, context = profile
    candidate["candidate_id"] = "selected"
    other = copy.deepcopy(candidate)
    other["candidate_id"] = "other"
    candidates = tmp_path / "candidates.json"
    from merlin.compile.fusion_enumeration import stratify_candidates

    candidates.write_bytes(C.canonical(stratify_candidates(dict(rows=[candidate, other]), {})))
    monkeypatch.setattr(V, "smoke_ids", lambda *args: ["selected"])
    rows = C._generate_candidate(candidate, schemes, facts, frozenset(), context, tmp_path)
    monkeypatch.setattr(V, "_case_inputs", lambda *args: (schemes, facts, frozenset(), context))
    suite = dict(
        schema="merlin.fusion_cases.v1",
        target="synthetic",
        identity=dict(
            candidate_manifest_sha256=C.file_digest(candidates),
            reference_implementation_sha256=C.reference_identity(),
            private_exclusion_sha256=C.digest([]),
        ),
        profile={} if scope is None else dict(selected_candidate_ids=scope, execution_scope="smoke"),
        candidate_count=2,
        candidate_case_count=1,
        case_count=len(rows),
        rows=rows,
    )
    suite["seal"] = dict(canonicalization="merlin.canonical_json.v1", payload_sha256=C.digest(suite))
    path = tmp_path / "suite.json"
    path.write_bytes(C.canonical(suite))
    if error:
        with pytest.raises(ValueError, match=error):
            V.read_suite(path, candidates)
    else:
        assert V.read_suite(path, candidates) == suite
        (tmp_path / rows[0]["member"]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="bytes changed"):
            V.read_suite(path, candidates)
        candidates.write_bytes(candidates.read_bytes() + b" ")
        with pytest.raises(ValueError, match="binding changed"):
            V.read_suite(path, candidates)


@pytest.mark.parametrize(
    "smoke,scope,available,executed",
    [
        (True, ["base", "extra"], ["base", "extra"], 2),
        (True, ["extra"], ["extra"], None),
        (False, ["base", "extra"], ["base", "extra"], 2),
        (False, None, ["base", "extra"], None),
        (False, None, ["base", "extra", "full_only"], 3),
    ],
)
def test_campaign_executes_the_sealed_suite_scope_and_rejects_missing_cases(
    tmp_path, monkeypatch, smoke, scope, available, executed
):
    import json
    from concurrent.futures import ThreadPoolExecutor

    candidates = dict(
        target="synthetic",
        profile={},
        normalized_facts={},
        enumeration_complete=True,
        rows=[dict(candidate_id=name, state="candidate", semantic={}) for name in ("base", "extra", "full_only")],
    )
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_bytes(C.canonical(candidates))
    suite = dict(
        identity={},
        profile={} if scope is None else dict(selected_candidate_ids=scope),
        case_count=len(available),
        rows=[dict(case_id=name, candidate_id=name) for name in available],
    )
    identity = dict(facts_sha256="facts", contract_sha256="contract", simulator_extension_sha256="extension")
    monkeypatch.setattr(V, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(V, "read_suite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(V, "execution_identity", lambda *args: identity)
    monkeypatch.setattr(V, "smoke_ids", lambda *args: ["base"])
    monkeypatch.setattr(V, "worker_pool", lambda workers, **kwargs: ThreadPoolExecutor(workers, **kwargs))
    monkeypatch.setattr(V, "read_validation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        V,
        "_worker",
        lambda case, *args: (
            dict(case_id=case["case_id"], outcome="not_admitted", reason="test_refusal", evidence_members=[]),
            1.0,
        ),
    )
    (tmp_path / "spec.yaml").write_text("{}")
    args = SimpleNamespace(
        candidates=candidate_path,
        cases=tmp_path / "cases.json",
        output=tmp_path / "smoke",
        target="synthetic",
        facts=tmp_path / "facts.json",
        package=tmp_path,
        provider=tmp_path,
        spec=tmp_path / "spec.yaml",
        extension=tmp_path,
        timeout=1,
        workers=1,
        smoke=smoke,
    )
    if executed is None:
        with pytest.raises(ValueError, match="omits selected execution candidates"):
            V.campaign(args)
    else:
        V.campaign(args)
        summary = json.loads((args.output / "smoke-summary.json").read_bytes())
        assert summary["candidates"] == summary["cases"] == executed


def test_record_is_independent_of_worker_completion_order(profile):
    candidate = profile[0]
    candidates = dict(target="synthetic", profile={}, rows=[candidate], enumeration_complete=True)
    suite = dict(identity={}, case_count=2, rows=[dict(case_id=name, candidate_id="candidate") for name in ("a", "b")])
    identity = dict(facts_sha256="facts", contract_sha256="contract")
    failed = dict(
        outcome="not_admitted", reason="host_fallback", numeric=True, placement=False, trace=False, evidence_members=[]
    )
    results = {name: {**failed, "case_id": name} for name in ("a", "b")}
    assert C.canonical(V.record(candidates, suite, identity, results)) == C.canonical(
        V.record(candidates, suite, identity, dict(reversed(list(results.items()))))
    )


def test_resume_reopens_inputs_selection_and_evidence(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import fusion_execution as E

    suite = tmp_path / "suite"
    suite.mkdir()
    C.write_frozen(suite / "member.json", C.canonical({}))
    case = dict(
        case_id="case", candidate_id="candidate", member="member.json", sha256=C.file_digest(suite / "member.json")
    )
    root, calls = tmp_path / "campaign", []

    def execute(payload, candidate, selection, directory):
        calls.append(directory)
        (directory / "proof.txt").write_text("retained observation")
        return dict(outcome="not_admitted", reason="host_fallback", numeric=True, placement=False, trace=False)

    monkeypatch.setattr(E, "execute_case", execute)
    first = V._worker(case, {}, {}, root, suite, dict(selected="identity"))
    assert V._worker(case, {}, {}, root, suite, dict(selected="identity")) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="selection changed"):
        V._worker(case, {}, {}, root, suite, dict(selected="other"))
    (calls[0] / "proof.txt").write_text("changed observation")
    with pytest.raises(ValueError, match="evidence"):
        V._worker(case, {}, {}, root, suite, dict(selected="identity"))


def test_bundle_preserves_byte_storage_and_positional_abi(profile, tmp_path):
    import json

    import numpy as np
    from merlin_experiments.phase0.fusion_execution import _write_bundle

    frozen = payload(profile, tmp_path)
    directory = tmp_path / "bundle"
    directory.mkdir()
    _write_bundle(frozen, directory)
    with np.load(directory / "inputs.npz") as arrays:
        assert arrays["in0"].dtype == np.int8
        assert arrays["in0"].tolist() == frozen["input_values"][frozen["inputs"][0]["name"]]
    assert json.loads((directory / "input_order.json").read_bytes()) == {"arg0": 0, "arg1": 1}


def test_source_seed_is_a_guarded_initializer(profile, tmp_path):
    from merlin.common import mlir_query as mq

    profile[0]["semantic"]["nodes"][0]["seed"] = True
    frozen = payload(profile, tmp_path, "conversion_boundary")
    module, operations, stages, _ = source_members(frozen["source"])
    mac = next(op for op in operations if "mac" in getattr(op.attributes.get("prov.trace_role"), "data", ""))
    guard = mac.operands[2].owner
    assert any(mq.op_name(op) == "cf.assert" for op in guard.walk())
    assert guard not in [operations[i] for members in stages.values() for i in members]
    assert frozen["witnesses"][0]["conversion_boundary"]
    module.verify()


def test_os_seed_stimuli_respect_the_source_guard(profile, tmp_path):
    profile[0]["semantic"]["dataflow"] = "OS"
    profile[0]["semantic"]["nodes"][0]["seed"] = True
    frozen = payload(profile, tmp_path, "conversion_boundary")
    assert {v for row in frozen["input_values"]["node0_Bias"] for v in row} == {0}
    assert not frozen["required_witnesses"]


@pytest.mark.parametrize("label", ["aligned", "partial", "rne_saturation", "conversion_boundary"])
@pytest.mark.parametrize("residual", [False, True])
def test_constant_seed_is_an_initializer_without_a_runtime_argument(profile, tmp_path, label, residual):
    from merlin.compile.fusion_enumeration import seed_source

    node = profile[0]["semantic"]["nodes"][0]
    node.update(seed=True, seed_source="constant")
    if residual:
        node["residual"] = dict(coefficient_bound=127)
    frozen = payload(profile, tmp_path, label)
    module, operations, _, _ = source_members(frozen["source"])
    assert all(spec["name"] != "node0_Bias" for spec in frozen["inputs"])
    assert "node0_Bias" not in frozen["input_values"]
    if not residual:
        mac = next(op for op in operations if "mac" in getattr(op.attributes.get("prov.trace_role"), "data", ""))
        assert seed_source(mac.operands[2]) == "constant"
    assert frozen["witnesses"][0]["conversion_boundary"] == (label == "conversion_boundary")
    module.verify()


def test_shared_source_reaches_the_ordinary_group_recognizer(profile, tmp_path):
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups

    candidate = profile[0]
    candidate["semantic"].update(structure="shared_input", nodes=candidate["semantic"]["nodes"] * 2)
    module = source_members(payload(profile, tmp_path)["source"])[0]
    groups = find_shared_input_groups(module)
    assert len(groups) == 1 and len(groups[0].contraction_ordinals) == 2


@pytest.mark.parametrize("mutation", [None, "elf_bytes", "writable", "ambiguous", "unreferenced", "source_bytes"])
def test_constant_seed_audit_requires_the_source_bound_immutable_elf_object(tmp_path, monkeypatch, mutation):
    from merlin_experiments.phase0.fusion_execution import _constant_seed_input
    from xdsl.dialects import builtin, llvm
    from xdsl.ir import Block, Region

    from merlin.targetgen import elf_lanes, native_instruction_audit

    data = bytes([1, 0, 0, 0, 2, 0, 0, 0])
    names = ["arbitrary_literal", "other_literal"] if mutation == "ambiguous" else ["arbitrary_literal"]
    globals = [
        llvm.GlobalOp(
            llvm.LLVMArrayType(8, builtin.i8),
            name,
            "internal",
            constant=True,
            value=builtin.DenseIntOrFPElementsAttr(builtin.TensorType(builtin.i8, [8]), builtin.BytesAttr(data)),
        )
        for name in names
    ]
    addresses = [] if mutation == "unreferenced" else [llvm.AddressOfOp(name, llvm.LLVMPointerType()) for name in names]
    function = llvm.FuncOp("kernel", llvm.LLVMFunctionType([]), body=Region(Block([*addresses, llvm.ReturnOp()])))
    directory = tmp_path / "spike-build/region_offload"
    directory.mkdir(parents=True)
    (directory / "core_aten.llvm.mlir").write_text(str(builtin.ModuleOp([*globals, function])))
    elf = tmp_path / "model.elf"
    elf.write_bytes((b"wrong!!!" if mutation == "elf_bytes" else data) * len(names))
    monkeypatch.setattr(elf_lanes, "_endian", lambda _: "<")
    monkeypatch.setattr(
        elf_lanes,
        "_section_headers",
        lambda _: [(b"section", 1, 3 if mutation == "writable" else 2, 0, len(names) * 8, 4096)],
    )
    monkeypatch.setattr(
        native_instruction_audit,
        "elf_object_buffers",
        lambda _: [dict(symbol=name, base=4096 + index * 8, byte_extent=8) for index, name in enumerate(names)],
    )
    case = dict(constant_seed=[[99, 2]] if mutation == "source_bytes" else [[1, 2]])
    if mutation is None:
        assert _constant_seed_input(case, tmp_path, "kernel", elf) == dict(kind="global", symbol=names[0], offset=0)
    else:
        with pytest.raises(ValueError, match="unique immutable linked object"):
            _constant_seed_input(case, tmp_path, "kernel", elf)


def test_shared_pool_and_accumulator_use_the_same_input_geometry(profile, tmp_path):
    candidate = profile[0]
    plain = copy.deepcopy(candidate["semantic"]["nodes"][0])
    plain["readout"] = dict(exit="accumulator", stages=[])
    pooled = copy.deepcopy(candidate["semantic"]["nodes"][0])
    pooled["readout"]["stages"].append(dict(kind="pool", padding="zero_tap", reductions=["max"]))
    pooled["readout"]["pool_padding"] = "unpadded_only"
    candidate["semantic"].update(structure="shared_input", nodes=[plain, pooled])
    frozen = payload(profile, tmp_path)
    source_members(frozen["source"])[0].verify()
    assert next(spec for spec in frozen["inputs"] if spec["name"] == "A")["shape"] == [25, 4]
    assert [ref["shape"] for ref in frozen["references"]] == [[25, 4], [1, 2, 2, 4]]


def test_shared_readout_audit_preserves_each_siblings_obligations(profile, tmp_path):
    from merlin_experiments.phase0.fusion_execution import _trace_modes

    candidate, _, facts, _ = profile
    left, right = copy.deepcopy(candidate["semantic"]["nodes"]) * 2
    right = copy.deepcopy(right)
    right["readout"]["stages"] = [s for s in right["readout"]["stages"] if s["kind"] != "scale"]
    candidate["semantic"].update(structure="shared_input", nodes=[left, right])
    frozen = payload(profile, tmp_path)
    assert _trace_modes(frozen, candidate, facts["output_path"])[2] == dict(shared_input=True, readout_multipliers=True)
    right["readout"]["stages"].append(dict(kind="activation"))
    facts["output_path"]["readout"].append(dict(kind="activation", function="relu"))
    with pytest.raises(ValueError, match="siblings require output-scoped trace contexts"):
        _trace_modes(payload(profile, tmp_path), candidate, facts["output_path"])


@pytest.mark.parametrize("label", ["aligned", "rne_saturation"])
def test_mean_retains_its_source_window_and_reduction_identity(profile, tmp_path, label):
    from merlin.common import mlir_query as mq

    node = profile[0]["semantic"]["nodes"][0]
    node.update(form="window_movement", realization="mean")
    profile[1][node["scheme"]]["readout_strategy"] = "f32_product_rne"
    frozen = payload(profile, tmp_path, label)
    module, operations, stages, _ = source_members(frozen["source"])
    reduction = next(op for op in operations if op.name == "linalg.reduce")
    assert operations.index(reduction) in stages["node0_mean_mac"]
    assert any(mq.op_name(op) == "quant_ext.dequantize_per_tensor" for op in module.walk())
    assert frozen["inputs"][0]["name"] == "node0_Image"
    assert len(frozen["inputs"][0]["shape"]) == 3
    if label == "rne_saturation":
        assert frozen["witnesses"][0]["rne_tie"]


def test_campaign_selects_the_authored_readout_definition(profile):
    from merlin_experiments.phase0.fusion_execution import _selected_policy, readout_options

    from merlin.common.paths import repo_root
    from merlin.targetgen.software_spec import load_software_spec

    spec = load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml", target="gemmini")
    candidate = profile[0]
    node = candidate["semantic"]["nodes"][0]
    node.update(
        scheme="q_f32_tensor", regime=dict(readout_strategy="f32_product_rne", operand_sum_rounding="per_operand")
    )
    selected = _selected_policy(spec, candidate)
    definition = next(
        row["semantics"]["readout_definitions"]["f32_product_rne"]
        for row in spec["transfer_contracts"]["declarations"]
        if "f32_product_rne" in row.get("semantics", {}).get("readout_definitions", {})
    )
    assert selected["numerical_semantics"]["readout"] == definition
    options = readout_options(dict(candidates=dict(normalized_facts=dict(output_path={}))), selected, {}, candidate)
    assert options["scheme"]["operand_sum_rounding"] == "per_operand"
    node["scheme"] = "undeclared"
    with pytest.raises(ValueError, match="not declared"):
        _selected_policy(spec, candidate)


def test_packed_output_audit_requires_a_complete_partition():
    from merlin_experiments.phase0.fusion_execution import _bind_result_slices

    args = [dict(role="output", dtype="i32", shape=[8])]
    decoded = dict(instructions=[dict(runtime_buffer_bindings=[dict(arg_index=0, base=100, byte_extent=32)])])
    binding = dict(result_slices=[dict(offset=0, shape=[2, 2]), dict(offset=4, shape=[2, 2])])
    _bind_result_slices(decoded, binding, args)
    assert decoded["instructions"][0]["runtime_buffer_bindings"] == [
        dict(arg_index=1, base=100, byte_extent=16),
        dict(arg_index=2, base=116, byte_extent=16),
    ]
    binding["result_slices"][1]["offset"] = 3
    with pytest.raises(ValueError, match="partition"):
        _bind_result_slices(dict(instructions=[]), binding, args)


def test_residual_source_uses_a_direct_quantized_grid(profile, tmp_path):
    profile[0]["semantic"]["nodes"][0]["residual"] = dict(coefficient_bound=127)
    result = payload(profile, tmp_path)
    assert "quant_ext.dequantize_per_tensor" in result["source"]
    assert "arith.addf" in result["source"]
    assert "arith.muli %r" not in result["source"]


@pytest.mark.parametrize("form", ["contraction", "movement", "operand_sum", "mean"])
@pytest.mark.parametrize("padded", [False, True])
def test_pool_sources_use_nhwc_windows_and_preserve_references(profile, tmp_path, form, padded):
    node = profile[0]["semantic"]["nodes"][0]
    stage = dict(kind="pool", padding="zero_tap", reductions=["max"])
    node["readout"]["stages"].append(stage)
    node["readout"]["pool_padding"] = "zero_or_padded" if padded else "unpadded_only"
    node["readout"]["pool_padding_class"] = "padded" if padded else "unpadded"
    if padded:
        node["readout"]["stages"].insert(0, dict(kind="activation", function="relu", domain="accumulator"))
    if form == "movement":
        node.update(form="window_movement", realization="pool")
    elif form == "operand_sum":
        node.update(form="operand_sum", realization="scaled_loads")
    elif form == "mean":
        node.update(form="window_movement", realization="mean")
        profile[1][node["scheme"]]["readout_strategy"] = "f32_product_rne"
    result = payload(profile, tmp_path, "conversion_boundary" if padded else "aligned")
    assert "arith.maximumf" in result["source"]
    pool = next(
        op
        for op in source_members(result["source"])[1]
        if getattr(op.attributes.get("prov.trace_role"), "data", "") == "node0_pool"
    )
    assert len(pool.get_iterator_types()) == 6
    assert result["references"][0]["shape"] == [1, 3 if padded else 2, 3 if padded else 2, 4]
    assert payload(profile, tmp_path, "partial")["references"][0]["shape"] == [
        1,
        3 if padded else 2,
        3 if padded else 2,
        5,
    ]


def test_readout_columns_bind_to_source_shape_without_assumed_store_counts(profile, tmp_path):
    import struct

    from merlin_experiments.phase0.fusion_execution import _readout_bindings

    candidate, schemes, _, _ = profile
    scheme = next(iter(schemes.values()))
    scheme.update(weight_scale_granularity="column_block", columns_per_scale=4)
    result = payload(profile, tmp_path, "partial")
    proof = dict(bindings=[dict(symbol="kernel")])
    args = [dict(role="input"), dict(role="input"), dict(role="output")]
    parameters = _readout_bindings(result, proof, "kernel", args)
    (part,) = parameters["partitions"]
    case = result["nodes"][0]["operation"]["attributes"]["case"]
    assert part["shape"] == [5, 5]
    assert part["output"] == dict(kind="argbase", arg_index=2, offset=0)
    assert part["column_scale_bits"] == [
        struct.unpack("<I", struct.pack("<f", case["multipliers"][c // 4]))[0] for c in range(5)
    ]


def test_operand_register_choices_need_independently_proven_numeric_admission():
    from merlin_experiments.phase0.fusion_execution import _operand_sum_proof

    proof = dict(kernels=[dict(symbol="kernel", status="declined", operand_sum=dict(status="proven"))])
    with pytest.raises(ValueError, match="no numeric proof"):
        _operand_sum_proof(proof, "kernel")
    proof["kernels"][0].update(status="proven", operand_sum=dict(status="declined"))
    with pytest.raises(ValueError, match="no numeric proof"):
        _operand_sum_proof(proof, "kernel")


@pytest.mark.parametrize("columns", [3, 4, 5, 9])
def test_diagonal_extension_work_is_per_output_tile(profile, columns):
    from merlin_experiments.phase0.fusion_execution import _extension_parameters

    from merlin.targetgen.composition_audit import _k_extension

    facts = profile[2]
    case = dict(
        coefficient=1, source_k=5, residual_width=columns, source_partial_bound=5 * 128**2, residual_range=[-128, 127]
    )
    parameters = _extension_parameters(case, facts)
    assert parameters == dict(source_k=5, extension_k=4)
    # Two source K tiles and one diagonal digit tile for this output.
    work = [(None, dict(c_addr=32))] * 3
    commits = [(dict(acc_addr=32), None, None, work)]
    _k_extension(commits, parameters, dict(tile_k=4, address_flags=0))
    with pytest.raises(ValueError, match="extra K tiles"):
        _k_extension([(dict(acc_addr=32), None, None, work[:-1])], parameters, dict(tile_k=4, address_flags=0))


@pytest.fixture
def sealed_sample(profile, tmp_path, monkeypatch):
    """Tiny sealed campaign; execution re-audits are mocked, reader custody is real."""
    from merlin_experiments.phase0 import composition_generation as G

    root = tmp_path / "sample"
    campaign = root / "campaign"
    campaign.mkdir(parents=True)
    original, schemes, facts, context = copy.deepcopy(profile)
    facts.update(dtype="int8", producer_placement={"contraction": ["accelerator"]})
    candidates, cases, results = [], [], {}
    all_schemes = {}
    for index, admitted in enumerate((True, True, False)):
        candidate = copy.deepcopy(original)
        candidate["candidate_id"] = f"candidate-{index}"
        node = candidate["semantic"]["nodes"][0]
        scheme = copy.deepcopy(next(iter(schemes.values())))
        scheme.pop("columns_per_scale")  # Tensor scaling has no block-width premise.
        if index == 1:
            scheme["id"] = "second_scale"
        node.update(scheme=scheme["id"], regime=scheme)
        all_schemes[scheme["id"]] = scheme
        candidates.append(candidate)
        for label in ("aligned", "partial", "rne_saturation", "conversion_boundary"):
            frozen = C.build_payload(
                candidate, {scheme["id"]: scheme}, facts, label, None, frozenset(), context, root / "screen"
            )
            member = f"members/{index}-{label}.json"
            C.write_frozen(root / member, C.canonical(frozen))
            case_id = C.digest(dict(candidate_id=candidate["candidate_id"], shape_class=label, coefficient=None))
            cases.append(
                dict(
                    case_id=case_id,
                    candidate_id=candidate["candidate_id"],
                    member=member,
                    sha256=C.file_digest(root / member),
                    bytes=(root / member).stat().st_size,
                    shape_class=label,
                    positive=True,
                    required_engines=["spike"],
                    coefficient=None,
                )
            )
            results[case_id] = dict(
                case_id=case_id,
                candidate_id=candidate["candidate_id"],
                case_sha256=cases[-1]["sha256"],
                outcome="pass" if admitted else "not_admitted",
                numeric=True,
                placement=admitted,
                trace=admitted,
                evidence_members=[],
                **({} if admitted else {"reason": "split_kernels"}),
            )
    from merlin.compile.fusion_enumeration import stratify_candidates

    spec = dict(quantization=dict(schemes=dict(schemes=list(all_schemes.values()))))
    manifest = stratify_candidates(
        dict(target="synthetic", profile={}, rows=candidates, enumeration_complete=True), spec
    )
    C.write_frozen(root / "candidates.json", C.canonical(manifest))
    for name in ("facts", "spec", "extension", "provider", "backend"):
        C.write_frozen(root / name, C.canonical(spec if name == "spec" else {"selection": name}))
    suite = dict(
        schema="merlin.fusion_cases.v1",
        target="synthetic",
        identity=dict(
            candidate_manifest_sha256=C.file_digest(root / "candidates.json"),
            reference_implementation_sha256=C.reference_identity(),
            private_exclusion_sha256=C.digest(sorted(C.corpus.held_out_shapes())),
        ),
        profile={},
        candidate_count=3,
        candidate_case_count=3,
        case_count=len(cases),
        rows=cases,
    )
    suite["seal"] = dict(canonicalization="merlin.canonical_json.v1", payload_sha256=C.digest(suite))
    C.write_frozen(root / "cases.json", C.canonical(suite))
    identity = dict(
        facts_sha256=C.file_digest(root / "facts"),
        spec_sha256=C.file_digest(root / "spec"),
        contract_sha256="contract",
        candidate_manifest_sha256=C.file_digest(root / "candidates.json"),
        case_manifest_sha256=C.file_digest(root / "cases.json"),
        simulator_extension_sha256=C.file_digest(root / "extension"),
        reference_backend_sha256=C.file_digest(root / "backend"),
    )
    for case_id, result in results.items():
        result["identity"] = identity
        C.write_frozen(campaign / "workers" / case_id / "result.json", C.canonical(result))
    C.write_frozen(campaign / "execution-selection.json", C.canonical({}))
    ledger = V.record(manifest, suite, identity, results)
    path = campaign / "validation.json"
    path.write_bytes(C.canonical(ledger))
    binding = dict(
        path=str(path),
        sha256=C.file_digest(path),
        candidates=str(root / "candidates.json"),
        cases=str(root / "cases.json"),
        **{name: str(root / name) for name in ("facts", "spec", "extension", "provider", "backend")},
    )
    evidence = SimpleNamespace(
        target="synthetic",
        raw_facts_sha256=identity["facts_sha256"],
        source_snapshots=[
            SimpleNamespace(
                role="software-spec",
                sha256=identity["spec_sha256"],
                content=(root / "spec").read_bytes(),
                path=root / "spec",
            )
        ],
    )
    monkeypatch.setattr(V, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(V, "_case_inputs", lambda *args: (all_schemes, facts, C.corpus.held_out_shapes(), context))
    monkeypatch.setattr(V, "execution_identity", lambda *args: identity)
    monkeypatch.setattr(V, "_inventory", C.file_digest)
    from merlin_experiments.phase0 import fusion_execution as E

    monkeypatch.setattr(E, "verify_pass", lambda *args: None)
    verified, reopened, suite_root = G.read_bound_validation(binding, base=root, evidence=evidence)
    assert verified == ledger and reopened == suite
    return SimpleNamespace(
        ledger=ledger,
        suite=suite,
        root=suite_root,
        binding=binding,
        evidence=evidence,
        facts=facts,
        spec=dict(quantization=dict(schemes=dict(schemes=list(all_schemes.values())))),
    )


def test_validated_sample_determinism_cover_and_rejections(sealed_sample):
    from merlin.targetgen.composition_corpus import sample_validated

    fixture = sealed_sample
    first = sample_validated(fixture.ledger, fixture.suite, software_spec=fixture.spec)
    shuffled = copy.deepcopy(fixture.ledger)
    shuffled["rows"].reverse()
    suite = {**fixture.suite, "rows": list(reversed(fixture.suite["rows"]))}
    assert first == sample_validated(shuffled, suite, software_spec=fixture.spec)
    assert first["minimum_cover"] == 2 and first["composition_count"] == 8
    assert first["cross_checks"] == 6 and first["underfill"] == 42
    assert first["unsampled_regimes"] == [] and len(first["admitted_regimes"]) == 2
    for regime in first["admitted_regimes"]:
        assert {row["shape_class"] for row in first["selected"] if row["regime"] == regime} >= {"aligned", "partial"}
    assert all(row["candidate_id"] != "candidate-2" for row in first["selected"])
    assert first["not_admitted"] == [dict(candidate_id="candidate-2", reasons=["split_kernels"])]


@pytest.mark.parametrize("member", ["path", "candidates", "cases", "facts", "spec", "extension", "backend"])
def test_bound_record_digest_mismatch_fails_closed(sealed_sample, member):
    from pathlib import Path

    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fixture = sealed_sample
    with Path(fixture.binding[member]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest|selection"):
        read_bound_validation(fixture.binding, base=fixture.root, evidence=fixture.evidence)


def test_validated_entries_use_existing_builder_and_coverage(sealed_sample):
    from merlin_experiments.phase0.composition_generation import append_validated, bind_capsule, coverage

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves
    from merlin.targetgen.composition_golden import evaluate

    fixture = sealed_sample
    entries, selection = append_validated(
        [],
        {},
        fixture.facts,
        budget=160,
        validation=fixture.ledger,
        suite=fixture.suite,
        suite_root=fixture.root,
        record_sha256=fixture.binding["sha256"],
    )
    binding = CS.CorpusBinding(
        target="synthetic",
        tile_dim=4,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )
    for entry in entries:
        assert entry["generalization"]["composition_truth"]["outcome"] == "pass", entry["generalization"][
            "composition_truth"
        ]["reasons"]
        cap, source = CS.build(entry, binding)
        bind_capsule(cap, source)
        case_id = entry["generalization"]["fusion_validation"]["case_ids"][0]
        case = next(row for row in fixture.suite["rows"] if row["case_id"] == case_id)
        import json

        frozen = json.loads((fixture.root / case["member"]).read_bytes())
        leaves = materialize_capsule_leaves(cap)
        for name, value in leaves.items():
            assert value.to_list() == frozen["input_values"][f"node0_{name}"]
        assert list(evaluate(cap).values()) == [row["values"] for row in frozen["references"]]
        assert cap["expected"]["composition"]["source_op_indices"]
    rows = coverage(entries, validation=fixture.ledger)
    assert sum(row["count"] for row in rows) == selection["composition_count"]
    assert not any(row["status"] == "not_sampled" for row in rows)
    rejected = [row for row in rows if row["not_admitted"]]
    assert len(rejected) == 1 and rejected[0]["reasons"] == ["split_kernels"]
    assert rejected[0]["count"] == 0
    excluded = coverage([], validation=fixture.ledger, excluded_spec={})
    assert all(row["status"] == "out_of_scope" and row["count"] == 0 for row in excluded)


def test_bound_validation_rechecks_current_held_out_shapes(sealed_sample, monkeypatch):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    monkeypatch.setattr(C.corpus, "held_out_shapes", lambda: frozenset({(123, 456, 789)}))
    with pytest.raises(ValueError, match="held-out selection"):
        read_bound_validation(sealed_sample.binding, base=sealed_sample.root, evidence=sealed_sample.evidence)


def test_written_sample_cannot_lose_a_regime(sealed_sample, tmp_path):
    from merlin_experiments.phase0.composition_generation import append_validated, verify_written

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.corpus_synth import SynthesisError

    fixture = sealed_sample
    entries, selection = append_validated(
        [],
        {},
        fixture.facts,
        budget=160,
        validation=fixture.ledger,
        suite=fixture.suite,
        suite_root=fixture.root,
        record_sha256=fixture.binding["sha256"],
    )
    binding = CS.CorpusBinding(
        target="synthetic",
        tile_dim=4,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        classes_for=lambda **_: [],
    )
    written = []
    for entry in entries:
        cap, _ = CS.build(entry, binding)
        root = tmp_path / entry["name"]
        root.mkdir()
        (root / "capsule.yaml").write_text(yaml.safe_dump(cap))
        written.append(root)
    context = SimpleNamespace(source_root=tmp_path, held_out_inputs=[], workload_spec={})
    verify_written(entries, written, selection, context)
    with pytest.raises(SynthesisError, match="actual emitted"):
        verify_written(entries, written[1:], selection, context)
    member = written[0] / "capsule.yaml"
    cap = yaml.safe_load(member.read_text())
    values = cap["input_palette"]["inputs"][0]["values"]
    values[0] = values[0] + 1 if values[0] < 127 else values[0] - 1
    member.write_text(yaml.safe_dump(cap))
    with pytest.raises(SynthesisError, match="frozen validated stimuli"):
        verify_written(entries, written, selection, context)


def test_authored_names_are_only_display_aliases(sealed_sample):
    from merlin_experiments.phase0.composition_generation import append_validated

    fixture = sealed_sample
    arguments = dict(
        budget=160,
        validation=fixture.ledger,
        suite=fixture.suite,
        suite_root=fixture.root,
        record_sha256=fixture.binding["sha256"],
    )
    spec = dict(
        **fixture.spec,
        compositions=dict(
            chains=[
                dict(
                    id="display-name",
                    producer="contraction",
                    exits=["requant"],
                    quantization_schemes=dict(primary=["test_scale", "second_scale"]),
                )
            ]
        ),
    )
    entries, sample = append_validated([], spec, fixture.facts, **arguments)
    without_aliases, other = append_validated([], {}, fixture.facts, **arguments)
    assert sample == other
    for entry, plain in zip(entries, without_aliases, strict=True):
        assert entry["generalization"]["fusion_validation"].pop("display_aliases") == ["display-name"]
        assert plain["generalization"]["fusion_validation"].pop("display_aliases") == []
        assert entry == plain


@pytest.fixture
def fixed_suite(profile, tmp_path, monkeypatch):
    import json

    candidate, schemes, facts, context = profile
    candidate["semantic"]["nodes"][0]["residual"] = dict(coefficient_bound=127)
    candidates = tmp_path / "candidates.json"
    from merlin.compile.fusion_enumeration import stratify_candidates

    candidates.write_bytes(C.canonical(stratify_candidates(dict(rows=[candidate]), {})))
    rows = C._generate_candidate(candidate, schemes, facts, frozenset(), context, tmp_path)
    monkeypatch.setattr(V, "_case_inputs", lambda *args: (schemes, facts, frozenset(), context))
    suite = dict(
        schema="merlin.fusion_cases.v1",
        target="synthetic",
        identity=dict(
            candidate_manifest_sha256=C.file_digest(candidates),
            reference_implementation_sha256=C.reference_identity(),
            private_exclusion_sha256=C.digest([]),
        ),
        profile={},
        candidate_count=1,
        candidate_case_count=1,
        case_count=len(rows),
        rows=rows,
    )
    path = tmp_path / "suite.json"

    def save():
        suite.pop("seal", None)
        suite["seal"] = dict(canonicalization="merlin.canonical_json.v1", payload_sha256=C.digest(suite))
        path.write_bytes(C.canonical(suite))

    save()
    assert V.read_suite(path, candidates) == json.loads(path.read_bytes())
    return path, candidates, suite, save


@pytest.mark.parametrize("index", range(6))
def test_suite_refuses_each_missing_required_case(fixed_suite, index):
    path, candidates, suite, save = fixed_suite
    suite["rows"].pop(index)
    suite["case_count"] -= 1
    save()
    with pytest.raises(ValueError, match="required case identities"):
        V.read_suite(path, candidates)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("field", ["input_values", "references", "source", "nodes"])
def test_suite_refuses_resealed_forged_payloads(fixed_suite, field, workers):
    import json

    path, candidates, suite, save = fixed_suite
    row = suite["rows"][0]
    member = path.parent / row["member"]
    payload = json.loads(member.read_bytes())
    if field == "input_values":
        payload[field][next(iter(payload[field]))][0][0] += 1
        payload["inputs_sha256"] = C.digest(dict(inputs=payload["inputs"], values=payload[field]))
    elif field == "references":
        payload[field][0]["values"][0][0] += 1
        payload["references_sha256"] = C.digest(payload[field])
    elif field == "source":
        payload[field] += "\n"
        import hashlib

        payload["source_sha256"] = hashlib.sha256(payload[field].encode()).hexdigest()
    else:
        payload[field][0]["operation"]["attributes"]["case"]["multipliers"] = [0.25]
    member.write_bytes(C.canonical(payload))
    row.update(bytes=member.stat().st_size, sha256=C.file_digest(member))
    save()
    with pytest.raises(ValueError, match="deterministic generation"):
        V.read_suite(path, candidates, workers=workers)


_forked = pytest.mark.skipif(
    "fork" not in __import__("multiprocessing").get_all_start_methods(),
    reason="workers must inherit the patched builder",
)


@pytest.fixture
def forked_workers(monkeypatch):
    """Patched builders reach workers only through fork; production workers never fork."""
    monkeypatch.setattr(V, "POOL_START_METHOD", "fork")


def test_verification_workers_never_fork_the_reader(monkeypatch):
    seen = []

    class Pool:
        def __init__(self, max_workers, mp_context, **kwargs):
            seen.append(mp_context.get_start_method())

    monkeypatch.setattr(V, "ProcessPoolExecutor", Pool)
    V.worker_pool(2)
    assert seen == ["forkserver"]


def test_parallel_payload_verification_matches_serial(fixed_suite):
    path, candidates, suite, save = fixed_suite
    serial = V.read_suite(path, candidates)
    assert V.read_suite(path, candidates, workers=4) == serial == V.read_suite(path, candidates, workers=64)
    with pytest.raises(ValueError, match="positive integer"):
        V.read_suite(path, candidates, workers=0)


@_forked
@pytest.mark.parametrize("workers", [1, 3, 4])
def test_parallel_payload_verification_reports_the_first_canonical_failure(
    fixed_suite, monkeypatch, forked_workers, workers
):
    import time

    path, candidates, suite, save = fixed_suite
    first, later = suite["rows"][1], suite["rows"][4]
    original = C.build_payload

    def build(candidate, schemes, facts, label, coefficient, *args):
        if (label, coefficient) == (first["shape_class"], first["coefficient"]):
            time.sleep(0.5)  # the later failure finishes first under parallel scheduling
            raise ValueError("first canonical failure")
        if (label, coefficient) == (later["shape_class"], later["coefficient"]):
            raise ValueError("later failure")
        return original(candidate, schemes, facts, label, coefficient, *args)

    monkeypatch.setattr(C, "build_payload", build)
    with pytest.raises(ValueError, match="^first canonical failure$"):
        V.read_suite(path, candidates, workers=workers)


@_forked
def test_parallel_payload_worker_crash_fails_the_read(fixed_suite, monkeypatch, forked_workers):
    import os
    from concurrent.futures.process import BrokenProcessPool

    path, candidates, suite, save = fixed_suite
    parent = os.getpid()
    original = C.build_payload

    def build(*args):
        if os.getpid() != parent:
            os._exit(1)
        return original(*args)

    monkeypatch.setattr(C, "build_payload", build)
    with pytest.raises(BrokenProcessPool):
        V.read_suite(path, candidates, workers=4)


def test_padded_max_pool_uses_negative_infinity_for_negative_readouts(profile, tmp_path):
    import numpy as np
    from merlin_experiments.phase0.fusion_reference import exact_reference, unfused_reference

    node = profile[0]["semantic"]["nodes"][0]
    node["readout"].update(pool_padding="zero_or_padded", pool_padding_class="padded")
    node["readout"]["stages"].append(dict(kind="pool", padding="zero_tap", reductions=["max"]))
    frozen = payload(profile, tmp_path)
    assert "%pz = arith.constant 0xff800000 : f32" in frozen["source"]
    cap = frozen["nodes"][0]
    env = {spec["name"]: np.full(spec["shape"], -8 if spec["name"] == "A" else 8) for spec in cap["inputs"]}
    exact, _ = exact_reference(cap, env)
    assert exact == unfused_reference(cap, env)
    assert all(value < 0 for value in np.asarray(next(iter(exact.values()))).flat)


@pytest.mark.parametrize("padded", [False, True])
def test_trace_audit_preserves_source_declaration_metadata(profile, tmp_path, monkeypatch, padded):
    from merlin_experiments.phase0 import fusion_execution as E

    from merlin.llvmlower import device_build
    from merlin.targetgen import composition_audit, frozen_placement, native_instruction_audit
    from merlin.targetgen.admission_conditions import binding_declaration

    binding = dict(symbol="kernel", op="aten.mm.default", numeric=dict(integer_producer=True))
    proof = dict(bindings=[binding])
    placement = dict(symbol="kernel", program_sha256="source", source_op_indices=[0], stages=dict(mac=[0]))
    monkeypatch.setattr(device_build, "load_region_offload", lambda *args: dict(routed=[binding]))
    case = dict(producer="contraction", exit="accumulator", readout_stages=[])
    if padded:
        case.update(
            readout_stages=["pool"], pool_in_dims=[5, 7], pool_window=[3, 3], pool_stride=[2, 2], padding=[1, 2]
        )
    monkeypatch.setattr(E, "_trace_contexts", lambda *args: [(case, {}, {})])

    def audit(**kwargs):
        assert binding_declaration(kwargs["routed"][0]) == "contraction"
        assert kwargs["routed"][0]["numeric"] == binding["numeric"]
        return dict(status="pass")

    monkeypatch.setattr(native_instruction_audit, "audit_case_instructions", audit)
    monkeypatch.setattr(native_instruction_audit, "decode_retirement_trace", lambda **kwargs: {})
    monkeypatch.setattr(frozen_placement, "host_chain_consumers", lambda *args: [])

    def check(decoded, expected, **kwargs):
        if padded:
            fields = expected["mode_parameters"]["pooled_store"]["fields"]
            assert fields == dict(
                pool_stride=2, pool_size=3, pool_out_dim=5, porows=3, pocols=5, orows=5, ocols=7, upad=1, lpad=2
            )
            composition_audit._pool([(None, fields, None, None)], dict(fields=fields))
            with pytest.raises(ValueError, match="window differs"):
                composition_audit._pool([(None, {**fields, "lpad": 0}, None, None)], dict(fields=fields))
        return dict(status="pass")

    monkeypatch.setattr(composition_audit, "check_composition", check)
    result = E._trace_audit(
        dict(source="source", nodes=[dict(operation=dict(attributes=dict(N=16)))]),
        profile[0],
        placement,
        proof,
        tmp_path,
        dict(elf=str(tmp_path / "elf")),
        dict(output_path={}, raw={}),
        "synthetic",
    )
    assert result["status"] == "pass"


def test_shared_input_obligations_are_reconstructed_and_scope_is_explicit(profile):
    candidate = copy.deepcopy(profile[0])
    candidate["semantic"].update(
        structure="shared_input", nodes=[copy.deepcopy(candidate["semantic"]["nodes"][0]) for _ in range(2)]
    )
    manifest = dict(target="synthetic", profile={}, rows=[candidate], enumeration_complete=True)
    suite = dict(schema="merlin.fusion_cases.v1", profile={}, identity={}, rows=[], case_count=0)
    identity = dict(facts_sha256="facts", contract_sha256="contract")
    missing = V.record(manifest, suite, identity, {})
    assert len(missing["rows"][0]["required_case_ids"]) == 4
    assert missing["rows"][0]["reasons"] == ["pending_execution"]
    assert not missing["execution_complete"]
    suite["rows"] = [
        dict(case_id=case, candidate_id=candidate["candidate_id"], positive=True)
        for case in missing["rows"][0]["required_case_ids"]
    ]
    suite["case_count"] = len(suite["rows"])
    results = {
        case["case_id"]: dict(outcome="pass", numeric=True, placement=True, trace=True, evidence_members=[])
        for case in suite["rows"]
    }
    complete = V.record(manifest, suite, identity, results)
    assert complete["execution_complete"] and complete["rows"][0]["admitted"]
    suite.update(profile=dict(selected_candidate_ids=["other"]), rows=[], case_count=0)
    outside = V.record(manifest, suite, identity, {})["rows"][0]
    assert outside["state"] == "not_selected_for_execution"
    assert outside["required_case_ids"] == [] and outside["reasons"] == ["outside_execution_scope"]


def test_sampler_excludes_deferred_candidates(sealed_sample):
    from merlin.targetgen.composition_corpus import sample_validated

    fixture = sealed_sample
    fixture.ledger["rows"][1].update(state="deferred_by_stratification", admitted=True)
    fixture.ledger["scope"]["campaign"] = dict(per_stratum=2)
    from merlin.targetgen.corpus_synth import SynthesisError

    with pytest.raises(SynthesisError, match="required fusion regimes") as error:
        sample_validated(fixture.ledger, fixture.suite, software_spec=fixture.spec)
    assert error.value.selection["seal_blockers"][0]["regime"]["scheme"] == "second_scale"
    sample = sample_validated(fixture.ledger, fixture.suite, software_spec={})
    assert {row["candidate_id"] for row in sample["selected"]} == {"candidate-0"}
    assert len(sample["admitted_regimes"]) == 1


def test_stratified_reader_reproduces_smoke_scope(sealed_sample):
    import json

    from merlin.compile.fusion_enumeration import stratify_candidates

    fixture = sealed_sample
    candidate_path = fixture.root / "candidates.json"
    manifest = stratify_candidates(
        json.loads(candidate_path.read_bytes()), dict(compositions=dict(campaign_per_stratum=1))
    )
    candidate_path.write_bytes(C.canonical(manifest))
    suite = copy.deepcopy(fixture.suite)
    selected = V.smoke_ids(manifest)
    suite["rows"] = [row for row in suite["rows"] if row["candidate_id"] in selected]
    suite["case_count"] = len(suite["rows"])
    suite["candidate_case_count"] = len(selected)
    suite["profile"].update(selected_candidate_ids=selected, execution_scope="smoke")
    suite["identity"]["candidate_manifest_sha256"] = C.file_digest(candidate_path)
    path = fixture.root / "scope-cases.json"

    def write_suite():
        suite.pop("seal", None)
        suite["seal"] = dict(canonicalization="merlin.canonical_json.v1", payload_sha256=C.digest(suite))
        path.write_bytes(C.canonical(suite))

    write_suite()
    assert V.read_suite(path, candidate_path)["case_count"] == suite["case_count"]
    suite["profile"]["selected_candidate_ids"] = selected[:1]
    write_suite()
    with pytest.raises(ValueError, match="deterministic execution scope"):
        V.read_suite(path, candidate_path)


def test_bound_reader_uses_only_current_extension_identity(sealed_sample):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    assert "libgemmini_rne_sha256" not in fx.ledger["identity"]
    assert fx.ledger["identity"]["simulator_extension_sha256"] == C.file_digest(fx.binding["extension"])
    assert read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)[0] == fx.ledger


def test_bound_reader_rebuilds_payloads_from_the_bound_facts(sealed_sample, monkeypatch):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    original, seen = V.read_suite, []

    def read_suite(path, candidates, facts_path=None, **kwargs):
        seen.append(facts_path)
        return original(path, candidates, facts_path, **kwargs)

    monkeypatch.setattr(V, "read_suite", read_suite)
    read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)
    assert [C.file_digest(path) for path in seen] == [fx.ledger["identity"]["facts_sha256"]]


_fork_only = pytest.mark.skipif(
    "fork" not in __import__("multiprocessing").get_all_start_methods(),
    reason="workers must inherit the patched readers",
)


@_fork_only
def test_bound_reader_parallel_verification_matches_serial(sealed_sample, monkeypatch, forked_workers):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    parallel = []
    original = V._verify_payload_rows_parallel

    def spy(inputs, rows, workers):
        parallel.append(workers)
        return original(inputs, rows, workers)

    monkeypatch.setattr(V, "_verify_payload_rows_parallel", spy)
    monkeypatch.delenv(V.VERIFY_WORKERS_ENV, raising=False)
    serial = read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)
    assert parallel == []
    explicit = read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence, workers=4)
    monkeypatch.setenv(V.VERIFY_WORKERS_ENV, "3")
    from_env = read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)
    assert serial == explicit == from_env
    assert serial[0] == fx.ledger and parallel == [4, 3]


@_fork_only
@pytest.mark.parametrize("stage", ["payload", "pass_audit"])
def test_bound_reader_parallel_refuses_a_disagreeing_case(sealed_sample, monkeypatch, forked_workers, stage):
    from merlin_experiments.phase0 import fusion_execution as E
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    first, later = fx.suite["rows"][1], fx.suite["rows"][5]
    if stage == "payload":
        original = C.build_payload

        def build(candidate, schemes, facts, label, coefficient, *args):
            payload = original(candidate, schemes, facts, label, coefficient, *args)
            if (candidate["candidate_id"], label) in {
                (first["candidate_id"], first["shape_class"]),
                (later["candidate_id"], later["shape_class"]),
            }:
                payload["references"][0]["values"][0][0] += 1  # regeneration no longer matches frozen bytes
            return payload

        monkeypatch.setattr(C, "build_payload", build)
        message = "deterministic generation"
    else:

        def verify(payload, candidate, selection, directory):
            if directory.name == later["case_id"]:
                raise ValueError("later pass audit")
            if directory.name == first["case_id"]:
                __import__("time").sleep(0.5)
                raise ValueError("first pass audit")

        monkeypatch.setattr(E, "verify_pass", verify)
        message = "^first pass audit$"
    with pytest.raises(ValueError, match=message):
        read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence, workers=4)


@pytest.mark.parametrize("workers", [0, -2, True, 2.0, "4"])
def test_bound_reader_refuses_invalid_explicit_workers(sealed_sample, workers):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    with pytest.raises(ValueError, match="positive integer"):
        read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence, workers=workers)


@pytest.mark.parametrize("value", ["0", "-1", "four", "2.5"])
def test_bound_reader_refuses_invalid_environment_workers(sealed_sample, monkeypatch, value):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    monkeypatch.setenv(V.VERIFY_WORKERS_ENV, value)
    with pytest.raises(ValueError, match="positive integer"):
        read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)


def test_verification_workers_default_is_serial_not_host_scaled(monkeypatch):
    monkeypatch.delenv(V.VERIFY_WORKERS_ENV, raising=False)
    assert V.verification_workers() == 1
    monkeypatch.setenv(V.VERIFY_WORKERS_ENV, "")
    assert V.verification_workers() == 1
    monkeypatch.setenv(V.VERIFY_WORKERS_ENV, "16")
    assert V.verification_workers() == 16 and V.verification_workers(2) == 2


def test_validation_reader_refuses_invalid_workers(sealed_sample):
    fx = sealed_sample
    with pytest.raises(ValueError, match="positive integer"):
        V.read_validation(fx.binding["path"], {}, fx.suite, {}, fx.root, workers=0)


@pytest.mark.parametrize("mutation", ["smoke", "missing_campaign", "subset"])
def test_sealed_smoke_and_missing_campaign_cannot_be_complete(sealed_sample, mutation):
    from merlin_experiments.phase0.composition_generation import _require_complete_execution

    ledger = copy.deepcopy(sealed_sample.ledger)
    if mutation == "smoke":
        ledger["scope"]["execution_scope"] = "smoke"
    elif mutation == "missing_campaign":
        ledger["scope"].pop("campaign")
    else:
        ledger["rows"][0]["state"] = "not_selected_for_execution"
    with pytest.raises(ValueError, match="campaign|smoke"):
        _require_complete_execution(ledger)


def test_plain_reader_uses_selected_spec_campaign_limit(sealed_sample):
    import json

    from merlin.compile.fusion_enumeration import stratify_candidates

    fx = sealed_sample
    from pathlib import Path

    manifest = json.loads(Path(fx.binding["candidates"]).read_bytes())
    changed = stratify_candidates(manifest, dict(compositions=dict(campaign_per_stratum=1)))
    with pytest.raises(ValueError, match="stratification"):
        V.read_validation(
            fx.binding["path"], changed, fx.suite, fx.ledger["identity"], fx.root / "campaign", spec=fx.spec
        )
    manifest.pop("campaign")
    with pytest.raises(ValueError, match="campaign"):
        V.read_validation(
            fx.binding["path"], manifest, fx.suite, fx.ledger["identity"], fx.root / "campaign", spec=fx.spec
        )


def test_retained_evidence_location_does_not_define_case_suite_root(sealed_sample, tmp_path):
    import json
    import shutil
    from pathlib import Path

    fx = sealed_sample
    retained = tmp_path / "separate" / "retained" / "deep"
    shutil.copytree(fx.root / "campaign", retained)
    (retained / "execution-selection.json").write_bytes(C.canonical(dict(cases=str(Path(fx.binding["cases"])))))
    manifest = json.loads(Path(fx.binding["candidates"]).read_bytes())
    assert (
        V.read_validation(retained / "validation.json", manifest, fx.suite, V.execution_identity(None, None), retained)
        == fx.ledger
    )
    (retained / "execution-selection.json").write_bytes(C.canonical({}))
    with pytest.raises(ValueError, match="suite root"):
        V.read_validation(retained / "validation.json", manifest, fx.suite, V.execution_identity(None, None), retained)


def test_independent_reference_disagreement_refuses_only_its_candidate(profile, tmp_path, monkeypatch):
    import concurrent.futures
    import json

    from merlin.compile.fusion_enumeration import _unordered_inputs, canonical_sha256, stratify_candidates

    candidate, schemes, facts, context = profile
    other = copy.deepcopy(candidate)
    other["candidate_id"] = "other"
    raw_facts = {}
    manifest = dict(
        target="synthetic",
        profile={},
        enumeration_complete=True,
        identity=dict(facts_sha256=canonical_sha256(_unordered_inputs(raw_facts))),
        rows=[candidate, other],
        candidate_count=2,
        scheme_resolution=dict(schemes=list(schemes.values())),
    )
    manifest = stratify_candidates(manifest, {})
    candidates_path, facts_path = tmp_path / "candidates.json", tmp_path / "facts.json"
    candidates_path.write_bytes(C.canonical(manifest))
    facts_path.write_bytes(C.canonical(raw_facts))
    monkeypatch.setattr(C, "facts_for_cases", lambda *args: facts)
    monkeypatch.setattr(C, "exclusion_context", lambda: context)
    monkeypatch.setattr(V, "worker_pool", lambda workers, **kwargs: concurrent.futures.ThreadPoolExecutor(workers))
    real_build = C.build_payload

    def build(row, *args):
        if row["candidate_id"] == candidate["candidate_id"]:
            raise C.ReferenceDisagreement("independent unfused references disagree")
        return real_build(row, *args)

    monkeypatch.setattr(C, "build_payload", build)
    for spec in (None, dict(compositions=dict(campaign_per_stratum=1))):
        # The campaign is re-derived from the selected spec, not trusted from its record.
        with pytest.raises(ValueError, match="selected software spec|deterministic stratification"):
            C.generate(candidates_path, facts_path, tmp_path / "suite", spec=spec)
    suite = C.generate(candidates_path, facts_path, tmp_path / "suite", spec={})
    assert suite["profile"]["refused_candidates"] == [
        dict(candidate_id=candidate["candidate_id"], reason="independent unfused references disagree")
    ]
    assert {row["candidate_id"] for row in suite["rows"]} == {"other"}
    ledger = V.record(manifest, suite, dict(facts_sha256="facts", contract_sha256="contract"), {})
    refused = next(row for row in ledger["rows"] if row["candidate_id"] == candidate["candidate_id"])
    assert not refused["admitted"] and refused["outcome"] == "not_admitted"
    assert refused["reasons"] == ["independent unfused references disagree"]
    assert json.loads((tmp_path / "suite" / "fusion-cases.json").read_bytes())["profile"] == suite["profile"]


def test_reference_mismatch_has_candidate_refusal_type(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "unfused_reference", lambda *args: {})
    with pytest.raises(C.ReferenceDisagreement, match="independent unfused"):
        payload(profile, tmp_path)


def test_plain_sealed_reader_always_requires_campaign(sealed_sample):
    import json
    from pathlib import Path

    fx = sealed_sample
    manifest = json.loads(Path(fx.binding["candidates"]).read_bytes())
    manifest.pop("campaign")
    with pytest.raises(ValueError, match="campaign"):
        V.read_validation(
            fx.binding["path"], manifest, fx.suite, V.execution_identity(None, None), fx.root / "campaign"
        )


@pytest.mark.parametrize("disagreement", [True, False])
def test_refused_candidate_is_rechecked_by_plain_suite_reader(fixed_suite, monkeypatch, disagreement):
    import json

    path, candidates, suite, save = fixed_suite
    manifest = json.loads(candidates.read_bytes())
    identity = manifest["rows"][0]["candidate_id"]
    suite["profile"]["refused_candidates"] = [
        dict(candidate_id=identity, reason="independent unfused references disagree")
    ]
    suite.update(rows=[], case_count=0, candidate_case_count=0)
    save()
    if disagreement:

        def refuse(*args):
            raise C.ReferenceDisagreement("independent unfused references disagree")

        monkeypatch.setattr(C, "build_payload", refuse)
        assert V.read_suite(path, candidates) == suite
    else:
        with pytest.raises(ValueError, match="refusal differs"):
            V.read_suite(path, candidates)


def test_bound_reader_rejects_smoke_even_when_all_candidates_were_selected(sealed_sample, monkeypatch):
    from merlin_experiments.phase0.composition_generation import read_bound_validation

    fx = sealed_sample
    suite = copy.deepcopy(fx.suite)
    suite["profile"]["execution_scope"] = "smoke"
    monkeypatch.setattr(V, "read_suite", lambda *args, **kwargs: suite)
    with pytest.raises(ValueError, match="smoke"):
        read_bound_validation(fx.binding, base=fx.root, evidence=fx.evidence)


def test_sealed_suite_cannot_disable_campaign_verification(fixed_suite):
    import json

    path, candidates, suite, save = fixed_suite
    manifest = json.loads(candidates.read_bytes())
    manifest.pop("campaign")
    candidates.write_bytes(C.canonical(manifest))
    suite["identity"]["candidate_manifest_sha256"] = C.file_digest(candidates)
    save()
    with pytest.raises(ValueError, match="campaign"):
        V.read_suite(path, candidates)


@pytest.mark.parametrize("state", ["deferred_by_stratification", "not_selected_for_execution"])
def test_unexecuted_rows_are_deferred_not_refused_in_coverage(profile, state):
    from merlin.targetgen.composition_corpus import validated_coverage

    row = copy.deepcopy(profile[0])
    row["state"] = state
    ledger = dict(
        rows=[
            dict(
                candidate_id=row["candidate_id"],
                semantic=row["semantic"],
                state=state,
                admitted=False,
                outcome="not_admitted",
                reasons=[state],
            )
        ]
    )
    (coverage,) = validated_coverage([], ledger)
    assert (coverage["state"], coverage["status"]) == (state, "deferred")
    assert (coverage["admitted"], coverage["not_admitted"], coverage["deferred"]) == (0, 0, 1)
    ledger["rows"][0]["state"] = "incompatible"
    (coverage,) = validated_coverage([], ledger)
    assert (coverage["status"], coverage["not_admitted"], coverage["deferred"]) == ("not_admitted", 1, 0)
