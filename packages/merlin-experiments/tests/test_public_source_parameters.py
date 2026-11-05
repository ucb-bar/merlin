"""Fresh buffer roles and common numerical policies remain independent of answers."""

import copy

import pytest
from merlin_experiments.phase0 import public_boundaries as B
from merlin_experiments.phase0 import public_boundary_semantics as S
from merlin_experiments.phase0 import public_source_policy as P
from merlin_experiments.phase0.public_boundary_parameters import captured_inputs
from test_public_functional_sources import POLICY, derive, inputs, program, role_trace

from merlin.targetgen import quant_recipe as Q
from merlin.targetgen.frontend_trace import _digest


def redigest(trace):
    graph = trace["graphs"]["original"]
    graph["sha256"] = _digest({k: v for k, v in graph.items() if k != "sha256"})
    return trace


def test_fresh_buffers_keep_explicit_source_roles_and_legacy_is_unchanged():
    original = role_trace(program())
    legacy, _ = S.cuts(original, max_nodes=16, typed=True, shared_inputs=True)
    current, _ = S.cuts(original, max_nodes=16, typed=True, shared_inputs=True, parameter_inputs=True)
    assert all("source_parameter" not in row for row in legacy[0]["inputs"])
    params = [row for row in current[0]["inputs"] if "source_parameter" in row]
    assert len(params) == 2 and {row["source_parameter"]["target"] for row in params} == {"Wq", "Wk"}
    assert S.cuts(original, max_nodes=16, typed=True, shared_inputs=True)[0] == legacy
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    generated = derive()[0][0]["public_boundary"]["program"]
    scope = {}
    exec(B.render(generated), scope)
    model, args = scope["get_model_and_inputs"]()
    assert len(args) == 2 and set(model.state_dict()) == {"v1", "v2"}
    exported = torch.export.export(model, args)
    assert [str(row.kind).split(".")[-1] for row in exported.graph_signature.input_specs] == [
        "BUFFER",
        "BUFFER",
        "USER_INPUT",
        "USER_INPUT",
    ]
    before = tuple(value.detach().clone() for value in model.buffers())
    model(*args)
    assert all(torch.equal(a, b) for a, b in zip(before, model.buffers(), strict=True))


@pytest.mark.parametrize("kind", ["PARAMETER", "BUFFER", "CONSTANT_TENSOR", "ATTRIBUTE"])
def test_original_explicit_constant_roles_replay_without_value_reads(kind):
    source = role_trace(program())
    node = source["graphs"]["original"]["nodes"][1]
    node["input_kind"] = kind
    if kind == "ATTRIBUTE":
        node["op"], node["target"] = "get_attr", node.pop("input_target")
        node.pop("input_kind")
    regions, gaps = S.cuts(redigest(source), max_nodes=16, typed=True, shared_inputs=True, parameter_inputs=True)
    assert not gaps and regions[0]["inputs"][1]["source_parameter"]["kind"] == kind
    node.pop("input_target", None)
    if kind != "ATTRIBUTE":
        assert S.cuts(redigest(source), max_nodes=16, typed=True, shared_inputs=True, parameter_inputs=True)[1]


@pytest.mark.parametrize("defect", ["dynamic", "foreign_target", "duplicate", "extra", "foreign_role"])
def test_captured_buffer_roster_cannot_be_replaced_by_dynamic_or_foreign_constants(defect):
    p = derive()[0][0]["public_boundary"]["program"]
    actual = role_trace(p)
    nodes = actual["graphs"]["original"]["nodes"]
    for node, row in zip(nodes, p["inputs"]):
        if "source_parameter" in row:
            node["input_kind"], node["input_target"] = "BUFFER", row["name"]
    assert len(captured_inputs(nodes, p)[0]) == len(p["inputs"])
    parameter = nodes[1]
    if defect == "dynamic":
        parameter["input_kind"] = "USER_INPUT"
    elif defect == "foreign_target":
        parameter["input_target"] = "foreign"
    elif defect == "duplicate":
        nodes[2]["input_target"] = parameter["input_target"]
    elif defect == "extra":
        nodes.insert(0, copy.deepcopy(parameter))
    else:
        parameter["input_kind"] = "CONSTANT_TENSOR"
    with pytest.raises(ValueError):
        captured_inputs(nodes, p)


def test_original_runtime_weight_is_not_silently_frozen_or_called_static_work():
    source = program()
    for row in source["inputs"]:
        row["parameter"] = False
    entries, _, report = derive(source)
    assert entries == []
    assert any(row.get("source_class") == "host_handled_not_author_candidate" for row in report["missing"])


@pytest.mark.parametrize("scope", ["host", "unreviewed", "different_operator", "composed_only"])
def test_other_placement_or_operator_constraints_do_not_license_source_work(scope):
    origins, products, evidence = inputs()
    row = evidence.software_spec["operations"][0]
    if scope == "host":
        row["placement"] = "host"
    elif scope == "unreviewed":
        row["status"] = "unreviewed"
    elif scope == "different_operator":
        row["ops"] = ["matmul"]
    else:
        row["signature"]["composed_with"] = ["unproved_composite"]
    entries, _, report = B.derive(origins, products, evidence=evidence, policy=POLICY)
    assert entries == [] and report["constructed"] == 0 and report["missing"]


def test_cpu_only_supported_cut_retains_obligation_but_never_enters_author_corpus():
    source = {
        "inputs": [{"name": "X", "shape": [2, 4], "dtype": "float32"}],
        "nodes": [
            {
                "name": "S",
                "target": "aten.softmax.int",
                "args": [{"value": "X"}, -1],
                "kwargs": {},
                "shape": [2, 4],
                "dtype": "float32",
            }
        ],
        "outputs": ["S"],
    }
    entries, _, report = derive(source)
    assert entries == [] and report["requested"] == 2
    assert sum(row.get("source_class") == "host_handled_not_author_candidate" for row in report["missing"]) == 2


def test_noncontraction_can_be_selected_work_only_with_its_explicit_quantized_contract():
    _, _, evidence = inputs()
    selection = P.select(evidence, None)
    add = {
        "inputs": [{"name": "a"}, {"name": "b"}],
        "nodes": [{"name": "c", "target": "aten.add.Tensor", "args": [{"value": "a"}, {"value": "b"}], "kwargs": {}}],
    }
    assert P.applicable(add, selection) == []
    selection["recipe"]["families"].append(Q.OPERAND_SUM)
    assert P.applicable(add, selection) == []  # recipe alone is not a selected SW operation
    selection["operations"].append({"id": "sum", "families": [Q.OPERAND_SUM]})
    assert P.applicable(add, selection) == [{"node": "c", "family": Q.OPERAND_SUM, "declarations": ["sum"]}]


def native_inputs():
    origins, products, evidence = inputs()
    evidence.software_spec["quantization"]["methodology"] = {
        "schema": "merlin.quantization_methodology.v1",
        "scale_granularity": "from_output_path",
        "bias": "accumulator_preload_int32",
        "epilogue": "integer_domain",
        "rounding": "from_output_path",
        "clamp": "from_output_path",
        "calibration": {"activation": "mse", "weight": "minmax"},
        "recovery": {"cross_layer_equalization": False, "bias_correction": False},
        "float_island": {"dtype": "fp32", "ops": ["softmax"]},
    }
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"kind": "preload", "dtype": "i32", "range": [-2147483648, 2147483647], "rounding": "none"},
            {"kind": "accumulate", "dtype": "i32", "operand_scale": "none"},
        ],
        "readout": [
            {"kind": "activation", "function": "relu", "domain": "accumulator"},
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {
                "kind": "scale",
                "dtype": "f32",
                "rounding": "half_even",
                "granularity": "per_command",
                "columns_per_scale": 4,
            },
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
        "exits": {"accumulator": True, "requant": True},
    }
    evidence.contract["output_epilogue"] = {"output_path": path}
    policy = {
        "schema": "numerical_recipe_policy_v1",
        "weight_granularity": "tensor",
        "native_outputs": {"operators": ["linear"], "reader_policy": "compute"},
        "dynamic_activation_planes": None,
        "float_quality": "report_only",
        "integer_extensions": {},
    }
    return origins, products, evidence, policy


def test_common_native_policy_is_explicit_model_free_and_bound_to_every_member():
    origins, products, evidence, policy = native_inputs()
    entries, _, report = B.derive(origins, products, evidence=evidence, policy=POLICY, numerical_policy=policy)
    assert entries
    for entry in entries:
        member, recipe = entry["public_boundary"], entry["quant_recipe"]
        assert recipe["weight"]["granularity"] == "tensor" and recipe["methodology"]["model"] is None
        assert member["numerical_selection"]["policy"] == policy
        assert member["numerical_selection"] == report["numerical_selection"]
        assert member["extent_scope"] == "logical_source_probe_only"
        assert recipe["epilogue"]["output_quantization"]["operators"] == ["linear"]
    baseline = copy.deepcopy((entries, report))
    evidence.software_spec["quantization"]["methodology"]["model_variants"] = [
        {"malformed_private": "must not be inspected"}
    ]
    after = B.derive(origins, products, evidence=evidence, policy=POLICY, numerical_policy=policy)
    assert (after[0], after[2]) == baseline


def test_legacy_policy_selection_is_never_reinterpreted():
    origins, products, evidence, policy = native_inputs()
    with pytest.raises(ValueError, match="prospective"):
        B.derive(
            origins, products, evidence=evidence, policy={**POLICY, "schema": B.SCHEMA_V2}, numerical_policy=policy
        )


def test_parameter_role_changed_after_derivation_refuses_replay(tmp_path):
    from merlin_experiments.phase0.evidence import _materialize_evidence

    entries, products, _ = derive()
    _materialize_evidence(tmp_path, products)
    entry = copy.deepcopy(entries[0])
    row = entry["public_boundary"]["program"]["inputs"][1]
    row.pop("source_parameter")
    with pytest.raises(ValueError, match="original connected"):
        B.selected_loader(entry, tmp_path)
