"""One actual ordinary source derivation retains arithmetic and schema gaps."""

import copy
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import arithmetic_intake as R
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves


def _fixtures(name):
    source = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("private_composition_" + name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _fixtures("test_component_hw_arithmetic")
S = _fixtures("test_component_operator_schemas")
F = H.automatic_fixtures
automatic, independent, selected, tools = H.automatic, H.independent, H.selected, H.tools
native_sources, effect_generation = S.native_sources, S.effect_generation


@pytest.fixture
def combined(native_sources, monkeypatch, request, tools, selected, tmp_path):
    documents, _, _ = native_sources
    # The actual same original capture goes through the ordinary minimal-SW
    # review fixture before either intake is issued. No saved alias row grants
    # a source relation, and the local hardware observation stays scalar only.
    monkeypatch.setattr(F, "example", lambda **kwargs: copy.deepcopy(documents["alias"]["trace"]))
    options = request.getfixturevalue("effect_generation")
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"]["objectives"] = []
    F.write(options["recipe"], recipe)
    intake = R.issue_independent_arithmetic_intake(
        hardware=options["hardware_intake"],
        circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "combined-arithmetic",
    )
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy.update(schema=A.ARITHMETIC_POLICY_SCHEMA, arithmetic_intake_sha256=intake.sha256)
    F.write(options["component_coverage"], policy)
    options["arithmetic_intake"] = intake
    return options


def _all_original_outputs(root):
    capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
    inputs = materialize_capsule_leaves(capsule)
    values = {name: (tensor.shape, list(tensor.data)) for name, tensor in inputs.items()}
    program = capsule["component_program"]
    for node in program["nodes"]:
        if node["op"] == "matmul":
            (lhs_shape, lhs), (rhs_shape, rhs) = [values[name] for name in node["actual_inputs"]]
            m, k, n = lhs_shape[0], lhs_shape[1], rhs_shape[1]
            data = [sum(lhs[i * k + p] * rhs[p * n + j] for p in range(k)) for i in range(m) for j in range(n)]
            values[node["name"]] = ((m, n), data)
        elif node["op"] in {"copy", "alias"}:
            shape, data = values[node["actual_inputs"][0]]
            values[node["name"]] = (shape, list(data))
        else:
            raise AssertionError("unexpected original operation in complete independent output control")
    outputs = {}
    for output in program["outputs"]:
        (m, n), data = values[output["actual_value"]]
        outputs[output["name"]] = [data[i * n : (i + 1) * n] for i in range(m)]
    assert golden_store.load_golden(root)["outputs"] == outputs
    return capsule, outputs


def _resign(report):
    record = report["automatic_derivation"]
    record["sha256"] = A.digest({key: value for key, value in record.items() if key != "sha256"})
    report["generation_identity"]["automatic_derivation_sha256"] = A.digest(record)


@pytest.mark.parametrize("logical", [False, True])
def test_real_combined_generation_retains_both_sources_all_outputs_and_unknowns(combined, logical):
    if logical:
        policy = yaml.safe_load(combined["component_coverage"].read_bytes())
        policy["schema"] = A.LOGICAL_POLICY_SCHEMA
        F.write(combined["component_coverage"], policy)
    report = F.run(combined)
    record = report["automatic_derivation"]
    A.verify(record, report=report)
    assert report["status"] == "incomplete"
    assert record["schema"] == (A.LOGICAL_RECEIPT_SCHEMA if logical else A.ARITHMETIC_RECEIPT_SCHEMA)
    assert record["arithmetic_intake"] == combined["arithmetic_intake"].record()
    assert record["operator_schema_intake"] == combined["operator_schema_intake"].record()
    assert record["operator_effect_semantics"][0]["effect_classes"] == ["may_alias_result"]
    missing = record["required_unknowns"]
    numeric = next(row for row in missing if row["kind"] == "numeric_datapath")
    assert numeric["requirements"]["modular_result_bits"] == 20
    assert numeric["requirements"]["signed_addend_input_bits"] == 32
    assert numeric["requirements"]["compatible_reviewed_owners"] == ["contraction"]
    selectors = {(row["kind"], row["selector"]) for row in missing}
    assert ("physical_effect", "may_alias_result") in selectors
    assert ("effect_domain", "original_operator_effects") in selectors
    assert ("resource_role", "rtl_boundary_axis_mapping") in selectors
    assert any(row["kind"] == "numeric_domain" for row in missing)
    for wanted in missing:
        row = next(row for row in report["obligations"] if row["id"] == wanted["id"])
        assert row["mandatory"] is True and row["state"] == "unavailable" and row["members"] == []
    cohorts, sources, output_rosters = set(), set(), set()
    for row in report["obligations"]:
        for member in row["members"]:
            root = combined["output_root"] / member["member"]
            capsule, outputs = _all_original_outputs(root)
            sources.add((root / capsule["linalg_mlir"]).read_text())
            output_rosters.add(tuple(sorted(outputs)))
            if row["id"].startswith("auto_may_alias_result_"):
                cohorts.add(row["cohort"])
                assert set(outputs) == {"Yinput", "Yview", "Ycopy"}
                assert "return %A, %A," in (root / capsule["linalg_mlir"]).read_text()
    assert cohorts == {"functional_guard", "withheld_transfer"}
    assert len(sources) > 1 and len(output_rosters) > 1


@pytest.mark.parametrize("logical", [False, True])
def test_combined_generation_refuses_missing_substituted_and_unselected_bindings(combined, tmp_path, logical):
    if logical:
        policy = yaml.safe_load(combined["component_coverage"].read_bytes())
        policy["schema"] = A.LOGICAL_POLICY_SCHEMA
        F.write(combined["component_coverage"], policy)
    original_policy = combined["component_coverage"].read_bytes()
    attempts = 0

    def run(options):
        nonlocal attempts
        attempts += 1
        options = {**options, "output_root": tmp_path / ("refused-attempt-" + str(attempts))}
        F.generation.generate_target("fixture", **options)

    for key, message in (
        ("operator_schema_intake", "identical live independent schema"),
        ("arithmetic_intake", "identical live"),
    ):
        options = {name: value for name, value in combined.items() if name != key}
        with pytest.raises(ValueError, match=message):
            run(options)
        forged = dict(combined)
        forged[key] = dataclasses.replace(combined[key])
        with pytest.raises(RtlIntakeRefusal, match="live independent|live independently issued"):
            run(forged)
        changed = yaml.safe_load(original_policy)
        changed[key + "_sha256"] = "f" * 64
        F.write(combined["component_coverage"], changed)
        with pytest.raises(ValueError, match="differs from protected"):
            run(combined)
        combined["component_coverage"].write_bytes(original_policy)
    changed = yaml.safe_load(original_policy)
    changed.pop("operator_schema_intake_sha256")
    F.write(combined["component_coverage"], changed)
    with pytest.raises(ValueError, match="explicit versioned automatic policy"):
        run(combined)


def test_combined_receipt_cannot_drop_or_replace_original_facets_by_resigning(combined):
    report = F.run(combined)
    for keys, message in (
        (("operator_schema_intake",), "lost its selected original operator effect"),
        (("operator_effect_semantics",), "lost its selected original operator effect"),
        (("operator_schema_intake", "operator_effect_semantics"), "lost its selected original operator effect"),
        (("arithmetic_intake",), "lost its selected original arithmetic"),
    ):
        changed = copy.deepcopy(report)
        for key in keys:
            changed["automatic_derivation"].pop(key)
        _resign(changed)
        with pytest.raises(ValueError, match=message):
            A.verify(changed["automatic_derivation"], report=changed)
    for key in ("operator_schema_intake", "arithmetic_intake"):
        changed = copy.deepcopy(report)
        field = "software_intake_sha256" if key == "operator_schema_intake" else "hardware_intake_sha256"
        changed["automatic_derivation"][key][field] = "f" * 64
        _resign(changed)
        with pytest.raises((RtlIntakeRefusal, ValueError), match="correspondence changed|protected original hardware"):
            A.verify(changed["automatic_derivation"], report=changed)
    changed = copy.deepcopy(report)
    changed["automatic_derivation"]["operator_effect_semantics"][0]["effect_classes"] = []
    _resign(changed)
    with pytest.raises(ValueError, match="native source/argument/result replay"):
        A.verify(changed["automatic_derivation"], report=changed)
    for kind in ("numeric_datapath", "physical_effect"):
        changed = copy.deepcopy(report)
        rows = changed["automatic_derivation"]["required_unknowns"]
        rows.remove(next(row for row in rows if row["kind"] == kind))
        _resign(changed)
        with pytest.raises(ValueError, match="source factory"):
            A.verify(changed["automatic_derivation"], report=changed)
