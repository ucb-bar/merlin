"""Source development cases prepare complete references without timing grants."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_numerics, generation, sweeps
from merlin_experiments.phase0 import component_source_performance as P
from merlin_experiments.phase0.component_coverage import build_guard_link
from merlin_experiments.phase0.component_generation import digest

from merlin.targetgen import golden_store


def _fixtures():
    path = Path(__file__).with_name("test_component_source_binding.py")
    spec = importlib.util.spec_from_file_location("source_performance_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()
automatic = fixtures.automatic
independent = fixtures.independent
selected = fixtures.selected


def options_for(automatic, tmp_path):
    options = fixtures._source_options(automatic, tmp_path, version=fixtures.A.LOGICAL_POLICY_SCHEMA)
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"]["objectives"] = [
        {
            "family": "copy_publication",
            "operations": ["movement"],
            "objective": {
                "metric": "cycles",
                "unit": "cycles",
                "direction": "min",
                "basis": "explicit independent development source family",
            },
        }
    ]
    fixtures.fixtures.write(options["recipe"], recipe)
    performance = {
        "level": "source",
        "family": "copy_publication",
        "lever": "ordinary source scheduling",
        "member_class": "OBJECTIVE",
        "claim": "DIFFERENTIAL",
        "comparand": {
            "kind": "candidate",
            "against": "same complete original source",
            "cancels": "original inputs",
            "demand_equal": "all original outputs",
        },
        "falsifier": {
            "observation": "matched complete cost",
            "fires_when": "original gate fails",
            "negative_control": "unchanged compiler",
        },
        "gate": {
            "traits": ["managed_scratchpad"],
            "instrument": "explicit future counter",
            "capacity": "independent resource mapping missing",
            "on_missing": "skip_with_evidence",
        },
        "regime": {"separation": "independent literal source sizes", "layout": "original tensor semantics"},
        "emitter": {"status": "existing", "entry": "component_program", "knobs": {}},
        "cost": {"tier": "unqualified", "runs": 1, "projected_cycles": "unbounded", "basis": "unmeasured"},
        "acceptance": {"evidence": {"timing_simulator": "$target_oracle:L3"}},
    }
    program = {
        "inputs": [{"name": "X", "role": "input", "shape": [{"axis": "R"}, {"axis": "C"}], "dtype": "operand"}],
        "nodes": [
            {"name": "view", "op": "alias", "inputs": ["X"]},
            {"name": "first", "op": "copy", "inputs": ["view"]},
            {"name": "second", "op": "copy", "inputs": ["first"]},
        ],
        "outputs": [{"name": "snapshot", "value": "first"}, {"name": "final", "value": "second"}],
    }
    template = {
        "sweeps": [
            {
                "id": "copy_publication",
                "name": "copy_{R}_{C}",
                "axes": {"R": [2, 3], "C": [2, 5]},
                "fit_axes": ["R", "C"],
                "base": {
                    "kind": "isa",
                    "cat": "_perf",
                    "label": "dev",
                    "op": "component_program",
                    "program": program,
                    "performance": performance,
                },
            }
        ]
    }
    fixtures.fixtures.write(options["performance_template"], template)
    options["source_preparation"] = P.SCHEMA
    return options


def run(options):
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **options)
    coverage = fixtures.fixtures.report(options)
    path = options["output_root"] / "_evidence/coverage/source-performance-contracts.json"
    record = json.loads(path.read_bytes())
    inputs = dict(
        root=options["output_root"],
        coverage=coverage,
        hardware=options["hardware_intake"],
        software=options["software_intake"],
    )
    return record, inputs


def test_nonempty_development_full_roster_references_and_pending_guard_contract(automatic, tmp_path):
    options = options_for(automatic, tmp_path)
    record, inputs = run(options)
    assert record["source_checked_counts"]["development"] == 4
    assert all(record["source_checked_counts"][cohort] > 0 for cohort in ("functional_guard", "withheld_transfer"))
    assert record["original_required_ids"] == [row["id"] for row in inputs["coverage"]["obligations"]]
    assert record["mandatory_missing_ids"] and record["missing_producers"]
    assert record["hardware_guard_link"] == "not_established" and record["release_authority"] == "not_issued"
    assert build_guard_link(inputs["coverage"])["guards"] == []
    for row in record["requested_members"]:
        assert row["candidate_verdict"] == "not_evaluated"
        if row["cohort"] == "development":
            assert [node["source_operation"] for node in row["typed_node_owners"]] == ["alias", "copy", "copy"]
            assert row["original"]["complete_output_roster"] == ["final", "snapshot"]
            capsule = yaml.safe_load((inputs["root"] / row["member"] / "capsule.yaml").read_bytes())
            assert capsule["performance"]["source_preparation"]["hardware_admission"] == "not_established"
            assert capsule["performance"]["acceptance"]["evidence"]["timing_simulator"] == "$target_oracle:L3"
    assert P.verify_source_contracts(record, **inputs) == record


def test_altered_last_development_output_refuses_even_after_resigning_products(automatic, tmp_path):
    record, inputs = run(options_for(automatic, tmp_path))
    member = next(row for row in record["requested_members"] if row["cohort"] == "development")
    directory = inputs["root"] / member["member"]
    golden = golden_store.load_golden(directory)
    golden["outputs"]["snapshot"][-1][-1] += 1
    golden_store.write_golden(directory, golden)
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    with pytest.raises(ValueError, match="complete original independent reference"):
        P.prepare_source_contracts(**inputs)


def test_whole_roster_budget_denial_precedes_any_reference_evaluation(automatic, tmp_path, monkeypatch):
    options = options_for(automatic, tmp_path)
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["execution_budget"]["max_total_reference_work"] = 1
    fixtures.fixtures.write(options["component_coverage"], policy)
    monkeypatch.setattr(
        component_numerics, "evaluate", lambda *_: pytest.fail("denied aggregate allocated a reference")
    )
    monkeypatch.setattr(
        generation, "_write_capsule", lambda *_a, **_k: pytest.fail("denied aggregate reached shaped builder")
    )
    record, inputs = run(options)
    assert all(row["state"] == "unavailable" for row in record["requested_members"])
    assert record["source_checked_counts"]["development"] == 0
    assert len(record["requested_members"]) == len(inputs["coverage"]["execution_admission"]["decisions"])


def test_source_preparation_and_full_replay_never_discover_backend_encodings(automatic, tmp_path, monkeypatch):
    options = options_for(automatic, tmp_path)
    monkeypatch.setattr(
        sweeps, "target_encodings", lambda *_a, **_k: pytest.fail("source preparation discovered backend")
    )
    record, inputs = run(options)
    assert record["source_checked_counts"]["development"] == 4
    assert P.verify_source_contracts(record, **inputs) == record


def test_source_sweeps_without_explicit_gate_facts_refuse_before_discovery(automatic, tmp_path, monkeypatch):
    options = options_for(automatic, tmp_path)
    template = yaml.safe_load(options["performance_template"].read_bytes())
    binding = P._binding(options["software_intake"], options["hardware_intake"])
    monkeypatch.setattr(sweeps, "_performance_facts", lambda *_a, **_k: pytest.fail("source sweep discovered facts"))
    with pytest.raises(ValueError, match="explicit pending gate facts"):
        sweeps.expand_sweeps(template, binding, source_preparation=P.SCHEMA)


def test_recipe_authored_members_refuse_before_allocation(automatic, tmp_path, monkeypatch):
    options = options_for(automatic, tmp_path)
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["capsules"] = [{"name": "authored", "cat": "_perf", "op": "component_program"}]
    fixtures.fixtures.write(options["recipe"], recipe)
    monkeypatch.setattr(generation, "_write_capsule", lambda *_a, **_k: pytest.fail("authored recipe reached builder"))
    with pytest.raises(ValueError, match="shared independent template"):
        generation.generate_target("fixture", **options)


def test_development_work_consumes_same_aggregate_budget_as_guards(automatic, tmp_path):
    from merlin_experiments.phase0.component_execution_budget import measure

    from merlin.targetgen import component_program

    options = options_for(automatic, tmp_path)
    template = yaml.safe_load(options["performance_template"].read_bytes())
    sweep = template["sweeps"][0]
    complete_development_work = 0
    for rows in sweep["axes"]["R"]:
        for columns in sweep["axes"]["C"]:
            program = json.loads(json.dumps(sweep["base"]["program"]))
            program["inputs"][0]["shape"] = [rows, columns]
            typed = component_program.analyze(program, operand_dtype="i8", accumulator_dtype="i32")
            complete_development_work += measure(
                {"kind": "component_program", "program": typed, "input_palette": None, "stimulus_range": None}
            )["reference_work"]
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["execution_budget"]["max_total_reference_work"] = complete_development_work
    fixtures.fixtures.write(options["component_coverage"], policy)
    record, inputs = run(options)
    assert record["source_checked_counts"]["development"] == 4
    assert record["source_checked_counts"]["functional_guard"] == 0
    assert record["source_checked_counts"]["withheld_transfer"] == 0
    decisions = inputs["coverage"]["execution_admission"]["decisions"]
    assert inputs["coverage"]["execution_admission"]["totals"]["reference_work"] == complete_development_work
    for row in decisions:
        if row["requested_member"].startswith("_perf/"):
            assert row["state"] == "admitted"
        else:
            assert row["state"] == "unavailable" and "total_reference_work" in row["reason"]


def test_unsupported_tile_axis_retains_missing_family(automatic, tmp_path):
    options = options_for(automatic, tmp_path)
    template = yaml.safe_load(options["performance_template"].read_bytes())
    template["sweeps"][0]["axes"]["R"] = ["tile", "tile+1"]
    fixtures.fixtures.write(options["performance_template"], template)
    record, _ = run(options)
    assert record["source_checked_counts"]["development"] == 0
    assert record["missing_families"][0]["family"] == "copy_publication"
    assert "literal axes" in record["missing_families"][0]["reason"]


def test_wrong_node_owner_refuses_before_writer(automatic, tmp_path, monkeypatch):
    options = options_for(automatic, tmp_path)
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"]["objectives"][0]["operations"] = ["contraction"]
    fixtures.fixtures.write(options["recipe"], recipe)
    monkeypatch.setattr(generation, "_write_capsule", lambda *_a, **_k: pytest.fail("unowned source reached writer"))
    with pytest.raises(ValueError, match="every actual DAG node owner"):
        generation.generate_target("fixture", **options)


def test_absent_selection_preserves_legacy_tile_refusal(automatic, tmp_path):
    options = options_for(automatic, tmp_path)
    options.pop("source_preparation")
    with pytest.raises(ValueError, match="sweeps need a tile edge"):
        generation.generate_target("fixture", **options)


def test_source_preparation_refuses_foreign_inputs_before_source_access(tmp_path):
    with pytest.raises(ValueError, match="explicit v1 live backend-free"):
        generation.generate_target("fixture", output_root=tmp_path / "out", source_preparation=P.SCHEMA)


def test_missing_last_development_member_retains_exact_pending_roster(automatic, tmp_path):
    record, inputs = run(options_for(automatic, tmp_path))
    member = [row for row in record["requested_members"] if row["cohort"] == "development"][-1]
    (inputs["root"] / member["member"] / "capsule.yaml").unlink()
    replay = P.prepare_source_contracts(**inputs)
    assert [row["member"] for row in replay["requested_members"]] == [
        row["member"] for row in record["requested_members"]
    ]
    missing = next(row for row in replay["requested_members"] if row["member"] == member["member"])
    assert missing["state"] == "unavailable" and missing["candidate_verdict"] == "not_evaluated"
    with pytest.raises(ValueError, match="contracts changed"):
        P.verify_source_contracts(record, **inputs)


def test_resigned_source_status_cannot_turn_pending_hardware_guard_into_admission(automatic, tmp_path):
    record, inputs = run(options_for(automatic, tmp_path))
    record["hardware_guard_link"] = "established"
    record["release_authority"] = "issued"
    record["candidate_verdict"] = "accepted"
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    with pytest.raises(ValueError, match="contracts changed"):
        P.verify_source_contracts(record, **inputs)


def test_resigned_member_scope_refuses_before_independent_values(automatic, tmp_path, monkeypatch):
    record, inputs = run(options_for(automatic, tmp_path))
    member = next(row for row in record["requested_members"] if row["cohort"] == "development")
    path = inputs["root"] / member["member"] / "capsule.yaml"
    capsule = yaml.safe_load(path.read_bytes())
    capsule["performance"]["source_preparation"]["hardware_admission"] = "accepted"
    fixtures.fixtures.write(path, capsule)
    monkeypatch.setattr(component_numerics, "evaluate", lambda *_: pytest.fail("scope substitution reached reference"))
    with pytest.raises(ValueError, match="hardware/measurement scope"):
        P.prepare_source_contracts(**inputs)
