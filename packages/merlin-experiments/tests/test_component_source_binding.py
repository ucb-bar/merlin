"""Backend-free source generation still retains every missing hardware role."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import component_coverage as C
from merlin_experiments.phase0 import generation
from merlin_experiments.phase0.component_execution_budget import source_for_entry
from merlin_experiments.phase0.component_source_binding import derive, verify_prepared_sources
from merlin_experiments.phase0.component_source_binding import screen_written as source_screen
from merlin_experiments.phase0.evidence import select_evidence
from merlin_experiments.phase0.software_intake import issue_independent_software_intake
from merlin_experiments.phase0.sweeps import _resolve_flat_extents, resolve_extent

from merlin.targetgen import golden_store, target_registry
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves


def _fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixture_module("test_component_automatic")
automatic = fixtures.automatic
independent = fixtures.independent
selected = fixtures.selected


def test_live_source_only_generation_does_not_require_or_discover_backend(automatic, monkeypatch):
    options = automatic
    options.pop("capability_contract")
    hardware, software = options["hardware_intake"], options["software_intake"]
    monkeypatch.setattr(target_registry, "resolve", lambda *_: pytest.fail("source generation discovered backend"))
    monkeypatch.setattr(
        generation, "_ensure_contract_on_path", lambda *_: pytest.fail("source generation imported backend")
    )
    with pytest.raises(ValueError, match="requires an explicit same-target backend"):
        select_evidence(
            "fixture",
            descriptor=options["descriptor"],
            facts_path=options["rtl_facts"],
            software_spec=options["software_spec"],
            hardware_intake=hardware,
            software_intake=software,
        )
    evidence = select_evidence(
        "fixture",
        descriptor=options["descriptor"],
        facts_path=options["rtl_facts"],
        software_spec=options["software_spec"],
        hardware_intake=hardware,
        software_intake=software,
        source_components=True,
    )
    assert evidence.contract == {} and not evidence.isa_taxonomy.get("by_class")
    identity = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"]["hardware"] = identity
    recipe["component_performance"]["objectives"] = []
    options["recipe"].write_text(yaml.safe_dump(recipe))
    template = yaml.safe_load(options["performance_template"].read_bytes())
    template["sweeps"] = []
    template["families"] = []
    options["performance_template"].write_text(yaml.safe_dump(template))
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["hardware"] = identity
    options["component_coverage"].write_text(yaml.safe_dump(policy))
    binding = derive(software, hardware=hardware, datapath=evidence.datapath)
    assert binding.tile_dim is None and binding.classes_for(op="matmul") == [] and binding.tiers == ["L0"]
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **options)
    report = fixtures.report(options)
    A.verify(report["automatic_derivation"], report=report)
    assert report["status"] == "source_prepared_incomplete"
    assert report["generation_identity"]["source_semantics_admission"]["software_intake_sha256"] == software.sha256
    assert {row["selector"] for row in report["automatic_derivation"]["required_unknowns"]} >= {
        "rtl_boundary_axis_mapping",
        "original_operator_effects",
    }
    count = 0
    for obligation in report["obligations"]:
        for member in obligation["members"]:
            assert member["state"] == "source_generated"
            directory = options["output_root"] / member["member"]
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            assert capsule["expected"]["instruction_classes"] == []
            assert capsule["source_semantics_screen"] == source_screen(
                capsule, directory, software=software, hardware=hardware
            )
            if any(node["op"] == "matmul" for node in capsule["component_program"]["nodes"]):
                assert capsule["software_screen"]["status"] == "unknown"
            leaves = materialize_capsule_leaves(capsule)
            values = {name: list(tensor.data) for name, tensor in leaves.items()}
            program = capsule["component_program"]
            types = {row["name"]: row for row in program["inputs"]}
            for node in program["nodes"]:
                inputs = node["actual_inputs"]
                if node["op"] in {"copy", "alias"}:
                    values[node["name"]] = list(values[inputs[0]])
                else:
                    assert node["op"] == "matmul"
                    m, k = types[inputs[0]]["shape"]
                    n = types[inputs[1]]["shape"][1]
                    values[node["name"]] = [
                        sum(values[inputs[0]][i * k + p] * values[inputs[1]][p * n + j] for p in range(k))
                        for i in range(m)
                        for j in range(n)
                    ]
                types[node["name"]] = node
            expected = {}
            for output in program["outputs"]:
                m, n = output["shape"]
                data = values[output["actual_value"]]
                expected[output["name"]] = [data[i * n : (i + 1) * n] for i in range(m)]
            assert golden_store.load_golden(directory)["outputs"] == expected
            count += 1
    assert count > 0
    for key in ("hardware_intake_sha256", "software_intake_sha256"):
        inconsistent = copy.deepcopy(report)
        inconsistent[key] = "0" * 64
        inconsistent.pop("sha256")
        inconsistent["sha256"] = C.digest(inconsistent)
        with pytest.raises(ValueError, match="exact live independent source mode"):
            verify_prepared_sources(options["output_root"], inconsistent, software=software, hardware=hardware)
    missing = copy.deepcopy(report)
    missing["obligations"].pop()
    missing.pop("sha256")
    missing["sha256"] = C.digest(missing)
    with pytest.raises(ValueError, match="original required obligation"):
        verify_prepared_sources(options["output_root"], missing, software=software, hardware=hardware)
    # Even an all-source-ready aggregate is a different preparation scope.
    # It cannot establish the original hardware admission or a Phase 2 guard.
    only_sources = copy.deepcopy(report)
    only_sources["obligations"] = [row for row in only_sources["obligations"] if row["state"] == "source_generated"]
    prepared = C.finalize(
        only_sources,
        root=options["output_root"],
        failures=[],
        generation_identity=report["generation_identity"],
        written=[
            options["output_root"] / member["member"]
            for row in only_sources["obligations"]
            for member in row["members"]
        ],
    )
    assert prepared["status"] == "source_prepared"
    assert C.build_guard_link(prepared)["status"] == "not_established"
    assert C.build_guard_link(prepared)["guards"] == []
    re_signed = copy.deepcopy(prepared)
    re_signed["status"] = "complete"
    re_signed.pop("sha256")
    re_signed["sha256"] = C.digest(re_signed)
    with pytest.raises(ValueError, match="source-only preparation is not concrete"):
        C.verify_report(options["output_root"], re_signed)
    changed = copy.deepcopy(capsule)
    changed["operation"]["attributes"]["program"]["outputs"].pop()
    with pytest.raises(ValueError, match="complete original typed DAG"):
        source_screen(changed, directory, software=software, hardware=hardware)
    path = directory / "capsule.interface.mlir"
    path.write_text(path.read_text().replace("func.return", "func.call"))
    with pytest.raises(ValueError, match="complete original typed DAG"):
        source_screen(capsule, directory, software=software, hardware=hardware)
    with pytest.raises(ValueError, match="source-only preparation is not concrete"):
        C.verify_report(options["output_root"])


def test_source_only_rejects_exported_backend_evidence_before_access(automatic):
    options = {**automatic, "evidence_input": Path("nonexistent-old-evidence"), "capability_contract": None}
    with pytest.raises(ValueError, match="live source replay, not exported backend evidence"):
        generation.generate_target("fixture", **options)


def test_minimal_independent_descriptor_does_not_discover_legacy_corpus_siblings(tmp_path):
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    selected = load_target_experiment(descriptor)
    assert selected.capsule_corpus is None
    generation._require_distinct_corpus_destinations(selected, output_root=tmp_path / "fresh", evidence_root=None)


@pytest.mark.parametrize("choice", ["readout", "tier", "numeric_policy"])
def test_source_binding_refuses_unimplemented_numerics_and_target_claims(automatic, selected, tmp_path, choice):
    software, hardware = automatic["software_intake"], automatic["hardware_intake"]
    numeric = software.public_facts()["numerical_semantics"]
    if choice == "readout":
        spec = yaml.safe_load(Path(software.source.path).read_bytes())
        spec["numerical_semantics"]["readout_dtype"] = "i16"
        path = fixtures.write(tmp_path / "distinct-readout.json", spec)
        review = yaml.safe_load((tmp_path / "protected-review.json").read_bytes())
        review["source"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        review["numerical_choices"] = spec["numerical_semantics"]
        software = issue_independent_software_intake(
            hardware=hardware,
            source=path,
            review=fixtures.write(tmp_path / "distinct-review.json", review),
            forbidden_roots=selected["forbidden_roots"],
            output_root=tmp_path / "distinct-issued",
        )
        numeric = software.public_facts()["numerical_semantics"]
    datapath = {
        "operand_dtype": numeric["operand_dtype"],
        "accum_dtype": numeric["accumulator_dtype"],
        "subnormal_operand_flush": numeric["subnormal_operand_flush"],
        "numerical_semantics": numeric,
    }
    if choice == "tier":
        datapath["required_oracle_tiers"] = ["L0", "L3"]
    elif choice == "numeric_policy":
        datapath["requant_shift"] = 2
    with pytest.raises(ValueError, match="distinct readout|physical or simulator|undeclared target routing"):
        derive(software, hardware=hardware, datapath=datapath)


def test_source_binding_does_not_invent_geometry_or_accept_unissued_semantics():
    assert resolve_extent(3, None) == 3
    with pytest.raises(ValueError, match="bool, not an extent"):
        resolve_extent(True, None)
    with pytest.raises(ValueError, match="independently selected tile"):
        resolve_extent("tile+1", None)
    from merlin.targetgen.corpus_spec import CorpusBinding

    binding = CorpusBinding("fixture", None, "int8", "i32", True, ["L0"], "exact_int")
    assert _resolve_flat_extents({"M": 3}, binding) == {"M": 3}
    with pytest.raises(ValueError, match="bool, not an extent"):
        _resolve_flat_extents({"M": True}, binding)
    with pytest.raises(ValueError, match="independently selected tile"):
        _resolve_flat_extents({"M": "tile+1"}, binding)
    with pytest.raises(ValueError, match="explicit tensor DAGs"):
        source_for_entry({"op": "movement", "M": 3, "N": 2}, binding=binding)
    with pytest.raises(ValueError, match="live independent"):
        derive({}, hardware=None, datapath={})
    with pytest.raises(ValueError, match="live independent"):
        select_evidence("fixture", source_components=True)
