"""Record, bounded admission and derive resolve the same selected data."""

import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import evidence, spec_validation

from merlin.targetgen import composition_facts, spec_fact_drift
from merlin.targetgen import software_spec_resolution as resolution
from merlin.targetgen.core_aten_spec_int8 import select_spec
from merlin.targetgen.functional_truth_admission import select_admission


@pytest.mark.parametrize("width,bound", [(3, False), (7, True)])
def test_data_bound_record_and_derive_identity(monkeypatch, tmp_path, width, bound):
    target = "synthetic"
    contract = {"name": target, "runner": {"backend": "chipyard_rocc"}, "compute_units": []}
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract))
    facts = {"facts": {"isa_taxonomy": {"status": "derived", "by_class": {}}}}
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    descriptor = tmp_path / "descriptor.yaml"
    workload = {"fusion_validation": {"path": str(tmp_path / "validation.json"), "sha256": "a" * 64}} if bound else {}
    descriptor.write_text(yaml.safe_dump({"target": target, "workload_spec": workload}))
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "target": target,
                "inputs": {"target": str(descriptor)},
                "policy": {"prohibited_instruction_roles": ["synthetic_role"]},
            }
        )
    )
    spec = {
        "schema": "merlin.software_spec.v1",
        "target": target,
        "status": "reviewed",
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
            "readout": {"strategy": "exact_rational_host", "requant_rounding": "half_even"},
        },
        "operations": [{"id": "copy", "families": ["movement"], "placement": "host", "signature": {"ranks": [2]}}],
        "compositions": {
            "schema": "merlin.software_spec.compositions.v1",
            "status": "reviewed",
            "grading": "placement_and_instruction_audit",
            "chains": [
                {
                    "id": "carrier",
                    "producer": "contraction",
                    "readout_stages": "from_output_path",
                    "exits": ["accumulator"],
                    "constraints": ["accumulator_exit_applies_no_readout_stage"],
                }
            ],
        },
        "quantization": {
            "formats": [],
            "schemes": {
                "schema": "merlin.quantization_schemes.v1",
                "status": "reviewed",
                "axes": {
                    "scale_value": ["power_of_two"],
                    "weight_scale_granularity": ["tensor"],
                    "activation_scale_granularity": ["tensor"],
                    "readout_exit": ["accumulator_full"],
                    "operand_sum_rounding": ["single_rounding"],
                    "residual_coefficient": ["integer_constant"],
                },
                "constraints": [
                    {
                        "when": {"residual_coefficient": "integer_constant"},
                        "coefficient_digits": {"dtype": "from_facts", "max_k_extension": "from_facts"},
                    }
                ],
                "schemes": [
                    {
                        "id": "panel",
                        "scale_value": "power_of_two",
                        "weight_scale_granularity": "tensor",
                        "operand_sum_rounding": "single_rounding",
                        "weight_layout": {"kind": "packed_panel", "panel_width": "from_facts"},
                    }
                ],
            },
        },
    }
    spec_path = tmp_path / "software.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    host = tmp_path / "host.yaml"
    host.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "operations": [],
                "compiler": {"package_sha256": "b" * 64, "dtype_strategy": "int8"},
                "evidence": {},
            }
        )
    )
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setenv("MERLIN_EXPERIMENT_SPEC", str(experiment))
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract_path))
    monkeypatch.setattr(
        resolution,
        "_derived_output_epilogue",
        lambda *args: {
            "output_path": {
                "schema": "output_path_v1",
                "accumulator": {"dtype": "i32"},
                "accumulator_stages": [],
                "readout": [],
                "exits": {"accumulator": True, "requant": False},
            }
        },
    )
    monkeypatch.setattr(
        composition_facts,
        "derive",
        lambda **kwargs: {
            "facts": {
                "weight_panel_width": width,
                "operand_signed_bits": 8,
                "k_extension_within": {"max_k": width * 5},
            },
            "evidence": {
                key: "synthetic proof" for key in ("weight_panel_width", "operand_signed_bits", "k_extension_within")
            },
            "unknown": {},
        },
    )
    original = spec_fact_drift.fact_capabilities

    def selected_facts(**kwargs):
        assert kwargs["data_only"] is True
        assert tuple(kwargs["prohibited_roles"]) == ("synthetic_role",)
        assert kwargs["taxonomy"] == facts["facts"]["isa_taxonomy"]
        view = original(**kwargs)
        view["composition"]["producer_placement"]["contraction"] = spec_fact_drift.FactField(
            ("accelerator",),
            True,
            ("synthetic carrier observation",),
        )
        return view

    monkeypatch.setattr(spec_fact_drift, "fact_capabilities", selected_facts)
    # A descriptor object avoids loading unrelated host/corpus fixtures.
    te = SimpleNamespace(
        path=descriptor,
        isa_headers=[],
        source_root=tmp_path,
        selected_board_catalog=lambda: None,
        host_lanes=SimpleNamespace(
            profiles={
                "native": SimpleNamespace(
                    package="host-package",
                    resolve_capabilities=lambda **kwargs: (host, yaml.safe_load(host.read_text()), {}),
                ),
            }
        ),
    )
    derive = evidence.select_evidence(
        target,
        descriptor=te,
        software_spec=spec_path,
        capability_contract_path=contract_path,
        facts_path=facts_path,
        prohibited_roles=("synthetic_role",),
    )
    record = select_admission(target, rtl_facts=facts_path, software_spec=spec_path, host_capabilities=host)
    selected = select_spec(target, facts, spec_path, host)
    digest = spec_validation.selection_identity(
        resolved_spec=record["spec"],
        rtl_facts=facts,
        capability_contracts=[contract],
        host_capabilities=[record["host_capabilities"]],
    )
    assert (
        digest["resolved_software_spec_sha256"]
        == spec_validation.evidence_identity(derive)["resolved_software_spec_sha256"]
    )
    assert selected["spec"] == record["spec"] == derive.software_spec
    assert not derive.software_spec_derivation["unresolved"]
    assert selected["resolution"]["prohibited_instruction_roles"] == ["synthetic_role"]
    schemes = selected["spec"]["quantization_scheme_resolution"]
    assert schemes["schemes"][0]["weight_layout"]["panel_width"] == width
    assert schemes["constraints"][0]["coefficient_digits"] == {"dtype": "i8", "max_k_extension": width * 5}
    assert selected["spec"]["composition_resolution"]["status"] == ("validation_required" if bound else "resolved")


def test_context_without_descriptor_keeps_legacy_policy(monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    monkeypatch.setenv("SPEC", "/nonexistent/unselected-experiment.yaml")
    assert resolution.selected_context("synthetic") == ({}, ())


@pytest.mark.parametrize("field", ["target", "descriptor"])
def test_context_refuses_foreign_experiment(monkeypatch, tmp_path, field):
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "target": "foreign" if field == "target" else "synthetic",
                "inputs": {"target": str(tmp_path / "other.yaml") if field == "descriptor" else str(descriptor)},
            }
        )
    )
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setenv("MERLIN_EXPERIMENT_SPEC", str(experiment))
    with pytest.raises(ValueError, match="differs|different"):
        resolution.selected_context("synthetic")


@pytest.mark.parametrize("changed", ["descriptor", "experiment"])
def test_runner_stamp_binds_resolution_context(tmp_path, monkeypatch, changed):
    import importlib.util

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("resolution_stamp_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    paths = {name: tmp_path / name for name in ("descriptor", "experiment", "suite", "simulator")}
    for name, path in paths.items():
        path.write_text(name)
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(paths["descriptor"]))
    monkeypatch.setenv("MERLIN_EXPERIMENT_SPEC", str(paths["experiment"]))
    monkeypatch.setenv("MERLIN_SPIKE", str(paths["simulator"]))
    args = SimpleNamespace(
        repo=repo_root(),
        m2m_dir=repo_root(),
        suite=paths["suite"],
        pin=[],
        extension_library=None,
        facts=None,
        execution_provider=None,
        software_spec=None,
        host_capabilities=None,
        device_package=None,
        target=None,
        numeric_policy="spec_int8",
    )
    first = runner.stamp(args)
    paths[changed].write_text("changed selection")
    second = runner.stamp(args)
    assert first != second
    assert first["source_digest"] != second["source_digest"]
    assert str(paths[changed]) in second["sources"]
    assert str(repo_root() / "src/merlin/targetgen/software_spec_resolution.py") in second["sources"]
