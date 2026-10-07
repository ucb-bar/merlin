"""Run evidence remains usable after original sources disappear or change."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import evidence, sweeps
from merlin_experiments.phase0.provenance import _scrub_capsule_dir

from merlin.perf import profile
from merlin.runtime.backends import base
from merlin.targetgen import readout_facet, target_registry
from merlin.targetgen.rtl import facts, source_selection


def test_specir_source_inventory_excludes_unrelated_generated_project_files(tmp_path):
    package = tmp_path / "specir"
    package.mkdir()
    (package / "__init__.py").write_text("\n")
    (package / "oracle.py").write_text("VALUE = 1\n")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "generated.mlir").write_text("unrelated\n")
    selected = evidence._reference_inventory_root(
        "numerical_model", tmp_path, {"numerical_semantics": {"model": {"engine": "specir_fp_reduce"}}}
    )
    assert selected == package
    assert sorted(path.name for path in selected.rglob("*.py")) == ["__init__.py", "oracle.py"]
    (package / "__init__.py").unlink()
    with pytest.raises(ValueError, match="SpecIR source package is absent"):
        evidence._reference_inventory_root(
            "numerical_model", tmp_path, {"numerical_semantics": {"model": {"engine": "specir_fp_reduce"}}}
        )


def test_mx_reference_inventory_selects_only_loaded_file(tmp_path, monkeypatch):
    reference = tmp_path / "mlc" / "validate" / "mx_ref.py"
    reference.parent.mkdir(parents=True)
    reference.write_text("VALUE = 1\n")
    unrelated = tmp_path / "runs" / "generated.py"
    unrelated.parent.mkdir()
    unrelated.write_text("VALUE = 2\n")
    document = {"numerical_semantics": {"model": {"engine": "mx_block_reference"}}}
    assert evidence._reference_inventory_members("numerical_model", tmp_path, document, {".py"}, {"runs"}) == (
        reference,
    )
    monkeypatch.setenv("MERLIN_MLC_DIR", str(tmp_path))
    from merlin_experiments.phase0.numerics import _mx_ref

    from merlin.targetgen.mx_oracle import mx_reference

    assert _mx_ref().VALUE == 1
    assert mx_reference().VALUE == 1
    reference.unlink()
    with pytest.raises(ValueError, match="MX numerical reference is absent"):
        evidence._reference_inventory_members("numerical_model", tmp_path, document, {".py"}, set())


def _selection(monkeypatch, tmp_path, body=None):
    provider = tmp_path / "support"
    contract = provider / "contracts" / "target_contract.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text("name: fixture\ncompute_units: []\n")
    code = provider / "backend.py"
    code.write_text("# observed provider source\n")
    raw = tmp_path / "facts.json"
    raw.write_bytes(
        json.dumps({"facts": body or {"arrays": [{"rows": 4, "cols": 4}], "memories": []}}, indent=4).encode() + b"\n"
    )
    info = target_registry.TargetInfo(
        name="fixture",
        kind="external",
        base=provider,
        contract_path=contract,
        dialect_plan_path=provider / "contracts/dialect_plan.yaml",
        facts_path=raw,
        backend="fixture",
    )
    monkeypatch.setattr(target_registry, "resolve", lambda target: info)
    monkeypatch.setattr(facts, "find_facts", lambda target, explicit=None: raw)
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("selection must never extract"))
    monkeypatch.setattr(readout_facet, "capture_inputs", lambda *a, **k: {"scalar_abi": None, "readouts": None})
    monkeypatch.setattr(base, "execution_capability_facts", lambda target: {})
    return evidence.select_evidence("fixture", facts_path=raw), raw, code


def test_software_fact_derivation_binds_legacy_readers_to_selected_bytes(monkeypatch, tmp_path):
    _, raw, _ = _selection(monkeypatch, tmp_path)
    software = tmp_path / "software.yaml"
    software.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.software_spec.v1",
                "target": "fixture",
                "status": "reviewed",
                "numerical_semantics": {
                    "model": {"engine": "integer_reference"},
                    "operand_dtype": "int8",
                    "accumulator_dtype": "i32",
                    "readout_dtype": "i32",
                    "subnormal_operand_flush": False,
                    "overflow": "wrap_internal_mac",
                },
                "operations": {"contraction": {"placement": "accelerator", "dtypes": ["int8"]}},
            }
        )
    )
    from merlin.llvmlower.device_shim import tile_edge_for
    from merlin.targetgen import spec_fact_drift

    derive = spec_fact_drift.fact_capabilities
    observed = []

    def through_legacy_readers(**kwargs):
        # The real helper calls load_facts(target), not the supplied raw_facts.
        # _selection makes any attempted live extraction fail.
        assert tile_edge_for(kwargs["target"]) == 4
        assert facts.load_facts(kwargs["target"]) == kwargs["raw_facts"]
        assert target_registry.load_contract(kwargs["target"]) == kwargs["contract"]
        observed.append(kwargs["target"])
        return derive(**kwargs)

    monkeypatch.setattr(spec_fact_drift, "fact_capabilities", through_legacy_readers)
    selected = evidence.select_evidence("fixture", facts_path=raw, software_spec=software)
    assert observed == ["fixture"]
    assert selected.raw_facts == raw.read_bytes()
    # The selected-only context must not leak into the next caller.
    with pytest.raises(pytest.fail.Exception, match="selection must never extract"):
        tile_edge_for("fixture")


def test_phase0_accepts_symlink_alias_for_byte_bound_rtl_production(monkeypatch, tmp_path):
    _, facts_path, _ = _selection(monkeypatch, tmp_path)
    source_root = tmp_path / "rtl-source"
    source_root.mkdir()
    alias = tmp_path / "rtl-alias"
    alias.symlink_to(source_root, target_is_directory=True)
    core = source_root / "core.hw.mlir"
    core.write_text("module {}\n")
    generic = source_root / "core.generic.mlir"
    generic.write_text("module {}\n")
    tool = source_root / "circt-opt"
    tool.write_text("tool bytes\n")

    def member(path):
        return {"path": str(path), "sha256": source_selection.digest(path)}

    bundle = source_root / "source-selection.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": source_selection.SCHEMA,
                "target": "fixture",
                "sources": {role: member(alias / core.name) for role in ("core_hw", "soc_hw", "firrtl", "hierarchy")},
            }
        )
    )

    def production(selected):
        return {
            "status": "verified",
            "sources": [{"role": role, **data} for role, data in sorted(selected["sources"].items())],
        }

    monkeypatch.setattr(source_selection, "production_consistency", production)
    selected = source_selection.load_selection(bundle, target="fixture")
    recorded = production(selected)
    for source in recorded["sources"]:
        source["path"] = str(alias / core.name)
    genericization = {
        "kind": "circt_generic_serialization",
        "returncode": 0,
        "input": member(alias / core.name),
        "output": member(generic),
        "tool": member(tool),
        "command": [str(tool), "--mlir-print-op-generic", str(alias / core.name), "-o", str(generic)],
    }
    recorded["genericization"] = genericization
    recorded["sources"].append({"role": "core_hw_generic", **genericization["output"]})
    facts_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "target": "fixture",
                    "source_bundle_path": str(bundle),
                    "generic_hw_path": str(generic),
                    "generic_hw_sha256": source_selection.digest(generic),
                },
                "source_consistency": recorded,
                "facts": {"arrays": [{"rows": 4, "cols": 4}], "memories": []},
            }
        )
    )
    observed = evidence.select_evidence("fixture", facts_path=facts_path)
    assert not [row for row in observed.diagnostics if row["component"] == "source-consistency"]

    genericization["input"]["path"] = str(generic)
    facts_path.write_text(
        json.dumps(
            {
                "inputs": observed.loaded_facts["inputs"],
                "source_consistency": recorded,
                "facts": {"arrays": [{"rows": 4, "cols": 4}], "memories": []},
            }
        )
    )
    altered = evidence.select_evidence("fixture", facts_path=facts_path)
    assert any(
        row["component"] == "source-consistency" and row["status"] == "contradiction" for row in altered.diagnostics
    )

    genericization["input"].pop("path")
    facts_path.write_text(
        json.dumps(
            {
                "inputs": observed.loaded_facts["inputs"],
                "source_consistency": recorded,
                "facts": {"arrays": [{"rows": 4, "cols": 4}], "memories": []},
            }
        )
    )
    malformed = evidence.select_evidence("fixture", facts_path=facts_path)
    assert any(
        row["component"] == "source-consistency" and row["status"] == "contradiction" for row in malformed.diagnostics
    )


def test_phase0_snapshots_selected_elaboration_inputs(monkeypatch, tmp_path):
    _, facts_path, _ = _selection(monkeypatch, tmp_path)
    source_root = tmp_path / "selected"
    source_root.mkdir()
    config = source_root / "configs.py"
    config.write_text("class SelectedConfig: pass\n")
    tool = source_root / "elaborator"
    tool.write_text("selected tool bytes\n")
    runs = []
    for index in (1, 2):
        run = tmp_path / f"run{index}"
        run.mkdir()
        firrtl = run / "selected.fir"
        firrtl.write_text("circuit Top :\n  module Top :\n")
        (run / "stdout.log").write_text("")
        (run / "stderr.log").write_text("")
        runs.append(
            {
                "firrtl": str(firrtl),
                "firrtl_sha256": source_selection.digest(firrtl),
                "stdout_sha256": source_selection.digest(run / "stdout.log"),
                "stderr_sha256": source_selection.digest(run / "stderr.log"),
            }
        )
    receipt = tmp_path / "elaboration.json"
    receipt.write_text(
        json.dumps(
            {
                "source": {
                    "root": str(source_root),
                    "config_file": "configs.py",
                    "config_sha256": source_selection.digest(config),
                },
                "tool": {"path": str(tool), "sha256": source_selection.digest(tool)},
                "runs": runs,
            }
        )
    )
    core = tmp_path / "core.hw.mlir"
    core.write_text("module {}\n")
    bundle = tmp_path / "source-selection.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": source_selection.SCHEMA,
                "target": "fixture",
                "sources": {
                    role: {"path": str(core), "sha256": source_selection.digest(core)}
                    for role in ("core_hw", "soc_hw", "firrtl", "hierarchy")
                },
                "production": {"elaboration": {"path": str(receipt), "sha256": source_selection.digest(receipt)}},
            }
        )
    )
    monkeypatch.setattr(
        source_selection,
        "production_consistency",
        lambda selected: {
            "status": "verified",
            "sources": [{"role": role, **row} for role, row in selected["sources"].items()],
            "elaboration": {"status": "reproduced_exact_firrtl"},
        },
    )
    facts_path.write_text(
        json.dumps(
            {
                "inputs": {"target": "fixture", "source_bundle_path": str(bundle)},
                "facts": {"arrays": [{"rows": 4, "cols": 4}], "memories": []},
            }
        )
    )
    selected = evidence.select_evidence("fixture", facts_path=facts_path)
    observed = {source.role for source in selected.source_snapshots}
    assert {
        "rtl-elaboration-receipt",
        "rtl-elaboration-config",
        "rtl-elaboration-tool",
        "rtl-elaboration-output",
        "rtl-elaboration-stdout",
        "rtl-elaboration-stderr",
    } <= observed


def test_rtl_receipt_alias_comparison_refuses_absent_files(tmp_path):
    missing = {"path": str(tmp_path / "absent"), "sha256": "a" * 64}
    assert not evidence._same_selected_file(missing, missing)
    assert not evidence._same_source_consistency(
        {"sources": [{"role": "core_hw", **missing}]},
        {"sources": [{"role": "core_hw", **missing}]},
    )


def test_exact_raw_bytes_and_derived_hashes_are_distinct(monkeypatch, tmp_path):
    selected, raw, _ = _selection(monkeypatch, tmp_path)
    assert selected.raw_facts == raw.read_bytes()
    assert selected.raw_facts_sha256 == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert selected.performance_facts["target_profile_sha256"] != selected.raw_facts_sha256
    mutable = selected.loaded_facts
    mutable["facts"]["arrays"][0]["rows"] = 999
    assert selected.loaded_facts["facts"]["arrays"][0]["rows"] == 4


def test_explicit_capability_contract_does_not_require_executable_provider(monkeypatch, tmp_path):
    contract = tmp_path / "selected-contract.yaml"
    contract.write_text("name: fixture\ncompute_units: []\n")
    monkeypatch.setattr(target_registry, "resolve", lambda target: (_ for _ in ()).throw(KeyError(target)))
    monkeypatch.setattr(
        facts, "find_facts", lambda target, explicit=None: (_ for _ in ()).throw(FileNotFoundError(target))
    )
    selected = evidence.select_evidence("fixture", capability_contract_path=contract)
    assert selected.contract == {"name": "fixture", "compute_units": []}
    assert any(source.path == contract and source.role == "target-contract" for source in selected.source_snapshots)

    contract.write_text("name: other_target\ncompute_units: []\n")
    with pytest.raises(ValueError, match="differs from selected target"):
        evidence.select_evidence("fixture", capability_contract_path=contract)


def test_selected_readout_scale_conflict_is_a_bound_diagnostic(monkeypatch, tmp_path):
    body = {
        "arrays": [{"rows": 4, "cols": 4}],
        "memories": [],
        "interfaces": [
            {
                "name": "register_bundle_layouts",
                "bundles": {"StoreConfig": {"fields": {"acc_scale": {"width": 32}}}},
                "unresolved": {},
            }
        ],
    }
    _, raw, code = _selection(monkeypatch, tmp_path, body=body)
    contract = code.parent / "contracts/target_contract.yaml"
    contract.write_text(
        "name: fixture\ncompute_units:\n"
        "- {name: unit, kind: systolic, dtypes: [int8], ops: [matmul], scaling: per_channel}\n"
    )
    selected = evidence.select_evidence("fixture", facts_path=raw)
    (finding,) = [row for row in selected.diagnostics if row["component"] == "readout-scaling"]
    assert finding["status"] == "contradiction"
    assert finding["finding"]["kind"] == "scaling_exceeds_readout"
    assert finding["finding"]["derived"] == ["tensor"]
    assert finding["raw_facts_sha256"] == selected.raw_facts_sha256
    assert finding["contract_sha256"] == evidence._canonical_digest(selected.contract)
    assert finding["readout_facet_sha256"] == evidence._canonical_digest(selected.readout_facets[0])

    contract.write_text(
        "name: fixture\ncompute_units:\n"
        "- {name: unit, kind: systolic, dtypes: [int8], ops: [matmul], scaling: per_tensor}\n"
    )
    corrected = evidence.select_evidence("fixture", facts_path=raw)
    assert not [row for row in corrected.diagnostics if row["component"] == "readout-scaling"]
    assert corrected.raw_facts_sha256 == selected.raw_facts_sha256
    assert corrected.contract != selected.contract


def test_derivation_identity_binds_provider_bytes_but_not_checkout_location(monkeypatch, tmp_path):
    selected, _, _ = _selection(monkeypatch, tmp_path)
    relocated = evidence.EvidenceSelection(
        selected.target,
        tuple(
            evidence.EvidenceSource(
                tmp_path / "elsewhere" / source.path.relative_to(tmp_path / "support"),
                source.role,
                source.content,
            )
            if source.role == "support-source"
            else source
            for source in selected.source_snapshots
        ),
        selected.views_json,
        selected.raw_facts,
    )
    assert relocated.derivation_identity == selected.derivation_identity

    changed = evidence.EvidenceSelection(
        relocated.target,
        tuple(
            evidence.EvidenceSource(source.path, source.role, source.content + b"# changed\n")
            if source.role == "support-source" and source.path.name == "backend.py"
            else source
            for source in relocated.source_snapshots
        ),
        relocated.views_json,
        relocated.raw_facts,
    )
    assert (
        changed.derivation_identity["support_sources_sha256"] != selected.derivation_identity["support_sources_sha256"]
    )


def test_export_and_reload_never_reopen_original_inputs(monkeypatch, tmp_path):
    selected, raw, code = _selection(monkeypatch, tmp_path)
    explicit = tmp_path / "selected-contract.yaml"
    explicit.write_text("name: fixture\ncompute_units: []\nencoding:\n  corpus_issue_order: [LOAD, COMPUTE, STORE]\n")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(explicit))
    selected = evidence.select_evidence("fixture", facts_path=raw)
    assert selected.contract["encoding"]["corpus_issue_order"] == ["LOAD", "COMPUTE", "STORE"]
    assert any(source.path == code for source in selected.source_snapshots)
    output = tmp_path / "run"
    manifest = evidence.export_evidence(selected, output)
    raw.unlink()
    explicit.unlink()
    code.write_text("# changed live provider\n")
    monkeypatch.setattr(evidence, "select_evidence", lambda *a, **k: pytest.fail("ambient selection ran"))
    restored = evidence.load_exported_evidence(output)
    assert restored.views_json == selected.views_json
    assert restored.raw_facts == selected.raw_facts
    assert restored.source_snapshots == selected.source_snapshots
    assert manifest["raw_facts_sha256"] == selected.raw_facts_sha256
    assert (output / "hardware/circt/facts.json").read_bytes() == selected.raw_facts
    assert evidence.export_evidence(restored, output) == manifest


def test_selected_instruction_semantics_are_frozen_with_the_phase0_inputs(monkeypatch, tmp_path):
    selected, raw, code = _selection(monkeypatch, tmp_path)
    from merlin.targetgen import instruction_semantics

    contract = code.parent / "contracts/target_contract.yaml"
    contract.write_text("name: fixture\ncompute_units: []\ninstruction_semantics: contracts/instructions.yaml\n")
    authored = code.parent / "contracts/instructions.yaml"
    authored.write_bytes(b"schema: merlin.instruction_semantics.v1\ntarget: fixture\ninstructions: []\n")
    calls = []

    def normalize(document, *, software_spec, rtl_facts, target, source_bytes, software_source_bytes, rtl_source_bytes):
        calls.append(
            (document, software_spec, rtl_facts, target, source_bytes, software_source_bytes, rtl_source_bytes)
        )
        return {
            "schema": "merlin.instruction_semantics.v1",
            "target": target,
            "status": "UNKNOWN",
            "instructions": [],
            "unknowns": [{"reason": "no reviewed instruction semantics"}],
        }

    monkeypatch.setattr(instruction_semantics, "normalize_instruction_semantics", normalize)
    selected = evidence.select_evidence("fixture", facts_path=raw)
    assert calls[0][3] == "fixture"
    assert calls[0][4] == authored.read_bytes()
    assert calls[0][6] == raw.read_bytes()
    output = tmp_path / "run-with-instructions"
    manifest = evidence.export_evidence(selected, output)
    assert (output / "software/instruction-semantics-authored.yaml").read_bytes() == authored.read_bytes()
    assert "software/instruction-semantics.json" in manifest["consumers"]["instruction_selection"]
    authored.unlink()
    restored = evidence.load_exported_evidence(output)
    assert restored.derivation_identity == selected.derivation_identity
    assert evidence.export_evidence(restored, output) == manifest


def test_selected_instruction_semantics_cannot_escape_provider(monkeypatch, tmp_path):
    _, raw, code = _selection(monkeypatch, tmp_path)
    contract = code.parent / "contracts/target_contract.yaml"
    contract.write_text("name: fixture\ncompute_units: []\ninstruction_semantics: ../foreign.yaml\n")
    (tmp_path / "foreign.yaml").write_text("target: foreign\n")
    with pytest.raises(ValueError, match="escapes provider root|resource is not a file"):
        evidence.select_evidence("fixture", facts_path=raw)


def test_selected_application_accounting_is_digest_bound_and_replayed_without_framework_queries(monkeypatch, tmp_path):
    _, facts_path, _ = _selection(monkeypatch, tmp_path)
    inventory = {
        "schema_version": 1,
        "status": "incomplete",
        "coverage_status": "unverified",
        "n_operations": 1,
        "applications": {
            "app": {
                "capture": "app/model.mlir",
                "capture_sha256": "a" * 64,
                "n_operations": 1,
                "n_signatures": 1,
                "counts": {"unclassified": 1},
                "signatures": [
                    {
                        "operation": "aten.mm.default",
                        "frontend_op": "aten.mm.default",
                        "mlir_operation": "linalg.matmul",
                        "disposition": "unclassified",
                        "count": 1,
                        "ordinals": [0],
                        "semantic_family": "contraction",
                        "operand_format": "int8",
                        "ordered_result_types": [{"shape": [2, 2]}],
                    }
                ],
            }
        },
    }
    raw = (json.dumps(inventory, indent=4) + "\n").encode()
    sidecar = tmp_path / "demands.json"
    sidecar.write_bytes(raw)
    content_digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    requirement = tmp_path / "requirement.yaml"
    requirement.write_text(
        f"application_demands:\n  sidecar: demands.json\n  full_inventory_sha256: {content_digest}\n"
    )
    descriptor = {"workload_spec": {"applications": ["app"]}}
    selected = evidence.select_evidence(
        "fixture", descriptor=descriptor, facts_path=facts_path, conformance_spec=requirement
    )
    output = tmp_path / "report-run"
    manifest = evidence.export_evidence(selected, output)
    assert selected.application_inventory_identity["status"] == "digest_bound"
    assert (output / "coverage/application-inventory.json").read_bytes() == raw
    accounting = json.loads((output / "coverage/operation-accounting.json").read_bytes())
    assert accounting["overall"]["n_mlir_operations"] == 1
    assert accounting["overall"]["pytorch_provenance"]["original_pytorch_invocation_count"] is None
    assert "coverage/operation-accounting.json" in manifest["consumers"]["operation_accounting"]
    assert "hardware/effective-views/isa-taxonomy.json" in manifest["consumers"]["corpus_binding"]
    coverage_readme = (output / "coverage/README.md").read_text()
    assert "unreviewed inputs remain unknown" in coverage_readme
    assert "## Normalized-IR operations by partition" in coverage_readme
    assert "reviewed declaration screens" not in coverage_readme
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="inventory differs"):
        evidence.select_evidence("fixture", descriptor=descriptor, facts_path=facts_path, conformance_spec=requirement)
    sidecar.unlink()
    requirement.unlink()
    from merlin.targetgen import aten_coverage

    monkeypatch.setattr(aten_coverage, "observe_opset", lambda **kwargs: pytest.fail("frozen replay queried torch"))
    restored = evidence.load_exported_evidence(output)
    assert restored.application_inventory == inventory
    assert evidence.export_evidence(restored, output) == manifest


def test_reload_and_reexport_refuse_changed_saved_artifacts(monkeypatch, tmp_path):
    selected, _, _ = _selection(monkeypatch, tmp_path)
    output = tmp_path / "run"
    evidence.export_evidence(selected, output)
    (output / "hardware/circt/facts.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="evidence member changed"):
        evidence.load_exported_evidence(output)
    with pytest.raises(FileExistsError, match="changed evidence output"):
        evidence.export_evidence(selected, output)


def test_actual_native_producer_joins_exact_capture_and_replays_without_host_admission(monkeypatch, tmp_path):
    """One coarse producer/freeze workflow on an explicitly selected real capture."""
    capture_root = os.environ.get("MERLIN_TEST_NATIVE_CAPTURE")
    if not capture_root:
        pytest.skip("explicit real native capture and compiler tools required")
    from merlin_experiments import model_qualification as qualification

    from merlin.targetgen.application_inventory import application_demand_inventory

    capture = Path(capture_root) / "model.mlir"
    result = qualification.qualify(
        bundle=capture.parent,
        package=None,
        target=None,
        output=tmp_path / "native",
        native_host_only=True,
        numeric_policy={"atol": 1e-5, "rtol": 1e-4},
        timeout_seconds=90,
        stage_timeout_seconds=45,
        memory_gib=8,
    )
    assert result["native_host_numerical_verified"] is True
    _, facts_path, _ = _selection(monkeypatch, tmp_path)
    inventory = application_demand_inventory(
        {"iteration": capture},
        "fixture",
        detailed=True,
        include_graph=True,
        capability_contract={"name": "fixture", "compute_units": []},
        application_metadata={"iteration": {"workload_role": "iteration", "coverage_scope": "full_capture"}},
    )
    sidecar = tmp_path / "demands.json"
    sidecar.write_text(json.dumps(inventory))
    options = {
        "descriptor": {"workload_spec": {"applications": ["iteration"]}},
        "facts_path": facts_path,
        "inventory_path": sidecar,
    }
    selected = evidence.select_evidence(
        "fixture", **options, native_qualifications={"iteration": tmp_path / "native/qualification.json"}
    )
    baseline = selected.native_baseline_observations["iteration"]
    assert baseline["status"] == "whole_program_numerically_matched" and baseline["target_executed"] is False
    frozen = tmp_path / "frozen"
    manifest = evidence.export_evidence(selected, frozen)
    accounting = json.loads((frozen / "coverage/operation-accounting.json").read_bytes())
    assert accounting["applications"]["iteration"]["precision_execution"]["executor"] == "native_cpu"
    assert all(
        row["host_admission"]["status"] == "unknown" for row in accounting["applications"]["iteration"]["signatures"]
    )
    assert any(source["role"] == "native-artifact:iteration" for source in manifest["sources"])
    wrong = json.loads(sidecar.read_bytes())
    wrong["applications"]["iteration"]["capture_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="capture"):
        evidence.select_evidence(
            "fixture", **options, native_qualifications={"iteration": tmp_path / "native/qualification.json"}
        )
    sidecar.unlink()
    monkeypatch.setattr(qualification, "inspect_workflow", lambda *_: pytest.fail("frozen replay reopens live capture"))
    restored = evidence.load_exported_evidence(frozen)
    assert restored.native_baseline_observations == selected.native_baseline_observations
    assert evidence.export_evidence(restored, tmp_path / "replay") == manifest


def test_frontend_graph_catalog_and_receipt_survive_source_deletion(monkeypatch, tmp_path):
    """One joined export/replay check, including honest legacy/partial evidence."""
    from merlin.targetgen.application_inventory import application_demand_inventory
    from merlin.targetgen.capsule_source import _write_frontend_evidence

    _, facts_path, _ = _selection(monkeypatch, tmp_path)
    capture = tmp_path / "iteration" / "linalg.mlir"
    capture.parent.mkdir()
    capture.write_text(
        "builtin.module { func.func @forward(%a: f32, %b: f32) -> f32 {\n"
        "  %result = arith.addf %a, %b : f32\n  func.return %result : f32\n} }\n"
    )
    trace = capture.with_name("frontend-trace.json")
    trace.write_text(
        json.dumps(
            {
                "schema": "m2m.frontend_trace.v1",
                "status": "diagnostic",
                "graphs": {},
                "blockers": ["original source unavailable"],
            }
        )
    )
    catalog = capture.with_name("pytorch-opset.json")
    catalog.write_text(
        json.dumps(
            {
                "schema": "merlin.pytorch_opset.v1",
                "status": "not_available",
                "reason": "diagnostic fixture has no framework process",
            }
        )
    )
    inventory = application_demand_inventory(
        {"iteration": capture},
        "fixture",
        detailed=True,
        include_graph=True,
        capability_contract={"name": "fixture", "compute_units": []},
        application_metadata={"iteration": {"workload_role": "iteration", "coverage_scope": "full_capture"}},
    )
    sidecar = tmp_path / "demands.json"
    sidecar.write_text(json.dumps(inventory))
    descriptor = {"workload_spec": {"applications": ["iteration"]}}
    selected = evidence.select_evidence("fixture", descriptor=descriptor, facts_path=facts_path, inventory_path=sidecar)
    assert selected.frontend_traces["iteration"]["status"] == "diagnostic"
    assert (
        selected.application_graphs["iteration"]["capture_sha256"] == hashlib.sha256(capture.read_bytes()).hexdigest()
    )
    incompatible = json.loads(sidecar.read_bytes())
    incompatible["applications"]["iteration"]["capture_normalization"]["output_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="same frontend/MLIR environment"):
        evidence.select_evidence("fixture", descriptor=descriptor, facts_path=facts_path, inventory_path=sidecar)
    sidecar.write_text(json.dumps(inventory))
    bundle = tmp_path / "capsule"
    bundle.mkdir()
    artifact = SimpleNamespace(
        meta={
            "frontend_trace": {"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()},
            "framework_catalog": {"path": str(catalog), "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()},
        },
        linalg_mlir=capture.read_text(),
    )
    receipt = _write_frontend_evidence(artifact, bundle, artifact.linalg_mlir)
    assert "execution unverified" in receipt["qualification"]
    assert (bundle / "frontend-source.mlir").read_bytes() == capture.read_bytes()
    output = tmp_path / "report"
    manifest = evidence.export_evidence(selected, output)
    index = json.loads((output / "software/frontend/index.json").read_bytes())["applications"]["iteration"]
    assert all((output / member).is_file() for member in index.values())
    report = json.loads((output / "coverage/operation-accounting.json").read_bytes())
    assert report["applications"]["iteration"]["completeness"]["source_trace"]["status"] != "complete"
    for original in (capture, trace, catalog, sidecar):
        original.unlink()
    restored = evidence.load_exported_evidence(output)
    monkeypatch.setattr(evidence, "select_evidence", lambda *a, **k: pytest.fail("replay reobserved live sources"))
    assert evidence.export_evidence(restored, tmp_path / "replayed") == manifest
    (output / index["frontend_trace"]).write_text("{}")
    with pytest.raises(ValueError, match="evidence member changed"):
        evidence.load_exported_evidence(output)


def test_op_frontend_lineage_records_exact_no_sidecar_reference_omission(tmp_path):
    from merlin.targetgen.capsule_source import _portable_weights_reference, _write_frontend_evidence

    weight = str(tmp_path / "weights.safetensors")
    raw = f'builtin.module attributes {{prov.weights_file = "{weight}", prov.level = "linalg"}} {{}}\n'
    portable, edit = _portable_weights_reference(raw, weight, sidecar=False)
    assert edit["kind"] == "weights_reference_omission" and edit["edit_count"] == 1
    trace = tmp_path / "selected-trace.json"
    trace.write_text(json.dumps({"mlir": {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "bytes": len(raw)}}))
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "capsule.interface.mlir").write_text(portable)
    (capsule / "capsule.linalg.mlir").write_text(portable)
    artifact = SimpleNamespace(
        linalg_mlir=raw,
        weights_path=weight,
        meta={"frontend_trace": {"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}},
    )
    receipt = _write_frontend_evidence(artifact, capsule, portable, packaged_path="capsule.linalg.mlir")
    (capsule / "capsule.yaml").write_text(json.dumps({"frontend_trace": receipt}))
    _scrub_capsule_dir(capsule)
    assert receipt["source_mlir_sha256"] == hashlib.sha256((capsule / "frontend-source.mlir").read_bytes()).hexdigest()
    assert receipt["raw_source_mlir_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert receipt["raw_source_trace_bound"] is True
    assert receipt["source_portability"]["kind"] == "weights_reference_omission"
    assert receipt["packaged_mlir_sha256"] == hashlib.sha256((capsule / "capsule.linalg.mlir").read_bytes()).hexdigest()


def test_absent_facts_are_diagnostic_without_regeneration(monkeypatch, tmp_path):
    monkeypatch.setattr(target_registry, "resolve", lambda target: (_ for _ in ()).throw(KeyError(target)))
    monkeypatch.setattr(facts, "find_facts", lambda *a, **k: None)
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("unexpected extraction"))
    monkeypatch.setattr(readout_facet, "capture_inputs", lambda *a, **k: {})
    monkeypatch.setattr(base, "execution_capability_facts", lambda target: {})
    selected = evidence.select_evidence("unavailable")
    assert selected.loaded_facts == {}
    assert selected.raw_facts_sha256 is None
    assert selected.status == "diagnostic"
    assert any(item["component"] == "facts" and item["status"] == "unknown" for item in selected.diagnostics)
    evidence.export_evidence(selected, tmp_path / "run")
    assert evidence.load_exported_evidence(tmp_path / "run").raw_facts is None


def test_legacy_backend_hooks_observe_the_explicit_document_without_extraction(monkeypatch, tmp_path):
    selected, raw, _ = _selection(monkeypatch, tmp_path)
    observed = []

    def hook(target, **kwargs):
        doc = facts.load_facts(target)
        observed.append(doc)
        return {}

    monkeypatch.setattr(readout_facet, "capture_inputs", hook)
    monkeypatch.setattr(base, "execution_capability_facts", hook)
    repeated = evidence.select_evidence("fixture", facts_path=raw)
    assert observed == [repeated.refreshed_facts, repeated.refreshed_facts]
    assert selected.raw_facts == repeated.raw_facts
    assert not facts._OBSERVED.get()


def test_performance_sweeps_consume_selected_snapshot(monkeypatch, tmp_path):
    selected, raw, _ = _selection(monkeypatch, tmp_path)
    raw.write_bytes(b"{}")
    monkeypatch.setattr(sweeps, "derive_profile", lambda *a, **k: pytest.fail("live profile read"))
    monkeypatch.setattr(sweeps, "execution_capability_facts", lambda *a, **k: pytest.fail("live backend read"))
    assert sweeps._performance_facts("fixture", evidence=selected) == selected.performance_facts
    with pytest.raises(ValueError, match="target differs"):
        sweeps._performance_facts("other", evidence=selected)


def test_profile_read_only_facts_use_shared_selection(monkeypatch, tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text('{"facts": {"selected": true}}')
    monkeypatch.setattr(facts, "find_facts", lambda target: selected)
    monkeypatch.setattr(facts, "rtl_facts_path", lambda *a, **k: pytest.fail("direct cache lookup"))
    monkeypatch.setattr(facts, "load_facts", lambda *a, **k: pytest.fail("unrequested extraction"))
    assert profile._read_facts("fixture", allow_extraction=False) == {"facts": {"selected": True}}


def test_profile_snapshots_do_not_read_ports_or_contracts(monkeypatch):
    from merlin.targetgen import eligibility
    from merlin.targetgen.rtl import ports

    monkeypatch.setattr(ports, "port_facts", lambda *a, **k: pytest.fail("ambient port read"))
    monkeypatch.setattr(eligibility, "capability_map_for_target", lambda *a, **k: pytest.fail("ambient contract read"))
    result = profile.derive_profile("fixture", facts={}, residual={}, contract={})
    assert result.traits["explicit_completion"].satisfied is None


def test_banked_memory_profile_uses_supplied_facts_without_ambient_extraction(monkeypatch):
    monkeypatch.setattr(facts, "load_facts", lambda *a, **k: pytest.fail("ambient banked-memory facts read"))
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("ambient banked-memory extraction"))
    document = {
        "facts": {
            "arrays": [{"name": "array", "rows": 4, "cols": 4}],
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "scratchpad smem UInt<8>"},
                {"name": "accumulator", "dtype": "i32"},
            ],
            "memories": [{"name": "scratchpad", "bytes": 256, "depth": 32}],
        }
    }
    result = profile.derive_profile("fixture", facts=document, residual={}, contract={})
    assert result.traits["banked_memory"].satisfied is True


def test_readout_refresh_hashes_original_source_bytes(monkeypatch, tmp_path):
    from merlin.targetgen.rtl import circt_introspect

    source = tmp_path / "registers.scala"
    original = b"// opaque byte: \xff\n"
    source.write_bytes(b"// changed live file\n")
    monkeypatch.setattr(circt_introspect, "_bundle_layouts", lambda text: ({"packet": {}}, {}))
    document = {"facts": {"interfaces": [{"name": "register_bundle_layouts", "source": str(source), "bundles": {}}]}}
    refreshed = readout_facet.with_current_register_layouts(document, source_bytes={str(source): original})
    record = refreshed["facts"]["interfaces"][0]
    assert record["refreshed"]["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert document["facts"]["interfaces"][0]["bundles"] == {}


def test_supplied_readout_inputs_suppress_live_declarations(monkeypatch):
    monkeypatch.setattr(readout_facet, "capture_inputs", lambda *a, **k: pytest.fail("live readout metadata"))
    facets = readout_facet.for_target("fixture", contract={}, facts={}, readout_inputs={})
    assert len(facets) == 1
    assert facets[0].rounding is None
    assert facets[0].element_dtype is None


def test_memory_axis_does_not_reread_missing_selected_store(monkeypatch, tmp_path):
    selected, _, _ = _selection(monkeypatch, tmp_path)
    from merlin.targetgen import memory_regime

    monkeypatch.setattr(memory_regime, "operand_store", lambda *a, **k: pytest.fail("ambient store discovery"))
    with pytest.raises(sweeps.AxisDerivationUnavailable):
        sweeps._memory_regime_axis(
            {"regimes": ["spills"]},
            owner="test",
            target="fixture",
            tile=4,
            dtype="i8",
            fixed={"M": [4], "N": [4]},
            evidence=selected,
        )


def test_an_rtl_audit_beside_the_facts_clears_only_its_own_diagnostic(monkeypatch, tmp_path):
    from merlin_experiments.phase0 import evidence_status

    selected, raw, _ = _selection(monkeypatch, tmp_path)
    assert selected.status == "diagnostic"
    assert any(row["component"] == "rtl-audit" for row in selected.diagnostics)
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text("target: fixture\n")
    (raw.parent / evidence_status.AUDIT_MEMBER).write_text(
        json.dumps(
            {
                "schema": evidence_status.AUDIT_SCHEMA,
                "status": "verified",
                "facts_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "hardware_spec_sha256": hashlib.sha256(hardware.read_bytes()).hexdigest(),
                "checks": [{"source_status": "verified", "extraction": {"status": "agrees"}, "gap": None}],
            }
        )
    )
    audited = evidence.select_evidence("fixture", facts_path=raw, hardware_spec=hardware)
    assert not any(row["component"] == "rtl-audit" for row in audited.diagnostics)
    # Other reasons remain on record, so the selection is still not verified.
    assert audited.diagnostics and audited.status == "diagnostic"
