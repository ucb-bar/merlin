"""Fail-closed private complete-network gate and host-only input declarations."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_bucketize_support as bucketize
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation
from merlin_experiments.phase1.feedback import private_f32_maximum_support as maximum
from merlin_experiments.phase1.feedback import private_f32_reduction_support as f32_reductions
from merlin_experiments.phase1.feedback import private_full_models as gate
from merlin_experiments.phase1.feedback import private_index_host_support as prepared_index
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer_reductions
from merlin_experiments.phase1.feedback import private_linalg_support as linalg
from merlin_experiments.phase1.feedback import private_linkage_support as linkage
from merlin_experiments.phase1.feedback import private_literal_arange_admission as arange
from merlin_experiments.phase1.feedback import private_ordered_scan_support as ordered_scan
from merlin_experiments.phase1.feedback import private_pool_support as pool
from merlin_experiments.phase1.feedback.private_prebuilt_receipt import load_diagnostic_receipt

from merlin.targetgen.sandbox import bwrap, host_surfaces
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_prepared_index(raw, normalized, receipt, count, selected, linked=None):
    witness = {
        "status": prepared_index.LINKED if linked else prepared_index.PENDING,
        "scope": prepared_index.SCOPE,
        "raw_source_sha256": raw,
        "normalized_source_sha256": normalized,
        "capture_receipt_sha256": receipt,
        "n_source_operations": count,
        "selected_index_observation": deepcopy(selected),
        "source_proof": None,
        "source_proof_sha256": None,
        "expected_ordinals": [],
        "admissions": [],
        "source_verified": True,
    }
    if linked:
        witness["linked_build"] = dict(linked)
        witness["source_capture_path"] = None
    return witness


def _synthetic_host_groups(module, _target, **_kwargs):
    """Cover the real source while isolating placement from admission checks."""
    from merlin.xdsl_dialects.lowering import compute_groups

    roots = [op for op in module.walk() if op.name.startswith("linalg.") and compute_groups.classify(op) is not None]
    return [
        compute_groups.Group(
            index,
            compute_groups.HOST,
            None,
            members=[root],
            stages=[compute_groups.classify(root).kind],
        )
        for index, root in enumerate(roots)
    ]


def _synthetic_source_demand(raw, normalized, groups, *, linked=None):
    """Closed bookkeeping fixture, independent of native/source qualification."""
    from merlin_experiments.phase1.feedback import private_source_demand

    record = {
        "schema": private_source_demand.SCHEMA,
        "status": "source_demand_linked" if linked is not None else "source_demand_derived",
        "source_sha256": raw,
        "normalized_source_sha256": normalized,
        "declarations": {name: "a" * 64 for name in ("capability", "software", "rtl_facts", "numeric_policy")},
        "n_source_groups": groups,
        "n_demanded_groups": groups,
        "n_requested_groups": groups,
        "n_lowered_groups": groups if linked else 0,
        "n_covered_groups": groups if linked else 0,
        "n_host_groups": 0,
        "n_unknown_host_groups": 0,
        "n_missing_direct_groups": 0,
        "n_unresolved_raw_regions": 0,
        "n_raw_undetermined_regions": 0,
        "demanded_group_provenance": [],
    }
    if linked is not None:
        record["linked_build"] = dict(linked)
    return record


def test_private_capture_binds_sealed_and_compiler_tree_algorithms_independently(tmp_path):
    from merlin_experiments.capture_execution import sealed_m2m

    from merlin.compile.model_execution_inputs import strict_tree_sha256

    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}\n")
    sealed = sealed_m2m._snapshot_tree(capture)["sha256"]
    strict = strict_tree_sha256(capture)["sha256"]
    assert sealed != strict
    assert gate._capture_tree_bindings(capture, strict, sealed) == {
        "compiler_strict_tree_sha256": strict,
        "issuer_sealed_tree_sha256": sealed,
    }
    with pytest.raises(ValueError, match="explicitly pinned"):
        gate._capture_tree_bindings(capture, "0" * 64, sealed)
    with pytest.raises(ValueError, match="explicitly pinned"):
        gate._capture_tree_bindings(capture, None, sealed)
    with pytest.raises(ValueError, match="attestation"):
        gate._capture_tree_bindings(capture, strict, "0" * 64)
    (capture / "model.mlir").write_text("module { func.func @changed() }\n")
    with pytest.raises(ValueError, match="explicitly pinned"):
        gate._capture_tree_bindings(capture, strict, sealed)
    with pytest.raises(ValueError, match="attestation"):
        gate._capture_tree_bindings(capture, strict_tree_sha256(capture)["sha256"], sealed)


def test_target_declares_every_complete_program():
    from merlin.common.paths import repo_root

    archived = Path(__file__).with_name("source-inputs") / "examples/gemmini/target/descriptor.yaml"
    descriptor = archived if archived.is_file() else repo_root() / "examples/gemmini/target/descriptor.yaml"
    assert gate.requirements_for(descriptor) == ("resnet50", "smolvla", "tiny_llama")
    assert gate.program_requirements_for(descriptor) == {
        "resnet50": ("model",),
        "smolvla": ("prefix_encode", "flow_denoise", "action_decode"),
        "tiny_llama": ("prefill", "decode"),
    }
    assert "M2M_LLAMA_LAYERS" in gate.loader_env_requirements_for(descriptor)["tiny_llama"]["forbidden"]
    smol = gate.loader_env_requirements_for(descriptor)["smolvla"]
    assert smol["required"]["M2M_SMOLVLA_PAPER_READY"] == "1"
    assert smol["selected_roles"] == ("checkpoint", "input")


def test_private_capture_records_synthetic_input_without_accuracy_claim(tmp_path):
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "loader_provenance_status": "declared",
                "loader_paper_ready": False,
                "loader_provenance": {
                    "synthetic_inputs": True,
                    "input_source": "synthetic_seed",
                    "input_sha256": "a" * 64,
                    "input_path": "/private/not-for-feedback.npz",
                },
            }
        )
    )
    observed = gate._captured_input_provenance([("model", stage)], {"paper_ready": False, "synthetic_inputs": True})[
        "model"
    ]
    assert observed["paper_ready"] is False and observed["synthetic_inputs"] is True
    assert observed["scope"].endswith("no paper accuracy or full-model numerical result")
    assert "input_path" not in observed
    with pytest.raises(ValueError, match="differs"):
        gate._captured_input_provenance([("model", stage)], {"paper_ready": True})
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "loader_provenance_status": "declared",
                "loader_paper_ready": False,
                "loader_provenance": {
                    "synthetic_tokens": True,
                    "token_source": "synthetic_seed_0",
                    "token_sha256": "b" * 64,
                },
            }
        )
    )
    token_observed = gate._captured_input_provenance(
        [("prefill", stage)], {"paper_ready": False, "synthetic_inputs": True}
    )["prefill"]
    assert token_observed["input_source"] == "synthetic_seed_0"
    assert token_observed["input_sha256"] == "b" * 64
    assert gate._expected_input_provenance({"input_provenance": {"paper_ready": True, "synthetic_inputs": False}}) == {
        "paper_ready": True,
        "synthetic_inputs": False,
    }


def test_complete_requires_current_candidate_all_programs_and_device_work(monkeypatch):
    # This existing roster fixture is synthetic; native recipe integrity is
    # exercised independently by test_private_postbuild_recipe_checks_actual_link_inputs.
    verified_compilation = {
        "status": "completed_compilation_inputs_verified",
        "recipe_path": "/operator/build/compilation_recipe.json",
        "recipe_sha256": "5" * 64,
        "elf_path": "/operator/build/model.elf",
        "elf_sha256": "2" * 64,
        "n_commands": 1,
        "link_executable": {"path": "/operator/cc", "sha256": "6" * 64, "bytes": 1},
        "link_inputs": [{"path": "/operator/input.o", "sha256": "7" * 64, "bytes": 1}],
        "scope": compilation.SCOPE,
    }
    monkeypatch.setattr(compilation, "verify", lambda *args, **kwargs: deepcopy(verified_compilation))
    monkeypatch.setattr(maximum, "_captured", lambda *_args: True)  # noqa: PLC2701 -- synthetic fixture only
    programs = {"a": ("prefix", "decode"), "b": ("model",)}
    selected_index = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "synthetic-clang",
        "compiler_resolved": "/synthetic/clang",
        "compiler_sha256": "b" * 64,
        "cross_flags": ["--target=synthetic", "-march=synthetic"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "synthetic test fixture; no compiler executed",
    }
    indexed_passes = ",".join(
        f"{name}{{index-bitwidth=64}}"
        for name in (
            "convert-index-to-llvm",
            "convert-arith-to-llvm",
            "finalize-memref-to-llvm",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
        )
    )

    def model(name, stages):
        return {
            "model": name,
            "status": "pass",
            "checks": {
                "capture_identity": {
                    "compiler_strict_tree_sha256": "c" * 64,
                    "issuer_sealed_tree_sha256": "e" * 64,
                },
                "recipe_derivation": {
                    "evidence_manifest_sha256": "5" * 64,
                    "recipe_sha256": "6" * 64,
                    "recipe_semantic_sha256": "7" * 64,
                    "scope": "current-spec diagnostic recipe derivation only; no model or hardware admission",
                },
                "input_provenance": {
                    stage: {
                        "paper_ready": None,
                        "synthetic_inputs": None,
                        "meta_sha256": "a" * 64,
                        "scope": "input provenance only; no paper accuracy or full-model numerical result",
                    }
                    for stage in stages
                },
                "source": {
                    stage: {
                        "source_sha256": "d" * 64,
                        "normalized_source_sha256": "e" * 64,
                        "capture_receipt_sha256": "f" * 64,
                        "n_source_operations": 5,
                        "n_linalg_regions": 1,
                        "n_groups": 1,
                        "eligible_groups": [0],
                        "eligible_group_provenance": [],
                        "accelerator_source_demand": _synthetic_source_demand(
                            "d" * 64,
                            "e" * 64,
                            1,
                            linked={
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        ),
                        "selected_index_observation": deepcopy(selected_index),
                        "n_bounded_control_assertions": 0,
                        "n_internal_mask_compactions": 0,
                        "linalg_host_support": {
                            **linalg.begin("d" * 64, "e" * 64, 5, deepcopy(selected_index)),
                            "status": linalg.LINKED,
                            "actual_index_observation": {
                                **deepcopy(selected_index),
                                "effective_pipeline": indexed_passes,
                            },
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        linkage.FIELD: {
                            **linkage.begin("d" * 64, "e" * 64, 5),
                            "status": linkage.LINKED,
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        "literal_arange_host_support": {
                            **arange.begin("d" * 64, "e" * 64, 5, deepcopy(selected_index)),
                            "status": arange.LINKED,
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        integer_reductions.FIELD: {
                            **integer_reductions.begin("d" * 64, "e" * 64, 5, deepcopy(selected_index)),
                            "status": integer_reductions.LINKED,
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        f32_reductions.FIELD: {
                            **f32_reductions.begin("d" * 64, "e" * 64, 5, deepcopy(selected_index)),
                            "status": f32_reductions.LINKED,
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        maximum.FIELD: {
                            **maximum.begin(
                                "d" * 64,
                                "e" * 64,
                                "f" * 64,
                                tuple(SimpleNamespace(attributes={}, properties={}) for _ in range(5)),
                                deepcopy(selected_index),
                            ),
                            "status": maximum.LINKED,
                            "source_capture_path": "/operator/synthetic-capture",
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        prepared_index.FIELD: _empty_prepared_index(
                            "d" * 64,
                            "e" * 64,
                            "f" * 64,
                            5,
                            selected_index,
                            {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        ),
                        ordered_scan.FIELD: {
                            "status": ordered_scan.LINKED,
                            "scope": ordered_scan.SCOPE,
                            "raw_source_sha256": "d" * 64,
                            "normalized_source_sha256": "e" * 64,
                            "capture_receipt_sha256": "f" * 64,
                            "n_source_operations": 5,
                            "selected_index_observation": deepcopy(selected_index),
                            "count": 0,
                            "occurrences": [],
                            "occurrences_sha256": ordered_scan._digest([]),  # noqa: PLC2701 -- exact empty fixture
                            "source_verified": True,
                            "admission_verified": True,
                            "admissions": [],
                            "admissions_sha256": ordered_scan._digest([]),  # noqa: PLC2701 -- empty fixture
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        bucketize.FIELD: {
                            "status": bucketize.LINKED,
                            "scope": bucketize.SCOPE,
                            "raw_source_sha256": "d" * 64,
                            "normalized_source_sha256": "e" * 64,
                            "capture_receipt_sha256": "f" * 64,
                            "n_source_operations": 5,
                            "selected_index_observation": deepcopy(selected_index),
                            "source_proof": None,
                            "source_proof_sha256": None,
                            "original_to_prepared_equivalence": "not_proved",
                            "boundary_facts": [],
                            "expected_ordinals": [],
                            "admissions": [],
                            "actual_index_observation": {
                                **deepcopy(selected_index),
                                "effective_pipeline": indexed_passes,
                            },
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                            "source_capture_path": None,
                        },
                        "pool_value_support": {
                            "status": pool.LINKED,
                            "scope": pool.SCOPE,
                            "raw_source_sha256": "d" * 64,
                            "normalized_source_sha256": "e" * 64,
                            "count": 0,
                            "occurrences": [],
                            "device_ordinals": [],
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        "transpose_data_support": {
                            "status": "source_structural_data_support_linked",
                            "scope": gate.TRANSPOSE_DATA_SUPPORT_SCOPE,
                            "raw_source_sha256": "d" * 64,
                            "normalized_source_sha256": "e" * 64,
                            "count": 1,
                            "ordinals": [2],
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                        "generic_copy_data_support": {
                            "status": "source_structural_data_support_linked",
                            "scope": gate.TRANSPOSE_DATA_SUPPORT_SCOPE,
                            "raw_source_sha256": "d" * 64,
                            "normalized_source_sha256": "e" * 64,
                            "count": 1,
                            "ordinals": [3],
                            "linked_build": {
                                "capture_tree_sha256": "c" * 64,
                                "elf_sha256": "2" * 64,
                                "candidate_tree_sha256": "1" * 64,
                            },
                        },
                    }
                    for stage in stages
                },
                "build": {
                    "status": "capture_lower_codegen_link_verified",
                    "candidate_tree_sha256": "1" * 64,
                    "static_build_board": "build-only-board",
                    "static_build_board_catalog_source": "/operator/board-catalog.yaml",
                    "static_build_board_catalog_sha256": "8" * 64,
                    "static_build_board_dts_source": "/operator/selected.dts",
                    "static_build_board_dts_sha256": "9" * 64,
                    "static_build_board_scope": gate.BUILD_BOARD_SCOPE,
                    "accelerator_rtl_facts_sha256": "a" * 64,
                    "accelerator_rtl_config": "fixture-accelerator-config",
                    "linked_device_groups": len(stages),
                    "programs": [
                        {
                            "program": stage,
                            "status": "capture_lower_codegen_link_verified",
                            "candidate_tree_sha256": "1" * 64,
                            "capture_tree_sha256": "c" * 64,
                            "source_sha256": "d" * 64,
                            "elf_sha256": "2" * 64,
                            "compilation_recipe": deepcopy(verified_compilation),
                            "index_lowering": {**deepcopy(selected_index), "effective_pipeline": indexed_passes},
                            "linked_device_groups": 1,
                            "static_host_compute_audit": [
                                {
                                    "verdict": "clean_static_host_compute_audit",
                                    "artifact_sha256": "3" * 64,
                                    "object_sha256": "4" * 64,
                                    "audit": {
                                        "budget": {"arithmetic_per_element": 1.0},
                                        "groups": [{"verdict": "clean"}],
                                    },
                                }
                            ],
                        }
                        for stage in stages
                    ],
                },
            },
        }

    record = {
        "schema": gate.RESULT_SCHEMA,
        "passed": True,
        "private_spec_sha256": "f" * 64,
        "authored_source_freeze": {
            "schema": "merlin.phase1.private_source_freeze.v1",
            "spec_sha256": "f" * 64,
            "record_sha256": "0" * 64,
            "root": "/operator/run/private_full_model_input/sources",
        },
        "candidate_tree_sha256": "1" * 64,
        "required_models": ["a", "b"],
        "required_programs": {name: list(stages) for name, stages in programs.items()},
        "models": [model(name, stages) for name, stages in programs.items()],
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    kwargs = {"required_models": ("a", "b"), "required_programs": programs, "candidate_sha256": "1" * 64}
    assert gate.complete(record, **kwargs)
    source = record["models"][0]["checks"]["source"]["prefix"]
    demand = source.pop("accelerator_source_demand")
    assert not gate.complete(record, **kwargs)
    source["accelerator_source_demand"] = demand
    demand["n_requested_groups"] = 0
    assert not gate.complete(record, **kwargs)
    demand["n_requested_groups"] = 1
    assert gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v1"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v6"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v2"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v3"
    assert not gate.complete(record, **kwargs)
    record["schema"] = gate.RESULT_SCHEMA
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v7"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v8"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v9"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v10"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v12"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v13"
    assert not gate.complete(record, **kwargs)
    record["schema"] = gate.RESULT_SCHEMA
    record["models"][0]["checks"]["source"]["prefix"].pop(maximum.FIELD)
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"].pop("n_internal_mask_compactions")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    for malformed in (None, {}, {"status": "unavailable_in_prebuilt_receipt"}):
        record["models"][0]["checks"]["build"]["programs"][0]["compilation_recipe"] = malformed
        assert not gate.complete(record, **kwargs)
        record["models"][0] = model("a", programs["a"])
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v4"
    assert not gate.complete(record, **kwargs)
    record["schema"] = "merlin.phase1.private_full_model_build_gate.v5"
    assert not gate.complete(record, **kwargs)
    record["schema"] = gate.RESULT_SCHEMA
    record["models"][0]["checks"]["source"]["prefix"]["linalg_host_support"].pop("selected_index_observation")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["linalg_host_support"].pop("actual_index_observation")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    linalg_witness = record["models"][0]["checks"]["source"]["prefix"]["linalg_host_support"]
    linalg_witness["count"] = 1
    linalg_witness["occurrences"] = [
        {
            "ordinal": 0,
            "profile": "synthetic-host",
            "capability_spec_sha256": "b" * 64,
            "declaration": "synthetic-add",
            "schema": "merlin.static_pointwise_source_body.v1",
            "operation": "arith.addi",
            "predicate": None,
            "shape": [1 << 61],
            "ordered_types": ["i64", "i64", "i64"],
            "input_shapes": [],
            "input_maps": [],
        }
    ]
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    for field in (integer_reductions.FIELD, f32_reductions.FIELD, bucketize.FIELD, ordered_scan.FIELD, linkage.FIELD):
        record["models"][0]["checks"]["source"]["prefix"].pop(field)
        assert not gate.complete(record, **kwargs)
        record["models"][0] = model("a", programs["a"])
        record["models"][0]["checks"]["source"]["prefix"][field]["linked_build"]["elf_sha256"] = "0" * 64
        assert not gate.complete(record, **kwargs)
        record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"].pop("literal_arange_host_support")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["literal_arange_host_support"]["source_proof"] = {"count": 1}
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["literal_arange_host_support"]["linked_build"]["elf_sha256"] = (
        "0" * 64
    )
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"].pop("linalg_host_support")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["linalg_host_support"]["count"] = 1
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["linalg_host_support"]["occurrences"] = [{"ordinal": True}]
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    frozen = record.pop("authored_source_freeze")
    assert not gate.complete(record, **kwargs)
    record["authored_source_freeze"] = frozen
    record["models"][0]["checks"]["build"]["programs"][0]["index_lowering"].pop("effective_pipeline")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["selected_index_observation"]["cross_flags"].append("-drift")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["capture_identity"].pop("issuer_sealed_tree_sha256")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"].pop("generic_copy_data_support")
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["generic_copy_data_support"]["linked_build"]["elf_sha256"] = (
        "f" * 64
    )
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["generic_copy_data_support"]["ordinals"] = [True]
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][0]["checks"]["source"]["prefix"]["generic_copy_data_support"]["normalized_source_sha256"] = (
        "f" * 64
    )
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    assert not gate.complete(record, **{**kwargs, "candidate_sha256": "3" * 64})
    record["models"][0]["checks"]["build"]["programs"].pop()
    assert not gate.complete(record, **kwargs)
    record["models"][0] = model("a", programs["a"])
    record["models"][1]["checks"]["build"]["linked_device_groups"] = 0
    assert not gate.complete(record, **kwargs)
    record["models"][1] = model("b", programs["b"])
    record["models"][1]["checks"]["build"]["static_build_board_scope"] = "executed"
    assert not gate.complete(record, **kwargs)
    record["models"][1] = model("b", programs["b"])
    record["models"][1]["checks"]["source"]["model"]["transpose_data_support"]["linked_build"]["elf_sha256"] = "f" * 64
    assert not gate.complete(record, **kwargs)
    record["models"][1] = model("b", programs["b"])
    record["models"][1]["checks"]["source"]["model"]["transpose_data_support"]["linked_build"].pop(
        "capture_tree_sha256"
    )
    record["models"][1]["checks"]["build"]["programs"][0].pop("capture_tree_sha256")
    assert not gate.complete(record, **kwargs)


def test_official_grade_refuses_manifest_selected_narrow_roster(tmp_path):
    from merlin_experiments.phase1.feedback.certification import _official_grade_result

    (tmp_path / "submission").mkdir()
    (tmp_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "completion": {
                    "formal_grade_complete": True,
                    "required_tier": "L3",
                    "required_full_models": ["a"],
                    "required_full_programs": {"a": ["model"]},
                }
            }
        )
    )
    result = _official_grade_result(
        0,
        tmp_path,
        required_models=("a", "b"),
        required_programs={"a": ("model",), "b": ("prefix", "decode")},
    )
    assert "private_full_model_roster_mismatch" in result["failures"]


def test_built_group_artifact_vetoes_candidate_host_arithmetic(tmp_path, monkeypatch):
    from merlin.llvmlower import device_build

    monkeypatch.setattr(device_build, "executable_kernel_entry", lambda _target: "kernel")
    device = tmp_path / "device"
    device.mkdir()
    body = "\n".join(
        f'    %v{index} = "llvm.fmul"(%v{index - 1}, %v{index - 1}) : (f32, f32) -> f32' for index in range(1, 65)
    )
    artifact = device / "stem.device.mlir"
    artifact.write_text(
        "\n".join(
            [
                '"builtin.module"() ({',
                '  "llvm.func"() <{sym_name = "kernel", function_type = !llvm.func<void ()>}> ({',
                "  ^entry:",
                '    %v0 = "llvm.mlir.constant"() <{value = 1.000000e+00 : f32}> : () -> f32',
                body,
                '    "llvm.return"() : () -> ()',
                "  }) : () -> ()",
                "}) : () -> ()",
            ]
        )
    )
    (device / "stem.ll").write_text("linked source\n")
    (device / "stem.o").write_bytes(b"linked object")
    with pytest.raises(Exception, match="compute on the host"):
        gate._audit_built_device_host_compute(
            tmp_path,
            [{"symbol": "stem", "group": 1, "source_region_id": "region-a", "source_node_ids": ["node-a"]}],
            {
                "eligible_groups": [4],
                "eligible_group_provenance": [
                    {"source_group": 4, "source_region_id": "region-a", "source_node_ids": ["node-a"]}
                ],
                "eligible_group_metrics": {4: {"elements": 64, "element_bytes": 4}},
            },
            "fixture",
        )


@pytest.mark.parametrize("second_group_optional", [False, True])
def test_built_kernel_audit_reuses_exact_bytes_not_source_calls(tmp_path, monkeypatch, second_group_optional):
    from xdsl.parser import Parser

    from merlin.llvmlower import device_build, toolchain

    monkeypatch.setattr(device_build, "executable_kernel_entry", lambda _target: "kernel")
    monkeypatch.setattr(toolchain, "nm", lambda: "nm")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=""))
    device = tmp_path / "device"
    device.mkdir()
    (device / "shared.device.mlir").write_text(
        '"builtin.module"() ({ "llvm.func"() <{sym_name = "kernel", '
        'function_type = !llvm.func<void ()>}> ({ "llvm.return"() : () -> () }) : () -> () }) : () -> ()'
    )
    (device / "shared.ll").write_text("linked input")
    (device / "shared.o").write_bytes(b"linked object")
    assigned = [
        {"symbol": "shared", "group": i, "source_region_id": f"region-{i}", "source_node_ids": [f"node-{i}"]}
        for i in (3, 7)
    ]
    source = {
        "eligible_groups": [4, 8],
        "eligible_group_provenance": [
            {"source_group": i + 1, "source_region_id": f"region-{i}", "source_node_ids": [f"node-{i}"]} for i in (3, 7)
        ],
        "eligible_group_metrics": {i: {"elements": 64, "element_bytes": 4} for i in (4, 8)},
    }
    if second_group_optional:
        source["eligible_groups"] = [4]
        source["optional_groups"] = [8]
        source["optional_group_provenance"] = [source["eligible_group_provenance"].pop()]
        source["optional_group_metrics"] = {8: source["eligible_group_metrics"].pop(8)}
    second_metrics = source["optional_group_metrics"] if second_group_optional else source["eligible_group_metrics"]
    original = Parser.parse_module
    calls = []

    def counted(parser, *args, **kwargs):
        calls.append(parser)
        return original(parser, *args, **kwargs)

    monkeypatch.setattr(Parser, "parse_module", counted)
    records = gate._audit_built_device_host_compute(tmp_path, assigned, source, "fixture")
    assert [(row["group"], row["source_group"]) for row in records] == [(3, 4), (7, 8)]
    assert len(calls) == 1
    second_metrics[8]["elements"] = 32
    calls.clear()
    gate._audit_built_device_host_compute(tmp_path, assigned, source, "fixture")
    assert len(calls) == 2  # Another element-scale budget requires a separate audit.

    from merlin_experiments.phase1.feedback import private_device_audit as built_audit

    second_metrics[8]["elements"] = 64
    original_sha = built_audit.file_sha256
    reads = []

    def changed_between_calls(path):
        if path.name == "shared.device.mlir":
            reads.append(path)
            if len(reads) == 3:  # After the first audit's before/after byte checks.
                (device / "shared.ll").write_text("different linked input")
        return original_sha(path)

    monkeypatch.setattr(built_audit, "file_sha256", changed_between_calls)
    calls.clear()
    records = gate._audit_built_device_host_compute(tmp_path, assigned, source, "fixture")
    assert len(calls) == 2
    assert records[0]["llvm_sha256"] != records[1]["llvm_sha256"]

    def changed_during_parse(parser, *args, **kwargs):
        module = original(parser, *args, **kwargs)
        (device / "shared.o").write_bytes(b"modified object")
        return module

    monkeypatch.setattr(built_audit, "file_sha256", original_sha)
    monkeypatch.setattr(Parser, "parse_module", changed_during_parse)
    with pytest.raises(ValueError, match="changed during"):
        gate._audit_built_device_host_compute(tmp_path, assigned, source, "fixture")


def test_unknown_built_kernel_audit_preserves_actual_diagnostics_and_still_refuses(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback.private_program_checks import ProgramCheckFailure

    from merlin.llvmlower import device_build

    monkeypatch.setattr(device_build, "executable_kernel_entry", lambda _target: "kernel")
    device = tmp_path / "device"
    device.mkdir()
    artifact = device / "stem.device.mlir"
    artifact.write_text(""""builtin.module"() ({
      "llvm.func"() <{sym_name = "kernel", function_type = !llvm.func<void (i1)>}> ({
      ^entry(%condition: i1):
        "llvm.cond_br"(%condition) [^yes, ^no] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^yes:
        "llvm.return"() : () -> ()
      ^no:
        "llvm.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()""")
    (device / "stem.ll").write_text("linked source\n")
    (device / "stem.o").write_bytes(b"linked object")
    with pytest.raises(ProgramCheckFailure, match="audit is unknown, not clean; kernel:.*conditional") as refusal:
        gate._audit_built_device_host_compute(
            tmp_path,
            [{"symbol": "stem", "group": 1, "source_region_id": "region-a", "source_node_ids": ["node-a"]}],
            {
                "eligible_groups": [4],
                "eligible_group_provenance": [
                    {"source_group": 4, "source_region_id": "region-a", "source_node_ids": ["node-a"]}
                ],
                "eligible_group_metrics": {4: {"elements": 64, "element_bytes": 4}},
            },
            "fixture",
            diagnostics=tmp_path / "diagnostics",
        )
    assert refusal.value.check == "host_compute"
    diagnostics = list((tmp_path / "diagnostics").glob("stem.host-compute-audit-*.json"))
    assert len(diagnostics) == 1
    saved = json.loads(diagnostics[0].read_text())
    assert saved["artifacts"]["llvm_mlir"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert saved["report"]["proven_clean"] is False
    assert saved["report"]["groups"][0]["activity_problems"]


def test_routed_group_join_uses_complete_source_identity_not_preparation_indices():
    source = {
        "eligible_groups": [4, 8],
        "eligible_group_provenance": [
            {"source_group": 4, "source_region_id": "region-a", "source_node_ids": ["a", "b"]},
            {"source_group": 8, "source_region_id": "region-b", "source_node_ids": ["c"]},
        ],
    }
    routed = [
        {"group": 3, "source_region_id": "region-a", "source_node_ids": ["a", "b"]},
        {"group": 7, "source_region_id": "region-b", "source_node_ids": ["c"]},
    ]
    assert gate._join_routed_source_groups(routed, source) == {3: 4, 7: 8}
    for changed in (
        routed[:1],
        [*routed, {"group": 9, "source_region_id": "region-extra", "source_node_ids": ["x"]}],
        [{**routed[0], "source_node_ids": ["b"]}, routed[1]],
        [{**routed[0], "source_node_ids": ["a", "a"]}, routed[1]],
        [{**routed[0], "source_node_ids": []}, routed[1]],
        [{**routed[0], "source_region_id": "region-b"}, routed[1]],
        [{**routed[0], "group": 7}, routed[1]],
    ):
        with pytest.raises(ValueError, match="provenance|omit|ambiguous"):
            gate._join_routed_source_groups(changed, source)
    duplicate_source = {
        **source,
        "eligible_group_provenance": [
            source["eligible_group_provenance"][0],
            {**source["eligible_group_provenance"][1], "source_region_id": "region-a"},
        ],
    }
    with pytest.raises(ValueError, match="ambiguous"):
        gate._join_routed_source_groups(routed, duplicate_source)
    duplicate_group_source = {
        **source,
        "eligible_group_provenance": [
            source["eligible_group_provenance"][0],
            {**source["eligible_group_provenance"][1], "source_group": 4},
        ],
    }
    with pytest.raises(ValueError, match="ambiguous"):
        gate._join_routed_source_groups(routed, duplicate_group_source)
    for indices in ([4, 4], [True, 8], [-1, 8], [8, 4]):
        with pytest.raises(ValueError, match="ambiguous"):
            gate._join_routed_source_groups(routed, {**source, "eligible_groups": indices})
    for changed in (
        {**source["eligible_group_provenance"][0], "source_node_ids": ["b", "a"]},
        {**source["eligible_group_provenance"][0], "source_node_ids": ["a", 1]},
    ):
        with pytest.raises(ValueError, match="provenance"):
            gate._join_routed_source_groups(
                routed,
                {**source, "eligible_group_provenance": [changed, source["eligible_group_provenance"][1]]},
            )


def test_routed_calls_may_reuse_only_identical_kernel_signatures():
    from copy import deepcopy

    row = {"symbol": "shared_kernel", "parallel": [8, 16], "reduction": [32], "dtypes": ["i8", "i8", "i32"]}
    emitted = {"routed": [{**row, "group": 3}, {**row, "group": 7}], "signatures": {"shared_kernel": [8, 16, 32]}}
    assert gate._routed_kernel_symbols(emitted) == {"shared_kernel"}
    for key, value in (
        ("parallel", [16, 8]),
        ("reduction", [16]),
        ("dtypes", ["i8", "i8", "f32"]),
        ("symbol", ""),
        ("symbol", "../outside"),
        ("symbol", ["shared_kernel"]),
        ("parallel", [True, 16]),
        ("dtypes", []),
    ):
        changed = deepcopy(emitted)
        changed["routed"][1][key] = value
        with pytest.raises(ValueError, match="kernel"):
            gate._routed_kernel_symbols(changed)
    for signatures in ({}, {"shared_kernel": [8, 16, 16]}, {"shared_kernel": [8, 16, 32], "unused": [1]}):
        with pytest.raises(ValueError, match="kernel"):
            gate._routed_kernel_symbols({**emitted, "signatures": signatures})


def test_eligible_source_root_requires_unambiguous_typed_provenance():
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    root = SimpleNamespace(
        attributes={
            "prov.region_id": StringAttr("region-a"),
            "prov.source_node_ids": ArrayAttr([StringAttr("node-b"), StringAttr("node-a")]),
        }
    )
    assert gate._eligible_source_identity(root) == ("region-a", ("node-a", "node-b"))
    for attributes in (
        {},
        {**root.attributes, "prov.region_id": StringAttr("")},
        {**root.attributes, "prov.source_node_ids": ArrayAttr([])},
        {**root.attributes, "prov.source_node_ids": ArrayAttr([StringAttr("node-a"), StringAttr("node-a")])},
        {**root.attributes, "prov.source_node_ids": ArrayAttr([StringAttr("node-a"), StringAttr("")])},
    ):
        with pytest.raises(ValueError, match="provenance"):
            gate._eligible_source_identity(SimpleNamespace(attributes=attributes))


def test_multi_program_capture_refuses_slice_and_extra_stage(tmp_path):
    capture = tmp_path / "capture"
    stage_root = capture / "stages"
    names = ("prefix", "decode")
    rows = []
    for name in names:
        stage = stage_root / name
        stage.mkdir(parents=True)
        (stage / "capture_receipt.json").write_text("{}")
        rows.append({"name": name, "ok": True, "opaque": 0, "receipt_sha256": _sha(stage / "capture_receipt.json")})
    contract = {
        "version": 2,
        "stages": list(names),
        "programs": [{"name": name, "bundle": f"stages/{name}"} for name in names],
        "bindings": [{"name": "state"}],
    }
    (capture / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    receipt = {
        "schema": "merlin.model_session_capture.v1",
        "session_contract_sha256": _sha(capture / "session_contract.yaml"),
        "programs": rows,
    }
    (capture / "session-receipt.json").write_text(json.dumps(receipt))
    assert [name for name, _ in gate._captured_programs(capture, names)] == list(names)
    with pytest.raises(ValueError, match="roster"):
        gate._captured_programs(capture, ("decode",))
    (stage_root / "unlisted").mkdir()
    with pytest.raises(ValueError, match="extra"):
        gate._captured_programs(capture, names)


def test_single_complete_capture_may_carry_version_one_execution_contract(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}")
    (capture / "capture_receipt.json").write_text("{}")
    (capture / "session_contract.yaml").write_text("version: 1\nkind: image_stream\nstages: [classify]\n")
    assert gate._captured_programs(capture, ("model",)) == [("model", capture)]
    (capture / "session_contract.yaml").write_text("version: 2\nstages: [classify]\n")
    with pytest.raises(ValueError, match="not version 1"):
        gate._captured_programs(capture, ("model",))
    (capture / "session_contract.yaml").unlink()
    (capture / "stages").mkdir()
    with pytest.raises(ValueError, match="unbound stage"):
        gate._captured_programs(capture, ("model",))


def test_unknown_support_required_operation_is_not_waived():
    assert gate._noncompute_support({"disposition": "support_required", "mlir_operation": "tensor.extract"})
    assert not gate._noncompute_support({"disposition": "support_required", "mlir_operation": "linalg.copy"})
    with pytest.raises(ValueError, match="audited lowering class"):
        gate._noncompute_support({"disposition": "support_required", "mlir_operation": "mystery.perform"})
    with pytest.raises(ValueError, match="audited lowering class"):
        gate._noncompute_support({"disposition": "support_required", "mlir_operation": "tensor.future_unknown"})


def test_only_source_proved_contraction_zero_init_fill_is_support():
    from merlin.common import mlir_query as mq

    def proven(value: str, consumer: str) -> bool:
        source = f"""module {{
          func.func @f(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {{
            %e = tensor.empty() : tensor<2x2xf32>
            %z = arith.constant {value} : f32
            %i = linalg.fill ins(%z : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
            {consumer}
          }}
        }}"""
        parsed = tuple(mq.walk(mq.parse(source)))
        ordinal = next(i for i, op in enumerate(parsed) if op.name == "linalg.fill")
        row = {"disposition": "support_required", "mlir_operation": "linalg.fill", "ordinals": [ordinal], "count": 1}
        return gate._contraction_zero_init_fill(row, parsed)

    matmul = """%r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
                   outs(%i : tensor<2x2xf32>) -> tensor<2x2xf32>
                return %r : tensor<2x2xf32>"""
    assert proven("0.000000e+00", matmul)
    assert not proven("1.000000e+00", matmul)
    assert not proven("0.000000e+00", "return %i : tensor<2x2xf32>")


def _transpose_module(*, permutation="1, 0", output="3x2xi8", body=None, operation="linalg.transpose"):
    from merlin.common import mlir_query as mq

    if body is None:
        body = '"linalg.yield"(%a) : (i8) -> ()'
    properties = (
        f"permutation = array<i64: {permutation}>"
        if operation == "linalg.transpose"
        else "indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, "
        "affine_map<(d0, d1) -> (d0, d1)>], "
        "iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "
        "operandSegmentSizes = array<i32: 1, 1>"
    )
    text = f"""module {{
  func.func @f(%x: tensor<2x3xi8>, %init: tensor<{output}>) -> tensor<{output}> {{
    %t = "{operation}"(%x, %init) <{{{properties}}}> ({{
      ^bb0(%a: i8, %b: i8):
        {body}
    }}) : (tensor<2x3xi8>, tensor<{output}>) -> tensor<{output}>
    return %t : tensor<{output}>
  }}
}}
"""
    return mq.parse(text)


def test_transpose_data_support_joins_every_source_ordinal_and_refuses_computation():
    from merlin.common import mlir_query as mq

    module = _transpose_module()
    row = {"disposition": "support_required", "mlir_operation": "linalg.transpose", "ordinals": [2], "count": 1}
    inventory = {
        "n_operations": 5,
        "capture_sha256": "a" * 64,
        "capture_normalization": {"output_sha256": "b" * 64},
        "signatures": [row],
    }
    proof = gate._transpose_data_support(module, inventory, raw_sha256="a" * 64, normalized_sha256="b" * 64)
    assert proof["count"] == 1
    assert proof["ordinals"] == [2]
    assert proof["raw_source_sha256"] == "a" * 64
    assert proof["normalized_source_sha256"] == "b" * 64
    assert proof["status"] == "source_structural_data_support_pending_build"
    assert mq.op_name(tuple(mq.walk(module))[2]) == "linalg.transpose"
    for invalid in (
        {"permutation": "0, 0"},
        {"permutation": "0, 1"},
        {"output": "2x3xi8"},
        {"output": "?x2xi8"},
        {"output": "3x2xf32"},
        {"body": '"linalg.yield"(%b) : (i8) -> ()'},
        {"body": '"linalg.yield"(%a) {unknown.semantic = 1 : i32} : (i8) -> ()'},
        {"body": '%sum = "arith.addi"(%a, %b) : (i8, i8) -> i8\n        "linalg.yield"(%sum) : (i8) -> ()'},
        {"operation": "linalg.generic"},
    ):
        with pytest.raises(ValueError):
            bad_module = _transpose_module(**invalid)
            bad_op = tuple(mq.walk(bad_module))[2]
            gate._verify_transpose_data_support(bad_op, ordinal=2)
    from xdsl.dialects.builtin import StringAttr

    attr_module = _transpose_module()
    attr_op = tuple(mq.walk(attr_module))[2]
    attr_op.attributes["unknown.semantic"] = StringAttr("changed")
    attr_module.verify()
    with pytest.raises(ValueError, match="semantic"):
        gate._verify_transpose_data_support(attr_op, ordinal=2)
    for nested in (False, True):
        property_module = _transpose_module()
        property_op = tuple(mq.walk(property_module))[2]
        target = list(property_op.regions[0].blocks[0].ops)[0] if nested else property_op
        target.properties["unknown.semantic"] = StringAttr("changed")
        with pytest.raises(ValueError):
            gate._verify_transpose_data_support(property_op, ordinal=2)
    for invalid_row in (
        {**row, "disposition": "host_required"},
        {**row, "ordinals": [3]},
        {**row, "ordinals": [2, 2], "count": 2},
        {**row, "ordinals": [], "count": 0},
    ):
        with pytest.raises(ValueError):
            gate._transpose_data_support(
                module,
                {**inventory, "signatures": [invalid_row]},
                raw_sha256="a" * 64,
                normalized_sha256="b" * 64,
            )
    with pytest.raises(ValueError, match="normalized parsed source"):
        gate._transpose_data_support(
            module,
            {**inventory, "capture_normalization": {"output_sha256": "c" * 64}},
            raw_sha256="a" * 64,
            normalized_sha256="b" * 64,
        )


def _generic_copy_module(
    *,
    input_shape="1x4x1x8",
    output_shape="1x4x3x8",
    input_map="d0, d1, 0, d3",
    output_map="d0, d1, d2, d3",
    dtype="f32",
    output_dtype=None,
    body=None,
    iterators=None,
    segments="1, 1",
    attrs="",
):
    from merlin.common import mlir_query as mq

    if body is None:
        body = f'"linalg.yield"(%a) : ({dtype}) -> ()'
    if output_dtype is None:
        output_dtype = dtype
    if iterators is None:
        iterators = ", ".join(["#linalg.iterator_type<parallel>"] * 4)
    domain = ", ".join(f"d{i}" for i in range(len(output_shape.split("x"))))
    input_type = f"tensor<{input_shape}x{dtype}>"
    output_type = f"tensor<{output_shape}x{output_dtype}>"
    properties = (
        f"indexing_maps = [affine_map<({domain}) -> ({input_map})>, "
        f"affine_map<({domain}) -> ({output_map})>], "
        f"iterator_types = [{iterators}], operandSegmentSizes = array<i32: {segments}>"
    )
    return mq.parse(
        f"""module {{
  func.func @f(%x: {input_type}, %init: {output_type}) -> {output_type} {{
    %t = "linalg.generic"(%x, %init) <{{{properties}}}> ({{
      ^bb0(%a: {dtype}, %b: {output_dtype}):
        {body}
    }}) {attrs}: ({input_type}, {output_type}) -> {output_type}
    return %t : {output_type}
  }}
}}"""
    )


@pytest.mark.parametrize(
    "input_shape,input_map,accepted",
    [
        ("1x2x12x12", "d3, d0, d4 * 4 + d1, d5 * 4 + d2", True),
        ("1x2x11x12", "d3, d0, d4 * 4 + d1, d5 * 4 + d2", False),
        ("1x2x12x12", "d3, d0, d4 * 4 + d1 + 1, d5 * 4 + d2", False),
        ("1x2x12x12", "d3, d0, d4 * 4 + d1 - 1, d5 * 4 + d2", False),
        ("1x2x12x12", "d3, d0, d4 floordiv 2, d5 * 4 + d2", False),
    ],
)
def test_generic_copy_support_proves_linear_tiled_movement_bounds(input_shape, input_map, accepted):
    from merlin_experiments.phase1.feedback.private_data_movement import verify_generic_copy

    from merlin.common import mlir_query as mq

    # Independent geometry: patch coordinates are bounded by the identity output domain.
    module = _generic_copy_module(
        input_shape=input_shape,
        output_shape="2x4x4x1x3x3",
        input_map=input_map,
        output_map="d0, d1, d2, d3, d4, d5",
        iterators=", ".join(["#linalg.iterator_type<parallel>"] * 6),
    )
    op = tuple(mq.walk(module))[2]
    if accepted:
        verify_generic_copy(op, ordinal=2)
    else:
        with pytest.raises(ValueError):
            verify_generic_copy(op, ordinal=2)


def test_generic_copy_support_requires_typed_singleton_projection_and_exact_join():
    from merlin_experiments.phase1.feedback.private_data_movement import prove_generic_copy_source

    from merlin.common import mlir_query as mq

    module = _generic_copy_module()
    row = {
        "disposition": "support_required",
        "semantic_family": "movement",
        "mlir_operation": "linalg.generic",
        "ordinals": [2],
        "count": 1,
    }
    inventory = {
        "n_operations": 5,
        "capture_sha256": "a" * 64,
        "capture_normalization": {"output_sha256": "b" * 64},
        "signatures": [row],
    }
    proof = prove_generic_copy_source(module, inventory, raw_sha256="a" * 64, normalized_sha256="b" * 64)
    assert proof["count"] == 1 and proof["ordinals"] == [2]
    assert proof["status"] == "source_structural_data_support_pending_build"
    assert mq.op_name(tuple(mq.walk(module))[2]) == "linalg.generic"
    for valid in (
        {"input_shape": "1x4x3x8", "input_map": "d0, d1, d2, d3"},
        {"dtype": "i1"},
    ):
        other = _generic_copy_module(**valid)
        assert prove_generic_copy_source(other, inventory, raw_sha256="a" * 64, normalized_sha256="b" * 64)[
            "ordinals"
        ] == [2]
    for invalid in (
        {"input_shape": "1x4x2x8"},
        {"output_dtype": "i1"},
        {"input_map": "d0, d1, 1, d3"},
        {"input_map": "d0, d1, d2, d3"},
        {"input_map": "d0, d1, d1, d3"},
        {"input_map": "d0, d1, d2 + 1, d3"},
        {"output_map": "d0, d1, d3, d2"},
        {"segments": "2, 0"},
        {"iterators": ", ".join(["#linalg.iterator_type<parallel>"] * 3 + ["#linalg.iterator_type<reduction>"])},
        {"attrs": "{fastmath = #arith.fastmath<fast>} "},
        {"body": '"linalg.yield"(%b) : (f32) -> ()'},
        {"body": '%s = "arith.addf"(%a, %b) : (f32, f32) -> f32\n        "linalg.yield"(%s) : (f32) -> ()'},
    ):
        with pytest.raises(ValueError):
            other = _generic_copy_module(**invalid)
            prove_generic_copy_source(other, inventory, raw_sha256="a" * 64, normalized_sha256="b" * 64)
    for invalid_row in (
        {**row, "ordinals": []},
        {**row, "ordinals": [3]},
        {**row, "ordinals": [2, 2], "count": 2},
        {**row, "disposition": "host_required"},
        {**row, "semantic_family": "elementwise_map"},
    ):
        with pytest.raises(ValueError):
            prove_generic_copy_source(
                module,
                {**inventory, "signatures": [invalid_row]},
                raw_sha256="a" * 64,
                normalized_sha256="b" * 64,
            )
    with pytest.raises(ValueError, match="normalized parsed source"):
        prove_generic_copy_source(module, inventory, raw_sha256="a" * 64, normalized_sha256="c" * 64)


def test_source_obligations_prove_copy_before_hardware_eligibility(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_data_movement as movement

    from merlin.common import mlir_query as mq
    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import application_inventory as ai
    from merlin.targetgen import eligibility
    from merlin.targetgen.compute_units import SemanticCapability
    from merlin.xdsl_dialects.lowering import compute_groups as cg

    capture = tmp_path / "capture"
    capture.mkdir()
    source = capture / "model.mlir"
    provenance = '{prov.family = "movement", prov.op = "aten.expand.default"}'
    source.write_text(str(_generic_copy_module(attrs=provenance)))
    (capture / "frontend-trace.json").write_text(json.dumps({"graphs": {"prepared": {"nodes": []}}}))
    monkeypatch.setattr(
        ai, "verify_capture_receipt", lambda _source: {"status": "verified_materialized", "receipt_sha256": "a" * 64}
    )
    monkeypatch.setattr(bucketize, "verify_capture_receipt", ai.verify_capture_receipt)
    monkeypatch.setattr(cg, "form_groups", _synthetic_host_groups)
    monkeypatch.setattr(cg, "plan", lambda _module, _target, *, groups: {})
    monkeypatch.setattr(cg, "require_explained", lambda _plan: None)
    monkeypatch.setattr(
        eligibility,
        "capability_map_from_contract",
        lambda _contract: {"movement": SemanticCapability("movement", dtypes=("int8",))},
    )
    signature_row = {
        "mlir_operation": "linalg.generic",
        "semantic_family": "movement",
        "disposition": "support_required",
        "count": 1,
        "ordinals": [2],
    }

    def inventory(path, _target, _cap_map, **_kwargs):
        normalized, _ = normalize_capture_mlir(path.read_text())
        parsed = tuple(mq.walk(mq.parse(normalized)))
        return {
            "capture_sha256": file_sha256(path),
            "capture_normalization": {"output_sha256": hashlib.sha256(normalized.encode()).hexdigest()},
            "n_operations": len(parsed),
            "signatures": [
                dict(signature_row)
                if ordinal == 2
                else {
                    "mlir_operation": mq.op_name(op),
                    "disposition": "structural" if ordinal in {0, 1, len(parsed) - 1} else "component",
                    "count": 1,
                    "ordinals": [ordinal],
                }
                for ordinal, op in enumerate(parsed)
            ],
        }

    monkeypatch.setattr(ai, "_application_operation_inventory", inventory)
    proof = gate._source_obligations(capture, "fixture", {}, {}, {})
    assert proof["eligible_groups"] == []
    assert proof["n_support_lowering_operations"] == 1
    assert proof["generic_copy_data_support"]["ordinals"] == [2]
    assert proof["generic_copy_data_support"]["status"] == movement.PENDING

    source.write_text(
        str(
            _generic_copy_module(
                attrs=provenance,
                body='%s = "arith.addf"(%a, %b) : (f32, f32) -> f32\n        "linalg.yield"(%s) : (f32) -> ()',
            )
        )
    )
    with pytest.raises(ValueError, match="generic copy"):
        gate._source_obligations(capture, "fixture", {}, {}, {})
    signature_row["semantic_family"] = "elementwise_map"
    signature_row["disposition"] = "host_required"
    monkeypatch.setattr(
        eligibility,
        "capability_map_from_contract",
        lambda _contract: {"movement": SemanticCapability("movement", dtypes=("fp32",))},
    )
    with pytest.raises(gate.source_demand.SourceDemandIncomplete):
        gate._source_obligations(capture, "fixture", {}, {}, {})


def test_source_inventory_ordinal_join_requires_complete_exact_parsed_operations():
    from merlin_experiments.phase1.feedback.private_data_movement import source_inventory_by_ordinal

    from merlin.common import mlir_query as mq

    parsed = tuple(mq.walk(_generic_copy_module()))
    rows = [{"mlir_operation": mq.op_name(op), "count": 1, "ordinals": [ordinal]} for ordinal, op in enumerate(parsed)]
    inventory = {
        "capture_sha256": "a" * 64,
        "capture_normalization": {"output_sha256": "b" * 64},
        "n_operations": len(parsed),
        "signatures": rows,
    }
    assert set(source_inventory_by_ordinal(parsed, inventory, "a" * 64, "b" * 64)) == set(range(len(parsed)))
    for invalid in (
        rows[:-1],
        [*rows, dict(rows[0])],
        [*rows[:2], {**rows[2], "mlir_operation": "arith.addf"}, *rows[3:]],
    ):
        with pytest.raises(ValueError, match="source inventory"):
            source_inventory_by_ordinal(parsed, {**inventory, "signatures": invalid}, "a" * 64, "b" * 64)


def test_source_obligations_joins_typed_hardware_exclusion_before_host_check(tmp_path, monkeypatch):
    from merlin.targetgen import application_inventory as ai
    from merlin.targetgen import operation_accounting as oa
    from merlin.xdsl_dialects.lowering import compute_groups as cg

    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text(
        """module {
  func.func @f(%init: tensor<4xi64>) -> tensor<4xi64> {
    %t = "linalg.generic"(%init) <{indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
      ^bb0(%old: i64):
        %idx = "linalg.index"() <{dim = 0 : i64}> : () -> index
        %wide = "arith.index_cast"(%idx) : (index) -> i64
        %c = "arith.constant"() <{value = 1 : i64}> : () -> i64
        %value = "arith.addi"(%wide, %c) : (i64, i64) -> i64
        "linalg.yield"(%value) : (i64) -> ()
    }) {prov.family = "movement", prov.op = "arange", prov.aten = "aten.arange.start_step"} :
      (tensor<4xi64>) -> tensor<4xi64>
    return %t : tensor<4xi64>
  }
}"""
    )
    monkeypatch.setattr(
        ai, "verify_capture_receipt", lambda _source: {"status": "verified_materialized", "receipt_sha256": "a" * 64}
    )
    (capture / "frontend-trace.json").write_text(json.dumps({"graphs": {"prepared": {"nodes": []}}}))
    monkeypatch.setattr(bucketize, "verify_capture_receipt", ai.verify_capture_receipt)
    monkeypatch.setattr(cg, "form_groups", _synthetic_host_groups)
    monkeypatch.setattr(cg, "plan", lambda _module, _target, *, groups: {})
    monkeypatch.setattr(cg, "require_explained", lambda _plan: None)
    capability = {
        "name": "fixture",
        "compute_units": [
            {
                "name": "v",
                "kind": "vector",
                "dtypes": ["int8"],
                "ops": ["add"],
                "semantic_capabilities": [
                    {"family": "elementwise_map", "dtypes": ["int8"]},
                    {"family": "movement", "dtypes": ["int8"]},
                ],
            }
        ],
    }
    with pytest.raises(gate.SourceAdmissionError) as failure:
        gate._source_obligations(
            capture,
            "fixture",
            {"operations": []},
            capability,
            {},
            facts_sha256="a" * 64,
            numeric_policy_sha256="b" * 64,
        )
    assert str(failure.value) == "source has 1 unaccounted or unjustified operation signature(s)"
    private = failure.value._private_unresolved
    assert len(private) == 1
    assert private[0]["row"]["frontend_op"] == "aten.arange.start_step"
    assert private[0]["row"]["ordered_operand_types"]
    assert private[0]["admission"]["observed_admission_signature"]["ordered_result_dtypes"] == ["i64"]
    assert private[0]["admission"]["host_admission"]["status"] != "admitted"
    assert private[0]["reason"] == "host operation lacks exact reviewed admission"
    private[0]["row"]["frontend_op"] = "caller mutation"
    private[0]["admission"]["host_admission"]["status"] = "caller mutation"
    assert failure.value._private_unresolved[0]["row"]["frontend_op"] == "aten.arange.start_step"
    assert failure.value._private_unresolved[0]["admission"]["host_admission"]["status"] != "caller mutation"
    assert "aten.arange" not in str(failure.value)
    assert gate._public_failure_reason(failure.value) == (
        "ValueError: source has 1 unaccounted or unjustified operation signature(s)"
    )
    original_admission = oa.admit_operation_row
    hardware_status = "unsupported"
    missing_precision = False
    unknown_precision = False

    def reviewed_host(row, **kwargs):
        result = original_admission(row, **kwargs)
        if row.get("frontend_op") == "aten.arange.start_step":
            result["host_admission"] = {"status": "admitted", "reviewed": True}
            result["hardware_admission"]["status"] = hardware_status
            if missing_precision:
                result["observed_admission_signature"]["operand_dtype"] = None
            if unknown_precision:
                result["hardware_admission"]["refusal"] = "result_dtype_unknown"
        return result

    monkeypatch.setattr(oa, "admit_operation_row", reviewed_host)
    # This fixture isolates the independent hardware exclusion screen; the
    # arange source/selected-width join has its own real captured-IR tests.
    monkeypatch.setattr(gate.arange_support, "record", lambda *_args: None)
    proof = gate._source_obligations(
        capture,
        "fixture",
        {"operations": []},
        capability,
        {},
        facts_sha256="a" * 64,
        numeric_policy_sha256="b" * 64,
    )
    assert proof["eligible_groups"] == []
    for hardware_status in ("unknown", "admitted"):
        with pytest.raises(ValueError, match="unknown hardware eligibility"):
            gate._source_obligations(
                capture,
                "fixture",
                {"operations": []},
                capability,
                {},
                facts_sha256="a" * 64,
                numeric_policy_sha256="b" * 64,
            )
    hardware_status = "unsupported"
    missing_precision = True
    with pytest.raises(ValueError, match="unknown hardware eligibility"):
        gate._source_obligations(
            capture,
            "fixture",
            {"operations": []},
            capability,
            {},
            facts_sha256="a" * 64,
            numeric_policy_sha256="b" * 64,
        )
    missing_precision = False
    unknown_precision = True
    with pytest.raises(ValueError, match="unknown hardware eligibility"):
        gate._source_obligations(
            capture,
            "fixture",
            {"operations": []},
            capability,
            {},
            facts_sha256="a" * 64,
            numeric_policy_sha256="b" * 64,
        )


def test_source_admission_error_copies_mutable_input_records():
    records = [{"row": {"ordered_operand_types": ["i64"]}, "admission": {"host_admission": {"status": "unknown"}}}]
    failure = gate.SourceAdmissionError(records)
    records[0]["row"]["ordered_operand_types"][0] = "f32"
    records[0]["admission"]["host_admission"]["status"] = "admitted"
    assert failure._private_unresolved[0]["row"]["ordered_operand_types"] == ["i64"]
    assert failure._private_unresolved[0]["admission"]["host_admission"]["status"] == "unknown"


def test_selected_input_roster_requires_transitive_dependency_exactly():
    selected = [
        {"role": "checkpoint", "kind": "tree", "guest_member": "hf-hub/main", "tree": {"sha256": "1" * 64}},
        {"role": "input", "kind": "file", "guest_member": "corpus/input.npz", "sha256": "2" * 64},
        {"role": "input", "kind": "tree", "guest_member": "hf-hub/transitive", "tree": {"sha256": "3" * 64}},
    ]
    expected = [
        {"role": "checkpoint", "kind": "tree", "guest_member": "hf-hub/main", "sha256": "1" * 64},
        {"role": "input", "kind": "file", "guest_member": "corpus/input.npz", "sha256": "2" * 64},
        {"role": "input", "kind": "tree", "guest_member": "hf-hub/transitive", "sha256": "3" * 64},
    ]
    gate._require_selected_input_bindings({"selected_input_bindings": expected}, {"selected_inputs": selected})
    with pytest.raises(ValueError, match="omits or changes"):
        gate._require_selected_input_bindings({"selected_input_bindings": expected[:-1]}, {"selected_inputs": selected})
    with pytest.raises(ValueError, match="omits or changes"):
        gate._require_selected_input_bindings(
            {"selected_input_bindings": [*expected[:-1], {**expected[-1], "sha256": "4" * 64}]},
            {"selected_inputs": selected},
        )


def test_static_build_receipt_binds_board_without_execution_claim(tmp_path):
    catalog = tmp_path / "board-catalog.yaml"
    catalog.write_text("selected board\n")
    dts = tmp_path / "selected.dts"
    dts.write_text("actual source DTS\n")
    inputs = {
        "target": "fixture",
        "board": "large-static-board",
        "board_catalog": str(catalog),
        "board_catalog_sha256": _sha(catalog),
        "dts": str(dts),
        "dts_sha256": _sha(dts),
        "run": "none",
    }
    kwargs = {
        "target": "fixture",
        "board": "large-static-board",
        "catalog": catalog,
        "dts": dts,
    }
    gate._require_static_build_inputs({"inputs": inputs}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "run": "gsim"}}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "dts_sha256": "3" * 64}}, **kwargs)
    with pytest.raises(ValueError, match="another board"):
        gate._require_static_build_inputs({"inputs": {**inputs, "dts": "/unrelated/other.dts"}}, **kwargs)


@pytest.mark.parametrize(
    "changed",
    [
        "capture",
        "host",
        "candidate",
        "arena",
        "external_elf",
        "missing_sidecar",
        "external_sidecar",
        "malformed_sidecar",
    ],
)
def test_prebuilt_diagnostic_rehashes_inputs_and_confines_outputs(tmp_path, changed):
    capture, host, candidate = (tmp_path / name for name in ("capture", "host", "candidate"))
    for path in (capture, host, candidate):
        path.mkdir()
        (path / "input.txt").write_text(path.name)
    build = tmp_path / "old-build"
    (build / "build").mkdir(parents=True)
    elf = build / "build" / "model.elf"
    sidecar = build / "build" / "device.json"
    elf.write_bytes(b"ELF")
    sidecar.write_text("{}")
    receipt_path = build / "baremetal_model.json"
    host_tree = gate.strict_tree_sha256(host)
    candidate_tree = gate.strict_tree_sha256(candidate)
    receipt = {
        "schema": "merlin.baremetal-saved-model.v1",
        "inputs": {
            "capture": str(capture.resolve()),
            "package": str(host.resolve()),
            "capture_tree": gate.strict_tree_sha256(capture),
            "package_tree": host_tree,
            "arena_mb": 8,
            "reference_file": None,
            "rtl_facts": None,
            "device": {"name": "fixture", "package": str(candidate.resolve()), "package_tree": candidate_tree},
        },
        "output": {"elf": str(elf), "device_sidecar": {"path": str(sidecar)}},
    }
    receipt_path.write_text(json.dumps(receipt))
    kwargs = {
        "capture": capture,
        "host_package": host,
        "candidate": candidate,
        "host_package_tree": host_tree,
        "candidate_tree": candidate_tree,
        "arena_mb": 8,
        "target": "fixture",
        "device_selected": True,
    }
    assert load_diagnostic_receipt(receipt_path, **kwargs) == receipt
    if changed in {"capture", "host", "candidate"}:
        (locals()[changed] / "input.txt").write_text("changed")
        if changed == "candidate":
            kwargs["candidate_tree"] = gate.strict_tree_sha256(candidate)
    elif changed == "arena":
        kwargs["arena_mb"] = 9
    elif changed == "external_elf":
        external = tmp_path / "external.elf"
        external.write_bytes(b"ELF")
        receipt["output"]["elf"] = str(external)
        receipt_path.write_text(json.dumps(receipt))
    elif changed == "external_sidecar":
        external = tmp_path / "external.device.json"
        external.write_text("{}")
        receipt["output"]["device_sidecar"]["path"] = str(external)
        receipt_path.write_text(json.dumps(receipt))
    elif changed == "malformed_sidecar":
        receipt["output"]["device_sidecar"] = "not a record"
        receipt_path.write_text(json.dumps(receipt))
    else:
        sidecar.unlink()
    with pytest.raises(ValueError, match="prebuilt diagnostic receipt"):
        load_diagnostic_receipt(receipt_path, **kwargs)


def test_prebuilt_diagnostic_selected_rtl_facts_are_path_and_byte_bound(tmp_path):
    capture, host, candidate = (tmp_path / name for name in ("capture", "host", "candidate"))
    for path in (capture, host, candidate):
        path.mkdir()
        (path / "input.txt").write_text(path.name)
    facts = tmp_path / "selected-facts.json"
    facts.write_text('{"selected":"first"}')
    build = tmp_path / "old-build"
    (build / "build").mkdir(parents=True)
    elf = build / "build" / "model.elf"
    sidecar = build / "build" / "device.json"
    elf.write_bytes(b"ELF")
    sidecar.write_text("{}")
    receipt_path = build / "baremetal_model.json"
    receipt = {
        "schema": "merlin.baremetal-saved-model.v1",
        "inputs": {
            "capture": str(capture.resolve()),
            "package": str(host.resolve()),
            "capture_tree": gate.strict_tree_sha256(capture),
            "package_tree": gate.strict_tree_sha256(host),
            "arena_mb": 8,
            "reference_file": None,
            "rtl_facts": str(facts.resolve()),
            "device": {
                "name": "fixture",
                "package": str(candidate.resolve()),
                "package_tree": gate.strict_tree_sha256(candidate),
            },
        },
        "output": {"elf": str(elf), "device_sidecar": {"path": str(sidecar)}},
    }
    receipt_path.write_text(json.dumps(receipt))
    kwargs = {
        "capture": capture,
        "host_package": host,
        "candidate": candidate,
        "host_package_tree": gate.strict_tree_sha256(host),
        "candidate_tree": gate.strict_tree_sha256(candidate),
        "arena_mb": 8,
        "target": "fixture",
        "device_selected": True,
        "selected_rtl_facts": {"path": str(facts.resolve()), "sha256": _sha(facts)},
    }
    assert load_diagnostic_receipt(receipt_path, **kwargs) == receipt
    legacy = deepcopy(receipt)
    legacy["inputs"]["rtl_facts"] = None
    receipt_path.write_text(json.dumps(legacy))
    assert load_diagnostic_receipt(receipt_path, **kwargs) == legacy
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="RTL facts"):
        load_diagnostic_receipt(receipt_path, **{**kwargs, "selected_rtl_facts": None})
    with pytest.raises(ValueError, match="RTL facts"):
        load_diagnostic_receipt(
            receipt_path,
            **{**kwargs, "selected_rtl_facts": {"path": str(tmp_path / "other.json"), "sha256": _sha(facts)}},
        )
    receipt["inputs"]["rtl_facts_identity"] = {"sha256": "0" * 64}
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="RTL facts"):
        load_diagnostic_receipt(receipt_path, **kwargs)
    del receipt["inputs"]["rtl_facts_identity"]
    receipt_path.write_text(json.dumps(receipt))
    facts.write_text('{"selected":"changed"}')
    with pytest.raises(ValueError, match="RTL facts"):
        load_diagnostic_receipt(receipt_path, **kwargs)
    facts.unlink()
    with pytest.raises(ValueError, match="RTL facts"):
        load_diagnostic_receipt(receipt_path, **kwargs)


def test_prebuilt_subset_cannot_be_completed_as_full_gate(tmp_path):
    spec = tmp_path / "private.yaml"
    spec.write_text(yaml.safe_dump({"schema": gate.SCHEMA, "target": "fixture", "models": [{"id": "only"}]}))
    with pytest.raises(ValueError, match="required target roster"):
        gate.run(
            tmp_path / "candidate",
            spec,
            target="fixture",
            required_models=("only", "second"),
            required_programs={"only": ("model",), "second": ("model",)},
            loader_env_requirements={"only": {}, "second": {}},
            out=tmp_path / "out",
            diagnostic_model="only",
            prebuilt_receipts={"model": tmp_path / "old"},
        )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.txt").write_text("fixture")
    original_roster = ("only", "second", "third")
    spec.write_text(
        yaml.safe_dump(
            {"schema": gate.SCHEMA, "target": "fixture", "models": [{"id": name} for name in original_roster]}
        )
    )
    diagnostic = gate.run(
        candidate,
        spec,
        target="fixture",
        required_models=original_roster,
        required_programs={name: ("model",) for name in original_roster},
        loader_env_requirements={name: {} for name in original_roster},
        out=tmp_path / "diagnostic",
        diagnostic_model="only",
        prebuilt_receipts={"model": tmp_path / "old"},
    )
    assert diagnostic["required_models"] == list(original_roster)
    assert [row["model"] for row in diagnostic["models"]] == ["only"]
    assert diagnostic["passed"] is False and diagnostic["diagnostic_passed"] is False
    assert not gate.complete(
        {"schema": gate.RESULT_SCHEMA, "mode": "prebuilt_diagnostic", "passed": True},
        required_models=("only",),
        required_programs={"only": ("model",)},
        candidate_sha256="a" * 64,
    )


def test_private_gate_report_is_owner_only_and_not_overwritten(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "kernel.txt").write_text("fixture")
    spec = tmp_path / "private.yaml"
    spec.write_text(yaml.safe_dump({"schema": gate.SCHEMA, "target": "fixture", "models": [{"id": "only"}]}))
    kwargs = {
        "target": "fixture",
        "required_models": ("only",),
        "required_programs": {"only": ("model",)},
        "loader_env_requirements": {"only": {}},
        "out": tmp_path / "result",
    }
    with pytest.raises(ValueError, match="run-owned authored-source freeze"):
        gate.run(candidate, spec, **kwargs)
    kwargs.update(diagnostic_model="only", prebuilt_receipts={"model": tmp_path / "absent-receipt.json"})
    result = gate.run(candidate, spec, **kwargs)
    assert result["passed"] is False
    report = kwargs["out"] / "private_full_model_gate.json"
    assert report.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        gate.run(candidate, spec, **kwargs)


def test_shared_postbuild_verifier_rehashes_prebuilt_elf(tmp_path):
    catalog = tmp_path / "board.yaml"
    dts = tmp_path / "host.dts"
    elf = tmp_path / "model.elf"
    for path in (catalog, dts, elf):
        path.write_bytes(path.name.encode())
    stage_tree = {"sha256": "1" * 64, "n_files": 1}
    candidate_tree = {"sha256": "2" * 64, "n_files": 1}
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "synthetic-clang",
        "compiler_resolved": "/synthetic/clang",
        "compiler_sha256": "b" * 64,
        "cross_flags": ["--target=synthetic"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "synthetic test fixture; no compiler executed",
    }
    actual = {
        **selected,
        "effective_pipeline": ",".join(
            f"{name}{{index-bitwidth=64}}"
            for name in (
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            )
        ),
    }
    source = {
        "source_sha256": "3" * 64,
        "normalized_source_sha256": "3" * 64,
        "capture_receipt_sha256": "a" * 64,
        "n_source_operations": 3,
        "n_linalg_regions": 1,
        "eligible_group_provenance": [],
        "accelerator_source_demand": _synthetic_source_demand("3" * 64, "3" * 64, 0),
        "selected_index_observation": deepcopy(selected),
        "n_bounded_control_assertions": 0,
        "n_internal_mask_compactions": 0,
        "direct_return_support": None,
        "transpose_data_support": {"status": "source_structural_data_support_pending_build"},
        "generic_copy_data_support": {"status": "source_structural_data_support_pending_build"},
        "pool_value_support": {"status": pool.PENDING},
        "linalg_host_support": linalg.begin("3" * 64, "3" * 64, 3, deepcopy(selected)),
        "literal_arange_host_support": arange.begin("3" * 64, "3" * 64, 3, deepcopy(selected)),
        integer_reductions.FIELD: integer_reductions.begin("3" * 64, "3" * 64, 3, deepcopy(selected)),
        f32_reductions.FIELD: f32_reductions.begin("3" * 64, "3" * 64, 3, deepcopy(selected)),
        maximum.FIELD: maximum.begin(
            "3" * 64,
            "3" * 64,
            "a" * 64,
            tuple(SimpleNamespace(attributes={}, properties={}) for _ in range(3)),
            deepcopy(selected),
        ),
        prepared_index.FIELD: _empty_prepared_index("3" * 64, "3" * 64, "a" * 64, 3, selected),
        ordered_scan.FIELD: {
            "status": ordered_scan.PENDING,
            "scope": ordered_scan.SCOPE,
            "raw_source_sha256": "3" * 64,
            "normalized_source_sha256": "3" * 64,
            "capture_receipt_sha256": "a" * 64,
            "n_source_operations": 3,
            "selected_index_observation": deepcopy(selected),
            "count": 0,
            "occurrences": [],
            "occurrences_sha256": ordered_scan._digest([]),  # noqa: PLC2701 -- exact empty fixture
            "source_verified": True,
            "admission_verified": True,
            "admissions": [],
            "admissions_sha256": ordered_scan._digest([]),  # noqa: PLC2701 -- empty fixture
        },
        bucketize.FIELD: {
            "status": bucketize.PENDING,
            "scope": bucketize.SCOPE,
            "raw_source_sha256": "3" * 64,
            "normalized_source_sha256": "3" * 64,
            "capture_receipt_sha256": "a" * 64,
            "n_source_operations": 3,
            "selected_index_observation": deepcopy(selected),
            "source_proof": None,
            "source_proof_sha256": None,
            "original_to_prepared_equivalence": "not_proved",
            "boundary_facts": [],
            "expected_ordinals": [],
            "admissions": [],
        },
    }
    receipt = {
        "status": "compiled",
        "execution_route": "host_baseline",
        "inputs": {
            "capture_tree": stage_tree,
            "target": "fixture",
            "board": "selected",
            "board_catalog": str(catalog),
            "board_catalog_sha256": _sha(catalog),
            "dts": str(dts),
            "dts_sha256": _sha(dts),
            "run": "none",
        },
        "output": {"elf": str(elf), "elf_sha256": _sha(elf), "index_lowering": actual},
    }
    kwargs = {
        "program": "model",
        "source": source,
        "stage_tree": stage_tree,
        "package_digest": candidate_tree,
        "target": "fixture",
        "board": "selected",
        "catalog": catalog,
        "dts": dts,
        "device_selected": False,
    }
    with pytest.raises(ValueError, match="producer-bound compilation recipe"):
        gate._verify_compiled_program(receipt, **kwargs)
    kwargs["require_compilation_recipe"] = False
    inspected = gate._verify_compiled_program(receipt, **kwargs)
    assert inspected["elf_sha256"] == _sha(elf)
    assert inspected["compilation_recipe"]["status"] == "unavailable_in_prebuilt_receipt"
    assert integer_reductions.linked_complete(
        source,
        {
            "source_sha256": source["source_sha256"],
            "capture_tree_sha256": stage_tree["sha256"],
            "elf_sha256": _sha(elf),
            "index_lowering": actual,
        },
        candidate_tree["sha256"],
    )
    assert bucketize.linked_source_complete(
        source,
        {
            "source_sha256": source["source_sha256"],
            "capture_tree_sha256": stage_tree["sha256"],
            "elf_sha256": _sha(elf),
            "index_lowering": actual,
        },
        candidate_tree["sha256"],
    )
    elf.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="ELF is absent or changed"):
        gate._verify_compiled_program(receipt, **kwargs)


def test_private_postbuild_recipe_checks_actual_link_inputs(tmp_path):
    """Native tiny build, not model or simulator qualification."""
    from merlin.llvmlower.compilation_recipe import CompilationRecipe

    cc, ar = shutil.which("cc"), shutil.which("ar")
    if cc is None or ar is None:
        pytest.skip("native compiler/archive tools unavailable")
    source, archive, elf = tmp_path / "main.c", tmp_path / "selected.a", tmp_path / "model.elf"
    source.write_text("int main(void) { return 0; }\n")
    subprocess.run([ar, "rc", str(archive)], capture_output=True, check=True)
    recipe = CompilationRecipe(tmp_path, producer=source)
    recipe.run(
        [cc, str(source), str(archive), "-o", str(elf)],
        runner=lambda argv: subprocess.run(argv, capture_output=True),
        inputs=[source, archive],
        output=elf,
    )
    recipe.completed(elf)
    output = {"compilation_recipe": {"path": str(recipe.path), "sha256": _sha(recipe.path)}}
    bound = compilation.verify(output, elf, required=True)
    assert bound["elf_sha256"] == _sha(elf)
    assert compilation.complete({"compilation_recipe": bound, "elf_sha256": _sha(elf)})
    original_elf = _sha(elf)
    archive.write_bytes(archive.read_bytes() + b"postbuild mutation")
    assert _sha(elf) == original_elf
    with pytest.raises(ValueError, match="identity|changed"):
        compilation.verify(output, elf, required=True)
    assert not compilation.complete({"compilation_recipe": bound, "elf_sha256": original_elf})


def test_recipe_derivation_binds_current_sources_and_selected_capture_recipe(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import evidence as evidence_api

    from merlin.targetgen.quant_recipe import digest as recipe_digest

    root = tmp_path / "derived"
    root.mkdir()
    manifest = root / "evidence-manifest.json"
    manifest.write_text("verified by canonical evidence loader")
    body = {"schema": "quant_recipe_v1", "target": "fixture", "status": "derived", "format": "int8", "why": {}}
    body["recipe_sha256"] = recipe_digest(body)
    raw = json.dumps(body, sort_keys=True).encode()
    selected = tmp_path / "selected-recipe.json"
    selected.write_bytes(raw)
    member = f"software/quantization-recipes/{_sha(selected)}.json"
    index = {
        "schema": "merlin.phase0.capture_recipes.v1",
        "target": "fixture",
        "software_review": "reviewed",
        "recipes": [{"path": member, "sha256": _sha(selected), "recipe_sha256": body["recipe_sha256"]}],
    }
    sources = {
        "software-spec": "1" * 64,
        "target-contract": "2" * 64,
        "rtl-facts": "3" * 64,
    }
    host_package = tmp_path / "selected-host"
    host_package.mkdir()
    (host_package / "manifest.yaml").write_text("selected host package\n")
    host_package_digest = gate.strict_tree_sha256(host_package)["sha256"]
    host_document = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": host_package_digest, "dtype_strategy": "int8_w8a8"},
        "operations": [],
        "evidence": {},
    }
    host_path = tmp_path / "host-capabilities.yaml"
    host_path.write_text(yaml.safe_dump(host_document))
    software = {
        "schema": "merlin.software_spec.v1",
        "target": "fixture",
        "status": "reviewed",
        "fact_derivation": {"status": "resolved"},
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
        },
        "operations": [
            {
                "id": "resolved",
                "ops": ["linalg.matmul"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["int8"]},
            }
        ],
    }
    profiles = {
        "reviewed_scalar": {
            "status": "reviewed",
            "resolved_package": str(host_package),
            "package_sha256": host_package_digest,
            "capability_spec_sha256": _sha(host_path),
            "dtype_strategy": "int8_w8a8",
            "capability_spec": host_document,
        }
    }
    exported = SimpleNamespace(
        target="fixture",
        source_snapshots=[
            *(SimpleNamespace(role=role, sha256=sha) for role, sha in sources.items()),
            SimpleNamespace(role="host-capability-spec:reviewed_scalar", sha256=_sha(host_path)),
        ],
        software_spec=software,
        host_capabilities=profiles,
        archived_artifacts=[
            ("software/quantization-recipes.json", json.dumps(index).encode()),
            (member, raw),
        ],
    )
    monkeypatch.setattr(evidence_api, "load_exported_evidence", lambda _root: exported)
    row = {
        "recipe_derivation_root": str(root),
        "recipe_derivation_manifest_sha256": _sha(manifest),
    }
    plan = {"recipe": {"path": str(selected), "sha256": _sha(selected), "recipe_sha256": body["recipe_sha256"]}}
    kwargs = {
        "target": "fixture",
        "software_sha256": sources["software-spec"],
        "capability_sha256": sources["target-contract"],
        "facts_sha256": sources["rtl-facts"],
        "host_capabilities_sha256": _sha(host_path),
    }
    proof, resolved_software, selected_profiles = gate._recipe_derivation(
        tmp_path / "operator.yaml", row, plan, **kwargs
    )
    assert proof["recipe_semantic_sha256"] == body["recipe_sha256"]
    assert proof["evidence_manifest_sha256"] == _sha(manifest)
    assert resolved_software == software
    assert selected_profiles == profiles
    from merlin_experiments.phase1.feedback import private_recipe_variant as variants

    variant = {**body, "framework_capture_policy": {"calibration_count": 1024}}
    variant["recipe_sha256"] = recipe_digest(variant)
    selected.write_text(json.dumps(variant, sort_keys=True))
    review = {
        "schema": variants.SCHEMA,
        "status": "reviewed",
        "reviewed_by": "operator",
        "reason": "Approved calibration policy",
        "base_file_sha256": index["recipes"][0]["sha256"],
        "base_recipe_sha256": body["recipe_sha256"],
        "selected_file_sha256": _sha(selected),
        "selected_recipe_sha256": variant["recipe_sha256"],
        "changes": variants.changes(body, variant),
    }
    variant_plan = {
        "recipe": {"path": str(selected), "sha256": _sha(selected), "recipe_sha256": variant["recipe_sha256"]}
    }
    variant_row = {**row, "capture_recipe_variant": review}
    variant_proof, _, _ = gate._recipe_derivation(tmp_path / "operator.yaml", variant_row, variant_plan, **kwargs)
    assert variant_proof["recipe_semantic_sha256"] == variant["recipe_sha256"]
    assert variant_proof["variant"]["base_recipe_sha256"] == body["recipe_sha256"]
    with pytest.raises(ValueError, match="absent from current-spec derivation"):
        gate._recipe_derivation(tmp_path / "operator.yaml", row, variant_plan, **kwargs)
    with pytest.raises(ValueError, match="selected rtl-facts"):
        gate._recipe_derivation(
            tmp_path / "operator.yaml", variant_row, variant_plan, **{**kwargs, "facts_sha256": "5" * 64}
        )
    selected.write_bytes(raw)
    assert gate._selected_admission_views(
        resolved_software,
        selected_profiles,
        target="fixture",
        host_package=host_package,
        host_package_sha256=host_package_digest,
        host_capabilities_path=host_path,
        host_capabilities_sha256=_sha(host_path),
    ) == (software, profiles)
    relocated = {"reviewed_scalar": {**profiles["reviewed_scalar"], "resolved_package": "/relocated/host"}}
    assert gate._selected_admission_views(
        software,
        relocated,
        target="fixture",
        host_package=host_package,
        host_package_sha256=host_package_digest,
        host_capabilities_path=host_path,
        host_capabilities_sha256=_sha(host_path),
    ) == (software, relocated)
    for mutation in (
        {"package_sha256": "4" * 64},
        {"capability_spec": {**host_document, "evidence": {"foreign": True}}},
        {"dtype_strategy": "bf16"},
    ):
        foreign = {"reviewed_scalar": {**profiles["reviewed_scalar"], **mutation}}
        with pytest.raises(ValueError, match="selected host profile"):
            gate._selected_admission_views(
                software,
                foreign,
                target="fixture",
                host_package=host_package,
                host_package_sha256=host_package_digest,
                host_capabilities_path=host_path,
                host_capabilities_sha256=_sha(host_path),
            )
    authored = {key: value for key, value in software.items() if key != "fact_derivation"}
    authored["operations"] = [
        *software["operations"],
        {"id": "unknown", "ops": ["linalg.unknown"], "placement": "unknown", "signature": {"operand_dtypes": ["int8"]}},
    ]
    admitted, _ = gate._selected_admission_views(
        authored,
        profiles,
        target="fixture",
        host_package=host_package,
        host_package_sha256=host_package_digest,
        host_capabilities_path=host_path,
        host_capabilities_sha256=_sha(host_path),
    )
    assert admitted["operations"][-1]["placement"] == "unknown"
    from merlin.targetgen.software_spec import admit_operation

    assert admit_operation(admitted, "linalg.unknown", {"operand_dtype": "int8"}, "host")["status"] == "unknown"
    with pytest.raises(ValueError, match="normalized"):
        gate._selected_admission_views(
            {**software, "operations": {"linalg.matmul": {"placement": "accelerator"}}},
            profiles,
            target="fixture",
            host_package=host_package,
            host_package_sha256=host_package_digest,
            host_capabilities_path=host_path,
            host_capabilities_sha256=_sha(host_path),
        )
    (host_package / "manifest.yaml").write_text("changed host package\n")
    with pytest.raises(ValueError, match="selected host profile"):
        gate._selected_admission_views(
            software,
            profiles,
            target="fixture",
            host_package=host_package,
            host_package_sha256=host_package_digest,
            host_capabilities_path=host_path,
            host_capabilities_sha256=_sha(host_path),
        )
    with pytest.raises(ValueError, match="selected software-spec"):
        gate._recipe_derivation(tmp_path / "operator.yaml", row, plan, **{**kwargs, "software_sha256": "4" * 64})
    with pytest.raises(ValueError, match="selected host-capability"):
        gate._recipe_derivation(
            tmp_path / "operator.yaml", row, plan, **{**kwargs, "host_capabilities_sha256": "4" * 64}
        )
    with pytest.raises(ValueError, match="absent from current-spec derivation"):
        gate._recipe_derivation(
            tmp_path / "operator.yaml",
            row,
            {"recipe": {**plan["recipe"], "recipe_sha256": "5" * 64}},
            **kwargs,
        )
    selected.write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        gate._recipe_derivation(tmp_path / "operator.yaml", row, plan, **kwargs)


def test_private_paths_are_denied_without_copying_capture_into_candidate(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import capture_selection

    selection = tmp_path / "selection" / "capture-selection.json"
    selection.parent.mkdir()
    selection.write_text("{}")
    source = tmp_path / "private-source"
    source.mkdir()
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"private")
    recipe = tmp_path / "selected-recipe.json"
    recipe.write_text("{}")
    evidence = tmp_path / "recipe-derivation"
    evidence.mkdir()
    (evidence / "evidence-manifest.json").write_text("{}")
    run = tmp_path / "sealed-run"
    selected = {
        "schema": capture_selection.SCHEMA_V2,
        "run_dir": str(run),
        "plan": {
            "m2m_root": str(source),
            "workload_root": str(source),
            "selected_inputs": [
                {
                    "source": str(checkpoint),
                    "kind": "file",
                    "role": "checkpoint",
                    "guest_member": "weights.safetensors",
                    "sha256": _sha(checkpoint),
                }
            ],
            "loader_env": {"MODEL_MODE": "complete"},
            "recipe": {"path": str(recipe), "sha256": _sha(recipe), "recipe_sha256": "a" * 64},
        },
    }
    monkeypatch.setattr(capture_selection, "load", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(gate, "_recipe_derivation", lambda *_args, **_kwargs: {})
    for name in ("facts.json", "software.yaml", "capability.yaml", "host.yaml"):
        (tmp_path / name).write_text("public target contract\n")
    row = {
        "id": "complete",
        "capture_selection": str(selection),
        "capture_selection_sha256": _sha(selection),
        "capture": str(run / "capture"),
        "capture_execution_attestation": str(run / "attestation.json"),
        "recipe_derivation_root": str(evidence),
        "recipe_derivation_manifest_sha256": _sha(evidence / "evidence-manifest.json"),
        "rtl_facts": str(tmp_path / "facts.json"),
        "rtl_facts_sha256": _sha(tmp_path / "facts.json"),
        "host_package": str(tmp_path / "host"),
        "software_spec": str(tmp_path / "software.yaml"),
        "software_spec_sha256": _sha(tmp_path / "software.yaml"),
        "capability_contract": str(tmp_path / "capability.yaml"),
        "capability_contract_sha256": _sha(tmp_path / "capability.yaml"),
        "host_capabilities": str(tmp_path / "host.yaml"),
        "host_capabilities_sha256": _sha(tmp_path / "host.yaml"),
        "board_catalog": str(tmp_path / "boards.yaml"),
        "host_dts": str(tmp_path / "host.dts"),
        "deployment_dtype": "int8",
        "input_provenance": {"paper_ready": False, "synthetic_inputs": True},
        "selected_input_bindings": [
            {
                "role": "checkpoint",
                "kind": "file",
                "guest_member": "weights.safetensors",
                "sha256": _sha(checkpoint),
            }
        ],
    }
    selection_raw = b'{"target":"fixture"}\n'
    (evidence / "software").mkdir()
    (evidence / "software" / "selection.json").write_bytes(selection_raw)
    archive_sources = []
    artifacts = {
        "software/selection.json": {
            "sha256": hashlib.sha256(selection_raw).hexdigest(),
            "size_bytes": len(selection_raw),
        }
    }
    for index, (key, role) in enumerate(
        (("software_spec", "software-spec"), ("host_capabilities", "host-capability-spec:profile"))
    ):
        authored = Path(row[key])
        raw = authored.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        member = f"software/source-snapshots/{index:04d}-{digest}.bin"
        (evidence / "software" / "source-snapshots").mkdir(exist_ok=True)
        (evidence / member).write_bytes(raw)
        artifacts[member] = {"sha256": digest, "size_bytes": len(raw)}
        archive_sources.append(
            {"source": str(authored), "role": role, "path": member, "sha256": digest, "size_bytes": len(raw)}
        )
    (evidence / "evidence-manifest.json").write_text(
        json.dumps(
            {
                "schema": "phase0_evidence_v1",
                "target": "fixture",
                "raw_facts_sha256": None,
                "sources": archive_sources,
                "artifacts": artifacts,
            }
        )
    )
    row["recipe_derivation_manifest_sha256"] = _sha(evidence / "evidence-manifest.json")
    spec = tmp_path / "operator.yaml"
    spec.write_text(yaml.safe_dump({"schema": gate.SCHEMA, "target": "fixture", "models": [row]}))
    paths = gate.private_input_paths(spec, target="fixture", required_models=("complete",))
    assert {entry["path"] for entry in paths} >= {
        str(source),
        str(checkpoint),
        str(run),
        str(selection.parent),
        str(evidence),
    }
    assert str(row["software_spec"]) not in {entry["path"] for entry in paths}
    assert str(row["rtl_facts"]) not in {entry["path"] for entry in paths}
    assert str(row["host_package"]) not in {entry["path"] for entry in paths}
    ws = tmp_path / "authoring" / "workspace"
    ws.mkdir(parents=True)
    argv = bwrap.base_argv(
        ws,
        {"allowed": [], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert str(source) not in argv  # default-denied scratch needs no invalid nested mount
    assert str(run) not in argv  # future output is default-denied, never copied or mounted
    run.mkdir()  # once the parent exists, a broad grant must mask this exact directory
    exposed = bwrap.base_argv(
        ws,
        {"allowed": [{"path": str(tmp_path)}], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert exposed[exposed.index(str(source)) - 1 : exposed.index(str(source)) + 1] == ["--tmpfs", str(source)]
    assert str(checkpoint) in exposed
    public_spec = Path(row["software_spec"])
    public_spec.write_text("public target contract\n")
    bwrap.base_argv(
        ws,
        {"allowed": [{"path": str(public_spec)}], "denied": [], "private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    alias = tmp_path / "runtime-alias"
    surfaces = host_surfaces.host_input_surfaces(
        ["--ro-bind", str(tmp_path), str(alias)],
        ws,
        {"private_validation_paths": paths},
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    assert alias / "private-source" in {surface.path for surface in surfaces}
    assert alias / "weights.safetensors" in {surface.path for surface in surfaces}

    from merlin_experiments.phase1 import corpus_inputs as CI

    descriptor = tmp_path / "target.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "phase1_gates": {
                    "private_full_models": {
                        "required": True,
                        "models": ["complete"],
                        "programs": {"complete": ["model"]},
                        "source_workload_dirs": {"complete": source.name},
                        "required_selected_roles": {"complete": ["checkpoint"]},
                        "deployment_dtypes": {"complete": "int8"},
                        "required_loader_env": {"complete": {"MODEL_MODE": "complete"}},
                    }
                },
            }
        )
    )
    authored = tmp_path / "authored.yaml"
    authored_bundle = {"bundle_id": "fixture", "allowed": [], "denied": []}
    authored.write_text(yaml.safe_dump(authored_bundle))
    host_run = tmp_path / "host-run"
    host_run.mkdir()
    corpus_input = host_run / "private_corpus_input"
    corpus_input.mkdir()
    corpus_record = {"descriptor_sha256": "selected", "staging_path": str(corpus_input)}
    staged_bundle = {
        **authored_bundle,
        "host_inputs": [{"path": str(corpus_input), "note": "host-only run corpus view"}],
    }
    monkeypatch.setattr(CI, "stage", lambda *_args, **_kwargs: (dict(staged_bundle), corpus_record))
    prepared = CI.prepare_bundle(
        host_run,
        SimpleNamespace(path=descriptor, target="fixture", descriptor_sha256="selected"),
        authored,
        authored_bundle,
        contract=tmp_path,
        private_full_model_spec=spec,
    )
    assert prepared.private_full_model_record["source"] == str(spec)
    assert prepared.private_full_model_record["source_freeze"]["schema"].endswith("private_source_freeze.v1")
    assert prepared.private_full_model_record["path"] in [entry["path"] for entry in prepared.bundle["host_inputs"]]
    assert prepared.bundle["allowed"] == []
    assert {row["path"] for row in prepared.bundle["private_validation_paths"]} >= {
        str(host_run / "grading_private_full_models"),
        str(host_run / "run_manifest.yaml"),
        str(host_run / "environment.yaml"),
        str(row["software_spec"]),
        str(row["host_capabilities"]),
        str(host_run / "private_full_model_input" / "sources"),
    }
    (host_run / "grading_private_full_models").mkdir()
    (host_run / "run_manifest.yaml").write_text("private\n")
    (host_run / "environment.yaml").write_text("private\n")
    frozen_alias = tmp_path / "runtime-frozen-alias"
    private_surfaces = host_surfaces.host_input_surfaces(
        ["--ro-bind", str(tmp_path), str(frozen_alias)],
        ws,
        prepared.bundle,
        repo=tmp_path,
        _policy_test_live_inputs=True,
    )
    aliases = {surface.path for surface in private_surfaces}
    assert frozen_alias / "software.yaml" in aliases
    assert frozen_alias / "host.yaml" in aliases
    assert frozen_alias / "host-run/private_full_model_input/sources" in aliases
    Path(row["software_spec"]).write_text("new software declaration\n")
    Path(row["host_capabilities"]).write_text("new host declaration\n")
    rebound_paths = gate.private_input_paths(
        spec,
        target="fixture",
        required_models=("complete",),
        source_freeze=prepared.private_full_model_record["source_freeze"],
        source_freeze_root=host_run / "private_full_model_input" / "sources",
    )
    assert {item["path"] for item in rebound_paths} >= {
        str(row["software_spec"]),
        str(row["host_capabilities"]),
    }
    resumed = CI.prepare_bundle(
        host_run,
        SimpleNamespace(path=descriptor, target="fixture", descriptor_sha256="selected"),
        authored,
        authored_bundle,
        contract=tmp_path,
        environment={
            "authored_bundle_manifest_sha256": prepared.authored_sha256,
            "public_corpus_input": corpus_record,
            "private_full_model_spec": prepared.private_full_model_record,
            "bundle_manifest_sha256": prepared.effective_sha256,
        },
    )
    assert resumed.private_full_model_record == prepared.private_full_model_record
    bwrap.materialize_bundle_inputs(ws, prepared.bundle, repo=tmp_path)
    [frozen] = bwrap.snapshot_input_paths(
        ws,
        prepared.bundle,
        [Path(prepared.private_full_model_record["path"])],
        repo=tmp_path,
    )
    assert _sha(frozen) == _sha(spec)


def test_live_bwrap_masks_resumed_private_grade_under_broad_grant(tmp_path):
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = tmp_path / "public.txt"
    public.write_text("public\n")
    private = tmp_path / "host-run" / "grading_private_full_models"
    private.mkdir(parents=True)
    (private / "private-result.json").write_text('{"model": "private"}\n')
    manifest = tmp_path / "host-run" / "run_manifest.yaml"
    manifest.write_text("private: true\n")
    bundle = {
        "allowed": [{"path": str(tmp_path)}],
        "denied": [],
        "private_validation_paths": [
            {"path": str(private), "kind": "dir"},
            {"path": str(manifest), "kind": "file"},
        ],
    }
    argv = bwrap.base_argv(
        workspace,
        bundle,
        repo=tmp_path,
        _policy_test_live_inputs=True,
        include_claude_home=False,
        inherit_environment=False,
    )
    alias = Path("/tmp/private-gate-alias")
    argv += ["--dir", str(alias), "--ro-bind", str(tmp_path), str(alias)]
    argv = bwrap.apply_answer_masks(
        argv,
        host_surfaces.host_input_surfaces(argv, workspace, bundle, repo=tmp_path, _policy_test_live_inputs=True),
    )
    probe = subprocess.run(
        [
            *argv,
            "/bin/sh",
            "-c",
            (
                f'test -s "{public}" && test -s "{alias / public.name}" '
                f'&& test ! -s "{private}/private-result.json" '
                f'&& test ! -s "{alias / private.relative_to(tmp_path) / "private-result.json"}" '
                f'&& test ! -s "{manifest}" '
                f'&& test ! -s "{alias / manifest.relative_to(tmp_path)}"'
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "Creating new namespace failed: Operation not permitted" in probe.stderr:
        pytest.skip("bubblewrap user namespace is disabled on this host")
    assert probe.returncode == 0, probe.stderr


def test_late_installed_package_alias_masks_private_grader_and_oracle(tmp_path):
    """A runtime bind must not give installed private imports a second name."""
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    package = tmp_path / "installed" / "merlin_experiments"
    grader = package / "phase1" / "feedback"
    grader.mkdir(parents=True)
    (grader / "private_full_models.py").write_text("private gate\n")
    (grader / "private_group_provenance.py").write_text("private source join\n")
    (grader / "private_device_audit.py").write_text("private built artifact veto\n")
    (grader / "private_pure_stage_support.py").write_text("private alias evidence\n")
    oracle = package / "phase1" / "private_oracle.py"
    oracle.write_text("private oracle\n")
    public = package / "phase1" / "public_tool.py"
    public.write_text("public tool\n")
    alias = Path("/tmp/installed-private-alias")
    argv = bwrap.base_argv(
        workspace,
        {"allowed": [], "denied": []},
        repo=tmp_path,
        _policy_test_live_inputs=True,
        include_claude_home=False,
        inherit_environment=False,
    )
    argv += ["--dir", str(alias), "--ro-bind", str(package), str(alias)]
    surfaces = [
        AnswerSurface("grader:feedback", grader, "dir", "grader"),
        AnswerSurface("oracle:private_oracle", oracle, "file", "oracle"),
    ]
    masked = bwrap.apply_answer_masks(argv, surfaces)
    assert not bwrap.is_exposed(masked, alias / "phase1/feedback/private_full_models.py")
    assert not bwrap.is_exposed(masked, alias / "phase1/feedback/private_group_provenance.py")
    assert not bwrap.is_exposed(masked, alias / "phase1/feedback/private_device_audit.py")
    assert not bwrap.is_exposed(masked, alias / "phase1/feedback/private_pure_stage_support.py")
    assert not bwrap.is_exposed(masked, alias / "phase1/private_oracle.py")
    assert bwrap.is_exposed(masked, alias / "phase1/public_tool.py")
    probe = subprocess.run(
        [
            *masked,
            "/bin/sh",
            "-c",
            (
                f'test -s "{alias / "phase1/public_tool.py"}" '
                f'&& test ! -s "{alias / "phase1/feedback/private_full_models.py"}" '
                f'&& test ! -s "{alias / "phase1/feedback/private_group_provenance.py"}" '
                f'&& test ! -s "{alias / "phase1/feedback/private_device_audit.py"}" '
                f'&& test ! -s "{alias / "phase1/feedback/private_pure_stage_support.py"}" '
                f'&& test ! -s "{alias / "phase1/private_oracle.py"}"'
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "Creating new namespace failed: Operation not permitted" in probe.stderr:
        pytest.skip("bubblewrap user namespace is disabled on this host")
    assert probe.returncode == 0, probe.stderr


def test_missing_future_private_output_refuses_live_parent_or_runtime_alias(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = tmp_path / "host-run"
    parent.mkdir()
    future = parent / "grading_private_full_models"
    bundle = {"allowed": [{"path": str(parent)}], "private_validation_paths": [{"path": str(future), "kind": "dir"}]}
    with pytest.raises(RuntimeError, match="future operator-private"):
        bwrap.base_argv(workspace, bundle, repo=tmp_path, _policy_test_live_inputs=True)

    no_public_grant = {"allowed": [], "private_validation_paths": bundle["private_validation_paths"]}
    with pytest.raises(RuntimeError, match="runtime bind could expose a future"):
        host_surfaces.host_input_surfaces(
            ["--ro-bind", str(parent), "/tmp/private-gate-alias"],
            workspace,
            no_public_grant,
            repo=tmp_path,
            _policy_test_live_inputs=True,
        )
