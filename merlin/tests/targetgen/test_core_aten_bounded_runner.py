"""Contract tests without Torch or a simulator installation."""

import json

import pytest

from merlin.targetgen.core_aten_bounded_runner import (
    SavedResponseAdapter,
    atomic_json,
    bounded_loader_source,
    bundle_admission_reason,
    case_digest,
    grade_portable_outputs,
    semantic_observation,
)


def test_numeric_success_does_not_claim_complete_semantics():
    evidence = {"numeric_verdict": {"status": "pass"}, "lane": "host"}
    observed = semantic_observation({"status": "pass"}, evidence)
    assert observed["status"] == "ungradable"
    assert observed["evidence"] == evidence
    assert "aliases" in observed["reason"]


@pytest.mark.parametrize("selection", ["software_spec", "host_capabilities"])
def test_cache_stamp_binds_selected_policy_files_and_layout_sources(tmp_path, monkeypatch, selection):
    from merlin.common.paths import repo_root

    module = _measurement_script()
    simulator = tmp_path / "simulator"
    simulator.write_bytes(b"binary")
    software, host = tmp_path / "software.yaml", tmp_path / "host.yaml"
    software.write_text("original software declaration")
    host.write_text("original host declaration")
    monkeypatch.setenv("MERLIN_SPIKE", str(simulator))
    args = _stamp_args(tmp_path, repo=repo_root(), m2m_dir=repo_root(), software_spec=software, host_capabilities=host)
    first = module.stamp(args)
    getattr(args, selection).write_text("changed declaration")
    second = module.stamp(args)
    assert first != second
    assert first[selection + "_sha256"] != second[selection + "_sha256"]
    sources = second["sources"]
    assert any(path.endswith("src/merlin/llvmlower/input_layout.py") for path in sources)
    assert any(path.endswith("src/merlin/targetgen/core_aten_spec_int8.py") for path in sources)
    assert any(path.endswith("src/merlin/targetgen/core_aten_input_storage.py") for path in sources)


def test_failure_and_device_evidence_are_preserved():
    evidence = {"lane": "device", "executed_instructions": 7}
    observed = semantic_observation({"status": "execution_failed", "reason": "missing receipt"}, evidence)
    assert observed == {"status": "execution_failed", "reason": "missing receipt", "evidence": evidence}


def test_loader_preserves_layout_and_seed():
    case = {
        "overload": "aten.alias.default",
        "rng_seed": 123,
        "arguments": {
            "args": {
                "kind": "tuple",
                "items": [
                    {
                        "kind": "tensor",
                        "dtype": "float32",
                        "shape": [2],
                        "stride": [2],
                        "storage_offset": 1,
                        "requires_grad": False,
                        "values": [1, 2],
                    }
                ],
            },
            "kwargs": {},
        },
        "expected": {"kind": "tensor"},
    }
    source = bounded_loader_source(case)
    compile(source, "loader.py", "exec")
    assert "as_strided(shape, stride, offset)" in source
    assert "torch.manual_seed(123)" in source
    assert 'complex(_decode(value["real"]), _decode(value["imag"]))' in source


def test_loader_seed_hook_and_complex_nonfinite_decoding():
    import torch

    case = dict(
        overload="aten.scalar_tensor.default",
        arguments=dict(
            args=dict(kind="tuple", items=[dict(kind="complex", real=dict(kind="float", value="inf"), imag=2)]),
            kwargs=dict(dtype=dict(kind="dtype", value="complex64")),
        ),
        expected=dict(kind="tensor"),
        rng_seed=19,
    )
    namespace = {}
    exec(compile(bounded_loader_source(case), "<bounded loader>", "exec"), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    result = model(*inputs)
    assert torch.isinf(result.real)
    assert result.imag.item() == 2
    assert torch.initial_seed() == 19


def test_atomic_response_and_case_identity(tmp_path):
    path = tmp_path / "nested" / "response.json"
    document = {"schema_version": 1, "results": {"case": {"status": "capture_unavailable"}}}
    atomic_json(path, document)
    assert json.loads(path.read_text()) == document
    assert not path.with_suffix(".json.tmp").exists()
    assert SavedResponseAdapter(document).execute({}) == document["results"]
    assert case_digest({"a": 1, "b": 2}) == case_digest({"b": 2, "a": 1})
    assert case_digest({"a": 1}) != case_digest({"a": 2})


def test_portable_nonfinite_oracle_uses_same_comparator_without_mutation():
    import struct

    case = {
        "expected": {
            "kind": "tensor",
            "dtype": "float32",
            "shape": [2],
            "values": [{"kind": "float", "value": "nan"}, {"kind": "float", "value": "inf"}],
        },
        "comparison": "torch_close",
        "comparison_parameters": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": True},
    }
    batch = {
        "cases": [
            {
                "case_id": "case",
                "overload": "aten.alias.default",
                "case": case,
                "status": "bundled",
                "output_indices": [0],
                "output_abi": [{"dtype": "f32", "shape": [2]}],
            }
        ]
    }
    before = json.dumps(batch, sort_keys=True)
    assert (
        grade_portable_outputs(batch, [struct.pack("<ff", float("nan"), float("inf"))])["cases"]["case"]["status"]
        == "pass"
    )
    assert grade_portable_outputs(batch, [struct.pack("<ff", 1, float("inf"))])["cases"]["case"]["status"] == "mismatch"
    assert json.dumps(batch, sort_keys=True) == before


def test_bundle_refusal_reports_native_abi_reason():
    assert (
        bundle_admission_reason({"capture_meta": {"input_abi": [{"dtype": "f80", "shape": [2]}]}})
        == "ValueError: unsupported batch ABI dtype 'f80'"
    )
    assert bundle_admission_reason({"capture_meta": {"output_abi": [{"dtype": "f32", "shape": [2]}]}}) is None


def test_half_storage_abis_are_admitted():
    for dtype in ("bf16", "f16"):
        assert bundle_admission_reason({"capture_meta": {"input_abi": [{"dtype": dtype, "shape": [2]}]}}) is None


def test_function_and_capture_abi_counts_must_agree():
    source = """builtin.module {
  func.func @forward(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    func.return %arg0 : tensor<2xf32>
  }
}"""
    abi = {"dtype": "f32", "shape": [2]}
    assert "argument ABI" in bundle_admission_reason(
        {"capture_meta": {"input_abi": [abi, abi], "output_abi": [abi]}}, source
    )
    assert "result ABI" in bundle_admission_reason(
        {"capture_meta": {"input_abi": [abi], "output_abi": [abi, abi]}}, source
    )


def test_full_success_exports_observation_and_device_receipt():
    boundary = dict(
        status="executed",
        output={"values": [7]},
        post_arguments={},
        mutated_arguments=[],
        output_input_aliases=[],
        execution_kind="simulator",
        evidence={"boundary": "same_conversion_exported_program"},
    )
    evidence = {"lane": "device", "routing": {"execution_evidence": {"executed_instructions": 7}}}
    result = semantic_observation({"status": "pass", "semantic_scope": "full", "observation": boundary}, evidence)
    assert result["status"] == "executed"
    assert result["output"] == boundary["output"]
    assert result["evidence"]["boundary"] == boundary["evidence"]["boundary"]
    assert result["evidence"]["routing"] == evidence["routing"]


def test_admission_uses_same_conversion_mutation_result_roles():
    source = """builtin.module {
  func.func @forward(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    func.return %arg0, %arg0 : tensor<2xf32>, tensor<2xf32>
  }
}"""
    abi = {"dtype": "f32", "shape": [2]}
    capture = {
        "capture_meta": {
            "input_abi": [abi],
            "output_abi": [abi],
            "result_contract": {"results": [{"role": "user_input_mutation"}, {"role": "user_output"}]},
        }
    }
    assert bundle_admission_reason(capture, source) is None
    capture["capture_meta"]["result_contract"]["results"].pop()
    assert "result ABI" in bundle_admission_reason(capture, source)


def test_secondary_numeric_grade_does_not_apply_full_boundary():
    import struct

    case = {
        "expected": {"kind": "tensor", "dtype": "float32", "shape": [1], "values": [2.0]},
        "comparison": "torch_close",
    }
    record = dict(
        case_id="case",
        overload="aten.alias.default",
        case=case,
        status="bundled",
        output_indices=[0],
        output_abi=[{"dtype": "f32", "shape": [1]}],
        semantic_boundary={"post_indices": [1]},
    )
    batch = {"cases": [record]}
    assert grade_portable_outputs(batch, [struct.pack("<f", 2.0)])["cases"]["case"]["status"] == "pass"
    assert record["semantic_boundary"] == {"post_indices": [1]}


def test_secondary_numeric_grade_uses_runtime_dynamic_shape():
    import struct

    case = {
        "expected": {"kind": "tensor", "dtype": "float32", "shape": [2], "values": [2.0, 3.0]},
        "comparison": "torch_close",
    }
    batch = {
        "cases": [
            dict(
                case_id="case",
                overload="aten.alias.default",
                case=case,
                status="bundled",
                output_indices=[0],
                output_abi=[{"dtype": "f32", "shape": [2], "declared_shape": [-1]}],
            )
        ]
    }
    result = grade_portable_outputs(batch, [struct.pack("<ff", 2.0, 3.0)], output_shapes=[[2]])
    assert result["cases"]["case"]["status"] == "pass"


def test_failure_family_merges_locations_but_separates_runtime_symbols():
    from merlin.targetgen.core_aten_bounded_runner import failure_family

    first = failure_family("compile_lowering_failed", "/tmp/a.o: undefined reference to `helper'", "aten.alias.default")
    second = failure_family(
        "compile_lowering_failed", "/tmp/b.o: undefined reference to `helper'", "aten.clone.default"
    )
    other = failure_family("compile_lowering_failed", "/tmp/b.o: undefined reference to `other'", "aten.clone.default")
    assert first == second
    assert first != other
    assert first[1] == "Merlin"


def test_summary_handles_families_present_only_in_baseline(tmp_path):
    import importlib.util
    from argparse import Namespace

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("bounded_measurement_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = tmp_path / "baseline" / "scalar"
    baseline.mkdir(parents=True)
    old = dict(
        case_count=1,
        status_counts={"ungradable": 1},
        cases={"case": {"status": "ungradable", "reason": "byte readback cannot observe aliases"}},
    )
    (baseline / "report.json").write_text(json.dumps(old))
    current = dict(
        case_count=1,
        campaign_complete=True,
        status_counts={"pass": 1},
        cases={"case": {"status": "pass", "evidence": {"lane": "host", "numeric_verdict": {"status": "pass"}}}},
    )
    case = dict(case_id="case", overload="aten.alias.default", partition_assignment={})
    args = Namespace(output=tmp_path / "current", baseline=tmp_path / "baseline", label="scalar")
    module.summarize(args, {"selected_cases": [case]}, current, {}, {})
    result = json.loads((args.output / args.label / "summary.json").read_text())
    assert result["failure_families"] == []
    assert result["delta_by_family"]["legacy numeric-only boundary"] == {"baseline": 1, "current": 0, "delta": -1}
    assert result["full_semantic_passed_count"] == 1
    assert result["numeric_only_passed_count"] == 1


def test_summary_and_shard_aggregation_separate_spec_int8_host_fallback(tmp_path):
    import importlib.util
    from argparse import Namespace

    from merlin.common.paths import repo_root

    module = _measurement_script()
    current = dict(
        case_count=1,
        campaign_complete=True,
        status_counts={"pass": 1},
        cases={
            "case": dict(
                status="pass",
                numeric_policy="spec_int8",
                evidence=dict(lane="host", numeric_verdict=dict(status="pass")),
            )
        },
    )
    case = dict(case_id="case", overload="aten.mm.default", partition_assignment={})
    args = Namespace(output=tmp_path, baseline=None, label="spec-int8")
    module.summarize(args, {"selected_cases": [case]}, current, {}, {})
    result = json.loads((tmp_path / args.label / "summary.json").read_text())
    assert result["spec_int8_passed_count"] == 0
    assert result["spec_int8_host_fallback_passed_count"] == 1
    assert result["numeric_passed_by_lane_and_policy"]["host"]["original_host"] == 1
    assert "Spec-int8 results: 0" in (tmp_path / args.label / "summary.md").read_text()
    spec = importlib.util.spec_from_file_location(
        "batch_policy_summary", repo_root() / "build_tools/scripts/build_core_aten_batch.py"
    )
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    verdict = dict(status="pass", numeric_status="pass", numeric_policy="spec_int8", lane="host")
    counts = batch._semantic_counts(dict(cases={"case": verdict}, passed_count=1))
    assert counts["spec_int8_passed_count"] == 0
    assert counts["spec_int8_host_fallback_passed_count"] == 1


def test_evaluate_only_refuses_missing_torch_without_reexecution(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("bounded_evaluation_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"selected_cases": []}))
    pinned = tmp_path / "pinned-python"
    arguments = [
        str(script),
        "--suite",
        str(suite),
        "--output",
        str(tmp_path / "output"),
        "--repo",
        str(tmp_path),
        "--m2m-dir",
        str(tmp_path / "m2m"),
        "--m2m-python",
        str(pinned),
        "--label",
        "scalar",
        "--evaluate-only",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(module, "sys", SimpleNamespace(argv=arguments, executable="controller-python", modules={}))
    monkeypatch.setattr(module, "stamp", lambda _: {})

    def unavailable(name):
        assert name == "torch"
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(module, "__import__", unavailable, raising=False)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: commands.append(argv))
    import pytest

    with pytest.raises(RuntimeError, match="--evaluate-only requires Torch"):
        module.main()
    assert commands == []
    assert not (tmp_path / "output").exists()


def test_observed_boundary_scope_keeps_real_layout_and_excludes_only_autograd():
    import copy

    import pytest

    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_eval import _validate_observation

    tensor = dict(
        kind="tensor", dtype="float32", shape=[1], stride=[1], storage_offset=0, requires_grad=False, values=[2.0]
    )
    case = dict(
        expected=tensor,
        post_arguments={"args": {"kind": "tuple", "items": [tensor]}, "kwargs": {}},
        mutated_arguments=[],
        output_input_aliases=[],
        comparison="torch_close",
    )
    boundary = dict(
        status="executed",
        execution_kind="simulator",
        output=copy.deepcopy(tensor),
        post_arguments=copy.deepcopy(case["post_arguments"]),
        mutated_arguments=[],
        output_input_aliases=[],
    )
    boundary["output"]["requires_grad"] = None
    boundary["post_arguments"]["args"]["items"][0]["requires_grad"] = None
    result = semantic_observation(
        dict(
            status="pass",
            semantic_scope="values+mutation+observed-metadata",
            imposed_by_harness=["requires_grad"],
            observation=boundary,
        ),
        {"lane": "host"},
    )
    assert result["status"] == "executed"
    _validate_observation(case, result, allow_unobserved_autograd=True)
    with pytest.raises(AssertionError, match="unsupported"):
        _validate_observation(case, result)
    result["output"]["stride"] = [2]
    with pytest.raises(AssertionError, match="stride"):
        _validate_observation(case, result, allow_unobserved_autograd=True)
    result["imposed_by_harness"].append("stride")
    with pytest.raises(AssertionError, match="unsupported"):
        _validate_observation(case, result, allow_unobserved_autograd=True)


def _measurement_script():
    import importlib.util

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("bounded_cache_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamp_args(tmp_path, **overrides):
    """Stamp arguments over an empty suite in ``tmp_path`` with no optional sources selected."""
    from types import SimpleNamespace

    suite = tmp_path / "suite.json"
    suite.write_text("{}")
    fields = dict(repo=tmp_path, m2m_dir=tmp_path, suite=suite, pin=[], extension_library=None, facts=None)
    fields.update(execution_provider=None, software_spec=None, host_capabilities=None, device_package=None, target=None)
    return SimpleNamespace(**{**fields, **overrides})


def test_cache_compares_entire_provenance_stamp(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    from merlin.targetgen.core_aten_capture import case_capture_name

    module = _measurement_script()
    case = {"case_id": "test", "overload": "aten.alias.default"}
    args = SimpleNamespace(
        output=tmp_path, label="scalar", numeric_policy="fp32", repo=tmp_path, capture_timeout=1, worker_timeout=1
    )
    block = dict(
        source_digest="same",
        merlin_diff_sha256="same",
        model2mlir_commit="m2m",
        merlin_commit="merlin",
        device_package_commit="oot",
        execution_provider_sha256="provider",
        artifact_digests={"spike": "binary", "extension_library": "extension"},
        hardware_pins={"device": {"commit": "pin", "ok": True}},
        target=None,
        all_pins_ok=True,
        sources=["source"],
        merlin={"commit": "merlin", "dirty_files": 0},
    )
    launched = []
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *a, **k: launched.append(a) or SimpleNamespace(wait=lambda **_: 0)
    )
    for phase in ("capture", "execute"):
        directory = (
            tmp_path / "captures" if phase == "capture" else tmp_path / "scalar" / "shards"
        ) / case_capture_name(case)
        completion = directory / ("capture.json" if phase == "capture" else "result.json")
        saved = dict(case_sha256=case_digest(case), numeric_policy="fp32", provenance=block)
        atomic_json(completion, saved)
        assert module.launch(args, phase, case, block) == "cached"
        assert not launched
        for key in block:
            changed = deepcopy(block)
            changed[key] = {"changed": True} if isinstance(changed[key], dict) else "changed"
            assert module.launch(args, phase, case, changed) == "finished", key
        assert len(launched) == len(block)
        launched.clear()
        # A legacy receipt without any current identity field cannot earn a hit.
        for key in block:
            legacy = deepcopy(saved)
            del legacy["provenance"][key]
            atomic_json(completion, legacy)
            assert module.launch(args, phase, case, block) == "finished", key
        launched.clear()


def test_stamp_reads_descriptor_and_runtime_sources(tmp_path, monkeypatch):

    module = _measurement_script()
    monkeypatch.setenv("MERLIN_SPIKE", str(tmp_path / "spike"))
    monkeypatch.setattr(module, "git", lambda *_: "commit")
    monkeypatch.setattr(module.subprocess, "check_output", lambda argv, **_: "" if "ls-files" in argv else b"")
    monkeypatch.setattr(module.provenance, "record", lambda **kwargs: kwargs)
    args = _stamp_args(tmp_path, execution_provider=tmp_path / "provider.py")
    stamp = module.stamp(args)
    expected = [
        "src/merlin/llvmlower/" + name
        for name in ("result_layout.py", "pipeline.py", "result_buffers.py", "c_runtime.py")
    ]
    expected += ["merlin/runtime/abi/mlir_runtime.c", "src/merlin/runtime/backends/spike_model.py"]
    expected += [
        "src/merlin/targetgen/" + name
        for name in ("_recipe_quantizer.py", "_m2m_capture_worker.py", "quant_recipe.py", "capture_cache.py")
    ]
    expected.append("src/merlin/capture/integerization.py")
    assert all(tmp_path / path in stamp["sources"] for path in expected)
    assert args.execution_provider in stamp["sources"]


def test_disk_backed_records_keep_the_complete_response_protocol(tmp_path):
    from merlin.targetgen.core_aten_bounded_runner import JsonRecordMapping

    records = JsonRecordMapping(tmp_path / "records")
    records["one"] = {"status": "pass", "evidence": {"bytes": [0, 255]}}
    records["two"] = {"status": "fail", "reason": "measured mismatch"}
    target = tmp_path / "report.json"
    atomic_json(target, {"cases": records, "case_count": 2})
    assert json.loads(target.read_text()) == {"cases": dict(records), "case_count": 2}
    assert all(isinstance(value, type(target)) for value in records.paths.values())
    projected = JsonRecordMapping(tmp_path, {"one": target}, field="case_count")
    assert projected["one"] == 2
    with pytest.raises(TypeError, match="read-only"):
        projected["two"] = 3


def test_unfinished_disk_response_preserves_explicit_missing_case(tmp_path):
    from merlin.targetgen.core_aten_bounded_runner import JsonRecordMapping

    records = JsonRecordMapping(tmp_path, {"pending": tmp_path / "missing.json"}, missing_value={"status": "not_run"})
    assert dict(records) == {"pending": {"status": "not_run"}}
    assert not (tmp_path / "missing.json").exists()


def test_plain_export_stamp_binds_new_sources_without_claiming_a_git_commit(tmp_path, monkeypatch):

    module = _measurement_script()
    monkeypatch.setattr(module, "git", lambda *_: None)
    monkeypatch.setattr(module.provenance, "record", lambda **kwargs: kwargs)
    monkeypatch.setenv("MERLIN_SPIKE", str(tmp_path / "spike"))

    def no_git(*_args, **_kwargs):
        raise AssertionError("plain exports must not require git diff")

    monkeypatch.setattr(module.subprocess, "check_output", no_git)
    source = tmp_path / "src/new_semantic_rule.py"
    source.parent.mkdir()
    source.write_text("rule = 1\n")
    args = _stamp_args(tmp_path, output=tmp_path / "run")
    first = module.stamp(args)
    source.write_text("rule = 2\n")
    second = module.stamp(args)
    assert first["extra"]["merlin_source_kind"] == "plain_source_export"
    assert first["extra"]["merlin_commit"] is None
    assert first["extra"]["merlin_diff_sha256"] is None
    assert first["extra"]["merlin_source_export_sha256"] != second["extra"]["merlin_source_export_sha256"]
    assert args.output / "merlin-source-export.json" in second["sources"]
