"""Small policy checks; the real M2M/PyTorch process smoke is run separately."""

import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.precision_staging import staging_error
from merlin_experiments.capture_execution.sealed_m2m import (
    SealedM2MError,
    _capture_api_missing,
    _command,
    _command_v2,
    _fp32_stage_api_missing,
    _frontend_trace_api_missing,
    _ldd_library_path,
    _policy,
    _recipe_selection,
    _snapshot_tree,
    _source_tree,
    _static_integer_reference_api_missing,
    prepare_plan,
)
from merlin_experiments.phase0.capture_execution_attestation import AttestationNotVerified, require_verified_execution

from merlin.common.paths import module_source_path, schemas_dir
from merlin.targetgen import application_inventory
from merlin.targetgen._m2m_capture_worker import _diagnostic_model_copy
from merlin.targetgen.quant_recipe import digest as recipe_digest


def test_sealed_capture_checks_namespace_availability_before_copying_runtime(monkeypatch):
    observed = []

    def unavailable(argv, **kwargs):
        observed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 1, b"", b"bwrap: loopback: Operation not permitted")

    monkeypatch.setattr(sealed_m2m.subprocess, "run", unavailable)
    with pytest.raises(SealedM2MError, match="sandbox unavailable before snapshot:.*Operation not permitted"):
        sealed_m2m._probe_sandbox(Path("/usr/bin/bwrap"))
    argv, kwargs = observed[0]
    assert "--unshare-all" in argv and "--disable-userns" in argv
    assert kwargs["env"] == {} and kwargs["timeout"] == 15


def test_selected_capture_api_refuses_second_conversion_before_runtime_snapshot(tmp_path, monkeypatch):
    m2m = tmp_path / "selected-m2m"
    capture = m2m / "m2m/capture"
    capture.mkdir(parents=True)
    (m2m / "m2m/api.py").write_text(
        "def convert(model, inputs, *, backend, quantization, quantization_preapplied, "
        "level, func_name, weights_path): pass\n"
    )
    (capture / "bundle.py").write_text("def write_bundle(model, inputs, out, *, quantization_preapplied=False): pass\n")
    missing = _capture_api_missing(m2m)
    assert missing == (
        "m2m/capture/bundle.py:write_bundle(capture_trace)",
        "m2m/capture/bundle.py:write_bundle(conversion_result)",
        "m2m/capture/bundle.py:write_bundle(source_path)",
        "m2m/capture/provenance.py",
    )
    workload = tmp_path / "workload"
    workload.mkdir()
    (workload / "loader.py").write_text("def get_model_and_inputs(): pass\n")
    worker = module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py"
    monkeypatch.setattr(sealed_m2m, "_venv_home", lambda *_: pytest.fail("runtime inventory must not start"))
    with pytest.raises(SealedM2MError, match="same-conversion materialization/receipt API") as error:
        prepare_plan(m2m_root=m2m, workload_root=workload, worker=worker, venv=tmp_path, schemas_root=schemas_dir())
    assert "conversion_result" in str(error.value)
    assert "m2m/capture/provenance.py" in str(error.value)

    (capture / "bundle.py").write_text(
        "def write_bundle(model, inputs, out, *, source_path, capture_trace, conversion_result): pass\n"
    )
    (capture / "provenance.py").write_text("def write_capture_receipt(out, *, source_path): pass\n")
    assert _capture_api_missing(m2m) == ()


def test_selected_optional_capture_features_are_reported_separately(tmp_path):
    m2m = tmp_path / "selected-m2m"
    package = m2m / "m2m"
    package.mkdir(parents=True)
    (package / "api.py").write_text("def convert(model, inputs, *, capture_trace=False): pass\n")
    assert _frontend_trace_api_missing(m2m) == (
        "m2m/api.py:convert(original_frontend_snapshot)",
        "m2m/capture/trace.py",
    )
    assert _fp32_stage_api_missing(m2m) == ("m2m/capture/trace.py",)
    (package / "capture").mkdir()
    (package / "capture/trace.py").write_text(
        "def materialize_frontend_precision(model, inputs, *, dtype, original_frontend_snapshot): pass\n"
    )
    assert _fp32_stage_api_missing(m2m) == (
        "m2m/capture/trace.py:materialize_frontend_precision(retarget_float_dtype_arguments)",
    )
    (package / "capture/trace.py").write_text(
        "def materialize_frontend_precision(model, inputs, *, dtype, original_frontend_snapshot, "
        "retarget_float_dtype_arguments=False): pass\n"
    )
    assert _fp32_stage_api_missing(m2m) == ()
    assert _static_integer_reference_api_missing(m2m) == (
        "m2m/capture/pt2e_integerize.py",
        "m2m/capture/pt2e_integer_reference.py",
    )


def test_raw_model_copy_identifies_exact_missing_api_without_admitting_capture(tmp_path):
    output = tmp_path / "raw"
    output.mkdir()
    loader = tmp_path / "loader.py"
    loader.write_text("def get_model_and_inputs(): pass\n")
    for name in (
        "linalg.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.json",
        "golden.json",
        "frontend-trace.json",
        "pytorch-opset.json",
        "meta.json",
    ):
        (output / name).write_text("{}\n")
    report = {
        "same_conversion_missing": ["m2m/capture/bundle.py:write_bundle(conversion_result)"],
        "frontend_trace_missing": ["m2m/capture/trace.py"],
        "static_integerization_missing": ["m2m/capture/pt2e_integerize.py"],
    }
    _diagnostic_model_copy(output, loader, capture_api=report)
    record = json.loads((output / "diagnostic-capture.json").read_text())
    assert record["capture_api"] == report
    assert record["phase0_admission"] == "not_granted"
    assert record["source_closure_verified"] is False
    assert not (output / "capture_receipt.json").exists()


def test_normalized_venv_inventory_matches_copied_bytes(tmp_path):
    selected = tmp_path / "selected"
    (selected / "lib").mkdir(parents=True)
    (selected / "lib/value.py").write_bytes(b"value = 1\n")
    (selected / "lib64").symlink_to("lib", target_is_directory=True)
    expected = _source_tree(selected, skip_lib64=True)
    copied = tmp_path / "copied"
    shutil.copytree(
        selected,
        copied,
        symlinks=False,
        ignore=lambda directory, names: {"lib64"} if Path(directory) == selected else set(),
    )
    assert _snapshot_tree(copied) == expected
    (copied / "lib/value.py").write_bytes(b"value = 2\n")
    assert _snapshot_tree(copied) != expected


def test_selected_m2m_source_ignores_transient_bytecode(tmp_path):
    selected = tmp_path / "m2m"
    selected.mkdir()
    (selected / "api.py").write_text("value = 1\n")
    before = _source_tree(selected, skip_python_cache=True)
    cache = selected / "__pycache__"
    cache.mkdir()
    (cache / "api.cpython-312.pyc").write_bytes(b"diagnostic import")
    assert _source_tree(selected, skip_python_cache=True) == before
    copied = tmp_path / "copied"
    shutil.copytree(
        selected,
        copied,
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    assert _snapshot_tree(copied) == before
    (selected / "api.py").write_text("value = 2\n")
    assert _source_tree(selected, skip_python_cache=True) != before
    (cache / "unexpected.py").write_text("value = 3\n")
    with pytest.raises(SealedM2MError, match="unsupported member"):
        _source_tree(selected, skip_python_cache=True)


def test_selected_merlin_source_excludes_only_validated_bytecode_before_staging(tmp_path):
    m2m, workload, merlin, source = (tmp_path / name for name in ("m2m", "workload", "merlin", "source"))
    (m2m / "m2m").mkdir(parents=True)
    workload.mkdir()
    (merlin / "targetgen").mkdir(parents=True)
    (merlin / "_data/schemas").mkdir(parents=True)
    (merlin / "targetgen/worker.py").write_text("value = 1\n")
    (merlin / "_data/schemas/example.yaml").write_text("name: example\n")
    source.mkdir()
    selected = _source_tree(merlin, skip_python_cache=True)
    cache = merlin / "targetgen/__pycache__"
    cache.mkdir()
    (cache / "worker.cpython-312.pyc").write_bytes(b"generated bytecode")
    assert _source_tree(merlin, skip_python_cache=True) == selected
    plan = {
        "m2m_root": str(m2m),
        "workload_root": str(workload),
        "merlin_root": str(merlin),
        "schemas_root": str(merlin / "_data/schemas"),
        "selected_trees": {"merlin": selected},
    }
    sealed_m2m._stage_source(plan, source)
    assert _snapshot_tree(source / "merlin-src/merlin") == selected
    assert not (source / "merlin-src/merlin/targetgen/__pycache__").exists()
    (merlin / "targetgen/worker.py").write_text("value = 2\n")
    assert _source_tree(merlin, skip_python_cache=True) != selected
    (cache / "unexpected.py").write_text("value = 3\n")
    with pytest.raises(SealedM2MError, match="unsupported member"):
        _source_tree(merlin, skip_python_cache=True)


def test_staged_cpu_source_must_match_antecedent_selection_before_execution(tmp_path):
    source, runtime = tmp_path / "source", tmp_path / "guest-root"
    roots = {
        "venv": runtime / "opt/capture-venv",
        "base": runtime / "usr/local/base-python",
        "m2m": source / "m2m-src/m2m",
        "workload": source / "workload",
        "merlin": source / "merlin-src/merlin",
    }
    roots["schemas"] = roots["merlin"] / "_data/schemas"
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
        (path / "selected.txt").write_text(path.name)
    plan = {
        "base": "/usr/local/base-python",
        "merlin_root": "/selected/merlin",
        "schemas_root": "/selected/merlin/_data/schemas",
        "selected_trees": {name: _snapshot_tree(path) for name, path in roots.items()},
    }
    assert sealed_m2m._verify_staged_selection(plan, source, runtime) == plan["selected_trees"]["schemas"]
    (roots["workload"] / "selected.txt").write_text("changed after plan recheck")
    with pytest.raises(SealedM2MError, match="staged workload bytes differ from the pre-execution selection"):
        sealed_m2m._verify_staged_selection(plan, source, runtime)


def test_other_directory_alias_is_rejected(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SealedM2MError, match="directory link"):
        _source_tree(selected)


def test_guest_output_path_is_host_resolvable_and_command_is_fixed(tmp_path):
    output = tmp_path / "capture"
    command = _command(output)
    assert command[:5] == ("/opt/capture-venv/bin/python", "-I", "-S", "-B", "-c")
    assert "sys.path[:0]" in command[-1]
    assert "KeyValueRenderer(sort_keys=True)" in command[-1]
    assert repr(str(output)) in command[-1]
    assert "--out" in command[-1]
    assert _policy(command, output) != _policy(command, tmp_path / "other")


def test_preselected_runs_record_a_log_floor_so_their_stderr_replays(tmp_path, monkeypatch):
    output = tmp_path / "capture"
    command = _command_v2(output, dtype="int8", recipe=True)
    # The floor is part of the recorded policy; unselected historical receipts keep theirs.
    assert _policy(command, output, replayable_logs=True) != _policy(command, output)
    seen = []

    def fake_run(argv, **_kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(sealed_m2m.subprocess, "run", fake_run)
    for replayable in (False, True):
        sealed_m2m._execute(tmp_path / "bwrap", tmp_path, tmp_path, output, command, output, replayable_logs=replayable)
    plain, quiet = seen
    expected = {"TORCH_CPP_LOG_LEVEL": "ERROR", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"}
    assert dict(sealed_m2m._REPLAYABLE_LOG_ENV) == expected
    remaining = quiet
    for name, value in expected.items():
        setting = ["--setenv", name, value]
        assert not any(plain[i : i + 3] == setting for i in range(len(plain)))
        (at,) = [i for i in range(len(remaining)) if remaining[i : i + 3] == setting]
        remaining = remaining[:at] + remaining[at + 3 :]
    assert remaining == plain


def test_int8_command_binds_selected_recipe_and_never_uses_host_path(tmp_path):
    command = _command_v2(tmp_path / "capture", dtype="int8", recipe=True)
    assert "'/source/merlin-src'" in command[-1]
    assert "'--dtype', 'int8'" in command[-1]
    assert "'--recipe', '/source/inputs/quant_recipe.json'" in command[-1]
    assert str(tmp_path / "recipe.json") not in command[-1]
    assert _policy(command, tmp_path / "capture") != _policy(
        _command_v2(tmp_path / "capture", dtype="fp32", recipe=False), tmp_path / "capture"
    )
    with pytest.raises(SealedM2MError, match="requires fp32 without a recipe or int8"):
        _command_v2(tmp_path / "capture", dtype="int8", recipe=False)


def test_v2_guest_command_imports_worker_sibling_from_selected_package(tmp_path):
    source = tmp_path / "source"
    worker = source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).parent))\n"
        "import _recipe_quantizer\nprint(_recipe_quantizer.MARKER)\n"
    )
    (worker.parent / "_recipe_quantizer.py").write_text("MARKER = 'selected-sibling'\n")
    stub = source / "m2m-src/structlog.py"
    stub.parent.mkdir()
    stub.write_text(
        "class processors:\n"
        "    @staticmethod\n"
        "    def KeyValueRenderer(**kwargs): return object()\n"
        "def configure(**kwargs): pass\n"
    )
    command = _command_v2(Path("/capture-out"), dtype="int8", recipe=True)
    program = command[-1].replace("/source", str(source))
    observed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program], capture_output=True, text=True, timeout=10, check=False
    )
    assert observed.returncode == 0, observed.stderr
    assert observed.stdout.strip() == "selected-sibling"


def test_v2_materialized_receipt_binds_the_executed_package_worker(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "capture"
    worker_member = "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    for member, content in (
        (worker_member, b"worker\n"),
        ("workload/loader.py", b"loader\n"),
        ("m2m-src/m2m/api.py", b"api\n"),
    ):
        path = source / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    output.mkdir()
    (output / "model.mlir").write_text("prov.weights_file = " + json.dumps(str(output / "weights.safetensors")) + "\n")
    receipt = {
        "source": {
            "path": "/source/workload/loader.py",
            "sha256": sealed_m2m._file_digest(source / "workload/loader.py"),
        },
        "tool": {
            "executed_entrypoint": {
                "path": "/source/" + worker_member,
                "sha256": sealed_m2m._file_digest(source / worker_member),
            },
            "source_inventory_status": "complete",
            "source_sha256": {"m2m/api.py": sealed_m2m._file_digest(source / "m2m-src/m2m/api.py")},
        },
    }
    (output / "capture_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        application_inventory,
        "verify_capture_receipt",
        lambda *_: {"status": "verified_materialized", "receipt_sha256": "bound"},
    )
    assert sealed_m2m._materialized(output, source, output, worker_member=worker_member)["status"] == (
        "verified_materialized"
    )
    receipt["tool"]["executed_entrypoint"]["path"] = "/source/worker.py"
    (output / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(SealedM2MError, match="snapshotted entrypoints"):
        sealed_m2m._materialized(output, source, output, worker_member=worker_member)
    receipt["tool"]["executed_entrypoint"]["path"] = "/source/" + worker_member
    receipt["tool"]["source_sha256"] = {
        str(source / "m2m-src/m2m/api.py"): sealed_m2m._file_digest(source / "m2m-src/m2m/api.py")
    }
    (output / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(SealedM2MError, match="unsafe M2M source member"):
        sealed_m2m._materialized(output, source, output, worker_member=worker_member)


def test_v2_staged_source_loads_selected_quant_format_registry_without_host_checkout(tmp_path):
    m2m, workload, source = tmp_path / "m2m", tmp_path / "workload", tmp_path / "source"
    (m2m / "m2m").mkdir(parents=True)
    (m2m / "m2m/__init__.py").write_text("")
    workload.mkdir()
    (workload / "loader.py").write_text("pass\n")
    source.mkdir()
    plan = {
        "m2m_root": str(m2m),
        "workload_root": str(workload),
        "merlin_root": str(module_source_path("merlin").parent),
        "schemas_root": str(schemas_dir()),
        "selected_trees": {"merlin": _source_tree(module_source_path("merlin").parent, skip_python_cache=True)},
    }
    sealed_m2m._stage_source(plan, source)
    selected_package = source / "merlin-src"
    site_packages = sysconfig.get_paths()["purelib"]
    program = (
        "import sys;sys.path[:0]=" + repr([str(selected_package), site_packages]) + ";"
        "import merlin;from merlin.common.quant_formats import names;"
        "print(merlin.__file__);print(len(names()))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program],
        env={},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == str(selected_package / "merlin/__init__.py")
    assert int(result.stdout.splitlines()[1]) > 0


def test_int8_recipe_selection_is_explicit_and_content_bound(tmp_path):
    recipe = {
        "schema": "quant_recipe_v1",
        "status": "derived",
        "software_numerical_engine": "integer_reference",
        "activation": {"dtype": "int8", "mode": "static"},
        "weight": {"dtype": "int8"},
    }
    recipe["recipe_sha256"] = recipe_digest(recipe)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    selected = _recipe_selection(path, dtype="int8")
    assert selected["path"] == str(path)
    assert selected["recipe_sha256"] == recipe["recipe_sha256"]
    with pytest.raises(SealedM2MError, match="must not select"):
        _recipe_selection(path, dtype="fp32")
    with pytest.raises(SealedM2MError, match="explicit recipe"):
        _recipe_selection(None, dtype="int8")
    recipe["activation"]["mode"] = "dynamic"
    path.write_text(json.dumps(recipe))
    with pytest.raises(SealedM2MError, match="static W8A8"):
        _recipe_selection(path, dtype="int8")
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        _recipe_selection(path, dtype="int8")


def test_int8_materialization_refuses_recipe_or_integer_reference_mismatch(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "capture"
    (source / "merlin-src/merlin/targetgen").mkdir(parents=True)
    (source / "inputs").mkdir()
    output.mkdir()
    (source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py").write_bytes(b"worker\n")
    recipe = {
        "schema": "quant_recipe_v1",
        "status": "derived",
        "software_numerical_engine": "integer_reference",
        "activation": {"dtype": "int8", "mode": "static"},
        "weight": {"dtype": "int8"},
    }
    recipe["recipe_sha256"] = recipe_digest(recipe)
    (source / "inputs/quant_recipe.json").write_text(json.dumps(recipe))
    (output / "meta.json").write_text(
        json.dumps(
            {
                "dtype": "int8",
                "scheme": "int8_static_act_int8_weight",
                "recipe_sha256": recipe["recipe_sha256"],
                "quantization_stats": {"recipe_sha256": recipe["recipe_sha256"]},
                "integerization_receipt": {"golden_agreement": {"status": "passed", "reference": "pt2e_integer"}},
            }
        )
    )
    monkeypatch.setattr(sealed_m2m, "_materialized", lambda *_, **__: {"status": "verified_materialized"})
    plan = {
        "dtype": "int8",
        "recipe": {
            "sha256": sealed_m2m._file_digest(source / "inputs/quant_recipe.json"),
            "bytes": (source / "inputs/quant_recipe.json").stat().st_size,
            "recipe_sha256": recipe["recipe_sha256"],
        },
    }
    assert sealed_m2m._materialized_v2(output, source, output, plan)["status"] == "verified_materialized"
    plan["recipe"]["recipe_sha256"] = "different"
    with pytest.raises(SealedM2MError, match="selected plan"):
        sealed_m2m._materialized_v2(output, source, output, plan)
    plan["recipe"]["recipe_sha256"] = recipe["recipe_sha256"]
    meta = json.loads((output / "meta.json").read_text())
    meta["integerization_receipt"]["golden_agreement"]["status"] = "failed"
    (output / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SealedM2MError, match="integer-reference agreement"):
        sealed_m2m._materialized_v2(output, source, output, plan)


def test_selected_fp32_staging_requires_audited_original_and_staged_abi(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "capture"
    source.mkdir()
    output.mkdir()
    (output / "meta.json").write_text(json.dumps({"dtype": "fp32"}))
    monkeypatch.setattr(sealed_m2m, "_materialized", lambda *_, **__: {"status": "verified_materialized"})
    plan = {"dtype": "fp32", "recipe": None, "worker_options": {"stage_fp32": True}}
    with pytest.raises(SealedM2MError, match="FP32 staging"):
        sealed_m2m._materialized_v2(output, source, output, plan)
    abi = [{"shape": [1, 2], "dtype": "f32"}]
    meta = {
        "dtype": "fp32",
        "input_abi": abi,
        "output_abi": abi,
        "precision_conversion": {
            "original_graph_sha256": "a" * 64,
            "staged_graph_sha256": "b" * 64,
            "graph_dtype_retargeting": "schema_float_dtype_operands",
            "dtype_decisions": [{"source_node_id": "node"}],
            "staged_precision_audit": {
                "status": "complete",
                "target_dtype": "torch.float32",
                "non_target_floating_values": 0,
                "checked_floating_values": 1,
            },
        },
        "fp32_staging": {
            "status": "observed",
            "scope": "original computation versus staged FP32; not accuracy equivalence",
            "original_graph_sha256": "a" * 64,
            "staged_graph_sha256": "b" * 64,
            "original_input_abi": [{"shape": [1, 2], "dtype": "bf16"}],
            "staged_input_abi": abi,
            "original_output_abi": [{"shape": [1, 2], "dtype": "bf16"}],
            "staged_output_abi": abi,
            "output_cardinality": 1,
            "output_metrics": [
                {
                    "shape": [1, 2],
                    "original_dtype": "bf16",
                    "staged_dtype": "f32",
                    "comparison": "observed_floating",
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                }
            ],
        },
    }
    trace = {"graphs": {"original": {"status": "complete", "sha256": "a" * 64}}}
    (output / "frontend-trace.json").write_text(json.dumps(trace))
    (output / "meta.json").write_text(json.dumps(meta))
    assert sealed_m2m._materialized_v2(output, source, output, plan)["status"] == "verified_materialized"
    float_to_int = json.loads(json.dumps(meta))
    float_to_int["fp32_staging"]["staged_input_abi"][0]["dtype"] = "i32"
    float_to_int["input_abi"][0]["dtype"] = "i32"
    assert "non-FP32 floating" in staging_error(float_to_int, selected=True, recipe=False, trace=trace)
    integer_widened = json.loads(json.dumps(meta))
    integer_widened["fp32_staging"]["original_input_abi"][0]["dtype"] = "i32"
    integer_widened["fp32_staging"]["staged_input_abi"][0]["dtype"] = "i64"
    integer_widened["input_abi"][0]["dtype"] = "i64"
    assert "exact integer/bool tensor dtype" in staging_error(integer_widened, selected=True, recipe=False, trace=trace)
    exact_changed = json.loads(json.dumps(meta))
    exact_changed["fp32_staging"]["original_output_abi"][0]["dtype"] = "i32"
    exact_changed["fp32_staging"]["staged_output_abi"][0]["dtype"] = "i32"
    exact_changed["output_abi"][0]["dtype"] = "i32"
    exact_changed["fp32_staging"]["output_metrics"][0].update(
        original_dtype="i32", staged_dtype="i32", comparison="exact_nonfloating", max_abs=1.0
    )
    assert "changed an exact integer/bool output" in staging_error(
        exact_changed, selected=True, recipe=False, trace=trace
    )
    inventory = [{"fqn": "layer", "kind": "Linear", "weight_shape": [2, 2], "stores_operand": True}]
    meta["fp32_staging"].update(
        source_layer_inventory=inventory,
        source_layer_inventory_sha256=hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        source_layer_plan_sha256="d" * 64,
    )
    meta["quantization_stats"] = {"plan_sha256": "d" * 64}
    assert staging_error(meta, selected=True, recipe=True, trace=trace) is None
    meta["fp32_staging"]["source_layer_inventory"][0]["weight_shape"] = [3, 2]
    assert "source-layer" in staging_error(meta, selected=True, recipe=True, trace=trace)
    meta["fp32_staging"]["source_layer_inventory"][0]["weight_shape"] = [2, 2]
    meta["precision_conversion"]["staged_graph_sha256"] = "c" * 64
    (output / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SealedM2MError, match="FP32 staging"):
        sealed_m2m._materialized_v2(output, source, output, plan)
    plan.pop("worker_options")
    with pytest.raises(SealedM2MError, match="FP32 staging"):
        sealed_m2m._materialized_v2(output, source, output, plan)


@pytest.mark.parametrize("version", ["v1", "v2", "v2-timed", "v3"])
def test_cpu_receipts_replay_only_under_their_selected_policy(tmp_path, monkeypatch, version):
    run = tmp_path / "run"
    source, runtime, output = run / "snapshots/source", run / "snapshots/guest-root", run / "capture"
    for path in (source, runtime, output):
        path.mkdir(parents=True)
    input_identity, output_identity = {"sha256": "input"}, {"sha256": "output"}
    process, materialized = {"returncode": 0}, {"status": "verified_materialized"}
    old = version == "v1"
    full = version == "v3"
    timed = version == "v2-timed"
    timeout = 3600 if full else 900 if timed else 120
    schema = sealed_m2m.SCHEMA_V1 if old else sealed_m2m.SCHEMA_V3 if full else sealed_m2m.SCHEMA
    command = _command(output) if old else _command_v2(output, dtype="int8", recipe=True)
    template = (
        (sealed_m2m._LAUNCH_PREFIX + sealed_m2m._LAUNCH_SUFFIX)
        if old
        else _command_v2(Path("/capture-out"), dtype="int8", recipe=True)[-1]
    )
    plan = {"schema": schema, "status": "plan_only", "command_template_sha256": sealed_m2m._digest(template.encode())}
    if not old:
        plan.update(
            {
                "dtype": "int8",
                "recipe": {"sha256": "selected"},
                "m2m_commit": "a" * 40,
                "base": "/usr",
                "merlin_root": "/selected/merlin",
                "schemas_root": "/selected/merlin/_data/schemas",
                "worker_sha256": "receipt",
                "selected_trees": {
                    name: input_identity for name in ("venv", "base", "m2m", "workload", "merlin", "schemas")
                },
            }
        )
    if timed:
        plan["execution_timeout_seconds"] = timeout
    if full:
        plan.update(
            execution_timeout_seconds=3600,
            loader_env={},
            loader_env_reads=[],
            selected_inputs=[{"role": "checkpoint", "guest_member": "weights.bin"}],
        )
        (source / "workload").mkdir()
        (source / "workload/loader.py").write_text("def get_model_and_inputs(): pass\n")
    receipt = {
        "schema": schema,
        "status": "pending_replay",
        "issuer_sha256": sealed_m2m._V1_ISSUER_SHA256 if old else "current-issuer",
        "nonce": "0" * 32,
        "plan": plan,
        "command": list(command),
        "policy_sha256": _policy(command, output, loader_env={} if full else None, timeout_seconds=timeout),
        "scope": sealed_m2m._V1_SCOPE if old else sealed_m2m._V3_SCOPE if full else sealed_m2m._V2_SCOPE,
        "source": input_identity,
        "guest_root": input_identity,
        **({"schemas": input_identity} if not old else {}),
        "output": output_identity,
        "process": process,
        "materialized": materialized,
        "bwrap_sha256": "bwrap",
    }
    (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(sealed_m2m, "_validate_snapshots", lambda *_, **__: None)
    monkeypatch.setattr(
        sealed_m2m,
        "_snapshot_tree",
        lambda path: (
            input_identity
            if Path(path).is_relative_to(source) or Path(path).is_relative_to(runtime)
            else output_identity
        ),
    )
    monkeypatch.setattr(sealed_m2m, "_materialized", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_materialized_v2", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_materialized_v3", lambda *_: materialized)
    monkeypatch.setattr(sealed_m2m, "_declared_loader_env", lambda *_: ({}, []))
    monkeypatch.setattr(sealed_m2m, "_verify_selected_input", lambda *_: None)
    executed = []
    monkeypatch.setattr(sealed_m2m, "_execute", lambda *_args, **kwargs: executed.append(kwargs) or process)
    monkeypatch.setattr(sealed_m2m, "_bwrap_binary", lambda *_: tmp_path / "bwrap")
    monkeypatch.setattr(
        sealed_m2m,
        "_file_digest",
        lambda path: (
            "bwrap"
            if Path(path).name == "bwrap"
            else "current-issuer"
            if Path(path).name == "sealed_m2m.py"
            else "receipt"
        ),
    )
    result = sealed_m2m.replay_verify(run)
    assert result["schema"] == schema
    assert result.get("capture_dtype") == (None if old else "int8")
    assert result["phase0_admission"] == "not_granted"
    assert result["status"] == "verified_sandbox_replay"
    # The replay runs under exactly the timeout the receipt's selected plan bound.
    assert executed and {call.get("timeout_seconds", 120) for call in executed} == {timeout}
    if timed:
        pending = run / "sealed_m2m_pending.json"
        for unbounded in (119, 14_401, "900", True):
            receipt["plan"]["execution_timeout_seconds"] = unbounded
            pending.write_text(json.dumps(receipt))
            with pytest.raises(SealedM2MError, match="bounded execution timeout"):
                sealed_m2m.replay_verify(run)
        receipt["plan"]["execution_timeout_seconds"] = 600
        pending.write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="unsupported policy"):
            sealed_m2m.replay_verify(run)
        receipt["plan"]["execution_timeout_seconds"] = timeout
        pending.write_text(json.dumps(receipt))
    if version == "v2":
        # These exact historical issuers differ only in strict JSON reads and
        # historical replay admission. Selected v2 still rechecks every byte.
        legacy_policy = receipt["policy_sha256"]
        receipt["capture_selection_sha256"] = "a" * 64
        receipt["policy_sha256"] = _policy(command, output, replayable_logs=True)
        receipt["issuer_sha256"] = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"
        pending = run / "sealed_m2m_pending.json"
        pending.write_text(json.dumps(receipt))
        assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"
        for field, changed in (
            ("issuer_sha256", "f" * 64),
            ("policy_sha256", "f" * 64),
            ("command", ["/unselected/runner"]),
            ("source", {"sha256": "different"}),
        ):
            original = receipt[field]
            receipt[field] = changed
            pending.write_text(json.dumps(receipt))
            with pytest.raises(SealedM2MError):
                sealed_m2m.replay_verify(run)
            receipt[field] = original
        raw = json.dumps(receipt).encode()
        pending.write_bytes(raw.replace(b'"issuer_sha256":', b'"issuer_sha256":"unknown","issuer_sha256":', 1))
        with pytest.raises(ValueError, match="unreadable"):
            sealed_m2m.replay_verify(run)
        receipt["issuer_sha256"] = "f1f36bc57807fbc4e93360757e9b0f891f38a70326ffc10c8067bfb0b5b11920"
        pending.write_text(json.dumps(receipt))
        assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"
        receipt["issuer_sha256"] = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"
        receipt["policy_sha256"] = legacy_policy
        receipt.pop("capture_selection_sha256")
        pending.write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="unsupported policy"):
            sealed_m2m.replay_verify(run)
        receipt["issuer_sha256"] = "current-issuer"
        pending.write_text(json.dumps(receipt))
    elif version == "v3":
        legacy_policy = receipt["policy_sha256"]
        receipt["capture_selection_sha256"] = "a" * 64
        receipt["policy_sha256"] = _policy(command, output, replayable_logs=True, loader_env={}, timeout_seconds=3600)
        receipt["issuer_sha256"] = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"
        receipt.pop("capture_selection_sha256")
        receipt["policy_sha256"] = legacy_policy
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="unsupported policy"):
            sealed_m2m.replay_verify(run)
        receipt["issuer_sha256"] = "current-issuer"
    else:
        receipt["issuer_sha256"] = "73f2463303b3a7274c666e44c99845dbe44953663aff3bfd346633fc7d391fea"
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="unsupported policy"):
            sealed_m2m.replay_verify(run)
        receipt["issuer_sha256"] = sealed_m2m._V1_ISSUER_SHA256
    if not old:
        receipt["issuer_sha256"] = sealed_m2m._PRE_FROZEN_ORIGIN_ISSUER_SHA256
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        assert sealed_m2m.replay_verify(run)["status"] == "verified_sandbox_replay"
        receipt["issuer_sha256"] = "current-issuer"
        if full:
            receipt["plan"]["frozen_origin"] = {"schema": "unselected"}
            (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
            with pytest.raises(SealedM2MError, match="not a v2 source selection"):
                sealed_m2m.replay_verify(run)
            receipt["plan"].pop("frozen_origin")
        receipt["schemas"] = {"sha256": "unselected"}
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="schema bytes differ"):
            sealed_m2m.replay_verify(run)
        receipt["schemas"] = input_identity
        receipt["plan"]["selected_trees"]["m2m"] = {"sha256": "unselected"}
        (run / "sealed_m2m_pending.json").write_text(json.dumps(receipt))
        with pytest.raises(SealedM2MError, match="selected m2m bytes differ"):
            sealed_m2m.replay_verify(run)


def test_plan_cannot_raise_snapshot_cap(tmp_path):
    with pytest.raises(SealedM2MError, match="no larger than 16 GB"):
        prepare_plan(
            m2m_root=tmp_path,
            workload_root=tmp_path,
            worker=tmp_path,
            venv=tmp_path,
            schemas_root=tmp_path,
            max_snapshot_bytes=16_000_000_001,
        )


def test_plan_rejects_enclosing_git_repository_as_m2m_origin(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.email", "test@example.invalid"], check=True)
    (outer / "marker").write_text("first\n")
    subprocess.run(["git", "-C", str(outer), "add", "marker"], check=True)
    subprocess.run(["git", "-C", str(outer), "commit", "-qm", "outer only"], check=True)
    frozen = outer / "frozen-m2m"
    (frozen / "m2m").mkdir(parents=True)
    (frozen / "m2m/api.py").write_text("# frozen package\n")
    workload = tmp_path / "workload"
    workload.mkdir()
    (workload / "loader.py").write_text("def get_model_and_inputs(): pass\n")
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3.12").write_text("python\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(base / "bin/python3.12")
    site = venv / "lib/python3.12/site-packages"
    (site / "torch").mkdir(parents=True)
    (site / "torch/_C.so").write_bytes(b"torch")
    (site / "numpy/_core").mkdir(parents=True)
    (site / "numpy/_core/_multiarray_umath.so").write_bytes(b"numpy")
    monkeypatch.setattr(sealed_m2m, "_capture_api_missing", lambda *_: ())
    monkeypatch.setattr(sealed_m2m, "_venv_home", lambda *_: base)
    monkeypatch.setattr(sealed_m2m, "_system_libs", lambda *_: ())
    with pytest.raises(SealedM2MError, match="own its Git repository"):
        prepare_plan(
            m2m_root=frozen,
            workload_root=workload,
            worker=module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py",
            venv=venv,
            schemas_root=schemas_dir(),
        )


def test_frozen_m2m_origin_ignores_outer_git_head_but_binds_copied_bytes(tmp_path, monkeypatch):
    from merlin_experiments.capture_execution.m2m_origin import frozen_selector, verify_frozen_receipt
    from merlin_experiments.phase0 import m2m_runtime

    def commit(root, message):
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", message], check=True)

    def init(root):
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)

    source = tmp_path / "true-m2m"
    init(source)
    (source / "m2m").mkdir()
    (source / "m2m/__init__.py").write_text("# package\n")
    (source / "m2m/api.py").write_text("# source bytes\n")
    commit(source, "selected M2M")
    outer = tmp_path / "outer-merlin"
    init(outer)
    (outer / "marker").write_text("one\n")
    commit(outer, "first outer revision")
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3.12").write_text("python\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(base / "bin/python3.12")
    site = venv / "lib/python3.12/site-packages"
    (site / "torch").mkdir(parents=True)
    (site / "torch/_C.so").write_bytes(b"torch")
    (site / "numpy/_core").mkdir(parents=True)
    (site / "numpy/_core/_multiarray_umath.so").write_bytes(b"numpy")
    monkeypatch.setattr(
        m2m_runtime,
        "_runtime",
        lambda *_: {
            "base": str(base),
            "venv": {"members": 1, "bytes": 0, "sha256": "0" * 64},
            "base_python": {"members": 1, "bytes": 0, "sha256": "0" * 64},
            "python_sha256": "0" * 64,
        },
    )
    selected = m2m_runtime.observe(source, venv / "bin/python", require_source_origin=True)
    (source / "marker").write_text("new source revision\n")
    commit(source, "new selected M2M revision")
    with pytest.raises(ValueError, match="runtime changed before freezing"):
        m2m_runtime.stage(selected, outer / "private/m2m-source")
    selected = m2m_runtime.observe(source, venv / "bin/python", require_source_origin=True)
    (source / "m2m/api.py").write_text("# edited source bytes\n")
    with pytest.raises(ValueError, match="clean pinned commit"):
        m2m_runtime.stage(selected, outer / "private/m2m-source")
    (source / "m2m/api.py").write_text("# source bytes\n")
    frozen = m2m_runtime.stage(selected, outer / "private/m2m-source")
    receipt_path = outer / "private/m2m-runtime.json"
    receipt_path.write_bytes(m2m_runtime.receipt(frozen))
    receipt_path.chmod(0o444)
    selector = frozen_selector(receipt_path)
    workload = tmp_path / "workload"
    workload.mkdir()
    (workload / "loader.py").write_text("def get_model_and_inputs(): pass\n")
    monkeypatch.setattr(sealed_m2m, "_capture_api_missing", lambda *_: ())
    monkeypatch.setattr(sealed_m2m, "_venv_home", lambda *_: base)
    monkeypatch.setattr(sealed_m2m, "_system_libs", lambda *_: ())
    arguments = {
        "m2m_root": Path(frozen["frozen_root"]),
        "frozen_origin": selector,
        "workload_root": workload,
        "worker": module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py",
        "venv": venv,
        "schemas_root": schemas_dir(),
    }
    first = prepare_plan(**arguments)
    # An unselected v2 timeout leaves the historical plan bytes (and fixed 120 s) unchanged; a
    # selected one is bound into the plan and must be bounded.
    assert "execution_timeout_seconds" not in first
    assert prepare_plan(**arguments, execution_timeout_seconds=900) == {**first, "execution_timeout_seconds": 900}
    for unbounded in (119, 14_401, "900", True, 900.0):
        with pytest.raises(SealedM2MError, match="between 120 and 14400"):
            prepare_plan(**arguments, execution_timeout_seconds=unbounded)
    assert first["m2m_commit"] == selected["source_origin"]["commit"]
    assert first["selected_trees"]["m2m"] == frozen["frozen_package"]
    assert first["estimate_bytes"] == sum(row["bytes"] for row in first["selected_trees"].values()) + len(
        receipt_path.read_bytes()
    )
    staged_source = tmp_path / "staged-source"
    staged_source.mkdir()
    (staged_source / "m2m-origin.json").write_bytes(receipt_path.read_bytes())
    with pytest.raises(SealedM2MError, match="revision differs from selected plan"):
        sealed_m2m._verify_staged_selection({**first, "m2m_commit": "0" * 40}, staged_source, tmp_path / "runtime")
    (outer / "marker").write_text("two\n")
    commit(outer, "second outer revision")
    assert prepare_plan(**arguments) == first
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.chmod(0o644)
    forged = json.loads(receipt_bytes)
    forged["source_origin"]["readonly_package"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(forged))
    receipt_path.chmod(0o444)
    with pytest.raises(SealedM2MError, match="origin receipt changed"):
        prepare_plan(**arguments)
    with pytest.raises(SealedM2MError, match="does not bind the selected copy"):
        prepare_plan(**{**arguments, "frozen_origin": frozen_selector(receipt_path)})
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(receipt_bytes)
    receipt_path.chmod(0o444)
    with pytest.raises(SealedM2MError, match="only supported for v2"):
        prepare_plan(**arguments, checkpoint=source / "checkpoint", loader_env={}, execution_timeout_seconds=120)
    copied_root = Path(frozen["frozen_root"])
    old_mode = copied_root.stat().st_mode & 0o777
    copied_root.chmod(old_mode | 0o200)
    (copied_root / ".git").mkdir()
    with pytest.raises(SealedM2MError, match="invalid frozen M2M origin selector"):
        prepare_plan(**arguments)
    assert verify_frozen_receipt(selector, receipt_bytes, copied_root, frozen["frozen_package"]) == first["m2m_commit"]
    (copied_root / ".git").rmdir()
    copied_root.chmod(old_mode)
    copied = Path(frozen["frozen_root"]) / "m2m/api.py"
    copied.chmod(0o644)
    copied.write_text("# changed copied bytes\n")
    with pytest.raises(SealedM2MError, match="does not bind the selected copy"):
        prepare_plan(**arguments)
    receipt_path.unlink()
    assert (
        verify_frozen_receipt(selector, receipt_bytes, Path(frozen["frozen_root"]), frozen["frozen_package"])
        == first["m2m_commit"]
    )


def test_m2m_git_origin_is_own_repository_and_ignores_git_redirect_environment(tmp_path, monkeypatch):
    from merlin_experiments.capture_execution.m2m_origin import M2MOriginError, git_origin

    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(outer), "config", "user.email", "test@example.invalid"], check=True)
    (outer / "marker").write_text("outer\n")
    subprocess.run(["git", "-C", str(outer), "add", "marker"], check=True)
    subprocess.run(["git", "-C", str(outer), "commit", "-qm", "outer"], check=True)
    source = outer / "true-m2m"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    (source / "m2m").mkdir()
    (source / "m2m/__init__.py").write_text("# source\n")
    subprocess.run(["git", "-C", str(source), "add", "m2m"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "source"], check=True)
    selected = git_origin(source, clean=True)
    monkeypatch.setenv("GIT_DIR", str(outer / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(outer))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(outer))
    assert git_origin(source, clean=True) == selected
    (outer / "not-a-repository").mkdir()
    with pytest.raises(M2MOriginError, match="own its Git repository"):
        git_origin(outer / "not-a-repository", clean=True)
    (source / "m2m/__init__.py").write_text("# modified\n")
    with pytest.raises(M2MOriginError, match="clean pinned commit"):
        git_origin(source, clean=True)


def test_prefreeze_source_guard_recomputes_verified_m2m_origin(tmp_path, monkeypatch):
    from merlin_experiments import SpecError, runner
    from merlin_experiments.phase0 import m2m_runtime

    entrypoint = tmp_path / "installed/merlin_experiments/phase0/__main__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# selected Phase 0 entrypoint\n")
    name = "phase0:source:__main__.py"
    monkeypatch.setattr(runner, "_phase0_source_inputs", lambda: (entrypoint, {name: str(entrypoint)}))
    selected = {
        "root": str(tmp_path / "selected-m2m"),
        "python": str(tmp_path / "selected-venv/bin/python"),
        "source_origin": {"commit": "a" * 40},
    }
    observed = []

    def observe(*_args, require_source_origin=False, **_kwargs):
        observed.append(require_source_origin)
        return (
            selected
            if require_source_origin
            else {key: value for key, value in selected.items() if key != "source_origin"}
        )

    monkeypatch.setattr(m2m_runtime, "observe", observe)
    command = {
        "adapter": "capsule_derivation",
        "module": runner.PHASE0_MODULE,
        "argv": [str(tmp_path / "python"), "-m", runner.PHASE0_MODULE],
        "inputs": {},
        "phase0_m2m_selection": selected,
        "env": {"PYTHONSAFEPATH": "1", "PYTHONPATH": str(entrypoint.parent.parent.parent)},
        "entrypoint": str(entrypoint),
    }
    plan = {"phases": {"0": command}, "input_paths": {name: str(entrypoint)}}
    runner._verify_phase0_sources(plan)
    assert observed == [True]
    command["phase0_m2m_selection"] = {**selected, "source_origin": {"commit": "b" * 40}}
    with pytest.raises(SpecError, match="runtime changed before freezing"):
        runner._verify_phase0_sources(plan)


def test_ldd_dependency_parser_accepts_only_structural_library_paths():
    assert _ldd_library_path("libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x123)") == Path(
        "/lib/x86_64-linux-gnu/libc.so.6"
    )
    assert _ldd_library_path("/lib64/ld-linux-x86-64.so.2 (0x123)") == Path("/lib64/ld-linux-x86-64.so.2")
    assert _ldd_library_path("linux-vdso.so.1 (0x123)") is None
    assert _ldd_library_path("libmissing.so => not found") is None


@pytest.mark.parametrize("schema", [sealed_m2m.SCHEMA_V1, sealed_m2m.SCHEMA])
def test_scoped_replay_proof_cannot_be_used_as_phase0_admission(schema):
    with pytest.raises(AttestationNotVerified):
        require_verified_execution(
            {
                "schema": schema,
                "status": "verified_sandbox_replay",
                "sealed_source_closure_replayed": True,
                "phase0_admission": "not_granted",
            }
        )
