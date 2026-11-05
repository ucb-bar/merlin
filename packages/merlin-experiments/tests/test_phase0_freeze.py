"""Real guarded Phase 0 execution owns copied sources and selected evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from merlin_experiments.measured_launch import FROZEN_PHASE0_ENV_POLICY, execution_environment
from merlin_experiments.phase0 import freeze
from merlin_experiments.spec import SpecError

from merlin.common.paths import data_path


def test_phase0_evidence_observes_selected_capture_interpreter(monkeypatch):
    selected = Path("/selected/capture/bin/python")
    observed = {}

    def capture(target, **kwargs):
        observed.update(target=target, **kwargs)
        return object()

    monkeypatch.setattr(freeze, "select_evidence", capture)
    command = {
        "inputs": {"descriptor": "target.yaml"},
        "phase0_m2m_selection": {"python": str(selected)},
    }
    freeze.selection(command, "fixture")
    assert observed["target"] == "fixture"
    assert observed["capture_python"] == str(selected)


def test_frozen_phase0_launch_uses_only_selected_environment(monkeypatch):
    monkeypatch.setenv("MERLIN_M2M_PYTHON", "/unselected/python")
    monkeypatch.setenv("MERLIN_MODEL2MLIR", "/unselected/source")
    monkeypatch.setenv("SPECIR_ROOT", "/unselected/model")
    command = {
        "adapter": "capsule_derivation",
        "source_snapshot": "/frozen/source",
        "phase0_environment_policy": FROZEN_PHASE0_ENV_POLICY,
        "env": {"MERLIN_REPO_ROOT": "/frozen/source", "SPECIR_ROOT": "/frozen/model"},
    }
    selected = execution_environment(command)
    assert selected["SPECIR_ROOT"] == "/frozen/model"
    assert selected["MERLIN_REPO_ROOT"] == "/frozen/source"
    assert "MERLIN_M2M_PYTHON" not in selected
    assert "MERLIN_MODEL2MLIR" not in selected
    with pytest.raises(SpecError, match="freeze a new run"):
        execution_environment({**command, "phase0_environment_policy": None})
    assert execution_environment({**command, "source_snapshot": None})["MERLIN_M2M_PYTHON"] == "/unselected/python"


@pytest.fixture(autouse=True)
def _writable_test_copies(tmp_path):
    yield
    for path in (tmp_path, *tmp_path.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


def _fixture(tmp_path, model_selector):
    spec = importlib.util.spec_from_file_location(
        "reviewed_corpus_fixture", Path(__file__).with_name("reviewed_corpus_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    fixture = helper.build_phase0_handoff(tmp_path)
    installed = fixture["installed"]
    # Match actual wheel layout, not checkout-only resource siblings. The
    # frozen launch must resolve these from the copied package, never live code.
    for resource in ("schemas", "contract"):
        shutil.copytree(
            data_path(resource),
            installed / "merlin/_data" / resource,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("capsules", "__pycache__"),
        )
    support = fixture["workspace"] / "explicit-target-resources"
    contract = {
        "name": "fixture-device",
        "compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"], "accumulator_dtype": "i32"}],
        "capabilities": {"mesh": {"rows": 2, "cols": 2}},
        "encoding": {"semantic_class": {"0": "FIXTURE_ISSUE"}, "corpus_issue_order": ["FIXTURE_ISSUE"]},
    }
    (support / "contracts/target_contract.yaml").write_text(yaml.safe_dump(contract))
    (support / "__init__.py").write_text("# synthetic selected provider\n")
    (support / ".env").write_text("SHOULD_NEVER_BE_COPIED=private\n")
    (installed / "merlin/.env.local").write_text("SHOULD_NEVER_BE_COPIED=private\n")
    entrypoint = installed / "merlin_experiments/phase0/__main__.py"
    probe = (
        "import os\n"
        "from pathlib import Path\n"
        "from merlin.integrations.specir import importable\n"
        "model_root = Path(os.environ['MERLIN_PHASE0_NUMERICAL_MODEL_ROOT'])\n"
        "assert model_root.is_relative_to(Path(os.environ['MERLIN_REPO_ROOT']))\n"
        "with importable(model_root):\n"
        "    import specir\n"
        "    assert specir.IDENTITY == 'captured model'\n"
    )
    entrypoint.write_text(
        entrypoint.read_text().replace(
            "from __future__ import annotations\n", "from __future__ import annotations\n\n" + probe
        )
    )
    model = tmp_path / "selected-numerical-model"
    (model / "specir").mkdir(parents=True)
    (model / "specir/__init__.py").write_text("IDENTITY = 'captured model'\n")
    (model / ".env").write_text("SHOULD_NEVER_BE_COPIED=private\n")
    software = fixture["profiles"] / "software.yaml"
    software.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.software_spec.v1",
                "target": "fixture-device",
                "status": "unreviewed",
                "capability_contract": contract,
                "numerical_semantics": {
                    "model": {
                        "engine": "integer_reference",
                        **(
                            {"source_root_env": "SPECIR_ROOT"}
                            if model_selector == "env"
                            else {"source_root_path": str(model)}
                        ),
                    },
                    "operand_dtype": "int8",
                    "accumulator_dtype": "i32",
                    "readout_dtype": "i32",
                    "subnormal_operand_flush": False,
                },
                "operations": [
                    {
                        "id": "matmul",
                        "ops": ["matmul"],
                        "placement": "unknown",
                        "signature": {"operand_dtypes": ["int8"]},
                    }
                ],
                "evidence": {"status": "synthetic diagnostic fixture"},
            }
        )
    )
    facts = fixture["profiles"] / "facts.json"
    facts.write_text(json.dumps({"facts": {"arrays": [{"rows": 2, "cols": 2}], "memories": []}}))
    conformance = fixture["profiles"] / "conformance.yaml"
    contract_bytes = (json.dumps(contract, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    conformance.write_text(
        yaml.safe_dump(
            {
                "target": "fixture-device",
                "application_demands": {"sidecar": "demands.json"},
                "derivation": {"phase0_execution": {"contract_sha256": hashlib.sha256(contract_bytes).hexdigest()}},
            }
        )
    )
    (fixture["profiles"] / "demands.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_declared",
                "coverage_status": "not_applicable",
                "applications": {},
                "n_operations": 0,
            }
        )
        + "\n"
    )
    document = yaml.safe_load(fixture["definition"].read_text())
    config = document["phases"]["0"]["config"]
    del config["profiles_root"]
    config.update(
        recipe=str(fixture["profiles"] / "fixture-device.yaml"),
        performance_template=str(fixture["profiles"] / "_perf.yaml"),
        synth_profile=str(fixture["profiles"] / "absent-synth.yaml"),
        conformance_spec=str(conformance),
        software_spec=str(software),
        rtl_facts=str(facts),
        evidence_mode="diagnostic",
    )
    # Installed qualification can reuse one actual producer-bound capture. Its
    # copied synthesis tree must remain executable after its original owner is
    # removed, without recapturing the framework model.
    capture = os.environ.get("MERLIN_TEST_MATERIALIZED_CAPTURE")
    if capture:
        from merlin_experiments.phase0.evidence import _materialize_evidence
        from merlin_experiments.phase0.requirements import _materialized_iteration_capsules

        original = tmp_path / "original-materialized-capture"
        shutil.copytree(Path(capture).parent, original)
        fixture["capture_source"] = original
        source = original / "model.mlir"
        application = {
            "capture_source_path": str(source),
            "capture_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "capture_receipt": {
                "receipt_sha256": hashlib.sha256((source.parent / "capture_receipt.json").read_bytes()).hexdigest()
            },
            "n_operations": len(json.loads((original / "frontend-trace.json").read_bytes())["mlir"]["operations"]),
        }
        entries, outputs = _materialized_iteration_capsules({"applications": {"iteration": application}}, "0" * 64)
        _materialize_evidence(fixture["profiles"], outputs)
        synthesis = fixture["profiles"] / "synthesis.yaml"
        synthesis.write_text(yaml.safe_dump({"capsules": entries}))
        config["synth_profile"] = str(synthesis)
        guard = (
            "from merlin.targetgen import capsule_source\n"
            "def forbidden_recapture(*args, **kwargs):\n"
            "    raise AssertionError('frozen materialized capsule must not recapture')\n"
            "capsule_source.PytorchRefSource = forbidden_recapture\n"
        )
        entrypoint.write_text(
            entrypoint.read_text().replace(
                "from __future__ import annotations\n", "from __future__ import annotations\n\n" + guard
            )
        )
    fixture["definition"].write_text(yaml.safe_dump(document))
    fixture["environment"].update(PYTHONPATH=str(installed), MERLIN_TARGET_PATH=str(support), SPECIR_ROOT=str(model))
    fixture.update(support=support, model=model)
    return fixture


@pytest.mark.parametrize("model_selector", ["env", "path"])
def test_actual_guarded_runner_executes_after_original_sources_are_deleted(tmp_path, monkeypatch, model_selector):
    fixture = _fixture(tmp_path, model_selector)
    driver = tmp_path / "stage-and-delete.py"
    driver.write_text(
        "import json, os, shutil, subprocess\n"
        "from pathlib import Path\n"
        "from merlin_experiments.runner import resolve_plan, run\n"
        "from merlin_experiments.spec import load_spec\n"
        "from merlin_experiments.phase0 import freeze\n"
        "original_popen = subprocess.Popen\n"
        "def launch(argv, *args, **kwargs):\n"
        "    if '--execute' in argv:\n"
        "        for root in json.loads(os.environ['DELETE_ROOTS']): shutil.rmtree(root)\n"
        "    return original_popen(argv, *args, **kwargs)\n"
        "subprocess.Popen = launch\n"
        "raise SystemExit(run(resolve_plan(load_spec(os.environ['DEFINITION']), phase='0', "
        "run_dir=Path(os.environ['RUN_DIR']))))\n"
    )
    descriptor = Path(yaml.safe_load(fixture["definition"].read_text())["phases"]["0"]["config"]["descriptor"])
    env = dict(
        fixture["environment"],
        DEFINITION=str(fixture["definition"]),
        RUN_DIR=str(fixture["run"]),
        DELETE_ROOTS=json.dumps(
            [
                str(fixture[key])
                for key in ("installed", "support", "profiles", "model", "capture_source")
                if key in fixture
            ]
            + [str(descriptor.parent)]
        ),
    )
    result = subprocess.run(
        [sys.executable, "-P", str(driver)],
        cwd=fixture["workspace"],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    logs = "\n".join(path.read_text() for path in fixture["run"].glob("*.log"))
    assert result.returncode == 0, result.stdout + result.stderr + logs
    plan = json.loads((fixture["run"] / "resolved-plan.json").read_text())
    snapshot = Path(plan["phase0_source_snapshot"])
    command = plan["phases"]["0"]
    assert command["frozen_launch"][1:4] == ["-I", "-S", "-B"]
    assert Path(command["frozen_launch"][4]).is_relative_to(snapshot)
    assert Path(command["entrypoint"]).is_relative_to(snapshot)
    model_root = Path(command["env"]["MERLIN_PHASE0_NUMERICAL_MODEL_ROOT"])
    assert model_root.is_relative_to(snapshot)
    assert (model_root / "specir/__init__.py").read_text() == "IDENTITY = 'captured model'\n"
    if model_selector == "env":
        assert command["env"]["SPECIR_ROOT"] == str(model_root)
    assert not list(snapshot.rglob(".env*"))
    assert all(path.resolve().is_relative_to(snapshot) for path in snapshot.rglob("*") if path.is_symlink())
    captured_inventory = json.loads((Path(command["inputs"]["conformance_spec"]).parent / "demands.json").read_text())
    assert captured_inventory["status"] == "not_declared"
    accounting = json.loads((fixture["run"] / "phase0/coverage/operation-accounting.json").read_text())
    assert accounting["status"] == "not_declared"
    assert (fixture["run"] / "phase0/software/quantization-contract.json").is_file()
    if os.environ.get("MERLIN_TEST_MATERIALIZED_CAPTURE"):
        materialized = freeze._materialized_inputs(command)
        assert len(materialized) >= 8
        assert all(path.is_relative_to(snapshot) for path in materialized)
        assert all(relative.startswith("materialized/iteration/") for _, relative in materialized.values())
        assert {name for name in plan["inputs"] if name.startswith("phase0:materialized:")}
        assert (fixture["run"] / "phase0/capsules/model/SY_source_iteration/frontend-evidence.json").is_file()
    else:
        assert not Path(command["inputs"]["synth_profile"]).exists()
    golden = yaml.safe_load((fixture["run"] / "phase0/capsules/isa/generated_member/golden.yaml").read_text())
    assert golden["golden_source"] == "merlin_tensor_int"
    assert golden["outputs"]
    freeze.verify(plan)
    changed = copy.deepcopy(plan)
    changed["phases"]["0"]["env"]["MERLIN_PHASE0_NUMERICAL_MODEL_ROOT"] = "/missing-model"
    with pytest.raises(ValueError, match="numerical model routing changed"):
        freeze.verify(changed)
    rerun = subprocess.run(
        command["frozen_launch"],
        cwd=snapshot,
        env=dict(os.environ, **command["env"]),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    from merlin_experiments.runner import resume

    monkeypatch.setenv("MERLIN_OUT_ROOT", plan["storage_root"])
    assert resume(fixture["run"]) == 0


def test_legacy_metadata_does_not_trigger_source_freezing():
    plan = {"phases": {"0": {"phase0_evidence": {"status": "diagnostic"}, "inputs": {}}}}
    assert freeze.stage(plan) == plan


def test_explicit_software_selection_requires_planned_evidence():
    with pytest.raises(ValueError, match="lacks planned"):
        freeze.stage({"phases": {"0": {"inputs": {"software_spec": "explicit.yaml"}}}})


def test_selected_m2m_runtime_is_explicit_and_rechecked_without_original_source(tmp_path):
    from merlin_experiments.phase0 import m2m_runtime

    source = tmp_path / "model2mlir"
    (source / "m2m").mkdir(parents=True)
    (source / "m2m/__init__.py").write_text("# selected package\n")
    cache = source / "m2m/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"transient bytecode")
    (source / "workloads" / "small_model").mkdir(parents=True)
    (source / "workloads" / "small_model" / "loader.py").write_text("# selected model loader\n")
    synth = tmp_path / "synthesis.yaml"
    synth.write_text(yaml.safe_dump({"capsules": [{"kind": "model", "model": "small_model"}]}))
    base = tmp_path / "python-base"
    (base / "bin").mkdir(parents=True)
    executable = base / "bin/python3.12"
    executable.write_bytes(b"selected interpreter bytes")
    executable.chmod(0o755)
    venv = tmp_path / "capture-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    (venv / "bin/python").symlink_to(executable)
    (venv / "site-module.py").write_text("VERSION = 1\n")

    selected = m2m_runtime.observe(source, venv / "bin/python", synth_profile=synth)
    assert selected["same_conversion_capture_api"]["status"] == "incompatible"
    assert "m2m/capture/provenance.py" in selected["same_conversion_capture_api"]["missing"]
    assert selected["frontend_trace_api"]["status"] == "incompatible"
    assert "m2m/capture/trace.py" in selected["frontend_trace_api"]["missing"]
    assert selected["static_integer_reference_api"]["status"] == "incompatible"
    assert "m2m/capture/pt2e_integerize.py" in selected["static_integer_reference_api"]["missing"]
    (cache / "other.cpython-312.pyc").write_bytes(b"later transient bytecode")
    assert m2m_runtime.observe(source, venv / "bin/python", synth_profile=synth)["package"] == selected["package"]
    frozen = m2m_runtime.stage(selected, tmp_path / "run/m2m-source")
    from merlin_experiments.capture_execution.sealed_m2m import _source_tree

    assert frozen["frozen_package"] == _source_tree(Path(frozen["frozen_root"]) / "m2m")
    assert not (Path(frozen["frozen_root"]) / "m2m/__pycache__").exists()
    shutil.rmtree(source)
    m2m_runtime.verify(frozen)
    m2m_runtime.verify_frozen_copy(frozen)
    assert (Path(frozen["frozen_root"]) / "workloads/small_model/loader.py").is_file()
    assert m2m_runtime.environment(frozen)["MERLIN_MODEL2MLIR"] == frozen["frozen_root"]
    assert frozen["phase0_admission"] == "not_granted"
    (venv / "site-module.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="host runtime changed"):
        m2m_runtime.verify(frozen)
    m2m_runtime.verify_frozen_copy(frozen)
    (venv / "site-module.py").write_text("VERSION = 1\n")
    frozen_package = Path(frozen["frozen_root"]) / "m2m"
    frozen_package.chmod(frozen_package.stat().st_mode | 0o200)
    (frozen_package / "__pycache__").mkdir(mode=0o555)
    frozen_package.chmod(frozen_package.stat().st_mode & ~0o222)
    with pytest.raises(ValueError, match="source package changed"):
        m2m_runtime.verify(frozen)
    with pytest.raises(ValueError, match="source package changed"):
        m2m_runtime.verify_frozen_copy(frozen)


@pytest.mark.parametrize("sealed", [False, True])
def test_installed_phase0_freezes_selected_m2m_routing_and_resumes(tmp_path, monkeypatch, sealed):
    from merlin_experiments.phase0 import m2m_runtime

    monkeypatch.setattr(m2m_runtime, "preflight_capture_sandbox", lambda value=None: str(value or "/selected/bwrap"))
    fixture = _fixture(tmp_path, "path")
    runtime = fixture["installed"] / "merlin_experiments/phase0/m2m_runtime.py"
    runtime.write_text(
        runtime.read_text()
        + '\n# This synthetic installed fixture never executes a framework capture.\n'
        + 'def preflight_capture_sandbox(value=None):\n    return str(value or "/selected/bwrap")\n'
    )
    source = tmp_path / "selected-m2m"
    (source / "m2m").mkdir(parents=True)
    (source / "m2m/__init__.py").write_text("# selected package\n")
    base = tmp_path / "selected-base"
    (base / "bin").mkdir(parents=True)
    executable = base / "bin/python3.12"
    executable.write_bytes(b"selected interpreter bytes")
    executable.chmod(0o755)
    venv = tmp_path / "selected-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    (venv / "bin/python").symlink_to(executable)
    document = yaml.safe_load(fixture["definition"].read_text())
    document["phases"]["0"]["config"].update(m2m_root=str(source), m2m_python=str(venv / "bin/python"))
    fixture["definition"].write_text(yaml.safe_dump(document))
    driver = tmp_path / "run-selected.py"
    driver.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from merlin_experiments.runner import resolve_plan, run\n"
        "from merlin_experiments.spec import load_spec\n"
        "plan = resolve_plan(load_spec(os.environ['DEFINITION']), phase='0', run_dir=Path(os.environ['RUN_DIR']))\n"
        "if os.environ['SELECT_SEALED'] == '1':\n"
        "    selected = plan['phases']['0']['phase0_m2m_selection']\n"
        "    root = Path(plan['run_dir']) / 'phase0/private'\n"
        "    config = {'m2m_root': selected['root'], 'package': selected['package'], "
        "'venv': str(Path(selected['python']).parent.parent), 'runs_root': str(root / 'sealed-captures'), "
        "'tmp_root': str(root / 'tmp')}\n"
        "    plan['phases']['0']['env']['MERLIN_PHASE0_SEALED_CAPTURE'] = json.dumps(config, sort_keys=True)\n"
        "raise SystemExit(run(plan))\n"
    )
    environment = dict(
        fixture["environment"],
        DEFINITION=str(fixture["definition"]),
        RUN_DIR=str(fixture["run"]),
        SELECT_SEALED="1" if sealed else "0",
    )
    result = subprocess.run(
        [sys.executable, "-P", str(driver)],
        cwd=fixture["workspace"],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    logs = "\n".join(path.read_text() for path in fixture["run"].glob("*.log"))
    assert result.returncode == 0, result.stdout + result.stderr + logs
    plan = json.loads((fixture["run"] / "resolved-plan.json").read_text())
    selected = plan["phases"]["0"]["phase0_m2m_selection"]
    assert Path(selected["frozen_root"]).is_relative_to(fixture["run"])
    assert plan["phases"]["0"]["env"]["MERLIN_M2M_DIR"] == selected["frozen_root"]
    assert selected["phase0_admission"] == "not_granted"
    if sealed:
        config = json.loads(plan["phases"]["0"]["env"]["MERLIN_PHASE0_SEALED_CAPTURE"])
        assert config["m2m_root"] == selected["frozen_root"]
        assert config["package"] == selected["frozen_package"]
    shutil.rmtree(source)
    freeze.verify(plan)
    from merlin_experiments.corpus.preparation import source_run

    assert source_run(fixture["run"])[1]["state"] == "execution_succeeded"
    drift = venv / "later-installed-module.py"
    drift.write_text("# unrelated later venv installation\n")
    assert source_run(fixture["run"])[1]["state"] == "execution_succeeded"
    from merlin_experiments import adapters

    def live_citation(*_args):
        raise AssertionError("completed consumption reopened current producer citation")

    with monkeypatch.context() as historical:
        historical.setattr(adapters, "_legacy_script", live_citation)
        assert source_run(fixture["run"])[1]["state"] == "execution_succeeded"
    with pytest.raises(ValueError, match="host runtime changed"):
        freeze.verify(plan)
    from merlin_experiments.runner import _verify_inputs

    with pytest.raises(SpecError, match="host runtime changed"):
        _verify_inputs(plan)
    drift.unlink()
    record_path = fixture["run"] / "orchestration.json"
    original_record = record_path.read_bytes()
    for invalid_returncode in (1, False):
        changed_record = json.loads(original_record)
        changed_record["attempts"][-1]["returncode"] = invalid_returncode
        record_path.write_text(json.dumps(changed_record))
        with pytest.raises(SpecError, match="successful immutable output receipt"):
            source_run(fixture["run"])
    record_path.write_bytes(original_record)
    manifest = Path(plan["phases"]["0"]["engine_output"]) / "MANIFEST.yaml"
    original_manifest = manifest.read_bytes()
    manifest.write_bytes(original_manifest + b"\n# unreviewed output mutation\n")
    with pytest.raises(SpecError, match="output changed"):
        source_run(fixture["run"])
    manifest.write_bytes(original_manifest)
    copied_package = Path(selected["frozen_root"]) / "m2m"
    copied_package.chmod(copied_package.stat().st_mode | 0o200)
    unexpected = copied_package / "unselected.py"
    unexpected.write_text("# source membership changed\n")
    copied_package.chmod(copied_package.stat().st_mode & ~0o222)
    with pytest.raises(SpecError, match="writable member"):
        source_run(fixture["run"])
    unexpected.chmod(0o444)
    with pytest.raises(SpecError, match="source package changed"):
        source_run(fixture["run"])
    copied_package.chmod(copied_package.stat().st_mode | 0o200)
    unexpected.unlink()
    alias = copied_package / "unselected.py"
    alias.symlink_to("__init__.py")
    copied_package.chmod(copied_package.stat().st_mode & ~0o222)
    with pytest.raises(SpecError, match="indirect"):
        source_run(fixture["run"])
    copied_package.chmod(copied_package.stat().st_mode | 0o200)
    alias.unlink()
    copied_package.chmod(copied_package.stat().st_mode & ~0o222)
    from merlin_experiments import source_snapshot

    snapshot = Path(plan["phase0_source_snapshot"])
    prior_mode = snapshot.stat().st_mode
    prior_seal = next(snapshot.glob("snapshot.*.json"))
    prior_bytes = prior_seal.read_bytes()
    external = tmp_path / "external-dependency"
    external.mkdir()
    snapshot.chmod(prior_mode | 0o200)
    prior_seal.unlink()
    (snapshot / "out").symlink_to(external, target_is_directory=True)
    amended = json.loads(prior_bytes)
    amended["external_links"]["out"] = str(external.resolve())
    new_seal = source_snapshot.seal(snapshot, "snapshot", amended)
    snapshot.chmod(prior_mode & ~0o222)
    with pytest.raises(SpecError, match="live external dependency links"):
        source_run(fixture["run"])
    snapshot.chmod(prior_mode | 0o200)
    new_seal.unlink()
    (snapshot / "out").unlink()
    prior_seal.write_bytes(prior_bytes)
    prior_seal.chmod(0o444)
    snapshot.chmod(prior_mode)
    if sealed:
        import tempfile

        from merlin_experiments.phase0.sealed_generation import SealedCaptureSource

        monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
        adapter = SealedCaptureSource(config)
        assert adapter.available()
        assert adapter.m2m_dir == Path(selected["frozen_root"])
        changed = copy.deepcopy(plan)
        config["m2m_root"] = str(source)
        changed["phases"]["0"]["env"]["MERLIN_PHASE0_SEALED_CAPTURE"] = json.dumps(config, sort_keys=True)
        with pytest.raises(ValueError, match="sealed Model2MLIR.*routing"):
            freeze.verify(changed)
    from merlin_experiments.runner import resume

    monkeypatch.setenv("MERLIN_OUT_ROOT", plan["storage_root"])
    assert resume(fixture["run"]) == 0
