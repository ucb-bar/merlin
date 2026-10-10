"""One selected measured timing record for installed and native Phase 1 readers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import timing
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import certification

from merlin.common.digest import sha256_file


def test_selected_timing_is_bound_to_target_config_and_simulator_bytes(tmp_path, monkeypatch):
    target = "synthetic_device"
    config = "SyntheticConfig"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic_device\n")
    simulator = tmp_path / "chipyard/sims/verilator" / f"simulator-chipyard.harness-{config}"
    simulator.parent.mkdir(parents=True)
    simulator.write_bytes(b"observed simulator bytes")
    selected = timing.timing_path(tmp_path, target)
    selected.write_text(
        json.dumps(
            {
                "target": target,
                "config": config,
                "verilator_per_capsule_s": 1500.0,
                "simulator_sha256": sha256_file(simulator),
                "measured_by": "fixture observation",
            }
        )
    )
    import merlin.common.paths as paths
    import merlin.targetgen.oracle_policy as engines
    import merlin.targetgen.target_experiment as experiments

    monkeypatch.setattr(paths, "ext_path", lambda name: tmp_path / "chipyard")
    monkeypatch.setattr(engines, "select_chipyard_engine", lambda name: {"engine": "verilator"})
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="chipyard")
    )
    monkeypatch.setattr(
        experiments, "declared_vs_resolved_contract", lambda selected: (None, tmp_path / "contract.yaml", "agree")
    )
    monkeypatch.setattr(
        experiments,
        "load_capability_manifest",
        lambda name, **kw: SimpleNamespace(contract={"runtime": {"rtl_sim_config": config}}),
    )
    context = SimpleNamespace(target=target, descriptor=descriptor, experiment=tmp_path)
    record = timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    assert record["verilator_per_capsule_s"] == 1500
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == 3000

    monkeypatch.setattr(engines, "select_chipyard_engine", lambda name: {"engine": "gsim"})
    with pytest.raises(ValueError, match="engine"):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    monkeypatch.setattr(engines, "select_chipyard_engine", lambda name: {"engine": "verilator"})

    with pytest.raises(ValueError, match="not bound to target"):
        timing.read_verified_timing(selected, descriptor=descriptor, target="other_device")
    alias = tmp_path / "alias.json"
    alias.symlink_to(selected)
    with pytest.raises(ValueError, match="symlink"):
        timing.read_verified_timing(alias, descriptor=descriptor, target=target)
    simulator.write_bytes(b"different simulator bytes")
    with pytest.raises(ValueError, match="simulator bytes changed"):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 2400
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == simjob._CERT_TIMEOUT_FALLBACK_S


def test_non_chipyard_simulator_keeps_legacy_timeout_without_chipyard_claim(tmp_path, monkeypatch):
    import merlin.targetgen.target_experiment as experiments

    target = "arc_device"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: arc_device\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    selected = scripts / f".oracle_timing.{target}.json"
    selected.write_text(json.dumps({"verilator_per_capsule_s": 1500}))
    context = SimpleNamespace(target=target, descriptor=descriptor, experiment=tmp_path)
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="mlc_arc")
    )
    monkeypatch.setattr(
        timing, "read_verified_timing", lambda *args, **kwargs: pytest.fail("Chipyard verifier used for arc")
    )
    assert not timing.requires_chipyard_timing(descriptor)
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == 3000


@pytest.fixture
def selected_engine(tmp_path, monkeypatch):
    """Real strict resolver/file controls, invented files, no provider execution."""
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import gsim_emulator as gsim
    from merlin.targetgen import oracle_policy as engines
    from merlin.targetgen import target_experiment as experiments

    target, config = "fixture_timing", "FixtureConfig"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("selected fixture descriptor\n")
    contract = tmp_path / "contract.yaml"
    contract.write_text("selected fixture contract\n")
    home = tmp_path / "engine"
    home.mkdir()
    binary = home / "emulator"
    binary.write_bytes(b"invented engine, never executed\n")
    binary.chmod(0o700)
    pins = {}
    for name in ("firrtl", "model_manifest", "gsim_emitter", "cxx_wrapper", "cxx_compiler", "harness"):
        path = home / name
        path.write_text(f"selected {name}\n")
        pins[name] = {"path": str(path), "sha256": sha256_file(path)}
    inputs = [{"role": "harness", **pins["harness"]}]
    commands = [
        {"stage": stage, "cwd": str(home), "argv": [pins[tool]["path"], stage]}
        for stage, tool in (("emit", "gsim_emitter"), ("compile", "cxx_wrapper"), ("link", "cxx_compiler"))
    ]
    receipt = home / gsim.RECEIPT_NAME
    receipt_doc = {
        "schema_version": gsim.STRICT_RECEIPT_SCHEMA,
        "status": "complete",
        "provenance": {
            "firrtl_boundary": gsim.FIRRTL_BOUNDARY_ADOPTED,
            "elaboration_performed": False,
            "warning": gsim.ADOPTED_FIRRTL_WARNING,
        },
        "firrtl_sha256": pins["firrtl"]["sha256"],
        "model_manifest_sha256": pins["model_manifest"]["sha256"],
        "binary_sha256": sha256_file(binary),
        "artifacts": {
            "firrtl": pins["firrtl"],
            "model_manifest": pins["model_manifest"],
            "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        },
        "tools": {name: pins[name] for name in ("gsim_emitter", "cxx_wrapper", "cxx_compiler")},
        "inputs": inputs,
        "inputs_sha256": gsim._canonical_sha(inputs),
        "commands": commands,
        "commands_sha256": gsim._canonical_sha(commands),
    }
    receipt.write_text(json.dumps(receipt_doc))
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {"target": target, "fir_sha256": pins["firrtl"]["sha256"], "firrtl_inputs": [pins["firrtl"]]},
                "facts": {"source": {"config": config}},
            }
        )
    )
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(gsim, "_env", lambda name: "")
    monkeypatch.setattr(gsim, "emulator_path", lambda name, **kwargs: binary)
    state = {"engine": "gsim", "binary": binary}
    backend = SimpleNamespace(
        available=lambda sim: sim == state["engine"],
        gsim_path=lambda: state["binary"],
        prepare_gsim_command=lambda *args, **kwargs: pytest.fail("engine must not execute"),
    )
    monkeypatch.setattr(backends, "get_backend", lambda name: backend)
    monkeypatch.setattr(engines, "select_chipyard_engine", lambda name: {"engine": state["engine"]})
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="chipyard")
    )
    monkeypatch.setattr(experiments, "declared_vs_resolved_contract", lambda selected: (None, contract, "agree"))
    monkeypatch.setattr(
        experiments,
        "load_capability_manifest",
        lambda name, **kw: SimpleNamespace(contract={"runtime": {"rtl_sim_config": config}}),
    )
    return SimpleNamespace(
        target=target,
        config=config,
        descriptor=descriptor,
        binary=binary,
        receipt=receipt,
        receipt_doc=receipt_doc,
        facts=facts,
        pins=pins,
        state=state,
        root=tmp_path,
    )


def _passed_observation(binding):
    return {
        "sim": binding["engine"],
        "all_pass": True,
        "_readiness_returncode": 0,
        "n_capsules": 1,
        "per_capsule": [
            {
                "capsule": "fixture_capsule",
                "pass": True,
                "barrier_tier": "L3",
                "barrier_status": "pass",
                "barrier_timing_identity": timing.barrier_timing_identity(
                    {
                        "status": "pass",
                        "engine": binding["engine"],
                        "sim_provenance": {
                            "engine": binding["engine"],
                            "binary": binding["simulator_path"],
                            "sha256": binding["simulator_sha256"],
                        },
                    }
                ),
            }
        ],
    }


def _write_observation(fixture, *, report=None, seconds=1500.0, before=None):
    binding = before or timing.selected_engine_binding(descriptor=fixture.descriptor, target=fixture.target)
    path = timing.timing_path(fixture.root, fixture.target)
    timing.write_observed_timing(
        path,
        descriptor=fixture.descriptor,
        target=fixture.target,
        before=binding,
        elapsed_s=seconds,
        report=report or _passed_observation(binding),
        measured_capsule="fixture_capsule",
        measured_by="fixture only; no executed engine",
    )
    return path


def test_gsim_observation_uses_strict_selected_engine_and_both_timeout_readers(selected_engine, monkeypatch):
    fixture = selected_engine
    path = _write_observation(fixture)
    record = timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)
    assert record["engine"] == "gsim"
    assert "verilator_per_capsule_s" not in record
    assert timing.observed_seconds(record) == 1500
    assert record["engine_binding"]["receipt"]["schema_version"] == "merlin.gsim-model-build.v3"
    context = SimpleNamespace(target=fixture.target, descriptor=fixture.descriptor, experiment=fixture.root)
    assert certification._verilator_per_capsule_timeout(context, timing_file=path) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    timeout, reason = simjob._per_capsule_timeout(0, context=context, timing_file=path)
    assert timeout == 3000
    assert "selected-engine" in reason and "verilator measurement" not in reason


def test_new_verilator_observation_preserves_the_same_timeout_consumers(selected_engine, monkeypatch):
    from merlin.common import paths

    fixture = selected_engine
    fixture.state["engine"] = "verilator"
    chipyard = fixture.root / "chipyard"
    simulator = chipyard / "sims/verilator" / f"simulator-chipyard.harness-{fixture.config}"
    simulator.parent.mkdir(parents=True)
    simulator.write_bytes(b"unexecuted fixture Verilator")
    monkeypatch.setattr(paths, "ext_path", lambda name: chipyard)
    path = _write_observation(fixture)
    record = timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)
    assert record["engine"] == "verilator" and timing.observed_seconds(record) == 1500
    context = SimpleNamespace(target=fixture.target, descriptor=fixture.descriptor, experiment=fixture.root)
    assert certification._verilator_per_capsule_timeout(context, timing_file=path) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=path)[0] == 3000


@pytest.mark.parametrize("changed", ["binary", "receipt", "facts", "firrtl", "gsim_emitter", "cxx_compiler", "harness"])
def test_gsim_observation_refuses_changed_selected_products(selected_engine, changed):
    fixture = selected_engine
    path = _write_observation(fixture)
    product = getattr(fixture, changed, None) or Path(fixture.pins[changed]["path"])
    product.write_bytes(product.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError):
        timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)


@pytest.mark.parametrize("changed", ["engine", "target", "config", "duration", "binding", "observation"])
def test_observation_refuses_resigned_foreign_invalid_or_wrong_engine_record(selected_engine, changed):
    fixture = selected_engine
    path = _write_observation(fixture)
    record = json.loads(path.read_text())
    if changed == "duration":
        record["per_capsule_s"] = True
    elif changed == "binding":
        record["engine_binding"]["simulator_sha256"] = "0" * 64
    elif changed == "observation":
        record["barrier_timing_identity"]["engine"] = "verilator"
    else:
        record[changed] = "foreign"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)


@pytest.mark.parametrize("missing", ["facts", "receipt", "strict", "future_schema", "backend_path", "foreign_facts"])
def test_gsim_timing_requires_real_selected_facts_strict_receipt_and_backend_path(
    selected_engine, monkeypatch, missing
):
    fixture = selected_engine
    if missing == "facts":
        monkeypatch.delenv("MERLIN_RTL_FACTS")
    elif missing == "receipt":
        fixture.receipt.unlink()
    elif missing in ("strict", "future_schema"):
        fixture.receipt_doc["schema_version"] = (
            "merlin.gsim-model-build.v2" if missing == "strict" else "merlin.gsim-model-build.v99"
        )
        fixture.receipt.write_text(json.dumps(fixture.receipt_doc))
    elif missing == "foreign_facts":
        doc = json.loads(fixture.facts.read_text())
        doc["inputs"]["target"] = "foreign"
        fixture.facts.write_text(json.dumps(doc))
    else:
        fixture.state["binary"] = fixture.root / "different-emulator"
        fixture.state["binary"].write_bytes(fixture.binary.read_bytes())
    with pytest.raises(ValueError):
        timing.selected_engine_binding(descriptor=fixture.descriptor, target=fixture.target)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "wrong_engine",
        "wrong_binary",
        "no_provenance",
        "failed",
        "partial",
        "bool_count",
        "wrong_capsule",
        "failed_process",
        "wrong_sim",
    ],
)
def test_writer_refuses_missing_or_wrong_actual_passed_barrier(selected_engine, invalid):
    fixture = selected_engine
    before = timing.selected_engine_binding(descriptor=fixture.descriptor, target=fixture.target)
    report = _passed_observation(before)
    row = report["per_capsule"][0]
    if invalid == "missing":
        del row["barrier_timing_identity"]
    elif invalid == "wrong_engine":
        row["barrier_timing_identity"]["engine"] = "verilator"
    elif invalid == "wrong_binary":
        row["barrier_timing_identity"]["simulator_sha256"] = "0" * 64
    elif invalid == "no_provenance":
        row["barrier_timing_identity"] = timing.barrier_timing_identity({"status": "pass", "engine": "gsim"})
    elif invalid == "failed":
        row["barrier_status"] = "fail"
    elif invalid == "partial":
        report["n_capsules"] = 2
    elif invalid == "bool_count":
        report["n_capsules"] = True
    elif invalid == "failed_process":
        report["_readiness_returncode"] = 1
    elif invalid == "wrong_sim":
        report["sim"] = "verilator"
    else:
        row["capsule"] = "another_capsule"
    with pytest.raises(ValueError):
        _write_observation(fixture, report=report, before=before)
    assert not timing.timing_path(fixture.root, fixture.target).exists()


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf"), 2**2048])
def test_writer_requires_positive_finite_real_duration(selected_engine, seconds):
    with pytest.raises(ValueError, match="observation"):
        _write_observation(selected_engine, seconds=seconds)


def test_writer_reopens_selected_binding_after_the_observation(selected_engine):
    fixture = selected_engine
    before = timing.selected_engine_binding(descriptor=fixture.descriptor, target=fixture.target)
    fixture.binary.write_bytes(b"changed during observation")
    with pytest.raises(ValueError):
        _write_observation(fixture, before=before)
    assert not timing.timing_path(fixture.root, fixture.target).exists()


def test_identical_frozen_facts_copy_is_reopened_by_bytes_not_original_path(selected_engine, monkeypatch):
    fixture = selected_engine
    path = _write_observation(fixture)
    snapshot = fixture.root / "frozen-facts.json"
    snapshot.write_bytes(fixture.facts.read_bytes())
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(snapshot))
    record = timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)
    assert record["engine_binding"]["rtl_facts"]["sha256"] == sha256_file(snapshot)
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="binding"):
        timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)


def test_actual_barrier_projection_is_path_free_and_never_borrows_a_requested_label():
    private = "/operator/private/selected-engine"
    tier = {
        "status": "pass",
        "engine": "gsim",
        "cycles": 19,
        "sim_provenance": {"engine": "gsim", "binary": private, "sha256": "a" * 64},
    }
    identity = timing.barrier_timing_identity(tier)
    assert identity["engine"] == "gsim" and identity["simulator_sha256"] == "a" * 64
    assert private not in json.dumps(identity)
    assert timing.barrier_timing_identity({"status": "pass", "engine": "gsim"}) is None
    tier["sim_provenance"]["engine"] = "verilator"
    assert timing.barrier_timing_identity(tier) is None
    tier["engine"] = private
    tier["sim_provenance"]["engine"] = private
    assert timing.barrier_timing_identity(tier) is None


def test_required_engine_mismatch_never_falls_back(selected_engine, monkeypatch):
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "verilator")
    with pytest.raises(ValueError, match="required engine"):
        timing.selected_engine_binding(descriptor=selected_engine.descriptor, target=selected_engine.target)


@pytest.mark.parametrize("failure", ["receipt", "backend"])
def test_selection_errors_do_not_disclose_private_paths(selected_engine, monkeypatch, failure):
    fixture = selected_engine
    private_path = "/operator/private/timing-support"
    if failure == "receipt":
        fixture.receipt_doc["artifacts"]["binary"]["sha256"] = "0" * 64
        fixture.receipt.write_text(json.dumps(fixture.receipt_doc))
    else:
        from merlin.runtime.backends import base as backends

        def unavailable(target):
            raise RuntimeError(private_path)

        monkeypatch.setattr(backends, "get_backend", unavailable)
    with pytest.raises(ValueError) as error:
        timing.selected_engine_binding(descriptor=fixture.descriptor, target=fixture.target)
    assert private_path not in str(error.value)
    assert str(fixture.root) not in str(error.value)


def test_unknown_or_unavailable_selected_engine_refuses_without_other_artifact_probe(selected_engine, monkeypatch):
    from merlin.targetgen import oracle_policy as engines

    selected_engine.state["engine"] = "another_engine"
    with pytest.raises(ValueError, match="selected engine"):
        timing.selected_engine_binding(descriptor=selected_engine.descriptor, target=selected_engine.target)

    def absent(target):
        raise RuntimeError("unavailable fixture engine")

    monkeypatch.setattr(engines, "select_chipyard_engine", absent)
    with pytest.raises(ValueError, match="no selected engine"):
        timing.selected_engine_binding(descriptor=selected_engine.descriptor, target=selected_engine.target)


def test_resigned_binding_bool_integer_substitution_is_refused(selected_engine):
    fixture = selected_engine
    path = _write_observation(fixture)
    record = json.loads(path.read_text())
    record["engine_binding"]["receipt"]["provenance"]["elaboration_performed"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="binding"):
        timing.read_verified_timing(path, descriptor=fixture.descriptor, target=fixture.target)


def test_ordinary_selfcheck_projects_only_actual_passed_barrier_identity(tmp_path, monkeypatch, capsys):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.feedback import selfcheck

    from merlin.common.paths import data_path

    monkeypatch.chdir(tmp_path)
    submission, corpus = tmp_path / "submission", tmp_path / "corpus"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    capsule = corpus / "fixture_capsule"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        json.dumps(
            {
                "name": "fixture_capsule",
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": "public",
                "operation": {"op": "movement"},
                "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
                "expected": {"instruction_classes": []},
                "required_oracle_tiers": ["L3"],
            }
        )
    )
    context = InvocationContext(
        tmp_path,
        tmp_path / "target.yaml",
        tmp_path,
        "fixture",
        tmp_path / "runs",
        tmp_path / "reports",
        tmp_path / "bundles",
        (),
    )
    monkeypatch.setattr(selfcheck, "_adapters", lambda *args: ({"L3": object()}, "gsim"))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *args: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *args: "fixture-suite")
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *args: None)
    private_path = "/operator/private/fixture-engine"
    tier = {
        "status": "pass",
        "engine": "gsim",
        "cycles": 19,
        "sim_provenance": {
            "engine": "gsim",
            "binary": private_path,
            "sha256": "a" * 64,
            "resolution": "private receipt details",
        },
    }

    def grade(_submission, *, runs_root, **kwargs):
        parent = Path(runs_root) / "runs" / "fixture-suite" / "fixture_capsule"
        parent.mkdir(parents=True)
        (parent / "capsule_result.json").write_text(
            json.dumps(
                {
                    "capsule": "fixture_capsule",
                    "kind": "op",
                    "status": "pass",
                    "numeric": {"status": "pass"},
                    "tiers": {"L3": tier},
                }
            )
        )
        return {"n_capsules": 1, "n_passed": 1, "per_capsule": [{"capsule": "fixture_capsule", "status": "pass"}]}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck.main(
        ["--sim", "gsim", "--submission", str(submission)],
        context=context,
        capsules_root=corpus,
        contract=data_path("contract"),
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["all_pass"] is True
    assert report["per_capsule"][0]["barrier_timing_identity"] == timing.barrier_timing_identity(tier)
    assert report["per_capsule"][0]["barrier_cycles"] == 19
    assert private_path not in json.dumps(report)
    assert "private receipt details" not in json.dumps(report)
