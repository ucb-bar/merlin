"""Retained products gain only newly completed ordinary static-check evidence."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import private_model_chain as chain
from merlin_experiments.phase1.feedback import private_static_products as products
from merlin_experiments.phase1.feedback import private_static_revalidation as replay

from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower.compilation_recipe import CompilationRecipe


def _write(path, value):
    path.write_text(json.dumps(value))
    return products.pin(path)


@pytest.fixture
def selected(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("candidate", "host", "capture")}
    for name, root in roots.items():
        root.mkdir()
        (root / "input").write_text(name)
    spec = tmp_path / "spec"
    spec.write_text("selected private inputs")
    receipts = {}
    for name in ("encode", "decode"):
        output = tmp_path / name
        build = output / "build"
        build.mkdir(parents=True)
        source = build / "producer.py"
        source.write_text("from pathlib import Path\nimport sys\n"
                          "Path(sys.argv[-1]).write_bytes(b'unchanged product')\n")
        elf = build / "model.elf"
        recipe = CompilationRecipe(build, producer=source)
        recipe.run([sys.executable, source, "-o", elf], runner=lambda argv: subprocess.run(argv, capture_output=True),
                   inputs=(source,), output=elf)
        recipe.completed(elf)
        receipts[name] = _write(output / "baremetal_model.json", {
            "schema": "merlin.baremetal-saved-model.v1", "status": "compiled", "execution_route": "host_baseline",
            "inputs": {"capture": str(roots["capture"]), "package": str(roots["host"]),
                       "capture_tree": strict_tree_sha256(roots["capture"]),
                       "package_tree": strict_tree_sha256(roots["host"]), "arena_mb": 8,
                       "reference_file": None, "rtl_facts": None, "device": None,
                       "host_math_policy": "native", "readback": "full", "host_llvm_bitcode": True,
                       "respect_captured_quantization_scope": True},
            "output": {"elf": str(elf), "elf_sha256": products.pin(elf)["sha256"],
                       "compilation_recipe": products.pin(recipe.path)},
        })
    run = tmp_path / "original-run"
    profile = {"selected_original": "unchanged"}
    record = {"schema": chain.SCHEMA, "scope": "canonical_static_only", "run_dir": str(run),
              "selected_model": "network", "target": "unit", "compiler_profile": profile,
              "candidate_tree_sha256": strict_tree_sha256(roots["candidate"])["sha256"],
              "private_spec_sha256": products.pin(spec)["sha256"],
              "required_programs": {"network": ["encode", "decode"]},
              "receipts": receipts, "inputs": [products.pin(spec)], "instruction_path": str(spec)}
    old = tmp_path / "old-native-static.json"
    _write(old, record)
    # This fixture isolates the new product/recipe gate from independent RTL issuance.
    monkeypatch.setattr(chain.partitions, "_selection", lambda *a, **k: None)
    kwargs = dict(run_dir=run, name="network", candidate=roots["candidate"], spec=spec,
                  compiler_profile=profile, programs=record["required_programs"])
    value = products.RetainedProducts(old, **kwargs)
    inputs = dict(capture=roots["capture"], host_package=roots["host"], candidate=roots["candidate"],
                  package_digest=strict_tree_sha256(roots["candidate"]), arena_mb=8, target="unit", device=None,
                  facts={}, options={"readback": "full", "host_llvm_bitcode": True}, math_policy="native",
                  directory=tmp_path / "fresh-policy")
    return SimpleNamespace(value=value, inputs=inputs, old=old, record=record, kwargs=kwargs, roots=roots)


def test_actual_completed_recipe_products_reopen_without_executing_any_generator(selected):
    before = {name: Path(identity["path"]).read_bytes() for name, identity in selected.record["receipts"].items()}
    for name in ("encode", "decode"):
        observed = selected.value.load(name, **selected.inputs)
        assert observed["status"] == "compiled"
    result = selected.value.record()
    assert result["original_producer_completion"] == "not_claimed"
    assert result["linked_policy_observations"] == {"encode": None, "decode": None}
    assert result["receipts"] == selected.record["receipts"]
    assert {name: Path(identity["path"]).read_bytes() for name, identity in result["receipts"].items()} == before


@pytest.mark.parametrize("changed", ["receipt", "recipe", "elf", "capture", "candidate", "policy"])
def test_changed_products_source_and_selected_numerics_refuse(selected, changed):
    receipt_path = Path(selected.record["receipts"]["encode"]["path"])
    receipt = json.loads(receipt_path.read_bytes())
    if changed == "receipt":
        receipt_path.write_text("{}")
    elif changed in {"recipe", "elf"}:
        path = Path(receipt["output"]["elf"] if changed == "elf" else receipt["output"]["compilation_recipe"]["path"])
        path.write_bytes(b"changed")
    elif changed in {"capture", "candidate"}:
        (selected.roots[changed] / "input").write_text("changed")
    else:
        selected.inputs["math_policy"] = "unselected"
    with pytest.raises(ValueError, match="changed|differs|different|selected|identity"):
        selected.value.load("encode", **selected.inputs)


def test_unissued_foreign_partial_and_changed_original_selections_refuse(selected):
    with pytest.raises(ValueError, match="issued"):
        products.RetainedProducts.__new__(products.RetainedProducts).verify()
    with pytest.raises(ValueError, match="omitted"):
        selected.value.record()
    with pytest.raises(ValueError, match="source or roster"):
        selected.value.require("network", ["encode"], selected.record["candidate_tree_sha256"],
                               selected.record["private_spec_sha256"])
    with pytest.raises(ValueError, match="run/source/profile"):
        products.RetainedProducts(selected.old, **{**selected.kwargs, "compiler_profile": {}})
    selected.old.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        selected.value.verify()


def test_issued_selection_carrier_cannot_replace_the_original_record(selected):
    selected.value.verify()[0]["candidate_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="issued original selection"):
        selected.value.verify()


@pytest.mark.parametrize("diagnostic,source_only,model", [(True, False, "network"), (False, True, "network"),
                                                          (False, False, None)])
def test_retained_route_rejects_diagnostic_source_only_or_unselected_gate(selected, diagnostic, source_only, model):
    with pytest.raises(ValueError, match="explicit independent"):
        products.require_selection(selected.value, selected_model=model, diagnostic=diagnostic, source_only=source_only)


def test_revalidation_scope_calls_only_fixed_ordinary_compile_coordinator(selected, monkeypatch):
    seen = []

    def coordinator(**kwargs):
        assert kwargs["retained_products"] is selected.value
        seen.append(kwargs)
        # A fresh refusal, including HA unknown, must propagate, never use an old clean record.
        raise ValueError("fresh host-compute audit refused")

    monkeypatch.setattr(chain, "_compile", coordinator)
    with pytest.raises(ValueError, match="fresh host-compute"):
        replay._revalidate(retained_products=selected.value)
    assert len(seen) == 1


def test_canonical_consumer_requires_distinct_exact_completed_revalidation_issuer(tmp_path, monkeypatch):
    path = tmp_path / "native-static.json"
    path.write_text("{}")
    record = {"schema": replay.SCHEMA, "run_dir": str(tmp_path), "selected_model": "network",
              "candidate_tree_sha256": "a" * 64,
              "retained_product_revalidation": {"original_partition": {"sha256": "b" * 64}}}
    call = {"kind": "python_call", "stage": replay.STAGE, "callable": replay.__name__ + "._revalidate",
            "arguments": {"run_dir": str(tmp_path), "model": "network", "candidate_tree_sha256": "a" * 64,
                          "original_partition_sha256": "b" * 64}, "outputs": [products.pin(path)]}
    monkeypatch.setattr(replay, "verify_record", lambda record: None)
    monkeypatch.setattr(replay.IR, "verify", lambda _: call)
    chain._producer(path, record, tmp_path / "invocation.json")
    for key, value in (("stage", "native_static_chain_partition"), ("callable", chain.__name__ + "._compile"),
                       ("arguments", {}), ("outputs", [])):
        altered = {**call, key: value}
        monkeypatch.setattr(replay.IR, "verify", lambda _: altered)
        with pytest.raises(ValueError, match="independent producer"):
            chain._producer(path, record, tmp_path / "invocation.json")
    monkeypatch.setattr(replay.IR, "verify", lambda _: (_ for _ in ()).throw(ValueError("not completed")))
    with pytest.raises(ValueError, match="not completed"):
        chain._producer(path, record, tmp_path / "invocation.json")


def test_trusted_checker_timeout_refuses_and_restores_alarm(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import private_instruction_coordinator

    monkeypatch.setattr(replay, "TRUSTED_CHECKER_TIMEOUT_S", 0.02)
    monkeypatch.setattr(private_instruction_coordinator, "prepare", lambda *a, **k: time.sleep(1))
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="120 seconds"):
        replay.prepare_instructions(object(), target="unit", output=tmp_path)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler


@pytest.mark.parametrize("changed", [None, "issuer", "candidate", "interface", "output"])
def test_original_compiler_emission_must_be_actual_and_match_all_selected_bytes(tmp_path, monkeypatch, changed):
    from merlin.common import invocation_record
    from merlin.targetgen import package_runtime
    from merlin.verify import lean_emission_context

    candidate = tmp_path / "candidate.py"
    candidate.write_text("selected candidate bytes")
    artifact = tmp_path / "dispatch.device.mlir"
    artifact.write_text("actual emitted LLVM")
    interface = tmp_path / "dispatch.iface.mlir"
    interface.write_text("exact interface")
    root = tmp_path / "dispatch.lowering"
    (root / "emission-custody-one/invocations/one").mkdir(parents=True)
    observation = root / "emission-custody-one/invocations/one/invocation.json"
    observation.write_text("actual observation fixture")
    (root / "input.interface.mlir").write_bytes(interface.read_bytes())
    (root / "lowered.llvm.mlir").write_bytes(artifact.read_bytes())
    call = {"kind": "python_call", "stage": "ordinary_source_emission",
            "callable": "merlin.targetgen.capsule_common.lower_interface", "inputs": [products.pin(interface)],
            "outputs": [products.pin(root / "lowered.llvm.mlir")], "dependencies": [products.pin(candidate)]}
    profile = {"profile_sha256": "p" * 64}
    pending = observation.parent.parent.parent / "emission_proof_chain.json"
    _write(pending, {"schema": "merlin.emission_pending.v1", "original_profile_sha256": "p" * 64,
                     "compiler_source_invocation": products.pin(observation),
                     "source": products.pin(interface),
                     "emitted_artifact": products.pin(root / "lowered.llvm.mlir")})
    monkeypatch.setattr(products, "_package_commands", lambda *args: [products.pin(observation)])
    monkeypatch.setattr(package_runtime, "load_package", lambda _: object())
    monkeypatch.setattr(lean_emission_context, "_package_sources", lambda _: (candidate,))
    monkeypatch.setattr(invocation_record, "verify", lambda _: call)
    if changed == "issuer":
        call["callable"] = "foreign.emitter"
    elif changed == "candidate":
        candidate.write_text("different candidate")
    elif changed == "interface":
        (root / "input.interface.mlir").write_text("different input")
    elif changed == "output":
        artifact.write_text("different LLVM")
    if changed:
        with pytest.raises(ValueError, match="selected original compiler/source"):
            products._emission_custody(artifact, candidate, profile=profile)
    else:
        assert products._emission_custody(artifact, candidate, profile=profile)["source"] == products.pin(observation)


@pytest.mark.parametrize("changed", [None, "issuer", "selection", "elf", "report"])
def test_fresh_instruction_policy_reopens_both_actual_observers(tmp_path, monkeypatch, changed):
    from merlin_experiments.phase2.contracts import document_sha256

    directory = tmp_path / "invocations/outer"
    policy = directory / "policy"
    inner = policy / "invocations/inner/invocation.json"
    inner.parent.mkdir(parents=True)
    inner.write_text("inner observer fixture")
    (directory / "invocation.json").write_text("outer observer fixture")
    elf = tmp_path / "model.elf"
    elf.write_bytes(b"actual ELF fixture")
    report = policy / "instruction_audit.json"
    result = {"status": "accepted", "elf_sha256": products.pin(elf)["sha256"],
              "policy_sha256": "p" * 64, "accessor_intake_sha256": "a" * 64,
              "decoder_selection": {"field": "selected"}, "prohibited_hits": []}
    _write(report, result)
    result.update(report_path=str(report), report_sha256=products.pin(report)["sha256"])
    record = {"target": "unit", "instruction_path": str(tmp_path / "selection"), "instruction_selection": {
        "instruction_check_sha256": document_sha256({"policy_sha256": result["policy_sha256"],
            "accessor_sha256": result["accessor_intake_sha256"], "selection": result["decoder_selection"]})}}
    outer_call = {"stage": "device_final_linked_elf_policy",
                  "callable": "merlin.targetgen.contract.elf_admission.LinkedElfAdmissionService.evaluate",
                  "arguments": {"target": "unit", "scope": "selected final linked-artifact policy"},
                  "outputs": [products.pin(report)]}
    inner_call = {"stage": "whole_linked_instruction_policy",
                  "callable": "merlin_experiments.phase2.component_instruction_audit._scan",
                  "arguments": {key: result[key] for key in (
                      "decoder_selection", "policy_sha256", "accessor_intake_sha256")},
                  "inputs": [products.pin(elf)], "outputs": [products.pin(report)]}
    monkeypatch.setattr(replay.IR, "verify", lambda path: inner_call if path == inner else outer_call)
    monkeypatch.setattr(chain.partitions, "_selection", lambda *a, **k: None)
    receipt = {"output": {"elf": str(elf), "elf_sha256": result["elf_sha256"]}}
    if changed == "issuer":
        inner_call["callable"] = "foreign.checker"
    elif changed == "selection":
        record["instruction_selection"]["instruction_check_sha256"] = "0" * 64
    elif changed == "elf":
        receipt["output"]["elf_sha256"] = "0" * 64
    elif changed == "report":
        report.write_text("{}")
    if changed:
        with pytest.raises(ValueError, match="changed|selected fresh checker"):
            replay._policy_observation(result, record, receipt)
    else:
        replay._policy_observation(result, record, receipt)


def test_retained_selection_can_never_fall_back_to_candidate_compilation(selected, monkeypatch, tmp_path):
    from contextlib import nullcontext

    from merlin_experiments.phase1.feedback import private_facts, private_result_format_selection

    from merlin.targetgen import target_experiment

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    monkeypatch.setattr(chain, "build_dir", lambda: tmp_path / "fresh-products")
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: SimpleNamespace(target="unit"))
    monkeypatch.setattr(chain.models, "requirements_for", lambda _: ["network"])
    monkeypatch.setattr(chain.models, "program_requirements_for", lambda _: selected.record["required_programs"])
    monkeypatch.setattr(chain.models, "loader_env_requirements_for", lambda _: {})
    monkeypatch.setattr(chain.execution, "gate_for", lambda *a, **k: None)
    monkeypatch.setattr(chain, "_build_options", lambda *a, **k: {})
    monkeypatch.setattr(private_facts, "selected_input_facts", lambda *a, **k: nullcontext({}))
    monkeypatch.setattr(private_result_format_selection, "from_run", lambda *a, **k: None)
    monkeypatch.setattr(replay, "prepare_instructions", lambda *a, **k: (
        SimpleNamespace(require_facts=lambda _: None, service=object()), {}))
    # Set up the original source/facts binding before issuing the typed selector.
    selected.record.update(fact_reader_binding={}, authored_source_freeze={})
    _write(selected.old, selected.record)
    value = products.RetainedProducts(selected.old, **selected.kwargs)
    seen = []

    class ReachedOrdinarySourceChecks(Exception):
        pass

    def source_checks(*args, **kwargs):
        seen.append(kwargs["retained_products"])
        raise ReachedOrdinarySourceChecks

    monkeypatch.setattr(chain.models, "run", source_checks)
    monkeypatch.setattr(chain.models, "compile_saved_model", lambda *a, **k: pytest.fail("candidate compilation"))
    with pytest.raises(ReachedOrdinarySourceChecks):
        chain._compile(
            run_dir=tmp_path / "run", descriptor=tmp_path / "descriptor", submission=selected.roots["candidate"],
            name="network", out=metadata, owner=None, spec=selected.kwargs["spec"], selected=object(), freeze={},
            retained_products=value,
        )
    assert seen == [value]


@pytest.mark.parametrize("changed", [None, "missing", "foreign_role", "duplicate", "input", "stdout", "timing"])
def test_all_original_boxed_package_calls_match_manifest_and_actual_emitted_bytes(tmp_path, monkeypatch, changed):
    from merlin.targetgen import package_runtime

    root = tmp_path / "lowering"
    root.mkdir()
    (root / "input.interface.mlir").write_text("original source interface")
    (root / "lowered.target.mlir").write_text("candidate target IR")
    artifact = root / "lowered.llvm.mlir"
    artifact.write_text("candidate LLVM IR")
    manifest = {"commands": {name: {"argv": ["python3", "compiler.py", flag, "{input_mlir}"]} for name, flag in (
        ("parse", "--parse"), ("lower_interface_to_target", "--target"), ("emit_analysis_bundle", "--bundle"))}}
    package = SimpleNamespace(directory=tmp_path, manifest=manifest)
    calls = {}
    for number, (role, spec) in enumerate(manifest["commands"].items()):
        path = root / "invocations" / str(number) / "invocation.json"
        path.parent.mkdir(parents=True)
        path.write_text("actual completed package command fixture " + role)
        stdout = artifact if role == "emit_analysis_bundle" else root / "lowered.target.mlir"
        calls[path] = {"kind": "subprocess", "stage": "original_profile_package",
                       "inputs": [products.pin(root / "input.interface.mlir")],
                       "argv": ["bwrap", "original-package", *[
                           word.replace("{input_mlir}", str(tmp_path / "scope/input.mlir")) for word in spec["argv"]]],
                       "started_ns": 2, "finished_ns": 3, "stdout": products.pin(stdout)}
    monkeypatch.setattr(products, "ordinary", lambda path: path)
    monkeypatch.setattr(replay.IR, "verify", lambda path: calls[path])
    monkeypatch.setattr(package_runtime, "_needs_interpreter", lambda *args: False)
    monkeypatch.setattr(package_runtime, "_resolve_argv", lambda p, role, incoming, outgoing: [
        word.replace("{input_mlir}", str(incoming)) for word in p.manifest["commands"][role]["argv"]])
    first, second, last = calls.values()
    if changed == "missing":
        next(iter(calls)).unlink()
    elif changed == "foreign_role":
        first["argv"][4] = "foreign.py"
    elif changed == "duplicate":
        first["argv"] = list(second["argv"])
    elif changed == "input":
        first["inputs"] = []
    elif changed == "stdout":
        last["stdout"] = products.pin(root / "input.interface.mlir")
    elif changed == "timing":
        first["started_ns"] = 0
    if changed:
        with pytest.raises(ValueError, match="incomplete|original boxed|selected manifest|emitted device bytes"):
            products._package_commands(package, root, {"started_ns": 1, "finished_ns": 4}, artifact)
    else:
        assert products._package_commands(package, root, {"started_ns": 1, "finished_ns": 4}, artifact) == [
            products.pin(path) for path in calls]


@pytest.mark.parametrize("actual,accepted", [
    (["tool", "--input=/scope/input", "--again=/scope/input"], True),
    (["tool", "--input=/scope/input", "--again=/different/input"], False),
    (["tool", "--input=relative", "--again=relative"], False),
    (["tool", "--input=/scope/../input", "--again=/scope/../input"], False),
    (["other-tool", "--input=/scope/input", "--again=/scope/input"], False),
])
def test_package_argv_only_substitutes_consistent_absolute_io_paths(actual, accepted):
    assert products._matches_argv(["tool", "--input={in}", "--again={in}"], actual, ("{in}", "{out}")) is accepted


@pytest.mark.parametrize("changed", [None, "unlinked_object", "missing_routed_kernel"])
def test_proven_emission_objects_are_exactly_the_actual_routed_final_link_inputs(selected, monkeypatch, changed):
    from merlin_experiments.phase1.feedback import private_group_provenance

    from merlin.llvmlower import device_build
    from merlin.targetgen import owned_host_model_execution as audit

    receipt = json.loads(Path(selected.record["receipts"]["encode"]["path"]).read_bytes())
    output = receipt["output"]
    build = Path(output["elf"]).parent
    obj = build / "producer.py"  # This harmless recipe's real final-command input.
    artifact = build / "raw.mlir"
    artifact.write_text("original source emission fixture")
    translation = build / "translation.json"
    translation.write_text("completed device translation fixture")
    sidecar = build / "sidecar.json"
    sidecar.write_text("{}")
    if changed == "unlinked_object":
        obj = build / "foreign-object"
        obj.write_bytes(b"same symbol claim, absent from actual link inputs")
    def sized(path):
        return {**products.pin(path), "size_bytes": path.stat().st_size}

    kernel = {"artifacts": {key: sized(path) for key, path in (
        ("object", obj), ("llvm_mlir", artifact), ("llvm_ir", artifact))},
        "custody": {"translation": sized(translation)}, "kernel_symbol": "entry"}
    output.update(device_sidecar=products.pin(sidecar), device_artifacts={"target": "unit", "kernels": {
        "dispatch": kernel}})
    monkeypatch.setattr(private_group_provenance, "routed_kernel_symbols", lambda _: (
        {"dispatch", "missing"} if changed == "missing_routed_kernel" else {"dispatch"}))
    monkeypatch.setattr(device_build, "executable_kernel_entry", lambda _: "entry")
    monkeypatch.setattr(audit, "_verify_device_custody", lambda *a, **k: None)
    monkeypatch.setattr(replay.IR, "verify", lambda _: {"inputs": [products.pin(artifact)]})
    monkeypatch.setattr(products, "_emission_custody", lambda *a, **k: {"actual_original_emission": True})
    if changed:
        with pytest.raises(ValueError, match="final-link input|omitted a routed kernel"):
            products._device_custody(output, target="unit", candidate=selected.roots["candidate"], profile={})
    else:
        assert products._device_custody(
            output, target="unit", candidate=selected.roots["candidate"], profile={}) == [
                {"actual_original_emission": True}]


def test_successful_retained_checks_publish_original_receipts_without_copying_products(selected, monkeypatch, tmp_path):
    from contextlib import nullcontext

    from merlin_experiments.phase1.feedback import private_facts, private_result_format_selection

    from merlin.targetgen import target_experiment

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    selected.record.update(fact_reader_binding={}, authored_source_freeze={})
    _write(selected.old, selected.record)
    value = products.RetainedProducts(selected.old, **selected.kwargs)
    monkeypatch.setattr(chain, "build_dir", lambda: tmp_path / "fresh-products")
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: SimpleNamespace(target="unit"))
    monkeypatch.setattr(chain.models, "requirements_for", lambda _: ["network"])
    monkeypatch.setattr(chain.models, "program_requirements_for", lambda _: selected.record["required_programs"])
    monkeypatch.setattr(chain.models, "loader_env_requirements_for", lambda _: {})
    monkeypatch.setattr(chain.execution, "gate_for", lambda *a, **k: None)
    monkeypatch.setattr(chain, "_build_options", lambda *a, **k: {})
    monkeypatch.setattr(private_facts, "selected_input_facts", lambda *a, **k: nullcontext({}))
    monkeypatch.setattr(private_result_format_selection, "from_run", lambda *a, **k: None)
    selection = SimpleNamespace(require_facts=lambda _: None, record=lambda: {}, path=selected.kwargs["spec"], pins=())
    monkeypatch.setattr(replay, "prepare_instructions", lambda *a, **k: (SimpleNamespace(
        require_facts=lambda _: None, service=object(), record=lambda: {}), {}))

    def ordinary_verification(*args, out, retained_products, **kwargs):
        for name in ("encode", "decode"):
            retained_products.load(name, **selected.inputs)
        report = out / "private_full_model_partition-network.json"
        report.parent.mkdir(parents=True)
        report.write_text("fresh ordinary check result fixture")
        return {"candidate_tree_sha256": selected.record["candidate_tree_sha256"],
                "private_spec_sha256": selected.record["private_spec_sha256"], "authored_source_freeze": {},
                "retained_product_revalidation": retained_products.record()}

    monkeypatch.setattr(chain.models, "run", ordinary_verification)
    monkeypatch.setattr(chain.models, "compile_saved_model", lambda *a, **k: pytest.fail("candidate compilation"))
    monkeypatch.setattr(chain, "_partition_complete", lambda *a: True)
    monkeypatch.setattr(chain.profile, "compiler_profile_binding", lambda *a, **k: selected.kwargs["compiler_profile"])
    monkeypatch.setattr(chain.NativePartition, "_verify_static_record", lambda *a: None)
    issuance = metadata / "instruction-selection/invocations/one/invocation.json"
    issuance.parent.mkdir(parents=True)
    issuance.write_text("trusted instruction issuance fixture")
    for name in ("environment.yaml", "input_bundle_manifest.yaml", "descriptor"):
        (tmp_path / name).write_text("{}")
    result = chain._compile(
        run_dir=tmp_path, descriptor=tmp_path / "descriptor", submission=selected.roots["candidate"],
        name="network", out=metadata, owner=None, spec=selected.kwargs["spec"], selected=selection, freeze={},
        retained_products=value,
    )
    assert json.loads(result.path.read_bytes())["receipts"] == selected.record["receipts"]
    assert not tuple((tmp_path / "fresh-products").rglob("baremetal_model.json"))
