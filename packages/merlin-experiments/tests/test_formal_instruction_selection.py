"""Owned coordinator wiring controls; native issuers/builds are substituted.

No fixture below certifies source roles, ISA, hardware, runtime or a model.
The real declaration, selection custody and formal caller are exercised while
every formal attempt remains incomplete and no native tool is launched.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import formal
from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
from merlin_experiments.phase1.feedback import private_instruction_declaration as D

from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService


def _never_substituted_evaluator(*, elf, evidence_root):
    raise AssertionError("diagnostic fixture cannot execute an ISA evaluator")


@pytest.fixture
def selected(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    public = inputs / "public"
    public.mkdir()

    def pin(name, content="owned selection only\n", *, parent=inputs):
        path = parent / name
        path.write_text(content)
        return {"path": str(path), "sha256": file_digest(path)}

    firrtl = pin("original.fir")
    document = {
        "schema": D.SCHEMA,
        "status": "reviewed",
        "target": "owned_instruction_control",
        "hardware": {
            "descriptor": pin("descriptor.yaml", "target: owned_instruction_control\n"),
            "source_bundle": pin("source-bundle.json", json.dumps({"config": "OwnedConfig"})),
        },
        "command": {
            "checkout": str(public),
            "commit": "a" * 40,
            "isa_source": pin("commands.scala", parent=public),
            "function_span": ["owned_begin", "owned_end"],
            "circt_opt": pin("circt-tool"),
        },
        "predicates": [
            {"source": pin("predicate.scala", parent=public), "binding": "owned_route", "operand": "owned_selector"}
        ],
        "accessor": {
            "checkout": str(public),
            "commit": "b" * 40,
            "include_root": str(public),
            "native_compiler": pin("native-compiler"),
            "reviewed_spec": pin("native-spec.json"),
        },
        "policy": pin("policy.json"),
        "decoder": {
            "match_constant": "OWNED_MATCH",
            "mask_constant": "OWNED_MASK",
            "selector_member": "owned_member",
            "expected_elf_machine": 913,
        },
        "forbidden_roots": [str(tmp_path / "excluded")],
    }
    path = inputs / "instruction-selection.json"
    path.write_text(json.dumps(document))
    return SimpleNamespace(
        document=document, path=path, firrtl=firrtl, target=document["target"], output=tmp_path / "prepared"
    )


def _substitute_issuers(selected, monkeypatch, *, fault=None):
    """Substituted public/native issuers confer no live production authority."""
    calls = []
    hardware = SimpleNamespace(
        target=selected.target,
        sha256="1" * 64,
        verify=lambda: None,
        public_facts=lambda: {"source_sha256": {"firrtl": selected.firrtl["sha256"]}},
    )
    commands = SimpleNamespace(hardware=hardware)
    predicates = SimpleNamespace(command_intake=commands)
    accessor = object()
    policy = SimpleNamespace(command_intake=commands)

    def issue(name, value):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            if fault == name + "_refuses":
                raise ValueError("owned substituted issuer refusal")
            if fault == name + "_mutates":
                Path(selected.document["policy"]["path"]).write_text("changed original selected input")
            return value

        return invoke

    for name, function, value in (
        ("hardware", "issue_independent_hardware_intake", hardware),
        ("command", "issue_independent_command_intake", commands),
        ("predicates", "issue_independent_source_predicate_intake", predicates),
        ("accessor", "issue_independent_accessor_intake", accessor),
        ("policy", "issue_independent_instruction_policy", policy),
    ):
        monkeypatch.setattr(C, function, issue(name, value))
    monkeypatch.setattr(C, "load_selection", lambda path, **kwargs: json.loads(path.read_text()))

    class DiagnosticCheck:
        def __init__(self, original_policy, original_accessor, decoder):
            assert original_policy is policy and original_accessor is accessor
            self.policy, self.selection, self.changed = policy, decoder, False
            self.service = None

        def verify(self):
            return ("8" if self.changed else "7") * 64

        def evaluate(self, *, elf, evidence_root):
            raise AssertionError("diagnostic cannot issue instruction admission")

        def admission_service(self):
            source = Path(__file__).resolve()
            self.service = LinkedElfAdmissionService(
                selected.target, self.evaluate, ((str(source), file_digest(source)),)
            )
            return self.service

    monkeypatch.setattr(C, "IndependentLinkedInstructionCheck", DiagnosticCheck)
    return calls, hardware


def _binding(selected):
    row = {"target": selected.target, "config": "OwnedConfig", "firrtl_sha256": selected.firrtl["sha256"]}
    return {"public_effective": dict(row), "private_input": dict(row)}


def test_fixed_chain_uses_exact_original_sources_and_live_service(selected, monkeypatch):
    calls, hardware = _substitute_issuers(selected, monkeypatch)
    original = D.read(selected.path, target=selected.target)
    prepared = C.prepare(original, target=selected.target, output=selected.output)
    assert [name for name, _kwargs in calls] == ["hardware", "command", "predicates", "accessor", "policy"]
    assert calls[1][1]["hardware"] is hardware
    assert calls[2][1]["command_intake"] is calls[4][1]["command_intake"]
    assert calls[1][1]["function_span"] == tuple(selected.document["command"]["function_span"])
    assert calls[3][1]["reviewed_spec"] == Path(selected.document["accessor"]["reviewed_spec"]["path"])
    assert prepared.check.selection.record() == selected.document["decoder"]
    assert prepared.service is prepared.check.service
    prepared.require_facts(_binding(selected))
    assert prepared.record()["scope"].endswith("no body, event, runtime or physical grant")


@pytest.mark.parametrize(
    "fault",
    [
        "hardware_refuses",
        "command_refuses",
        "predicates_refuses",
        "accessor_refuses",
        "policy_refuses",
        "command_mutates",
        "policy_mutates",
    ],
)
def test_fresh_issuer_refusal_or_input_change_never_produces_selection(selected, monkeypatch, fault):
    _substitute_issuers(selected, monkeypatch, fault=fault)
    with pytest.raises(ValueError):
        C.prepare(D.read(selected.path, target=selected.target), target=selected.target, output=selected.output)
    assert not (selected.output / "selection.json").exists()


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("public_effective", "firrtl_sha256", "f" * 64),
        ("private_input", "config", "OtherConfig"),
        ("private_input", "target", "other_owned_control"),
    ],
)
def test_original_firrtl_and_configuration_cannot_be_substituted(selected, monkeypatch, role, field, value):
    _substitute_issuers(selected, monkeypatch)
    prepared = C.prepare(D.read(selected.path, target=selected.target), target=selected.target, output=selected.output)
    binding = _binding(selected)
    binding[role][field] = value
    with pytest.raises(ValueError, match="original selected FIRRTL/configuration"):
        prepared.require_facts(binding)


@pytest.mark.parametrize("change", ["callback", "check", "request", "policy"])
def test_prepared_selection_reopens_original_callback_and_source_inputs(selected, monkeypatch, change):
    _substitute_issuers(selected, monkeypatch)
    prepared = C.prepare(D.read(selected.path, target=selected.target), target=selected.target, output=selected.output)
    if change == "callback":
        object.__setattr__(prepared.service, "evaluator", _never_substituted_evaluator)
    elif change == "check":
        prepared.check.changed = True
    else:
        path = selected.path if change == "request" else Path(selected.document["policy"]["path"])
        path.write_text("changed selection")
    with pytest.raises(ValueError):
        prepared.record()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unreviewed",
        "saved_authority",
        "boolean_machine",
        "duplicate_predicate",
        "digest",
        "missing_file",
        "excluded",
        "symlink",
        "other_target",
        "escaped_command",
        "escaped_predicate",
    ],
)
def test_malformed_or_unreviewed_declaration_refuses_before_issuance(selected, monkeypatch, change):
    calls, _hardware = _substitute_issuers(selected, monkeypatch)
    document = copy.deepcopy(selected.document)
    if change == "missing":
        del document["decoder"]
    elif change == "unreviewed":
        document["status"] = "pending"
    elif change == "saved_authority":
        document["intake_receipt"] = {"status": "passed"}
    elif change == "boolean_machine":
        document["decoder"]["expected_elf_machine"] = True
    elif change == "duplicate_predicate":
        document["predicates"].append(copy.deepcopy(document["predicates"][0]))
    elif change == "digest":
        document["policy"]["sha256"] = "f" * 64
    elif change == "missing_file":
        Path(document["policy"]["path"]).unlink()
    elif change == "excluded":
        document["forbidden_roots"] = [str(selected.path.parent)]
    elif change == "other_target":
        document["target"] = "other_owned_control"
    elif change == "escaped_command":
        document["command"]["isa_source"] = document["policy"]
    elif change == "escaped_predicate":
        document["predicates"][0]["source"] = document["policy"]
    elif change == "symlink":
        original = Path(document["policy"]["path"])
        link = original.with_name("policy-link.json")
        link.symlink_to(original)
        document["policy"]["path"] = str(link)
    selected.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unchanged reviewed public source selections"):
        D.read(selected.path, target=selected.target)
    assert not calls and not selected.output.exists()


def test_declaration_byte_limit_precedes_json_parse(selected, monkeypatch):
    monkeypatch.setattr(D, "MAX_DECLARATION_BYTES", 32)
    monkeypatch.setattr(D, "loads", lambda *args, **kwargs: pytest.fail("oversized declaration reached JSON parsing"))
    with pytest.raises(ValueError, match="unchanged reviewed"):
        D.read(selected.path, target=selected.target)


def test_duplicate_raw_fields_are_not_a_reviewed_declaration(selected):
    raw = selected.path.read_text()
    selected.path.write_text('{"status":"pending",' + raw[1:])
    with pytest.raises(ValueError, match="unchanged reviewed"):
        D.read(selected.path, target=selected.target)


@pytest.mark.parametrize("change", ["unreviewed", "missing_pins", "missing_roots"])
def test_direct_declaration_construction_cannot_bypass_closed_review_or_membership(selected, monkeypatch, change):
    calls, _hardware = _substitute_issuers(selected, monkeypatch)
    original = D.read(selected.path, target=selected.target)
    pins, roots = original.pins, original.roots
    if change == "unreviewed":
        document = original.document()
        document["status"] = "pending"
        selected.path.write_text(json.dumps(document))
    elif change == "missing_pins":
        pins = ()
    else:
        roots = ()
    constructed = D.Declaration(selected.path, selected.path.read_bytes(), pins, roots)
    with pytest.raises(ValueError):
        C.prepare(constructed, target=selected.target, output=selected.output)
    assert not calls and not selected.output.exists()


def test_output_cannot_overwrite_or_contain_original_selections(selected, monkeypatch):
    calls, _hardware = _substitute_issuers(selected, monkeypatch)
    declaration = D.read(selected.path, target=selected.target)
    with pytest.raises(ValueError, match="disjoint output owner"):
        C.prepare(declaration, target=selected.target, output=selected.path.parent)
    assert not calls


def test_coordinator_readers_remain_in_registered_private_grader_identity():
    from merlin.common.access import declared_modules, module_matches

    for module in (D, C):
        assert any(module_matches(module.__name__, prefix) for prefix in declared_modules("grader"))
        assert not any(module_matches(module.__name__, prefix) for prefix in declared_modules("agent"))


def _formal_attempt(selected, monkeypatch, *, choose=True, fault=None, build_passes=False):
    calls, _hardware = _substitute_issuers(selected, monkeypatch)
    run = selected.output.with_name("formal-run")
    (run / "submission").mkdir(parents=True)
    (run / "submission/manifest.yaml").write_text("{}")
    spec = run.parent / "owned-private-spec.yaml"
    spec.write_text("{}")
    context = SimpleNamespace(
        target=selected.target,
        descriptor=Path(selected.document["hardware"]["descriptor"]["path"]),
        repo=run.parent,
        readback_policy=None,
    )
    monkeypatch.setattr(
        formal, "_install_formal_model_simulator", lambda target: ({"engine": "diagnostic_not_executed"}, None)
    )
    monkeypatch.setattr(formal, "_score", lambda *args, **kwargs: {"per_capsule": [], "gradeable": False})
    monkeypatch.setattr(formal.freeze_run, "repo_sha", lambda **kwargs: "owned_diagnostic_source")
    monkeypatch.setattr(
        formal, "_private_source_freeze_for_formal", lambda *args, **kwargs: {"diagnostic_not_qualified": True}
    )
    monkeypatch.setattr(formal.PFM, "requirements_for", lambda descriptor: ("first", "second"))
    monkeypatch.setattr(
        formal.PFM, "program_requirements_for", lambda descriptor: {"first": ("model",), "second": ("model",)}
    )
    monkeypatch.setattr(formal.PFM, "loader_env_requirements_for", lambda descriptor: {})
    binding = _binding(selected)
    if fault == "facts":
        binding["public_effective"]["firrtl_sha256"] = "f" * 64

    @contextmanager
    def selected_facts(*args, **kwargs):
        yield binding

    monkeypatch.setattr(formal, "selected_input_facts", selected_facts)
    seen = []
    original_prepare = C.prepare
    prepared = []

    def retain(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(C, "prepare", retain)

    def build(*args, **kwargs):
        seen.append(kwargs)
        if fault == "callback":
            object.__setattr__(kwargs["linked_elf_admission"], "evaluator", _never_substituted_evaluator)
        return {"passed": build_passes, "models": [], "reason": "owned diagnostic: no actual build or qualification"}

    monkeypatch.setattr(formal.PFM, "run", build)
    arguments = [
        "--run-dir",
        str(run),
        "--arm",
        "owned_control",
        "--capsules",
        str(run.parent / "public"),
        "--skip-hidden",
        "--no-oracle",
        "--private-full-model-spec",
        str(spec),
    ]
    if choose:
        arguments += ["--instruction-selection", str(selected.path)]
    assert formal.main(arguments, context=context) == 1
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    assert manifest["completion"]["formal_grade_complete"] is False
    assert manifest["private_full_models"]["passed"] is build_passes
    return manifest, calls, seen, prepared


def test_actual_formal_caller_forwards_the_fresh_service_in_process(selected, monkeypatch):
    manifest, calls, seen, prepared = _formal_attempt(selected, monkeypatch)
    assert len(calls) == 5 and len(seen) == len(prepared) == 1
    assert seen[0]["linked_elf_admission"] is prepared[0].service
    assert manifest["private_full_models"]["instruction_selection"]["schema"] == D.SCHEMA


def test_instruction_selection_preserves_required_numerical_execution_gate(selected, monkeypatch):
    gate = {"required": True}
    options = {"first": {"model": {"readback": "full"}}}
    executed = []
    monkeypatch.setattr(formal.PFX, "gate_for", lambda *args, **kwargs: gate)
    monkeypatch.setattr(formal.PFX, "build_options", lambda selected_gate: options)
    monkeypatch.setattr(formal.PFX, "not_run", lambda *args: {"passed": False})

    def execute(models, selected_gate, **kwargs):
        executed.append((models, selected_gate))
        return {"passed": False, "reason": "diagnostic numerical execution refusal"}

    monkeypatch.setattr(formal.PFX, "run", execute)
    manifest, _calls, seen, prepared = _formal_attempt(selected, monkeypatch, build_passes=True)
    assert seen[0]["linked_elf_admission"] is prepared[0].service
    assert seen[0]["build_options"] is options
    assert len(executed) == 1 and executed[0][1] is gate
    assert "private_full_model_execution:incomplete" in manifest["completion"]["failures"]


@pytest.mark.parametrize("fault", ["facts", "callback"])
def test_actual_formal_caller_refuses_facts_or_postbuild_selection_drift(selected, monkeypatch, fault):
    manifest, _calls, seen, _prepared = _formal_attempt(selected, monkeypatch, fault=fault)
    assert len(seen) == (0 if fault == "facts" else 1)
    assert "instruction_selection" not in manifest["private_full_models"]
    assert (
        "coordinator" in manifest["private_full_models"]["reason"]
        or "linked ELF admission" in manifest["private_full_models"]["reason"]
    )


def test_absent_selection_keeps_legacy_formal_call_shape_unqualified(selected, monkeypatch):
    _manifest, calls, seen, prepared = _formal_attempt(selected, monkeypatch, choose=False)
    assert not calls and not prepared and len(seen) == 1
    assert "linked_elf_admission" not in seen[0]


def test_run_freeze_keeps_instruction_selection_private_and_reopens_dependencies(selected, tmp_path):
    run = tmp_path / "run"
    frozen, record = D.freeze_for_run(selected.path, target=selected.target, run_dir=run)
    assert frozen.read_bytes() == selected.path.read_bytes()
    assert frozen.stat().st_mode & 0o077 == 0
    assert record["sha256"] == file_digest(frozen)
    assert D.freeze_for_run(selected.path, target=selected.target, run_dir=run, prior=record, resuming=True) == (
        frozen,
        record,
    )
    with pytest.raises(ValueError, match="removed"):
        D.freeze_for_run("", target=selected.target, run_dir=run, prior=record, resuming=True)
    Path(selected.document["policy"]["path"]).write_text("changed independently selected policy")
    with pytest.raises(ValueError):
        D.freeze_for_run(selected.path, target=selected.target, run_dir=run, prior=record, resuming=True)


def test_run_freeze_refuses_changed_frozen_instruction_declaration(selected, tmp_path):
    run = tmp_path / "run"
    frozen, record = D.freeze_for_run(selected.path, target=selected.target, run_dir=run)
    frozen.chmod(0o600)
    frozen.write_text("changed frozen selection")
    with pytest.raises(ValueError, match="changed"):
        D.freeze_for_run(selected.path, target=selected.target, run_dir=run, prior=record, resuming=True)
