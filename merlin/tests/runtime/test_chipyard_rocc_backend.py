"""Neutral process/data controls, never simulator or hardware qualification."""

import base64
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime.backends import chipyard_rocc as B
from merlin.runtime.backends import rocc_selection as S
from merlin.targetgen import gsim_emulator as GE
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


def _pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _unit(identifier, roster):
    return {"id": identifier, "sha256": _pin(B._resource(identifier, roster))["sha256"]}


def _file(tmp, name, text):
    path = tmp / name
    path.write_text(text)
    return path


def _console():
    return "\n".join(
        (
            "OUT_B64_BEGIN v1 Y 1 2 1 s",
            "OUT_B64_CHUNK 00000000 0002 " + base64.b64encode(bytes([127, 128])).decode(),
            "OUT_B64_END",
            "METRIC raw_sample 3",
            "DONE",
            "",
        )
    )


@pytest.fixture
def selection(tmp_path):
    binary = _file(tmp_path, "owned-process", "#!/bin/sh\nprintf '%s\\n' '" + _console().rstrip() + "'\n")
    binary.chmod(0o700)
    compiler = _file(tmp_path, "selected-compiler", "#!/bin/sh\nexit 19\n")
    compiler.chmod(0o700)
    library = _file(tmp_path, "selected-model.so", "owned diagnostic bytes only\n")
    script = _file(tmp_path, "layout.ld", "ENTRY(_start)\nSECTIONS { .text : { *(.text*) } }\n")
    engine = {
        "binary": _pin(binary),
        "argv": ["--isa=rv64gc", "{elf}"],
        "environment": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        "cwd": str(tmp_path),
        "max_timeout_s": 5,
        "max_console_bytes": 4096,
        "failure_markers": ["DECLARED_FAILURE"],
    }
    contract = {
        "name": "synthetic",
        "runner": {
            "backend": "chipyard_rocc",
            "chipyard_rocc": {
                "version": 1,
                "config": "selected_config",
                "toolchain": {
                    "compiler": _pin(compiler),
                    "runtime_units": [_unit(x, B._RUNTIME) for x in ("startup", "htif_console")],
                    "headers": [_unit(x, B._HEADERS) for x in ("htif", "out_b64")],
                    "link_script": _pin(script),
                    "load_address": 4096,
                    "cflags": ["-march=rv64gc", "-mabi=lp64d", "-ffreestanding"],
                    "ldflags": [],
                    "kernel_stack_frame": {"entry_symbol": "entry", "max_static_bytes": 8192},
                },
                "engines": {"spike": engine},
            },
            "spike_extension": {
                "extension_name": "selected_extension",
                "extlib": str(library),
                "sha256": _pin(library)["sha256"],
            },
        },
        "harness_abi": {
            "version": 2,
            "kind": "logical_pointer",
            "entry_symbol": "entry",
            "fence_symbol": None,
            "tensor_alignment": 8,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": FULL_VALUES_B64,
        },
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": [{"funct": 1, "class": "TRANSPORT", "operands": {}}],
        },
        "rtl_checks": ["decode_clean", "legal_funct"],
    }
    facts = {
        "inputs": {"target": "synthetic", "fir_sha256": "f" * 64},
        "source_consistency": {"status": "unverified", "config": "selected_config"},
        "facts": {
            "source": {"config": "selected_config"},
            "interfaces": [
                {
                    "name": "funct_decode_table",
                    "legal_funct": [1],
                    "custom_opcode": 11,
                    "funct3": 7,
                    "scope": "complete_rocc_funct7",
                    "complete_isa": True,
                }
            ],
        },
    }
    return contract, facts


def _bound(selection):
    return B.bind(target="synthetic", contract=selection[0], facts=selection[1])


def _engine(selection):
    return selection[0]["runner"]["chipyard_rocc"]["engines"]["spike"]


def _program():
    return {
        "tensors": {
            "X": {"shape": [1, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [1, 2], "dtype": "i8", "role": "output"},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
    }


def _deployment(selection):
    contract = copy.deepcopy(selection[0])
    block = contract["runner"]["chipyard_rocc"]
    for name in ("compiler", "link_script"):
        block["toolchain"][name].pop("sha256")
    for name in ("runtime_units", "headers"):
        for row in block["toolchain"][name]:
            row.pop("sha256")
    for engine in block["engines"].values():
        engine["binary"].pop("sha256")
        if "receipt" in engine:
            engine["receipt"].pop("sha256")
    if "spike" in block["engines"]:
        contract["runner"]["spike_extension"].pop("sha256")
    return contract


def test_prepare_deployment_preserves_semantics_and_reaches_ordinary_selection(selection, monkeypatch):
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.targetgen.target_registry import observed_contract

    declaration = _deployment(selection)
    before = copy.deepcopy(declaration)
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    prepared = S.prepare_contract(target="synthetic", contract=declaration, facts=selection[1])
    assert prepared == selection[0]
    assert declaration == before
    with observed_contract("synthetic", prepared), observed_facts("synthetic", selection[1]):
        bound = get_backend("synthetic")
        assert (
            bound.verify_execution_inputs()["spike"]["extension"]["path"]
            == prepared["runner"]["spike_extension"]["extlib"]
        )
        assert "entry(tensor_0, tensor_1);" in bound.render_harness(
            _program(), target="synthetic", inputs={"X": [[1, 2]]}
        )
        assert bound.rocc_semantics.isa_constants("synthetic")["CUSTOM_OPCODE"] == 11


def test_prepare_deployment_reopens_strict_receipt_without_native_work(selection, tmp_path, monkeypatch):
    _gsim(selection, tmp_path)
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    prepared = S.prepare_contract(target="synthetic", contract=_deployment(selection), facts=selection[1])
    assert prepared == selection[0]
    assert (
        B.bind(target="synthetic", contract=prepared, facts=selection[1]).verify_execution_inputs()["gsim"]["receipt"][
            "schema_version"
        ]
        == GE.STRICT_RECEIPT_SCHEMA
    )


@pytest.mark.parametrize("member", ["compiler", "link_script", "runtime_units", "headers", "binary", "extension"])
def test_prepare_deployment_never_restamps_existing_pins(selection, member, monkeypatch):
    declaration = _deployment(selection)
    block = declaration["runner"]["chipyard_rocc"]
    if member in ("compiler", "link_script"):
        row = block["toolchain"][member]
    elif member in ("runtime_units", "headers"):
        row = block["toolchain"][member][0]
    elif member == "binary":
        row = block["engines"]["spike"]["binary"]
    else:
        row = declaration["runner"]["spike_extension"]
    row["sha256"] = "0" * 64
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    with pytest.raises(ValueError, match="changed"):
        S.prepare_contract(target="synthetic", contract=declaration, facts=selection[1])


@pytest.mark.parametrize(
    "mutation", ["plugin", "legacy_abi", "missing_layout", "vendor_unit", "foreign_config", "runtime_config"]
)
def test_prepare_deployment_cannot_supply_or_invent_compiler_inputs(selection, mutation, monkeypatch):
    declaration = _deployment(selection)
    facts = copy.deepcopy(selection[1])
    if mutation == "plugin":
        declaration["plugin"] = {"backend": "private.py"}
    elif mutation == "legacy_abi":
        declaration["harness_abi"] = {"entry_symbol": "entry", "includes": ["vendor.h"]}
    elif mutation == "missing_layout":
        declaration["rocc_operand_roles"]["instructions"][0]["operands"] = {"rs2": {"bundle": "Absent"}}
    elif mutation == "vendor_unit":
        declaration["runner"]["chipyard_rocc"]["toolchain"]["runtime_units"].append({"id": "device_kernel"})
    elif mutation == "runtime_config":
        declaration["runtime"] = {"rtl_sim_config": "other_config"}
    else:
        facts["facts"]["source"]["config"] = "other_config"
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    with pytest.raises(ValueError):
        S.prepare_contract(target="synthetic", contract=declaration, facts=facts)


def test_prepare_cli_writes_only_a_fresh_operator_contract(selection, tmp_path, monkeypatch, capsys):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(S, "artifacts_dir", lambda: root)
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    contract = _file(tmp_path, "deployment.json", json.dumps(_deployment(selection)))
    facts = _file(tmp_path, "facts.json", json.dumps(selection[1]))
    output = root / "handoff/contract.json"
    argv = ["--target", "synthetic", "--contract", str(contract), "--facts", str(facts), "--output", str(output)]
    assert S.main(argv) == 0
    assert json.loads(output.read_text()) == selection[0]
    report = json.loads(capsys.readouterr().out)
    assert report["native_executed"] is False
    assert report["sha256"] == _pin(output)["sha256"]
    with pytest.raises(SystemExit):
        S.main(argv)
    assert json.loads(output.read_text()) == selection[0]


def test_prepare_cli_refuses_source_drift_before_publishing(selection, tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(S, "artifacts_dir", lambda: root)
    contract = _file(tmp_path, "deployment.json", json.dumps(_deployment(selection)))
    facts = _file(tmp_path, "facts.json", json.dumps(selection[1]))
    output = root / "handoff/contract.json"
    original = S.prepare_contract

    def changed(**kwargs):
        prepared = original(**kwargs)
        facts.write_text("{}")
        return prepared

    monkeypatch.setattr(S, "prepare_contract", changed)
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    with pytest.raises(SystemExit):
        S.main(["--target", "synthetic", "--contract", str(contract), "--facts", str(facts), "--output", str(output)])
    assert not output.exists()


def test_bound_metadata_has_no_binary_dependency_or_default_provider(selection, monkeypatch):
    bound = _bound(selection)
    Path(_engine(selection)["binary"]["path"]).unlink()
    selection[0]["runner"]["chipyard_rocc"]["engines"]["spike"]["argv"].clear()
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    assert bound.readout_scalar_abi() is None
    assert bound.readout_epilogue_capability() is None
    assert bound.readout_operand_sum() is None
    assert bound.epilogue_stage_routes() == []
    assert bound.ORACLE["spike"]["derived_from_rtl"] is False
    assert bound.available("spike") is False
    assert bound.available("verilator") is False
    assert "warm_single_counter_region_cycles" not in bound.EXECUTION_CAPABILITIES


def test_metadata_and_renderer_preserve_original_data_without_tools(selection, monkeypatch):
    before = copy.deepcopy(selection)
    bound = _bound(selection)
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    source = bound.render_harness(_program(), target="synthetic", inputs={"X": [[127, -128]]})
    assert "entry(tensor_0, tensor_1);" in source
    assert "static unsigned char tensor_0[2]" in source
    assert "OUT_B64_BEGIN v1 Y 1 2" in source
    assert selection == before


def test_binding_and_data_queries_do_not_open_execution_members(selection, monkeypatch):
    monkeypatch.setattr(B, "_check_pin", lambda *a, **k: pytest.fail("execution member reached"))
    bound = _bound(selection)
    assert bound.readback_policy().transport == FULL_VALUES_B64
    assert bound.readout_scalar_abi() is None
    assert bound.ORACLE["spike"]["kind"] == "functional"


def test_explicit_recipe_uses_only_selected_core_units(selection):
    recipe = _bound(selection).harness_build_recipe()
    assert [path.name for path in recipe.support_sources] == ["crt.S", "htif.c"]
    assert recipe.header_dependencies == tuple(B._resource(x, B._HEADERS) for x in ("htif", "out_b64"))
    assert recipe.march() == "-march=rv64gc"
    assert recipe.mabi() == "-mabi=lp64d"
    assert recipe.load_address == 4096
    assert recipe.require_kernel_stack_frame().entry_symbol == "entry"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "bool_version",
        "facts_target",
        "config",
        "engine",
        "legacy_abi",
        "vendor_unit",
        "vendor_header",
        "include_flag",
        "object_flag",
        "plugin_flag",
        "library",
        "no_isa",
        "no_abi",
        "double_elf",
        "other_placeholder",
        "no_extension",
        "extension_env",
        "extension_argv",
        "bool_budget",
        "no_codec",
        "legacy_facts",
        "missing_config",
        "ambiguous_config",
        "unresolved_firrtl",
    ],
)
def test_incomplete_or_compiler_bearing_selections_refuse_before_any_process(selection, monkeypatch, mutation):
    contract, facts = selection
    block = contract["runner"]["chipyard_rocc"]
    toolchain = block["toolchain"]
    if mutation == "missing":
        contract["runner"].pop("chipyard_rocc")
    elif mutation == "bool_version":
        block["version"] = True
    elif mutation == "facts_target":
        facts["inputs"]["target"] = "other"
    elif mutation == "config":
        facts["facts"]["source"]["config"] = "other"
    elif mutation == "legacy_facts":
        facts["source"] = facts["facts"].pop("source")
    elif mutation == "missing_config":
        facts["source_consistency"].pop("config")
    elif mutation == "ambiguous_config":
        facts["source"] = {"config": "other"}
    elif mutation == "unresolved_firrtl":
        facts["inputs"]["fir_sha256"] = "unresolved"
    elif mutation == "engine":
        block["engines"]["verilator"] = block["engines"].pop("spike")
    elif mutation == "legacy_abi":
        contract["harness_abi"]["version"] = 1
    elif mutation == "vendor_unit":
        toolchain["runtime_units"].append({"id": "compute", "sha256": "1" * 64})
    elif mutation == "vendor_header":
        toolchain["headers"].append({"id": "vendor_primitives", "sha256": "1" * 64})
    elif mutation in {"include_flag", "object_flag", "plugin_flag"}:
        toolchain["cflags"].append(
            {
                "include_flag": "-include=compute.h",
                "object_flag": "/foreign/code.o",
                "plugin_flag": "-fplugin=compute.so",
            }[mutation]
        )
    elif mutation == "library":
        toolchain["ldflags"].append("-lcompute")
    elif mutation == "no_isa":
        toolchain["cflags"].remove("-march=rv64gc")
    elif mutation == "no_abi":
        toolchain["cflags"].remove("-mabi=lp64d")
    elif mutation == "double_elf":
        _engine(selection)["argv"].append("{elf}")
    elif mutation == "other_placeholder":
        _engine(selection)["argv"].append("{configuration}")
    elif mutation == "no_extension":
        contract["runner"].pop("spike_extension")
    elif mutation == "extension_env":
        contract["runner"]["spike_extension"]["extlib_env"] = "MODEL_PATH"
    elif mutation == "extension_argv":
        _engine(selection)["argv"].append("--extension=other")
    elif mutation == "no_codec":
        toolchain["headers"].pop()
    else:
        _engine(selection)["max_console_bytes"] = True
    monkeypatch.setattr(B.subprocess, "Popen", lambda *a, **k: pytest.fail("process reached"))
    with pytest.raises(ValueError):
        _bound(selection)


@pytest.mark.parametrize(
    "directive", ["INPUT(code.o)", "GROUP(-lcompute)", "INCLUDE foreign.ld", "SEARCH_DIR(/foreign)", "STARTUP(code.o)"]
)
def test_linker_layout_cannot_inject_hidden_code(selection, directive):
    row = selection[0]["runner"]["chipyard_rocc"]["toolchain"]["link_script"]
    path = Path(row["path"])
    path.write_text(directive)
    row.update(_pin(path))
    with pytest.raises(ValueError, match="external compilation inputs"):
        _bound(selection).harness_build_recipe()


def test_harmless_selected_process_has_actual_elf_argv_environment_and_full_output(selection, tmp_path):
    bound = _bound(selection)
    elf = _file(tmp_path, "requested.elf", "owned input artifact\n")
    console = bound.run_elf(elf, simulator="spike", timeout=2)
    outputs, raw = bound.parse_output(console)
    assert outputs == {"Y": [[127, -128]]}
    assert raw == {"raw_sample": 3}
    records = list(tmp_path.glob("invocations/*/invocation.json"))
    assert len(records) == 1
    actual = I.verify(records[0])
    I.require_environment(records[0], environment=_engine(selection)["environment"])
    assert actual["argv"][-1] == str(elf)
    assert actual["inputs"] == [_pin(elf)]
    assert actual["environment"] == I.environment_identity(_engine(selection)["environment"])
    assert "values" not in actual["environment"]
    assert records[0].with_name("stdout.bin").read_text() == console
    assert records[0].with_name("stderr.bin").read_bytes() == b""


@pytest.mark.parametrize("change", ["binary", "extension", "elf"])
def test_changed_selected_bytes_cannot_complete(selection, tmp_path, change):
    engine = _engine(selection)
    elf = _file(tmp_path, "requested.elf", "original\n")
    if change == "elf":
        binary = Path(engine["binary"]["path"])
        binary.write_text("#!/bin/sh\nfor last do :; done\nprintf changed > \"$last\"\nprintf 'DONE\\n'\n")
        engine["binary"] = _pin(binary)
        bound = _bound(selection)
    else:
        bound = _bound(selection)
        path = Path(
            engine["binary"]["path"] if change == "binary" else selection[0]["runner"]["spike_extension"]["extlib"]
        )
        path.write_text("changed\n")
    with pytest.raises(ValueError):
        bound.run_elf(elf, simulator="spike", timeout=2)
    if change != "elf":
        assert not list(tmp_path.glob("invocations/*/invocation.json"))


@pytest.mark.parametrize(
    "option", [{"memory_readback": {}}, {"reference": "unselected"}, {"timeout": True}, {"timeout": 601}]
)
def test_unsupported_invocation_options_refuse_without_execution(selection, tmp_path, option):
    bound = _bound(selection)
    options = {"simulator": "spike", "timeout": 2, **option}
    with pytest.raises(ValueError):
        bound.run_elf(_file(tmp_path, "requested.elf", "owned\n"), **options)
    assert not list(tmp_path.glob("invocations/*/invocation.json"))


@pytest.mark.parametrize(
    "body,exception",
    [
        ("printf 'DECLARED_FAILURE\\nDONE\\n'", ValueError),
        ("printf 'partial\\n'; exit 3", subprocess.CalledProcessError),
        ("while :; do printf 'xxxxxxxxxxxxxxxx'; done", ValueError),
        ("while :; do :; done", subprocess.TimeoutExpired),
    ],
)
def test_failure_deadline_and_capture_limit_retain_actual_partial_stream(selection, tmp_path, body, exception):
    engine = _engine(selection)
    binary = Path(engine["binary"]["path"])
    binary.write_text("#!/bin/sh\n" + body + "\n")
    engine["binary"] = _pin(binary)
    engine["max_console_bytes"] = 256
    with pytest.raises(exception):
        _bound(selection).run_elf(_file(tmp_path, "requested.elf", "owned\n"), simulator="spike", timeout=0.2)
    record = next(tmp_path.glob("invocations/*/invocation.json"))
    document = json.loads(record.read_text())
    assert document["status"] in {"completed", "failed"}
    assert record.with_name("stdout.bin").stat().st_size <= 257


@pytest.mark.parametrize(
    "console", ["", "DONE\n", "OUTSUM Y ignored\nDONE\n", "OUT_B64_BEGIN v1 Y 1 2 1 s\nDONE\n", "OUT Y 1 2 7\nDONE\n"]
)
def test_partial_sampled_or_missing_console_refuses(selection, console):
    with pytest.raises(ValueError):
        _bound(selection).parse_output(console)


@pytest.mark.parametrize(
    "console",
    [
        "OUT Y 1 1 3\nOUT Y 1 1 4\nDONE\n",
        "OUT Y 1 1 3\nDONE\nDONE\n",
        "OUT Y 1 1 3\nMETRIC sample 1\nMETRIC sample 2\nDONE\n",
        "OUT Y 1 1 nan\nDONE\n",
        "OUT Y 1 1 inf\nDONE\n",
        "OUT Y 1 1 3\nMETRIC sample nan\nDONE\n",
        "OUT Y 1 1 3\nDONE malformed\n",
    ],
)
def test_duplicate_or_nonfinite_plain_framing_is_not_silently_accepted(selection, console):
    with pytest.raises(ValueError):
        _bound(selection).parse_output(console)


def test_omitted_logical_inputs_and_warm_profile_are_explicit_refusals(selection):
    bound = _bound(selection)
    with pytest.raises(ValueError):
        bound.render_harness(_program(), target="synthetic")
    with pytest.raises(NotImplementedError):
        bound.render_harness(_program(), target="synthetic", inputs={"X": [[1, 2]]}, warm_profile={})


def test_coherent_execution_requires_a_separate_selected_reader(selection):
    selection[0]["harness_abi"]["readback_transport"] = COHERENT_DUMP_V1
    with pytest.raises(ValueError, match="coherent readback transport"):
        _bound(selection)
    selection[0]["runner"]["chipyard_rocc"]["toolchain"]["headers"] = [
        _unit(x, B._HEADERS) for x in ("htif", "out_bin", "out_bin_memory")
    ]
    with pytest.raises(ValueError, match="coherent readback transport"):
        _bound(selection)


def test_readback_policy_is_typed_and_derived_only_from_frozen_selection(selection):
    bound = _bound(selection)
    selection[0]["harness_abi"]["readback_transport"] = COHERENT_DUMP_V1
    policy = bound.readback_policy()
    assert type(policy) is ReadbackPolicy
    assert policy.transport == FULL_VALUES_B64
    assert policy.record() == bound.readback_policy().record()


def _gsim(selection, tmp_path):
    contract, facts = selection
    engines = contract["runner"]["chipyard_rocc"]["engines"]
    engine = engines.pop("spike")
    engine["argv"] = ["{elf}"]
    engines["gsim"] = engine
    pins = {
        name: _pin(_file(tmp_path, name, "owned " + name))
        for name in ("firrtl", "model_manifest", "gsim_emitter", "cxx_wrapper", "cxx_compiler", "harness")
    }
    inputs = [{"role": "harness", **pins["harness"]}]
    commands = [
        {"stage": stage, "cwd": str(tmp_path), "argv": [pins[tool]["path"], "owned-input"]}
        for stage, tool in (("emit", "gsim_emitter"), ("compile", "cxx_wrapper"), ("link", "cxx_wrapper"))
    ]
    doc = {
        "schema_version": GE.STRICT_RECEIPT_SCHEMA,
        "status": "complete",
        "binary_sha256": engine["binary"]["sha256"],
        "firrtl_sha256": pins["firrtl"]["sha256"],
        "model_manifest_sha256": pins["model_manifest"]["sha256"],
        "artifacts": {"binary": engine["binary"], **{name: pins[name] for name in ("firrtl", "model_manifest")}},
        "tools": {name: pins[name] for name in ("gsim_emitter", "cxx_wrapper", "cxx_compiler")},
        "inputs": inputs,
        "inputs_sha256": GE._canonical_sha(inputs),
        "commands": commands,
        "commands_sha256": GE._canonical_sha(commands),
        "provenance": {
            "firrtl_boundary": GE.FIRRTL_BOUNDARY_ADOPTED,
            "elaboration_performed": False,
            "warning": GE.ADOPTED_FIRRTL_WARNING,
        },
    }
    receipt = _file(tmp_path, "receipt.json", json.dumps(doc))
    engine["receipt"] = _pin(receipt)
    facts["inputs"]["fir_sha256"] = pins["firrtl"]["sha256"]
    return engine, doc, receipt


def test_gsim_uses_strict_existing_checks_and_exact_selected_firrtl(selection, tmp_path):
    engine, doc, receipt = _gsim(selection, tmp_path)
    bound = _bound(selection)
    assert bound.available("gsim") is True
    assert bound.gsim_path() == Path(engine["binary"]["path"])
    assert bound.gsim_selected_firrtl_status()[0] is True
    Path(doc["tools"]["gsim_emitter"]["path"]).write_text("changed tool\n")
    assert bound.available("gsim") is False
    assert bound.gsim_selected_firrtl_status()[0] is False


def test_gsim_resolution_uses_exact_selected_receipt_without_global_discovery(selection, tmp_path, monkeypatch):
    engine, doc, receipt = _gsim(selection, tmp_path)
    monkeypatch.setattr(GE, "resolve", lambda *a, **k: pytest.fail("global resolver reached"))
    bound = _bound(selection)
    resolved = bound.gsim_resolution()
    assert type(resolved) is GE.Resolution
    assert resolved.target == "synthetic"
    assert resolved.path == Path(engine["binary"]["path"])
    assert resolved.source == "contract"
    assert resolved.digest == engine["binary"]["sha256"]
    assert resolved.receipt_status == "bound"
    assert resolved.receipt["receipt_sha256"] == engine["receipt"]["sha256"]
    assert resolved.receipt["firrtl_sha256"] == selection[1]["inputs"]["fir_sha256"]
    assert resolved.ok and resolved.flavour == "binary"
    Path(doc["tools"]["gsim_emitter"]["path"]).write_text("changed tool\n")
    with pytest.raises(ValueError):
        bound.gsim_resolution()


@pytest.mark.parametrize("change", ["legacy", "wrong_firrtl", "sibling_binary", "changed_receipt", "missing_receipt"])
def test_gsim_refuses_weak_foreign_or_changed_lineage(selection, tmp_path, change):
    engine, doc, receipt = _gsim(selection, tmp_path)
    if change == "legacy":
        doc["schema_version"] = "merlin.gsim-model-build.v2"
    elif change == "wrong_firrtl":
        selection[1]["inputs"]["fir_sha256"] = "0" * 64
    elif change == "sibling_binary":
        other = tmp_path / "other-binary"
        other.write_bytes(Path(engine["binary"]["path"]).read_bytes())
        doc["artifacts"]["binary"] = _pin(other)
    receipt.write_text(json.dumps(doc))
    engine["receipt"] = _pin(receipt)
    bound = _bound(selection)
    if change == "changed_receipt":
        receipt.write_text("{}")
    elif change == "missing_receipt":
        receipt.unlink()
    assert bound.available("gsim") is False
