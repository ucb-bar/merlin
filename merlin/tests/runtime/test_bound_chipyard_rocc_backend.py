"""Neutral process/data controls, never simulator or hardware qualification."""

import base64
import copy
import fcntl
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime import harness_render as H
from merlin.runtime.backends import chipyard_rocc_bound as B
from merlin.runtime.backends import rocc_selection as S
from merlin.targetgen import gsim_emulator as GE
from merlin.targetgen.contract.readback_policy import (
    COHERENT_DUMP_V1,
    COHERENT_PACKET_V1,
    FULL_VALUES_B64,
    ReadbackPolicy,
)

pytest_plugins = ("test_rtl_build_receipt",)


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


def test_explicit_execution_metadata_never_discovers_provider(monkeypatch, selection):
    from merlin.runtime.backends import base
    from merlin.targetgen import target_registry

    def forbidden(*args, **kwargs):
        pytest.fail("explicit execution data discovered a provider")

    monkeypatch.setattr(base, "get_backend", forbidden)
    monkeypatch.setattr(target_registry, "resolve", forbidden)
    contract, facts = selection
    observed = base.execution_capability_facts("synthetic", contract=contract, facts=facts)
    assert observed["whole_program_kernel_abi"]["satisfied"] is True
    assert observed["whole_program_kernel_abi"]["tier"] == "backend_declared"
    assert all(row["tier"] == "backend_declared" for row in observed.values())
    incomplete = base.execution_capability_facts("synthetic", contract=contract, facts={})
    assert all(row["satisfied"] is None for row in incomplete.values())


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


def test_native_build_service_keeps_renderer_and_recipe_from_one_selection(selection, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen.native_model_execution import _build_service_for

    first = _bound(selection)
    changed = copy.deepcopy(selection)
    changed[0]["harness_abi"]["entry_symbol"] = "later_entry"
    changed[0]["runner"]["chipyard_rocc"]["toolchain"]["kernel_stack_frame"]["entry_symbol"] = "later_entry"
    later = _bound(changed)
    selections = iter((first, later))
    monkeypatch.setattr(base, "get_backend", lambda _target: next(selections))
    service = _build_service_for("synthetic")
    assert service.recipe.require_kernel_stack_frame().entry_symbol == "entry"
    rendered = service.render(
        _program(), target="synthetic", inputs={"X": [[1, 2]]}, readback_policy=ReadbackPolicy(FULL_VALUES_B64)
    )
    assert "extern void entry(" in rendered and "later_entry(" not in rendered
    # A later public lookup still selects the current backend; there is no cache.
    assert base.harness_build_recipe("synthetic").require_kernel_stack_frame().entry_symbol == "later_entry"


@pytest.mark.parametrize("consumer", ["public", "native_service"])
def test_selected_recipe_reopens_tool_source_after_backend_binding(selection, monkeypatch, consumer):
    from merlin.runtime.backends import base
    from merlin.targetgen.native_model_execution import _build_service_for

    bound = _bound(selection)
    compiler = Path(selection[0]["runner"]["chipyard_rocc"]["toolchain"]["compiler"]["path"])

    def selected(_target):
        compiler.write_text("changed selected compiler bytes\n")
        return bound

    monkeypatch.setattr(base, "get_backend", selected)
    with pytest.raises(B.RoCCExecutionError, match="changed"):
        if consumer == "public":
            base.harness_build_recipe("synthetic")
        else:
            _build_service_for("synthetic")


@pytest.mark.parametrize("consumer", ["public", "native_service"])
def test_recipe_missing_capability_keeps_target_named_refusal(tmp_path, monkeypatch, consumer):
    from types import SimpleNamespace

    from merlin.runtime.backends import base
    from merlin.targetgen.native_model_execution import _build_service_for

    source = tmp_path / "backend.py"
    source.write_text("# neutral backend with no bare-metal recipe\n")
    backend = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(base, "get_backend", lambda _target: backend)
    with pytest.raises(NotImplementedError, match="target 'synthetic' declares no harness_build_recipe"):
        if consumer == "public":
            base.harness_build_recipe("synthetic")
        else:
            _build_service_for("synthetic")


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


def test_job_budget_respects_selected_process_ceiling(selection, monkeypatch):
    from merlin.common import execution_deadline as D
    from merlin.targetgen.contract.compile import _selected_run_timeout

    contract, facts = copy.deepcopy(selection)
    contract["runner"]["chipyard_rocc"]["engines"]["spike"]["max_timeout_s"] = 600
    bound = B.bind(target="synthetic", contract=contract, facts=facts)
    monkeypatch.setattr(D.time, "monotonic", lambda: 100.0)
    assert _selected_run_timeout(bound, "spike", 900, D.ExecutionDeadline(100, 900)) == 600
    assert _selected_run_timeout(bound, "spike", 900, D.ExecutionDeadline(0, 150)) == 50


@pytest.mark.parametrize("resources", [None, {}, {"max_native_timeout_s": True},
    {"max_native_timeout_s": 0}, {"max_native_timeout_s": -1},
    {"max_native_timeout_s": 86401}, {"max_native_timeout_s": 1800.0},
    {"max_native_timeout_s": 1800, "extra": 1}])
def test_native_resource_selection_refuses_malformed_bounds(selection, resources):
    selection[0]["runner"]["chipyard_rocc"]["execution_resources"] = resources
    with pytest.raises(B.RoCCExecutionError):
        _bound(selection)


def test_legacy_native_bound_requires_explicit_host_extension(selection, tmp_path):
    engine, _, _ = _gsim(selection, tmp_path)
    engine["max_timeout_s"] = 601
    with pytest.raises(B.RoCCExecutionError, match="bounded process limit"):
        _bound(selection)
    block = selection[0]["runner"]["chipyard_rocc"]
    block["execution_resources"] = {"max_native_timeout_s": 1800}
    engine["max_timeout_s"] = 1801
    with pytest.raises(B.RoCCExecutionError, match="bounded process limit"):
        _bound(selection)


def test_native_extension_does_not_extend_functional_engine(selection):
    block = selection[0]["runner"]["chipyard_rocc"]
    block["execution_resources"] = {"max_native_timeout_s": 1800}
    block["engines"]["spike"]["max_timeout_s"] = 601
    with pytest.raises(B.RoCCExecutionError, match="bounded process limit"):
        _bound(selection)


def test_model_spike_resource_selection_extends_only_functional_process_budget(selection):
    original, facts = selection
    selected = S.select_model_spike_timeout(original, 172800)
    assert original["runner"]["chipyard_rocc"]["engines"]["spike"]["max_timeout_s"] == 5
    block = selected["runner"]["chipyard_rocc"]
    assert block["execution_resources"] == {
        "max_native_timeout_s": 600, "max_model_spike_timeout_s": 172800,
    }
    assert block["engines"]["spike"]["max_timeout_s"] == 172800
    bound = B.bind(target="synthetic", contract=selected, facts=facts)
    assert bound.execution_timeout_limit("spike") == 172800


@pytest.mark.parametrize("budget", [True, 0, -1, 172801, 600.0])
def test_model_spike_resource_selection_refuses_unbounded_or_nonnumeric_budget(selection, budget):
    with pytest.raises(B.RoCCExecutionError, match="model Spike process ceiling"):
        S.select_model_spike_timeout(selection[0], budget)


def test_explicit_native_budget_keeps_frozen_selection_and_remaining_deadline(selection, tmp_path, monkeypatch):
    from merlin.common import execution_deadline as D
    from merlin.targetgen.contract.compile import _selected_run_timeout

    engine, _, _ = _gsim(selection, tmp_path)
    engine["max_timeout_s"] = 1800
    block = selection[0]["runner"]["chipyard_rocc"]
    block["execution_resources"] = {"max_native_timeout_s": 1800}
    bound = _bound(selection)
    original_identity = bound.execution_engine_identity("gsim")
    block["execution_resources"]["max_native_timeout_s"] = 3600
    engine["max_timeout_s"] = 3600
    assert bound.execution_timeout_limit("gsim") == 1800
    assert bound.execution_engine_identity("gsim") == original_identity
    assert original_identity["execution_resources"] == {
        "max_native_timeout_s": 1800, "selected_engine_timeout_s": 1800,
    }
    monkeypatch.setattr(D.time, "monotonic", lambda: 100.0)
    assert _selected_run_timeout(bound, "gsim", 3600, D.ExecutionDeadline(100, 3600)) == 1800
    assert _selected_run_timeout(bound, "gsim", 1200, D.ExecutionDeadline(100, 3600)) == 1200
    assert _selected_run_timeout(bound, "gsim", 3600, D.ExecutionDeadline(0, 150)) == 50
    with pytest.raises(TimeoutError, match="wall budget expired"):
        _selected_run_timeout(bound, "gsim", 3600, D.ExecutionDeadline(0, 99))


def test_selected_extended_native_budget_reaches_actual_process(selection, tmp_path, monkeypatch):
    from merlin.targetgen import rtl_engine_policy

    engine, _, _ = _gsim(selection, tmp_path)
    engine["max_timeout_s"] = 1800
    selection[0]["runner"]["chipyard_rocc"]["execution_resources"] = {"max_native_timeout_s": 1800}
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kwargs: nullcontext())
    bound = _bound(selection)
    elf = _file(tmp_path, "requested.elf", "owned original executable")
    assert bound.run_elf(elf, simulator="gsim", timeout=1200) == _console()
    with pytest.raises(B.RoCCExecutionError, match="bounded deadline"):
        bound.run_elf(elf, simulator="gsim", timeout=1801)


def test_private_gsim_cycle_budget_reaches_actual_process_without_changing_selection(
    selection, tmp_path, monkeypatch
):
    from merlin.targetgen import rtl_engine_policy

    engine, _, _ = _gsim(selection, tmp_path)
    engine["argv"].append("+max-cycles=20000000")
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kwargs: nullcontext())
    bound = _bound(selection)
    selected = bound.execution_engine_identity("gsim")
    elf = _file(tmp_path, "requested.elf", "owned original executable")
    assert bound.run_elf(elf, simulator="gsim", timeout=2, private_max_cycles=150000000) == _console()
    record = I.verify(next(tmp_path.glob("invocations/*/invocation.json")))
    assert record["argv"][-1] == "+max-cycles=150000000"
    assert bound.execution_engine_identity("gsim") == selected
    with pytest.raises(ValueError, match="cycle budget"):
        bound.run_elf(elf, simulator="gsim", timeout=2, private_max_cycles=0)
    with pytest.raises(ValueError, match="cycle budget"):
        bound.run_elf(elf, simulator="gsim", timeout=2, private_max_cycles=True)


def test_private_gsim_cycle_budget_requires_source_bound_cycle_argument(selection, tmp_path):
    _gsim(selection, tmp_path)
    bound = _bound(selection)
    elf = _file(tmp_path, "requested.elf", "owned original executable")
    with pytest.raises(ValueError, match="unique source-bound cycle limit"):
        bound.run_elf(elf, simulator="gsim", timeout=2, private_max_cycles=150000000)
    assert not list(tmp_path.glob("invocations/*/invocation.json"))


def test_expired_job_budget_refuses_before_selected_process(selection, monkeypatch):
    from merlin.common import execution_deadline as D
    from merlin.targetgen.contract.compile import _selected_run_timeout

    monkeypatch.setattr(D.time, "monotonic", lambda: 100.0)
    with pytest.raises(TimeoutError, match="wall budget expired"):
        _selected_run_timeout(_bound(selection), "spike", 900, D.ExecutionDeadline(0, 99))


def test_ordinary_integer_runner_stimulus_reaches_strict_harness(selection):
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.targetgen.contract.compile import _ordinary_integer_harness_inputs

    cb = _program()
    del cb["kernel_abi"]
    cb["commands"] = [{"opcode": "MOVE", "operands": {"src": "X", "dst": "Y"}}]
    inputs = _ordinary_integer_harness_inputs(cb)
    assert inputs == {"X": materialize_inputs(cb)["X"].to_list()}
    assert "entry(tensor_0, tensor_1);" in _bound(selection).render_harness(
        cb, target="synthetic", inputs=inputs
    )
    with pytest.raises(ValueError, match="exact complete logical input roster"):
        _bound(selection).render_harness(cb, target="synthetic", inputs={})


def test_integer_stimulus_fallback_does_not_replace_model_or_float_inputs():
    from merlin.targetgen.contract.compile import _ordinary_integer_harness_inputs

    with pytest.raises(ValueError, match="whole-program"):
        _ordinary_integer_harness_inputs(_program())
    cb = _program()
    del cb["kernel_abi"]
    cb["tensors"]["X"]["dtype"] = "f32"
    with pytest.raises(ValueError, match="non-integer"):
        _ordinary_integer_harness_inputs(cb)


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


def test_engine_override_keeps_default_parser_and_spike_policy(selection, tmp_path):
    contract, facts = copy.deepcopy(selection)
    spike = copy.deepcopy(contract["runner"]["chipyard_rocc"]["engines"]["spike"])
    engine, _, _ = _gsim((contract, facts), tmp_path)
    contract["runner"]["chipyard_rocc"]["engines"]["spike"] = spike
    engine["readback_transport"] = COHERENT_DUMP_V1
    bound = B.bind(target="synthetic", contract=contract, facts=facts)
    policy = bound.engine_readback_policy("gsim")
    assert policy.transport == COHERENT_DUMP_V1
    assert bound.engine_readback_policy("spike").transport == FULL_VALUES_B64
    assert bound.readback_policy().transport == FULL_VALUES_B64
    assert bound.execution_engine_identity("gsim")["readback_transport"] == COHERENT_DUMP_V1
    assert bound.memory_readback_transport("gsim") == "gsim_coherent_dump_v1"
    assert bound.memory_readback_transport("spike") is None
    unsupported = _program()
    unsupported["tensors"]["Y"]["dtype"] = "f64"
    assert bound.engine_readback_policy("gsim", cb=unsupported).transport == FULL_VALUES_B64
    with pytest.raises(ValueError, match="omitted complete output"):
        bound.parse_output("DONE\n")
    assert bound.parse_output("DONE\n", readback_policy=policy) == ({}, {})
    text = bound.render_harness(_program(), target="synthetic", inputs={"X": [[1, 2]]}, readback_policy=policy)
    assert 'htif_puts("DONE\\n")' in text and "OUT_B64_BEGIN" not in text
    assert "tensor_1[2]" in text
    with pytest.raises(ValueError, match="readback differs"):
        H.render_harness(_program(), target="synthetic", inputs={"X": [[1, 2]]},
                         contract=selection[0], readback_policy=policy)


@pytest.fixture
def logical_memory_case(selection, tmp_path, monkeypatch, request):
    import shutil

    from merlin_experiments.phase1.feedback.native_memory_readback import NativeLogicalMemoryReadback

    from merlin.runtime.backends import base

    engine, _, _ = _gsim(selection, tmp_path)
    transport = getattr(request, "param", COHERENT_DUMP_V1)
    engine["readback_transport"] = transport
    if transport == COHERENT_PACKET_V1:
        from merlin.runtime.direct_kernel_harness import packet_output_blocks
        from merlin.runtime.harness_render import _roster
        from merlin.runtime.out_packet import out_bin_packet_capacity

        selection[0]["runner"]["chipyard_rocc"]["toolchain"]["headers"].extend(
            _unit(name, B._HEADERS) for name in ("out_bin", "out_bin_memory")
        )
        normalized, _ = _roster(_program())
        capacity = out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(normalized)))
    bound = _bound(selection)
    # Factory intentionally returns fresh genuine bound objects.
    monkeypatch.setattr(base, "get_backend", lambda target: _bound(selection))
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(selection[1]))
    source = tmp_path / "objects.c"
    packet_source = (f"unsigned char merlin_readback_packet[{capacity}];\n"
                     "volatile unsigned long long merlin_readback_packet_used __attribute__((aligned(8)));\n"
                     if transport == COHERENT_PACKET_V1 else "")
    source.write_text(packet_source + "static volatile unsigned char tensor_0[2]={1,2};\n"
                      "static volatile unsigned char tensor_1[2]={127,128};\n"
                      "int main(void){return tensor_0[0]+tensor_1[0];}\n")
    compiler = shutil.which("cc")
    assert compiler, "layout controls need an actual host ELF compiler"
    elf = tmp_path / "objects.elf"
    subprocess.run([compiler, "-no-pie", "-O0", str(source), "-o", str(elf)], check=True, capture_output=True)
    cb = _program()
    del cb["kernel_abi"]
    reader = NativeLogicalMemoryReadback(facts_path=facts_path, policy=ReadbackPolicy(transport))
    request = reader.prepare(cb=cb, target="synthetic", elf_path=elf, workdir=tmp_path,
                             simulator="gsim", backend=bound)["memory_readback"]
    return reader, request, cb, bound, selection, facts_path, elf


def _neutral_packet_dump(request, *, defect=None):
    import struct

    data, checksum = b"\x7f\x80", 0xCBF29CE484222325
    for byte in data:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    packet = b"OUT_BIN_BEGIN v1 merlin_block_0_0 1 2 1 s 2\n" + data
    packet += f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    if defect == "checksum":
        packet = packet.replace(f"{checksum:016x}".encode(), b"0000000000000000")
    if defect == "wrong_roster":
        packet = packet.replace(b"merlin_block_0_0", b"merlin_block_0_1")
    metadata, arena = request["regions"]
    raw = b"GSIMPKT1" + struct.pack("<QQQQ", metadata["base"] + int(defect == "address"),
                                   arena["base"], arena["bytes"], len(packet) + int(defect == "used"))
    raw += packet + b"PKTEND1\n"
    if defect == "truncated":
        raw = raw[:-1]
    if defect == "extra":
        raw += b"extra"
    Path(request["output_path"]).write_bytes(raw)


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
def test_neutral_packet_reader_preserves_original_full_output_roster(logical_memory_case):
    reader, request, cb, bound, *_ = logical_memory_case
    _neutral_packet_dump(request)
    outputs, evidence = reader.decode("DONE\n")
    assert outputs == {"Y": [[127, -128]]}
    assert evidence["transport"] == "gsim_coherent_packet_v1" and evidence["status"] == "complete"
    assert evidence["block_words"] == 256 and evidence["packet_framing"] == "neutral_dense_blocks_out_bin_v1"
    assert evidence["native_filesystem_view"] is None
    assert bound.engine_readback_policy("gsim", cb=cb).transport == COHERENT_PACKET_V1
    assert bound.readback_policy().transport == FULL_VALUES_B64
    assert request["regions"][0]["bytes"] == 8


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
@pytest.mark.parametrize("defect", ["address", "used", "checksum", "wrong_roster", "truncated", "extra"])
def test_neutral_packet_reader_refuses_malformed_full_value_envelope(logical_memory_case, defect):
    reader, request, *_ = logical_memory_case
    _neutral_packet_dump(request, defect=defect)
    with pytest.raises(ValueError):
        reader.decode("DONE\n")


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
@pytest.mark.parametrize("mutation", ["elf", "facts", "cb", "policy", "request", "regions", "binary"])
def test_neutral_packet_reader_reopens_original_source_authority(logical_memory_case, mutation):
    reader, request, cb, bound, _selection, facts, elf = logical_memory_case
    _neutral_packet_dump(request)
    if mutation == "elf":
        elf.write_bytes(elf.read_bytes() + b"changed")
    elif mutation == "facts":
        facts.write_text("{}")
    elif mutation == "cb":
        cb["tensors"]["Y"]["shape"] = [1, 3]
    elif mutation == "policy":
        reader.policy = ReadbackPolicy(COHERENT_DUMP_V1)
    elif mutation == "request":
        request["regions"][1]["bytes"] += 1
    elif mutation == "regions":
        Path(request["regions_path"]).write_text("0x1000 2\n")
    else:
        Path(bound._engine("gsim")["binary"]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        reader.decode("DONE\n")


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
def test_packet_process_uses_only_declared_original_mode(logical_memory_case, monkeypatch):
    from merlin.targetgen import rtl_engine_policy

    _reader, request, _cb, bound, *_ = logical_memory_case

    def run(argv, *, observed, **kwargs):
        assert argv[-1] == "+dump-mode=coherent-packet"
        _neutral_packet_dump(request)
        result = subprocess.CompletedProcess(argv, 0, stdout=b"DONE\n", stderr=b"")
        observed.complete(result)
        return result

    monkeypatch.setattr(B, "_bounded_process", run)
    monkeypatch.setattr(rtl_engine_policy, "simulation_runtime_slot", lambda *a, **k: nullcontext())
    assert bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request) == "DONE\n"


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
def test_packet_process_cannot_downgrade_original_native_view(logical_memory_case, monkeypatch):
    from merlin.runtime import native_runtime

    _reader, request, _cb, bound, *_ = logical_memory_case
    monkeypatch.setattr(native_runtime, "active_native_runtime", lambda: object())
    monkeypatch.setattr(B, "_bounded_process", lambda *a, **k: pytest.fail("view was downgraded"))
    with pytest.raises(ValueError, match="native filesystem view has no approved coherent-export mapping"):
        bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request)


def test_packet_selection_refuses_missing_original_codec_pins(selection, tmp_path):
    engine, *_ = _gsim(selection, tmp_path)
    engine["readback_transport"] = COHERENT_PACKET_V1
    bound = _bound(selection)
    with pytest.raises(ValueError, match="complete codec header roster"):
        bound.engine_readback_policy("gsim")


@pytest.mark.parametrize("logical_memory_case", [COHERENT_PACKET_V1], indirect=True)
def test_packet_process_refuses_a_different_native_transport(logical_memory_case, monkeypatch):
    _reader, request, _cb, bound, *_ = logical_memory_case
    request["transport"] = "gsim_coherent_dump_v1"
    # A separately well-formed raw request cannot substitute for the selected packet mode.
    request["regions"] = [request["regions"][1]]
    Path(request["regions_path"]).write_text(
        f"{request['regions'][0]['base']:#x} {request['regions'][0]['bytes']}\n"
    )
    monkeypatch.setattr(B, "_bounded_process", lambda *a, **k: pytest.fail("changed mode executed"))
    with pytest.raises(ValueError, match="declared engine transport"):
        bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request)


def _logical_dump(request, *, source=0, data=b"\x7f\x80", trailer=True):
    import struct

    region = request["regions"][0]
    raw = b"GSIMDMP1" + struct.pack("<QQQQ", source, 1, region["base"], region["bytes"]) + data
    if trailer:
        raw += b"GSIMEND1" + struct.pack("<QQ", 1, len(data))
    Path(request["output_path"]).write_bytes(raw)


def test_neutral_coherent_reader_resolves_actual_symbols_and_complete_values(logical_memory_case):
    reader, request, cb, *_ = logical_memory_case
    _logical_dump(request)
    outputs, evidence = reader.decode("DONE\n")
    assert outputs == {"Y": [[127, -128]]}
    assert evidence["status"] == "complete" and evidence["source"] == "coherent"
    assert evidence["outputs"] == [{"name": "Y", "shape": [1, 2], "bytes": 2}]
    with pytest.raises(ValueError, match="one original"):
        reader.prepare(cb=cb, target="synthetic", elf_path=logical_memory_case[-1],
                       workdir=Path(request["output_path"]).parent, simulator="gsim", backend=logical_memory_case[3])


@pytest.mark.parametrize("mutation", ["elf", "facts", "cb_shape", "policy", "request", "regions", "binary"])
def test_neutral_coherent_reader_rejects_changed_authority(logical_memory_case, mutation):
    reader, request, cb, bound, selection, facts_path, elf = logical_memory_case
    _logical_dump(request)
    if mutation == "elf":
        elf.write_bytes(elf.read_bytes() + b"changed")
    elif mutation == "facts":
        facts_path.write_text("{}")
    elif mutation == "cb_shape":
        cb["tensors"]["Y"]["shape"] = [1, 3]
    elif mutation == "policy":
        reader.policy = ReadbackPolicy(FULL_VALUES_B64)
    elif mutation == "request":
        request["regions"][0]["bytes"] = 3
    elif mutation == "regions":
        Path(request["regions_path"]).write_text("0x1000 2\n")
    else:
        Path(bound._engine("gsim")["binary"]["path"]).write_text("changed binary")
    with pytest.raises(ValueError):
        reader.decode("DONE\n")


@pytest.mark.parametrize("defect", ["truncated", "backing_store", "extra_byte", "missing_done", "serial_values"])
def test_neutral_coherent_reader_refuses_incomplete_or_noncoherent_values(logical_memory_case, defect):
    reader, request, *_ = logical_memory_case
    _logical_dump(request, source=1 if defect == "backing_store" else 0,
                  trailer=defect != "truncated", data=b"\x7f\x80\x00" if defect == "extra_byte" else b"\x7f\x80")
    console = "" if defect == "missing_done" else "OUT Y 1 2 1 2\nDONE\n" if defect == "serial_values" else "DONE\n"
    with pytest.raises(ValueError):
        reader.decode(console)


def test_bound_memory_process_uses_original_command_and_revalidates_export(logical_memory_case, monkeypatch):
    from merlin.targetgen import rtl_engine_policy

    _reader, request, _cb, bound, *_ = logical_memory_case
    seen = []

    def run(argv, *, cwd, env, timeout, max_bytes, observed):
        seen.append((argv, cwd, env, timeout, max_bytes))
        assert argv[0] == str(bound.gsim_path())
        assert argv[1] == str(logical_memory_case[-1])
        assert argv[2:] == [f"+dump-regions={request['regions_path']}",
                            f"+dump-out={request['output_path']}", "+dump-mode=coherent"]
        _logical_dump(request)
        result = subprocess.CompletedProcess(argv, 0, stdout=b"DONE\n", stderr=b"")
        observed.complete(result)
        return result

    monkeypatch.setattr(B, "_bounded_process", run)
    monkeypatch.setattr(rtl_engine_policy, "simulation_runtime_slot", lambda *a, **k: nullcontext())
    assert bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request) == "DONE\n"
    assert len(seen) == 1 and 0 < seen[0][3] <= 5
    # A preexisting result cannot be accepted as a fresh execution output.
    with pytest.raises(ValueError, match="not fresh"):
        bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request)
    assert len(seen) == 1


def test_bound_memory_process_refuses_changed_regions_after_process(logical_memory_case, monkeypatch):
    from merlin.targetgen import rtl_engine_policy

    _reader, request, _cb, bound, *_ = logical_memory_case

    def run(argv, *, observed, **kwargs):
        _logical_dump(request)
        Path(request["regions_path"]).write_text("0x1000 2\n")
        result = subprocess.CompletedProcess(argv, 0, stdout=b"DONE\n", stderr=b"")
        observed.complete(result)
        return result

    monkeypatch.setattr(B, "_bounded_process", run)
    monkeypatch.setattr(rtl_engine_policy, "simulation_runtime_slot", lambda *a, **k: nullcontext())
    with pytest.raises(ValueError, match="region manifest changed"):
        bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request)


def test_bound_coherent_export_cannot_silently_downgrade_selected_native_view(logical_memory_case, monkeypatch):
    from merlin.runtime import native_runtime

    _reader, request, _cb, bound, *_ = logical_memory_case
    monkeypatch.setattr(native_runtime, "active_native_runtime", lambda: object())
    monkeypatch.setattr(B, "_bounded_process", lambda *a, **k: pytest.fail("selected view was downgraded"))
    with pytest.raises(ValueError, match="native filesystem view has no approved coherent-export mapping"):
        bound.run_elf(logical_memory_case[-1], simulator="gsim", timeout=5, memory_readback=request)
    assert not Path(request["output_path"]).exists()


def test_memory_backend_binding_rejects_foreign_or_derived_class(logical_memory_case):
    from merlin_experiments.phase1.feedback.native_memory_readback import _same_selected_backend

    _reader, _request, _cb, bound, selection, *_ = logical_memory_case
    assert _same_selected_backend(bound, _bound(selection), "gsim")
    replacement = copy.deepcopy(selection)
    replacement[0]["runner"]["chipyard_rocc"]["engines"]["gsim"]["environment"]["EXTRA"] = "foreign"
    assert not _same_selected_backend(bound, _bound(replacement), "gsim")

    class Foreign(B.BoundRoCCBackend):
        pass

    derived = Foreign(bound.target, bound._selection_json, bound.rocc_semantics)
    assert not _same_selected_backend(derived, bound, "gsim")


def test_ordinary_oracle_resolves_engine_policy_without_changing_adapter_consensus(logical_memory_case, monkeypatch):
    from merlin_experiments.phase1.feedback import native_memory_readback as MR

    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import oracle_readback as OR
    from merlin.targetgen.contract import compile as C
    from merlin.targetgen.rtl import facts as F

    _reader, _request, cb, bound, _selection, facts_path, _elf = logical_memory_case
    monkeypatch.setattr(F, "rtl_facts_path", lambda target: facts_path)
    monkeypatch.setattr(MR, "select_memory_engine", lambda **kwargs: ({"original": True}, lambda: {"original": True}))
    calls = []
    monkeypatch.setattr(C, "run_on_oracle", lambda *args, **kwargs: calls.append(kwargs) or {"status": "diagnostic"})
    assert OR._run(cb, "original LLVM", sim="gsim", target=bound.target, backend=bound,
                   workdir=facts_path.parent, timeout=5, policy=None) == {"status": "diagnostic"}
    assert calls[-1]["readback_policy"].transport == COHERENT_DUMP_V1
    assert type(calls[-1]["memory_readback"]) is MR.NativeLogicalMemoryReadback
    # An explicit invocation choice still overrides engine defaults.
    OR._run(cb, "original LLVM", sim="gsim", target=bound.target, backend=bound,
            workdir=facts_path.parent, timeout=5, policy=ReadbackPolicy(FULL_VALUES_B64))
    assert "memory_readback" not in calls[-1] and calls[-1]["readback_policy"].transport == FULL_VALUES_B64
    adapters = {
        tier: CR.simulator_adapter(engine, bound.target) for tier, engine in {"L2": "spike", "L3": "gsim"}.items()
    }
    assert CR._adapter_readback_policy(adapters) is None


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


@pytest.mark.parametrize("simulator", ("spike", "gsim"))
@pytest.mark.parametrize("failure", (False, True))
def test_selected_execution_guards_only_gsim_and_releases_after_failure(
    selection,
    tmp_path,
    monkeypatch,
    simulator,
    failure,
):
    from merlin.runtime.backends import chipyard_rocc_bound as owner
    from merlin.targetgen import rtl_engine_policy

    if simulator == "gsim":
        _gsim(selection, tmp_path)
    active = False
    events = []

    @contextmanager
    def guarded(*, wait_timeout_s):
        nonlocal active
        assert wait_timeout_s == 2
        active = True
        events.append("enter")
        try:
            yield
        finally:
            active = False
            events.append("exit")

    def execute(argv, **kwargs):
        assert active is (simulator == "gsim")
        if failure:
            raise subprocess.TimeoutExpired(argv, 2)
        result = subprocess.CompletedProcess(argv, 0, stdout=_console().encode(), stderr=b"")
        kwargs["observed"].complete(result)
        return result

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", guarded)
    monkeypatch.setattr(owner, "_bounded_process", execute)
    elf = _file(tmp_path, "requested.elf", "owned original executable")
    bound = _bound(selection)
    if failure:
        with pytest.raises(subprocess.TimeoutExpired):
            bound.run_elf(elf, simulator=simulator, timeout=2)
    else:
        assert bound.run_elf(elf, simulator=simulator, timeout=2) == _console()
    assert active is False
    assert events == (["enter", "exit"] if simulator == "gsim" else [])


@pytest.mark.parametrize("simulator", ("spike", "gsim"))
def test_selected_native_process_uses_shared_compute_cpu(selection, tmp_path, monkeypatch, simulator):
    from merlin.targetgen import rtl_engine_policy

    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "compute"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    monkeypatch.setenv("MERLIN_GSIM_SLOT_ROOT", str(tmp_path / "native"))
    monkeypatch.setenv("MERLIN_GSIM_CPUS", "obsolete-fixed-roster")
    monkeypatch.setitem(rtl_engine_policy.CAPSULE_WORKER_CAP, "gsim", 1)
    monkeypatch.setattr(rtl_engine_policy, "_native_gsim_census", lambda **_: rtl_engine_policy._NativeCensus(0, ()))
    if simulator == "gsim":
        _gsim(selection, tmp_path)

    def execute(argv, **kwargs):
        assert os.sched_getaffinity(0) == {cpu}
        with (slots / f"cpu-{cpu}.lock").open("rb") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.CompletedProcess(argv, 0, stdout=_console().encode(), stderr=b"")
        kwargs["observed"].complete(result)
        return result

    monkeypatch.setattr(B, "_bounded_process", execute)
    elf = _file(tmp_path, "requested.elf", "owned original executable")
    assert _bound(selection).run_elf(elf, simulator=simulator, timeout=2) == _console()


@pytest.mark.parametrize("engine", ["spike", "gsim"])
def test_oracle_metadata_keeps_concrete_engine_through_actual_tier_projection(selection, tmp_path, engine):
    from merlin_experiments.phase1.timing import barrier_timing_identity

    from merlin.targetgen.capsule_runner import _sim_telemetry
    from merlin.targetgen.contract.compile import simulator_provenance

    if engine == "gsim":
        _gsim(selection, tmp_path)
    bound = _bound(selection)
    metadata = {**bound.ORACLE[engine], "provenance": simulator_provenance(bound, engine)}
    tier = {"status": "pass", **_sim_telemetry(metadata, "diagnostic_console.log")}
    assert tier["engine"] == engine
    assert tier["sim_provenance"]["engine"] == engine
    if engine == "gsim":
        identity = barrier_timing_identity(tier)
        assert identity["engine"] == engine
        assert identity["simulator_sha256"] == tier["sim_provenance"]["sha256"]


def test_interleaved_stderr_does_not_interrupt_complete_stdout_frames(selection, tmp_path):
    engine = _engine(selection)
    binary = Path(engine["binary"]["path"])
    lines = _console().splitlines()
    binary.write_text(
        "#!/bin/sh\n"
        + "".join("printf '%s\\n' '" + line + "'\nprintf '%s\\n' 'RTL diagnostic' >&2\n" for line in lines)
    )
    engine["binary"] = _pin(binary)
    elf = tmp_path / "input.elf"
    elf.write_bytes(b"owned input")
    bound = _bound(selection)
    text = bound.run_elf(elf, simulator="spike", timeout=2)
    assert text == _console()
    assert bound.parse_output(text)[0] == {"Y": [[127, -128]]}
    record = next(tmp_path.glob("invocations/*/invocation.json"))
    assert record.with_name("stderr.bin").read_text() == "RTL diagnostic\n" * len(lines)


def test_declared_failure_on_stderr_still_refuses(selection, tmp_path):
    engine = _engine(selection)
    binary = Path(engine["binary"]["path"])
    binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + _console().rstrip() + "'\nprintf '%s\\n' DECLARED_FAILURE >&2\n")
    engine["binary"] = _pin(binary)
    elf = tmp_path / "input.elf"
    elf.write_bytes(b"owned input")
    with pytest.raises(ValueError, match="declared failure"):
        _bound(selection).run_elf(elf, simulator="spike", timeout=2)


def test_stderr_flood_obeys_combined_console_budget(selection, tmp_path):
    engine = _engine(selection)
    binary = Path(engine["binary"]["path"])
    binary.write_text("#!/bin/sh\nwhile :; do printf 012345678901234567890123456789 >&2; done\n")
    engine["binary"] = _pin(binary)
    engine["max_console_bytes"] = 256
    elf = tmp_path / "input.elf"
    elf.write_bytes(b"owned input")
    with pytest.raises(ValueError, match="byte budget"):
        _bound(selection).run_elf(elf, simulator="spike", timeout=2)
    record = next(tmp_path.glob("invocations/*/invocation.json"))
    assert record.with_name("stdout.bin").stat().st_size + record.with_name("stderr.bin").stat().st_size <= 257
    assert json.loads(record.read_bytes())["status"] == "failed"


def test_selected_clang_flags_are_explicit_and_unambiguous(selection):
    flags = selection[0]["runner"]["chipyard_rocc"]["toolchain"]["cflags"]
    flags += ["--target=riscv64-unknown-elf", "-fuse-ld=lld"]
    assert _bound(selection).harness_build_recipe().cflags == tuple(flags)
    for bad in ("--target=other", "-fuse-ld=other", "--target=", "-fuse-ld=/unselected/linker"):
        test = copy.deepcopy(selection)
        test[0]["runner"]["chipyard_rocc"]["toolchain"]["cflags"].append(bad)
        with pytest.raises(ValueError):
            _bound(test)


def test_dram_placement_annotation_preserves_exact_logical_storage():
    spec = {"shape": [2, 3], "dtype": "i32", "role": "input"}
    expected = H._tensor(spec)
    assert H._tensor({**spec, "base": 0x81000000}) == expected
    for base in (True, -1, 2**64, "0x81000000", None):
        with pytest.raises(ValueError, match="annotation"):
            H._tensor({**spec, "base": base})
    for field, value in (("storage_shape", [2, 16]), ("layout", "packed"), ("strides", [1, 2])):
        with pytest.raises(ValueError):
            H._tensor({**spec, "base": 0x81000000, field: value})


def _native(selection, chain):
    doc, validate, _, tmp = chain
    validate()
    binary = Path(doc["binary"]["path"])
    binary.chmod(0o700)
    contract, facts = copy.deepcopy(selection)
    block = contract["runner"]["chipyard_rocc"]
    block["config"] = "SyntheticConfig"
    facts["facts"]["source"]["config"] = "SyntheticConfig"
    facts["source_consistency"]["config"] = "SyntheticConfig"
    facts["inputs"]["fir_sha256"] = doc["firrtl"]["sha256"]
    engine = block["engines"].pop("spike")
    engine["binary"] = doc["binary"]
    engine["argv"] = ["{elf}"]
    engine["receipt"] = _pin(tmp / "receipt.json")
    block["engines"]["verilator"] = engine
    contract["runner"].pop("spike_extension")
    return contract, facts, binary


def test_native_deployment_joins_actual_observations_without_spike_extension(selection, chain):
    contract, facts, binary = _native(selection, chain)
    bound = B.bind(target="synthetic", contract=contract, facts=facts)
    assert bound.verilator_path() == binary
    identity = bound.execution_engine_identity("verilator")
    assert identity["receipt"]["firrtl"]["sha256"] == facts["inputs"]["fir_sha256"]
    assert "extension" not in identity and bound.ORACLE["verilator"]["derived_from_rtl"] is True
    assert S.prepare_contract(target="synthetic", contract=_deployment((contract, facts)), facts=facts) == contract


def test_native_receipt_cannot_be_reused_for_another_firrtl(selection, chain):
    contract, facts, _ = _native(selection, chain)
    facts["inputs"]["fir_sha256"] = "a" * 64
    bound = B.bind(target="synthetic", contract=contract, facts=facts)
    assert bound.available("verilator") is False
    with pytest.raises(ValueError, match="build identity"):
        bound.verilator_path()


def test_declared_model_spike_machine_is_explicit_and_recorded(selection, tmp_path, monkeypatch):
    from merlin.runtime.backends import spike_model

    bound = _bound(selection)
    elf = _file(tmp_path, "saved-model.elf", "owned model bytes\n")
    monkeypatch.setattr(spike_model, "declared_memory", lambda path: (0x80000000, 256 << 20))
    monkeypatch.setattr(spike_model, "declared_isa", lambda path: "rv64gc_zba_zbb_zfh")
    monkeypatch.setattr(spike_model, "declared_harts", lambda path: 1)
    flags, library = bound.spike_extension()
    assert flags and library == Path(selection[0]["runner"]["spike_extension"]["extlib"]).parent
    bound.run_declared_model_elf(elf, timeout=2)
    record = I.verify(next(tmp_path.glob("invocations/*/invocation.json")))
    assert "--isa=rv64gc_zba_zbb_zfh" in record["argv"]
    assert "--isa=rv64gc" not in record["argv"]
    assert "-m0x80000000:0x10000000" in record["argv"]
    assert "-p1" in record["argv"]
    assert record["inputs"] == [_pin(elf)]
    assert _engine(selection)["argv"] == ["--isa=rv64gc", "{elf}"]
