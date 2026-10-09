"""Invocation-only full-value readback binds builds without editing command buffers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import parse_console
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.readback_policy import (
    BUILD_RECEIPT,
    FULL_VALUES_B64,
    FULL_VALUES_BIN,
    ReadbackPolicy,
    read_console,
    require_build_receipt,
    require_full_value_roster,
    selected_build_inputs,
)


def _cb():
    return {
        "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
        "tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}},
        "params": {"console_value_cap": 1},
    }


def _service(tmp_path, renderer):
    source = tmp_path / "renderer.py"
    source.write_text("SOURCE=1\n", encoding="utf-8")
    script = tmp_path / "script.ld"
    script.write_text("SECTIONS {}\n", encoding="utf-8")
    recipe = HarnessBuildRecipe(
        compiler=tmp_path / "unused-gcc",
        include_roots=(),
        support_sources=(),
        link_script=script,
        load_address=0,
        cflags=("-march=rv64gc", "-mabi=lp64d"),
    )
    pins = tuple((str(path), file_digest(path)) for path in (source, Path(__file__).resolve()))
    return BuildOnlyService("fixture", recipe, renderer, pins), source


def test_policy_is_strictly_versioned_and_does_not_modify_capsule():
    policy = ReadbackPolicy(FULL_VALUES_B64)
    assert ReadbackPolicy.from_record(policy.record()) == policy
    for bad in ({"schema": "wrong", "transport": FULL_VALUES_B64}, {"schema": policy.schema}, {}):
        with pytest.raises(ValueError):
            ReadbackPolicy.from_record(bad)
    with pytest.raises(ValueError):
        ReadbackPolicy("other")
    assert ReadbackPolicy.from_record(ReadbackPolicy(FULL_VALUES_BIN).record()) == ReadbackPolicy(FULL_VALUES_BIN)


@pytest.mark.parametrize("transport", ["coherent_dump_v1", "coherent_packet_v1"])
def test_memory_readback_is_explicit_and_never_admits_serial_values(transport):
    policy = ReadbackPolicy(transport)
    assert ReadbackPolicy.from_record(policy.record()) == policy
    with pytest.raises(ValueError, match="memory admission"):
        require_full_value_roster(_cb(), "DONE\n", {"out": [[1, 2]]}, policy=policy)


@pytest.mark.parametrize("transport", ["coherent_dump_v1", "coherent_packet_v1"])
def test_memory_build_receipt_rechecks_selected_transport_and_header_bytes(monkeypatch, tmp_path, transport):
    def render(_cb, *, inputs, readback_policy):
        assert readback_policy == ReadbackPolicy(transport)
        return "int main(void) { return 0; }\n"

    service, _source = _service(tmp_path, render)
    build = tmp_path / "build"
    build.mkdir()
    obj = build / "kernel.o"
    obj.write_bytes(b"kernel")
    monkeypatch.setattr(
        "merlin.targetgen.runtime_build.derived_link_script", lambda *_args, **_kwargs: service.recipe.link_script
    )

    def fake_compile(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"elf" if "-T" in command else b"harness-object")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_compile)
    policy = ReadbackPolicy(transport)
    cb = _cb()
    saved = copy.deepcopy(cb)
    elf = compiler.link_elf(
        cb, obj, build, target="fixture", inputs={"arg": [1]}, _build_service=service, readback_policy=policy
    )
    recipe, source_pins = selected_build_inputs("fixture", service.recipe.with_effective_abi(), service, policy=policy)

    def verified():
        return require_build_receipt(
            build / BUILD_RECEIPT,
            policy=policy,
            cb=cb,
            target="fixture",
            recipe_record=recipe,
            source_pins=source_pins,
            object_path=obj,
            harness_path=build / "harness.c",
            elf_path=elf,
        )

    receipt = verified()
    packet = transport == "coherent_packet_v1"
    assert receipt["schema"] == ("merlin_readback_build_v3" if packet else "merlin_readback_build_v2")
    assert bool(receipt["staged_codec_sha256"]) == packet
    assert recipe["readback_transport"] == policy.record()
    codecs = ("out_b64.h", "out_bin.h", "out_bin_memory.h")
    assert all((build / name).exists() == packet for name in codecs)
    assert cb == saved
    for path in (obj, elf, build / "harness.c", *((build / name for name in codecs) if packet else ())):
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="receipt"):
            verified()
        path.write_bytes(original)


@pytest.mark.parametrize("transport", [None, FULL_VALUES_B64, FULL_VALUES_BIN])
def test_console_reader_uses_explicit_policy_not_filename(tmp_path, transport):
    path = tmp_path / "console.bin"
    path.write_bytes(b"DONE\n")
    policy = ReadbackPolicy(transport) if transport else None
    expected = b"DONE\n" if transport == FULL_VALUES_BIN else "DONE\n"
    assert read_console(path, policy=policy) == expected
    path.write_bytes(b"\xff\x00")
    if transport == FULL_VALUES_BIN:
        assert read_console(path, policy=policy) == b"\xff\x00"
    else:
        with pytest.raises(UnicodeDecodeError):
            read_console(path, policy=policy)


def test_console_reader_refuses_untyped_choice_before_filesystem_access(tmp_path):
    with pytest.raises(ValueError, match="explicit trusted"):
        read_console(tmp_path / "absent.bin", policy={"transport": FULL_VALUES_BIN})


@pytest.mark.parametrize("transport", ["coherent_dump_v1", "coherent_packet_v1"])
def test_console_reader_refuses_coherent_memory_before_filesystem_access(tmp_path, transport):
    with pytest.raises(ValueError, match="independent memory audit"):
        read_console(tmp_path / "absent.txt", policy=ReadbackPolicy(transport))


def test_binary_receipt_binds_both_staged_headers_and_declared_values(monkeypatch, tmp_path):
    from merlin.runtime.out_bin import parse_binary_console

    def render(_cb, *, inputs, readback_policy):
        assert readback_policy == ReadbackPolicy(FULL_VALUES_BIN)
        return '#include "out_bin.h"\nint main(void) { return 0; }\n'

    service, _source = _service(tmp_path, render)
    build = tmp_path / "build"
    build.mkdir()
    obj = build / "kernel.o"
    obj.write_bytes(b"kernel")
    monkeypatch.setattr(
        "merlin.targetgen.runtime_build.derived_link_script",
        lambda *_args, **_kwargs: service.recipe.link_script,
    )

    def fake_compile(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"elf" if "-T" in command else b"harness-object")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_compile)
    policy = ReadbackPolicy(FULL_VALUES_BIN)
    cb = _cb()
    elf = compiler.link_elf(
        cb,
        obj,
        build,
        target="fixture",
        inputs={"arg": [1]},
        _build_service=service,
        readback_policy=policy,
    )
    recipe, source_pins = selected_build_inputs(
        "fixture",
        service.recipe.with_effective_abi(),
        service,
        policy=policy,
    )
    assert len(recipe["readback_codecs"]) == 2

    def verified():
        return require_build_receipt(
            build / BUILD_RECEIPT,
            policy=policy,
            cb=cb,
            target="fixture",
            recipe_record=recipe,
            source_pins=source_pins,
            object_path=obj,
            harness_path=build / "harness.c",
            elf_path=elf,
        )

    assert verified()["staged_range_sha256"]
    for name in ("out_b64.h", "out_bin.h"):
        path = build / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="receipt"):
            verified()
        path.write_bytes(original)

    raw = b"\x01\x02"
    checksum = 0xCBF29CE484222325
    for byte in raw:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    console = b"OUT_BIN_BEGIN v1 out 1 2 1 u 2\n" + raw + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    outputs, _metrics = parse_binary_console(console)
    require_full_value_roster(cb, console, outputs, policy=policy)
    wide = b"\x01\x00\x02\x00"
    checksum = 0xCBF29CE484222325
    for byte in wide:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    wide_console = b"OUT_BIN_BEGIN v1 out 1 2 2 u 4\n" + wide + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    wide_outputs, _metrics = parse_binary_console(wide_console)
    with pytest.raises(ValueError, match="wire width"):
        require_full_value_roster(cb, wide_console, wide_outputs, policy=policy)


def test_optin_bypasses_legacy_build_cache(monkeypatch, tmp_path):
    from merlin.targetgen import build_cache

    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(build_cache, name, lambda *args, **kwargs: pytest.fail("legacy cache reached"))
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *args, **kwargs: tmp_path / "kernel.o")
    calls = []
    monkeypatch.setattr(compiler, "link_elf", lambda cb, obj, workdir, **kw: calls.append(kw) or tmp_path / "elf")
    cb = _cb()
    saved = copy.deepcopy(cb)
    policy = ReadbackPolicy(FULL_VALUES_B64)
    assert (
        compiler.compile_lowered_to_elf(cb, "llvm", tmp_path, target="fixture", readback_policy=policy)
        == tmp_path / "elf"
    )
    assert calls == [{"target": "fixture", "inputs": None, "warm_profile": None, "readback_policy": policy}]
    assert cb == saved


def test_real_link_seam_records_exact_selected_bytes_and_rejects_changed_source(monkeypatch, tmp_path):
    def render(cb, *, inputs, readback_policy):
        assert readback_policy == ReadbackPolicy(FULL_VALUES_B64)
        return '#include "out_b64.h"\nint main(void) { return 0; }\n'

    service, source = _service(tmp_path, render)
    build = tmp_path / "build"
    build.mkdir()
    obj = build / "kernel.o"
    obj.write_bytes(b"kernel")
    monkeypatch.setattr(
        "merlin.targetgen.runtime_build.derived_link_script",
        lambda *_args, **_kwargs: service.recipe.link_script,
    )

    def fake_compile(command, **_kwargs):
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"elf" if "-T" in command else b"harness-object")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_compile)
    cb = _cb()
    saved = copy.deepcopy(cb)
    policy = ReadbackPolicy(FULL_VALUES_B64)
    elf = compiler.link_elf(
        cb, obj, build, target="fixture", inputs={"arg": [1]}, _build_service=service, readback_policy=policy
    )
    recipe_record, source_pins = selected_build_inputs("fixture", service.recipe.with_effective_abi(), service)
    receipt = require_build_receipt(
        build / BUILD_RECEIPT,
        policy=policy,
        cb=cb,
        target="fixture",
        recipe_record=recipe_record,
        source_pins=source_pins,
        object_path=obj,
        harness_path=build / "harness.c",
        elf_path=elf,
    )
    assert receipt["build_identity_sha256"] and receipt["staged_codec_sha256"]
    assert cb == saved
    (build / "harness.c").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        require_build_receipt(
            build / BUILD_RECEIPT,
            policy=policy,
            cb=cb,
            target="fixture",
            recipe_record=recipe_record,
            source_pins=source_pins,
            object_path=obj,
            harness_path=build / "harness.c",
            elf_path=elf,
        )
    source.write_text("SOURCE=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        selected_build_inputs("fixture", service.recipe.with_effective_abi(), service)


def test_selected_source_mutation_during_link_never_finishes_receipt(monkeypatch, tmp_path):
    def render(cb, *, inputs, readback_policy):
        return '#include "out_b64.h"\nint main(void) { return 0; }\n'

    service, source = _service(tmp_path, render)
    build = tmp_path / "build"
    build.mkdir()
    obj = build / "kernel.o"
    obj.write_bytes(b"kernel")
    monkeypatch.setattr(
        "merlin.targetgen.runtime_build.derived_link_script",
        lambda *_args, **_kwargs: service.recipe.link_script,
    )

    def mutate(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"object")
        source.write_text("SOURCE=2\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(compiler.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="source"):
        compiler.link_elf(
            _cb(),
            obj,
            build,
            target="fixture",
            inputs={"arg": [1]},
            _build_service=service,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        )
    assert json.loads((build / BUILD_RECEIPT).read_text()) == {"status": "incomplete"}


def test_full_value_roster_requires_packed_complete_values():
    cb = _cb()
    full = "OUT_B64_BEGIN v1 out 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\nDONE\n"
    outputs, _ = parse_console(full)
    require_full_value_roster(cb, full, outputs)
    with pytest.raises(ValueError, match="digest-only"):
        require_full_value_roster(cb, "OUTSUM out 1 2 1234567890abcdef\nDONE\n", {})
    with pytest.raises(ValueError, match="omitted"):
        require_full_value_roster(cb, "DONE\n", {})
    with pytest.raises(ValueError, match="size"):
        require_full_value_roster(cb, full.replace("out 1 2", "out 2 1"), outputs)


def test_binary_oracle_preserves_raw_console_and_parses_text_counters(monkeypatch, tmp_path):
    from merlin.runtime.backends import base as backends
    from merlin.runtime.out_bin import parse_binary_console
    from merlin.targetgen.contract import readback_policy as readback

    raw = b"\x01\x02"
    checksum = 0xCBF29CE484222325
    for byte in raw:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    console = (
        b"METRIC cycles 7\nOUT_BIN_BEGIN v1 out 1 2 1 u 2\n" + raw + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    )
    elf = tmp_path / "model.elf"
    elf.write_bytes(b"ELF")
    backend = SimpleNamespace(
        run_elf=lambda *_args, **_kwargs: console,
        parse_output=parse_binary_console,
        ORACLE={"spike": {"kind": "spike", "derived_from_rtl": False}},
    )
    monkeypatch.setattr(backends, "get_backend", lambda _target: backend)
    monkeypatch.setattr(
        backends,
        "harness_build_recipe",
        lambda _target: SimpleNamespace(
            with_effective_abi=lambda: "recipe",
        ),
    )
    monkeypatch.setattr(compiler, "compile_lowered_to_elf", lambda *_args, **_kwargs: elf)
    monkeypatch.setattr(compiler, "simulator_provenance", lambda *_args: None)
    monkeypatch.setattr(readback, "selected_build_inputs", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(readback, "require_build_receipt", lambda *_args, **_kwargs: {"selected": True})

    result = compiler.run_on_oracle(
        _cb(),
        "llvm",
        simulator="spike",
        target="fixture",
        workdir=tmp_path,
        readback_policy=ReadbackPolicy(FULL_VALUES_BIN),
    )
    assert result["outputs"] == {"out": [[1, 2]]}
    assert result["raw_metrics"]["cycles"] == 7
    assert result["console"] == console
    assert (tmp_path / "oracle_console.bin").read_bytes() == console


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "no_reader",
        "no_engine",
        "changed_engine",
        "changed_engine_during_decode",
        "extra_kw",
        "mixed_serial",
        "mixed_digest",
        "incomplete",
        "changed_build",
        "changed_preflight",
        "partial_values",
        "unproved",
    ],
)
def test_memory_oracle_requires_prelaunch_admission_completion_and_postrun_identity(monkeypatch, tmp_path, fault):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.contract import readback_policy as readback

    elf = tmp_path / "model.elf"
    elf.write_bytes(b"ELF")
    seen = []
    console = "METRIC cycles 7\nDONE\n"
    if fault == "mixed_serial":
        console = "OUT out 1 2 1 2\n" + console
    elif fault == "mixed_digest":
        console = "OUTSUM out 1 2 0000000000000000\n" + console
    elif fault == "incomplete":
        console = "METRIC cycles 7\n"

    def run_elf(_elf, **kwargs):
        seen.append(("run", kwargs))
        return console

    def verify(*_args, **_kwargs):
        seen.append(("verify", None))
        if fault == "changed_build" and any(step == "run" for step, _ in seen):
            raise ValueError("readback build receipt changed")
        return {"selected": True, "elf_sha256": file_digest(elf)}

    def prepare(**kwargs):
        seen.append(("prepare", kwargs))
        if fault == "changed_preflight":
            elf.write_bytes(b"different ELF")
        return {
            "memory_readback": {"schema": "fixture", "elf_sha256": file_digest(elf)},
            **({"timeout": 9999} if fault == "extra_kw" else {}),
        }

    def decode(text):
        assert text == console
        seen.append(("decode", None))
        return {"out": [[1]] if fault == "partial_values" else [[1, 2]]}, {
            "schema": "fixture_admission",
            "status": "unproved" if fault == "unproved" else "complete",
        }

    def revalidate_engine():
        seen.append(("engine", None))
        changed = (fault == "changed_engine" and any(step == "run" for step, _ in seen)) or (
            fault == "changed_engine_during_decode" and any(step == "decode" for step, _ in seen)
        )
        return {"engine_sha256": "changed" if changed else "selected"}

    backend = SimpleNamespace(run_elf=run_elf, parse_output=parse_console, ORACLE={"spike": {"kind": "spike"}})
    monkeypatch.setattr(backends, "get_backend", lambda _target: backend)
    monkeypatch.setattr(
        backends, "harness_build_recipe", lambda _target: SimpleNamespace(with_effective_abi=lambda: "recipe")
    )
    monkeypatch.setattr(compiler, "compile_lowered_to_elf", lambda *_args, **_kwargs: elf)
    monkeypatch.setattr(compiler, "simulator_provenance", lambda *_args: None)
    monkeypatch.setattr(readback, "selected_build_inputs", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(readback, "require_build_receipt", verify)
    kwargs = {"memory_readback": SimpleNamespace(prepare=prepare, decode=decode)} if fault != "no_reader" else {}
    if fault != "no_engine":
        kwargs["oracle_revalidate"] = revalidate_engine
    if fault is not None:
        with pytest.raises((ValueError, RuntimeError)):
            compiler.run_on_oracle(
                _cb(),
                "llvm",
                simulator="spike",
                target="fixture",
                workdir=tmp_path,
                readback_policy=ReadbackPolicy("coherent_dump_v1"),
                **kwargs,
            )
        if fault in {"no_reader", "no_engine", "extra_kw", "changed_preflight"}:
            assert not any(step == "run" for step, _ in seen)
        if fault in {"mixed_serial", "mixed_digest", "incomplete", "changed_engine"}:
            assert not any(step == "decode" for step, _ in seen)
        return
    result = compiler.run_on_oracle(
        _cb(),
        "llvm",
        simulator="spike",
        target="fixture",
        workdir=tmp_path,
        readback_policy=ReadbackPolicy("coherent_dump_v1"),
        **kwargs,
    )
    assert result["outputs"] == {"out": [[1, 2]]}
    assert result["readback_memory"]["status"] == "complete"
    assert result["raw_metrics"] == {"cycles": 7}
    assert result["oracle"]["memory_engine"] == {"engine_sha256": "selected"}
    assert [step for step, _ in seen] == [
        "verify",
        "prepare",
        "engine",
        "run",
        "engine",
        "verify",
        "decode",
        "verify",
        "engine",
    ]
    assert (tmp_path / "oracle_console.log").read_text() == console


def test_binary_counter_projection_excludes_validated_payload_markers():
    from merlin.perf.hw_counters import parse_counter_output
    from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console

    payload = b"\x00\nMERLIN_HWCOUNTER forged 5\n\x00"
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    console = (
        f"OUT_BIN_BEGIN v1 out 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload
        + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    )
    assert parse_counter_output(console.decode("utf-8")) == {"forged": 5}
    parse_binary_console(console)
    text = binary_console_diagnostics(console).decode("utf-8")
    assert parse_counter_output(text) == {}


def test_native_simulator_adapter_forwards_policy_and_foreign_adapter_refuses(monkeypatch, tmp_path):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import capsule_runner

    policy = ReadbackPolicy(FULL_VALUES_B64)
    seen = []
    monkeypatch.setattr(backends, "get_backend", lambda _target: SimpleNamespace(available=lambda _sim: True))
    monkeypatch.setattr(
        capsule_runner.oot_compile,
        "run_on_oracle",
        lambda *_args, **kwargs: seen.append(kwargs) or {"oracle": {"kind": "spike"}},
    )
    adapter = capsule_runner.simulator_adapter("spike", "fixture")
    assert capsule_runner._with_readback_policy({"L2": adapter}, None)["L2"] is adapter
    assert capsule_runner._adapter_readback_policy({"L2": adapter}) is None
    selected_adapter = capsule_runner._with_readback_policy({"L2": adapter}, policy)["L2"]
    assert capsule_runner._adapter_readback_policy({"L2": selected_adapter}) == policy
    selected_adapter(_cb(), "llvm", tmp_path, 1)
    assert seen[-1]["readback_policy"] == policy
    with pytest.raises(NotImplementedError, match="cannot consume"):
        capsule_runner._with_readback_policy({"L3": lambda *_args: {}}, policy)
    with pytest.raises(NotImplementedError, match="no oracle"):
        capsule_runner._with_readback_policy({}, policy)
    with pytest.raises(ValueError, match="disagree"):
        capsule_runner._adapter_readback_policy({"L2": selected_adapter, "L3": adapter})
    with pytest.raises(ValueError, match="disagree"):
        capsule_runner._adapter_readback_policy({"L2": selected_adapter, "L3": lambda: None})


def test_qa_factories_carry_explicit_policy_without_changing_default(monkeypatch):
    from merlin.targetgen import capsule_runner

    policy = ReadbackPolicy(FULL_VALUES_B64)
    calls = []

    def adapters(_target, _sim_via, *, readback_policy=None):
        calls.append(readback_policy)
        return {"L2": lambda: None, "L3": lambda: None}

    monkeypatch.setattr(capsule_runner, "oracle_adapters", adapters)
    assert set(capsule_runner.qa_loop_adapters("fixture", readback_policy=policy)) == {"L2"}
    assert set(capsule_runner.qa_checkpoint_adapters("fixture", readback_policy=policy)) == {"L2", "L3"}
    assert calls == [policy, policy]

    def legacy_adapters(_target, _sim_via):
        calls.append("legacy")
        return {"L2": lambda: None}

    monkeypatch.setattr(capsule_runner, "oracle_adapters", legacy_adapters)
    assert set(capsule_runner.qa_loop_adapters("fixture")) == {"L2"}
    assert set(capsule_runner.qa_checkpoint_adapters("fixture")) == {"L2"}
    assert calls[-2:] == ["legacy", "legacy"]


def test_native_postrun_build_mutation_cannot_retain_a_numerical_pass(monkeypatch, tmp_path):
    from merlin.compile import model_execution_inputs
    from merlin.runtime import route_quality
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import bundle_harness, capsule_golden, golden_store, native_model_execution, oracle_policy
    from merlin.targetgen.contract import compile as compiler
    from merlin.targetgen.contract import readback_policy as readback

    source = tmp_path / "source"
    capture = tmp_path / "capture"
    source.mkdir()
    capture.mkdir()
    for path in (
        source / "capsule.yaml",
        source / "golden.yaml",
        capture / "model.mlir",
        capture / "weights.safetensors",
        capture / "weights.safetensors.manifest.json",
        capture / "inputs.npz",
    ):
        path.write_bytes(b"selected-neutral-input")
    cb = _cb()
    cb["kernel_abi"]["args"] = []
    monkeypatch.setattr(bundle_harness, "is_executable_emission", lambda *_args, **_kw: (True, ""))
    monkeypatch.setattr(bundle_harness, "emitted_entry_arity", lambda _text, **_kw: 0)
    monkeypatch.setattr(golden_store, "load_golden", lambda _source: {"outputs": {"out": [[1, 2]]}})
    monkeypatch.setattr(
        native_model_execution,
        "_bind_inputs",
        lambda *_args, **_kw: (
            {"arg": [[1, 2]]},
            {"source_entry_binding": {"source_owned_mutables": []}},
        ),
    )
    monkeypatch.setattr(native_model_execution, "_frozen_model_policy", lambda *_args, **_kw: {})
    monkeypatch.setattr(
        native_model_execution,
        "_host_compute_report",
        lambda *_args, **_kw: SimpleNamespace(
            to_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(route_quality, "require_clean_host_compute", lambda _report: None)
    recipe = SimpleNamespace(
        require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="entry"),
        with_effective_abi=lambda: "recipe",
    )
    service = SimpleNamespace(recipe=recipe, source_pins=(("pinned", "digest"),))
    monkeypatch.setattr(backends, "harness_build_recipe", lambda _target: recipe)
    monkeypatch.setattr(native_model_execution, "_build_service_for", lambda *_args, **_kw: service)

    def compile_selected(_cb, _lowered, work, **_kwargs):
        work.mkdir()
        (work / "kernel.o").write_bytes(b"object")
        (work / "harness.c").write_text("original\n")
        elf = work / "model.elf"
        elf.write_bytes(b"ELF")
        return elf

    monkeypatch.setattr(compiler, "compile_lowered_to_elf", compile_selected)
    monkeypatch.setattr(readback, "selected_build_inputs", lambda *_args, **_kwargs: ({"recipe": "selected"}, []))
    checks = []

    def receipt_check(_path, **kwargs):
        checks.append(kwargs["harness_path"].read_text())
        if checks[-1] != "original\n":
            raise ValueError("readback build receipt changed during native run")
        return {"schema": "merlin_readback_build_v1"}

    monkeypatch.setattr(readback, "require_build_receipt", receipt_check)
    monkeypatch.setattr(
        oracle_policy,
        "selected_l3_engine_report",
        lambda _target: {
            "available": True,
            "engine": "verilator",
        },
    )
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda *_args, **_kw: {})
    monkeypatch.setattr(
        native_model_execution,
        "_functional_engine",
        lambda _target: (_ for _ in ()).throw(
            RuntimeError("no functional probe"),
        ),
    )
    harness = tmp_path / "out" / "build" / "harness.c"

    def mutate_after_run(*_args, **_kwargs):
        harness.write_text("mutated\n")
        return "DONE\n"

    backend = SimpleNamespace(
        run_elf=mutate_after_run,
        parse_output=lambda _console: pytest.fail("numeric parsing preceded the post-run build join"),
    )
    monkeypatch.setattr(
        model_execution_inputs,
        "native_engine",
        lambda *_args, **_kw: (
            backend,
            {"engine": "verilator"},
            lambda: None,
            None,
        ),
    )
    monkeypatch.setattr(
        capsule_golden, "compare", lambda *_args, **_kw: pytest.fail("numeric comparison preceded join")
    )

    result = native_model_execution.execute_candidate_model(
        command_buffer=cb,
        lowered_mlir_text="neutral-llvm",
        capsule_dir=source,
        capture_bundle=capture,
        target="fixture",
        out_dir=tmp_path / "out",
        simulator="verilator",
        rtl_facts="selected-facts",
        board_config="selected-board",
        numeric_policy={"compare": "exact_int"},
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
    )
    assert checks == ["original\n", "mutated\n"]
    assert result["status"] == "incomplete" and result["failure"]["type"] == "ValueError"
    assert result.get("tiers", {}).get("L3", {}).get("status") != "pass"
