"""Explicit pointer diagnostics use original build/full-value execution boundaries."""

import base64
import copy
import json
import struct
from types import SimpleNamespace

import pytest
import yaml
from test_native_pointer_source_binding import frozen  # noqa: F401

from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import native_pointer_source_execution as E
from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy


@pytest.fixture
def execution(request, tmp_path, monkeypatch):
    root, capture, cb, a, b = request.getfixturevalue("frozen")
    contract = tmp_path / "target.yaml"
    contract.write_text(
        yaml.safe_dump(
            dict(
                name="neutral",
                harness_abi=dict(
                    kind="logical_pointer",
                    version=2,
                    byte_order="little",
                    entry_symbol="selected_entry",
                    fence_symbol="selected_done",
                    main_convention="void",
                    tensor_alignment=64,
                ),
            )
        )
    )
    service = SimpleNamespace(
        recipe=SimpleNamespace(require_kernel_stack_frame=lambda: KernelStackFramePolicy("selected_entry", 8192)),
        source_pins=((str(contract), "pinned-source-for-neutral-test"),),
        verify=lambda _: None,
    )
    monkeypatch.setattr(E.original, "_build_service_for", lambda _: service)
    monkeypatch.setattr(
        E.original, "_frozen_model_policy", lambda *_, **__: dict(required_tiers=["L0", "L1", "L2", "L3"])
    )
    monkeypatch.setattr(
        "merlin.compile.model_execution_inputs.selected_firrtl",
        lambda *_, **__: dict(path="selected-facts", config="selected-config"),
    )
    monkeypatch.setattr(
        "merlin.targetgen.target_registry.resolve", lambda _: SimpleNamespace(capability_contract_path=contract)
    )
    compiled, ran = [], []

    def compile_original(buffer, lowered, destination, **kwargs):
        assert buffer == cb and "kernel_abi" not in buffer
        assert kwargs["inputs"]["left"]["values"] == a.reshape(-1).tolist()
        assert kwargs["_build_service"] is service
        destination.mkdir()
        (destination / "harness.c").write_text("original selected logical harness")
        elf = destination / "kernel.elf"
        elf.write_bytes(b"independently selected test ELF bytes")
        compiled.append(elf)
        return elf

    monkeypatch.setattr(E, "compile_lowered_to_elf", compile_original)
    build_record = ({"original_recipe": True}, [{"original_build_source": True}], {"original_build_receipt": True})
    monkeypatch.setattr("merlin.targetgen.native_readback.verify_build", lambda **_: build_record)
    value = (a.astype("int32") @ b.astype("int32")).reshape(-1).tolist()
    data = base64.b64encode(struct.pack("<4i", *value)).decode()
    console = f"OUT_B64_BEGIN v1 result 2 2 4 s\nOUT_B64_CHUNK 00000000 0010 {data}\nOUT_B64_END\n"

    class Backend:
        def parse_output(self, text):
            actual, decoder = {}, OutB64Decoder()
            for line in text.splitlines():
                decoder.consume(line.split(), actual)
            decoder.require_closed()
            return actual, {}

    backend = Backend()
    monkeypatch.setattr(
        E.original, "_functional_engine", lambda _: (backend, {"engine": "spike", "target": "neutral"}, lambda: None)
    )

    def observe_original(actual_backend, elf, **kwargs):
        assert actual_backend is backend and elf == compiled[0]
        assert kwargs["simulator"] == "spike"
        ran.append(elf)
        return console, {"path": "neutral-original-execution-observation", "sha256": "not-an-admission"}

    monkeypatch.setattr(E, "observed_execution", observe_original)
    lowered = (
        "builtin.module { llvm.func @selected_entry(%a: !llvm.ptr, %b: !llvm.ptr, %y: !llvm.ptr) { llvm.return } }"
    )
    args = dict(
        command_buffer=cb,
        lowered_mlir_text=lowered,
        capsule_dir=root,
        capture_bundle=capture,
        target="neutral",
        out_dir=tmp_path / "execution",
        simulator="spike",
        timeout=30,
        oracle_tier_ceiling="L2",
        complete_call_timing=False,
        rtl_facts=tmp_path / "facts.json",
        board_config="selected-config",
    )
    return args, compiled, ran, observe_original, contract


def test_explicit_pointer_mode_uses_unchanged_candidate_and_complete_outputs(execution):
    args, compiled, ran, _, _ = execution
    cb = copy.deepcopy(args["command_buffer"])
    record = E.execute_candidate_pointer_source(**args)
    assert record.get("failure") is None
    assert record["status"] == "numeric_match_diagnostic"
    assert record["numeric"]["status"] == "pass"
    assert len(compiled) == len(ran) == 1
    assert record["pointer_execution"]["entry_symbol"] == "selected_entry"
    assert record["pointer_execution"]["completion_symbol"] == "selected_done"
    assert record["source_execution_protocol"] == "merlin.native_pointer_source.v1"
    assert record["hardware_execution_qualified"] is False
    assert record["timing_authority"] == "NONE"
    assert args["command_buffer"] == cb and "kernel_abi" not in cb
    assert json.loads((args["out_dir"] / "result.json").read_text())["numeric"]["status"] == "pass"


def test_prospective_intake_uses_streamed_inputs_and_original_complete_output_boundary(execution, monkeypatch):
    import numpy as np

    from merlin.runtime.tensor_payload import TensorPayload
    from merlin.targetgen.native_pointer_source_binding import LARGE_INTAKE_PROTOCOL

    args, compiled, ran, _, _ = execution
    original_service = E.original._build_service_for
    original_compile = E.compile_lowered_to_elf
    selections = []

    def service(target, **kwargs):
        selections.append(kwargs)
        return original_service(target)

    def compile_streamed(buffer, lowered, destination, **kwargs):
        assert all(type(value) is TensorPayload for value in kwargs["inputs"].values())
        inputs = {name: dict(shape=list(value.shape), values=np.frombuffer(
            b"".join(value.payload.chunks()), dtype=np.int8).astype(np.int64).tolist())
            for name, value in kwargs["inputs"].items()}
        return original_compile(buffer, lowered, destination, **(kwargs | {"inputs": inputs}))

    monkeypatch.setattr(E.original, "_build_service_for", service)
    monkeypatch.setattr(E, "compile_lowered_to_elf", compile_streamed)
    record = E.execute_candidate_pointer_source(**args, source_intake_protocol=LARGE_INTAKE_PROTOCOL)
    assert record.get("failure") is None and record["numeric"]["status"] == "pass"
    assert record["source_intake"]["protocol"] == LARGE_INTAKE_PROTOCOL
    assert selections == [dict(source_intake_protocol=LARGE_INTAKE_PROTOCOL)]
    assert len(compiled) == len(ran) == 1


@pytest.mark.parametrize(
    "change",
    [
        dict(simulator="gsim"),
        dict(oracle_tier_ceiling="L3"),
        dict(complete_call_timing=True),
        dict(input_storage="streamed"),
        dict(timeout=True),
        dict(timeout=0),
        dict(timeout=601),
        dict(timeout=900),
        dict(numeric_policy={"compare": "tolerance_float", "atol": 1}),
    ],
)
def test_unsupported_mode_or_numeric_override_never_reaches_build(execution, change):
    args, compiled, ran, _, _ = execution
    record = E.execute_candidate_pointer_source(**(args | change))
    assert record.get("failure") and not compiled and not ran
    assert "numeric" not in record


def test_native_leaf_accepts_exact_original_timeout_ceiling(execution):
    args, compiled, ran, _, _ = execution
    record = E.execute_candidate_pointer_source(**(args | {"timeout": E.MAX_TIMEOUT_SECONDS}))
    assert E.MAX_TIMEOUT_SECONDS == 600
    assert record["status"] == "numeric_match_diagnostic"
    assert record["numeric"]["status"] == "pass"
    assert len(compiled) == len(ran) == 1


def test_mutation_during_execution_cannot_retain_numerical_pass(execution, monkeypatch):
    args, _, _, observe, _ = execution

    def mutate(*a, **kw):
        result = observe(*a, **kw)
        (args["capsule_dir"] / "source.mlir").write_text("changed source")
        return result

    monkeypatch.setattr(E, "observed_execution", mutate)
    record = E.execute_candidate_pointer_source(**args)
    assert record.get("failure") and "namespace changed" in record["failure"]["detail"]
    assert "numeric" not in record


def test_incomplete_full_value_frame_cannot_match_independent_reference(execution, monkeypatch):
    args, _, _, _, _ = execution
    monkeypatch.setattr(E, "observed_execution", lambda *_, **__: ("OUT_B64_BEGIN v1 result 2 2 4 s\n", {}))
    record = E.execute_candidate_pointer_source(**args)
    assert record.get("failure") and "numeric" not in record


def test_changed_pointer_entry_arity_is_refused_before_build(execution):
    args, compiled, ran, _, _ = execution
    args["lowered_mlir_text"] = "builtin.module { llvm.func @selected_entry(%a: !llvm.ptr) { llvm.return } }"
    record = E.execute_candidate_pointer_source(**args)
    assert record.get("failure") and not compiled and not ran


def test_grader_imports_are_withheld():
    from merlin.common.access import declared_modules

    for name in ("merlin.targetgen.native_pointer_source_binding", "merlin.targetgen.native_pointer_source_execution"):
        assert name in declared_modules("grader")


def test_changed_harness_contract_after_build_refuses_before_execution(execution, monkeypatch):
    args, _, ran, _, contract = execution
    compile_original = E.compile_lowered_to_elf

    def mutate(*a, **kw):
        elf = compile_original(*a, **kw)
        contract.write_text(contract.read_text() + "\nchanged_selection: true\n")
        return elf

    monkeypatch.setattr(E, "compile_lowered_to_elf", mutate)
    record = E.execute_candidate_pointer_source(**args)
    assert record.get("failure") and "contract changed" in record["failure"]["detail"]
    assert not ran and "numeric" not in record
