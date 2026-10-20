"""Original data-bound layout checks; no synthetic target is hardware-qualified."""

import copy
import hashlib
import json
import shutil
import struct
import subprocess
from types import SimpleNamespace

import pytest
import test_bound_chipyard_rocc_backend as bound_tests
import test_coherent_output_dump as dump_tests
from merlin_experiments.phase1.feedback import caller_layout as C
from merlin_experiments.phase1.feedback import native_output_readback as R

from merlin.runtime import bound_caller_layout as L
from merlin.runtime import harness_render
from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import bind
from merlin.targetgen import plugins, target_registry

selection = bound_tests.selection


@pytest.fixture
def bound_layout(tmp_path, monkeypatch, selection):
    contract, facts = selection
    contract_path = tmp_path / "selected-contract.json"
    contract_path.write_text(json.dumps(contract))
    facts_path = tmp_path / "selected-facts.json"
    facts_path.write_text(json.dumps(facts))
    backend = bind(target="synthetic", contract=contract, facts=facts)
    info = SimpleNamespace(capability_contract_path=contract_path,
                           load_contract=lambda: json.loads(contract_path.read_bytes()))
    monkeypatch.setattr(target_registry, "resolve", lambda target: info)
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(plugins, "resolve_support", lambda target: pytest.fail("data runtime discovered an OOT hook"))
    cb = dict(target="synthetic", commands=[], tensors={
        "A": dict(shape=[2, 3], dtype="i8", role="input"),
        "Y": dict(shape=[2, 3], dtype="i32", role="output")}, kernel_abi=dict(
            kind="whole_program", args=[dict(tensor="A", access="read"), dict(tensor="Y", access="write")],
            outputs=["Y"]))
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    return SimpleNamespace(backend=backend, cb=cb, facts=facts, contract=contract, contract_path=contract_path,
                           facts_path=facts_path, submission=submission, monkeypatch=monkeypatch)


def test_exact_bound_runtime_uses_original_dense_renderer_without_oot_callbacks(bound_layout):
    fixture = bound_layout
    receipt = C.inspect_caller_layout(submission=fixture.submission, command_buffer_member="command_buffer.json",
                                      target="synthetic", facts_path=fixture.facts_path)
    assert receipt["schema"] == "public_caller_layout_receipt_v2"
    assert receipt["policy"] == dict(mode="logical_dense_row_major_v2")
    assert receipt["tensors"] == [dict(tensor=name, symbol=f"tensor_{ordinal}", dtype=dtype,
                                       logical_shape=[2, 3], physical_extents=[2, 3],
                                       logical_strides_elements=[3, 1], storage_elements=6, offset_elements=0)
                                  for ordinal, (name, dtype) in enumerate((("A", "i8"), ("Y", "i32")))]
    assert receipt["status"] == "layout_only" and "timing authority" in receipt["scope"]
    assert len(receipt["runtime_selection_sha256"]) == len(receipt["selected_contract_file_sha256"]) == 64
    assert not any(key in receipt for key in ("values", "expected", "all_pass", "qualification"))


@pytest.mark.parametrize("mutation", ["facts", "contract", "runtime", "fake_type", "other_target"])
def test_original_binding_is_reselected_and_fake_or_stale_wrappers_refuse(bound_layout, mutation):
    fixture = bound_layout
    backend, target, facts = fixture.backend, "synthetic", copy.deepcopy(fixture.facts)
    if mutation == "facts":
        facts["facts"]["source"]["config"] = "foreign"
    elif mutation == "contract":
        fixture.contract["logical_harness"] = dict(storage_budget_bytes=10)
        fixture.contract_path.write_text(json.dumps(fixture.contract))
    elif mutation == "runtime":
        changed = copy.deepcopy(fixture.contract)
        changed["logical_harness"] = dict(storage_budget_bytes=1024)
        fixture.monkeypatch.setattr(base, "get_backend", lambda target: bind(
            target="synthetic", contract=changed, facts=fixture.facts))
    elif mutation == "fake_type":
        backend = SimpleNamespace(target=target, _selection_json=backend._selection_json,
                                  __file__=backend.__file__)
    else:
        target = "other"
    with pytest.raises(ValueError, match="bound|original|selected"):
        L.inspect(backend, fixture.cb, target=target, facts=facts)


@pytest.mark.parametrize("mutation", ["padding", "strides", "transformed", "budget", "pointer_order", "scalar",
                                      "byte_order", "alignment"])
def test_new_layout_path_preserves_original_storage_and_pointer_obligations(bound_layout, mutation):
    fixture = bound_layout
    cb = copy.deepcopy(fixture.cb)
    if mutation == "padding":
        cb["tensors"]["Y"]["storage_shape"] = [2, 4]
    elif mutation == "strides":
        cb["tensors"]["Y"]["strides"] = [1, 2]
    elif mutation == "transformed":
        cb["params"] = dict(storage_encodings={})
    elif mutation in {"budget", "byte_order", "alignment"}:
        if mutation == "budget":
            fixture.contract["logical_harness"] = dict(storage_budget_bytes=10)
        else:
            fixture.contract["harness_abi"][mutation if mutation == "byte_order" else "tensor_alignment"] = (
                "big" if mutation == "byte_order" else 1)
        fixture.contract_path.write_text(json.dumps(fixture.contract))
        fixture.backend = bind(target="synthetic", contract=fixture.contract, facts=fixture.facts)
        fixture.monkeypatch.setattr(base, "get_backend", lambda target: fixture.backend)
    elif mutation == "pointer_order":
        cb["kernel_abi"]["args"][0]["tensor"] = "Y"
    else:
        cb["tensors"]["Y"]["shape"] = []
    with pytest.raises(ValueError):
        L.inspect(fixture.backend, cb, target="synthetic", facts=fixture.facts)


def test_source_or_command_mutation_during_the_actual_query_refuses(bound_layout, monkeypatch):
    fixture = bound_layout
    original = harness_render._roster

    def mutate(cb):
        result = original(cb)
        path = fixture.submission / "command_buffer.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(harness_render, "_roster", mutate)
    with pytest.raises(ValueError, match="changed"):
        C.inspect_caller_layout(submission=fixture.submission, command_buffer_member="command_buffer.json",
                               target="synthetic", facts_path=fixture.facts_path)


def test_logical_v2_policy_cannot_reinterpret_original_v1_projection(bound_layout):
    fixture = bound_layout
    layout = L.inspect(fixture.backend, fixture.cb, target="synthetic", facts=fixture.facts)["layout"]
    layout["schema"] = "caller_storage_layout_v1"
    with pytest.raises(ValueError, match="unsupported"):
        C._checked_projection(layout, fixture.cb)


@pytest.mark.parametrize("field,value", [("logical_strides_elements", [1, 2]), ("offset_elements", 1),
                                       ("physical_extents", [2, 4]), ("storage_elements", 8),
                                       ("symbol", "tensor_1")])
def test_bound_projection_reconstructs_exact_dense_extents_and_offsets(bound_layout, field, value):
    fixture = bound_layout
    layout = L.inspect(fixture.backend, fixture.cb, target="synthetic", facts=fixture.facts)["layout"]
    layout["tensors"][0][field] = value
    with pytest.raises(ValueError, match="bounds|dense|symbol"):
        C._checked_projection(layout, fixture.cb)


def test_bound_symbols_and_pointer_order_match_the_actual_original_renderer(bound_layout):
    fixture = bound_layout
    result = L.inspect(fixture.backend, fixture.cb, target="synthetic", facts=fixture.facts)
    text = fixture.backend.render_harness(fixture.cb, target="synthetic", inputs={"A": [[-3, 2, -1], [2, -1, 3]]})
    for row, extent in zip(result["layout"]["tensors"], (6, 24), strict=True):
        assert f'static unsigned char {row["symbol"]}[{extent}]' in text
    assert "entry(tensor_0, tensor_1);" in text


def test_bound_v2_rechecks_actual_sized_elf_symbol_and_all_signed_dump_values(bound_layout, tmp_path):
    fixture = bound_layout
    cc = shutil.which("cc")
    assert cc, "independent writable-symbol check requires the existing host compiler"
    source = tmp_path / "actual-output.c"
    source.write_text("#include <stdint.h>\nstatic int32_t tensor_1[6] __attribute__((used));\n"
                      "int main(void){return 0;}\n")
    elf = tmp_path / "actual-output.elf"
    subprocess.run([cc, "-O0", "-no-pie", str(source), "-o", str(elf)], check=True, capture_output=True)
    arguments = dict(submission=fixture.submission, command_buffer_member="command_buffer.json",
                     target="synthetic", facts_path=fixture.facts_path, elf_path=elf)
    admission = R.admit_coherent_output_layout(
        **arguments, expected_elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest())
    assert admission["schema"] == "coherent_output_layout_admission_v2"
    assert admission["outputs"][0]["symbol"] == "tensor_1" and admission["outputs"][0]["bytes"] == 24
    fixed = R.admit_fixed_address_output_layout(admission=admission, **arguments)
    expected = [-2**31, -17, -1, 0, 7, 2**31 - 1]
    dump = dump_tests._dump(tmp_path / "output.dump", address=admission["outputs"][0]["address"],
                            payload=struct.pack("<6i", *expected))
    actual = R.decode_coherent_output_dump(admission=admission, fixed_address_admission=fixed,
                                           dump_path=dump, **arguments)
    assert actual["outputs"]["Y"] == [expected[:3], expected[3:]]
    stale = {**admission, "schema": "coherent_output_layout_admission_v1"}
    with pytest.raises(ValueError, match="changed"):
        R.decode_coherent_output_dump(admission=stale, fixed_address_admission=fixed,
                                      dump_path=dump, **arguments)
