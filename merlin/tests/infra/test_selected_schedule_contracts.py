"""Selected RTL/software inputs drive actual configuration cleanup and emission.

Facts here are synthetic; the native machine is independent test qualification
of the shared mechanism, never support evidence for a real hardware target.
"""

import hashlib
import json
import time
from dataclasses import replace

import pytest
import yaml
from test_configuration_elimination import STORAGE, codec, command, source
from test_configuration_elimination import native as native

from merlin.compile.scheduling.program_limits import AddressWindow, ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Call, Const, evaluate
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction


@pytest.fixture
def native_machine(request):
    return request.getfixturevalue("native")


def inputs(tmp_path, width=4, endian="<", *, ordered=True, capacity=2):
    word_bits = (2 + 2 * width) * 8
    offsets = (0, 8, 16, 16 + width * 8) if endian == "<" else (word_bits - 8, word_bits - 16, width * 8, 0)
    facts = {"facts": {"target": "neutral", "word_bits": word_bits, "capacity": capacity,
                       "offsets": offsets, "widths": [8, 8, width * 8, width * 8], "codes": list(range(5))}}
    fact_path, software_path = tmp_path / "facts.json", tmp_path / "software.yaml"
    fact_path.write_text(json.dumps(facts))

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    def item(index, value=None):
        row = {"resource": "configuration", "index": index}
        if value is not None:
            row["value"] = value
        return row

    def argument(name):
        return {"argument": name}

    def constant(value):
        return {"constant": value}
    rows = {
        "set": dict(reads=[], writes=[item(argument("slot"), argument("a"))],
                    clobbers=[], ordered=True, pure_write=True),
        "work": dict(reads=[item(constant(0), constant(3)), item(constant(1), constant(2))],
                     writes=[], clobbers=[], ordered=ordered, pure_write=False),
        "reset": dict(reads=[], writes=[], clobbers=[item(argument("slot"))], ordered=True, pure_write=False),
        "pulse": dict(reads=[], writes=[item(argument("slot"), argument("a"))],
                      clobbers=[], ordered=True, pure_write=False),
    }
    instructions = []
    for code, name in enumerate(("set", "work", "reset", "pulse", "opaque")):
        operands = [dict(name=key, type=dict(kind="integer", bits=ref("widths", i), signed=False), unit="integer")
                    for i, key in enumerate(("op", "slot", "a", "b")) if i]
        fields = [dict(offset=ref("offsets", i), width=ref("widths", i),
                       **({"fixed": ref("codes", code)} if i == 0 else {"argument": key}))
                  for i, key in enumerate(("op", "slot", "a", "b"))]
        instruction = dict(name=name, operands=operands, results=[], immediates=[], encoding=fields)
        if name != "opaque":
            instruction["effects"] = dict(
                reads=["input", "configuration"] if name == "work" else [],
                writes=["output"] if name == "work" else ["configuration"],
            )
        instructions.append(instruction)
    resource = dict(name="configuration", kind="state", unit="fields")
    if capacity is not None:
        resource["capacity"] = ref("capacity")
    software = dict(schema="merlin.software_spec.v1", target="neutral", status="reviewed", isa=dict(
        schema="merlin.isa_spec.v1", namespace="neutral_isa",
        facts_sha256=hashlib.sha256(fact_path.read_bytes()).hexdigest(),
        word_bits=ref("word_bits"), instructions=instructions,
        resources=[resource, dict(name="input", kind="memory", unit="bytes"),
                   dict(name="output", kind="memory", unit="bytes")],
    ), scheduling=dict(schema="merlin.schedule_effects.v1", numerical_policy="u8-wrap", configuration=rows,
                       instruction_byteorder="little" if endian == "<" else "big"))
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    encoding, word = codec(endian, width)

    def encode(kernel):
        order = "little" if endian == "<" else "big"
        return b"".join(encode_instruction(selected.isa, call.instr, {k: evaluate(v, {}) for k, v in call.args})
                        .to_bytes(word.size, order) for call in kernel.body)

    def decode(data):
        calls = []
        for offset in range(0, len(data), word.size):
            name, values = decode_instruction(selected.isa, int.from_bytes(
                data[offset:offset + word.size], "little" if endian == "<" else "big"))
            calls.append(Call(name, tuple((k, Const(v)) for k, v in values.items())))
        return tuple(calls)

    encoding = replace(
        encoding, encode=encode, decode_calls=decode,
        program_decoder=replace(encoding.program_decoder, configuration_identity=selected.hardware_identity),
    )
    storage = replace(STORAGE, configuration_identity=selected.hardware_identity)
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    options = dict(private_ownership=True, external_windows=windows, selected_contracts=selected)
    return selected, storage, encoding, ProgramLimits(word.size * 64, 64, 1, windows, ()), options


def compile_case(model, *, enabled=True, qualified=True, kernel=None, encoding=None):
    selected, storage, original_encoding, limits, options = model
    return compile_schedule(
        source() if kernel is None else kernel, (), (StorageBank("unused", "other", 8, "bytes"),), storage,
        original_encoding if encoding is None else encoding, limits,
        budgets=ScheduleBudgets(eliminate_configuration=enabled), **options,
        configuration_qualification_identity="independent-neutral-state-machine-v1" if qualified else None,
    )


@pytest.mark.parametrize("width,endian,ordered", [(4, "<", True), (8, ">", True), (8, "<", False)])
def test_two_inputs_drive_ordinary_cleanup_and_complete_native_output(
    tmp_path, native_machine, record_property, width, endian, ordered
):
    model = inputs(tmp_path, width, endian, ordered=ordered)
    started = time.perf_counter()
    baseline = compile_case(model, enabled=False)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized = compile_case(model)
    optimized_seconds = time.perf_counter() - started
    measurements = []
    for result, expected_count in ((baseline, 30), (optimized, 12 if ordered else 30)):
        image = b"".join(result.program.segments)
        started = time.perf_counter()
        actual, counters = native_machine(image, width, endian == ">")
        execution_seconds = time.perf_counter() - started
        expected = bytes((((31 * i + 253) % 256) * 2 + 3) % 256 for i in range(67)) + bytes([91] * 5)
        assert actual == expected and counters == (expected_count, 67, 67, 3, 2, 0)
        assert result.report["final_conformance"]["actual_commands"] == expected_count
        assert result.report["selected_contracts"]["resolved_isa_identity"] == model[0].isa.identity()
        measurements.append(dict(static_bytes=len(image), actual_commands=counters[0],
                                 requested_read_bytes=counters[1], requested_write_bytes=counters[2],
                                 native_emulator_call_seconds=execution_seconds))
    for key in ("hardware", "numerical_policy", "execution", "library"):
        assert baseline.report["context"][key] == optimized.report["context"][key]
    record_property("paired_measurements", json.dumps(dict(
        schedule_compile_seconds=[baseline_seconds, optimized_seconds], peak_memory="not_measured",
        arms=measurements, scope="neutral whole transfer program; no full model or accelerator timing qualification",
    )))


def test_missing_qualification_and_unknown_capacity_keep_original_program(tmp_path, native_machine):
    for name, capacity, qualified in (("unqualified", 2, False), ("unknown", None, True)):
        root = tmp_path / name
        root.mkdir()
        result = compile_case(inputs(root, capacity=capacity), qualified=qualified)
        assert native_machine(b"".join(result.program.segments), 4, False)[1][0] == 30
        if not qualified:
            assert result.report["selected_contracts"]["configuration_pass"] == "unavailable_missing_qualification"


@pytest.mark.parametrize("which", ["facts_path", "software_path"])
def test_stale_inputs_refuse_before_emission_and_during_emission(tmp_path, which):
    model = inputs(tmp_path)
    selected = model[0]
    path = getattr(selected, which)
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model)
    path.write_bytes(original)
    def changed(kernel):
        data = model[2].encode(kernel)
        path.write_bytes(original + b"\n")
        return data
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model, encoding=replace(model[2], encode=changed))


def test_rtl_capacity_and_software_policy_must_match_actual_selection(tmp_path):
    model = inputs(tmp_path, capacity=1)
    with pytest.raises(ValueError, match="RTL state capacity"):
        compile_case(model)
    selected, storage, *rest = model
    with pytest.raises(ValueError, match="disagree"):
        compile_case((selected, replace(storage, numerical_policy_identity="different"), *rest))


@pytest.mark.parametrize(
    "fault", ["facts_pin", "foreign_target", "unknown_effects", "memory_purity", "operand", "unit",
              "unencoded", "inline_capacity"]
)
def test_mutated_specifications_do_not_authorize_cleanup(tmp_path, fault):
    selected = inputs(tmp_path)[0]
    spec = yaml.safe_load(selected.software_bytes)
    if fault == "facts_pin":
        spec["isa"]["facts_sha256"] = "0" * 64
    elif fault == "foreign_target":
        spec["target"] = "foreign"
    elif fault == "unknown_effects":
        spec["isa"]["instructions"][0].pop("effects")
    elif fault == "memory_purity":
        spec["isa"]["instructions"][0]["effects"]["writes"].append("output")
    elif fault == "operand":
        spec["scheduling"]["configuration"]["set"]["writes"][0]["value"] = {"argument": "missing"}
    elif fault == "unit":
        spec["isa"]["resources"][0]["unit"] = "bytes"
    elif fault == "inline_capacity":
        spec["isa"]["resources"][0]["capacity"] = 2
    else:
        extra = {**spec["isa"]["instructions"][0]["operands"][0], "name": "unencoded"}
        spec["isa"]["instructions"][0]["operands"].append(extra)
        spec["scheduling"]["configuration"]["set"]["writes"][0]["value"] = {"argument": "unencoded"}
    selected.software_path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        read_scheduling_contracts(selected.facts_path, selected.software_path, target="neutral")


@pytest.mark.parametrize("barrier", [command("opaque"), command("reset"), command("pulse", 1, 2)])
def test_unknown_reset_and_nonidempotent_effects_preserve_full_output_and_state(tmp_path, native_machine, barrier):
    model = inputs(tmp_path)
    body = (command("set", 0, 3), command("set", 1, 2), barrier,
            command("set", 0, 3), command("set", 1, 2), command("work", a=0, b=67))
    kernel = replace(source(), body=body)
    baseline = compile_case(model, enabled=False, kernel=kernel)
    optimized = compile_case(model, kernel=kernel)
    a = native_machine(b"".join(baseline.program.segments), 4, False)
    b = native_machine(b"".join(optimized.program.segments), 4, False)
    assert a[0] == b[0] and a[1][1:] == b[1][1:]
    assert barrier in optimized.kernel.body


def test_final_bytes_cannot_change_an_actual_register_value(tmp_path):
    model = inputs(tmp_path)
    def corrupted(kernel):
        body = list(kernel.body)
        body[0] = command("set", 0, 4)
        return model[2].encode(replace(kernel, body=tuple(body)))
    with pytest.raises(ValueError):
        compile_case(model, encoding=replace(model[2], encode=corrupted))


def test_false_execution_qualification_is_exposed_by_independent_native_execution(tmp_path, native_machine):
    model = inputs(tmp_path)
    baseline, optimized = compile_case(model, enabled=False), compile_case(model)
    # This different machine pulses on every setter. Its semantics contradict
    # the declared persistent pure write, although the structural input is valid.
    left = native_machine(b"".join(baseline.program.segments), 4, False, pulse_on_set=True)
    right = native_machine(b"".join(optimized.program.segments), 4, False, pulse_on_set=True)
    assert left[0] != right[0]
    assert optimized.report["selected_contracts"]["execution_qualification"] == "independent_evidence_required"


def test_software_change_reselects_an_alternative_and_changes_analysis_context(tmp_path):
    model = inputs(tmp_path)
    first = compile_case(model)
    selected = model[0]
    spec = yaml.safe_load(selected.software_bytes)
    spec["scheduling"]["configuration"]["work"]["ordered"] = False
    selected.software_path.write_text(yaml.safe_dump(spec))
    new = read_scheduling_contracts(selected.facts_path, selected.software_path, target="neutral")
    updated = (new, *model[1:4], {**model[4], "selected_contracts": new})
    second = compile_case(updated)
    assert len(first.kernel.body) == 12 and len(second.kernel.body) == 30
    assert first.report["context"]["hardware"] == second.report["context"]["hardware"]
    assert first.report["context"]["execution"] != second.report["context"]["execution"]


def test_cached_resolved_fields_cannot_be_replaced_independently(tmp_path):
    selected = inputs(tmp_path)[0]
    with pytest.raises(ValueError, match="init=False"):
        replace(selected, numerical_policy="invented")


def test_provider_decoder_cannot_hide_changed_actual_encoded_operands(tmp_path):
    model = inputs(tmp_path)
    def changed(kernel):
        body = list(kernel.body)
        body[0] = command("set", 0, 4)
        return model[2].encode(replace(kernel, body=tuple(body)))
    def lying_decode(image):
        calls = list(model[2].decode_calls(image))
        calls[0] = command("set", 0, 3)
        return tuple(calls)
    with pytest.raises(ValueError, match="provider decoding disagrees"):
        compile_case(model, encoding=replace(model[2], encode=changed, decode_calls=lying_decode))


def test_unknown_software_byteorder_refuses_and_wrong_declared_byteorder_fails_final_decode(tmp_path):
    model = inputs(tmp_path)
    selected = model[0]
    spec = yaml.safe_load(selected.software_bytes)
    spec["scheduling"]["instruction_byteorder"] = "unknown"
    selected.software_path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="byte order"):
        read_scheduling_contracts(selected.facts_path, selected.software_path, target="neutral")
    spec["scheduling"]["instruction_byteorder"] = "big"
    selected.software_path.write_text(yaml.safe_dump(spec))
    changed = read_scheduling_contracts(selected.facts_path, selected.software_path, target="neutral")
    updated = (changed, *model[1:4], {**model[4], "selected_contracts": changed})
    with pytest.raises(ValueError):
        compile_case(updated)


def test_rtl_width_drives_wide_values_without_truncation(tmp_path, native_machine):
    model = inputs(tmp_path, 8, ">")
    value = (1 << 40) + 17
    kernel = replace(source(), body=(command("set", 0, value),) * 64)
    for enabled, count in ((False, 64), (True, 1)):
        result = compile_case(model, enabled=enabled, kernel=kernel)
        output, observed = native_machine(b"".join(result.program.segments), 8, True)
        assert output == bytes([91] * 72) and observed == (count, 0, 0, value, 73, 0)
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    with pytest.raises(ValueError, match="outside range"):
        compile_case(inputs(narrow, 4), kernel=kernel)
