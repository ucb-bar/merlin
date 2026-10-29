"""Selected device spills, actual encoded execution and independent full state."""

import copy
import json
import time
from dataclasses import replace

import pytest
from test_private_copy_elimination import SPACES, command
from test_selected_storage_contracts import inputs as storage_inputs
from test_selected_storage_contracts import native  # noqa: F401

from merlin.compile.scheduling.program_limits import AddressWindow, DecodedInstruction, ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.compile.scheduling.spill_conformance import SpillClass
from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable, TemporaryObject
from merlin.sched.ir import Kernel, TensorArg
from merlin.semantic_compiler.allocation_types import Reservation


def emit_copy(read, write):
    return command("copy", read.start, write.start, read.size, SPACES.index(read.space), SPACES.index(write.space))


def inputs(tmp_path, *, width=8, register_slots=1, word_format="<BBBQQQ", mutate=None, segments=1,
           distinct_reload=False):
    def declare(facts, software, storage):
        alignment = 8 if width == 16 else 1
        facts.update(register_bits=width * 8, spill_alignment=alignment)
        facts["starts"][:2] = [0, 0]
        facts["alignments"][:] = [alignment, alignment]
        facts["sizes"][:2] = facts["capacities"][:2] = [register_slots * width, 3 * width]
        facts["sizes"][2] = facts["capacities"][2] = 3 * width * segments + 8
        software["scheduling"]["spills"] = dict(schema="merlin.device_spill_contract.v1", classes={
            "vector_values": dict(register_bank=SPACES[0], spill_bank=SPACES[1],
                                  width_bits={"fact": ["facts", "register_bits"]},
                                  alignment={"fact": ["facts", "spill_alignment"]},
                                  store="copy", reload="copy", execution="synchronous_bit_preserving_bytes")})
        if distinct_reload:
            facts["codes"].append(4)
            instruction = copy.deepcopy(software["isa"]["instructions"][0])
            instruction["name"] = "restore"
            instruction["encoding"][0]["fixed"] = {"fact": ["facts", "codes", 4]}
            software["isa"]["instructions"].append(instruction)
            storage["instructions"]["restore"] = copy.deepcopy(storage["instructions"]["copy"])
            software["scheduling"]["spills"]["classes"]["vector_values"]["reload"] = "restore"
        if mutate:
            mutate(facts, software, storage)

    selected, _, _, adapter, codec, _, word, _ = storage_inputs(
        tmp_path, width=width, separate=True, reserve=False, word_format=word_format, mutate=declare)
    if distinct_reload:
        original_project = adapter.project
        def project(call):
            return original_project(replace(call, instr="copy") if call.instr == "restore" else call)
        adapter = replace(adapter, project=project)
        def decode_program(data):
            result = []
            for offset in range(0, len(data), word.size):
                op, r, w, s, d, size = word.unpack_from(data, offset)
                if op not in (0, 1, 4):
                    raise ValueError("unknown actual opcode")
                effect = project(command("restore" if op == 4 else "increment" if op else "copy", s, d, size, r, w))
                result.append(DecodedInstruction(offset, word.size, effect.accesses,
                                                  completion="synchronous", relocatable=True))
            return tuple(result)
        codec = replace(codec, program_decoder=replace(codec.program_decoder, decode=decode_program))
    length = 3 * width * segments + 8
    calls, objects = [], []
    for segment in range(segments):
        addresses = [32 + 64 * (3 * segment + i) for i in range(3)]
        offsets = [(3 * segment + i) * width for i in range(3)]
        a, b, c = addresses
        x, y, z = offsets
        sizes = [width, width, max(1, width - 1)]
        objects += [TemporaryObject(f"value_{segment}_{i}", SPACES[0], address, size)
                    for i, (address, size) in enumerate(zip(addresses, sizes))]
        calls += [command("copy", x, a, width, 2, 0), command("increment", a, a, width),
                  command("copy", y, b, width, 2, 0), command("increment", b, b, width),
                  command("copy", b, y, width, 0, 3), command("copy", z, c, sizes[2], 2, 0),
                  command("copy", c, z, sizes[2], 0, 3), command("copy", a, x, width, 0, 3)]
    kernel = Kernel("device_state", (TensorArg("input", (length,), "i8", "read"),
                                     TensorArg("output", (length,), "i8", "write")), tuple(calls))
    windows = tuple(AddressWindow(w.resource, w.start, w.size)
                    for w in (*selected.storage.banks, *selected.storage.external))
    limits = ProgramLimits(word.size * 128, 128, 1, windows, ())
    return selected, kernel, tuple(objects), adapter, codec, limits, word, length


def compile_case(model, *, legacy=False, emitters=None, qualification="neutral-independent-device-spills-v1",
                 budgets=None, **kwargs):
    selected, kernel, objects, adapter, codec, limits, _, _ = model
    row = json.loads(selected.spills.declarations_json)["vector_values"]
    def selected_transfer(read, write):
        call = emit_copy(read, write)
        return replace(call, instr=row["store"] if read.space == row["register_bank"] else row["reload"])
    if legacy:
        banks, reservations, external_windows, bound = selected.storage.bind(
            selected.isa, adapter, banks=None, reservations=(), external_windows=None, private_ownership=True)
        # Same source, physical hardware, library and transfer implementation.
        # This arm exercises the existing explicit-capability entry; selected
        # binding is expected to preserve its commands, not claim a speedup.
        classes = (SpillClass(SPACES[0], SPACES[1], row["width_bits"] // 8,
                              row["alignment"], selected_transfer, qualification),)
        return compile_schedule(kernel, objects, banks, bound, codec, limits,
                                budgets=budgets or ScheduleBudgets(eliminate_copies=False),
                                spill_classes=classes, reservations=reservations, external_windows=external_windows,
                                private_ownership=True, **kwargs)
    return compile_schedule(kernel, objects, None, adapter, codec, limits,
                            selected_contracts=selected, spill_emitters=({"vector_values": selected_transfer}
                                                                       if emitters is None else emitters),
                            spill_qualification_identity=qualification,
                            budgets=budgets or ScheduleBudgets(eliminate_copies=False),
                            private_ownership=True, **kwargs)


@pytest.mark.parametrize("width,word_format", [(3, "<BBBQQQ"), (16, ">BBBIII")])
def test_selected_spills_execute_complete_outputs_and_carried_state(
    tmp_path, native, record_property, width, word_format  # noqa: F811
):
    model = inputs(tmp_path, width=width, word_format=word_format, distinct_reload=width == 16)
    selected, _, _, _, _, _, word, length = model
    results, times = [], []
    for legacy in (True, False):
        start = time.perf_counter()
        results.append(compile_case(model, legacy=legacy))
        times.append(time.perf_counter() - start)
    baseline, result = results
    assert baseline.report["library_sources_sha256"] == result.report["library_sources_sha256"]
    assert baseline.program.segments == result.program.segments
    assert result.report["selected_contracts"]["spills"]["selected_classes"] == ["vector_values"]
    assert result.report["repair_rounds"] == 1
    assert result.report["storage"]["spills"]["inserted_exact_copies"] == 2
    assert set(result.report["storage"]["repair_rounds"][0]["invalidated"]) == {
        "liveness", "allocation", "completion", "emission", "program_size"}
    data = bytearray((i * 71 + 253) % 256 for i in range(length))
    state = bytearray([91] * length)
    arms = []
    for name, compiled, elapsed in zip(("explicit_capability", "selected_contract"), results, times):
        current, output = bytearray(data), bytearray(state)
        counts = None
        for invocation in range(3):
            expected = bytearray(output)
            expected[:2 * width] = bytes((value + 1) % 256 for value in current[:2 * width])
            expected[2 * width:3 * width - 1] = current[2 * width:3 * width - 1]
            original_input = bytes(current)
            first, second = bytearray([167] * 3 * width), bytearray([199] * 3 * width)
            start = time.perf_counter()
            counts = native(b"".join(compiled.program.segments), first, second, current, output,
                            width=8 if word_format.endswith("Q") else 4,
                            big=word_format[0] == ">", separate=True)
            wall = time.perf_counter() - start
            assert output == expected and bytes(current) == original_input
            assert first[width:] == bytes([167] * 2 * width)
            assert counts[:3] == (10, 10 * width - 2, 10 * width - 2)
            assert compiled.report["final_conformance"]["requested_read_bytes"] == counts[1]
            assert compiled.report["final_conformance"]["requested_write_bytes"] == counts[2]
            current[:] = output
            current[invocation] ^= 37  # A new invocation may change input/state.
        arms.append(dict(arm=name, compile_seconds=elapsed, encoded_bytes=len(compiled.program.segments[0]),
                         commands=counts[0], read_bytes=counts[1], write_bytes=counts[2],
                         native_host_call_seconds=wall))
    record_property("selected_spill_pair", json.dumps(dict(
        width_bytes=width, measurements=arms, selected_contracts=result.report["selected_contracts"],
        shared_library_sources=result.report["library_sources_sha256"],
        compile_peak_memory="unmeasured", accelerator_timing="unmeasured",
        scope="complete neutral stateful byte graph; existing explicit vs selected capability; no speedup claim")))
    assert selected.spills is not None and word.size in (15, 27)


@pytest.mark.parametrize("fault,reason", [("qualification", "missing_execution_qualification"),
                                         ("emitter", "missing_physical_emitter"),
                                         ("width", "unknown_RTL_width_or_alignment"),
                                         ("alignment", "unknown_RTL_width_or_alignment"),
                                         ("execution", "unsupported_or_unknown_execution"),
                                         ("completion", "unsupported_transfer_completion")])
def test_unavailable_spill_routes_preserve_feasible_program_or_diagnose_pressure(tmp_path, fault, reason):
    def mutate(facts, software, storage):
        row = software["scheduling"]["spills"]["classes"]["vector_values"]
        if fault in ("width", "alignment"):
            row["width_bits" if fault == "width" else "alignment"] = None
        if fault == "execution":
            row["execution"] = "unknown"
        if fault == "completion":
            row["store"] = row["reload"] = "async_copy"
            storage["instructions"]["async_copy"]["exact_copy"] = True
    extra = (dict(qualification=None) if fault == "qualification" else
             dict(emitters={}) if fault == "emitter" else {})
    tmp_path.joinpath("feasible").mkdir()
    feasible = inputs(tmp_path / "feasible", register_slots=2, mutate=mutate)
    result = compile_case(feasible, **extra)
    assert result.report["repair_rounds"] == 0
    assert result.report["selected_contracts"]["spills"]["unavailable_classes"] == {"vector_values": reason}
    tmp_path.joinpath("pressure").mkdir()
    with pytest.raises(StorageOptimizationUnavailable, match=reason):
        compile_case(inputs(tmp_path / "pressure", mutate=mutate), **extra)


@pytest.mark.parametrize("fault", ["no_storage", "alias", "duplicate", "subbyte", "constant_width",
                                  "direction", "not_copy", "extra_effect", "unknown_instruction"])
def test_invalid_selected_spill_contracts_reject_instead_of_inferring_support(tmp_path, fault):
    def mutate(facts, software, storage):
        section = software["scheduling"]["spills"]
        row = section["classes"]["vector_values"]
        if fault == "no_storage":
            del software["scheduling"]["storage"]
        elif fault == "alias":
            storage["banks"][SPACES[1]]["backing"] = storage["banks"][SPACES[0]]["backing"]
        elif fault == "duplicate":
            section["classes"]["another"] = dict(row)
        elif fault == "subbyte":
            facts["register_bits"] = 7
        elif fault == "constant_width":
            row["width_bits"] = 64
        elif fault == "direction":
            access = storage["instructions"]["copy"]["accesses"][1]
            access["space"]["choices"] = [c for c in access["space"]["choices"] if c["resource"] != SPACES[1]]
            software["isa"]["instructions"][0]["effects"]["writes"].remove(SPACES[1])
        elif fault == "not_copy":
            row["store"] = "increment"
        elif fault == "extra_effect":
            software["isa"]["resources"].append(dict(name="queue", kind="execution", unit="entries"))
            software["isa"]["instructions"][0]["effects"]["writes"].append("queue")
        else:
            row["reload"] = "absent"
    with pytest.raises(ValueError):
        inputs(tmp_path, mutate=mutate)


@pytest.mark.parametrize("fault", ["route", "size", "direction"])
def test_actual_spill_emitter_must_match_selected_instructions_and_operands(tmp_path, fault):
    def wrong(read, write):
        call = emit_copy(read, write)
        if fault == "route":
            return replace(call, instr="increment")
        return command("copy", read.start, write.start, read.size - (fault == "size"),
                       SPACES.index(write.space) if fault == "direction" else SPACES.index(read.space),
                       SPACES.index(read.space) if fault == "direction" else SPACES.index(write.space))
    with pytest.raises(ValueError, match="spill"):
        compile_case(inputs(tmp_path), emitters={"vector_values": wrong})


def test_spill_repair_obeys_capacity_reservations_and_final_program_limits(tmp_path):
    model = inputs(tmp_path, segments=2)
    result = compile_case(model, budgets=ScheduleBudgets(eliminate_copies=False, max_repairs=2))
    assert result.report["repair_rounds"] == 2
    assert result.report["storage"]["spills"]["inserted_exact_copies"] == 4
    for budgets in (ScheduleBudgets(eliminate_copies=False, max_repairs=1),
                    ScheduleBudgets(eliminate_copies=False, max_spills=0)):
        with pytest.raises((ValueError, StorageOptimizationUnavailable)):
            compile_case(model, budgets=budgets)
    narrow = (*model[:5], replace(model[5], max_instructions=16), *model[6:])
    with pytest.raises(ValueError, match="budget"):
        compile_case(narrow)
    with pytest.raises(StorageOptimizationUnavailable):
        compile_case(model, reservations=(Reservation(SPACES[1], 0, 3 * 8),))


def test_spill_binding_does_not_accept_unselected_emitters_or_capability_overrides(tmp_path):
    model = inputs(tmp_path)
    with pytest.raises(ValueError, match="declared classes"):
        compile_case(model, emitters={"guessed": emit_copy})
    capability = SpillClass(SPACES[0], SPACES[1], 8, 1, emit_copy, "unselected")
    with pytest.raises(ValueError, match="overridden"):
        compile_case(model, spill_classes=(capability,))


def test_changed_software_during_spill_emission_is_stale(tmp_path):
    model = inputs(tmp_path)
    def emit(read, write):
        model[0].software_path.write_bytes(model[0].software_bytes + b"\n")
        return emit_copy(read, write)
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model, emitters={"vector_values": emit})


def test_qualification_label_does_not_prove_machine_copy_semantics(tmp_path, native):  # noqa: F811
    model = inputs(tmp_path)
    result = compile_case(model)
    initial = bytearray(range(model[-1]))
    expected = bytearray([91] * len(initial))
    expected[:16] = bytes((value + 1) % 256 for value in initial[:16])
    expected[16:23] = initial[16:23]
    outcomes = []
    for fault in (False, True):
        output = bytearray([91] * len(initial))
        counts = native(result.program.segments[0], bytearray(24), bytearray(24), bytearray(initial), output,
                        separate=True, faulty_spill=fault)
        assert counts[:3] == (10, 78, 78)
        outcomes.append(output)
    assert outcomes[0] == expected
    assert outcomes[1] != expected  # Same declarations/counters; independent execution disproves qualification.


@pytest.mark.parametrize("bank", [0, 1])
def test_selected_capacity_below_one_register_cannot_be_repaired(tmp_path, bank):
    def mutate(facts, software, storage):
        facts["sizes"][bank] = facts["capacities"][bank] = 7
    with pytest.raises(StorageOptimizationUnavailable):
        compile_case(inputs(tmp_path, mutate=mutate))


def test_final_spill_word_mutation_is_redecoded_independently(tmp_path):
    model = inputs(tmp_path)
    codec, word = model[4], model[6]
    original = codec.encode
    def corrupt(kernel):
        image = bytearray(original(kernel))
        for index, call in enumerate(kernel.body):
            if call.arg("read_bank").value == 0 and call.arg("write_bank").value == 1:
                image[index * word.size] = 1  # Actual increment instead of exact spill.
                break
        return bytes(image)
    broken = (*model[:4], replace(codec, encode=corrupt), *model[5:])
    with pytest.raises(ValueError, match="differ|disagrees"):
        compile_case(broken)


def test_stricter_spill_alignment_replans_the_saved_register_too(tmp_path, native):  # noqa: F811
    def mutate(facts, software, storage):
        facts["spill_alignment"] = 16
        facts["starts"][:2] = [8, 8]
        facts["sizes"][:2] = [24, 24]
        facts["capacities"][:2] = [32, 32]
    model = inputs(tmp_path, width=16, mutate=mutate)
    result = compile_case(model)
    for call in result.kernel.body:
        if {call.arg("read_bank").value, call.arg("write_bank").value} == {0, 1}:
            assert call.arg("source").value % 16 == call.arg("destination").value % 16 == 0
    data, output = bytearray(range(model[-1])), bytearray([91] * model[-1])
    first, second = bytearray([167] * 48), bytearray([199] * 48)
    assert native(result.program.segments[0], first, second, data, output, separate=True)[:3] == (10, 158, 158)
    expected = bytearray([91] * model[-1])
    expected[:32] = bytes(value + 1 for value in data[:32])
    expected[32:47] = data[32:47]
    assert output == expected
    assert first[:8] == bytes([167] * 8) and second[:8] == bytes([199] * 8)
    assert first[32:] == bytes([167] * 16) and second[32:] == bytes([199] * 16)
