"""Observe real encoded setup removal, complete outputs and persistent state."""

import ctypes
import shutil
import struct
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.scheduling.configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    ConfigurationUnavailable,
    eliminate_redundant_configuration,
    verify_configuration_rewrite,
    verify_configuration_state,
)
from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    Continuation,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding, compile_schedule
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate, loop
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank


def command(op, slot=0, a=0, b=0):
    return Call(op, tuple((key, Const(value)) for key, value in (("slot", slot), ("a", a), ("b", b))))


def configuration(call, *, ordered_work=True):
    slot, a, _ = (evaluate(call.arg(key), {}) for key in ("slot", "a", "b"))
    if call.instr == "set":
        return ConfigurationEffects(writes=((f"field{slot}", a),), ordered=True, pure_write=True)
    if call.instr == "work":
        return ConfigurationEffects(reads=(("field0", 3), ("field1", 2)), ordered=ordered_work)
    if call.instr == "reset":
        return ConfigurationEffects(clobbers=(f"field{slot}",), ordered=True)
    if call.instr == "pulse":
        return ConfigurationEffects(writes=((f"field{slot}", a),), ordered=True)
    return None


def adapter(ordered_work=True):
    return ConfigurationAdapter(
        lambda call: configuration(call, ordered_work=ordered_work), "neutral-state-v1", "synthetic"
    )


def storage(call):
    a, b = (evaluate(call.arg(key), {}) for key in ("a", "b"))
    accesses = (EncodedAccess("input", a, b), EncodedAccess("output", a, b, True)) if call.instr == "work" else ()
    return StorageEffects(repr((call.instr, call.args)), accesses, "synchronous")


def rewrite(call, accesses):
    assert accesses == storage(call).accesses
    return call


STORAGE = StorageAdapter(storage, rewrite, "neutral-state-v1", "u8-wrap", "synthetic")
OPS = ("set", "work", "reset", "pulse", "opaque")


def codec(endian, integer_bytes):
    word = struct.Struct(endian + "BB" + ("I" if integer_bytes == 4 else "Q") * 2)

    def encode(kernel):
        return b"".join(
            word.pack(OPS.index(c.instr), *(evaluate(c.arg(k), {}) for k in ("slot", "a", "b"))) for c in kernel.body
        )

    def decode(data):
        return tuple(command(OPS[op], slot, a, b) for op, slot, a, b in word.iter_unpack(data))

    def decode_program(data):
        return tuple(
            DecodedInstruction(
                i * word.size, word.size, storage(c).accesses, completion="synchronous", relocatable=True
            )
            for i, c in enumerate(decode(data))
        )

    return ScheduleEncoding(
        encode, decode, ProgramDecoder(decode_program, "neutral-state-v1", "synthetic"), "u8-wrap", "synthetic"
    ), word


def source(length=67, chunk=7):
    body = []
    for offset in range(0, length, chunk):
        body.extend(
            (command("set", 0, 3), command("set", 1, 2), command("work", a=offset, b=min(chunk, length - offset)))
        )
    return Kernel(
        "whole_program",
        (TensorArg("input", (length + 5,), "i8", "read"), TensorArg("output", (length + 5,), "i8", "write")),
        tuple(body),
    )


_EXECUTOR = r"""
#include <stdint.h>
#include <stddef.h>
static uint64_t number(const uint8_t *p,int n,int big) {
 uint64_t v=0;for(int i=0;i<n;i++)v|=(uint64_t)p[i]<<(8*(big?n-1-i:i));return v;
}
int execute(const uint8_t *image,size_t length,int width,int big,const uint8_t *input,
            uint8_t *output,size_t capacity,uint64_t *observation,int pulse_on_set) {
 uint64_t state[2]={91,73},commands=0,reads=0,writes=0,pulses=0;
 size_t stride=2+2*width;if(length%stride)return 1;
 for(size_t cursor=0;cursor<length;cursor+=stride) {
  const uint8_t *p=image+cursor;unsigned op=p[0],slot=p[1];
  uint64_t a=number(p+2,width,big),b=number(p+2+width,width,big);commands++;
  if(op==0){if(slot>=2)return 2;state[slot]=a;pulses+=pulse_on_set;}
  else if(op==1){if(a>capacity||b>capacity-a)return 3;
   for(uint64_t j=a;j<a+b;j++) output[j]=(uint8_t)(input[j]*state[1]+state[0]+pulses);
   reads+=b;writes+=b;
  } else if(op==2){if(slot>=2)return 4;state[slot]=0;}
  else if(op==3){if(slot>=2)return 5;state[slot]=a;pulses++;}
  else if(op==4){state[0]=17;state[1]=19;}
  else return 6;
 }
 observation[0]=commands;observation[1]=reads;observation[2]=writes;
 observation[3]=state[0];observation[4]=state[1];observation[5]=pulses;return 0;
}
"""


@pytest.fixture
def native(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    src, lib = tmp_path / "execute.c", tmp_path / "execute.so"
    src.write_text(_EXECUTOR)
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(src), "-o", str(lib)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(lib)).execute
    bytep = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        bytep,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        bytep,
        bytep,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]

    def execute(image, width, big, *, pulse_on_set=False):
        inputs = bytes((31 * i + 253) % 256 for i in range(72))
        input_buffer = (ctypes.c_uint8 * 72).from_buffer_copy(inputs)
        output = (ctypes.c_uint8 * 72)(*([91] * 72))
        encoded = (ctypes.c_uint8 * len(image)).from_buffer_copy(image)
        counts = (ctypes.c_uint64 * 6)()
        assert fn(encoded, len(image), width, big, input_buffer, output, 72, counts, pulse_on_set) == 0
        assert bytes(input_buffer) == inputs
        return bytes(output), tuple(counts)

    return execute


@pytest.mark.parametrize("endian,width,ordered", [("<", 4, True), (">", 8, True), ("<", 8, False)])
def test_ordinary_pipeline_removes_actual_setup_and_executes(endian, width, ordered, native):
    original = source()
    encoding, word = codec(endian, width)
    banks = (StorageBank("unused", "separate", 8, "bytes"),)
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    limits = ProgramLimits(64 * word.size, 64, 1, windows, ())
    results = []
    for enabled in (False, True):
        result = compile_schedule(
            original,
            (),
            banks,
            STORAGE,
            encoding,
            limits,
            configuration_adapter=adapter(ordered),
            budgets=ScheduleBudgets(eliminate_configuration=enabled),
            private_ownership=True,
            external_windows=windows,
        )
        image = b"".join(result.program.segments)
        output, counts = native(image, width, endian == ">")
        expected = bytes((((31 * i + 253) % 256) * 2 + 3) % 256 for i in range(67)) + bytes([91] * 5)
        assert output == expected
        assert counts[1:] == (67, 67, 3, 2, 0)
        assert counts[0] == (12 if enabled and ordered else 30)
        assert len(image) == counts[0] * word.size
        assert result.report["final_conformance"]["actual_commands"] == counts[0]
        assert result.report["storage"]["configuration"]["removed_calls"] == 30 - counts[0]
        assert result.report["final_conformance"]["requested_read_bytes"] == 67
        assert [p["name"] for p in result.report["passes"]][:3] == ["semantic_admission", "configuration", "storage"]
        results.append(result)
    assert results[0].report["context"]["library"] == results[1].report["context"]["library"]


@pytest.mark.parametrize("barrier", [command("reset"), command("opaque"), command("pulse", 0, 7)])
def test_resets_unknown_effects_and_nonidempotent_writes_are_retained(barrier):
    kernel = replace(source(), body=(command("set", 0, 3), barrier, command("set", 0, 3)))
    result = eliminate_redundant_configuration(kernel, adapter())
    assert result.kernel == kernel


def test_partial_register_change_preserves_whole_write_and_final_state():
    pair = ConfigurationAdapter(
        lambda c: ConfigurationEffects(
            writes=(("a", evaluate(c.arg("a"), {})), ("b", evaluate(c.arg("b"), {}))), ordered=True, pure_write=True
        ),
        "multi",
        "synthetic",
    )
    k = replace(source(), body=(command("set", a=3, b=4), command("set", a=3, b=5), command("set", a=3, b=5)))
    result = eliminate_redundant_configuration(k, pair)
    assert result.kernel.body == k.body[:2] and result.report["removed_calls"] == 1


@pytest.mark.parametrize("fault", ["first_write", "consumer", "value", "interface", "extra", "reset"])
def test_actual_output_mutations_fail(fault):
    k = replace(
        source(), body=(command("set", 0, 3), command("set", 0, 3), command("set", 1, 2), command("work", a=0, b=3))
    )
    actual = eliminate_redundant_configuration(k, adapter()).kernel
    if fault == "first_write":
        actual = replace(actual, body=actual.body[1:])
    elif fault == "consumer":
        actual = replace(actual, body=actual.body[:-1])
    elif fault == "value":
        actual = replace(actual, body=(command("set", 0, 4), *actual.body[1:]))
    elif fault == "interface":
        actual = replace(actual, name="changed")
    elif fault == "extra":
        actual = replace(actual, body=(*actual.body, command("set", 0, 3)))
    else:
        k = replace(k, body=(command("set", 0, 3), command("reset"), command("set", 0, 3)))
        actual = replace(k, body=(k.body[0], k.body[2]))
    with pytest.raises(ValueError, match="configuration conformance"):
        verify_configuration_rewrite(k, actual, adapter())


def test_unsupported_loops_fields_budgets_and_wrong_state():
    k = source()
    for kernel, kwargs in (
        (replace(k, body=(loop("i", 3, *k.body),)), {}),
        (k, {"max_calls": 2}),
        (k, {"max_fields": 1}),
    ):
        with pytest.raises(ConfigurationUnavailable):
            eliminate_redundant_configuration(kernel, adapter(), **kwargs)
    k = replace(k, body=(command("set", 0, 4), command("work", a=0, b=1)))
    with pytest.raises(ConfigurationUnavailable, match="requires field0=3"):
        eliminate_redundant_configuration(k, adapter())


def test_false_pure_write_claim_is_exposed_by_independent_execution(native):
    k = source()
    encoding, _ = codec("<", 4)
    result = eliminate_redundant_configuration(k, adapter())
    # A pulse on every setup is a different device; a matching register value
    # does not qualify this instruction as a removable pure write.
    before = native(encoding.encode(k), 4, False, pulse_on_set=True)
    after = native(encoding.encode(result.kernel), 4, False, pulse_on_set=True)
    assert before[0] != after[0] and before[1][-1] != after[1][-1]


def test_sixty_four_identical_writes_become_one_and_preserve_final_state(native):
    value = (1 << 40) + 3
    k = replace(source(), body=(command("set", 0, value),) * 64)
    encoding, word = codec(">", 8)
    result = eliminate_redundant_configuration(k, adapter())
    before = native(encoding.encode(k), 8, True)
    after = native(encoding.encode(result.kernel), 8, True)
    assert before[0] == after[0]
    assert before[1][1:] == after[1][1:] == (0, 0, value, 73, 0)
    assert before[1][0] == 64 and after[1][0] == 1
    assert len(encoding.encode(result.kernel)) == word.size


def test_encoder_cannot_reintroduce_setup_that_the_pass_removed():
    k = source()
    encoding, word = codec("<", 4)
    corrupt = replace(encoding, encode=lambda ignored: encoding.encode(k))
    banks = (StorageBank("unused", "separate", 8, "bytes"),)
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    limits = ProgramLimits(64 * word.size, 64, 1, windows, ())
    with pytest.raises(ValueError, match="effects differ from the verified schedule"):
        compile_schedule(
            k,
            (),
            banks,
            STORAGE,
            corrupt,
            limits,
            configuration_adapter=adapter(),
            private_ownership=True,
            external_windows=windows,
        )


def test_pure_setup_cannot_also_touch_memory():
    k = source()
    encoding, word = codec("<", 4)

    def bad_project(call):
        effect = storage(call)
        if call.instr == "set":
            return replace(effect, accesses=(EncodedAccess("input", 0, 1),))
        return effect

    unsafe = replace(STORAGE, project=bad_project)
    banks = (StorageBank("unused", "separate", 8, "bytes"),)
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    with pytest.raises(ValueError, match="pure configuration claim disagrees"):
        compile_schedule(
            k,
            (),
            banks,
            unsafe,
            encoding,
            ProgramLimits(64 * word.size, 64, 1, windows, ()),
            configuration_adapter=adapter(),
            private_ownership=True,
            external_windows=windows,
        )


def test_configuration_precedes_fusion_and_survives_allocation_repair():
    from test_physical_fusion import fixture

    k, objects, banks, storage_adapter, capability, encoding, limits, model = fixture(
        count=3, private_output=True, strict=True
    )
    model["reservations"] = (Reservation("normal", 0, 1),)
    cfg = ConfigurationAdapter(
        lambda call: ConfigurationEffects(ordered=True), storage_adapter.configuration_identity, "synthetic"
    )
    result = compile_schedule(
        k,
        objects,
        banks,
        storage_adapter,
        encoding,
        limits,
        configuration_adapter=cfg,
        fusion_capabilities=(capability,),
        **model,
    )
    assert result.report["repair_rounds"] == 1
    assert result.report["storage"]["fusion"]["status"] == "allocation_fallback"
    assert result.report["storage"]["configuration"]["removed_calls"] == 0
    assert [p["name"] for p in result.report["passes"]][:4] == [
        "semantic_admission",
        "configuration",
        "fusion",
        "storage",
    ]


def test_invalid_configuration_order_is_rejected_before_target_callbacks():
    k = source()
    encoding, word = codec("<", 4)
    cfg = replace(adapter(), project=lambda call: pytest.fail("invalid order reached target decoding"))
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    with pytest.raises(ValueError):
        compile_schedule(
            k,
            (),
            (StorageBank("unused", "separate", 8, "bytes"),),
            STORAGE,
            encoding,
            ProgramLimits(64 * word.size, 64, 1, windows, ()),
            configuration_adapter=cfg,
            pass_order=("semantic_admission", "storage", "configuration", "completion", "emission", "final_program"),
            private_ownership=True,
            external_windows=windows,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(writes=(("x", True),)),
        dict(writes=(("x", 1), ("x", 2))),
        dict(writes=(("x", 1),), pure_write=True),
        dict(pure_write=True, ordered=True),
    ],
)
def test_malformed_or_unqualified_state_declarations_refuse(kwargs):
    with pytest.raises(ValueError):
        ConfigurationEffects(**kwargs)


def test_unknown_inserted_transfer_invalidates_config_uses_until_reestablished():
    setters = (command("set", 0, 3), command("set", 1, 2))
    k = replace(source(), body=(*setters, command("opaque"), command("work", a=0, b=7)))
    with pytest.raises(ConfigurationUnavailable, match="lacks an established"):
        verify_configuration_state(k, adapter())
    repaired = replace(k, body=(*k.body[:-1], *setters, k.body[-1]))
    assert verify_configuration_state(repaired, adapter())["checked_reads"] == 2


@pytest.mark.parametrize("preserved", [(), ("field0",), ("field0", "field1")])
def test_segmented_launches_require_explicit_register_persistence(preserved, native):
    k = source()
    encoding, word = codec("<", 4)
    banks = (StorageBank("unused", "separate", 8, "bytes"),)
    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    limits = ProgramLimits(4 * word.size, 4, 1, windows, ())
    continuation = Continuation("synthetic", ("input", "output"), True, True, 8, preserved_configuration=preserved)

    def compile():
        return compile_schedule(
            k,
            (),
            banks,
            STORAGE,
            encoding,
            limits,
            configuration_adapter=adapter(),
            continuation=continuation,
            private_ownership=True,
            external_windows=windows,
        )

    if len(preserved) < 2:
        with pytest.raises(ValueError, match="configuration-register persistence"):
            compile()
        return
    result = compile()
    assert len(result.program.segments) == 3
    # One persistent native interpreter state realizes sequential launches
    # without resetting the declared fields. All final encoded slices run.
    output, counts = native(b"".join(result.program.segments), 4, False)
    expected = bytes((((31 * i + 253) % 256) * 2 + 3) % 256 for i in range(67)) + bytes([91] * 5)
    assert output == expected and counts == (12, 67, 67, 3, 2, 0)
    # Counterexample: retaining memory alone while resetting configuration
    # each launch violates the independent reference on the later launches.
    reset_output = bytearray([91] * 72)
    for image in result.program.segments:
        per_launch, _ = native(image, 4, False)
        for c in encoding.decode_calls(image):
            if c.instr == "work":
                start, count = (evaluate(c.arg(key), {}) for key in ("a", "b"))
                reset_output[start : start + count] = per_launch[start : start + count]
    assert bytes(reset_output) != expected


def test_final_configuration_decoder_checks_fields_separately_from_memory_effects():
    k = source()
    encoding, word = codec("<", 4)

    def memory_projection(call):
        effect = storage(call)
        return replace(effect, semantic_key="setup") if call.instr == "set" else effect

    # Deliberately incomplete storage semantics cannot conceal a changed
    # register operand from the independently selected configuration decoder.
    incomplete = replace(STORAGE, project=memory_projection)

    def corrupt(kernel):
        body = tuple(command("set", evaluate(c.arg("slot"), {}), a=4) if c.instr == "set" else c for c in kernel.body)
        return encoding.encode(replace(kernel, body=body))

    windows = (AddressWindow("input", 0, 72), AddressWindow("output", 0, 72))
    with pytest.raises(ValueError, match="emitted configuration effects differ"):
        compile_schedule(
            k,
            (),
            (StorageBank("unused", "separate", 8, "bytes"),),
            incomplete,
            replace(encoding, encode=corrupt),
            ProgramLimits(64 * word.size, 64, 1, windows, ()),
            configuration_adapter=adapter(),
            private_ownership=True,
            external_windows=windows,
        )
