"""Source provenance through qualified exact async copies and actual encoded bytes.

The neutral evaluator defers source reads and writes until its matching wait.
These tests qualify that closed completion model, not instruction reordering,
concurrent engine support, physical traffic, or accelerator latency.
"""

import ctypes
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace

import pytest
from test_pointwise_source import source_program
from test_private_copy_elimination import SPACES, project
from test_private_copy_elimination import command as original_command
from test_selected_storage_contracts import STORAGE_EXECUTOR, inputs, provider_project
from test_spilled_pointwise_source import INCREMENT, source_text

from merlin.compile.scheduling.pointwise_source import PointwiseBindings, compile_source_pointwise
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.compile.scheduling.temporary_storage import TemporaryObject
from merlin.sched.ir import Const, Kernel, TensorArg, evaluate

# Select a unique encoded token independently of transfer size. The original
# executor already delays reading source bytes until completion; additionally
# qualify the independent six-bit token encoded above the two-bit bank selector.
assert STORAGE_EXECUTOR.count("if(!pending || n!=pn) return 2;") == 1
ASYNC_EXECUTOR = (
    STORAGE_EXECUTOR.replace("int pending=0,pr=0,pw=0;", "int pending=0,pr=0,pw=0,pt=0;")
    .replace("r=code[pc+1],w=code[pc+2];", "r=code[pc+1]&3,w=code[pc+2],token=code[pc+1]>>2;")
    .replace("if(!pending || n!=pn) return 2;", "if(!pending || n!=pn || token!=pt) return 2;")
    .replace("pending=1; pr=r;", "pending=1; pt=token; pr=r;")
)


def command(op, source, destination, size, read_bank=0, write_bank=0, *, token=0):
    call = original_command(op, source, destination, size, read_bank, write_bank)
    return replace(call, args=(*call.args, ("token", Const(token))))


def case(tmp_path, *, m=1, n=47, width=16, separate=False, word_format="<BBBQQQ", asynchronous=True, mutate=None):
    """Three useful source fragments, including the tail; no moved instruction."""
    count = m * n
    assert 2 * width < count <= 3 * width

    def declare(facts, software, storage):
        facts["byte_bits"] = 8
        facts["widths"][1] = 2
        facts["token_bits"], facts["token_offset"] = 6, facts["offsets"][1] + 2
        for instruction in software["isa"]["instructions"]:
            instruction["operands"].append(
                dict(
                    name="token",
                    type=dict(kind="integer", signed=False, bits={"fact": ["facts", "token_bits"]}),
                    unit="integer",
                )
            )
            instruction["encoding"].append(
                dict(
                    offset={"fact": ["facts", "token_offset"]},
                    width={"fact": ["facts", "token_bits"]},
                    argument="token",
                )
            )
            instruction["semantics"]["bindings"]["token"] = "token"
        storage["instructions"]["async_copy"].update(exact_copy=True, token={"argument": "token"})
        storage["instructions"]["wait"]["waits"] = [{"argument": "token"}]
        software["scheduling"]["fusion"] = dict(
            schema="merlin.physical_fusion_contract.v1",
            elementwise={
                "increment": dict(
                    scalar_ir=INCREMENT.decode(),
                    elements=[{"argument": "size"}],
                    element_bits=[{"fact": ["facts", "byte_bits"]}] * 2,
                    stages=["increment"],
                    byte_orders=["little"] * 2,
                    readout=None,
                    execution="contiguous_read_then_write",
                )
            },
            readouts={},
        )
        if mutate:
            mutate(facts, software, storage)

    selected, original, _, adapter, codec, limits, word, _ = inputs(
        tmp_path, width=width, separate=separate, word_format=word_format, reserve=False, mutate=declare
    )

    def effects(call):
        if call.instr == "async_copy":
            return replace(
                project(replace(call, instr="copy")),
                completion="asynchronous",
                token=str(evaluate(call.arg("token"), {})),
            )
        if call.instr == "wait":
            return replace(provider_project(call), waits=(str(evaluate(call.arg("token"), {})),))
        return provider_project(call)

    adapter = replace(adapter, project=effects, qualification_identity="neutral-deferred-exact-copy-and-token-v1")
    original_decode = codec.program_decoder.decode

    def decode_program(data):
        memory_words = bytearray(data)
        for offset in range(0, len(memory_words), word.size):
            memory_words[offset + 1] &= 3
        return original_decode(bytes(memory_words))

    codec = replace(codec, program_decoder=replace(codec.program_decoder, decode=decode_program))
    calls, objects = [], []
    for index, offset in enumerate(range(0, count, width)):
        size, address = min(width, count - offset), 80 + 32 * index
        bank, view = index % 2, index % 2 if separate else 1
        objects.append(TemporaryObject("fragment_" + str(index), SPACES[bank], address, size))
        transfer = "async_copy" if asynchronous else "copy"
        calls.append(command(transfer, offset, address, size, 2, bank, token=2 * index + 1))
        if asynchronous:
            calls.append(command("wait", 0, 0, size, token=2 * index + 1))
        calls.append(command("increment", address, address, size, view, view))
        calls.append(command(transfer, address, offset, size, view, 3, token=2 * index + 2))
        if asynchronous:
            calls.append(command("wait", 0, 0, size, token=2 * index + 2))
    kernel = Kernel(
        original.name,
        (TensorArg("input", (m, n), "i8", "read"), TensorArg("output", (m, n), "i8", "write")),
        tuple(calls),
    )
    return dict(
        kernel=kernel,
        objects=tuple(objects),
        adapter=adapter,
        encoding=codec,
        limits=limits,
        selected_contracts=selected,
        private_ownership=True,
        bindings=PointwiseBindings(
            (AddressWindow("input", 0, count),), AddressWindow("output", 0, count), ("little", "little"), True
        ),
        fusion_qualification_identity="neutral-native-byte-increment-v1",
    ), word


@pytest.fixture
def execute(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    source, library = tmp_path / "deferred.c", tmp_path / "deferred.so"
    source.write_text(ASYNC_EXECUTOR)
    subprocess.run(
        [compiler, "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    machine = ctypes.CDLL(str(library))
    bytep = ctypes.POINTER(ctypes.c_ubyte)
    machine.run.argtypes = [
        bytep,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        bytep,
        bytep,
        bytep,
        bytep,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]

    def run(image, first, second, data, output, *, word_format, separate, faulty=False, expected_status=0):
        def buffer(array):
            return (ctypes.c_ubyte * len(array)).from_buffer(array)

        raw, counters = bytearray(image), (ctypes.c_uint64 * 5)()
        status = machine.run(
            buffer(raw),
            len(raw),
            8 if word_format.endswith("Q") else 4,
            word_format[0] == ">",
            separate,
            2 if faulty else 0,
            buffer(first),
            buffer(second),
            buffer(data),
            buffer(output),
            192,
            len(data),
            counters,
        )
        assert status == expected_status
        return tuple(counters)

    return run


def compile_case(tmp_path, **choices):
    kwargs, word = case(tmp_path, **choices)
    module, operation = source_program(source_text(choices.get("m", 1), choices.get("n", 47)))
    result = compile_source_pointwise(
        module, operation, budgets=ScheduleBudgets(max_fusions=0, eliminate_copies=False), **kwargs
    )
    return result, kwargs, word


@pytest.mark.parametrize(
    "m,n,width,separate,word_format",
    [
        (1, 8, 3, False, ">BBBIII"),
        (1, 47, 16, True, "<BBBQQQ"),
    ],
)
def test_exact_async_copy_keeps_complete_source_and_native_state(
    tmp_path, execute, record_property, m, n, width, separate, word_format
):
    arms = []
    for asynchronous in (False, True):
        directory = tmp_path / str(asynchronous)
        directory.mkdir()
        result, kwargs, word = compile_case(
            directory, m=m, n=n, width=width, separate=separate, word_format=word_format, asynchronous=asynchronous
        )
        image = b"".join(result.images())
        (directory / "program.bin").write_bytes(image)
        selected = kwargs["selected_contracts"]
        selected.verify_image(image, kwargs["encoding"].decode_calls(image))
        actual = [word.unpack_from(image, pos) for pos in range(0, len(image), word.size)]
        assert len(actual) == (15 if asynchronous else 9)
        assert sum(row[0] == 2 for row in actual) == sum(row[0] == 3 for row in actual) == (6 if asynchronous else 0)
        data = bytearray((i * 71 + 253) % 256 for i in range(m * n + 8))
        history, counters = [], []
        for _ in range(3):
            original = bytes(data)
            output = bytearray([91] * len(data))
            first, second = bytearray([167] * 200), bytearray([199] * 200)
            counts = execute(image, first, second, data, output, word_format=word_format, separate=separate)
            expected = bytes((value + 1) % 256 for value in original[: m * n]) + bytes([91] * 8)
            assert bytes(output) == expected and bytes(data) == original
            assert first[:8] == bytes([167] * 8) and first[192:] == bytes([167] * 8)
            assert second[:16] == bytes([199] * 16) and second[192:] == bytes([199] * 8)
            assert counts[:3] == tuple(
                result.conformance[k] for k in ("actual_commands", "requested_read_bytes", "requested_write_bytes")
            )
            assert counts[:3] == (15 if asynchronous else 9, 3 * m * n, 3 * m * n)
            history.append(bytes(output))
            counters.append(counts)
            data[: m * n] = output[: m * n]
        if asynchronous:
            wrong = bytearray([91] * len(data))
            execute(
                image,
                bytearray(200),
                bytearray(200),
                data,
                wrong,
                word_format=word_format,
                separate=separate,
                faulty=True,
            )
            assert wrong[: m * n] != bytes((value + 1) % 256 for value in data[: m * n])
            changed = bytearray(image)
            changed[-word.size + 1] = (changed[-word.size + 1] & 3) | (63 << 2)
            execute(
                bytes(changed),
                bytearray(200),
                bytearray(200),
                data,
                bytearray(len(data)),
                word_format=word_format,
                separate=separate,
                expected_status=2,
            )
        arms.append(
            dict(
                asynchronous=asynchronous,
                program_sha256=hashlib.sha256(image).hexdigest(),
                decoded_words=actual,
                counters=counters,
                complete_output_sha256=[hashlib.sha256(v).hexdigest() for v in history],
                facts_sha256=hashlib.sha256(selected.facts_bytes).hexdigest(),
                software_sha256=hashlib.sha256(selected.software_bytes).hexdigest(),
                library_sources=dict(result.library_sources),
                source_conformance=result.conformance,
                executor_sha256=hashlib.sha256(ASYNC_EXECUTOR.encode()).hexdigest(),
            )
        )
    assert arms[0]["facts_sha256"] == arms[1]["facts_sha256"]
    assert arms[0]["software_sha256"] == arms[1]["software_sha256"]
    assert arms[0]["library_sources"] == arms[1]["library_sources"]
    assert arms[0]["complete_output_sha256"] == arms[1]["complete_output_sha256"]
    report = dict(arms=arms, scope="neutral encoded source completion correctness; no reordering or latency claim")
    (tmp_path / "async_source_pair.json").write_text(json.dumps(report, indent=2))
    record_property("async_source_pair", json.dumps(report))


@pytest.mark.parametrize(
    "fault", ["missing_wait", "wrong_wait", "read_before_wait", "destination_overwrite", "source_clobber"]
)
def test_pending_source_and_destination_cannot_be_observed_or_overwritten(tmp_path, fault):
    kwargs, _ = case(tmp_path)
    body = list(kwargs["kernel"].body)
    if fault == "missing_wait":
        del body[1]
    elif fault == "wrong_wait":
        body[1] = command("wait", 0, 0, 16, token=63)
    elif fault == "read_before_wait":
        body[1:1] = [command("increment", 80, 80, 16)]
    elif fault == "destination_overwrite":
        body[1:1] = [command("copy", 16, 80, 16, 2, 0)]
    else:
        body[4:4] = [command("increment", 80, 80, 16)]
    kwargs["kernel"] = replace(kwargs["kernel"], body=tuple(body))
    module, operation = source_program(source_text())
    with pytest.raises(ValueError, match="completion|wait|pending|conflicts"):
        compile_source_pointwise(module, operation, **kwargs)


def test_unqualified_async_operation_and_unknown_effect_remain_barriers(tmp_path):
    for name, unknown in (("not_exact", False), ("unknown", True)):
        path = tmp_path / name
        path.mkdir()

        def mutate(_facts, _software, storage):
            storage["instructions"]["async_copy"]["exact_copy"] = False
            if unknown:
                storage["instructions"]["async_copy"].update(completion="unknown", token=None)

        kwargs, _ = case(path, mutate=mutate)
        old = kwargs["adapter"]
        kwargs["adapter"] = replace(
            old,
            project=lambda c: replace(old.project(c), exact_copy=False) if c.instr == "async_copy" else old.project(c),
        )
        module, operation = source_program(source_text())
        with pytest.raises(ValueError, match="asynchronous arithmetic|unknown instruction or completion"):
            compile_source_pointwise(module, operation, **kwargs)


def test_async_arithmetic_is_not_admitted_as_sync_elementwise(tmp_path):
    def mutate(_facts, _software, storage):
        storage["instructions"]["increment"].update(completion="asynchronous", token={"argument": "destination"})

    with pytest.raises(ValueError, match="fusion requires synchronous"):
        case(tmp_path, mutate=mutate)


@pytest.mark.parametrize("fault", ["missing_wait", "wrong_wait", "wrong_source"])
def test_final_source_conformance_decodes_actual_async_program(tmp_path, monkeypatch, fault):
    from merlin.compile.scheduling import pointwise_source

    result, kwargs, _ = compile_case(tmp_path)
    calls = list(kwargs["encoding"].decode_calls(b"".join(result.images())))
    if fault == "missing_wait":
        del calls[-1]
    elif fault == "wrong_wait":
        calls[-1] = replace(calls[-1], args=tuple((k, Const(63) if k == "token" else v) for k, v in calls[-1].args))
    else:
        call = calls[3]
        calls[3] = replace(
            call, args=tuple((k, type(v)(evaluate(v, {}) + 1) if k == "source" else v) for k, v in call.args)
        )
    image = kwargs["encoding"].encode(replace(result.compiled.kernel, body=tuple(calls)))
    false_result = replace(result.compiled, program=replace(result.compiled.program, segments=(image,)))
    monkeypatch.setattr(pointwise_source, "compile_schedule", lambda *args, **kwargs: false_result)
    module, operation = source_program(source_text())
    with pytest.raises(ValueError, match="completion|wait|undefined|overwritten|complete typed source"):
        compile_source_pointwise(module, operation, **kwargs)
