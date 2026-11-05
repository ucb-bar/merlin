"""Actual async completion through ordinary allocation, encoding and final cuts.

Neutral byte machines establish the mechanism, not accelerator qualification.
"""

import hashlib
import json
import resource
import sys
import time
from dataclasses import replace

import pytest
from test_private_copy_elimination import command
from test_selected_program_contracts import QUALIFICATION, inputs
from test_selected_storage_contracts import compile_case
from test_selected_storage_contracts import native as native_machine_fixture

from merlin.compile.scheduling.program_limits import Continuation
from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable


@pytest.fixture
def native(tmp_path):
    return native_machine_fixture.__wrapped__(tmp_path)


def asynchronous_model(tmp_path, *, width=7, separate=False, word_format="<BBBQQQ", segmented=False):
    def declare(facts, software, program):
        if segmented:
            program["continuation"]["preserved_backings"] = sorted(
                {window["backing"] for window in program["windows"].values()}
            )

    model = list(inputs(tmp_path, width=width, separate=separate, word_format=word_format, count=128, mutate=declare))
    calls = model[1].body
    # The first save reads its temporary until the explicit wait. The second
    # branch may load independently before that wait, but cannot reuse storage.
    model[1] = replace(
        model[1],
        body=(
            *calls[:2],
            replace(calls[2], instr="async_copy"),
            calls[3],
            command("wait", 0, 0, width),
            *calls[4:],
        ),
    )
    return model


@pytest.mark.parametrize(
    "width,separate,word_format,segmented",
    [
        (7, False, "<BBBQQQ", False),
        (13, True, ">BBBIII", False),
        (7, False, "<BBBQQQ", True),
        (13, True, ">BBBIII", True),
    ],
)
def test_async_ordinary_allocation_executes_complete_graph_and_carried_state(
    tmp_path, native, record_property, width, separate, word_format, segmented
):
    model = asynchronous_model(tmp_path, width=width, separate=separate, word_format=word_format, segmented=segmented)
    selected, _, _, _, _, limits, word, length = model
    kwargs = {}
    if segmented:
        model[5] = replace(limits, max_bytes=5 * word.size, max_instructions=5)
        kwargs["continuation"] = Continuation(QUALIFICATION, tuple(w.space for w in limits.windows), True, True, 16)
    results, compile_seconds, compile_memory = [], [], []
    for enabled in (False, True):
        started = time.perf_counter()
        results.append(compile_case(model, optimize=enabled, **kwargs))
        compile_seconds.append(time.perf_counter() - started)
        compile_memory.append(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 if sys.platform.startswith("linux") else None
        )
    baseline, optimized = results
    assert baseline.report["context"]["library"] == optimized.report["context"]["library"]
    assert optimized.report["storage"]["lifetime_calls"]["value_0"] == [0, 4]
    assert optimized.report["storage"]["addresses"]["value_0"] != optimized.report["storage"]["addresses"]["value_1"]
    assert optimized.report["storage"]["addresses"]["value_0"] == optimized.report["storage"]["addresses"]["value_2"]
    observations, states = [], []
    (tmp_path / "source.sched").write_text(model[1].text())
    for arm, result in zip(("baseline", "optimized"), results):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "schedule.sched").write_text(result.kernel.text())
        (directory / "compilation.json").write_text(json.dumps(result.report, sort_keys=True))
        (directory / "program.json").write_text(json.dumps(result.program.report, sort_keys=True))
        images = []
        for index, image in enumerate(result.program.segments):
            path = directory / f"segment_{index}.bin"
            path.write_bytes(image)
            images.append(dict(path=str(path), sha256=hashlib.sha256(image).hexdigest(), bytes=len(image)))
        first, second = bytearray([99] * 192), bytearray([99] * 192)
        state = bytes((i * 29 + 249) % 256 for i in range(length))
        measurements = []
        for invocation in range(3):
            raw = bytes((i * 31 + 253 + invocation) % 256 for i in range(length))
            packed = raw[::-1] + state + bytes([73] * 8)
            data, output = bytearray(packed), bytearray([91] * len(packed))
            counts = [0] * 5
            for image in result.program.segments:
                observed = native(
                    image,
                    first,
                    second,
                    data,
                    output,
                    width=8 if word_format.endswith("Q") else 4,
                    big=word_format[0] == ">",
                    separate=separate,
                )
                for index in range(3):
                    counts[index] += observed[index]
                for index in (3, 4):
                    counts[index] = max(counts[index], observed[index])
            expected = bytes((x + 1) % 256 for x in packed[: 2 * length]) + bytes([91] * 8)
            assert bytes(output) == expected and bytes(data) == packed
            state = bytes(output[length : 2 * length])
            assert first[:8] == bytes([99] * 8) and second[:16] == bytes([99] * 16)
            assert first[32:40] == first[64:72] == bytes([99] * 8)
            conformance = result.report["final_conformance"]
            assert tuple(counts[:3]) == (
                conformance["actual_commands"],
                conformance["requested_read_bytes"],
                conformance["requested_write_bytes"],
            )
            assert conformance["completion_model"] == "explicit_global_drains"
            input_path, output_path = (directory / f"{name}_{invocation}.bin" for name in ("input", "output"))
            input_path.write_bytes(data)
            output_path.write_bytes(output)
            measurements.append(
                dict(
                    invocation=invocation,
                    actual_commands=counts[0],
                    requested_read_bytes=counts[1],
                    requested_write_bytes=counts[2],
                    memory_high_water=list(counts[3:]),
                    complete_output_sha256=hashlib.sha256(output).hexdigest(),
                    complete_input_sha256=hashlib.sha256(data).hexdigest(),
                    input_path=str(input_path),
                    output_path=str(output_path),
                )
            )
        states.append(state)
        observations.append(
            dict(
                arm=arm,
                segments=len(result.program.segments),
                encoded_bytes=sum(map(len, result.program.segments)),
                measurements=measurements,
                storage=result.report["storage"],
                images=images,
            )
        )
    assert states[0] == states[1]
    assert sum(observations[1]["measurements"][0]["memory_high_water"]) < sum(
        observations[0]["measurements"][0]["memory_high_water"]
    )
    pair = dict(
        scope="complete neutral transfer/arithmetic graph with three carried-state invocations; no real target/model",
        selected_contracts=selected.record(),
        library_sources=optimized.report["library_sources_sha256"],
        compile_seconds=compile_seconds,
        compile_memory=dict(
            process_high_water_bytes=compile_memory,
            scope="cumulative process RSS high-water after each compilation; includes earlier work, "
            "excludes children; not an isolated per-arm peak; Linux only",
        ),
        accelerator_timing="unmeasured",
        host_work="identical explicit input reversal/concatenation; native host glue, no host compiler timing claim",
        arms=observations,
        word_format=word_format,
        separate=separate,
        width=width,
        length=length,
        native_source_sha256=hashlib.sha256((tmp_path / "machine.c").read_bytes()).hexdigest(),
        native_library_sha256=hashlib.sha256((tmp_path / "machine.so").read_bytes()).hexdigest(),
    )
    (tmp_path / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")
    record_property("async_schedule_pair", json.dumps(pair))


@pytest.mark.parametrize("mutation", ["missing_wait", "wrong_token", "early_wait", "overwrite"])
def test_invalid_async_source_refuses_before_encoding(tmp_path, mutation):
    model = asynchronous_model(tmp_path)
    calls = list(model[1].body)
    if mutation == "missing_wait":
        calls.pop(4)
    elif mutation == "wrong_token":
        calls[4] = command("wait", 0, 0, 3)
    elif mutation == "early_wait":
        calls.insert(2, calls.pop(4))
    else:
        calls.insert(3, calls[1])
    model[1] = replace(model[1], body=tuple(calls))
    encoded = []
    model[4] = replace(model[4], encode=lambda kernel: encoded.append(kernel))
    with pytest.raises(ValueError, match="completion|wait"):
        compile_case(model)
    assert encoded == []


@pytest.mark.parametrize("mutation", ["issue_as_sync", "wait_as_sync", "wait_removed"])
def test_final_bytes_and_decoder_must_agree_on_actual_completion(tmp_path, mutation):
    model = asynchronous_model(tmp_path)
    codec = model[4]
    if mutation == "wait_removed":

        def encode(kernel):
            return codec.encode(replace(kernel, body=tuple(c for c in kernel.body if c.instr != "wait")))

        model[4] = replace(codec, encode=encode)
    else:
        wrong = "asynchronous" if mutation == "issue_as_sync" else "drain"

        def decode(image):
            return tuple(
                replace(i, completion="synchronous") if i.completion == wrong else i
                for i in codec.program_decoder.decode(image)
            )

        model[4] = replace(codec, program_decoder=replace(codec.program_decoder, decode=decode))
    with pytest.raises(ValueError, match="completion semantics|before actual completion|final decoding disagrees"):
        compile_case(model)


def test_partial_wait_cannot_claim_a_global_drain_or_trigger_encoding(tmp_path):
    model = list(inputs(tmp_path, count=128))
    calls = list(model[1].body)
    calls[2] = replace(calls[2], instr="async_copy")
    calls[-1] = replace(calls[-1], instr="async_copy")
    calls += [command("wait", 0, 0, 7), command("wait", 0, 0, 3)]
    model[1] = replace(model[1], body=tuple(calls))
    encoded = []
    model[4] = replace(model[4], encode=lambda kernel: encoded.append(kernel))
    with pytest.raises(StorageOptimizationUnavailable, match="partial waits need a qualified final model"):
        compile_case(model)
    assert encoded == []


def test_false_runtime_wait_fails_independent_complete_numerical_execution(tmp_path, native):
    model = asynchronous_model(tmp_path)
    result = compile_case(model)
    length = model[-1]
    data = bytearray((i * 31 + 253) % 256 for i in range(2 * length + 8))
    output = bytearray([91] * len(data))
    counts = native(
        b"".join(result.program.segments),
        bytearray([99] * 192),
        bytearray([99] * 192),
        data,
        output,
        faulty_completion=True,
    )
    conformance = result.report["final_conformance"]
    assert counts[:3] == (
        conformance["actual_commands"],
        conformance["requested_read_bytes"],
        conformance["requested_write_bytes"],
    )
    expected = bytes((x + 1) % 256 for x in data[: 2 * length]) + bytes([91] * 8)
    assert bytes(output) != expected


def test_code_budget_cannot_split_an_outstanding_transfer(tmp_path):
    model = asynchronous_model(tmp_path, segmented=True)
    limits, word = model[5:7]
    model[5] = replace(limits, max_bytes=2 * word.size, max_instructions=2)
    continuation = Continuation(QUALIFICATION, tuple(w.space for w in limits.windows), True, True, 16)
    with pytest.raises(ValueError, match="no completion-safe, state-preserving cut"):
        compile_case(model, continuation=continuation)
