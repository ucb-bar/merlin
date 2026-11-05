"""Actual fixed harness boundaries/full histories; no hardware or timing roles."""

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime import direct_kernel_counter as C
from merlin.runtime import direct_kernel_harness as H
from merlin.runtime import direct_kernel_invocation as V
from merlin.runtime import direct_kernel_phases as P
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def selection(extent=3, byte_order="little"):
    original = CompileOnlySourceAbi(
        tuple(CompileOnlyTensor(name, (extent,), "i8") for name in ("A", "B")),
        tuple(CompileOnlyTensor(name, (extent,), "i8") for name in ("Y", "Z")),
    )
    cb = {
        "tensors": {name: {"shape": [extent], "dtype": "i8"} for name in ("A", "B", "Y", "Z")},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": name, "access": "read" if name in ("A", "B") else "write"} for name in ("A", "B", "Y", "Z")
            ],
            "outputs": ["Y", "Z"],
        },
    }
    inputs = {
        name: {"shape": [extent], "values": [((index * factor) % 251) - 125 for index in range(extent)]}
        for name, factor in (("A", 1), ("B", 7))
    }
    abi = H.DirectKernelAbi("enqueue_copy", "finish_copy", 8, byte_order, "void")
    invocations = V.DirectKernelInvocationPlan(original, 2, "history", "invocations")
    counters = C.DirectKernelCounterPlan("raw_counter", "raw_control", "calls", 2, 256)
    phases = P.DirectKernelPhasePlan(counters, "phases", 8192)
    return cb, inputs, abi, invocations, counters, phases


@pytest.mark.parametrize("extent", [3, 257])
@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("clock", ["markers", "monotonic"])
def test_actual_native_fixed_boundaries_preserve_complete_inputs_outputs_and_two_call_histories(
    tmp_path, extent, byte_order, clock
):
    observation, actual, _ = native(tmp_path, extent, byte_order, clock, False)
    assert observation.invocations.observed_count == observation.counters.completed_count == 2
    for name, rows in observation.invocations.output_bytes:
        assert rows == (actual["A" if name == "Y" else "B"],) * 2
    for source, output in (("A", "Y"), ("B", "Z")):
        assert actual[source] == actual[output]
    if clock == "markers":
        values = dict(observation.phases)
        modulus = 1 << 64
        # These deltas are the declared CPU marker fixture's side effects only.
        # The production observation deliberately computes no delta or units.
        assert (values["console_setup"][0][1] - values["console_setup"][0][0]) % modulus == 6
        assert (values["calibration_loop"][0][1] - values["calibration_loop"][0][0]) % modulus == 5
        assert (values["done_publication"][0][1] - values["done_publication"][0][0]) % modulus == 8
        assert all((row[1] - row[0]) % modulus == 1 for row in values["history_copy"])
        assert all((row[1] - row[0]) % modulus == 29 for row in observation.counters.call_samples)
        assert values["main_body"][0][0] > values["main_body"][0][1]  # actual fixture uint64 wrap
        assert all(row[2] != row[3] for row in observation.counters.call_samples)
    else:
        main = dict(observation.phases)["main_body"][0]
        for _, samples in observation.phases:
            assert all(main[0] <= row[0] <= row[1] <= main[1] for row in samples)
        assert all(main[0] <= row[0] <= row[1] <= main[1] for row in observation.counters.call_samples)
    assert "static_initialization_allocation_and_loader_startup" in observation.unknown
    assert "return_exit_and_parent_readback" in observation.unknown
    assert "outer_sample_storage_and_printing" in observation.unknown


def test_actual_empty_completion_retains_wrong_histories_despite_correct_final_outputs_and_entry_samples(tmp_path):
    observation, actual, _ = native(tmp_path, 3, "little", "markers", True)
    assert actual["Y"] == actual["A"] and actual["Z"] == actual["B"]
    assert all(row == (bytes(3),) * 2 for _, row in observation.invocations.output_bytes)
    assert all(row[2] == row[3] for row in observation.counters.call_samples)
    assert all((row[1] - row[0]) % (1 << 64) == 18 for row in observation.counters.call_samples)
    assert observation.counters.completed_count == 2  # timestamps/count alone do not prove completion
    assert "completion_and_device_synchronization_semantics" in observation.unknown


def native(root, extent, byte_order, clock, omit_completion):
    selected = os.environ.get("MERLIN_TEST_CLANG") or shutil.which("cc")
    if not selected:
        pytest.skip("harness phase controls require an explicit or local native C compiler")
    compiler = Path(selected).resolve(strict=True)
    symbols_tool = Path(os.environ.get("MERLIN_TEST_READELF", "/usr/bin/readelf")).resolve(strict=True)
    cb, inputs, abi, invocations, counters, phases = selection(extent, byte_order)
    phase_roster = phases.bind(cb, abi=abi, invocation_plan=invocations)
    counter_roster = counters.bind(cb, abi=abi, invocation_plan=invocations)
    histories = invocations.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
    roster = (
        phase_roster
        | counter_roster
        | {invocations.count_symbol: 8}
        | {row.symbol: row.byte_extent for row in histories}
    )
    choice = root / "selection.json"
    choice.write_text(
        json.dumps(
            {
                "cb": cb,
                "inputs": inputs,
                "abi": dataclasses.asdict(abi),
                "invocations": invocations.record(),
                "phases": phases.record(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    sources = tuple(Path(module.__file__).resolve(strict=True) for module in (H, C, V, P))
    pins = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (*sources, choice, compiler, symbols_tool)}

    def reopen():
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in pins.items())

    reopen()
    harness = root / "harness.c"
    with I.observe_call(
        root,
        stage="render_fixed_harness_phases",
        function=H.render_direct_kernel,
        arguments=json.loads(choice.read_text()),
        inputs=(choice,),
        outputs=(harness,),
        dependencies=sources,
    ) as observed:
        source = H.render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            abi=abi,
            invocation_plan=invocations,
            counter_plan=counters,
            phase_plan=phases,
        )
        harness.write_text(source)
        observed.returned(stdout=source)
    header = root / "htif.h"
    header.write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
    declarations = "".join(f"extern volatile unsigned char {name}[{size}];\n" for name, size in roster.items())
    # History objects are nonvolatile in the original renderer; match their actual declarations.
    for row in histories:
        declarations = declarations.replace(f"volatile unsigned char {row.symbol}", f"unsigned char {row.symbol}")
    writer = "".join(f"for(unsigned i=0;i<{size};i++)fputc({name}[i],stdout);" for name, size in roster.items())
    getter = (
        "return tick++;"
        if clock == "markers"
        else "struct timespec now;if(clock_gettime(CLOCK_MONOTONIC,&now))exit(20);"
        "return (uint64_t)now.tv_sec*UINT64_C(1000000000)+(uint64_t)now.tv_nsec;"
    )
    control = root / "control.c"
    control.write_text(
        "#define _POSIX_C_SOURCE 200809L\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
        "#include <string.h>\n#include <time.h>\n"
        + declarations
        + "static uint64_t tick=UINT64_MAX-3,state=1234;static unsigned calls,completions;"
        + "static unsigned char *a,*b,*y,*z;\n"
        + "uint64_t raw_counter(void){"
        + getter
        + "}\nuint64_t raw_control(void){return state;}\n"
        + "void console_init(void){tick+=5;state^=17;}\n"
        + "void enqueue_copy(void*pa,void*pb,void*py,void*pz){calls++;tick+=17;a=pa;b=pb;y=py;z=pz;}\n"
        + "void finish_copy(void){completions++;"
        + ("" if omit_completion else f"memcpy(y,a,{extent});memcpy(z,b,{extent});state^=31;tick+=11;")
        + "}\n"
        + 'void htif_puts(const char*s){if(strcmp(s,"DONE\\n"))exit(21);tick+=7;}\n'
        + "void htif_exit(int code){if(code||calls!=2||completions!=2)exit(22);"
        + f"memcpy(y,a,{extent});memcpy(z,b,{extent});"  # end-only correctness cannot hide stale history
        + writer
        + f"fwrite(a,1,{extent},stdout);fwrite(b,1,{extent},stdout);"
        + f"fwrite(y,1,{extent},stdout);fwrite(z,1,{extent},stdout);exit(0);}}\n"
        + "int harness_main(void);int main(void){return harness_main();}\n"
    )
    objects = []
    for name, flags in (("harness", ("-Dmain=harness_main",)), ("control", ())):
        product = root / (name + ".o")
        result = I.run(
            [
                str(compiler),
                "-std=c11",
                "-O2",
                "-I",
                str(root),
                *flags,
                "-c",
                str(root / (name + ".c")),
                "-o",
                str(product),
            ],
            directory=root,
            cwd=root,
            stage="native_phase_object",
            inputs=(root / (name + ".c"), header, choice),
            outputs=(product,),
            dependencies=sources,
            env=ENVIRONMENT,
            capture_output=True,
            timeout=30,
        )
        result.check_returncode()
        objects.append(product)
    elf = root / "control.elf"
    result = I.run(
        [str(compiler), *(str(path) for path in objects), "-o", str(elf)],
        directory=root,
        cwd=root,
        stage="native_phase_link",
        inputs=(*objects, choice),
        outputs=(elf,),
        dependencies=sources,
        env=ENVIRONMENT,
        capture_output=True,
        timeout=30,
    )
    result.check_returncode()
    symbol_result = I.run(
        [str(symbols_tool), "-sW", str(elf)],
        directory=root,
        cwd=root,
        stage="native_phase_symbols",
        inputs=(elf, choice),
        dependencies=sources,
        env=ENVIRONMENT,
        capture_output=True,
        timeout=10,
    )
    symbol_result.check_returncode()
    complete_symbol_roster = roster | {"tensor_" + str(index): extent for index in range(4)}
    found = {}
    for line in symbol_result.stdout.decode("ascii").splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[3] == "OBJECT" and fields[-1] in complete_symbol_roster:
            assert fields[-1] not in found
            found[fields[-1]] = int(fields[2])
    assert found == complete_symbol_roster
    payload = root / "raw-readback.bin"
    result = I.run(
        [str(elf)],
        directory=root,
        cwd=root,
        stage="native_phase_execute",
        inputs=(elf, choice),
        dependencies=sources,
        env=ENVIRONMENT,
        capture_output=True,
        timeout=10,
    )
    result.check_returncode()
    assert result.stderr == b"" and len(result.stdout) == sum(roster.values()) + 4 * extent
    executions = [
        path for path in root.glob("invocations/*/invocation.json") if I.verify(path)["stage"] == "native_phase_execute"
    ]
    assert len(executions) == 1
    executed = I.require_environment(executions[0], environment=ENVIRONMENT)
    stdout = Path(executed["stdout"]["path"])
    assert stdout.read_bytes() == result.stdout
    raw, offset = {}, 0
    for name, size in roster.items():
        raw[name] = result.stdout[offset : offset + size]
        offset += size
    actual = {
        name: result.stdout[offset + index * extent : offset + (index + 1) * extent]
        for index, name in enumerate(("A", "B", "Y", "Z"))
    }
    assert actual["A"] == bytes(value % 256 for value in inputs["A"]["values"])
    assert actual["B"] == bytes(value % 256 for value in inputs["B"]["values"])
    reopen()
    decoded = root / "decoded-readback.json"
    with I.observe_call(
        root,
        stage="decode_fixed_harness_phases",
        function=phases.decode,
        arguments=json.loads(choice.read_text()),
        inputs=(elf, choice, stdout, harness, control, header),
        outputs=(payload, decoded),
        dependencies=sources,
    ) as observed:
        observation = phases.decode(raw, cb=cb, abi=abi, invocation_plan=invocations)
        payload.write_bytes(result.stdout)
        encoded = json.dumps(dataclasses.asdict(observation), sort_keys=True, default=lambda value: value.hex()) + "\n"
        decoded.write_text(encoded)
        observed.returned(stdout=encoded)
    for path in sorted(root.glob("invocations/*/invocation.json")):
        row = I.verify(path)
        if row["kind"] == "subprocess":
            I.require_environment(path, environment=ENVIRONMENT)
        if row["stage"] == "native_phase_execute":
            assert row["stdout"]["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    reopen()
    return observation, actual, raw


@pytest.mark.parametrize("defect", ["missing", "extra", "partial", "phase_count", "call_count", "history_count"])
def test_changed_or_incomplete_original_object_membership_refuses(defect):
    cb, _, abi, invocations, counters, phases = selection()
    roster = phases.bind(cb, abi=abi, invocation_plan=invocations) | counters.bind(
        cb, abi=abi, invocation_plan=invocations
    )
    histories = invocations.bind(cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol)
    roster |= {invocations.count_symbol: 8, **{row.symbol: row.byte_extent for row in histories}}
    objects = {name: bytes(size) for name, size in roster.items()}
    for name in ("phases_completed", "calls_completed", "invocations"):
        objects[name] = (2).to_bytes(8, abi.byte_order)
    if defect == "missing":
        del objects["history_0"]
    elif defect == "extra":
        objects["other"] = b""
    elif defect == "partial":
        objects["phases_history_copy_end"] = bytes(8)
    else:
        name = {"phase_count": "phases_completed", "call_count": "calls_completed", "history_count": "invocations"}[
            defect
        ]
        objects[name] = (1).to_bytes(8, abi.byte_order)
    with pytest.raises(ValueError):
        phases.decode(objects, cb=cb, abi=abi, invocation_plan=invocations)


@pytest.mark.parametrize("defect", ["counter", "repeat", "serial", "budget", "accessor_alias", "history_alias", "huge"])
def test_unsupported_selection_or_whole_roster_budget_refuses_before_rendering(defect):
    cb, inputs, abi, invocations, counters, phases = selection()
    if defect == "huge":
        original = dataclasses.replace(
            invocations.original_abi,
            inputs=tuple(dataclasses.replace(row, shape=(1 << 40,)) for row in invocations.original_abi.inputs),
            outputs=tuple(dataclasses.replace(row, shape=(1 << 40,)) for row in invocations.original_abi.outputs),
        )
        invocations = dataclasses.replace(invocations, original_abi=original)
        for row in cb["tensors"].values():
            row["shape"] = [1 << 40]
    if defect == "counter":
        counters = dataclasses.replace(counters)
    if defect == "repeat":
        invocations = None
    if defect == "budget":
        phases = dataclasses.replace(phases, max_observation_bytes=1)
    if defect == "accessor_alias":
        abi = dataclasses.replace(abi, completion_symbol="phase_value_start")
    if defect == "history_alias":
        invocations = dataclasses.replace(invocations, history_prefix="phases_history_copy")
        # An exact generated symbol collision, not an assumed address/name role.
        phases = dataclasses.replace(phases, storage_prefix="history")
        invocations = dataclasses.replace(invocations, count_symbol="history_completed")
    policy = ReadbackPolicy(FULL_VALUES_B64 if defect == "serial" else COHERENT_DUMP_V1)
    with pytest.raises(ValueError):
        H.render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=policy,
            abi=abi,
            invocation_plan=invocations,
            counter_plan=counters,
            phase_plan=phases,
        )
