"""Real CPU ELF/object readback custody; no runtime or timing qualification."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.runtime.direct_kernel_counter import DirectKernelCounterPlan
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.runtime.direct_kernel_phases import DirectKernelPhasePlan
from merlin.targetgen.contract import process_execution as P
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.contract.prepared_process_readback import PreparedProcessReadbackPlan
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, ReadbackPolicy

ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def parse_completion(console):
    assert console == "DONE\n"
    return {}, {}


def selection():
    original = CompileOnlySourceAbi(
        tuple(CompileOnlyTensor(name, (3,), "i8") for name in ("A", "B")),
        tuple(CompileOnlyTensor(name, (3,), "i8") for name in ("Y", "Z")),
    )
    cb = {
        "tensors": {name: {"shape": [3], "dtype": "i8"} for name in ("A", "B", "Y", "Z")},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": name, "access": "read" if name in ("A", "B") else "write"} for name in ("A", "B", "Y", "Z")
            ],
            "outputs": ["Y", "Z"],
        },
    }
    abi = DirectKernelAbi("enqueue_copy", "finish_copy", 8, "little", "void")
    calls = DirectKernelInvocationPlan(original, 2, "history", "invocations")
    counters = DirectKernelCounterPlan("raw_counter", "raw_control", "calls", 2, 256)
    phases = DirectKernelPhasePlan(counters, "phases", 8192)
    plan = PreparedProcessReadbackPlan(abi, calls, counters, phases, 0, 8192, 16384)
    inputs = {"A": {"shape": [3], "values": [-17, 5, 31]}, "B": {"shape": [3], "values": [31, -8, -17]}}
    return cb, inputs, plan


def compile_control(root, mode):
    root.mkdir(mode=0o700)
    compiler = Path(os.environ.get("MERLIN_CLANG", "/usr/bin/clang-18")).resolve(strict=True)
    symbols = Path(os.environ.get("MERLIN_TEST_READELF", "/usr/bin/readelf")).resolve(strict=True)
    cb, inputs, plan = selection()
    roster = plan.bind(cb)
    choice = root / "original.json"
    choice.write_text(json.dumps({"cb": cb, "inputs": inputs, "plan": plan.record()}, sort_keys=True))
    header = root / "htif.h"
    header.write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
    harness = root / "harness.c"
    with I.observe_call(
        root,
        stage="render_original_prepared_harness",
        function=render_direct_kernel,
        arguments={"plan": plan.record()},
        inputs=(choice,),
        outputs=(harness,),
        dependencies=plan.source_paths(),
    ) as observed:
        source = render_direct_kernel(
            cb,
            inputs=inputs,
            abi=plan.abi,
            invocation_plan=plan.invocation_plan,
            counter_plan=plan.counter_plan,
            phase_plan=plan.phase_plan,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
        )
        harness.write_text(source)
        observed.returned(stdout=source)
    history_names = {
        row.symbol
        for row in plan.invocation_plan.bind(
            cb, entry_symbol=plan.abi.entry_symbol, completion_symbol=plan.abi.completion_symbol
        )
    }
    declarations = "".join(
        f"extern {'volatile ' if name not in history_names and not name.startswith('tensor_') else ''}"
        f"unsigned char {name}[{size}];\n"
        for name, size in roster
        if not name.startswith("tensor_")
    )
    pointers = {"tensor_" + str(index): pointer for index, pointer in enumerate(("a", "b", "y", "z"))}
    writer = "".join(f"if(fwrite({pointers.get(name, name)},1,{size},out)!={size})exit(24);" for name, size in roster)
    if mode == "partial":
        writer = writer.rsplit("if(fwrite", 1)[0]
    control = root / "control.c"
    control.write_text(
        "#define _POSIX_C_SOURCE 200809L\n#include <stdio.h>\n#include <stdint.h>\n#include <stdlib.h>\n"
        "#include <fcntl.h>\n#include <unistd.h>\n#include <string.h>\n"
        + declarations
        + "static char *request_path,*output_path;static unsigned char *a,*b,*y,*z;static uint64_t tick,state;"
        "uint64_t raw_counter(void){return tick++;}uint64_t raw_control(void){return state;}"
        "void console_init(void){}void enqueue_copy(void*pa,void*pb,void*py,void*pz){a=pa;b=pb;y=py;z=pz;}"
        "void finish_copy(void){" + ("" if mode == "completion" else "memcpy(y,a,3);memcpy(z,b,3);state++;") + "}"
        'void htif_puts(const char*s){if(strcmp(s,"DONE\\n"))exit(22);puts("DONE");}'
        "void htif_exit(int code){if(code)exit(code);memcpy(y,a,3);memcpy(z,b,3);"
        'FILE *request=fopen(request_path,"rb");if(!request)exit(23);unsigned bytes=0;'
        "while(fgetc(request)!=EOF)if(++bytes>16384)exit(23);if(fclose(request)||!bytes)exit(23);"
        + (
            "exit(0);"
            if mode == "missing"
            else 'int fd=open(output_path,O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0)exit(24);FILE*out=fdopen(fd,"wb");'
            + writer
            + "if(fclose(out))exit(24);exit(0);"
        )
        + "}int harness_main(void);int main(int argc,char**argv){if(argc!=3)return 25;"
        "request_path=argv[1];output_path=argv[2];return harness_main();}\n"
    )
    dependencies = (*plan.source_paths(), compiler, symbols, Path(__file__).resolve())
    objects = []
    for path, flags in ((harness, ("-Dmain=harness_main",)), (control, ())):
        product = root / (path.stem + ".o")
        I.run(
            [str(compiler), "-std=c11", "-O2", "-I", str(root), *flags, "-c", str(path), "-o", str(product)],
            directory=root,
            stage="prepared_native_object",
            cwd=root,
            env=ENVIRONMENT,
            inputs=(path, header, choice),
            outputs=(product,),
            dependencies=dependencies,
            capture_output=True,
            timeout=30,
            check=True,
        )
        objects.append(product)
    elf = root / "control.elf"
    I.run(
        [str(compiler), *(str(path) for path in objects), "-o", str(elf)],
        directory=root,
        stage="prepared_native_link",
        cwd=root,
        env=ENVIRONMENT,
        inputs=(*objects, choice),
        outputs=(elf,),
        dependencies=dependencies,
        capture_output=True,
        timeout=30,
        check=True,
    )
    result = I.run(
        [str(symbols), "-sW", str(elf)],
        directory=root,
        stage="prepared_native_symbols",
        cwd=root,
        env=ENVIRONMENT,
        inputs=(elf, choice),
        dependencies=dependencies,
        capture_output=True,
        timeout=10,
        check=True,
    )
    found = {}
    for line in result.stdout.decode("ascii").splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[3] == "OBJECT" and fields[-1] in dict(roster):
            assert fields[-1] not in found
            found[fields[-1]] = int(fields[2])
    assert found == dict(roster) and len(found) == 37
    return elf


@pytest.fixture(scope="module")
def native_elfs(tmp_path_factory):
    root = tmp_path_factory.mktemp("prepared-native-elfs")
    return {mode: compile_control(root / mode, mode) for mode in ("complete", "completion", "missing", "partial")}


@pytest.fixture
def prepared(tmp_path, native_elfs):
    cb, inputs, plan = selection()
    executable = Path("/usr/bin/env").resolve(strict=True)
    pins = tuple(
        (str(path), file_digest(path))
        for path in (
            executable,
            module_source_path("merlin.targetgen.contract.process_execution"),
            *plan.source_paths(),
        )
    )
    process = P.RecordedProcessExecution(
        executable,
        ("{elf}", "{request}", "{output}"),
        tmp_path,
        tuple(ENVIRONMENT.items()),
        tmp_path / "records",
        "stdout",
        pins,
        plan,
    )
    return cb, inputs, plan, process, native_elfs


def request_for(prepared, tmp_path, mode="complete"):
    cb, _, plan, _, elfs = prepared
    elf = elfs[mode]
    with I.observe_call(
        tmp_path,
        stage="prepare_original_process_request",
        function=plan.prepare,
        arguments={"plan": plan.record(), "original_cb": cb},
        inputs=(elf, elf.parent / "original.json"),
        dependencies=(*plan.source_paths(), Path(__file__).resolve()),
    ) as observed:
        request = plan.prepare(cb=cb, elf_path=elf, workdir=tmp_path)["memory_readback"]
        observed.outputs = tuple(Path(request[name]) for name in ("cb_path", "request_path"))
        observed.returned(stdout=json.dumps(request, sort_keys=True))
    return request


def objects(plan, cb, output):
    raw = output.read_bytes()
    found, offset = {}, 0
    for name, extent in plan.bind(cb):
        found[name] = raw[offset : offset + extent]
        offset += extent
    assert len(raw) == offset
    return found


def test_actual_fixed_service_consumes_full_original_elf_request_objects_and_environment(prepared, tmp_path):
    cb, inputs, plan, process, elfs = prepared
    request = request_for(prepared, tmp_path)
    service = FunctionalExecutionService(
        "cpu_diagnostic",
        "native_diagnostic",
        process.run_elf,
        parse_completion,
        (*process.source_pins, (str(Path(__file__).resolve()), file_digest(Path(__file__)))),
        '{"scope":"CPU object custody only"}',
        process,
    )
    console = service.run_elf(elfs["complete"], simulator=service.simulator, timeout=10, memory_readback=request)
    assert service.parse_output(console) == ({}, {})
    joined = service.consumption(elf=elfs["complete"], console=console)
    actual = I.require_environment(Path(joined["record"]["path"]), environment=ENVIRONMENT)
    assert actual["argv"] == [
        str(process.executable),
        str(elfs["complete"]),
        request["request_path"],
        request["output_path"],
    ]
    assert len(actual["inputs"]) == 3 and actual["outputs"] == [joined["prepared_readback"]["output"]]
    assert joined["prepared_readback"]["product_bytes"] == 368
    raw = objects(plan, cb, Path(request["output_path"]))
    observation = plan.phase_plan.decode(
        {name: value for name, value in raw.items() if not name.startswith("tensor_")},
        cb=cb,
        abi=plan.abi,
        invocation_plan=plan.invocation_plan,
    )
    assert observation.invocations.observed_count == observation.counters.completed_count == 2
    for index, name in enumerate(("A", "B")):
        expected = bytes(value % 256 for value in inputs[name]["values"])
        assert raw["tensor_" + str(index)] == raw["tensor_" + str(index + 2)] == expected
        assert dict(observation.invocations.output_bytes)[("Y", "Z")[index]] == (expected,) * 2
    wire = json.loads(Path(request["request_path"]).read_text())
    assert all(set(row) == {"symbol", "bytes"} for row in wire["objects"])
    assert "values" not in json.dumps(wire) and "address" not in wire
    assert "unqualified" in joined["scope"] and observation.unknown


@pytest.mark.parametrize("mode", ["missing", "partial"])
def test_actual_zero_exit_done_with_missing_or_partial_product_refuses(prepared, tmp_path, mode):
    _, _, _, process, elfs = prepared
    with pytest.raises(ValueError, match="ordinary file|complete original readback"):
        process.run_elf(elfs[mode], timeout=10, memory_readback=request_for(prepared, tmp_path, mode))
    record = next(process.record_root.rglob("invocation.json"))
    actual = I.require_environment(record, environment=ENVIRONMENT)
    assert actual["returncode"] == 0 and Path(actual["stdout"]["path"]).read_bytes() == b"DONE\n"


def test_actual_completion_defect_keeps_correct_final_buffers_but_wrong_complete_histories(prepared, tmp_path):
    cb, _, plan, process, elfs = prepared
    request = request_for(prepared, tmp_path, "completion")
    console = process.run_elf(elfs["completion"], timeout=10, memory_readback=request)
    process.consumption(elf=elfs["completion"], console=console)
    raw = objects(plan, cb, Path(request["output_path"]))
    assert raw["tensor_0"] == raw["tensor_2"] and raw["tensor_1"] == raw["tensor_3"]
    observation = plan.phase_plan.decode(
        {name: value for name, value in raw.items() if not name.startswith("tensor_")},
        cb=cb,
        abi=plan.abi,
        invocation_plan=plan.invocation_plan,
    )
    assert observation.invocations.observed_count == 2
    assert all(rows == (bytes(3), bytes(3)) for _, rows in observation.invocations.output_bytes)
    assert "completion_and_device_synchronization_semantics" in observation.unknown


def test_default_and_missing_selection_keep_option_refusal(prepared, tmp_path):
    _, _, _, process, elfs = prepared
    with pytest.raises(ValueError, match="unsupported invocation options"):
        replace(process, prepared_readback=None, argv_template=("{elf}",)).run_elf(
            elfs["complete"], timeout=5, memory_readback=request_for(prepared, tmp_path)
        )
    with pytest.raises(ValueError, match="unsupported closed request"):
        process.run_elf(elfs["complete"], timeout=5)
    with pytest.raises(ValueError, match="unsupported invocation options"):
        process.run_elf(elfs["complete"], timeout=5, unrelated=True)
    assert not process.record_root.exists()


@pytest.mark.parametrize("defect", ["wrong_elf", "roster", "request", "preexisting", "alias", "hardlink"])
def test_preflight_refuses_changed_or_indirect_request_before_native_dispatch(prepared, tmp_path, defect):
    _, _, _, process, elfs = prepared
    request = request_for(prepared, tmp_path)
    selected = elfs["complete"]
    if defect == "wrong_elf":
        selected = elfs["completion"]
    elif defect == "roster":
        path = Path(request["request_path"])
        wire = json.loads(path.read_text())
        wire["objects"][-1]["bytes"] += 1
        path.write_text(json.dumps(wire))
        request["request_sha256"] = file_digest(path)
    elif defect == "request":
        request["unsupported"] = True
    elif defect == "preexisting":
        Path(request["output_path"]).write_bytes(b"old packet")
    else:
        path = Path(request["request_path"])
        original = path.with_suffix(".original")
        path.rename(original)
        path.symlink_to(original) if defect == "alias" else os.link(original, path)
    with pytest.raises(
        ValueError, match="selected ELF|typed roster|closed request|absent before|unlinked|individually owned"
    ):
        process.run_elf(selected, timeout=5, memory_readback=request)
    assert not process.record_root.exists()


@pytest.mark.parametrize("member", ["request", "cb", "output", "tool", "environment"])
def test_actual_consumption_reopens_every_request_product_and_selection(prepared, tmp_path, member):
    _, _, _, process, elfs = prepared
    request = request_for(prepared, tmp_path)
    console = process.run_elf(elfs["complete"], timeout=5, memory_readback=request)
    if member in ("request", "cb", "output"):
        path = Path(request[{"request": "request_path", "cb": "cb_path", "output": "output_path"}[member]])
        with path.open("ab") as output:
            output.write(b"changed")
    else:
        object.__setattr__(
            process,
            "environment" if member == "environment" else "argv_template",
            (("PATH", "/bin"), ("LC_ALL", "C"))
            if member == "environment"
            else ("--", "{elf}", "{request}", "{output}"),
        )
    with pytest.raises((ValueError, json.JSONDecodeError), match="changed|roster|budget|Extra data"):
        process.consumption(elf=elfs["complete"], console=console)


@pytest.mark.parametrize(
    "argv",
    [
        ("{elf}",),
        ("{elf}", "{request}", "{request}", "{output}"),
        ("{elf}", "{request}", "--out={output}"),
        ("{elf}", "{elf}", "{request}", "{output}"),
    ],
)
def test_prepared_selection_has_only_exact_single_operands(prepared, argv):
    with pytest.raises(ValueError, match="exact explicit command"):
        replace(prepared[3], argv_template=argv).verify()


def test_huge_original_extent_is_bounded_before_creating_request(prepared, tmp_path):
    cb, _, plan, _, elfs = prepared
    huge = 1 << 63
    original = CompileOnlySourceAbi(
        tuple(replace(slot, shape=(huge,)) for slot in plan.invocation_plan.original_abi.inputs),
        tuple(replace(slot, shape=(huge,)) for slot in plan.invocation_plan.original_abi.outputs),
    )
    changed = replace(plan, invocation_plan=replace(plan.invocation_plan, original_abi=original))
    for spec in cb["tensors"].values():
        spec["shape"] = [huge]
    with pytest.raises(ValueError, match="budget|uint64 storage"):
        changed.prepare(cb=cb, elf_path=elfs["complete"], workdir=tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_actual_native_redirected_product_refuses_original_output_custody(prepared, tmp_path, monkeypatch):
    _, _, _, process, elfs = prepared
    request = request_for(prepared, tmp_path)
    redirected = Path(request["output_path"]).with_name("redirected.bin")
    native_run = I.run

    def redirect(argv, **kwargs):
        return native_run([*argv[:-1], str(redirected)], **kwargs)

    monkeypatch.setattr(P.I, "run", redirect)
    with pytest.raises(ValueError, match="ordinary file"):
        process.run_elf(elfs["complete"], timeout=10, memory_readback=request)
    assert redirected.stat().st_size == 368 and not Path(request["output_path"]).exists()
    actual = I.require_environment(next(process.record_root.rglob("invocation.json")), environment=ENVIRONMENT)
    assert actual["argv"][-1] == str(redirected)


@pytest.mark.parametrize("member", ["elf", "output", "request_dict", "tool_bytes"])
def test_changed_actual_same_extent_bytes_or_request_identity_refuses(prepared, tmp_path, member):
    _, _, _, process, elfs = prepared
    elf = tmp_path / "selected.elf"
    elf.write_bytes(elfs["complete"].read_bytes())
    elf.chmod(0o700)
    (tmp_path / "original.json").write_bytes((elfs["complete"].parent / "original.json").read_bytes())
    if member == "tool_bytes":
        executable = tmp_path / "selected-tool"
        executable.write_bytes(process.executable.read_bytes())
        executable.chmod(0o700)
        process = replace(
            process,
            executable=executable,
            source_pins=tuple(
                (str(executable), file_digest(executable)) if path == str(process.executable) else (path, digest)
                for path, digest in process.source_pins
            ),
        )
    request = request_for((*prepared[:-1], {"complete": elf}), tmp_path)
    console = process.run_elf(elf, timeout=10, memory_readback=request)
    if member == "request_dict":
        request["request_sha256"] = "0" * 64
    else:
        path = (
            elf if member == "elf" else process.executable if member == "tool_bytes" else Path(request["output_path"])
        )
        with path.open("r+b") as output:
            first = output.read(1)
            output.seek(0)
            output.write(bytes((first[0] ^ 1,)))
    with pytest.raises(ValueError, match="changed|selected source/tool|selected tool or fixed source owner"):
        process.consumption(elf=elf, console=console)


def test_file_growth_at_open_refuses_exact_product_byte_limit(prepared, tmp_path, monkeypatch):
    _, _, _, process, elfs = prepared
    request = request_for(prepared, tmp_path)
    console = process.run_elf(elfs["complete"], timeout=10, memory_readback=request)
    selected = Path(request["output_path"])
    original_open, grew = Path.open, []

    def grow(path, mode="r", *args, **kwargs):
        if path == selected and mode == "rb" and not grew:
            grew.append(True)
            with original_open(path, "ab") as output:
                output.write(b"growth")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow)
    with pytest.raises(ValueError, match="byte budget"):
        process.consumption(elf=elfs["complete"], console=console)
    assert grew and selected.stat().st_size == 374


@pytest.mark.parametrize("change", ["output", "dtype", "order", "depth", "scalar"])
def test_original_typed_contract_or_metadata_change_refuses_before_files(prepared, tmp_path, change):
    cb, _, plan, _, elfs = prepared
    if change == "output":
        cb["kernel_abi"]["outputs"].pop()
    elif change == "dtype":
        cb["tensors"]["Y"]["dtype"] = "i16"
    elif change == "order":
        cb["kernel_abi"]["args"].reverse()
    elif change == "depth":
        value = {}
        cb["extra"] = value
        for _ in range(34):
            value["child"] = {}
            value = value["child"]
    else:
        cb["extra"] = 1 << 80
    with pytest.raises(ValueError):
        plan.prepare(cb=cb, elf_path=elfs["complete"], workdir=tmp_path)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("ordering", ["inputs", "dependencies"])
def test_actual_sibling_prefix_members_preserve_recorder_path_order(prepared, tmp_path, ordering):
    cb, _, plan, process, elfs = prepared
    if ordering == "inputs":
        workspace, artifact_owner = tmp_path / "installed", tmp_path / "installed-venv"
        workspace.mkdir(mode=0o700)
        artifact_owner.mkdir(mode=0o700)
        elf = artifact_owner / "control.elf"
        elf.write_bytes(elfs["complete"].read_bytes())
        elf.chmod(0o700)
        (artifact_owner / "original.json").write_bytes((elfs["complete"].parent / "original.json").read_bytes())
        request = request_for((*prepared[:-1], {"complete": elf}), workspace)
    else:
        selected = []
        for name in ("installed", "installed-venv"):
            owner = tmp_path / name
            owner.mkdir(mode=0o700)
            member = owner / "selected.source"
            member.write_text("explicit diagnostic dependency\n")
            selected.append(member)
        process = replace(process, source_pins=(*process.source_pins, *((str(p), file_digest(p)) for p in selected)))
        elf = elfs["complete"]
        request = request_for(prepared, tmp_path)
    console = process.run_elf(elf, timeout=10, memory_readback=request)
    actual = I.require_environment(next(process.record_root.rglob("invocation.json")), environment=ENVIRONMENT)
    rows = actual[ordering]
    assert rows == sorted(rows, key=lambda row: Path(row["path"]))
    assert rows != sorted(rows, key=lambda row: row["path"])
    process.consumption(elf=elf, console=console)
    raw = objects(plan, cb, Path(request["output_path"]))
    assert raw["tensor_0"] == raw["tensor_2"] and raw["tensor_1"] == raw["tensor_3"]
