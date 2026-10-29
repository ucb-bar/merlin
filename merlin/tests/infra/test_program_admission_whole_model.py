"""Real ordinary build/cache/relink checks on an interpreted neutral program.

This fixture executes the linked bytes as model parameters; it establishes no
accelerator support, device mapping or performance qualification.
"""

import inspect
import json
import struct
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_dense_source_catalog import compilation_memory
from test_selected_program_contracts import inputs

from merlin.model_check import full_build as FB
from merlin.model_check.program_admission import service
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.elf_lanes import read_program_section


def admit_interpreter_subset(*, elf, evidence_root, word_format, extent):
    """Independent executor contract: this interpreter implements scalar copy/inc."""
    word = struct.Struct(word_format)
    image, _ = read_program_section(elf, ".weights", max_bytes=64 * word.size)
    accepted = len(image) == 64 * word.size
    for offset in range(0, len(image), word.size):
        op, r, w, source, destination, size = word.unpack_from(image, offset)
        accepted &= op in (0, 1) and r in (2, 3) and w == 3 and size == 1
        accepted &= source < extent and destination < extent
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "interpreter-admission.json"
    result = dict(status="accepted" if accepted else "refused", elf_sha256=FB.sha256(elf))
    path.write_text(json.dumps(result))
    return {**result, "report_path": str(path), "report_sha256": FB.sha256(path)}


def source_ir(code_size, word_size, field_bytes, big, extent):
    low = field_bytes - 1 if big else 0
    return (
        """module {
func.func private @htif_putc(i8)
func.func @forward(%code: tensor<CODE x i8>, %input: tensor<EXT x i8>,
                   %state: tensor<EXT x i8>, %position: tensor<i32>)
 -> (tensor<EXT x i8>, tensor<i32>, tensor<EXT x i8>) {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %count = arith.constant 64 : index
 %stride = arith.constant STRIDE : index
 %sourcefield = arith.constant SOURCE : index
 %destfield = arith.constant DEST : index
 %one = arith.constant 1 : i8
 %inputbank = arith.constant 2 : i8
 %newline = arith.constant 10 : i8
 %trace_p = arith.constant 80 : i8
 %trace_colon = arith.constant 58 : i8
 %trace_zero = arith.constant 48 : i8
 func.call @htif_putc(%newline) : (i8) -> ()
 func.call @htif_putc(%trace_p) : (i8) -> ()
 func.call @htif_putc(%trace_colon) : (i8) -> ()
 %out = scf.for %pc = %c0 to %count step %c1 iter_args(%s = %state) -> (tensor<EXT x i8>) {
   %base = arith.muli %pc, %stride : index
   %op = tensor.extract %code[%base] : tensor<CODE x i8>
   %ri = arith.addi %base, %c1 : index
   %r = tensor.extract %code[%ri] : tensor<CODE x i8>
   %si = arith.addi %base, %sourcefield : index
   %sb = tensor.extract %code[%si] : tensor<CODE x i8>
   %source = arith.index_castui %sb : i8 to index
   %di = arith.addi %base, %destfield : index
   %db = tensor.extract %code[%di] : tensor<CODE x i8>
   %dest = arith.index_castui %db : i8 to index
   %external = arith.cmpi eq, %r, %inputbank : i8
   %v = scf.if %external -> (i8) {
     %x = tensor.extract %input[%source] : tensor<EXT x i8>
     scf.yield %x : i8
   } else {
     %x = tensor.extract %s[%source] : tensor<EXT x i8>
     scf.yield %x : i8
   }
   %increment = arith.cmpi eq, %op, %one : i8
   %plus = arith.addi %v, %one : i8
   %value = arith.select %increment, %plus, %v : i8
   %updated = tensor.insert %value into %s[%dest] : tensor<EXT x i8>
   %trace_char = arith.addi %op, %trace_zero : i8
   func.call @htif_putc(%trace_char) : (i8) -> ()
   scf.yield %updated : tensor<EXT x i8>
 }
 func.call @htif_putc(%newline) : (i8) -> ()
 %position0 = tensor.extract %position[] : tensor<i32>
 %one32 = arith.constant 1 : i32
 %position1 = arith.addi %position0, %one32 : i32
 %positionout = tensor.from_elements %position1 : tensor<i32>
 return %out, %positionout, %input : tensor<EXT x i8>, tensor<i32>, tensor<EXT x i8>
}}
""".replace("CODE", str(code_size))
        .replace("EXT", str(extent))
        .replace("STRIDE", str(word_size))
        .replace("SOURCE", str(3 + low))
        .replace("DEST", str(3 + field_bytes + low))
        .replace(" x ", "x")
    )


def observed_requests(console, image, word_format):
    """Count executed interpreter operations, never physical CPU/bus traffic.

    The selected interpreter admits only one-byte copy/increment instructions.
    Its opaque IO calls print each actual fetched opcode after the state update;
    unknown IO effects remain compiler barriers. Decode the actual ELF payload
    independently and require exact event membership/order before counting.
    """
    word = struct.Struct(word_format)
    assert len(image) % word.size == 0
    rows = [word.unpack_from(image, offset) for offset in range(0, len(image), word.size)]
    assert rows and all(
        op in (0, 1) and r in (2, 3) and w == 3 and size == 1 for op, r, w, source, destination, size in rows
    )
    events = [line[2:] for line in console.splitlines() if line.startswith(b"P:")]
    assert events == [bytes(48 + row[0] for row in rows)], "missing, reordered or changed actual program events"
    return dict(
        executed_program_commands=len(events[0]),
        requested_read_bytes=sum(row[-1] for row in rows),
        requested_write_bytes=sum(row[-1] for row in rows),
        scope="executed admitted interpreter data accesses; "
        "excludes code fetch, packing, native accesses and bus traffic",
    )


@pytest.mark.parametrize("mutation", ["missing", "short", "extra", "order", "unsupported", "extent"])
def test_program_trace_cannot_replace_actual_instruction_effects(mutation):
    word_format = "<BBBQQQ"
    word = struct.Struct(word_format)
    rows = [(0, 2, 3, 0, 0, 1), (1, 3, 3, 0, 0, 1), (0, 2, 3, 1, 1, 1)]
    image = b"".join(word.pack(*row) for row in rows)
    console = b"P:010\n"
    assert observed_requests(console, image, word_format)["executed_program_commands"] == 3
    if mutation == "missing":
        console = b""
    elif mutation == "short":
        console = b"P:01\n"
    elif mutation == "extra":
        console *= 2
    elif mutation == "order":
        console = b"P:100\n"
    else:
        row = list(rows[0])
        row[0 if mutation == "unsupported" else -1] = 2
        image = word.pack(*row) + image[word.size :]
    with pytest.raises(AssertionError):
        observed_requests(console, image, word_format)


@pytest.mark.parametrize("word_format,separate", [("<BBBQQQ", False), (">BBBIII", True)])
def test_ordinary_model_checks_linked_program_and_complete_carried_state(
    tmp_path, monkeypatch, record_property, word_format, separate
):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model
    from merlin.targetgen.rtl.facts import observed_facts

    if not toolchain.available() or not spike.available():
        pytest.skip("selected native lowering, cross toolchain and functional simulator required")
    model = inputs(tmp_path, width=31, word_format=word_format, separate=separate, count=64)
    selected, _, _, _, codec, _, word, length = model
    extent = 2 * length + 8
    rows = [(0, 2, 3, 0, 0, 1)] + [(1, 3, 3, 0, 0, 1)] * 62 + [(0, 2, 3, 1, 1, 1)]
    payload = b"".join(word.pack(*row) for row in rows)
    root, capture, host = (tmp_path / name for name in ("out", "capture", "host"))
    for path in (root, capture, host):
        path.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", "streamed")
    (host / "manifest.yaml").write_text(
        yaml.safe_dump(
            dict(
                target="host",
                run_id="neutral-interpreter",
                family="scalar_linalg",
                authoring=dict(mode="deterministic_generated_from_spec", generated_by_agent=False),
                outputs=dict(knobs="knobs.yaml"),
            )
        )
    )
    (host / "knobs.yaml").write_text(
        yaml.safe_dump(
            dict(
                backend="scalar",
                dtype_strategy="fp32",
                cflags=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
            )
        )
    )
    dts, catalog = tmp_path / "cpu.dts", tmp_path / "boards.yaml"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    catalog.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                boards=dict(
                    selected=dict(
                        target="neutral",
                        dram_bytes=64 << 20,
                        dram_base=spike_model.DRAM_BASE,
                        harts=1,
                        console="htif",
                        flow="baremetal",
                        code_reserve=16 << 20,
                        host_dts_sha256=FB.sha256(dts),
                    )
                ),
            )
        )
    )
    settings = FB.BuildSettings("selected", catalog, dts, "fp32", 1, host)
    header = json.dumps(dict(code=dict(dtype="I8", shape=[len(payload)], data_offsets=[0, len(payload)]))).encode()
    (capture / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": dict(kind="param", weight="code"),
                "1": dict(kind="input", name="input"),
                "2": dict(kind="input", name="state"),
                "3": dict(kind="input", name="position"),
            }
        )
    )
    (capture / "model.mlir").write_text(
        source_ir(len(payload), word.size, 8 if word_format.endswith("Q") else 4, word_format.startswith(">"), extent)
    )
    initial = np.arange(extent, dtype=np.uint8).view(np.int8)
    np.savez(capture / "inputs.npz", in0=initial, in1=initial[::-1].copy(), in2=np.array(4, np.int32))
    (capture / "input_order.json").write_text('{"input":0,"state":1,"position":2}')
    initial_reference = bytearray(initial[::-1].tobytes())
    initial_reference[0], initial_reference[1] = 62, 1
    np.save(capture / "golden.npy", np.frombuffer(initial_reference, dtype=np.int8))
    artifacts = {p.name: dict(bytes=p.stat().st_size, sha256=FB.sha256(p)) for p in capture.iterdir()}
    (capture / "capture_receipt.json").write_text(
        json.dumps(dict(schema="m2m.capture-receipt.v1", materialized_abi=dict(complete=True), artifacts=artifacts))
    )
    owner = Path(__file__).resolve()
    original = LinkedElfAdmissionService(
        selected.target,
        partial(admit_interpreter_subset, word_format=word_format, extent=extent),
        ((str(owner), FB.sha256(owner)),),
    )
    decoder_owner = Path(inspect.getsourcefile(codec.program_decoder.decode)).resolve()
    composed = service(
        prerequisite=original,
        selected_contracts=selected,
        decoder=codec.program_decoder,
        decoder_sources=((str(decoder_owner), FB.sha256(decoder_owner)),),
        section=".weights",
        reader_max_bytes=len(payload),
    )
    reports = []
    with observed_facts(selected.target, json.loads(selected.facts_bytes)):
        started = time.perf_counter()
        with compilation_memory() as memory:
            baseline = FB.build(
                capture,
                target=selected.target,
                label="interpreter",
                submission=host,
                submission_sha256="unused",
                settings=settings,
                directory=root / "cache",
                offload=False,
                admission=composed,
            )
        assert baseline["host_admission"]["program"]["instruction_count"] == 64
        compile_seconds = time.perf_counter() - started
        checked = FB.build(
            capture,
            target=selected.target,
            label="interpreter",
            submission=host,
            submission_sha256="unused",
            settings=settings,
            directory=root / "cache",
            offload=False,
            admission=composed,
        )
        assert not baseline["reused"] and checked["reused"]
        assert baseline["elf_sha256"] == checked["elf_sha256"]
        assert checked["cache_admission"]["program"]["instruction_count"] == 64
        assert read_program_section(Path(checked["elf"]), ".weights", max_bytes=len(payload))[0] == payload
        initial_report = Path(baseline["host_admission"]["report_path"])
        initial_report_digest = FB.sha256(initial_report)
        initial_recipe = Path(baseline["elf"]).parent / "compilation_recipe.json"
        recipe = json.loads(initial_recipe.read_text())
        assert recipe["status"] == "completed"
        assert all(command["status"] == "returned" for command in recipe["commands"])
        arms = []
        for name, policy in (("original", original), ("checked", composed)):
            state, position = initial[::-1].copy(), 4
            values = []
            for step in range(3):
                data = (initial.astype(np.int16) + step).astype(np.int8)
                bound = dict(in0=data, in1=state.copy(), in2=np.array(position, np.int32))
                started = time.perf_counter()
                elf = FB.relink(checked, capture, bound, root / f"{name}-{step}", admission=policy)
                relink_seconds = time.perf_counter() - started
                console, wall = FB.run_host(elf, timeout_s=30)
                (elf.parent / "console.bin").write_bytes(console)
                actual_image, _ = read_program_section(elf, ".weights", max_bytes=len(payload))
                counters = observed_requests(console, actual_image, word_format)
                outputs, observed = FB.read(console, capture, checked["build_hash"])
                # Independent full-word reference; no use of the shared decoder/planner.
                expected = bytearray(state.tobytes())
                for offset in range(0, len(payload), word.size):
                    op, r, _, source, dest, _ = word.unpack_from(payload, offset)
                    value = (data.view(np.uint8) if r == 2 else expected)[source]
                    expected[dest] = (int(value) + op) % 256
                np.testing.assert_array_equal(outputs[0], np.frombuffer(expected, dtype=np.int8))
                np.testing.assert_array_equal(outputs[1], np.array(position + 1, np.int32))
                np.testing.assert_array_equal(outputs[2], data)
                np.testing.assert_array_equal(bound["in1"], state)
                state, position = outputs[0].copy(), position + 1
                values.append([output.tolist() for output in outputs])
                reports.append(
                    dict(
                        arm=name,
                        step=step,
                        elf_sha256=FB.sha256(elf),
                        elf_bytes=elf.stat().st_size,
                        elf_sections=FB.elf_sections(elf),
                        relink_wall_seconds=relink_seconds,
                        functional_simulator_process_wall_seconds=wall,
                        functional_host_instructions=int(observed["metrics"]["cycles"]),
                        actual_program_requests=counters,
                    )
                )
            arms.append(values)
        assert arms[0] == arms[1]
        # Removing the originally selected host policy cannot authorize a relink.
        with pytest.raises(FB.FullBuildError, match="currently selected"):
            FB.relink(checked, capture, bound, root / "missing-policy")
        # Tighter current hardware must refuse the actual cached image, without recompiling it.
        facts = json.loads(selected.facts_bytes)
        facts["facts"]["program_count"] = 63
        facts_path, software_path = tmp_path / "tighter-facts.json", tmp_path / "tighter-software.yaml"
        facts_path.write_text(json.dumps(facts))
        software = yaml.safe_load(selected.software_bytes)
        software["isa"]["facts_sha256"] = FB.sha256(facts_path)
        software_path.write_text(yaml.safe_dump(software))
        from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

        current = read_scheduling_contracts(facts_path, software_path, target=selected.target)
        tighter = service(
            prerequisite=original,
            selected_contracts=current,
            decoder=replace_decoder(codec.program_decoder, current.hardware_identity),
            decoder_sources=((str(decoder_owner), FB.sha256(decoder_owner)),),
            section=".weights",
            reader_max_bytes=len(payload),
        )
        with pytest.raises(FB.FullBuildError, match="admission refused"):
            FB.build(
                capture,
                target=current.target,
                label="interpreter",
                submission=host,
                submission_sha256="unused",
                settings=settings,
                directory=root / "cache",
                offload=False,
                admission=tighter,
            )
        assert FB.sha256(initial_report) == initial_report_digest
        refused = [
            json.loads(path.read_text()) for path in (root / "cache/cache-admission").glob("program-admission-*.json")
        ]
        refused = [report for report in refused if report["status"] == "refused"]
        assert len(refused) == 1 and refused[0]["selection"]["selected"] == current.record()
    report = dict(
        word_format=word_format,
        program_bytes=len(payload),
        initial_compilation_wall_seconds=compile_seconds,
        compilation_memory=memory,
        initial_compilation_recipe=dict(path=str(initial_recipe), sha256=FB.sha256(initial_recipe)),
        observed_parent_compile_and_link_commands=len(recipe["commands"]),
        compiler_command_scope="actual returned parent recipe commands; nested lowering records retained separately",
        accepted_report=dict(path=str(initial_report), sha256=initial_report_digest),
        baseline_and_checked_same_image=True,
        complete_outputs_and_state_equal=True,
        calls=reports,
        accounting=dict(
            execution_route="host scalar interpreter; no accelerator selected",
            static_decoded_program_instructions=64,
            executed_program_commands=[row["actual_program_requests"]["executed_program_commands"] for row in reports],
            requested_read_bytes=[row["actual_program_requests"]["requested_read_bytes"] for row in reports],
            requested_write_bytes=[row["actual_program_requests"]["requested_write_bytes"] for row in reports],
            physical_bus_bytes=None,
            packing_bytes=None,
            host_arithmetic_seconds=None,
            host_dispatch_seconds=None,
            accelerator_seconds=None,
            units="bytes, instructions and seconds as named",
            unknown_reason="no physical-bus, packing or component timing observer selected",
            observed_scope="opcode trace joined to actual linked program under one-byte interpreter effects; "
            "IO overhead included",
            policy="UNKNOWN costs supply no optimization ranking or speedup claim",
        ),
        timing_qualification="UNKNOWN",
    )
    (tmp_path / "whole_model.json").write_text(json.dumps(report, indent=2))
    record_property("whole_model", json.dumps(report))


def replace_decoder(decoder, identity):
    from dataclasses import replace

    return replace(decoder, configuration_identity=identity)
