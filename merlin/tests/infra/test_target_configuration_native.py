"""Selected typed target IR actually reaches native code and recurrent state.

This is a neutral package/translation/native mechanism test, not Gemmini or
accelerator qualification and not an ordinary compile_saved_model experiment.
"""

import ctypes
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_selected_schedule_contracts import inputs

from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P

COUNT = 47
QUALIFICATION = "independent-neutral-native-persistent-settings-v1"
CONSUMER = dict(
    schema="merlin.target_ir_consumer.v1",
    command="emit_target_bundle",
    input="selected_target_mlir",
    output="command_buffer_and_target_artifact",
)
COMMANDS = (
    ("set", 0, 3, 0),
    ("set", 0, 3, 0),
    ("set", 1, 2, 0),
    ("set", 1, 2, 0),
    ("work", 0, 0, 23),
    ("reset", 0, 0, 0),
    ("set", 0, 3, 0),
    ("set", 0, 3, 0),
    ("work", 0, 23, 13),
    ("opaque", 0, 0, 0),
    ("set", 0, 3, 0),
    ("set", 0, 3, 0),
    ("set", 1, 2, 0),
    ("set", 1, 2, 0),
    ("work", 0, 36, 11),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_text(bits):
    lines = ['"builtin.module"() ({']
    for i, (op, slot, a, b) in enumerate(COMMANDS):
        for name, value, width in (("slot", slot, 8), ("a", a, bits), ("b", b, bits)):
            lines.append(f'%{name}{i} = "arith.constant"() <{{value = {value} : i{width}}}> : () -> i{width}')
        lines.append(f'"neutral_isa.{op}"(%slot{i}, %a{i}, %b{i}) : (i8, i{bits}, i{bits}) -> ()')
    return "\n".join((*lines, "}) : () -> ()", ""))


DRIVER = r"""
import hashlib
import json
import sys
from pathlib import Path
from xdsl.context import Context
from xdsl.dialects import arith, builtin
from xdsl.parser import Parser
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.targetgen.isa_dialect_gen import build_dialect

command, source, *rest = sys.argv[1:]
source = Path(source)
if command == 'parse':
    assert source.read_text().strip() == 'module {}'
elif command == 'lower_interface_to_target':
    sys.stdout.write(Path('original.target.mlir').read_text())
elif command in ('emit_target_artifact', 'lower_target_to_llvm'):
    raise RuntimeError('legacy interface emission must never bypass selected target IR')
elif command == 'emit_target_bundle':
    paths = json.loads(Path('selection.json').read_text())
    selected = read_scheduling_contracts(*paths, target='neutral')
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, build_dialect(selected.isa)):
        context.load_dialect(dialect)
    module = Parser(context, source.read_text()).parse_module()
    module.verify()
    actual = []
    for op in module.body.block.ops:
        if op.name == 'arith.constant':
            continue
        assert op.name.startswith(selected.isa.namespace + '.')
        values = []
        for value in op.operands:
            assert value.owner.name == 'arith.constant'
            values.append(value.owner.value.value.data)
        actual.append((op.name.split('.')[1], *values))
    buffer = json.loads(Path('buffer.json').read_text())
    Path(rest[0]).write_text(json.dumps(buffer))
    source.with_name('consumption.json').write_text(json.dumps(dict(
        input_path=str(source.resolve()), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actual_commands=actual,
    )))
    lines = ['module {',
        'llvm.func @neutral_set(i64, i64)', 'llvm.func @neutral_reset(i64)',
        'llvm.func @neutral_opaque()',
        'llvm.func @neutral_work(!llvm.ptr, !llvm.ptr, i64, i64)',
        'llvm.func @neutral_kernel(%input: !llvm.ptr, %output: !llvm.ptr) {']
    for i, (name, slot, a, b) in enumerate(actual):
        for field, value in (('slot', slot), ('a', a), ('b', b)):
            lines.append(f'%{field}{i} = llvm.mlir.constant({value} : i64) : i64')
        if name == 'set':
            lines.append(f'llvm.call @neutral_set(%slot{i}, %a{i}) : (i64, i64) -> ()')
        elif name == 'reset':
            lines.append(f'llvm.call @neutral_reset(%slot{i}) : (i64) -> ()')
        elif name == 'opaque':
            lines.append('llvm.call @neutral_opaque() : () -> ()')
        elif name == 'work':
            lines.append(f'llvm.call @neutral_work(%input, %output, %a{i}, %b{i})'
                         ' : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()')
        else:
            raise ValueError('unqualified native fixture primitive')
    sys.stdout.write('\n'.join((*lines, 'llvm.return', '}', '}', '')))
else:
    raise ValueError(command)
"""

RUNTIME = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static uint64_t settings[2], counts[5], pulses;
static int faulty;
void neutral_set(uint64_t slot,uint64_t value) {
 if(slot>=2)__builtin_trap(); settings[slot]=value; counts[0]++; pulses+=faulty;
}
void neutral_reset(uint64_t slot) {
 if(slot>=2)__builtin_trap(); settings[slot]=0; counts[2]++;
}
void neutral_opaque(void) {settings[0]=17;settings[1]=19;counts[3]++;}
void neutral_work(const uint8_t *input,uint8_t *output,uint64_t start,uint64_t count) {
 if(start>47 || count>47-start)__builtin_trap(); counts[1]++;
 for(uint64_t i=start;i<start+count;i++) output[i]=(uint8_t)(input[i]*settings[1]+settings[0]+pulses);
 counts[4]+=count;
}
extern void neutral_kernel(const uint8_t*,uint8_t*);
void model(const uint8_t *input,uint8_t *history,uint8_t *state,uint32_t *position,
           uint64_t *observation,uint64_t *final_settings,int mutate) {
 uint8_t mixed[47];faulty=mutate;pulses=0;settings[0]=91;settings[1]=73;
 memset(counts,0,sizeof(counts));
 for(unsigned step=0;step<3;step++) {
  for(unsigned i=0;i<47;i++)mixed[i]=(uint8_t)(input[i]+state[i]+*position);
  neutral_kernel(mixed,history+step*47);
  memcpy(state,history+step*47,47);(*position)++;
 }
 memcpy(observation,counts,sizeof(counts));memcpy(final_settings,settings,sizeof(settings));
}
"""


def package(tmp_path, selected, bits, *, consumer=True):
    directory = tmp_path / "package"
    directory.mkdir()
    (directory / "original.target.mlir").write_text(target_text(bits))
    (directory / "selection.json").write_text(json.dumps([str(selected.facts_path), str(selected.software_path)]))
    (directory / "driver.py").write_text(DRIVER)
    commands = {
        name: dict(argv=[sys.executable, "-B", "driver.py", name, "{input_mlir}", *extra])
        for name, extra in (
            ("parse", []),
            ("lower_interface_to_target", []),
            ("emit_command_buffer", ["{output_json}"]),
            ("emit_target_artifact", []),
            ("lower_target_to_llvm", []),
            ("emit_target_bundle", ["{output_json}"]),
        )
    }
    manifest = dict(
        artifact_type="mlir_oot_target_backend",
        target="neutral",
        language="python",
        authoring=dict(mode="hand_curated"),
        integrity_exempt=False,
        entrypoints=dict(tool="driver.py"),
        commands=commands,
    )
    if consumer:
        manifest["target_ir_consumer"] = CONSUMER
    (directory / "manifest.yaml").write_text(json.dumps(manifest))
    (directory / "buffer.json").write_text(
        json.dumps(
            dict(
                abi_version="0.1",
                target="neutral",
                commands=[],
                tensors={
                    name: dict(shape=[COUNT], dtype="i8", role=role)
                    for name, role in (("input", "input"), ("output", "output"))
                },
                kernel_abi=dict(
                    kind="whole_program",
                    args=[dict(tensor="input", access="read"), dict(tensor="output", access="write")],
                    outputs=["output"],
                ),
            )
        )
    )
    return P.load_package(directory)


def execute(library, *, faulty=False):
    library.model.argtypes = (
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    )
    library.model.restype = None
    image = [(31 * i + 253) % 256 for i in range(COUNT)]
    initial = [(3 * i + 1) % 256 for i in range(COUNT)]
    data = (ctypes.c_uint8 * COUNT)(*image)
    state = (ctypes.c_uint8 * COUNT)(*initial)
    history = (ctypes.c_uint8 * (3 * COUNT))()
    position = ctypes.c_uint32(7)
    counters = (ctypes.c_uint64 * 5)()
    settings = (ctypes.c_uint64 * 2)()
    library.model(data, history, state, ctypes.byref(position), counters, settings, ctypes.c_int(faulty))
    expected_history = []
    expected_state = initial
    for p in range(7, 10):
        expected_state = [(((x + s + p) % 256) * 2 + 3) % 256 for x, s in zip(image, expected_state, strict=True)]
        expected_history.extend(expected_state)
    actual = [list(history), list(state), [position.value], list(data), list(settings)]
    expected = [expected_history, expected_state, [10], image, [3, 2]]
    return actual, expected, list(counters)


@pytest.mark.parametrize("width,endian", [(4, "<"), (8, ">")])
def test_selected_target_ir_drives_real_translation_and_complete_native_state(tmp_path, record_property, width, endian):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.llvmlower import toolchain

    translator = Path(toolchain.mlir_translate())
    compiler = Path(toolchain.clang())
    if not translator.is_file() or not compiler.is_file():
        pytest.skip("native LLVM translator and C compiler required")
    facts = tmp_path / "capabilities"
    facts.mkdir()
    selected, *_ = inputs(facts, width=width, endian=endian)
    pkg = package(tmp_path, selected, width * 8)
    interface = tmp_path / "input.mlir"
    interface.write_text("module {}\n")
    runtime = tmp_path / "runtime.c"
    runtime.write_text(RUNTIME)
    qualification = dict(
        identity=QUALIFICATION,
        runtime_source_sha256=sha(runtime),
        hardware_identity=selected.hardware_identity,
        scope="independently reviewed neutral C persistent setter/read/reset/opaque semantics",
        limitations="not a physical target qualification; counters are separately observed side effects",
    )
    (tmp_path / "original_qualification.json").write_text(json.dumps(qualification, indent=2))
    results = []
    for enabled in (False, True):
        work = tmp_path / ("optimized" if enabled else "baseline")
        selection = ConfigurationIRSelection(selected, QUALIFICATION, enabled=enabled)
        started = time.perf_counter()
        _, artifact = CC.lower_interface(
            pkg, interface, work, contract=None, timeout=30, target_configuration=selection
        )
        assert (work / "lowered.target.mlir").read_text() == target_text(width * 8)
        actual = json.loads((work / "consumption.json").read_text())
        assert actual["input_path"] == str((work / "selected.target.mlir").resolve())
        assert actual["input_sha256"] == sha(work / "selected.target.mlir")
        setters = sum(row[0] == "set" for row in actual["actual_commands"])
        assert setters == (5 if enabled else 10)
        assert artifact.count("llvm.call @neutral_set(") == setters
        mlir = work / "lowered.llvm.mlir"
        llvm = work / "kernel.ll"
        obj = work / "kernel.o"
        elf = work / "model.so"
        commands = [
            [str(translator), "--mlir-to-llvmir", str(mlir), "-o", str(llvm)],
            [str(compiler), "-O2", "-fPIC", "-c", str(llvm), "-o", str(obj)],
            [str(compiler), "-O2", "-shared", "-fPIC", str(obj), str(runtime), "-o", str(elf)],
        ]
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)
        compilation_seconds = time.perf_counter() - started
        # Reopening and executing the linked object establishes that target IR
        # was consumed, rather than merely written beside interface emission.
        assert elf.read_bytes().startswith(b"\x7fELF")
        library = ctypes.CDLL(str(elf))
        actual_outputs, expected, counters = execute(library)
        assert actual_outputs == expected
        assert counters == [3 * setters, 9, 3, 3, 3 * COUNT]
        mutant, _, _ = execute(library, faulty=True)
        assert mutant != expected, "non-idempotent setter mutation escaped numerical gate"
        selection.revalidate()
        receipt = json.loads((work / "target_configuration.json").read_text())
        results.append(
            dict(
                enabled=enabled,
                complete_outputs=actual_outputs,
                counters=counters,
                executed_target_operations=sum(counters[:4]),
                logical_kernel_reads_bytes=counters[4],
                logical_kernel_writes_bytes=counters[4],
                compilation_seconds=compilation_seconds,
                compilation_scope="package parse/lower/shared pass/emission, translation, object compilation and link",
                peak_memory_bytes=None,
                physical_transfer_bytes=None,
                host_execution_seconds=None,
                accelerator_execution_seconds=None,
                target_sha256=actual["input_sha256"],
                original_target_sha256=sha(work / "lowered.target.mlir"),
                emitted_mlir_sha256=sha(mlir),
                llvm_sha256=sha(llvm),
                object_sha256=sha(obj),
                elf_sha256=sha(elf),
                elf_bytes=elf.stat().st_size,
                commands=commands,
                receipt=receipt,
                compiler_sha256=sha(compiler),
                translator_sha256=sha(translator),
                original_qualification=qualification,
                physical_device_support=False,
                timing_seconds=None,
                scope="neutral native recurrent model through ordinary package lowering",
            )
        )
    assert results[0]["complete_outputs"] == results[1]["complete_outputs"]
    assert results[0]["original_target_sha256"] == results[1]["original_target_sha256"]
    (tmp_path / "pair.json").write_text(json.dumps(results, indent=2))
    record_property("actual_target_ir_native_pair", json.dumps(results))


def test_selected_cleanup_without_consuming_package_protocol_refuses(tmp_path):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

    facts = tmp_path / "capabilities"
    facts.mkdir()
    selected, *_ = inputs(facts)
    pkg = package(tmp_path, selected, 32, consumer=False)
    interface = tmp_path / "input.mlir"
    interface.write_text("module {}\n")
    with pytest.raises(P.CertFailure, match="merlin.target_ir_consumer.v1"):
        CC.lower_interface(
            pkg,
            interface,
            tmp_path / "output",
            contract=None,
            timeout=30,
            target_configuration=ConfigurationIRSelection(selected, QUALIFICATION),
        )
    assert not (tmp_path / "output" / "consumption.json").exists()
