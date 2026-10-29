"""An exact source plan narrows real package calls in a complete saved model.

This explicit neutral scalar endpoint tests the target-package transport. It
qualifies neither an accelerator nor arbitrary generated instruction effects.
"""

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_dense_source_catalog import compilation_memory
from test_native_offload_model import source
from test_source_choice_broker_model import reference
from test_source_model_api import setup
from xdsl.dialects import func

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower.device_build import DeviceRouting
from merlin.model_check import full_build as FB
from merlin.runtime.backends import base, spike_model
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.rtl.facts import observed_facts
from merlin.transforms.certificates import ir_text

NAMES = ("first", "second", "state", "position", "input", "weight0", "weight1")
POLICY = dict(compare="exact_int", operand_dtype="i8", accum_dtype="i32")
EXECUTOR_ISA = "rv64gc"
EXECUTOR_DRAM_BYTES = 64 << 20
EXECUTOR_HARTS = 1
_EXECUTION_OBSERVATION = ContextVar("neutral_package_execution_observation", default=None)


@contextmanager
def execution_observation(started, memory):
    observation = {}
    token = _EXECUTION_OBSERVATION.set((started, memory, observation))
    try:
        yield observation
    finally:
        _EXECUTION_OBSERVATION.reset(token)


def spike_extension():
    """The original scalar fixture needs no accelerator instruction extension."""
    return (), str(Path(__file__).resolve().parent)


def run_elf(elf, *, simulator, timeout, capture_bytes):
    assert simulator == "spike" and capture_bytes is True
    started = time.perf_counter()
    observed = _EXECUTION_OBSERVATION.get()
    assert observed is not None
    before, memory, record = observed
    assert not record
    record.update(
        compilation_and_preexecution_seconds=started - before,
        compilation_and_preexecution_memory=dict(memory),
        scope=(
            "entry to ordinary compile_saved_model through completed build, admission and pre-execution checks; "
            "memory is a 100ms sampled tree-RSS snapshot before simulation, not a guaranteed peak"
        ),
    )
    try:
        return spike_model.run_raw(
            Path(elf),
            harts=EXECUTOR_HARTS,
            mem_bytes=EXECUTOR_DRAM_BYTES,
            isa=EXECUTOR_ISA,
            timeout=timeout,
        )
    finally:
        record["functional_simulator_wall_seconds"] = time.perf_counter() - started
        record["execution_scope"] = (
            "functional simulator process including diagnostic operand/result logging; "
            "not accelerator latency or qualified whole-call timing"
        )
        record["qualified_whole_call_timing"] = "UNKNOWN"


def harness_build_recipe():
    """Declare this fixture's scalar ISA, entry and bounded synchronous call frame."""
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy

    runtime = spike_model._harness_dir()
    # The selected scalar runtime gives each hart 16 KiB (crt.S); reserve at
    # most one quarter for this tiny private leaf, leaving caller/runtime room.
    assert "li    t1, 16384" in (runtime / "crt.S").read_text()
    return HarnessBuildRecipe(
        compiler=spike_model._spike.gcc_path(),
        include_roots=(runtime,),
        support_sources=tuple(runtime / name for name in ("crt.S", "htif.c", "printf_min.c", "libc_min.c")),
        link_script=runtime / "model_link.ld",
        load_address=spike_model.DRAM_BASE,
        cflags=("-march=" + EXECUTOR_ISA, "-mabi=lp64d", "-mcmodel=medany", "-nostdlib", "-nostartfiles"),
        kernel_stack_frame=KernelStackFramePolicy(neutral_contract()["harness_abi"]["entry_symbol"], 4096),
    )


@contextmanager
def original_scalar_executor(target):
    """Register actual execution support for this explicit neutral endpoint only."""
    assert target == "neutral"
    previous = base._REGISTRY.get(target)
    base.register(base.BackendInfo(target, base.TargetClass.CPU, base.BackendKind.WHOLE_MODEL, __name__))
    try:
        yield
    finally:
        if previous is None:
            base._REGISTRY.pop(target)
        else:
            base.register(previous)


# This is the fixture's target-owned compiler, not a shared target decoder. The
# actual emitter parses its typed intermediate and produces an executable MAC.
DRIVER = (
    r"""
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, attr_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from merlin.llvmlower import toolchain
from merlin.llvmlower.device_shim import logical_argument_roles
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.schemas import validate_command_buffer

@irdl_op_definition
class Matmul(IRDLOperation):
    name = 'neutral_target.matmul'
    m = attr_def(builtin.IntegerAttr)
    k = attr_def(builtin.IntegerAttr)
    n = attr_def(builtin.IntegerAttr)
    buffer = attr_def(builtin.StringAttr)
    def verify_(self):
        dims = tuple(getattr(self, key).value.data for key in ('m', 'k', 'n'))
        if any(type(x) is not int or not 0 < x <= 32 for x in dims):
            raise ValueError('neutral target shape outside original fixture envelope')
        if geometry(json.loads(self.buffer.data)) != dims:
            raise ValueError('target shape does not match exact command buffer')

def geometry(cb):
    validate_command_buffer(cb)
    commands = cb['commands']
    assert [row['opcode'] for row in commands] == ['RES_PACK','MATMUL_RESIDENT','COMMIT','EVICT']
    packed, product, commit, evict = commands
    assert product.get('attributes', {}).get('accumulate', False) is False
    assert not commit.get('attributes', {}).get('epilogue')
    assert product['operands']['rhs'] == packed['operands']['dst']
    assert commit['operands']['src'] == product['operands']['dst']
    lhs = cb['tensors'][product['operands']['lhs']]
    rhs = cb['tensors'][packed['operands']['src']]
    assert lhs['dtype'] in ('int8','i8') and rhs['dtype'] in ('int8','i8')
    assert commit['attributes']['output_dtype'] in ('int32','i32')
    m,k = lhs['shape']; k1,n = rhs['shape']
    assert k == k1
    assert evict['operands']['handle'] == packed['operands']['dst']
    return m,k,n

def target(cb):
    m,k,n = geometry(cb)
    op = Matmul(attributes={**{key:builtin.IntegerAttr(value, builtin.i64)
        for key,value in zip(('m','k','n'),(m,k,n))},
        'buffer':builtin.StringAttr(json.dumps(cb, sort_keys=True))})
    module = builtin.ModuleOp([op]); module.verify()
    stream=io.StringIO(); Printer(stream=stream,print_generic_format=True).print_op(module)
    return stream.getvalue()+'\n'

def consume(text):
    context=Context(); context.load_dialect(builtin.Builtin)
    context.load_dialect(Dialect('neutral_target',[Matmul],[]))
    module=Parser(context,text).parse_module(); module.verify()
    ops=tuple(module.body.block.ops)
    assert len(ops)==1 and type(ops[0]) is Matmul
    op=ops[0]
    return json.loads(op.buffer.data), tuple(getattr(op,key).value.data for key in ('m','k','n'))

command, source, *rest = sys.argv[1:]
source=Path(source)
cb=parse_interface_mlir(source.read_text())
typed=target(cb)
if command=='parse':
    consume(typed)
elif command=='lower_interface_to_target':
    sys.stdout.write(typed)
elif command=='emit_analysis_bundle':
    output=Path(rest[0]); directory=output.parent
    consumed=directory/'consumed.target.mlir'; consumed.write_text(typed)
    actual,dims=consume(consumed.read_text())
    m,k,n=dims
    code=r"""
    + '"""'
    + r"""
#include <stdint.h>
extern void htif_puts(const char *);
extern void htif_putd(long);
void neutral_kernel(@PARAMETERS@) {
 htif_puts("MERLIN_PACKAGE_CALL @M@ @K@ @N@\n");
 htif_puts("MERLIN_PACKAGE_A");
 for(unsigned i=0;i<@M@*@K@;i++){htif_puts(" ");htif_putd(a[i]);}htif_puts("\n");
 htif_puts("MERLIN_PACKAGE_B");
 for(unsigned i=0;i<@K@*@N@;i++){htif_puts(" ");htif_putd(b[i]);}htif_puts("\n");
 for(unsigned i=0;i<@M@;i++)for(unsigned j=0;j<@N@;j++){
  uint32_t acc=0;
  for(unsigned t=0;t<@K@;t++)acc+=(uint32_t)((int32_t)a[i*@K@+t]*(int32_t)b[t*@N@+j]);
  y[i*@N@+j]=(int32_t)acc;
 }
 htif_puts("MERLIN_PACKAGE_Y");
 for(unsigned i=0;i<@M@*@N@;i++){htif_puts(" ");htif_putd(y[i]);}htif_puts("\n");
}
"""
    + '"""'
    + r"""
    for key,value in zip(('@M@','@K@','@N@'),dims): code=code.replace(key,str(value))
    roles,reason=logical_argument_roles(source.read_text())
    assert roles is not None,reason
    parameters={'lhs':'const int8_t *a','rhs':'const int8_t *b','out':'int32_t *y'}
    code=code.replace('@PARAMETERS@',','.join(parameters[role] for role in roles))
    c=directory/'neutral_kernel.c'; c.write_text(code)
    llvm=directory/'neutral_kernel.ll'
    argv=[str(toolchain.clang()),'--target=riscv64-unknown-elf','-march=rv64gc',
          '-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin',
          '-S','-emit-llvm',str(c),'-o',str(llvm)]
    subprocess.run(argv,check=True,capture_output=True)
    translated=subprocess.run([str(toolchain.llvm_install()/'bin'/'mlir-translate'),
       '--import-llvm',str(llvm)],check=True,capture_output=True,text=True)
    output.write_text(json.dumps(actual))
    (directory/'consumption.json').write_text(json.dumps(dict(
        schema='neutral.target-consumption.v1',shape=dims,logical_argument_roles=roles,
        files={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (source,consumed,c,llvm)},c_command=argv,
        primitive_input_payload_bytes=m*k+k*n,primitive_output_payload_bytes=4*m*n,
        static_command_buffer_commands=len(actual['commands']),
        requested_transfer_bytes='UNKNOWN',total_host_traffic_bytes='UNKNOWN',
        scope='logical scalar input/output payload, not total host traffic or bus accesses')))
    sys.stdout.write(translated.stdout)
elif command in ('emit_command_buffer','emit_target_artifact','lower_target_to_llvm'):
    raise ValueError('bundled typed target consumer is required')
else: raise ValueError(command)
"""
)


def complete_source(m=3, k=5):
    """Three dependent steps, represented in the existing flat source domain."""
    one = source(m, k, "i8")
    original = next(op for op in one.body.block.ops if isinstance(op, func.FuncOp))
    module = one.clone()
    entry = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    block = entry.body.block
    for op in tuple(block.ops):
        block.erase_op(op, safe_erase=False)
    returned = None
    for _ in range(3):
        mapping = dict(zip(original.body.block.args, block.args, strict=True))
        if returned is not None:
            mapping[original.body.block.args[3]] = returned[2]
            mapping[original.body.block.args[4]] = returned[3]
        for op in original.body.block.ops:
            if isinstance(op, func.ReturnOp):
                returned = tuple(mapping[v] for v in op.arguments)
            else:
                block.add_op(op.clone(value_mapper=mapping))
    block.add_op(func.ReturnOp(*returned))
    module.verify()
    return module


def package(directory):
    directory.mkdir()
    (directory / "driver.py").write_text(DRIVER)
    names = (
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
        "lower_target_to_llvm",
        "emit_analysis_bundle",
    )
    commands = {
        name: dict(
            argv=[
                sys.executable,
                "-B",
                "driver.py",
                name,
                "{input_mlir}",
                *(["{output_json}"] if name in ("emit_command_buffer", "emit_analysis_bundle") else []),
            ]
        )
        for name in names
    }
    (directory / "manifest.yaml").write_text(
        json.dumps(
            dict(
                artifact_type="mlir_oot_target_backend",
                target="neutral",
                language="python",
                authoring=dict(mode="hand_curated"),
                integrity_exempt=False,
                entrypoints=dict(tool="driver.py"),
                commands=commands,
            )
        )
    )
    return directory


def neutral_contract():
    return dict(
        name="neutral",
        target="neutral",
        endpoint_kind="upstream_target",
        compute_units=[
            dict(
                name="scalar_mac",
                kind="scalar",
                dtypes=["int8"],
                ops=["matmul"],
                accumulate=[{"in": "int8", "weight": "int8", "acc": "i32"}],
                semantic_capabilities=[dict(family="contraction", dtypes=["int8"], ranks=[2])],
            )
        ],
        encoding=dict(semantic_class={"matmul": "COMPUTE"}, corpus_issue_order=["COMPUTE"]),
        harness_abi=dict(entry_symbol="neutral_kernel", kernel_abi_version=2),
    )


def admit_neutral_elf(*, elf, evidence_root):
    """Inspect actual linked architecture/sections, not a numerical certificate."""
    from merlin.llvmlower.toolchain import llvm_nm

    with Path(elf).open("rb") as stream:
        header = stream.read(64)
    # ELF gABI fields, not accelerator instruction encodings.
    assert len(header) == 64 and header[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(header[18:20], "little") == 243  # EM_RISCV.
    nm = llvm_nm()
    command = [str(nm), "--defined-only", "--extern-only", "--format=posix", str(elf)]
    symbols = subprocess.run(command, check=True, capture_output=True, text=True)
    names = {line.split()[0] for line in symbols.stdout.splitlines() if line.split()}
    assert "_start" in names and "_mlir_ciface_forward" in names
    assert any("neutral" in name and "kernel" in name for name in names)
    sections = FB.elf_sections(Path(elf))
    assert sections[".text"] > 0
    evidence_root.mkdir(parents=True, exist_ok=True)
    report = evidence_root / "neutral-linked-image.json"
    result = dict(
        status="accepted",
        elf_sha256=sha256_file(Path(elf)),
        sections=sections,
        symbol_observation=dict(command=command, tool_sha256=sha256_file(nm), stdout=symbols.stdout),
        scope="reviewed neutral fixture ELF architecture and linked-entry custody; not arbitrary machine semantics",
    )
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": sha256_file(report)}


class PreparedSnapshot:
    def __init__(self):
        self.paths = []

    def save(self, source_path, work):
        # The ordinary rewrite mutates source_path in place. Retain the exact
        # original pre-rewrite bytes before returning that same ordinary path.
        directory = Path(work)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / "original.prepared.mlir"
        snapshot.write_bytes(Path(source_path).read_bytes())
        self.paths.append(snapshot)
        return Path(source_path)


def original_selector(_contraction):
    return True


def exact_plan(owner):
    from merlin.llvmlower.offload_plan import realize_offload_plan
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan

    # Every second exact contraction is selected. Identical shapes cannot
    # distinguish these six sites or establish this source selection.
    accepted = owner.original_ordinals
    assert len(accepted) == 6
    alternatives = tuple(
        replace(
            owner.source.cpu_alternatives[owner.source.operation_ordinals.index(index)],
            id=f"package_{index}",
            placement=owner.routing.device,
            implementation=owner.implementation_identity,
        )
        for index in accepted
    )
    chosen = {alt.nodes[0]: alt for alt in alternatives[1::2]}
    plan = GlobalPlan(
        tuple(chosen.get(i, alt) for i, alt in enumerate(owner.source.cpu_alternatives)),
        (),
        CycleInterval.unknown("no target execution timing qualification"),
    )
    return realize_offload_plan(owner.source, plan, device_alternatives=alternatives, context=owner.context)


def observed_calls(console, trace, *, narrowed):
    lines = [line.split() for line in console.splitlines() if line.startswith(b"MERLIN_PACKAGE_")]
    wanted = [item for item in trace if not narrowed or item[0] == 1]
    assert len(lines) == 4 * len(wanted)
    rows = []
    for call, (_, a, b, y) in zip(range(0, len(lines), 4), wanted, strict=True):
        header, left, right, result = lines[call : call + 4]
        assert header == [b"MERLIN_PACKAGE_CALL", *[str(v).encode() for v in (a.shape[0], a.shape[1], b.shape[1])]]
        for values, prefix, expected in (
            (left, b"MERLIN_PACKAGE_A", a),
            (right, b"MERLIN_PACKAGE_B", b),
            (result, b"MERLIN_PACKAGE_Y", y),
        ):
            assert values[0] == prefix
            np.testing.assert_array_equal(np.array(list(map(int, values[1:]))).reshape(expected.shape), expected)
        rows.append(
            dict(
                shape=[a.shape[0], a.shape[1], b.shape[1]],
                lhs=list(map(int, left[1:])),
                rhs=list(map(int, right[1:])),
                result=list(map(int, result[1:])),
            )
        )
    return rows


def test_exact_source_plan_narrows_real_target_package_calls(tmp_path, monkeypatch, record_property):
    from merlin.llvmlower.device_offload_selection import bind_device_offload_source

    model, _, _, root, options = setup(tmp_path, monkeypatch, (3, 5, 5), (2, 3, 4), 1, False)
    assert options["target"] == model[0].target == "neutral"
    values, expected_history, trace = reference(3, 5, 1)
    expected = expected_history[-1]
    bundle = options["capture"]
    save_file(dict(w0=values[0], w1=values[1]), bundle / "weights.safetensors")
    np.savez(bundle / "inputs.npz", in0=values[2], in1=values[3], in2=values[4])
    (bundle / "model.mlir").write_text(ir_text(complete_source()))
    np.save(bundle / "golden.npy", expected[0])
    np.savez(bundle / "goldens.npz", **dict(zip(NAMES, expected, strict=True)))
    (bundle / "output_order.json").write_text(json.dumps(NAMES))
    artifacts = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p))
        for p in bundle.iterdir()
        if p.name != "capture_receipt.json"
    }
    (bundle / "capture_receipt.json").write_text(
        json.dumps(dict(schema="m2m.capture-receipt.v1", materialized_abi=dict(complete=True), artifacts=artifacts))
    )
    contract = tmp_path / "neutral_contract.json"
    contract.write_text(json.dumps(neutral_contract()))
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract))
    providers = tmp_path / "original-runtime-only"
    providers.mkdir()
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(providers))
    pkg = package(tmp_path / "target_package")
    snapshot = PreparedSnapshot()
    routing = DeviceRouting(
        "neutral",
        pkg,
        "i8",
        "i32",
        select=original_selector,
        numeric_policy=POLICY,
        prepared_transform=snapshot.save,
        linked_elf_admission=LinkedElfAdmissionService(
            "neutral", admit_neutral_elf, ((str(Path(__file__).resolve()), sha256_file(Path(__file__))),)
        ),
    )
    observations = []
    selected = None
    with original_scalar_executor(model[0].target), observed_facts(model[0].target, json.loads(model[0].facts_bytes)):
        for name, narrowed in (("original", False), ("explicit_candidate", True)):
            if narrowed:
                prepared = snapshot.paths[-1]
                owner = bind_device_offload_source(
                    mq.parse(prepared.read_text()),
                    routing=routing,
                    configuration=(
                        ("index_bits", 64),
                        ("runtime", "spike-functional-rv64gc"),
                        ("executor", f"{__name__}.run_elf"),
                        ("harts", EXECUTOR_HARTS),
                        ("dram_bytes", EXECUTOR_DRAM_BYTES),
                    ),
                    source_pins=(
                        (str(contract), sha256_file(contract)),
                        (str(bundle / "model.mlir"), sha256_file(bundle / "model.mlir")),
                        (str(Path(__file__).resolve()), sha256_file(Path(__file__))),
                        (str(prepared), sha256_file(prepared)),
                    ),
                )
                selected = owner.select(exact_plan(owner), explicit=True)
                chosen = replace(routing, offload_selection=selected)
            else:
                chosen = routing
            started = time.perf_counter()
            with compilation_memory() as memory, execution_observation(started, memory) as phases:
                receipt = compile_saved_model(**options, output=root / name, device=chosen)
            elapsed = time.perf_counter() - started
            assert receipt["status"] == "verified_complete_output"
            result = receipt["output"]
            console = Path(result["console"]).read_bytes()
            actual, metadata = FB.read(console, bundle, result["build_hash"])
            assert len(actual) == len(expected) == 7
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(got, want)
            calls = observed_calls(console, trace, narrowed=narrowed)
            elf = Path(result["elf"])
            assert result["elf_sha256"] == sha256_file(elf)
            sidecar = json.loads(Path(result["device_sidecar"]["path"]).read_text())
            static_buffers = []
            for consumption_path in sorted((root / name / "build").rglob("consumption.json")):
                consumption = json.loads(consumption_path.read_text())
                buffer_path = consumption_path.with_name("command_buffer.json")
                commands = json.loads(buffer_path.read_text())["commands"]
                assert len(commands) == consumption["static_command_buffer_commands"]
                static_buffers.append(
                    dict(path=str(buffer_path), sha256=sha256_file(buffer_path), commands=len(commands))
                )
            assert len(static_buffers) == 1
            observations.append(
                dict(
                    name=name,
                    receipt=receipt,
                    calls=calls,
                    sidecar=sidecar,
                    compile_and_execute_seconds=elapsed,
                    execution_phases=phases,
                    sampled_compilation_and_execution_memory=memory,
                    original_prepared_snapshot=dict(
                        path=str(snapshot.paths[-1]), sha256=sha256_file(snapshot.paths[-1])
                    ),
                    elf_bytes=elf.stat().st_size,
                    allocated_sections=FB.elf_sections(elf),
                    functional_forward_instructions=metadata["metrics"]["cycles"],
                    primitive_input_payload_bytes=sum(len(row["lhs"]) + len(row["rhs"]) for row in calls),
                    primitive_output_payload_bytes=sum(4 * len(row["result"]) for row in calls),
                    static_command_buffers=static_buffers,
                    requested_transfer_bytes="UNKNOWN",
                    total_host_traffic_bytes="UNKNOWN",
                    source_selection=None if selected is None else selected.record(),
                    scope="neutral scalar target package; total host traffic and accelerator latency UNKNOWN",
                )
            )
    assert [len(row["calls"]) for row in observations] == [6, 3]
    path = tmp_path / "pair.json"
    path.write_text(
        json.dumps(
            dict(
                schema="neutral.device-package-pair.v1",
                observations=observations,
                capture=str(bundle),
                inputs_sha256=sha256_file(bundle / "inputs.npz"),
                contract=dict(path=str(contract), sha256=sha256_file(contract)),
                numerical_policy=POLICY,
                source_route_context=json.loads(selected.owner._record),
                measurement_scope=(
                    "compile_and_execute includes qualification/preparation/link/console; "
                    "functional instructions are not cycles"
                ),
            ),
            indent=2,
        )
    )
    record_property("pair", str(path))


def test_unrolled_source_keeps_six_same_shape_contractions_and_all_results():
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul

    module = complete_source()
    recognized = []
    for operation in module.walk():
        try:
            recognized.append(recognize_signed_integer_matmul(operation, ("i8", "i8", "i32")))
        except ValueError:
            pass
    assert len(recognized) == 6
    function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    assert len(function.body.block.last_op.arguments) == 7
    compile(DRIVER, "neutral_driver.py", "exec")


def test_original_prepared_snapshot_survives_in_place_routing(tmp_path):
    prepared = tmp_path / "ordinary.mlir"
    original = ir_text(complete_source()).encode()
    prepared.write_bytes(original)
    saved = PreparedSnapshot()
    assert saved.save(prepared, tmp_path / "snapshot") == prepared
    prepared.write_text("module {}\n")
    assert saved.paths[0] != prepared and saved.paths[0].read_bytes() == original


def test_original_scalar_executor_declares_no_extension_and_restores_registry():
    target = "neutral"
    previous = base._REGISTRY.get(target)
    with original_scalar_executor(target):
        selected = base._REGISTRY[target]
        assert selected.module == __name__ and selected.target_class == base.TargetClass.CPU
        assert selected.kind == base.BackendKind.WHOLE_MODEL
        flags, directory = spike_extension()
        assert flags == () and Path(directory).is_dir()
        with pytest.raises(AssertionError):
            run_elf(Path("not-opened.elf"), simulator="unselected", timeout=1, capture_bytes=True)
    assert base._REGISTRY.get(target) is previous


def test_neutral_typed_consumer_checks_public_interface_geometry(tmp_path, monkeypatch):
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    contract = tmp_path / "neutral_contract.json"
    contract.write_text(json.dumps(neutral_contract()))
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract))
    binding = _mesh_tile_binding("neutral", "i8", "i32", numeric_policy=POLICY)
    _, interface = CS.build(
        dict(
            name="neutral_test",
            op="matmul",
            kind="op",
            source_role="mesh_tile_synthesized",
            source_reference="original neutral fixture geometry",
            M=3,
            K=5,
            N=5,
        ),
        binding,
    )
    namespace = {}
    exec(DRIVER.split("command, source, *rest = sys.argv[1:]")[0], namespace)
    cb = namespace["parse_interface_mlir"](interface)
    text = namespace["target"](cb)
    recovered, dims = namespace["consume"](text)
    assert recovered == cb and dims == (3, 5, 5)
    assert '"neutral_target.matmul"' in text
    wrong = text.replace("m = 3 : i64", "m = 2 : i64")
    assert wrong != text
    with pytest.raises(ValueError, match="target shape does not match"):
        namespace["consume"](wrong)
