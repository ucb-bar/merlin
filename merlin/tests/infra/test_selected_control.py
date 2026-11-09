"""Actual compiled selected-region control, exact artifacts and refusal boundaries."""

import ctypes
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.compiler.backend import layout
from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    admit_original_cpu,
    admit_proposal,
    source_numerical_identity,
)
from merlin.compiler.build.image import CImageLink
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext

SOURCE = """module {func.func @forward(%x:tensor<2x3xf32>)->tensor<2x3xf32> {
 %a=tensor.collapse_shape %x [[0,1]] : tensor<2x3xf32> into tensor<6xf32>
 %b=tensor.expand_shape %a [[0,1]] output_shape [2,3] : tensor<6xf32> into tensor<2x3xf32>
 func.return %b:tensor<2x3xf32>
}}"""


def selected(tmp_path):
    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump({
        "schema": "merlin.execution_topology.v1",
        "resources": [{"id": "cpu0", "kind": "cpu", "features": [], "shared_group": "cpu0"}],
        "memory_domains": [{"id": "ram", "capacity_bytes": 192, "alignment_bytes": 64}],
        "access": [{"resource": "cpu0", "domain": "ram", "read": True, "write": True,
                    "addressability": "shared", "visibility": "coherent", "completion": "synchronous"}],
    }))
    hardware = select_topology(path)
    source = bind_source(parse_mlir_text(SOURCE), entry="forward", context=PipelineContext(), graph_mode="ssa")
    def approve_layout(bound, proposal):
        assert len(proposal.alternative.nodes) == 1
        assert bound.operations[proposal.alternative.nodes[0]].name in {"tensor.collapse_shape", "tensor.expand_shape"}
        # This check admits the source family. Compilation output and actual C
        # behavior are checked separately; this is not accelerator qualification.
        bound.module.verify()
    candidates = []
    for index, original in enumerate(source.cpu_alternatives):
        alternative = replace(original, id=f"layout_{index}", implementation="dense-layout", placement="cpu0")
        claim = SourceProposal(alternative, layout.name, "cpu0", "ram", "logical_tensor_v1",
                               source.stamp.source_sha256, (source.operation_ordinals[index],),
                               source_numerical_identity(source, (index,)))
        candidates.append(admit_proposal(source, claim, hardware, validate_implementation=approve_layout))
    return source, hardware, tuple(candidates)


OPTIONS = api.CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"})


def compile_control(tmp_path, *, mutate=None, image=None, options=OPTIONS):
    source, _, candidates = selected(tmp_path)
    provider = layout
    if mutate:
        def changed(request):
            result = layout.compile(request)
            mutate(result)
            return result
        provider = SimpleNamespace(name=layout.name, validate=layout.validate, compile=changed)
    backend = PlannedCompiler(select_execution(source, candidates), {layout.name: RegionBackend(provider, "kernel")},
                              image=image)
    return api.compile(ir_text(source.module), backend=backend, output=tmp_path / "compiled", options=options)


def test_selected_c_control_links_and_executes_preserving_bits_and_original_input(tmp_path):
    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC required for actual C control execution")
    result = compile_control(tmp_path)
    report = json.loads((result.directory / "control/control.json").read_text())
    assert result.metadata["stage"] == "c_dispatch" and result.deployment is None
    assert report["total_tensor_bytes"] == 152 and report["external_bytes"] == 128
    assert len(report["calls"]) == 2 and {b["resource"] for b in report["bindings"]} == {"cpu0"}
    assert report["backend_correctness"] == "not_established" and not report["executed"]
    library = tmp_path / "model.so"
    subprocess.run([gcc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    *(str(result.directory / path) for path in report["link_inputs"]), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    class State(ctypes.Structure):
        _fields_ = [("active", ctypes.c_int), ("completed", ctypes.c_size_t), ("kernel_status", ctypes.c_int)]
    class Buffer(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("bytes", ctypes.c_size_t)]
    run = ctypes.CDLL(str(library)).model_run
    run.argtypes = [ctypes.POINTER(State), ctypes.c_void_p, ctypes.c_size_t,
                    ctypes.POINTER(Buffer), ctypes.c_size_t, ctypes.c_void_p]
    run.restype = ctypes.c_int
    arena = ctypes.create_string_buffer(128)
    address = (ctypes.addressof(arena) + 63) & ~63
    bits = (0x7F800001, 0x7FC12345, 0x80000000, 0x7F800000, 1, 0xFF800001)
    incoming = (ctypes.c_uint32 * 6)(*bits)
    outgoing = (ctypes.c_uint32 * 6)(*[0xDEADBEEF] * 6)
    buffers = (Buffer * 2)(Buffer(ctypes.addressof(incoming), 24), Buffer(ctypes.addressof(outgoing), 24))
    state = State()
    assert run(ctypes.byref(state), address, 24, buffers, 2, None) == 0
    assert state.active == 0 and state.completed == 2 and tuple(incoming) == tuple(outgoing) == bits
    # Failed preflight must not invoke either region or overwrite user data.
    before = bytes(state)
    assert run(ctypes.byref(state), address, 23, buffers, 2, None) == -1
    assert bytes(state) == before and tuple(outgoing) == bits
    api.inspect(result.manifest)
    (result.directory / report["link_inputs"][1]).write_text("changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        api.inspect(result.manifest)


@pytest.mark.parametrize("mutation,reason", [
    (lambda result: result.metadata.pop("c_call"), "no supported explicit C call"),
    (lambda result: result.metadata["c_call"]["inputs"][0].update(shape=[1, 6]), "ordered original tensor"),
    (lambda result: result.metadata["c_call"].update(completion="async"), "synchronous completion"),
    (lambda result: result.metadata["c_call"].update(link_inputs=["layout.json"]), "link input needs C source"),
    (lambda result: result.metadata["c_call"].update(buffer_alignment=128), "selected memory contract"),
    (lambda result: result.metadata["c_call"].update(scratch_bytes=64), "zero additional host scratch"),
    (lambda result: result.metadata["c_call"].update(resource_kind="accelerator"), "selected resource"),
])
def test_c_control_refuses_unimplemented_or_substituted_abi_before_success_manifest(tmp_path, mutation, reason):
    with pytest.raises(ValueError, match=reason):
        compile_control(tmp_path, mutate=mutation)
    assert not (tmp_path / "compiled/compilation.json").exists()
    assert not (tmp_path / "compiled/control").exists()


def test_c_control_never_drops_original_inline_cpu_operations(tmp_path):
    source, hardware, candidates = selected(tmp_path)
    original = admit_original_cpu(source, 0, hardware, resource="cpu0", memory_domain="ram")
    backend = PlannedCompiler(select_execution(source, (original, candidates[1])),
                              {layout.name: RegionBackend(layout, "kernel")})
    with pytest.raises(ValueError, match="inline CPU cannot be omitted"):
        api.compile(ir_text(source.module), backend=backend, output=tmp_path / "rejected", options=OPTIONS)
    assert not (tmp_path / "rejected").exists()


def test_later_provider_cannot_change_a_previous_region_receipt(tmp_path):
    previous = []
    def mutate(result):
        if previous:
            previous[0].metadata["c_call"]["completion"] = "async"
        previous.append(result)
    with pytest.raises(ValueError, match="receipt changed after invocation"):
        compile_control(tmp_path, mutate=mutate)
    assert not (tmp_path / "compiled/compilation.json").exists()


def image_selection(tmp_path):
    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC required for actual C image execution")
    directory = tmp_path / "startup"
    directory.mkdir()
    path = directory / "main.c"
    path.write_text('''#include <stdint.h>
#include <stdio.h>
#include "model.h"
int main(void) {
  _Alignas(64) unsigned char arena[64];
  uint32_t incoming[6] = {0x7f800001u, 0x7fc12345u, 0x80000000u, 0x7f800000u, 1u, 0xff800001u};
  uint32_t outgoing[6] = {0};
  model_buffer buffers[2] = {{incoming, sizeof incoming}, {outgoing, sizeof outgoing}};
  model_state state = {0};
  if (model_run(&state, arena, sizeof arena, buffers, 2, 0)) return 1;
  if (state.completed != 2 || state.active) return 2;
  for (int i = 0; i < 6; ++i) if (incoming[i] != outgoing[i]) return 3;
  puts("selected-layout-bits-preserved");
  return 0;
}
''')
    product = Artifact.from_file(path, kind="c-source", role="entry")
    startup = Compilation(directory, "caller.startup", product.sha256, (product,))
    startup.write()
    return CImageLink(compiler=Path(gcc).resolve(), flags=("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"),
                      startup=startup, link_inputs=("main.c",),
                      deployment=Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1))


def test_selected_image_links_actual_wrappers_runs_and_prepares_existing_external_handoff(tmp_path):
    from merlin.runtime.service import RunOptions, RunResult

    image = image_selection(tmp_path)
    options = replace(OPTIONS, system="fixture-cpu", platform="linux", harts=1)
    result = compile_control(tmp_path, image=image, options=options)
    assert result.deployment == image.deployment and result.metadata["stage"] == "linked_image"
    report = json.loads((result.directory / "plan.json").read_text())
    assert "link" not in report["remaining"] and "execution_evidence" in report["remaining"]
    linked = json.loads((result.directory / "image/link.json").read_text())
    assert linked["placement_execution"] == "not_observed" and linked["elf"]["machine"] == 62
    assert len(list((result.directory / "image/invocations").rglob("invocation.json"))) == 1

    def validate(request):
        assert request.deployment == image.deployment and request.artifact.kind == "elf"

    def run(request):
        actual = subprocess.run([request.artifact.path], check=True, capture_output=True, text=True)
        return RunResult("fixture.host", request.artifact.sha256, "completed", actual.stdout,
                         {"scope": "actual host CPU layout test; no accelerator or FireSim execution"})

    executed = api.run(result, runner=SimpleNamespace(name="fixture.host", validate=validate, run=run))
    assert executed.stdout == "selected-layout-bits-preserved\n"
    prepared = api.run(result, runner="firesim", options=RunOptions(mode="prepare", output=tmp_path / "handoff"))
    request = json.loads((prepared.directory / "request.json").read_text())
    assert request["entry"] == "payload/image/program.elf" and request["execution"] == "external_operator_required"
    assert (prepared.directory / "payload/image/startup/main.c").is_file()
    actual = subprocess.run(["sh", str(prepared.directory / "invoke.sh")], capture_output=True, text=True, check=True)
    assert actual.stdout == executed.stdout
    api.inspect(result.manifest)


def test_image_selection_and_link_inputs_cannot_change_or_hide_unimplemented_payloads(tmp_path, monkeypatch):
    from merlin.compiler.build import process

    image = image_selection(tmp_path)
    with pytest.raises(ValueError, match="explicitly selected harts"):
        compile_control(tmp_path, image=image, options=replace(OPTIONS, harts=2))
    assert not (tmp_path / "compiled").exists()
    with pytest.raises(ValueError, match="no admitted payload"):
        compile_control(tmp_path, image=image,
                        mutate=lambda result: result.metadata["c_call"].update(payloads=["layout.json"]))
    assert not (tmp_path / "compiled/compilation.json").exists()

    directory = tmp_path / "changed"
    directory.mkdir()
    selected_image = image_selection(directory)
    original = process.run
    def change_after_link(*args, **kwargs):
        result = original(*args, **kwargs)
        selected_image.startup.artifacts[0].path.write_text("changed after actual link")
        return result
    monkeypatch.setattr(process, "run", change_after_link)
    with pytest.raises(ValueError, match="bytes do not match"):
        compile_control(directory, image=selected_image)
    assert not (directory / "compiled/compilation.json").exists()



def stateful_control(tmp_path, *, corrupt=None, share=False, split_host=False, topology_edit=None,
                     platform=None, image_factory=None):
    """Source-bound software ABI fixture; no device math/execution claim."""
    from merlin.compiler.backend.cabi import CRegionABI
    from merlin.compiler.backend.stateful import CContext, CRegionABI2, EmbeddedPayload, render_u32_payload

    source, hardware, original = selected(tmp_path)
    document = yaml.safe_load(hardware.path.read_text())
    document["memory_domains"][0]["capacity_bytes"] = 1024
    if split_host:
        document["memory_domains"].append({"id": "host", "capacity_bytes": 256, "alignment_bytes": 64})
        document["access"].append({"resource": "cpu0", "domain": "host", "read": True, "write": True,
                                   "addressability": "shared", "visibility": "coherent", "completion": "synchronous"})
    if topology_edit is not None:
        topology_edit(document)
    hardware.path.write_text(yaml.safe_dump(document))
    hardware = select_topology(hardware.path)
    requirements = []
    admitted = []
    for index, item in enumerate(original):
        requirement = CContext(
            "shared" if share else f"context_{index}",
            "a" * 64 if platform is None else next(a.sha256 for a in platform.artifacts if a.kind == "json"),
            "cpu0", "host" if split_host else "ram",
            64, 16, 64, 16, "context.h", "software_context", "software_context_init")
        requirements.append(requirement)
        alternative = replace(item.alternative, metadata=tuple(sorted((requirement.selection_metadata(),))))
        proposal = replace(item.proposal, alternative=alternative)
        def admit(bound, claim):
            assert bound is source and claim.alternative.nodes in ((0,), (1,))
            assert claim.alternative.scratch_bytes == 0
        admitted.append(admit_proposal(source, proposal, hardware, validate_implementation=admit))

    def compile_region(request):
        result = layout.compile(request)
        index = int(request.output.name)
        context = requirements[index]
        original_call = CRegionABI.from_compilation(result)
        header = request.output / "context.h"
        header.write_text("""#include <stddef.h>
#include <stdint.h>
typedef struct { unsigned calls; uint32_t loaded; void *workspace; size_t bytes; } software_context;
_Static_assert(sizeof(software_context)<=64 && _Alignof(software_context)<=16, "context reservation");
static inline int software_context_init(software_context *c, void *workspace, size_t bytes) {
 if(!c || !workspace || bytes<64) return 1;
 *c=(software_context){0,0,workspace,bytes}; return 0;
}
""")
        binary = request.output / "program.bin"
        binary.write_bytes(bytes.fromhex("0403020108070605"))
        embedded = request.output / "program.h"
        embedded.write_text(render_u32_payload("software_program", binary.read_bytes()))
        wrapper = request.output / "context.c"
        symbol = original_call.symbol + "_context"
        wrapper.write_text(f"""#include "context.h"
#include "program.h"
extern int {original_call.symbol}(void *, const void *const *, void *const *);
int {symbol}(void *opaque, const void *const *inputs, void *const *outputs) {{
 software_context *c=opaque;
 if(!c || !c->workspace || c->bytes<64) return 37;
 c->loaded=software_program[0];
 int result={original_call.symbol}(0,inputs,outputs);
 if(!result) ++c->calls;
 return result;
}}
""")
        call = replace(original_call, symbol=symbol, link_inputs=(*original_call.link_inputs, "context.c"),
                       payloads=("program.bin",))
        abi = CRegionABI2(call, context, (EmbeddedPayload("program.bin", "program.h", "software_program"),))
        artifacts = (*result.artifacts, Artifact.from_file(header, kind="c-header", role="context"),
                     Artifact.from_file(binary, kind="device-binary", role="device-program"),
                     Artifact.from_file(embedded, kind="c-header", role="embedded-program"),
                     Artifact.from_file(wrapper, kind="c-source", role="context-wrapper"))
        result = replace(result, artifacts=artifacts, metadata={**result.metadata, "c_call": abi.record()})
        if corrupt:
            corrupt(result)
        return result

    provider = SimpleNamespace(name=layout.name, validate=layout.validate, compile=compile_region)
    selection = select_execution(source, tuple(admitted))
    backend = PlannedCompiler(selection, {layout.name: RegionBackend(provider, "kernel")},
                              image=image_factory(tuple(requirements)) if image_factory is not None else None)
    return api.compile(ir_text(source.module), backend=backend, output=tmp_path / "compiled", options=OPTIONS)


def test_selected_stateful_contexts_reserve_host_storage_embed_exact_bytes_and_execute(tmp_path):
    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC required for actual stateful C control execution")
    result = stateful_control(tmp_path, split_host=True)
    report = json.loads((result.directory / "control/control.json").read_text())
    assert report["context_table"]["minimum_count"] == 2
    assert [row["context_index"] for row in report["calls"]] == [0, 1]
    assert report["context_indices"] == {"context_0": 0, "context_1": 1}
    assert report["host_context_reservations"]["domains"] == {
        "host": {"state_bytes": 128, "workspace_bytes": 64, "total_bytes": 192}}
    assert report["total_reserved_host_bytes"] == 0  # Separate host domain, not tensor RAM.
    assert report["context_initialization"] == "caller_obligation_not_observed"
    assert report["completion_visibility"] == "runtime_obligation; no_execution_observed"
    assert len(report["payloads"]) == 2 and result.deployment is None
    first = result.directory / "regions/0"
    startup = tmp_path / "stateful.c"
    startup.write_text("""#include <stdint.h>
#include <string.h>
#include "model.h"
#include "context.h"
int main(void) {
 _Alignas(64) unsigned char arena[128], workspace0[64], workspace1[64];
 _Alignas(64) uint32_t input[6]={0x7f800001,0x7fc12345,0x80000000,0x7f800000,1,0xff800001};
 _Alignas(64) uint32_t output[6]={0};
 software_context first,second;
 if(software_context_init(&first,workspace0,sizeof(workspace0)) ||
    software_context_init(&second,workspace1,sizeof(workspace1))) return 1;
 void *items[]={&first,&second}; model_contexts contexts={items,2};
 model_buffer buffers[]={{input,sizeof(input)},{output,sizeof(output)}};
 model_state state={0};
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)) return 2;
 if(first.calls!=1 || second.calls!=1 || first.loaded!=0x01020304u || second.loaded!=0x01020304u ||
    memcmp(input,output,sizeof(input)) || state.active || state.completed!=2) return 3;
 /* Missing context is a real wrapper error. Driver remains active and cannot
  * reuse buffers or reenter until its owner drains/resets the endpoint. */
 state=(model_state){0}; items[0]=0;
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_KERNEL_ERROR || !state.active ||
    state.completed || state.kernel_status!=37) return 4;
 if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_BUSY) return 5;
 return 0;
}
""")
    executable = tmp_path / "stateful"
    subprocess.run([gcc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(result.directory / "control"), "-I" + str(first), str(startup),
                    *(str(result.directory / name) for name in report["link_inputs"]), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_stateful_receipt_refuses_changed_embedding_context_and_unadmitted_sharing(tmp_path):
    from merlin.compiler.backend.stateful import CRegionABI2

    compiled = stateful_control(tmp_path)
    region = Compilation.load(compiled.directory / "regions/0/compilation.json")
    header = region.directory / "program.h"
    header.write_text(header.read_text().replace("01020304", "01020305"))
    resealed = replace(region, artifacts=tuple(Artifact.from_file(a.path, kind=a.kind, role=a.role)
                       if a.path == header else a for a in region.artifacts))
    with pytest.raises(ValueError, match="exact selected binary bytes"):
        CRegionABI2.from_compilation(resealed)
    context_case = tmp_path / "context-case"
    context_case.mkdir()
    def changed_context(result):
        result.metadata["c_call"]["context"]["workspace_bytes"] = 128
    with pytest.raises(ValueError, match="original selected host reservation"):
        stateful_control(context_case, corrupt=changed_context)
    sharing_case = tmp_path / "sharing-case"
    sharing_case.mkdir()
    with pytest.raises(ValueError, match="different compiled initialization or products"):
        stateful_control(sharing_case, share=True)


    # The named host domain must supply its own actual capacity/access. Device
    # boundary RAM cannot silently stand in for an unavailable host reservation.
    cases = (
        ("device-context", lambda d: d["resources"][0].update(kind="accelerator"), "CPU execution resource"),
        ("small-host", lambda d: d["memory_domains"][1].update(capacity_bytes=128), "host contexts/workspace"),
        ("no-host-access", lambda d: d["access"].pop(), "no declared access"),
        ("explicit-visibility", lambda d: d["access"][-1].update(visibility="explicit"), "unimplemented transfer"),
    )
    for name, edit, reason in cases:
        destination = tmp_path / name
        destination.mkdir()
        with pytest.raises(ValueError, match=reason):
            stateful_control(destination, split_host=True, topology_edit=edit)
        assert not (destination / "compiled").exists()


def stateful_image(tmp_path, *, corrupt=None, reservations=None, validator_edit=None, entry_bytes=128, pie=False):
    """Original caller's software-only platform and generated image entry."""
    from merlin.compiler.backend.stateful import CRegionABI2
    from merlin.compiler.build.contextentry import CImageContext, CImageEntry, ReservedImageInterval

    compiler = shutil.which("gcc")
    if not compiler:
        pytest.skip("GCC required for actual stateful image construction")
    platform_directory = tmp_path / "platform"
    platform_directory.mkdir()
    contract = platform_directory / "contract.json"
    contract.write_text(json.dumps({"scope": "software-only", "initialize": "contents_not_retained"}))
    artifact = Artifact.from_file(contract, kind="json", role="platform-contract")
    platform = Compilation(platform_directory, "caller.software-platform", artifact.sha256, (artifact,))
    platform.write()
    startup_directory = tmp_path / "startup"
    startup_directory.mkdir()
    source = startup_directory / "main.c"
    source.write_text('''#include <stdint.h>
#include <string.h>
#include "entry.h"
int main(void) {
 _Alignas(64) unsigned char arena[128];
 _Alignas(64) uint32_t input[6]={0x7f800001,0x7fc12345,0x80000000,0x7f800000,1,0xff800001}, output[6]={0};
 model_buffer buffers[]={{input,sizeof(input)},{output,sizeof(output)}}; model_state state={0};
 if(image_run(&state,arena,sizeof(arena),buffers,2) || memcmp(input,output,sizeof(input)) ||
    state.active || state.completed!=2) return 1;
 /* Any error retains persistent entry state; clearing a caller state cannot retry it. */
 if(image_run(&state,arena,sizeof(arena),buffers,1)!=MODEL_INVALID) return 2;
 state=(model_state){0};
 if(image_run(&state,arena,sizeof(arena),buffers,2)!=MODEL_BUSY) return 3;
 return 0;
}
''')
    configuration = startup_directory / "configuration.json"
    configuration.write_text('{"platform":"software-test","runtime_state_pre_reserved":128}')
    script = startup_directory / "layout.ld"
    script.write_text("SECTIONS { .image_marker : { BYTE(0x42) } } INSERT AFTER .rodata;\n")
    products = (Artifact.from_file(source, kind="c-source", role="entry"),
                Artifact.from_file(configuration, kind="json", role="configuration"),
                Artifact.from_file(script, kind="data", role="linker-script"))
    startup = Compilation(startup_directory, "caller.startup", products[0].sha256, products)
    startup.write()
    admissions = []
    flags = ("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pie" if pie else "-no-pie")
    def validate(binding, region, image_selection, options):
        assert binding.platform is platform and image_selection.startup is startup
        assert json.loads(contract.read_text()) == {"scope": "software-only", "initialize": "contents_not_retained"}
        assert binding.context.contract_sha256 == artifact.sha256
        assert CRegionABI2.from_compilation(region).context == binding.context
        assert image_selection.deployment == Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1)
        assert image_selection.compiler == Path(compiler).resolve()
        assert image_selection.flags == flags
        assert options == {"scope": "software-only; no hardware execution qualification"}
        admissions.append(binding.context.identity)
        if validator_edit:
            validator_edit(binding, region, options)
    entry_holder = []
    def image_factory(contexts):
        entry = CImageEntry(contexts=tuple(CImageContext(c, platform, "contract.json",
            "int(context*,void*,size_t)", "contents_not_retained", "host_state_only") for c in contexts),
            symbol="image_run",
            validate_platform=validate, options={"scope": "software-only; no hardware execution qualification"},
            reservations=reservations or (ReservedImageInterval("physical", 0x90000000, 4096),),
            configuration=("configuration.json",), linker_scripts=("layout.ld",),
            entry_state_bytes=entry_bytes, entry_state_pre_reserved=True)
        entry_holder.append(entry)
        return CImageLink(compiler=Path(compiler).resolve(), flags=flags,
            startup=startup, link_inputs=("main.c",), context_entry=entry,
            deployment=Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1))
    result = stateful_control(tmp_path, platform=platform, image_factory=image_factory, corrupt=corrupt)
    return result, admissions, entry_holder[0]


def test_selected_stateful_image_initializes_exact_contexts_links_and_retains_failure(tmp_path):
    result, admissions, entry = stateful_image(tmp_path)
    assert result.metadata["stage"] == "linked_image" and result.deployment.platform == "linux"
    assert set(admissions) == {"context_0", "context_1"} and len(admissions) >= 4
    subprocess.run([str(result.directory / "image/program.elf")], check=True, capture_output=True, text=True)
    record = json.loads((result.directory / "image/link.json").read_text())
    assert record["placement_execution"] == "not_observed"
    assert record["context_entry"]["platform_admission"] == (
        "original_caller_conditional_contract; hardware_execution_unobserved")
    assert record["context_entry"]["selection"]["configuration"] == ["configuration.json"]
    generated = (result.directory / "image/context/entry.c").read_text()
    assert generated.count("static unsigned char workspace_") == 1  # Serial maximum, not a hidden duplicate.
    assert generated.index("software_context_init(&context_0") < generated.index("result=model_run(")
    assert any("-Wl,-T," in token and token.endswith("image/startup/layout.ld") for token in record["command"])
    with pytest.raises(ValueError, match="retained explicit members"):
        entry.validate_flags(("-Wl,-T,/tmp/unselected.ld",))
    entry.options["scope"] = "changed"
    with pytest.raises(ValueError, match="selection changed"):
        entry.revalidate()


def test_selected_stateful_image_refuses_unadmitted_signature_mutation_and_reserved_segments(tmp_path):
    from merlin.compiler.build.contextentry import ReservedImageInterval

    def wrong_signature(result):
        path = result.directory / "context.h"
        path.write_text(path.read_text().replace("static inline int software_context_init",
                                                 "static inline unsigned software_context_init"))
        artifacts = tuple(Artifact.from_file(a.path, kind=a.kind, role=a.role) if a.path == path else a
                          for a in result.artifacts)
        object.__setattr__(result, "artifacts", artifacts)
    signature = tmp_path / "signature"
    signature.mkdir()
    with pytest.raises(RuntimeError, match="admitted image initializer signature"):
        stateful_image(signature, corrupt=wrong_signature)
    assert not (signature / "compiled/compilation.json").exists()
    mutation = tmp_path / "mutation"
    mutation.mkdir()
    with pytest.raises(ValueError, match="selection changed"):
        stateful_image(mutation, validator_edit=lambda binding, region, options: options.update(scope="changed"))
    assert not (mutation / "compiled/image").exists()
    reserved = tmp_path / "reserved"
    reserved.mkdir()
    with pytest.raises(ValueError, match="overlaps an admitted platform reservation"):
        stateful_image(reserved, reservations=(ReservedImageInterval("physical", 0, 1 << 64),))
    assert not (reserved / "compiled/compilation.json").exists()
    state = tmp_path / "state"
    state.mkdir()
    with pytest.raises(RuntimeError, match="image startup state ceiling"):
        stateful_image(state, entry_bytes=1)

    pie = tmp_path / "pie"
    pie.mkdir()
    with pytest.raises(ValueError, match="fixed-address executable ELF"):
        stateful_image(pie, pie=True)
    assert not (pie / "compiled/compilation.json").exists()
