"""Resident host backing and deployment admission as one table of isolated rows.

Compact host backing keeps port identities and authenticated bytes, host backing leaves only with every
consumer's resident proof, and a resident deployment is published only through the capabilities a target
declares. Each failure names its `check N` label, which the comment at that check describes. The accelerator
regions publish synthetic programs, not device code; the generated host C executes here.
"""
import hashlib
import json
import shutil
import struct
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from corefixtures.compile_input_fixtures import _write_weighted_capture
from corefixtures.row_table import run_rows
from semantic_application_fixtures import plan_for
from shared_loop_providers import (
    WINDOW_BASE,
    WINDOW_BYTES,
    AdmittingProvider,
    DeployedProvider,
    FlatProvider,
    _deployed,
    _onchip,
    _snapshot,
    _source,
)

from merlin.compiler.backend.protocol import ResidentParameters
from merlin.compiler.build.regions import (
    CompiledCRegion, compile_semantic_application, verify_semantic_application,
)
from merlin.compiler.lower.c.application import emit_application_c
from merlin.compiler.lower.c.compact_parameters import compact_parameters
from merlin.compiler.lower.c.dispatch import DispatchCall, build_program, emit_program_c
from merlin.compiler.lower.c.parameters import ApplicationParameters
from merlin.compiler.lower.c.weights import WeightEntry
from merlin.compiler.partition.global_plan_emission import dispatch_digest
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.schedule.storage import application_residency as residency
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options, current_codegen_options
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType
from merlin.compiler.semantic.target_binding import NativeCompilationError, verify_native_publication


@contextmanager
def _refused(label, error=ValueError, match=None):
    """`pytest.raises` whose failure names the label of its check."""
    try:
        with pytest.raises(error, match=match) as caught:
            yield caught
    except (AssertionError, pytest.fail.Exception) as failure:
        raise AssertionError(f"check {label}: {failure}") from failure


def _run(command, label, **options):
    """Run one generated C step and fail with its check label."""
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, **options)
    assert done.returncode == 0, f"check {label}: {done.stdout}{done.stderr}"
    return done


def _compiler(label):
    compiler = shutil.which("cc")
    assert compiler, f"check {label}: a C compiler is required; this execution check cannot skip"
    return compiler


def _parameters(directory):
    """A four-port archive: two ports share bytes 0 to 4, one 8-byte port, and one 4-byte host port."""
    payload = bytes(range(16))
    header = json.dumps({
        "shared": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "resident": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        "host": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
    }).encode()
    archive = struct.pack("<Q", len(header)) + header + payload
    path = directory / "weights.safetensors"
    path.write_bytes(archive)
    entries = (WeightEntry(0, "shared", "F32", [1], 0, 4),
               WeightEntry(1, "shared", "F32", [1], 0, 4),
               WeightEntry(2, "resident", "F32", [2], 4, 8),
               WeightEntry(3, "host", "F32", [1], 12, 4))
    nodes = tuple(SemanticNode(f"arg{i}", "input", (), TensorType(tuple(e.shape), "f32", "raw-bit-pattern"),
                               effect="input") for i, e in enumerate(entries))
    source = KernelRequest(nodes, ("arg3",), ("external",),
                           tuple((n.id, "external") for n in nodes), "synthetic")
    original = ApplicationParameters(source.digest(), "a" * 64, sha256(archive).hexdigest(),
                                     sha256(payload).hexdigest(), len(payload), entries, path)
    original.check(source)
    return source, original


def _subdirectory(directory, name):
    path = directory / name
    path.mkdir()
    return path


# ---- compact_bytes ----

# check 39: compact backing keeps the source digest and port identities, packs the kept bytes at their offsets,
# refuses an existing output, and its emitted C runs.
def _compact_bytes(directory, patch):
    for index, (removed, expected, offsets) in enumerate([
        (("arg2",), bytes(range(4)) + bytes(range(12, 16)), (0, 0, None, 4)),
        ((), bytes(range(16)), (0, 0, 4, 12)),
        (("arg0", "arg1", "arg2", "arg3"), b"\0", (None,) * 4),
    ]):
        _compact_bytes_case(_subdirectory(directory, f"case_{index}"), removed, expected, offsets)


def _compact_bytes_case(directory, removed, expected, offsets):
    source, original = _parameters(directory)
    compact = compact_parameters(original, source, resident_only=removed)
    compact.check(source, resident_only=removed)
    assert compact.source_request_digest == original.source_request_digest, "check 39"
    assert compact.node_ids == original.node_ids, "check 39"
    assert compact.compact_offsets == offsets, "check 39"
    assert compact.payload_bytes == len(expected), "check 39"
    assert compact.payload_sha256 == sha256(expected).hexdigest(), "check 39"
    record = compact.record()
    assert record["source_parameters"] == original.record(), "check 39"
    assert record["payload_sha256"] != original.payload_sha256 or not removed, "check 39"
    destination = directory / "compact.bin"
    compact.write_payload(destination)
    assert destination.read_bytes() == expected, "check 39"
    with _refused(39, FileExistsError):
        compact.write_payload(destination)
    assert destination.read_bytes() == expected, "check 39"
    for node, offset in zip(original.node_ids, offsets):
        assert compact.pointer_expression(node) == ("0" if offset is None else f"parameters->data+{offset}u"), (
            "check 39")
    declarations, implementation, headers = compact.emit_binding("check")
    for name, text in headers.items():
        (directory / name).write_text(text)
    actual = ",".join(map(str, expected))
    expressions = "\n".join(
        f"assert(({compact.pointer_expression(node)})=="
        + ("0);" if offset is None else f"bytes+{offset}u);")
        for node, offset in zip(original.node_ids, offsets))
    c = """
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include "parameter_sha256.h"
static int check_valid(const void *p,size_t n) { return p && (uintptr_t)p<=UINTPTR_MAX-n; }
static int check_overlap(const void *p,size_t n,const void *q,size_t m) {
  return (uintptr_t)p<(uintptr_t)q+m && (uintptr_t)q<(uintptr_t)p+n;
}
""" + declarations + implementation + f"""
int main(void) {{
  _Alignas(8) unsigned char bytes[]={{{actual}}};
  check_parameters descriptor={{0}},before;
  check_parameters *parameters=&descriptor;
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes))==0);
  assert(check_parameters_ready(parameters));
  {expressions}
  before=descriptor;bytes[0]^=1;
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes))==-2);
  assert(memcmp(&before,parameters,sizeof(before))==0);
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes)+1)==-1);
  return 0;
}}
"""
    (directory / "check.c").write_text(c)
    _run([_compiler(39), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "check.c", "-o", "check"], 39,
         cwd=directory)
    _run([str(directory / "check")], 39)


# ---- compact_application_execution ----

# check 46: application C over compact backing refuses a driver without the resident-only ports, publishes only
# the kept bytes, and runs.
def _compact_application_execution(directory, patch):
    source, original = _parameters(directory)
    result = SemanticNode("out", "concat", original.node_ids, TensorType((5,), "f32", "raw-bit-pattern"),
                          (("axis", 0),))
    source = replace(source, nodes=(*source.nodes, result), outputs=("out",))
    original = replace(original, source_request_digest=source.digest())
    compact = compact_parameters(original, source, resident_only=("arg2",))
    plan = plan_for(source, ((0,),))
    plan = replace(plan, selected=tuple(replace(row, placement="host") for row in plan.selected))
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    program = build_program(outlined.logical, global_plan=plan, global_plan_emitter=outlined,
                            memory_requirements=lambda graph: ArenaRequirements(
                                1024, (0,), (1,), dispatch_digest=dispatch_digest(graph)))
    driver, driver_files = emit_program_c(program, "model_driver", {0: DispatchCall("consume", "host")},
                                         read_only_inputs=compact.node_ids,
                                         resident_only_inputs=compact.resident_only)
    with _refused(46, match="resident-only"):
        emit_application_c(source, {**driver, "resident_only_inputs": []}, driver_files,
                           "model", capacity=1024, parameters=compact)
    interface, files = emit_application_c(source, driver, driver_files, "model", capacity=1024, parameters=compact)
    assert interface["inputs"] == [], "check 46"
    assert interface["model_parameters"]["payload_bytes"] == 8, "check 46"
    for name, text in {**driver_files, **files}.items():
        (directory / name).write_text(text)
    compact.write_payload(directory / "parameters.bin")
    assert (directory / "parameters.bin").read_bytes() == bytes(range(4)) + bytes(range(12, 16)), "check 46"
    (directory / "main.c").write_text(r'''
#include "model.h"
#include <assert.h>
#include <string.h>
static unsigned visits;
int consume(void *context,const void *const *inputs,void *const *outputs) {
  (void)context;
  assert(inputs[0] && inputs[0]==inputs[1] && inputs[2]==0 && inputs[3]);
  const unsigned char *host=inputs[0],*tail=inputs[3];
  assert(tail==host+4);
  assert(host[0]==0 && host[3]==3 && tail[0]==12 && tail[3]==15);
  unsigned char *out=outputs[0];
  memcpy(out,host,4);memcpy(out+4,host,4);
  memset(out+8,0x5a,8);memcpy(out+16,tail,4);
  ++visits;return 0;
}
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  _Alignas(8) unsigned char bytes[]={0,1,2,3,12,13,14,15};
  float output[5];float *outputs[]={output};
  model_parameters parameters={0};model_driver_state state={0};
  assert(model_parameters_bind(&parameters,bytes,sizeof(bytes))==0);
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,0,outputs,0)==0);
  const unsigned char expected[]={0,1,2,3,0,1,2,3,90,90,90,90,90,90,90,90,12,13,14,15};
  assert(visits==1 && memcmp(output,expected,sizeof(expected))==0);
  return 0;
}
''')
    _run([_compiler(46), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
          "model.c", "model_driver.c", "main.c", "-o", "check"], 46, cwd=directory)
    _run([str(directory / "check")], 46)


# ---- fresh_replay ----

def _replay_setup(directory):
    """A two-region application whose accelerator region keeps arg2 resident and whose host region reads it."""
    source, original = _parameters(directory)
    resident_result = SemanticNode("resident_result", "reshape", ("arg2",), source.node("arg2").type)
    result = SemanticNode("out", "concat", ("arg0", "arg1", "resident_result", "arg3"),
                          TensorType((5,), "f32", "raw-bit-pattern"),
                          (("axis", 0),))
    source = replace(source, nodes=(*source.nodes, resident_result, result), outputs=("out",))
    parameters = replace(original, source_request_digest=source.digest())
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=tuple(replace(row, placement=placement)
                 for row, placement in zip(plan.selected, ("accelerator", "host"))))

    class Adapter(ResidentParameters):
        compiles = 0
        exports = 0

        def plan(self, request):
            assert request == source, "check 100"
            return plan

        def resident_memory_geometry(self):
            return {"dma_base": 0, "dma_capacity": 65536, "alignment": 4}

        def inventory_resident_inputs(self, region, alternative):
            return [{"node_id": "arg2", "bytes": 8, "alignment": 4,
                     "layout": {"schema": "test.dense-resident.v1"}}]

        def pack_resident_input(self, region, node_id, dense, alternative):
            assert node_id == "arg2" and len(dense) == 8, "check 100"
            return dense

        def admit_resident_deployment(self, *args, **kwargs):
            raise AssertionError("check 100: this application is not deployed on chip")

        def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
            self.compiles += 1
            if alternative.placement == "accelerator":
                root = segments / name
                root.mkdir()
                binary = b"synthetic identity fixture; not device code\n"
                (root / "program.bin").write_bytes(binary)
                manifest = {"engine": "merlin_native", "request_digest": region.digest(),
                            "target_identity": source.target_identity,
                            "binary_sha256": sha256(binary).hexdigest()}
                (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            return self.export_region(region, alternative, name, segments, target_source)

        def export_region(self, region, alternative, name, segments, target_source):
            self.exports += 1
            device = alternative.placement == "accelerator"
            description, programs = {}, ()
            if device:
                root = segments / name
                manifest = json.loads((root / "manifest.json").read_bytes())
                verify_native_publication(root, manifest, engine="merlin_native", request_digest=region.digest(),
                                          target_identity=source.target_identity)
                binding = current_codegen_options().resident_bindings[region.digest()]["arg2"]
                description = {"resident_bindings": [{"node_id": "arg2", **binding}]}
                programs = (f"segments/{name}",)
            header = f"int {name}(void *,const void *const *,void *const *);\n"
            inputs = tuple(key for key, _ in region.input_storages)
            resident_index = inputs.index("resident_result") if not device else None
            tail_index = inputs.index("arg3") if not device else None
            body = f'''#include <assert.h>
#include <string.h>
int {name}(void *context,const void *const *in,void *const *out) {{
  (void)context;
  assert(in[0] && in[0]==in[1] && in[2] && in[3]);
  unsigned char *result=out[0];
  memcpy(result,in[0],4);memcpy(result+4,in[1],4);
  memcpy(result+8,in[{resident_index}],8);memcpy(result+16,in[{tail_index}],4);
  return 0;
}}
'''
            if device:
                body = f'''#include <assert.h>
#include <string.h>
int {name}(void *context,const void *const *in,void *const *out) {{
  assert(context && !in[0]);memcpy(out[0],context,8);return 0;
}}
'''
            return CompiledCRegion(name, name + ".h", inputs,
                                   region.outputs, {name + ".h": header.encode(), name + ".c": body.encode()},
                                   (), programs, description)

    options = CodegenOptions(application_residency=True, io_memory_base=4096, io_memory_bytes=256,
                             io_memory_capacity_base=4096, io_memory_capacity_bytes=4096)
    return source, parameters, Adapter(), _snapshot(directory / "snapshot"), options


def _replay_compile(directory):
    source, parameters, adapter, snapshot, options = _replay_setup(directory)
    destination = directory / "application"
    with codegen_options(options):
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=directory, destination=destination, limits=None, name="model", capacity=1024,
            parameters=parameters)
    return source, parameters, adapter, snapshot, options, destination, manifest


# check 100: a fresh compile with resident backing publishes the kept parameter bytes, and its combined C runs.
def _fresh_replay(directory, patch):
    source, parameters, adapter, snapshot, options, destination, manifest = _replay_compile(
        _subdirectory(directory, "check_100"))
    root = destination.parent
    assert adapter.compiles == 2 and adapter.exports >= 4, "check 100"
    # The handoff/parameters.bin layout parts of checks 100 and 101 move with the handoff code.
    assert (destination / "handoff/parameters.bin").read_bytes() == bytes(range(4)) + bytes(range(12, 16)), (
        "check 100")
    published = manifest["application_interface"]["model_parameters"]
    assert published["schema"] == "merlin.compact_application_parameters.v1", "check 100"
    assert published["resident_only"] == ["arg2"], "check 100"
    assert published["source_parameters"]["payload_sha256"] == parameters.payload_sha256, "check 100"
    assert manifest["request_digest"] == source.digest() == parameters.source_request_digest, "check 100"
    assert json.loads((destination / "dispatch-c.json").read_bytes())["resident_only_inputs"] == ["arg2"], (
        "check 100")
    record = json.loads((destination / "residency.json").read_bytes())
    load, = record["loads"]
    assert (destination / load["path"]).read_bytes() == bytes(range(4, 12)), "check 100"
    with codegen_options(options):
        checked = verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=root, artifact=destination, parameters=parameters)
    assert checked["status"] == "checked", "check 100"
    assert adapter.exports >= 6, "check 100"
    package = destination / "c-package"
    (root / "main.c").write_text(r'''
#include "combined/model_combined.h"
#include <assert.h>
#include <string.h>
int main(void) {
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  _Alignas(8) unsigned char host[]={0,1,2,3,12,13,14,15};
  unsigned char resident[]={4,5,6,7,8,9,10,11};void *items[]={resident,0};
  model_driver_contexts contexts={items,2};
  model_driver_state state={0};model_parameters parameters={0};
  float output[5];float *outputs[]={output};
  assert(model_parameters_bind(&parameters,host,sizeof(host))==0);
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,0,outputs,&contexts)==0);
  const unsigned char expected[]={0,1,2,3,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  assert(memcmp(output,expected,sizeof(expected))==0 && state.completed==2);
  resident[7]^=1;
  assert(model_invoke(&state,workspace,sizeof(workspace),&parameters,0,outputs,&contexts)==0);
  assert(memcmp(output,expected,sizeof(expected))!=0);
  return 0;
}
''')
    _run([_compiler(100), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "combined/model_combined.c",
          str(root / "main.c"), "-I", str(package), "-o", str(root / "check")], 100, cwd=package)
    _run([str(root / "check")], 100)
    # check 101: a changed compact payload or a changed resident byte of the original source refuses replay,
    # and the refusal names what changed.
    for changed, reason in (("compact", "changed or is missing: handoff/parameters.bin|model parameter blob differs"),
                            ("resident_source", "weight archive changed after compiler input validation")):
        source, parameters, adapter, snapshot, options, destination, _ = _replay_compile(
            _subdirectory(directory, f"check_101_{changed}"))
        if changed == "compact":
            path = destination / "handoff/parameters.bin"
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
        else:
            path = parameters.archive_path
            raw = bytearray(path.read_bytes())
            # Target a removed resident-only byte; compact host backing is unchanged.
            raw[-5] ^= 1
        path.write_bytes(raw)
        with codegen_options(options), _refused(101, match=reason):
            verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
                target_source=destination.parent, artifact=destination, parameters=parameters)


# ---- compact_refusals ----

# check 41: compacting refuses invalid resident-only ports, and a compact descriptor refuses changed fields or a
# changed source.
def _compact_refusals(directory, patch):
    source, original = _parameters(_subdirectory(directory, "check_41"))
    compact = compact_parameters(original, source, resident_only=("arg2",))
    for removed in (("arg2", "arg2"), ("unknown",), (1,)):
        with _refused(41):
            compact_parameters(original, source, resident_only=removed)
    for changed in (replace(compact, resident_only=("arg0",)),
                    replace(compact, compact_offsets=(0, 0, None, 0)),
                    replace(compact, payload_sha256="0" * 64)):
        with _refused(41):
            changed.check(source)
    with _refused(41):
        compact.check(source, resident_only=())
    with _refused(41):
        compact.check(replace(source, source_identity="changed"))
    with _refused(41):
        compact.pointer_expression("unknown")
    with _refused(41):
        compact.pointer_expression("arg0", "bad->name")
    # check 40: a changed or truncated archive is refused, and a refused publication leaves no output.
    for damage in ("changed", "truncated"):
        case = _subdirectory(directory, f"check_40_{damage}")
        source, original = _parameters(case)
        compact = compact_parameters(original, source, resident_only=("arg2",))
        raw = original.archive_path.read_bytes()
        original.archive_path.write_bytes(raw[:-1] if damage == "truncated" else raw[:-1] + bytes([raw[-1] ^ 1]))
        with _refused(40):
            compact_parameters(original, source, resident_only=("arg2",))
        with _refused(40):
            compact.check(source)
        output = case / "bad.bin"
        with _refused(40):
            compact.write_payload(output)
        assert not output.exists(), "check 40"
    with patch.context() as own:
        _aligned_projection(_subdirectory(directory, "check_42"), own)


def _aligned_projection(directory, patch):
    """check 42: an aligned projection pads with zeros and streams without a full-file read."""
    payload = bytes(range(24))
    header = json.dumps({
        "bytes": {"dtype": "I8", "shape": [3], "data_offsets": [0, 3]},
        "removed": {"dtype": "I8", "shape": [13], "data_offsets": [3, 16]},
        "wide": {"dtype": "I64", "shape": [1], "data_offsets": [16, 24]},
    }).encode()
    archive = struct.pack("<Q", len(header)) + header + payload
    path = directory / "weights.safetensors"
    path.write_bytes(archive)
    entries = (WeightEntry(0, "bytes", "I8", [3], 0, 3),
               WeightEntry(1, "removed", "I8", [13], 3, 13),
               WeightEntry(2, "wide", "I64", [1], 16, 8))
    nodes = tuple(SemanticNode(f"arg{i}", "input", (), TensorType(tuple(e.shape), dtype, "raw-bit-pattern"),
                               effect="input") for i, (e, dtype) in enumerate(zip(entries, ("i8", "i8", "i64"))))
    source = KernelRequest(nodes, ("arg2",), ("external",),
                           tuple((n.id, "external") for n in nodes), "synthetic")
    original = ApplicationParameters(source.digest(), "a" * 64, sha256(archive).hexdigest(),
                                     sha256(payload).hexdigest(), len(payload), entries, path)
    # Full-file byte reads are forbidden even for this tiny archive: construction,
    # checking, and publication must use the verified streaming snapshot route.
    with patch.context() as reads:
        reads.setattr(Path, "read_bytes", lambda _: pytest.fail("check 42: unbounded payload read"))
        compact = compact_parameters(original, source, resident_only=("arg1",))
        compact.check(source)
        compact.write_payload(directory / "compact.bin")
    assert compact.compact_offsets == (0, None, 8), "check 42"
    assert (directory / "compact.bin").read_bytes() == payload[:3] + b"\0" * 5 + payload[16:], "check 42"


# ---- host_backing_removal ----

@contextmanager
def _proof(rows):
    """Regions that read `inputs`, whose packages report `resident` ports bound, under a published table."""
    regions, packages, bindings = [], [], {}
    for index, (inputs, resident) in enumerate(rows):
        digest = f"region-{index}"
        region = SimpleNamespace(digest=lambda digest=digest: digest,
                                 input_storages=tuple((key, "external") for key in inputs))
        regions.append((digest, region))
        reports = [{"node_id": key, "address": 4096 + 64 * i, "bytes": 32, "immutable": True}
                   for i, key in enumerate(resident)]
        packages.append(SimpleNamespace(description={"resident_bindings": reports}))
        if reports:
            bindings[digest] = {r["node_id"]: {key: r[key] for key in ("address", "bytes", "immutable")}
                                for r in reports}
    record = {"entries": [], "loads": [], "bindings": bindings}
    token = residency._PUBLICATION.set({"residency.json": json.dumps(record).encode()})
    try:
        with codegen_options(CodegenOptions(resident_bindings=bindings)):
            yield tuple(regions), tuple(packages)
    finally:
        residency._PUBLICATION.reset(token)


# check 43: host backing leaves only for a port that every consuming region proves resident.
def _host_backing_removal(directory, patch):
    for index, (rows, removed) in enumerate([
        ([(("arg0", "arg2"), ("arg2",))], ("arg2",)),
        ([(("arg2",), ("arg2",)), (("arg2",), ())], ()),
        ([(("arg2",), ("arg2",)), (("arg2",), ("arg2",))], ("arg2",)),
        ([(("arg3",), ("arg3",))], ()),
        ([], ()),
    ]):
        source, original = _parameters(_subdirectory(directory, f"check_43_{index}"))
        with _proof(rows) as (regions, packages):
            compact = residency.compact_runtime_parameters(source, regions, packages, original)
        if not removed:
            assert compact is original, f"check 43 case {index}"
        else:
            compact.check(source, resident_only=removed)
            assert compact.resident_only == removed, f"check 43 case {index}"
            assert compact.pointer_expression("arg2") == "0", f"check 43 case {index}"
            assert compact.pointer_expression("arg1") != "0", f"check 43 case {index}"
            assert compact.pointer_expression("arg3") != "0", f"check 43 case {index}"
    # check 44: no regions leaves the original backing, and an invalid report keeps it.
    with patch.context():
        source, original = _parameters(_subdirectory(directory, "check_44"))
        assert residency.compact_runtime_parameters(source, (), (), original) is original, "check 44"
        with _proof([(("arg2",), ("arg2",))]) as (regions, packages):
            packages[0].description["resident_bindings"][0]["immutable"] = False
            with _refused(44, match="did not consume"):
                residency.compact_runtime_parameters(source, regions, packages, original)
    # check 45: compacting a compact descriptor again rechecks the original consumer proof.
    with patch.context():
        source, original = _parameters(_subdirectory(directory, "check_45"))
        with _proof([(("arg2",), ("arg2",))]) as (regions, packages):
            compact = residency.compact_runtime_parameters(source, regions, packages, original)
        with _proof([(("arg2",), ("arg2",)), (("arg2",), ())]) as (regions, packages):
            kept = residency.compact_runtime_parameters(source, regions, packages, compact)
        assert kept is original, "check 45"


# ---- residency_decline_tree ----

# check 95: selected residency without external parameters declines and publishes the residency-off tree, and a
# replay that drops the reason is refused.
def _residency_decline_tree(directory, patch):
    """This fixture has no external parameters, so selected residency declines and publishes the residency-off tree."""
    manifest, tree = _deployed(directory / "selected", application_residency=True)
    expected_manifest, expected = _deployed(directory / "off")
    reason = manifest.pop("application_residency_declined")
    assert reason == "the application has no external parameters to keep resident", "check 95"
    assert "application_residency" not in manifest and manifest == expected_manifest, "check 95"
    assert sorted(tree) == sorted(expected) and "resident" not in json.loads(tree["deployment.json"]), "check 95"
    assert {name for name in tree if tree[name] != expected[name]} == {"manifest.json"}, "check 95"
    # Replay recomputes the decision, so a manifest that drops the reason is refused.
    source, plan = _source(2)
    (directory / "selected/application/manifest.json").write_text(json.dumps(manifest))
    with _onchip(application_residency=True), _refused(95, match="residency differs from compilation"):
        verify_semantic_application(source, adapter=DeployedProvider(plan),
                                    snapshot=_snapshot(directory / "selected/snapshot"),
                                    target_source=directory / "selected", artifact=directory / "selected/application")


# ---- resident_section ----

# check 96: an applied residency whose provider names no resident section is refused.
def _resident_section(directory, patch):
    with patch.context() as own:
        # Make the fixture's residency apply, so place() sees it selected; this provider names no resident section.
        own.setattr(residency, "_resident_layouts", lambda *args: ())
        own.setattr(residency, "_resident_publication", lambda *args: nullcontext())
        with _refused(96, NativeCompilationError,
                      match="deployment_unsupported_residency: .* disjoint nonresident"):
            _deployed(directory / "unnamed", application_residency=True)
        named = DeployedProvider(_source(2)[1])
        named.deployment_requirements = lambda *args: {**DeployedProvider.deployment_requirements(named, *args),
                                                       "resident_section": ".device_resident"}
        with _refused(96, Exception) as other:
            _deployed(directory / "named", named, application_residency=True)
        assert "disjoint nonresident storage" not in str(other.value), "check 96"


# ---- resident_arena ----

def _resident_arena(directory, patch):
    """check 131: one read-only arena after the window and inside the capacity, from the active publication."""
    from merlin.compiler.build import deployment

    files, entries, loads = {}, [], []
    for i, (address, data) in enumerate(((4096, b"ab"), (4100, b"cd"))):
        files[f"residency/parameter_{i}.bin"] = data
        entries.append(dict(name=f"parameter_{i}", address=address, bytes=len(data)))
        loads.append(dict(name=f"parameter_{i}", path=f"residency/parameter_{i}.bin", sha256=sha256(data).hexdigest()))
    files["residency.json"] = json.dumps(dict(entries=entries, loads=loads)).encode()
    token = residency._PUBLICATION.set(files)
    try:
        arena = deployment._resident_arena(4096, 8192)
        assert (arena["address"], arena["bytes"]) == (4096, 6), "check 131"
        assert deployment._resident_bytes(arena, residency.publication()) == b"ab\0\0cd", "check 131"
        for window_end, capacity_end in ((4097, 8192), (4096, 4101)):
            with _refused(131, NativeCompilationError, match="resident loads leave the capacity"):
                deployment._resident_arena(window_end, capacity_end)
    finally:
        residency._PUBLICATION.reset(token)


# ---- deployment_admission ----

# checks 97 to 99: a resident deployment without a provider admission hook is refused (97), a refusing hook stops
# it (98), and an admitting hook lets publication go on (99).
def _compile_admission(directory, patch, admit, label):
    """Compile a deployment whose place() publishes a resident arena, with a provider that admits, refuses, or has no hook."""
    from merlin.compiler.build import deployment

    place, calls = deployment.place, []

    def resident(*args, **kwargs):
        published = place(*args, **kwargs)
        return {**published, "resident": dict(address=WINDOW_BASE + WINDOW_BYTES, bytes=64)}

    patch.setattr(deployment, "place", resident)
    adapter = (AdmittingProvider if admit else DeployedProvider)(_source(2)[1])
    if admit:
        def admit_resident_deployment(summaries, regions, native_root, target_source, *, source_request_digest):
            calls.append(source_request_digest)
            if admit == "refuses":
                raise NativeCompilationError("unsupported_semantics", "provider refuses this resident deployment")
        adapter.admit = admit_resident_deployment
    message = {None: "resident deployment requires provider admission",
               "refuses": "provider refuses this resident deployment"}.get(admit)
    if message:
        with _refused(label, NativeCompilationError, match=message):
            _deployed(directory, adapter)
        assert len(calls) == (admit == "refuses"), f"check {label}"
        return
    # Admitted, publication goes on and stops later at this fixture's arena, which has no packed bytes.
    with _refused(label, Exception) as later:
        _deployed(directory, adapter)
    assert "admission" not in str(later.value) and "refuses" not in str(later.value) and calls, (
        f"check {label}: {later.value}")


def _deployment_admission(directory, patch):
    from merlin.compiler.build.regions import _admit_resident_deployment

    with patch.context() as own:
        _compile_admission(_subdirectory(directory, "check_98"), own, "refuses", 98)
    with patch.context() as own:
        _compile_admission(_subdirectory(directory, "check_97"), own, None, 97)
    with patch.context() as own:
        _compile_admission(_subdirectory(directory, "check_99"), own, "admits", 99)
    # check 133: direct admission is skipped without a resident arena and passes the source digest.
    with patch.context():
        source = SimpleNamespace(digest=lambda: "source-digest")
        calls = []
        admitting = AdmittingProvider(_source(2)[1])
        admitting.admit = lambda *args, **kwargs: calls.append((args, kwargs))
        resident = dict(resident=dict(address=8192, bytes=2))
        for published in (None, dict(resident=None)):
            _admit_resident_deployment(published, SimpleNamespace(), "summaries", "regions", "root", None, source)
        for adapter, target_source in ((SimpleNamespace(), "target"), (admitting, None)):
            with _refused(133, NativeCompilationError, match="resident deployment requires provider admission"):
                _admit_resident_deployment(resident, adapter, "summaries", "regions", "root", target_source, source)
        assert calls == [], "check 133"
        _admit_resident_deployment(resident, admitting, "summaries", "regions", "root", "target", source)
        assert calls == [(("summaries", "regions", "root", "target"), dict(source_request_digest="source-digest"))], (
            "check 133")


# ---- application_source_hooks ----

# check 30: a custom staged capture reaches the application source with its own identity, and capability hooks
# that a provider does not declare are refused or never called.
class _UndeclaredHooks(FlatProvider):
    """Every capability hook by name, without declaring LoopBodyBinding, ResidentParameters, or PlatformDeployment."""

    region_loop_parameters = ()
    supports_deployment_loop_sharing = True

    def __init__(self, plan):
        super().__init__(plan)
        self.called = []

    def _hook(self, name):
        self.called.append(name)
        raise AssertionError(f"check 30: undeclared hook {name} was called")

    def bind_loop_region(self, *args, **kwargs):
        return self._hook("bind_loop_region")

    def finalize_loop_regions(self, *args, **kwargs):
        return self._hook("finalize_loop_regions")

    def application_binding(self, *args, **kwargs):
        return self._hook("application_binding")

    def deployment_requirements(self, *args, **kwargs):
        return self._hook("deployment_requirements")

    def resident_memory_geometry(self, *args, **kwargs):
        return self._hook("resident_memory_geometry")

    def inventory_resident_inputs(self, *args, **kwargs):
        return self._hook("inventory_resident_inputs")

    def pack_resident_input(self, *args, **kwargs):
        return self._hook("pack_resident_input")

    def admit_resident_deployment(self, *args, **kwargs):
        return self._hook("admit_resident_deployment")


def _application_source_hooks(directory, patch):
    from merlin.compiler.build.regions import _admit_resident_deployment
    from merlin.compiler.frontend.inputs import stage_compile_inputs
    from merlin.compiler.lower.c.parameters import parameter_source_identity
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError
    from merlin.compiler.semantic.source_preparation import verify_source_preparation
    from merlin.development.targetgen.execution.native.native_application import _source as application_source

    capture, stage = directory / "custom_capture", directory / "custom_stage"
    _write_weighted_capture(capture)
    original = (capture / "model.mlir").read_text()
    custom = original.replace(
        "func.return %input : tensor<2x2xf32>",
        '%unused = "fixture.combine"(%weight, %input) : '
        "(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
        '        %result = "fixture.combine"(%weight, %input) : '
        "(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
        "        func.return %result : tensor<2x2xf32>",
    )
    assert custom != original, "check 30"
    (capture / "model.mlir").write_text(custom)
    trace = json.loads((capture / "frontend-trace.json").read_text())
    trace["mlir"]["sha256"] = hashlib.sha256(custom.encode()).hexdigest()
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    metadata = json.loads((capture / "meta.json").read_text())
    metadata["frontend_trace"]["sha256"] = hashlib.sha256((capture / "frontend-trace.json").read_bytes()).hexdigest()
    (capture / "meta.json").write_text(json.dumps(metadata))
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    for member in ("model.mlir", "frontend-trace.json", "meta.json"):
        raw = (capture / member).read_bytes()
        receipt["artifacts"][member] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    stage_compile_inputs(capture, stage)
    args = SimpleNamespace(
        compile_inputs=stage, mode="diagnostic", linalg_entry=None, application_parameters="external"
    )
    adapter = object()
    received = []

    def translate(op, operands, results):
        return (SemanticNode(results[0], "fixture_combined", operands, TensorType((2, 2), "f32", "raw-bit-pattern")),)

    def factory(policy, *, original_source_sha256, source, staged, compile_inputs):
        assert policy is None and original_source_sha256 == hashlib.sha256(custom.encode()).hexdigest(), "check 30"
        assert compile_inputs == stage, "check 30"
        assert source.source_identity == parameter_source_identity(staged.program_sha256, staged), "check 30"
        assert tuple(n.op for n in source.nodes) == ("input", "input", "fixture_combined"), "check 30"
        assert source.node(source.outputs[0]).inputs == ("arg0", "arg1"), "check 30"
        received.append(source)
        return adapter

    binding = SimpleNamespace(
        application_source_translators=lambda policy: {"fixture.combine": translate},
        application_adapter=factory,
    )
    staged, source, selected, state, folding, parameters = application_source(args, binding, "fixture")
    # The one adapter factory is called once, after source preparation, with the prepared source.
    assert selected is adapter and received == [source] and state is None, "check 30"
    assert parameters.node_ids == ("arg0",), "check 30"
    parameters.check(source)
    unprepared = verify_source_preparation(source, folding)
    assert tuple(node.op for node in unprepared.nodes).count("fixture_combined") == 2, "check 30"

    # A registration supplies semantics only within the original typed boundary.
    def changed_type(op, operands, results):
        return (replace(translate(op, operands, results)[0], type=TensorType((2, 2), "i32", "i32-wrap-k-ascending")),)

    with _refused(30, LinalgBridgeError, match="result types"):
        application_source(
            args,
            SimpleNamespace(
                application_source_translators=lambda policy: {"fixture.combine": changed_type},
                application_adapter=factory,
            ),
            "fixture",
        )
    assert received == [source], "check 30"

    # A binding without the translator hook cannot make an unrecognized source operation enter intake.
    with _refused(30, LinalgBridgeError):
        application_source(args, SimpleNamespace(application_adapter=factory), "fixture")
    assert received == [source], "check 30"

    # A target carrying every capability hook by name but declaring none gets no loop binding, no
    # platform binding, no on-chip placement, no resident admission, and no residency.
    loop_source, plan = _source(2)
    flat = _UndeclaredHooks(plan)
    root = _subdirectory(directory, "undeclared_flat")
    manifest = compile_semantic_application(loop_source, adapter=flat, snapshot=_snapshot(root / "snapshot"),
                                            target_source=root, destination=root / "application", limits=None,
                                            name="model", capacity=4096)
    assert flat.compiles == 3 and flat.binds == 0 and flat.called == [], "check 30"
    assert all("compiled_body_index" not in row for row in manifest["regions"]), "check 30"
    assert "dispatch_loop" not in manifest, "check 30"
    assert not (root / "application/application-loop-plan.json").exists(), "check 30"
    # Only a declared PlatformDeployment can require an on-chip deployment, so the on-chip window compiles flat.
    onchip = _UndeclaredHooks(plan)
    _, tree = _deployed(_subdirectory(directory, "undeclared_onchip"), onchip)
    assert "deployment.json" not in tree and onchip.called == [], "check 30"
    admitted = _UndeclaredHooks(plan)
    with _refused(30, NativeCompilationError, match="resident deployment requires provider admission"):
        _admit_resident_deployment(dict(resident=dict(address=8192, bytes=2)), admitted, "summaries", "regions",
                                   "root", "target", loop_source)
    assert admitted.called == [], "check 30"
    resident = _UndeclaredHooks(plan)
    request, model_parameters = _parameters(_subdirectory(directory, "undeclared_residency"))
    with _refused(30, match="does not declare an immutable residency ABI"):
        residency._resident_layouts(request, resident, model_parameters, None,
                                    CodegenOptions(application_residency=True, io_memory_base=4096, io_memory_bytes=256,
                                                   io_memory_capacity_base=4096, io_memory_capacity_bytes=4096))
    assert resident.called == [], "check 30"


def test_resident_host_backing_and_deployment_admission(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "compact_bytes": _compact_bytes,
        "compact_application_execution": _compact_application_execution,
        "fresh_replay": _fresh_replay,
        "compact_refusals": _compact_refusals,
        "host_backing_removal": _host_backing_removal,
        "residency_decline_tree": _residency_decline_tree,
        "resident_section": _resident_section,
        "resident_arena": _resident_arena,
        "deployment_admission": _deployment_admission,
        "application_source_hooks": _application_source_hooks,
    })
