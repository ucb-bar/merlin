"""Counted C loops and shared region bodies match the flat schedule they replace.

Each row is one table entry of `corefixtures.row_table.run_rows`; every failure names its `check N` label, which
the comment at that check describes. The accelerator region of the shared-loop fixture publishes a one-word
synthetic program, not device code; the loop bodies are host C that executes here.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import re
import shutil
import subprocess

import pytest
from corefixtures.row_table import run_rows

from merlin.compiler.backend.protocol import LoopBodyBinding
from merlin.compiler.build.regions import compile_semantic_application, verify_semantic_application
from merlin.compiler.lower.c.dispatch import DispatchCall, build_program, emit_program_c, prepare_dispatch_loop
from merlin.compiler.partition.dispatch_program import Buffer, DispatchProgram, Node
from merlin.compiler.partition.global_plan_emission import dispatch_digest
from merlin.compiler.schedule.region_loop import analyze_region_loop, expand_region_loop, structural_signature
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.semantic.target_binding import NativeCompilationError
from shared_loop_providers import (
    FlatProvider,
    OnchipFlatProvider,
    OnchipProvider,
    Provider,
    UntypedBindingProvider,
    _onchip,
    _snapshot,
    _source,
    _verify,
    compile_fixture,
)


@contextmanager
def tagged(tag):
    """Name the check label `tag` on any failure of the block, including an exception the library raises."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as failure:
        raise AssertionError(f"{tag}: {failure!r}") from failure


@contextmanager
def refuses(error, tag, match=None):
    """`pytest.raises` whose failure, a missing or a different refusal, names the check label `tag`."""
    try:
        with pytest.raises(error, match=match) as raised:
            yield raised
    except (Exception, pytest.fail.Exception) as failure:
        raise AssertionError(f"{tag}: {failure}") from failure


def _cc(tag):
    compiler = shutil.which("cc")
    assert compiler, f"{tag}: a C compiler is required; this execution gate cannot skip"
    return compiler


def _run(command, tag, **options):
    done = subprocess.run(command, capture_output=True, text=True, **options)
    assert done.returncode == 0, f"{tag}: {done.stdout}{done.stderr}"


# ---- One dispatch program, flat or counted ----

def fixture(count=6, capacity=64):
    buffers = {"x": Buffer("x", [4], "i8", "arg", 0),
               "constant": Buffer("constant", [4], "i8", "const")}
    nodes = []
    previous = "x"
    for i in range(count):
        key = f"v{i}"
        buffers[key] = Buffer(key, [4], "i8", "intermediate")
        nodes.append(Node("dispatch", f"op{i}", [previous, "constant"], [key]))
        previous = key
    graph = DispatchProgram("forward", [0], buffers, nodes, [previous])
    program = build_program(graph, memory_requirements=ArenaRequirements(
        capacity, tuple(2*i for i in range(count)), tuple(2*i+1 for i in range(count)),
        alignment=1, dispatch_digest=dispatch_digest(graph)))
    calls = {i: DispatchCall(f"variant{i}", "device" if i % 2 else "host", context_index=i)
             for i in range(count)}
    return program, calls


def planned():
    """Two repeated regions emitted from a selected plan, as the application render builds them."""
    from merlin.compiler.partition.semantic_regions import outline_semantic_regions
    from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    from semantic_application_fixtures import plan_for

    tensor = TensorType((1,), "f32", "raw-bit-pattern")
    nodes = [SemanticNode(key, "input", (), tensor, effect="input") for key in ("x", "w0", "w1")]
    nodes += [SemanticNode(f"r{i}", "f32_add", (("x", "r0")[i], f"w{i}"), tensor,
                           (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                           index_maps=pointwise_identity_maps(1, 2)) for i in range(2)]
    source = KernelRequest(tuple(nodes), ("r1",), ("external",), tuple((key, "external") for key in ("x", "w0", "w1")),
                           "synthetic")
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=(replace(plan.selected[0], placement="host", implementation="shared_kernel"),
                                   replace(plan.selected[1], placement="host")))
    outlined = outline_semantic_regions(source, entry="test", plan=plan)
    program = build_program(outlined.logical, global_plan=plan, global_plan_emitter=outlined,
                            memory_requirements=lambda graph: ArenaRequirements(
                                256, (0, 2), (1, 3), dispatch_digest=dispatch_digest(graph)))
    return program, {i: DispatchCall(f"variant{i}", "host", context_index=i) for i in range(2)}


COUNTED_CHECK = r'''
#include "flat.h"
#include "counted.h"
#include <assert.h>
#include <string.h>
typedef struct { int variant; } Context;
typedef struct { int variant; void *context; const void *input, *constant; void *output; } Log;
static Context ctx[6]={{0},{1},{2},{3},{4},{5}};
static Log logs[6], expected[6];
static int used, fail=-1, admissions, decision=1;
static int execute(int i, void *c, const void *const *in, void *const *out) {
  assert(c == &ctx[i] && ((Context *)c)->variant == i);
  logs[used++]=(Log){i,c,in[0],in[1],out[0]};
  if (i==fail) return 37;
  for(unsigned j=0;j<4;++j) ((unsigned char *)out[0])[j]=
    ((const unsigned char *)in[0])[j]+((const unsigned char *)in[1])[j]+i;
  return 0;
}
#define WRAPPER(i) int variant##i(void *c,const void *const *in,void *const *out) { return execute(i,c,in,out); }
WRAPPER(0) WRAPPER(1) WRAPPER(2) WRAPPER(3) WRAPPER(4) WRAPPER(5)
static int admit(void *opaque, const counted_loop *loop) {
  assert(opaque==ctx && used==0);
  assert(loop->count==6 && loop->offset==1 && loop->period==2 && loop->trip_count==2);
  for (unsigned i=0;i<6;++i) {
    assert(loop->calls[i].context==&ctx[i]);
    assert(loop->calls[i].input_count==2 && loop->calls[i].output_count==1);
    assert(loop->calls[i].inputs[0]==expected[i].input);
    assert(loop->calls[i].inputs[1]==expected[i].constant);
    assert(loop->calls[i].outputs[0]==expected[i].output);
  }
  ++admissions; return decision;
}
int main(void) {
  unsigned char arena[64], input[4]={1,2,3,4}, constant[4]={7,6,5,4}, result[4], reference[4];
  void *contexts[6]; for(unsigned i=0;i<6;++i) contexts[i]=&ctx[i];
  flat_contexts flat_ctx={contexts,6}; counted_contexts loop_ctx={contexts,6};
  flat_buffer a[]={{input,4},{constant,4},{result,4}};
  counted_buffer b[]={{input,4},{constant,4},{result,4}};
  flat_state fs={0}; counted_state ls={0};
  memset(arena,0xcd,64);
  assert(flat_run(&fs,arena,64,a,3,&flat_ctx)==FLAT_OK);
  memcpy(expected,logs,sizeof(logs)); memcpy(reference,result,4);
  assert(fs.completed==6 && !fs.active);
  for (unsigned j=0;j<4;++j) assert(reference[j]==input[j]+6*constant[j]+15);
  for(int mode=0;mode<3;++mode) {
    used=0; memset(arena,0xcd,64); memset(result,0,4); ls=(counted_state){0};
    decision=mode==0?0:1;
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,mode==2?0:admit,ctx)==COUNTED_OK);
    assert(used==6 && ls.completed==6 && !ls.active);
    assert(!memcmp(reference,result,4));
    for(int i=0;i<6;++i) {
      assert(logs[i].variant==expected[i].variant && logs[i].context==expected[i].context);
      assert(logs[i].input==expected[i].input && logs[i].constant==expected[i].constant && logs[i].output==expected[i].output);
    }
    for(unsigned j=0;j<4;++j) assert(input[j]==j+1 && constant[j]==7-j);
  }
  assert(admissions==2);
  used=0; decision=2; counted_state before=ls;
  assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_INVALID);
  assert(!used && !memcmp(&before,&ls,sizeof(ls)));
  for(int mode=0;mode<2;++mode) {
    used=0; fail=4; decision=mode; ls=(counted_state){0};
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_KERNEL_ERROR);
    assert(ls.active && ls.completed==4 && ls.kernel_status==37 && used==5);
    assert(counted_run_loop(&ls,arena,64,b,3,&loop_ctx,admit,ctx)==COUNTED_BUSY && used==5);
  }
  return 0;
}
'''


def _counted_loop_execution(directory, patch):
    # check 61: the counted loop matches the flat schedule's calls, pointers, admission, and failure.
    root = directory / "check-61"
    root.mkdir()
    compiler = _cc("check 61")
    program, calls = fixture()
    schedule = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    flat, flat_files = emit_program_c(program, "flat", calls)
    loop, loop_files = emit_program_c(program, "counted", calls, loop=schedule)
    assert flat["calls"] == loop["calls"], "check 61"
    assert flat["memory_plan"] == loop["memory_plan"], "check 61"
    assert loop["loop"]["trip_count"] == 2, "check 61"
    for member, content in {**flat_files, **loop_files}.items():
        (root / member).write_text(content)
    # The call table is static storage: a board host declares no heap, so the loop must not allocate.
    assert not any(word in loop_files["counted.c"] for word in ("malloc", "free(", "stdlib")), "check 61"
    (root / "check.c").write_text(COUNTED_CHECK)
    _run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "flat.c", "counted.c", "check.c", "-o", "check"],
         "check 61", cwd=root, timeout=60)
    _run([str(root / "check")], "check 61", timeout=10)

    # check 62: 2,048 trips keep a bounded frame. The original stack table alone exceeded 96 KiB at this
    # scale; check the unoptimized frame so scalar replacement cannot hide an automatic table.
    root = directory / "check-62"
    root.mkdir()
    compiler = _cc("check 62")
    program, calls = fixture(count=2048, capacity=2048*64)
    schedule = prepare_dispatch_loop(program, calls, offset=0, period=1, trip_count=2048)
    _, files = emit_program_c(program, "counted", calls, loop=schedule)
    for member, content in files.items():
        (root / member).write_text(content)
    _run([compiler, "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror", "-Wframe-larger-than=4096",
          "-c", "counted.c", "-o", "counted.o"], "check 62", cwd=root, timeout=60)

    # check 66: a static stack bound follows a function pointer to every address-taken function, so the
    # table holds none and the loop calls each kernel directly.
    program, calls = fixture()
    schedule = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    _, files = emit_program_c(program, "counted", calls, loop=schedule)
    source = files["counted.c"]
    for call in calls.values():
        uses = re.findall(r"\b" + call.symbol + r"\b.{0,24}", source)
        assert uses and all(use.startswith((call.symbol + "(void *", call.symbol + "(call->context"))
                            for use in uses), f"check 66: {uses}"
    assert "unsigned body;" in files["counted.h"] and not re.search(r"\(\*\w+\)\s*\(", source), "check 66"


RESIDENT_CHECK = '''
#include "ordinary.h"
#include "resident.h"
#include <assert.h>
#include <string.h>
static unsigned visits;
int variant0(void *context,const void *const *inputs,void *const *outputs) {
  (void)context; assert(!inputs[0]); ++visits;
  memcpy(outputs[0],inputs[1],4); return 0;
}
int main(void) {
  unsigned char arena[64],constant[4]={1,2,3,4},out[4]={0};
  void *items[1]={0}; resident_contexts contexts={items,1};
  resident_state state={0}; ordinary_state old={0};
  resident_buffer buffers[]={{0,4},{constant,4},{out,4}};
  ordinary_buffer old_buffers[]={{0,4},{constant,4},{out,4}};
  assert(ordinary_run(&old,arena,sizeof(arena),old_buffers,3,&contexts)==ORDINARY_INVALID);
  assert(!visits);
  assert(resident_run(&state,arena,sizeof(arena),buffers,3,&contexts)==RESIDENT_OK);
  assert(visits==1 && memcmp(out,constant,4)==0);
  buffers[1].data=0;
  assert(resident_run(&state,arena,sizeof(arena),buffers,3,&contexts)==RESIDENT_INVALID);
  assert(visits==1);
  return 0;
}
'''


def _resident_only_ports(directory, patch):
    # check 67: only a compiler-declared resident-only port may be null, and required buffers still refuse.
    program, calls = fixture(count=1)
    compiler = _cc("check 67")
    _, ordinary = emit_program_c(program, "ordinary", calls, read_only_inputs=("x",))
    record, resident = emit_program_c(program, "resident", calls, read_only_inputs=("x",),
                                      resident_only_inputs=("x",))
    assert record["resident_only_inputs"] == ["x"], "check 67"
    for key, value in {**ordinary, **resident}.items():
        (directory / key).write_text(value)
    (directory / "check.c").write_text(RESIDENT_CHECK)
    _run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "ordinary.c", "resident.c", "check.c", "-o", "check"],
         "check 67", cwd=directory, timeout=60)
    _run([str(directory / "check")], "check 67", timeout=10)

    # check 68: a resident-only port must be a distinct read-only source port.
    program, calls = fixture(count=1)
    for readonly, resident in (((), ("x",)), (("x",), ("x", "x")), (("x",), ("constant",))):
        with refuses(ValueError, "check 68", match="resident-only"):
            emit_program_c(program, "resident", calls, read_only_inputs=readonly, resident_only_inputs=resident)


# ---- One shared C body with distinct weights and contexts per trip ----

SHARED_CHECK = r"""
#include "combined/model_combined.h"
#include <assert.h>
int main(int argc, char **argv) {
  (void)argv;
  _Alignas(64) unsigned char workspace[MODEL_WORKSPACE_BYTES];
  model_driver_state state={0}; float input=1.0f, output=0, x[2]={0}, y[2]={0}, sum[2]={0};
  float weights[TRIPS]; unsigned visits[TRIPS+1]={0}; void *items[TRIPS+1];
  const float *inputs[TRIPS+3]; float *outputs[]={&output,sum};
  inputs[0]=&input;
  for(unsigned i=0;i<TRIPS;++i) { weights[i]=(float)(i+1); inputs[i+1]=&weights[i]; }
  for(unsigned i=0;i<=TRIPS;++i) items[i]=&visits[i];
  inputs[TRIPS+1]=x; inputs[TRIPS+2]=y;
  if(argc>1) weights[TRIPS-1]+=1.0f;
  model_driver_contexts contexts={items,TRIPS+1};
  assert(model_invoke(&state,workspace,sizeof(workspace),inputs,outputs,&contexts)==0);
  assert(input==1.0f && state.completed==TRIPS+1 && !state.active);
  for(unsigned i=0;i<TRIPS;++i) assert(visits[i]==1);
  return output==(float)(1+TRIPS*(TRIPS+1)/2) ? 0 : 1;
}
"""


def _shared_body_execution(directory, patch):
    # check 75: one shared body runs each trip's weights, replays, and detects a changed manifest.
    tag = "check 75"
    for trips in (2, 22):
        root = directory / f"check-75-trips-{trips}"
        root.mkdir()
        with tagged(tag):
            source, adapter, snapshot, destination, manifest = compile_fixture(root, trips)
        assert adapter.compiles == 2 and adapter.binds == 2 * (trips - 1), tag  # compile, then its own replay
        assert manifest["dispatch_loop"] is True, tag
        assert manifest["application_interface"]["dispatch_loop"]["trip_count"] == trips, tag
        rows = json.loads((destination / "compiled-regions.json").read_bytes())
        assert [row.get("compiled_body_index") for row in rows] == [None] + [0] * (trips - 1) + [None], tag
        assert sorted(path.name for path in (destination / "c-package/regions").iterdir()) == [
            "region_000", f"region_{trips:03d}"], tag
        exports = adapter.exports
        with tagged(tag):
            assert _verify(source, adapter, snapshot, root, destination)["status"] == "checked"
        assert adapter.exports - exports == 2, tag
        package = destination / "c-package"
        (root / "check.c").write_text(SHARED_CHECK)
        compiler = _cc(tag)
        _run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", f"-DTRIPS={trips}",
              "combined/model_combined.c", str(root / "check.c"), "-I", str(package), "-o", str(root / "check")],
             tag, cwd=package, timeout=60)
        _run([str(root / "check")], tag, timeout=10)
        negative = subprocess.run([str(root / "check"), "mutate-last-weight"], capture_output=True, timeout=10)
        assert negative.returncode == 1, tag
        manifest["dispatch_loop"] = False
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with refuses(ValueError, tag, match="differs from replayed"):
            _verify(source, adapter, snapshot, root, destination)

    # check 78: a complete provider binding table is finalized, replayed, and a changed table is refused.
    root = directory / "check-78"
    with tagged("check 78"):
        source, adapter, snapshot, destination, _ = compile_fixture(root, 3, finalize=True)
        table_path = destination / "c-package/regions/region_000/bindings.json"
        table = json.loads(table_path.read_bytes())
        assert len(table) == 3 and len({tuple(ports) for ports in table}) == 3
        assert adapter.compiles == 2
        assert _verify(source, adapter, snapshot, root, destination)["status"] == "checked"
    table[-1] = table[0]
    table_path.write_text(json.dumps(table))
    with refuses(ValueError, "check 78"):
        _verify(source, adapter, snapshot, root, destination)


def _schedule_refusals(directory, patch):
    # check 63: a schedule is refused after any change to what it was prepared from. Each field mutates its
    # own fixture, so this source builds them inside its own context.
    with patch.context():
        for field in ("source", "placement", "symbol", "context", "binding", "memory", "authority", "plan"):
            tag = f"check 63 ({field})"
            program, calls = planned() if field == "plan" else fixture()
            if field == "plan":
                loop = prepare_dispatch_loop(program, calls, offset=0, period=1, trip_count=2)
                emit_program_c(program, "test", calls, loop=loop)
                # The same dispatch under a different selected implementation, with an emission updated to match.
                plan = program.global_plan
                program.global_plan = replace(plan, selected=(
                    replace(plan.selected[0], implementation="different_kernel"), *plan.selected[1:]))
                program.global_plan_emission = replace(program.global_plan_emission,
                                                       plan_digest=program.global_plan.digest)
                with refuses(ValueError, tag, match="loop differs"):
                    emit_program_c(program, "test", calls, loop=loop)
                continue
            loop = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
            if field == "source":
                program.dispatch.nodes[0].op = "changed"
            elif field == "placement":
                calls[3] = replace(calls[3], placement="other")
            elif field == "symbol":
                calls[3] = replace(calls[3], symbol="other")
            elif field == "context":
                calls[3] = replace(calls[3], context_index=0)
            elif field == "binding":
                program.dispatch.nodes[3].inputs.reverse()
            elif field == "memory":
                program.memory_plan.offsets["v0"] += 1
            else:
                loop = replace(loop, authority_sha256="stale")
            with refuses(ValueError, tag):
                emit_program_c(program, "test", calls, loop=loop)

    # check 64: invalid bounds are refused.
    program, calls = fixture()
    for bounds in ((-1, 2, 2), (True, 2, 2), (1, 0, 2), (1, 2, 1), (1, 3, 2)):
        with refuses(ValueError, f"check 64 {bounds}", match="bounds"):
            prepare_dispatch_loop(program, calls, offset=bounds[0], period=bounds[1], trip_count=bounds[2])

    # check 65: a kernel symbol that collides with the loop entry is refused.
    program, calls = fixture()
    calls[0] = replace(calls[0], symbol="test_run_loop")
    loop = prepare_dispatch_loop(program, calls, offset=1, period=2, trip_count=2)
    with refuses(ValueError, "check 65", match="conflicts"):
        emit_program_c(program, "test", calls, loop=loop)


# ---- Region loop records ----

def request(name, op='copy', immediate=1):
    return dict(nodes=[dict(id=name, inputs=[], attrs={'value': immediate}, op=op, effect='pure',
                            type={'dtype': 'i32', 'shape': [1]})],
                outputs=[name], constants=[], input_storages={}, source_identity=name)


def boundary_request(left='left', right='right', output='out'):
    """A complete valid request with declared ordered input boundaries."""
    from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    tensor = TensorType((2,), 'bf16', 'raw-bit-pattern')
    return KernelRequest(
        nodes=(SemanticNode(left, 'input', (), tensor, effect='input'),
               SemanticNode(right, 'input', (), tensor, effect='input'),
               SemanticNode(output, 'bf16_add', (left, right), tensor,
                            attrs=(('numeric_semantics', 'exact'),), index_maps=pointwise_identity_maps(1, 2))),
        outputs=(output,), output_storages=('external',),
        input_storages=((left, 'external'), (right, 'external')), target_identity='test-target',
    ).record()


def _region_loop_expansion(directory, patch):
    # check 69: a region loop expands to the exact flat wiring and refuses changed iterations or bounds.
    flat = [request('prefix', 'start')] + [request(str(i), op, i) for i in range(4) for op in ('a', 'b')]
    flat += [request('suffix', 'end')]
    record = analyze_region_loop(flat, parameter_attributes=('value',))
    assert (record['offset'], record['period'], record['trip_count']) == (1, 2, 4), "check 69"
    assert expand_region_loop(record) == flat, "check 69"
    changed = deepcopy(record)
    changed['iterations'][1][0]['outputs'] = ['wrong']
    with refuses(ValueError, "check 69", match='expansion'):
        expand_region_loop(changed)
    changed = deepcopy(record)
    changed['trip_count'] += 1
    with refuses(ValueError, "check 69", match='bounds'):
        expand_region_loop(changed)

    # check 74: a changed iteration whose flat digest is recomputed is still refused by its structure.
    loop = analyze_region_loop([boundary_request(), boundary_request('x', 'y', 'z')])
    changed = deepcopy(loop)
    changed['iterations'][1][0]['nodes'][-1]['op'] = 'bf16_sub'
    flat = changed['prefix'] + [item for row in changed['iterations'] for item in row] + changed['suffix']
    changed['flat_sha256'] = sha256(json.dumps(flat, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
    with refuses(ValueError, "check 74", match='body structure'):
        expand_region_loop(changed)


def _loop_signature(directory, patch):
    # check 70: signatures keep semantics unless an attribute is declared a parameter.
    a = request('x')
    b = request('y', immediate=2)
    assert structural_signature(a) != structural_signature(b), "check 70"
    assert (structural_signature(a, parameter_attributes=('value',))
            == structural_signature(b, parameter_attributes=('value',))), "check 70"
    b['nodes'][0]['type']['shape'] = [2]
    assert (structural_signature(a, parameter_attributes=('value',))
            != structural_signature(b, parameter_attributes=('value',))), "check 70"
    assert analyze_region_loop([request('x', 'a'), request('y', 'b')]) is None, "check 70"

    # check 71: reordering the declared boundary can make record() select v5 even when all physical
    # metadata is empty. Analysis canonicalizes by operand use; exact expansion still keeps the original
    # ABI and versioned records.
    old = boundary_request()
    ordered = boundary_request('zleft', 'aright', 'result')
    assert old['schema'] == 'merlin.semantic_kernel.v2', "check 71"
    assert ordered['schema'] == 'merlin.semantic_kernel.v5', "check 71"
    assert ordered['physical_extents'] == {}, "check 71"
    assert ordered['boundary_layouts'] == {}, "check 71"
    assert ordered['output_layouts'] == [], "check 71"
    assert structural_signature(old) == structural_signature(ordered), "check 71"
    flat = [deepcopy(old), deepcopy(ordered), deepcopy(old), deepcopy(ordered)]
    loop = analyze_region_loop(flat)
    assert (loop['offset'], loop['period'], loop['trip_count']) == (0, 1, 4), "check 71"
    assert expand_region_loop(loop) == flat, "check 71"

    # check 72: constants keep their bytes, encoding, and binding association.
    source = boundary_request()
    source['nodes'][0].update(op='constant', effect='constant')
    source['input_storages'].pop('left')
    source['constants'] = [dict(node_id='left', encoding='bf16-le-raw', data_hex='803f0040')]
    renamed = deepcopy(source)
    renamed['nodes'][0]['id'] = 'literal'
    renamed['nodes'][-1]['inputs'][0] = 'literal'
    renamed['constants'][0]['node_id'] = 'literal'
    assert structural_signature(source) == structural_signature(renamed), "check 72"
    for field, value in [('data_hex', '803f0140'), ('encoding', 'i32-le'), ('node_id', 'right')]:
        changed = deepcopy(source)
        changed['constants'][0][field] = value
        assert structural_signature(source) != structural_signature(changed), f"check 72 ({field})"

    # check 73: index maps, types, and physical layouts stay significant.
    source = boundary_request()
    changed = deepcopy(source)
    changed['nodes'][-1]['index_maps'][0]['offsets'][0] = 1
    assert structural_signature(source) != structural_signature(changed), "check 73 (index map)"
    for field, value in [('shape', [1, 2]), ('dtype', 'f32'), ('numerical_policy', 'different')]:
        changed = deepcopy(source)
        changed['nodes'][-1]['type'][field] = value
        assert structural_signature(source) != structural_signature(changed), f"check 73 ({field})"
    for field, value in [('physical_extents', {'left': 2, 'right': 2}),
                         ('boundary_layouts', {'left': 'contiguous', 'right': 'contiguous'}),
                         ('output_layouts', ['contiguous'])]:
        changed = deepcopy(source)
        changed[field] = value
        assert structural_signature(source) != structural_signature(changed), f"check 73 ({field})"


# ---- Provider bindings ----

def _provider_port_integrity(directory, patch):
    # check 77: a binding may not reorder ports or change the shared body.
    with refuses(ValueError, "check 77", match="ports differ"):
        compile_fixture(directory / "check-77-ports", 2, bad_ports=True)
    with refuses(ValueError, "check 77", match="differs from its representative"):
        compile_fixture(directory / "check-77-body", 2, changed_body=True)

    # check 79: finalization may not change the ordered runtime ports.
    with refuses(ValueError, "check 79", match="finalization changed region binding"):
        compile_fixture(directory / "check-79", 2, bad_finalize=True)


def _flat_fallback(directory, patch):
    # check 76: a provider that declines, lacks the binding hooks, or names them without declaring
    # LoopBodyBinding compiles every region flat.
    for case, kind in (("decline", Provider), ("no_hook", FlatProvider), ("untyped", UntypedBindingProvider)):
        tag = f"check 76 ({case})"
        root = directory / f"check-76-{case}"
        root.mkdir()
        source, plan = _source(2)
        adapter = kind(plan, decline=case == "decline")
        assert isinstance(adapter, LoopBodyBinding) is (case == "decline"), tag
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=_snapshot(root / "snapshot"),
                                                target_source=root, destination=root / "application", limits=None,
                                                name="model", capacity=4096)
        assert adapter.compiles == 3, tag
        assert (adapter.binds >= 1) if case == "decline" else (adapter.binds == 0), tag
        assert all("compiled_body_index" not in row for row in manifest["regions"]), tag
        if case != "decline":
            assert "dispatch_loop" not in manifest, tag
            assert not (root / "application/application-loop-plan.json").exists(), tag


# ---- A loop whose staging is found after binding publishes exactly what an unbound compile publishes ----

WIDE = (1, 4, 8, 4)  # fits the window, so only the provider can refuse it whole
STAGED = {  # checks by trigger and trip count
    ("region", 2): 82, ("region", 3): 83, ("render", 2): 84, ("render", 3): 85,
    ("first_region", 2): 86, ("first_region", 3): 87, ("bound_body", 2): 88, ("bound_body", 3): 89,
    ("bind_raises", 2): 90, ("bind_raises", 3): 91,
}


def _staged(root, *, bind, trigger, tail_first=False, trips=3, **options):
    """Compile on-chip; `trigger` is "region" (a later region refuses) or "render" (the flat arena overflows)."""
    source, plan = _source(trips, wide=WIDE if trigger == "region" else None, tail_first=tail_first,
                           body=options.pop("body", (1,)))
    adapter = (OnchipProvider if bind else OnchipFlatProvider)(
        plan, whole_limit=WIDE if trigger == "region" else None, **options)
    destination = root / "application"
    with _onchip():
        manifest = compile_semantic_application(
            source, adapter=adapter, snapshot=_snapshot(root / "snapshot"), target_source=root,
            destination=destination, limits=None, name="model", capacity=8 if trigger == "render" else 4096)
        checked = verify_semantic_application(source, adapter=adapter, snapshot=_snapshot(root / "snapshot"),
                                              target_source=root, artifact=destination)
    tree = {path.relative_to(destination).as_posix(): path.read_bytes()
            for path in sorted(destination.rglob("*")) if path.is_file()}
    return adapter, manifest, checked, tree


def _staged_matches_unbound(directory, trigger, trips):
    """`bound_body`: a bound body also refuses when it compiles itself, so the first refusal names the reason.

    `bind_raises`: the provider refuses the binding itself for resource_limit, so that region compiles itself.
    """
    tag = f"check {STAGED[trigger, trips]}"
    root = directory / f"check-{STAGED[trigger, trips]}"
    case = dict(trigger="render" if trigger in ("render", "bind_raises") else "region",
                tail_first=trigger == "first_region", trips=trips)
    if trigger == "bound_body":
        case.update(refuse_label=f"region_{trips - 1:03d}", body=WIDE)
    if trigger == "bind_raises":
        case.update(bind_raises="resource_limit")
    with tagged(tag):
        bound, manifest, checked, tree = _staged(root / "bound", bind=True, **case)
        unbound, expected_manifest, expected_checked, expected = _staged(root / "unbound", bind=False, **case)
    # A region refused before the loop stops binding: those exports would be discarded.
    assert (bound.binds == 0) if trigger == "first_region" else (bound.binds >= 1), tag
    assert unbound.binds == 0, tag
    assert manifest["staged_deployment"] is True and "dispatch_loop" not in manifest, tag
    assert all("compiled_body_index" not in row for row in manifest["regions"]), tag
    assert checked["status"] == expected_checked["status"] == "checked", tag
    assert manifest == expected_manifest, tag
    if trigger == "bound_body":
        assert json.loads(tree["deployment.json"])["staging_reason"].startswith("the body's own compile"), tag
    assert sorted(tree) == sorted(expected), tag
    assert {name for name in tree if tree[name] != expected[name]} == set(), tag


def _staged_equals_unbound(directory, patch):
    # checks 82 to 87: a later-region, render, or first-region refusal, at two and three trips.
    for trigger in ("region", "render", "first_region"):
        for trips in (2, 3):
            _staged_matches_unbound(directory, trigger, trips)


def _staged_binding_refusals(directory, patch):
    # checks 88 to 91: a bound body's own refusal and a binding refusal, at two and three trips.
    for trigger in ("bound_body", "bind_raises"):
        for trips in (2, 3):
            _staged_matches_unbound(directory, trigger, trips)

    # check 81: on-chip loop discovery needs the declared sharing capability. It builds its own on-chip provider
    # without that capability.
    from merlin.compiler.build.regions import _normal_loop
    from merlin.compiler.build.staging import LoopBound
    from merlin.compiler.partition.semantic_regions import outline_semantic_regions

    source, plan = _source(2)
    adapter = OnchipProvider(plan)
    adapter.supports_deployment_loop_sharing = False
    outlined = outline_semantic_regions(source, entry="model", plan=plan)
    with tagged("check 81"):
        assert _normal_loop(outlined, plan, adapter) == (None, {})
        adapter.supports_deployment_loop_sharing = True
        loop, bound = _normal_loop(outlined, plan, adapter)
    assert loop["trip_count"] == 2 and tuple(bound) == (1,) and isinstance(bound[1], LoopBound), "check 81"

    # check 92: a provider that writes while binding is refused when staging is found.
    with refuses(ValueError, "check 92", match="while binding a shared body"):
        _staged(directory / "check-92", bind=True, trigger="render", writes_on_bind=True)

    # check 93: only an on-chip resource-limit binding refusal is a decline.
    with refuses(NativeCompilationError, "check 93") as onchip:
        _staged(directory / "check-93-onchip", bind=True, trigger="render", bind_raises="unsupported_semantics")
    assert onchip.value.status == "unsupported_semantics", "check 93"
    with refuses(NativeCompilationError, "check 93") as offchip:
        compile_fixture(directory / "check-93-offchip", 2, bind_raises="resource_limit")
    assert offchip.value.status == "resource_limit", "check 93"

    # check 94: a reason found before region compilation is published even when a region refuses later.
    # It patches the staging reason, so it opens its own context and builds its own fixtures.
    with patch.context() as own:
        own.setattr("merlin.compiler.build.region_fit.staging_reason", lambda *args: "pre-compile bound")
        for bind in (False, True):
            tag = f"check 94 (bind={bind})"
            with tagged(tag):
                adapter, manifest, checked, tree = _staged(directory / f"check-94-{bind}", bind=bind,
                                                           trigger="region", trips=2)
            assert adapter.binds == 0 and manifest["staged_deployment"] is True, tag
            assert checked["status"] == "checked", tag
            assert json.loads(tree["deployment.json"])["staging_reason"] == "pre-compile bound", tag


def test_counted_and_shared_loops_match_their_flat_authority(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "counted_loop_execution": _counted_loop_execution,
        "resident_only_ports": _resident_only_ports,
        "shared_body_execution": _shared_body_execution,
        "schedule_refusals": _schedule_refusals,
        "region_loop_expansion": _region_loop_expansion,
        "loop_signature": _loop_signature,
        "provider_port_integrity": _provider_port_integrity,
        "flat_fallback": _flat_fallback,
        "staged_equals_unbound": _staged_equals_unbound,
        "staged_binding_refusals": _staged_binding_refusals,
    })
