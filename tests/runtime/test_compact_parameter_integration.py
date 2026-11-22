"""Host backing removal requires every current regional consumer's proof."""
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import shutil
import subprocess

import pytest

from merlin.compiler.schedule.storage import application_residency as residency
from merlin.compiler.lower.c.application import emit_application_c
from merlin.compiler.lower.c.compact_parameters import compact_parameters
from merlin.compiler.lower.c.dispatch import DispatchCall, emit_program_c
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.lower.c.dispatch import build_program
from merlin.compiler.semantic.codegen import CodegenOptions, codegen_options
from merlin.compiler.semantic.model import SemanticNode, TensorType
from semantic_application_fixtures import plan_for
from test_compact_parameters import fixture
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.partition.global_plan_emission import dispatch_digest


@contextmanager
def proof(rows):
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


def test_every_consumer_output_and_unused_port_rules(tmp_path):
    for index, (rows, removed) in enumerate([
        ([(("arg0", "arg2"), ("arg2",))], ("arg2",)),
        ([(("arg2",), ("arg2",)), (("arg2",), ())], ()),
        ([(("arg2",), ("arg2",)), (("arg2",), ("arg2",))], ("arg2",)),
        ([(("arg3",), ("arg3",))], ()),
        ([], ()),
    ]):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        _every_consumer_output_and_unused_port_rules(case, rows, removed)


def _every_consumer_output_and_unused_port_rules(tmp_path, rows, removed):
    source, original = fixture(tmp_path)
    with proof(rows) as (regions, packages):
        compact = residency.compact_runtime_parameters(source, regions, packages, original)
    if not removed:
        assert compact is original
    else:
        compact.check(source, resident_only=removed)
        assert compact.resident_only == removed
        assert compact.pointer_expression("arg2") == "0"
        assert compact.pointer_expression("arg1") != "0"
        assert compact.pointer_expression("arg3") != "0"


def test_disabled_publication_and_invalid_report_do_not_remove_backing(tmp_path):
    source, original = fixture(tmp_path)
    assert residency.compact_runtime_parameters(source, (), (), original) is original
    with proof([(("arg2",), ("arg2",))]) as (regions, packages):
        packages[0].description["resident_bindings"][0]["immutable"] = False
        with pytest.raises(ValueError, match="did not consume"):
            residency.compact_runtime_parameters(source, regions, packages, original)


def test_repeated_compaction_rechecks_original_consumer_proof(tmp_path):
    source, original = fixture(tmp_path)
    with proof([(("arg2",), ("arg2",))]) as (regions, packages):
        compact = residency.compact_runtime_parameters(source, regions, packages, original)
    with proof([(("arg2",), ("arg2",)), (("arg2",), ())]) as (regions, packages):
        kept = residency.compact_runtime_parameters(source, regions, packages, compact)
    assert kept is original


def test_emitted_application_dispatches_null_resident_and_compact_host_ports(tmp_path):
    source, original = fixture(tmp_path)
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
    with pytest.raises(ValueError, match="resident-only"):
        emit_application_c(source, {**driver, "resident_only_inputs": []}, driver_files,
                           "model", capacity=1024, parameters=compact)
    interface, files = emit_application_c(source, driver, driver_files, "model", capacity=1024, parameters=compact)
    assert interface["inputs"] == []
    assert interface["model_parameters"]["payload_bytes"] == 8
    for name, text in {**driver_files, **files}.items():
        (tmp_path / name).write_text(text)
    compact.write_payload(tmp_path / "parameters.bin")
    assert (tmp_path / "parameters.bin").read_bytes() == bytes(range(4)) + bytes(range(12, 16))
    (tmp_path / "main.c").write_text(r'''
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
    compiler = shutil.which("cc")
    assert compiler, "C compiler is required for generated application execution"
    built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            "model.c", "model_driver.c", "main.c", "-o", "check"],
                           cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
