"""Frozen arguments compose with ordered, runtime static concat transport."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.common.paths import runtime_dir
from merlin.compile.frozen_inputs import FrozenInput, FrozenInputs, specialize
from merlin.compile.outlined_program import build_outlined_program
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.deployable_outline import canonical_kernel, plan_deployable_chain
from merlin.llvmlower.lower import lower_model
from merlin.runtime.staged_dispatch import StagedRoles

SOURCE = """module {
func.func @forward(%w:tensor<2x3xi32>,%x:tensor<2x5xi32>)->tensor<2x11xi32> {
 %cat="tensor.concat"(%w,%x,%w) <{dim=1:i64}> :
   (tensor<2x3xi32>,tensor<2x5xi32>,tensor<2x3xi32>)->tensor<2x11xi32>
 %e=tensor.empty():tensor<2x11xi32>
 %out=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
   iterator_types=["parallel","parallel"]}
   ins(%cat:tensor<2x11xi32>)outs(%e:tensor<2x11xi32>){
 ^bb0(%a:i32,%old:i32):
   %seven=arith.constant 7:i32
   %v=arith.addi %a,%seven:i32
   linalg.yield %v:i32
 }->tensor<2x11xi32>
 func.return %out:tensor<2x11xi32>
}}
"""


def _plan():
    return plan_deployable_chain(
        parse_mlir_text(SOURCE),
        worker_slots=(0,),
        worker_count=1,
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        data_preparation=True,
    )


def _inputs(tmp_path, plan, all_frozen):
    # Independent bit patterns include both signed extremes and wrap on +7.
    weight = np.array([[2**31 - 1, -1, -(2**31)], [17, 255, -257]], dtype="<i4")
    dynamic = np.array([[0, 2**31 - 4, -5, 99, -101], [3, -4, 5, -6, 7]], dtype="<i4")
    bound = []
    for index, value in enumerate((weight, dynamic)):
        if index == 1 and not all_frozen:
            continue
        path = tmp_path / f"source_arg_{index}.bin"
        path.write_bytes(value.tobytes())
        bound.append(FrozenInput(index, "i32", value.shape, path, hashlib.sha256(value.tobytes()).hexdigest()))
    return (
        FrozenInputs(plan.source_ir_sha256, tuple(bound), 1 << 20, 4096),
        weight,
        dynamic,
    )


@pytest.mark.parametrize("all_frozen", [False, True])
def test_pending_concat_keeps_all_ordered_operands_and_remaps_external_abi(tmp_path, all_frozen):
    original = _plan()
    before = original.to_dict()
    request, _, _ = _inputs(tmp_path, original, all_frozen)
    derived, blob = specialize(original, request, tmp_path / "offline")
    assert original.to_dict() == before
    assert len(derived.dispatches) == 1
    original_view = original.dispatches[0]["views_before"][0]
    view = derived.dispatches[0]["views_before"][0]
    assert view == original_view
    assert view["kind"] == "pack_concat"
    assert len(view["inputs"]) == 3 and view["inputs"][0] == view["inputs"][2]
    assert view["input_block_bytes"] == [12, 20, 12]
    assert derived.frozen_inputs["external_input_source_indices"] == ([] if all_frozen else [1])
    assert derived.frozen_inputs["dispatch_mapping"] == [
        {
            "source_dispatch_index": 0,
            "symbol": original.dispatches[0]["symbol"],
            "folded": False,
            "runtime_dispatch_index": 0,
        }
    ]
    assert hashlib.sha256(blob).hexdigest() == derived.frozen_inputs["published_blob_sha256"]


def _bytes(values):
    return ",".join(map(str, values))


@pytest.mark.parametrize("all_frozen", [False, True])
def test_real_build_and_generated_concat_execute_full_words_and_guards(tmp_path, all_frozen):
    if not os.getenv("MERLIN_MLIR_INSTALL") or not os.getenv("MERLIN_COMPILER_PYTHON"):
        pytest.skip("explicit selected compiler required for real host/target objects")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    plan = _plan()
    request, weight, dynamic = _inputs(tmp_path, plan, all_frozen)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,))
    work = tmp_path / "build"
    built = build_outlined_program(
        source,
        work,
        roles=roles,
        worker_slots=(0,),
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
        data_preparation=True,
        frozen_inputs=request,
    )
    assert built["status"] == "built" and built["dispatches"] == 1
    assert built["driver_metadata"]["external_input_source_indices"] == ([] if all_frozen else [1])
    assert all(path in built["input_sha256"] for path in built["link_objects"])
    assert all(str(item.file) in built["input_sha256"] for item in request.inputs)
    driver = (work / "merlin_chain.c").read_text()
    header = (work / "merlin_chain.h").read_text()
    if all_frozen:
        # C reserves one inert array slot rather than a nonstandard zero-sized
        # member; the actual external ABI roster and every access are empty.
        assert "void *inputs[1]" in header and "c->inputs[" not in driver
    else:
        assert "void *inputs[1]" in header and "c->inputs[1]" not in driver
        assert "c->inputs[0]" in driver

    # Compile the very same isolated arithmetic IR for the host, rather than
    # substituting a handwritten arithmetic callback for the transport check.
    entry = built["kernel_builds"][0]["entry"]
    kernel = canonical_kernel(plan.outlined.module, plan.dispatches[0]["symbol"])
    function = next(op for op in kernel.body.block.ops if op.name == "func.func")
    function.properties["sym_name"] = StringAttr(entry)
    host = lower_model(checked_generic_text(kernel), tmp_path / "host", textual=True, targets=("host",))
    blob = (work / "harness_blob_merlin_chain_frozen_blob.bin").read_bytes()
    record = json.loads(Path(built["plan"]).read_text())
    positions = {buffer["id"]: index for index, buffer in enumerate(record["buffers"])}
    view = record["dispatches"][0]["views_before"][0]
    concatenated = np.concatenate((weight, dynamic, weight), axis=1)
    expected = np.add(concatenated, np.int32(7), dtype=np.int32)
    arena = record["workspace_bytes"]
    # The fixture exercises generated callbacks with explicit role transitions.
    # It verifies transport and real host kernel arithmetic, not a thread/target
    # timing claim. Staged-runtime concurrency has its own existing tests.
    dynamic_binding = "" if all_frozen else ("c.inputs[0]=input+64;c.input_capacity[0]=sizeof input-128;")
    fixture = tmp_path / "transport.c"
    fixture.write_text(
        '#include "merlin_chain.c"\n#include <assert.h>\n'
        + "const _Alignas(64) unsigned char merlin_chain_frozen_blob["
        + str(len(blob))
        + "]={"
        + _bytes(blob)
        + "};\n"
        + "static unsigned self;static unsigned current_hart(void*p){(void)p;return self;}\n"
        + "static _Alignas(64) unsigned char arena["
        + str(arena + 128)
        + "],input[168],output[216];static merlin_chain_context c;\n"
        + "int main(void){const unsigned char original[]={"
        + _bytes(dynamic.tobytes())
        + "},packed[]={"
        + _bytes(concatenated.tobytes())
        + "},expected[]={"
        + _bytes(expected.tobytes())
        + "};unsigned char saved_blob[sizeof merlin_chain_frozen_blob];"
        + "memcpy(saved_blob,merlin_chain_frozen_blob,sizeof saved_blob);"
        + "memset(arena,0xa7,sizeof arena);memset(input,0xa7,sizeof input);memset(output,0xa7,sizeof output);"
        + "memcpy(input+64,original,sizeof original);"
        + dynamic_binding
        + "c.outputs[0]=output+64;c.output_capacity[0]=sizeof expected;"
        + "c.workspace=arena+64;c.workspace_capacity="
        + str(arena)
        + ";c.platform_hooks.current_hart=current_hart;"
        + "c.team.config.controller_hart=2;c.team.config.worker_count=1;"
        + "c.team.config.worker_harts[0]=0;c.team.config.sequence=1;self=2;"
        + "assert(model_init(&c)==0);assert(prepare(&c)==0);"
        + "assert(!memcmp(c.buffers["
        + str(positions[view["output"]])
        + "],packed,sizeof packed));"
        + "self=0;assert(compute(&c,0)==0);"
        + "for(unsigned i=0;i<sizeof expected;i++)assert(output[64+i]==0xa7);"
        + "atomic_store_explicit(&c.team.done,1,memory_order_release);self=2;assert(finish(&c)==0);"
        + "assert(!memcmp(output+64,expected,sizeof expected));"
        + "assert(!memcmp(input+64,original,sizeof original));"
        + "assert(!memcmp(saved_blob,merlin_chain_frozen_blob,sizeof saved_blob));"
        + "for(unsigned i=0;i<64;i++){assert(arena[i]==0xa7&&arena["
        + str(arena + 64)
        + "+i]==0xa7);assert(input[i]==0xa7&&input[104+i]==0xa7);"
        + "assert(output[i]==0xa7&&output[152+i]==0xa7);}"
        + "assert(c.kernel_calls[0]==1&&c.completed_dispatches==1);return 0;}\n"
    )
    runtime = runtime_dir() / "c"
    exe = tmp_path / "transport"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(work),
            "-I" + str(runtime),
            str(fixture),
            str(runtime / "merlin_descriptors.c"),
            str(runtime / "merlin_staged.c"),
            str(host.host_so),
            "-Wl,-rpath," + str(host.host_so.parent),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)
