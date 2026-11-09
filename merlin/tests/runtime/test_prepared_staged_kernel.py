"""Explicit staged selections bind source/bytes, arena lifetime and callback ABI."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.common.paths import runtime_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.deployable_outline import canonical_kernel, plan_deployable_chain
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.prepared_staged_kernel import PreparedStagedKernel, reserve_states, verify_selection
from merlin.runtime.staged_dispatch import StagedRoles

SOURCE = """builtin.module {
func.func @forward(%a:tensor<2x4xi32>,%b:tensor<2x4xi32>)->tensor<2x4xi32> {
 %e=tensor.empty():tensor<2x4xi32>
 %r=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}
 ins(%a,%b:tensor<2x4xi32>,tensor<2x4xi32>) outs(%e:tensor<2x4xi32>) {
 ^bb0(%x:i32,%y:i32,%o:i32):
 %z=arith.addi %x,%y:i32
 linalg.yield %z:i32
 }->tensor<2x4xi32>
 func.return %r:tensor<2x4xi32>
}}
"""


def selected(tmp_path):
    planned = plan_deployable_chain(
        parse_mlir_text(SOURCE),
        worker_slots=(0,),
        worker_count=2,
        workspace_budget_bytes=4096,
        alignment_bytes=64,
    )
    kernel = canonical_kernel(planned.outlined.module, planned.dispatches[0]["symbol"])
    header, obj = tmp_path / "provider.h", tmp_path / "provider.o"
    header.write_text("""#include "merlin_staged_kernel.h"
typedef struct {unsigned char data[128];} provider_state;
int provider_prepare(void*,const merlin_staged_kernel_invocation*);
int provider_compute(void*,const merlin_staged_kernel_invocation*,unsigned);
int provider_finish(void*,const merlin_staged_kernel_invocation*);
""")
    # Byte-binding admission is separate from the selected platform's object/link gate.
    obj.write_bytes(b"test-only-object-identity")
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (header, obj)}
    value = PreparedStagedKernel(
        source_ir_sha256=hashlib.sha256(checked_generic_text(kernel).encode()).hexdigest(),
        header=header,
        state_type="provider_state",
        state_bytes=128,
        state_alignment=64,
        prepare="provider_prepare",
        compute="provider_compute",
        finish="provider_finish",
        worker_slots=(0, 1),
        objects=(obj,),
        input_sha256=pins,
        provenance={"schema": "test.explicit_callback_identity.v1", "numerical_gate": False},
    )
    return planned, value


def test_reserves_provider_state_in_whole_chain_budget(tmp_path):
    plan, kernel = selected(tmp_path)
    expanded, layout = reserve_states(plan, {0: kernel}, budget_bytes=192)
    assert plan.workspace_bytes == 64 and expanded.workspace_bytes == 192
    assert layout == {0: {"offset": 64, "bytes": 128, "alignment": 64}}
    verify_selection(expanded, {0: kernel}, layout)
    with pytest.raises(ValueError, match="whole-chain"):
        reserve_states(plan, {0: kernel}, budget_bytes=191)


@pytest.mark.parametrize(
    "changes",
    [
        {"worker_slots": (0, 0)},
        {"worker_slots": (2,)},
        {"state_alignment": 3},
        {"state_bytes": 0},
        {"compute": "bad();"},
        {"objects": ()},
    ],
)
def test_invalid_contract_refuses(tmp_path, changes):
    _, kernel = selected(tmp_path)
    with pytest.raises(ValueError):
        replace(kernel, **changes).verify(worker_count=2)


def test_changed_source_or_bytes_and_overlapping_state_refuse(tmp_path):
    plan, kernel = selected(tmp_path)
    expanded, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    with pytest.raises(ValueError, match="source identity"):
        verify_selection(expanded, {0: replace(kernel, source_ir_sha256="0" * 64)}, layout)
    with pytest.raises(ValueError, match="overlaps"):
        verify_selection(expanded, {0: kernel}, {0: {"offset": 0, "bytes": 128, "alignment": 64}})
    kernel.header.write_text(kernel.header.read_text() + "\n/*changed*/\n")
    with pytest.raises(ValueError, match="input changed"):
        verify_selection(expanded, {0: kernel}, layout)


def test_renderer_separates_provider_callbacks_and_counts_selected_workers(tmp_path):
    plan, kernel = selected(tmp_path)
    plan, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    generated = render_outlined_chain(
        plan, ("unused",), roles=roles, prepared_kernels={0: kernel}, provider_state_layout=layout
    )
    source = generated["source"]
    prepare = source.split("static int prepare", 1)[1].split("static int compute", 1)[0]
    compute = source.split("static int compute", 1)[1].split("static int finish", 1)[0]
    finish = source.split("static int finish", 1)[1]
    assert "provider_prepare(" in prepare and "provider_compute(" not in prepare
    assert "provider_compute(" in compute and "provider_prepare(" not in compute
    assert "provider_finish(" in finish and "provider_compute(" not in finish
    assert "assigned_masks[1]={3}" in source and "_mlir_ciface_unused" not in source
    assert "c->workspace+64" in source and "c->invocation" in source


def test_actual_c_compile_checks_state_size_and_callback_type(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    plan, kernel = selected(tmp_path)
    plan, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    output = render_outlined_chain(
        plan, ("unused",), roles=roles, prepared_kernels={0: kernel}, provider_state_layout=layout
    )
    (tmp_path / "merlin_chain.h").write_text(output["header"])
    (tmp_path / "driver.c").write_text(output["source"])
    argv = [
        cc,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(tmp_path),
        "-I" + str(runtime_dir() / "c"),
        "-c",
        str(tmp_path / "driver.c"),
        "-o",
        str(tmp_path / "driver.o"),
    ]
    subprocess.run(argv, check=True, capture_output=True, timeout=30)
    kernel.header.write_text(kernel.header.read_text().replace("data[128]", "data[129]"))
    failed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert failed.returncode and "prepared state budget" in failed.stderr
    kernel.header.write_text(
        kernel.header.read_text()
        .replace("data[129]", "data[128]")
        .replace("int provider_compute", "unsigned provider_compute")
    )
    failed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert failed.returncode and "prepared callback ABI" in failed.stderr


def test_explicit_prepared_selection_never_falls_back(tmp_path, monkeypatch):
    from merlin.compile.outlined_program import build_outlined_program
    from merlin.llvmlower import kernel_backend, toolchain

    if not toolchain.available():
        pytest.skip("MLIR compiler unavailable")
    plan, kernel = selected(tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError("explicit selection fell back")

    monkeypatch.setattr(kernel_backend, "compile_deployable_kernel", forbidden)
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    kwargs = dict(
        roles=roles,
        worker_slots=(0,),
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
        prepared_staged_kernels={0: kernel},
    )
    report = build_outlined_program(source, tmp_path / "built", **kwargs)
    assert report["status"] == "built" and report["kernel_builds"] == []
    assert report["prepared_kernel_builds"][0]["worker_slots"] == [0, 1]
    assert report["workspace_bytes"] == 192 and report["parent_forward_compiled"] is False
    kernel.header.write_text(kernel.header.read_text() + "\n/*changed*/\n")
    with pytest.raises(ValueError, match="input changed"):
        build_outlined_program(source, tmp_path / "failed", **kwargs)
    assert json.loads((tmp_path / "failed/build.json").read_text())["status"] == "failed"


def test_actual_pitched_descriptor_bounds_are_transactional(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    source = r"""
#include "merlin_model.h"
#include <assert.h>
#include <string.h>
static _Alignas(64) unsigned char bytes[4096];
int main(void){
 merlin_descriptor_t d,before;int64_t sizes[2]={2,17},strides[2]={1,3},words[19];
 memset(&d,0x5a,sizeof d);
 assert(!merlin_prepare_strided_descriptor(bytes,51,1,2,0,sizes,strides,&d));
 memcpy(words,&d,sizeof words);assert(words[2]==0&&words[3]==2&&words[4]==17&&words[5]==1&&words[6]==3);
 sizes[0]=1;sizes[1]=1;strides[0]=65;strides[1]=1;
 assert(!merlin_prepare_strided_descriptor(bytes,780,4,2,194,sizes,strides,&d));
 memcpy(words,&d,sizeof words);assert(words[2]==194&&words[3]==1&&words[4]==1&&words[5]==65&&words[6]==1);
 before=d;
 assert(merlin_prepare_strided_descriptor(bytes,779,4,2,194,sizes,strides,&d));
 assert(!memcmp(&d,&before,sizeof d));
 strides[1]=0;assert(merlin_prepare_strided_descriptor(bytes,780,4,2,0,sizes,strides,&d));
 strides[1]=-1;assert(merlin_prepare_strided_descriptor(bytes,780,4,2,0,sizes,strides,&d));strides[1]=1;
 assert(merlin_prepare_strided_descriptor(bytes,780,4,2,-1,sizes,strides,&d));
 assert(merlin_prepare_strided_descriptor(bytes,780,3,2,0,sizes,strides,&d));
 assert(merlin_prepare_strided_descriptor(bytes+1,780,4,2,0,sizes,strides,&d));
 assert(merlin_prepare_strided_descriptor((void*)(UINTPTR_MAX-3),780,4,2,0,sizes,strides,&d));
 sizes[0]=INT64_MAX;sizes[1]=2;assert(merlin_prepare_strided_descriptor(bytes,SIZE_MAX,4,2,0,sizes,strides,&d));
 assert(!memcmp(&d,&before,sizeof d));
 sizes[0]=1;sizes[1]=1;
 assert(merlin_prepare_strided_descriptor(bytes,sizeof bytes,1,2,0,sizes,strides,(merlin_descriptor_t*)(bytes+128)));
 assert(!memcmp(&d,&before,sizeof d));return 0;
}
"""
    (tmp_path / "pitched.c").write_text(source)
    runtime = runtime_dir() / "c"
    exe = tmp_path / "pitched"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(runtime),
            str(tmp_path / "pitched.c"),
            str(runtime / "merlin_descriptors.c"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)
