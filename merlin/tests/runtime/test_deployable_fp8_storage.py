"""Standard FP8 storage/descriptor admission is not arithmetic qualification."""

import hashlib
import shutil
import subprocess

import pytest
from xdsl.dialects.builtin import Float16Type, TensorType

from merlin.common.paths import runtime_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.deployable_outline import canonical_kernel, plan_deployable_chain, static_tensor_contract
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.prepared_staged_kernel import PreparedStagedKernel, reserve_states, verify_selection
from merlin.runtime.staged_dispatch import StagedRoles


def source(dtype):
    return f"""builtin.module {{
func.func @forward(%a:tensor<3x17x{dtype}>)->tensor<3x17xf32> {{
 %e=tensor.empty():tensor<3x17xf32>
 %r=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]}}
 ins(%a:tensor<3x17x{dtype}>) outs(%e:tensor<3x17xf32>) {{
 ^bb0(%x:{dtype},%o:f32):
 %z=arith.extf %x:{dtype} to f32
 linalg.yield %z:f32
 }}->tensor<3x17xf32>
 func.return %r:tensor<3x17xf32>
}}}}
"""


def plan(dtype):
    return plan_deployable_chain(
        parse_mlir_text(source(dtype)),
        worker_slots=(0,),
        worker_count=1,
        workspace_budget_bytes=4096,
        alignment_bytes=64,
    )


@pytest.mark.parametrize("dtype", ["f8E4M3FN", "f8E5M2"])
def test_standard_fp8_parses_verifies_and_keeps_one_byte_storage(dtype):
    module = parse_mlir_text(source(dtype))
    module.verify()
    before = checked_generic_text(module)
    function = next(iter(module.body.block.ops))
    element = function.function_type.inputs.data[0].element_type
    assert element.bitwidth == 8
    assert static_tensor_contract(function.function_type.inputs.data[0]) == ((3, 17), dtype, 51)
    planned = plan(dtype)
    assert [(b.shape, b.dtype, b.bytes) for b in planned.buffers if b.arg_index is not None] == [((3, 17), dtype, 51)]
    assert any(b.dtype == "f32" and b.bytes == 204 for b in planned.buffers)
    assert checked_generic_text(module) == before
    # Accepting encoded storage does not silently encode Python float values.
    with pytest.raises(NotImplementedError, match="packing"):
        element.pack([1.0])


def test_other_unadmitted_storage_is_not_inferred_by_width():
    with pytest.raises(ValueError, match="dtypes"):
        static_tensor_contract(TensorType(Float16Type(), (3, 17)))


def selected(tmp_path, planned):
    header, obj = tmp_path / "provider.h", tmp_path / "provider.o"
    header.write_text("""#include "merlin_staged_kernel.h"
typedef struct {unsigned char data[64];} provider_state;
int provider_prepare(void*,const merlin_staged_kernel_invocation*);
int provider_compute(void*,const merlin_staged_kernel_invocation*,unsigned);
int provider_finish(void*,const merlin_staged_kernel_invocation*);
""")
    # Metadata-only test: object execution/link admission is explicitly absent.
    obj.write_bytes(b"test-only-identity-not-executable")
    kernel = canonical_kernel(planned.outlined.module, planned.dispatches[0]["symbol"])
    return PreparedStagedKernel(
        hashlib.sha256(checked_generic_text(kernel).encode()).hexdigest(),
        header,
        "provider_state",
        64,
        64,
        "provider_prepare",
        "provider_compute",
        "provider_finish",
        (0,),
        (obj,),
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (header, obj)},
        {"schema": "test.explicit_storage_only.v1", "execution_qualified": False},
    )


def test_equal_byte_width_does_not_merge_distinct_fp8_source_formats(tmp_path):
    original = plan("f8E4M3FN")
    kernel = selected(tmp_path, original)
    expanded, layout = reserve_states(original, {0: kernel}, budget_bytes=4096)
    verify_selection(expanded, {0: kernel}, layout)
    other, other_layout = reserve_states(plan("f8E5M2"), {0: kernel}, budget_bytes=4096)
    with pytest.raises(ValueError, match="source identity"):
        verify_selection(other, {0: kernel}, other_layout)


@pytest.mark.parametrize("dtype", ["f8E4M3FN", "f8E5M2"])
def test_actual_driver_c_abi_compile_requires_no_hidden_float_encoder(tmp_path, dtype):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    planned = plan(dtype)
    kernel = selected(tmp_path, planned)
    planned, layout = reserve_states(planned, {0: kernel}, budget_bytes=4096)
    roles = StagedRoles((0, 1), 1, 1, 1, 1, (0,), (0,))
    generated = render_outlined_chain(
        planned,
        ("not_executed",),
        roles=roles,
        prepared_kernels={0: kernel},
        provider_state_layout=layout,
    )
    (tmp_path / "merlin_chain.h").write_text(generated["header"])
    source_file = tmp_path / "driver.c"
    source_file.write_text(generated["source"])
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(tmp_path),
            "-I" + str(runtime_dir() / "c"),
            "-c",
            str(source_file),
            "-o",
            str(tmp_path / "driver.o"),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert "provider_prepare(" in generated["source"] and "provider_compute(" in generated["source"]
    assert "_mlir_ciface_not_executed" not in generated["source"]
