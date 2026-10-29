"""Live selected sibling effects reach normal native lowering.

Neutral C kernels test host ownership and every result. They establish no
device implementation qualification or performance result.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_sibling_source_route import route
from test_source_region import sibling_source
from xdsl.dialects import builtin, func

from merlin.common import invocation_record as I
from merlin.common.digest import sha256_file
from merlin.llvmlower import device_preparation as P
from merlin.llvmlower import device_shim as D
from merlin.llvmlower import sibling_source_route as S
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.device_build import DeviceBuild
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.toolchain import clang
from merlin.runtime.device_borrowed_calls import sibling_borrowed_calls


@pytest.fixture
def issued(tmp_path, monkeypatch, upstream_host_tools, request):
    module = sibling_source()
    forward = module.body.block.first_op
    block = forward.body.block
    cast = next(op for op in block.ops if op.name == "linalg.generic" and str(op.result_types[0]) == "tensor<2x2xf32>")
    empty = cast.operands[-1].owner.clone()
    second = cast.clone(value_mapper={cast.operands[0]: block.last_op.arguments[1], cast.operands[-1]: empty.results[0]})
    if not getattr(request, "param", False):
        block.insert_ops_before((empty, second), block.last_op)
        returned = func.ReturnOp(block.last_op.arguments[0], second.results[0])
        block.insert_op_before(returned, block.last_op)
        block.erase_op(block.last_op)
        forward.properties["function_type"] = builtin.FunctionType.from_lists(
            forward.function_type.inputs, (cast.result_types[0], second.result_types[0]))
    _, source, selected, device, work = route(tmp_path, monkeypatch, module=module)
    invocation = P.freeze_device_offload_selection(device)
    prepared = P.prepare_sibling_device_host(device, source, work, invocation)
    P.prepare_device_route(device, prepared, work, selection=invocation)
    P.retain_device_offload_source(invocation, prepared)
    host = invocation.sibling_host
    buffers, reason = D.logical_buffers(host.interface)
    assert not reason
    unit = D.emit_logical_translation_unit(device.device,
        {host.device_symbol: (buffers, host.rewrite.call_buffers[host.device_symbol])},
        kernel_symbol_for=lambda _: "neutral_kernel")
    unit = S.with_borrowed_bridges(unit, selected, host)
    kernel = work / "kernel.c"
    kernel.write_text("""#include <stdint.h>
void neutral_kernel(int8_t *a,int8_t *b,int8_t *c,int32_t *x,int32_t *y) {
 for(int m=0;m<2;m++) for(int n=0;n<2;n++) {
  int32_t u=0,v=0; for(int k=0;k<2;k++) {u+=a[m*2+k]*b[k*2+n];v+=a[m*2+k]*c[k*2+n];}
  x[m*2+n]=u; y[m*2+n]=v;
 }
}
""")
    kernel_o, shim_c, shim_o = work / "kernel.o", work / "shim.c", work / "shim.o"
    shim_c.write_text(unit.text)

    def compile(source, output, stage):
        result = I.run([clang(), "-fPIC", "-O2", "-c", str(source), "-o", str(output)],
            directory=work / "invocations", stage=stage, inputs=(source,), outputs=(output,), capture_output=True)
        assert result.returncode == 0, result.stderr
        path = max((work / "invocations").rglob("invocation.json"), key=lambda p: p.stat().st_mtime_ns)
        return dict(path=str(path), sha256=sha256_file(path))

    kernel_pin = compile(kernel, kernel_o, "neutral_conditional_kernel_compilation")
    shim_pin = compile(shim_c, shim_o, "device_shim_compilation")
    # Repeated roles explicitly describe a neutral custody fixture, not an
    # ordinary device backend's translation/optimization qualification.
    custody = {role: kernel_pin for role in ("translation", "optimization", "optimized_import", "compilation", "rename")}
    custody["sibling_source"] = dict(source_ir_sha256=selected.source_ir_sha256,
        interface_sha256=S._sha(host.interface), borrowed_effects=selected.record()["effects"],
        original_epilogues_and_all_live_outputs_preserved=True,
        descriptor_bridge={
            **{role: dict(path=str(path.resolve()), sha256=sha256_file(path), size_bytes=path.stat().st_size)
               for role, path in (("source", shim_c), ("object", shim_o))},
            "compilation": shim_pin,
        })
    built = DeviceBuild(device.device, (kernel_o, shim_o), shim_o,
        {host.device_symbol: "neutral_kernel"}, custody={host.device_symbol: custody},
        shim_unit=unit, shim_invocation=shim_pin)
    calls = sibling_borrowed_calls(device, invocation, built, prepared)
    return device, invocation, built, prepared, calls


def test_actual_sibling_roster_reaches_native_lifetime_and_complete_outputs(issued, tmp_path):
    _, invocation, built, prepared, calls = issued
    assert len(calls.calls) == 1 and calls.calls[0].result_arguments == (3, 4)
    assert calls.calls[0].effects is invocation.sibling_source[0].effects
    lowered = lower_model_file(prepared, tmp_path / "lower", targets=(), vectorize=False,
                               borrowed_buffer_calls=calls, features=["inline_tensor_helpers_before_bufferization"])
    assert lowered.borrowed_arena_lifetime is not None
    accepted, receipt = lowered.borrowed_arena_lifetime.checked(lowered.ll_path.read_text())
    assert calls.calls[0].borrowed_symbol in accepted and receipt["status"] == "checked"
    image = tmp_path / "host.so"
    result = I.run([clang(), "-shared", "-fPIC", "-O2", str(lowered.ll_path),
                    *map(str, built.objects), "-lm", "-o", str(image)],
                   directory=tmp_path / "link", stage="neutral_sibling_host_link",
                   inputs=(lowered.ll_path, *built.objects), outputs=(image,), capture_output=True)
    assert result.returncode == 0, result.stderr
    model = HostModel.load(image, image_policy=PrivateHostImagePolicy(tmp_path))
    a = np.array([[1, -2], [3, 4]], dtype=np.int8)
    b = np.array([[2, 5], [-3, 1]], dtype=np.int8)
    c = np.array([[-2, 3], [7, -1]], dtype=np.int8)
    guarded = [np.full(8, -999.0, dtype=np.float32) for _ in range(2)]
    outputs = [v[2:-2].reshape(2, 2) for v in guarded]
    originals = [v.copy() for v in (a, b, c)]
    model([(v.ctypes.data, v.shape) for v in (a, b, c, *outputs)])
    for actual, rhs in zip(outputs, (b, c.T), strict=True):
        np.testing.assert_array_equal(actual, a.astype(np.int32) @ rhs.astype(np.int32))
    assert all(np.all(v[:2] == -999.0) and np.all(v[-2:] == -999.0) for v in guarded)
    assert all(np.array_equal(actual, original) for actual, original in zip((a, b, c), originals, strict=True))
    # The original live native observer can reopen precisely these products.
    calls.observe_native(tuple(Path(p["path"]) for p in receipt["native_products"]
                               if Path(p["path"]).name != "lower_to_llvm.py"), False)


@pytest.mark.parametrize("issued", [True], indirect=True)
def test_public_plural_output_has_no_implicit_disjointness_permission(issued, tmp_path):
    _, _, _, prepared, calls = issued
    from merlin.llvmlower.pipeline import PipelineError

    with pytest.raises(PipelineError, match="plural caller results require distinct fresh allocation owners"):
        lower_model_file(prepared, tmp_path / "lower", targets=(), vectorize=False,
                         borrowed_buffer_calls=calls, features=["inline_tensor_helpers_before_bufferization"])


@pytest.mark.parametrize("damage", ["effects_copy", "both_effects_copy", "effects_flag", "writer", "writer_copy",
                                    "target", "host", "source", "shim", "object", "roster"])
def test_changed_live_owner_or_real_compilation_cannot_refresh_effects(issued, damage):
    device, invocation, built, prepared, calls = issued
    host = invocation.sibling_host
    if damage == "effects_copy":
        object.__setattr__(host.writer, "borrowed_effects", replace(host.writer.borrowed_effects))
    elif damage == "both_effects_copy":
        copied = replace(host.writer.borrowed_effects)
        object.__setattr__(host.writer, "borrowed_effects", copied)
        object.__setattr__(device.sibling_source, "effects", copied)
    elif damage == "effects_flag":
        object.__setattr__(host.writer.borrowed_effects, "complete_before_return", False)
    elif damage == "writer":
        object.__setattr__(host.writer, "result_arguments", (4, 3))
    elif damage == "writer_copy":
        object.__setattr__(host, "writer", replace(host.writer))
    elif damage == "target":
        object.__setattr__(device, "device", "another-target")
    elif damage == "host":
        invocation.sibling_host = replace(host)
    elif damage == "source":
        prepared.write_text(prepared.read_text() + "\n//changed\n")
    elif damage == "shim":
        object.__setattr__(built.shim_unit, "text", built.shim_unit.text + "\n")
    elif damage == "object":
        built.objects[0].write_bytes(built.objects[0].read_bytes() + b"changed")
    else:
        built.kernels.clear()
    with pytest.raises(ValueError, match="changed|original|effects"):
        calls.revalidate()


def test_unselected_route_is_inert():
    assert sibling_borrowed_calls(None, None, None, None) is None


@pytest.mark.parametrize("damage", ["missing", "source", "object", "compilation", "extra"])
def test_ordinary_descriptor_bridge_custody_is_exact_before_issuance(issued, damage):
    device, invocation, built, prepared, _ = issued
    row = built.custody[invocation.sibling_host.device_symbol]["sibling_source"]
    if damage == "missing":
        del row["descriptor_bridge"]
    elif damage == "extra":
        row["descriptor_bridge"]["unobserved"] = True
    elif damage == "compilation":
        row["descriptor_bridge"][damage] = {**built.shim_invocation, "sha256": "0" * 64}
    else:
        row["descriptor_bridge"][damage]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original writer custody"):
        sibling_borrowed_calls(device, invocation, built, prepared)
