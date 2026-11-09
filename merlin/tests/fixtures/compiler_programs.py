"""Neutral compiler program builders shared across test subsystem boundaries."""

import inspect
import json
from pathlib import Path

import numpy as np

from merlin.llvmlower.device_offload import DeviceRewrite, Routed
from merlin.llvmlower.host_transform_chain import (
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformChain,
    HostLLVMTransformStage,
)


def contraction_source(
    a_shape,
    b_shape,
    c_shape,
    *,
    transpose=False,
    alias=False,
    fastmath=False,
    reversed_add=False,
):
    rank = len(c_shape)
    dims = [f"d{i}" for i in range(rank + 1)]
    batch = dims[:-3]
    i, j, k = dims[-3:]
    maps = [batch + [i, k], batch + ([j, k] if transpose else [k, j]), batch + [i, j]]
    amap = ",".join(f"affine_map<({','.join(dims)})->({','.join(m)})>" for m in maps)
    iters = ",".join([*(['"parallel"'] * rank), '"reduction"'])

    def typ(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    ta, tb, tc = map(typ, [a_shape, b_shape, c_shape])
    arguments = f"%a: {ta}, %b: {tb}" + ("" if alias else f", %c: {tc}")
    initial = "%a" if alias else "%c"
    flags = " fastmath<contract>" if fastmath else ""
    add_args = "%z,%p" if reversed_add else "%p,%z"
    return f"""module {{ func.func @forward({arguments}) -> {tc} attributes {{llvm.emit_c_interface}} {{
      %r = linalg.generic {{indexing_maps=[{amap}], iterator_types=[{iters}]}}
      ins(%a,%b:{ta},{tb}) outs({initial}:{tc}) {{
      ^bb0(%x:f32,%y:f32,%z:f32):
        %p = arith.mulf %x,%y{flags}:f32
        %s = arith.addf {add_args}:f32
        linalg.yield %s:f32
      }} -> {tc}
      return %r:{tc}
    }} }}"""


def ordered_contraction(a, b, c):
    result = c.copy()
    for k in range(a.shape[-1]):
        product = a[..., :, k, None] * b[..., k, None, :]
        result = np.float32(product + result)
    return result


def device_signature_source(work):
    work.mkdir(parents=True, exist_ok=True)
    DeviceRewrite(
        "synthetic",
        signatures={"a": (2, 2, 2)},
        routed=(Routed("a", (2, 2), (2,), ("i8", "i8", "i32")),),
    ).write_sidecar(work)


def host_transform_stage(tmp_path, name, *, transform=None, verify=None):
    proof = tmp_path / (name + "_permission.json")
    proof.write_text('{"source_arithmetic_unchanged":true}\n')

    def copy(original, work):
        selected = work / "selected.ll"
        selected.write_bytes(original.read_bytes())
        return selected

    def verify_copy(original, selected, work):
        if original.read_bytes() != selected.read_bytes():
            raise ValueError("copy stage changed the complete original IR")
        evidence = work / "emission.json"
        evidence.write_text(json.dumps({"complete_IR_bytes_identical": True}))
        return HostLLVMEmission(
            HostLLVMArtifact.capture(original),
            HostLLVMArtifact.capture(selected),
            (HostLLVMArtifact.capture(evidence),),
        )

    transform, verify = transform or copy, verify or verify_copy
    sources = tuple(
        HostLLVMArtifact.capture(path)
        for path in sorted({Path(inspect.getsourcefile(callback)).resolve() for callback in (transform, verify)})
    )
    contract = HostLLVMStageContract(name, sources, (HostLLVMArtifact.capture(proof),))
    return HostLLVMTransformStage(contract, transform, verify)


def host_transform_chain(*stages):
    return HostLLVMTransformChain(tuple(stage.contract for stage in stages), tuple(stages))


def bounded_roundeven_source(bits=8):
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return f"""define i{bits} @quant(float %x) {{
  %lo = call float @llvm.maximum.f32(float %x, float {float(lower)})
  %hi = call float @llvm.minimum.f32(float %lo, float {float(upper)})
  %rounded = call float @llvm.roundeven.f32(float %hi)
  %result = fptosi float %rounded to i{bits}
  ret i{bits} %result
}}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.roundeven.f32(float)
"""


def inert_package(root, *, target):
    """Real package inputs for aggregation tests; executing this tool is an error.

    This admits only manifest/source readiness. Test-supplied suite results are
    synthetic observations, never compiler, numerical or hardware qualification.
    """
    directory = Path(root) / "package"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tool.py").write_text(
        'raise RuntimeError("inert aggregation package must not execute")\n', encoding="utf-8"
    )
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": target,
        "language": "python",
        "authoring": {"mode": "agent_generated_from_recipe"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "tool.py"},
        "commands": {
            name: {"argv": ["python3", "{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    (directory / "manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    return directory
