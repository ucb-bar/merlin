"""Merlin-FAITHFUL vector codegen: command buffer -> MLIR (linalg/arith) -> merlin's real
MLIR→LLVM compiler (lower_model). NO hand-written C kernel.

This replaces the C-emitter (`saturn_vec_codegen.py`) with the merlin way: the vector ops are
expressed as `linalg.generic` (elementwise) and a reduction, and lowered through
`merlin.llvmlower.lower_model`. We certify on the host target (x86 via ctypes) against the
Merlin reference — proving the compiler path, not a kernel. (The same module lowers to a
riscv object; native-RVV vectorization currently covers contractions, so elementwise/reduction
lower to scalar RVV-target code — correct, just not yet vector-instruction'd; see SV findings.)
"""
from __future__ import annotations

from typing import Any

from ..commandbuffer import materialize_inputs

ID1 = "affine_map<(d0) -> (d0)>"
RED1 = "affine_map<(d0) -> (0)>"

# Transform-dialect schedule that RVV-vectorizes the vector family (1-D elementwise + reduction
# linalg.generic) — the matmul-only RVV_TRANSFORM_SCHEDULE skips these. Tile + vectorize at a
# fixed lane width; the reduction lowers via lower-vector-multi-reduction in build_rvv_pipeline.
ELEMENTWISE_RVV_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l = transform.structured.tile_using_for %g tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [8] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _shapes(cb: dict[str, Any]) -> dict[str, int]:
    sh = {n: s["shape"][0] for n, s in cb.get("tensors", {}).items()}
    for c in cb.get("commands", []):
        op, o = c["opcode"], c.get("operands", {})
        if op == "VECTOR_MAP":
            sh[o["dst"]] = sh[o["lhs"]]
        elif op == "VREDUCE":
            sh[o["dst"]] = 1
    return sh


def emit_mlir(cb: dict[str, Any]) -> tuple[str, list[str], str]:
    """Return (mlir_text, input_order, output_name) for the vector command buffer."""
    sh = _shapes(cb)
    tensors = cb.get("tensors", {})
    inputs = sorted(n for n, s in tensors.items() if s.get("role") == "input")
    outputs = [n for n, s in tensors.items() if s.get("role") == "output"]
    assert len(outputs) == 1, "single-output vector workloads (v1)"
    out = outputs[0]

    ssa = {n: f"%arg{i}" for i, n in enumerate(inputs)}
    arg_decl = ", ".join(f"{ssa[n]}: tensor<{sh[n]}xi32>" for n in inputs)
    body: list[str] = []
    tmp = 0

    def fresh() -> str:
        nonlocal tmp
        tmp += 1
        return f"%v{tmp}"

    for c in cb.get("commands", []):
        op, o, a = c["opcode"], c.get("operands", {}), c.get("attributes", {})
        if op == "VECTOR_MAP":
            n = sh[o["lhs"]]
            arith = "arith.muli" if a.get("combine") == "mul" else "arith.addi"
            relu = "relu" in a.get("activation", [])
            e, r = fresh(), fresh()
            body.append(f"  {e} = tensor.empty() : tensor<{n}xi32>")
            inner = [f"      %s = {arith} %in, %in0 : i32"]
            yld = "%s"
            if relu:
                inner += ["      %z = arith.constant 0 : i32",
                          "      %rl = arith.maxsi %s, %z : i32"]
                yld = "%rl"
            body += [
                f"  {r} = linalg.generic {{indexing_maps = [{ID1}, {ID1}, {ID1}], "
                f'iterator_types = ["parallel"]}} ins({ssa[o["lhs"]]}, {ssa[o["rhs"]]} : '
                f"tensor<{n}xi32>, tensor<{n}xi32>) outs({e} : tensor<{n}xi32>) {{",
                "    ^bb0(%in: i32, %in0: i32, %out: i32):",
                *inner,
                f"      linalg.yield {yld} : i32",
                f"  }} -> tensor<{n}xi32>",
            ]
            ssa[o["dst"]] = r
        elif op == "VREDUCE":
            n = sh[o["src"]]
            e, f, r = fresh(), fresh(), fresh()
            body += [
                f"  {e} = tensor.empty() : tensor<1xi32>",
                "  %zc = arith.constant 0 : i32",
                f"  {f} = linalg.fill ins(%zc : i32) outs({e} : tensor<1xi32>) -> tensor<1xi32>",
                f"  {r} = linalg.generic {{indexing_maps = [{ID1}, {RED1}], "
                f'iterator_types = ["reduction"]}} ins({ssa[o["src"]]} : tensor<{n}xi32>) '
                f"outs({f} : tensor<1xi32>) {{",
                "    ^bb0(%in: i32, %acc: i32):",
                "      %a = arith.addi %in, %acc : i32",
                "      linalg.yield %a : i32",
                f"  }} -> tensor<1xi32>",
            ]
            ssa[o["dst"]] = r
        else:
            raise ValueError(f"unsupported vector opcode {op!r}")

    on = sh[out]
    text = (f"builtin.module {{\n  func.func @forward({arg_decl}) -> tensor<{on}xi32> {{\n"
            + "\n".join(body)
            + f"\n    func.return {ssa[out]} : tensor<{on}xi32>\n  }}\n}}\n")
    return text, inputs, out


def run_host(cb: dict[str, Any], workdir: str | Path | None = None) -> dict[str, Any]:
    """Lower the vector cb through merlin's MLIR→LLVM compiler and run on host; gate vs reference."""
    import tempfile
    from pathlib import Path
    import numpy as np

    from ...llvmlower.lower import lower_model
    from ...llvmlower.abi import HostModel
    from ..reference import outputs_match, reference_outputs

    sh = _shapes(cb)
    text, inputs, out = emit_mlir(cb)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="vecmlir_"))
    res = lower_model(text, work, targets=("host",))
    model = HostModel.load(str(res.host_so))

    leaves = materialize_inputs(cb)
    args = []
    arrays = []
    for n in inputs:
        arr = np.array(leaves[n].data, dtype=np.int32)
        arrays.append(arr)
        args.append((arr.ctypes.data, (sh[n],)))
    y = np.zeros((sh[out],), dtype=np.int32)
    args.append((y.ctypes.data, (sh[out],)))
    model(args)

    got = {out: y.tolist()}
    ref = reference_outputs(cb)
    return {"outputs": got, "correct": outputs_match(got, ref),
            "mlir": text, "host_so": str(res.host_so), "oracle": {"kind": "merlin_mlir_host"}}


def lower_rvv(cb: dict[str, Any], workdir: str | Path | None = None) -> dict[str, Any]:
    """Lower the vector cb to a rv64gcv object via merlin's native-RVV path using the
    elementwise/reduction transform schedule; report whether real RVV vector ops were emitted."""
    import tempfile
    from pathlib import Path

    from ...llvmlower.lower import lower_model
    from ...llvmlower.custom_isa import disassemble

    text, _, _ = emit_mlir(cb)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="vecrvv_"))
    res = lower_model(text, work, targets=("riscv",), vectorize=True,
                      transform_schedule=ELEMENTWISE_RVV_SCHEDULE)
    dis = disassemble(res.riscv_obj)
    rvv = [m for m in ("vsetvli", "vsetivli", "vle32.v", "vadd.vv", "vmul.vv",
                       "vredsum", "vfadd", "vmv") if m in dis]
    return {"riscv_obj": str(res.riscv_obj), "has_rvv": bool(rvv),
            "rvv_ops": rvv, "ll_path": str(res.ll_path)}
