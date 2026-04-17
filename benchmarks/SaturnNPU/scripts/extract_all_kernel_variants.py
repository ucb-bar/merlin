#!/usr/bin/env python3
"""Extract ALL kernel shape variants + fused patterns from global-opt MLIR.

Uses MLIR Python bindings to walk the IR and write standalone .mlir files
for every shape variant of every kernel type, with fully resolved affine maps.

Also identifies fusible composite patterns and writes them as separate kernels.
"""

from __future__ import annotations

from pathlib import Path

import iree.compiler.ir as ir


def classify(child):
    if child.name != "linalg.generic":
        return child.name
    body_ops = set()
    it_str = str(child.attributes["iterator_types"]) if "iterator_types" in child.attributes else ""
    has_red = "reduction" in it_str
    for region in child.regions:
        for b in region:
            for body_op in b:
                body_ops.add(body_op.name)
    body = body_ops - {"linalg.yield"}
    if "arith.extf" in body and "arith.mulf" in body and "arith.addf" in body and has_red:
        return "quantized_matmul_fp8"
    if "math.rsqrt" in body:
        return "rms_norm"
    if "math.tanh" in body:
        return "gelu_tanh"
    if "math.exp" in body and "arith.negf" in body:
        return "silu"
    if "arith.fptosi" in body:
        return "int8_quantize"
    if body <= {"arith.extf"}:
        return "type_conversion"
    if body <= {"arith.truncf"}:
        return "type_conversion"
    if body == {"arith.addf"} and not has_red:
        return "elementwise_add"
    if body == {"arith.mulf"} and not has_red:
        return "elementwise_mul"
    if body == {"arith.divf"} and not has_red:
        return "elementwise_div"
    if body == {"arith.subf"} and not has_red:
        return "elementwise_sub"
    if body == {"arith.addf"} and has_red:
        return "reduction_sum"
    if "arith.addf" in body and "arith.truncf" in body:
        return "bias_add_cast"
    if "math.fpowi" in body:
        return "rope_frequency"
    return "other"


SKIP = {
    "tensor.empty",
    "tensor.expand_shape",
    "tensor.collapse_shape",
    "arith.constant",
    "tensor.extract_slice",
    "tensor.insert_slice",
    "util.global.load",
    "linalg.fill",
    "linalg.transpose",
}


def main():
    gopt_path = Path(
        "build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir"
    )
    kernels_dir = Path("benchmarks/SaturnNPU/kernels")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    print(f"Parsing {gopt_path.name}...")
    with open(gopt_path) as f:
        module = ir.Module.parse(f.read(), context=ctx)

    main_func = None
    for op in module.body:
        if "sym_name" in op.attributes:
            sym = str(op.attributes["sym_name"]).strip('"')
            if "main" in sym and "async" in sym:
                main_func = op
                break

    block = list(main_func.regions[0])[0]

    # Build op sequence
    ops = []
    for child in block:
        cls = classify(child)
        if child.name in SKIP:
            continue
        result_types = [str(r.type) for r in child.results]
        operand_types = [str(o.type) for o in child.operands]
        result = result_types[0] if result_types else ""
        ops.append(
            {
                "cls": cls,
                "result": result,
                "operands": operand_types,
                "child": child,
            }
        )

    # Collect unique shape variants per kernel type
    variants: dict[str, dict[str, dict]] = {}
    for o in ops:
        cls = o["cls"]
        if not cls or cls == "other":
            continue
        result = o["result"]
        if cls not in variants:
            variants[cls] = {}
        if result not in variants[cls]:
            variants[cls][result] = {
                "result": result,
                "operands": o["operands"],
                "mlir": str(o["child"]),
            }

    # Write each kernel type
    print("\n=== Kernel variants ===")
    for cls in sorted(variants.keys()):
        vs = variants[cls]
        d = kernels_dir / cls
        d.mkdir(parents=True, exist_ok=True)

        for i, (result_key, v) in enumerate(sorted(vs.items())):
            shape_label = (result_key.replace("tensor<", "").replace(">", "").replace("x", "_"))[:50]
            fname = f"variant_{i}_{shape_label}.mlir"
            content = (
                f"// Kernel: {cls}, variant {i}\n"
                f"// Result: {v['result']}\n"
                f"// Operands: {v['operands'][:3]}\n"
                f"// Affine maps fully resolved\n\n"
                f"{v['mlir']}\n"
            )
            (d / fname).write_text(content)

        # Write README
        readme_lines = [f"# `{cls}`", "", f"**{len(vs)} shape variants** in the model.", ""]
        readme_lines.append("| # | Result Type | Operand Types |")
        readme_lines.append("| --- | --- | --- |")
        for i, (rk, v) in enumerate(sorted(vs.items())):
            rt = v["result"][:60]
            ot = ", ".join(v["operands"][:2])[:60]
            readme_lines.append(f"| {i} | `{rt}` | `{ot}` |")
        readme_lines.append("")
        readme_lines.append("Each `.mlir` file has the full op with resolved affine maps.")
        (d / "README.md").write_text("\n".join(readme_lines))

        print(f"  {cls}: {len(vs)} variants")

    # Find and write fused patterns
    fused_defs = [
        ("fused_matmul_bias", "quantized_matmul_fp8", "elementwise_add"),
        ("fused_quant_matmul", "int8_quantize", "linalg.matmul"),
        ("fused_silu_gate", "silu", "elementwise_mul"),
        ("fused_norm_scale", "rms_norm", "elementwise_mul"),
    ]

    print("\n=== Fused composite kernels ===")
    for fuse_name, a_cls, b_cls in fused_defs:
        count = 0
        example = None
        for i in range(len(ops) - 1):
            if ops[i]["cls"] == a_cls and ops[i + 1]["cls"] == b_cls:
                count += 1
                if example is None:
                    example = (ops[i], ops[i + 1])
        if count == 0:
            continue

        d = kernels_dir / fuse_name
        d.mkdir(parents=True, exist_ok=True)

        a_data, b_data = example
        (d / "op_a.mlir").write_text(
            f"// {fuse_name} part A ({a_cls})\n" f"// Output feeds into op B\n\n" f"{str(a_data['child'])}\n"
        )
        (d / "op_b.mlir").write_text(
            f"// {fuse_name} part B ({b_cls})\n" f"// Consumes output of op A\n\n" f"{str(b_data['child'])}\n"
        )
        (d / "README.md").write_text(
            f"# `{fuse_name}` (Fused Composite Kernel)\n\n"
            f"**Pattern**: `{a_cls}` -> `{b_cls}`\n\n"
            f"**{count}x** in the model.\n\n"
            f"These ops always appear consecutively. "
            f"Fusing them eliminates intermediate materialization.\n\n"
            f"- `op_a.mlir` — the {a_cls} op\n"
            f"- `op_b.mlir` — the {b_cls} op that consumes A's output\n"
        )
        print(f"  {fuse_name}: {count}x ({a_cls} -> {b_cls})")

    # Top-level index
    all_dirs = sorted(d.name for d in kernels_dir.iterdir() if d.is_dir() and (d / "README.md").exists())
    index_lines = ["# SmolVLA Kernel Catalog", ""]
    index_lines.append("| Kernel | Variants | Type |")
    index_lines.append("| --- | --- | --- |")
    for dirname in all_dirs:
        mlir_count = len(list((kernels_dir / dirname).glob("*.mlir")))
        fused = "fused" if dirname.startswith("fused_") else "atomic"
        index_lines.append(f"| [`{dirname}`]({dirname}/) | {mlir_count} | {fused} |")
    (kernels_dir / "README.md").write_text("\n".join(index_lines))

    total_variants = sum(len(v) for v in variants.values())
    fused_count = sum(
        1
        for fn, ac, bc in fused_defs
        if any(ops[i]["cls"] == ac and ops[i + 1]["cls"] == bc for i in range(len(ops) - 1))
    )
    print(f"\nTotal: {total_variants} variants across {len(variants)} types + {fused_count} fused patterns")


if __name__ == "__main__":
    main()
