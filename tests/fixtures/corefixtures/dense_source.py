from merlin.compiler.frontend.linalg_mlir import parse_mlir_text


def contraction(shape, dtypes, *, lhs="a", rhs="b", output="r", tag="matmul"):
    m, k, n = shape
    x, y, z = dtypes
    expressions, values = ([], [])
    for source, dtype, name in (("x", x, "u"), ("y", y, "v")):
        if dtype != z:
            expressions.append(f"%{output}_{name} = arith.extsi %{output}_{source} : {dtype} to {z}")
            values.append(output + "_" + name)
        else:
            values.append(output + "_" + source)
    scalar = "\n".join(expressions)
    return f'\n %zero_{output} = arith.constant 0 : {z}\n %empty_{output} = tensor.empty() : tensor<{m}x{n}x{z}>\n %fill_{output} = linalg.fill ins(%zero_{output} : {z})\n   outs(%empty_{output} : tensor<{m}x{n}x{z}>) -> tensor<{m}x{n}x{z}>\n %{output} = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,\n   affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],\n   iterator_types = ["parallel", "parallel", "reduction"], prov.region_id = "{tag}"}}\n   ins(%{lhs}, %{rhs} : tensor<{m}x{k}x{x}>, tensor<{k}x{n}x{y}>)\n   outs(%fill_{output} : tensor<{m}x{n}x{z}>) {{\n ^bb0(%{output}_x: {x}, %{output}_y: {y}, %{output}_s: {z}):\n {scalar}\n %{output}_product = arith.muli %{values[0]}, %{values[1]} : {z}\n %{output}_sum = arith.addi %{output}_s, %{output}_product : {z}\n linalg.yield %{output}_sum : {z}\n }} -> tensor<{m}x{n}x{z}>\n'


def source(shape, dtypes):
    m, k, n = shape
    x, y, z = dtypes
    text = f"module {{func.func @forward(%a:tensor<{m}x{k}x{x}>, %b:tensor<{k}x{n}x{y}>)\n      -> tensor<{m}x{n}x{z}> {{\n      {contraction(shape, dtypes)}\n      return %r : tensor<{m}x{n}x{z}>\n    }}}}"
    module = parse_mlir_text(text)
    return (module, next((op for op in module.walk() if op.name == "linalg.generic")))
