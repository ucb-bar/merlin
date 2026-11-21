import numpy as np


def contraction_source(a_shape, b_shape, c_shape, *, transpose=False, alias=False, fastmath=False, reversed_add=False):
    rank = len(c_shape)
    dims = [f"d{i}" for i in range(rank + 1)]
    batch = dims[:-3]
    i, j, k = dims[-3:]
    maps = [batch + [i, k], batch + ([j, k] if transpose else [k, j]), batch + [i, j]]
    amap = ",".join((f"affine_map<({','.join(dims)})->({','.join(m)})>" for m in maps))
    iters = ",".join([*['"parallel"'] * rank, '"reduction"'])

    def typ(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    ta, tb, tc = map(typ, [a_shape, b_shape, c_shape])
    arguments = f"%a: {ta}, %b: {tb}" + ("" if alias else f", %c: {tc}")
    initial = "%a" if alias else "%c"
    flags = " fastmath<contract>" if fastmath else ""
    add_args = "%z,%p" if reversed_add else "%p,%z"
    return f"module {{ func.func @forward({arguments}) -> {tc} attributes {{llvm.emit_c_interface}} {{\n      %r = linalg.generic {{indexing_maps=[{amap}], iterator_types=[{iters}]}}\n      ins(%a,%b:{ta},{tb}) outs({initial}:{tc}) {{\n      ^bb0(%x:f32,%y:f32,%z:f32):\n        %p = arith.mulf %x,%y{flags}:f32\n        %s = arith.addf {add_args}:f32\n        linalg.yield %s:f32\n      }} -> {tc}\n      return %r:{tc}\n    }} }}"


def ordered_contraction(a, b, c):
    result = c.copy()
    for k in range(a.shape[-1]):
        product = a[..., :, k, None] * b[..., k, None, :]
        result = np.float32(product + result)
    return result
