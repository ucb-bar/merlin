from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _Capture:
    """A quantized model spelled the way a capture lowers it on the host (generic MLIR form).

    A convolution is strided ``extract_slice``s of the zero-padded NCHW image, one per tap, each
    transposed to NHWC and flattened to ``[positions, channels]``, concatenated into the patch matrix
    an integer matmul reads against the weight's ``[Co, C, kh, kw] -> [kh*kw*C, Co]`` view, then the
    NCHW readout: two scales, a per-channel bias, relu and a quantize. A residual connection is two
    dequantizes, a sum, relu and a quantize. Every tensor between layers is NCHW int8."""

    def __init__(self):
        self.lines: list[str] = []
        self.count = 0
        self.args: list[tuple[str, str]] = []

    def fresh(self):
        self.count += 1
        return f"%v{self.count}"

    @staticmethod
    def ty(shape, dt):
        return f"tensor<{''.join((f'{d}x' for d in shape))}{dt}>"

    @staticmethod
    def arr(values):
        return f"array<i64: {', '.join(map(str, values))}>"

    @staticmethod
    def reassoc(groups):
        return "[" + ", ".join(("[" + ", ".join((f"{i} : i64" for i in g)) + "]" for g in groups)) + "]"

    def arg(self, name, shape, dt):
        self.args.append((f"%{name}", self.ty(shape, dt)))
        return f"%{name}"

    def op(self, text):
        name = self.fresh()
        self.lines.append(f"    {name} = {text}")
        return name

    def empty(self, shape, dt):
        return self.op(f'"tensor.empty"() : () -> {self.ty(shape, dt)}')

    def transpose(self, src, shape, perm, dt):
        out = [shape[i] for i in perm]
        init = self.empty(out, dt)
        region = f'({{\n    ^bb0(%a: {dt}, %b: {dt}):\n      "linalg.yield"(%a) : ({dt}) -> ()\n    }})'
        return (
            self.op(
                f'"linalg.transpose"({src}, {init}) <{{permutation = {self.arr(perm)}}}> {region} : ({self.ty(shape, dt)}, {self.ty(out, dt)}) -> {self.ty(out, dt)}'
            ),
            out,
        )

    def collapse(self, src, shape, groups, dt):
        out = [int(np.prod([shape[i] for i in g])) for g in groups]
        return (
            self.op(
                f'"tensor.collapse_shape"({src}) <{{reassociation = {self.reassoc(groups)}}}> : ({self.ty(shape, dt)}) -> {self.ty(out, dt)}'
            ),
            out,
        )

    def expand(self, src, shape, groups, out, dt):
        attrs = f"reassociation = {self.reassoc(groups)}, static_output_shape = {self.arr(out)}"
        return (
            self.op(f'"tensor.expand_shape"({src}) <{{{attrs}}}> : ({self.ty(shape, dt)}) -> {self.ty(out, dt)}'),
            out,
        )

    def extract(self, src, shape, offsets, sizes, strides, dt):
        attrs = f"static_offsets = {self.arr(offsets)}, static_sizes = {self.arr(sizes)}, static_strides = {self.arr(strides)}, operandSegmentSizes = array<i32: 1, 0, 0, 0>"
        return self.op(f'"tensor.extract_slice"({src}) <{{{attrs}}}> : ({self.ty(shape, dt)}) -> {self.ty(sizes, dt)}')

    def splat(self, value, dt, shape):
        constant = self.op(f'"arith.constant"() <{{value = {value} : {dt}}}> : () -> {dt}')
        return self.op(f'"tensor.splat"({constant}) : ({dt}) -> {self.ty(shape, dt)}')

    def elementwise(self, inputs, types, shape, body, maps=None):
        init = self.empty(shape, "f32")
        dims = ", ".join((f"d{i}" for i in range(len(shape))))
        identity = f"affine_map<({dims}) -> ({dims})>"
        maps = [*(maps or [identity] * len(inputs)), identity]
        kinds = ", ".join(["#linalg.iterator_type<parallel>"] * len(shape))
        args = ", ".join((f"%a{i}: {t.split('x')[-1][:-1]}" for i, t in enumerate(types))) + ", %o: f32"
        attrs = f"indexing_maps = [{', '.join(maps)}], iterator_types = [{kinds}], operandSegmentSizes = array<i32: {len(inputs)}, 1>"
        signature = f"({', '.join([*types, self.ty(shape, 'f32')])}) -> {self.ty(shape, 'f32')}"
        return self.op(
            f""""linalg.generic"({", ".join(inputs)}, {init}) <{{{attrs}}}> ({{\n    ^bb0({args}):\n{body}\n      "linalg.yield"(%r) : (f32) -> ()\n    }}) : {signature}"""
        )

    def quantize(self, value, shape, scale):
        q_scale, q_zero = (self.splat(f"{scale:e}", "f32", []), self.splat(0, "i64", []))
        return self.op(
            f""""quant_ext.quantize_per_tensor"({value}, {q_scale}, {q_zero}) <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> : ({self.ty(shape, "f32")}, tensor<f32>, tensor<i64>) -> {self.ty(shape, "i8")}"""
        )

    def dequantize(self, value, shape, scale):
        q_scale, q_zero = (self.splat(f"{scale:e}", "f32", []), self.splat(0, "i64", []))
        return self.op(
            f""""quant_ext.dequantize_per_tensor"({value}, {q_scale}, {q_zero}) <{{quant_min = -128 : i64, quant_max = 127 : i64}}> : ({self.ty(shape, "i8")}, tensor<f32>, tensor<i64>) -> {self.ty(shape, "f32")}"""
        )

    def relu(self, value, shape):
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        return self.elementwise(
            [value],
            [self.ty(shape, "f32")],
            shape,
            f'      %z = "arith.constant"() <{{value = 0.000000e+00 : f32}}> : () -> f32\n      %r = "arith.maximumf"(%a0, %z) {fastmath} : (f32, f32) -> f32',
        )

    def maxpool(self, value, shape, size, stride, pad):
        """A framework max-pool of an NCHW float tensor: a ``-inf`` pad, then a NaN-propagating window max."""
        n, c, h, w = shape
        hp, wp = (h + 2 * pad, w + 2 * pad)
        if pad:
            fill = self.splat("0xff800000", "f32", [n, c, hp, wp])
            attrs = f"static_offsets = {self.arr([0, 0, pad, pad])}, static_sizes = {self.arr(shape)}, static_strides = {self.arr([1, 1, 1, 1])}, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>"
            value = self.op(
                f""""tensor.insert_slice"({value}, {fill}) <{{{attrs}}}> : ({self.ty(shape, "f32")}, {self.ty([n, c, hp, wp], "f32")}) -> {self.ty([n, c, hp, wp], "f32")}"""
            )
        ho, wo = ((hp - size) // stride + 1, (wp - size) // stride + 1)
        out = [n, c, ho, wo]
        init = self.splat("0xff800000", "f32", out)
        window = self.empty([size, size], "f32")
        maps = f"affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * {stride} + d4, d3 * {stride} + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>"
        kinds = ", ".join(["#linalg.iterator_type<parallel>"] * 4 + ["#linalg.iterator_type<reduction>"] * 2)
        fm = "#arith.fastmath<none>"
        body = f'    ^bb0(%a: f32, %b: f32, %o: f32):\n      %u = "arith.cmpf"(%a, %a) <{{predicate = 14 : i64, fastmath = {fm}}}> : (f32, f32) -> i1\n      %g = "arith.cmpf"(%a, %o) <{{predicate = 2 : i64, fastmath = {fm}}}> : (f32, f32) -> i1\n      %e = "arith.ori"(%u, %g) : (i1, i1) -> i1\n      %r = "arith.select"(%e, %a, %o) : (i1, f32, f32) -> f32\n      "linalg.yield"(%r) : (f32) -> ()\n'
        padded = [n, c, hp, wp]
        return (
            self.op(
                f""""linalg.generic"({value}, {window}, {init}) <{{indexing_maps = [{maps}], iterator_types = [{kinds}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n{body}    }}) : ({self.ty(padded, "f32")}, {self.ty([size, size], "f32")}, {self.ty(out, "f32")}) -> {self.ty(out, "f32")}"""
            ),
            out,
        )

    def linear(self, x, m, k, n, weight, bias, s1, s2, so):
        """``x[m, k] @ weight[n, k]^T`` + bias, relu, quantize -- the weight transposed on the host."""
        wt, _ = self.transpose(weight, [n, k], [1, 0], "i8")
        init = self.splat(0, "i32", [m, n])
        maps = "affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>"
        kinds = "#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"
        overflow = "<{overflowFlags = #arith.overflow<none>}>"
        acc = self.op(
            f""""linalg.generic"({x}, {wt}, {init}) <{{indexing_maps = [{maps}], iterator_types = [{kinds}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n    ^bb0(%a: i8, %b: i8, %o: i32):\n      %ea = "arith.extsi"(%a) : (i8) -> i32\n      %eb = "arith.extsi"(%b) : (i8) -> i32\n      %m = "arith.muli"(%ea, %eb) {overflow} : (i32, i32) -> i32\n      %s = "arith.addi"(%o, %m) {overflow} : (i32, i32) -> i32\n      "linalg.yield"(%s) : (i32) -> ()\n    }}) : ({self.ty([m, k], "i8")}, {self.ty([k, n], "i8")}, {self.ty([m, n], "i32")}) -> {self.ty([m, n], "i32")}"""
        )
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        value = self.elementwise(
            [acc], [self.ty([m, n], "i32")], [m, n], '      %r = "arith.sitofp"(%a0) : (i32) -> f32'
        )
        for scale in (s1, s2):
            factor = self.splat(f"{scale:e}", "f32", [m, n])
            value = self.elementwise(
                [value, factor],
                [self.ty([m, n], "f32")] * 2,
                [m, n],
                f'      %r = "arith.mulf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            )
        value = self.elementwise(
            [value, bias],
            [self.ty([m, n], "f32"), self.ty([n], "f32")],
            [m, n],
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            maps=["affine_map<(d0, d1) -> (d0, d1)>", "affine_map<(d0, d1) -> (d1)>"],
        )
        return self.quantize(self.relu(value, [m, n]), [m, n], so)

    def conv(self, image, c, h, w, co, k, stride, pad, weight, bias, s1, s2, so, pool=None):
        """A conv + bias + relu (+ max-pool) + quantize layer of the NCHW int8 ``image``;
        returns ``(value, rows, cols)``. ``pool`` is ``(size, stride, pad)``."""
        hp, wp = (h + 2 * pad, w + 2 * pad)
        ho, wo = ((hp - k) // stride + 1, (wp - k) // stride + 1)
        p, kk = (ho * wo, k * k * c)
        if pad:
            zero = self.splat(0, "i8", [1, c, hp, wp])
            attrs = f"static_offsets = {self.arr([0, 0, pad, pad])}, static_sizes = {self.arr([1, c, h, w])}, static_strides = {self.arr([1, 1, 1, 1])}, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>"
            image = self.op(
                f""""tensor.insert_slice"({image}, {zero}) <{{{attrs}}}> : ({self.ty([1, c, h, w], "i8")}, {self.ty([1, c, hp, wp], "i8")}) -> {self.ty([1, c, hp, wp], "i8")}"""
            )
        pieces = []
        for dh in range(k):
            for dw in range(k):
                rows = self.extract(image, [1, c, hp, wp], [0, 0, dh, 0], [1, c, ho, wp], [1, 1, stride, 1], "i8")
                tap = self.extract(rows, [1, c, ho, wp], [0, 0, 0, dw], [1, c, ho, wo], [1, 1, 1, stride], "i8")
                nhwc, shape = self.transpose(tap, [1, c, ho, wo], [0, 2, 3, 1], "i8")
                flat, shape = self.collapse(nhwc, shape, [[0, 1, 2, 3]], "i8")
                pieces.append(self.expand(flat, shape, [[0, 1]], [p, c], "i8")[0])
        patches = pieces[0]
        if len(pieces) > 1:
            inputs = ", ".join([self.ty([p, c], "i8")] * len(pieces))
            patches = self.op(
                f""""tensor.concat"({", ".join(pieces)}) <{{dim = 1 : i64}}> : ({inputs}) -> {self.ty([p, kk], "i8")}"""
            )
        wv, shape = self.transpose(weight, [co, c, k, k], [0, 2, 3, 1], "i8")
        wv, shape = self.collapse(wv, shape, [[0, 1, 2, 3]], "i8")
        wv, shape = self.expand(wv, shape, [[0, 1]], [co, kk], "i8")
        wv, _ = self.transpose(wv, [co, kk], [1, 0], "i8")
        init = self.splat(0, "i32", [p, co])
        maps = "affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>"
        kinds = "#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"
        overflow = "<{overflowFlags = #arith.overflow<none>}>"
        acc = self.op(
            f""""linalg.generic"({patches}, {wv}, {init}) <{{indexing_maps = [{maps}], iterator_types = [{kinds}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n    ^bb0(%a: i8, %b: i8, %o: i32):\n      %ea = "arith.extsi"(%a) : (i8) -> i32\n      %eb = "arith.extsi"(%b) : (i8) -> i32\n      %m = "arith.muli"(%ea, %eb) {overflow} : (i32, i32) -> i32\n      %s = "arith.addi"(%o, %m) {overflow} : (i32, i32) -> i32\n      "linalg.yield"(%s) : (i32) -> ()\n    }}) : ({self.ty([p, kk], "i8")}, {self.ty([kk, co], "i8")}, {self.ty([p, co], "i32")}) -> {self.ty([p, co], "i32")}"""
        )
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        value = self.elementwise(
            [acc], [self.ty([p, co], "i32")], [p, co], '      %r = "arith.sitofp"(%a0) : (i32) -> f32'
        )
        for scale in (s1, s2):
            factor = self.splat(f"{scale:e}", "f32", [p, co])
            value = self.elementwise(
                [value, factor],
                [self.ty([p, co], "f32")] * 2,
                [p, co],
                f'      %r = "arith.mulf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            )
        value, shape = self.collapse(value, [p, co], [[0, 1]], "f32")
        value, shape = self.expand(value, shape, [[0, 1, 2, 3]], [1, ho, wo, co], "f32")
        value, out = self.transpose(value, shape, [0, 3, 1, 2], "f32")
        bias_view, _ = self.expand(bias, [co], [[0, 1, 2, 3]], [1, co, 1, 1], "f32")
        value = self.elementwise(
            [value, bias_view],
            [self.ty(out, "f32"), self.ty([1, co, 1, 1], "f32")],
            out,
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            maps=["affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>", "affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>"],
        )
        value = self.relu(value, out)
        if pool is not None:
            value, out = self.maxpool(value, out, *pool)
        return (self.quantize(value, out, so), out[2], out[3])

    def residual(self, a, b, shape, sa, sb, so):
        """``quantize(relu(dequantize(a) + dequantize(b)))`` over two NCHW int8 tensors."""
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        total = self.elementwise(
            [self.dequantize(a, shape, sa), self.dequantize(b, shape, sb)],
            [self.ty(shape, "f32")] * 2,
            shape,
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
        )
        return self.quantize(self.relu(total, shape), shape, so)

    def module(self, result, shape) -> str:
        types = ", ".join((t for _n, t in self.args))
        params = ", ".join((f"{n}: {t}" for n, t in self.args))
        return (
            f""""builtin.module"() ({{\n  "func.func"() <{{sym_name = "forward", function_type = ({types}) -> {self.ty(shape, "i8")}}}> ({{\n  ^bb0({params}):\n"""
            + "\n".join(self.lines)
            + f"""\n    "func.return"({result}) : ({self.ty(shape, "i8")}) -> ()\n  }}) : () -> ()\n}}) : () -> ()\n"""
        )


def _write_capture(directory: Path, text: str, stored: dict) -> None:
    """``model.mlir`` plus the weights manifest and safetensors the route's prepack reads."""
    import struct as _struct

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.mlir").write_text(text, encoding="utf-8")
    header, payload = ({}, b"")
    for name, array in stored.items():
        raw = array.tobytes()
        offsets = [len(payload), len(payload) + len(raw)]
        spelled = "I8" if array.dtype == np.int8 else "F32"
        header[name] = {"dtype": spelled, "shape": list(array.shape), "data_offsets": offsets}
        payload += raw
    blob = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(_struct.pack("<Q", len(blob)) + blob + payload)
    manifest = {"0": {"kind": "input"}, **{str(i + 1): {"kind": "weight", "weight": n} for i, n in enumerate(stored)}}
    (directory / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
