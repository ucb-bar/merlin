"""Projection sources and an ordered numerical recipe independent of Merlin's lowering."""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.frontends.linalg_mlir import parse_mlir_text


@dataclass(frozen=True)
class Projection:
    ranges: tuple[int, ...]
    parallel: tuple[bool, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]

    @property
    def output(self):
        return tuple(i for i, keep in enumerate(self.parallel) if keep)

    def shape(self, dims):
        return tuple(self.ranges[i] for i in dims)

    def source(self, *, integer_left=False, accumulator_first=False):
        def tensor(shape, element):
            return "tensor<" + "".join(f"{size}x" for size in shape) + element + ">"

        left = tensor(self.shape(self.left), "i8" if integer_left else "f32")
        right = tensor(self.shape(self.right), "f32")
        output = tensor(self.shape(self.output), "f32")
        dimensions = ", ".join(f"d{i}" for i in range(len(self.ranges)))

        def affine(dims):
            return f"affine_map<({dimensions}) -> ({', '.join(f'd{i}' for i in dims)})>"

        maps = ", ".join(affine(dims) for dims in (self.left, self.right, self.output))
        iters = ", ".join('"parallel"' if keep else '"reduction"' for keep in self.parallel)
        conversion = "%float = arith.sitofp %a : i8 to f32" if integer_left else ""
        product = "%float" if integer_left else "%a"
        summands = "%acc, %product" if accumulator_first else "%product, %acc"
        return parse_mlir_text(f"""func.func @forward(%left: {left}, %right: {right}) -> {output} {{
          %empty = tensor.empty() : {output}
          %zero = arith.constant 0.0 : f32
          %initial = linalg.fill ins(%zero : f32) outs(%empty : {output}) -> {output}
          %result = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{iters}]}}
            ins(%left, %right : {left}, {right}) outs(%initial : {output}) {{
            ^bb0(%a: {"i8" if integer_left else "f32"}, %b: f32, %acc: f32):
              {conversion}
              %product = arith.mulf {product}, %b : f32
              %sum = arith.addf {summands} : f32
              linalg.yield %sum : f32
          }} -> {output}
          return %result : {output}
        }}""")

    def recipe(self, left, right, *, integer_left=False):
        """Keep operand scale multiplication order and each tensor operation's f32 rounding."""
        import numpy as np

        quantized, scales = [], []
        for array, dimensions, integer in [(left, self.left, integer_left), (right, self.right, False)]:
            if integer:
                quantized.append(np.asarray(array, dtype=np.int8))
                scales.append(None)
                continue
            values = np.asarray(array, dtype=np.float32)
            axes = tuple(position for position, dim in enumerate(dimensions) if not self.parallel[dim])
            amax = np.max(np.abs(values), axis=axes, keepdims=True, initial=np.float32(0))
            divided = np.asarray(amax / np.float32(127), dtype=np.float32)
            scale = np.where(
                amax == np.float32(0), np.float32(0.00001), np.maximum(divided, np.float32(0.00001))
            ).astype(np.float32)
            inverse = np.asarray(np.float64(1) / scale.astype(np.float64), dtype=np.float32)
            rounded = np.rint(np.asarray(values * inverse, dtype=np.float32))
            quantized.append(np.clip(rounded, -127, 127).astype(np.int8))
            scales.append(scale)
        accumulator = np.zeros(self.shape(self.output), dtype=np.int64)
        for point in np.ndindex(self.ranges):
            out = tuple(point[dim] for dim in self.output)
            lhs = tuple(point[dim] for dim in self.left)
            rhs = tuple(point[dim] for dim in self.right)
            accumulator[out] += int(quantized[0][lhs]) * int(quantized[1][rhs])
        assert np.all(accumulator >= -(2**31)) and np.all(accumulator < 2**31)
        result = accumulator.astype(np.float32)
        for scale, dimensions in zip(scales, (self.left, self.right), strict=True):
            if scale is None:
                continue
            following = np.empty_like(result)
            for out in np.ndindex(result.shape):
                parameter = tuple(out[self.output.index(dim)] if self.parallel[dim] else 0 for dim in dimensions)
                following[out] = np.float32(result[out] * scale[parameter])
            result = following
        return result


CASES = {
    "batch": Projection((2, 2, 3, 4), (True, True, True, False), (0, 1, 3), (0, 3, 2)),
    "attention": Projection((2, 2, 3, 4), (True, True, True, False), (0, 1, 3), (0, 2, 3)),
    "two_batches": Projection((2, 2, 2, 3, 4), (True, True, True, True, False), (0, 1, 2, 4), (0, 1, 4, 3)),
    "interleaved": Projection((2, 3, 2, 4), (True, False, True, True), (0, 2, 1), (0, 1, 3)),
    "two_reductions": Projection((2, 3, 4, 2), (True, False, True, False), (0, 1, 3), (1, 2, 3)),
    "scalar": Projection((3, 2), (False, False), (0, 1), (1, 0)),
    "matvec": Projection((2, 3), (True, False), (0, 1), (1,)),
    "broadcast": Projection((2, 3, 4), (True, True, False), (0,), (1, 2)),
    "empty_batch": Projection((0, 2, 3, 4), (True, True, True, False), (0, 1, 3), (0, 3, 2)),
    "empty_reduction": Projection((2, 2, 3, 0), (True, True, True, False), (0, 1, 3), (0, 3, 2)),
}


def changed_source_pair(operand):
    original = Projection((2, 2, 2, 3), (True, True, True, False), (0, 1, 3), (0, 3, 2))
    return original, replace(original, **{operand: (1, 0, 3) if operand == "left" else (1, 3, 2)})
