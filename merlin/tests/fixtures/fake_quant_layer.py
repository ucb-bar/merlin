"""A fake-quantized layer and a synthetic target oracle, shared by the compute-group tests.

``module`` prints dq(x), dq(w) -> matmul -> bias -> relu -> quantize in the form a capture has;
``Oracle`` is an integer unit that takes elementwise stages only fused, at the scale granularities
it holds.
"""

from __future__ import annotations

from merlin.xdsl_dialects.lowering import compute_groups as CG

_ELEMENTWISE = (
    'linalg.generic {{indexing_maps = [{maps}], iterator_types = ["parallel", '
    '"parallel"]}} ins({ins}) outs({out} : tensor<4x16xf32>) {{\n{body}\n    }} -> '
    "tensor<4x16xf32>"
)
_ID = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"


def _generic(result: str, ins: str, maps: list[str], body: str, out: str) -> str:
    return f"    {result} = " + _ELEMENTWISE.format(maps=", ".join(maps), ins=ins, out=out, body=body)


def module(
    *,
    weight_dequantize: str = "per_channel",
    weight_axis: int = 1,
    reshape_weight: bool = False,
    relu_fan_out: bool = False,
    residual: bool = False,
) -> str:
    """dq(x), dq(w) -> matmul -> bias -> relu -> quantize, in the fake-quantized form a capture has."""
    if weight_dequantize == "per_channel":
        weight = (
            '    %wd = "quant_ext.dequantize_per_channel"(%w, %ws, %wz) <{axis = '
            f"{weight_axis} : i64, quant_min = -127 : i64, quant_max = 127 : i64}}> : "
            "(tensor<8x16xi8>, tensor<16xf32>, tensor<16xi64>) -> tensor<8x16xf32>"
        )
    else:
        weight = (
            '    %wd = "quant_ext.dequantize_per_tensor"(%w, %s, %z) <{quant_min = -127 : '
            "i64, quant_max = 127 : i64}> : (tensor<8x16xi8>, tensor<f32>, tensor<i64>) "
            "-> tensor<8x16xf32>"
        )
    weight_value = "%wd"
    if reshape_weight:
        # Merges the scaled axis into its neighbour and splits it back differently: the axis the
        # scale varies over does not survive, so its extent cannot be read.
        weight += (
            "\n    %wf = tensor.collapse_shape %wd [[0, 1]] : tensor<8x16xf32> into "
            "tensor<128xf32>\n    %wr = tensor.expand_shape %wf [[0, 1]] output_shape "
            "[8, 16] : tensor<128xf32> into tensor<8x16xf32>\n    %wt = tensor.empty() : "
            "tensor<8x16xf32>\n    %wp = linalg.transpose ins(%wr : tensor<8x16xf32>) "
            "outs(%wt : tensor<8x16xf32>) permutation = [0, 1]"
        )
        weight_value = "%wp"
    lines = [
        "builtin.module {",
        "  func.func @forward(%x: tensor<4x8xi8>, %w: tensor<8x16xi8>, %ws: tensor<16xf32>, "
        "%wz: tensor<16xi64>, %b: tensor<16xf32>, %skip: tensor<4x16xf32>) -> tensor<4x16xi8> {",
        "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
        "    %z = arith.constant dense<0> : tensor<i64>",
        '    %xd = "quant_ext.dequantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, '
        "quant_max = 127 : i64}> : (tensor<4x8xi8>, tensor<f32>, tensor<i64>) -> tensor<4x8xf32>",
        weight,
        "    %e0 = tensor.empty() : tensor<4x16xf32>",
        "    %c0 = arith.constant 0.000000e+00 : f32",
        "    %f = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x16xf32>) -> tensor<4x16xf32>",
        f"    %mm = linalg.matmul ins(%xd, {weight_value} : tensor<4x8xf32>, tensor<8x16xf32>) "
        "outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>",
        "    %e1 = tensor.empty() : tensor<4x16xf32>",
        _generic(
            "%ba",
            "%mm, %b : tensor<4x16xf32>, tensor<16xf32>",
            [_ID, _COL, _ID],
            "    ^bb0(%p: f32, %q0: f32, %o: f32):\n      %r = arith.addf %p, %q0 : f32\n      linalg.yield %r : f32",
            "%e1",
        ),
        "    %e2 = tensor.empty() : tensor<4x16xf32>",
        _generic(
            "%relu",
            "%ba : tensor<4x16xf32>",
            [_ID, _ID],
            "    ^bb0(%p: f32, %o: f32):\n      %zero = arith.constant 0.000000e+00 : f32\n"
            "      %r = arith.maximumf %p, %zero : f32\n      linalg.yield %r : f32",
            "%e2",
        ),
    ]
    quantized = "%relu"
    if relu_fan_out:
        lines += [
            "    %e3 = tensor.empty() : tensor<4x16xf32>",
            _generic(
                "%sum",
                "%relu, %skip : tensor<4x16xf32>, tensor<4x16xf32>",
                [_ID, _ID, _ID],
                "    ^bb0(%p: f32, %q0: f32, %o: f32):\n      %r = arith.addf %p, %q0 : f32"
                "\n      linalg.yield %r : f32",
                "%e3",
            ),
            "    %e4 = tensor.empty() : tensor<4x16xf32>",
            _generic(
                "%mix",
                "%sum, %relu : tensor<4x16xf32>, tensor<4x16xf32>",
                [_ID, _ID, _ID],
                "    ^bb0(%p: f32, %q0: f32, %o: f32):\n      %r = arith.addf %p, %q0 : f32"
                "\n      linalg.yield %r : f32",
                "%e4",
            ),
        ]
        quantized = "%mix"
    elif residual:
        # ONE residual add, read by nothing else: the chain stays single-use, so what stops the group
        # here is a verdict about the stage and never the graph's shape.
        lines += [
            "    %e3 = tensor.empty() : tensor<4x16xf32>",
            _generic(
                "%sum",
                "%relu, %skip : tensor<4x16xf32>, tensor<4x16xf32>",
                [_ID, _ID, _ID],
                "    ^bb0(%p: f32, %q0: f32, %o: f32):\n      %r = arith.addf %p, %q0 : f32"
                "\n      linalg.yield %r : f32",
                "%e3",
            ),
        ]
        quantized = "%sum"
    lines += [
        f'    %out = "quant_ext.quantize_per_tensor"({quantized}, %s, %z) <{{quant_min = -128 : '
        'i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<4x16xf32>, tensor<f32>, '
        "tensor<i64>) -> tensor<4x16xi8>",
        "    func.return %out : tensor<4x16xi8>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


def residual_module(
    *, lhs_scale: float = 0.5, rhs_scale: float = 0.25, out_scale: float = 1.0, zero_point: int = 0, relu: bool = True
) -> str:
    """dq(a), dq(b) -> add -> [relu] -> quantize: a residual connection whose add was quantized."""

    def scale(name: str, value: float) -> str:
        return f"    {name} = arith.constant dense<{value!r}> : tensor<f32>"

    def dequantize(result: str, operand: str, scale_name: str) -> str:
        return (
            f'    {result} = "quant_ext.dequantize_per_tensor"({operand}, {scale_name}, %z) <{{quant_min = '
            "-128 : i64, quant_max = 127 : i64}> : (tensor<4x16xi8>, tensor<f32>, tensor<i64>) -> tensor<4x16xf32>"
        )

    lines = [
        "builtin.module {",
        "  func.func @forward(%a: tensor<4x16xi8>, %b: tensor<4x16xi8>) -> tensor<4x16xi8> {",
        scale("%sa", lhs_scale),
        scale("%sb", rhs_scale),
        scale("%so", out_scale),
        f"    %z = arith.constant dense<{zero_point}> : tensor<i64>",
        dequantize("%ad", "%a", "%sa"),
        dequantize("%bd", "%b", "%sb"),
        "    %e0 = tensor.empty() : tensor<4x16xf32>",
        _generic(
            "%sum",
            "%ad, %bd : tensor<4x16xf32>, tensor<4x16xf32>",
            [_ID, _ID, _ID],
            "    ^bb0(%p: f32, %q0: f32, %o: f32):\n      %r = arith.addf %p, %q0 : f32\n      linalg.yield %r : f32",
            "%e0",
        ),
    ]
    quantized = "%sum"
    if relu:
        lines += [
            "    %e1 = tensor.empty() : tensor<4x16xf32>",
            _generic(
                "%relu",
                "%sum : tensor<4x16xf32>",
                [_ID, _ID],
                "    ^bb0(%p: f32, %o: f32):\n      %zero = arith.constant 0.000000e+00 : f32\n"
                "      %r = arith.maximumf %p, %zero : f32\n      linalg.yield %r : f32",
                "%e1",
            ),
        ]
        quantized = "%relu"
    lines += [
        f'    %out = "quant_ext.quantize_per_tensor"({quantized}, %so, %z) <{{quant_min = -128 : '
        'i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<4x16xf32>, tensor<f32>, '
        "tensor<i64>) -> tensor<4x16xi8>",
        "    func.return %out : tensor<4x16xi8>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


def _mean_lines(
    source: str, shape: tuple[int, ...], dims: tuple[int, ...], scales, count: float, tag: str
) -> list[str]:
    """dq(source) -> sum over ``dims`` -> divide by ``count`` -> quantize, defining ``%pooled{tag}``."""
    kept = [extent for axis, extent in enumerate(shape) if axis not in dims]
    full, part = "x".join(map(str, shape)), "x".join(map(str, kept))
    axes = ", ".join(f"d{i}" for i in range(len(kept)))
    identity = f"affine_map<({axes}) -> ({axes})>"
    parallel = ", ".join(['"parallel"'] * len(kept))
    return [
        f"    %si{tag} = arith.constant dense<{scales[0]!r}> : tensor<f32>",
        f"    %so{tag} = arith.constant dense<{scales[1]!r}> : tensor<f32>",
        f"    %mz{tag} = arith.constant dense<0> : tensor<i64>",
        f'    %xd{tag} = "quant_ext.dequantize_per_tensor"({source}, %si{tag}, %mz{tag}) <{{quant_min = -128 : i64, '
        f"quant_max = 127 : i64}}> : (tensor<{full}xi8>, tensor<f32>, tensor<i64>) -> tensor<{full}xf32>",
        f"    %zero{tag} = arith.constant 0.000000e+00 : f32",
        f"    %init{tag} = tensor.splat %zero{tag} : tensor<{part}xf32>",
        f"    %total{tag} = linalg.reduce ins(%xd{tag}:tensor<{full}xf32>) outs(%init{tag}:tensor<{part}xf32>) "
        f"dimensions = [{', '.join(map(str, dims))}]",
        "      (%mp: f32, %mq: f32) {",
        "        %mr = arith.addf %mp, %mq : f32",
        "        linalg.yield %mr : f32",
        "      }",
        f"    %n{tag} = arith.constant {count!r} : f32",
        f"    %ns{tag} = tensor.splat %n{tag} : tensor<{part}xf32>",
        f"    %me{tag} = tensor.empty() : tensor<{part}xf32>",
        f"    %mean{tag} = linalg.generic {{indexing_maps = [{identity}, {identity}, {identity}], iterator_types = "
        f"[{parallel}]}} ins(%total{tag}, %ns{tag} : tensor<{part}xf32>, tensor<{part}xf32>) "
        f"outs(%me{tag} : tensor<{part}xf32>) {{",
        "    ^bb0(%dp: f32, %dq: f32, %do: f32):",
        "      %dr = arith.divf %dp, %dq : f32",
        "      linalg.yield %dr : f32",
        f"    }} -> tensor<{part}xf32>",
        f'    %pooled{tag} = "quant_ext.quantize_per_tensor"(%mean{tag}, %so{tag}, %mz{tag}) <{{quant_min = -128 : '
        f'i64, quant_max = 127 : i64, output_dtype = "int8"}}> : (tensor<{part}xf32>, tensor<f32>, tensor<i64>) '
        f"-> tensor<{part}xi8>",
    ]


def mean_module(
    *,
    shape: tuple[int, ...] = (2, 8, 3, 4),
    dims: tuple[int, ...] = (2, 3),
    in_scale: float = 0.5,
    out_scale: float = 0.25,
    count: float = 12.0,
) -> str:
    """An average pool in the form a capture has it, between a dequantize and a quantize."""
    kept = "x".join(str(extent) for axis, extent in enumerate(shape) if axis not in dims)
    full = "x".join(map(str, shape))
    return "\n".join(
        [
            "builtin.module {",
            f"  func.func @forward(%x: tensor<{full}xi8>) -> tensor<{kept}xi8> {{",
            *_mean_lines("%x", shape, dims, (in_scale, out_scale), count, ""),
            f"    func.return %pooled : tensor<{kept}xi8>",
            "  }",
            "}",
        ]
    )


def residual_then_mean_module() -> str:
    """A quantized residual whose quantized result feeds an average pool: two integer regions."""
    lines = residual_module(lhs_scale=0.5, rhs_scale=0.25, out_scale=1.0).split("\n")
    body = lines[:-3]  # everything up to the residual's own return
    body[1] = body[1].replace("-> tensor<4x16xi8>", "-> tensor<4xi8>")
    return "\n".join(
        [
            *body,
            *_mean_lines("%out", (4, 16), (1,), (1.0, 0.5), 16.0, "_m"),
            "    func.return %pooled_m : tensor<4xi8>",
            "  }",
            "}",
        ]
    )


class Oracle:
    """An integer unit that takes elementwise stages only fused, at the granularities it holds.

    ``readout`` is the target's derived readout, asked only about what no capability row states:
    whether a unit's load can sum separately scaled operands.
    """

    def __init__(
        self,
        holds=("tensor",),
        families=("contraction", "elementwise_map"),
        readout=None,
        applies=("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool"),
    ):
        self.holds, self.families, self.target, self.readout = holds, families, "synthetic", readout
        #: What this synthetic target DECLARES its readout applies, in the command-buffer ABI's stage
        #: vocabulary -- the same declaration a real target makes in ``readout_epilogue_capability``.
        #: ``None`` is a target that declares nothing at all.
        self.applies = applies

    def unit_for(self, *_args) -> str:
        return "unit0"

    def absorbs(self, kind: str):
        return CG.readout_absorbs(kind, self.declared_readout(), target=self.target)

    def declared_readout(self):
        """This synthetic target's ``readout_epilogue_capability``, as the facet a real one derives."""
        from merlin.targetgen import readout_facet as RF

        declared = ({"selector": "out", "applies": list(self.applies)},) if self.applies is not None else ()
        return RF.TargetReadout((RF.ReadoutFacet(target=self.target, readouts=declared),))

    def ask(self, *, op, family, in_dtype, weight_dtype=None, rank=None, attached, granularity=None):
        if family not in self.families:
            return CG.Admission(False, "undeclared_family", f"no capability for {family}")
        if in_dtype != "int8":
            return CG.Admission(False, "input_dtype", f"{in_dtype} is not an integer format here")
        if family != "contraction" and not attached:
            return CG.Admission(False, "fused_only", "available only fused with a contraction")
        if granularity is not None and granularity not in self.holds:
            return CG.Admission(
                False, "scale_granularity", f"scales per {granularity}; the readout holds {list(self.holds)}"
            )
        return CG.Admission(True, units=("unit0",))
