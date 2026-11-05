"""Inspect unary epilogue chains in captured linalg graphs before specialization."""

from xdsl.ir import BlockArgument, Operation

from .integer_gemm import match_integer_gemm
from .requantization import f32


def scalar(value):
    op = value.owner
    if not isinstance(op, Operation) or op.name != "tensor.splat":
        raise ValueError("not scalar splat")
    c = op.operands[0].owner
    if not isinstance(c, Operation) or c.name != "arith.constant":
        raise ValueError("not constant scalar")
    return c.properties.get("value", c.attributes.get("value")).value.data


def inspect_chain(op, *, allow_maxpool=False, pool_inspector=None):
    dims = match_integer_gemm(op)
    if dims is None:
        raise ValueError("not signed integer GEMM")
    v = op.results[0]
    scales = []
    bias = None
    relu = False
    operations = []
    layouts = []
    converted = False
    pool_proof = None
    while True:
        uses = list(v.uses)
        if len(uses) != 1:
            raise ValueError("fanout before quantization")
        u = uses[0].operation
        name = u.name
        if name == "builtin.unregistered":
            name = u.attributes["op_name__"].data
        operations.append(u)
        if name == "quant_ext.quantize_per_tensor":
            if u.operands[0] is not v or str(u.results[0].type.get_element_type()) != "i8":
                raise ValueError("noncanonical quantizer")
            if bias is None or len(scales) != 2:
                raise ValueError("expected two dequant scales and one bias")
            if scalar(u.operands[2]) != 0:
                raise ValueError("nonzero output zero point")
            if u.properties["quant_min"].value.data != -128 or u.properties["quant_max"].value.data != 127:
                raise ValueError("non-int8 quant limits")
            return dict(
                dimensions=dims,
                scales=scales,
                bias=bias,
                reciprocal=1.0 / f32(scalar(u.operands[1])),
                relu=relu,
                operations=operations,
                layouts=layouts,
                quantize=u,
                pool=pool_proof,
            )
        if name == "tensor.insert_slice" and allow_maxpool:
            if not relu or bias is None or len(scales) != 2 or pool_proof is not None:
                raise ValueError("pool needs proven ReLU epilogue")
            if pool_inspector is None:
                raise ValueError("explicit pool source inspector required")
            pool, pool_proof = pool_inspector(u, v)
            operations.append(pool)
            v = pool.results[0]
            continue
        if name in ("tensor.collapse_shape", "tensor.expand_shape", "linalg.transpose"):
            if name.startswith("tensor.") and len(u.operands) != 1:
                raise ValueError("dynamic layout is not supported")
            layouts.append(u)
            v = u.results[0]
            continue
        if name != "linalg.generic":
            raise ValueError("boundary " + name)
        from xdsl.ir.affine import AffineMap

        rank = len(v.type.get_shape())
        slot = list(u.operands[:-1]).index(v)
        maps = u.properties["indexing_maps"].data
        if (
            maps[slot].data != AffineMap.identity(rank)
            or maps[-1].data != AffineMap.identity(rank)
            or any(x.data.value != "parallel" for x in u.properties["iterator_types"].data)
        ):
            raise ValueError("nonidentity pointwise path")
        for scalar_op in u.regions[0].block.ops:
            fm = scalar_op.properties.get("fastmath")
            if fm is not None and str(fm) != "#arith.fastmath<none>":
                raise ValueError("fastmath not supported")
        ops = [x for x in u.regions[0].block.ops if x.name != "arith.constant"]
        names = [x.name for x in ops]
        if len(ops) != 2 or list(ops[-1].operands) != list(ops[0].results):
            raise ValueError("noncanonical scalar yield")
        blockarg = u.regions[0].block.args[slot]
        if blockarg not in ops[0].operands:
            raise ValueError("scalar path disconnected")
        if names == ["arith.sitofp", "linalg.yield"] and not converted and not scales and bias is None:
            if list(ops[0].operands) != [blockarg] or str(ops[0].results[0].type) != "f32":
                raise ValueError("noncanonical conversion")
            converted = True
        elif names == ["arith.mulf", "linalg.yield"] and bias is None and converted:
            if len(u.operands) != 3 or set(ops[0].operands) != set(u.regions[0].block.args[:2]):
                raise ValueError("scalar multiplication operand mismatch")
            other = next(x for x in u.operands[:-1] if x is not v)
            scales.append(f32(scalar(other)))
        elif names == ["arith.addf", "linalg.yield"] and bias is None and len(scales) == 2:
            if len(u.operands) != 3 or set(ops[0].operands) != set(u.regions[0].block.args[:2]):
                raise ValueError("scalar bias operand mismatch")
            other = next(x for x in u.operands[:-1] if x is not v)
            while isinstance(other.owner, Operation) and other.owner.name in (
                "tensor.collapse_shape",
                "tensor.expand_shape",
            ):
                other = other.owner.operands[0]
            if (
                not isinstance(other, BlockArgument)
                or tuple(other.type.get_shape()) != (dims.n,)
                or str(other.type.get_element_type()) != "f32"
            ):
                raise ValueError("not captured channel bias")
            bias = other
        elif names == ["arith.maximumf", "linalg.yield"] and bias is not None:
            maximum = ops[0]
            other = next(x for x in maximum.operands if not isinstance(x, BlockArgument))
            if other.owner.name != "arith.constant" or other.owner.properties["value"].value.data != 0:
                raise ValueError("nonzero activation clamp")
            relu = True
        elif "arith.addf" in names and bias is not None:
            raise ValueError("residual addition before quantization")
        else:
            raise ValueError("non-unary epilogue " + ",".join(names))
        v = u.results[0]
