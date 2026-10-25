"""Standard Linalg sources for the composition axis; no device implementation."""

from __future__ import annotations

import copy


def _type(shape, dtype):
    return "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"


class _Source:
    def __init__(self):
        self.inputs, self.lines, self.types = [], [], {}

    def input(self, name, shape, dtype, role="input"):
        self.inputs.append(dict(name=name, shape=list(shape), dtype=dtype, role=role))
        self.types[name] = _type(shape, dtype)
        return name

    def constant(self, name, shape, dtype, values):
        import json

        self.types[name] = _type(shape, dtype)
        self.lines.append(f"    %{name} = arith.constant dense<{json.dumps(values)}> : {self.types[name]}")
        return name

    def generic(self, name, args, shape, dtype, body, *, maps=None, reduction=False, seed=0):
        rank = 3 if reduction else 2
        identity = "affine_map<(d0, d1) -> (d0, d1)>"
        if maps is None:
            maps = [identity] * (len(args) + 1)
        self.types[name] = result = _type(shape, dtype)
        self.lines += [f"    %{name}_empty = tensor.empty() : {result}"]
        destination = f"%{name}_empty"
        if reduction:
            self.lines += [
                f"    %{name}_zero = arith.constant {seed} : {dtype}",
                f"    %{name}_init = linalg.fill ins(%{name}_zero : {dtype}) "
                f"outs({destination} : {result}) -> {result}",
            ]
            destination = f"%{name}_init"
        iterator = ", ".join('"reduction"' if reduction and i == rank - 1 else '"parallel"' for i in range(rank))
        arg_values = ", ".join("%" + arg for arg in args)
        arg_types = ", ".join(self.types[arg] for arg in args)
        scalars = ", ".join(f"%x{i}: {self.types[arg].rsplit('x', 1)[-1][:-1]}" for i, arg in enumerate(args))
        self.lines += [
            f"    %{name} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = [{iterator}]}} "
            f"ins({arg_values} : {arg_types}) outs({destination} : {result}) "
            f'attrs = {{prov.composition_stage = "{name}"}} {{',
            f"    ^bb0({scalars}, %dest: {dtype}):",
            *["      " + line for line in body],
            "    } -> " + result,
        ]
        return name

    def matmul(self, name, lhs, rhs, m, k, n):
        return self.generic(
            name,
            [lhs, rhs],
            [m, n],
            "i32",
            [
                "%a = arith.extsi %x0 : i8 to i32",
                "%b = arith.extsi %x1 : i8 to i32",
                "%p = arith.muli %a, %b : i32",
                "%v = arith.addi %p, %dest : i32",
                "linalg.yield %v : i32",
            ],
            maps=[
                "affine_map<(d0,d1,d2)->(d0,d2)>",
                "affine_map<(d0,d1,d2)->(d2,d1)>",
                "affine_map<(d0,d1,d2)->(d0,d1)>",
            ],
            reduction=True,
        )

    def readout(self, name, value, m, n, multipliers, width, *, relu=False):
        gains = [multipliers[min(i // width, len(multipliers) - 1)] for i in range(n)]
        gain = self.constant(name + "_gains", [n], "f32", gains)
        source_dtype = self.types[value].rsplit("x", 1)[-1][:-1]
        body = [f"%f = arith.sitofp %x0 : {source_dtype} to f32"]
        if relu:
            body += ["%zero = arith.constant 0.0 : f32", "%act = arith.maximumf %f, %zero : f32"]
        body += [
            f"%p = arith.mulf %{'act' if relu else 'f'}, %x1 : f32",
            "%r = math.roundeven %p : f32",
            "%lo = arith.constant -128.0 : f32",
            "%hi = arith.constant 127.0 : f32",
            "%l = arith.maximumf %r, %lo : f32",
            "%h = arith.minimumf %l, %hi : f32",
            "%v = arith.fptosi %h : f32 to i8",
            "linalg.yield %v : i8",
        ]
        return self.generic(
            name,
            [value, gain],
            [m, n],
            "i8",
            body,
            maps=["affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(d0,d1)>"],
        )

    def pool(self, value, case, n):
        side = case["pool_in_dims"][0]
        window, stride = case["pool_window"][0], case["pool_stride"][0]
        if any(case["padding"]):
            raise ValueError("composition source requires explicitly unpadded pool geometry")
        out_side = (side - window) // stride + 1
        # One reduction dimension enumerates a complete square window.
        source_map = (
            f"affine_map<(d0,d1,d2)->(((d0 floordiv {out_side}) * {stride} + "
            f"d2 floordiv {window}) * {side} + (d0 mod {out_side}) * {stride} + d2 mod {window},d1)>"
        )
        window_name = self.constant("pool_window_shape", [window * window], "i8", [0] * (window * window))
        return self.generic(
            "pool",
            [value, window_name],
            [out_side * out_side, n],
            "i8",
            ["%v = arith.maxsi %x0, %dest : i8", "linalg.yield %v : i8"],
            maps=[source_map, "affine_map<(d0,d1,d2)->(d2)>", "affine_map<(d0,d1,d2)->(d0,d1)>"],
            reduction=True,
            seed=-128,
        )

    def render(self, outputs):
        args = ", ".join(f"%{row['name']}: {self.types[row['name']]}" for row in self.inputs)
        returns = ", ".join(self.types[value] for value in outputs)
        values = ", ".join("%" + value for value in outputs)
        return "\n".join(
            [
                "builtin.module {",
                f"  func.func @forward({args}) -> ({returns}) {{",
                *self.lines,
                f"    func.return {values} : {returns}",
                "  }",
                "}",
                "",
            ]
        )


def _sum(source, case, m, n):
    lhs = source.input("Left", [m, n], "i8")
    rhs = source.input("Right", [m, n], "i8")
    scales = case["operand_multipliers"]
    if case["operand_sum_rounding"] == "per_operand":
        lhs = source.readout("left_requant", lhs, m, n, [scales[0]], n)
        rhs = source.readout("right_requant", rhs, m, n, [scales[1]], n)
        body = [
            "%a = arith.extsi %x0 : i8 to i32",
            "%b = arith.extsi %x1 : i8 to i32",
            "%v = arith.addi %a, %b : i32",
            "linalg.yield %v : i32",
        ]
    else:
        body = [
            "%a = arith.sitofp %x0 : i8 to f32",
            "%b = arith.sitofp %x1 : i8 to f32",
            f"%s0 = arith.constant {scales[0]} : f32",
            f"%s1 = arith.constant {scales[1]} : f32",
            "%p0 = arith.mulf %a, %s0 : f32",
            "%p1 = arith.mulf %b, %s1 : f32",
            "%p = arith.addf %p0, %p1 : f32",
            "%r = math.roundeven %p : f32",
            "%v = arith.fptosi %r : f32 to i32",
            "linalg.yield %v : i32",
        ]
    return source.generic("operand_sum", [lhs, rhs], [m, n], "i32", body)


def _extension(source, value, case, m, n):
    residual = source.input("Residual", [m, n], "i8")
    coefficient = case["coefficient"]
    if float(coefficient).is_integer():
        body = [
            "%r = arith.extsi %x1 : i8 to i32",
            f"%c = arith.constant {int(coefficient)} : i32",
            "%p = arith.muli %r, %c : i32",
            "%v = arith.addi %x0, %p : i32",
        ]
    else:
        body = [
            "%r = arith.sitofp %x1 : i8 to f32",
            f"%c = arith.constant {float(coefficient)} : f32",
            "%p = arith.mulf %r, %c : f32",
            "%rounded = math.roundeven %p : f32",
            "%i = arith.fptosi %rounded : f32 to i32",
            "%v = arith.addi %x0, %i : i32",
        ]
    return source.generic("extension", [value, residual], [m, n], "i32", body + ["linalg.yield %v : i32"])


def build_composition(entry, binding):
    from .composition_corpus import assert_generic_shape, held_out_shapes

    assert_generic_shape(entry, held_out_shapes())
    if binding.cap_dtype(binding.operand_dtype) != "i8" or binding.cap_dtype(binding.accum_dtype) != "i32":
        raise ValueError("composition source requires selected signed i8/i32 formats")
    m, k, n = (entry[key] for key in ("M", "K", "N"))
    case = copy.deepcopy(entry["composition_instance"])
    scheme = entry["generalization"]["quantization_scheme"]
    case["operand_sum_rounding"] = scheme["operand_sum_rounding"]
    source = _Source()
    producer = case["producer"]
    outputs = []
    if producer == "operand_sum":
        value = _sum(source, case, m, n)
    elif producer == "movement":
        value = source.input("Image", [m, n], "i8")
    else:
        lhs = source.input("A", [m, k], "i8")
        siblings = case.get("siblings", [dict(exit=case["exit"])])
        for index, sibling in enumerate(siblings):
            weight_name = f"W{index}"
            if case.get("constant_weights") is not None:
                weight = source.constant(weight_name, [k, n], "i8", case["constant_weights"])
            else:
                weight = source.input(weight_name, [k, n], "i8", "weight")
            value = source.matmul(f"mac{index}", lhs, weight, m, k, n)
            if "coefficient" in case:
                value = _extension(source, value, case, m, n)
            if "preload" in case["accumulator_stages"]:
                bias = source.input("Bias", [m, n], "i32", "bias")
                value = source.generic(
                    "bias", [value, bias], [m, n], "i32", ["%v = arith.addi %x0, %x1 : i32", "linalg.yield %v : i32"]
                )
            if len(siblings) > 1:
                if sibling["exit"] == "requant":
                    value = _readout(source, f"readout{index}", value, case, scheme, m, n)
                outputs.append(value)
    if not outputs:
        if case["exit"] == "requant":
            value = _readout(source, "readout", value, case, scheme, m, n)
        if "pool" in case["readout_stages"]:
            value = source.pool(value, case, n)
        outputs.append(value)
    program = source.render(outputs)
    cap = dict(
        name=entry["name"],
        kind=entry["kind"],
        label=entry.get("label", "public"),
        source_role=entry["source_role"],
        source_reference=entry["source_reference"],
        interface_mlir="capsule.interface.mlir",
        linalg_mlir="capsule.interface.mlir",
        inputs=source.inputs,
        operation=dict(
            op="composition_program",
            attributes=dict(case=case, M=m, K=k, N=n, quantization_scheme=copy.deepcopy(scheme), outputs=outputs),
        ),
        numeric_policy=dict(compare="exact_int", dtype="i32"),
        expected=dict(instruction_classes=binding.classes_for(op="matmul" if producer == "contraction" else producer)),
        required_oracle_tiers=list(binding.tiers),
        vcs="optional",
        firesim="optional",
    )
    return cap, program


def _readout(source, name, value, case, scheme, m, n):
    width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else n
    return source.readout(name, value, m, n, case["multipliers"], width, relu="activation" in case["readout_stages"])
