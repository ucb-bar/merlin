"""Independent bounded loop fixtures and structural, verifier-valid mutations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopCase:
    name: str
    factor: int
    tile: int
    carries: int
    lower: int
    step: int
    count: int
    nonaligned_upper: bool = False

    @property
    def source(self):
        upper = self.lower + self.step * self.count - int(self.nonaligned_upper)
        types = ", ".join(["i32"] * self.carries)
        initial = ", ".join(f"%c{i} = %{'x' if i % 2 == 0 else 'y'}" for i in range(self.carries))
        lines = []
        for i in range(self.carries):
            lines += [f"%a{i} = arith.addi %c{i}, %y : i32", f"%b{i} = func.call @step(%a{i}, %i) : (i32,index) -> i32"]
        lines += ["func.call @observe(%c0, %y) : (i32,i32) -> ()"]
        if not (self.factor == 4 and self.carries == 3):
            lines += ["func.call @observe(%y, %c0) : (i32,i32) -> ()"]
        lines += ["scf.yield " + ", ".join(f"%b{i}" for i in range(self.carries)) + " : " + types]
        returned = ", ".join(f"%r#{i}" for i in range(self.carries))
        return f"""builtin.module {{
func.func private @step(i32, index) -> i32
func.func private @other(i32, index) -> i32
func.func private @observe(i32, i32)
func.func private @watch(i32)
func.func @main(%x:i32, %y:i32) -> ({types}) {{
%lb = arith.constant {self.lower} : index
%ub = arith.constant {upper} : index
%st = arith.constant {self.step} : index
%r:{self.carries} = scf.for %i = %lb to %ub step %st iter_args({initial}) -> ({types}) {{
{chr(10).join(lines)}
}}
func.call @watch(%r#0) : (i32) -> ()
func.return {returned} : {types}
}}
func.func @untouched(%x:i32) -> i32 {{ func.return %x : i32 }}
}}"""


POSITIVES = (
    LoopCase("factor2_carry1_short", 2, 3, 1, 3, 2, 14),
    LoopCase("factor3_carry2_nonaligned", 3, 2, 2, 5, 3, 15, True),
    LoopCase("factor4_carry3_short", 4, 4, 3, 2, 2, 20),
    LoopCase("factor2_carry4_short", 2, 2, 4, 1, 1, 10),
    LoopCase("tile16_short", 2, 16, 2, 3, 2, 34),
    LoopCase("tile7_short", 3, 7, 1, 0, 1, 27),
    LoopCase("factor4_nonunit", 4, 2, 2, 9, 4, 12),
)


def mutations(buffers):
    """Mutate actual structured IR; no production algorithm or source is edited."""
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.passes_xdsl import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    def change(stage, mutation):
        module = parse_mlir_text(buffers[stage])
        entry = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "main")
        loops = [op for op in entry.walk() if isinstance(op, scf.ForOp)]
        mutation(entry, loops)
        module.verify()
        result = list(buffers)
        result[stage] = text(module, generic=True)
        if result[stage] == buffers[stage]:
            raise ValueError("mutation changed no bytes")
        return tuple(result)

    def constant(value, delta):
        owner = value.owner
        if not isinstance(owner, arith.ConstantOp):
            raise ValueError("control fixture requires literal owner")
        old = owner.value.value.data
        owner.properties["value"] = builtin.IntegerAttr(old + delta, builtin.IndexType())

    def replace_operand(op, position, value):
        values = list(op.operands)
        values[position] = value
        op.operands = values

    result = {}
    result["source_wrong_cutoff"] = change(0, lambda entry, loops: constant(loops[0].ub, 9))
    result["middle_wrong_stride"] = change(1, lambda entry, loops: constant(loops[0].step, 1))
    result["middle_wrong_cutoff"] = change(1, lambda entry, loops: constant(loops[0].ub, -1))
    result["final_wrong_tile_stride"] = change(2, lambda entry, loops: constant(loops[0].step, 1))
    result["middle_carry_order"] = change(
        1,
        lambda entry, loops: setattr(
            loops[0], "operands", [*list(loops[0].operands)[:3], *reversed(list(loops[0].operands)[3:])]
        ),
    )
    result["middle_yield_order"] = change(
        1,
        lambda entry, loops: setattr(
            loops[0].body.block.last_op, "operands", list(reversed(loops[0].body.block.last_op.operands))
        ),
    )
    result["final_return_order"] = change(
        2,
        lambda entry, loops: setattr(
            entry.body.block.last_op, "operands", list(reversed(entry.body.block.last_op.operands))
        ),
    )

    def lane(entry, loops):
        op = next(
            op
            for op in loops[0].body.block.ops
            if isinstance(op, arith.AddiOp) and isinstance(op.result.type, builtin.IndexType)
        )
        replace_operand(op, 0, loops[0].ub)

    result["middle_lane_iv"] = change(1, lane)

    def capture(entry, loops):
        op = next(
            op
            for op in entry.walk()
            if isinstance(op, arith.AddiOp) and isinstance(op.result.type, builtin.IntegerType)
        )
        replace_operand(op, 1, entry.args[0])

    result["final_wrong_capture"] = change(2, capture)

    def body(entry, loops):
        op = next(
            op
            for op in entry.walk()
            if isinstance(op, arith.AddiOp) and isinstance(op.result.type, builtin.IntegerType)
        )
        Rewriter.replace_op(op, arith.SubiOp(op.operands[0], op.operands[1]))

    result["final_wrong_body"] = change(2, body)

    def callee(entry, loops):
        call = next(
            op for op in entry.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "step"
        )
        call.properties["callee"] = builtin.SymbolRefAttr("other")

    result["middle_wrong_call"] = change(1, callee)

    def observe_order(entry, loops):
        calls = [
            op for op in entry.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "observe"
        ]
        first, second = calls[:2]
        a, b = list(first.operands), list(second.operands)
        first.operands, second.operands = b, a

    result["final_partial_effect_order"] = change(2, observe_order)

    def observer(entry, loops):
        call = next(op for op in entry.body.block.ops if isinstance(op, func.CallOp))
        Rewriter.erase_op(call)

    result["final_missing_observer"] = change(2, observer)

    def inner_bound(entry, loops):
        replace_operand(loops[1], 1, loops[0].ub)

    result["final_wrong_inner_cutoff"] = change(2, inner_bound)

    def entry_name(entry, loops):
        entry.properties["sym_name"] = builtin.StringAttr("changed_entry")

    result["final_changed_entry"] = change(2, entry_name)
    return result


def unsupported_sources():
    base = LoopCase("unsupported", 2, 3, 1, 3, 2, 12).source
    dynamic = "\n".join(line for line in base.splitlines() if not line.startswith("%ub ="))
    dynamic = dynamic.replace("%y:i32)", "%y:i32, %dynamic:index)").replace("to %ub", "to %dynamic")
    nested = base.replace("%a0 =", "scf.for %j = %lb to %ub step %st { scf.yield }\n%a0 =", 1)
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.passes_xdsl import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    module = parse_mlir_text(base)
    entry = next(op for op in module.ops if op.name == "func.func" and op.sym_name.data == "main")
    extra = entry.clone()
    extra.properties["sym_name"] = StringAttr("main2")
    module.body.block.add_op(extra)
    return {
        "empty": LoopCase("empty", 2, 3, 1, 3, 2, 0).source,
        "inline": LoopCase("inline", 2, 3, 1, 3, 2, 2).source,
        "tail": LoopCase("tail", 2, 3, 1, 3, 2, 7).source,
        "negative_lower": LoopCase("negative", 2, 3, 1, -2, 1, 12).source,
        "dynamic": dynamic,
        "nested_general_tiling": nested,
        "two_changed_loops": text(module, generic=True),
    }
