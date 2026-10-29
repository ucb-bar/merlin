"""Independent bounded rank-two epochs and complete-IR adversaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BufferCase:
    name: str
    rows: int
    columns: int
    tile: int
    epochs: int = 2
    alignment: int = 16

    @property
    def source(self):
        kind = f"memref<{self.rows}x{self.columns}xi32>"
        body = []
        for epoch in range(self.epochs):
            seed = epoch % 2
            body.append(f"""
  %buffer{epoch} = memref.alloc() {{alignment = {self.alignment} : i64}} : {kind}
  %view{epoch} = "memref.cast"(%buffer{epoch}) : ({kind}) -> {kind}
  %again{epoch} = "memref.cast"(%view{epoch}) : ({kind}) -> {kind}
  %rows{epoch} = memref.dim %again{epoch}, %zero : {kind}
  %columns{epoch} = memref.dim %again{epoch}, %one : {kind}
  scf.for %iv{epoch}_0 = %zero to %bound0 step %one {{
   scf.for %iv{epoch}_1 = %zero to %bound1 step %one {{
    memref.store %seed{seed}, %again{epoch}[%iv{epoch}_0, %iv{epoch}_1] : {kind}
    scf.yield
   }}
   scf.yield
  }}
  %first{epoch} = memref.load %again{epoch}[%zero, %zero] : {kind}
  %updated{epoch} = arith.addi %first{epoch}, %seed{seed} : i32
  memref.store %updated{epoch}, %again{epoch}[%zero, %zero] : {kind}
  %value{epoch} = memref.load %again{epoch}[%zero, %zero] : {kind}
  %dim{epoch} = memref.dim %again{epoch}, %zero : {kind}
  %observed{epoch} = func.call @touch(%value{epoch}, %caller) : (i32, memref<1xi32>) -> i32
  memref.dealloc %buffer{epoch} : {kind}
""")
        result = "%observed0"
        for epoch in range(1, self.epochs):
            body.append(f"%result{epoch} = arith.addi {result}, %observed{epoch} : i32")
            result = f"%result{epoch}"
        return f"""builtin.module {{
 func.func @kernel(%seed0: i32, %seed1: i32, %caller: memref<1xi32>) -> (i32, index) {{
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %bound0 = arith.constant {self.rows} : index
  %bound1 = arith.constant {self.columns} : index
  {chr(10).join(body)}
  func.return {result}, %dim{self.epochs - 1} : i32, index
 }}
 func.func @touch(%value: i32, %caller: memref<1xi32>) -> i32 {{
  %zero = arith.constant 0 : index
  %old = memref.load %caller[%zero] : memref<1xi32>
  %new = arith.addi %old, %value : i32
  memref.store %new, %caller[%zero] : memref<1xi32>
  func.return %new : i32
 }}
 func.func @untouched(%input: i32) -> i32 {{
  %one = arith.constant 1 : i32
  %result = arith.addi %input, %one : i32
  func.return %result : i32
 }}
}}"""


VARIANTS = (
    BufferCase("rows3_columns2_tile2_short", 3, 2, 2),
    BufferCase("rows1_columns1_tile3", 1, 1, 3),
    BufferCase("three_disjoint_epochs", 2, 3, 2, epochs=3),
    BufferCase("alignment32_tile3", 2, 3, 3, alignment=32),
)


def mutations(buffers):
    from xdsl.dialects import arith, builtin, func, memref, scf
    from xdsl.rewriter import Rewriter

    from merlin.verify.buffer_reuse_bridge import parse
    from merlin.xdsl_dialects._common import text

    def change(stage, update):
        module = parse(buffers[stage])
        entry = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "kernel")
        update(module, entry)
        module.verify()
        output = list(buffers)
        output[stage] = text(module, generic=True)
        if output[stage] == buffers[stage]:
            raise ValueError("mutation changed no bytes")
        return tuple(output)

    def replace_operand(op, index, value):
        values = list(op.operands)
        values[index] = value
        op.operands = values

    def constant(op, value):
        op.properties["value"] = builtin.IntegerAttr(value, builtin.IndexType())

    def index_constant(entry, value):
        return next(
            op
            for op in entry.body.block.ops
            if isinstance(op, arith.ConstantOp)
            and isinstance(op.result.type, builtin.IndexType)
            and op.value.value.data == value
        )

    def initializer_store(entry, which=0):
        return [op for op in entry.walk() if isinstance(op, memref.StoreOp)][which]

    def first_loop(entry):
        return next(op for op in entry.body.block.ops if isinstance(op, scf.ForOp))

    def touch(entry):
        return [op for op in entry.body.block.ops if isinstance(op, func.CallOp)]

    output = {}
    output["source_wrong_seed"] = change(
        0, lambda module, entry: replace_operand(initializer_store(entry), 0, entry.args[1])
    )
    output["normalized_wrong_seed"] = change(
        1, lambda module, entry: replace_operand(initializer_store(entry), 0, entry.args[1])
    )
    output["reused_wrong_seed"] = change(
        2, lambda module, entry: replace_operand(initializer_store(entry), 0, entry.args[1])
    )
    output["final_wrong_seed"] = change(
        3, lambda module, entry: replace_operand(initializer_store(entry), 0, entry.args[1])
    )
    output["final_missing_initializer_store"] = change(
        3, lambda module, entry: Rewriter.erase_op(initializer_store(entry))
    )
    output["final_wrong_tile_stride"] = change(
        3, lambda module, entry: replace_operand(first_loop(entry), 2, index_constant(entry, 1).result)
    )
    output["final_wrong_inner_cutoff"] = change(
        3,
        lambda module, entry: replace_operand(
            next(op for op in first_loop(entry).body.block.ops if isinstance(op, scf.ForOp)), 1, first_loop(entry).ub
        ),
    )
    output["final_wrong_load_address"] = change(
        3,
        lambda module, entry: replace_operand(
            [op for op in entry.body.block.ops if isinstance(op, memref.LoadOp)][1], 1, index_constant(entry, 1).result
        ),
    )
    output["final_wrong_store_address"] = change(
        3,
        lambda module, entry: replace_operand(
            next(op for op in entry.body.block.ops if isinstance(op, memref.StoreOp)),
            2,
            index_constant(entry, 1).result,
        ),
    )
    output["final_wrong_scalar_return"] = change(
        3, lambda module, entry: replace_operand(entry.body.block.last_op, 0, entry.args[0])
    )
    output["source_wrong_scalar_return"] = change(
        0, lambda module, entry: replace_operand(entry.body.block.last_op, 0, entry.args[0])
    )
    output["final_wrong_observer_operand"] = change(
        3, lambda module, entry: replace_operand(touch(entry)[0], 0, entry.args[1])
    )
    output["final_early_free"] = change(
        3,
        lambda module, entry: entry.body.block.insert_op_before(
            entry.body.block.detach_op(next(op for op in entry.body.block.ops if isinstance(op, memref.DeallocOp))),
            first_loop(entry),
        ),
    )
    output["final_missing_free"] = change(
        3,
        lambda module, entry: Rewriter.erase_op(
            next(op for op in entry.body.block.ops if isinstance(op, memref.DeallocOp))
        ),
    )
    output["final_alignment1"] = change(
        3,
        lambda module, entry: next(
            op for op in entry.body.block.ops if isinstance(op, memref.AllocOp)
        ).properties.__setitem__("alignment", builtin.IntegerAttr(1, builtin.i64)),
    )
    output["final_visibility"] = change(
        3, lambda module, entry: entry.properties.__setitem__("sym_visibility", builtin.StringAttr("private"))
    )
    output["final_entry_renamed"] = change(
        3, lambda module, entry: entry.properties.__setitem__("sym_name", builtin.StringAttr("changed_kernel"))
    )
    output["final_missing_unchanged_entry"] = change(
        3,
        lambda module, entry: Rewriter.erase_op(
            next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "untouched")
        ),
    )

    def observer_order(module, entry):
        first, second = touch(entry)
        a, b = list(first.operands), list(second.operands)
        # Both values dominate the second call, while only the first value
        # dominates the first call. Use initialized common seeds to keep SSA valid.
        first.operands = [entry.args[0], a[1]]
        second.operands = [entry.args[1], b[1]]

    output["final_observer_arguments_order"] = change(3, observer_order)

    def reordered(module, entry):
        first, second = touch(entry)
        entry.body.block.detach_op(first)
        entry.body.block.insert_op_after(first, second)

    output["final_observer_reordered"] = change(3, reordered)
    output["final_missing_observer"] = change(
        3, lambda module, entry: Rewriter.replace_op(touch(entry)[0], [], [entry.args[0]])
    )

    def private_type(module, entry, *, strided=False):
        kind = (
            builtin.MemRefType(builtin.i32, [2, 3], layout=builtin.StridedLayoutAttr([4, 1]))
            if strided
            else builtin.MemRefType(builtin.i32, [2, 2])
        )
        for operation in entry.walk():
            for value in tuple(operation.results):
                if isinstance(value.type, builtin.MemRefType) and len(value.type.get_shape()) == 2:
                    Rewriter.replace_value_with_new_type(value, kind)

    output["final_wrong_allocation_shape"] = change(3, private_type)
    output["reused_wrong_allocation_shape"] = change(2, private_type)
    output["final_nonidentity_strides"] = change(3, lambda module, entry: private_type(module, entry, strided=True))
    return output
