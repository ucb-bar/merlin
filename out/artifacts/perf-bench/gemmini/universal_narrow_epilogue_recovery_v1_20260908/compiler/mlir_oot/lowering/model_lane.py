"""Mixed-lane lowering: one program that drives the mesh AND the scalar lane.

A model does not belong to one lane.  Its int8 contractions are exactly what this target's
capability manifest admits on the 16x16 mesh; the normalizations, casts and elementwise maps
between them are families the mesh has no datapath for and belong on the scalar lane.  Leaving
the admitted work on the host is a compiler defect, not a placement choice, so this pass splits
the module's own dataflow into an ordered list of SEGMENTS -- a mesh contraction, or a run of
host ops -- and gives every value that crosses a segment boundary a DRAM buffer.

Nothing here is keyed on a capsule: which ops go to the mesh is decided by
`frontend.linalg_reader.place` (family + operand dtype against the RTL-derived datapath) and by
whether the op IS a contraction the tile schedule can express, and every extent is read from the
op's own operand types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Block, Operation, SSAValue
from ..codegen.host_linalg import HOST_LINALG_ELEMENT_BUDGET, attr_of, estimate_cost
from ..frontend.direct_conv import DirectConv, recognize as recognize_direct_conv
from ..frontend.linalg_reader import HOST_LANE, MESH_LANE, LinalgWorkload
from ..tables import rtl_facts as F
from ..xdsl_utils import is_integer_matmul
from .plan import Buffer, Contraction, Convolution, Epilogue, LoweringDeclined, Plan, kernel_args


@dataclass
class HostSegment:
    """A run of host-lane ops, with the DRAM bindings that carry values in and out."""

    ops: list[Operation] = field(default_factory=list)
    inputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    outputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)


def _shape(ty) -> tuple[int, ...]:
    if not isinstance(ty, TensorType):
        raise LoweringDeclined(f"expected a tensor type, got {ty}", op="model_lane")
    return tuple(int(d) for d in ty.get_shape())


def _elem(ty) -> str:
    ety = ty.get_element_type()
    if isinstance(ety, IntegerType):
        return f"i{int(ety.width.data)}"
    return str(ety)


def _func_of(module) -> Operation:
    for op in module.walk():
        if op.name == "func.func":
            return op
    raise LoweringDeclined("the module declares no func.func to lower", op="model_lane")


def _region_id(op: Operation) -> str:
    attr = op.attributes.get("prov.region_id")
    return getattr(attr, "data", "") or ""


def mesh_eligible(op: Operation, lane_of: dict[str, str]) -> bool:
    """Is this op a contraction the mesh both ADMITS and this backend can schedule?

    Both halves matter.  The placement rule says which (family, dtype) the datapath admits; this
    adds the second question a placement cannot answer -- whether the op's own shape is one the
    tile schedule expresses (a rank-2 contraction at the mesh operand dtype).
    """
    if lane_of.get(_region_id(op)) != MESH_LANE:
        return False
    conv = recognize_direct_conv(op)
    if conv is not None:
        return _is_zero_init(op.operands[2])
    if not is_integer_matmul(op):
        return False
    try:
        lhs, rhs = _shape(op.operands[0].type), _shape(op.operands[1].type)
        out = _shape(op.results[0].type)
    except LoweringDeclined:
        return False
    if len(lhs) != 2 or len(rhs) != 2 or len(out) != 2:
        return False
    if lhs[1] != rhs[0] or out != (lhs[0], rhs[1]):
        return False
    if not all(_elem(t.type) == F.OPERAND_DTYPE
               for t in (op.operands[0], op.operands[1])):
        return False
    # The mesh contraction starts from a ZERO accumulator.  An `outs` operand that is anything
    # else carries an initial value the tile schedule would silently drop, so such a matmul is
    # not one this lowering can express and stays on the host.
    return _is_zero_init(op.operands[2])


def _is_zero_init(value: SSAValue) -> bool:
    """Is `value` a fill of the additive identity (what a from-zero accumulation needs)?"""
    producer = value.owner if isinstance(value.owner, Operation) else None
    if producer is None or producer.name != "linalg.fill":
        return False
    scalar = producer.operands[0]
    src = scalar.owner if isinstance(scalar.owner, Operation) else None
    if src is None or src.name != "arith.constant":
        return False
    attr = attr_of(src, "value")
    data = getattr(getattr(attr, "value", None), "data", None)
    return data is not None and float(data) == 0.0


def _zero_init_chain(value: SSAValue) -> list[Operation]:
    """Return the source ops implementing a from-zero tensor initializer.

    A mesh contraction consumes the semantic fact that its accumulator starts at
    zero; it does not consume a materialized DRAM tensor containing those zeros.
    Consequently the ``empty -> constant zero -> fill`` chain is owned by that
    contraction task, rather than becoming a spurious host task in the global
    dispatch ledger.
    """
    fill = value.owner if isinstance(value.owner, Operation) else None
    if fill is None or fill.name != "linalg.fill" or not _is_zero_init(value):
        return []
    empty = (fill.operands[1].owner
             if isinstance(fill.operands[1].owner, Operation) else None)
    zero = (fill.operands[0].owner
            if isinstance(fill.operands[0].owner, Operation) else None)
    return [op for op in (empty, zero, fill) if op is not None]


def _constant_weight_transpose(op: Operation) -> tuple[Operation, SSAValue] | None:
    """Return a sole-use ``[N,K] -> [K,N]`` i8 RHS transpose and its source.

    This is a storage-layout choice, not model compute. Recognizing it here lets
    the bundle packer store the immutable weight in consumer layout once while
    target-neutral scalar/RVV paths retain their explicit transpose.
    """
    if not is_integer_matmul(op):
        return None
    rhs = op.operands[1]
    tr = rhs.owner if isinstance(rhs.owner, Operation) else None
    if tr is None or tr.name != "linalg.transpose" or len(tr.inputs) != 1:
        return None
    if list(tr.permutation.get_values()) != [1, 0]:
        return None
    source = tr.inputs[0]
    if not isinstance(source.owner, Block) or source.owner is not op.parent_block():
        return None
    if _elem(source.type) != "i8" or len(_shape(source.type)) != 2:
        return None
    live = [use for use in rhs.uses if use.operation.parent_block() is not None]
    if len(live) != 1 or live[0].operation is not op:
        return None
    return tr, source


class MixedBuilder:
    """Splits one `linalg-on-tensors` function into mesh contractions and host segments."""

    def __init__(self, module, wl: LinalgWorkload):
        self.module = module
        self.wl = wl
        self.func = _func_of(module)
        self.block = self.func.regions[0].blocks[0]
        self.lane_of = {r.region_id: r.lane for r in wl.regions}
        self.buffers: dict[str, Buffer] = {}
        self.order: list[str] = []
        self.of_value: dict[SSAValue, str] = {}
        self.ordered: list[Any] = []
        self.commands: list[dict[str, Any]] = []
        self.residents: dict[str, str] = {}
        self.weight_prepack: list[dict[str, Any]] = []
        self.absorbed_by: dict[Operation, list[Operation]] = {}
        self._scratch = 0
        self.source_indices = {op: i for i, op in enumerate(self.block.ops)
                               if op.name != "func.return"}

    # -- buffers -----------------------------------------------------------------------------
    def declare(self, name: str, shape, dtype: str, role: str) -> str:
        self.buffers[name] = Buffer(name, [int(d) for d in shape] or [1], dtype, role)
        if name not in self.order:
            self.order.append(name)
        return name

    #: The role a compiler-owned intermediate is declared with.  NOT `input`: a leaf input is a
    #: tensor some command consumes and none produces, and the dataflow binder reads the buffer's
    #: leaves as the model's own inputs.  An intermediate IS produced -- by a COMMIT, or by the
    #: host segment that spills it -- so declaring it `input` would add a tensor nobody passes in
    #: to the list of things the runner thinks it has to supply.
    INTERMEDIATE_ROLE = "intermediate"

    def buffer_for(self, value: SSAValue, role: str = "") -> str:
        """The DRAM buffer holding `value`, declaring an intermediate one on first use."""
        role = role or self.INTERMEDIATE_ROLE
        hit = self.of_value.get(value)
        if hit is not None:
            return hit
        name = f"t{self._scratch}"
        self._scratch += 1
        self.of_value[value] = self.declare(name, _shape(value.type), _elem(value.type), role)
        return name

    # -- entry point -------------------------------------------------------------------------
    def build(self) -> Plan:
        ops = [op for op in self.block.ops if op.name != "func.return"]
        returns = [op for op in self.block.ops if op.name == "func.return"]
        if not returns:
            raise LoweringDeclined("the entry function returns nothing to write", op="model_lane")
        result_values = list(returns[0].operands)

        mesh_ops = [op for op in ops if mesh_eligible(op, self.lane_of)]
        if not mesh_ops:
            raise LoweringDeclined(
                "no region of this module is a contraction the mesh admits", op="model_lane")
        # Source-native host loops no longer expand in proportion to tensor size.

        # interface tensors: the entry's arguments, then its results, in declaration order
        rhs_values = {
            (_constant_weight_transpose(op) or (None, op.operands[1]))[1]
            for op in mesh_ops
        }
        for i, arg in enumerate(self.block.args):
            role = "weight" if arg in rhs_values else "input"
            self.of_value[arg] = self.declare(f"arg{i}", _shape(arg.type), _elem(arg.type), role)
        out_names: list[str] = []
        for i, value in enumerate(result_values):
            name = self.declare(f"Y{i}", _shape(value.type), _elem(value.type), "output")
            # A result the mesh commits to IS this buffer; a result the host computes is stored
            # into it.  Either way the interface's own name is the one the runner reads back.
            self.of_value.setdefault(value, name)
            if self.of_value[value] != name:
                self.of_value[value] = name
            out_names.append(name)

        # Which values each host run must LEAVE in DRAM: whatever a later mesh op reads.
        needed_by_mesh: dict[SSAValue, None] = {}
        for op in mesh_ops:
            conv = recognize_direct_conv(op)
            needed_by_mesh[conv.activation if conv is not None else op.operands[0]] = None
            folded_transpose = _constant_weight_transpose(op)
            needed_by_mesh[(folded_transpose[1] if folded_transpose is not None
                            else op.operands[1])] = None

        # Padding and zero-init chains are not host work when a convolution target task owns
        # their exact semantics. Give those source ops to the task receipt and omit them from the
        # host stream; otherwise Phase 1 would both generate padding in im2col AND materialize the
        # full padded tensor it replaced.
        absorbed_by: dict[Operation, list[Operation]] = {}
        absorbed: set[Operation] = set()
        for op in mesh_ops:
            conv = recognize_direct_conv(op)
            owned: list[Operation] = []
            folded_transpose = _constant_weight_transpose(op)
            if folded_transpose is not None:
                transpose, _source = folded_transpose
                empty = (transpose.outputs[0].owner
                         if isinstance(transpose.outputs[0].owner, Operation) else None)
                owned.extend(x for x in (empty, transpose) if x is not None)
            if conv is None:
                # The tile schedule initializes the accumulator itself.  Leaving the
                # linalg zero-fill chain in the host stream would report host work that
                # neither contributes a buffer to nor is read by this contraction.
                owned.extend(_zero_init_chain(op.operands[2]))
                absorbed_by[op] = owned
                absorbed.update(owned)
                continue
            pad = op.operands[0].owner if isinstance(op.operands[0].owner, Operation) else None
            if pad is not None and pad.name == "tensor.insert_slice":
                base = pad.operands[1].owner if isinstance(pad.operands[1].owner, Operation) else None
                zero = (base.operands[0].owner if base is not None and base.operands
                        and isinstance(base.operands[0].owner, Operation) else None)
                owned.extend(x for x in (zero, base, pad) if x is not None)
            owned.extend(_zero_init_chain(op.operands[2]))
            absorbed_by[op] = owned
            absorbed.update(owned)
        self.absorbed_by = absorbed_by

        segments: list[HostSegment] = []
        pending: list[Operation] = []
        mesh_set = set(mesh_ops)
        schedule: list[Any] = []
        for op in ops:
            if op in absorbed:
                continue
            if op in mesh_set:
                schedule.append(("host", pending))
                pending = []
                schedule.append(("mesh", op))
            else:
                pending.append(op)
        schedule.append(("host", pending))

        # Two different places a value can already be live when a host run needs it: in DRAM
        # (an interface tensor, or what a mesh contraction committed) or as a scalar SSA value.
        # Tensor temporaries may NOT survive a segment boundary implicitly: LoopHostLinalg backs
        # them with reusable scratch slots, so a later segment would read a slot that intervening
        # segments have overwritten. Any tensor with a use outside its host run is therefore a
        # real spill output and is rebound from DRAM at the later run.
        in_dram: set[SSAValue] = set(self.block.args)
        scalar_ssa: set[SSAValue] = set()
        carry: list[Operation] = []
        for kind, payload in schedule:
            if kind == "mesh":
                self._mesh(payload, in_dram)
                in_dram.add(payload.results[0])
                continue
            host_ops = carry + [op for op in payload if op.name != "func.return"]
            host_set = set(host_ops)
            defined = [r for op in host_ops for r in op.results]
            defined_set = set(defined)
            outputs: list[tuple[SSAValue, str]] = []
            for value in defined:
                used_outside_run = any(use.operation not in host_set for use in value.uses)
                if value in needed_by_mesh:
                    outputs.append((value, self.buffer_for(value)))
                elif value in result_values:
                    outputs.append((value, self.of_value[value]))
                elif isinstance(value.type, TensorType) and used_outside_run:
                    outputs.append((value, self.buffer_for(value)))
            if not outputs:
                # This run hands nothing to the mesh and nothing to the interface -- it is the
                # glue that initialises the next contraction's accumulator.  Carry its ops into
                # the next run rather than dropping them: a later run may still read them.
                carry = host_ops
                continue
            carry = []
            inputs: list[tuple[SSAValue, str]] = []
            seen: set[SSAValue] = set()
            for op in host_ops:
                for value in op.operands:
                    if value in defined_set or value in seen or value in scalar_ssa:
                        continue
                    if value not in in_dram:
                        raise LoweringDeclined(
                            f"the host run reads a value that is neither an interface tensor, a "
                            f"mesh result, nor computed by an earlier run: {value.type}",
                            op="model_lane")
                    seen.add(value)
                    inputs.append((value, self.of_value[value]))
            scalar_ssa |= {value for value in defined_set
                           if not isinstance(value.type, TensorType)}
            in_dram |= {v for v, _ in outputs}
            segments.append(HostSegment(
                host_ops, inputs, outputs,
                sorted({_region_id(op) for op in host_ops if _region_id(op)})))
            segments[-1].source_op_indices = [self.source_indices[op] for op in host_ops]
            self.ordered.append(segments[-1])

        # a mesh result that is the entry's own result needs no host store; one that is not is
        # read back by the segment after it, which the binding loop above already wired.
        cb = self._command_buffer(segments)
        return Plan("gemmini", self.buffers, self.ordered, cb,
                    kernel_args(cb, list(self.order)))

    # -- one mesh contraction ----------------------------------------------------------------
    def _mesh(self, op: Operation, in_dram: set[SSAValue]) -> None:
        from .plan import Contraction

        conv = recognize_direct_conv(op)
        if conv is not None:
            self._mesh_conv(op, conv, in_dram)
            return

        lhs_v, rhs_v = op.operands[0], op.operands[1]
        folded_transpose = _constant_weight_transpose(op)
        rhs_source_v = folded_transpose[1] if folded_transpose is not None else rhs_v
        out_v = op.results[0]
        for operand in (lhs_v, rhs_source_v):
            if operand not in in_dram:
                raise LoweringDeclined(
                    "a mesh contraction reads an operand no earlier segment left in DRAM",
                    op="model_lane")
        lhs = self.of_value.get(lhs_v) or self.buffer_for(lhs_v)
        rhs = self.of_value.get(rhs_source_v) or self.buffer_for(rhs_source_v)
        dst = self.of_value.get(out_v) or self.buffer_for(out_v)
        m, k = _shape(lhs_v.type)
        n = _shape(out_v.type)[1]
        if folded_transpose is not None:
            original_shape = list(self.buffers[rhs].shape)
            self.buffers[rhs].shape = [k, n]
            recipe = {
                "tensor": rhs, "source_shape": original_shape,
                "packed_shape": [k, n], "source_layout": "NK",
                "packed_layout": "KN_dim_padded",
            }
            if recipe not in self.weight_prepack:
                self.weight_prepack.append(recipe)
        out_dtype = _elem(out_v.type)
        # Integer tensor arithmetic wraps at its declared width. Native narrowed
        # accelerator readout saturates, so preserve a full accumulator and let
        # submitted host code truncate it. This policy follows source types.
        modular = int(out_dtype[1:]) < int(F.ACCUMULATOR_DTYPE[1:])
        temporary = self.declare(f"{dst}_accumulator", [m, n], F.ACCUMULATOR_DTYPE,
                                 self.INTERMEDIATE_ROLE) if modular else None
        handle = self.residents.get(rhs)
        if handle is None:
            handle = f"{rhs}_res"
            self.residents[rhs] = handle
            self.commands.append({"opcode": "RES_PACK",
                                  "operands": {"src": rhs, "dst": handle},
                                  "attributes": {"layout": "packed_rhs"}})
        acc = f"acc_{dst}"
        self.commands.append({"opcode": "MATMUL_RESIDENT",
                              "operands": {"lhs": lhs, "rhs": handle, "dst": acc}})
        self.commands.append({"opcode": "COMMIT",
                              "operands": {"src": acc, "dst": temporary or dst},
                              "attributes": {"epilogue": [], "output_dtype":
                                             F.ACCUMULATOR_DTYPE if modular else out_dtype}})
        self.ordered.append(
            Contraction(lhs, rhs, dst, m, k, n, lhs_row_elems=k, rhs_row_elems=n,
                        epilogue=Epilogue(stages=[], output_dtype=out_dtype,
                                          integer_output_policy="modular"),
                        accumulator_temporary=temporary))
        self.ordered[-1].source_op_indices = [
            *[self.source_indices[item] for item in self.absorbed_by.get(op, [])],
            self.source_indices[op],
        ]

    def _mesh_conv(self, op: Operation, spec: DirectConv, in_dram: set[SSAValue]) -> None:
        for operand in (spec.activation, spec.weight):
            if operand not in in_dram:
                raise LoweringDeclined(
                    "a streamed convolution reads an operand no earlier segment left in DRAM",
                    op="conv2d")
        activation = self.of_value.get(spec.activation) or self.buffer_for(spec.activation)
        weight = self.of_value.get(spec.weight) or self.buffer_for(spec.weight)
        dst = self.of_value.get(spec.output) or self.buffer_for(spec.output)
        # The compiler ABI consumes an offline-packed [Co,K] view. OIHW flattening preserves
        # element order; only end-of-row DIM padding is added by the bundle packer.
        original_shape = list(self.buffers[weight].shape)
        self.buffers[weight].shape = [spec.co, spec.k]
        recipe = {"tensor": weight, "source_shape": original_shape,
                  "packed_shape": [spec.co, spec.k], "source_layout": "OIHW",
                  "packed_layout": "CoK_dim_padded"}
        if recipe not in self.weight_prepack:
            self.weight_prepack.append(recipe)
        self.commands.append({"opcode": "RES_PACK", "operands": {"src": weight,
                              "dst": f"{weight}_res"},
                              "attributes": {"layout": "packed_conv_lhs"}})
        self.commands.append({"opcode": "CONV2D",
                              "operands": {"ifm": activation, "weight": f"{weight}_res",
                                           "dst": dst},
                              "attributes": {"kernel": [spec.kh, spec.kw, spec.ci, spec.co],
                                             "stride": [spec.stride_h, spec.stride_w],
                                             "padding": [spec.pad_top, spec.pad_left,
                                                         spec.pad_bottom, spec.pad_right],
                                             "dilation": [spec.dilation_h, spec.dilation_w],
                                             "layout": "nchw_streamed_row_im2col",
                                             "output_dtype": "i32"}})
        task = Convolution(
            activation, weight, dst, spec.batch, spec.ci, spec.hi, spec.wi, spec.co,
            spec.kh, spec.kw, spec.ho, spec.wo, spec.stride_h, spec.stride_w,
            spec.dilation_h, spec.dilation_w, spec.pad_top, spec.pad_left,
            spec.pad_bottom, spec.pad_right, spec.direct_dma)
        task.source_op_indices = [self.source_indices[item]
                                  for item in [*self.absorbed_by.get(op, []), op]]
        self.ordered.append(task)

    # -- serialisation -----------------------------------------------------------------------
    def _command_buffer(self, segments: list[HostSegment]) -> dict[str, Any]:
        # Bind host inputs explicitly; resident packs describe accelerator work only.
        commands = self.commands
        tensors = {n: {"shape": self.buffers[n].shape, "dtype": self.buffers[n].dtype,
                       "role": self.buffers[n].role} for n in self.order}
        placement = [{"region": r.region_id, "family": r.family, "op": r.op, "dtype": r.dtype,
                      "lane": r.lane, "reason": r.reason} for r in self.wl.regions]
        return {
            "abi_version": "0.1",
            "target": "gemmini",
            "backend": "mlir_oot_xdsl_gemmini",
            "tensors": tensors,
            "kernel_abi": {
                "kind": "whole_program",
                "args": [{"tensor": n, "access": "write" if self.buffers[n].role
                          in ("output", "intermediate") else "read"} for n in self.order],
                "outputs": [n for n in self.order if self.buffers[n].role == "output"],
            },
            "commands": commands,
            "params": {
                "global_program_plan": {
                    "schema": "mixed_program_plan_v1",
                    "source_op_count": len(self.source_indices),
                    "compiler_temporaries": [
                        {"tensor": task.accumulator_temporary,
                         "source_op_index": task.source_op_indices[0],
                         "source_result_index": 0, "purpose": "accumulator_readout"}
                        for task in self.ordered if isinstance(task, Contraction)
                        and task.accumulator_temporary],
                    "tasks": [
                        {"task_index": i,
                         "kind": ("host" if isinstance(task, HostSegment) else
                                  "convolution" if isinstance(task, Convolution) else
                                  "contraction"),
                         "source_op_indices": list(task.source_op_indices),
                         "reads": [n for _, n in task.inputs] if isinstance(task, HostSegment)
                                  else ([task.activation, task.weight]
                                        if isinstance(task, Convolution)
                                        else [task.lhs, task.rhs]),
                         "writes": [n for _, n in task.outputs] if isinstance(task, HostSegment)
                                   else ([task.dst] if isinstance(task, Convolution) else
                                         [task.dst] + ([task.accumulator_temporary]
                                                       if task.accumulator_temporary else [])),
                         **({} if isinstance(task, (HostSegment, Convolution)) else {
                             "integer_output_policy": task.epilogue.integer_output_policy,
                             "accumulator_temporary": task.accumulator_temporary})}
                        for i, task in enumerate(self.ordered)],
                    "source_values": [
                        {"op_index": self.source_indices[value.owner],
                         "result_index": value.index, "tensor": name}
                        for value, name in self.of_value.items()
                        if isinstance(value.owner, Operation) and value.owner in self.source_indices],
                    "entry_bindings": [self.of_value[arg] for arg in self.block.args],
                    "output_bindings": [n for n in self.order if self.buffers[n].role == "output"],
                },
                "lane_placement": placement,
                "mesh_regions": [r.region_id for r in self.wl.mesh_regions],
                "host_lane_regions": [r.region_id for r in self.wl.host_regions],
                "lanes": {
                    "reported": sorted({r.lane for r in self.wl.regions}),
                    MESH_LANE: [r.region_id for r in self.wl.mesh_regions],
                    HOST_LANE: [r.region_id for r in self.wl.host_regions],
                },
                "host_lane_segments": [
                    {"regions": seg.regions,
                     "reads": [n for _, n in seg.inputs],
                     "writes": [n for _, n in seg.outputs]} for seg in segments],
                "weight_prepack_recipes": list(self.weight_prepack),
                "convolution_lowering": {
                    "owner": "generated_phase1_target_dialect",
                    "strategy": "direct_dma_or_streamed_output_row_im2col",
                    "full_im2col_materialized": False,
                },
            },
        }


def build(module, wl: LinalgWorkload) -> Plan:
    return MixedBuilder(module, wl).build()
