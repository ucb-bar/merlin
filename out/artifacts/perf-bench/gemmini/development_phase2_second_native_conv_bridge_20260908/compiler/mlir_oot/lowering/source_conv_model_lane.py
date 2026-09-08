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
from math import prod
from typing import Any

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Block, Operation, SSAValue
from merlin.xdsl_dialects.lowering.canonical_matmul import is_integer_matmul

from ..codegen.host_linalg import HOST_LINALG_ELEMENT_BUDGET, attr_of, estimate_cost
from ..frontend.direct_conv import DirectConv, recognize as recognize_direct_conv
from ..frontend.linalg_reader import HOST_LANE, MESH_LANE, LinalgWorkload
from ..tables import rtl_facts as F
from .plan import Buffer, Contraction, Epilogue, LoweringDeclined, Plan, kernel_args, row_pitch
from .source_conv_plan import Convolution
from .model_lane import MixedBuilder, HostSegment


def place_source_convolutions(module, workload):
    """Place the actual integer contraction, not incidental f32 region furniture."""
    from ..frontend.linalg_reader import place
    regions = {region.region_id: region for region in workload.regions}
    for operation in module.walk():
        if recognize_direct_conv(operation) is None:
            continue
        region = regions.get(_region_id(operation))
        if region is None:
            raise LoweringDeclined("source convolution lacks an owned placement region", op="conv2d")
        region.family = "contraction"
        region.dtype = _elem(operation.operands[0].type)
        place(region)


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
    if any(use.operation is not tr for use in source.uses):
        return None
    if _elem(source.type) != "i8" or len(_shape(source.type)) != 2:
        return None
    live = [use for use in rhs.uses if use.operation.parent_block() is not None]
    if len(live) != 1 or live[0].operation is not op:
        return None
    return tr, source



class SourceConvBuilder(MixedBuilder):
    """Explicit source-convolution route; default mixed planner remains unchanged."""

    def __init__(self, module, wl, *, diagnostic_transposed_compute_only=False):
        super().__init__(module, wl)
        self.weight_prepack = []
        self.absorbed_by = {}
        self.diagnostic_transposed_compute_only = bool(diagnostic_transposed_compute_only)

    def _encode_buffer(self, name, logical_shape, axis_groups):
        """One explicit contract drives both the caller ABI and device row pitches.

        Groups preserve their stated mixed-radix order; equal element counts do
        not authorize an axis permutation. Only source-derived routes call this.
        """
        buffer = self.buffers[name]
        logical_shape = list(logical_shape)
        physical = [prod(logical_shape[axis] for axis in group) for group in axis_groups]
        if physical != buffer.shape:
            raise LoweringDeclined("source axis grouping differs from physical buffer shape",
                                   op="storage_encoding")
        pitch = row_pitch(physical[-1])
        strides = [1] * len(physical)
        if len(physical) > 1:
            strides[-2] = pitch
            for axis in range(len(physical) - 3, -1, -1):
                strides[axis] = strides[axis + 1] * physical[axis + 1]
        buffer.storage_encoding = {
            "schema": "grouped_axes_storage_v1", "logical_shape": logical_shape,
            "dtype": buffer.dtype, "axis_groups": [list(group) for group in axis_groups],
            "physical_shape": list(physical), "strides_elements": strides,
            "storage_elements": prod(physical[:-1]) * pitch, "offset_elements": 0,
        }

    def declare(self, name, shape, dtype, role):
        logical_shape = [int(dim) for dim in shape]
        result = super().declare(name, logical_shape, dtype, role)
        groups = [[axis] for axis in range(len(logical_shape))] or [[]]
        self._encode_buffer(result, logical_shape, groups)
        return result

    def build(self) -> Plan:
        ops = [op for op in self.block.ops if op.name != "func.return"]
        returns = [op for op in self.block.ops if op.name == "func.return"]
        if not returns:
            raise LoweringDeclined("the entry function returns nothing to write", op="model_lane")
        result_values = list(returns[0].operands)

        mesh_ops = [op for op in ops if mesh_eligible(op, self.lane_of)]
        if not any(recognize_direct_conv(op) is not None for op in mesh_ops):
            # Enabling a device route does not make other source graphs illegal.
            # Reuse the canonical whole-program planner, including source-native
            # host loops, original scalar/tensor dtypes and source ownership.
            # A fresh builder avoids source-convolution storage/grouping rules
            # leaking into ordinary matmul/host graphs or introducing artificial
            # mesh work. Use this extension only when it owns an admitted conv.
            return MixedBuilder(self.module, self.wl).build()
        self._select_exact_native_epilogues(mesh_ops)
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
                absorbed_by[op] = owned
                absorbed.update(owned)
                continue
            pad = op.operands[0].owner if isinstance(op.operands[0].owner, Operation) else None
            if pad is not None and pad.name == "tensor.insert_slice":
                base = pad.operands[1].owner if isinstance(pad.operands[1].owner, Operation) else None
                zero = (base.operands[0].owner if base is not None and base.operands
                        and isinstance(base.operands[0].owner, Operation) else None)
                owned.extend(x for x in (zero, base, pad) if x is not None)
            fill = op.operands[2].owner if isinstance(op.operands[2].owner, Operation) else None
            if fill is not None and fill.name == "linalg.fill":
                empty = fill.operands[1].owner if isinstance(fill.operands[1].owner, Operation) else None
                zero = fill.operands[0].owner if isinstance(fill.operands[0].owner, Operation) else None
                owned.extend(x for x in (empty, zero, fill) if x is not None)
            absorbed_by[op] = owned
            absorbed.update(owned)
        all_owners = {}
        for sink, owned in absorbed_by.items():
            for item in owned:
                all_owners.setdefault(item, []).append(sink)
        for sink, owned in list(absorbed_by.items()):
            exclusive = {item for item in owned if len(all_owners[item]) == 1}
            while True:
                kept = {item for item in exclusive if all(
                    use.operation is sink or use.operation in exclusive
                    for value in item.results for use in value.uses)}
                if kept == exclusive:
                    break
                exclusive = kept
            absorbed_by[sink] = [item for item in owned if item in exclusive]
        for producer, formation in self.native_epilogues.items():
            owned = self._exclusive_epilogue_owners(producer, formation)
            absorbed_by[producer] = sorted(
                set(absorbed_by.get(producer, ())) | owned,
                key=lambda item: list(item.parent_block().ops).index(item))
        absorbed = {item for owned in absorbed_by.values() for item in owned}
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
                formation = self.native_epilogues.get(payload)
                if formation is not None:
                    in_dram.add(formation.output)
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
                source_operands = (value for nested in op.walk() for value in nested.operands
                                   if value.owner is self.block or value.owner in self.source_indices)
                for value in source_operands:
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
            self._encode_buffer(rhs, original_shape, [[1], [0]])
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
        if (spec.weight.owner is not self.block
                or any(use.operation is not op for use in spec.weight.uses)):
            raise LoweringDeclined(
                "offline convolution prepacking requires an exclusive entry weight; "
                "shared or computed weight needs a separate explicit encoding adapter", op="conv2d")
        for operand in (spec.activation, spec.weight):
            if operand not in in_dram:
                raise LoweringDeclined(
                    "a streamed convolution reads an operand no earlier segment left in DRAM",
                    op="conv2d")
        activation = self.of_value.get(spec.activation) or self.buffer_for(spec.activation)
        weight = self.of_value.get(spec.weight) or self.buffer_for(spec.weight)
        formation = self.native_epilogues.get(op)
        out_value = formation.output if formation is not None else spec.output
        dst = self.of_value.get(out_value) or self.buffer_for(out_value)
        # The compute-only LOOP_CONV route is intentionally restricted to the exact NCHW case
        # its training-convolution transpose bit represents: batch one.  All other cases retain
        # the established streamed-row fallback and its CoK packing.
        compute_only_candidate = bool(
            formation is None and F.HAS_LOOP_CONV and F.HAS_TRAINING_CONVS
            and self.buffers[dst].role != "output"
            and self.buffers[activation].pitch == spec.wi
            and spec.batch == 1 and spec.kh == spec.kw
            and spec.stride_h == spec.stride_w
            and spec.dilation_h == spec.dilation_w
            and spec.pad_top == spec.pad_left == spec.pad_bottom == spec.pad_right
            and 0 <= spec.pad_top < spec.kh
            and max(spec.batch, spec.ci, spec.hi, spec.wi, spec.co, spec.kh, spec.kw,
                    spec.ho, spec.wo) < (1 << 16)
            and spec.stride_h < (1 << 8) and spec.dilation_h < (1 << 10)
            and spec.kh * spec.kw * spec.ci * 128 * 128 <= (1 << 31) - 1)
        compute_only = compute_only_candidate and self.diagnostic_transposed_compute_only
        original_shape = list(self.buffers[weight].shape)
        if compute_only:
            # LOOP_CONV's default weight stream is HWIO.  This is an offline constant transform,
            # never runtime work.  K-major rows are padded only along the output-channel axis.
            self.buffers[weight].shape = [spec.k, spec.co]
            self._encode_buffer(weight, original_shape, [[2, 3, 1], [0]])
            recipe = {"tensor": weight, "source_shape": original_shape,
                      "packed_shape": [spec.k, spec.co], "source_layout": "OIHW",
                      "packed_layout": "HWIO_dim_padded",
                      "permutation": [2, 3, 1, 0]}
            out = self.buffers[dst]
            logical = [spec.batch, spec.co, spec.ho, spec.wo]
            physical = [spec.batch, spec.ho, spec.wo, spec.co]
            pitch = row_pitch(spec.co)
            out.storage_encoding = {
                "schema": "permuted_axes_storage_v1", "logical_shape": logical,
                "dtype": out.dtype, "permutation": [0, 2, 3, 1],
                "physical_shape": physical,
                "strides_elements": [spec.ho * spec.wo * pitch,
                                     spec.wo * pitch, pitch, 1],
                "storage_elements": spec.batch * spec.ho * spec.wo * pitch,
                "offset_elements": 0,
            }
        else:
            # The fallback ABI consumes an offline-packed [Co,K] view. OIHW flattening preserves
            # element order; only end-of-row DIM padding is added by the bundle packer.
            self.buffers[weight].shape = [spec.co, spec.k]
            self._encode_buffer(weight, original_shape, [[0], [1, 2, 3]])
            recipe = {"tensor": weight, "source_shape": original_shape,
                      "packed_shape": [spec.co, spec.k], "source_layout": "OIHW",
                      "packed_layout": "CoK_dim_padded"}
        if recipe not in self.weight_prepack:
            self.weight_prepack.append(recipe)
        self.commands.append({"opcode": "RES_PACK", "operands": {"src": weight,
                              "dst": f"{weight}_res"},
                              "attributes": {"layout": "packed_conv_lhs"}})
        attributes = {"kernel": [spec.kh, spec.kw, spec.ci, spec.co],
                      "stride": [spec.stride_h, spec.stride_w],
                      "padding": [spec.pad_top, spec.pad_left,
                                  spec.pad_bottom, spec.pad_right],
                      "dilation": [spec.dilation_h, spec.dilation_w],
                      "layout": ("nchw_batch1_loop_conv_compute_only"
                                 if compute_only else "nchw_streamed_row_im2col"),
                      "output_dtype": ("i8" if formation is not None else "i32")}
        # Keep an unselected command byte-for-byte stable.  An empty epilogue list is redundant
        # semantics, but adding it to every existing convolution obscures whether any physical work
        # changed in command-buffer A/B evidence.
        if formation is not None:
            attributes["epilogue"] = ["relu"] if formation.relu else []
        self.commands.append({"opcode": "CONV2D",
                              "operands": {"ifm": activation, "weight": f"{weight}_res",
                                           "dst": dst},
                              "attributes": attributes})
        task = Convolution(
            activation, weight, dst, spec.batch, spec.ci, spec.hi, spec.wi, spec.co,
            spec.kh, spec.kw, spec.ho, spec.wo, spec.stride_h, spec.stride_w,
            spec.dilation_h, spec.dilation_w, spec.pad_top, spec.pad_left,
            spec.pad_bottom, spec.pad_right, spec.direct_dma,
            input_layout="NCHW_batch1" if compute_only else "NCHW",
            weight_layout="HWIO" if compute_only else "CoK",
            output_layout="NHWC_accumulator" if compute_only else "NCHW",
            output_dtype=("i8" if formation is not None else "i32"),
            relu=bool(formation is not None and formation.relu),
            compute_only_native=compute_only,
            diagnostic_transposed_compute_only=compute_only,
            native_performance_refusal=(
                "hardware_cost_guard_transposed_nchw_underfills_systolic_rows"
                if compute_only_candidate and not compute_only else None))
        if formation is not None:
            self.of_value[spec.output] = dst
            self.of_value[formation.output] = dst
        task.source_op_indices = [self.source_indices[item]
                                  for item in [*self.absorbed_by.get(op, []), op]]
        self.ordered.append(task)

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
                "storage_encodings": {name: self.buffers[name].storage_encoding
                                      for name in self.order},
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
                "target_neutral_quantized_epilogues": self._epilogue_receipt(),
                "weight_prepack_recipes": list(self.weight_prepack),
                "convolution_lowering": {
                    "owner": "generated_phase1_target_dialect",
                    "strategy": "direct_dma_or_streamed_output_row_im2col",
                    "full_im2col_materialized": False,
                },
            },
        }


def build(module, wl, *, diagnostic_transposed_compute_only=False):
    return SourceConvBuilder(
        module, wl,
        diagnostic_transposed_compute_only=diagnostic_transposed_compute_only).build()
