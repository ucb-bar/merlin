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
from xdsl.ir.affine import AffineDimExpr
from merlin.xdsl_dialects.lowering.canonical_matmul import is_integer_matmul

from ..codegen.host_linalg import HOST_LINALG_ELEMENT_BUDGET, attr_of, estimate_cost
from ..frontend.linalg_reader import HOST_LANE, MESH_LANE, LinalgWorkload, place
from ..frontend.quantized_epilogue import (
    QuantizedEpilogue,
    Refusal as EpilogueRefusal,
    exact_identity_narrowing,
    recognize as recognize_quantized_epilogue,
)
from ..frontend.native_aligned_epilogue import (
    NativeAlignedEpilogue,
    recognize as recognize_native_aligned_epilogue,
)
from ..frontend.requantization_contract import (
    AccuracyPolicy,
    NativeNarrowCapability,
    RequantizationDecision,
    contract_from_formation,
    evaluate_accuracy_gate,
)
from ..frontend.residual_epilogue import (
    Refusal as ResidualRefusal,
    ResidualEpilogue,
    recognize_all as recognize_residual_epilogues,
)
from ..tables import rtl_facts as F
from .plan import (BLOCK_DIAGONAL_ELEM_BUDGET, Buffer, Epilogue, LoweringDeclined,
                   Plan, kernel_args)


# Target adapter facts, separate from the source-semantic contract and generic accuracy gate.
# This endpoint has one scalar scale, no output zero-point adder, and applies activation after
# round/saturation.  A richer backend supplies a different NativeNarrowCapability.
NATIVE_NARROW_CAPABILITY = NativeNarrowCapability(
    name="accumulator_narrow_i8_scalar_scale_v1",
    scale_granularities=("per_tensor",),
    scale_axis=None,
    bias_domains=("none", "accumulator_i32"),
    output_zero_points=(0,),
    rounding=("round_to_nearest_even",),
    saturation=(-128, 127),
    ordered_stage_templates=(
        ("acc_scale", "round_to_nearest_even", "saturate_i8"),
        ("acc_scale", "round_to_nearest_even", "saturate_i8", "relu"),
    ),
)


@dataclass
class HostSegment:
    """A run of host-lane ops, with the DRAM bindings that carry values in and out."""

    ops: list[Operation] = field(default_factory=list)
    inputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    outputs: list[tuple[SSAValue, str]] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    residual_fusions: list[ResidualEpilogue] = field(default_factory=list)


@dataclass(frozen=True)
class IntegerContraction:
    """The source-derived geometry of one integer contraction.

    ``batch_shape`` is empty for ordinary matmul.  A batched lhs with one shared rhs is already
    a 2-D contraction in row-major storage: flattening the leading batch axes into M preserves
    every product.  When both operands vary by batch, ``block_diagonal`` records the general
    block-diagonal reduction used by the interface planner.
    """

    lhs: SSAValue
    rhs: SSAValue
    output: SSAValue
    batch_shape: tuple[int, ...]
    m: int
    k: int
    n: int
    lhs_batched: bool
    rhs_batched: bool

    @property
    def batch(self) -> int:
        return prod(self.batch_shape) if self.batch_shape else 1

    @property
    def block_diagonal(self) -> bool:
        return bool(self.batch_shape) and self.lhs_batched and self.rhs_batched


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


def _map_positions(attr) -> tuple[int, ...] | None:
    amap = getattr(attr, "data", None)
    if amap is None or amap.num_symbols:
        return None
    positions: list[int] = []
    for expr in amap.results:
        if not isinstance(expr, AffineDimExpr):
            return None
        positions.append(expr.position)
    return tuple(positions)


def _is_integer_mac_body(op: Operation) -> bool:
    """Recognise the same exact integer MAC body as canonical rank-2 matmul.

    The shared canonical recogniser intentionally accepts only rank two.  Batched generics have
    different indexing maps but the scalar computation must still be exactly
    ``acc + sext(lhs) * sext(rhs)`` before this lowering may discard the source region.
    """
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    body = op.regions[0].blocks[0]
    if len(body.args) != 3:
        return False
    operations = list(body.ops)
    if (not operations or operations[-1].name != "linalg.yield"
            or any(item.name not in {
                "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"
            } for item in operations)):
        return False
    addition = operations[-1].operands[0].owner
    if not isinstance(addition, Operation) or addition.name != "arith.addi":
        return False
    terms = list(addition.operands)
    if body.args[2] not in terms:
        return False
    product_op = terms[1 - terms.index(body.args[2])].owner
    if not isinstance(product_op, Operation) or product_op.name != "arith.muli":
        return False

    def base(value: SSAValue) -> SSAValue:
        producer = value.owner
        if isinstance(producer, Operation) and producer.name == "arith.extsi":
            return producer.operands[0]
        return value

    return {base(value) for value in product_op.operands} == set(body.args[:2])


def _leading_broadcast_source(value: SSAValue, batch_rank: int) -> SSAValue | None:
    """Peel an explicit broadcast that adds exactly the contraction's leading batch axes."""
    producer = value.owner if isinstance(value.owner, Operation) else None
    if producer is None or producer.name != "linalg.broadcast" or len(producer.operands) < 1:
        return None
    dimensions = attr_of(producer, "dimensions")
    get_values = getattr(dimensions, "get_values", None)
    if get_values is None or tuple(int(v) for v in get_values()) != tuple(range(batch_rank)):
        return None
    source = producer.operands[0]
    return source if len(_shape(source.type)) == 2 else None


def _integer_contraction(op: Operation) -> IntegerContraction | None:
    """Recover rank-2 or batched matmul geometry from types and exact indexing semantics."""
    if len(op.operands) != 3 or len(op.results) != 1:
        return None
    try:
        lhs_shape = _shape(op.operands[0].type)
        rhs_shape = _shape(op.operands[1].type)
        out_shape = _shape(op.results[0].type)
    except LoweringDeclined:
        return None

    if len(out_shape) == 2:
        if not is_integer_matmul(op) or len(lhs_shape) != 2 or len(rhs_shape) != 2:
            return None
        m, n = out_shape
        if lhs_shape[0] != m:
            return None
        k = lhs_shape[1]
        if rhs_shape != (k, n):
            return None
        return IntegerContraction(op.operands[0], op.operands[1], op.results[0], (),
                                  m, k, n, False, False)

    if len(out_shape) < 3:
        return None
    batch_shape, m, n = out_shape[:-2], out_shape[-2], out_shape[-1]
    if not batch_shape or len(lhs_shape) not in (2, len(out_shape)):
        return None
    lhs_batched = len(lhs_shape) == len(out_shape)
    rhs_batched = len(rhs_shape) == len(out_shape)
    if lhs_shape != ((*batch_shape, m, lhs_shape[-1]) if lhs_batched
                     else (m, lhs_shape[-1])):
        return None
    k = lhs_shape[-1]
    if rhs_shape != ((*batch_shape, k, n) if rhs_batched else (k, n)):
        return None

    # Named batch_matmul owns these maps by definition.  For a generic, prove the precise maps
    # and scalar body instead of trusting provenance labels.
    if op.name != "linalg.batch_matmul":
        if op.name != "linalg.generic" or not _is_integer_mac_body(op):
            return None
        props = {**op.attributes, **op.properties}
        maps = props.get("indexing_maps")
        iters = props.get("iterator_types")
        rank = len(out_shape)
        if maps is None or len(maps) != 3 or iters is None:
            return None
        iterator_names = [getattr(getattr(item, "data", None), "value", None)
                          for item in iters]
        if iterator_names != [*(["parallel"] * rank), "reduction"]:
            return None
        batch_dims = tuple(range(rank - 2))
        m_dim, n_dim, k_dim = rank - 2, rank - 1, rank
        expected = (
            (*batch_dims, m_dim, k_dim) if lhs_batched else (m_dim, k_dim),
            (*batch_dims, k_dim, n_dim) if rhs_batched else (k_dim, n_dim),
            (*batch_dims, m_dim, n_dim),
        )
        if tuple(_map_positions(item) for item in maps) != expected:
            return None

    lhs, rhs = op.operands[0], op.operands[1]
    # A materialised leading-axis broadcast carries no per-batch information.  Contract its
    # source once as a shared matrix instead of block-diagonalising duplicated bytes.
    if lhs_batched and (source := _leading_broadcast_source(lhs, len(batch_shape))) is not None:
        if _shape(source.type) == (m, k):
            lhs, lhs_batched = source, False
    if rhs_batched and (source := _leading_broadcast_source(rhs, len(batch_shape))) is not None:
        if _shape(source.type) == (k, n):
            rhs, rhs_batched = source, False
    return IntegerContraction(lhs, rhs, op.results[0], batch_shape,
                              m, k, n, lhs_batched, rhs_batched)


def _block_diagonal_is_safe(op: Operation, spec: IntegerContraction) -> bool:
    """Whether the ABI's harness-derived block diagonal is semantically available here."""
    if not spec.batch_shape:
        return True
    # Flattening leading batch axes into M is valid precisely when the lhs varies per batch and
    # the rhs is shared.  A shared lhs with varying rhs (or two shared operands whose result must
    # be replicated) needs output/input slice addressing this plan does not claim to provide.
    if not spec.lhs_batched:
        return False
    if not spec.rhs_batched:
        return True
    rows, kdim = spec.batch * spec.m, spec.batch * spec.k
    if rows * kdim > BLOCK_DIAGONAL_ELEM_BUDGET:
        return False
    # im2col recipes are materialised before program execution.  Therefore both operands must be
    # entry leaves, not values a preceding host task computes, and changing their row-major view
    # must not alter another source consumer.
    if not isinstance(spec.lhs.owner, Block) or not isinstance(spec.rhs.owner, Block):
        return False
    return (all(use.operation is op for use in spec.lhs.uses)
            and all(use.operation is op for use in spec.rhs.uses))


def _place_integer_contractions(module, workload: LinalgWorkload) -> None:
    """Correct region placement from the contraction op's operand type, not its zero init.

    Captured generic matmuls commonly put the provenance region on ``arith.constant 0`` first.
    Region discovery then sees the accumulator's i32 type before it reaches the i8 operands.  The
    operation is the authoritative placement unit: re-run the central target placement rule with
    its actual lhs type for every contraction this backend can faithfully construct.
    """
    regions = {region.region_id: region for region in workload.regions}
    for op in module.walk():
        spec = _integer_contraction(op)
        if (spec is None or not _block_diagonal_is_safe(op, spec)
                or not _is_zero_init(op.operands[2])
                or any(_elem(value.type) != F.OPERAND_DTYPE
                       for value in (spec.lhs, spec.rhs))):
            continue
        region = regions.get(_region_id(op))
        if region is None:
            continue
        region.family = "contraction"
        region.dtype = _elem(spec.lhs.type)
        place(region)


def _absorbed_candidates(op: Operation, spec: IntegerContraction) -> list[Operation]:
    """Source ops whose exact semantics are owned by the constructed contraction."""
    owned: list[Operation] = []
    init = op.operands[2].owner if isinstance(op.operands[2].owner, Operation) else None
    if init is not None and init.name in ("linalg.fill", "tensor.splat"):
        scalar = init.operands[0].owner if isinstance(init.operands[0].owner, Operation) else None
        owned.extend(item for item in (scalar, init) if item is not None)
        if init.name == "linalg.fill" and len(init.operands) > 1:
            empty = (init.operands[1].owner
                     if isinstance(init.operands[1].owner, Operation) else None)
            if empty is not None:
                owned.insert(-1, empty)
    for original, effective in ((op.operands[0], spec.lhs), (op.operands[1], spec.rhs)):
        if original is effective:
            continue
        broadcast = original.owner if isinstance(original.owner, Operation) else None
        if broadcast is None or broadcast.name != "linalg.broadcast":
            continue
        empty = (broadcast.operands[1].owner if len(broadcast.operands) > 1
                 and isinstance(broadcast.operands[1].owner, Operation) else None)
        owned.extend(item for item in (empty, broadcast) if item is not None)
    return list(dict.fromkeys(owned))


def _exclusive_absorbed(mesh_ops: list[Operation]) -> dict[Operation, list[Operation]]:
    """Keep only derived/init ops whose every live use is owned by one mesh operation."""
    candidates = {op: _absorbed_candidates(op, _integer_contraction(op)) for op in mesh_ops}
    owners: dict[Operation, list[Operation]] = {}
    for sink, items in candidates.items():
        for item in items:
            owners.setdefault(item, []).append(sink)
    result: dict[Operation, list[Operation]] = {}
    for sink, items in candidates.items():
        exclusive = {item for item in items if len(owners[item]) == 1}
        while True:
            kept = {item for item in exclusive if all(
                use.operation is sink or use.operation in exclusive
                for value in item.results for use in value.uses)}
            if kept == exclusive:
                break
            exclusive = kept
        result[sink] = sorted(exclusive, key=lambda item: list(item.parent_block().ops).index(item))
    return result


def mesh_eligible(op: Operation, lane_of: dict[str, str]) -> bool:
    """Is this op a contraction the mesh both ADMITS and this backend can schedule?

    Both halves matter.  The placement rule says which (family, dtype) the datapath admits; this
    adds the second question a placement cannot answer -- whether the op's own shape is one the
    tile schedule expresses: rank two directly, batched with a shared RHS by flattening its
    leading axes, or independent batches through the ABI's bounded block-diagonal derivation.
    """
    if lane_of.get(_region_id(op)) != MESH_LANE:
        return False
    spec = _integer_contraction(op)
    if spec is None or not _block_diagonal_is_safe(op, spec):
        return False
    try:
        operand_dtypes = (_elem(spec.lhs.type), _elem(spec.rhs.type))
    except LoweringDeclined:
        return False
    if not all(dtype == F.OPERAND_DTYPE for dtype in operand_dtypes):
        return False
    # The mesh contraction starts from a ZERO accumulator.  An `outs` operand that is anything
    # else carries an initial value the tile schedule would silently drop, so such a matmul is
    # not one this lowering can express and stays on the host.
    return _is_zero_init(op.operands[2])


def _is_zero_init(value: SSAValue) -> bool:
    """Is `value` a fill of the additive identity (what a from-zero accumulation needs)?"""
    producer = value.owner if isinstance(value.owner, Operation) else None
    if producer is None or producer.name not in ("linalg.fill", "tensor.splat"):
        return False
    scalar = producer.operands[0]
    src = scalar.owner if isinstance(scalar.owner, Operation) else None
    if src is None or src.name != "arith.constant":
        return False
    attr = attr_of(src, "value")
    data = getattr(getattr(attr, "value", None), "data", None)
    return data is not None and float(data) == 0.0


class MixedBuilder:
    """Splits one `linalg-on-tensors` function into mesh contractions and host segments."""

    def __init__(self, module, wl: LinalgWorkload, *,
                 requantization_policy: AccuracyPolicy | None = None,
                 normalized_source_sha256: str = ""):
        self.module = module
        self.wl = wl
        self.func = _func_of(module)
        self.block = self.func.regions[0].blocks[0]
        _place_integer_contractions(module, wl)
        self.lane_of = {r.region_id: r.lane for r in wl.regions}
        self.buffers: dict[str, Buffer] = {}
        self.order: list[str] = []
        self.of_value: dict[SSAValue, str] = {}
        self.ordered: list[Any] = []
        self.commands: list[dict[str, Any]] = []
        self.residents: dict[str, str] = {}
        self.im2col_recipes: list[dict[str, Any]] = []
        self.batched_contractions: list[dict[str, Any]] = []
        self.host_tensor_spills: dict[str, SSAValue] = {}
        self.native_epilogues: dict[Operation, QuantizedEpilogue | NativeAlignedEpilogue] = {}
        self.native_epilogue_accuracy: dict[Operation, RequantizationDecision] = {}
        self.epilogue_refusals: list[dict[str, Any]] = []
        self.requantization_policy = requantization_policy
        self.normalized_source_sha256 = normalized_source_sha256
        self.residual_epilogues: list[ResidualEpilogue] = []
        self.selected_residual_epilogues: list[ResidualEpilogue] = []
        self.residual_refusals: list[ResidualRefusal] = []
        self.deferred_residual_ops: set[Operation] = set()
        self.deferred_before: dict[Operation, list[Operation]] = {}
        self.deferred_branch_producers: dict[Operation, tuple[int, ...]] = {}
        self._scratch = 0
        self.source_indices = {op: i for i, op in enumerate(self.block.ops)
                               if op.name != "func.return"}

    def prepare_residual_schedule(self, ops: list[Operation], mesh_ops: list[Operation]) -> None:
        """Delay exclusive pure branch epilogues until their residual join.

        This is ordinary dependence scheduling.  A branch may move only when its exact
        recognizer proves single-use pure tensor operations, and only when at least one
        independent mesh operation lies between the branch and the join.  Moving it later
        removes an f32 boundary tensor; all source scalar operations remain ordered inside
        the eventual fused host traversal.
        """
        formed, refused = recognize_residual_epilogues(ops)
        self.residual_epilogues = formed
        self.residual_refusals = refused
        mesh = set(mesh_ops)
        for formation in formed:
            add_index = self.source_indices[formation.residual_add]
            deferred_producers: list[int] = []
            for branch in formation.branches:
                if branch.producer not in mesh:
                    continue
                branch_ops = sorted(set(branch.operations), key=self.source_indices.__getitem__)
                last = max(self.source_indices[item] for item in branch_ops)
                if not any(last < self.source_indices[item] < add_index for item in mesh_ops):
                    continue
                if any(item in self.deferred_residual_ops for item in branch_ops):
                    raise LoweringDeclined(
                        "one source operation belongs to two deferred residual branches",
                        op="residual_epilogue")
                self.deferred_residual_ops.update(branch_ops)
                self.deferred_before.setdefault(formation.residual_add, []).extend(branch_ops)
                deferred_producers.append(self.source_indices[branch.producer])
            self.deferred_branch_producers[formation.residual_add] = tuple(deferred_producers)

    def append_scheduled_host_op(self, pending: list[Operation], op: Operation) -> None:
        """Append one host op, injecting any source-proven delayed branch first."""
        pending.extend(self.deferred_before.get(op, ()))
        pending.append(op)

    def residual_receipt(self) -> dict[str, Any]:
        formed = []
        for item in self.residual_epilogues:
            formed.append(item.receipt(
                self.source_indices,
                self.deferred_branch_producers.get(item.residual_add, ())))
        return {
            "schema": "target_neutral_residual_epilogue_formations_v1",
            "formed": formed,
            "refused": [
                {"source_op_index": self.source_indices[item.operation],
                 "reason": item.reason}
                for item in self.residual_refusals
            ],
            "formed_site_count": len(formed),
            "selected_site_count": len(self.selected_residual_epilogues),
            "selected_residual_add_source_op_indices": [
                self.source_indices[item.residual_add]
                for item in self.selected_residual_epilogues],
            "selection_refusals": [
                {"residual_add_source_op_index": self.source_indices[item.residual_add],
                 "reason": "f32 value has another same-segment consumer; dual-write is not sufficient"}
                for item in self.residual_epilogues
                if item not in self.selected_residual_epilogues],
            "deferred_branch_count": sum(
                len(row["deferred_branch_producers"]) for row in formed),
            "selected_lowering": "single_exact_host_loop_i8_with_optional_live_f32_write",
            "accelerator_semantics_changed": False,
        }

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
        self.prepare_residual_schedule(ops, mesh_ops)
        self.absorbed_by = _exclusive_absorbed(mesh_ops)
        self._select_exact_native_epilogues(mesh_ops)
        for producer, formation in self.native_epilogues.items():
            owned = self._exclusive_epilogue_owners(producer, formation)
            prior = self.absorbed_by.setdefault(producer, [])
            self.absorbed_by[producer] = sorted(
                set(prior) | owned,
                key=lambda item: list(item.parent_block().ops).index(item))
        absorbed = {item for items in self.absorbed_by.values() for item in items}
        # Source-native host loops no longer expand in proportion to tensor size.

        # interface tensors: the entry's arguments, then its results, in declaration order
        rhs_values = {_integer_contraction(op).rhs for op in mesh_ops}
        returned_arguments = set(self.block.args) & set(result_values)
        for i, arg in enumerate(self.block.args):
            # A directly returned argument has one physical in/out buffer.  Declaring both argN and
            # YN leaves the former materialized but unreachable after the SSA-to-buffer map is updated.
            if arg in returned_arguments:
                continue
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

        segments: list[HostSegment] = []
        pending: list[Operation] = []
        mesh_set = set(id(op) for op in mesh_ops)
        schedule: list[Any] = []
        for op in ops:
            if op in absorbed:
                continue
            if op in self.deferred_residual_ops:
                continue
            if id(op) in mesh_set:
                schedule.append(("host", pending))
                pending = []
                schedule.append(("mesh", op))
            else:
                self.append_scheduled_host_op(pending, op)
        schedule.append(("host", pending))

        # Which values each host run must LEAVE in DRAM: whatever a later mesh op reads.
        needed_by_mesh: dict[SSAValue, None] = {}
        for op in mesh_ops:
            spec = _integer_contraction(op)
            needed_by_mesh[spec.lhs] = None
            needed_by_mesh[spec.rhs] = None

        # Tensor storage is reusable after a host run.  Consequently a tensor used by any later
        # run must cross the boundary through an explicit DRAM buffer; only scalar SSA values may
        # stay live in registers across accelerator work.  This is a lifetime rule, independent of
        # operation family, tensor rank, model, and target geometry.
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
                    name = self.buffer_for(value)
                    outputs.append((value, name))
                    self.host_tensor_spills[name] = value
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
                # Scalar regions may capture source tensors without listing them as
                # linalg inputs (e.g. embedding gathers). Bind those actual source
                # operands too, but not region-local scalar block arguments/results.
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
            output_values = {value for value, _ in outputs}
            residual_fusions = [
                item for item in self.residual_epilogues
                if (item.residual_add in host_set
                    and item.output in output_values
                    and (item.exposed_float in output_values
                         or all(use.operation is item.quantize
                                for use in item.exposed_float.uses)))]
            self.selected_residual_epilogues.extend(residual_fusions)
            segments.append(HostSegment(
                host_ops, inputs, outputs,
                sorted({_region_id(op) for op in host_ops if _region_id(op)}),
                residual_fusions))
            segments[-1].source_op_indices = [self.source_indices[op] for op in host_ops]
            self.ordered.append(segments[-1])

        # a mesh result that is the entry's own result needs no host store; one that is not is
        # read back by the segment after it, which the binding loop above already wired.
        cb = self._command_buffer(segments)
        return Plan("gemmini", self.buffers, self.ordered, cb,
                    kernel_args(cb, list(self.order)))

    def _select_exact_native_epilogues(self, mesh_ops: list[Operation]) -> None:
        """Prefer universal proof; consult an explicit held-out budget only when opted in."""
        for producer in mesh_ops:
            native_aligned = recognize_native_aligned_epilogue(producer)
            if native_aligned is not None:
                self.native_epilogues[producer] = native_aligned
                continue
            result = recognize_quantized_epilogue(producer)
            if isinstance(result, EpilogueRefusal):
                row = result.receipt(self.source_indices)
                row["stage"] = "structural_formation"
                self.epilogue_refusals.append(row)
                continue
            if not isinstance(result, QuantizedEpilogue):
                continue
            eligible, reason = exact_identity_narrowing(result)
            if not eligible:
                source_contract = contract_from_formation(result)
                decision = evaluate_accuracy_gate(
                    self.requantization_policy,
                    normalized_source_sha256=self.normalized_source_sha256,
                    producer_source_op_index=self.source_indices[producer],
                    source_contract=source_contract,
                    capability=NATIVE_NARROW_CAPABILITY,
                )
                if not decision.selected:
                    row = {
                        "producer_source_op_index": self.source_indices[producer],
                        "reason": (reason if self.requantization_policy is None
                                   else decision.reason),
                        "stage": ("exact_native_equivalence" if self.requantization_policy is None
                                  else "accuracy_bounded_requantization"),
                    }
                    if self.requantization_policy is not None:
                        row["universal_exact_proof_refusal"] = reason
                        row["accuracy_gate"] = dict(decision.receipt)
                    self.epilogue_refusals.append(row)
                    continue
                self.native_epilogue_accuracy[producer] = decision
            self.native_epilogues[producer] = result

    def _native_requant_scale(self, producer: Operation) -> float:
        formation = self.native_epilogues.get(producer)
        if isinstance(formation, NativeAlignedEpilogue):
            return formation.multiplier
        decision = self.native_epilogue_accuracy.get(producer)
        if decision is None:
            return 1.0
        if decision.candidate is None or decision.candidate.scale_granularity != "per_tensor":
            raise LoweringDeclined(
                "selected accuracy candidate exceeds target scalar-scale adapter",
                op="requantization")
        return float(decision.candidate.scales[0])

    def _native_requant_stages(self, producer: Operation,
                               formation: QuantizedEpilogue | NativeAlignedEpilogue) -> list[str]:
        stages = ["bias"] if isinstance(formation, NativeAlignedEpilogue) else []
        if self._native_requant_scale(producer) != 1.0:
            stages.append("acc_scale")
        if formation.relu:
            stages.append("relu")
        return stages

    def accelerator_region_receipt(self, segments: list[HostSegment]) -> dict[str, Any]:
        """Form maximal dependency-connected accelerator regions after all legal admissions.

        This is deliberately post-selection and graph driven.  It does not use source/model
        names, and it cannot join across a host task. A real producer/consumer dependency remains
        internal only when it names one selected buffer/encoding; an adapter is a distinct buffer
        and therefore remains explicit. Since each accuracy candidate has passed an independent
        site budget and narrowing only removes host work/storage, admitting every passing site
        maximizes accelerator coverage without a competing capacity cost in this lowering.
        """
        def io(task):
            if isinstance(task, HostSegment):
                return ([name for _, name in task.inputs], [name for _, name in task.outputs])
            if hasattr(task, "activation"):
                return ([task.activation, task.weight]
                        + ([task.bias] if getattr(task, "bias", None) else []), [task.dst])
            return ([task.lhs, task.rhs]
                    + ([task.epilogue.bias] if task.epilogue.bias else []), [task.dst])

        def encoding(name: str):
            value = self.buffers[name].storage_encoding
            return value if value is not None else {
                "logical_shape": list(self.buffers[name].shape),
                "dtype": self.buffers[name].dtype,
                "layout": "logical_row_major",
            }

        bounded_indices = {
            self.source_indices[producer] for producer in self.native_epilogue_accuracy}
        exact_narrow_indices = {
            self.source_indices[producer] for producer in self.native_epilogues
            if producer not in self.native_epilogue_accuracy}
        consumers: dict[str, int] = {}
        for task in self.ordered:
            for name in io(task)[0]:
                consumers[name] = consumers.get(name, 0) + 1

        regions: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        prior_writes: list[str] = []
        residual_host_ops = []
        region_by_id = {region.region_id: region for region in self.wl.regions}
        accelerator_task_count = bounded_task_count = 0
        for task_index, task in enumerate(self.ordered):
            reads, writes = io(task)
            if isinstance(task, HostSegment):
                current = None
                prior_writes = []
                for operation in task.ops:
                    source_index = self.source_indices[operation]
                    region_id = _region_id(operation)
                    placement = region_by_id.get(region_id)
                    matching_refusals = [
                        row for row in self.epilogue_refusals
                        if row.get("producer_source_op_index") == source_index
                        or row.get("blocker_source_op_index") == source_index]
                    residual_host_ops.append({
                        "source_op_index": source_index,
                        "op": operation.name,
                        "region": region_id or None,
                        "reason": (placement.reason if placement is not None
                                   else "not_an_accelerator_semantic_operation"),
                        "quantized_epilogue_refusals": matching_refusals,
                    })
                continue

            accelerator_task_count += 1
            task_sources = list(task.source_op_indices)
            bounded = any(index in bounded_indices for index in task_sources)
            exact_narrow = any(index in exact_narrow_indices for index in task_sources)
            bounded_task_count += int(bounded)
            shared = sorted(set(prior_writes).intersection(reads))
            compatible = [name for name in shared if name in self.buffers]
            # A buffer name denotes exactly one dtype and selected physical encoding.  An inserted
            # conversion has distinct source/target buffers, so it cannot appear in ``shared`` and
            # therefore breaks the region.  This avoids inferring compatibility from equal shapes.
            join = current is not None and compatible == shared
            if not join:
                current = {
                    "region_index": len(regions),
                    "maximal": True,
                    "task_indices": [],
                    "source_op_indices": [],
                    "internal_boundaries": [],
                    "numeric_admission": [],
                }
                regions.append(current)
            else:
                current["internal_boundaries"].extend({
                    "tensor": name,
                    "dtype": self.buffers[name].dtype,
                    "storage_encoding": encoding(name),
                    "fanout": consumers.get(name, 0),
                    "materialized_host_boundary": False,
                } for name in shared)
            current["task_indices"].append(task_index)
            current["source_op_indices"].extend(task_sources)
            current["numeric_admission"].append(
                "heldout_accuracy_bounded" if bounded else
                "universal_exact_narrow" if exact_narrow else "source_exact_accumulator")
            prior_writes = writes

        for region in regions:
            region["source_op_indices"] = sorted(set(region["source_op_indices"]))
            region["numeric_admission"] = sorted(set(region["numeric_admission"]))
        return {
            "schema": "target_neutral_maximal_accelerator_regions_v1",
            "selection_objective": [
                "maximize_legally_admitted_accelerator_source_operations",
                "maximize_direct_encoding_compatible_accelerator_boundaries",
                "minimize_residual_host_operations",
            ],
            "selection_scope": "whole_entry_graph_post_fusion",
            "model_or_shape_rules": False,
            "accelerator_task_count": accelerator_task_count,
            "accuracy_bounded_task_count": bounded_task_count,
            "maximal_region_count": len(regions),
            "regions": regions,
            "residual_host_operation_count": len(residual_host_ops),
            "residual_host_operations": residual_host_ops,
            "global_refusals": list(self.epilogue_refusals),
        }

    def _exclusive_epilogue_owners(
            self, producer: Operation,
            formation: QuantizedEpilogue | NativeAlignedEpilogue) -> set[Operation]:
        """Return the complete, single-use source slice erased by native formation.

        Besides the four linalg stages, the slice owns their tensor.empty initializers,
        scalar splats/constants, and reciprocal computation.  Backward closure is admitted only
        when every live use is already inside the slice, preventing a constant or derived tensor
        shared with another branch from disappearing.
        """
        owned = set(formation.operations)
        changed = True
        while changed:
            changed = False
            for operation in tuple(owned):
                for operand in operation.operands:
                    candidate = operand.owner if isinstance(operand.owner, Operation) else None
                    if (candidate is None or candidate is producer or candidate in owned
                            or candidate.parent_block() is not self.block):
                        continue
                    if all(use.operation in owned for value in candidate.results for use in value.uses):
                        owned.add(candidate)
                        changed = True
        return owned

    # -- one mesh contraction ----------------------------------------------------------------
    def _mesh(self, op: Operation, in_dram: set[SSAValue]) -> None:
        from .plan import Contraction

        spec = _integer_contraction(op)
        if spec is None or not _block_diagonal_is_safe(op, spec):
            raise LoweringDeclined("integer contraction geometry changed after placement",
                                   op="model_lane")
        lhs_v, rhs_v = spec.lhs, spec.rhs
        formation = self.native_epilogues.get(op)
        native_scale = self._native_requant_scale(op) if formation is not None else 1.0
        native_stages = (self._native_requant_stages(op, formation)
                         if formation is not None else [])
        out_v = formation.output if formation is not None else spec.output
        native_bias_v = formation.bias if isinstance(formation, NativeAlignedEpilogue) else None
        for operand in (lhs_v, rhs_v, *([native_bias_v] if native_bias_v is not None else [])):
            if operand not in in_dram:
                raise LoweringDeclined(
                    "a mesh contraction reads an operand no earlier segment left in DRAM",
                    op="model_lane")
        lhs = self.of_value.get(lhs_v) or self.buffer_for(lhs_v)
        rhs = self.of_value.get(rhs_v) or self.buffer_for(rhs_v)
        native_bias = ((self.of_value.get(native_bias_v)
                        or self.buffer_for(native_bias_v, "bias"))
                       if native_bias_v is not None else None)
        dst = self.of_value.get(out_v) or self.buffer_for(out_v)
        m, k, n = spec.batch * spec.m, spec.k, spec.n
        if spec.block_diagonal:
            k = spec.batch * spec.k
            # Both leaves keep identical row-major bytes under these views.  The lhs's derived
            # block diagonal is the same generic im2col construction used by PlanBuilder.
            self.buffers[lhs].shape = [1, spec.batch, spec.m, spec.k]
            self.buffers[rhs].shape = [k, n]
            block_lhs = self.declare(f"{dst}_lhs_batchdiag", [m, k],
                                     self.buffers[lhs].dtype, "input")
            self.im2col_recipes.append(
                {"source": lhs, "target": block_lhs, "kh": spec.batch, "kw": 1,
                 "ci": spec.k, "stride": [spec.batch + 1, 1],
                 "padding": [0, 0, 0, 0], "dilation": [-spec.batch, 1],
                 "layout": "nhwc"})
            contraction_lhs = block_lhs
        else:
            contraction_lhs = lhs
        if spec.batch_shape:
            self.batched_contractions.append(
                {"lhs": lhs, "rhs": rhs, "dst": dst, "batch_shape": list(spec.batch_shape),
                 "batch": spec.batch, "m": spec.m, "k": spec.k, "n": spec.n,
                 "strategy": "block_diagonal" if spec.block_diagonal else "flatten_shared_rhs",
                 **({"block_diagonal_lhs": contraction_lhs}
                    if spec.block_diagonal else {})})
        out_dtype = _elem(out_v.type)
        # Integer tensor arithmetic wraps at its declared width. Native narrowed
        # accelerator readout saturates, so preserve a full accumulator and let
        # submitted host code truncate it. This policy follows source types.
        modular = formation is None and int(out_dtype[1:]) < int(F.ACCUMULATOR_DTYPE[1:])
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
                              "operands": {"lhs": contraction_lhs,
                                           "rhs": handle, "dst": acc}})
        command_attributes = {"epilogue": native_stages,
                              "output_dtype": F.ACCUMULATOR_DTYPE if modular else out_dtype}
        if "acc_scale" in native_stages:
            command_attributes["acc_scale"] = native_scale
        if native_bias is not None:
            command_attributes["bias"] = native_bias
        self.commands.append({"opcode": "COMMIT",
                              "operands": {"src": acc, "dst": temporary or dst},
                              "attributes": command_attributes})
        self.ordered.append(
            Contraction(contraction_lhs, rhs, dst, m, k, n,
                        lhs_row_elems=k, rhs_row_elems=n,
                        epilogue=Epilogue(
                            stages=native_stages,
                            output_dtype=out_dtype, acc_scale=native_scale,
                            bias=native_bias,
                            integer_output_policy=("saturate" if formation is not None
                                                   else "modular")),
                        accumulator_temporary=temporary))
        if formation is not None:
            self.of_value[spec.output] = dst
            self.of_value[formation.output] = dst
        self.ordered[-1].source_op_indices = [
            *[self.source_indices[item] for item in self.absorbed_by.get(op, [])],
            self.source_indices[op],
        ]

    # -- serialisation -----------------------------------------------------------------------
    def _command_buffer(self, segments: list[HostSegment]) -> dict[str, Any]:
        # Bind host inputs explicitly; resident packs describe accelerator work only.
        commands = self.commands
        tensors = {n: {"shape": self.buffers[n].shape, "dtype": self.buffers[n].dtype,
                       "role": self.buffers[n].role} for n in self.order}
        entry_bindings = [self.of_value[arg] for arg in self.block.args]
        output_bindings = [n for n in self.order if self.buffers[n].role == "output"]
        placement = [{"region": r.region_id, "family": r.family, "op": r.op, "dtype": r.dtype,
                      "lane": r.lane, "reason": r.reason} for r in self.wl.regions]
        return {
            "abi_version": "0.1",
            "target": "gemmini",
            "backend": "mlir_oot_xdsl_gemmini",
            "tensors": tensors,
            "kernel_abi": {
                "kind": "whole_program",
                "args": [{"tensor": n,
                          "access": ("readwrite" if n in entry_bindings and n in output_bindings
                                     else "write" if self.buffers[n].role
                                     in ("output", "intermediate") else "read")}
                         for n in self.order],
                "outputs": output_bindings,
            },
            "commands": commands,
            "params": {
                **({"im2col_recipes": list(self.im2col_recipes)}
                   if self.im2col_recipes else {}),
                **({"batched_contractions": list(self.batched_contractions)}
                   if self.batched_contractions else {}),
                "global_program_plan": {
                    "schema": "mixed_program_plan_v1",
                    "source_op_count": len(self.source_indices),
                    "compiler_temporaries": [
                        {"tensor": task.accumulator_temporary,
                         "source_op_index": task.source_op_indices[0],
                         "source_result_index": 0, "purpose": "accumulator_readout"}
                        for task in self.ordered if not isinstance(task, HostSegment)
                        and task.accumulator_temporary],
                    "host_tensor_spills": [
                        {"tensor": name,
                         "source_op_index": self.source_indices[value.owner],
                         "source_result_index": value.index,
                         "bytes": self.buffers[name].nbytes}
                        for name, value in self.host_tensor_spills.items()],
                    "tasks": [
                        {"task_index": i,
                         "kind": "host" if isinstance(task, HostSegment) else "contraction",
                         "source_op_indices": list(task.source_op_indices),
                         "reads": [n for _, n in task.inputs] if isinstance(task, HostSegment)
                                  else ([task.lhs, task.rhs]
                                        + ([task.epilogue.bias]
                                           if task.epilogue.bias else [])),
                         "writes": [n for _, n in task.outputs] if isinstance(task, HostSegment)
                                   else [task.dst] + ([task.accumulator_temporary]
                                                     if task.accumulator_temporary else []),
                         **({} if isinstance(task, HostSegment) else {
                             "integer_output_policy": task.epilogue.integer_output_policy,
                             "accumulator_temporary": task.accumulator_temporary})}
                        for i, task in enumerate(self.ordered)],
                    "source_values": [
                        {"op_index": self.source_indices[value.owner],
                         "result_index": value.index, "tensor": name}
                        for value, name in self.of_value.items()
                        if isinstance(value.owner, Operation) and value.owner in self.source_indices],
                    "entry_bindings": entry_bindings,
                    "output_bindings": output_bindings,
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
                     "writes": [n for _, n in seg.outputs],
                     "residual_fusion_sites": [
                         self.source_indices[item.residual_add]
                         for item in seg.residual_fusions]}
                    for seg in segments],
                "target_neutral_quantized_epilogues": self._epilogue_receipt(),
                "target_neutral_residual_epilogues": self.residual_receipt(),
                "maximal_accelerator_regions": self.accelerator_region_receipt(segments),
            },
        }

    def _epilogue_receipt(self) -> dict[str, Any]:
        formed = []
        boundary_bytes = 0
        source_ops = 0
        for producer, formation in self.native_epilogues.items():
            full_width_bytes = prod(formation.shape) * 4
            boundary_bytes += full_width_bytes
            source_ops += len(self.absorbed_by.get(producer, ()))
            row = formation.receipt(self.source_indices)
            source_contract = (
                {"schema": "native_aligned_i32_epilogue_contract_v1",
                 "bias_domain": "accumulator_i32",
                 "scale_granularity": "per_tensor",
                 "scales": [formation.multiplier],
                 "rounding": "round_to_nearest_even",
                 "saturation": [-128, 127],
                 "output_zero_point": 0}
                if isinstance(formation, NativeAlignedEpilogue)
                else contract_from_formation(formation).receipt())
            decision = self.native_epilogue_accuracy.get(producer)
            row.update({
                "semantic_contract": source_contract,
                "selected_lowering": (
                    "native_i32_bias_scalar_scale_i8_readout"
                    if isinstance(formation, NativeAlignedEpilogue)
                    else "native_identity_scale_i8_readout" if decision is None
                    else "accuracy_bounded_native_narrow_i8_readout"),
                "selection_proof": ("canonical_source_semantics"
                                    if isinstance(formation, NativeAlignedEpilogue)
                                    else "universal_exact" if decision is None
                                    else "empirical_heldout_accuracy_budget"),
                **({} if decision is None else {"accuracy_gate": dict(decision.receipt)}),
                "logical_device_to_host_boundary_eliminated": 1,
                "full_width_intermediate_bytes_eliminated": full_width_bytes,
                "output_dma_bytes_before": full_width_bytes,
                "output_dma_bytes_after": full_width_bytes // 4,
                "output_dma_bytes_eliminated": full_width_bytes * 3 // 4,
            })
            formed.append(row)
        accuracy_metrics = [
            decision.receipt["metrics"] for decision in self.native_epilogue_accuracy.values()]
        accuracy_samples = sum(row["sample_count"] for row in accuracy_metrics)
        accuracy_mismatches = sum(row["mismatch_count"] for row in accuracy_metrics)
        saturation_mismatches = sum(
            row["saturation_mismatch_count"] for row in accuracy_metrics)
        accuracy_summary = {
            "heldout_sample_count": accuracy_samples,
            "maximum_site_abs_error_lsb": max(
                (row["max_abs_error_lsb"] for row in accuracy_metrics), default=0),
            "weighted_mean_abs_error_lsb": (
                sum(row["mean_abs_error_lsb"] * row["sample_count"]
                    for row in accuracy_metrics) / accuracy_samples
                if accuracy_samples else 0.0),
            "aggregate_mismatch_fraction": (
                accuracy_mismatches / accuracy_samples if accuracy_samples else 0.0),
            "aggregate_saturation_mismatch_fraction": (
                saturation_mismatches / accuracy_samples if accuracy_samples else 0.0),
        }
        return {
            "schema": "target_neutral_quantized_epilogue_lowering_v2",
            "formed": formed,
            "refused": list(self.epilogue_refusals),
            "selected_count": len(formed),
            "source_ops_absorbed": source_ops,
            "full_width_intermediate_bytes_eliminated": boundary_bytes,
            "logical_device_to_host_boundaries_eliminated": len(formed),
            "output_dma_bytes_eliminated": boundary_bytes * 3 // 4,
            # Physical fence count is a schedule-level fact.  A terminal result still needs the
            # final completion fence, so formation alone must never claim one disappeared.
            "physical_fences_eliminated": 0,
            "numeric_contract": ("calibration_bounded_opt_in" if self.native_epilogue_accuracy
                                 else "exact_for_every_i32_accumulator"),
            "approximation": bool(self.native_epilogue_accuracy),
            "accuracy_bounded_selected_count": len(self.native_epilogue_accuracy),
            "accuracy_metrics": accuracy_summary,
            "accuracy_policy_sha256": (None if self.requantization_policy is None
                                       else self.requantization_policy.policy_sha256),
        }


def build(module, wl: LinalgWorkload, *, requantization_policy: AccuracyPolicy | None = None,
          normalized_source_sha256: str = "") -> Plan:
    return MixedBuilder(
        module, wl, requantization_policy=requantization_policy,
        normalized_source_sha256=normalized_source_sha256).build()
