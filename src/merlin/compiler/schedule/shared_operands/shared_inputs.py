"""Find closed source initialization groups without selecting a target schedule.

An exact zero splat used only as a signed contraction's seed can remain virtual
when the target independently realizes zero accumulation and complete readout.
These source proposals grant no hardware, ownership or numerical authority.
The selected shared-library releases and their import rosters stay unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from io import StringIO

from xdsl.dialects import arith, builtin, func
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.printer import Printer

from merlin.compiler.schedule.blocks import Compute, Load, Preload, Store
from merlin.compiler.schedule.dense.compact import DenseLoopStorage

# ---- zero_initializer_groups -------------------------------------------------------------------


def _digest(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def _name(operation):
    return getattr(operation.attributes.get("op_name__"), "data", operation.name)


def _provenance_only(operation, *, properties=()):
    return all(name.startswith("prov.") or name == "op_name__" for name in operation.attributes) and not set(
        operation.properties
    ) - set(properties)


@dataclass(frozen=True)
class ZeroInitializerGroup:
    """Current source operations that one checked device task may own together."""

    source_ir_sha256: str
    source_op_indices: tuple[int, ...]
    contraction_ordinal: int
    initializer_ordinal: int
    virtual_result_positions: tuple[tuple[int, int], ...]
    geometry_m_k_n: tuple[int, int, int]
    initializer_elements: int
    initializer_bytes: int

    def record(self):
        return {
            "schema": "merlin.zero_initializer_group.v1",
            **asdict(self),
            "required_target_realization": [
                "exact zero accumulator initialization",
                "complete original signed products and accumulator continuation",
                "complete source output readout and completion",
                "no physical read of the virtual initializer",
                "original external buffer ownership and alias checks",
            ],
            "cost_scope": "logical host materialization avoided; not physical traffic or cycles",
            "authority": "source grouping proposal only; original final execution checks required",
        }

    def verify_current(self, module, *, max_source_operations=10_000, max_source_elements=64 << 20):
        if self not in find_zero_initializer_groups(
            module, max_source_operations=max_source_operations, max_source_elements=max_source_elements
        ):
            raise ValueError("zero initializer grouping differs from the complete current typed source")


def find_zero_initializer_groups(module, *, max_source_operations=10_000, max_source_elements=64 << 20):
    """Propose contiguous one-use zero splat and exact signed rank-two MAC groups.

    The source remains unchanged, including every scalar operation and result.
    A consumer must keep full source-op/task accounting while omitting virtual
    storage, and independently check its actual initializer/readout realization.
    Shared constants may stay in another task; shared or returned splats cannot
    disappear. Unknown scalar, layout, aliasing or arithmetic semantics retain
    the original route. No model identity or provenance label selects a group.
    """
    from merlin.compiler.transforms.dense.ir import recognize_signed_integer_matmul

    if type(module) is not builtin.ModuleOp:
        raise ValueError("zero initializer grouping requires the current original typed module")
    if any(type(value) is not int or value <= 0 for value in (max_source_operations, max_source_elements)):
        raise ValueError("zero initializer grouping requires explicit positive source budgets")
    functions = [op for op in module.body.block.ops if type(op) is func.FuncOp]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        return ()
    block = functions[0].body.block
    operations = tuple(block.ops)
    if len(operations) > max_source_operations:
        raise ValueError("zero initializer source operation budget exceeded")
    module.verify()
    if not operations or type(operations[-1]) is not func.ReturnOp:
        return ()
    if any(op.successors or type(op) is func.ReturnOp for op in operations[:-1]):
        return ()
    ordinals = {op: index for index, op in enumerate(operations[:-1])}
    groups, source_digest = [], None
    for ordinal, operation in enumerate(operations[:-1]):
        if not isinstance(operation, GenericOp):
            continue
        # Frontends may register a parser-only subclass of the standard generic
        # operation. Reconstruct its complete standard shell before applying
        # the strict recognizer; preserve all fields and unknown attributes.
        detached = GenericOp.create(
            operands=operation.operands,
            result_types=[value.type for value in operation.results],
            attributes=dict(operation.attributes),
            properties=dict(operation.properties),
            successors=operation.successors,
            regions=[region.clone() for region in operation.regions],
        )
        for inner in detached.walk():
            for key in tuple(inner.attributes):
                if key.startswith("prov."):
                    del inner.attributes[key]
        try:
            recognize_signed_integer_matmul(detached, ("i8", "i8", "i32"))
        except ValueError:
            continue
        finally:
            # Cloned operations temporarily add uses to their external SSA
            # operands. Restore those exact lists before testing single use.
            detached.drop_all_references()
        lhs, rhs, initializer = operation.operands
        shapes = [tuple(value.type.get_shape()) for value in (lhs, rhs, initializer)]
        if any(len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes):
            continue
        (m, k), (rhs_k, n), output = shapes
        if rhs_k != k or output != (m, n) or m * k + k * n + m * n > max_source_elements:
            continue
        splat = initializer.owner
        index = ordinals.get(splat)
        if (
            index != ordinal - 1
            or index is None
            or _name(splat) != "tensor.splat"
            or len(splat.operands) != 1
            or tuple(splat.results) != (initializer,)
            or splat.regions
            or splat.successors
            or not _provenance_only(splat)
            or len(tuple(initializer.uses)) != 1
            or next(iter(initializer.uses)).operation is not operation
        ):
            continue
        scalar = splat.operands[0]
        constant = scalar.owner
        if (
            type(constant) is not arith.ConstantOp
            or constant.parent is not block
            or ordinals.get(constant, ordinal) >= index
            or tuple(constant.results) != (scalar,)
            or type(constant.value) is not builtin.IntegerAttr
            or constant.value.value.data != 0
            or scalar.type != initializer.type.element_type
            or not _provenance_only(constant, properties=("value",))
        ):
            continue
        indices = (index, ordinal)
        virtual = ((index, 0),)
        if ordinals[constant] == index - 1 and len(tuple(scalar.uses)) == 1:
            indices = (index - 1, *indices)
            virtual = ((index - 1, 0), *virtual)
        if source_digest is None:
            source_digest = _digest(module)
        groups.append(
            ZeroInitializerGroup(source_digest, indices, ordinal, index, virtual, (m, k, n), m * n, m * n * 4)
        )
    return tuple(groups)


# ---- shared_input_groups -----------------------------------------------------------------------
# Recognize closed sibling contractions of one immutable typed tensor value.
#
# These source proposals preserve every operation and each independent result.
# They establish no physical aliases, target instruction effects or timing.


@dataclass(frozen=True)
class SharedInputGroup:
    source_ir_sha256: str
    source_op_indices: tuple[int, ...]
    contraction_ordinals: tuple[int, ...]
    m: int
    k: int
    widths: tuple[int, ...]

    def record(self):
        return {
            "schema": "merlin.shared_input_group.v1",
            **asdict(self),
            "source_semantics": "immutable tensor SSA; separate zero-seeded signed modular contractions",
            "result_boundary": "every original contraction result and later epilogue remains independent",
            "authority": "source proposal only; physical ownership and actual target effects remain required",
        }

    def verify_current(self, module, *, max_source_operations=10_000, max_source_elements=64 << 20):
        groups = find_shared_input_groups(
            module, max_source_operations=max_source_operations, max_source_elements=max_source_elements
        )
        if not any(self == prefix for group in groups for prefix in grow_shared_input_group(group)):
            raise ValueError("shared-input group differs from complete current typed source")


def grow_shared_input_group(group):
    """Yield consecutive prefixes, adding one sibling at a time; never subsets.

    Each prefix retains its own exact zero initializers and output identities.
    Unselected siblings and external consumers stay in the original module.
    ``verify_current`` rederives the maximal group before accepting a prefix.
    """
    from dataclasses import replace

    for count in range(2, len(group.widths) + 1):
        last = group.contraction_ordinals[count - 1]
        yield replace(
            group,
            source_op_indices=tuple(i for i in group.source_op_indices if i <= last),
            contraction_ordinals=group.contraction_ordinals[:count],
            widths=group.widths[:count],
        )


def find_shared_input_groups(module, *, max_source_operations=10_000, max_source_elements=64 << 20):
    """Group adjacent checked zero-seed regions with the identical LHS SSA value.

    No operation moves across unknown effects or an intervening consumer. The
    existing strict source checker establishes signed products, increasing K,
    ordinary tensor layouts and one-use exact zero initializers. Only source
    zero/initializer/contraction operations may occur within a group. Grouping
    does not erase them or change any use list, including later epilogues.
    """
    seeds = find_zero_initializer_groups(
        module, max_source_operations=max_source_operations, max_source_elements=max_source_elements
    )
    if not seeds:
        return ()
    # The original checker already validates the single-function/block shell.
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    operations = tuple(function.body.block.ops)
    groups, run, covered, shared = [], [], set(), None

    def finish():
        if len(run) < 2:
            return
        first = run[0]
        groups.append(
            SharedInputGroup(
                first.source_ir_sha256,
                tuple(sorted(covered)),
                tuple(seed.contraction_ordinal for seed in run),
                first.geometry_m_k_n[0],
                first.geometry_m_k_n[1],
                tuple(seed.geometry_m_k_n[2] for seed in run),
            )
        )

    for seed in seeds:
        lhs = operations[seed.contraction_ordinal].operands[0]
        # The source recognizer requires immutable, unencoded tensor SSA. A
        # repeated name or equal shape never substitutes for value identity.
        assert isinstance(lhs.type, builtin.TensorType)
        new = set(seed.source_op_indices)
        contiguous = not run or min(new) == max(covered) + 1
        if run and (lhs is not shared or not contiguous):
            finish()
            run, covered = [], set()
        shared = lhs
        run.append(seed)
        covered.update(new)
    finish()
    return tuple(groups)


# ---- shared_input_schedule ---------------------------------------------------------------------
# Retain one full-reduction input stripe across independent sibling outputs.
#
# The source supplies immutable tensor value identity, exact products and zeros;
# the caller selects primitive dimensions and already assigned private storage.
# No target instruction, completion rule or physical source/output alias proof
# is supplied. Actual target projection must pass the independent checker.


@dataclass(frozen=True)
class SharedInputEffect:
    """Actual logical effect, with distinct RHS/output ownership per sibling.

    ``member=None`` is reserved for reads of the common immutable input.
    A member index selects the source contraction ordinal in the checked group.
    """

    member: int | None
    operation: Load | Preload | Compute | Store


@dataclass(frozen=True)
class SharedInputBlock:
    m: int
    k: int
    n: int
    operand_bytes: int
    accumulator_bytes: int


@dataclass(frozen=True)
class SharedInputSchedule:
    group: SharedInputGroup
    primitive_tile: tuple[int, int, int]
    block: SharedInputBlock
    operations: tuple[SharedInputEffect, ...]
    candidates_considered: int
    realization: dict


def _ceil(value, unit):
    return (value + unit - 1) // unit


def _storage(problem, tile, block, storage, resources):
    """Check all assigned intervals, including padding and shared backings."""
    if type(storage) is not DenseLoopStorage or storage.accumulator is None:
        return False
    tm, tk, tn = tile
    kt = _ceil(problem.k, tk)
    views = storage.lhs, storage.rhs, storage.accumulator
    rows = block.m * kt * tm, block.k * block.n * tk, block.m * block.n * tm
    widths = tk, tn, tn * 4  # Derived from the checked source's i8/i8/i32 formats.
    if any(
        count > view.capacity or width > view.unit_bytes or stride % view.alignment
        for count, width, stride, view in zip(rows, widths, (tm, tk, tm), views)
    ):
        return False
    used = tuple(view.unit_bytes * count for view, count in zip(views, rows))
    if used[0] + used[1] > resources.operand_bytes or used[2] > resources.accumulator_bytes:
        return False
    intervals = [(v.backing, v.base * v.unit_bytes, (v.base + v.capacity) * v.unit_bytes) for v in views]
    return not any(
        a[0] == b[0] and a[1] < b[2] and b[1] < a[2]
        for index, a in enumerate(intervals)
        for b in intervals[index + 1 :]
    )


def schedule_shared_input(
    module,
    group,
    *,
    primitive_tile,
    resources,
    storage,
    selected_block=None,
    max_commands=1_000_000,
    max_rows_visited=16 << 20,
    max_source_operations=10_000,
    max_source_elements=64 << 20,
):
    """Construct and check a bounded raw-core sibling schedule, or decline.

    All K tiles of an M panel stay in LHS storage while distinct RHS panels
    stream through a bounded output wave. Each output retains increasing K,
    its own exact zero seed and its original result identity. Later epilogues
    remain separate. No weight/output concatenation or host packing is needed.

    ``selected_block=(M,K,N)`` exposes a legal choice to an external tuner.
    Without it, requested logical bytes, command count and storage break ties.
    Counts are not physical traffic/cycles. This finite family declines when
    a full-K LHS stripe does not fit; it emits no compact machine loop or fence.
    """
    from merlin.compiler.transforms.dense.planning import DenseResources

    if type(group) is not SharedInputGroup or type(resources) is not DenseResources:
        raise ValueError("shared-input scheduling requires a current source group and selected resources")
    group.verify_current(module, max_source_operations=max_source_operations, max_source_elements=max_source_elements)
    if (
        type(primitive_tile) is not tuple
        or len(primitive_tile) != 3
        or any(type(v) is not int or v <= 0 for v in primitive_tile)
        or any(type(v) is not int or v <= 0 for v in (max_commands, max_rows_visited))
    ):
        raise ValueError("shared-input primitive and verification budgets must be positive and explicit")
    if selected_block is not None and (
        type(selected_block) is not tuple
        or len(selected_block) != 3
        or any(type(v) is not int or v <= 0 for v in selected_block)
    ):
        raise ValueError("selected sibling block must contain three positive tile counts")
    tm, tk, tn = primitive_tile
    mt, kt = _ceil(group.m, tm), _ceil(group.k, tk)
    nt = sum(_ceil(width, tn) for width in group.widths)
    if not resources.partial_output_tiles and (group.m % tm or any(n % tn for n in group.widths)):
        return None
    # Every member has its own output tile boundaries; padded virtual lanes
    # never become a source read, arithmetic product or physical output cell.
    bm_max, bn_max, bk_max = (
        min(mt, resources.max_m_blocks),
        min(nt, resources.max_n_blocks),
        min(kt, resources.max_k_blocks),
    )
    if selected_block is not None and any(
        value > maximum for value, maximum in zip(selected_block, (bm_max, bk_max, bn_max), strict=True)
    ):
        return None
    candidate_count = 1 if selected_block is not None else bm_max * bn_max * bk_max
    if candidate_count > resources.candidate_budget:
        raise ValueError("shared-input blocking search exceeds its candidate budget")
    commands = mt * kt + _ceil(mt, bm_max) * nt * kt + 2 * mt * nt * kt + mt * nt
    if commands > max_commands:
        raise ValueError("shared-input schedule exceeds its command budget before materialization")
    candidates, considered = [], 0
    # An explicit alternative tests only its selected block, not the entire
    # search product. Native storage/effect checks below still apply unchanged.
    m_choices = (selected_block[0],) if selected_block is not None else range(1, bm_max + 1)
    n_choices = (selected_block[2],) if selected_block is not None else range(1, bn_max + 1)
    k_choices = (selected_block[1],) if selected_block is not None else range(1, bk_max + 1)
    for bm in m_choices:
        for bn in n_choices:
            for bk in k_choices:
                considered += 1
                if not resources.partial_output_blocks and (mt % bm or nt % bn):
                    continue
                lhs = bm * kt * tm * storage.lhs.unit_bytes if type(storage) is DenseLoopStorage else 0
                rhs = bk * bn * tk * storage.rhs.unit_bytes if type(storage) is DenseLoopStorage else 0
                acc = (
                    bm * bn * tm * storage.accumulator.unit_bytes
                    if type(storage) is DenseLoopStorage and storage.accumulator
                    else 0
                )
                block = SharedInputBlock(bm, bk, bn, lhs + rhs, acc)
                if not _storage(group, primitive_tile, block, storage, resources):
                    continue
                count = mt * kt + _ceil(mt, bm) * nt * kt + 2 * mt * nt * kt + mt * nt
                if count <= max_commands:
                    requested = group.m * group.k + _ceil(mt, bm) * group.k * sum(group.widths)
                    candidates.append(((requested, count, lhs + rhs + acc, bm, bn, bk), block))
    if not candidates:
        return None
    _, block = min(candidates, key=lambda item: item[0])
    bm, bk, bn = block.m, block.k, block.n
    lanes = tuple((member, n) for member, width in enumerate(group.widths) for n in range(_ceil(width, tn)))
    effects = []
    for m0 in range(0, mt, bm):
        for mi in range(m0, min(mt, m0 + bm)):
            for ki in range(kt):
                effects.append(
                    SharedInputEffect(
                        None,
                        Load(
                            "lhs",
                            (mi, ki),
                            mi * tm,
                            ki * tk,
                            min(tm, group.m - mi * tm),
                            min(tk, group.k - ki * tk),
                            storage.lhs.base + ((mi - m0) * kt + ki) * tm,
                            ki,
                        ),
                    )
                )
        for n0 in range(0, nt, bn):
            wave = lanes[n0 : n0 + bn]
            for k0 in range(0, kt, bk):
                for ki in range(k0, min(kt, k0 + bk)):
                    for slot, (member, ni) in enumerate(wave):
                        effects.append(
                            SharedInputEffect(
                                member,
                                Load(
                                    "weight",
                                    (ki, ni),
                                    ki * tk,
                                    ni * tn,
                                    min(tk, group.k - ki * tk),
                                    min(tn, group.widths[member] - ni * tn),
                                    storage.rhs.base + ((ki - k0) * bn + slot) * tk,
                                    ki,
                                ),
                            )
                        )
                for ki in range(k0, min(kt, k0 + bk)):
                    depth = min(tk, group.k - ki * tk)
                    for slot, (member, ni) in enumerate(wave):
                        cols = min(tn, group.widths[member] - ni * tn)
                        for mi in range(m0, min(mt, m0 + bm)):
                            rows = min(tm, group.m - mi * tm)
                            fresh = mi == m0
                            effects.append(
                                SharedInputEffect(
                                    member,
                                    Preload(
                                        storage.rhs.base + ((ki - k0) * bn + slot) * tk if fresh else None,
                                        (ki, ni) if fresh else None,
                                        storage.accumulator.base + ((mi - m0) * bn + slot) * tm,
                                        ki != 0,
                                        rows,
                                        cols,
                                        depth,
                                    ),
                                )
                            )
                            effects.append(
                                SharedInputEffect(
                                    member,
                                    Compute(
                                        storage.lhs.base + ((mi - m0) * kt + ki) * tm,
                                        (mi, ki),
                                        rows,
                                        depth,
                                        fresh,
                                    ),
                                )
                            )
            for slot, (member, ni) in enumerate(wave):
                for mi in range(m0, min(mt, m0 + bm)):
                    effects.append(
                        SharedInputEffect(
                            member,
                            Store(
                                mi * tm,
                                ni * tn,
                                min(tm, group.m - mi * tm),
                                min(tn, group.widths[member] - ni * tn),
                                storage.accumulator.base + ((mi - m0) * bn + slot) * tm,
                            ),
                        )
                    )
    effects = tuple(effects)
    realization = verify_shared_input(
        effects,
        module=module,
        group=group,
        primitive_tile=primitive_tile,
        resources=resources,
        storage=storage,
        max_commands=max_commands,
        max_rows_visited=max_rows_visited,
        max_source_operations=max_source_operations,
        max_source_elements=max_source_elements,
    )
    return SharedInputSchedule(group, primitive_tile, block, effects, considered, realization)


# ---- shared_input_verification -----------------------------------------------------------------
# Independent finite source/lifetime induction over actual sibling effects.
#
# This checks the synchronous logical effect family, not target admission or
# overlap. An adapter must decode actual emitted addresses and source bindings;
# retaining planned operations instead of projecting emitted ones is no proof.


def verify_shared_input(
    operations,
    *,
    module,
    group,
    primitive_tile,
    resources,
    storage,
    max_commands=1_000_000,
    max_rows_visited=16 << 20,
    max_source_operations=10_000,
    max_source_elements=64 << 20,
):
    """Check all source products, separate results and actual private addresses.

    Loaded row intervals carry original tensor identities. Every consumed
    rectangle is reconstructed from them, including retained array contents.
    Accumulator rows carry output ownership and the next required K position.
    This proves increasing, complete products without expanding M*K*N scalar
    products. Complete output intervals are checked independently at the end.
    """
    from merlin.compiler.transforms.dense.planning import DenseResources

    if type(group) is not SharedInputGroup or type(resources) is not DenseResources:
        raise ValueError("shared-input checking requires original source and selected resources")
    group.verify_current(module, max_source_operations=max_source_operations, max_source_elements=max_source_elements)
    if (
        type(operations) is not tuple
        or type(primitive_tile) is not tuple
        or len(primitive_tile) != 3
        or any(type(v) is not int or v <= 0 for v in (*primitive_tile, max_commands, max_rows_visited))
        or len(operations) > max_commands
        or type(storage) is not DenseLoopStorage
        or storage.accumulator is None
    ):
        raise ValueError("shared-input checking requires bounded actual effects and assigned private storage")
    tm, tk, tn = primitive_tile
    views = {"lhs": storage.lhs, "weight": storage.rhs, "accumulator": storage.accumulator}
    for role, width in (("lhs", tk), ("weight", tn), ("accumulator", 4 * tn)):
        if views[role].unit_bytes < width:
            raise ValueError("selected primitive exceeds assigned physical row width")
    intervals = [(v.backing, v.base * v.unit_bytes, (v.base + v.capacity) * v.unit_bytes) for v in views.values()]
    if any(
        a[0] == b[0] and a[1] < b[2] and b[1] < a[2]
        for index, a in enumerate(intervals)
        for b in intervals[index + 1 :]
    ):
        raise ValueError("shared-input assigned physical storage aliases")
    operands, accumulators, outputs = {}, {}, {}
    pending, stationary = None, None
    visited = products = lhs_loads = rhs_loads = requested = retained = 0
    high = dict.fromkeys(views, 0)

    def rectangle(role, row, rows, cols):
        nonlocal visited
        view = views[role]
        max_r, max_c = (tk, tn) if role == "weight" else (tm, tk if role == "lhs" else tn)
        if (
            any(type(v) is not int for v in (row, rows, cols))
            or not 0 < rows <= max_r
            or not 0 < cols <= max_c
            or row < view.base
            or row + rows > view.base + view.capacity
            or row % view.alignment
        ):
            raise ValueError("actual sibling rectangle exceeds primitive or assigned private storage")
        if not resources.partial_output_tiles and (
            (role in ("lhs", "accumulator") and rows != tm) or (role in ("weight", "accumulator") and cols != tn)
        ):
            raise ValueError("actual sibling rectangle requires unselected partial output support")
        visited += rows
        if visited > max_rows_visited:
            raise ValueError("actual sibling effects exceed independent row-checking budget")
        high[role] = max(high[role], row + rows - view.base)
        if (
            high["lhs"] * storage.lhs.unit_bytes + high["weight"] * storage.rhs.unit_bytes > resources.operand_bytes
            or high["accumulator"] * storage.accumulator.unit_bytes > resources.accumulator_bytes
        ):
            raise ValueError("actual sibling high-water marks exceed selected resource bytes")

    for effect in operations:
        if type(effect) is not SharedInputEffect:
            raise ValueError("unknown sibling effect cannot be erased")
        member, op = effect.member, effect.operation
        if member is not None and (type(member) is not int or not 0 <= member < len(group.widths)):
            raise ValueError("actual sibling effect selects unknown original result")
        if pending is not None and type(op) is not Compute:
            raise ValueError("synchronous sibling preload must have its immediate matching compute")
        if type(op) is Load:
            if op.role not in ("lhs", "weight") or (member is None) != (op.role == "lhs") or op.gather is not None:
                raise ValueError("actual sibling load has wrong source ownership or unknown gather")
            rectangle(op.role, op.row, op.rows, op.cols)
            source_rows, source_cols = (group.m, group.k) if member is None else (group.k, group.widths[member])
            if (
                any(type(v) is not int or v < 0 for v in (op.dram_row, op.dram_col))
                or op.dram_row + op.rows > source_rows
                or op.dram_col + op.cols > source_cols
            ):
                raise ValueError("actual sibling load exceeds original tensor")
            for r in range(op.rows):
                operands[op.role, op.row + r] = member, op.dram_row + r, op.dram_col, op.cols
            requested += op.rows * op.cols
            lhs_loads += member is None
            rhs_loads += member is not None
        elif type(op) is Preload:
            if member is None or type(op.accumulate) is not bool or type(op.weight_rows) is not int:
                raise ValueError("actual sibling descriptor has unknown source or arithmetic continuation")
            rectangle("accumulator", op.accumulator_row, op.rows, op.cols)
            if not 0 < op.weight_rows <= tk:
                raise ValueError("actual sibling descriptor has invalid reduction depth")
            if op.weight_row is not None:
                rectangle("weight", op.weight_row, op.weight_rows, op.cols)
                weights = tuple(operands.get(("weight", op.weight_row + r)) for r in range(op.weight_rows))
                if None in weights:
                    raise ValueError("actual sibling preload reads uninitialized weight rows")
                owner, k0, n0, width = weights[0]
                if (
                    owner != member
                    or width < op.cols
                    or any(row != (member, k0 + r, n0, width) for r, row in enumerate(weights))
                    or op.weight_block != (k0 // tk, n0 // tn)
                ):
                    raise ValueError("actual sibling weights have wrong source identity or rectangle")
                stationary = member, k0, n0, op.weight_rows, op.cols
            elif op.weight_block is not None:
                raise ValueError("retained sibling descriptor supplies an unrelated weight identity")
            if stationary is None or stationary[0] != member or stationary[3:] != (op.weight_rows, op.cols):
                raise ValueError("actual retained weights have wrong sibling ownership or geometry")
            pending = member, op, stationary
        elif type(op) is Compute:
            if (
                member is None
                or pending is None
                or member != pending[0]
                or type(op.fresh_weights) is not bool
                or op.input_gather is not None
            ):
                raise ValueError("actual sibling compute lacks its matching descriptor")
            _, descriptor, weights = pending
            if (op.rows, op.cols) != (descriptor.rows, descriptor.weight_rows) or op.fresh_weights != (
                descriptor.weight_row is not None
            ):
                raise ValueError("actual sibling compute changes primitive or retained-weight semantics")
            rectangle("lhs", op.input_row, op.rows, op.cols)
            _, k0, n0, depth, width = weights
            for r in range(op.rows):
                source = operands.get(("lhs", op.input_row + r))
                if source is None:
                    raise ValueError("actual sibling compute reads uninitialized shared input")
                owner, m, k, cols = source
                if (
                    owner is not None
                    or k != k0
                    or cols < depth
                    or op.input_block != ((m - r) // tm, k0 // tk)
                    or (r and m != operands["lhs", op.input_row][1] + r)
                ):
                    raise ValueError("actual sibling product differs from original input/weight geometry")
                address = descriptor.accumulator_row + r
                expected = member, m, n0, width, k0
                if descriptor.accumulate:
                    if k0 == 0 or accumulators.get(address) != expected:
                        raise ValueError("actual sibling continuation loses products or aliases a live output")
                elif k0 != 0:
                    raise ValueError("actual sibling initialization skips required source products")
                elif address in accumulators:
                    raise ValueError("actual sibling initialization overwrites an unread output")
                accumulators[address] = member, m, n0, width, k0 + depth
            products += op.rows * depth * width
            retained += not op.fresh_weights
            pending = None
        elif type(op) is Store:
            if member is None or op.readout is not None:
                raise ValueError("sibling raw-core checker cannot replace an independent original epilogue")
            rectangle("accumulator", op.accumulator_row, op.rows, op.cols)
            if (
                any(type(v) is not int or v < 0 for v in (op.dram_row, op.dram_col))
                or op.dram_row + op.rows > group.m
                or op.dram_col + op.cols > group.widths[member]
            ):
                raise ValueError("actual sibling readout exceeds original result")
            for r in range(op.rows):
                expected = member, op.dram_row + r, op.dram_col, op.cols, group.k
                if accumulators.get(op.accumulator_row + r) != expected:
                    raise ValueError("actual sibling readout changes or omits original products")
                outputs.setdefault((member, op.dram_row + r), []).append((op.dram_col, op.dram_col + op.cols))
                # Completion releases ownership; a different output may now
                # reuse the row, while a duplicate store still fails coverage.
                del accumulators[op.accumulator_row + r]
        else:
            raise ValueError("unknown sibling effects cannot be erased or approximated")
    if pending is not None or accumulators or len(outputs) != group.m * len(group.widths):
        raise ValueError("actual sibling final state leaves pending or missing original outputs")
    for (member, _), intervals in outputs.items():
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("actual sibling outputs contain gaps or duplicate intervals")
            cursor = end
        if cursor != group.widths[member]:
            raise ValueError("actual sibling outputs omit required source columns")
    if products != group.m * group.k * sum(group.widths):
        raise ValueError("actual sibling products differ from complete original work")
    return {
        "status": "verified_finite_shared_input_effects",
        "products": products,
        "output_elements": group.m * sum(group.widths),
        "lhs_loads": lhs_loads,
        "rhs_loads": rhs_loads,
        "requested_operand_bytes": requested,
        "requested_store_bytes": group.m * sum(group.widths) * 4,
        "retained_computes": retained,
        "commands": len(operations),
        "rows_visited": visited,
        "machine_emission": "not_established",
        "timing_authority": "NONE",
    }
