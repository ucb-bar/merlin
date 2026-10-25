"""Retain complete LHS panels while double-buffering bounded RHS K segments.

This is a counted command transformation, not an implicit hardware permission.
The selected owner must independently admit ordered accumulator commits WHILE
other engines write disjoint future operand windows. The old disjoint-output
and pure-reduction capabilities alone do not supply that conjunction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseBiasSource, _digest, _edge, _emit_lhs_panel, _Group, _groups
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    ISSUE_ORDERS,
    DensePanelOverlapAdmission,
    DenseRetainedReductionAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, check_structure, select
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.transforms.dense_planning import dense_blocks

SEGMENTED_ISSUE_ORDERS = (*ISSUE_ORDERS, "prefetch_rhs_between_k_tiles")

@dataclass(frozen=True)
class DenseSegmentedAdmission:
    """Separate, explicit conjunction of overlap and ordered reduction laws.

    The identity names an independently reviewed source/target binding covering
    the combined state. Merely supplying the two previous laws is insufficient.
    Issued counts remain compiler bounds; pending counts require selected
    ready/valid backpressure, not a fabricated token or fixed cycle spacing.
    """

    overlap: DensePanelOverlapAdmission
    reduction: DenseRetainedReductionAdmission
    qualification_identity: str
    execution: str = "ordered_accumulator_commits_disjoint_future_loads_full_drain"

    def __post_init__(self):
        if (type(self.overlap) is not DensePanelOverlapAdmission
                or type(self.reduction) is not DenseRetainedReductionAdmission
                or type(self.qualification_identity) is not str or not self.qualification_identity.strip()
                or self.execution != "ordered_accumulator_commits_disjoint_future_loads_full_drain"):
            raise ValueError("segmented reduction needs a separate selected combined execution law")
        self.overlap.__post_init__()
        self.reduction.__post_init__()
        for field in ("hardware_identity", "numerical_policy_identity", "load_configuration", "descriptor",
                      "pending_compute_pairs", "pending_loads", "pending_readouts", "readout_configuration"):
            if getattr(self.overlap, field) != getattr(self.reduction, field):
                raise ValueError("segmented component laws disagree on selected execution or numerical state")


def _assignment(problem, block, resources, storage, window_offsets=None):
    """Exact original geometry and private byte ownership, without a new grant."""
    if storage.accumulator is None or block.k > resources.max_k_blocks:
        raise ValueError("segmented reduction requires original private accumulators and declared K domain")
    candidates = dense_blocks(problem, replace(resources, max_k_blocks=block.k))
    if block not in candidates or not storage.admits(problem, block, resources):
        raise ValueError("segmented block differs from original geometry, domain or resource derivation")
    kt = problem.tiles[1]
    counts = (block.m * kt * problem.tile_m, block.k * block.n * problem.tile_k)
    if window_offsets is None:
        window_offsets = ((0, 0), (0, counts[1]))
    if (type(window_offsets) is not tuple or len(window_offsets) != 2
            or any(type(slot) is not tuple or len(slot) != 2 for slot in window_offsets)
            or any(type(x) is not int or x < 0 for slot in window_offsets for x in slot)
            or window_offsets[0][0] != window_offsets[1][0]):
        raise ValueError("segmented windows require one retained LHS offset and two bounded RHS offsets")
    lhs = replace(storage.lhs, base=storage.lhs.base + window_offsets[0][0], capacity=counts[0])
    rhs = tuple(replace(storage.rhs, base=storage.rhs.base + offset[1], capacity=counts[1])
                for offset in window_offsets)
    for view, parent in ((lhs, storage.lhs), *((v, storage.rhs) for v in rhs)):
        if (view.base < parent.base or view.base + view.capacity > parent.base + parent.capacity
                or view.base % parent.alignment):
            raise ValueError("segmented live window escapes or misaligns its original assigned view")
    views = (lhs, *rhs, storage.accumulator)
    for i, left in enumerate(views):
        for right in views[i + 1:]:
            if (left.backing == right.backing
                    and left.base * left.unit_bytes < (right.base + right.capacity) * right.unit_bytes
                    and right.base * right.unit_bytes < (left.base + left.capacity) * left.unit_bytes):
                raise ValueError("segmented simultaneously live windows alias")
    if (lhs.capacity * lhs.unit_bytes + sum(v.capacity * v.unit_bytes for v in rhs) > resources.operand_bytes
            or block.m * block.n * problem.tile_m * storage.accumulator.unit_bytes > resources.accumulator_bytes):
        raise ValueError("retained LHS plus two RHS segments exceed original byte capacity")
    return lhs, rhs


def build_dense_segmented_pipeline(
    problem, block, resources, storage, adapter, *, name, dtypes,
    prefetch=True, outer_order=("m", "n"), window_offsets=None, issue_order="compute_then_prefetch",
    output_readout=None, bias_source=None,
):
    """Emit real initial/frontier loads, ordered sums, stores and full drains.

    Full-K LHS is staged once per M group and reused across N groups. The two
    RHS slots alternate only after the previous complete epoch drained. Within
    one epoch, current-segment increasing-K compute and future-segment movement
    use disjoint windows. The serialized control emits identical data commands
    with one extra completion boundary between those phases. No modulo address
    expression, dynamic instruction expansion or full-image buffer is needed.
    """
    if type(issue_order) is not str or issue_order not in SEGMENTED_ISSUE_ORDERS:
        raise ValueError("unsupported explicit segmented issue order")
    if issue_order != "compute_then_prefetch" and not prefetch:
        raise ValueError("frontier interleaving requires an explicit prefetched epoch")
    if type(prefetch) is not bool or outer_order != ("m", "n"):
        raise ValueError("retained-LHS segmented family requires explicit issue policy and M/N outer order")
    if problem.bias != (bias_source is not None):
        raise ValueError("segmented bias presence differs from original source")
    if bias_source is not None:
        if type(bias_source) is not DenseBiasSource:
            raise ValueError("segmented initialization requires the original typed bias source")
        bias_source.validate(problem, dtypes)
    if type(output_readout) is DenseTiledIntegerReadout:
        output_readout.validate_problem(problem)
    from merlin.sched.ir.kernel import DTYPE_BYTES

    if (tuple(DTYPE_BYTES.get(t) for t in dtypes)
            != (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes)
            or not adapter.synchronous_completion or not adapter.exact_accumulator_continuation):
        raise ValueError("segmented source needs exact original integer widths and serial continuation semantics")
    lhs, rhs = _assignment(problem, block, resources, storage, window_offsets)
    kt = problem.tiles[1]
    segments = (kt + block.k - 1) // block.k
    last = kt - (segments - 1) * block.k
    serial = 0
    used = {"lhs", "rhs", "bias", "output", "outer_m", "outer_n"}

    def variable(prefix):
        nonlocal serial
        serial += 1
        value = prefix + str(serial)
        if value in used:
            raise ValueError("segmented variable collides with source boundary")
        used.add(value)
        return value

    def call(operation):
        result = adapter.encode(operation)
        if type(result) is not Call:
            raise ValueError("segmented adapter must emit actual instruction calls")
        return result

    def nest(extents, action):
        names = tuple(variable("segment_tile_") for _ in extents)
        body = tuple(action(*(Var(n) for n in names)))
        for n, extent in reversed(tuple(zip(names, extents))):
            body = (Loop(n, extent, body),)
        return body

    def epoch(body):
        return Loop(variable("segment_epoch_"), 1, (*body, Call(DRAIN, ())))

    def output_group(mg, ng):
        def depth(k, count, final):
            return (select(k, count - 1, problem.k % problem.tile_k, problem.tile_k)
                    if final and problem.k % problem.tile_k else Const(problem.tile_k))

        def loads(slot, base, count, final=False):
            def load(k, n):
                return (call(Load("weight", (base + k, ng.base + n), (base + k) * problem.tile_k,
                                  (ng.base + n) * problem.tile_n, depth(k, count, final),
                                  _edge(ng, n, problem.n, problem.tile_n),
                                  rhs[slot].base + (k * block.n + n) * problem.tile_k, 1)),)
            return nest((count, ng.blocks), load)

        def computes(slot, base, count, first=False, final=False):
            def compute(k, m, n):
                rows, cols = _edge(mg, m, problem.m, problem.tile_m), _edge(ng, n, problem.n, problem.tile_n)
                continuation = (select(k, 0, 0, 1) if first and bias_source is None else Const(1))
                acc = storage.accumulator.base + (m * block.n + n) * problem.tile_m
                return (call(Preload(rhs[slot].base + (k * block.n + n) * problem.tile_k,
                                     (base + k, ng.base + n), acc, continuation, rows, cols, depth(k, count, final))),
                        call(Compute(lhs.base + (m * kt + base + k) * problem.tile_m,
                                     (mg.base + m, base + k), rows, depth(k, count, final), True)))
            return nest((count, mg.blocks, ng.blocks), compute)

        def stage(slot, base, *, first=False, final_frontier=False):
            current = computes(slot, base, block.k, first=first)
            future = loads(1 - slot, base + block.k, last if final_frontier else block.k, final_frontier)
            return (epoch((*current, *future)),) if prefetch else (epoch(current), epoch(future))

        result = []
        if bias_source is not None:
            def initialize(m, n):
                return (epoch((call(InitializeAccumulator(
                    (mg.base + m, ng.base + n), (ng.base + n) * problem.tile_n,
                    _edge(mg, m, problem.m, problem.tile_m), _edge(ng, n, problem.n, problem.tile_n),
                    storage.accumulator.base + (m * block.n + n) * problem.tile_m)),)),)
            result.extend(nest((mg.blocks, ng.blocks), initialize))
        result.append(epoch(loads(0, Const(0), last if segments == 1 else block.k, segments == 1)))
        if segments >= 3:
            result.extend(stage(0, Const(0), first=True))
            pairs, tail = divmod(segments - 3, 2)
            if pairs:
                period = variable("segment_period_")
                result.append(Loop(period, pairs, (
                    *stage(1, (Var(period) * 2 + 1) * block.k),
                    *stage(0, (Var(period) * 2 + 2) * block.k))))
            if tail:
                index = 1 + 2 * pairs
                result.extend(stage(index % 2, Const(index * block.k)))
        if segments >= 2:
            index = segments - 2
            result.extend(stage(index % 2, Const(index * block.k), first=index == 0, final_frontier=True))
        result.append(epoch(computes((segments - 1) % 2, Const((segments - 1) * block.k), last,
                                     first=segments == 1, final=True)))

        def store(m, n):
            tiled = type(output_readout) is DenseTiledIntegerReadout
            operation = TiledIntegerStore if tiled else Store
            return (epoch((call(operation(
                (mg.base + m) * problem.tile_m, (ng.base + n) * problem.tile_n,
                _edge(mg, m, problem.m, problem.tile_m), _edge(ng, n, problem.n, problem.tile_n),
                storage.accumulator.base + (m * block.n + n) * problem.tile_m,
                output_readout.identity if output_readout is not None else None,
                **({"readout_tile": ng.base + n} if tiled else {}))),)),)
        result.extend(nest((mg.blocks, ng.blocks), store))
        return tuple(result)

    body = []
    full = replace(block, k=kt)
    for mg in _groups(problem.tiles[0], block.m, "outer_m"):
        retained = _emit_lhs_panel(problem, full, replace(storage, lhs=lhs), call, nest, mg,
                                   _Group(None, 1, Const(0), kt, True))
        row = [epoch(retained)]
        for ng in _groups(problem.tiles[2], block.n, "outer_n"):
            group = output_group(mg, ng)
            row.extend((Loop(ng.var, ng.trips, group),) if ng.var is not None else group)
        body.extend((Loop(mg.var, mg.trips, tuple(row)),) if mg.var is not None else row)
    kernel = Kernel(name, (
        TensorArg("lhs", (problem.m, problem.k), dtypes[0], "read"),
        TensorArg("rhs", (problem.k, problem.n), dtypes[1], "read"),
        *((TensorArg("bias", (problem.n,), bias_source.dtype, "read"),) if bias_source is not None else ()),
        TensorArg("output", (problem.m, problem.n),
                  dtypes[2] if output_readout is None else output_readout.output_dtype,
                  "write")), tuple(body))
    if check_structure(kernel):
        raise ValueError("segmented construction changed source control or completion scopes")
    check_integer_domain(kernel, adapter.index_bits)
    if issue_order != "compute_then_prefetch":
        kernel, _ = interleave_dense_segmented_frontier(kernel, adapter, issue_order=issue_order)
    return kernel


def interleave_dense_segmented_frontier(kernel, adapter, *, issue_order="prefetch_rhs_between_columns"):
    """Propose a bounded RHS frontier load after each current column's M work.

    The actual projected axes identify a perfect K/M/N compute nest and the
    following K/N weight-load nest in one existing drained epoch. The typed
    choice either fuses the K loops, or interchanges M/N then zips the future
    load into the matching counted K/N loop by exact coordinate substitution. A shorter final frontier
    uses a counted compute-only suffix; no invalid speculative load is emitted.
    Full source and epoch checks MUST follow: this pass grants no target law.
    """
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
    from merlin.compile.scheduling.tile_load_coalescing import _linear, _plain
    from merlin.sched.ir.expr import free_vars
    from merlin.sched.primitives import _bound_vars, _loop_at, _splice

    if (type(kernel) is not Kernel or check_structure(kernel)
            or issue_order not in SEGMENTED_ISSUE_ORDERS[1:]):
        raise ValueError("frontier interleaving requires actual closed commands and an explicit interleaving order")
    check_integer_domain(kernel, adapter.index_bits)
    paths = []

    def discover(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            current = (*path, statement.var)
            if statement.body and statement.body[-1] == Call(DRAIN, ()):
                ops = [adapter.project(c) for c in _calls(statement.body[:-1])]
                if any(type(op) is Load for op in ops) and any(type(op) is Compute for op in ops):
                    paths.append(current)
            else:
                discover(statement.body, current)
    discover(kernel.body)
    if not paths:
        raise ValueError("frontier issue order requires at least one actual mixed prefetch epoch")
    actual = kernel
    used = _bound_vars(kernel.body) | {arg.name for arg in kernel.args}
    records = []

    def chain(statement, count):
        loops = []
        for index in range(count):
            if type(statement) is not Loop:
                raise ValueError("frontier issue requires complete actual K/M/N and K/N counted nests")
            loops.append(statement)
            if index + 1 < count:
                if len(statement.body) != 1:
                    raise ValueError("frontier issue cannot move an effect out of an imperfect nest")
                statement = statement.body[0]
        return loops, statement.body

    def equivalent(a, b):
        return _linear(a) == _linear(b)

    for path in paths:
        epoch = _loop_at(actual.body, path)
        if epoch.extent != 1 or len(epoch.body) != 3:
            raise ValueError("frontier issue requires one complete current stream then one future stream and drain")
        (ck, cm, cn), paired = chain(epoch.body[0], 3)
        (fk, fn), loads = chain(epoch.body[1], 2)
        if (len(paired) != 2 or len(loads) != 1
                or any(type(c) is not Call or not _plain(c) for c in (*paired, *loads))):
            raise ValueError("frontier issue cannot split a physical pair or move unknown effects")
        preload, compute, load = (adapter.project(c) for c in (*paired, *loads))
        coordinates = {ck.var, cm.var, cn.var}
        if (type(preload) is not Preload or type(compute) is not Compute or type(load) is not Load
                or load.role != "weight" or fk.extent > ck.extent or fn.extent != cn.extent
                or coordinates & free_vars(compute.input_block[0]) != {cm.var}
                or coordinates & free_vars(compute.input_block[1]) != {ck.var}
                or coordinates & free_vars(preload.weight_block[0]) != {ck.var}
                or coordinates & free_vars(preload.weight_block[1]) != {cn.var}):
            raise ValueError("frontier issue axes differ from actual independent K/M/N source domains")
        renamed = _remap(_remap(loads, fk.var, ck.var), fn.var, cn.var)
        projected = adapter.project(renamed[0])
        if (not equivalent(projected.block[0], preload.weight_block[0] + ck.extent)
                or not equivalent(projected.block[1], preload.weight_block[1])):
            raise ValueError("future RHS frontier is not the exact following K segment with the same columns")
        following = ()
        if fk.extent < ck.extent:
            name = ck.var + "_frontier_suffix"
            while name in used:
                name += "_"
            used.add(name)
            following = (Loop(name, ck.extent - fk.extent,
                              _remap(ck.body, ck.var, name, offset=fk.extent)),)
        if issue_order == "prefetch_rhs_between_k_tiles":
            # The existing typed fusion primitive substitutes the future K
            # coordinate into the current K loop. All M/N compute order stays
            # exactly as before. A short frontier leaves its suffix after the
            # fused loop, never manufacturing an out-of-domain DMA request.
            prefix = replace(ck, extent=fk.extent)
            rewritten = replace(epoch, body=(prefix, epoch.body[1], *following, Call(DRAIN, ())))
            actual = replace(actual, body=_splice(actual.body, path, (rewritten,)))
            request = LoopTransform("fuse", (*path, ck.var))
            actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
            pairs_before_load = cm.extent * cn.extent
        else:
            request = LoopTransform("reorder", (*path, ck.var, cm.var))
            actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
            epoch = _loop_at(actual.body, path)
            (ck, cn, cm), _ = chain(epoch.body[0], 3)
            columns = replace(cn, body=(cm, *renamed))
            prefix = replace(ck, extent=fk.extent, body=(columns,))
            rewritten = replace(epoch, body=(prefix, *following, Call(DRAIN, ())))
            actual = replace(actual, body=_splice(actual.body, path, (rewritten,)))
            pairs_before_load = cm.extent
        records.append(dict(loop_transform=asdict(request), epoch=path,
                            interleaved_k_tiles=fk.extent, compute_only_suffix=ck.extent - fk.extent,
                            output_columns=cn.extent, compute_pairs_before_frontier=pairs_before_load))
    if check_structure(actual):
        raise ValueError("frontier interleaving changed loop scope or completion ownership")
    check_integer_domain(actual, adapter.index_bits)
    return actual, dict(schema="merlin.segmented_frontier_issue.v1", source_kernel_sha256=kernel.digest(),
                        actual_kernel_sha256=actual.digest(), regions=records, issue_order=issue_order,
                        completion="original full epoch drain retained", target_execution="not_established")


def verify_dense_segmented_pipeline(
    kernel, problem, block, resources, storage, adapter, admission, *, static_budget,
    timeout_ms=5000, output_readout=None, bias_source=None,
):
    """Check actual complete source, epochs and addresses under the NEW law."""
    if type(admission) is not DenseSegmentedAdmission:
        raise ValueError("segmented pipeline needs explicit combined execution admission")
    admission.__post_init__()
    if (admission.overlap.hardware_identity != adapter.configuration_identity
            or admission.overlap.numerical_policy_identity != adapter.numerical_policy_identity):
        raise ValueError("segmented law differs from original selected adapter")
    _assignment(problem, block, resources, storage)
    bound = replace(admission.overlap,
                    max_issued_compute_pairs=min(admission.overlap.max_issued_compute_pairs,
                                                 admission.reduction.max_issued_compute_pairs),
                    max_issued_loads=min(admission.overlap.max_issued_loads, admission.reduction.max_issued_loads),
                    max_issued_readouts=min(admission.overlap.max_issued_readouts,
                                           admission.reduction.max_issued_readouts))
    report = _verify_dense_epochs(
        kernel, problem, resources, storage, adapter, bound,
        static_budget=static_budget, timeout_ms=timeout_ms, output_readout=output_readout,
        bias_source=bias_source, ordered_accumulator_commits=True, source_phase_fission=True)
    report.update(schema="merlin.dense_segmented_pipeline.v1", admission=asdict(admission),
                  ordered_accumulator_commits=admission.qualification_identity,
                  retained_operand="lhs", segment_k_tiles=block.k)
    return report


@dataclass(frozen=True)
class DenseSegmentedPipelineResult:
    """Checked emitted commands; no target qualification is fabricated."""

    kernel: Kernel
    realization: dict
    source_identity: str
    admission: DenseSegmentedAdmission
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_owner_sha256: str
    checked_library_sources: tuple
    output_readout: object = None
    bias_source: object = None
    readout_qualification_identity: str | None = None
    checked_boundary_sha256: str | None = None

    def revalidate(self):
        from hashlib import sha256
        from pathlib import Path

        from merlin.compile.scheduling.library_sources import shared_library_sources

        if (self.kernel.digest() != self.checked_kernel_sha256
                or _digest(asdict(self.admission)) != self.checked_admission_sha256
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.checked_owner_sha256
                or tuple(sorted(shared_library_sources(version=15).items())) != self.checked_library_sources
                or _digest((None if self.output_readout is None else asdict(self.output_readout),
                            None if self.bias_source is None else asdict(self.bias_source),
                            self.readout_qualification_identity)) != self.checked_boundary_sha256):
            raise ValueError("segmented checked commands, capability, source boundary or owner changed")

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program

        self.revalidate()
        result = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return result


def schedule_dense_segmented_pipeline(
    problem, resources, storage, adapter, admission, *, name, dtypes, schedule_plan,
    source_identity, external_buffers_disjoint, static_budget=128, timeout_ms=5000,
    output_readout=None, bias_source=None, readout_qualification_identity=None,
):
    """Consume the ordinary typed source-bound block/placement/issue decision."""
    from hashlib import sha256
    from pathlib import Path

    from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.compile.scheduling.schedule_ir import check_dense_schedule_commands, read_dense_schedule_plan

    if external_buffers_disjoint is not True:
        raise ValueError("segmented source requires original external-buffer nonalias ownership")
    if output_readout is not None:
        if (type(output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout)
                or type(readout_qualification_identity) is not str or not readout_qualification_identity.strip()):
            raise ValueError("segmented readout requires original typed numerical execution qualification")
    elif readout_qualification_identity is not None:
        raise ValueError("segmented readout qualification has no original numerical boundary")
    decision = read_dense_schedule_plan(
        schedule_plan, problem, resources, storage, adapter, source_identity=source_identity,
        dtypes=dtypes, static_budget=static_budget, output_readout=output_readout, bias_source=bias_source)
    if (decision.pipeline_kind != "retained_lhs_segmented_k" or decision.pipeline_prefetch is None
            or decision.pipeline_issue_order not in SEGMENTED_ISSUE_ORDERS):
        raise ValueError("segmented consumer requires its exact explicit typed pipeline decision")
    kernel = build_dense_segmented_pipeline(
        problem, decision.block, resources, decision.storage, adapter, name=name, dtypes=dtypes,
        prefetch=decision.pipeline_prefetch, outer_order=decision.outer_order,
        window_offsets=decision.pipeline_window_offsets, output_readout=output_readout, bias_source=bias_source)
    issue_report = None
    if decision.pipeline_issue_order != "compute_then_prefetch":
        if not decision.pipeline_prefetch:
            raise ValueError("frontier interleaving requires an explicit prefetched epoch")
        kernel, issue_report = interleave_dense_segmented_frontier(
            kernel, adapter, issue_order=decision.pipeline_issue_order)
    transform_source = kernel.digest()
    if decision.loop_transforms:
        from merlin.compile.scheduling.loop_schedule import propose_loop_transforms

        # Typed requests propose real command rewrites. The unchanged complete
        # source and epoch checks below still establish their legality, including
        # descriptor pairing, reduction cover and actual operand noninterference.
        kernel = propose_loop_transforms(kernel, decision.loop_transforms, index_bits=adapter.index_bits)
    check_dense_schedule_commands(decision, sum(1 for _ in _calls(kernel.body)), static_budget)
    report = verify_dense_segmented_pipeline(
        kernel, problem, decision.block, resources, decision.storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms, output_readout=output_readout, bias_source=bias_source)
    report["frontier_issue"] = issue_report
    report["loop_transforms"] = dict(
        source_kernel_sha256=transform_source, actual_kernel_sha256=kernel.digest(),
        requests=[asdict(request) for request in decision.loop_transforms],
        legality="original complete source and actual epoch hazard checks",
    )
    report["schedule_ir"] = dict(
        sha256=decision.plan_sha256, text=decision.plan_text, source_identity=source_identity,
        pipeline_kind=decision.pipeline_kind, prefetch=decision.pipeline_prefetch,
        pipeline_issue_order=decision.pipeline_issue_order,
        decisions_consumed=("block_tiles", "outer_order", "row_bases", "panel_pipeline", "pipeline_window_offsets",
                            "pipeline_issue_order", "loop_transforms"))
    result = DenseSegmentedPipelineResult(
        kernel, report, source_identity, admission, kernel.digest(), _digest(asdict(admission)),
        sha256(Path(__file__).read_bytes()).hexdigest(), tuple(sorted(shared_library_sources(version=15).items())),
        output_readout, bias_source, readout_qualification_identity,
        _digest((None if output_readout is None else asdict(output_readout),
                 None if bias_source is None else asdict(bias_source), readout_qualification_identity)))
    result.revalidate()
    return result


def admit_segmented_tile_loads(
    original, proposal, problem, block, resources, storage, adapter, *,
    project, encode_band, static_budget=128, timeout_ms=5000, max_static_nodes=4096,
):
    """Compose exact wide addressing with the original combined execution law.

    Every actual band is projected back into its original scalar tile address,
    then all source/reduction/epoch checks run under the unchanged segmented
    admission. Wider commands supply no additional queue or overlap permission.
    Store batching, when selected, must already be checked before this final
    address-only pass. The old disjoint-output consumer remains separate.
    """
    from merlin.compile.scheduling.panel_tile_loads import _admit_dense_tile_loads

    if type(original) is not DenseSegmentedPipelineResult:
        raise ValueError("segmented wide loads require their original checked combined-law result")
    return _admit_dense_tile_loads(
        original, proposal, problem, block, resources, storage, adapter,
        verifier=verify_dense_segmented_pipeline, project=project, encode_band=encode_band,
        static_budget=static_budget, timeout_ms=timeout_ms, max_static_nodes=max_static_nodes,
    )


def batch_dense_segmented_readouts(
    result, problem, block, resources, storage, adapter, admission, *, static_budget=128, timeout_ms=5000,
):
    """Reuse the existing pure-store pass under unchanged combined compute law.

    Only the two component declarations' independent store snapshot/backpressure
    limits may change. Every compute/load epoch and its completion stays intact.
    Varying tiled numerical configuration keeps its separately drained route.
    """
    from merlin.compile.scheduling.dense_panel_readout import _propose

    if type(result) is not DenseSegmentedPipelineResult or type(admission) is not DenseSegmentedAdmission:
        raise ValueError("segmented readout batching needs original checked result and combined admission")
    result.revalidate()
    admission.__post_init__()
    if (admission.qualification_identity != result.admission.qualification_identity
            or admission.execution != result.admission.execution):
        raise ValueError("readout batching cannot change the combined compute law")
    for chosen, original in ((admission.overlap, result.admission.overlap),
                             (admission.reduction, result.admission.reduction)):
        if replace(chosen, max_issued_readouts=original.max_issued_readouts,
                   pending_readouts=original.pending_readouts,
                   readout_configuration=original.readout_configuration) != original:
            raise ValueError("readout batching cannot change original operand or reduction capability")
    if type(result.output_readout) is DenseTiledIntegerReadout:
        raise ValueError("varying per-tile readout configuration has no segmented batch law")
    budget = min(admission.overlap.max_issued_readouts, admission.reduction.max_issued_readouts)
    proposal, groups = _propose(result.kernel, adapter, budget)
    report = verify_dense_segmented_pipeline(
        proposal, problem, block, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=result.output_readout, bias_source=result.bias_source)
    report["schedule_ir"] = result.realization["schedule_ir"]
    report["frontier_issue"] = result.realization.get("frontier_issue")
    report["input_schedule"] = result.realization
    report["readout_batching"] = dict(
        source_kernel_sha256=result.kernel.digest(), actual_kernel_sha256=proposal.digest(), groups=groups,
        removed_dynamic_drains=result.realization["dynamic_epoch_drains"] - report["dynamic_epoch_drains"],
        target_execution="not_established")
    rewritten = replace(result, kernel=proposal, realization=report, admission=admission,
                        checked_kernel_sha256=proposal.digest(), checked_admission_sha256=_digest(asdict(admission)))
    result.revalidate()
    rewritten.revalidate()
    return rewritten
