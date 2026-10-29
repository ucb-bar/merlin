"""Bounded transport issue with original rolling consumers and real drains.

This source transform grants no target overlap or asynchronous completion.
Future writes occupy disjoint live ring slots. A caller must separately bind
ordered, backpressured transport and complete both transport and consumer work.
"""

from dataclasses import asdict, dataclass
import hashlib

from xdsl.dialects import arith, builtin, func, scf
from xdsl.ir import Block, Region

from merlin.xdsl_dialects.rolling_window import (
    AwaitOp, BeginOp, CompleteOp, ConsumeGroupOp, ConsumerType, ResidentType,
    StageRowOp, TransferType,
    StartConsumerOp, AdvanceConsumerOp, FinishConsumerOp, BeginRhsOp, StageRhsOp,
)
from .rolling_window import (
    RollingWindowPlan, _constant, _group_horizon, materialize_grouped_rolling_window,
    rolling_window_text, verify_grouped_rolling_window,
)


@dataclass(frozen=True)
class RollingLookaheadAdmission:
    """Caller-owned permission for disjoint transport and consumer issue."""
    hardware_identity: str
    qualification_identity: str
    disjoint_local_effects: bool = False
    immutable_input: bool = False
    ordered_configuration: bool = False
    finite_queue_progress: bool = False
    complete_consumer_and_transport: bool = False
    retained_consumer_continuations: bool = False

    def require(self):
        if (any(type(v) is not str or not v.strip() for v in
                (self.hardware_identity, self.qualification_identity))
                or any(type(v) is not bool or not v for v in
                       (self.disjoint_local_effects, self.immutable_input,
                        self.ordered_configuration, self.finite_queue_progress,
                        self.complete_consumer_and_transport))):
            raise ValueError("lookahead needs separate complete target effect/progress admission")
        if type(self.retained_consumer_continuations) is not bool:
            raise ValueError("lookahead continuation admission must be an explicit Boolean")


@dataclass(frozen=True)
class RollingRhsFrontierAdmission:
    """Separate target law for replacing RHS rows after their original reads.

    Cross-lane WAR/RAW dependencies must retain actual data commands until
    completion. The current consumer's real full drain also completes future
    DMA before its phase can consume these same rows. No drain is removed.
    """
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    scratchpad_read_write_dependencies: bool = False
    ordered_configuration: bool = False
    finite_queue_progress: bool = False
    immutable_weights: bool = False
    full_consumer_transport_completion: bool = False

    def require(self):
        values = asdict(self)
        if any(type(values[k]) is not str or not values[k].strip() for k in
               ("hardware_identity", "numerical_policy_identity", "qualification_identity")) or any(
            type(values[k]) is not bool or not values[k] for k in values if not k.endswith("identity")
        ):
            raise ValueError("RHS frontier requires separately bound WAR/RAW, configuration, progress and full completion")


@dataclass(frozen=True)
class RollingRhsFrontier:
    """Next original phase mapped into a dead prefix of the current RHS rows."""
    source_identity: str
    rhs_base: int
    tile: int
    output_channels: int
    current_column_blocks: int
    next_columns: tuple[tuple[int, int], ...]

    def require(self, plan, runs):
        if (self.source_identity != plan.source_identity
                or any(type(v) is not int or not 0 < v < 1 << 63 for v in
                       (self.rhs_base, self.tile, self.output_channels, self.current_column_blocks))
                or type(self.next_columns) is not tuple
                or not 0 < len(self.next_columns) <= self.current_column_blocks
                or any(type(t) is not tuple or len(t) != 2 or any(type(v) is not int for v in t)
                       or not 0 <= t[0] < t[0]+t[1] <= self.output_channels or not 0 < t[1] <= self.tile
                       for t in self.next_columns)
                or any(a[0]+a[1] > b[0] for a,b in zip(self.next_columns,self.next_columns[1:]))):
            raise ValueError("RHS frontier needs original ordered columns within the current dead-prefix allocation")
        oh,ow=plan.window.output_spatial
        expected=plan.window.kernel[0]*plan.window.kernel[1]*((plan.window.input_shape[-1]+self.tile-1)//self.tile)
        if (type(runs) is not int or runs != expected or self.rhs_base+expected*self.current_column_blocks*self.tile >= 1 << 63
                or (plan.output_x_first,plan.output_x_count,plan.output_y_first,plan.output_y_count)!=(0,ow,0,oh)):
            raise ValueError("RHS frontier requires every original spatial/batch consumer and complete K runs")


@dataclass(frozen=True)
class RollingBatchFrontierAdmission:
    """Original target law for replacing input rows after their final reads."""
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    scratchpad_read_write_dependencies: bool = False
    ordered_configuration: bool = False
    finite_queue_progress: bool = False
    immutable_input: bool = False
    full_consumer_transport_completion: bool = False
    retained_consumer_continuations: bool = False

    def require(self):
        values = asdict(self)
        if any(type(values[k]) is not str or not values[k].strip() for k in
               ("hardware_identity", "numerical_policy_identity", "qualification_identity")) or any(
            type(values[k]) is not bool or not values[k] for k in values if not k.endswith("identity")
        ):
            raise ValueError("batch frontier requires original input WAR/RAW, progress, continuation and completion admission")


@dataclass(frozen=True)
class RollingBatchFrontier:
    """Derive next-image row writes from source K and actual window ownership.

    A single window waits each row's final original reader. Distinct image
    windows permit future rows between original complete kernel-tap traversals;
    real cohort completion still precedes cycling back to a previous window.
    """
    source_identity: str
    channel_tile: int

    def stages(self, plan, output_rows, runs):
        if (self.source_identity != plan.source_identity or type(self.channel_tile) is not int
                or not 0 < self.channel_tile < 1 << 63):
            raise ValueError("batch frontier requires its original source and channel tile")
        w = plan.window
        oh, ow = w.output_spatial
        chunks = (w.input_shape[-1] + self.channel_tile - 1) // self.channel_tile
        if (output_rows != oh or (plan.output_x_first, plan.output_x_count,
                plan.output_y_first, plan.output_y_count) != (0, ow, 0, oh)
                or runs != w.kernel[0] * w.kernel[1] * chunks):
            raise ValueError("batch frontier needs one complete original spatial consumer and K traversal")
        live = _validate(plan, output_rows, False)
        result = []
        for row in range(live):
            taps = [ky for ky in range(w.kernel[0])
                    if any(y * w.stride[0] + ky * w.dilation[0] == row for y in range(oh))]
            # An untouched stride/dilation hole has no original readers. Its
            # unchanged preparation can follow the first complete source run.
            # Original kernel-tap boundaries keep the channel traversal in one
            # counted loop. These are source work boundaries, not latency cuts.
            last = (max(1,w.kernel[0]*w.kernel[1]*(row+1)//(live+1))*chunks-1 if plan.image_windows>1 else
                    ((max(taps) + 1) * w.kernel[1] * chunks - 1) if taps else 0)
            result.append((last, row))
        return tuple(result)


def _validate(plan, output_rows, future_input=True):
    if type(plan) is not RollingWindowPlan:
        raise ValueError("lookahead needs the original typed rolling source")
    _group_horizon(plan, output_rows)
    if type(future_input) is not bool:
        raise ValueError("future input selection must be an explicit Boolean")
    # The union of two adjacent original groups is a contiguous source-row
    # interval. Modulo slots are injective on any interval no wider than slots.
    # This includes invalid halo rows, which still write their original zeros.
    count = min((2 if future_input else 1) * output_rows, plan.output_y_count)
    live = plan.span_rows + (count - 1) * plan.window.stride[0]
    if live > plan.image_slots:
        raise ValueError("lookahead future rows overlap the current live source union")
    return live


def materialize_lookahead_rolling_window(plan, *, output_rows, consumer_runs=None,
                                        rhs_frontier=None, future_input=True, batch_frontier=None):
    """Build actual counted SCF using the existing typed transport effects.

    Prologue awaits current rows. Each cohort issues future frontier rows,
    consumes the current group, completes *both* lifetimes, then awaits the
    future transfer. The carried resident is an actual binding, never a delay.
    The final group has no speculative transport beyond the original domain.
    ``consumer_runs`` optionally splits the original consumer into complete
    ordered reduction traversals. Its target must retain accumulator/array
    state across future strips and initialize/read out exactly once. The run
    count comes from the original consumer, never a timing or queue estimate.
    An optional RHS frontier replaces only a dead physical prefix after each
    complete K/N/M run of the last spatial group in the last source batch.
    Its separate target law must cover real scratchpad read/write completion.
    ``future_input=False`` serializes the original input rows while retaining
    this RHS continuation. Every original consumer completion remains present.
    A separately admitted ``batch_frontier`` instead carries the next image's
    original rows through the existing drain, replacing each ring row only
    after its derived final source K run. It requires a full spatial consumer.
    """
    _validate(plan, output_rows, future_input)
    _runs(consumer_runs)
    if rhs_frontier is not None:
        if type(rhs_frontier) is not RollingRhsFrontier:
            raise ValueError("RHS frontier requires its typed original phase map")
        rhs_frontier.require(plan, consumer_runs)
    if batch_frontier is not None:
        if type(batch_frontier) is not RollingBatchFrontier or future_input:
            raise ValueError("batch frontier is a separate complete-image input alternative")
        batch_frontier.stages(plan, output_rows, consumer_runs)
    return _materialize(plan, output_rows, consumer_runs, rhs_frontier, future_input, batch_frontier)


def _runs(count):
    if count is not None and (type(count) is not int or not 0 < count < 1 << 63):
        raise ValueError("split consumer requires its positive original reduction-run count")


def _materialize(plan, output_rows, consumer_runs=None, rhs_frontier=None, future_input=True, batch_frontier=None):
    w = plan.window
    batches, h, width, ci = w.input_shape
    root = Block()

    def binary(out, cls, a, b):
        a = _constant(out, a) if type(a) is int else a
        b = _constant(out, b) if type(b) is int else b
        op = cls(a, b); out.add_op(op); return op.result

    def loop(out, lo, hi, body, initial=()):
        lo, hi, step = (_constant(out, n) for n in (lo, hi, 1))
        inner = Block(arg_types=[builtin.i64, *(v.type for v in initial)])
        result = body(inner, *inner.args) or ()
        inner.add_op(scf.YieldOp(*result))
        op = scf.ForOp(lo, hi, step, initial, Region(inner)); out.add_op(op)
        return op.results

    def stage(out, image, y, first, count, transfer=None):
        if transfer is None:
            begin = BeginOp.create(result_types=[TransferType()]); out.add_op(begin)
            transfer = begin.transfer
        def row(inner, index):
            unbiased = binary(inner, arith.AddiOp, binary(inner, arith.MuliOp, y, w.stride[0]),
                              binary(inner, arith.AddiOp, index, first))
            source_y = binary(inner, arith.SubiOp, unbiased, w.padding[0])
            slot=binary(inner,arith.RemUIOp,unbiased,plan.image_slots)
            if plan.image_windows>1:
                slot=binary(inner,arith.AddiOp,slot,binary(inner,arith.MuliOp,
                    binary(inner,arith.RemUIOp,image,plan.image_windows),plan.image_slots))
            cache = binary(inner, arith.AddiOp, plan.cache.start,
                           binary(inner, arith.MuliOp, slot, plan.row_stride_bytes))
            source = binary(inner, arith.AddiOp,
                            binary(inner, arith.MuliOp,
                                   binary(inner, arith.AddiOp, binary(inner, arith.MuliOp, image, h), source_y), width * ci),
                            plan.source_columns[0] * ci)
            valid = arith.CmpiOp(source_y, _constant(inner, h), "ult"); inner.add_op(valid)
            inner.add_op(StageRowOp.create(operands=[transfer, image, source_y, source, cache, valid.result],
                                          properties={"row_bytes": builtin.IntegerAttr(plan.row_bytes, 64)}))
        loop(out, 0, count, row)
        return transfer

    def ready(out, transfer):
        op = AwaitOp.create(operands=[transfer], result_types=[ResidentType()]); out.add_op(op); return op.resident

    def consume(out, resident, image, y, count, future=None, next_image=None):
        if type(y) is int: y = _constant(out, y)
        if consumer_runs is not None:
            op = StartConsumerOp.create(operands=[resident, image, y], result_types=[ConsumerType()],
                                        properties={"output_rows": builtin.IntegerAttr(count, 64)})
            out.add_op(op)
            current = op.consumer
            batch_transfer = None
            if next_image is not None:
                begin = BeginOp.create(result_types=[TransferType()]); out.add_op(begin)
                batch_transfer = begin.transfer
            rhs_transfer = None
            if rhs_frontier is not None:
                begin_rhs = BeginRhsOp.create(result_types=[TransferType()],
                    properties={"rhs_row":builtin.IntegerAttr(rhs_frontier.rhs_base,64)})
                out.add_op(begin_rhs); rhs_transfer=begin_rhs.transfer

            def advance(first, stop):
                regions=[]
                if rhs_frontier is not None:
                    body=Block(arg_types=[builtin.i64,builtin.i64,builtin.i64])
                    run,reduction,depth=body.args
                    last_image=arith.CmpiOp(image,_constant(body,batches-1),"eq");body.add_op(last_image)
                    yes,no=Block(),Block()
                    last_y=arith.CmpiOp(binary(yes,arith.AddiOp,y,count),
                                       _constant(yes,plan.output_y_count),"eq");yes.add_op(last_y)
                    stage_body,empty=Block(),Block()
                    for n,(column,columns)in enumerate(rhs_frontier.next_columns):
                        local=binary(stage_body,arith.AddiOp,rhs_frontier.rhs_base,
                            binary(stage_body,arith.MuliOp,
                                binary(stage_body,arith.AddiOp,
                                    binary(stage_body,arith.MuliOp,run,len(rhs_frontier.next_columns)),n),rhs_frontier.tile))
                        stage_body.add_op(StageRhsOp.create(operands=[rhs_transfer,reduction,
                            _constant(stage_body,column),local,depth],
                            properties={"columns":builtin.IntegerAttr(columns,64)}))
                    stage_body.add_op(scf.YieldOp());empty.add_op(scf.YieldOp())
                    yes.add_op(scf.IfOp(last_y.result,[],Region(stage_body),Region(empty)))
                    yes.add_op(scf.YieldOp());no.add_op(scf.YieldOp())
                    body.add_op(scf.IfOp(last_image.result,[],Region(yes),Region(no)))
                    regions=[Region(body)]
                operation=AdvanceConsumerOp.create(operands=[current],result_types=[ConsumerType()],
                    properties={"first_run":builtin.IntegerAttr(first,64),"run_count":builtin.IntegerAttr(stop-first,64)},
                    regions=regions)
                out.add_op(operation)
                return operation.consumer

            position = 0
            transfer = None
            if next_image is not None:
                stages = batch_frontier.stages(plan, output_rows, consumer_runs)
                for last in sorted({last for last, _ in stages}):
                    stop = last + 1
                    current = advance(position, stop); position = stop
                    more = arith.CmpiOp(image, _constant(out, batches - 1), "ult"); out.add_op(more)
                    yes, no = Block(), Block()
                    for boundary, row in stages:
                        if boundary == last:
                            stage(yes, next_image, 0, row, 1, batch_transfer)
                    yes.add_op(scf.YieldOp()); no.add_op(scf.YieldOp())
                    out.add_op(scf.IfOp(more.result, [], Region(yes), Region(no)))
            if future is not None:
                next_y, rows = future
                begin = BeginOp.create(result_types=[TransferType()]); out.add_op(begin)
                transfer = begin.transfer
                # Each strip follows a whole original reduction traversal.
                # Repeated boundaries collect strips when runs < future rows.
                for row in range(rows):
                    stop = max(1, consumer_runs * (row + 1) // (rows + 1))
                    if stop > position:
                        current = advance(position,stop); position = stop
                    stage(out, image, next_y, plan.span_rows - w.stride[0] + row, 1, transfer)
            if position < consumer_runs:
                current = advance(position,consumer_runs)
            finish = FinishConsumerOp.create(operands=[current], result_types=[ConsumerType()])
            out.add_ops((finish, CompleteOp.create(operands=[finish.consumer])))
            return batch_transfer if next_image is not None else transfer
        op = ConsumeGroupOp.create(operands=[resident, image, y], result_types=[ConsumerType()],
                                   properties={"output_rows": builtin.IntegerAttr(count, 64)})
        out.add_ops((op, CompleteOp.create(operands=[op.consumer])))

    def image(out, image):
        groups, tail = divmod(plan.output_y_count, output_rows)
        total = groups + bool(tail)
        first = min(output_rows, plan.output_y_count)
        resident = ready(out, stage(out, image, plan.output_y_first, 0,
                                    plan.span_rows + (first - 1) * w.stride[0]))
        # All current groups in the steady loop have exactly output_rows rows.
        def cohort(inner, index, current):
            y = binary(inner, arith.AddiOp, plan.output_y_first, binary(inner, arith.MuliOp, index, output_rows))
            next_y = binary(inner, arith.AddiOp, y, output_rows)
            if not future_input:
                consume(inner,current,image,y,output_rows)
                transfer=stage(inner,image,next_y,plan.span_rows-w.stride[0],output_rows*w.stride[0])
            elif consumer_runs is None:
                transfer = stage(inner, image, next_y, plan.span_rows - w.stride[0], output_rows * w.stride[0])
                consume(inner, current, image, y, output_rows)
            else:
                transfer = consume(inner, current, image, y, output_rows,
                                   (next_y, output_rows * w.stride[0]))
            return (ready(inner, transfer),)
        full_pairs = groups - 1 if groups else 0
        if full_pairs:
            resident, = loop(out, 0, full_pairs, cohort, (resident,))
        y = plan.output_y_first + max(0, groups - 1) * output_rows
        if groups and tail:
            next_y = plan.output_y_first + groups * output_rows
            if not future_input:
                consume(out,resident,image,y,output_rows)
                transfer=stage(out,image,next_y,plan.span_rows-w.stride[0],tail*w.stride[0])
            elif consumer_runs is None:
                transfer = stage(out, image, next_y, plan.span_rows - w.stride[0], tail * w.stride[0])
                consume(out, resident, image, y, output_rows)
            else:
                transfer = consume(out, resident, image, y, output_rows, (next_y, tail * w.stride[0]))
            resident = ready(out, transfer)
            y = next_y
        consume(out, resident, image, y, tail or output_rows)

    if batch_frontier is not None:
        live = _validate(plan, output_rows, False)
        resident = ready(root, stage(root, _constant(root, 0), 0, 0, live))
        def next_batch(inner, index, current):
            next_image = binary(inner, arith.AddiOp, index, 1)
            transfer = consume(inner, current, index, 0, output_rows, next_image=next_image)
            return (ready(inner, transfer),)
        # One counted consumer body also handles the final image. Its guarded
        # future transfer is empty, but still receives a real completion drain.
        # This avoids duplicating the complete original K/N/M target program.
        loop(root, 0, batches, next_batch, (resident,))
    else:
        loop(root, 0, batches, image)
    root.add_op(func.ReturnOp())
    module = builtin.ModuleOp([func.FuncOp("rolling_lookahead", ([], []), Region(root))])
    module.verify()
    return module


def verify_lookahead_rolling_window(module, plan, *, output_rows, consumer_runs=None,
                                   rhs_frontier=None,future_input=True,batch_frontier=None):
    """Check closed actual SCF conformance plus independent source/lifetime laws.

    The existing checker independently proves original stage addresses, row
    coverage, halo guards and retained identity. The only extra effect order
    admitted here is a future frontier before or between complete original
    reduction runs of a disjoint current consumer.
    Canonical conformance rejects altered operands, effects, yields or drains;
    it is not permission for arbitrary user-edited schedules.
    """
    live = _validate(plan, output_rows,future_input)
    _runs(consumer_runs)
    if rhs_frontier is not None:
        if type(rhs_frontier) is not RollingRhsFrontier:
            raise ValueError("RHS frontier requires its typed original phase map")
        rhs_frontier.require(plan,consumer_runs)
    batch_stages = ()
    if batch_frontier is not None:
        if type(batch_frontier) is not RollingBatchFrontier or future_input:
            raise ValueError("batch frontier is a separate complete-image input alternative")
        batch_stages = batch_frontier.stages(plan, output_rows, consumer_runs)
    module.verify()
    actual = rolling_window_text(module)
    if actual != rolling_window_text(_materialize(plan, output_rows, consumer_runs,rhs_frontier,future_input,batch_frontier)):
        raise ValueError("actual lookahead SCF changes checked source/effect/completion structure")
    base = verify_grouped_rolling_window(materialize_grouped_rolling_window(plan, output_rows=output_rows),
                                         plan, output_rows=output_rows)
    return {**base, "schema": "merlin.rolling_lookahead.v1",
            "actual_ir_sha256": hashlib.sha256(actual.encode()).hexdigest(),
            "actual_counted_ir_operations": sum(1 for _ in module.walk()),
            "live_source_row_union": live,
            "completion": "current consumer and future transport fully complete before next consumer or slot reuse",
            "consumer_runs": consumer_runs,
            "future_input":future_input,
            "batch_frontier": None if batch_frontier is None else asdict(batch_frontier),
            "batch_row_issue_runs": batch_stages,
            "batch_input_image_windows": plan.image_windows,
            "batch_final_empty_transfer_drains": int(batch_frontier is not None),
            "rhs_frontier":None if rhs_frontier is None else asdict(rhs_frontier),
            "rhs_frontier_tiles":0 if rhs_frontier is None else consumer_runs*len(rhs_frontier.next_columns),
            "issue_order": ("next original image rows after derived final K/N/M reads; final image retains selected RHS frontier; all original full completions"
                            if batch_frontier is not None else
                            "last batch/spatial group: complete original K/N/M run; next RHS dead-prefix transfer; original full completion"
                            if rhs_frontier is not None else
                            "original reduction slices interleaved with disjoint future strips; finish; complete both; await future"
                            if consumer_runs is not None and future_input else
                            "original reduction slices; finish; complete; stage and await next original rows"
                            if consumer_runs is not None else
                            "future frontier; current consumer; complete both; await future"),
            "target_qualification": False}


def emit_lookahead_rolling_window(module, plan, builder, lower, *, output_rows, admission, consumer_runs=None,
                                rhs_frontier=None,future_input=True,rhs_frontier_admission=None,
                                batch_frontier=None,batch_frontier_admission=None):
    """Consume the verified actual SCF, preserving its real completion edges."""
    if type(admission) is not RollingLookaheadAdmission:
        raise ValueError("lookahead requires its separate target effect admission")
    admission.require()
    if consumer_runs is not None and admission.retained_consumer_continuations is not True:
        raise ValueError("split consumer requires separate retained-state continuation admission")
    if rhs_frontier is not None:
        if type(rhs_frontier_admission) is not RollingRhsFrontierAdmission:
            raise ValueError("RHS frontier requires its separate target admission")
        rhs_frontier_admission.require()
        if rhs_frontier_admission.hardware_identity != admission.hardware_identity:
            raise ValueError("RHS frontier and input transport have different target identities")
    frontier_identity=None if rhs_frontier_admission is None else asdict(rhs_frontier_admission)
    if batch_frontier is not None:
        if type(batch_frontier_admission) is not RollingBatchFrontierAdmission:
            raise ValueError("batch frontier requires its separate target admission")
        batch_frontier_admission.require()
        if batch_frontier_admission.hardware_identity != admission.hardware_identity:
            raise ValueError("batch frontier and input transport have different target identities")
    batch_identity=None if batch_frontier_admission is None else asdict(batch_frontier_admission)
    before = verify_lookahead_rolling_window(module, plan, output_rows=output_rows, consumer_runs=consumer_runs,
                                           rhs_frontier=rhs_frontier,future_input=future_input,batch_frontier=batch_frontier)
    identity = asdict(admission)
    if not callable(lower) or not callable(getattr(builder, "loop_carried", None)):
        raise ValueError("lookahead needs actual effect bindings and carried-loop emission")

    def handle(value):
        if value is None: raise ValueError("lookahead binding did not return an actual handle")
        return value

    def emit(block, inherited):
        env = dict(inherited)
        for op in block.ops:
            if type(op) is arith.ConstantOp: env[op.result] = op.value.value.data
            elif type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp):
                name = {arith.AddiOp:"add",arith.SubiOp:"sub",arith.MuliOp:"mul",arith.RemUIOp:"urem"}[type(op)]
                env[op.result] = builder.op(name, env[op.lhs], env[op.rhs])
            elif type(op) is arith.CmpiOp:
                env[op.result] = builder.cmp({0:"eq",6:"ult"}[op.predicate.value.data],env[op.lhs],env[op.rhs])
            elif type(op) is scf.IfOp:
                builder.branch(env[op.cond],lambda:emit(op.true_region.block,env),
                               lambda:emit(op.false_region.block,env))
            elif type(op) is scf.ForOp:
                def body(index, carried=()):
                    return emit(op.body.block, {**env, **dict(zip(op.body.block.args,(index,*carried),strict=True))})
                if op.iter_args:
                    result = builder.loop_carried(env[op.lb],env[op.ub],env[op.step],tuple(env[x]for x in op.iter_args),body)
                    env.update(zip(op.results,result,strict=True))
                else: builder.loop(env[op.lb],env[op.ub],env[op.step],body)
            elif type(op) is BeginOp: env[op.transfer] = handle(lower("begin_transport", {}))
            elif type(op) is BeginRhsOp:
                env[op.transfer] = handle(lower("begin_rhs_prefetch",{"rhs_row":op.rhs_row.value.data}))
            elif type(op) is StageRhsOp:
                fields={k:env[getattr(op,k)]for k in("transfer","reduction_row","output_column","local_row","rows")}
                lower("stage_rhs",{**fields,"columns":op.columns.value.data})
            elif type(op) is StageRowOp:
                fields = {k:env[getattr(op,k)]for k in ("transfer","image","input_row","source_byte","cache_byte","valid")}
                lower("stage_row",{**fields,"row_bytes":op.row_bytes.value.data})
            elif type(op) is AwaitOp: env[op.resident] = handle(lower("await_transport",{"transfer":env[op.transfer]}))
            elif type(op) is ConsumeGroupOp:
                fields = {k:env[getattr(op,k)]for k in ("resident","image","output_y")}
                env[op.consumer] = handle(lower("consume_group",{**fields,"output_rows":op.output_rows.value.data}))
            elif type(op) is StartConsumerOp:
                fields = {k:env[getattr(op,k)]for k in ("resident","image","output_y")}
                env[op.consumer] = handle(lower("start_consumer",{**fields,"output_rows":op.output_rows.value.data,
                                                                            "consumer_runs":consumer_runs}))
            elif type(op) is AdvanceConsumerOp:
                fields={"consumer":env[op.previous],"first_run":op.first_run.value.data,
                        "run_count":op.run_count.value.data}
                if op.after_run is not None:
                    region=op.after_run.block
                    def after_run(run,reduction,depth):
                        emit(region,{**env,**dict(zip(region.args,(run,reduction,depth),strict=True))})
                    fields["after_run"]=after_run
                env[op.consumer] = handle(lower("advance_consumer",{"consumer":env[op.previous],
                    **fields}))
            elif type(op) is FinishConsumerOp:
                env[op.consumer] = handle(lower("finish_consumer",{"consumer":env[op.previous]}))
            elif type(op) is CompleteOp: lower("complete",{"consumer":env[op.consumer]})
            elif type(op) is scf.YieldOp: return tuple(env[x]for x in op.arguments)
            elif type(op) is not func.ReturnOp: raise ValueError("unhandled checked lookahead operation")
        return ()

    emit(tuple(module.body.block.ops)[0].body.block,{})
    admission.require()
    if rhs_frontier_admission is not None:
        rhs_frontier_admission.require()
    if frontier_identity != (None if rhs_frontier_admission is None else asdict(rhs_frontier_admission)):
        raise ValueError("RHS frontier target admission changed during emission")
    if batch_frontier_admission is not None:
        batch_frontier_admission.require()
    if batch_identity != (None if batch_frontier_admission is None else asdict(batch_frontier_admission)):
        raise ValueError("batch frontier target admission changed during emission")
    if asdict(admission)!=identity or verify_lookahead_rolling_window(module,plan,output_rows=output_rows,
            consumer_runs=consumer_runs,rhs_frontier=rhs_frontier,future_input=future_input,batch_frontier=batch_frontier)!=before:
        raise ValueError("lookahead source/admission changed during emission")
    return {**before,"admission":identity,"emission":"actual counted SCF; real binding handles and full completion"}
