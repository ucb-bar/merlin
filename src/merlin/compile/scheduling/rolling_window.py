"""Counted initial/frontier row transport for source-exact NHWC windows.

The compact checker reads actual SCF/arith/typed-effect IR. It proves address
maps, coverage, retained row identity and completion placement independently
of runtime domain size. Payload correctness, input immutability, nonaliasing
and the original consumer's lowering remain separately qualified premises.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass

from xdsl.dialects import arith, builtin, func, scf
from xdsl.ir import Block, Region

from merlin.xdsl_dialects.rolling_window import (
    AwaitOp,
    BeginOp,
    CompleteOp,
    ConsumeGroupOp,
    ConsumeOp,
    ConsumerType,
    ResidentType,
    StageRowOp,
    TransferType,
)

from .program_limits import AddressWindow
from .window_transfer import Window2D

SCHEMA = "merlin.rolling_window.v1"
GROUP_SCHEMA = "merlin.grouped_rolling_window.v1"


@dataclass(frozen=True)
class RollingWindowPlan:
    window: Window2D
    source_identity: str
    output_x_first: int
    output_x_count: int
    output_y_first: int
    output_y_count: int
    cache: AddressWindow
    slots: int
    row_stride_bytes: int
    alignment: int = 1
    image_windows: int = 1

    def __post_init__(self):
        if type(self.window) is not Window2D or type(self.cache) is not AddressWindow:
            raise ValueError("rolling cache requires original typed window and caller-owned byte capacity")
        self.window.__post_init__()
        self.cache.__post_init__()
        if (type(self.slots) is not int or type(self.image_windows) is not int or not 0 < self.image_windows < 1 << 63
                or self.slots % self.image_windows):
            raise ValueError("image windows require distinct equal caller-owned row ranges")
        if (
            type(self.source_identity) is not str
            or len(self.source_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.source_identity)
        ):
            raise ValueError("rolling cache requires the original caller's source identity")
        oh, ow = self.window.output_spatial
        if (
            any(
                type(v) is not int
                for v in (self.output_x_first, self.output_x_count, self.output_y_first, self.output_y_count)
            )
            or not 0 <= self.output_x_first < self.output_x_first + self.output_x_count <= ow
            or not 0 <= self.output_y_first < self.output_y_first + self.output_y_count <= oh
        ):
            raise ValueError("rolling cache must cover an explicit original output rectangle")
        if (
            any(type(v) is not int or not 0 < v < 1 << 63 for v in (self.slots, self.row_stride_bytes, self.alignment))
            or self.alignment & (self.alignment - 1)
            or self.image_slots < self.span_rows
            or self.row_bytes <= 0
            or self.row_stride_bytes < self.row_bytes
            or self.row_stride_bytes % self.alignment
            or self.cache.start % self.alignment
            or self.required_bytes > self.cache.size
        ):
            raise ValueError("rolling cache slots, alignment or complete row capacity are insufficient")
        b, h, w, ci = self.window.input_shape
        last_y = self.output_y_first + self.output_y_count - 1
        last_unbiased = last_y * self.window.stride[0] + self.span_rows - 1
        low_y = self.output_y_first * self.window.stride[0] - self.window.padding[0]
        high_y = last_unbiased - self.window.padding[0]
        # Every actual SSA index fits signed i64, including invalid source-row
        # offsets which the explicit padding predicate prevents dereferencing.
        if (
            max(
                last_unbiased,
                self.cache.start + self.required_bytes,
                abs(low_y) * w * ci + self.row_bytes,
                abs((b - 1) * h + high_y) * w * ci + w * ci,
            )
            >= 1 << 63
        ):
            raise ValueError("rolling cache index arithmetic exceeds signed i64")

    @property
    def span_rows(self):
        return (self.window.kernel[0] - 1) * self.window.dilation[0] + 1

    @property
    def source_columns(self):
        _, _, width, _ = self.window.input_shape
        first = self.output_x_first * self.window.stride[1] - self.window.padding[1]
        last = (
            (self.output_x_first + self.output_x_count - 1) * self.window.stride[1]
            - self.window.padding[1]
            + (self.window.kernel[1] - 1) * self.window.dilation[1]
            + 1
        )
        return max(0, first), min(width, last)

    @property
    def row_bytes(self):
        first, last = self.source_columns
        return (last - first) * self.window.input_shape[-1]

    @property
    def required_bytes(self):
        return self.slots * self.row_stride_bytes

    @property
    def image_slots(self):
        return self.slots // self.image_windows

    @property
    def frontier_rows(self):
        return min(self.span_rows, self.window.stride[0])

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def propose_rolling_window(
    window,
    cache,
    *,
    source_identity,
    output_x_first=0,
    output_x_count=None,
    output_y_first=0,
    output_y_count=None,
    slots=None,
    alignment=1,
):
    """Derive a bounded source row view; this proposal grants no target effect."""
    if type(window) is not Window2D or type(cache) is not AddressWindow:
        raise ValueError("rolling cache needs explicit source geometry and assigned storage")
    window.__post_init__()
    oh, ow = window.output_spatial
    if any(type(v) is not int for v in (output_x_first, output_y_first)):
        raise ValueError("rolling cache output origins must be integers")
    nx = ow - output_x_first if output_x_count is None else output_x_count
    ny = oh - output_y_first if output_y_count is None else output_y_count
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("rolling cache alignment must be a positive power of two")
    first = max(0, output_x_first * window.stride[1] - window.padding[1])
    last = min(
        window.input_shape[2],
        (output_x_first + nx - 1) * window.stride[1]
        - window.padding[1]
        + (window.kernel[1] - 1) * window.dilation[1]
        + 1,
    )
    row_bytes = (last - first) * window.input_shape[-1]
    pitch = (row_bytes + alignment - 1) // alignment * alignment
    count = (window.kernel[0] - 1) * window.dilation[0] + 1 if slots is None else slots
    return RollingWindowPlan(
        window, source_identity, output_x_first, nx, output_y_first, ny, cache, count, pitch, alignment
    )


def rolling_window_text(module):
    from xdsl.printer import Printer

    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def read_rolling_window(text, plan):
    """Parse editable typed SCF transport, then check its original source plan."""
    return _read_rolling_window(text, plan, None)


def read_grouped_rolling_window(text, plan, *, output_rows):
    """Parse and independently check actual grouped transport and consumers."""
    return _read_rolling_window(text, plan, output_rows)


def _read_rolling_window(text, plan, output_rows):
    from xdsl.context import Context
    from xdsl.parser import Parser

    from merlin.xdsl_dialects.rolling_window import RollingWindow

    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, scf.Scf, RollingWindow):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    _verify_rolling_window(module, plan, output_rows)
    return module


def _constant(out, value):
    op = arith.ConstantOp.from_int_and_width(value, 64)
    out.add_op(op)
    return op.result


def materialize_rolling_window(plan):
    """Emit actual counted SCF and typed, linearly ordered effect tokens."""
    return _materialize_rolling_window(plan, None)


def _group_horizon(plan, output_rows):
    if type(plan) is not RollingWindowPlan:
        raise ValueError("grouped transport requires its original rolling-window plan")
    plan.__post_init__()
    if type(output_rows) is not int or not 1 <= output_rows <= 16:
        raise ValueError("grouped transport needs an explicit output-row bound 1..16")
    # Preserve precisely the original contiguous initial/frontier row transfers.
    # A strided window with gaps needs a different checked union representation.
    if plan.window.stride[0] > plan.span_rows:
        raise ValueError("grouped transport requires overlapping or adjacent source-row windows")
    horizon = plan.span_rows + (min(output_rows, plan.output_y_count) - 1) * plan.window.stride[0]
    if horizon > plan.image_slots:
        raise ValueError("grouped source-row union exceeds the actual ring slots")
    return horizon


def _group_identity(plan, output_rows):
    return hashlib.sha256(json.dumps((plan.identity, output_rows)).encode()).hexdigest()


def materialize_grouped_rolling_window(plan, *, output_rows):
    """Stage original row unions, await once, consume, and complete before reuse.

    The caller separately binds original independent consumer semantics and
    bounded target issue/progress laws. A source-row bound is not queue capacity.
    The existing one-row materializer and its completion contract are unchanged.
    """
    _group_horizon(plan, output_rows)
    return _materialize_rolling_window(plan, output_rows)


def _materialize_rolling_window(plan, output_rows):
    if type(plan) is not RollingWindowPlan:
        raise ValueError("original typed rolling-window proposal required")
    plan.__post_init__()
    w = plan.window
    b, h, width, ci = w.input_shape
    block = Block()

    def binary(out, cls, lhs, rhs):
        lhs = _constant(out, lhs) if type(lhs) is int else lhs
        rhs = _constant(out, rhs) if type(rhs) is int else rhs
        op = cls(lhs, rhs)
        out.add_op(op)
        return op.result

    def loop(out, low, high, body):
        lb, ub, step = (_constant(out, v) for v in (low, high, 1))
        inner = Block(arg_types=[builtin.i64])
        body(inner, inner.args[0])
        inner.add_op(scf.YieldOp())
        out.add_op(scf.ForOp(lb, ub, step, [], Region(inner)))

    def epoch(out, image, y, first, count, consumers=1):
        if type(y) is int:
            y = _constant(out, y)
        begin = BeginOp.create(result_types=[TransferType()])
        out.add_op(begin)

        def stage(inner, row):
            unbiased = binary(
                inner,
                arith.AddiOp,
                binary(inner, arith.MuliOp, y, w.stride[0]),
                binary(inner, arith.AddiOp, row, first),
            )
            source_y = binary(inner, arith.SubiOp, unbiased, w.padding[0])
            slot = binary(inner, arith.RemUIOp, unbiased, plan.image_slots)
            if plan.image_windows > 1:
                slot = binary(inner, arith.AddiOp, slot,
                    binary(inner, arith.MuliOp, binary(inner, arith.RemUIOp, image, plan.image_windows), plan.image_slots))
            destination = binary(
                inner, arith.AddiOp, plan.cache.start, binary(inner, arith.MuliOp, slot, plan.row_stride_bytes)
            )
            source = binary(
                inner,
                arith.AddiOp,
                binary(
                    inner,
                    arith.MuliOp,
                    binary(inner, arith.AddiOp, binary(inner, arith.MuliOp, image, h), source_y),
                    width * ci,
                ),
                plan.source_columns[0] * ci,
            )
            valid = arith.CmpiOp(source_y, _constant(inner, h), "ult")
            inner.add_op(valid)
            inner.add_op(
                StageRowOp.create(
                    operands=[begin.transfer, image, source_y, source, destination, valid.result],
                    properties={"row_bytes": builtin.IntegerAttr(plan.row_bytes, 64)},
                )
            )

        loop(out, 0, count, stage)
        ready = AwaitOp.create(operands=[begin.transfer], result_types=[ResidentType()])
        consume = (
            ConsumeOp.create(operands=[ready.resident, image, y], result_types=[ConsumerType()])
            if output_rows is None else
            ConsumeGroupOp.create(operands=[ready.resident, image, y], result_types=[ConsumerType()],
                properties={"output_rows": builtin.IntegerAttr(consumers, 64)})
        )
        out.add_ops((ready, consume, CompleteOp.create(operands=[consume.consumer])))

    def image_body(out, image):
        if output_rows is not None:
            full, tail = divmod(plan.output_y_count, output_rows)
            first_count = output_rows if full else tail
            advance = w.stride[0]
            epoch(out, image, plan.output_y_first, 0,
                  plan.span_rows + (first_count - 1) * advance, first_count)
            if full > 1:
                def group(inner, index):
                    y = binary(inner, arith.AddiOp, plan.output_y_first,
                               binary(inner, arith.MuliOp, index, output_rows))
                    epoch(inner, image, y, plan.span_rows - advance, output_rows * advance, output_rows)
                loop(out, 1, full, group)
            if full and tail:
                epoch(out, image, plan.output_y_first + full * output_rows,
                      plan.span_rows - advance, tail * advance, tail)
            return
        epoch(out, image, plan.output_y_first, 0, plan.span_rows)
        if plan.output_y_count > 1:
            loop(
                out,
                plan.output_y_first + 1,
                plan.output_y_first + plan.output_y_count,
                lambda inner, y: epoch(inner, image, y, plan.span_rows - plan.frontier_rows, plan.frontier_rows),
            )

    loop(block, 0, b, image_body)
    block.add_op(func.ReturnOp())
    module = builtin.ModuleOp(
        [func.FuncOp("rolling_window", ([], []), Region(block))],
        attributes={SCHEMA: builtin.StringAttr(plan.identity)} if output_rows is None else
                   {GROUP_SCHEMA: builtin.StringAttr(_group_identity(plan, output_rows))},
    )
    _verify_rolling_window(module, plan, output_rows)
    return module


def verify_rolling_window(module, plan):
    """Prove actual addresses and effect order, plus the inductive reuse law.

    No dynamic loops are enumerated and no source-body or hardware law is
    inferred. A consumer must use the original checked lookup and complete at
    its explicit token boundary; a transport binding must copy/zero exactly.
    """
    return _verify_rolling_window(module, plan, None)


def verify_grouped_rolling_window(module, plan, *, output_rows):
    """Check actual grouped SCF, exact row effects, union residency and custody."""
    _group_horizon(plan, output_rows)
    return _verify_rolling_window(module, plan, output_rows)


def _verify_rolling_window(module, plan, output_rows):
    import z3

    if type(plan) is not RollingWindowPlan:
        raise ValueError("rolling-window check needs its original caller-owned proposal")
    plan.__post_init__()
    module.verify()

    def require(condition, reason):
        if not condition:
            raise ValueError("rolling-window check: " + reason)

    horizon = plan.span_rows if output_rows is None else _group_horizon(plan, output_rows)
    schema = SCHEMA if output_rows is None else GROUP_SCHEMA
    identity = plan.identity if output_rows is None else _group_identity(plan, output_rows)
    require(
        type(module) is builtin.ModuleOp
        and set(module.attributes) == {schema}
        and module.attributes[schema] == builtin.StringAttr(identity),
        "original source/layout identity changed",
    )
    functions = tuple(module.body.block.ops)
    require(len(functions) == 1 and type(functions[0]) is func.FuncOp, "one closed entry required")
    function = functions[0]
    require(
        function.sym_name.data == "rolling_window"
        and len(function.body.blocks) == 1
        and not function.function_type.inputs
        and not function.function_type.outputs
        and not function.attributes,
        "entry cannot introduce a payload ABI or hidden function semantics",
    )
    arithmetic = (arith.ConstantOp, arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp, arith.CmpiOp)
    consumer_class = ConsumeOp if output_rows is None else ConsumeGroupOp
    effects = (BeginOp, StageRowOp, AwaitOp, consumer_class, CompleteOp)
    allowed = (*arithmetic, *effects, scf.ForOp, scf.YieldOp, func.ReturnOp)
    for op in function.body.walk():
        require(type(op) in allowed and not op.attributes, "unknown operation or semantic attribute")
        if isinstance(op, (arith.AddiOp, arith.SubiOp, arith.MuliOp)):
            require(op.overflow_flags == arith.IntegerOverflowAttr("none"), "unproved overflow/poison flags")
        require(not op.successors and (not op.regions or type(op) is scf.ForOp), "unproved control flow")
    queries = 0
    checked_arithmetic = set()

    def prove(claim, domain, reason):
        nonlocal queries
        queries += 1
        solver = z3.Solver()
        solver.set(timeout=2000)
        solver.add(*domain, z3.Not(claim))
        require(solver.check() == z3.unsat, reason)

    def expr(value, env, domain, memo):
        if value in env:
            return env[value]
        if value in memo:
            return memo[value]
        owner = value.owner
        require(type(owner) in arithmetic, "captured or unknown address expression")
        checked_arithmetic.add(owner)
        if type(owner) is arith.ConstantOp:
            require(type(owner.value) is builtin.IntegerAttr and value.type == builtin.i64, "non-i64 constant")
            result = z3.IntVal(owner.value.value.data)
        else:
            lhs, rhs = (expr(v, env, domain, memo) for v in owner.operands)
            if type(owner) is arith.AddiOp:
                result = lhs + rhs
            elif type(owner) is arith.SubiOp:
                result = lhs - rhs
            elif type(owner) is arith.MuliOp:
                require(
                    z3.is_int_value(z3.simplify(lhs)) or z3.is_int_value(z3.simplify(rhs)),
                    "non-affine address multiplication",
                )
                result = lhs * rhs
            elif type(owner) is arith.RemUIOp:
                divisor = z3.simplify(rhs)
                require(z3.is_int_value(divisor) and divisor.as_long() > 0, "unproved modulo divisor")
                prove(lhs >= 0, domain, "unsigned cache slot operand can be negative")
                result = lhs % divisor
            else:
                require(owner.predicate.value.data == 6, "only explicit unsigned source-row bounds admitted")

                def unsigned(x):
                    return z3.If(x < 0, x + (1 << 64), x)

                result = unsigned(lhs) < unsigned(rhs)
        if value.type == builtin.i64:
            prove(z3.And(result >= -(1 << 63), result < 1 << 63), domain, "actual address arithmetic can overflow")
        memo[value] = z3.simplify(result)
        return memo[value]

    def effect_list(block):
        return tuple(
            op for op in block.ops if type(op) not in arithmetic and type(op) not in (scf.YieldOp, func.ReturnOp)
        )

    def bounds(loop, expected):
        require(
            type(loop) is scf.ForOp
            and not loop.iter_args
            and not loop.results
            and len(loop.body.blocks) == 1
            and len(loop.body.block.args) == 1,
            "counted loop boundary changed",
        )
        values = []
        for value in (loop.lb, loop.ub, loop.step):
            require(type(value.owner) is arith.ConstantOp, "loop bounds must remain explicit constants")
            checked_arithmetic.add(value.owner)
            values.append(value.owner.value.value.data)
        require(
            tuple(values) == expected and values[0] < values[1] and values[2] == 1, "missing/duplicated loop coverage"
        )

    w = plan.window
    b, h, width, ci = w.input_shape
    image = z3.Int("rolling_image")
    outer = effect_list(function.body.block)
    require(len(outer) == 1, "one original complete image loop required")
    bounds(outer[0], (0, b, 1))
    env = {outer[0].body.block.args[0]: image}
    image_domain = [image >= 0, image < b]

    def epoch(block, expected_y, env, domain, first, count, consumers=1):
        commands = effect_list(block)
        require(
            len(commands) == 5
            and tuple(type(op) for op in commands) == (BeginOp, scf.ForOp, AwaitOp, consumer_class, CompleteOp),
            "transport/await/consume/complete order changed",
        )
        begin, loop, ready, consume, complete = commands
        if output_rows is not None:
            require(consume.output_rows == builtin.IntegerAttr(consumers, 64), "original grouped consumer rows changed")
        bounds(loop, (0, count, 1))
        require(
            ready.transfer == begin.transfer
            and consume.resident == ready.resident
            and complete.consumer == consume.consumer,
            "completion token custody changed",
        )
        row = z3.Int("rolling_frontier_row")
        inner_env = {**env, loop.body.block.args[0]: row}
        inner_domain = [*domain, row >= 0, row < count]
        stages = effect_list(loop.body.block)
        require(len(stages) == 1 and type(stages[0]) is StageRowOp, "each original frontier row needs one transport")
        stage = stages[0]
        require(
            stage.transfer == begin.transfer and stage.row_bytes == builtin.IntegerAttr(plan.row_bytes, 64),
            "row transport changed epoch or byte extent",
        )
        memo = {}
        actual = [
            expr(v, inner_env, inner_domain, memo)
            for v in (stage.image, stage.input_row, stage.source_byte, stage.cache_byte, stage.valid)
        ]
        unbiased = expected_y * w.stride[0] + row + first
        source_y = unbiased - w.padding[0]
        expected = [
            image,
            source_y,
            ((image * h + source_y) * width + plan.source_columns[0]) * ci,
            plan.cache.start + ((unbiased % plan.image_slots)
                + (image % plan.image_windows) * plan.image_slots) * plan.row_stride_bytes,
            z3.And(source_y >= 0, source_y < h),
        ]
        prove(
            z3.And(*(a == e for a, e in zip(actual, expected, strict=True))),
            inner_domain,
            "actual row source/padding/slot address differs from original window",
        )
        prove(
            z3.And(actual[3] >= plan.cache.start, actual[3] + plan.row_bytes <= plan.cache.start + plan.cache.size),
            inner_domain,
            "actual row leaves assigned private cache",
        )
        observed = [expr(v, env, domain, {}) for v in (consume.image, consume.output_y)]
        prove(z3.And(observed[0] == image, observed[1] == expected_y), domain, "consumer output domain changed")

    body = outer[0].body.block
    operations = effect_list(body)
    initial = operations[:5]
    if output_rows is None:
        require(len(operations) == 5 + int(plan.output_y_count > 1), "missing or extra input/consumer epochs")

    # Inspect the initial epoch through a block view without cloning or trusting
    # the emitter's requested address list.
    class View:
        ops = initial

    if output_rows is not None:
        full, tail = divmod(plan.output_y_count, output_rows)
        first_count = output_rows if full else tail
        advance = w.stride[0]
        require(len(operations) == 5 + int(full > 1) + 5 * int(bool(full and tail)),
                "missing or extra grouped transport/consumer epochs")
        epoch(View(), z3.IntVal(plan.output_y_first), env, image_domain, 0,
              plan.span_rows + (first_count - 1) * advance, first_count)
        if full > 1:
            loop = operations[5]
            bounds(loop, (1, full, 1))
            group = z3.Int("rolling_output_group")
            epoch(loop.body.block, plan.output_y_first + group * output_rows,
                  {**env, loop.body.block.args[0]: group},
                  [*image_domain, group >= 1, group < full],
                  plan.span_rows - advance, output_rows * advance, output_rows)
        if full and tail:
            class TailView:
                ops = operations[-5:]
            epoch(TailView(), z3.IntVal(plan.output_y_first + full * output_rows), env, image_domain,
                  plan.span_rows - advance, tail * advance, tail)
    else:
        epoch(View(), z3.IntVal(plan.output_y_first), env, image_domain, 0, plan.span_rows)
    if output_rows is None and plan.output_y_count > 1:
        loop = operations[-1]
        low, high = plan.output_y_first + 1, plan.output_y_first + plan.output_y_count
        bounds(loop, (low, high, 1))
        y = z3.Int("rolling_output_y")
        epoch(
            loop.body.block,
            y,
            {**env, loop.body.block.args[0]: y},
            [*image_domain, y >= low, y < high],
            plan.span_rows - plan.frontier_rows,
            plan.frontier_rows,
        )

    require(
        all(op in checked_arithmetic for op in function.body.walk() if type(op) in arithmetic),
        "unverified or dead address arithmetic",
    )

    # Independent inductive laws: injective live rows, retained row identity,
    # complete original dilated taps, and horizontal cropped-row coverage.
    a, c, base = z3.Ints("rolling_live_a rolling_live_c rolling_base_residue")
    domain = [a >= 0, a < horizon, c >= 0, c < horizon, base >= 0, base < plan.image_slots]
    prove(z3.Implies((base + a) % plan.image_slots == (base + c) % plan.image_slots, a == c), domain, "live cache rows alias")
    if plan.image_windows > 1:
        prove((image % plan.image_windows) * plan.image_slots + a
              != ((image + 1) % plan.image_windows) * plan.image_slots + c,
              [image >= 0, image < b, a >= 0, a < plan.image_slots, c >= 0, c < plan.image_slots],
              "adjacent original images share a private input row")
    advance = w.stride[0]
    if output_rows is not None:
        member, tap = z3.Ints("rolling_group_member rolling_group_tap")
        prove(z3.And(member * advance + tap * w.dilation[0] >= 0,
                     member * advance + tap * w.dilation[0] < horizon),
              [member >= 0, member < min(output_rows, plan.output_y_count), tap >= 0, tap < w.kernel[0]],
              "group union misses an original consumer row tap")
        # The previous complete group's union contains exactly the retained
        # prefix of the next group. Every other current-union row is written
        # once by the actual checked frontier loop; injectivity above prevents
        # those writes from destroying the retained prefix before consumption.
        group, retained = z3.Ints("rolling_group_inductive rolling_group_retained")
        old = (group - output_rows) * advance + output_rows * advance + retained
        current = group * advance + retained
        prove(z3.And(old == current, old % plan.image_slots == current % plan.image_slots,
                     output_rows * advance + retained < horizon),
              [group >= plan.output_y_first + output_rows,
               group < plan.output_y_first + plan.output_y_count,
               retained >= 0, retained < plan.span_rows - advance],
              "previous group does not preserve the current retained source rows")
    if advance < plan.span_rows:
        y = z3.Int("rolling_inductive_y")
        old = (y - 1) * advance + (advance + a)
        current = y * advance + a
        prove(
            z3.And(old - w.padding[0] == current - w.padding[0], old % plan.image_slots == current % plan.image_slots),
            [
                y >= plan.output_y_first + 1,
                y < plan.output_y_first + plan.output_y_count,
                a >= 0,
                a < plan.span_rows - advance,
            ],
            "retained row identity changed",
        )
        prove(
            (base + a) % plan.image_slots != (base + c) % plan.image_slots,
            [
                a >= 0,
                a < plan.span_rows - advance,
                c >= plan.span_rows - advance,
                c < horizon,
                base >= 0,
                base < plan.image_slots,
            ],
            "frontier overwrites a retained live row",
        )
    ky, kx, x = z3.Ints("rolling_ky rolling_kx rolling_x")
    source_x = x * w.stride[1] - w.padding[1] + kx * w.dilation[1]
    prove(
        z3.And(ky * w.dilation[0] >= 0, ky * w.dilation[0] < plan.span_rows),
        [ky >= 0, ky < w.kernel[0]],
        "original dilated row tap missing",
    )
    prove(
        z3.Implies(
            z3.And(source_x >= 0, source_x < width),
            z3.And(source_x >= plan.source_columns[0], source_x < plan.source_columns[1]),
        ),
        [x >= plan.output_x_first, x < plan.output_x_first + plan.output_x_count, kx >= 0, kx < w.kernel[1]],
        "original horizontal input tap missing",
    )
    staged = b * (plan.span_rows + (plan.output_y_count - 1) * plan.frontier_rows)
    return dict(
        schema=schema,
        plan_sha256=plan.identity,
        actual_ir_sha256=hashlib.sha256(rolling_window_text(module).encode()).hexdigest(),
        symbolic_queries=queries,
        dynamic_stage_rows=staged,
        dynamic_stage_bytes=staged * plan.row_bytes,
        consumers=b * plan.output_y_count,
        unstaged_reference_rows=b * plan.output_y_count * plan.span_rows,
        retained_rows_per_advance=plan.span_rows - plan.frontier_rows,
        cache_bytes=plan.required_bytes,
        actual_counted_ir_operations=sum(1 for _ in module.walk()),
        conditional_on=[
            "original source identity and immutable input",
            "disjoint caller-owned cache",
            "exact guarded row copy/zero completed at await",
            "original consumer reads only checked window view and completes at complete",
        ],
        target_qualification=False,
        **({} if output_rows is None else dict(
            output_rows_per_group=output_rows,
            consumer_groups=b * ((plan.output_y_count + output_rows - 1) // output_rows),
            source_row_union=horizon,
            max_staged_rows_per_epoch=horizon,
            completion="await all actual group transports before consumer; full consumer completion before any reuse",
        )),
    )


def emit_rolling_window(module, plan, builder, lower):
    """Lower the verified *actual* SCF and typed effects through ordinary callbacks.

    ``lower(kind, fields)`` implements begin_transport, stage_row,
    await_transport, consume and complete. Begin/await/consume must return
    actual binding handles; await and complete must realize transport and
    consumer completion before returning. Those payload/effect contracts are
    separate from this address/lifetime proof, with no inferred target grant.
    """
    return _emit_rolling_window(module, plan, builder, lower, None)


def emit_grouped_rolling_window(module, plan, builder, lower, *, output_rows):
    """Emit actual grouped SCF with consume_group(output_y, output_rows).

    Each binding must complete every stage at await and the complete original
    consumer interval at complete. Target issue/progress bounds and arithmetic
    independence require their separate admissions; this proof grants neither.
    """
    _group_horizon(plan, output_rows)
    return _emit_rolling_window(module, plan, builder, lower, output_rows)


def _emit_rolling_window(module, plan, builder, lower, output_rows):
    before = _verify_rolling_window(module, plan, output_rows)
    if not callable(lower) or any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop")):
        raise ValueError("rolling cache emission needs an ordinary integer builder and effect binding")

    def produced(value, operation):
        if value is None:
            raise ValueError(f"rolling cache {operation} must return its binding handle")
        return value

    def emit(block, inherited):
        env = dict(inherited)
        for op in block.ops:
            if type(op) is arith.ConstantOp:
                env[op.result] = op.value.value.data
            elif type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp, arith.RemUIOp):
                name = {arith.AddiOp: "add", arith.SubiOp: "sub", arith.MuliOp: "mul", arith.RemUIOp: "urem"}[type(op)]
                env[op.result] = builder.op(name, env[op.lhs], env[op.rhs])
            elif type(op) is arith.CmpiOp:
                env[op.result] = builder.cmp("ult", env[op.lhs], env[op.rhs])
            elif type(op) is scf.ForOp:
                builder.loop(
                    env[op.lb],
                    env[op.ub],
                    env[op.step],
                    lambda index: emit(op.body.block, {**env, op.body.block.args[0]: index}),
                )
            elif type(op) is BeginOp:
                env[op.transfer] = produced(lower("begin_transport", {}), "begin_transport")
            elif type(op) is StageRowOp:
                fields = {
                    name: env[getattr(op, name)]
                    for name in ("transfer", "image", "input_row", "source_byte", "cache_byte", "valid")
                }
                lower("stage_row", {**fields, "row_bytes": op.row_bytes.value.data})
            elif type(op) is AwaitOp:
                env[op.resident] = produced(lower("await_transport", {"transfer": env[op.transfer]}), "await_transport")
            elif type(op) is ConsumeOp:
                fields = {name: env[getattr(op, name)] for name in ("resident", "image", "output_y")}
                env[op.consumer] = produced(lower("consume", fields), "consume")
            elif type(op) is ConsumeGroupOp:
                fields = {name: env[getattr(op, name)] for name in ("resident", "image", "output_y")}
                fields["output_rows"] = op.output_rows.value.data
                env[op.consumer] = produced(lower("consume_group", fields), "consume_group")
            elif type(op) is CompleteOp:
                lower("complete", {"consumer": env[op.consumer]})

    emit(tuple(module.body.block.ops)[0].body.block, {})
    if _verify_rolling_window(module, plan, output_rows) != before:
        raise ValueError("rolling-window source/effect IR changed during actual emission")
    return {**before, "emission": "actual counted SCF with explicit bound transport/consumer handles"}


def emit_rolling_window_lookup(plan, builder, output_y, output_x, ky, kx, channel, *, image=None):
    """Address one original NHWC tap in the current resident output row.

    The caller must execute this within the matching consume/complete boundary.
    Only a true predicate permits loading the byte; false means original zero
    padding (or an out-of-domain invocation). No MAC/readout or source-body
    qualification is supplied here.
    """
    if type(plan) is not RollingWindowPlan:
        raise ValueError("rolling lookup requires its original typed source plan")
    plan.__post_init__()
    if plan.image_windows > 1 and image is None:
        raise ValueError("multiple input windows require the original source image coordinate")
    w = plan.window
    _, h, width, ci = w.input_shape
    unbiased_y = builder.op("add", builder.op("mul", output_y, w.stride[0]), builder.op("mul", ky, w.dilation[0]))
    iy = builder.op("sub", unbiased_y, w.padding[0])
    ix = builder.op(
        "add",
        builder.op("sub", builder.op("mul", output_x, w.stride[1]), w.padding[1]),
        builder.op("mul", kx, w.dilation[1]),
    )
    checks = (
        builder.cmp("ult", builder.op("sub", output_y, plan.output_y_first), plan.output_y_count),
        builder.cmp("ult", builder.op("sub", output_x, plan.output_x_first), plan.output_x_count),
        builder.cmp("ult", ky, w.kernel[0]),
        builder.cmp("ult", kx, w.kernel[1]),
        builder.cmp("ult", channel, ci),
        builder.cmp("ult", iy, h),
        builder.cmp("ult", ix, width),
    )
    if plan.image_windows > 1:
        checks = (*checks, builder.cmp("ult", image, w.input_shape[0]))
    predicate = checks[0]
    for check in checks[1:]:
        predicate = builder.op("and", predicate, check, "i1")
    offset = builder.op("add", builder.op("mul", builder.op("sub", ix, plan.source_columns[0]), ci), channel)
    slot = builder.op("urem", unbiased_y, plan.image_slots)
    if plan.image_windows > 1:
        slot = builder.op("add", slot, builder.op("mul",
            builder.op("urem", image, plan.image_windows), plan.image_slots))
    address = builder.op(
        "add",
        builder.op(
            "add",
            plan.cache.start,
            builder.op("mul", slot, plan.row_stride_bytes),
        ),
        offset,
    )
    return {"predicate": predicate, "cache_byte": address}
