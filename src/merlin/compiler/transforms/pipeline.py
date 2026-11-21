"""Common callable transformations; verification authority belongs to the host."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import Rewriter

from merlin.compiler.transforms.analyses import (
    DependencyPlan,
    StoragePlan,
    apply_order,
    plan_dependencies,
    plan_storage,
    reuse_storage,
)
from merlin.compiler.transforms.certificates import (
    TRANSFORM_SCOPES,
    V2_KINDS,
    LoopWitness,
    TransformationChange,
    Witness,
    ir_digest,
    ir_text,
    make_change,
    parse_ir,
)
from merlin.compiler.transforms.pipeline_state import (
    AnalysisStamp,
    PassRecord,
    PassSpec,
    PipelineContext,
    PipelineSession,
    validate_pass_order,
)
from merlin.compiler.transforms.structural_rewrites import (
    constant_integer,
    normalize_shapes,
    simplify_integers,
)

# ---- loops -------------------------------------------------------------------------------------
# Ordered loop segmentation with constant-size generated IR.


@dataclass(frozen=True)
class Segment:
    lower: int
    upper: int
    step: int


def iter_segments(lower: int, upper: int, step: int, tile_iterations: int) -> Iterator[Segment]:
    """Lazy execution/reference utility; compiler emission never enumerates this."""
    if lower < 0 or upper < lower or step <= 0 or tile_iterations <= 0:
        raise ValueError("segmentation requires nonnegative bounds and positive steps/tiles")
    for start in range(lower, upper, step * tile_iterations):
        yield Segment(start, min(start + step * tile_iterations, upper), step)


def _loop_depth(op: scf.ForOp) -> int:
    depth = 0
    parent = op.parent_op()
    while parent is not None:
        depth += isinstance(parent, scf.ForOp)
        parent = parent.parent_op()
    return depth


def segment_loops(
    module: builtin.ModuleOp, tile_iterations: Sequence[int], *, index_bits: int
) -> tuple[LoopWitness, ...]:
    """Strip-mine loops by their original nesting depth, retaining exact order.

    Every iter_arg is forwarded through the inner and outer yields. Thus ordered
    reductions retain the original accumulator after each element, including
    wrapping arithmetic. Arbitrary body effects remain in their original order.
    Negative/dynamic bounds and non-index induction types are left unchanged.
    Requested static loops whose generated index arithmetic would overflow refuse.
    """
    if index_bits <= 1 or any(v <= 0 for v in tile_iterations):
        raise ValueError("positive tile iterations and an explicit index width are required")
    candidates = [
        (index, op, _loop_depth(op), ir_digest(op))
        for index, op in enumerate(module.walk())
        if isinstance(op, scf.ForOp)
    ]
    witnesses = []
    for index, loop, depth, original_digest in reversed(candidates):
        if depth >= len(tile_iterations):
            continue
        lower, upper, step = (constant_integer(v) for v in (loop.lb, loop.ub, loop.step))
        if lower is None or upper is None or step is None:
            continue
        if lower < 0 or upper < lower or step <= 0 or not isinstance(loop.lb.type, builtin.IndexType):
            continue
        tile = tile_iterations[depth]
        block_step = step * tile
        maximum = (1 << (index_bits - 1)) - 1
        # Both the bound add and final loop increments fit the selected index type.
        if upper + block_step > maximum or upper + step > maximum:
            raise ValueError("segmented loop index arithmetic exceeds the selected signed index width")
        if tile == 1:
            continue
        body = loop.body.block
        witness = LoopWitness(
            index, lower, upper, step, tile, index_bits, tuple(str(v.type) for v in loop.iter_args), original_digest
        )
        block_step_op = arith.ConstantOp.from_int_and_width(block_step, builtin.IndexType())
        outer_body = Block(arg_types=[builtin.IndexType(), *(v.type for v in loop.iter_args)])
        upper_add = arith.AddiOp(outer_body.args[0], block_step_op.result)
        upper_min = arith.MinUIOp(upper_add.result, loop.ub)
        inner_body = Block(arg_types=[v.type for v in body.args])
        mapper: dict[SSAValue, SSAValue] = dict(zip(body.args, inner_body.args, strict=True))
        inner_body.add_ops(op.clone(mapper) for op in body.ops)
        inner = scf.ForOp(outer_body.args[0], upper_min.result, loop.step, outer_body.args[1:], inner_body)
        inner.attributes.update(loop.attributes)
        outer_body.add_ops([upper_add, upper_min, inner, scf.YieldOp(*inner.results)])
        outer = scf.ForOp(loop.lb, loop.ub, block_step_op.result, loop.iter_args, outer_body)
        Rewriter.replace_op(loop, [block_step_op, outer], list(outer.results))
        witnesses.append(witness)
    return tuple(reversed(witnesses))


def ordered_loop_obligation(witness: LoopWitness) -> dict[str, int]:
    """The numerical projection consumed by a parameterized partition theorem."""
    return {
        "lower": witness.lower,
        "upper": witness.upper,
        "step": witness.step,
        "tile_iterations": witness.tile_iterations,
        "index_bits": witness.index_bits,
    }


# ---- validation --------------------------------------------------------------------------------
# Recognize actual generated structure before issuing abstract proof obligations.
#
# This is a structural guard, never an equivalence proof receipt. In particular,
# the loop recognizer checks actual IV substitution, carried state, tail bounds,
# body preservation and unrelated surrounding IR independently of code emission.


class _LoopRecognizer:
    def __init__(self, before: builtin.ModuleOp, witnesses: Sequence[LoopWitness]):
        operations = tuple(before.walk())
        self.selected = {}
        for witness in witnesses:
            if not 0 <= witness.operation_index < len(operations):
                raise ValueError("loop witness operation index is outside the actual IR")
            op = operations[witness.operation_index]
            if not isinstance(op, scf.ForOp) or ir_digest(op) != witness.body_sha256:
                raise ValueError("loop witness does not identify the actual source loop")
            if op in self.selected:
                raise ValueError("duplicate loop witness")
            self.selected[op] = witness
        self.context: dict[SSAValue, SSAValue] = {}
        self.seen: set[Operation] = set()

    def values(self, before: Sequence[SSAValue], after: Sequence[SSAValue]) -> bool:
        return len(before) == len(after) and all(self.context.get(a) is b for a, b in zip(before, after, strict=True))

    def block(self, before: Block, after: Block) -> bool:
        if tuple(a.type for a in before.args) != tuple(a.type for a in after.args):
            return False
        self.context.update(zip(before.args, after.args, strict=True))
        left, right = tuple(before.ops), tuple(after.ops)
        cursor = 0
        for op in left:
            if op in self.selected:
                if cursor + 1 >= len(right) or not self.segment(op, right[cursor], right[cursor + 1]):
                    return False
                cursor += 2
            else:
                if cursor >= len(right) or not self.operation(op, right[cursor]):
                    return False
                cursor += 1
        return cursor == len(right)

    def operation(self, before: Operation, after: Operation) -> bool:
        if (
            before.name != after.name
            or before.attributes != after.attributes
            or before.properties != after.properties
            or tuple(r.type for r in before.results) != tuple(r.type for r in after.results)
            or not self.values(before.operands, after.operands)
            or before.successors
            or after.successors
            or len(before.regions) != len(after.regions)
        ):
            return False
        self.context.update(zip(before.results, after.results, strict=True))
        for a, b in zip(before.regions, after.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                return False
            if not all(self.block(x, y) for x, y in zip(a.blocks, b.blocks, strict=True)):
                return False
        return True

    def segment(self, source: Operation, stride: Operation, target: Operation) -> bool:
        if (
            not isinstance(source, scf.ForOp)
            or not isinstance(stride, arith.ConstantOp)
            or not isinstance(target, scf.ForOp)
        ):
            return False
        witness = self.selected[source]
        actual_bounds = tuple(constant_integer(v) for v in (source.lb, source.ub, source.step))
        if actual_bounds != (witness.lower, witness.upper, witness.step):
            return False
        if witness.tile_iterations <= 1 or witness.step <= 0 or witness.lower < 0 or witness.upper < witness.lower:
            return False
        if (
            witness.index_bits <= 1
            or witness.upper + witness.step * witness.tile_iterations > (1 << (witness.index_bits - 1)) - 1
        ):
            return False
        if tuple(str(v.type) for v in source.iter_args) != witness.carried_types:
            return False
        if (
            not isinstance(stride.result.type, builtin.IndexType)
            or stride.attributes
            or constant_integer(stride.result) != witness.step * witness.tile_iterations
            or target.lb is not self.context.get(source.lb)
            or target.ub is not self.context.get(source.ub)
            or target.step is not stride.result
            or not self.values(source.iter_args, target.iter_args)
            or target.attributes
            or target.properties != source.properties
            or tuple(r.type for r in target.results) != tuple(r.type for r in source.results)
        ):
            return False
        block = target.body.block
        if tuple(v.type for v in block.args) != tuple(v.type for v in source.body.block.args):
            return False
        body = tuple(block.ops)
        if len(body) != 4:
            return False
        add, minimum, inner, outer_yield = body
        if not isinstance(add, arith.AddiOp) or not isinstance(minimum, arith.MinUIOp):
            return False
        if add.attributes or add.overflow_flags != arith.IntegerOverflowAttr("none"):
            return False
        if add.lhs is not block.args[0] or add.rhs is not stride.result:
            return False
        if minimum.attributes or minimum.lhs is not add.result or minimum.rhs is not self.context.get(source.ub):
            return False
        if not isinstance(inner, scf.ForOp) or not isinstance(outer_yield, scf.YieldOp):
            return False
        if (
            inner.lb is not block.args[0]
            or inner.ub is not minimum.result
            or inner.step is not self.context.get(source.step)
            or tuple(inner.iter_args) != tuple(block.args[1:])
            or inner.attributes != source.attributes
            or inner.properties != source.properties
            or tuple(v.type for v in inner.results) != tuple(v.type for v in source.results)
            or tuple(outer_yield.operands) != tuple(inner.results)
            or outer_yield.attributes
        ):
            return False
        self.context.update(zip(source.results, target.results, strict=True))
        self.seen.add(source)
        return self.block(source.body.block, inner.body.block)


def validate_change(change: TransformationChange) -> None:
    """Reject stale identities, forged witnesses or different actual rewrites."""
    change.validate_identity()
    if change.kind not in TRANSFORM_SCOPES or change.certificate.scope != TRANSFORM_SCOPES[change.kind]:
        raise ValueError("transformation coverage cannot be changed by candidate declarations")
    expected_schema = "merlin.shared_transform.v2" if change.kind in V2_KINDS else "merlin.shared_transform.v1"
    if change.certificate.schema != expected_schema:
        raise ValueError("transformation schema differs from its immutable kind")
    if change.kind in {"scalar_redundancy_elimination", "scalar_loop_invariant_motion"}:
        from merlin.compiler.transforms.structural_rewrites import validate_scalar_change

        validate_scalar_change(change)
        return
    if change.kind == "ordered_loop_unrolling":
        from merlin.compiler.transforms.structural_rewrites import validate_unrolled_change

        validate_unrolled_change(change)
        return
    expected_parameters = {
        "ordered_loop_segmentation": {"index_bits", "tile_iterations"},
        "sequential_dependency_schedule": {"block_index", "order"},
    }.get(change.kind, set())
    if set(change.parameters) != expected_parameters:
        raise ValueError("unexpected transformation parameters")
    before, after = parse_ir(change.before_ir), parse_ir(change.after_ir)
    if change.kind == "ordered_loop_segmentation":
        if any(not isinstance(w, LoopWitness) for w in change.certificate.witnesses):
            raise ValueError("unexpected loop witness type")
        recognizer = _LoopRecognizer(before, change.certificate.witnesses)
        if not recognizer.operation(before, after) or recognizer.seen != set(recognizer.selected):
            raise ValueError("actual IR is not the witnessed order-preserving loop segmentation")
        for witness in change.certificate.witnesses:
            if witness.index_bits != change.parameters.get("index_bits"):
                raise ValueError("loop index width does not match requested compilation parameters")
            loop = tuple(before.walk())[witness.operation_index]
            depth = 0
            parent = loop.parent_op()
            while parent is not None:
                depth += isinstance(parent, scf.ForOp)
                parent = parent.parent_op()
            tiles = change.parameters.get("tile_iterations", ())
            if depth >= len(tiles) or witness.tile_iterations != tiles[depth]:
                raise ValueError("loop tile does not match requested compilation parameters")
        return
    # These guards regenerate a structural result from current IR facts. Their
    # corresponding abstract semantics remain separate host proof obligations.
    if change.kind == "shape_view_normalization":
        witnesses = normalize_shapes(before)
    elif change.kind == "exact_integer_simplification":
        witnesses = simplify_integers(before)
    elif change.kind == "buffer_reuse":
        witnesses = reuse_storage(before, plan_storage(before))
    elif change.kind == "sequential_dependency_schedule":
        witnesses = (apply_order(before, int(change.parameters["block_index"]), change.parameters["order"]),)
    else:
        raise ValueError("unknown shared transformation kind")
    before.verify()
    if ir_text(before) != ir_text(after) or witnesses != change.certificate.witnesses:
        raise ValueError("actual IR or witnesses differ from the current structural transformation")


# ---- pipeline ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineOptions:
    index_bits: int
    tile_iterations: tuple[int, ...] = ()
    normalize_shapes: bool = True
    simplify_integers: bool = True
    reuse_buffers: bool = True
    dependency_orders: tuple[tuple[int, tuple[int, ...]], ...] = ()
    eliminate_redundancy: bool = False
    hoist_invariants: bool = False
    unroll_factors: tuple[int, ...] = ()
    max_added_operations: int = 4096
    pass_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    module: builtin.ModuleOp
    changes: tuple[TransformationChange, ...]
    storage_plan: StoragePlan
    dependency_plan: DependencyPlan
    plan_only: bool
    analyses_stamp: AnalysisStamp | None = None
    passes: tuple[PassRecord, ...] = ()


def apply_common_pipeline(
    module: builtin.ModuleOp,
    options: PipelineOptions,
    *,
    verifier: Callable[[TransformationChange], object] | None = None,
    plan_only: bool = False,
    context: PipelineContext | None = None,
) -> PipelineResult:
    """Transform a private clone; dispatch every covered change to verification.

    ``verifier`` is supplied by the host's trusted broker. This API creates no
    formal receipts and cannot turn a callback return value into proof authority.
    The broker must supply live checker completion. Without
    that hook, only explicit ``plan_only`` invocations can produce changed IR.
    No callback, successful numerical test or structural certificate proves a
    transformation by itself.
    """
    if options.index_bits <= 1 or any(tile <= 0 for tile in options.tile_iterations):
        raise ValueError("positive tiles and an explicit signed index width are required")
    module.verify()
    current = module.clone()
    changes: list[TransformationChange] = []
    session = PipelineSession(current, context=context)
    actions: dict[str, tuple[PassSpec, Callable[[], Mapping[str, object]]]] = {}

    def register(name, stage, action, *, requires=(), produces=(), mutates=True):
        actions[name] = (PassSpec(name, stage, requires, produces, ("*",) if mutates else (), mutates), action)

    def run(kind: str, action: Callable[[], tuple[Witness, ...]], parameters: dict[str, Any], scope: tuple[str, ...]):
        before = ir_text(current)
        witnesses = action()
        if not witnesses:
            if before != ir_text(current):
                raise ValueError("transformation changed IR without a structural witness")
            return {}
        if scope != TRANSFORM_SCOPES[kind]:
            raise ValueError("internal transformation coverage differs from its registry")
        change = make_change(kind, before, current, parameters, witnesses, scope)
        validate_change(change)
        if verifier is None and not plan_only:
            raise ValueError("covered transformation requires the host verifier; use plan_only for unverified planning")
        if verifier is not None:
            receipt = verifier(change)
            if not plan_only:
                from merlin.verification.lean.checker import accepted_pipeline_receipt

                if not accepted_pipeline_receipt(receipt, change):
                    raise ValueError("covered transformation lacks fresh host Lean completion for this exact change")
        changes.append(change)
        return {}

    def transform(kind, stage, action, parameters=None, requires=()):
        register(kind, stage, lambda: run(kind, action, parameters or {}, TRANSFORM_SCOPES[kind]), requires=requires)

    if options.normalize_shapes:
        transform("shape_view_normalization", 0, lambda: normalize_shapes(current))
    if options.simplify_integers:
        transform("exact_integer_simplification", 0, lambda: simplify_integers(current))
    if options.eliminate_redundancy:
        from merlin.compiler.transforms.structural_rewrites import eliminate_scalar_redundancy

        transform("scalar_redundancy_elimination", 0, lambda: eliminate_scalar_redundancy(current))
    if options.hoist_invariants:
        from merlin.compiler.transforms.structural_rewrites import hoist_scalar_invariants

        transform("scalar_loop_invariant_motion", 0, lambda: hoist_scalar_invariants(current))
    if options.tile_iterations:
        transform(
            "ordered_loop_segmentation",
            2,
            lambda: segment_loops(current, options.tile_iterations, index_bits=options.index_bits),
            {"tile_iterations": options.tile_iterations, "index_bits": options.index_bits},
        )
    if options.unroll_factors:
        from merlin.compiler.transforms.structural_rewrites import unroll_ordered_loops

        transform(
            "ordered_loop_unrolling",
            2,
            lambda: unroll_ordered_loops(
                current,
                options.unroll_factors,
                index_bits=options.index_bits,
                max_added_operations=options.max_added_operations,
            ),
            {
                "factors": options.unroll_factors,
                "index_bits": options.index_bits,
                "max_added_operations": options.max_added_operations,
            },
        )

    if options.reuse_buffers:
        # This narrow source rewrite needs the original rectangular initializer.
        # Physical allocation belongs after tiling; this is normalization of
        # already explicit, disjoint source lifetimes, not device placement.
        register(
            "storage_analysis", 1, lambda: {"storage": plan_storage(current)}, produces=("storage",), mutates=False
        )
        transform("buffer_reuse", 1, lambda: reuse_storage(current, session.analysis("storage")), requires=("storage",))
    if options.dependency_orders:
        register(
            "dependency_analysis",
            4,
            lambda: {"dependencies": plan_dependencies(current)},
            produces=("dependencies",),
            mutates=False,
        )

        def schedule():
            # Each requested order is separately witnessed against the IR at
            # that point. apply_order independently reconstructs dependencies.
            session.analysis("dependencies")
            for block_index, order in options.dependency_orders:
                run(
                    "sequential_dependency_schedule",
                    lambda: (apply_order(current, block_index, order),),
                    {"block_index": block_index, "order": order},
                    TRANSFORM_SCOPES["sequential_dependency_schedule"],
                )
            return {}

        register("sequential_dependency_schedule", 4, schedule, requires=("dependencies",))

    def final_analyses():
        # Keep the historical standalone source closure intact. Rich alias and
        # liveness analyses remain separately callable; this fact records the
        # verified current structure, not a new semantic proof.
        current.verify()
        return {
            "structure": ir_digest(current),
            "storage": plan_storage(current),
            "dependencies": plan_dependencies(current),
        }

    register("final_analyses", 5, final_analyses, produces=("structure", "storage", "dependencies"), mutates=False)
    register("final_verification", 5, lambda: {}, requires=("structure", "storage", "dependencies"), mutates=False)
    names = options.pass_order or tuple(sorted(actions, key=lambda name: actions[name][0].stage))
    if type(names) is not tuple or len(names) != len(actions) or set(names) != set(actions):
        raise ValueError("explicit pass order must name every selected pass exactly once")
    specs = tuple(actions[name][0] for name in names)
    validate_pass_order(specs)
    for name in names:
        session.run(*actions[name])
    return PipelineResult(
        current,
        tuple(changes),
        session.analysis("storage"),
        session.analysis("dependencies"),
        plan_only,
        session.stamp(),
        tuple(session.records),
    )


# ---- client ------------------------------------------------------------------------------------
# Public transport for host-owned transformations and preservation checks.
#
# Clients supply actual IR only. Semantics, coverage, proof tools and acceptance
# remain with the original host broker; returned summaries create no authority.

ACTION = "common-transform"
SCHEMA = "merlin.common_transform_request.v1"
CHECK_ACTION = "common-check"
CHECK_SCHEMA = "merlin.common_check_request.v1"
MAX_REQUEST_BYTES = 256 * 1024


def _requester_identity() -> dict:
    """The existing original broker compares process leases within one namespace."""
    try:
        tail = Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return {"pid": os.getpid(), "start_ticks": int(tail[19]), "pid_namespace": os.readlink("/proc/self/ns/pid")}
    except (OSError, ValueError, IndexError):
        return {}  # The original finite deadline remains authoritative.


def _response_identity(result, expected, *, request_id=None, optional_source=False):
    if not isinstance(result, dict):
        raise RuntimeError("invalid common transformation response")
    if request_id is not None and result.get("selfcheck_request_id") != request_id:
        raise RuntimeError("common transformation response belongs to a different request")
    for name, digest in expected.items():
        if optional_source and name == "source_sha256" and result.get(name) is None:
            continue
        if result.get(name) != digest:
            raise RuntimeError("common transformation response describes different input IR")
    return result


def _submit(workspace, payload, *, action, expected, wait_seconds, transport_command, strict):
    if type(wait_seconds) is not int or not 1 <= wait_seconds <= 540:
        raise ValueError("common transform wait must be 1..540 seconds")
    if transport_command is not None:
        if (
            not isinstance(transport_command, tuple)
            or not transport_command
            or any(not isinstance(argument, str) or not argument for argument in transport_command)
        ):
            raise ValueError("common transform requires its declared broker transport command")
        encoded = json.dumps(payload).encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("common transformation request exceeds the admitted byte budget")
        workspace = Path(workspace).resolve()
        directory = workspace / "build"
        if directory.is_symlink():
            raise ValueError("common transformation request directory cannot be linked")
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".json", delete=False) as temporary:
            temporary.write(encoded)
            temporary.flush()
            path = Path(temporary.name)
        try:
            done = subprocess.run(
                [*transport_command, action, "request=" + str(path.relative_to(workspace))],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=wait_seconds,
            )
            if done.returncode:
                return {"status": "unknown", "reason": "common transformation transport refused the request"}
            if len(done.stdout.encode()) > 2 * MAX_REQUEST_BYTES:
                raise RuntimeError("invalid common transformation response")
            result = json.loads(done.stdout)
            if isinstance(result, dict) and isinstance(result.get("stdout"), str):
                result = json.loads(result["stdout"])
            return _response_identity(result, expected, request_id=path.name if strict else None)
        except subprocess.TimeoutExpired:
            return {"status": "unknown", "reason": "common transformation transport timed out"}
        finally:
            path.unlink(missing_ok=True)
    channel = Path(workspace) / ".qa_channel"
    if channel.is_symlink() or not channel.is_dir() or (channel / "STOP").exists():
        raise RuntimeError("the admitted common transform broker is unavailable")
    request_id = f"{os.getpid()}_{time.time_ns()}_{secrets.token_hex(16)}"
    deadline = time.time_ns() + wait_seconds * 1_000_000_000
    request = {
        "protocol": 3,
        "request_id": request_id,
        "requester": _requester_identity(),
        "requested_at_unix_ns": time.time_ns(),
        "deadline_unix_ns": deadline,
        "action": action,
        "transform": payload,
    }
    encoded = json.dumps(request, sort_keys=True).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("common transformation request exceeds the admitted byte budget")
    destination, temporary = channel / f"req_{request_id}.json", channel / f".{request_id}.tmp"
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    response, done = channel / f"resp_{request_id}.json", channel / f"done_{request_id}"
    while time.time_ns() < deadline:
        if done.exists() and response.exists():
            if response.is_symlink() or response.stat().st_size > 2 * MAX_REQUEST_BYTES:
                raise RuntimeError("invalid common transformation response")
            result = json.loads(response.read_text(encoding="utf-8"))
            return _response_identity(result, expected, request_id=request_id, optional_source=not strict)
        if (channel / "STOP").exists():
            raise RuntimeError("common transformation broker stopped")
        time.sleep(0.05)
    return {"status": "unknown", "reason": "common transformation request timed out", "request_id": request_id}


def apply_transform(
    workspace: Path,
    source_ir: str,
    options: dict,
    *,
    wait_seconds: int = 120,
    transport_command: tuple[str, ...] | None = None,
) -> dict:
    """Apply admitted common passes; receipt summaries are descriptive only."""
    if not isinstance(source_ir, str) or not source_ir.strip() or not isinstance(options, dict):
        raise ValueError("common transformation requires actual IR and explicit options")
    return _submit(
        workspace,
        {"schema": SCHEMA, "source_ir": source_ir, "options": options},
        action=ACTION,
        expected={"source_sha256": hashlib.sha256(source_ir.encode()).hexdigest()},
        wait_seconds=wait_seconds,
        transport_command=transport_command,
        strict=False,
    )


def check_transform(
    workspace: Path,
    before_ir: str,
    after_ir: str,
    *,
    entry: str,
    wait_seconds: int = 120,
    transport_command: tuple[str, ...] | None = None,
) -> dict:
    """Check an actual rewrite under the frozen host's semantic and proof policy."""
    if (
        not isinstance(before_ir, str)
        or not before_ir.strip()
        or not isinstance(after_ir, str)
        or not after_ir.strip()
        or not isinstance(entry, str)
        or not entry.strip()
        or len(entry.encode()) > 1024
    ):
        raise ValueError("common check requires actual before/after IR and a bounded entry symbol")
    return _submit(
        workspace,
        {"schema": CHECK_SCHEMA, "before_ir": before_ir, "after_ir": after_ir, "entry": entry},
        action=CHECK_ACTION,
        expected={
            "source_sha256": hashlib.sha256(before_ir.encode()).hexdigest(),
            "target_sha256": hashlib.sha256(after_ir.encode()).hexdigest(),
        },
        wait_seconds=wait_seconds,
        transport_command=transport_command,
        strict=True,
    )
