"""Cost-derived OpenMP team widths for whole-model parallel regions.

ModelBlaster does not dispatch every operator to the full worker pool.  It keeps one
persistent pool and chooses among 1/2/4/8 workers from the operator geometry.  Merlin's
OpenMP lowering previously omitted ``num_threads`` on every ``omp.parallel``; consequently
all regions inherited one process-wide width, including regions too small to amortize it.

The default-off family ``parallel_team_cost_<work>`` makes that decision explicit.  Before
``convert-scf-to-openmp`` it prices each ``scf.parallel`` with the same structural cost model
as :mod:`parallel_grain`.  A region needing fewer than two workers is serialized.  Every
remaining cost is rounded up to the smallest power-of-two team that supplies at most
``work`` lane-operations per worker, capped by ``parallel_harts``.  Immediately after the
conversion, the corresponding ``omp.parallel`` receives a real ``num_threads`` operand.
OpenMP-to-LLVM lowering therefore emits ``__kmpc_push_num_threads`` at each dispatch.

The mapping fails closed: pre/post region counts must agree exactly, and unpriceable regions
use the full requested team.  Empty features leave the pass manager unsplit and the emitted
module byte-identical to the existing path.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


FEATURE_PREFIX = "parallel_team_cost_"


def feature_name(work_per_thread: int) -> str:
    work = int(work_per_thread)
    if work < 1:
        raise ValueError(f"parallel-team work per thread must be >= 1, got {work}")
    return f"{FEATURE_PREFIX}{work}"


def work_of(features) -> int | None:
    names = sorted(n for n in (features or ()) if n.startswith(FEATURE_PREFIX))
    if not names:
        return None
    if len(names) != 1:
        raise ValueError(f"{len(names)} parallel-team policies named at once ({names}); a build "
                         "has one team policy")
    try:
        work = int(names[0][len(FEATURE_PREFIX):])
    except ValueError as exc:
        raise ValueError(f"invalid parallel-team feature {names[0]!r}") from exc
    if work < 1:
        raise ValueError(f"parallel-team work per thread must be >= 1, got {work}")
    return work


def team_width(cost: int, work_per_thread: int, max_team: int) -> int:
    """Smallest power-of-two team meeting the cost target, capped by ``max_team``."""
    cost, work, cap = int(cost), int(work_per_thread), int(max_team)
    if cost < 0 or work < 1 or cap < 1:
        raise ValueError("cost must be >= 0 and work_per_thread/max_team must be >= 1")
    needed = max(1, (cost + work - 1) // work)
    width = 1
    while width < needed and width < cap:
        width = min(width * 2, cap)
    return width


def ensure_registered(work_per_thread: int) -> str:
    from .impr_features import ImprFeature, known, register

    name = feature_name(work_per_thread)
    if name not in known():
        register(ImprFeature(
            name=name,
            action_class="HEURISTIC",
            description=(
                "Cost-derived OpenMP dispatch width: serialize regions cheaper than one worker "
                f"unit ({int(work_per_thread)} lane-operations), otherwise emit an explicit "
                "num_threads operand rounded through 2/4/... up to parallel_harts. Implements "
                "the per-dispatch width policy used by persistent inference runtimes; default "
                "off and hardware-calibrated."),
        ))
    return name


# Uses `_pg_cost` and `_pg_serialize` from parallel_grain.RUNNER_PRELUDE, which is deliberately
# spliced first.  Sharing the implementation prevents the grain and team policies from silently
# assigning different costs to the same region.
RUNNER_PRELUDE = r'''
_PT_PLAN = []


def _pt_walk_named(module, wanted):
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    inner = handle.operation
                    if inner.name == wanted:
                        found.append(inner)
                    walk(inner)

    walk(module.operation)
    return found


def _pt_width(cost, work, cap):
    needed = max(1, (cost + work - 1) // work)
    width = 1
    while width < needed and width < cap:
        width = min(width * 2, cap)
    return width


def _parallel_team_plan(ctx, module):
    """Record widths in structural order and serialize regions assigned one worker."""
    global _PT_PLAN
    found = _pt_walk_named(module, "scf.parallel")
    _PT_PLAN = []
    serialized = unpriceable = 0
    histogram = {}
    for op in found:
        cost = _pg_cost(op)
        if cost is None or len(op.results):
            width = _PARALLEL_TEAM_CAP
            unpriceable += 1
        else:
            width = _pt_width(cost, _PARALLEL_TEAM_WORK, _PARALLEL_TEAM_CAP)
        if width == 1 and not len(op.results):
            _pg_serialize(op, ctx)
            serialized += 1
            continue
        _PT_PLAN.append(width)
        histogram[width] = histogram.get(width, 0) + 1
    print("OK parallel_team_plan regions", len(found), "serialized", serialized,
          "remaining", len(_PT_PLAN), "unpriceable", unpriceable,
          "widths", ",".join(str(k) + ":" + str(histogram[k]) for k in sorted(histogram)))
    return len(_PT_PLAN)


def _pt_set_num_threads(ctx, old, width):
    """Replace one result-free omp.parallel with an equivalent op carrying num_threads."""
    from torch_mlir import ir as _ptir
    if len(old.results):
        raise RuntimeError("parallel-team policy cannot replace result-carrying omp.parallel")
    block = old.regions[0].blocks[0]
    if len(block.arguments):
        raise RuntimeError("parallel-team policy found omp.parallel block arguments")
    body = [handle.operation for handle in block.operations]
    if not body or body[-1].name != "omp.terminator":
        raise RuntimeError("parallel-team policy found malformed omp.parallel region")
    with ctx, _ptir.Location.unknown(), _ptir.InsertionPoint(old):
        i32 = _ptir.IntegerType.get_signless(32)
        constant = _ptir.Operation.create(
            "arith.constant", results=[i32],
            attributes={"value": _ptir.IntegerAttr.get(i32, int(width))})
        replacement = _ptir.Operation.create(
            "omp.parallel", operands=[constant.results[0]], regions=1,
            attributes={"operandSegmentSizes":
                        _ptir.DenseI32ArrayAttr.get([0, 0, 0, 1, 0, 0])})
        target = replacement.regions[0].blocks.append()
        with _ptir.InsertionPoint(target):
            terminator = _ptir.Operation.create("omp.terminator")
    for inner in body[:-1]:
        inner.move_before(terminator)
    old.erase()


def _parallel_team_apply(ctx, module):
    found = _pt_walk_named(module, "omp.parallel")
    if len(found) != len(_PT_PLAN):
        raise RuntimeError("parallel-team pre/post OpenMP region census differs: planned "
                           + str(len(_PT_PLAN)) + ", converted " + str(len(found)))
    for op, width in zip(found, _PT_PLAN):
        _pt_set_num_threads(ctx, op, width)
    module.operation.verify()
    print("OK parallel_team_apply regions", len(found))
    return len(found)
'''


STAGE_SRC = r'''
_PARALLEL_TEAM_WORK = int(sys.argv[11]) if len(sys.argv) > 11 else 0
_PARALLEL_TEAM_CAP = int(sys.argv[12]) if len(sys.argv) > 12 else 0
if bool(_PARALLEL_TEAM_WORK) != bool(_PARALLEL_TEAM_CAP):
    raise RuntimeError("parallel-team work and cap must be supplied together")
if _PARALLEL_TEAM_WORK < 0 or _PARALLEL_TEAM_CAP < 0:
    raise RuntimeError("parallel-team work and cap must be non-negative")
if _PARALLEL_TEAM_WORK:
    _LATE_STAGES = [*_LATE_STAGES, ("parallel_team_plan", _parallel_team_plan)]
    _POST_OPENMP_STAGES = [("parallel_team_apply", _parallel_team_apply)]
else:
    _POST_OPENMP_STAGES = []
'''


def apply_for_test(mlir_text: str, work_per_thread: int, max_team: int) -> tuple[str, dict]:
    """Exercise the exact pre/post rewrite around upstream's OpenMP conversion."""
    from .parallel_grain import RUNNER_PRELUDE as grain_prelude
    from .toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_parallel_team_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        + grain_prelude + RUNNER_PRELUDE +
        "_PARALLEL_TEAM_WORK = int(sys.argv[2])\n"
        "_PARALLEL_TEAM_CAP = int(sys.argv[3])\n"
        "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f: module = ir.Module.parse(f.read(), ctx)\n"
        "_parallel_team_plan(ctx, module)\n"
        "PassManager.parse('builtin.module(convert-scf-to-openmp)', ctx).run(module.operation)\n"
        "_parallel_team_apply(ctx, module)\n"
        "print('MERLIN_TEAM ' + json.dumps({'plan': _PT_PLAN}))\n"
        "print('MERLIN_MODULE_BEGIN')\n"
        "print(module.operation)\n", encoding="utf-8")
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(int(work_per_thread)), str(int(max_team))],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"parallel-team test rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    record = next(line for line in proc.stdout.splitlines() if line.startswith("MERLIN_TEAM "))
    module = proc.stdout.split("MERLIN_MODULE_BEGIN\n", 1)[1]
    return module, json.loads(record[len("MERLIN_TEAM "):])
