"""Fork/join GRAIN for the multicore lowering: keep a parallel region only when it is worth
entering (default-off feature family ``parallel_grain_<threshold>``).

WHAT THE DEFECT IS
------------------
The multicore RVV pipeline creates parallelism in two independent places:

  * ``scf-forall-to-parallel`` -- the outer ``scf.forall`` that ``pipeline.parallel_transform_schedule``
    wraps around each named contraction, and
  * ``func.func(convert-linalg-to-parallel-loops)`` -- EVERY remaining ``linalg`` op, i.e. the whole
    elementwise / broadcast / reduction tail.

``convert-scf-to-openmp`` then gives each resulting ``scf.parallel`` its OWN ``omp.parallel`` +
``omp.wsloop``, i.e. its own ``__kmpc_fork_call``. There is no cost model anywhere on that path: a
loop that stores 32 floats gets a fork, a barrier and a join exactly like a 25-million-element
contraction does.

MEASURED, on the whole-model int8 lowerings this repo builds for an 8-hart RVV board. Counted two
ways that agree exactly: statically off the IR at the point ``convert-scf-to-openmp`` runs (weighting
each operation by its enclosing loops' static trip counts and by its vector width), and dynamically
by EXECUTING the emitted program off-board against a counting stand-in for the five ``__kmpc_*``
symbols it declares.

    model         omp.parallel   fork/join events   of those,   work in regions   share of
                  regions        per inference      NESTED      under 1e4 units   total work
    ------------  ------------   ----------------   ---------   ---------------   ----------
    lstmnetvit         960             23,344         14,340       22,700 forks      2.6 %
    deepjscc           413              5,160              0        4,101 forks      1.6 %

So ~97% (lstmnetvit) and ~79% (deepjscc) of all fork/join events in an inference are entered for a
combined 2.6% / 1.6% of the work. The Amdahl serial fraction of those builds is 1.4% and 0.001%
respectively -- parallel coverage is NOT the ceiling; the number of times the program enters a
parallel region is.

The NESTED column is the sharpest form of it. 61% of lstmnetvit's regions are entered from inside an
already-active parallel region (an `scf.parallel` from `convert-linalg-to-parallel-loops` sitting
inside the `scf.forall` the contraction was wrapped in). Under the default OpenMP nesting policy --
`max-active-levels = 1`, the default since OpenMP 5.0 -- every one of those runs with a team of ONE.
They cost a fork and a join and buy no parallelism at all, by construction, on any machine.

WHAT THIS FEATURE DOES
----------------------
Between ``convert-linalg-to-parallel-loops`` and ``convert-scf-to-openmp`` (a window the runner owns;
see ``selfcopy._run_stages``), every ``scf.parallel`` whose STATIC cost is below the threshold is
rewritten into an equivalent serial ``scf.for`` nest. ``convert-scf-to-openmp`` then never sees it,
so no fork is emitted for it and its work runs on the calling thread inside whatever region encloses
it. Everything above the threshold is untouched.

Cost is ``product(trip counts of the parallel dims) * body cost``, where the body cost counts each
operation once, weights a vector-typed result by its element count, and multiplies by the static trip
count of any enclosing ``scf.for``. It is a proxy for retired lane-operations, not for time -- which
is the honest thing to key a GRAIN decision on, since the alternative (time) is not available to a
compiler.

FAIL CLOSED. A loop whose bounds are not resolvable index constants, or whose body contains a nested
``scf.parallel`` or an op this cannot price, keeps its parallelism: the rewrite never guesses a cost.
The counts of serialized / kept / unpriceable regions are printed by the runner, so a threshold that
matched nothing is visible in the build log instead of passing for applied.

WHY IT IS A KNOB AND NOT A DEFAULT
----------------------------------
Threading is workload-dependent -- on the same board, the same 8 cores, the same session: it bought
2.54x on one model, 1.24x on a second and 0.82x (a LOSS) on a third. A grain threshold inherits that:
it trades Amdahl coverage for fork count, and which side of that trade wins is a property of the
model and the machine's fork cost, neither of which is derivable here. So the threshold is a
continuous, beam-searchable point (``parallel_grain_1000``, ``parallel_grain_10000``, ...), default
absent, and a lowering that names none is byte-identical to the baseline.

COST/CEILING, swept on the two models above (fork events per inference, and the Amdahl ceiling at
8 harts implied by the work that becomes serial). The fork counts at the 10,000 point are the
EXECUTED ones, and they match the static prediction exactly:

    threshold      lstmnetvit forks   ceiling     deepjscc forks   ceiling
    -----------    ----------------   --------    --------------   --------
    (none)              23,344          7.51x          5,160         8.00x
    1,000                7,219          7.20x          5,016         8.00x
    10,000                 749          6.87x          1,059         7.18x
    100,000                397          6.15x             98         5.54x

At 10,000 the nested-fork count goes 14,340 -> 280 on lstmnetvit, i.e. most of what disappears is
the class of region that could never have run in parallel in the first place.

NOT MEASURED ON HARDWARE. The fork/join cost of this board's libomp is not known to this module, so
which threshold (if any) wins is an open question a board run has to answer. Nothing here claims a
speedup. What IS verified off-board, by executing the emitted program at 1 and 8 threads, repeated:
the output is bit-identical with the feature on and off, and identical to the serial (single-hart)
build's output.

RACE ANALYSIS
-------------
The rewrite only ever REMOVES concurrency: an ``scf.parallel`` whose iterations were already required
to be independent (that is what the op means, and it is what ``convert-linalg-to-parallel-loops``
asserts about a ``linalg`` op's parallel iterators) is executed in sequential index order instead of
in an arbitrary interleaving. Any schedule the serial nest produces is one of the schedules the
parallel op already permitted, so no new write-sharing is introduced and no ordering guarantee is
weakened. The converse direction -- turning a serial loop parallel -- is the one that needs a
dependence proof, and this feature never does it. ``scf.parallel`` ops that carry REDUCTIONS (results
/ a non-empty ``scf.reduce``) are refused outright rather than serialized, because their terminator
would have to be rewritten too. A region CONTAINING a nested ``scf.parallel`` is priced as
unpriceable and left alone -- the inner one is a candidate on its own terms, but collapsing the
outer one would decide the inner one's fate as a side effect.
"""
from __future__ import annotations

#: Feature-name prefix. The full name carries the threshold: ``parallel_grain_10000``.
FEATURE_PREFIX = "parallel_grain_"


def feature_name(threshold: int) -> str:
    """The feature name for ``threshold`` lane-operations."""
    t = int(threshold)
    if t < 1:
        raise ValueError(f"parallel-grain threshold must be >= 1, got {t}")
    return f"{FEATURE_PREFIX}{t}"


def threshold_of(features) -> int | None:
    """The threshold named by ``features``, or None when the family is absent.

    Two different thresholds in one feature set is a configuration error, not a merge: the two
    describe incompatible grains and picking either silently would make the build unattributable.
    """
    names = sorted(n for n in (features or ()) if n.startswith(FEATURE_PREFIX))
    if not names:
        return None
    if len(names) > 1:
        raise ValueError(f"{len(names)} parallel-grain thresholds named at once ({names}); a build "
                         "has one grain")
    return int(names[0][len(FEATURE_PREFIX):])


def ensure_registered(threshold: int) -> str:
    """Register (idempotently) the grain point for ``threshold`` and return its name.

    Registered from HERE and re-registered on demand from the name, because the lowering runs in a
    child process that re-imports the registry: a point registered only in the parent fails to
    resolve in the child.
    """
    from .impr_features import ImprFeature, known, register
    name = feature_name(threshold)
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="HEURISTIC",
        description=(
            f"Multicore fork/join GRAIN: rewrite every `scf.parallel` whose static cost is below "
            f"{int(threshold)} lane-operations into a serial `scf.for` nest, between "
            f"convert-linalg-to-parallel-loops and convert-scf-to-openmp, so no `omp.parallel` "
            f"(and no __kmpc_fork_call) is emitted for it. Attacks the fork COUNT, not the Amdahl "
            f"fraction: the measured whole-model int8 lowerings enter 23,344 (lstmnetvit) and 5,160 "
            f"(deepjscc) parallel regions per inference, of which 97%/79% carry 2.6%/1.6% of the "
            f"work. Only ever removes concurrency, so it introduces no write-sharing. Requires the "
            f"multicore lowering (parallel_harts); with a serial pipeline there is no `scf.parallel` "
            f"and it reports 0. NOT MEASURED ON HARDWARE."),
    ))
    return name


# --------------------------------------------------------------------------------------------------
# The runner half. Executes in the m2m venv (which owns the MLIR Python bindings), spliced into every
# lowering-runner variant. Defines `_parallel_grain(ctx, module)`; `_run_stages` calls it in the LATE
# window -- after the forall/linalg -> scf.parallel conversions have created the regions, and before
# convert-scf-to-openmp turns each one into a fork.
# --------------------------------------------------------------------------------------------------
RUNNER_PRELUDE = r'''
_PG_TERMINATORS = frozenset((
    "scf.yield", "scf.reduce", "scf.reduce.return", "scf.condition",
    "memref.alloca_scope.return", "omp.yield", "omp.terminator", "func.return", "cf.br"))


def _pg_const_index(value):
    """The integer of an index-typed `arith.constant` defining `value`, else None (fail closed)."""
    from torch_mlir import ir as _pgir
    try:
        owner = value.owner.operation
    except AttributeError:                      # a block argument has no defining operation
        return None
    if owner.name != "arith.constant":
        return None
    try:
        return int(_pgir.IntegerAttr(owner.attributes["value"]))
    except Exception:                           # noqa: BLE001 - not an integer attribute
        return None


def _pg_trip(lb, ub, step):
    """ceil((ub - lb) / step) when all three are index constants and step > 0, else None."""
    l, u, s = _pg_const_index(lb), _pg_const_index(ub), _pg_const_index(step)
    if l is None or u is None or s is None or s <= 0:
        return None
    return max(0, -(-(u - l) // s))


def _pg_lanes(op):
    """Lane count of the widest vector result of `op` (1 for a scalar op).

    A vector operation retires one instruction but does that many lanes of work; pricing it as 1
    would make a vectorized region look 16x cheaper than the scalar spelling of the same work."""
    from torch_mlir import ir as _pgir
    n = 1
    for res in op.results:
        try:
            vt = _pgir.VectorType(res.type)
        except Exception:                       # noqa: BLE001 - not a vector type
            continue
        c = 1
        for d in vt.shape:
            c *= int(d)
        n = max(n, c)
    return n


def _pg_body_cost(block):
    """Static lane-operation cost of `block`, or None when something in it cannot be priced."""
    total = 0
    for handle in block.operations:
        op = handle.operation
        name = op.name
        if name in _PG_TERMINATORS:
            continue
        if name == "scf.parallel":
            return None                          # a nested parallel: leave the outer one alone
        if name == "scf.for":
            trip = _pg_trip(op.operands[0], op.operands[1], op.operands[2])
            if trip is None:
                return None
            inner = _pg_body_cost(op.regions[0].blocks[0])
            if inner is None:
                return None
            total += trip * inner
            continue
        if len(op.regions):
            sub = 0
            for region in op.regions:
                for blk in region.blocks:
                    cost = _pg_body_cost(blk)
                    if cost is None:
                        return None
                    # `scf.if` takes ONE of its regions; everything else (alloca_scope, ...) all.
                    sub = max(sub, cost) if name == "scf.if" else sub + cost
            total += sub + 1
            continue
        total += _pg_lanes(op)
    return total


def _pg_cost(op):
    """Static cost of one `scf.parallel`: parallel trip product * body cost. None => unpriceable."""
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    if rank == 0 or len(operands) < 3 * rank:
        return None
    trip = 1
    for i in range(rank):
        t = _pg_trip(operands[i], operands[rank + i], operands[2 * rank + i])
        if t is None:
            return None
        trip *= t
    body = _pg_body_cost(block)
    if body is None:
        return None
    return trip * body


def _pg_serialize(op, ctx):
    """Rewrite one reduction-free `scf.parallel` into an equivalent serial `scf.for` nest."""
    from torch_mlir import ir as _pgir
    from torch_mlir.dialects import scf as _pgscf
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    with ctx, _pgir.Location.unknown():
        point = _pgir.InsertionPoint(op)
        ivs = []
        terminator = None
        for i in range(rank):
            with point:
                loop = _pgscf.ForOp(operands[i], operands[rank + i], operands[2 * rank + i])
            ivs.append(loop.induction_variable)
            with _pgir.InsertionPoint(loop.regions[0].blocks[0]):
                terminator = _pgscf.YieldOp([])
            point = _pgir.InsertionPoint(terminator)
    for i in range(rank):
        block.arguments[i].replace_all_uses_with(ivs[i])
    # Everything but the `scf.reduce` terminator moves, in order, into the innermost body.
    for handle in [o.operation for o in block.operations][:-1]:
        handle.move_before(terminator.operation)
    op.erase()


def _parallel_grain(ctx, module):
    """Serialize every `scf.parallel` cheaper than `_PARALLEL_GRAIN` lane-operations.

    Returns the number serialized; prints the kept / unpriceable / reduction-carrying counts so a
    threshold that matched nothing is visible instead of passing for applied."""
    if not _PARALLEL_GRAIN:
        return 0
    found = []

    def _walk(op):
        for region in op.regions:
            for blk in region.blocks:
                for handle in blk.operations:
                    inner = handle.operation
                    if inner.name == "scf.parallel":
                        found.append(inner)
                    _walk(inner)

    _walk(module.operation)
    serialized = kept = unpriceable = reducing = 0
    for op in found:
        if len(op.results):                      # carries a reduction: refuse rather than rewrite
            reducing += 1
            continue
        cost = _pg_cost(op)
        if cost is None:
            unpriceable += 1
            continue
        if cost < _PARALLEL_GRAIN:
            _pg_serialize(op, ctx)
            serialized += 1
        else:
            kept += 1
    print("OK parallel_grain threshold", _PARALLEL_GRAIN, "regions", len(found),
          "serialized", serialized, "kept", kept, "unpriceable", unpriceable,
          "reducing", reducing)
    return serialized
'''

#: Spliced after :data:`RUNNER_PRELUDE`: reads the threshold off argv and builds the LATE stage list.
LATE_STAGE_SRC = r"""
_PARALLEL_GRAIN = int(sys.argv[9]) if len(sys.argv) > 9 else 0
_LATE_STAGES = [("parallel_grain", _parallel_grain)] if _PARALLEL_GRAIN else []
"""
