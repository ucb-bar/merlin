"""Apply an RVV package's codegen knobs to a workload build, via the existing build_app seam.

This is the plug-back-in point: an isolated :class:`HostPackage` supplies its transform schedule
and cflags into ``build_app(rvv_schedule=..., cflags_override=...)`` — WITHOUT touching the
``pipeline.RVV_TRANSFORM_SCHEDULE`` / ``RVV_CFLAGS`` module defaults. ``apply_rvv_package(hand_v0)``
is therefore byte-identical to today's ``build_app(backend="rvv")`` (asserted by test_rvv_package).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.compiler.backend.host import HostPackage, load_host_package
from merlin.runtime.backends.zephyr import application as zm
from merlin.runtime.backends.zephyr import image_build as zephyr_build

# ---- apply -------------------------------------------------------------------------------------


class _SplitShape:
    """A contraction shape whose parallel extents are what ONE HART actually sees."""

    __slots__ = ("op", "parallel", "reduction")

    def __init__(self, op: str, parallel, reduction=()):
        self.op = op
        self.parallel = tuple(parallel)
        self.reduction = tuple(reduction)


def _harts_split_shapes(shapes, harts: int) -> list:
    """``shapes`` re-expressed as the per-hart tiles a multicore build actually blocks.

    The multicore stage wraps each contraction in an ``scf.forall`` BEFORE the package's schedule
    runs (see ``pipeline.parallel_transform_schedule``), splitting ``linalg.matmul`` over N with
    ``tile_using_forall ... num_threads``. So the register block has to cover ``ceil(N / harts)`` and
    the smaller remainder tile — NOT the whole N. Choosing the block from the unsplit extents is how
    ``--harts 3`` on a 2-wide N ended up masking a parallel dim and failing to lower, while the same
    model built fine at 1 and 4 harts (N/4 of an even N stays even).

    ``batch_matmul`` splits over its BATCH dim, which is not part of the (M, N) block, so its shapes
    pass through unchanged.
    """
    if harts < 2:
        return list(shapes)
    out = []
    for s in shapes:
        par = list(s.parallel)
        if s.op != "linalg.matmul" or len(par) < 2:
            out.append(s)
            continue
        n = int(par[-1])
        tile = -(-n // harts)  # ceil: the largest tile any hart gets
        rem = n - tile * (harts - 1)  # the last hart's tile (may be <= 0 = idle)
        for extent in {tile, rem} if rem > 0 else {tile}:
            out.append(_SplitShape(s.op, [*par[:-1], extent], getattr(s, "reduction", ())))
    return out


def shape_adapted_features(pkg: HostPackage, model_dir: str | Path, *, harts: int = 1) -> list[str]:
    """``pkg``'s compiler features re-resolved against the CONTRACTIONS ``model_dir`` actually has.

    This is the seam where a package and a workload first meet, so it is where a register block
    stops being an abstract knob and becomes a claim about specific extents. A package pins one
    ``microkernel`` block; a whole model carries many matmul shapes, and a block that masks a
    parallel dim of any one of them fails to lower on the int8 path (and degrades ~34x on fp32) —
    see ``from_strategy._rvv_blocking_lowers`` for the measured predicate. Re-resolving here reads
    the pinned block as an upper BOUND and derives the largest legal one per op class.

    Returns the package's features unchanged when the model has no readable contractions, so an
    unobservable workload degrades to the shape-blind behavior rather than failing.

    A package may pin its block one of TWO ways, and both are adapted here. A ``microkernel`` knob
    block goes through the target-agnostic space. A HAND-FROZEN named point (the champion's
    ``accumulator_resident_wholemodel_vf``) carries its block as constants inside the feature, so
    it is translated to the equivalent caps first -- otherwise the one package anyone actually
    compiles with is the one package that cannot adapt, and a workload whose extents the frozen
    tails do not fit fails to lower with a masked-parallel-dim PipelineError.

    ``harts`` is the hart count the image will be built for. It matters because the multicore stage
    splits each matmul over N BEFORE this schedule runs, so what the block must cover is the per-hart
    tile, not the model's N — see :func:`_harts_split_shapes`."""
    provider = getattr(pkg, "codegen_provider", None)
    if provider is not None:
        return list(provider.shape_adapted_features(pkg, model_dir, harts=harts))

    from merlin.compiler.backend.host import resolve_features
    from merlin.compiler.ir.contraction_shapes import contraction_shapes

    shapes = contraction_shapes(Path(model_dir) / "model.mlir")
    if not shapes:
        return list(pkg.compiler_features)
    shapes = _harts_split_shapes(shapes, int(harts))
    feats = resolve_features(pkg.knobs, pkg.manifest, shapes=shapes)
    return _adapt_frozen_points(feats, shapes, target=str(pkg.manifest.get("target", "rvv")))


def _schedule_pinned_blocks(schedule_text: str) -> dict[str, tuple[int, int]]:
    """``{op class: (M tile, N tile)}`` a transform SCHEDULE TEXT pins, read structurally.

    Some packages carry no compiler feature at all — their register block lives in the schedule's
    ``tile_sizes`` literals. Those are invisible to :func:`shape_adapted_features` (there is no
    feature to re-resolve), so a block that masks a parallel dim of THIS workload cannot be detected
    the same way. Reading the sizes back out is what lets the caller at least SAY so.

    Parsed by following handles, not by pattern-matching a fixed spelling: a ``match ops{["X"]}``
    binds a handle, and a later ``tile_using_for <handle> tile_sizes [...]`` gives that class's tile.
    Unparseable text yields ``{}`` (the caller then stays silent rather than guessing).
    """
    handles: dict[str, str] = {}
    blocks: dict[str, tuple[int, int]] = {}
    for line in schedule_text.splitlines():
        s = line.strip()
        if "transform.structured.match" in s and 'ops{["' in s and "=" in s:
            handle = s.split("=", 1)[0].strip().split()[-1]
            cls = s.split('ops{["', 1)[1].split('"', 1)[0]
            handles[handle] = cls
        elif "tile_using_for" in s and "tile_sizes" in s:
            after = s.split("tile_using_for", 1)[1]
            handle = after.strip().split()[0]
            cls = handles.get(handle)
            if cls is None:
                continue
            nums = after.split("tile_sizes", 1)[1].split("[", 1)[1].split("]", 1)[0]
            try:
                sizes = [int(t.strip()) for t in nums.split(",")]
            except ValueError:
                continue
            # matmul tiles [M, N, K]; batch_matmul tiles [B, M, N, K] -> take the two parallel tiles
            pos = 1 if len(sizes) >= 4 else 0
            if len(sizes) > pos + 1 and cls not in blocks:
                mt, nt = sizes[pos], sizes[pos + 1]
                if mt > 0 and nt > 0:
                    blocks[cls] = (mt, nt)
    return blocks


def blocking_risks(pkg: HostPackage, model_dir: str | Path) -> list[str]:
    """Op classes whose extents this package's SCHEDULE-PINNED block would mask, as messages.

    A masked parallel dim is not a style question: it fails to lower outright on the integer path and
    degrades ~34x on fp32 (see ``from_strategy._rvv_blocking_lowers``). When the block comes from a
    feature we re-derive it; when it is baked into the schedule text we cannot, so the least we owe
    the caller is to name the risk instead of letting a 34x slowdown look like the model being slow.
    Empty when the package has an adaptable feature, when nothing is masked, or when the schedule is
    unreadable.
    """
    from merlin.analysis.mining.from_strategy import _rvv_blocking_lowers
    from merlin.compiler.backend import providers as provided
    from merlin.compiler.ir.contraction_shapes import contraction_shapes

    blocks = _schedule_pinned_blocks(pkg.schedule_text)
    if not blocks:
        return []
    is_frozen_block = provided.component("frozen_blocks").is_frozen_block
    if any(is_frozen_block(f) for f in pkg.compiler_features):
        return []  # the resolver handles this one
    shapes = contraction_shapes(Path(model_dir) / "model.mlir")
    out: list[str] = []
    for op, (mt, nt) in blocks.items():
        bad = [
            (m, n)
            for s in shapes
            if s.op == op and len(s.parallel) >= 2
            for m, n in [(s.parallel[-2], s.parallel[-1])]
            if not _rvv_blocking_lowers(mt, nt, m, n)
        ]
        if bad:
            uniq = sorted(set(bad))[:4]
            out.append(
                f"{op} block [{mt}, {nt}] masks a parallel dim of {len(bad)} contraction(s) "
                f"(e.g. extents {uniq}) — that fails to lower on int8 and costs ~34x on fp32"
            )
    return out


def _adapt_frozen_points(feats: list[str], shapes, *, target: str) -> list[str]:
    """Re-derive a hand-frozen register block ONLY for the op classes where it cannot lower.

    Deliberately minimal. Adapting a class whose frozen block already lowers would change the
    emitted code for workloads that work today -- including the model the block was tuned on,
    whose measurements other people's results depend on. So each contraction op class is checked
    against the measured predicate independently, the frozen block is kept wherever it holds, and
    only a class that would mask a parallel dim (and therefore fail to lower, not merely run slow)
    is replaced by the largest block legal for its own extents.

    Emits a per-op-class feature when anything changed, and returns the list unchanged otherwise,
    so a workload the frozen point already fits compiles byte-identically.
    """
    from merlin.analysis.mining.from_strategy import _rvv_best_block, _rvv_blocking_lowers
    from merlin.compiler.backend import providers as provided

    frozen_blocks = provided.component("frozen_blocks")
    frozen_block_caps, frozen_block_per_class = frozen_blocks.frozen_block_caps, frozen_blocks.frozen_block_per_class
    out: list[str] = []
    for name in feats:
        caps, frozen = frozen_block_caps(name), frozen_block_per_class(name)
        if caps is None or frozen is None:
            out.append(name)
            continue
        chosen: dict[str, tuple[int, int] | None] = {}
        changed = False
        for op, block in frozen.items():
            ext = [(s.parallel[-2], s.parallel[-1]) for s in shapes if s.op == op and len(s.parallel) >= 2]
            if not ext or all(_rvv_blocking_lowers(block[0], block[1], m, n) for m, n in ext):
                chosen[op] = block
                continue
            best = _rvv_best_block(int(caps["MR"]), int(caps["NR"]), ext)
            # A 1-lane N block is not a vectorization: it buys no lanes, and vectorizing that tile
            # emits a parallel-dim-free `vector.contract` (vector<1xT> dot into a scalar) that no
            # lowering strategy matches, so the build dies late. Leave the class un-tiled instead —
            # its contractions go through convert-linalg-to-loops (scalar), which is correct, and the
            # OTHER class keeps its vectors. Reported by the caller, never silent.
            chosen[op] = None if best[1] <= 1 else best
            changed = True
        if not changed:
            out.append(name)
            continue
        from merlin.compiler.optimizations.register_blocking import ensure_v3_perop_microkernel

        mm, bmm = chosen["linalg.matmul"], chosen["linalg.batch_matmul"]
        if mm is None and bmm is None:  # nothing vectorizable -> keep the pinned point as-is
            out.append(name)
            continue
        out.append(ensure_v3_perop_microkernel(*(mm or (None, None)), *(bmm or (None, None)), int(caps["KC"])))
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def apply_rvv_package(
    pkg: HostPackage | str | Path,
    model_dir: str | Path,
    work: str | Path,
    *,
    board: str | None = None,
    harts: int = 2,
    int8_compute: bool | None = None,
    shape_adapt: bool = False,
    **kw: Any,
) -> dict:
    """Build the Zephyr image for ``model_dir`` using THIS package's codegen knobs.

    The package owns the RVV-specific cflags; ``_CFLAGS_COMMON`` is appended here. ``dtype_strategy``
    selects an EXISTING lowering path (``int8_w8a8`` -> ``int8_compute=True`` -> passes_quant_int;
    ``bf16_f32acc`` -> the existing bf16 f32-accumulate path) — no new lowering logic. Returns the
    ``build_app`` dict (elf, app_dir, build_dir, ram_bytes, ...); ``model.o`` lives at ``work/model.o``.

    ``shape_adapt=True`` (default OFF -> byte-identical to today) resolves the package's
    ``microkernel`` block against THIS workload's contraction shapes instead of pinning it — see
    :func:`shape_adapted_features`. Opt-in because it changes the emitted schedule for workloads the
    pinned block does not fit, which is a measurement-visible change.
    """
    if board is None:
        raise ValueError("apply_rvv_package requires a board from MERLIN_BOARD_CATALOG")
    if not isinstance(pkg, HostPackage):
        pkg = load_host_package(pkg)
    if int8_compute is None:
        int8_compute = pkg.is_int8
    feats = shape_adapted_features(pkg, model_dir) if shape_adapt else list(pkg.compiler_features)
    return zephyr_build.build_app(
        model_dir,
        work,
        board=board,
        backend="rvv",
        rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        int8_compute=int8_compute,
        cpus=max(harts, 1),
        features=frozenset(feats) or None,
        **kw,
    )


# ---- lever_applicability -----------------------------------------------------------------------
# Can this lever reach the IR at all? — a preflight the search runs before spending board time.
#
# A transform-dialect lever finds its work with ``transform.structured.match ops{["linalg.matmul"]}``.
# When the module contains none of the ops it names, the match yields an EMPTY handle and every op
# downstream of it becomes a vacuous no-op: the lever builds, runs, gates clean, and changes nothing.
# It reports as applied.
#
# That is not hypothetical. ``passes_quant_int`` rewrote every ``linalg.matmul`` into a
# ``linalg.generic`` (measured: 15 -> 0 on ``small_llama_int8``), while ``impr_features`` matches
# ``linalg.matmul``/``linalg.batch_matmul`` in 39 places -- so the entire register-blocking and
# accumulator-resident family was inert on the int8 datapath. An 87-fork beam search over exactly
# those levers emitted 20 distinct binaries, flagged 34 nodes inert, and could not improve on the two
# levers that happen to work at the ``linalg.generic`` level. Every one of those forks cost board time.
#
# The check is deliberately STRUCTURAL and target-agnostic: it reads the op names out of the lever's
# own schedule text and counts those ops in the prepared module. It knows nothing about which ops
# matter, so it keeps working when a new lever or a new dialect appears.

_MATCH_TOKEN = "transform.structured.match"
_OPS_OPEN = "ops{["
_OPS_CLOSE = "]}"


def _match_handles(schedule_text: str) -> dict[str, tuple[str, ...]]:
    """``{result handle -> op names}`` for every ``transform.structured.match ops{[...]}`` line."""
    out: dict[str, tuple[str, ...]] = {}
    for line in (schedule_text or "").splitlines():
        if _MATCH_TOKEN not in line or _OPS_OPEN not in line:
            continue
        lhs, _, _rhs = line.partition("=")
        handle = lhs.strip().split()[0].strip() if "%" in lhs else ""
        open_at = line.find(_OPS_OPEN)
        close_at = line.find(_OPS_CLOSE, open_at)
        if close_at < 0:
            continue
        names = tuple(
            sorted(
                {
                    t.strip().strip('"').strip()
                    for t in line[open_at + len(_OPS_OPEN) : close_at].split(",")
                    if t.strip().strip('"').strip()
                }
            )
        )
        if handle and names:
            out[handle] = names
    return out


def _payload_handles(schedule_text: str) -> set[str]:
    """Handles a STRUCTURED transform actually consumes -- i.e. the lever's real work targets.

    A schedule also matches container ops it never transforms: ``%f = ...match ops{["func.func"]}``
    exists only to receive ``transform.apply_patterns to %f``. Counting those as work targets is
    what made the first version of this check report ``applicable`` for the exact case it was
    written to catch -- ``func.func`` is always present, so a lever whose every payload op had
    vanished still looked fine. Payload-ness is read off the schedule's own dataflow rather than
    from a list of container op names, which would go stale.
    """
    used: set[str] = set()
    for line in (schedule_text or "").splitlines():
        stripped = line.strip()
        if "transform.structured." not in stripped or _MATCH_TOKEN in stripped:
            continue
        _lhs, _, rhs = stripped.partition("=")
        body = rhs if _ else stripped
        for tok in body.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if tok.startswith("%"):
                used.add(tok.split(":")[0].strip())
    return used


def matched_op_names(schedule_text: str) -> tuple[str, ...]:
    """The op names this schedule actually TRANSFORMS, parsed structurally (no regex).

    De-duplicated and sorted, and restricted to handles a structured transform consumes, so a
    container match (``func.func`` for ``apply_patterns``) does not count as a work target. A
    schedule that matches by interface or attribute rather than by name yields an empty tuple, which
    this module treats as "cannot be judged" -- never as "inapplicable", because refusing a lever we
    cannot analyse would be worse than running it.
    """
    handles = _match_handles(schedule_text)
    payload = _payload_handles(schedule_text)
    names: set[str] = set()
    for handle, ops in handles.items():
        if handle in payload:
            names.update(ops)
    return tuple(sorted(names))


def all_matched_op_names(schedule_text: str) -> tuple[str, ...]:
    """Every op name the schedule matches, including containers it only reads. Diagnostic only."""
    names: set[str] = set()
    for ops in _match_handles(schedule_text).values():
        names.update(ops)
    return tuple(sorted(names))


def module_op_counts(module) -> dict[str, int]:
    """How many of each op the prepared module actually contains."""
    counts: dict[str, int] = {}
    for op in module.walk():
        counts[op.name] = counts.get(op.name, 0) + 1
    return counts


def applicability(schedule_text: str, op_counts: dict[str, int]) -> dict:
    """``{status, needs, present, reason}`` for one lever against one prepared module.

    ``status`` is ``applicable`` | ``inapplicable`` | ``unknown``. ``unknown`` means the schedule
    does not match by op name, so this check has nothing to say -- reported honestly rather than
    guessed, on the same fail-open terms the rest of the search uses for what it cannot measure.
    """
    needs = matched_op_names(schedule_text)
    if not needs:
        return {
            "status": "unknown",
            "needs": (),
            "present": {},
            "reason": "schedule does not match by op name; applicability not decidable here",
        }
    present = {n: int(op_counts.get(n, 0)) for n in needs}
    if any(v > 0 for v in present.values()):
        return {"status": "applicable", "needs": needs, "present": present, "reason": ""}
    return {
        "status": "inapplicable",
        "needs": needs,
        "present": present,
        "reason": (
            f"the module contains none of {list(needs)}, so every "
            f"transform.structured.match on them yields an empty handle and the lever "
            f"is a no-op; it would still build, gate clean and report as applied"
        ),
    }


def inapplicable_features(features, op_counts: dict[str, int]) -> dict[str, dict]:
    """``{feature name -> applicability}`` for every named feature that CANNOT fire on this module.

    Resolves each feature's own schedule text through :mod:`merlin.compiler.optimizations.registry`, so it
    stays correct as levers are added and needs no table of its own. Features with no schedule, or
    whose schedule matches by interface rather than by name, are omitted -- this reports only what it
    can actually establish, on the same terms as :func:`applicability`.
    """
    from merlin.compiler.optimizations.registry import get, known

    out: dict[str, dict] = {}
    for name in sorted(set(features or ())):
        if name not in known():
            continue
        edit = getattr(get(name), "edit_schedule", None)
        if edit is None:
            continue
        try:
            text = edit("")
        except Exception:  # a schedule we cannot render, we cannot judge
            continue
        verdict = applicability(text, op_counts)
        if verdict["status"] == "inapplicable":
            out[name] = verdict
    return out
