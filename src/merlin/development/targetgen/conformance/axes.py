"""What a target's whole-model corpus MUST cover, derived — never typed.

A corpus of hand-picked whole-model capsules describes whichever architectures somebody happened to
author. Radiance's did exactly that: the only model capsule ever graded is a LLaMA decoder, so its
captured linalg carries no ``tanh``, no ``erf`` and no convolution, and nothing in the scorecard said the
evidence was one architecture wide. Widening it by hand-writing "the corpus shall contain these eleven
op families" trades one overfit for another: the list is stale the moment a target or a target-model is
added, and it cannot report what it is missing.

So the REQUIREMENT is computed from three independent sources, and the corpus is measured against it:

``admitted``
    what the silicon declares it can compute — the target's capability manifest
    (:func:`merlin.target.capabilities.eligibility.capability_map_for_target`), family x dtype x rank. A target
    with no fp8 datapath does not get fp8 cells.

``observed``
    what the real target-models actually contain — the family census of each captured model
    (:func:`merlin.development.targetgen.conformance.model_coverage.regions_from_module`). A family no real model uses is not
    required of the corpus, however capable the hardware is.

``boundaries``
    where the hardware's own edges are — the mesh/array edge and the MX block-scale group, read from the
    target's facts. Edge cases are generated at those boundaries rather than imagined.

The required set is ``admitted INTERSECT observed``, expressed in the SAME cell vocabulary
:func:`merlin.development.targetgen.corpus.materialization.cert_capsule_cover` already uses — ``(semantic_family,
dtype, tile_alignment)`` — so the existing cover machinery measures conformance for free and the two can
never disagree about what a cell is.

Every cell carries its provenance: which capture observed the family, which compute unit admitted the
dtype, or ``DECLARED`` plus a citation where a target-model could not be captured at all. A declared cell
is never silently indistinguishable from an observed one — that distinction is the whole point of
deriving, and DeepSeek-R1-Distill-Qwen has no capture, so its column can only be declared until a
``model2MLIR`` loader for it exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from merlin.development.targetgen.applications.demand_inventory import application_demand_inventory
from merlin.development.targetgen.conformance.gaps import _scope_gap
from merlin.development.targetgen.conformance.requirements import (
    DECLARED,
    OBSERVED,
    OBSERVED_VIA_PRIMITIVES,
    Boundaries,
    Cell,
    CellOrigin,
)

# ---- conformance -------------------------------------------------------------------------------


def _capsule_geometry(cap: dict) -> str | None:
    """The geometry class of a capsule's own contraction extents, or ``None`` when it has none.

    Read off the DECLARED inputs, which is what the harness materialises -- the same source
    ``cert_capsule_cover`` classifies alignment from. A capsule with two rank-2 inputs sharing an inner
    extent is a contraction: ``A0[M, K]`` and ``W[K, N]``. Anything else (one input, a rank-4 image, a
    broadcast row) is not a contraction and returns ``None`` rather than a class it does not have.
    """
    from merlin.capture import demands as ST

    shapes = [
        [int(x) for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()] for t in (cap.get("inputs") or [])
    ]
    twod = [sh for sh in shapes if len(sh) == 2 and all(d > 0 for d in sh)]
    for i, a in enumerate(twod):
        for j, b in enumerate(twod):
            if i != j and a[1] == b[0]:
                return ST.classify_geometry(M=a[0], N=b[1], K=a[1])
    return None


def _geometry_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required geometry classes no capsule in the corpus presents."""
    import yaml

    from merlin.development.targetgen.conformance.requirements import _capsule_paths

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    have: dict[str, list[str]] = {}
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude:
                continue
            klass = _capsule_geometry(cap)
            if klass:
                have.setdefault(klass, []).append(name)
    want = [str(r.get("class")) for r in (required or ()) if r.get("class")]
    missing = sorted(set(want) - set(have))
    by_class = {str(r.get("class")): r for r in (required or ()) if r.get("class")}
    return {
        "status": "ok",
        "n_required": len(set(want)),
        "n_covered": len(set(want) & set(have)),
        "uncovered": missing,
        "covered_by": {k: sorted(v) for k, v in sorted(have.items())},
        "mac_fraction_uncovered": round(sum(float(by_class[k].get("mac_fraction") or 0.0) for k in missing), 6),
        "note": (
            "a geometry class real models present that no capsule reproduces means the corpus "
            "cannot tell a compiler that tiles that aspect ratio well from one that does not; "
            "`mac_fraction_uncovered` is the share of real contraction work sitting in classes "
            "nothing tests"
        ),
    }


def _builder_epilogue_stages() -> tuple[str, ...]:
    """Epilogue stages the capsule builder can actually emit. Not a guess about hardware -- it is the
    writer's vocabulary, and a stage outside it could be required and never written.

    READ FROM THE BUILDER, not re-typed here. This was a local tuple annotated as mirroring
    ``corpus_synth._EPILOGUE_STAGES``, which was annotated as mirroring this one, and the pair drifted
    from the builders they describe: ``requant`` was in the ABI, the schema, all three engines and the
    interface verifier, and in neither copy -- so the one epilogue stage an external hard-kernel campaign
    actually found dropped could not be demanded by any capsule.
    """
    from merlin.development.targetgen.conformance.requirements import _cs

    return _cs().BUILDER_EPILOGUE_STAGES


def scope_axis(
    captures: dict[str, str | Path], target: str, *, min_occurrences: int = 4, max_signatures: int = 12
) -> dict:
    """WHICH ADJACENCIES a target's own captures present -- the unit a phase-2 obligation is made of.

    Every other axis here describes ONE region: its family, its dtype, its extents, its geometry. That is
    the right unit for a correctness obligation, because a compiler gets one op wrong at a time. It is
    the wrong unit for the four upper rungs of the optimisation ladder -- inter-layer scheduling,
    boundary crossing, fusion, global scope -- because those are properties of ADJACENT regions.

    The corpus has zero members on those rungs, and that was never an oversight: nothing in this repo
    observed that two regions were adjacent, so a requirement for a chain could not be derived at all
    rather than merely being unpopulated. This derives it the way every other axis here is derived --
    from what the target's own captured models contain.

    A signature is the chain's shape in the semantic-family vocabulary, not its extents, so two attention
    blocks at different sequence lengths present ONE obligation rather than two. Rare chains are not
    required: ``min_occurrences`` keeps a one-off out of the requirement, because a corpus cannot be
    asked to cover every accident of a graph. Both the threshold and what it dropped are reported, since
    an axis that filters silently is indistinguishable from one that saw nothing.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.development.targetgen.applications import operation_plan as SC

    unreadable: dict[str, str] = {}
    seen: dict[str, dict] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            module = mq.parse(path)
            census = SC.chain_census(module, max_length=64)
        except Exception as exc:  # noqa: BLE001 - an unreadable capture evidences nothing, and says so
            unreadable[str(label)] = f"{type(exc).__name__}: {exc}"
            continue
        for signature, count in (census.get("by_signature") or {}).items():
            row = seen.setdefault(signature, {"signature": signature, "occurrences": 0, "observed_in": []})
            row["occurrences"] += int(count)
            if str(label) not in row["observed_in"]:
                row["observed_in"].append(str(label))

    ranked = sorted(seen.values(), key=lambda r: (-r["occurrences"], r["signature"]))
    frequent = [r for r in ranked if r["occurrences"] >= int(min_occurrences)]
    kept = frequent[: int(max_signatures)]
    for row in kept:
        row["length"] = len(row["signature"].split(" -> "))
    return {
        "required": kept,
        "min_occurrences": int(min_occurrences),
        "max_signatures": int(max_signatures),
        "dropped_below_threshold": len(ranked) - len(frequent),
        "dropped_over_cap": [r["signature"] for r in frequent[int(max_signatures) :]],
        "captures_unreadable": unreadable,
    }


def _epilogue_axis(target: str) -> dict:
    """Which epilogue stages ``target`` must be asked to fuse onto a contraction.

    TWO SOURCES, because either alone is demonstrably incomplete.

    The capability manifest declares a family ``composed_with`` a contraction when that family exists
    only as an epilogue -- which is exactly a fusion capability. But only one target in this repo
    declares it, and the others are not fusion-less: atlas's own ISA resolves ``TensorComputeUnary`` for
    a relu stage and ``TensorComputeBinary`` for a bias stage, so its manifest under-declares against
    its RTL. Deriving from the manifest alone would have produced an epilogue requirement for one target
    and called the rest incapable, which is the single-target overfit this axis exists to remove.

    So a stage is required when the manifest says its family is fused-only, OR the target's own
    instruction taxonomy resolves a class for the role the stage needs, OR a readout the target
    declares APPLIES it, OR a composition-scoped route applies it before readout. The stage-granular
    sources are needed because ``composed_with`` is family-granular in the other
    direction too: a family that ALSO runs standalone (an elementwise map licensed by accumulate-on-load
    evidence) is no longer declared fused-only, yet a readout can still fuse activation and scale,
    while a separate route can seed bias before the contraction. A corpus that stopped asking for those
    stages would let a backend that cannot fuse them fail nothing. Each stage records which source
    evidenced it; a readout claim and a pre-accumulator route are different claims.
    """
    from merlin.development.targetgen.conformance.gaps import _declared_output_path, _path_stages
    from merlin.target.capabilities import semantic_families as sf
    from merlin.target.isa import taxonomy as IT

    try:
        from merlin.target.capabilities.eligibility import capability_map_for_target

        cap_map = capability_map_for_target(target) or {}
    except Exception:  # noqa: BLE001 -- unreadable manifest evidences nothing
        cap_map = {}
    fused_families = {f for f, c in cap_map.items() if tuple(getattr(c, "composed_with", ()) or ())}

    base: set = set()
    taxonomy = None
    try:
        taxonomy = IT.taxonomy_for_target(target)
        base = set(IT.required_classes_for_op(taxonomy, op="matmul"))
    except Exception:  # noqa: BLE001 -- no taxonomy is not "no capability"
        taxonomy = None

    # Stage-granular sources: the target's readout and any composition-scoped non-readout routes.
    # The manifest's `composed_with` is evidence at FAMILY granularity -- "elementwise_map exists only
    # as an epilogue" -- so it makes every elementwise stage required together. These declarations
    # distinguish activation on readout from bias seeded before contraction. Consulting only the
    # coarse source put a stage in the requirement that no application path performs, and the grade
    # refused that capsule hours later. Absent declarations leave the original sources deciding.
    from merlin.target.capabilities.readout_derivation import (
        ReadoutSupportError,
        epilogue_readouts,
        epilogue_stage_routes,
    )
    from merlin.verification.references.epilogue_applicability import selectors_applying

    try:
        readouts = epilogue_readouts(target)
        routes = epilogue_stage_routes(target)
    except ReadoutSupportError:
        raise  # selected support owes readout facts and cannot serve them: a refusal, not "absent"
    except Exception:  # noqa: BLE001 -- an unreadable declaration evidences nothing
        readouts, routes = None, ()

    # Whether the two stage-granular sources were READ at all. ``epilogue_readouts`` returns ``None`` for
    # a target that declares nothing (or whose declaration could not be read), which its own contract
    # calls UNKNOWN; an ungrounded taxonomy is likewise ``unknown`` / ``not_applicable``, never "derived
    # with no fusion class". When neither was read, a stage the manifest does not make fused-only has NO
    # evidence either way. Recording it as rejected wrote a definitive "the target cannot fuse this" into
    # the requirement, and the derived corpus then demanded no epilogue at all.
    taxonomy_status = IT.status_of(taxonomy)
    unread = readouts is None and taxonomy_status != IT.STATUS_DERIVED

    # The derived output path (when the selected contract declares one) evidences each stage it
    # performs: its readout stages, and a bias wherever it preloads the accumulator. It never
    # overrides a declared readout that refuses a stage: that capsule would not be writable.
    from merlin.runtime.engine.command_buffer import BIAS_STAGES

    output_path = _declared_output_path(target)
    path_stages = set(_path_stages(output_path)) if output_path is not None else set()
    if output_path is not None and any(
        stage.get("kind") == "preload" for stage in output_path.get("accumulator_stages") or ()
    ):
        path_stages |= set(BIAS_STAGES)
    required, rejected, unresolved = [], [], []
    for stage in _builder_epilogue_stages():
        family = sf.from_op(stage)
        by_manifest = family in fused_families
        classes: list = []
        if taxonomy is not None:
            try:
                classes = sorted(set(IT.required_classes_for_op(taxonomy, op="matmul", epilogue=(stage,))) - base)
            except Exception:  # noqa: BLE001
                classes = []
        applying = selectors_applying(readouts, [stage], routes=routes, composition="contraction") if readouts else []
        on_readout = selectors_applying(readouts, [stage]) if readouts else []
        routed = any(
            r.stage == stage
            and r.composed_with == "contraction"
            and any(readout.selector in r.readouts for readout in readouts or ())
            for r in routes
        )
        no_readout_applies = bool(readouts) and not applying
        by_path = stage in path_stages
        if no_readout_applies:
            rejected.append(
                {
                    "stage": stage,
                    "family": family,
                    "why": (
                        "no readout or contraction-scoped stage route this target declares APPLIES the stage ("
                        + "; ".join(f"{r.selector!r} applies {sorted(r.applies)}" for r in readouts)
                        + "), so fusing it would declare a computation the hardware does not perform. "
                        "Stage-granular RTL-evidenced declaration overrides the family-granular "
                        + ("manifest evidence" if by_manifest else "taxonomy evidence")
                        + " that would otherwise require it."
                    ),
                }
            )
        elif by_manifest or classes or applying or by_path:
            required.append(
                {
                    "stage": stage,
                    "family": family,
                    "evidenced_by": (["manifest_composed_with"] if by_manifest else [])
                    + (["isa_instruction_class"] if classes else [])
                    + (["readout_applies"] if on_readout else [])
                    + (["contraction_stage_route"] if routed else [])
                    + (["derived_output_path"] if by_path else []),
                    "isa_classes": classes,
                }
            )
        elif unread:
            unresolved.append(
                {
                    "stage": stage,
                    "family": family,
                    "taxonomy_status": taxonomy_status,
                    "readouts": "undeclared",
                    "why": (
                        "the manifest declares no family fused-only for it, no readout declaration could be "
                        f"read for this target and its instruction taxonomy is {taxonomy_status!r}; whether "
                        "the stage fuses onto a contraction is UNDETERMINED, which is neither a requirement "
                        "nor a refusal"
                    ),
                }
            )
        else:
            rejected.append(
                {
                    "stage": stage,
                    "family": family,
                    "why": (
                        "the manifest declares no family fused-only for it, this target's "
                        "instruction taxonomy resolves no class for the role it needs, and it "
                        "declares no readout that applies it"
                    ),
                }
            )
    return {
        "required": required,
        "rejected": rejected,
        "unresolved": unresolved,
        "axis_basis": (
            "the epilogue stages this target can fuse onto a contraction, evidenced by its capability "
            "manifest declaring the stage's family fused-only OR by its own instruction taxonomy "
            "resolving a class for the role the stage needs, and then INTERSECTED with the stages its "
            "declared readout or contraction-scoped non-readout route actually applies -- "
            "the first two are family-granular and the last is "
            "stage-granular, so a target that fuses an activation but not an integer requantize is "
            "described correctly instead of being asked for both. A (family, dtype, alignment) cell "
            "cannot express WHICH epilogue rides the contraction, so a corpus derived from cells alone "
            "tests one stage and calls the fusion capability covered"
        ),
    }


def _capsule_dtype_or(token):
    """``capsule_dtype(token)`` where the registry knows the token, else the token unchanged."""
    from merlin.development.targetgen.conformance.requirements import capsule_dtype

    if not token:
        return token
    try:
        return capsule_dtype(str(token))
    except Exception:  # noqa: BLE001 — an unmappable token is not a dtype
        return token


def _shape_axis(target: str) -> dict:
    """The rank and layout regions ``target``'s own manifest declares it handles.

    Derived from :mod:`merlin.development.targetgen.policies.form_regimes`, which already turns each family's declared
    capability into region descriptors -- ``batch`` into a rank-3 region, ``transpose`` and each declared
    layout into an operand-layout variant. Nothing consumed them outside the fuzzer, so a target could
    declare batching and never be asked for a batched region by anything that grades it.

    Fails soft: a target whose capability map cannot be read contributes an empty axis with the reason,
    because an unreadable manifest is not a target that declares no batching.
    """
    from merlin.development.targetgen.conformance.requirements import _SHAPE_AXIS_BASIS

    try:
        from merlin.development.targetgen.policies.form_regimes import synthesize as _probes
        from merlin.target.capabilities.eligibility import capability_map_for_target

        cap_map = capability_map_for_target(target) or {}
        probes = _probes(cap_map, target=target)
    except Exception as exc:  # noqa: BLE001 -- unreadable is not empty
        return {
            "required": [],
            "unavailable": f"{type(exc).__name__}: {exc}",
            # The basis is stated even when nothing could be derived. An axis with no basis reads as an
            # axis nobody thought about, and "the manifest could not be read" is a different and more
            # useful fact than "this target declares no batched or transposed region".
            "axis_basis": _SHAPE_AXIS_BASIS,
        }

    required = []
    for pr in probes:
        d = pr.descriptor
        rank = int(getattr(d, "rank", 2) or 2)
        layout = getattr(d, "layout", None)
        if rank < 3 and not layout:
            continue  # a plain 2-D region: the cells already say it
        required.append(
            {
                "probe": pr.name,
                "axis": "rank" if rank >= 3 else "layout",
                "family": d.family,
                # THE CAPSULE SPELLING, like every other axis in this spec. `capability_probes` reads the
                # MANIFEST's vocabulary (`int8`) and the cells carry the capsule's (`i8`); emitting the
                # manifest spelling here put an `int8` entry into a corpus whose admitted set is `{i8}`,
                # so the entry read as using a dtype the requirement does not admit.
                "dtype": _capsule_dtype_or(d.in_dtype),
                "rank": rank,
                "layout": layout,
                "m": d.m,
                "k": d.k,
                "n": d.n,
                "batch": int(getattr(d, "batch", 1) or 1),
            }
        )
    return {
        "required": sorted(required, key=lambda r: (r["axis"], r["probe"])),
        "axis_basis": _SHAPE_AXIS_BASIS,
        "why_shape_corners_are_excluded": (
            "the corner probes (tile, tile+-1, prime, skinny) and the dtype probes restate the alignment "
            "and dtype axes the cells already carry; requiring them again would inflate the requirement "
            "with points already required under another name. NOTE the alignment axis now carries three "
            "occupancy classes rather than two: `partial` rags one axis by one element (a nearly-full "
            "tile) and `sub_tile` leaves the tile barely occupied, which the hand-authored corpus probes "
            "45 times and the derived corpus could not express at all. That is a genuinely distinct "
            "point, not a corner probe under another name"
        ),
    }


def classify_alignment(extents, tile_dim: int) -> str | None:
    """Which of the three occupancy classes a capsule's extents fall in. ``None`` with no tile edge.

    THE ONE DEFINITION, because the requirement and the thing that measures it must not each carry
    their own. `Cell.alignment` demands `aligned`/`partial`/`sub_tile`, and `cert_capsule_cover`
    produced only the first two -- so the four `*_i8_sub_tile` cells were demanded by the requirement
    and uncoverable by construction, which is precisely the "gap no capsule could close, reported
    forever as debt" that the cover's own comment warns about two lines further down.

    The classes are separated by how full the RAGGED tile is, which is what a tiling compiler actually
    branches on:

    * ``aligned``   every extent is a whole number of tiles; the tail path is never entered.
    * ``partial``   the ragged tile is nearly full (`tile-1` leaves 15 of 16 lanes live).
    * ``sub_tile``  the ragged tile is barely occupied -- at most half the edge.

    Half the edge is the boundary rather than an invented constant: it is exactly what separates the
    two shapes the synthesizer emits (`tile-1` for partial, `tile/4` and `tile/2` for sub_tile), so
    the classifier and the generator cannot disagree about what they are naming.
    """
    if not tile_dim or tile_dim <= 0:
        return None
    # A DEGENERATE AXIS IS NOT AN OCCUPANCY. An extent of 1 is a broadcast/parameter axis -- an rmsnorm's
    # `1 x K` gain vector, a bias row -- and nothing tiles it, so there is no tile for it to partially
    # occupy. Counting it dragged the WHOLE capsule into `sub_tile` on the strength of the parameter:
    # measured, `SY_normalization_bf16_aligned` is a 16x32 problem with a 1x32 gain, and it classified
    # `sub_tile`, so `normalization/bf16/aligned` was required, built, and reported uncovered while the
    # capsule that covers it sat on disk under a class it does not exercise.
    sizes = [int(e) for e in extents if int(e) > 1]
    if not sizes:
        return None
    remainders = [e % tile_dim for e in sizes]
    if not any(remainders):
        return "aligned"
    return "sub_tile" if min(r for r in remainders if r) <= tile_dim // 2 else "partial"


def _certified_depth(target: str, *, tile: int, budget_s: float | None):
    """A multi-pass reduction this target certifies, as a FALLBACK. ``(point, refusal)``.

    Only used where the residency regimes yield no depth of their own; where they do, they produce
    deeper capsules and this one would be a shallower duplicate.

    WHAT DECIDES ITS SIZE, and what does not. A capsule writes ``M x N`` elements whatever ``K`` is,
    and the measured cost law is a function of that written output -- so the reduction depth is very
    nearly free and there is no budget to clip ``K`` against. An earlier version did clip it, priced
    by operand size, and produced the wrong answer in the expensive direction: it capped the deepest
    reductions out of the cycle-accurate tier while calling the number "measured".

    So the depth is derived from what makes a reduction MULTI-PASS -- two tiles of ``K``, the first
    depth at which an accumulator must survive being written twice -- and the affordability question
    is asked of the OUTPUT TILE instead, which is the thing that actually costs. Refuses when the
    tile edge is unknown or when a single output tile already exceeds the budget, because then no
    capsule of any depth is certifiable here.
    """
    from merlin.development.targetgen.conformance.requirements import _DEFAULT_CERT_BUDGET_S
    from merlin.development.targetgen.oracles import certification_cost as CC

    if not tile:
        return None, "no tile edge, so there is no whole-tile depth to size"
    budget = float(budget_s if budget_s else _DEFAULT_CERT_BUDGET_S)
    seconds, extrapolated = CC.predict_seconds_from_output(int(tile) * int(tile))
    if seconds is None:
        return None, "no measured certification cost law, so one output tile cannot be priced"
    if seconds > budget:
        return None, (
            f"a single {tile}x{tile} output tile already costs {seconds:.0f}s against a "
            f"{budget}s budget on this target, so no reduction of any depth is certifiable"
        )
    return {
        "M": int(tile),
        "K": 2 * int(tile),
        "N": int(tile),
        "K_tiles": 2,
        "predicted_seconds": round(seconds, 1),
        "budget_s": budget,
        "sized_by": "multi_pass_minimum_at_one_output_tile",
        "extrapolated": bool(extrapolated),
        "why": (
            "two tiles of K is the shallowest reduction that writes the accumulator twice; the "
            "cost is the output tile, which K does not move"
        ),
    }, None


def _binding_engine_fit(target: str):
    """The cost fit of the SLOWEST engine this target has certified on, or ``None``.

    One source for "which line prices a capsule here", so the application axis and the certification
    ceiling cannot disagree. ``None`` when no engine has a usable fit, which is a real answer: the
    caller then falls back to the unattributed whole-history fit rather than refusing to size anything.
    """
    from merlin.development.targetgen.conformance.requirements import _DEFAULT_CERT_BUDGET_S
    from merlin.development.targetgen.oracles import cert_affordability as CA

    try:
        priced = CA.affordability(target, budget_s=_DEFAULT_CERT_BUDGET_S)
    except Exception:  # noqa: BLE001 -- unreadable run history is "no per-engine fit"
        return None
    affordable = {
        engine: row
        for engine, row in (priced.get("engines") or {}).items()
        if row.get("max_elements") and row.get("fit")
    }
    if not affordable:
        return None
    chosen = min(affordable, key=lambda e: (affordable[e]["max_elements"], e))
    try:
        # Ask for the fit OBJECT rather than rebuilding one from the reported dict: the dict is a
        # rendering (it carries `sources`, and its field set is free to change), and a reconstruction
        # that silently fails would drop this axis back onto the mixed fit with nothing said.
        return CA.fit_for(target, engine=chosen)
    except Exception:  # noqa: BLE001 -- an engine we cannot re-fit is one we do not price with
        return None


def _cert_affordability(target: str, *, budget_s: float | None) -> dict:
    """The largest WRITTEN OUTPUT this target can certify inside the budget. Never operand size.

    Which size metric drives certification cost was measured, and the first answer was wrong. An
    observational fit over graded runs picked the largest operand, because in a corpus where every
    capsule is a tile-multiple square the operand and the output move together. A calibration ladder
    that deliberately broke that correlation -- growing the weight while holding the output nearly
    fixed -- showed time following the OUTPUT, and refitting over this target's own 20 shape-resolvable
    certifications agrees decisively:

        written output  r2 0.924      max operand  r2 0.226      work (M*K*N)  r2 0.655

    The consequence is the useful part, and it is the opposite of what the operand metric implied:
    REDUCTION DEPTH IS CHEAP TO CERTIFY. A capsule writes ``M x N`` elements whatever ``K`` is, so a
    deep accumulation drains the same result tile as a shallow one. Measured here: ``PK03_k128``
    (K=128) took 161.5 s against 121.1 s for the same shape at K=16 -- eight times the reduction for a
    third more time, where the operand metric predicted a capsule nobody could afford. So the deepest
    accumulation a compiler must survive can be certified cycle-accurately at one output tile, which
    is exactly the minimal-but-representative shape a functional corpus wants.

    ``max_elements`` is ``None`` when the law cannot be applied, and the caller must then leave every
    tier alone rather than cap on a cost nobody measured.
    """
    from merlin.development.targetgen.conformance.requirements import _DEFAULT_CERT_BUDGET_S
    from merlin.development.targetgen.oracles import cert_affordability as CA
    from merlin.development.targetgen.oracles import certification_cost as CC

    budget = float(budget_s if budget_s else _DEFAULT_CERT_BUDGET_S)
    why = (
        "the largest written output whose predicted certification fits the budget. A capsule above "
        "it is graded at the loop tier and rests on a certified sibling -- not because it is "
        "uninteresting, but because nobody can afford to run it cycle-accurately. Reduction depth "
        "does NOT count against this: K moves the operands, not the result"
    )

    # THIS TARGET'S OWN CERTIFICATIONS FIRST. A simulation rate is a property of the DESIGN and the
    # simulator, not of the corpus, so pricing one target's capsules with another's measurements is a
    # claim with no evidence behind it -- and that is what happened: the ladder constants below were
    # measured on one target and every target got them, so three targets' capsules were sized against
    # a device none of them is. Where a target has its own history the two methods can be compared,
    # and on the calibrated target they agree to within 2%, which is why the fallback is usable at all.
    # ...AND PRICED PER ENGINE, because the engine IS the cost. This used to call `cert_cost.fit_for`
    # with no `engine=`, which draws one line through every certification the target has, whatever
    # answered it. Measured here (2026-09-21, gemmini): that mixed line is r2 0.149 and puts the
    # ceiling at 19,245 written elements, while the same history split by engine gives 6,500 on gsim
    # (r2 0.230) and 923 on verilator (r2 0.604) -- a 21x spread between the two machines the mixture
    # was averaging, and the corpus's own tracked ceiling of 875 sits next to the verilator figure.
    # The mixed number is not a newer measurement of anything; it is two devices added together.
    #
    # THE BINDING ENGINE IS THE SLOWEST ONE THIS TARGET HAS CERTIFIED ON, not the cheapest and not an
    # average. This ceiling decides whether a capsule may DEMAND certification, and the demand has to
    # be payable on whichever engine the run actually uses -- so the constraint that binds is the
    # engine that takes longest. Choosing the cheapest would emit capsules nobody can afford the
    # moment a run picks the other one, which is the failure this function exists to prevent.
    #
    # `max_elements_within` does the inversion, so the extrapolation clamp (no answer beyond the
    # measured range times the margin) and the "floor alone exceeds the budget" refusal live in one
    # place. The old inline `(budget - intercept) / per_element` here bypassed both.
    engines: dict = {}
    fit = None
    try:
        priced = CA.affordability(target, budget_s=budget)
        engines = {
            engine: {"max_elements": row.get("max_elements"), "n_samples": row.get("n_samples"), "fit": row.get("fit")}
            for engine, row in (priced.get("engines") or {}).items()
        }
        affordable = {e: row["max_elements"] for e, row in engines.items() if row.get("max_elements")}
        if affordable:
            chosen = min(affordable, key=lambda e: (affordable[e], e))
            row = (priced.get("engines") or {})[chosen]
            f = row["fit"] or {}
            lo, hi = (f.get("measured_range_elements") or [0, 0])[:2]
            return {
                "max_elements": int(affordable[chosen]),
                "budget_s": budget,
                "metric": "written_output_elements",
                "engine": chosen,
                "engine_choice": (
                    "the slowest engine this target has measured history on: the ceiling must be "
                    "payable on whichever engine the run picks, so the binding constraint is the "
                    "engine that takes longest"
                ),
                "engines": {e: row["max_elements"] for e, row in engines.items()},
                "law": f"seconds = {f.get('intercept_s')} + {f.get('per_element_s')} * output",
                "calibrated_to_elements": int(hi or 0),
                "extrapolated": bool(int(affordable[chosen]) > int(hi or 0)),
                "basis": (
                    f"fitted on {f.get('n_samples')} cycle-accurate {chosen} certification(s) of THIS "
                    f"target (r2 {f.get('r2')}, {lo}..{hi} written elements); other engine ceilings: "
                    f"{ {e: r['max_elements'] for e, r in engines.items()} }"
                ),
                "why": why,
            }
    except Exception:  # noqa: BLE001 -- unreadable run history is "none"
        engines = {}
    # No engine could be priced. Fall back to the UNATTRIBUTED whole-history fit only to say how
    # little it is worth: it is reported, never returned as the ceiling, because a mixed fit prices a
    # capsule at no machine that exists.
    try:
        fit = CC.fit_for(target)
    except Exception:  # noqa: BLE001 -- unreadable run history is "none"
        fit = None
    if fit is not None:
        engines.setdefault(
            "<mixed, not used>",
            {
                "max_elements": CC.max_elements_within(fit, budget),
                "n_samples": fit.n_samples,
                "fit": fit.to_dict(),
                "mixed_engines": bool(fit.mixed_engines),
            },
        )

    coeff = getattr(CC, "MEASURED_COEFFICIENT_S", None)
    exponent = getattr(CC, "MEASURED_EXPONENT", None)
    if not coeff or not exponent:
        return {
            "max_elements": None,
            "budget_s": budget,
            "metric": "written_output_elements",
            "engines": engines,
            "basis": "no measured certification history for this target and no fallback law",
            "why": "no measured certification cost law is available in this checkout",
        }
    # Invert the power law: the output at which the predicted certification exactly spends the budget.
    ceiling = int((budget / float(coeff)) ** (1.0 / float(exponent)))
    calibrated_to = getattr(CC, "MEASURED_MAX_OUTPUT_ELEMENTS", 0)
    return {
        "max_elements": max(1, ceiling),
        "budget_s": budget,
        "metric": "written_output_elements",
        "law": f"seconds = {coeff} * output ** {exponent}",
        "calibrated_to_elements": calibrated_to,
        "extrapolated": True,
        "engines": engines,
        "basis": (
            "BORROWED: this target has certified nothing cycle-accurately, so the ceiling comes "
            "from the deliberate calibration ladder measured on another device. It bounds the "
            "budget rather than describing this device, and a capsule sized by it is sized by a "
            "rate nobody has measured here -- run one cert on this target to replace it"
        ),
        "why": why,
    }


def regime_dtype_selection(cells: list[Cell]) -> tuple[str | None, dict]:
    """Choose the most represented required-cell dtype with a stable width tie-break.

    Cell frequency remains the primary evidence. Equal counts are resolved by the
    registered operand storage width, then by token spelling. Unknown widths sort
    after known widths; they remain visible in the provenance rather than vanishing.
    No target name, preference, or inferred capability enters this choice.
    """
    from merlin.development.targetgen.grading.golden_inputs import dtype_bits

    counts = Counter(cell.dtype for cell in cells)
    widths: dict[str, int | None] = {}
    for dtype in sorted(counts):
        try:
            widths[dtype] = int(dtype_bits(dtype))
        except (KeyError, ValueError):
            widths[dtype] = None

    selected = min(
        counts,
        key=lambda dtype: (
            -counts[dtype],
            widths[dtype] is None,
            widths[dtype] if widths[dtype] is not None else 0,
            dtype,
        ),
        default=None,
    )
    greatest_count = max(counts.values(), default=0)
    evidence = {
        "policy": (
            "highest required-cell frequency, then narrowest registered storage width, "
            "then lexical dtype token; unknown widths last"
        ),
        "counts": dict(sorted(counts.items())),
        "storage_bits": widths,
        "tied_by_count": sorted(dtype for dtype, count in counts.items() if count == greatest_count),
        "selected": selected,
    }
    return selected, evidence


def _accumulator_output_boundary(target: str, tile: int) -> dict:
    """Smallest-output contraction exceeding one addressable accumulator.

    An output-capacity test is distinct from deep-K and operand-store tests: a
    backend may tile both inputs correctly while keeping every output tile live.
    The address-space resolver, not a target name or a hand-authored capacity,
    decides whether this boundary exists.
    """
    from merlin.target.capabilities import address_space as AS

    if tile <= 0:
        return {"status": "unknown", "reason": "no derived compute tile edge"}
    try:
        kind = AS.accumulator_kind(AS.derive_address_space(target))
    except Exception as exc:  # noqa: BLE001 -- preserve missing RTL evidence
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    if kind.kind == AS.IN_DATAPATH:
        return {"status": "not_applicable", "reason": "no separately addressable accumulator rows"}
    if kind.kind != AS.ADDRESSABLE or not kind.rows:
        reason = kind.unknown.reason if kind.unknown else "accumulator row capacity is not derivable"
        return {"status": "unknown", "reason": reason}
    n_tiles = int(kind.rows) // tile + 1
    return {
        "status": "resolved",
        "capacity_rows": int(kind.rows),
        "store": kind.store.name if kind.store else None,
        "buffers": int(kind.buffers),
        "tile_edge": tile,
        "M": 1,
        "K": tile,
        "N": n_tiles * tile,
        "N_tiles": n_tiles,
        "output_rows_if_resident": n_tiles * tile,
        "basis": "one output row, the first whole N tile beyond one accumulator buffer's row capacity",
    }


def derive_spec(
    target: str,
    captures: dict[str, str | Path],
    *,
    oracle_tiers: list[str] | Callable[[], list[str]],
    declared: dict[str, dict] | None = None,
    personas: dict[str, dict] | None = None,
    applications: dict[str, str | Path] | None = None,
    corpus_roots=None,
    cert_budget_s: float | None = None,
    certification_floor: str | None = None,
    application_inventory_options: dict | None = None,
) -> dict:
    """Derive a conformance spec from explicitly supplied tier evidence.

    Regenerable and tracked: a reviewer diffs it to see the requirement change when a target's manifest
    or a target-model's capture changes, which is exactly the drift a hand-authored list hides.
    ``oracle_tiers`` must be the observed constructed inventory, not advertised metadata.
    An empty observation preserves the existing unknown/no-cap policy.
    A caller may supply a zero-argument observer instead of a list. It is called
    exactly once, at the historical field's observation point after preceding
    derivation succeeds. It is never cached; core performs no default discovery.
    """
    from merlin.development.targetgen.conformance.gaps import _carried_state_axis, _conv_geometry_axis, _group_axis
    from merlin.development.targetgen.conformance.requirements import (
        _application_axis,
        _cs,
        _host_only_observed_ops,
        boundaries,
        geometry_axis,
        host_lane_cells,
        host_only_dtypes,
        required_cells,
    )

    if certification_floor is not None and (
        not isinstance(certification_floor, str)
        or not certification_floor.startswith("L")
        or not certification_floor[1:].isdigit()
    ):
        raise ValueError("certification_floor must be a fidelity tier")
    cells, diag = required_cells(target, captures, declared=declared)
    bnd = boundaries(target)
    from merlin.development.targetgen.grading import boundary as BD
    from merlin.development.targetgen.policies import memory_regime as MR

    comp = BD.required_boundaries(captures, target)
    mem = MR.required_regimes(captures, target)
    # The extents that REACH each required regime, resolved here because the search needs the target's
    # address space -- exactly the I/O `corpus_synth` is not allowed to do. Emitting them into the spec
    # keeps the synthesizer pure and keeps one definition of what a regime costs.
    _regime_dtype, _regime_dtype_selection = regime_dtype_selection(list(cells))
    HL = host_lane_cells(captures, target, corpus_roots=corpus_roots)
    HO_DTYPES = host_only_dtypes(captures, diag.get("families_needed_but_not_admitted") or ())
    try:
        _reduction_depth = MR.reduction_depth_regimes(
            target, sorted((mem.get("by_regime") or {}).keys()), tile_dim=bnd.tile_edge or 0, dtype=_regime_dtype
        )
    except Exception as _exc:  # noqa: BLE001 -- an underivable depth is not zero
        _reduction_depth = {"unavailable": f"{type(_exc).__name__}: {_exc}"}
    if certification_floor is None:
        _reduction_depth["certified"], _reduction_depth["certified_refusal"] = _certified_depth(
            target, tile=bnd.tile_edge or 0, budget_s=cert_budget_s
        )
    else:
        # A minimum direct tier is a requirement, not a measured-cost certificate.
        # Retain the two-pass functional anchor without reading ambient cost history.
        tile = int(bnd.tile_edge or 0)
        _reduction_depth["certified"] = (
            {
                "M": tile,
                "K": 2 * tile,
                "N": tile,
                "K_tiles": 2,
                "predicted_seconds": None,
                "budget_s": None,
                "sized_by": "functional_two_pass_minimum",
                "cost_status": "unknown_not_priced",
                "why": "two K tiles require at least two accumulation passes",
            }
            if tile > 0
            else None
        )
        _reduction_depth["certified_refusal"] = None if tile > 0 else "no tile edge for two-pass anchor"
    _regime_extents = MR.required_regime_extents(
        target, sorted((mem.get("by_regime") or {}).keys()), tile_dim=bnd.tile_edge or 0, dtype=_regime_dtype
    )
    return {
        "target": target,
        **(
            {
                "certification_floor": {
                    "tier": certification_floor,
                    "source": "derive_spec.certification_floor argument",
                    "scope": "direct per-capsule certification; no execution or affordability claim",
                }
            }
            if certification_floor is not None
            else {}
        ),
        "generated_by": "merlin.development.targetgen.conformance.requirements.spec",
        "derivation": {
            "admitted": "capability manifest compute_units[].semantic_capabilities (family x dtype)",
            "observed": "model_coverage.regions_from_module over each captured model (name-first, "
            "provenance-tag fallback; unresolved regions not counted)",
            "required": "admitted INTERSECT observed, x tile alignment where the target tiles",
            "cell_vocabulary": "(semantic_family, dtype, tile_alignment) — identical to "
            "contract.materialize.cert_capsule_cover, so the existing cover measures "
            "this spec without a second definition of a cell",
        },
        "boundaries": bnd.to_dict(),
        "composition": {
            "required": comp["by_kind"],
            "whole_model_shape": comp["whole_model_shape"],
            "captures_unreadable": comp["captures_unreadable"],
            "axis_basis": (
                "the composition shapes the real captures CONTAIN — not just the shape each capture has "
                "as a whole. A model classifies as `routing` end to end, yet it contains isolated "
                "dispatches, adjacent accelerator pairs and host islands, and each of those is a "
                "composition the corpus must exercise somewhere. Taking only the whole-model label would "
                "demand `routing` and nothing else, the narrowest reading of the richest evidence"
            ),
            "why_orthogonal": (
                "the composition axis is NOT crossed with family/dtype/alignment. A cross product would "
                "demand cells like `movement/i8/partial/routing` that no real model presents, "
                "manufacturing uncovered cells nobody should build; composition is a property of how a "
                "program is assembled, not of the arithmetic in it"
            ),
        },
        # THE SHAPE-GENERALIZATION AXIS. `capability_probes` already enumerates, per family and from the
        # manifest's own declarations, the region shapes a target claims to handle -- and until now it
        # fed only the fuzzer, so nothing in the REQUIREMENT said the corpus had to contain them.
        #
        # Only two of its axes reach here, deliberately. The shape-corner probes (tile, tile+-1, prime,
        # skinny) restate the alignment axis the cells already carry, and the dtype probes restate the
        # cells themselves; adding either would inflate the requirement with points already required
        # under another name. Rank and layout are the two the cell vocabulary genuinely cannot express:
        # a `(family, dtype, alignment)` cell says nothing about whether the unit was asked for a
        # BATCHED region or a TRANSPOSED operand, and both are things a manifest declares and a compiler
        # gets wrong independently of arithmetic.
        "shape_generalization": _shape_axis(target),
        # THE HOST LANE, AT FULL WIDTH. `host_only` below carries families with no admitted dtype at
        # all; this carries every (family, dtype) pair the captures contain and the manifest does not
        # admit, which is a far larger and more representative set -- and it is the work a real model
        # actually hands the host.
        "host_lane": HL,
        # WHICH epilogue rides the contraction. The cells say a fused-only family must be carried as an
        # epilogue; they cannot say which stage, so a corpus derived from them tests one and reports the
        # capability covered.
        "epilogue": _epilogue_axis(target),
        # THE GROUPS REAL MODELS FORM. The epilogue axis asks for each stage once; it cannot ask
        # for the COMBINATION a model needs on one contraction, and a backend that fuses each stage
        # alone and no two together passes it. This axis is the compute groups the captures
        # actually form on this target, stated in capsule words.
        "groups": _group_axis(captures, target),
        # STATE CARRIED FROM ONE COMMAND INTO THE NEXT. Every other axis describes one command on a
        # freshly configured unit. A stage that is pure configuration stays switched on until
        # something switches it off, and only a SECOND command, without that stage, can show it.
        "carried_state": _carried_state_axis(target),
        # THE USER'S OWN APPLICATIONS. Empty unless the target declares some; when it does, this is
        # the only axis whose capsules carry a shape a real model contains rather than a tile
        # multiple, and the only one whose sizing is bounded by what a certification costs.
        "application_shapes": _application_axis(
            target, captures=applications, budget_s=cert_budget_s, certification_floor=certification_floor
        ),
        # All operations in each DECLARED derivation application, including host and unknown work.
        # Grouping identical signatures keeps the tracked requirement reviewable; `ordinals` still
        # accounts for every parsed operation. Claim/held-out models never enter this input mapping.
        "application_demands": application_demand_inventory(
            applications or {}, target, **(application_inventory_options or {})
        ),
        # WHAT A CERTIFICATION COSTS HERE, so an axis can size against it instead of assuming every
        # capsule it derives is affordable at the deepest tier.
        "cert_affordability": (
            _cert_affordability(target, budget_s=cert_budget_s)
            if certification_floor is None
            else {
                "status": "unknown_not_priced",
                "max_elements": None,
                "budget_s": cert_budget_s,
                "basis": "explicit direct certification floor; no ambient cost history selected",
            }
        ),
        # THE TIERS THIS TARGET ACTUALLY DECLARES. Published because a capsule too large to certify has
        # to be capped to a tier that EXISTS here, and "L2" is not a universal name: one target in this
        # repo declares `[L3]` alone, so capping to L2 there names a tier nothing can run. The
        # synthesizer is pure and cannot ask the adapter registry, so the answer travels with the spec.
        "oracle_tiers": oracle_tiers() if callable(oracle_tiers) else oracle_tiers,
        # THE ASPECT-RATIO AXIS. Emitted beside the cells rather than crossed with them, for the same
        # reason the composition axis is: a cross product would demand geometries at dtypes no model
        # presents. See `geometry_axis` for why a corpus that is entirely square proves nothing about
        # the tall-skinny convolutions that carry a real vision model's work.
        "shape_geometry": geometry_axis(captures, target),
        # A full-output-resident schedule can exceed the accumulator even when
        # its input operands fit and K is shallow. This independent boundary is
        # derived from the RTL-backed address-space role resolution.
        "accumulator_output_boundary": _accumulator_output_boundary(target, bnd.tile_edge or 0),
        # THE ADJACENCY AXIS. Every axis above describes ONE region; the optimisation ladder's
        # upper rungs are about adjacent ones, and until this nothing observed adjacency at all,
        # so those rungs could not be required rather than merely being unpopulated.
        "scope": scope_axis(captures, target),
        # THE CONVOLUTION-WINDOW AXIS. A cell says a contraction runs at int8; it cannot say that the
        # contraction is a 3x3 stride-2 convolution whose border is padded, and the padded border is
        # where a lost padding identity is wrong and nowhere else. Derived structurally because
        # torch-mlir emits im2col: the captured program has a gather and a matmul and no convolution
        # op, so there is no `padding` attribute anywhere to read. See `conv_geometry`.
        "conv_geometry": _conv_geometry_axis(captures),
        "memory_mapping": {
            "required": mem.get("by_regime") or {},
            "region_counts": mem.get("region_counts") or {},
            "capacity_rows": mem.get("capacity_rows"),
            "captures_unreadable": mem.get("captures_unreadable") or {},
            "why": mem.get("why", ""),
            # Tile-relative extents that land a capsule in each required regime, derived with the same
            # sizing the coverage gate measures with. A regime whose value is null is one no capsule
            # shape can reach on this target -- reported, never silently dropped. `fits_on_reuse` is
            # always null here: a capsule's declared inputs are all live at once, so peak-live and total
            # coincide and the regime that separates them cannot arise from inputs alone.
            "regime_extents": _regime_extents,
            "regime_dtype": _regime_dtype,
            "regime_dtype_selection": _regime_dtype_selection,
            # ACCUMULATION DEPTH, per regime. `regime_extents` gives ONE shape that reaches each
            # residency band; this gives the deepest K within it. They are different questions: a
            # regime capsule proves the store is loaded to that level, a deep-K capsule proves the
            # accumulator survives the reduction that fills it. The hand-authored corpus tests
            # K_tiles 1, 2 and 4 by hand for exactly this reason and nothing derived it.
            "reduction_depth": _reduction_depth,
            "axis_basis": (
                "the regimes real captured models put this target's operand store in, derived from the "
                "store's own geometry (bytes / row width from the compute array and the datapath "
                "element type). A corpus whose capsules all fit the store many times over cannot detect "
                "a memory-mapping failure of any kind, and on a hardware-interlocked target nothing "
                "else will report it either -- the schedule is correct whatever it chooses"
            ),
        },
        "host_only": {
            # THE NEGATIVE LANE. Families real captures contain that this target's manifest does NOT
            # admit -- so the compiler must place them on the host, and a target that accelerates one is
            # as wrong as one that misses admitted work. The set was already derived; it sat in
            # diagnostics where nothing could require it, which is why `H` was only ever covered
            # incidentally.
            "families": list(diag.get("families_needed_but_not_admitted") or ()),
            # The dtype each host family is actually observed in. It cannot come from the manifest -- the
            # hardware declares no capability for these -- so it comes from the captures.
            "dtypes": HO_DTYPES,
            # The operations each host-only family's work consists of, at its observed dtype.
            "observed_ops": _host_only_observed_ops(captures, HO_DTYPES),
            "observed_in": {
                f: n
                for f, n in (diag.get("families_observed") or {}).items()
                if f in set(diag.get("families_needed_but_not_admitted") or ())
            },
            "basis": (
                "families a real capture contains and this target's capability manifest does not admit. "
                "The compiler must route them to the host lane; accelerating one is as much a defect as "
                "failing to accelerate an admitted family. EMPTY means no negative lane is derivable "
                "for this target -- every family its captures contain is admitted -- and never that "
                "none is needed: a target with an empty complement simply cannot be asked this "
                "question, which is a different fact from passing it"
            ),
        },
        "diagnostics": diag,
        "personas": personas or {},
        # `shape_quantum` travels WITH the cell because the synthesizer is pure: it cannot ask a
        # manifest what granularity this dtype's datapath imposes, and a capsule sized on the tile edge
        # alone is a shape the golden refuses (sub-byte MX) or answers with zeros. Same channel, same
        # reason, as `memory_mapping.regime_extents`.
        "cells": [
            {
                "cell": c.key(),
                "family": c.family,
                "dtype": c.dtype,
                "alignment": c.alignment,
                "shape_quantum": _cs().shape_quantum(
                    c.dtype, tile_dim=bnd.tile_edge, scale_block=(diag.get("scale_block_elements"))
                ),
                **o.to_dict(),
            }
            for c, o in sorted(cells.items(), key=lambda kv: kv[0].key())
        ],
    }
