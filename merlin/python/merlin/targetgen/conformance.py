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
    (:func:`merlin.targetgen.eligibility.capability_map_for_target`), family x dtype x rank. A target
    with no fp8 datapath does not get fp8 cells.

``observed``
    what the real target-models actually contain — the family census of each captured model
    (:func:`merlin.targetgen.model_coverage.regions_from_module`). A family no real model uses is not
    required of the corpus, however capable the hardware is.

``boundaries``
    where the hardware's own edges are — the mesh/array edge and the MX block-scale group, read from the
    target's facts. Edge cases are generated at those boundaries rather than imagined.

The required set is ``admitted INTERSECT observed``, expressed in the SAME cell vocabulary
:func:`merlin.targetgen.contract.materialize.cert_capsule_cover` already uses — ``(semantic_family,
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
from dataclasses import dataclass, field
from pathlib import Path

#: How a cell's origin was established. ``observed`` = a real capture contains the family;
#: ``declared`` = asserted from an external source because no capture exists. Kept as data on every cell
#: so a reader can tell evidence from assertion without consulting this docstring.
OBSERVED = "observed"
DECLARED = "declared"
#: A composite family evidenced by observing every primitive it decomposes into, because the importer
#: lowers it away before merlin sees the IR. Distinct from OBSERVED so a reader can see the inference.
OBSERVED_VIA_PRIMITIVES = "observed_via_primitives"


def _sf():
    """:mod:`merlin.targetgen.semantic_families`, imported lazily to keep this module import-light."""
    from merlin.targetgen import semantic_families
    return semantic_families


def _cs():
    """:mod:`merlin.targetgen.corpus_spec`, imported lazily for the same reason as :func:`_sf`. It owns
    the ONE definition of a dtype's numeric regime and of the shape granularity that regime imposes, so
    the requirement and the synthesizer cannot disagree about which shapes exist."""
    from merlin.targetgen import corpus_spec
    return corpus_spec


def capsule_dtype(token: str) -> str:
    """A manifest dtype token in the spelling a CAPSULE writes, via the one canonical mapping.

    The capability manifest declares canonical tokens (``fp32``, ``int8``); a capsule's
    ``inputs[].dtype`` carries the capsule spelling (``f32``, ``i8``) that ``binding.cap_dtype`` produces,
    and ``cert_capsule_cover`` builds its cells from the latter. Comparing the two spellings directly
    makes every float32 and int8 cell look simultaneously required and uncovered -- measured: it reported
    40 of 56 cells missing while the corpus plainly had them, with `contraction/fp32/aligned` "uncovered"
    beside `contraction/f32/aligned` "extra". Route both sides through ``corpus_spec.dtype_info`` so
    there is exactly one spelling authority. An unknown token is returned unchanged rather than dropped:
    a dtype we cannot map is still a real requirement, and it should surface as an uncovered cell instead
    of vanishing.
    """
    try:
        from merlin.targetgen.corpus_spec import dtype_info
        return dtype_info(str(token))[0]
    except Exception:                                      # noqa: BLE001 — unmapped token, keep as-is
        return str(token)


@dataclass(frozen=True)
class Cell:
    """One coverage requirement, in ``cert_capsule_cover``'s vocabulary.

    ``alignment`` is ``"aligned"`` / ``"partial"`` / ``"sub_tile"``, or ``None`` when the target
    declares no tiling edge at all. All are required wherever an edge exists: a unit that only ever
    sees whole tiles has never exercised its tail path, and the tail is where tiling bugs live.

    ``sub_tile`` IS NOT ``partial`` UNDER ANOTHER NAME, and the difference was measured. ``partial``
    rags one axis by a single element, so the tile it produces is NEARLY FULL -- 15 of 16 lanes live
    on a 16-wide edge. A tile that is BARELY OCCUPIED is a different question for a tiling compiler:
    whether it still issues a full tile, whether it skips the pass, whether the tail predicate is
    even reached. Across the hand-authored corpus, extents of 2/4/7/8/9/10/12 against a 16-wide tile
    occur 45 times; across the derived corpus they occurred ZERO times, because the two alignment
    classes could not express one. That is the single largest structural gap between the two corpora,
    and by the measured certification cost law it is also the cheapest to close: a tile/4 capsule
    writes 16 elements and certifies in seconds, where the corpus median is 256 elements.
    """

    family: str
    dtype: str
    alignment: str | None

    def key(self) -> str:
        """The ``"/"``-joined spelling ``cert_capsule_cover`` reports, dropping absent axes."""
        return "/".join(x for x in (self.family, self.dtype, self.alignment) if x)


@dataclass
class CellOrigin:
    """Why a cell is required. Separate from :class:`Cell` so the cell stays hashable and comparable."""

    basis: str = OBSERVED                       # OBSERVED | OBSERVED_VIA_PRIMITIVES | DECLARED
    observed_in: tuple[str, ...] = ()           # captures whose census contains this family
    admitted_by: tuple[str, ...] = ()           # compute units declaring the family+dtype
    citation: str = ""                          # required when basis == DECLARED
    via_primitives: tuple[str, ...] = ()        # set when basis == OBSERVED_VIA_PRIMITIVES
    n_regions: int = 0                          # how many regions across all captures carried it

    def to_dict(self) -> dict:
        out: dict = {"basis": self.basis}
        if self.observed_in:
            out["observed_in"] = list(self.observed_in)
        if self.admitted_by:
            out["admitted_by"] = list(self.admitted_by)
        if self.via_primitives:
            out["via_primitives"] = list(self.via_primitives)
        if self.n_regions:
            out["n_regions"] = self.n_regions
        if self.citation:
            out["citation"] = self.citation
        return out


@dataclass
class Boundaries:
    """The target's own edges, and where each came from.

    ``tile_edge`` is the value the corpus generator tiles against. On a target with no fixed hardware
    mesh it is a SOFTWARE default, and ``tile_edge_is_hardware_fact`` says so — reporting a software
    default as a hardware boundary is how a derived artifact starts lying.
    """

    tile_edge: int | None = None
    tile_edge_is_hardware_fact: bool = False
    tile_edge_source: str = ""
    block_scale_group: int | None = None        # MX E8M0 K-group, when the target has one
    block_scale_source: str = ""
    operand_store_bytes: int | None = None
    operand_store_source: str = ""

    def extent_probes(self) -> list[dict]:
        """Extents that straddle each real boundary — the derived edge cases.

        ``tile_edge`` catches the tiling tail; ``block_scale_group`` catches a K that does not divide
        the block-scale group, which is not a hypothetical: a captured LLaMA's own hidden size (344)
        is not a multiple of a 32-element group.

        SUB-BOUNDARY POINTS ARE A SEPARATE CLASS, and leaving them out was measured as the structural
        gap between this corpus and the hand-authored one. The points used to be
        ``[edge-1, edge, edge+1, edge*2]`` -- straddling the edge and doubling it, but never sampling
        WELL below it. Across the hand-authored capsules, extents of 2/4/7/8/9/10/12 against a 16-wide
        tile occur 45 times; across the derived corpus they occurred ZERO times, because no token in
        this list could produce one. A partial tile that is nearly full (``edge-1``) and one that is
        barely occupied (``edge//4``) are different cases for a tiling compiler: the first exercises
        the tail predicate, the second exercises whether a mostly-empty tile is issued at all.

        So each boundary now also contributes ``1`` (the degenerate extent -- a single row or element,
        which the hand set probes 26 times), ``edge//2`` and ``edge//4``. Deduplicated and ordered, so
        a small edge that collapses these onto each other simply yields fewer points rather than
        repeats.
        """
        out: list[dict] = []
        for name, edge, src in (("tile_edge", self.tile_edge, self.tile_edge_source),
                                ("block_scale_group", self.block_scale_group, self.block_scale_source)):
            if not edge or edge < 2:
                continue
            e = int(edge)
            points = {1, e // 4, e // 2, e - 1, e, e + 1, e * 2}
            out.append({"boundary": name, "edge": e, "source": src,
                        "points": sorted(x for x in points if x >= 1)})
        return out

    def to_dict(self) -> dict:
        return {
            "tile_edge": self.tile_edge,
            "tile_edge_is_hardware_fact": self.tile_edge_is_hardware_fact,
            "tile_edge_source": self.tile_edge_source,
            "block_scale_group": self.block_scale_group,
            "block_scale_source": self.block_scale_source,
            "operand_store_bytes": self.operand_store_bytes,
            "operand_store_source": self.operand_store_source,
            "extent_probes": self.extent_probes(),
        }


# ---------------------------------------------------------------------------------------------------
# the three derivation sources
# ---------------------------------------------------------------------------------------------------

def admitted(target: str) -> dict[str, tuple[str, ...]]:
    """``family -> dtypes`` the target's capability manifest DECLARES the silicon can compute.

    Fused-only families are included: the hardware can run them, just not standalone, and whether the
    corpus exercises them fused is a coverage question rather than an admissibility one. Raises nothing —
    an unresolvable contract yields ``{}`` so the caller reports "nothing admitted" instead of inventing
    a denominator.
    """
    return admitted_with_reason(target)[0]


def admitted_with_reason(target: str) -> tuple[dict[str, tuple[str, ...]], str]:
    """:func:`admitted` plus WHY it is what it is: ``"resolved"`` / ``"unresolvable: ..."``.

    ⚠️ AN EMPTY MAP HAS TWO CAUSES AND THEY LICENSE OPPOSITE ACTIONS. "This target's manifest admits no
    family a capture contains" is a real, final answer. "This target has no generated contract to read"
    is a missing artifact, and a requirement derived from it is not empty -- it is UNKNOWN. Both used to
    return ``{}`` and every caller downstream saw zero cells, so a target with no contract reported the
    same clean nothing as one whose families genuinely do not intersect.

    Measured: ``saturn_opu`` and ``saturn_opu_rvv`` have no
    ``out/artifacts/targets/<target>/contracts/`` at all, derived zero cells, and were reported as
    having no requirement rather than as unverifiable.
    """
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(target)
    except Exception as exc:                               # noqa: BLE001 — unresolvable contract
        return {}, f"unresolvable: {type(exc).__name__}: {str(exc)[-160:]}"
    return ({fam: tuple(cap.dtypes or ()) for fam, cap in sorted(cap_map.items())}, "resolved")


def admitting_units(target: str) -> dict[tuple[str, str], tuple[str, ...]]:
    """``(family, dtype) -> unit names`` — which compute unit admits each pair, for cell provenance."""
    out: dict[tuple[str, str], list[str]] = {}
    try:
        from merlin.targetgen import compute_units as cu
        from merlin.targetgen import target_registry as tr
        units = cu.compute_units(tr.load_contract(target))
    except Exception:                                      # noqa: BLE001
        return {}
    for unit in units:
        for cap in getattr(unit, "semantic_capabilities", ()) or ():
            for dt in (getattr(cap, "dtypes", ()) or ()):
                # keyed in the CAPSULE spelling, so it joins the cells built below
                out.setdefault((cap.family, capsule_dtype(dt)), []).append(getattr(unit, "name", "?"))
    return {k: tuple(sorted(set(v))) for k, v in out.items()}


def observed(capture: str | Path, target: str) -> Counter:
    """``family -> n_regions`` actually present in one captured model's linalg.

    Delegates to :mod:`merlin.targetgen.model_coverage`, which resolves a region's family from the op's
    own NAME first and only falls back to provenance tags — the right order, because real captures leave a
    large fraction of regions untagged and the tags disagree with the IR often enough to be a hint rather
    than an authority. Regions whose family cannot be resolved are NOT counted: an unnamed region is
    evidence neither for nor against a requirement.
    """
    from merlin.targetgen import model_coverage as mc

    module = mc.load_module(capture)
    regions = mc.regions_from_module(module)
    rep = mc.coverage_for(regions, target, model=str(Path(capture).parent.name))
    return Counter(dict(rep.by_family))


def observed_pairs(capture: str | Path, target: str) -> Counter:
    """``(family, dtype) -> n_regions`` present in one capture, in CAPSULE dtype spelling.

    :func:`observed` collapses the same walk to a family histogram, and that collapse is where the
    host-lane requirement went missing: a region's dtype decides whether the hardware may take it, so
    dropping it makes "contraction, which this target admits" indistinguishable from "contraction at f32,
    which it does not". Measured on the real captures: gemmini admits four families and every one of them
    also appears at f32, 10,719 regions of work that must run on the host and that the requirement asked
    for no capsule of.

    A region whose family or dtype could not be resolved is not counted, exactly as in :func:`observed`:
    an unreadable region is evidence neither way.
    """
    from merlin.targetgen import model_coverage as mc

    out: Counter = Counter()
    for region in mc.regions_from_module(mc.load_module(capture)):
        family, dtype = getattr(region, "family", None), getattr(region, "in_dtype", None)
        if not family or not dtype:
            continue
        try:
            out[(str(family), capsule_dtype(str(dtype)))] += 1
        except Exception:                          # noqa: BLE001 -- an unmappable token stays visible
            out[(str(family), str(dtype))] += 1
    return out


def host_lane_cells(captures: dict, target: str) -> dict:
    """The ``(family, dtype)`` work real models contain that ``target`` may NOT accelerate.

    The negative lane, at full width. The existing ``host_only`` block carries only families with no
    admitted dtype AT ALL, which is the narrow case: a target admitting ``contraction`` at int8 still
    cannot take an f32 contraction, and a real model is full of them. Every pair here is work the
    compiler must place on the host, and a corpus with no capsule for it cannot tell a compiler that
    routes it correctly from one that does not.

    Derived by intersecting what the captures CONTAIN with what the manifest ADMITS, both as
    (family, dtype) pairs -- the same two sources the cells come from, read at the resolution the cells
    throw away.
    """
    adm = admitted(target)
    admitted_pairs = set()
    for family, dtypes in (adm or {}).items():
        for d in dtypes or ():
            try:
                admitted_pairs.add((str(family), capsule_dtype(str(d))))
            except Exception:                      # noqa: BLE001
                admitted_pairs.add((str(family), str(d)))

    seen: Counter = Counter()
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            seen.update(observed_pairs(path, target))
        except Exception as e:                     # noqa: BLE001 -- reported, never skipped silently
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"

    required = [{"family": f, "dtype": d, "n_regions": n}
                for (f, d), n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
                if (f, d) not in admitted_pairs]
    return {
        "required": required,
        "admitted_pairs": sorted(f"{f}/{d}" for f, d in admitted_pairs),
        "captures_unreadable": unreadable,
        "axis_basis": (
            "the (family, dtype) work real captures CONTAIN that this target's manifest does not admit. "
            "The cells intersect admitted with observed and keep only what survives; this keeps what does "
            "NOT, which is precisely the work the compiler has to place on the host. Both sides come from "
            "the same two sources the cells do, read at the resolution the cells discard -- a region's "
            "dtype decides whether the hardware may take it, so a family histogram cannot express it"),
    }


def boundaries(target: str) -> Boundaries:
    """The target's edges, each tagged with the artifact it was read from."""
    b = Boundaries()
    # Tile edge, and whether it is a hardware fact or the software-tiling default. `_tile_dim` resolves
    # a declared matrix unit / capabilities.mesh / capabilities.tile / an RTL `arrays` fact, and only then
    # falls back to a constant -- so "did it come from hardware" is decidable by asking for the fact
    # sources directly rather than by trusting the number.
    try:
        from merlin.targetgen.corpus_spec import _DEFAULT_SW_TILE, _tile_dim
        from merlin.targetgen.target_experiment import load_capability_manifest
        contract = load_capability_manifest(target).contract
        edge = int(_tile_dim(target, contract) or 0) or None
        b.tile_edge = edge
        caps = contract.get("capabilities") or {}
        hw = bool((caps.get("mesh") or {}).get("rows") or (caps.get("tile") or {}).get("rows"))
        # A target may DECLINE to restate its geometry in the contract and leave it to RTL discovery --
        # one does exactly that, with `capabilities: {}` and a comment saying the mesh is a CIRCT-extracted
        # fact. Asking only the contract then reports a real hardware mesh as a software guess, and it is
        # worst precisely when the mesh edge happens to equal the default, because the numeric fallback
        # below cannot tell them apart either. The value already comes from the RTL fact; the PROVENANCE
        # has to come from the same place or the two disagree about what is known.
        rtl_rows = None
        if not hw:
            try:
                from merlin.targetgen.rtl.facts import load_facts
                arrays = ((load_facts(target) or {}).get("facts") or {}).get("arrays") or []
                mesh = next((a for a in arrays if a.get("rows") and a.get("cols")), None)
                rtl_rows = int(mesh["rows"]) if mesh else None
            except Exception:                              # noqa: BLE001 — absent facts: not a hardware fact
                rtl_rows = None
        from_rtl = rtl_rows is not None and edge is not None and int(rtl_rows) == int(edge)
        b.tile_edge_is_hardware_fact = hw or from_rtl or (edge is not None and edge != _DEFAULT_SW_TILE)
        b.tile_edge_source = ("capability manifest (declared mesh/tile rows)" if hw else
                              "RTL facts arrays[].rows (the target leaves geometry to discovery rather "
                              "than restating it in the contract)" if from_rtl else
                              f"software-tiling default ({_DEFAULT_SW_TILE}); this target declares no "
                              f"fixed hardware mesh, so it is NOT a hardware boundary")
    except Exception as e:                                 # noqa: BLE001
        b.tile_edge_source = f"unavailable: {type(e).__name__}"

    # MX block-scale K-group. Lives in the target's MX MMIO contract, not in rtl/facts -- read it from the
    # one accessor so a second hardcoded 32 does not enter the tree.
    try:
        from merlin.targetgen.rtl.mlc_bridge import mx_mmio_for
        mx = mx_mmio_for(target) or {}
        grp = mx.get("group")
        if grp:
            b.block_scale_group = int(grp)
            b.block_scale_source = "target contract mx_mmio.group (one E8M0 scale per K group)"
    except Exception as e:                                 # noqa: BLE001
        b.block_scale_source = f"unavailable: {type(e).__name__}"

    # On-chip operand store, for capacity-fit extents. The facts artifact carries it as a MEMORIES LIST
    # keyed by name, not as a `shared_memory` mapping -- reading the mapping spelling silently yielded
    # None here, which would have published a spec claiming the target declares no operand store. Match
    # on the entry's name, then fall back to the contract, and record which one answered.
    try:
        from merlin.targetgen.rtl.facts import load_facts
        mems = ((load_facts(target) or {}).get("facts") or {}).get("memories") or []
        for m in mems:
            if isinstance(m, dict) and str(m.get("name")) == "shared_memory" and m.get("bytes"):
                b.operand_store_bytes = int(m["bytes"])
                b.operand_store_source = 'rtl facts memories[name="shared_memory"].bytes'
                break
    except Exception as e:                                 # noqa: BLE001
        b.operand_store_source = f"rtl facts unavailable: {type(e).__name__}"
    if b.operand_store_bytes is None:
        try:
            from merlin.targetgen.target_experiment import load_capability_manifest
            mm = (load_capability_manifest(target).contract.get("memory_model") or {})
            if mm.get("shared_memory_bytes"):
                b.operand_store_bytes = int(mm["shared_memory_bytes"])
                b.operand_store_source = "capability manifest memory_model.shared_memory_bytes"
        except Exception as e:                             # noqa: BLE001
            b.operand_store_source += f"; manifest unavailable: {type(e).__name__}"
    return b


# ---------------------------------------------------------------------------------------------------
# the requirement
# ---------------------------------------------------------------------------------------------------

def required_cells(target: str, captures: dict[str, str | Path], *,
                   declared: dict[str, dict] | None = None
                   ) -> tuple[dict[Cell, CellOrigin], dict]:
    """``admitted INTERSECT observed`` as cells, plus a diagnostics block.

    ``captures`` maps a label (the model it came from) to a captured ``.mlir``. ``declared`` adds families
    that no available capture can evidence, as ``{family: {"citation": str, "dtypes": [...]}}`` — used for
    a target-model with no importable workload. Declared entries are marked, never merged into observed.

    The diagnostics record what was DROPPED and why, because the interesting failure of a derived
    requirement is a silent narrowing: a family the models need that the hardware does not admit is a
    real coverage hole, and it must be visible rather than absent.
    """
    adm, adm_reason = admitted_with_reason(target)
    units = admitting_units(target)
    bnd = boundaries(target)
    # A tile edge admits three occupancy classes, not two -- see `Cell.alignment`. `sub_tile` is
    # required only where the edge is wide enough for "barely occupied" to differ from "ragged by
    # one": at edge 2, `tile-1` and `tile//2` are the same extent and a third cell would be a repeat.
    #
    # The bar is 8 rather than 4 because the SHAPE has to exist as well as the class: the sub-tile
    # extents are `tile/8` x `tile/4` (see `corpus_synth.extents_for`, which is spelled that way
    # because the smaller square spellings make the deterministic stimulus degenerate), and below
    # edge 8 that collapses to a zero extent. Demanding a cell no shape can express would be the
    # uncoverable-by-construction failure this axis was added to fix.
    if not bnd.tile_edge:
        aligns: tuple[str | None, ...] = (None,)
    elif int(bnd.tile_edge) >= 8:
        aligns = ("aligned", "partial", "sub_tile")
    else:
        aligns = ("aligned", "partial")

    # AND THE DTYPE NARROWS IT AGAIN. The classes above are what the TILE EDGE admits; a block-scaled
    # format imposes a granularity of its own on top (one E8M0 exponent per whole run of K, sub-byte
    # codes addressed in nibble pairs), and on this target that granularity is >= the tile edge. Every
    # extent the datapath can execute is then a whole multiple of it, so "ragged by one" and "barely
    # occupied" name shapes that CANNOT EXIST -- the golden refuses them and the reference returns an
    # all-zero result. Requiring them produced 11 capsules that failed to generate and 11 cells nothing
    # could ever cover, which is the uncoverable-by-construction failure the sub_tile guard above exists
    # to prevent, arriving through the dtype instead of through the edge.
    scale_block = None
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest
        scale_block = _cs()._scale_block_elems(load_capability_manifest(target).contract)
    except Exception:                                  # noqa: BLE001 — no manifest: no granularity known
        scale_block = None

    def _aligns_for(dt: str) -> tuple[str | None, ...]:
        q = _cs().shape_quantum(dt, tile_dim=bnd.tile_edge, scale_block=scale_block)
        if max(int(q.get("row") or 1), int(q.get("reduction") or 1)) <= 1:
            return aligns
        return tuple(a for a in aligns if a in (None, "aligned"))

    seen: Counter = Counter()
    seen_in: dict[str, list[str]] = {}
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            hist = observed(path, target)
        except Exception as e:                             # noqa: BLE001 — an unreadable capture is
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"   # reported, never skipped silently
            continue
        for fam, n in hist.items():
            seen[fam] += n
            seen_in.setdefault(fam, []).append(label)

    # A COMPOSITE FAMILY IS DECOMPOSED BY THE CAPTURE, so it never appears as a region of its own.
    # Measured: `attention` and `softmax` occur in ZERO of the four real captures, because the importer
    # lowers attention into contraction + reduction + elementwise_map before merlin ever sees it. Taking
    # the census literally therefore dropped attention from the requirement for a corpus whose target
    # models are transformers -- the single most important thing to cover, excluded by an accounting
    # artifact. A composite counts as observed when EVERY primitive it decomposes into is observed
    # (`semantic_families.primitives_of`), which is evidence from the same captures rather than a family
    # added by hand.
    composite_via: dict[str, tuple[str, ...]] = {}
    for fam in sorted(set(adm) - set(seen)):
        prims = _sf().primitives_of(fam)
        if not prims or tuple(prims) == (fam,):
            continue                                       # a primitive that is simply absent
        if all(p in seen for p in prims):
            composite_via[fam] = tuple(prims)
            seen[fam] = min(int(seen[p]) for p in prims)    # bounded by its scarcest primitive
            seen_in[fam] = sorted({lb for p in prims for lb in seen_in.get(p, ())})

    cells: dict[Cell, CellOrigin] = {}
    #: dtypes whose own format granularity removed alignment classes the tile edge would have admitted.
    #: Recorded, never silent: an absent cell and a cell nobody could cover read the same downstream.
    quantized_dtypes: dict[str, list] = {}
    for fam in sorted(set(seen) | set(declared or {})):
        is_declared = fam not in seen
        dtypes = tuple(adm.get(fam) or ())
        if is_declared:
            want = tuple((declared or {}).get(fam, {}).get("dtypes") or dtypes)
            dtypes = tuple(d for d in want if d in dtypes) or dtypes
        if not dtypes:
            continue                                       # not admitted at all: recorded in diagnostics
        if is_declared:
            basis = DECLARED
        elif fam in composite_via:
            basis = OBSERVED_VIA_PRIMITIVES
        else:
            basis = OBSERVED
        for dt in dtypes:
            cdt = capsule_dtype(dt)
            dt_aligns = _aligns_for(cdt)
            if dt_aligns != aligns:
                quantized_dtypes[cdt] = list(dt_aligns)
            for al in dt_aligns:
                cell = Cell(fam, cdt, al)
                cells[cell] = CellOrigin(
                    basis=basis,
                    observed_in=tuple(sorted(seen_in.get(fam, ()))),
                    admitted_by=units.get((fam, capsule_dtype(dt)), ()),
                    citation=(declared or {}).get(fam, {}).get("citation", "") if is_declared else "",
                    via_primitives=composite_via.get(fam, ()),
                    n_regions=int(seen.get(fam, 0)))

    needed_not_admitted = sorted(f for f in seen if not adm.get(f))
    admitted_not_needed = sorted(f for f in adm if f not in seen and f not in (declared or {}))
    diagnostics = {
        "n_cells": len(cells),
        "captures_read": sorted(k for k in (captures or {}) if k not in unreadable),
        "captures_unreadable": unreadable,
        "families_observed": {f: int(n) for f, n in sorted(seen.items())},
        "families_observed_via_primitives": {f: list(p) for f, p in sorted(composite_via.items())},
        "families_needed_but_not_admitted": needed_not_admitted,
        "families_admitted_but_no_model_uses": admitted_not_needed,
        "alignment_axis": list(aligns),
        # The per-dtype narrowing of that axis. EMPTY means "checked, no dtype narrows it" -- not
        # "nobody looked": the key is always present once this derivation has run.
        "alignment_axis_narrowed_by_dtype": {d: list(a) for d, a in sorted(quantized_dtypes.items())},
        "scale_block_elements": scale_block,
        # Why the admitted side is what it is. Load-bearing when it is EMPTY: "no family intersects" is
        # an answer, "no contract to read" is a missing artifact, and a caller that cannot tell them
        # apart reports a target with no generated package as cleanly requiring nothing.
        "admitted_status": adm_reason,
        "notes": [],
    }
    if adm_reason != "resolved":
        diagnostics["notes"].append(
            f"this target's capability contract did not resolve ({adm_reason}), so NOTHING is admitted "
            f"and the requirement below is UNKNOWN rather than empty; generate the target's package "
            f"before reading any coverage number for it")
    if composite_via:
        diagnostics["notes"].append(
            f"{len(composite_via)} composite family/families ({', '.join(sorted(composite_via))}) appear "
            f"in NO capture as a region of their own because the importer decomposes them; they are "
            f"required on the evidence that every primitive they decompose into is observed. Taking the "
            f"region census literally would drop attention from a transformer corpus")
    if needed_not_admitted:
        diagnostics["notes"].append(
            f"{len(needed_not_admitted)} family/families appear in a real capture but the hardware "
            f"declares no capability for them ({', '.join(needed_not_admitted)}); those are compiler "
            f"work for the scalar/vector lane, not accelerator cells, and are excluded from the "
            f"requirement rather than silently counted as covered")
    if unreadable:
        diagnostics["notes"].append(
            f"{len(unreadable)} capture(s) could not be read; the requirement below is NARROWER than the "
            f"evidence would support and must not be read as complete")
    if not bnd.tile_edge_is_hardware_fact and bnd.tile_edge:
        diagnostics["notes"].append(
            f"the alignment axis uses tile edge {bnd.tile_edge}, which is a SOFTWARE tiling default for "
            f"this target, not a hardware boundary")
    # THE DTYPE AXIS IS ADMITTED-ONLY, NOT OBSERVED, and that asymmetry is deliberate. The available
    # captures are single-precision (fp32), so observing dtypes would collapse the requirement onto f32
    # and drop exactly the MX/int8 paths this accelerator exists for. Requiring every admitted dtype of
    # an observed family over-requires instead -- a cell like `attention/f32/partial` may be one no real
    # deployment runs. Over-requiring is the safe direction (it surfaces as an uncovered cell to argue
    # about, not as unearned coverage), but a reader must not take cell count as demand.
    diagnostics["axis_basis"] = {
        "semantic_family": "observed in a real capture (or all of its primitives were)",
        "dtype": "ADMITTED ONLY — the captures are single-precision, so dtype demand is not observable "
                 "from them; every dtype the hardware declares for an observed family is required",
        "tile_alignment": ("both, wherever the target tiles: a unit that only ever sees whole tiles has "
                           "never exercised its tail path"),
    }
    return cells, diagnostics


def host_only_dtypes(captures: dict, families) -> dict:
    """``family -> capsule dtype`` for families the HOST must carry, read from the captures themselves.

    A host-only family has no admitted dtype by construction -- the hardware declares no capability for
    it -- so the dtype cannot come from the manifest the way a cell's does. It comes from the real
    captures instead: the dtype those regions actually carry, most frequent first. A family nobody could
    size is ABSENT from the result rather than defaulted, because a host capsule emitted at a dtype no
    model uses tests a program nobody runs.
    """
    from collections import Counter

    from merlin.targetgen import model_coverage as mc

    want = {str(f) for f in families}
    if not want:
        return {}
    seen: dict[str, Counter] = {f: Counter() for f in want}
    for path in captures.values():
        try:
            regions = mc.regions_from_module(mc.load_module(Path(path)))
        except Exception:                          # noqa: BLE001 -- an unreadable capture is not evidence
            continue
        for region in regions:
            fam = region.resolved_family()
            if fam in want and getattr(region, "in_dtype", None):
                seen[fam][str(region.in_dtype)] += 1
    out = {}
    for fam, counts in seen.items():
        if counts:
            try:
                out[fam] = capsule_dtype(counts.most_common(1)[0][0])
            except Exception:                      # noqa: BLE001 -- an unmappable spelling is not a dtype
                continue
    return out


#: How long a single cycle-accurate certification may take before a capsule is too big to be one.
#: A POLICY number, not a hardware fact -- it says how much simulator time this experiment is willing
#: to spend, so it is declarable per target and defaulted here rather than derived. The default is
#: recorded in the axis so a reader can tell a declared budget from an inherited one.
_DEFAULT_CERT_BUDGET_S = 300.0


def _application_axis(target: str, *, captures: dict | None = None,
                      budget_s: float | None = None) -> dict:
    """The shapes this target's declared APPLICATIONS contain, sized to what a cert costs.

    The corpus gives every synthesized capsule one of two tile-relative shapes, and real models do
    not look like that -- 757 contraction regions across six captures carry shapes like
    ``(8,2048,32000)``, against a corpus that tests 16x16x16. This axis is where a user's own
    applications reach the requirement.

    It emits at most two capsules per behavioural class, and the split is what makes the whole thing
    runnable: one sized to be affordable cycle-accurately, and -- where the application is larger --
    one at the true shape that EXTENDS it. See :mod:`merlin.targetgen.applications` for why K is
    never clamped and why a class with no cost model is refused rather than sized by convention.

    Empty and inert for a target declaring no applications, which is every target today.
    """
    from merlin.targetgen import applications as APP
    from merlin.targetgen import cert_cost as CC

    budget = float(budget_s) if budget_s else _DEFAULT_CERT_BUDGET_S
    basis = {
        "axis_basis": (
            "the contraction shapes this target's DECLARED applications contain, grouped by what the "
            "compiler must do with them and sized to what a certification costs. A capsule at an "
            "application's real shape is worthless if nobody can afford to certify it, so each class "
            "yields a cycle-accurate capsule at an affordable size plus, when the application is "
            "larger, an L2 capsule at the true shape that extends it -- never one without the other"),
        "cert_budget_s": budget,
        "budget_source": "declared" if budget_s else "default",
    }
    if not captures:
        return {"required": [], "refused": [], "declared_applications": 0, **basis}

    grouped = APP.classify_captures(captures, target)
    fit = CC.fit_for(target)
    try:
        from merlin.targetgen.corpus_spec import _tile_dim  # noqa: PLC2701
        from merlin.targetgen.target_registry import load_contract
        tile = int(_tile_dim(target, load_contract(target)) or 0)
    except Exception:                              # noqa: BLE001 -- no edge is a real answer
        tile = 0

    required, refused = [], []
    for row in grouped.get("classes") or ():
        evidence = APP.ClassEvidence(
            region_class=APP.RegionClass(
                family=row["family"], dtype=row["dtype"], alignment=row["alignment"],
                regime=row["regime"], rank=int(row["rank"]), geometry=row["geometry"]),
            m=int(row["M"]), k=int(row["K"]), n=int(row["N"]), batch=int(row["batch"]),
            multiplicity=int(row["multiplicity"]), work=int(row["work"]),
            work_complete=bool(row["work_complete"]), source=str(row["source"]))
        sized, refusal = APP.size_class(evidence, target=target, budget_s=budget,
                                        tile=tile or None, fit=fit)
        if refusal:
            refused.append(refusal)
            continue
        required.extend(cap.to_dict() for cap in sized)
    return {
        "required": required, "refused": refused,
        "declared_applications": len(captures),
        "n_classes": grouped.get("n_classes", 0),
        "n_regions": grouped.get("n_regions", 0),
        "total_work": grouped.get("total_work", 0),
        "captures_unreadable": grouped.get("captures_unreadable") or {},
        "cost_model": fit.to_dict() if fit is not None else None,
        **basis,
    }


#: What the shape axis rests on. One string, because it is stated on the derived path and on the
#: unreadable-manifest path alike, and two copies would drift into two claims.
_SHAPE_AXIS_BASIS = (
    "the rank and operand-layout regions this target's capability manifest DECLARES it handles, via "
    "capability_probes. A (family, dtype, alignment) cell cannot express either: it says what arithmetic "
    "the corpus must contain and nothing about whether the unit was ever asked for a batched region or a "
    "transposed operand. Both are declared capabilities, so a target that claims them and is never asked "
    "for one has an untested claim")


def _mkn(shape) -> "tuple[int, int, int] | None":
    """``(M, K, N)`` for a contraction named by ITERATOR TYPE, or ``None``.

    ``ContractionShape`` deliberately does not say M/N/K -- it says which extents iterate in parallel
    and which reduce. The geometry taxonomy is written in M/N/K, so the two are joined HERE, once: the
    innermost two parallel extents are ``(M, N)``, every LEADING parallel extent is a batch and
    multiplies into M (a batched matmul writes ``B x M`` result rows, and that is what the cost of
    draining it follows), and the reduction extents multiply into K.
    """
    par = tuple(int(x) for x in (getattr(shape, "parallel", ()) or ()))
    red = tuple(int(x) for x in (getattr(shape, "reduction", ()) or ()))
    if len(par) < 2 or not red:
        return None
    lead = 1
    for x in par[:-2]:
        lead *= x
    m, n = par[-2] * lead, par[-1]
    k = 1
    for x in red:
        k *= x
    return (m, k, n) if m > 0 and k > 0 and n > 0 else None


#: Largest written output a capsule may declare when the target's OWN operand store is not derivable.
#: MEASURED, not chosen: it is the largest operand any capsule this corpus already materialises
#: (`SY_regime_spills` on the one target with a derivable store, 262,144 elements behind a 2.4 MB
#: golden), so it is the only size this repo has evidence it can actually build. It is a fallback, not
#: a default -- a target whose store IS readable is bounded by that store, which is the physically
#: meaningful number. Without it a geometry class drawn at its real extents emitted a 128 x 256,000
#: capsule whose golden is 32.7 MILLION elements: the generator ran for 83 minutes on that one capsule
#: and had produced no golden when it was killed.
_MATERIALIZABLE_ELEMENTS_FALLBACK = 262144


def _materialization_ceiling(target: str) -> "int | None":
    """The largest written output a capsule of this target may declare, in elements, or ``None``.

    DERIVED, not declared: it is the capacity of the target's own operand store, which is exactly the
    size the memory-regime axis already sizes its largest capsule against. A capsule bigger than the
    store it runs on tests nothing about residency that `SY_regime_spills` does not, and its golden --
    written as text, one number per element -- stops being materializable long before it stops being
    interesting. ``None`` where the store is not derivable: then nothing is scaled, and an oversized
    class is reported by the caller rather than silently shrunk.
    """
    try:
        from merlin.targetgen import memory_regime as MR
        bnd = boundaries(target)
        dtype = None
        from merlin.targetgen.target_experiment import load_capability_manifest
        units = (load_capability_manifest(target).contract.get("compute_units") or [{}])
        for u in units:
            for d in (u.get("dtypes") or ()):
                dtype = capsule_dtype(str(d))
                break
            if dtype:
                break
        store, rows = MR.operand_store(target, dtype=dtype)
        if store is None or not rows:
            return _MATERIALIZABLE_ELEMENTS_FALLBACK
        per_row = store.elems_per_row(dtype)
        return int(rows) * int(per_row) if per_row else _MATERIALIZABLE_ELEMENTS_FALLBACK
    except Exception:                                       # noqa: BLE001 — no store: the fallback
        return _MATERIALIZABLE_ELEMENTS_FALLBACK


def geometry_axis(captures: dict[str, str | Path], target: str) -> dict:
    """Which GEOMETRY CLASSES real models present, with the representative shape of each.

    The axis the cells cannot express, and the one that decides whether the corpus resembles the work.
    A ``(family, dtype, alignment)`` cell says WHAT arithmetic runs and how it sits against the tile
    edge; it says nothing about the aspect ratio of the problem, and aspect ratio is what a tiling
    compiler mostly gets wrong. Measured on a real capture: every synthesized capsule in this corpus
    is ``projection_like`` (M ~ N ~ tile), while the model it is derived from presents FIVE classes,
    and the largest single block of its convolution work is ``tall_skinny`` at ``M=3584, K=14, N=8``
    -- an aspect ratio of 448:1 that nothing in the corpus comes within two orders of magnitude of.

    The representative of a class is its highest-MAC-mass shape, so the capsule tests the geometry at
    the size the work actually occurs at rather than at an arbitrary small one. Every entry carries
    ``out_elements`` because that is what decides its tier: a class whose representative writes more
    than the target can certify inside the budget is a LOOP-tier capsule resting on a certified
    sibling, and saying so here is what keeps the axis from demanding a cert nobody can run.
    """
    from merlin.dse_guidance import shape_taxonomy as ST
    from merlin.kernels import shapes as KS

    by_class: dict[str, dict] = {}
    unreadable: dict[str, str] = {}
    total_macs = 0
    for label, path in sorted((captures or {}).items()):
        try:
            pairs = KS.observe_contractions(path)
        except Exception as e:                              # noqa: BLE001 — reported, never skipped
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        for _op, shape in pairs:
            mkn = _mkn(shape)
            if mkn is None:
                continue
            m, k, n = mkn
            macs = m * k * n
            total_macs += macs
            klass = ST.classify_geometry(M=m, N=n, K=k)
            row = by_class.setdefault(klass, {"class": klass, "family": "contraction",
                                              "n_regions": 0, "macs": 0, "observed_in": set(),
                                              "M": m, "K": k, "N": n, "rep_macs": 0})
            row["n_regions"] += 1
            row["macs"] += macs
            row["observed_in"].add(label)
            if macs > row["rep_macs"]:                      # the representative is the heaviest shape
                row["M"], row["K"], row["N"], row["rep_macs"] = m, k, n, macs

    # A CLASS IS A RATIO, NOT A SIZE, and that is what makes this axis affordable. The heaviest real
    # shape in `wide_skinny` carries a 590-MILLION-element weight -- a golden nothing in this repo can
    # materialize, let alone simulate -- but the thing the capsule has to reproduce is the 1:2000
    # aspect ratio, not the absolute extent. So the representative is scaled DOWN by a common factor on
    # ALL THREE extents until its largest tensor fits what the target's own operand store holds, and
    # the result is re-classified: a capsule is emitted only if the scaled shape is still in the class
    # it was drawn from. Same minimal-but-representative rule the accumulation-depth axis uses, applied
    # to aspect ratio.
    #
    # THE BOUND IS THE LARGEST TENSOR, NOT THE WRITTEN OUTPUT, and the difference is not academic.
    # Certification cost tracks the output (measured, r2 0.92) and reduction depth is therefore nearly
    # free to certify -- but the GOLDEN has to synthesize and write every operand element, and a skewed
    # aspect ratio has a colossal operand behind a small output. Bounding the output alone admitted an
    # 11 x 2304 by 2304 x 23272 capsule whose output is 255,992 elements and whose weight is 53.6
    # MILLION: the generator held 14.7 GB and had produced nothing after 27 minutes of CPU.
    # `cert_cost.MEASURED_MAX_OPERAND_ELEMENTS` already says this in as many words -- "treating output
    # as the only bound there would price an enormous transfer at zero, so a caller must refuse rather
    # than extrapolate" -- and this axis was the caller that did not.
    ceiling = _materialization_ceiling(target)

    def _largest(mm: int, kk: int, nn: int) -> int:
        return max(mm * kk, kk * nn, mm * nn)

    # A CEILING WITHOUT A FLOOR PRODUCES A CAPSULE THAT PROVES NOTHING. Searching only downward, the
    # 1:2000 class scaled to `1 x 1 x 31` -- still `wide_skinny` by the taxonomy, and a contraction
    # with no reduction and one output row, which is exactly the "too small to be representative"
    # end of the trade this axis exists to sit inside. So an extent may shrink to the tile edge and no
    # further -- EXCEPT one that was already smaller, because that is the class rather than a
    # reduction of it: a GEMV's M is 1 in the application, and flooring it at the tile would make
    # every gemv-like class unreachable by definition.
    edge = int(boundaries(target).tile_edge or 0) or 1

    def _floor(original: int) -> int:
        return min(int(original), edge)

    required = []
    for klass, row in sorted(by_class.items()):
        m, k, n = int(row["M"]), int(row["K"]), int(row["N"])
        scaled_from = None
        if ceiling and _largest(m, k, n) > ceiling:
            # The SMALLEST factor that fits, searched over every integer rather than over powers of
            # two: the class is a ratio, so any common divisor preserves it in principle, and the
            # coarse ladder overshot far enough to push a shape out of its own class and lose the
            # coverage entirely. The first factor that both fits and re-classifies to the same class
            # wins; if none does, the class is genuinely unrepresentable at this store size.
            fm, fk, fn = (_floor(row["M"]), _floor(row["K"]), _floor(row["N"]))
            m = n = 0
            for factor in range(2, 8193):
                cm = max(1, int(row["M"]) // factor)
                cn = max(1, int(row["N"]) // factor)
                ck = max(1, int(row["K"]) // factor)
                if cm < fm or ck < fk or cn < fn:
                    break                               # past the floor: shrinking further is not a
                if _largest(cm, ck, cn) > ceiling:      # smaller representative, it is a different
                    continue                            # and degenerate problem
                if ST.classify_geometry(M=cm, N=cn, K=ck) == klass:
                    m, n, k = cm, cn, ck
                    break
            if not m or not n:
                required.append({
                    "class": klass, "family": row["family"],
                    "M": None, "K": None, "N": None,
                    "out_elements": int(row["M"]) * int(row["N"]),
                    "n_regions": int(row["n_regions"]),
                    "mac_fraction": round(row["macs"] / float(total_macs), 6) if total_macs else None,
                    "observed_in": sorted(row["observed_in"]),
                    "unreachable": (
                        f"the heaviest shape in this class carries a "
                        f"{_largest(int(row['M']), int(row['K']), int(row['N']))}-element tensor, more "
                        f"than the {ceiling} this target's operand store holds, and no common divisor "
                        f"of its extents fits while every extent stays at or above its floor "
                        f"(min(original, tile edge {edge})) and the shape stays in its class -- so no "
                        f"capsule of this aspect ratio is both representative and buildable here"),
                })
                continue
            scaled_from = {"M": int(row["M"]), "K": int(row["K"]), "N": int(row["N"]),
                           "factor": factor}
        out_elems = m * n
        required.append({
            "class": klass,
            "family": row["family"],
            "M": m, "K": k, "N": n,
            "out_elements": out_elems,
            "n_regions": int(row["n_regions"]),
            "mac_fraction": round(row["macs"] / float(total_macs), 6) if total_macs else None,
            "observed_in": sorted(row["observed_in"]),
            "scaled_from": scaled_from,
        })
    return {
        "required": required,
        "captures_unreadable": unreadable,
        "materialization_ceiling_elements": ceiling,
        "total_macs": total_macs,
        "axis_basis": (
            "the geometric classes (shape_taxonomy.classify_geometry) that real captured models "
            "present, each with the highest-MAC-mass shape in the class. A (family, dtype, alignment) "
            "cell cannot express aspect ratio at all, and aspect ratio is where a tiling compiler "
            "fails: a 448:1 tall-skinny convolution and a square projection are the same cell. EMPTY "
            "means the captures contain no readable contraction, never that geometry does not matter"),
    }


def _capsule_geometry(cap: dict) -> "str | None":
    """The geometry class of a capsule's own contraction extents, or ``None`` when it has none.

    Read off the DECLARED inputs, which is what the harness materialises -- the same source
    ``cert_capsule_cover`` classifies alignment from. A capsule with two rank-2 inputs sharing an inner
    extent is a contraction: ``A0[M, K]`` and ``W[K, N]``. Anything else (one input, a rank-4 image, a
    broadcast row) is not a contraction and returns ``None`` rather than a class it does not have.
    """
    from merlin.dse_guidance import shape_taxonomy as ST

    shapes = [[int(x) for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()]
              for t in (cap.get("inputs") or [])]
    twod = [sh for sh in shapes if len(sh) == 2 and all(d > 0 for d in sh)]
    for i, a in enumerate(twod):
        for j, b in enumerate(twod):
            if i != j and a[1] == b[0]:
                return ST.classify_geometry(M=a[0], N=b[1], K=a[1])
    return None


def _geometry_gap(required, corpus_roots, *, labels=None, exclude=None) -> dict:
    """Which required geometry classes no capsule in the corpus presents."""
    import yaml

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    have: dict[str, list[str]] = {}
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    for root in roots:
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
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
        "mac_fraction_uncovered": round(
            sum(float(by_class[k].get("mac_fraction") or 0.0) for k in missing), 6),
        "note": ("a geometry class real models present that no capsule reproduces means the corpus "
                 "cannot tell a compiler that tiles that aspect ratio well from one that does not; "
                 "`mac_fraction_uncovered` is the share of real contraction work sitting in classes "
                 "nothing tests"),
    }


#: Epilogue stages the capsule builder can actually emit. Not a guess about hardware -- it is the
#: writer's vocabulary, and a stage outside it could be required and never written.
_BUILDER_EPILOGUE_STAGES = ("relu", "acc_scale", "bias_add", "maxpool")


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
    instruction taxonomy resolves a class for the role the stage needs. Each stage records which of the
    two evidenced it, because "the manifest declares it" and "the ISA has an instruction for it" are
    different claims and a reader must be able to tell them apart.
    """
    from merlin.targetgen import isa_taxonomy as IT
    from merlin.targetgen import semantic_families as sf

    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(target) or {}
    except Exception:                              # noqa: BLE001 -- unreadable manifest evidences nothing
        cap_map = {}
    fused_families = {f for f, c in cap_map.items() if tuple(getattr(c, "composed_with", ()) or ())}

    base: set = set()
    taxonomy = None
    try:
        taxonomy = IT.taxonomy_for_target(target)
        base = set(IT.required_classes_for_op(taxonomy, op="matmul"))
    except Exception:                              # noqa: BLE001 -- no taxonomy is not "no capability"
        taxonomy = None

    required, rejected = [], []
    for stage in _BUILDER_EPILOGUE_STAGES:
        family = sf.from_op(stage)
        by_manifest = family in fused_families
        classes: list = []
        if taxonomy is not None:
            try:
                classes = sorted(set(IT.required_classes_for_op(
                    taxonomy, op="matmul", epilogue=(stage,))) - base)
            except Exception:                      # noqa: BLE001
                classes = []
        if by_manifest or classes:
            required.append({
                "stage": stage, "family": family,
                "evidenced_by": ([  "manifest_composed_with"] if by_manifest else [])
                                + (["isa_instruction_class"] if classes else []),
                "isa_classes": classes,
            })
        else:
            rejected.append({
                "stage": stage, "family": family,
                "why": ("the manifest declares no family fused-only for it and this target's "
                        "instruction taxonomy resolves no class for the role it needs"),
            })
    return {
        "required": required,
        "rejected": rejected,
        "axis_basis": (
            "the epilogue stages this target can fuse onto a contraction, evidenced by its capability "
            "manifest declaring the stage's family fused-only OR by its own instruction taxonomy "
            "resolving a class for the role the stage needs. A (family, dtype, alignment) cell cannot "
            "express WHICH epilogue rides the contraction, so a corpus derived from cells alone tests "
            "one stage and calls the fusion capability covered"),
    }


def _shape_axis(target: str) -> dict:
    """The rank and layout regions ``target``'s own manifest declares it handles.

    Derived from :mod:`merlin.targetgen.capability_probes`, which already turns each family's declared
    capability into region descriptors -- ``batch`` into a rank-3 region, ``transpose`` and each declared
    layout into an operand-layout variant. Nothing consumed them outside the fuzzer, so a target could
    declare batching and never be asked for a batched region by anything that grades it.

    Fails soft: a target whose capability map cannot be read contributes an empty axis with the reason,
    because an unreadable manifest is not a target that declares no batching.
    """
    try:
        from merlin.targetgen.capability_probes import synthesize as _probes
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(target) or {}
        probes = _probes(cap_map, target=target)
    except Exception as exc:                               # noqa: BLE001 -- unreadable is not empty
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
            continue                                       # a plain 2-D region: the cells already say it
        required.append({
            "probe": pr.name,
            "axis": "rank" if rank >= 3 else "layout",
            "family": d.family,
            "dtype": d.in_dtype,
            "rank": rank,
            "layout": layout,
            "m": d.m, "k": d.k, "n": d.n,
            "batch": int(getattr(d, "batch", 1) or 1),
        })
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
            "point, not a corner probe under another name"),
    }


def classify_alignment(extents, tile_dim: int) -> "str | None":
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
    from merlin.targetgen import cert_cost as CC

    if not tile:
        return None, "no tile edge, so there is no whole-tile depth to size"
    budget = float(budget_s if budget_s else _DEFAULT_CERT_BUDGET_S)
    seconds, extrapolated = CC.predict_seconds_from_output(int(tile) * int(tile))
    if seconds is None:
        return None, "no measured certification cost law, so one output tile cannot be priced"
    if seconds > budget:
        return None, (f"a single {tile}x{tile} output tile already costs {seconds:.0f}s against a "
                      f"{budget}s budget on this target, so no reduction of any depth is certifiable")
    return {
        "M": int(tile), "K": 2 * int(tile), "N": int(tile), "K_tiles": 2,
        "predicted_seconds": round(seconds, 1), "budget_s": budget,
        "sized_by": "multi_pass_minimum_at_one_output_tile",
        "extrapolated": bool(extrapolated),
        "why": ("two tiles of K is the shallowest reduction that writes the accumulator twice; the "
                "cost is the output tile, which K does not move"),
    }, None


def _declared_oracle_tiers(target: str) -> list:
    """The oracle tiers ``target`` declares, cheapest first, or ``[]`` when none can be resolved.

    Read from the same adapter registry the runner grades through, so the requirement cannot name a
    tier the run does not have. ``[]`` is "we could not resolve them" and the caller must then leave a
    capsule's tier alone rather than cap it onto a guess.
    """
    try:
        from merlin.common.paths import repo_root
        from merlin.targetgen import capsule_runner as CR
        from merlin.targetgen.target_experiment import load_target_experiment
        desc = (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / str(target)
                / "target_experiment.yaml")
        if not desc.is_file():
            return []
        te = load_target_experiment(desc)
        return sorted(CR.oracle_adapters(target, te.sim_via) or {})
    except Exception:                                  # noqa: BLE001 — unresolvable: report nothing
        return []


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
    from merlin.targetgen import cert_cost as CC

    budget = float(budget_s if budget_s else _DEFAULT_CERT_BUDGET_S)
    why = ("the largest written output whose predicted certification fits the budget. A capsule above "
           "it is graded at the loop tier and rests on a certified sibling -- not because it is "
           "uninteresting, but because nobody can afford to run it cycle-accurately. Reduction depth "
           "does NOT count against this: K moves the operands, not the result")

    # THIS TARGET'S OWN CERTIFICATIONS FIRST. A simulation rate is a property of the DESIGN and the
    # simulator, not of the corpus, so pricing one target's capsules with another's measurements is a
    # claim with no evidence behind it -- and that is what happened: the ladder constants below were
    # measured on one target and every target got them, so three targets' capsules were sized against
    # a device none of them is. Where a target has its own history the two methods can be compared,
    # and on the calibrated target they agree to within 2%, which is why the fallback is usable at all.
    fit = None
    try:
        fit = CC.fit_for(target)
    except Exception:                              # noqa: BLE001 -- unreadable run history is "none"
        fit = None
    if fit is not None and getattr(fit, "per_element_s", 0):
        ceiling = int((budget - float(fit.intercept_s)) / float(fit.per_element_s))
        if ceiling >= 1:
            return {
                "max_elements": ceiling,
                "budget_s": budget,
                "metric": "written_output_elements",
                "law": f"seconds = {fit.intercept_s:.6g} + {fit.per_element_s:.6g} * output",
                "calibrated_to_elements": int(fit.elements_max),
                "extrapolated": bool(ceiling > int(fit.elements_max)),
                "basis": (f"fitted on {fit.n_samples} cycle-accurate certification(s) of THIS target "
                          f"(r2 {fit.r2:.3f}, {fit.elements_min}..{fit.elements_max} written elements)"),
                "why": why,
            }

    coeff = getattr(CC, "MEASURED_COEFFICIENT_S", None)
    exponent = getattr(CC, "MEASURED_EXPONENT", None)
    if not coeff or not exponent:
        return {"max_elements": None, "budget_s": budget, "metric": "written_output_elements",
                "basis": "no measured certification history for this target and no fallback law",
                "why": "no measured certification cost law is available in this checkout"}
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
        "basis": ("BORROWED: this target has certified nothing cycle-accurately, so the ceiling comes "
                  "from the deliberate calibration ladder measured on another device. It bounds the "
                  "budget rather than describing this device, and a capsule sized by it is sized by a "
                  "rate nobody has measured here -- run one cert on this target to replace it"),
        "why": why,
    }


def spec(target: str, captures: dict[str, str | Path], *,
         declared: dict[str, dict] | None = None,
         personas: dict[str, dict] | None = None,
         applications: dict[str, str | Path] | None = None,
         cert_budget_s: float | None = None) -> dict:
    """The full derived conformance spec, ready to serialize.

    Regenerable and tracked: a reviewer diffs it to see the requirement change when a target's manifest
    or a target-model's capture changes, which is exactly the drift a hand-authored list hides.
    """
    cells, diag = required_cells(target, captures, declared=declared)
    bnd = boundaries(target)
    from merlin.targetgen import boundary as BD
    from merlin.targetgen import memory_regime as MR
    comp = BD.required_boundaries(captures, target)
    mem = MR.required_regimes(captures, target)
    # The extents that REACH each required regime, resolved here because the search needs the target's
    # address space -- exactly the I/O `corpus_synth` is not allowed to do. Emitting them into the spec
    # keeps the synthesizer pure and keeps one definition of what a regime costs.
    _dtypes = [c.dtype for c in cells]
    _regime_dtype = max(set(_dtypes), key=_dtypes.count) if _dtypes else None
    HL = host_lane_cells(captures, target)
    try:
        _reduction_depth = MR.reduction_depth_regimes(
            target, sorted((mem.get("by_regime") or {}).keys()),
            tile_dim=bnd.tile_edge or 0, dtype=_regime_dtype)
    except Exception as _exc:                      # noqa: BLE001 -- an underivable depth is not zero
        _reduction_depth = {"unavailable": f"{type(_exc).__name__}: {_exc}"}
    _reduction_depth["certified"], _reduction_depth["certified_refusal"] = _certified_depth(
        target, tile=bnd.tile_edge or 0, budget_s=cert_budget_s)
    _regime_extents = MR.required_regime_extents(
        target, sorted((mem.get("by_regime") or {}).keys()),
        tile_dim=bnd.tile_edge or 0, dtype=_regime_dtype)
    return {
        "target": target,
        "generated_by": "merlin.targetgen.conformance.spec",
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
                "demand `routing` and nothing else, the narrowest reading of the richest evidence"),
            "why_orthogonal": (
                "the composition axis is NOT crossed with family/dtype/alignment. A cross product would "
                "demand cells like `movement/i8/partial/routing` that no real model presents, "
                "manufacturing uncovered cells nobody should build; composition is a property of how a "
                "program is assembled, not of the arithmetic in it"),
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
        # THE USER'S OWN APPLICATIONS. Empty unless the target declares some; when it does, this is
        # the only axis whose capsules carry a shape a real model contains rather than a tile
        # multiple, and the only one whose sizing is bounded by what a certification costs.
        "application_shapes": _application_axis(target, captures=applications,
                                                budget_s=cert_budget_s),
        # WHAT A CERTIFICATION COSTS HERE, so an axis can size against it instead of assuming every
        # capsule it derives is affordable at the deepest tier.
        "cert_affordability": _cert_affordability(target, budget_s=cert_budget_s),
        # THE TIERS THIS TARGET ACTUALLY DECLARES. Published because a capsule too large to certify has
        # to be capped to a tier that EXISTS here, and "L2" is not a universal name: one target in this
        # repo declares `[L3]` alone, so capping to L2 there names a tier nothing can run. The
        # synthesizer is pure and cannot ask the adapter registry, so the answer travels with the spec.
        "oracle_tiers": _declared_oracle_tiers(target),
        # THE ASPECT-RATIO AXIS. Emitted beside the cells rather than crossed with them, for the same
        # reason the composition axis is: a cross product would demand geometries at dtypes no model
        # presents. See `geometry_axis` for why a corpus that is entirely square proves nothing about
        # the tall-skinny convolutions that carry a real vision model's work.
        "shape_geometry": geometry_axis(captures, target),
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
                "else will report it either -- the schedule is correct whatever it chooses"),
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
            "dtypes": host_only_dtypes(captures, diag.get("families_needed_but_not_admitted") or ()),
            "observed_in": {f: n for f, n in (diag.get("families_observed") or {}).items()
                            if f in set(diag.get("families_needed_but_not_admitted") or ())},
            "basis": (
                "families a real capture contains and this target's capability manifest does not admit. "
                "The compiler must route them to the host lane; accelerating one is as much a defect as "
                "failing to accelerate an admitted family. EMPTY means no negative lane is derivable "
                "for this target -- every family its captures contain is admitted -- and never that "
                "none is needed: a target with an empty complement simply cannot be asked this "
                "question, which is a different fact from passing it"),
        },
        "diagnostics": diag,
        "personas": personas or {},
        # `shape_quantum` travels WITH the cell because the synthesizer is pure: it cannot ask a
        # manifest what granularity this dtype's datapath imposes, and a capsule sized on the tile edge
        # alone is a shape the golden refuses (sub-byte MX) or answers with zeros. Same channel, same
        # reason, as `memory_mapping.regime_extents`.
        "cells": [{"cell": c.key(), "family": c.family, "dtype": c.dtype,
                   "alignment": c.alignment,
                   "shape_quantum": _cs().shape_quantum(
                       c.dtype, tile_dim=bnd.tile_edge,
                       scale_block=(diag.get("scale_block_elements"))),
                   **o.to_dict()}
                  for c, o in sorted(cells.items(), key=lambda kv: kv[0].key())],
    }


def uncovered(spec_doc: dict, corpus_roots, *, labels=None, tile_dim: int | None = None,
              exclude=None) -> dict:
    """Which required cells the corpus does NOT cover — the gate's question.

    Coverage is measured with :func:`contract.materialize.cert_capsule_cover`, the same function the cert
    tier uses to pick a representative subset, so "covered" means exactly what it means everywhere else.
    """
    from merlin.targetgen.contract.materialize import cert_capsule_cover

    got = cert_capsule_cover(corpus_roots, labels=labels, tile_dim=tile_dim, exclude=exclude)
    have = set(got.get("cells") or ())
    want = [c["cell"] for c in (spec_doc.get("cells") or ())]
    missing = sorted(set(want) - have)
    out = {
        "n_required": len(want),
        "n_covered": len(set(want) & have),
        "uncovered": missing,
        "corpus_cells": sorted(have),
        "extra_cells": sorted(have - set(want)),
        "note": ("a required cell with no capsule means the corpus cannot evidence a family/dtype/"
                 "alignment the hardware admits and a real target-model uses"),
    }
    # THE COMPOSITION AXIS, measured on the same corpus and reported beside the cells rather than folded
    # into them. A spec written before this axis existed carries no `composition` block; that is reported
    # as "not measured", never as "nothing required" -- an axis a stale spec cannot express must not read
    # as an axis with no gaps.
    comp_req = (spec_doc.get("composition") or {}).get("required")
    if comp_req is None:
        out["composition"] = {"status": "not_measured",
                              "detail": "this spec predates the composition axis; regenerate it with "
                                        "--write to derive the requirement"}
    else:
        from merlin.targetgen import boundary as BD
        corpus = BD.corpus_boundaries(corpus_roots, str(spec_doc.get("target") or ""),
                                      labels=labels, exclude=exclude)
        gap = BD.uncovered_boundaries({"by_kind": comp_req}, corpus)
        gap["status"] = "ok"
        gap["covered_by"] = corpus["by_kind"]
        out["composition"] = gap

    # THE GEOMETRY AXIS, measured by classifying each capsule's own contraction extents with the SAME
    # taxonomy the requirement is written in. A stale spec carrying no block reads "not measured",
    # never "no geometry is required".
    geom_req = (spec_doc.get("shape_geometry") or {}).get("required")
    if geom_req is None:
        out["shape_geometry"] = {"status": "not_measured",
                                 "detail": "this spec predates the geometry axis; regenerate it with "
                                           "--write to derive the requirement"}
    else:
        out["shape_geometry"] = _geometry_gap(geom_req, corpus_roots, labels=labels, exclude=exclude)

    # THE NEGATIVE LANE, measured the same way and reported beside the others. A family the hardware
    # does not admit must be shown landing on the HOST -- and shown by a capsule built to prove it, not
    # by one that merely happens to contain a host stretch. `corpus_boundaries` credits `H` to any
    # capsule containing one, so without the family clause below `H` is trivially covered by every
    # routing-shaped capsule and means nothing as a requirement.
    host_only = spec_doc.get("host_only")
    if host_only is None:
        out["host_only"] = {"status": "not_measured",
                            "detail": "this spec predates the negative-lane axis; regenerate it with "
                                      "--write to derive the requirement"}
    elif not (host_only.get("families") or ()):
        out["host_only"] = {"status": "undeterminable", "families": [],
                            "detail": "every family this target's captures contain is admitted by its "
                                      "manifest, so no negative lane is derivable here. NOT the same as "
                                      "a negative lane that passed"}
    else:
        import yaml as _yaml
        from pathlib import Path as _P

        from merlin.targetgen import boundary as BD
        want_fams = set(host_only["families"])
        covered_by: dict[str, list[str]] = {}
        roots = [corpus_roots] if isinstance(corpus_roots, (str, _P)) else list(corpus_roots)
        labelset = set(labels or {"public"})
        skip = set(exclude or ())
        for root in roots:
            for cy in sorted(_P(root).glob("*/capsule.yaml")):
                try:
                    cap = _yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
                except _yaml.YAMLError:
                    continue
                name = str(cap.get("name") or cy.parent.name)
                if name in skip or str(cap.get("label")) not in labelset:
                    continue
                fam = str((cap.get("semantic") or {}).get("semantic_family") or "")
                if fam not in want_fams:
                    continue
                prof = BD.profile_capsule(cy.parent, str(spec_doc.get("target") or ""))
                if prof.kind == BD.HOST_ONLY:
                    covered_by.setdefault(fam, []).append(name)
        out["host_only"] = {
            "status": "ok",
            "families": sorted(want_fams),
            "n_required": len(want_fams),
            "n_covered": len(covered_by),
            "uncovered": sorted(want_fams - set(covered_by)),
            "covered_by": covered_by,
            "note": ("a family the hardware does not admit, shown landing on the host lane by a capsule "
                     "whose OWN family is that one -- not merely by a capsule that contains a host "
                     "stretch, which every routing-shaped capsule does"),
        }

    mem_req = (spec_doc.get("memory_mapping") or {}).get("required")
    if mem_req is None:
        out["memory_mapping"] = {"status": "not_measured",
                                 "detail": "this spec predates the memory-mapping axis; regenerate it "
                                           "with --write to derive the requirement"}
    else:
        from merlin.targetgen import memory_regime as MR
        mem_corpus = MR.corpus_regimes(corpus_roots, str(spec_doc.get("target") or ""),
                                       labels=labels, exclude=exclude)
        mgap = MR.uncovered_regimes({"by_regime": mem_req}, mem_corpus)
        mgap["status"] = "ok"
        mgap["covered_by"] = mem_corpus["by_regime"]
        mgap["region_counts"] = (spec_doc.get("memory_mapping") or {}).get("region_counts") or {}
        out["memory_mapping"] = mgap
    return out
