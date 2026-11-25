"""Coverage cells, origins and target-derived boundary data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from merlin.development.targetgen.applications.demand_inventory import application_demand_inventory

# ---- conformance_types -------------------------------------------------------------------------

OBSERVED = "observed"
DECLARED = "declared"
OBSERVED_VIA_PRIMITIVES = "observed_via_primitives"


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

    basis: str = OBSERVED  # OBSERVED | OBSERVED_VIA_PRIMITIVES | DECLARED
    observed_in: tuple[str, ...] = ()  # captures whose census contains this family
    admitted_by: tuple[str, ...] = ()  # compute units declaring the family+dtype
    citation: str = ""  # required when basis == DECLARED
    via_primitives: tuple[str, ...] = ()  # set when basis == OBSERVED_VIA_PRIMITIVES
    n_regions: int = 0  # how many regions across all captures carried it

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
    block_scale_group: int | None = None  # MX E8M0 K-group, when the target has one
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
        for name, edge, src in (
            ("tile_edge", self.tile_edge, self.tile_edge_source),
            ("block_scale_group", self.block_scale_group, self.block_scale_source),
        ):
            if not edge or edge < 2:
                continue
            e = int(edge)
            points = {1, e // 4, e // 2, e - 1, e, e + 1, e * 2}
            out.append({"boundary": name, "edge": e, "source": src, "points": sorted(x for x in points if x >= 1)})
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


# ---- conformance -------------------------------------------------------------------------------
# What a target's whole-model corpus MUST cover, derived — never typed.
#
# A corpus of hand-picked whole-model capsules describes whichever architectures somebody happened to
# author. Radiance's did exactly that: the only model capsule ever graded is a LLaMA decoder, so its
# captured linalg carries no ``tanh``, no ``erf`` and no convolution, and nothing in the scorecard said the
# evidence was one architecture wide. Widening it by hand-writing "the corpus shall contain these eleven
# op families" trades one overfit for another: the list is stale the moment a target or a target-model is
# added, and it cannot report what it is missing.
#
# So the REQUIREMENT is computed from three independent sources, and the corpus is measured against it:
#
# ``admitted``
#     what the silicon declares it can compute — the target's capability manifest
#     (:func:`merlin.target.capabilities.eligibility.capability_map_for_target`), family x dtype x rank. A target
#     with no fp8 datapath does not get fp8 cells.
#
# ``observed``
#     what the real target-models actually contain — the family census of each captured model
#     (:func:`merlin.development.targetgen.conformance.model_coverage.regions_from_module`). A family no real model uses is not
#     required of the corpus, however capable the hardware is.
#
# ``boundaries``
#     where the hardware's own edges are — the mesh/array edge and the MX block-scale group, read from the
#     target's facts. Edge cases are generated at those boundaries rather than imagined.
#
# The required set is ``admitted INTERSECT observed``, expressed in the SAME cell vocabulary
# :func:`merlin.development.targetgen.corpus.materialization.cert_capsule_cover` already uses — ``(semantic_family,
# dtype, tile_alignment)`` — so the existing cover machinery measures conformance for free and the two can
# never disagree about what a cell is.
#
# Every cell carries its provenance: which capture observed the family, which compute unit admitted the
# dtype, or ``DECLARED`` plus a citation where a target-model could not be captured at all. A declared cell
# is never silently indistinguishable from an observed one — that distinction is the whole point of
# deriving, and DeepSeek-R1-Distill-Qwen has no capture, so its column can only be declared until a
# ``model2MLIR`` loader for it exists.


def _capsule_paths(root: str | Path) -> list[Path]:
    """Accept an exact capsule directory without broadening it to its siblings."""
    directory = Path(root)
    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents):
        raise ValueError("capsule coverage selection cannot traverse symlinks")
    direct = directory / "capsule.yaml"
    paths = [direct] if direct.is_file() else sorted(directory.glob("*/capsule.yaml"))
    if any(path.is_symlink() or path.parent.is_symlink() for path in paths):
        raise ValueError("capsule coverage members cannot be symlinked")
    return paths


def _sf():
    """:mod:`merlin.target.capabilities.semantic_families`, imported lazily to keep this module import-light."""
    from merlin.target.capabilities import semantic_families

    return semantic_families


def _cs():
    """:mod:`merlin.development.targetgen.corpus.binding`, imported lazily for the same reason as :func:`_sf`. It owns
    the ONE definition of a dtype's numeric regime and of the shape granularity that regime imposes, so
    the requirement and the synthesizer cannot disagree about which shapes exist."""
    from merlin.development.targetgen.corpus import binding as corpus_spec

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
        from merlin.development.targetgen.corpus.binding import dtype_info

        return dtype_info(str(token))[0]
    except Exception:  # noqa: BLE001 — unmapped token, keep as-is
        return str(token)


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
        from merlin.target.capabilities.eligibility import capability_map_for_target

        cap_map = capability_map_for_target(target)
    except Exception as exc:  # noqa: BLE001 — unresolvable contract
        return {}, f"unresolvable: {type(exc).__name__}: {str(exc)[-160:]}"
    return ({fam: tuple(cap.dtypes or ()) for fam, cap in sorted(cap_map.items())}, "resolved")


def admitting_units(target: str) -> dict[tuple[str, str], tuple[str, ...]]:
    """``(family, dtype) -> unit names`` — which compute unit admits each pair, for cell provenance."""
    out: dict[tuple[str, str], list[str]] = {}
    try:
        from merlin.target.capabilities import compute_units as cu
        from merlin.target.registry import identity as tr

        units = cu.compute_units(tr.load_contract(target))
    except Exception:  # noqa: BLE001
        return {}
    for unit in units:
        for cap in getattr(unit, "semantic_capabilities", ()) or ():
            for dt in getattr(cap, "dtypes", ()) or ():
                # keyed in the CAPSULE spelling, so it joins the cells built below
                out.setdefault((cap.family, capsule_dtype(dt)), []).append(getattr(unit, "name", "?"))
    return {k: tuple(sorted(set(v))) for k, v in out.items()}


def observed(capture: str | Path, target: str) -> Counter:
    """``family -> n_regions`` actually present in one captured model's linalg.

    Delegates to :mod:`merlin.development.targetgen.conformance.model_coverage`, which resolves a region's family from the op's
    own NAME first and only falls back to provenance tags — the right order, because real captures leave a
    large fraction of regions untagged and the tags disagree with the IR often enough to be a hint rather
    than an authority. Regions whose family cannot be resolved are NOT counted: an unnamed region is
    evidence neither for nor against a requirement.
    """
    from merlin.development.targetgen.conformance import model_coverage as mc

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
    from merlin.development.targetgen.conformance import model_coverage as mc

    out: Counter = Counter()
    for region in mc.regions_from_module(mc.load_module(capture)):
        family, dtype = getattr(region, "family", None), getattr(region, "in_dtype", None)
        if not family or not dtype:
            continue
        try:
            out[(str(family), capsule_dtype(str(dtype)))] += 1
        except Exception:  # noqa: BLE001 -- an unmappable token stays visible
            out[(str(family), str(dtype))] += 1
    return out


def corpus_presented_pairs(corpus_roots, *, labels=None) -> Counter:
    """``(family, dtype) -> n`` for the work the CORPUS ITSELF presents, from its capsules' declarations.

    A second evidence source for the negative lane, and on one target the only one that sees a real
    gap. The captures are one model zoo; the corpus is what the grader will actually run, and it
    already contains capsules the manifest does not admit -- a bf16 contraction on an int8-only array,
    for instance. Those must land on the host, nothing declares that they must, and so a compiler that
    quietly accelerated one and a compiler that correctly declined would score identically.

    Read off each capsule's own ``semantic.semantic_family`` and declared input dtypes, which is the
    same resolution `cert_capsule_cover` classifies at -- so a pair this reports is a pair the cover
    can see, rather than one derived at a resolution nothing else uses.
    """
    import yaml

    labels = set(labels or {"public"})
    out: Counter = Counter()
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots or ())
    for root in roots:
        for cy in _capsule_paths(root):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            fam = (cap.get("semantic") or {}).get("semantic_family")
            if not fam:
                continue
            # ⚠️ A WHOLE MODEL'S `inputs[]` IS ITS ENTRY TENSOR, NOT AN OPERAND. A language model is
            # entered with a token-id vector -- i64 -- and computes in none of that: it declares what it
            # computes in separately, at `operation.attributes.dtype`. Reading the entry tensor as an
            # operand dtype minted a `contraction/i64` host-lane obligation on all three targets, for
            # arithmetic no capsule performs and no hardware was ever asked to take. Measured: 24
            # capsules declare a compute dtype, 20 of them disagree with their own `inputs[]`, and ALL
            # 20 are `kind: model`.
            #
            # So a DECLARED compute dtype wins outright. It is the capsule's own statement about its
            # arithmetic, and where it exists there is nothing to infer.
            declared = str(((cap.get("operation") or {}).get("attributes") or {}).get("dtype") or "")
            if declared:
                out[(str(fam), declared)] += 1
                continue
            for t in cap.get("inputs") or []:
                if t.get("dtype") and t.get("role") in ("input", "weight"):
                    out[(str(fam), str(t["dtype"]))] += 1
    return out


def host_lane_cells(captures: dict, target: str, corpus_roots=None) -> dict:
    """The ``(family, dtype)`` work real models contain that ``target`` may NOT accelerate.

    The negative lane, at full width. The existing ``host_only`` block carries only families with no
    admitted dtype AT ALL, which is the narrow case: a target admitting ``contraction`` at int8 still
    cannot take an f32 contraction, and a real model is full of them. Every pair here is work the
    compiler must place on the host, and a corpus with no capsule for it cannot tell a compiler that
    routes it correctly from one that does not. Expressed tensor types outside the format registry
    are separate host/support obligations: no typed capsule writer can represent them faithfully,
    but they are never dropped or treated as accelerator-admissible.

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
            except Exception:  # noqa: BLE001
                admitted_pairs.add((str(family), str(d)))

    seen: Counter = Counter()
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            seen.update(observed_pairs(path, target))
        except Exception as e:  # noqa: BLE001 -- reported, never skipped silently
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"

    # THE CORPUS IS EVIDENCE TOO, and on an int8-only target it is the only source that sees the gap:
    # its captures are int8 models, so no bf16 contraction is ever OBSERVED, while the corpus ships
    # bf16 contraction capsules that the array cannot take. Those sit on the host lane by necessity and
    # nothing asserted that they must, so accelerating one -- a real type-contract miss -- cost nothing.
    from_corpus = corpus_presented_pairs(corpus_roots) if corpus_roots else Counter()
    from_captures = set(seen)  # recorded BEFORE the merge, so the two sources stay
    for pair, n in from_corpus.items():  # distinguishable in what each pair says it rests on
        seen[pair] = max(int(seen.get(pair, 0)), int(n))

    # A tensor with an MLIR element type outside the quant-format registry is still HOST work, but
    # this capsule-pair axis cannot faithfully materialize it: the corpus schema has no format or
    # golden policy for `unsupported_mlir:i64`/`i1`, for example. Keep the measured work as explicit
    # support-lane debt rather than turning it into an impossible typed capsule or silently treating
    # the previously missing dtype as accelerator-admissible.
    unsupported_formats = [
        {
            "family": f,
            "dtype": d,
            "n_regions": n,
            "reason": (
                "the captured tensor element type is outside the quant-format registry; this is "
                "host/support work, but no typed capsule writer can materialize this pair"
            ),
        }
        for (f, d), n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        if str(d).startswith("unsupported_mlir:")
    ]
    required = [
        {
            "family": f,
            "dtype": d,
            "n_regions": n,
            "evidenced_by": (
                "captures_and_corpus"
                if (f, d) in from_corpus and (f, d) in from_captures
                else "corpus"
                if (f, d) in from_corpus
                else "captures"
            ),
        }
        for (f, d), n in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        if (f, d) not in admitted_pairs and not str(d).startswith("unsupported_mlir:")
    ]
    # Which operations each pair's host work consists of, so a probe is built from one of them.
    from merlin.development.targetgen.applications import operation_plan as host_lane_ops

    observed_ops = host_lane_ops.observed_host_ops(captures or {}, [(row["family"], row["dtype"]) for row in required])
    for row in required:
        row["observed_ops"] = observed_ops.get(host_lane_ops.pair_key(row["family"], row["dtype"]), [])
    return {
        "required": required,
        "unsupported_format_host_obligations": unsupported_formats,
        "admitted_pairs": sorted(f"{f}/{d}" for f, d in admitted_pairs),
        "captures_unreadable": unreadable,
        "axis_basis": (
            "the (family, dtype) work real captures CONTAIN that this target's manifest does not admit. "
            "The cells intersect admitted with observed and keep only what survives; this keeps what does "
            "NOT, which is precisely the work the compiler has to place on the host. Both sides come from "
            "the same two sources the cells do, read at the resolution the cells discard -- a region's "
            "dtype decides whether the hardware may take it, so a family histogram cannot express it. "
            "Unsupported MLIR element types remain visible as unsupported_format_host_obligations, "
            "not as impossible typed-capsule requirements"
        ),
    }


def boundaries(target: str) -> Boundaries:
    """The target's edges, each tagged with the artifact it was read from."""
    b = Boundaries()
    # Tile edge, and whether it is a hardware fact or the software-tiling default. `_tile_dim` resolves
    # a declared matrix unit / capabilities.mesh / capabilities.tile / an RTL `arrays` fact, and only then
    # falls back to a constant -- so "did it come from hardware" is decidable by asking for the fact
    # sources directly rather than by trusting the number.
    try:
        from merlin.development.targetgen.corpus.binding import (
            _DEFAULT_SW_TILE,
            _structural_mesh_rows,
            _tile_dim,
        )
        from merlin.target.registry.identity import load_contract

        contract = load_contract(target)
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
        structural_rows = None
        if not hw:
            try:
                from merlin.target.rtl.facts import load_facts

                selected_facts = load_facts(target) or {}
                arrays = (selected_facts.get("facts") or {}).get("arrays") or []
                mesh = next((a for a in arrays if a.get("rows") and a.get("cols")), None)
                rtl_rows = int(mesh["rows"]) if mesh else None
                if rtl_rows is None:
                    structural_rows = _structural_mesh_rows(selected_facts)
            except Exception:  # noqa: BLE001 — absent facts: not a hardware fact
                rtl_rows = None
        from_rtl = rtl_rows is not None and edge is not None and int(rtl_rows) == int(edge)
        from_structure = structural_rows is not None and edge is not None and structural_rows == edge
        b.tile_edge_is_hardware_fact = (
            hw or from_rtl or from_structure or (edge is not None and edge != _DEFAULT_SW_TILE)
        )
        b.tile_edge_source = (
            "capability manifest (declared mesh/tile rows)"
            if hw
            else "RTL facts arrays[].rows (the target leaves geometry to discovery rather "
            "than restating it in the contract)"
            if from_rtl
            else "RTL facts structural_observations[].rows (source-bound grid geometry only; "
            "compute capability uncorroborated)"
            if from_structure
            else f"software-tiling default ({_DEFAULT_SW_TILE}); this target declares no "
            f"fixed hardware mesh, so it is NOT a hardware boundary"
        )
    except Exception as e:  # noqa: BLE001
        b.tile_edge_source = f"unavailable: {type(e).__name__}"

    # MX block-scale K-group. Lives in the target's MX MMIO contract, not in rtl/facts -- read it from the
    # one accessor so a second hardcoded 32 does not enter the tree.
    try:
        from merlin.target.rtl.extraction.mlc_toolchain import mx_mmio_for

        mx = mx_mmio_for(target) or {}
        grp = mx.get("group")
        if grp:
            b.block_scale_group = int(grp)
            b.block_scale_source = "target contract mx_mmio.group (one E8M0 scale per K group)"
    except Exception as e:  # noqa: BLE001
        b.block_scale_source = f"unavailable: {type(e).__name__}"

    # On-chip operand store, for capacity-fit extents. Select its role from the
    # RTL-derived address space; memory names are extractor labels, not an ABI.
    try:
        from merlin.target.capabilities import address_space as AS

        resolved = AS.operand_store(AS.derive_address_space(target))
        if resolved.store is not None and resolved.store.nbytes:
            b.operand_store_bytes = int(resolved.store.nbytes)
            b.operand_store_source = f"RTL-derived operand store {resolved.store.name!r} ({resolved.basis})"
        else:
            b.operand_store_source = f"RTL operand store unresolved: {resolved.reason}"
    except Exception as e:  # noqa: BLE001
        b.operand_store_source = f"rtl facts unavailable: {type(e).__name__}"
    if b.operand_store_bytes is None:
        try:
            from merlin.target.registry.experiment_descriptor import load_capability_manifest

            mm = load_capability_manifest(target).contract.get("memory_model") or {}
            if mm.get("shared_memory_bytes"):
                b.operand_store_bytes = int(mm["shared_memory_bytes"])
                b.operand_store_source = "capability manifest memory_model.shared_memory_bytes"
        except Exception as e:  # noqa: BLE001
            b.operand_store_source += f"; manifest unavailable: {type(e).__name__}"
    return b


def required_cells(
    target: str, captures: dict[str, str | Path], *, declared: dict[str, dict] | None = None
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
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        scale_block = _cs()._scale_block_elems(load_capability_manifest(target).contract)
    except Exception:  # noqa: BLE001 — no manifest: no granularity known
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
        except Exception as e:  # noqa: BLE001 — an unreadable capture is
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"  # reported, never skipped silently
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
            continue  # a primitive that is simply absent
        if all(p in seen for p in prims):
            composite_via[fam] = tuple(prims)
            seen[fam] = min(int(seen[p]) for p in prims)  # bounded by its scarcest primitive
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
            continue  # not admitted at all: recorded in diagnostics
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
                    n_regions=int(seen.get(fam, 0)),
                )

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
            f"before reading any coverage number for it"
        )
    if composite_via:
        diagnostics["notes"].append(
            f"{len(composite_via)} composite family/families ({', '.join(sorted(composite_via))}) appear "
            f"in NO capture as a region of their own because the importer decomposes them; they are "
            f"required on the evidence that every primitive they decompose into is observed. Taking the "
            f"region census literally would drop attention from a transformer corpus"
        )
    if needed_not_admitted:
        diagnostics["notes"].append(
            f"{len(needed_not_admitted)} family/families appear in a real capture but the hardware "
            f"declares no capability for them ({', '.join(needed_not_admitted)}); those are compiler "
            f"work for the scalar/vector lane, not accelerator cells, and are excluded from the "
            f"requirement rather than silently counted as covered"
        )
    if unreadable:
        diagnostics["notes"].append(
            f"{len(unreadable)} capture(s) could not be read; the requirement below is NARROWER than the "
            f"evidence would support and must not be read as complete"
        )
    if not bnd.tile_edge_is_hardware_fact and bnd.tile_edge:
        diagnostics["notes"].append(
            f"the alignment axis uses tile edge {bnd.tile_edge}, which is a SOFTWARE tiling default for "
            f"this target, not a hardware boundary"
        )
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
        "tile_alignment": (
            "both, wherever the target tiles: a unit that only ever sees whole tiles has never exercised its tail path"
        ),
    }
    return cells, diagnostics


def _host_only_observed_ops(captures: dict, dtypes: dict) -> dict:
    """``family -> [{op, frontend_op, n_regions}]`` at each host-only family's observed dtype."""
    from merlin.development.targetgen.applications import operation_plan as host_lane_ops

    observed = host_lane_ops.observed_host_ops(captures or {}, list(dtypes.items()))
    return {family: observed.get(host_lane_ops.pair_key(family, dtype), []) for family, dtype in dtypes.items()}


def host_only_dtypes(captures: dict, families) -> dict:
    """``family -> capsule dtype`` for families the HOST must carry, read from the captures themselves.

    A host-only family has no admitted dtype by construction -- the hardware declares no capability for
    it -- so the dtype cannot come from the manifest the way a cell's does. It comes from the real
    captures instead: the dtype those regions actually carry, most frequent first and lexical token on
    a tie. A family nobody could size is ABSENT from the result rather than defaulted, because a host
    capsule emitted at a dtype no model uses tests a program nobody runs.
    """
    from collections import Counter

    from merlin.development.targetgen.conformance import model_coverage as mc

    want = {str(f) for f in families}
    if not want:
        return {}
    seen: dict[str, Counter] = {f: Counter() for f in sorted(want)}
    for path in captures.values():
        try:
            regions = mc.regions_from_module(mc.load_module(Path(path)))
        except Exception:  # noqa: BLE001 -- an unreadable capture is not evidence
            continue
        for region in regions:
            fam = region.resolved_family()
            if fam in want and getattr(region, "in_dtype", None):
                seen[fam][str(region.in_dtype)] += 1
    out = {}
    for fam, counts in seen.items():
        if counts:
            try:
                out[fam] = capsule_dtype(min(counts, key=lambda dtype: (-counts[dtype], dtype)))
            except Exception:  # noqa: BLE001 -- an unmappable spelling is not a dtype
                continue
    return out


_DEFAULT_CERT_BUDGET_S = 300.0


def _application_axis(
    target: str,
    *,
    captures: dict | None = None,
    budget_s: float | None = None,
    certification_floor: str | None = None,
) -> dict:
    """The shapes this target's declared APPLICATIONS contain.

    The corpus gives every synthesized capsule one of two tile-relative shapes, and real models do
    not look like that -- 757 contraction regions across six captures carry shapes like
    ``(8,2048,32000)``, against a corpus that tests 16x16x16. This axis is where a user's own
    applications reach the requirement.

    It emits at most two capsules per behavioural class, and the split is what makes the whole thing
    runnable: one sized to be affordable cycle-accurately, and -- where the application is larger --
    one at the true shape that EXTENDS it. An explicit direct-certification floor instead preserves
    the full shape and a tile-scale functional anchor without reading ambient cost history or claiming
    either affordable. See :mod:`merlin.development.targetgen.conformance.claim_applications` for why K is never clamped.

    Empty and inert for a target declaring no applications, which is every target today.
    """
    from merlin.development.targetgen.conformance import claim_applications as APP
    from merlin.development.targetgen.conformance.axes import _binding_engine_fit
    from merlin.development.targetgen.oracles import certification_cost as CC

    budget = float(budget_s) if budget_s else _DEFAULT_CERT_BUDGET_S
    basis = {
        "axis_basis": (
            "declared application contraction classes retain a tile-scale functional anchor and "
            "their full observed shape; both require direct certification and their cost is unknown"
            if certification_floor is not None
            else "the contraction shapes this target's DECLARED applications contain, grouped by what the "
            "compiler must do with them and sized to what a certification costs. A capsule at an "
            "application's real shape is worthless if nobody can afford to certify it, so each class "
            "yields a cycle-accurate capsule at an affordable size plus, when the application is "
            "larger, an L2 capsule at the true shape that extends it -- never one without the other"
        ),
        "cert_budget_s": budget if certification_floor is None else None,
        "budget_source": (
            "not_used_direct_floor" if certification_floor is not None else "declared" if budget_s else "default"
        ),
        **(
            {"cost_status": "unknown_not_priced", "declared_budget_s": budget_s}
            if certification_floor is not None
            else {}
        ),
    }
    if not captures:
        return {"required": [], "refused": [], "declared_applications": 0, **basis}

    # ⚠️ AN APPLICATION CLASS THE HARDWARE DOES NOT ADMIT IS HOST-LANE WORK, NOT AN ACCELERATOR
    # OBLIGATION. The classes come from what the captures CONTAIN, and a real model contains plenty of
    # arithmetic this target cannot take -- measured on an int8-only array, 32 of 46 sized classes were
    # f32 and 2 were bf16. Emitting those as accelerator capsules demands acceleration the datapath
    # cannot provide, which is a capsule that cannot pass under any correct behaviour. They are already
    # an obligation on the `host_lane` axis, at the same (family, dtype) resolution, so they are moved
    # there rather than dropped: the work is real, it is just the host's.
    #
    # This was invisible for as long as the axis produced nothing. Turning it on is what surfaced it.
    adm = admitted(target) or {}
    admitted_pairs = set()
    for _fam, _dts in adm.items():
        for _d in _dts or ():
            try:
                admitted_pairs.add((str(_fam), capsule_dtype(str(_d))))
            except Exception:  # noqa: BLE001
                admitted_pairs.add((str(_fam), str(_d)))

    # A block-scaled format admitted by the target is not automatically present in a model whose
    # checkpoint merely stores narrow weights. Ask the application reader to retain an explicit
    # missing-capability record when none of the declared bundles carries the compute + scale
    # semantics. This does not widen the required cells or relabel a float contraction.
    from merlin.common import target_data as _qf

    block_scaled_formats = set()
    for family, dtype in admitted_pairs:
        if family != "contraction":
            continue
        try:
            fmt = _qf.get(dtype)
        except Exception:  # noqa: BLE001 -- unknown cannot prove block scaling
            continue
        if fmt.scale.kind == "block_e8m0":
            block_scaled_formats.add(fmt.name)
    grouped = APP.classify_captures(captures, target, required_block_scaled_formats=block_scaled_formats)
    # PRICE THIS AXIS ON THE ENGINE THAT BINDS, the same way `_cert_affordability` does. `fit_for` with
    # no engine draws ONE line through every certification the target has, whatever answered it, and on
    # a target with two engines that line describes neither: measured here, the mixed fit is r2 0.149
    # over a 3..150528-element range and puts the ceiling at 19,245 written elements, while the same
    # history split by engine gives 6,500 on gsim and 923 on verilator. A spec carrying both numbers --
    # `cert_affordability.max_elements: 923` beside `application_shapes.budget_elements: 19245` -- is
    # not two opinions, it is one axis priced at no machine that exists.
    #
    # The binding engine is the SLOWEST the target has certified on, not the cheapest and not an
    # average, because the shape has to be payable on whichever engine the run picks. Falls back to the
    # unattributed fit only when no engine can be priced, which keeps a target with no per-engine
    # history working exactly as before.
    fit = None if certification_floor is not None else (_binding_engine_fit(target) or CC.fit_for(target))
    try:
        from merlin.development.targetgen.corpus.binding import _tile_dim  # noqa: PLC2701
        from merlin.target.registry.identity import load_contract

        tile = int(_tile_dim(target, load_contract(target)) or 0)
    except Exception:  # noqa: BLE001 -- no edge is a real answer
        tile = 0
    if certification_floor is not None and tile <= 0:
        needs_anchor = any(
            not admitted_pairs or (str(row["family"]), str(row["dtype"])) in admitted_pairs
            for row in grouped.get("classes") or ()
        )
        if needs_anchor:
            raise ValueError(
                f"{target}: direct {certification_floor} application anchors require selected positive tile geometry"
            )

    required, refused = [], []
    for row in grouped.get("classes") or ():
        if admitted_pairs and (str(row["family"]), str(row["dtype"])) not in admitted_pairs:
            # Refusals are strings here, matching what `size_class` returns, so a caller printing the
            # list does not have to know which refusal it is looking at.
            refused.append(
                f"{row.get('class') or (row['family'] + '/' + row['dtype'])}: this target's manifest "
                f"does not admit {row['family']}/{row['dtype']}, so the work in this class belongs on "
                f"the HOST lane and not on the accelerator. It is already required by the host_lane "
                f"axis at the same resolution ({row.get('multiplicity')} region(s), "
                f"{row.get('work')} MACs); an accelerator capsule here would demand an offload the "
                f"datapath cannot perform and could not pass under any correct behaviour"
            )
            continue
        evidence = APP.ClassEvidence(
            region_class=APP.RegionClass(
                family=row["family"],
                dtype=row["dtype"],
                alignment=row["alignment"],
                regime=row["regime"],
                rank=int(row["rank"]),
                geometry=row["geometry"],
            ),
            m=int(row["M"]),
            k=int(row["K"]),
            n=int(row["N"]),
            batch=int(row["batch"]),
            multiplicity=int(row["multiplicity"]),
            work=int(row["work"]),
            work_complete=bool(row["work_complete"]),
            source=str(row["source"]),
        )
        sized, refusal = APP.size_class(
            evidence,
            target=target,
            budget_s=budget,
            tile=tile or None,
            fit=fit,
            certification_floor=certification_floor,
        )
        if refusal:
            refused.append(refusal)
            continue
        required.extend(cap.to_dict() for cap in sized)
    return {
        "required": required,
        "refused": refused,
        "declared_applications": len(captures),
        "n_classes": grouped.get("n_classes", 0),
        "n_regions": grouped.get("n_regions", 0),
        "total_work": grouped.get("total_work", 0),
        "work_coverage": grouped.get("work_coverage"),
        "work_coverage_basis": grouped.get("work_coverage_basis"),
        "capture_audit": grouped.get("capture_audit") or {},
        "region_audit": grouped.get("region_audit") or {},
        "captures_unreadable": grouped.get("captures_unreadable") or {},
        "missing_capabilities": grouped.get("missing_capabilities") or [],
        "cost_model": fit.to_dict() if fit is not None else None,
        **basis,
    }


_SHAPE_AXIS_BASIS = (
    "the rank and operand-layout regions this target's capability manifest DECLARES it handles, via "
    "capability_probes. A (family, dtype, alignment) cell cannot express either: it says what arithmetic "
    "the corpus must contain and nothing about whether the unit was ever asked for a batched region or a "
    "transposed operand. Both are declared capabilities, so a target that claims them and is never asked "
    "for one has an untested claim"
)


def _mkn(shape) -> tuple[int, int, int] | None:
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


_MATERIALIZABLE_ELEMENTS_FALLBACK = 262144


def _materialization_ceiling(target: str) -> int | None:
    """The largest written output a capsule of this target may declare, in elements, or ``None``.

    DERIVED, not declared: it is the capacity of the target's own operand store, which is exactly the
    size the memory-regime axis already sizes its largest capsule against. A capsule bigger than the
    store it runs on tests nothing about residency that `SY_regime_spills` does not, and its golden --
    written as text, one number per element -- stops being materializable long before it stops being
    interesting. ``None`` where the store is not derivable: then nothing is scaled, and an oversized
    class is reported by the caller rather than silently shrunk.
    """
    try:
        from merlin.development.targetgen.policies import memory_regime as MR

        _bnd = boundaries(target)
        dtype = None
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        units = load_capability_manifest(target).contract.get("compute_units") or [{}]
        for u in units:
            for d in u.get("dtypes") or ():
                dtype = capsule_dtype(str(d))
                break
            if dtype:
                break
        store, rows = MR.operand_store(target, dtype=dtype)
        if store is None or not rows:
            return _MATERIALIZABLE_ELEMENTS_FALLBACK
        per_row = store.elems_per_row(dtype)
        return int(rows) * int(per_row) if per_row else _MATERIALIZABLE_ELEMENTS_FALLBACK
    except Exception:  # noqa: BLE001 — no store: the fallback
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
    from merlin.capture import demands as ST
    from merlin.compiler.ir import contraction_shapes as KS

    by_class: dict[str, dict] = {}
    unreadable: dict[str, str] = {}
    total_macs = 0
    for label, path in sorted((captures or {}).items()):
        try:
            pairs = KS.observe_contractions(path)
        except Exception as e:  # noqa: BLE001 — reported, never skipped
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
            row = by_class.setdefault(
                klass,
                {
                    "class": klass,
                    "family": "contraction",
                    "n_regions": 0,
                    "macs": 0,
                    "observed_in": set(),
                    "M": m,
                    "K": k,
                    "N": n,
                    "rep_macs": 0,
                },
            )
            row["n_regions"] += 1
            row["macs"] += macs
            row["observed_in"].add(label)
            if macs > row["rep_macs"]:  # the representative is the heaviest shape
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
                    break  # past the floor: shrinking further is not a
                if _largest(cm, ck, cn) > ceiling:  # smaller representative, it is a different
                    continue  # and degenerate problem
                if ST.classify_geometry(M=cm, N=cn, K=ck) == klass:
                    m, n, k = cm, cn, ck
                    break
            if not m or not n:
                required.append(
                    {
                        "class": klass,
                        "family": row["family"],
                        "M": None,
                        "K": None,
                        "N": None,
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
                            f"capsule of this aspect ratio is both representative and buildable here"
                        ),
                    }
                )
                continue
            scaled_from = {"M": int(row["M"]), "K": int(row["K"]), "N": int(row["N"]), "factor": factor}
        out_elems = m * n
        required.append(
            {
                "class": klass,
                "family": row["family"],
                "M": m,
                "K": k,
                "N": n,
                "out_elements": out_elems,
                "n_regions": int(row["n_regions"]),
                "mac_fraction": round(row["macs"] / float(total_macs), 6) if total_macs else None,
                "observed_in": sorted(row["observed_in"]),
                "scaled_from": scaled_from,
            }
        )
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
            "means the captures contain no readable contraction, never that geometry does not matter"
        ),
    }
