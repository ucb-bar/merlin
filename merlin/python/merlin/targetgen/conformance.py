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

    ``alignment`` is ``"aligned"`` / ``"partial"`` (extents that are, or are not, whole multiples of the
    target's tile edge) or ``None`` when the target declares no tiling edge at all. Both alignments are
    required wherever an edge exists: a unit that only ever sees whole tiles has never exercised its tail
    path, and the tail is where tiling bugs live.
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

        One point below, one on, one above each edge. ``tile_edge`` catches the tiling tail;
        ``block_scale_group`` catches a K that does not divide the block-scale group, which is not a
        hypothetical: a captured LLaMA's own hidden size (344) is not a multiple of a 32-element group.
        """
        out: list[dict] = []
        for name, edge, src in (("tile_edge", self.tile_edge, self.tile_edge_source),
                                ("block_scale_group", self.block_scale_group, self.block_scale_source)):
            if not edge or edge < 2:
                continue
            out.append({"boundary": name, "edge": int(edge), "source": src,
                        "points": [int(edge) - 1, int(edge), int(edge) + 1, int(edge) * 2]})
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
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(target)
    except Exception:                                      # noqa: BLE001 — unresolvable contract
        return {}
    return {fam: tuple(cap.dtypes or ()) for fam, cap in sorted(cap_map.items())}


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
    adm = admitted(target)
    units = admitting_units(target)
    bnd = boundaries(target)
    aligns: tuple[str | None, ...] = ("aligned", "partial") if bnd.tile_edge else (None,)

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
            for al in aligns:
                cell = Cell(fam, capsule_dtype(dt), al)
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
        "notes": [],
    }
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


def spec(target: str, captures: dict[str, str | Path], *,
         declared: dict[str, dict] | None = None,
         personas: dict[str, dict] | None = None) -> dict:
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
        "cells": [{"cell": c.key(), "family": c.family, "dtype": c.dtype,
                   "alignment": c.alignment, **o.to_dict()}
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
