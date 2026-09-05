"""What a capture contains BEYOND one op at a time -- the unit a phase-2 obligation is made of.

The existing census describes a capture one region at a time: family, dtype, extents. That is the right
unit for a correctness obligation, because a compiler gets one op wrong at a time. It is the wrong unit
for a performance obligation, because the levers that matter above the tile -- inter-layer scheduling,
epilogue fusion, the host/accelerator seam, whole-layer residency -- are properties of ADJACENT regions,
and adjacency is exactly what a per-region descriptor throws away.

Two things are lost, and each has a measured consequence.

**Op configuration.** ``RegionDescriptor`` carries ``(family, dtype, extents)`` and nothing else, so a
convolution's padding, stride and dilation never reach the requirement. Every convolution capsule in
this corpus therefore declared the SAME geometry -- zero padding, unit stride, unit dilation, an
identical window -- four times over, while the builder had accepted all three parameters all along and
nothing ever passed a non-default. The defect that hid there is a lowering that reuses whatever the
staging buffer last held instead of reading an out-of-bounds tap as zero: wrong only in the border
rows, and unreachable by a corpus with no border. A requirement cannot ask what its vocabulary cannot
express, so the vocabulary is widened here rather than the capsules being hand-written.

**Adjacency.** The optimisation ladder declares rungs for inter-layer scheduling, boundary crossing,
fusion and global scope. The corpus has ZERO members between them, and it is not an oversight: no
observation in this repo records that two regions were adjacent, so a requirement for a chain cannot be
derived at all. ``chains`` recovers it from the use-def graph the capture already carries.

WHAT IS DERIVED AND WHAT IS NOT. Both structures read only what the capture states -- an op's own
attribute mapping, verbatim, and the producer/consumer edges between region ops. Nothing is normalised
against a list of known attribute names, because such a list is a per-target fact in disguise: it would
have to be edited for the next target's spelling, and the edit is what gets forgotten. A configuration
this module cannot interpret is still RECORDED, so a downstream requirement can ask for coverage of it
without this module having to understand it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

__all__ = ["RegionConfig", "Chain", "region_configs", "chains", "config_axes", "chain_census"]


@dataclass(frozen=True)
class RegionConfig:
    """One region's declared configuration, verbatim.

    ``attrs`` is whatever the op carries. It is deliberately not filtered against a known-key list:
    an unknown key is evidence about the capture, and dropping it is how an axis becomes invisible.
    """

    index: int
    op: str
    family: str | None
    attrs: dict[str, str] = field(default_factory=dict)

    def axis(self, key: str) -> str | None:
        return self.attrs.get(key)


@dataclass(frozen=True)
class Chain:
    """A maximal run of regions where each consumes the previous one's result.

    ``ops`` is the sequence of short op names in program order. ``signature`` is what a requirement
    would be written against -- the shape of the chain rather than its extents -- so two attention
    blocks at different sequence lengths present the same obligation.
    """

    ops: tuple[str, ...]
    indices: tuple[int, ...]
    families: tuple[str | None, ...] = ()

    @property
    def signature(self) -> str:
        """The chain's shape in the vocabulary a requirement is written in.

        An unnamed ``linalg.generic`` carries no information in its op name, and a real capture is
        mostly unnamed generics -- a signature built from op names alone reads "generic -> generic ->
        generic" for every chain in the model, which is a count of nothing. The semantic family is the
        cross-target vocabulary the conformance cells already use, so it is what the signature says
        wherever it resolves.
        """
        parts = []
        for i, op in enumerate(self.ops):
            fam = self.families[i] if i < len(self.families) else None
            parts.append(fam or op)
        return " -> ".join(parts)

    @property
    def length(self) -> int:
        return len(self.ops)


def _region_ops(module) -> list[Any]:
    """The computation-carrying regions, in program order.

    Delegates the predicate to the existing census rather than re-deriving it: two copies of "is this a
    region" drift, and a drifted copy silently changes which ops a chain can span.
    """
    from merlin.targetgen.model_coverage import region_ops

    return list(region_ops(module))


def region_configs(module) -> tuple[RegionConfig, ...]:
    """Every region's declared configuration, in the same order the region census uses."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.model_coverage import _short_op  # the one spelling of "short name"

    out: list[RegionConfig] = []
    for i, op in enumerate(_region_ops(module)):
        short = _short_op(mq.op_name(op))
        attrs: dict[str, str] = {}
        for table in mq._attr_tables(op):
            try:
                items = table.items()
            except AttributeError:
                continue
            for k, v in items:
                key = str(k)
                if key.startswith("prov."):
                    continue  # provenance is recorded elsewhere and is a hint, not configuration
                attrs[key] = str(v)
        out.append(RegionConfig(index=i, op=short, family=sf.from_op(short), attrs=attrs))
    return tuple(out)


def config_axes(configs: Sequence[RegionConfig]) -> dict[str, dict[str, int]]:
    """``op -> attribute -> number of DISTINCT values observed``.

    This is the number that decides whether an axis is worth a capsule. An attribute a capture only
    ever presents at one value is not an axis the corpus is failing to cover; an attribute presenting
    several is one the corpus must span, and today does not.
    """
    seen: dict[str, dict[str, set[str]]] = {}
    for c in configs:
        per_op = seen.setdefault(c.op, {})
        for k, v in c.attrs.items():
            per_op.setdefault(k, set()).add(v)
    return {op: {k: len(vs) for k, vs in sorted(axes.items())} for op, axes in sorted(seen.items())}


def chains(module, *, max_length: int = 8) -> tuple[Chain, ...]:
    """Maximal producer -> consumer runs over the capture's region ops.

    A chain is grown greedily from each region that no other region feeds, following the single
    consumer while there is exactly one. Branching ends a chain: where a value feeds two regions the
    run is no longer a line, and calling it one would invent an ordering the program does not state.

    ``max_length`` bounds the walk so a deep residual stack cannot produce one chain per model. It is a
    reporting bound, not a claim about the program.
    """
    ops = _region_ops(module)
    if not ops:
        return ()
    index_of = {id(op): i for i, op in enumerate(ops)}

    # consumers[i] = the region indices that read a result of region i
    consumers: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    producers: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    for i, op in enumerate(ops):
        for operand in _operands(op):
            owner = _owner(operand)
            j = index_of.get(id(owner)) if owner is not None else None
            if j is not None and j != i:
                consumers[j].append(i)
                producers[i].append(j)

    out: list[Chain] = []
    truncated = False
    for i in range(len(ops)):
        # A region heads a run unless it is the SINGLE continuation of exactly one producer. Starting
        # only at regions with no producer at all missed every run that begins after a branch -- in a
        # residual block that is most of them, so the census saw the stem and none of the arms.
        prods = producers[i]
        if len(prods) == 1 and len(consumers.get(prods[0]) or []) == 1:
            continue
        run = [i]
        cur = i
        while len(run) < max_length:
            nxt = consumers.get(cur) or []
            if len(nxt) != 1:
                break
            cur = nxt[0]
            if cur in run:  # defensive: a cycle is not a chain
                break
            run.append(cur)
        if len(run) == max_length and len(consumers.get(run[-1]) or []) == 1:
            truncated = True  # the bound bound: the real chain continues past what is reported
        if len(run) > 1:
            from merlin.common import mlir_query as mq
            from merlin.targetgen import semantic_families as sf
            from merlin.targetgen.model_coverage import _short_op

            names = tuple(_short_op(mq.op_name(ops[k])) for k in run)
            out.append(Chain(ops=names, indices=tuple(run),
                             families=tuple(_family_of(ops[k], names[j]) for j, k in enumerate(run))))
    _LAST_WALK["truncated"] = truncated
    return tuple(out)


#: Whether the most recent :func:`chains` walk hit its length bound. A census that silently reports a
#: truncated longest-chain as the model's longest chain states a property of the BOUND as a property of
#: the program, so the flag travels with the census rather than being inferred from ``longest``.
_LAST_WALK: dict[str, bool] = {"truncated": False}


def _family_of(op, short: str) -> "str | None":
    """The op's semantic family, resolved the SAME way the region census resolves it.

    Structural first -- an op's own name settles it whatever a tag claims -- then the capture's
    provenance tags for the unnamed ``linalg.generic`` case, which is most of a real capture. Reading
    only the name reported "generic" for 90% of a model, which is a count of nothing.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen import semantic_families as sf

    fam = sf.from_op(short)
    if fam is not None:
        return fam
    return sf.from_prov(mq.attr_str(op, "prov.family"), mq.attr_str(op, "prov.op"))


def _operands(op) -> Iterator[Any]:
    try:
        yield from op.operands
    except (AttributeError, TypeError):
        return


def _owner(value) -> Any | None:
    owner = getattr(value, "owner", None)
    if owner is None:
        return None
    # An xDSL result's owner is the op; a block argument's owner is a block, which is not a region op.
    return owner if hasattr(owner, "operands") else None


def chain_census(module, *, max_length: int = 8) -> dict[str, Any]:
    """What a phase-2 requirement would be written against: which chain shapes a capture presents.

    Reports the signature counts rather than the chains themselves, because the obligation is the
    SHAPE of the chain -- two attention blocks at different sequence lengths present one obligation,
    not two -- while the extents belong to the member that witnesses it.
    """
    cs = chains(module, max_length=max_length)
    by_sig: dict[str, int] = {}
    for c in cs:
        by_sig[c.signature] = by_sig.get(c.signature, 0) + 1
    lengths: dict[int, int] = {}
    for c in cs:
        lengths[c.length] = lengths.get(c.length, 0) + 1
    return {
        "n_regions": len(_region_ops(module)),
        "n_chains": len(cs),
        "longest": max((c.length for c in cs), default=0),
        "length_bound": max_length,
        "longest_is_truncated": _LAST_WALK["truncated"],
        "by_length": dict(sorted(lengths.items())),
        "by_signature": dict(sorted(by_sig.items(), key=lambda kv: -kv[1])),
    }
