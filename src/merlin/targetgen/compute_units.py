"""Compute-unit capability model — the datatype -> unit -> op mapping a target declares.

A hardware target is a set of **compute units** (a systolic mesh, SIMT lanes, a vector unit, a scalar
core, ...), each of which supports a set of numeric **formats** (named against the target-agnostic
:mod:`merlin.common.quant_formats` registry) on a set of **ops**, with an allowed
``(input, weight) -> accumulator`` matrix and an optional **scaling** format. This is what a target
contract's ``compute_units`` field expresses; :mod:`merlin.targetgen.routing` reads it to bind each op
in a model to a legal ``(unit, dtype)`` — or report an honest gap.

Two deliberate boundaries keep this generic and un-overfit:
- **Formats are referenced by name, never redefined here.** "This unit runs mxfp6" is a reference to
  the registry entry, not a copy of its encoding.
- **Requant/quantize semantics are NOT modelled here.** A unit carries only an opaque ``requant``
  reference (``{ref: <lowering-id>}``) pointing at the target's own out-of-tree lowering. The exact
  arithmetic (a rounding-shift, an E8M0 block-exponent add + LUT, or none at all) is mutually
  incompatible across targets and lives with the target, not in shared code.

Compute units may **compose** (`contains`): a unit can embed others (e.g. an MX systolic unit
inside a SIMT cluster), and its *effective* capability is the union of itself and what it
contains — so the MX unit works standalone or as a sub-unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from merlin.common import quant_formats as qf
from merlin.targetgen import semantic_families as _sf

#: Coarse compute-unit kinds (aligned with runtime.backends.base.TargetClass at the silicon level).
#: ``spatial`` is the non-systolic spatial tensor tile (an OuterProductUnit-style grid of accumulator
#: cells that reduces via rank-1 outer-product accumulate, not a stationary-weight systolic wavefront) —
#: NPU-class silicon like ``systolic`` but a distinct datapath, so it carries its own fact family.
KINDS: frozenset[str] = frozenset({"systolic", "simt", "vector", "scalar", "spatial"})


@dataclass(frozen=True)
class AccumRule:
    """One allowed ``(input, weight) -> accumulator`` triple for a unit.

    ``inp``/``weight`` are quant-format names; ``acc`` is a raw MLIR accumulator type token
    (e.g. ``i32``/``f32``) — accumulators are not storage formats, so they are not registry entries.
    """

    inp: str
    weight: str
    #: ``None`` is "this pairing is legal and accumulates nothing", which is what a unit with no
    #: declared matrix means for its own formats once it is composed with a unit that has one.
    acc: str | None


@dataclass(frozen=True)
class EvidencePath:
    """One way a capability runs standalone: the derived fact that licenses it, and the roles it uses."""

    fact: str
    roles: tuple[str, ...]


def _evidence_paths(raw: Any, unit_name: str, family: str) -> tuple[EvidencePath, ...]:
    from merlin.kernels.roles import ROLES

    if raw in (None, []):
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"compute unit {unit_name!r}: {family} standalone_evidence must be a list")
    out = []
    for row in raw:
        fact = row.get("fact") if isinstance(row, dict) else None
        roles = row.get("roles") if isinstance(row, dict) else None
        if not isinstance(fact, str) or not fact or not isinstance(roles, list) or not roles:
            raise ValueError(f"compute unit {unit_name!r}: {family} evidence needs a fact and a roles list")
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ValueError(f"compute unit {unit_name!r}: {family} evidence names unknown role(s) {unknown}")
        out.append(EvidencePath(fact=fact, roles=tuple(roles)))
    return tuple(out)


@dataclass(frozen=True)
class SemanticCapability:
    """What a unit can compute at the **semantic-family** level — the HARDWARE-truth declaration the
    eligibility oracle (:mod:`merlin.targetgen.eligibility`) reads as the ARR *denominator*.

    Deliberately independent of ``ComputeUnit.ops`` (which is what the generated compiler's *routing*
    binds against, the ARR *numerator*): this says "the silicon can run this family over these formats
    and shapes", not "the compiler currently has a lowering for it". Authored per target in the
    residual, from hardware facts — the gap between this and what routing achieves IS the compiler
    deficiency ARR is meant to surface. ``family`` is a :mod:`merlin.targetgen.semantic_families` name;
    ``dtypes`` reference the quant-format registry.
    """

    family: str
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()  # legal tensor ranks (e.g. (2, 3) for 2D + batched); () = any
    transpose: bool = True  # transposed-operand variants legal where applicable
    arbitrary_mnk: bool = True  # M/N/K need not be tile multiples (tails handled)
    batch: bool = True  # batch dimensions supported
    layouts: tuple[str, ...] = ()  # legal layout tags (coarse); () = unconstrained
    #: Semantic operation forms within a family, distinct from storage layout. For
    #: example, DMA copy does not establish a standalone axis permutation. Empty
    #: retains the legacy unconstrained declaration for older contracts.
    forms: tuple[str, ...] = ()
    #: Which compute-unit KINDS provide this family on this target — the engine attribution.
    #:
    #: The fold below merges every unit's capabilities into one family -> capability map and, until
    #: this field existed, dropped WHICH unit provided each one. So the oracle knew a target could do
    #: ``elementwise_map``; it did not know that this happens on the VPU rather than the MXU. That is
    #: the recorded atlas failure mode (`atlas-0of11-is-agent-not-tooling`): the accumulate never fired
    #: because nothing the agent was handed said which engine owned which family.
    #:
    #: Populated by :func:`semantic_capability_map` from the declaring unit's ``kind`` — it is derived,
    #: not authored, so a contract cannot drift from its own compute_units. Empty means unconstrained
    #: (see ``eligibility._EMPTY_IS_NARROWING``), which is the honest reading for a contract written
    #: before the attribution existed.
    engines: tuple[str, ...] = ()
    #: Families this one is only available FUSED WITH — () means it is available standalone.
    #: A systolic mesh whose only elementwise hardware is the readout epilogue (requant/relu on the
    #: accumulator pop) can run ``elementwise_map`` as a contraction epilogue and CANNOT run a standalone
    #: ``gelu``. Declaring it standalone makes every elementwise region eligible, so the target carries a
    #: permanent false_fallback no compiler change can clear; omitting the family entirely hides real
    #: hardware. This field is how a capability says "yes, but only attached to that".
    composed_with: tuple[str, ...] = ()
    #: How the family runs STANDALONE, as evidence paths: each names the derived hardware fact that
    #: licenses it and the instruction ROLES a program uses to drive it. A policy that prohibits a
    #: role (``prohibited_instruction_roles``) then knows whether the standalone form survives it:
    #: see :mod:`merlin.targetgen.capability_roles`. () means the declaration predates the evidence.
    standalone_evidence: tuple[EvidencePath, ...] = ()
    notes: str = ""
    #: Optional proven result formats for dtype-changing operations. An absent
    #: declaration does not license a conversion merely because its input is legal.
    result_dtypes: tuple[str, ...] | None = None
    #: Correlated input/weight support, projected from each declaring unit before
    #: merging. None retains the older direct-construction API. Appended so the
    #: existing positional shape arguments keep their meaning.
    operand_pairs: tuple[tuple[str, str], ...] | None = None
    #: Which KINDS of reduction this row computes (:data:`REDUCTION_KINDS`): a sum unit is not a max
    #: unit. Empty is unconstrained, the reading for every row written before the field existed.
    kinds: tuple[str, ...] = ()
    #: The individually declared rows a merged capability was folded from, in declaration order.
    #: The fold unions fields (and intersects ``composed_with``), so a family declared once fused
    #: and once standalone for a narrower kind reads as standalone for every kind. Eligibility checks
    #: these rows ANY-OF, so the correlation within each row survives the merge. Empty on an
    #: unmerged row. Not part of equality: two folds of the same rows are the same capability.
    alternatives: tuple[SemanticCapability, ...] = field(default=(), compare=False, repr=False)

    def rows(self) -> tuple[SemanticCapability, ...]:
        """The declared rows this capability stands for: its alternatives, or itself."""
        return self.alternatives or (self,)


#: The closed vocabulary of reduction kinds, the structural classes ``classify_generic`` separates.
REDUCTION_KINDS: frozenset[str] = frozenset({"sum", "max", "absmax", "other"})


@dataclass(frozen=True)
class ComputeUnit:
    name: str
    kind: str
    dtypes: tuple[str, ...] = ()  # quant_format names this unit computes on
    ops: tuple[str, ...] = ()
    accumulate: tuple[AccumRule, ...] = ()
    scaling: str | None = None  # a quant_formats SCALE_KIND (per_channel/block_e8m0/none/...)
    requant: dict[str, Any] | None = None  # opaque {ref: <out-of-tree lowering id>}; not interpreted
    contains: tuple[str, ...] = ()
    semantic_capabilities: tuple[SemanticCapability, ...] = ()
    #: How SOFTWARE drives this unit (a ``families.ENDPOINT_KINDS`` token), independent of ``kind``.
    #: None means "not declared" and defers to the target, then to the family default — see
    #: :func:`resolve_exposure`. This is a PER-UNIT axis on purpose: a hybrid target has two units with
    #: two different exposures at once, and a single target-wide endpoint cannot express that.
    exposure: str | None = None

    def supports_dtype(self, fmt_name: str) -> bool:
        return fmt_name in self.dtypes

    def supports_op(self, op: str, *, family: str | None = None) -> bool:
        """Can this unit compute ``op``? By NAME, or by the FAMILY the op belongs to.

        The name test alone made a spelling decide hardware legality. A target declares its unit's ops
        as ``[matmul]`` while torch-MLIR spells a convolution's contraction
        ``convolution_im2col_matmul``, so every ResNet convolution was refused to the host by a string
        comparison -- 53 of them, on a mesh whose own contract declares
        ``{family: contraction, dtypes: [int8], ranks: [2,3,4]}``.

        ``family`` is the producer's statement, not an inference from the name, and ``None`` never
        widens: an op whose family is unknown is matched by name exactly as before. The declared
        capability still has to admit it -- dtype and rank are checked separately by
        :func:`~merlin.targetgen.routing._legal_on` and this only answers "is this KIND of work mine".
        """
        if not self.ops:
            return True
        if op in self.ops:
            return True
        if family is None:
            return False
        # A COMPOSED capability is not a standalone one. This mesh declares `elementwise_map` and
        # `reduction` with `composed_with: [contraction]` -- it can apply them to a contraction's
        # accumulator on the way out (the COMMIT epilogue), and it cannot run one on its own. Claiming
        # the composed form here would route a free-standing batch norm onto hardware that has no way
        # to execute it.
        return any(cap.family == family and not cap.composed_with for cap in self.semantic_capabilities)


def _accum(raw: Any) -> AccumRule:
    return AccumRule(inp=raw.get("in", ""), weight=raw.get("weight", raw.get("in", "")), acc=raw.get("acc", ""))


def _sem_cap(raw: dict[str, Any], unit_name: str) -> SemanticCapability:
    family = raw.get("family")
    if not _sf.is_family(family):
        raise ValueError(
            f"compute unit {unit_name!r}: semantic_capabilities family {family!r} not in {sorted(_sf.FAMILIES)}"
        )
    if raw.get("engines"):
        # Refused rather than honoured: engine attribution is DERIVED from the declaring unit's kind
        # (see SemanticCapability.engines). Letting a contract author it would permit exactly the drift
        # the derivation exists to prevent -- a unit of kind "vector" declaring that its capability
        # runs on a systolic array, which no reader could then contradict.
        raise ValueError(
            f"compute unit {unit_name!r}: semantic_capability {family!r} declares "
            f"'engines'; attribution is derived from the unit's kind, not authored"
        )
    dtypes = tuple(raw.get("dtypes", ()) or ())
    unknown = [d for d in dtypes if not qf.has(d)]
    if unknown:
        raise ValueError(
            f"compute unit {unit_name!r}: semantic_capability {family!r} unknown quant "
            f"formats {unknown} (known: {qf.names()})"
        )
    raw_pairs = raw.get("operand_pairs")
    if raw_pairs is not None and (
        not isinstance(raw_pairs, list)
        or any(
            not isinstance(pair, dict)
            or set(pair) != {"in", "weight"}
            or not all(isinstance(pair[key], str) and qf.has(pair[key]) for key in ("in", "weight"))
            for pair in raw_pairs
        )
    ):
        raise ValueError(f"compute unit {unit_name!r}: {family} operand_pairs must name known input/weight formats")
    operand_pairs = (
        tuple((qf.get(pair["in"]).name, qf.get(pair["weight"]).name) for pair in raw_pairs)
        if raw_pairs is not None
        else None
    )
    if operand_pairs is not None and any(
        left not in {qf.get(name).name for name in dtypes} or right not in {qf.get(name).name for name in dtypes}
        for left, right in operand_pairs
    ):
        raise ValueError(f"compute unit {unit_name!r}: {family} operand_pairs exceed semantic dtypes")
    result_dtypes = raw.get("result_dtypes")
    if result_dtypes is not None:
        if not isinstance(result_dtypes, list) or not result_dtypes or any(not qf.has(d) for d in result_dtypes):
            raise ValueError(f"compute unit {unit_name!r}: {family} result_dtypes must name known formats")
    kinds = raw.get("kinds")
    if kinds is not None and (
        family != "reduction"
        or not isinstance(kinds, list)
        or not kinds
        or any(kind not in REDUCTION_KINDS for kind in kinds)
    ):
        raise ValueError(
            f"compute unit {unit_name!r}: {family} kinds must be a nonempty reduction kind list "
            f"from {sorted(REDUCTION_KINDS)}"
        )
    return SemanticCapability(
        family=family,
        dtypes=dtypes,
        kinds=tuple(kinds or ()),
        operand_pairs=operand_pairs,
        result_dtypes=tuple(result_dtypes) if result_dtypes is not None else None,
        ranks=tuple(raw.get("ranks", ()) or ()),
        transpose=bool(raw.get("transpose", True)),
        arbitrary_mnk=bool(raw.get("arbitrary_mnk", True)),
        batch=bool(raw.get("batch", True)),
        composed_with=tuple(raw.get("composed_with", ()) or ()),
        layouts=tuple(raw.get("layouts", ()) or ()),
        forms=tuple(raw.get("forms", ()) or ()),
        standalone_evidence=_standalone_evidence(raw, unit_name, family),
        notes=raw.get("notes", "") or "",
    )


def _standalone_evidence(raw: dict[str, Any], unit_name: str, family: str) -> tuple[EvidencePath, ...]:
    paths = _evidence_paths(raw.get("standalone_evidence"), unit_name, family)
    if paths and raw.get("composed_with"):
        raise ValueError(f"compute unit {unit_name!r}: {family} declares standalone evidence and composed_with at once")
    return paths


def _derived_sem_caps(raw: dict[str, Any], unit_name: str) -> tuple[SemanticCapability, ...]:
    """Semantic capabilities implied by a unit's OWN declared ``ops`` + ``dtypes``.

    Used only when a unit declares no ``semantic_capabilities``. A generated contract may describe a unit
    fully in ``ops``/``dtypes`` and omit the semantic block, and the omission is indistinguishable from
    "this unit can accelerate nothing": every region is then ineligible, so eligibility and recall read
    0/N and look like a measurement instead of a missing field. Measured: a systolic MXU declaring
    ``ops: [matmul]`` scored 0% routable on six real models purely because the block was absent.

    Deriving is not widening the claim -- ``ops`` is the unit's own declaration and the family router is the
    canonical op->family map, so a unit gets exactly the families its declared ops imply and no others. An
    op with no known family is SKIPPED rather than guessed, so an unrecognized op still contributes nothing
    (fail closed). A unit that genuinely accelerates nothing declares no ops and still derives nothing.
    """
    ops = tuple(raw.get("ops", ()) or ())
    dtypes = tuple(raw.get("dtypes", ()) or ())
    families: list[str] = []
    for op in ops:
        family = _sf.from_op(op)
        if family is not None and family not in families:
            families.append(family)
    return tuple(SemanticCapability(family=f, dtypes=dtypes, notes=f"derived from {unit_name!r} ops") for f in families)


def _unit(raw: dict[str, Any]) -> ComputeUnit:
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"compute unit {raw.get('name')!r}: kind {kind!r} not in {sorted(KINDS)}")
    dtypes = tuple(raw.get("dtypes", ()) or ())
    unknown = [d for d in dtypes if not qf.has(d)]
    if unknown:
        raise ValueError(f"compute unit {raw.get('name')!r}: unknown quant formats {unknown} (known: {qf.names()})")
    scaling = raw.get("scaling")
    if scaling is not None and scaling not in qf.SCALE_KINDS:
        raise ValueError(f"compute unit {raw.get('name')!r}: scaling {scaling!r} not in {sorted(qf.SCALE_KINDS)}")
    name = raw["name"]
    exposure = raw.get("exposure")
    if exposure is not None:
        from . import families  # lazy: families is a sibling registry, not a dependency

        if exposure not in families.ENDPOINT_KINDS:
            raise ValueError(
                f"compute unit {raw.get('name')!r}: exposure {exposure!r} not in {list(families.ENDPOINT_KINDS)}"
            )
    return ComputeUnit(
        name=name,
        kind=kind,
        dtypes=dtypes,
        ops=tuple(raw.get("ops", ()) or ()),
        accumulate=tuple(_accum(a) for a in raw.get("accumulate", ()) or ()),
        scaling=scaling,
        requant=raw.get("requant"),
        contains=tuple(raw.get("contains", ()) or ()),
        # An explicit declaration always wins; derive from this unit's own ops only when absent, so a
        # target that DOES declare its semantic block is completely unaffected.
        semantic_capabilities=(
            tuple(_sem_cap(s, name) for s in raw["semantic_capabilities"])
            if raw.get("semantic_capabilities")
            else _derived_sem_caps(raw, name)
        ),
        exposure=exposure,
    )


def resolve_exposure(unit: ComputeUnit, *, target_endpoint_kind: str | None = None) -> str:
    """How software drives ``unit``: the unit's own declaration, else the target's, else its family's.

    The precedence is the same FACTS > residual > family-default shape the target-wide ``endpoint_kind``
    already uses, which is what makes this change inert for existing targets: a single-unit target that
    declares no per-unit exposure resolves to exactly what it resolved to before.

    The reason this is per-unit at all is that a datapath class does not imply a software exposure. A
    spatial tensor tile is driven by one-hot command ports *inside* a vector unit, but software issues
    vector instructions to reach it — so its family default (``command_buffer``) describes the datapath
    correctly and the exposure wrongly. Letting the class imply the exposure makes a hybrid target
    inexpressible, which is a taxonomy failure rather than a missing special case.
    """
    if unit.exposure is not None:
        return unit.exposure
    if target_endpoint_kind is not None:
        return target_endpoint_kind
    from . import families

    return families.family_profile(unit.kind).endpoint_kind_default


def compute_units(contract: dict[str, Any]) -> list[ComputeUnit]:
    """Parse + validate a contract's ``compute_units`` (empty list if the field is absent)."""
    raw = contract.get("compute_units")
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("compute_units must be a list")
    return [_unit(u) for u in raw]


def effective(unit: ComputeUnit, all_units: list[ComputeUnit]) -> ComputeUnit:
    """Return ``unit`` with contained units' dtypes/ops/accumulate folded in (composition union).

    A composed unit (``contains: [...]``) has the combined capability of itself plus everything it
    embeds — so an MX systolic unit contributes its fp4/fp6/fp8 support when embedded in a larger target.
    """
    if not unit.contains:
        return unit
    by_name = {u.name: u for u in all_units}
    dtypes = list(unit.dtypes)
    ops = list(unit.ops)
    accum = list(unit.accumulate)
    if not accum:
        # A unit with no accumulate matrix takes every pairing of its own formats and accumulates
        # nothing. Folding a child's matrix into it would turn that into a whitelist of the CHILD's
        # formats, and the parent would refuse what it computes natively: measured, a cluster that
        # contains a block-scaled array routed its own float contraction to the host. State the
        # parent's freedom as rules before the child's are added, so the union widens and never
        # narrows.
        accum = [AccumRule(inp=a, weight=b, acc=None) for a in unit.dtypes for b in unit.dtypes]
    sem = list(unit.semantic_capabilities)
    for child_name in unit.contains:
        child = by_name.get(child_name)
        if child is None:
            raise ValueError(f"compute unit {unit.name!r} contains unknown unit {child_name!r}")
        child = effective(child, all_units)
        dtypes += [d for d in child.dtypes if d not in dtypes]
        ops += [o for o in child.ops if o not in ops]
        accum += [a for a in child.accumulate if a not in accum]
        sem += [s for s in child.semantic_capabilities if s not in sem]
    return ComputeUnit(
        name=unit.name,
        kind=unit.kind,
        dtypes=tuple(dtypes),
        ops=tuple(ops),
        accumulate=tuple(accum),
        scaling=unit.scaling,
        requant=unit.requant,
        contains=unit.contains,
        semantic_capabilities=tuple(sem),
        # Composition unions CAPABILITY (what can be computed), not exposure: how software drives this
        # unit is a property of this unit, and inheriting a child's would silently retarget the parent.
        exposure=unit.exposure,
    )


def datatype_tokens(unit: ComputeUnit) -> set[str]:
    """MLIR-ish element tokens a unit's storage dtypes present as (for RTL-fact cross-checks).

    Resolves each format name to its canonical name + aliases, so a datapath token like ``i8``
    matches the ``int8`` format (whose aliases include ``i8``).
    """
    tokens: set[str] = set()
    for name in unit.dtypes:
        fmt = qf.get(name)
        tokens.add(fmt.name)
        tokens.update(fmt.aliases)
    return tokens


def _merge_caps(a: SemanticCapability, b: SemanticCapability, *, keep_rows: bool = False) -> SemanticCapability:
    """Union two same-family capabilities into the most-permissive combined capability (a target is
    capable of a family if ANY of its units is): union dtypes/layouts/ranks, OR the boolean flags.

    ``composed_with`` is the one field that INTERSECTS rather than unions. It is a restriction, not a
    capability: if one unit runs the family standalone and another only as an epilogue, the TARGET runs
    it standalone, so the restriction must not survive the merge. Unioning it would let one unit's
    limitation constrain another unit's freedom; an empty intersection correctly means "standalone".

    ``keep_rows`` retains the folded rows as ``alternatives`` so eligibility can check them any-of.
    :func:`semantic_capability_map` keeps them for the rows ONE unit declares for a family (a fused
    and a standalone form of the same hardware); a fold across units keeps the legacy union view.
    """

    def _u(x: tuple, y: tuple) -> tuple:
        out = list(x)
        out += [v for v in y if v not in out]
        return tuple(out)

    def _i(x: tuple, y: tuple) -> tuple:
        return tuple(v for v in x if v in y)

    notes = "; ".join(n for n in (a.notes, b.notes) if n)
    return SemanticCapability(
        family=a.family,
        dtypes=_u(a.dtypes, b.dtypes),
        operand_pairs=(
            _u(a.operand_pairs, b.operand_pairs)
            if a.operand_pairs is not None and b.operand_pairs is not None
            else None
        ),
        result_dtypes=(
            _u(a.result_dtypes, b.result_dtypes)
            if a.result_dtypes is not None and b.result_dtypes is not None
            else None
        ),
        ranks=_u(a.ranks, b.ranks),
        transpose=a.transpose or b.transpose,
        arbitrary_mnk=a.arbitrary_mnk or b.arbitrary_mnk,
        batch=a.batch or b.batch,
        layouts=_u(a.layouts, b.layouts),
        forms=(() if not a.forms or not b.forms else _u(a.forms, b.forms)),
        # Engines UNION: two units providing one family means the target has two ways to run it, and
        # that is exactly the fact a single folded map used to destroy.
        engines=_u(a.engines, b.engines),
        composed_with=_i(a.composed_with, b.composed_with),
        # Evidence UNIONS: either unit's way of running the family standalone is a way the target has.
        standalone_evidence=_u(a.standalone_evidence, b.standalone_evidence),
        notes=notes,
        # A row with no kind restriction computes every kind, so the union is unconstrained.
        kinds=(() if not a.kinds or not b.kinds else _u(a.kinds, b.kinds)),
        alternatives=a.rows() + b.rows() if keep_rows else (),
    )


def semantic_capability_map(units: list[ComputeUnit]) -> dict[str, SemanticCapability]:
    """The target's HARDWARE semantic capability, folded across all units (composition resolved): a
    ``family -> merged SemanticCapability`` map. This is the independent ARR denominator source the
    eligibility oracle reads — derived only from the declared contract, never from routing/lowering.
    """
    merged: dict[str, SemanticCapability] = {}
    # Validate containment as effective() historically did, but project each
    # declaration from its OWN unit. Projecting after composition would allow
    # a parent's float operand to pair with a child's integer operand.
    for u in units:
        effective(u, units)
    for u in units:
        own: dict[str, SemanticCapability] = {}
        for cap in u.semantic_capabilities:
            # Attribute to the unit that DECLARED it before merging. `effective` folds a containing
            # unit's capabilities together with what it contains, so the attribution is taken from the
            # declaring unit found by `providers_of`, not from the outer unit's kind -- otherwise a
            # SIMT cluster containing a systolic mesh would report every family as SIMT.
            cap = _attribute(_with_operand_pairs(cap, u), providers_of(cap.family, units))
            # Several rows of one family on ONE unit are alternative forms of that unit's hardware
            # (fused vs standalone, or a narrower kind): keep each row for an any-of check.
            own[cap.family] = _merge_caps(own[cap.family], cap, keep_rows=True) if cap.family in own else cap
        for family, cap in own.items():
            merged[family] = _merge_caps(merged[family], cap) if family in merged else cap
    return merged


def _with_operand_pairs(cap: SemanticCapability, unit: ComputeUnit) -> SemanticCapability:
    """Project only pairs licensed by this declaring unit and this semantic family."""
    allowed = {qf.get(name).name for name in cap.dtypes}
    if cap.operand_pairs is not None:
        pairs = cap.operand_pairs  # independently declared hardware capability
    elif unit.accumulate:
        pairs = tuple(
            dict.fromkeys(
                (qf.get(rule.inp).name, qf.get(rule.weight).name)
                for rule in unit.accumulate
                if qf.has(rule.inp)
                and qf.has(rule.weight)
                and qf.get(rule.inp).name in allowed
                and qf.get(rule.weight).name in allowed
            )
        )
    else:
        # With no matrix, the semantic capability's dtypes are the independent
        # hardware declaration. They may intentionally be broader than the
        # router's unit.dtypes; intersecting would hide compiler false fallback.
        pairs = tuple((left, right) for left in sorted(allowed) for right in sorted(allowed))
    return replace(cap, operand_pairs=pairs)


def _attribute(cap: SemanticCapability, providers: tuple[tuple[str, str], ...]) -> SemanticCapability:
    """``cap`` with its ``engines`` set from the units that declare it (kept if already declared)."""
    if cap.engines or not providers:
        return cap
    kinds = tuple(dict.fromkeys(k for _, k in providers))
    return replace(cap, engines=kinds)


def providers_of(family: str, units: list[ComputeUnit]) -> tuple[tuple[str, str], ...]:
    """The ``(unit_name, kind)`` pairs that DECLARE ``family``, in declaration order.

    Reads each unit's OWN ``semantic_capabilities`` rather than its effective ones, because the
    question is which piece of silicon computes the family -- a cluster does not acquire a mesh's
    contraction by containing it, it acquires the mesh.
    """
    return tuple((u.name, u.kind) for u in units if any(c.family == family for c in u.semantic_capabilities))


def semantic_engine_map(units: list[ComputeUnit]) -> dict[str, tuple[tuple[str, str], ...]]:
    """``family -> ((unit_name, kind), ...)``: which units of a target cover which semantic family.

    The companion to :func:`semantic_capability_map`, which answers WHETHER and deliberately not WHICH.
    """
    fams: set[str] = set()
    for u in units:
        fams.update(c.family for c in effective(u, units).semantic_capabilities)
    return {f: providers_of(f, units) for f in sorted(fams)}
