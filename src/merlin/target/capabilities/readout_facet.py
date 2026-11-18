"""What a unit's readout can absorb, derived from the target and audited against what it declares.

A contraction leaves a wide accumulator. Whether the scale, the rounding, the clamp and the
activation that turn it into a stored element happen ON the unit or on the host is decided by one
question nothing in the compiler used to ask: at what granularity can this hardware hold a scale?

The cost of not asking is a whole model's performance. A quantizer that emits one scale per output
channel, lowered onto a store path that holds one scale per command, leaves the compiler two
choices: read the raw accumulator out and finish every element on the host, or compute something
other than what the model says. The first is what happened; it is correct, it passes every gate, and
it is two orders of magnitude slower than the hardware.

So the granularity a readout accepts is a FACT, derived here and compared with the contract's
``scaling:`` declaration, which until now nothing checked.

**Evidence ladder.** Each field records the rung that produced it and the thing observed.

``format``
    A unit whose operand format carries its own scale structure (a block-scaled format) fixes the
    granularity by definition: the block length and the scale dtype are the format's.
``rtl_datapath``
    Extracted element and accumulator dtypes, and whether the accumulator is an addressable memory
    or exists only inside the datapath.
``register_layout``
    The target's own command-register layouts. A field whose name carries the ``scale`` role and
    that is a register field of bounded width holds ONE value per command, so every store it
    configures has one scale. Roles are read from the tokens of the target's own field names
    against a closed, target-neutral vocabulary; nothing here names a target, a register or a
    command.
``isa_role``
    On a self-hosted ISA, the structural role census. A scaled-readout class proves a scale is an
    instruction OPERAND. It does not say how much of the accumulator one such instruction drains,
    so this rung records the carrier and leaves the granularity unknown with that as the reason.
``scalar_abi``
    A scalar readout contract the target's backend derived from its generated parameter header
    (accumulator, element and scale types, clamp bounds, round-half-to-even). Supplied by the
    backend through an optional ``readout_scalar_abi`` hook, because the header is the target's.
``backend_declared``
    Which stages each readout selector applies (``readout_epilogue_capability``), weakest because
    it is declared, and the only source for it.

**Three states.** A field no rung could decide is ``None`` with a reason under ``unknown``. It is
never defaulted: a granularity assumed to be per-tensor on a target that can do better loses
accuracy silently, and one assumed to be per-channel on a target that cannot is the defect above.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.common import target_data as qf

# ---- readout_facet -----------------------------------------------------------------------------

SCHEMA = "readout_facet_v1"


SCALAR_ABI_SCHEMA = "scalar_narrow_readout_contract_v1"


HANDOFF_SCHEMA = "readout_numerics_handoff_v1"


OPERAND_SUM_SCHEMA = "operand_sum_contract_v1"


GRANULARITIES = ("tensor", "row", "column", "rank1", "block")


GRANULARITY_OF_SCALE_KIND: dict[str, str] = {
    "per_tensor": "tensor",
    "per_channel": "column",
    "per_group": "block",
    "block_affine": "block",
    "block_e8m0": "block",
    "nvfp4_block": "block",
    "kquant_superblock": "block",
}


EPILOGUE_GRANULARITY: dict[str, str] = {"tensor": "per_tensor", "column": "per_axis"}


TORCHAO_GRANULARITY: dict[str, str] = {"tensor": "PerTensor", "row": "PerRow", "column": "PerAxis", "block": "PerGroup"}


_FIELD_ROLE_TOKENS: dict[str, tuple[str, ...]] = {
    "scale": ("scale",),
    "shift": ("shift",),
    "activation": ("activation", "act"),
    "pool": ("pool",),
    "zero_point": ("zp", "zeropoint"),
}


_REFERENCE_TOKENS = ("addr", "address", "ptr", "pointer", "base")


def field_roles(name: str) -> tuple[str, ...]:
    """The roles a register field's own name states, by token; ``()`` when it states none."""
    tokens = [token for token in name.lower().split("_") if token]
    return tuple(role for role, spellings in _FIELD_ROLE_TOKENS.items() if any(token in spellings for token in tokens))


def _is_reference(name: str) -> bool:
    return any(token in _REFERENCE_TOKENS for token in name.lower().split("_"))


@dataclass(frozen=True)
class Evidence:
    field: str
    rung: str
    observed: str


@dataclass
class ReadoutFacet:
    target: str
    unit: str | None = None
    element_dtype: str | None = None
    accumulator_dtype: str | None = None
    accumulator_kind: str | None = None  # "addressable" | "in_datapath"
    scale_granularities: tuple[str, ...] | None = None
    #: ``False`` when ``scale_granularities`` is a derived LOWER BOUND: what is listed is held, and a
    #: granularity that is not listed is unknown, not refused. Refusing on a lower bound would
    #: report a hardware gap nobody established.
    scale_granularities_complete: bool = True
    scale_block: int | None = None
    scale_dtype: str | None = None
    scale_carriers: tuple[dict[str, Any], ...] = ()
    rounding: str | None = None
    clamp: tuple[int, int] | None = None
    register_stages: tuple[str, ...] = ()  # roles with a register field: what is ENCODABLE
    zero_point_carried: bool | None = None  # a register field carries an output zero point
    readouts: tuple[dict[str, Any], ...] = ()  # selector -> the stages that readout applies
    stage_routes: tuple[dict[str, Any], ...] = ()  # non-readout stages, scoped to a composition path
    unknown: dict[str, str] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    scalar_abi: dict[str, Any] | None = None
    #: What a LOAD does to an operand on its way into the accumulator, when the design's load
    #: multiplies: the licence for an integer sum of separately scaled tensors. ``None`` with the
    #: reason beside it otherwise. The reason is NOT an ``unknown``: a unit that cannot sum operands
    #: has a fully derived readout, and ``unknown`` is what stops a readout from licensing anything.
    operand_sum: dict[str, Any] | None = None
    operand_sum_absent: str | None = None
    #: Command-buffer stages the contract's DERIVED output epilogue says this unit's output path
    #: applies (:mod:`merlin.target.rtl.firrtl.output_epilogue`), and whether it accumulates an operand on
    #: load. Read from the design, so a derived stage is applied even where no readout is declared.
    derived_stages: tuple[str, ...] = ()
    accumulates: bool = False

    def note(self, name: str, rung: str, observed: str) -> None:
        self.evidence.append(Evidence(name, rung, observed))
        self.unknown.pop(name, None)

    def rungs_for(self, name: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.rung for e in self.evidence if e.field == name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "target": self.target,
            "unit": self.unit,
            "element_dtype": self.element_dtype,
            "accumulator_dtype": self.accumulator_dtype,
            "accumulator_kind": self.accumulator_kind,
            "scale": {
                "granularities": (list(self.scale_granularities) if self.scale_granularities is not None else None),
                "granularities_complete": self.scale_granularities_complete,
                "block": self.scale_block,
                "dtype": self.scale_dtype,
                "carriers": [dict(carrier) for carrier in self.scale_carriers],
            },
            "rounding": self.rounding,
            "clamp": list(self.clamp) if self.clamp else None,
            "register_stages": list(self.register_stages),
            "zero_point_carried": self.zero_point_carried,
            "readouts": [dict(readout) for readout in self.readouts],
            "stage_routes": [dict(route) for route in self.stage_routes],
            "operand_sum": ({**self.operand_sum, "bound_lsb": self.operand_sum_bound()} if self.operand_sum else None),
            "operand_sum_absent": self.operand_sum_absent,
            "derived_stages": list(self.derived_stages),
            "accumulates": self.accumulates,
            "unknown": dict(sorted(self.unknown.items())),
            "evidence": [{"field": e.field, "rung": e.rung, "observed": e.observed} for e in self.evidence],
        }

    def scalar_readout_facts(self) -> dict[str, Any] | None:
        """The scalar readout contract a numerics contract is built from, or ``None``.

        Only the backend's header-verified record licenses a rounding, so only that record is
        returned; the dtypes this facet derived elsewhere are not promoted into one.
        """
        return dict(self.scalar_abi) if self.scalar_abi else None

    def numerics_handoff(self) -> dict[str, Any]:
        """What a kernel scheduler may build its numerics contract from, with a digest to compare by.

        Two pieces of work read this facet: group formation, which decides what a readout absorbs,
        and kernel scheduling, which decides how the absorbed stages are issued. They must not hold
        two opinions of one readout. This is the whole of what the second may assume: the
        header-verified scalar record (the only thing that licenses a rounding), the granularities
        a contract may be built at, and whether that list is complete. ``sha256`` is over the
        canonical JSON of everything else, so the two sides agree by comparing one string.
        """
        import hashlib
        import json

        body = {
            "schema": HANDOFF_SCHEMA,
            "target": self.target,
            "unit": self.unit,
            "readout_facts": self.scalar_readout_facts(),
            "granularities": list(self.scale_granularities) if self.scale_granularities is not None else None,
            "granularities_complete": self.scale_granularities_complete,
            "scale_block": self.scale_block,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return {**body, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}

    def sum_fits_accumulator(self, count: int) -> bool | None:
        """Whether ``count`` elements of the element type, summed, always fit the accumulator.

        ``None`` when either width is not derived as an integer format: not knowing is not a no.
        """
        widths = []
        for name in (self.element_dtype, self.accumulator_dtype):
            digits = str(name or "").lstrip("iu")
            if not name or str(name)[0] not in "iu" or not digits.isdigit():
                return None
            widths.append(int(digits))
        return int(count) * (1 << (widths[0] - 1)) <= (1 << (widths[1] - 1)) - 1

    def operand_sum_bound(self, multipliers: Sequence[float] = ()) -> int | None:
        """How many output steps this unit's sum may lie from the single-rounding reference.

        Computed, never declared. Each of ``n`` operands is rounded on its way in, so the integer
        sum carries up to ``n/2`` of error, and the reference rounds its own sum by up to a half:
        with every multiplier at or below one the two integers differ by at most ``(n+1)//2``.
        A multiplier above one cannot go through the load (see :meth:`operand_sum_refusal`), so
        the common factor ``m`` is divided out of every operand and applied by the readout's own
        scale instead. The operands' rounding error is then multiplied by ``m`` and the readout
        rounds once more, which gives ``floor(m*n/2 + 1)``.
        """
        if not self.operand_sum:
            return None
        import math

        n = int(self.operand_sum["operands"])
        m = max([1.0, *(float(value) for value in multipliers)])
        return (n + 1) // 2 if m == 1.0 else math.floor(m * n / 2 + 1)

    def operand_sum_refusal(self, multipliers: Sequence[float]) -> str | None:
        """Why this unit cannot sum operands under these multipliers, or ``None`` when it can."""
        if not self.operand_sum:
            return self.operand_sum_absent or "the unit's load applies no scale"
        if len(multipliers) > int(self.operand_sum["operands"]):
            return f"the unit sums {self.operand_sum['operands']} operands, not {len(multipliers)}"
        if any(float(m) < 0.0 for m in multipliers):
            return f"multiplier(s) {[m for m in multipliers if float(m) < 0.0]} are negative"
        wide = [m for m in multipliers if float(m) > 1.0]
        if wide and self.operand_sum.get("operand_saturates") and not self.scale_granularities:
            # A scaled operand is saturated to the element type BEFORE it is added, so a multiplier
            # above one can clip an operand whose share of the sum the other would have cancelled:
            # an error no bound covers. The way out is to divide the common factor out of the
            # loads and give it to the readout's scale, and this readout is not derived to have one.
            return (
                f"multiplier(s) {wide} exceed one: each operand saturates to "
                f"{self.operand_sum.get('operand_dtype')} before the add, and no readout scale is "
                f"derived to carry the common factor instead"
            )
        return None

    def admits_granularity(self, granularity: str) -> bool | None:
        """``True`` / ``False`` when derived, ``None`` when the target's granularity is unknown."""
        if self.scale_granularities is None:
            return None
        if granularity in self.scale_granularities:
            return True
        return False if self.scale_granularities_complete else None

    def capabilities(self) -> tuple[Any, ...]:
        """This unit's ``readouts`` as the rule module's objects, so one type states the declaration."""
        from merlin.verification.references.epilogue_applicability import ReadoutCapability

        return tuple(
            ReadoutCapability(
                selector=str(r.get("selector")),
                applies=frozenset(str(s) for s in (r.get("applies") or ())),
                evidence=str(r.get("evidence") or ""),
            )
            for r in self.readouts
            if isinstance(r, Mapping) and r.get("selector")
        )

    def applies_stage(self, stage: str) -> bool | None:
        """Does a readout this unit declares APPLY ``stage``? ``None`` when it declares none.

        ``stage`` is in the command-buffer ABI's vocabulary
        (:data:`merlin.runtime.engine.command_buffer.EPILOGUE_STAGES`), the same words
        ``readout_epilogue_capability`` is declared in. The rule is
        :func:`merlin.verification.references.epilogue_applicability.selectors_applying`: a stage is applied when SOME
        declared readout applies it, and a readout that applies nothing (the full-width path that
        writes the raw accumulator) contributes nothing, exactly as it should.

        ``None`` is UNKNOWN and never "applies everything" -- assuming would reproduce the silent
        discard this declaration exists to catch, and assuming the other way would report a hardware
        gap nobody established.
        """
        from merlin.verification.references.epilogue_applicability import selectors_applying

        if stage in self.derived_stages:
            return True  # read from the design's own output path: stronger than any declaration
        capabilities = self.capabilities()
        if not capabilities:
            return None
        return bool(selectors_applying(capabilities, (stage,)))


def _from_formats(facet: ReadoutFacet, dtypes: Sequence[str]) -> None:
    for name in dtypes:
        try:
            scale = qf.get(name).scale
        except (KeyError, ValueError):
            continue
        if not scale.is_block:
            continue
        facet.scale_granularities = ("block",)
        facet.scale_block, facet.scale_dtype = scale.block, scale.dtype
        if facet.element_dtype is None:
            # A block-scaled format IS its element format; no datapath extraction is needed to know
            # what the unit stores.
            facet.element_dtype = name
            facet.note("element_dtype", "format", f"the unit's operand format is {name!r}")
        facet.note(
            "scale_granularities",
            "format",
            f"operand format {name!r} is block-scaled ({scale.kind}, block {scale.block}, "
            f"scale dtype {scale.dtype}) by its own definition",
        )
        return


def _from_datapath(facet: ReadoutFacet, facts: Mapping[str, Any]) -> None:
    datapaths = {str(d.get("name")): d for d in facts.get("datapaths") or () if isinstance(d, Mapping)}
    for role, attribute in (("input", "element_dtype"), ("accumulator", "accumulator_dtype")):
        entry = datapaths.get(role)
        if entry and entry.get("dtype"):
            setattr(facet, attribute, str(entry["dtype"]))
            facet.note(attribute, "rtl_datapath", str(entry.get("evidence") or entry["dtype"]))
    if "accumulator" in datapaths:
        memories = {str(m.get("name")) for m in facts.get("memories") or () if isinstance(m, Mapping)}
        facet.accumulator_kind = "addressable" if "accumulator" in memories else "in_datapath"
        facet.note(
            "accumulator_kind",
            "rtl_datapath",
            "an accumulator memory is extracted beside the accumulator datapath"
            if facet.accumulator_kind == "addressable"
            else "the accumulator exists as a datapath width and as no extracted memory",
        )


def _from_address_space(facet: ReadoutFacet, target: str, facts: Mapping[str, Any]) -> None:
    """Where the accumulator lives, from the one module that derives it.

    :func:`merlin.target.capabilities.address_space.accumulator_kind` resolves the accumulator store by row
    width AND requires an accumulate datapath, which is stricter than reading a memory's name off
    the facts. Where it reaches a verdict that verdict replaces the datapath rung's; where it does
    not, the datapath rung's reading stands and says which rung it came from.
    """
    try:
        from merlin.target.capabilities import address_space as AS

        kind = AS.accumulator_kind(AS.derive_address_space(target, facts=dict(facts)))
    except Exception:  # noqa: BLE001 -- no derivable address space: the datapath rung stands
        return
    if kind.kind not in (AS.ADDRESSABLE, AS.IN_DATAPATH):
        return
    agrees = facet.accumulator_kind in (None, kind.kind)
    facet.accumulator_kind = kind.kind
    facet.note(
        "accumulator_kind",
        "address_space",
        f"the address space resolves the accumulator as {kind.kind}"
        + (f" ({kind.rows} rows x {kind.buffers} buffer(s) of {kind.dtype})" if kind.rows else "")
        + ("" if agrees else "; the datapath rung read it differently and is overruled"),
    )


def _register_bundles(facts: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for interface in facts.get("interfaces") or ():
        if isinstance(interface, Mapping) and interface.get("name") == "register_bundle_layouts":
            return interface
    return None


def with_current_register_layouts(
    facts: Mapping[str, Any], *, source_bytes: Mapping[str, bytes] | None = None
) -> Mapping[str, Any]:
    """``facts`` with its register layouts re-read by the CURRENT reader, when they predate it.

    A cached fact bundle outlives the reader that wrote it. One written before parametric Bundles
    were placed carries no scale register at all, and a facet derived from it can only say
    "unknown". The bundle records the source file the layouts were read from, so they are read
    again from that same file, in memory: nothing is written back, the cache is another process's
    to regenerate, and the record says it was refreshed and from what bytes. A bundle that is
    already current, or whose source is not on this machine, is returned as it is.
    """
    import hashlib
    from pathlib import Path

    body = facts.get("facts", facts) if isinstance(facts, Mapping) else {}
    record = _register_bundles(body)
    if record is None or "unresolved" in record:
        return facts
    source = Path(str(record.get("source") or ""))
    if source_bytes is not None:
        raw = source_bytes.get(str(source), source_bytes.get(str(source.absolute())))
        if raw is None:
            return facts
    else:
        if not source.is_file():
            return facts
        raw = source.read_bytes()
    from merlin.target.rtl.extraction.circt_decode import _bundle_layouts

    text = raw.decode("utf-8", errors="replace")
    layouts, unresolved = _bundle_layouts(text)
    refreshed = {
        **record,
        "bundles": layouts,
        "unresolved": unresolved,
        "refreshed": {
            "why": "the cached layouts predate the parametric-bundle reader",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    interfaces = [refreshed if item is record else item for item in body.get("interfaces") or ()]
    new_body = {**body, "interfaces": interfaces}
    return {**facts, "facts": new_body} if "facts" in facts else new_body


def _from_registers(facet: ReadoutFacet, facts: Mapping[str, Any]) -> None:
    record = _register_bundles(facts)
    if record is None:
        return
    unresolved = record.get("unresolved")
    carriers: list[dict[str, Any]] = []
    stages: list[str] = []
    for bundle, layout in sorted((record.get("bundles") or {}).items()):
        for name, placed in sorted((layout.get("fields") or {}).items()):
            roles = field_roles(name)
            for role in roles:
                if role != "scale" and role not in stages:
                    stages.append(role)
            if "scale" in roles:
                bits = placed.get("width") or placed.get("slot_width")
                carriers.append({"bundle": bundle, "field": name, "bits": bits, "by_reference": _is_reference(name)})
    facet.register_stages = tuple(stages)
    for role in stages:
        facet.note(f"register_stage:{role}", "register_layout", f"a command-register field carries the {role!r} role")
    facet.scale_carriers = tuple(carriers)
    if unresolved is not None:
        # Only a reader that places EVERY Bundle (or names the ones it could not) can say a role
        # is absent; the older one dropped the parametric Bundles and would call anything absent.
        facet.zero_point_carried = "zero_point" in stages
        facet.note(
            "zero_point_carried",
            "register_layout",
            "a command-register field carries a zero point"
            if facet.zero_point_carried
            else f"no command-register field carries a zero point"
            + (f" (layouts not derived for {sorted(unresolved)})" if unresolved else ""),
        )
    if facet.scale_granularities is not None:
        return  # a stronger rung already fixed it
    if not carriers:
        if unresolved is None:
            facet.unknown["scale_granularities"] = (
                "the register layouts were extracted by a reader that dropped every Bundle "
                "parameterised by its field widths, which is where a design's scale fields live; "
                "re-derive the target's facts"
            )
        else:
            facet.unknown["scale_granularities"] = "no command-register field carries the scale role" + (
                f"; layouts not derived for {sorted(unresolved)}" if unresolved else ""
            )
        return
    if any(carrier["by_reference"] for carrier in carriers):
        facet.unknown["scale_granularities"] = (
            "a scale is passed by reference, so its extent is a property of the memory it points "
            "at, which no rung here reads"
        )
        return
    facet.scale_granularities = ("tensor",)
    spelled = ", ".join(f"{c['bundle']}.{c['field']} ({c['bits']} bits)" for c in carriers)
    if record.get("refreshed"):
        spelled += (
            f"; layouts re-read from the recorded source (sha256 {str(record['refreshed'].get('source_sha256'))[:12]})"
        )
    facet.note(
        "scale_granularities",
        "register_layout",
        f"every scale the design accepts is one bounded register field per command "
        f"({spelled}); none is passed by reference, so a configured store has one scale",
    )


def _from_isa_roles(facet: ReadoutFacet, taxonomy: Mapping[str, Any] | None) -> None:
    from merlin.target.capabilities.readout_derivation import _PLAIN_READOUT_ROLE, _SCALED_READOUT_ROLE

    if not taxonomy or facet.scale_granularities is not None:
        return
    scaled = sorted(
        name
        for name, members in (taxonomy.get("by_class") or {}).items()
        if any(isinstance(m, Mapping) and m.get("role") == _SCALED_READOUT_ROLE for m in members or ())
    )
    if not scaled:
        return
    facet.scale_carriers += tuple({"instruction_class": name, "by_reference": False} for name in scaled)
    facet.evidence.append(
        Evidence(
            "scale_carriers", "isa_role", f"scaled-readout instruction class(es) {scaled} take the scale as an operand"
        )
    )
    # WHAT ONE SCALED READOUT CAN ADDRESS. Read off the encodings, never off a name: when every
    # scaled readout carries exactly the operand fields a plain readout of the same ISA carries, it
    # has nothing with which to select a PART of what a plain readout drains. One such instruction
    # is then one drain under one scale, so a scale that is constant over the tensor is always
    # expressible: issue every drain with it. That is a lower bound and is recorded as one. How
    # much one drain covers decides anything finer, and it is not in an encoding.
    by_class = taxonomy.get("by_class") or {}
    members = [m for group in by_class.values() for m in group or () if isinstance(m, Mapping)]
    scaled_fields = [frozenset(m.get("fields") or ()) for m in members if m.get("role") == _SCALED_READOUT_ROLE]
    plain_fields = {frozenset(m.get("fields") or ()) for m in members if m.get("role") == _PLAIN_READOUT_ROLE}
    if scaled_fields and plain_fields and all(fields in plain_fields for fields in scaled_fields):
        facet.scale_granularities, facet.scale_granularities_complete = ("tensor",), False
        facet.note(
            "scale_granularities",
            "isa_role",
            f"every scaled readout ({', '.join(scaled)}) carries exactly a plain readout's operand "
            f"fields, so it cannot address part of a drain: one scale per drain, hence a per-tensor "
            f"scale is expressible. A LOWER BOUND; anything finer depends on what one drain covers",
        )
        facet.unknown["scale_granularities_finer"] = (
            "how much of the accumulator one readout instruction drains is not derived, so whether a "
            "scale finer than per tensor is held is unknown (it is not refused)"
        )
        return
    facet.unknown["scale_granularities"] = (
        f"the ISA's scaled readout ({', '.join(scaled)}) takes its scale as an instruction operand; "
        f"how much of the accumulator one such instruction drains is not derived, so neither is "
        f"the granularity"
    )


def _from_scalar_abi(facet: ReadoutFacet, abi: Mapping[str, Any] | None) -> None:
    if not abi or abi.get("schema") != SCALAR_ABI_SCHEMA:
        return
    facet.scalar_abi = dict(abi)
    source = f"scalar readout contract ({abi.get('provenance', {}).get('scope', 'backend-derived')})"
    for key, attribute in (
        ("accumulator_dtype", "accumulator_dtype"),
        ("output_dtype", "element_dtype"),
        ("scale_dtype", "scale_dtype"),
    ):
        value = abi.get(key)
        if value is None:
            continue
        current = getattr(facet, attribute)
        if current is None:
            setattr(facet, attribute, str(value))
        facet.note(
            attribute,
            "scalar_abi",
            f"{source}: {key}={value}" + ("" if current in (None, str(value)) else f" (rtl says {current})"),
        )
    if abi.get("clamp_min") is not None and abi.get("clamp_max") is not None:
        facet.clamp = (int(abi["clamp_min"]), int(abi["clamp_max"]))
        facet.note("clamp", "scalar_abi", f"{source}: clamp [{facet.clamp[0]}, {facet.clamp[1]}]")
    # The schema is only produced after the target's scale and rounding sources were matched
    # against the round-half-to-even construction; that match is what licenses the value.
    facet.rounding = "half_even"
    facet.note("rounding", "scalar_abi", f"{source}: rounding matched round-half-to-even")


def _from_operand_sum(facet: ReadoutFacet, contract: Mapping[str, Any] | None) -> None:
    if not contract:
        facet.operand_sum_absent = "its backend supplies no load-scale contract"
        return
    if contract.get("schema") != OPERAND_SUM_SCHEMA:
        facet.operand_sum_absent = f"load-scale contract schema {contract.get('schema')!r} is not one this reads"
        return
    if facet.accumulator_kind != "addressable":
        # Two loads can only meet in a store a load can address. A design whose accumulator is
        # state inside the datapath has a scaled load and still no place to sum two of them.
        facet.operand_sum_absent = (
            f"the load multiplies, but the accumulator is {facet.accumulator_kind or 'not derived'}, "
            f"and separately loaded operands can only be summed in an addressable one"
        )
        return
    facet.operand_sum = {
        key: contract[key]
        for key in ("operands", "operand_dtype", "scale_dtype", "operand_rounding", "operand_saturates")
        if key in contract
    }
    scope = (contract.get("provenance") or {}).get("scope", "backend-derived")
    facet.note(
        "operand_sum",
        "scalar_abi",
        f"load-scale contract ({scope}): a load multiplies by an {contract.get('scale_dtype')} scale, rounds "
        f"{contract.get('operand_rounding')}"
        + (f" and saturates to {contract.get('operand_dtype')}" if contract.get("operand_saturates") else "")
        + f"; {contract.get('operands')} loads accumulate in the addressable accumulator",
    )
