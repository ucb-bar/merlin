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

from merlin.common import quant_formats as qf

SCHEMA = "readout_facet_v1"
SCALAR_ABI_SCHEMA = "scalar_narrow_readout_contract_v1"
HANDOFF_SCHEMA = "readout_numerics_handoff_v1"
OPERAND_SUM_SCHEMA = "operand_sum_contract_v1"

#: How a scale varies over the accumulator tile. ``row`` / ``column`` are one scale per tile row /
#: column, ``rank1`` their outer product, ``block`` one per fixed-length run of elements.
GRANULARITIES = ("tensor", "row", "column", "rank1", "block")

#: The contract's ``scaling:`` vocabulary (``quant_formats.SCALE_KINDS``) in facet terms. ``none``
#: is absent on purpose: a unit that declares no scaling makes no claim to audit.
GRANULARITY_OF_SCALE_KIND: dict[str, str] = {
    "per_tensor": "tensor",
    "per_channel": "column",
    "per_group": "block",
    "block_affine": "block",
    "block_e8m0": "block",
    "nvfp4_block": "block",
    "kquant_superblock": "block",
}
#: Facet granularity -> the quantized-epilogue contract's spelling (the ones it can express).
EPILOGUE_GRANULARITY: dict[str, str] = {"tensor": "per_tensor", "column": "per_axis"}
#: Facet granularity -> the observer granularity a PT2E / ``quantize_`` recipe asks TorchAO for.
TORCHAO_GRANULARITY: dict[str, str] = {"tensor": "PerTensor", "row": "PerRow", "column": "PerAxis", "block": "PerGroup"}

#: Register-field roles, read from the tokens of a target's own field names.
_FIELD_ROLE_TOKENS: dict[str, tuple[str, ...]] = {
    "scale": ("scale",),
    "shift": ("shift",),
    "activation": ("activation", "act"),
    "pool": ("pool",),
    "zero_point": ("zp", "zeropoint"),
}
#: Tokens that make a field a REFERENCE to values held elsewhere rather than the value itself.
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
    unknown: dict[str, str] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    scalar_abi: dict[str, Any] | None = None
    #: What a LOAD does to an operand on its way into the accumulator, when the design's load
    #: multiplies: the licence for an integer sum of separately scaled tensors. ``None`` with the
    #: reason beside it otherwise. The reason is NOT an ``unknown``: a unit that cannot sum operands
    #: has a fully derived readout, and ``unknown`` is what stops a readout from licensing anything.
    operand_sum: dict[str, Any] | None = None
    operand_sum_absent: str | None = None

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
            "operand_sum": ({**self.operand_sum, "bound_lsb": self.operand_sum_bound()} if self.operand_sum else None),
            "operand_sum_absent": self.operand_sum_absent,
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
        from merlin.verify.epilogue_applicability import ReadoutCapability

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
        (:data:`merlin.runtime.commandbuffer.EPILOGUE_STAGES`), the same words
        ``readout_epilogue_capability`` is declared in. The rule is
        :func:`merlin.verify.epilogue_applicability.selectors_applying`: a stage is applied when SOME
        declared readout applies it, and a readout that applies nothing (the full-width path that
        writes the raw accumulator) contributes nothing, exactly as it should.

        ``None`` is UNKNOWN and never "applies everything" -- assuming would reproduce the silent
        discard this declaration exists to catch, and assuming the other way would report a hardware
        gap nobody established.
        """
        from merlin.verify.epilogue_applicability import selectors_applying

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

    :func:`merlin.targetgen.address_space.accumulator_kind` resolves the accumulator store by row
    width AND requires an accumulate datapath, which is stricter than reading a memory's name off
    the facts. Where it reaches a verdict that verdict replaces the datapath rung's; where it does
    not, the datapath rung's reading stands and says which rung it came from.
    """
    try:
        from merlin.targetgen import address_space as AS

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


def with_current_register_layouts(facts: Mapping[str, Any]) -> Mapping[str, Any]:
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
    if not source.is_file():
        return facts
    from merlin.targetgen.rtl.circt_introspect import _bundle_layouts

    text = source.read_text(encoding="utf-8", errors="replace")
    layouts, unresolved = _bundle_layouts(text)
    refreshed = {
        **record,
        "bundles": layouts,
        "unresolved": unresolved,
        "refreshed": {
            "why": "the cached layouts predate the parametric-bundle reader",
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
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


_NO_RUNG = "no rung produced it"
#: The structural ISA role of a readout that applies a scale (``semantic_families.ISA_ROLE_FAMILY``).
_SCALED_READOUT_ROLE = "acc_readout_scaled"
_PLAIN_READOUT_ROLE = "acc_readout"


def derive(
    target: str,
    *,
    facts: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
    scalar_abi: Mapping[str, Any] | None = None,
    readouts: Sequence[Mapping[str, Any]] | None = None,
    taxonomy: Mapping[str, Any] | None = None,
    operand_sum: Mapping[str, Any] | None = None,
) -> ReadoutFacet:
    """Derive one unit's readout facet from supplied evidence. Pure: it loads nothing."""
    facet = ReadoutFacet(target=target, unit=str(unit.get("name")) if unit else None)
    for name in (
        "element_dtype",
        "accumulator_dtype",
        "accumulator_kind",
        "scale_granularities",
        "scale_dtype",
        "rounding",
        "clamp",
        "zero_point_carried",
    ):
        facet.unknown[name] = _NO_RUNG
    body = facts.get("facts", facts) if isinstance(facts, Mapping) else {}
    _from_formats(facet, tuple((unit or {}).get("dtypes") or ()))
    _from_datapath(facet, body)
    if facts:
        _from_address_space(facet, target, facts)
    _from_registers(facet, body)
    _from_scalar_abi(facet, scalar_abi)
    _from_operand_sum(facet, operand_sum)
    # An unknown says which rungs were silent and why, because "not derived" tells the next person
    # nothing about whether to extract more or to stop looking.
    silent = []
    if not body.get("datapaths"):
        silent.append("the target's facts carry no extracted datapath")
    if _register_bundles(body) is None:
        silent.append("its facts carry no command-register layouts")
    if not scalar_abi:
        silent.append("its backend supplies no scalar readout contract")
    _from_isa_roles(facet, taxonomy)
    for name, why in list(facet.unknown.items()):
        if why == _NO_RUNG:
            facet.unknown[name] = "; ".join(silent) or "every rung ran and none decides this field"
    if readouts:
        facet.readouts = tuple(dict(readout) for readout in readouts)
        facet.note(
            "readouts",
            "backend_declared",
            f"{len(facet.readouts)} readout selector(s) declared by the target's backend",
        )
    return facet


def _backend_hook(target: str, name: str) -> Callable[[], Any] | None:
    from merlin.runtime.backends import base

    try:
        backend = base.get_backend(target)
    except Exception:  # noqa: BLE001 -- a target with no runtime backend has no hook, not an error
        return None
    hook = getattr(backend, name, None)
    return hook if callable(hook) else None


def epilogue_readouts(target: str):
    """The target's own ``readout_epilogue_capability`` declaration, as rule-module objects.

    ONE accessor, so the grade-time check and the capsule GENERATOR read the same declaration. They did
    not before: the check consulted it and the generator did not, which is how a corpus came to hold
    capsules whose declared epilogue no readout applies -- found, one at a time, by the check that runs
    hours later instead of by the writer that had the answer in hand.

    ``None`` means the target declares nothing, which is UNKNOWN and never "applies everything".
    """
    hook = _backend_hook(target, "readout_epilogue_capability")
    if hook is None:
        return None
    try:
        raw = hook()
    except Exception:  # noqa: BLE001 -- an unreadable declaration is an absent one, not a crash
        return None
    if not raw:
        return None
    from merlin.verify.epilogue_applicability import ReadoutCapability

    return [
        ReadoutCapability(
            selector=str(r.get("selector")),
            applies=frozenset(str(s) for s in (r.get("applies") or ())),
            evidence=str(r.get("evidence") or ""),
        )
        for r in raw
        if isinstance(r, Mapping) and r.get("selector")
    ]


def for_target(
    target: str, *, contract: Mapping[str, Any] | None = None, facts: Mapping[str, Any] | None = None
) -> list[ReadoutFacet]:
    """One facet per compute unit the target declares (one unit-less facet if it declares none)."""
    if contract is None:
        from merlin.targetgen.target_experiment import load_capability_manifest

        contract = load_capability_manifest(target).contract
    if facts is None:
        from merlin.targetgen.rtl import facts as rtl_facts

        try:
            facts = with_current_register_layouts(rtl_facts.load_facts(target))
        except Exception:  # noqa: BLE001 -- underivable facts leave every RTL rung silent
            facts = {}
    abi_hook, readout_hook = (
        _backend_hook(target, "readout_scalar_abi"),
        _backend_hook(target, "readout_epilogue_capability"),
    )
    scalar_abi = None
    if abi_hook is not None:
        try:
            scalar_abi = abi_hook()
        except Exception:  # noqa: BLE001 -- an unreadable header is an absent rung
            scalar_abi = None
    readouts = readout_hook() if readout_hook is not None else None
    sum_hook, operand_sum = _backend_hook(target, "readout_operand_sum"), None
    if sum_hook is not None:
        try:
            operand_sum = sum_hook()
        except Exception:  # noqa: BLE001 -- an unreadable header is an absent rung
            operand_sum = None
    taxonomy = None
    if _register_bundles(facts.get("facts", facts) if isinstance(facts, Mapping) else {}) is None:
        # Only a target with no command registers can be a self-hosted ISA worth a role census.
        try:
            from merlin.targetgen import isa_taxonomy

            taxonomy = isa_taxonomy.taxonomy_for_target(target)
        except Exception:  # noqa: BLE001 -- no ISA definition: the rung is silent
            taxonomy = None
    units = [u for u in contract.get("compute_units") or () if isinstance(u, Mapping)] or [None]
    return [
        derive(
            target,
            facts=facts,
            unit=u,
            scalar_abi=scalar_abi,
            readouts=readouts,
            taxonomy=taxonomy,
            operand_sum=operand_sum,
        )
        for u in units
    ]


@dataclass(frozen=True)
class TargetReadout:
    """Every unit's facet, asked as one: can ANY readout of this target hold the granularity?

    ``True`` if one can, ``False`` only if every facet was derived and none can, else ``None``.
    An underived facet beside a derived refusal is unknown, not a refusal: the unit nobody could
    read may be the one that holds it.
    """

    facets: tuple[ReadoutFacet, ...]

    @property
    def scale_granularities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(g for f in self.facets for g in f.scale_granularities or ()))

    @property
    def unknown(self) -> dict[str, str]:
        return {key: why for f in self.facets for key, why in f.unknown.items()}

    def operand_sum(self, multipliers: Sequence[float]) -> tuple[ReadoutFacet | None, str | None]:
        """``(facet, None)`` for the first unit that can sum operands under ``multipliers``, else
        ``(None, why)`` with the first refusal."""
        refusals = []
        for facet in self.facets:
            refusal = facet.operand_sum_refusal(multipliers)
            if refusal is None:
                return facet, None
            refusals.append(refusal)
        return None, (refusals[0] if refusals else "the target has no unit")

    def admits_granularity(self, granularity: str) -> bool | None:
        answers = [f.admits_granularity(granularity) for f in self.facets]
        if any(answer is True for answer in answers):
            return True
        return False if answers and all(answer is False for answer in answers) else None

    def applies_stage(self, stage: str) -> bool | None:
        """Can ANY readout of this target apply ``stage``? Same tri-state as
        :meth:`admits_granularity`: ``True`` if one can, ``False`` only when every unit declared its
        readouts and none applies it, ``None`` when no unit declared any -- the unit nobody described
        may be the one that applies it."""
        answers = [f.applies_stage(stage) for f in self.facets]
        if any(answer is True for answer in answers):
            return True
        return False if answers and all(answer is False for answer in answers) else None


def reconcile(unit: Mapping[str, Any], facet: ReadoutFacet) -> list[dict[str, Any]]:
    """Compare a unit's declared ``scaling`` with what its readout was derived to hold.

    ``scaling_exceeds_readout`` is the defect this module exists for: the contract promises a
    granularity the hardware cannot hold, so everything generated from the contract (capsules,
    goldens, quantization) asks for arithmetic that can only happen on the host.
    """
    declared = unit.get("scaling")
    if not declared or str(declared) == "none":
        return []
    wanted = GRANULARITY_OF_SCALE_KIND.get(str(declared))
    base = {
        "unit": unit.get("name"),
        "declared": declared,
        "declared_granularity": wanted,
        "derived": (list(facet.scale_granularities) if facet.scale_granularities is not None else None),
    }
    if wanted is None:
        return [{**base, "kind": "unmapped_scaling", "why": f"scaling {declared!r} has no facet granularity"}]
    admitted = facet.admits_granularity(wanted)
    if admitted is None:
        return [
            {
                **base,
                "kind": "unaudited_scaling",
                "why": facet.unknown.get("scale_granularities", "granularity not derived"),
            }
        ]
    if admitted:
        return []
    return [
        {
            **base,
            "kind": "scaling_exceeds_readout",
            "rungs": list(facet.rungs_for("scale_granularities")),
            "why": "; ".join(e.observed for e in facet.evidence if e.field == "scale_granularities"),
        }
    ]


def epilogue_capability(facet: ReadoutFacet, *, name: str | None = None):
    """The quantized-epilogue capability this facet licenses, or a ``ValueError`` naming the gap.

    Every limit is a derived one. A facet with an unknown in any field the capability needs
    produces no capability: a planner handed invented limits would plan against them.
    """
    from merlin.perf.quantization_contract import QuantizedEpilogueCapability
    from merlin.runtime.commandbuffer import BIAS_STAGES

    needed = {
        "accumulator_dtype": facet.accumulator_dtype,
        "element_dtype": facet.element_dtype,
        "scale_granularities": facet.scale_granularities,
        "scale_dtype": facet.scale_dtype,
        "rounding": facet.rounding,
        "clamp": facet.clamp,
        "zero_point_carried": facet.zero_point_carried,
    }
    missing = sorted(key for key, value in needed.items() if value is None)
    if missing:
        raise ValueError(
            f"readout facet of {facet.target!r} cannot license an epilogue capability: "
            + ", ".join(f"{key} ({facet.unknown.get(key, 'not derived')})" for key in missing)
        )
    granularities = tuple(EPILOGUE_GRANULARITY[g] for g in facet.scale_granularities if g in EPILOGUE_GRANULARITY)
    if not granularities:
        raise ValueError(
            f"readout facet of {facet.target!r} holds scales at "
            f"{list(facet.scale_granularities)}, which the epilogue contract cannot "
            f"express"
        )
    if facet.scale_dtype != "f32" or facet.rounding != "half_even":
        raise ValueError(
            f"the epilogue contract evaluates an f32 scale rounded half to even; "
            f"{facet.target!r} derives scale {facet.scale_dtype} / {facet.rounding}"
        )
    if facet.zero_point_carried:
        raise ValueError(
            f"{facet.target!r} carries an output zero point in a register; its domain "
            f"is not derived, so no capability is licensed"
        )
    applied = {stage for readout in facet.readouts for stage in readout.get("applies") or ()}
    has_bias, has_relu = bool(applied & set(BIAS_STAGES)), "relu" in applied
    core = ("scale_f32", "round_to_nearest_even", "clamp")
    templates = tuple(
        (("bias_i32",) if bias else ()) + core + (("relu",) if relu else ())
        for bias in ((False, True) if has_bias else (False,))
        for relu in ((False, True) if has_relu else (False,))
    )
    return QuantizedEpilogueCapability(
        name=name or f"{facet.target}:{facet.unit or 'readout'}",
        accumulator_dtypes=(facet.accumulator_dtype,),
        output_dtypes=(facet.element_dtype,),
        scale_granularities=granularities,
        bias_domains=("none", "accumulator") if has_bias else ("none",),
        output_zero_points=(0,),
        roundings=("round_to_nearest_even",),
        saturations=(facet.clamp,),
        ordered_stage_templates=templates,
        activations=("none", "relu") if has_relu else ("none",),
    )
