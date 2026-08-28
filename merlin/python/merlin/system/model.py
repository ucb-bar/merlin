"""The compiled system: one host, N devices, and the LINK that joins each device to the host.

Why this exists. Everywhere else in the compiler a "target" is one flat name, and that name means
the DEVICE. The host is then not modelled at all: a routing gap silently *becomes* the host lane,
so "this op is illegal on the host" is inexpressible, and the host's vector length reaches codegen
as a ``-march`` substring pinned independently in three places. Meanwhile the device's own facts
(mesh, scratchpad, ISA, endpoint) live in a second vocabulary that shares nothing with the first --
five separate places answer "where does memory start", each derived differently.

A real configuration is a host plus one or more devices plus a way for them to talk. That is what
this models, and every field is DERIVED or ``None``; nothing is defaulted into existence. ``None``
means "not derivable here", which is a usable answer -- a caller may refuse, or may fall back to a
path that does not need it -- whereas a fabricated default is a wrong answer that looks right.

Vocabularies are closed sets validated fail-closed, exactly as ``families.ENDPOINT_KINDS`` is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- link vocabularies
#
# ``endpoint_kind`` answers FOUR independent questions with one token, which is why a hybrid
# device (a mesh reached by .insn whose operands arrive by DMA) cannot be described by it. The
# axes below separate them. Each is derived independently; each may be None.

#: HOW a command reaches the device.
#:   host_instruction -- the host executes an instruction the device decodes (RoCC .insn)
#:   mmio_store       -- the host stores to a control aperture
#:   command_buffer   -- a command buffer is handed to the device's own runtime
#:   device_native    -- the device fetches and decodes its own instruction stream
COMMAND_TRANSPORTS: tuple[str, ...] = ("host_instruction", "mmio_store", "command_buffer",
                                       "device_native")

#: WHERE the operands live, and who is responsible for putting them there.
#:   pointer_args      -- host allocates; the device is handed pointers and pulls (DMA)
#:   preload_at_base   -- operands are staged at agreed addresses before launch, results captured back
#:   host_materialized -- the host writes operand values into the device's own storage inline
OPERAND_PLACEMENTS: tuple[str, ...] = ("pointer_args", "preload_at_base", "host_materialized")

#: How a host address becomes a device address.
#:   identity       -- one address space
#:   offset         -- device address = host address - ``offset`` (see Link.address_offset)
#:   separate_space -- disjoint; an explicit transfer is required
ADDRESS_TRANSLATIONS: tuple[str, ...] = ("identity", "offset", "separate_space")


def _checked(value: str | None, allowed: tuple[str, ...], what: str) -> str | None:
    """Fail closed on an unknown token; ``None`` (not derivable) is always legal."""
    if value is not None and value not in allowed:
        raise ValueError(f"unknown {what}: {value!r} (known: {', '.join(allowed)})")
    return value


@dataclass(frozen=True)
class Link:
    """How one device is reached from the host. Every field derived, or None."""

    command_transport: str | None = None
    operand_placement: str | None = None
    address_translation: str | None = None
    #: The translation constant when ``address_translation == "offset"``; None otherwise.
    address_offset: int | None = None
    #: Where the device's memory window starts, when that is derivable.
    device_dram_base: int | None = None
    #: The artifact the compiler emits for this device (``runner_config.ENDPOINT_ARTIFACT``).
    emitted_artifact: str | None = None
    #: Per-field provenance: {field_name: "how this was derived"}. A field present here but None in
    #: the dataclass is a field we LOOKED for and could not ground -- distinct from never looking.
    evidence: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _checked(self.command_transport, COMMAND_TRANSPORTS, "command_transport")
        _checked(self.operand_placement, OPERAND_PLACEMENTS, "operand_placement")
        _checked(self.address_translation, ADDRESS_TRANSLATIONS, "address_translation")
        if self.address_translation == "offset" and self.address_offset is None:
            raise ValueError("address_translation='offset' needs an address_offset")
        if self.address_offset is not None and self.address_translation != "offset":
            raise ValueError("address_offset is only meaningful with address_translation='offset'")

    def unknowns(self) -> tuple[str, ...]:
        """The axes that could not be derived. A caller that needs one must refuse, not assume."""
        return tuple(n for n in ("command_transport", "operand_placement", "address_translation",
                                 "emitted_artifact") if getattr(self, n) is None)

    def to_device_address(self, host_addr: int) -> int | None:
        """Translate a host address, or None when the translation is not derivable."""
        if self.address_translation == "identity":
            return host_addr
        if self.address_translation == "offset":
            return host_addr - int(self.address_offset or 0)
        return None            # separate_space or underivable: the caller must move the bytes


@dataclass(frozen=True)
class Host:
    """The core(s) running the model's non-device work. Derived from the board descriptor."""

    name: str
    harts: int | None = None
    #: Harts that can execute VECTOR code, when that differs from ``harts``.
    vector_harts: int | None = None
    vector_hart_ids: tuple[int, ...] | None = None
    #: Hardware vector length in BITS. None = unknown; codegen must then not pin a ``zvl``.
    vlen: int | None = None
    dram_base: int | None = None
    dram_bytes: int | None = None
    console: str | None = None
    evidence: dict[str, str] = field(default_factory=dict)

    def vector_capable(self) -> bool | None:
        """Whether any hart can run vector code -- ``None`` when the board does not say.

        Tri-state on purpose. ``vlen=None`` in a board descriptor means "unknown, assume the V
        minimum", NOT "no vector unit", and a board may carry vectors while declaring neither field.
        Collapsing unknown to False would quietly route vectorizable work to the scalar lane and look
        like a placement decision rather than a missing fact.
        """
        if self.vector_harts is not None:
            return self.vector_harts > 0
        if self.vector_hart_ids is not None:
            return len(self.vector_hart_ids) > 0
        return None            # nothing declared either way: not derivable, say so


@dataclass(frozen=True)
class Device:
    """One accelerator, and how it is reached."""

    name: str
    kind: str | None = None            # a compute_units KIND (systolic/simt/spatial/vector/scalar)
    endpoint_kind: str | None = None   # the legacy token, kept: the Link is derived FROM it + facts
    link: Link = field(default_factory=Link)
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class System:
    """One host and the devices attached to it.

    ``devices`` is a tuple because a configuration with two accelerators is the normal case, not an
    exotic one -- two cores with different accelerators, or one core with a mesh and a vector engine.
    Nothing here assumes a length of one; code that can only handle one device must say so itself.
    """

    host: Host | None = None
    devices: tuple[Device, ...] = ()

    def device(self, name: str) -> Device | None:
        return next((d for d in self.devices if d.name == name), None)

    @property
    def is_single_device(self) -> bool:
        return len(self.devices) == 1

    def unknowns(self) -> dict[str, tuple[str, ...]]:
        """Per-device link axes that could not be derived, for a caller that must fail closed."""
        return {d.name: d.link.unknowns() for d in self.devices if d.link.unknowns()}
