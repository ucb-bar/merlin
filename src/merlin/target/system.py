"""Host+device system model: what we compile for, and how the pieces are reached."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.record_formats import safe_load_text

# ---- model -------------------------------------------------------------------------------------
# The compiled system: one host, N devices, and the LINK that joins each device to the host.
#
# Why this exists. Everywhere else in the compiler a "target" is one flat name, and that name means
# the DEVICE. The host is then not modelled at all: a routing gap silently *becomes* the host lane,
# so "this op is illegal on the host" is inexpressible, and the host's vector length reaches codegen
# as a ``-march`` substring pinned independently in three places. Meanwhile the device's own facts
# (mesh, scratchpad, ISA, endpoint) live in a second vocabulary that shares nothing with the first --
# five separate places answer "where does memory start", each derived differently.
#
# A real configuration is a host plus one or more devices plus a way for them to talk. That is what
# this models, and every field is DERIVED or ``None``; nothing is defaulted into existence. ``None``
# means "not derivable here", which is a usable answer -- a caller may refuse, or may fall back to a
# path that does not need it -- whereas a fabricated default is a wrong answer that looks right.
#
# Vocabularies are closed sets validated fail-closed, exactly as ``families.ENDPOINT_KINDS`` is.

# ---- model -------------------------------------------------------------------------------------

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
COMMAND_TRANSPORTS: tuple[str, ...] = ("host_instruction", "mmio_store", "command_buffer", "device_native")

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
        return tuple(
            n
            for n in ("command_transport", "operand_placement", "address_translation", "emitted_artifact")
            if getattr(self, n) is None
        )

    def to_device_address(self, host_addr: int) -> int | None:
        """Translate a host address, or None when the translation is not derivable."""
        if self.address_translation == "identity":
            return host_addr
        if self.address_translation == "offset":
            return host_addr - int(self.address_offset or 0)
        return None  # separate_space or underivable: the caller must move the bytes


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
        return None  # nothing declared either way: not derivable, say so


@dataclass(frozen=True)
class Device:
    """One accelerator, and how it is reached."""

    name: str
    kind: str | None = None  # a compute_units KIND (systolic/simt/spatial/vector/scalar)
    endpoint_kind: str | None = None  # the legacy token, kept: the Link is derived FROM it + facts
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


# ---- derive ------------------------------------------------------------------------------------
# Build a :class:`~merlin.target.system.System` from what the repo already derives.
#
# Nothing here discovers a new fact. Every value comes from an existing deriver -- the board
# descriptor, the capability manifest, the RTL facts -- and the contribution is putting them in one
# object with one vocabulary, so a caller can ask "how do I reach this device" without knowing which
# of five places happens to answer that for this particular target.
#
# Two rules, both load-bearing:
#
# * **Derived or None.** A fact we cannot ground is ``None`` with a note in ``evidence`` saying what we
#   looked at. It is never defaulted, because a default here is indistinguishable from a measurement
#   at the call site and silently produces a wrong address or a wrong transport.
# * **No target names.** Everything is keyed on derived properties (endpoint kind, decoder facts,
#   declared interfaces), never on which target it happens to be.

# ------------------------------------------------------------------ host


def host_from_board(board_name: str, *, board_catalog: Path | None = None, **overrides) -> Host:
    """Derive the Host from a board descriptor (``runtime.boards``).

    The board is where the host facts already live -- harts, which of them carry a vector unit, the
    hardware VLEN, the DRAM window. They are DECLARED there (the docstrings explain why: nothing
    readable states the vector-hart mapping), but they are declared in exactly one place, which is
    what makes them usable as the single source for codegen's ``zvl`` pin instead of the three
    independent pins that exist today.
    """
    from merlin.runtime.boards import catalog as _b

    if board_catalog is None:
        brd = _b.board(board_name, **overrides)
    else:
        selected = _b.load_boards(board_catalog)
        if board_name not in selected:
            raise _b.BoardRegistryError(f"board {board_name!r} is absent from {board_catalog}")
        brd = selected[board_name]
        if overrides:
            from dataclasses import replace

            brd = replace(brd, **overrides)
    return Host(
        name=getattr(brd, "name", board_name),
        harts=getattr(brd, "harts", None),
        vector_harts=getattr(brd, "vector_harts", None),
        vector_hart_ids=getattr(brd, "vector_hart_ids", None),
        vlen=getattr(brd, "vlen", None),
        dram_base=getattr(brd, "dram_base", None),
        dram_bytes=getattr(brd, "dram_bytes", None),
        console=getattr(brd, "console", None),
        evidence={"source": f"runtime.boards.board({board_name!r})"},
    )


# ------------------------------------------------------------------ link

#: What each endpoint kind implies about the transport. This is the ONLY thing ``endpoint_kind``
#: genuinely determines; the other axes are derived separately below, which is the point of the
#: decomposition -- two targets sharing an endpoint kind can differ in where their operands live.
_TRANSPORT_FOR_ENDPOINT = {
    "inline_asm_insn": "host_instruction",  # the host executes an instruction the device decodes
    "command_buffer": "command_buffer",  # no command ISA at all; a buffer is handed over
    "external_backend": "device_native",  # the device fetches and decodes its own stream
    "upstream_target": None,  # not a separate device: it lowers through stock LLVM
}


def _facts(target: str) -> dict[str, Any]:
    """The target's RTL facts body, or ``{}`` when the extractor never grounded them."""
    try:
        from merlin.target.rtl import facts as _f

        return _f.body_if_present(target)
    except Exception:  # noqa: BLE001 -- absent facts are a real answer, not an error
        return {}


def _interfaces(body: dict) -> set[str]:
    return {str(i.get("name")) for i in (body.get("interfaces") or []) if i.get("name")}


def link_for(target: str, endpoint_kind: str | None) -> Link:
    """Derive how ``target`` is reached, from its endpoint kind plus its own RTL facts."""
    body = _facts(target)
    ifaces = _interfaces(body)
    ev: dict[str, str] = {}

    transport = _TRANSPORT_FOR_ENDPOINT.get(endpoint_kind or "", None)
    ev["command_transport"] = (
        f"endpoint_kind={endpoint_kind!r}"
        if transport
        else f"endpoint_kind={endpoint_kind!r} implies no distinct transport"
    )

    # Operand placement. A DMA/TLB interface means the device pulls from host memory given pointers;
    # this is the `interfaces` fact the extractor already derives and that nothing has ever read.
    placement = None
    if ifaces & {"dma_tlb"}:
        placement, ev["operand_placement"] = "pointer_args", "facts.interfaces contains dma_tlb"
    elif transport == "command_buffer":
        placement, ev["operand_placement"] = "preload_at_base", "command-buffer tensors are staged by the runner"
    elif transport == "device_native":
        placement, ev["operand_placement"] = "preload_at_base", "device-native stream: operands staged before launch"
    else:
        ev["operand_placement"] = f"no dma interface among {sorted(ifaces) or 'none'}; not derivable"

    # Address space. Only claim identity when there is positive evidence of a shared window.
    translation, offset, dram_base = None, None, None
    try:
        from merlin.target.isa.machine_model import dram_base_for

        db = dram_base_for(target)
        if db:
            dram_base, ev["device_dram_base"] = int(db), f"dram_facts.dram_base_for({target!r})"
    except Exception:  # noqa: BLE001
        ev["device_dram_base"] = "dram_facts unavailable"
    if placement == "pointer_args":
        # The device walks host page tables (that is what a TLB interface is for), so an address
        # handed to it is a host address.
        translation, ev["address_translation"] = "identity", "device translates host addresses (dma_tlb)"
    else:
        ev["address_translation"] = "no evidence of a shared address space; not derivable"

    artifact = None
    try:
        from merlin.target.registry.identity import ENDPOINT_ARTIFACT

        artifact = ENDPOINT_ARTIFACT.get(endpoint_kind or "")
        ev["emitted_artifact"] = f"runner_config.ENDPOINT_ARTIFACT[{endpoint_kind!r}]"
    except Exception:  # noqa: BLE001
        ev["emitted_artifact"] = "runner_config unavailable"

    return Link(
        command_transport=transport,
        operand_placement=placement,
        address_translation=translation,
        address_offset=offset,
        device_dram_base=dram_base,
        emitted_artifact=artifact,
        evidence=ev,
    )


# ------------------------------------------------------------------ device / system


def device_for(target: str) -> Device:
    """Derive one Device from its capability manifest (kind + endpoint), plus its Link."""
    kind = endpoint = None
    ev: dict[str, str] = {}
    try:
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        man = load_capability_manifest(target)
        kind, endpoint = getattr(man, "kind", None), getattr(man, "endpoint_kind", None)
        ev["source"] = f"capability manifest for {target!r}"
    except Exception as exc:  # noqa: BLE001 -- an unresolvable target yields an empty Device
        ev["source"] = f"no capability manifest ({type(exc).__name__})"
    return Device(name=target, kind=kind, endpoint_kind=endpoint, link=link_for(target, endpoint), evidence=ev)


def system_for(
    target: str | None = None,
    *,
    targets=None,
    board: str | None = None,
    board_catalog: Path | None = None,
    **board_overrides,
) -> System:
    """The System a compile runs on: one host, and the device(s) named.

    ``target``/``targets`` keep today's single-target callers working unchanged -- a single name
    yields a one-device System -- while the shape is already plural, so nothing downstream has to
    change when a second device appears.
    """
    names = list(targets) if targets is not None else ([target] if target else [])
    host = host_from_board(board, board_catalog=board_catalog, **board_overrides) if board else None
    return System(host=host, devices=tuple(device_for(n) for n in names))


def host_board_for_experiment(target: str) -> tuple[str | None, str]:
    """``(board name, why)`` for ``target``'s experiment descriptor.

    Separate from :func:`system_for_experiment` so a caller can ask what is DECLARED without building a
    System, and so the reason travels with the answer. ``None`` means the descriptor names no board --
    reported, never defaulted.
    """
    try:
        # Through `corpora`, not by building the path here: it is the one module allowed to know the
        # `experiments/` layout, and -- load-bearing -- it honors MERLIN_TARGET_EXPERIMENT. The
        # convention path built by hand ignored that override, so a caller pointed at an out-of-tree
        # descriptor got the board from the in-tree one instead, silently.
        from merlin.target.registry.descriptor_loading import descriptor_path, load_target_experiment

        desc = descriptor_path(target)
        if not desc.is_file():
            return None, f"no experiment descriptor for {target!r} at {desc}"
        board = load_target_experiment(desc).host_board
    except Exception as exc:  # noqa: BLE001 -- an unreadable descriptor is not a board
        return None, f"could not read {target}'s experiment descriptor: {type(exc).__name__}: {exc}"
    if not board:
        return None, (
            f"{target}'s descriptor declares no `host: {{board: ...}}`, so the host this "
            f"target's lane compiles for is unknown"
        )
    return board, f"declared by {target}'s experiment descriptor"


def system_for_experiment(target: str, **board_overrides) -> tuple[System, str]:
    """``(System, why)`` for ``target``, with the host taken from its experiment descriptor.

    The plain :func:`system_for` needs the board passed in, and no caller had one -- so every System
    built for a real target carried ``host=None`` and placement silently became scalar-only. This is the
    seam that gives it a host, and it FAILS OPEN WITH A REASON rather than closed: a target with no
    declared board still yields a usable one-device System, but the caller is handed the sentence
    explaining why its host is absent, so "we could not tell" cannot be mistaken for "there is no
    vector host".
    """
    board, why = host_board_for_experiment(target)
    if board is None:
        return system_for(target), why
    # A DECLARED board must be a KNOWN board. ``runtime.boards.board`` deliberately falls back to
    # conservative defaults for an unnamed board -- that is right for a caller deliberately trying a new
    # one with explicit overrides, and wrong here: a typo in a descriptor would yield a plausible Host
    # (harts=2, vlen=None) and every placement would be measured against hardware nobody has. Strict at
    # this seam only; the general helper keeps its documented behaviour.
    try:
        from merlin.runtime.boards.catalog import BOARDS, load_boards
        from merlin.target.registry.descriptor_loading import descriptor_path, load_target_experiment

        catalog = load_target_experiment(descriptor_path(target)).selected_board_catalog()
        known = set(load_boards(catalog)) if catalog is not None else set(BOARDS)
    except Exception as exc:  # noqa: BLE001 -- an unreadable registry is not a board
        return system_for(target), f"could not read the board registry: {type(exc).__name__}: {exc}"
    if board not in known:
        return system_for(target), (
            f"{target!r} declares board {board!r}, which did not resolve: it is not in "
            f"merlin.runtime.boards.catalog (known: {', '.join(sorted(known))}). Refusing the conservative "
            f"fallback here, because a defaulted host would be measured as if it were real hardware"
        )
    try:
        return system_for(target, board=board, board_catalog=catalog, **board_overrides), why
    except Exception as exc:  # noqa: BLE001 -- a bad board name is not a host
        return system_for(target), (
            f"{target!r} declares board {board!r}, which did not resolve: {type(exc).__name__}: {exc}"
        )


# ---- placement ---------------------------------------------------------------------------------
# Device offload legality and the selected system topology.

# ---- offload -----------------------------------------------------------------------------------
# Which contractions a DEVICE could take — derived from that device, not from constants.
#
# The existing offload path answers this with two module-level literals: a dtype triple
# ``("i8","i8","i32")`` and an op table ``{"linalg.matmul": 2, "linalg.batch_matmul": 3}``. They are
# correct for the one device that path was built for, and they are *its* facts: the triple is literally
# that device's first declared ``accumulate`` rule, and the ranks are what its ``contraction`` semantic
# capability declares. Written as literals they belong to nobody, so a second device either inherits
# another device's datapath or needs a second copy of the pass.
#
# Here the same question is asked of a named device and answered from what that device already
# declares. Adding a device adds no code.
#
# **Fail closed.** A device whose datapath cannot be derived offloads NOTHING. The alternative --
# assuming a triple -- produces a rewrite that compiles and silently computes in the wrong precision,
# which is far worse than declining. ``why_not`` says which of the three gates rejected a contraction
# so a caller can report it rather than guess.


def _mlir(token: str) -> str | None:
    """A dtype token in MLIR spelling, or None when the registry does not know it (never guessed)."""
    try:
        from merlin.common.target_data import machine_bits
        from merlin.development.targetgen.corpus.binding import dtype_info

        # Plain signed integer widths are type spellings, not a closed corpus
        # vocabulary or a claim that a device executes this arithmetic.
        bits = machine_bits(str(token))
        if bits and str(token).startswith(("int", "i")):
            return f"i{bits}"
        return dtype_info(str(token))[1]
    except Exception:  # noqa: BLE001 -- an unknown token is a real answer: skip that rule
        return None


def _units(device_name: str) -> list[dict]:
    """The device's compute units as declared/derived in its capability manifest."""
    try:
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        man = load_capability_manifest(device_name)
        return list((man.contract or {}).get("compute_units") or [])
    except Exception:  # noqa: BLE001
        return []


def _triples_from_facts(device_name: str) -> tuple[tuple[str, str, str], ...]:
    """The datapath triple read straight off the RTL facts' ``datapaths`` block.

    The accumulate matrix in a manifest is a PROJECTION of these facts, and a hand-written contract
    may simply not carry one -- measured: one target's curated contract declares ``accumulate: []``
    while its facts plainly record ``input i8`` and ``accumulator i32``. Reading the facts directly
    means a device is not treated as having no datapath merely because nobody transcribed one.

    Operand and weight share the ``input`` datapath: a contraction datapath feeds both sides of the
    multiply from the same storage class. Where that is untrue the manifest's accumulate matrix says
    so and takes precedence (it is consulted first).
    """
    try:
        from merlin.target.rtl import facts as _f

        body = _f.body_if_present(device_name)
    except Exception:  # noqa: BLE001 -- ungrounded facts are a real answer
        return ()
    by_name = {
        str(d.get("name")): str(d.get("dtype"))
        for d in (body.get("datapaths") or ())
        if d.get("name") and d.get("dtype")
    }
    inp, acc = by_name.get("input"), by_name.get("accumulator")
    if not inp or not acc:
        return ()
    lhs, a = _mlir(inp), _mlir(acc)
    return ((lhs, lhs, a),) if lhs and a else ()


def device_dtype_triples(device_name: str) -> tuple[tuple[str, str, str], ...]:
    """``(lhs, rhs, acc)`` in MLIR spelling for every datapath this device declares.

    Two sources, in order. The unit's ``accumulate`` matrix is the richer one -- it can express
    several datapaths and mixed operand/weight formats -- so it wins where present. Where a contract
    carries none, the RTL facts' own ``datapaths`` block still grounds one, and using it is what keeps
    "this contract never transcribed an accumulate matrix" from reading as "this device has no
    datapath". A rule naming a dtype the registry cannot spell is skipped rather than approximated:
    a triple is a precision claim, and a wrong one is silent.
    """
    out: list[tuple[str, str, str]] = []
    for unit in _units(device_name):
        for rule in unit.get("accumulate") or ():
            lhs, rhs, acc = (_mlir(rule.get("in") or ""), _mlir(rule.get("weight") or ""), _mlir(rule.get("acc") or ""))
            if lhs and rhs and acc and (lhs, rhs, acc) not in out:
                out.append((lhs, rhs, acc))
    if out:
        return tuple(out)
    return _triples_from_facts(device_name)


def device_contraction_ranks(device_name: str) -> tuple[int, ...] | None:
    """Legal output ranks for a contraction on this device; ``None`` means unconstrained.

    ``None`` and ``()`` mean the same thing in the capability model (unconstrained) and that reading
    is kept here: a device that never narrowed its ranks has not thereby forbidden every rank.
    """
    ranks: set[int] = set()
    saw = False
    for unit in _units(device_name):
        for cap in unit.get("semantic_capabilities") or ():
            if str(cap.get("family")) != "contraction":
                continue
            saw = True
            for r in cap.get("ranks") or ():
                ranks.add(int(r))
    return tuple(sorted(ranks)) if (saw and ranks) else None


def why_not(shape, *, triples, ranks) -> str | None:
    """Why this contraction is not offloadable here, or None when it is. Reported, never inferred."""
    if not triples:
        return "device declares no derivable datapath (no accumulate rule this registry can spell)"
    if tuple(shape.dtypes) not in triples:
        return f"dtypes {tuple(shape.dtypes)} not among the device's datapaths {sorted(triples)}"
    if len(shape.reduction) != 1:
        return f"{len(shape.reduction)} reduction dims; a contraction datapath takes exactly one"
    if ranks is not None and len(shape.parallel) not in ranks:
        return f"output rank {len(shape.parallel)} not among the device's legal ranks {list(ranks)}"
    return None


def offloadable_contractions(module, device_name: str, *, require_zero_init: bool = True) -> list[tuple[Any, Any]]:
    """``[(op, shape)]`` for every contraction ``device_name`` COULD take. No decision is made.

    "Could" means legal on this device's declared datapath and shape envelope -- never profitable.
    Whether a legal contraction is worth moving is a placement decision, and keeping the two apart is
    why this enumerates instead of rewriting.

    ``require_zero_init`` keeps the correctness condition the offload ABI depends on: ``linalg.matmul``
    computes ``C_init + A@B`` while a device kernel that OVERWRITES its output computes ``A@B``, and
    those agree only when ``C_init`` is zero. A contraction accumulating onto a live init is declined
    rather than silently losing the addend.
    """
    triples = device_dtype_triples(device_name)
    if not triples:
        return []  # fail closed: an underivable datapath offloads nothing

    from merlin.compiler.planning.source_coverage import contractions_for_dtypes

    return contractions_for_dtypes(module, device_name, triples, require_zero_init=require_zero_init)


def catalog_source_contractions(
    module, device_name: str, source_dtypes: tuple[str, str, str], *, require_zero_init: bool = True
) -> list[tuple[Any, Any]]:
    """Compatibility entrypoint for the shared planning query."""
    from merlin.compiler.planning.source_coverage import catalog_source_contractions as catalog

    return catalog(module, device_name, source_dtypes, require_zero_init=require_zero_init)


# ---- topology ----------------------------------------------------------------------------------
# Explicit selected execution resources and synchronous shared memory access.
#
# The original caller selects this contract; provider proposals cannot amend it.
# No relationship is inferred from a backend name or from a device's presence.

SCHEMA = "merlin.execution_topology.v1"


def _name(value, field):
    if type(value) is not str or not value.strip():
        raise ValueError(f"topology requires an explicit {field}")


def _names(value, field):
    if type(value) is not tuple or len(set(value)) != len(value):
        raise ValueError(f"topology {field} must be an immutable unique tuple")
    for item in value:
        _name(item, field)


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    features: tuple[str, ...]
    shared_group: str

    def __post_init__(self):
        for field in ("id", "kind", "shared_group"):
            _name(getattr(self, field), field)
        _names(self.features, "features")


@dataclass(frozen=True)
class MemoryDomain:
    """Available selected byte budget after external/runtime reservations."""

    id: str
    capacity_bytes: int
    alignment_bytes: int

    def __post_init__(self):
        _name(self.id, "memory domain")
        if type(self.capacity_bytes) is not int or self.capacity_bytes <= 0:
            raise ValueError("memory domain requires an explicit positive available byte capacity")
        alignment = self.alignment_bytes
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("memory domain alignment must be an explicit positive power of two")


@dataclass(frozen=True)
class MemoryAccess:
    resource: str
    domain: str
    read: bool
    write: bool
    addressability: str
    visibility: str
    completion: str

    def __post_init__(self):
        for field in ("resource", "domain", "addressability", "visibility", "completion"):
            _name(getattr(self, field), field)
        if type(self.read) is not bool or type(self.write) is not bool:
            raise ValueError("memory access read/write permissions must be explicit booleans")
        if self.addressability not in ("shared", "separate", "unknown"):
            raise ValueError("unknown memory addressability vocabulary")
        if self.visibility not in ("coherent", "explicit", "unknown"):
            raise ValueError("unknown memory visibility vocabulary")
        if self.completion not in ("synchronous", "explicit", "unknown"):
            raise ValueError("unknown memory completion vocabulary")


def _document(resources, memories, access):
    return json.dumps(
        {
            "resources": [asdict(r) for r in resources],
            "memory_domains": [asdict(m) for m in memories],
            "access": [asdict(a) for a in access],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse(payload):
    document = safe_load_text(payload.decode("utf-8"))
    if type(document) is not dict or set(document) != {"schema", "resources", "memory_domains", "access"}:
        raise ValueError("execution topology requires its exact schema and resource/memory/access fields")
    if document["schema"] != SCHEMA:
        raise ValueError("unsupported execution topology schema")
    for field in ("resources", "memory_domains", "access"):
        if type(document[field]) is not list or len(document[field]) > 4096:
            raise ValueError(f"topology {field} must be a bounded list")
    resources = []
    for row in document["resources"]:
        if type(row) is not dict or set(row) != {"id", "kind", "features", "shared_group"}:
            raise ValueError("resource requires explicit identity, kind, features and shared group")
        if type(row["features"]) is not list:
            raise ValueError("resource features must be an explicit list")
        resources.append(Resource(**{**row, "features": tuple(row["features"])}))
    try:
        memories = tuple(MemoryDomain(**row) for row in document["memory_domains"])
        access = tuple(MemoryAccess(**row) for row in document["access"])
    except TypeError as error:
        raise ValueError("memory domains/access require exact explicitly typed fields") from error
    resources = tuple(resources)
    if not resources or not memories:
        raise ValueError("execution topology requires resources and memory domains")
    if len({r.id for r in resources}) != len(resources) or len({m.id for m in memories}) != len(memories):
        raise ValueError("execution topology repeats a resource or memory identity")
    if len({(a.resource, a.domain) for a in access}) != len(access):
        raise ValueError("execution topology repeats a resource/domain access edge")
    for edge in access:
        if edge.resource not in {r.id for r in resources} or edge.domain not in {m.id for m in memories}:
            raise ValueError("memory access names an undeclared resource or domain")
    return resources, memories, access


def _read(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("execution topology requires an original unlinked regular file")
    if path.stat().st_size > 1 << 20:
        raise ValueError("execution topology exceeds its byte budget")
    payload = path.read_bytes()
    if len(payload) > 1 << 20:
        raise ValueError("execution topology exceeds its byte budget")
    return payload


@dataclass(frozen=True)
class SelectedTopology:
    path: Path
    sha256: str
    resources: tuple[Resource, ...]
    memory_domains: tuple[MemoryDomain, ...]
    access: tuple[MemoryAccess, ...]
    _record: str

    def revalidate(self):
        payload = _read(self.path)
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError("selected execution topology changed; replan")
        parsed = _parse(payload)
        if (
            _document(*parsed) != self._record
            or _document(self.resources, self.memory_domains, self.access) != self._record
        ):
            raise ValueError("selected execution topology bindings changed; replan")

    def resource(self, identity: str) -> Resource:
        self.revalidate()
        found = next((row for row in self.resources if row.id == identity), None)
        if found is None:
            raise ValueError(f"undeclared execution resource {identity!r}")
        return found

    def memory(self, identity: str) -> MemoryDomain:
        self.revalidate()
        found = next((row for row in self.memory_domains if row.id == identity), None)
        if found is None:
            raise ValueError(f"undeclared memory domain {identity!r}")
        return found

    def require_access(self, resource: str, domain: str, *, read: bool, write: bool):
        """Admit this slice's synchronous coherent shared-address operation."""
        self.resource(resource)
        self.memory(domain)
        edge = next((a for a in self.access if (a.resource, a.domain) == (resource, domain)), None)
        if edge is None or read and not edge.read or write and not edge.write:
            raise ValueError("resource has no declared access to its boundary memory")
        if (edge.addressability, edge.visibility, edge.completion) != ("shared", "coherent", "synchronous"):
            raise ValueError("selected execution requires an unimplemented transfer, visibility or completion action")
        return edge


def select_topology(path: str | Path) -> SelectedTopology:
    """Bind an original caller-selected hardware contract, never a provider claim."""
    path = Path(path).absolute()
    payload = _read(path)
    resources, memories, access = _parse(payload)
    result = SelectedTopology(
        path, hashlib.sha256(payload).hexdigest(), resources, memories, access, _document(resources, memories, access)
    )
    result.revalidate()
    return result


__all__ = [
    "catalog_source_contractions",
    "device_contraction_ranks",
    "device_dtype_triples",
    "offloadable_contractions",
    "why_not",
]
