"""Build a :class:`~merlin.system.model.System` from what the repo already derives.

Nothing here discovers a new fact. Every value comes from an existing deriver -- the board
descriptor, the capability manifest, the RTL facts -- and the contribution is putting them in one
object with one vocabulary, so a caller can ask "how do I reach this device" without knowing which
of five places happens to answer that for this particular target.

Two rules, both load-bearing:

* **Derived or None.** A fact we cannot ground is ``None`` with a note in ``evidence`` saying what we
  looked at. It is never defaulted, because a default here is indistinguishable from a measurement
  at the call site and silently produces a wrong address or a wrong transport.
* **No target names.** Everything is keyed on derived properties (endpoint kind, decoder facts,
  declared interfaces), never on which target it happens to be.
"""
from __future__ import annotations

from typing import Any

from .model import Device, Host, Link, System

# ------------------------------------------------------------------ host

def host_from_board(board_name: str, **overrides) -> Host:
    """Derive the Host from a board descriptor (``runtime.boards``).

    The board is where the host facts already live -- harts, which of them carry a vector unit, the
    hardware VLEN, the DRAM window. They are DECLARED there (the docstrings explain why: nothing
    readable states the vector-hart mapping), but they are declared in exactly one place, which is
    what makes them usable as the single source for codegen's ``zvl`` pin instead of the three
    independent pins that exist today.
    """
    from merlin.runtime import boards as _b

    brd = _b.board(board_name, **overrides)
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
    "inline_asm_insn": "host_instruction",   # the host executes an instruction the device decodes
    "command_buffer": "command_buffer",      # no command ISA at all; a buffer is handed over
    "external_backend": "device_native",     # the device fetches and decodes its own stream
    "upstream_target": None,                 # not a separate device: it lowers through stock LLVM
}


def _facts(target: str) -> dict[str, Any]:
    """The target's RTL facts body, or ``{}`` when the extractor never grounded them."""
    try:
        from merlin.targetgen.rtl import facts as _f
        return (_f.load_facts(target) or {}).get("facts") or {}
    except Exception:            # noqa: BLE001 -- absent facts are a real answer, not an error
        return {}


def _interfaces(body: dict) -> set[str]:
    return {str(i.get("name")) for i in (body.get("interfaces") or []) if i.get("name")}


def link_for(target: str, endpoint_kind: str | None) -> Link:
    """Derive how ``target`` is reached, from its endpoint kind plus its own RTL facts."""
    body = _facts(target)
    ifaces = _interfaces(body)
    ev: dict[str, str] = {}

    transport = _TRANSPORT_FOR_ENDPOINT.get(endpoint_kind or "", None)
    ev["command_transport"] = (f"endpoint_kind={endpoint_kind!r}" if transport
                               else f"endpoint_kind={endpoint_kind!r} implies no distinct transport")

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
        from merlin.targetgen.dram_facts import dram_base_for
        db = dram_base_for(target)
        if db:
            dram_base, ev["device_dram_base"] = int(db), f"dram_facts.dram_base_for({target!r})"
    except Exception:            # noqa: BLE001
        ev["device_dram_base"] = "dram_facts unavailable"
    if placement == "pointer_args":
        # The device walks host page tables (that is what a TLB interface is for), so an address
        # handed to it is a host address.
        translation, ev["address_translation"] = "identity", "device translates host addresses (dma_tlb)"
    else:
        ev["address_translation"] = "no evidence of a shared address space; not derivable"

    artifact = None
    try:
        from merlin.targetgen.runner_config import ENDPOINT_ARTIFACT
        artifact = ENDPOINT_ARTIFACT.get(endpoint_kind or "")
        ev["emitted_artifact"] = f"runner_config.ENDPOINT_ARTIFACT[{endpoint_kind!r}]"
    except Exception:            # noqa: BLE001
        ev["emitted_artifact"] = "runner_config unavailable"

    return Link(command_transport=transport, operand_placement=placement,
                address_translation=translation, address_offset=offset,
                device_dram_base=dram_base, emitted_artifact=artifact, evidence=ev)


# ------------------------------------------------------------------ device / system

def device_for(target: str) -> Device:
    """Derive one Device from its capability manifest (kind + endpoint), plus its Link."""
    kind = endpoint = None
    ev: dict[str, str] = {}
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest
        man = load_capability_manifest(target)
        kind, endpoint = getattr(man, "kind", None), getattr(man, "endpoint_kind", None)
        ev["source"] = f"capability manifest for {target!r}"
    except Exception as exc:     # noqa: BLE001 -- an unresolvable target yields an empty Device
        ev["source"] = f"no capability manifest ({type(exc).__name__})"
    return Device(name=target, kind=kind, endpoint_kind=endpoint,
                  link=link_for(target, endpoint), evidence=ev)


def system_for(target: str | None = None, *, targets=None, board: str | None = None,
               **board_overrides) -> System:
    """The System a compile runs on: one host, and the device(s) named.

    ``target``/``targets`` keep today's single-target callers working unchanged -- a single name
    yields a one-device System -- while the shape is already plural, so nothing downstream has to
    change when a second device appears.
    """
    names = list(targets) if targets is not None else ([target] if target else [])
    host = host_from_board(board, **board_overrides) if board else None
    return System(host=host, devices=tuple(device_for(n) for n in names))


def host_board_for_experiment(target: str) -> tuple[str | None, str]:
    """``(board name, why)`` for ``target``'s experiment descriptor.

    Separate from :func:`system_for_experiment` so a caller can ask what is DECLARED without building a
    System, and so the reason travels with the answer. ``None`` means the descriptor names no board --
    reported, never defaulted.
    """
    try:
        from merlin.common.paths import merlin_dir
        from merlin.targetgen.target_experiment import load_target_experiment
        desc = merlin_dir() / "experiments/capsule_bench/targets" / target / "target_experiment.yaml"
        if not desc.is_file():
            return None, f"no experiment descriptor for {target!r} at {desc}"
        board = load_target_experiment(desc).host_board
    except Exception as exc:                       # noqa: BLE001 -- an unreadable descriptor is not a board
        return None, f"could not read {target!r}'s experiment descriptor: {type(exc).__name__}: {exc}"
    if not board:
        return None, (f"{target!r}'s descriptor declares no `host: {{board: ...}}`, so the host this "
                      f"target's lane compiles for is unknown")
    return board, f"declared by {target!r}'s experiment descriptor"


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
        from merlin.runtime.boards import BOARDS
        known = set(BOARDS)
    except Exception as exc:                       # noqa: BLE001 -- an unreadable registry is not a board
        return system_for(target), f"could not read the board registry: {type(exc).__name__}: {exc}"
    if board not in known:
        return system_for(target), (
            f"{target!r} declares board {board!r}, which did not resolve: it is not in "
            f"merlin.runtime.boards (known: {', '.join(sorted(known))}). Refusing the conservative "
            f"fallback here, because a defaulted host would be measured as if it were real hardware")
    try:
        return system_for(target, board=board, **board_overrides), why
    except Exception as exc:                       # noqa: BLE001 -- a bad board name is not a host
        return system_for(target), (f"{target!r} declares board {board!r}, which did not resolve: "
                                    f"{type(exc).__name__}: {exc}")
