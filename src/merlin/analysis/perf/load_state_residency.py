"""Did the emitted stream KEEP its load configurations, or re-establish them per block?

THE LEVER. A target whose load path holds several configurations live lets a program configure each
operand's movement once and then address it by a selector carried in the configuration command. A
program that does not use the selector has one slot, and a contraction moves two operands whose row
pitches differ (the activation's pitch is its reduction extent, the weight's is its output extent),
so the one slot alternates and never holds what the next transfer needs: the configuration command is
re-issued on every block of the nest. Nothing in this corpus asked for it. The measured consequence
of nobody asking is the whole reason this module exists -- a correct compiler that re-configures the
load path on every ``(m, n, k)`` block is exactly as correct as one that does not, and the numeric
oracle cannot tell them apart.

THE PROPERTY, and why it is decidable from the candidate's OWN trace with no second arm:

    A configuration command carries no address. Two such commands with identical payloads therefore
    configure identical state, and the second one cannot be made to matter by any machine state the
    first did not already establish -- which is precisely the argument
    :func:`merlin.development.targetgen.trace_check._stale_mode_config_findings` already makes for the execution
    mode, applied to the load path. So a load configuration that re-establishes what a state already
    holds is PROVABLY inert work, not a proxy for it.

    Hence the demand: **the number of load-configuration commands must not exceed the number of
    DISTINCT load configurations the program uses, whenever that number is within the target's
    derived load-state capacity.** Stated over the property, never over a mechanism -- a schedule
    that avoids the reissue by staging both operands under one configuration satisfies it just as a
    schedule that uses two selectors does, and neither is named here.

WHY A STRUCTURAL PASS CRITERION AND NOT A CYCLE BOUND.

1. The quantity is exact. Every excess command is one dispatched instruction that establishes state
   already held; there is no modelling step between the observation and the claim, so no cost model
   can be wrong about it.
2. A cycle bound needs a second arm, and the second arm for this lever is a different compiler --
   not a different shape or a different emitter knob. A paired cycle delta between two emitter
   settings would measure the machine; it would not require the candidate to do anything.
3. It cannot be satisfied by being slow but correct, which is the failure mode that makes a
   performance demand harder than a correctness one.

WHAT PASSING DOES NOT PROVE, said plainly because the alternative is the failure this repo keeps
repeating: passing shows that the redundant configuration work is ABSENT. It is not a speedup claim
and it does not certify that the movement schedule is good. A family that quoted it as a win would be
overclaiming.

FAIL CLOSED EVERYWHERE. The selector's position and width come from the target's own extracted
register-bundle layout; the capacity comes from that width. A target that publishes no such layout,
or a trace whose configuration operands do not resolve to constants, yields ``REFUSED`` with the
reason -- never a pass, and never a substituted default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.development.targetgen import semantic_families as SF

__all__ = [
    "CONFIG_LOAD_CLASS",
    "PASS",
    "FAIL",
    "REFUSED",
    "SELECTOR_FIELD",
    "capacity_from_selector",
    "load_state_selector",
    "movement_in_classes_for",
    "residency_findings",
    "residency_verdict",
]

PASS = "PASS"
FAIL = "FAIL"
REFUSED = "REFUSED"

#: The instruction class a load configuration decodes to: the one shared plumbing class that configures
#: INBOUND movement, read from the shared class vocabulary
#: (:func:`merlin.development.targetgen.semantic_families.configuration_classes`) the capability manifest's
#: ``config_subtype`` maps its own encoding onto -- not an encoding and not a target's spelling. A target
#: whose manifest declares no such subtype simply never produces one, and the verdict below refuses for
#: want of evidence rather than passing an empty cohort. Unpacked, so a vocabulary that ever named two
#: fails at import instead of silently checking one.
(CONFIG_LOAD_CLASS,) = SF.configuration_classes("in")

#: The EXTRACTOR'S field name for the load-state selector inside the load-configuration register
#: bundle. This is the RTL introspection vocabulary (the same vocabulary that names the bundle itself
#: and the store-side ``activation`` field), not a bit position and not an ISA constant: the offset
#: and width are read from whatever the extraction published under this name, and a target that
#: publishes nothing under it has no derivable selector and is refused.
SELECTOR_FIELD = "state_id"


def load_state_selector(target: str) -> dict[str, int] | None:
    """``{"offset", "width", "capacity"}`` of ``target``'s load-state selector, or ``None``.

    The capacity travels WITH the selector rather than beside it, because the two are derived from
    different facts and a caller that threaded only the field would silently get the wider of the two
    bounds. One object, both facts, no way to carry half of it. The capacity is the SMALLER of two
    derived bounds, because over-reading it is the one way the demand could ask for something the
    machine cannot do:

    * the selector's own span -- how many states the ENCODING can name;
    * how many distinct inbound-movement instruction classes the target's decode table declares --
      how many states a program can actually ADDRESS, since a state it has no instruction to reach is
      a state it cannot use.

    Measured on the target this was written against, those are 4 and 2: the selector is two bits wide
    while the manifest declares two load classes. Taking the width alone would have demanded four
    held configurations from a program that has two ways to ask for one. Under-reading is the safe
    direction -- it only ever makes the verdict REFUSE where it might have failed -- so a target whose
    classes cannot be counted falls back to the selector's span and nothing is invented.

    Read from the target's extracted load-configuration register bundle -- the same source the store
    configuration's field layout comes from. ``None`` is a real answer and callers must treat it as a
    refusal: a target whose RTL we could not read has not been shown to have one state, it has been
    shown to be unreadable.
    """
    layout = _load_config_layout(target)
    if not isinstance(layout, Mapping):
        return None
    spec = (layout.get("fields") or {}).get(SELECTOR_FIELD)
    if not isinstance(spec, Mapping):
        return None
    offset, width = spec.get("offset"), spec.get("width")
    if not isinstance(offset, int) or not isinstance(width, int):
        return None
    if offset < 0 or width < 1:
        return None
    found = {"offset": offset, "width": width}
    addressable = _addressable_movement_classes(target)
    found["capacity"] = (1 << width) if addressable is None else min(1 << width, addressable)
    return found


def capacity_from_selector(selector: Mapping[str, Any] | None) -> int | None:
    """How many load configurations this selector says can be held at once, or ``None``.

    A selector carrying a derived ``capacity`` is taken at its word -- that is the narrowed bound
    :func:`load_state_selector` computed from two facts. Otherwise this falls back to the selector's
    own span, which is the ENCODING's bound and may be wider than what the machine can address; the
    verdict only ever uses the number to ADMIT a program, never to require one, so the fallback is
    conservative in the only direction that matters.
    """
    if not isinstance(selector, Mapping):
        return None
    declared = selector.get("capacity")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 1:
        return declared
    width = selector.get("width")
    if not isinstance(width, int) or width < 1:
        return None
    return 1 << width


def movement_in_classes_for(target: str) -> frozenset:
    """The DECLARED inbound-movement instruction classes ``target``'s own decode table names.

    Read through the selected support's ISA facts (:func:`merlin.development.targetgen.rocc.decode.funct_class_for`)
    and intersected with the shared vocabulary's inbound-movement classes
    (:func:`merlin.development.targetgen.semantic_families.movement_classes`), so a target declaring one load class
    is never assumed to have three. Raises when the table cannot be read; callers decide.
    """
    from merlin.development.targetgen.rocc import decode as _decode

    declared = {str(name) for name in _decode.funct_class_for(str(target)).values()}
    return frozenset(SF.movement_classes("in") & declared)


def _addressable_movement_classes(target: str) -> int | None:
    """How many distinct inbound-movement classes ``target`` declares, or ``None``."""
    try:
        declared = movement_in_classes_for(str(target))
    except Exception:  # noqa: BLE001 -- a target whose table cannot be read declares no count
        return None
    return len(declared) or None


def _load_config_layout(target: str) -> Mapping[str, Any] | None:
    """The target's load-configuration register-bundle layout, or ``None`` if it has none.

    Published by the SELECTED support's ISA facts under ``CONFIG_LD_LAYOUT`` -- the load-side twin of
    the store layout it already publishes -- because the bundle's name in the RTL is the target's own
    spelling and belongs with its support, not here. Support that publishes none leaves the selector
    underivable, and every verdict that needs it refuses.
    """
    try:
        from merlin.development.targetgen.rocc import decode as _decode

        return _decode.isa_constants(str(target)).get("CONFIG_LD_LAYOUT")
    except Exception:  # noqa: BLE001 -- a target whose ISA cannot be resolved publishes no layout
        return None


def _operand_raw(value: Any) -> int | None:
    """The resolved constant behind one decoded operand, or ``None`` when it is not a constant.

    A configuration payload that is not a constant cannot be compared to another payload, and
    guessing would decide the claim on an operand nobody read. ``None`` propagates to a refusal.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping) and value.get("kind") == "const":
        raw = value.get("raw")
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    return None


def _split(raw: int, selector: Mapping[str, int]) -> tuple[int, int]:
    """``(state, payload)`` -- the selector's value, and the payload with its bits removed."""
    offset, width = int(selector["offset"]), int(selector["width"])
    mask = ((1 << width) - 1) << offset
    return (raw & mask) >> offset, raw & ~mask


def residency_verdict(
    trace: Any,
    *,
    selector: Mapping[str, int] | None,
    capacity: int | None = None,
) -> dict[str, Any]:
    """Decide the load-state residency property for one emitted trace.

    Returns ``{"verdict", "reason", ...}``. ``FAIL`` names every command that re-established a
    configuration already held and reports the excess; ``REFUSED`` names what could not be derived.
    An empty cohort -- a trace with no load configuration at all -- is REFUSED, not passed: a
    property checked over nothing is the check that cannot fail.
    """
    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    instructions = trace.get("instructions")
    if not isinstance(instructions, Sequence):
        raise TypeError("trace instructions must be a sequence")

    if selector is None:
        return {
            "verdict": REFUSED,
            "reason": (
                "the target publishes no load-state selector in its extracted load-configuration "
                "layout, so the number of configurations it can hold live is UNKNOWN and no "
                "residency demand can be decided from this trace"
            ),
        }
    if capacity is None:
        capacity = capacity_from_selector(selector)
    if not isinstance(capacity, int) or capacity < 1:
        return {
            "verdict": REFUSED,
            "reason": "the load-state capacity is not derivable from the selector's width",
            "selector": dict(selector),
        }

    configs = [
        (index, instruction)
        for index, instruction in enumerate(instructions)
        if isinstance(instruction, Mapping) and instruction.get("class") == CONFIG_LOAD_CLASS
    ]
    if not configs:
        return {
            "verdict": REFUSED,
            "reason": (
                f"the trace carries no {CONFIG_LOAD_CLASS} instruction, so there is no load "
                "configuration to hold resident and nothing this demand could have observed"
            ),
            "config_count": 0,
        }

    # Per state, every payload that state has ALREADY held -- not merely the one it holds now. The
    # single-slot signature is not an immediate repeat: one state carrying two operands' configurations
    # in turn never repeats itself back to back, it EVICTS and RESTORES, and a check that only compared
    # a command with its predecessor would report that thrashing as clean.
    held: dict[int, set[tuple]] = {}
    redundant: list[dict[str, Any]] = []
    payloads: set[tuple] = set()
    for index, instruction in configs:
        raw = _operand_raw(instruction.get("rs1"))
        if raw is None:
            return {
                "verdict": REFUSED,
                "reason": (
                    f"{CONFIG_LOAD_CLASS} at #{index} carries a configuration word that does not "
                    "resolve to a constant, so its payload cannot be compared with another's"
                ),
                "config_count": len(configs),
            }
        state, payload_bits = _split(raw, selector)
        second = instruction.get("rs2")
        second_raw = _operand_raw(second)
        # A non-constant second word is not a refusal on its own: it is compared by its DECODED
        # identity (kind + reference), exactly as a movement operand is, so a configuration built
        # from a runtime stride still compares equal to itself and different from another's.
        payload = (payload_bits, second_raw if second_raw is not None else _reference_identity(second))
        payloads.add(payload)
        if payload in held.setdefault(state, set()):
            redundant.append({"index": index, "state": state})
        held[state].add(payload)

    distinct = len(payloads)
    issued = len(configs)
    row: dict[str, Any] = {
        "config_count": issued,
        "distinct_configurations": distinct,
        "states_used": sorted(held),
        "capacity": capacity,
        "selector": dict(selector),
        "redundant": redundant,
    }
    if distinct > capacity:
        return {
            **row,
            "verdict": REFUSED,
            "reason": (
                f"the program uses {distinct} distinct load configurations but the target can "
                f"address only {capacity}; holding them all is not something this machine can do, "
                "so no reissue here has been shown to be avoidable"
            ),
        }
    if issued > distinct:
        return {
            **row,
            "verdict": FAIL,
            "reason": (
                f"{issued} load configurations are issued for {distinct} distinct "
                f"configuration(s), which this target can hold live at once ({capacity} "
                f"addressable state(s)); {issued - distinct} of them re-establish state the load "
                f"path already held and do provably no work. "
                + (
                    f"{len(redundant)} of them re-establish a configuration THAT STATE HAS HELD "
                    f"BEFORE (at {', '.join('#' + str(r['index']) for r in redundant[:4])}), across "
                    f"{len(held)} state(s) -- the signature of one slot carrying each operand's "
                    f"configuration in turn and never holding the next one."
                    if redundant
                    else "No state re-established a configuration it had held, so the excess is a "
                    "duplicate configuration spread across states rather than one slot thrashing."
                )
            ),
            "excess": issued - distinct,
        }
    return {
        **row,
        "verdict": PASS,
        "excess": 0,
        "reason": (
            f"{issued} load configuration(s) establish {distinct} distinct configuration(s): none "
            "re-establishes state the load path already held. This says the redundant configuration "
            "work is absent; it is not a claim that the movement schedule is otherwise good"
        ),
    }


def _reference_identity(value: Any) -> tuple:
    """A hashable identity for a non-constant operand, compared structurally (never by text)."""
    if isinstance(value, Mapping):
        return ("ref", value.get("kind"), value.get("raw"), value.get("arg_index"), value.get("offset"))
    return ("lit", value)


def residency_findings(trace: Any, *, selector: Mapping[str, int] | None, capacity: int | None = None) -> list[str]:
    """The verdict as advisory diagnostic lines, for callers that collect findings rather than verdicts.

    A ``PASS`` yields no line. Both ``FAIL`` and ``REFUSED`` yield one, because a refusal that
    produced silence would read exactly like a clean trace -- which is how a check that cannot fail
    gets shipped.
    """
    verdict = residency_verdict(trace, selector=selector, capacity=capacity)
    if verdict["verdict"] == PASS:
        return []
    return [f"load-state residency ({verdict['verdict'].lower()}): {verdict['reason']}"]
