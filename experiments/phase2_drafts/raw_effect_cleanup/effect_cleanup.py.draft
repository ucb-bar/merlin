"""Bounded straight-line configuration/barrier proposals; no execution authority.

Target adapters supply exact effects. Unknown operations forget configuration
and require a later barrier. A retained configuration command is itself pending
work: only adjacent barriers, or barriers made adjacent by a redundant write,
can disappear. This module is standalone for byte-identical candidate vendoring.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Effect:
    kind: str
    state: str = ""
    value: tuple[int, ...] = ()
    protected: bool = False

    def __post_init__(self):
        if (
            self.kind not in ("configuration", "activity", "barrier", "unknown", "pure")
            or type(self.state) is not str
            or type(self.value) is not tuple
            or any(type(n) is not int for n in self.value)
            or type(self.protected) is not bool
            or (self.kind == "configuration") != bool(self.state and self.value)
            or self.kind != "configuration" and (self.state or self.value)
            or self.protected and self.kind != "barrier"
        ):
            raise ValueError("cleanup requires closed typed effects")


def _checked(effects, maximum):
    if type(effects) is not tuple or any(type(e) is not Effect for e in effects):
        raise ValueError("cleanup requires original ordered typed effects")
    if type(maximum) is not int or not 0 < maximum <= 1 << 20 or len(effects) > maximum:
        raise ValueError("cleanup operation budget exceeded")


def cleanup_indices(effects, *, maximum=65536):
    """Return retained original indices; never move, insert or modify work."""
    _checked(effects, maximum)
    live, pending, had_barrier, retained = {}, True, False, []
    for index, effect in enumerate(effects):
        omit = effect.kind == "configuration" and live.get(effect.state) == effect.value
        omit |= effect.kind == "barrier" and had_barrier and not pending and not effect.protected
        if omit:
            continue
        retained.append(index)
        if effect.kind == "configuration":
            live[effect.state] = effect.value
            pending = True
        elif effect.kind == "barrier":
            pending, had_barrier = False, True
        elif effect.kind != "pure":
            pending = True
            if effect.kind == "unknown":
                live.clear()
    result = tuple(retained)
    verify_cleanup(effects, result, maximum=maximum)
    return result


def verify_cleanup(effects, retained, *, maximum=65536):
    """Independently prove omissions against actual retained ordered effects."""
    _checked(effects, maximum)
    if (
        type(retained) is not tuple
        or any(type(i) is not int or not 0 <= i < len(effects) for i in retained)
        or tuple(sorted(set(retained))) != retained
    ):
        raise ValueError("cleanup changed original operation order or identity")
    live, pending, had_barrier, cursor = {}, True, False, 0
    for index, effect in enumerate(effects):
        keep = cursor < len(retained) and retained[cursor] == index
        if not keep:
            redundant_write = effect.kind == "configuration" and live.get(effect.state) == effect.value
            redundant_barrier = effect.kind == "barrier" and had_barrier and not pending and not effect.protected
            if not (redundant_write or redundant_barrier):
                raise ValueError("cleanup omitted an unproved operation or completion")
            continue
        cursor += 1
        if effect.kind == "configuration":
            live[effect.state], pending = effect.value, True
        elif effect.kind == "barrier":
            pending, had_barrier = False, True
        elif effect.kind != "pure":
            pending = True
            if effect.kind == "unknown":
                live.clear()
    return {"source_operations": len(effects), "retained_operations": len(retained),
            "removed_operations": len(effects) - len(retained), "execution_qualification": "NOT_ESTABLISHED"}
