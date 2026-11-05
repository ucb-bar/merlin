"""Decode a command stream into the timing space's price classes and whole-call feature columns.

The same decoder reads three sources -- a measured capsule's static instruction trace, a probe's
emitted command list and (in a test) a synthetic stream -- so a probe's predicted features and an
archived label's features are computed by one function. Configuration state is tracked exactly as
the timing space declares it (a selector field picks which state a config command writes); a config
command whose selector or operands are not constants makes every state field UNKNOWN, and a command
whose class depends on UNKNOWN state is counted as unclassified rather than guessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .space import COUNT, FIXED, TimingSpace

_WORD = 1 << 64


@dataclass(frozen=True)
class Command:
    """One accelerator command: its RTL funct name (or a pseudo class) and constant operands."""

    name: str
    rs1: int | None = None
    rs2: int | None = None


def _word(value) -> int | None:
    if type(value) is not int:
        return None
    return value % _WORD


def commands_from_trace(trace: dict, space: TimingSpace) -> list[Command]:
    """A static ``instruction_trace.json`` (funct codes, constant or symbolic operands) as commands."""
    out = []
    for row in trace.get("instructions") or ():
        funct = row.get("funct")
        if funct is None:
            name = str(row.get("class"))
            if name not in space.pseudo_classes:
                name = f"UNKNOWN:{name}"
        else:
            name = space.funct_names.get(funct, f"UNKNOWN_FUNCT:{funct}")
        operands = []
        for slot in ("rs1", "rs2"):
            value = row.get(slot) or {}
            operands.append(_word(value.get("raw")) if value.get("kind") == "const" else None)
        out.append(Command(name, *operands))
    return out


def _matches(predicate, value) -> bool | None:
    if value is None:
        return None
    if isinstance(predicate, dict):
        return value != predicate["not"]
    if isinstance(predicate, list):
        return value in predicate
    return value == predicate


@dataclass
class Decoded:
    counts: Counter = field(default_factory=Counter)
    quantities: dict = field(default_factory=dict)  # (class, quantity) -> int | None
    variants: Counter = field(default_factory=Counter)  # (class, instruction) -> commands
    unclassified: Counter = field(default_factory=Counter)
    commands: int = 0

    def features(self, space: TimingSpace) -> dict[str, float | None]:
        """Every column of ``space``; ``None`` marks a quantity some contributing command left UNKNOWN."""
        out: dict[str, float | None] = {}
        for column in space.columns:
            if column.kind == "fixed" and column.name == FIXED:
                out[column.name] = 1.0
            elif column.kind == "class":
                if column.quantity == COUNT:
                    out[column.name] = float(self.counts.get(column.price_class, 0))
                else:
                    value = self.quantities.get((column.price_class, column.quantity), 0)
                    out[column.name] = None if value is None else float(value)
            else:
                total: float | None = 0.0
                for source in column.sources:
                    cls, _, quantity = source.partition(".")
                    value = self.quantities.get((cls, quantity), 0)
                    total = None if value is None or total is None else total + value
                out[column.name] = total
        return out

    def aggregate(self, space: TimingSpace, quantity: str) -> float | None:
        """A domain quantity of the whole call: its rate column if the profile has one, else the sum."""
        rate = f"rate.{quantity}"
        if any(c.name == rate for c in space.columns):
            return self.features(space)[rate]
        total: float | None = 0.0
        for (cls, name), value in self.quantities.items():
            if name == quantity:
                total = None if value is None or total is None else total + value
        return total


def decode(space: TimingSpace, commands: list[Command]) -> Decoded:
    result = Decoded()
    classes = [c for c in space.classes if c.instructions]
    state: dict[str, int | None] = dict(space.reset)
    keys = {f"{s}.{f}" for s, entry in space.states.items() for f in entry["slots"]}
    by_code = {entry["code"]: name for name, entry in space.states.items()}
    for command in commands:
        result.commands += 1
        operands = {"rs1": command.rs1, "rs2": command.rs2}
        if command.name == space.config_instruction:
            code = space.selector.extract(operands[space.selector.operand]) if space.selector else None
            selected = by_code.get(code) if code is not None else None
            if selected is None:
                for key in keys:
                    state[key] = None
            else:
                for fname, slot in space.states[selected]["slots"].items():
                    state[f"{selected}.{fname}"] = slot.extract(operands[slot.operand])
        chosen, ambiguous = None, False
        for cls in classes:
            if command.name not in cls.instructions:
                continue
            verdicts = [_matches(p, state.get(k)) for k, p in cls.state.items()]
            if False in verdicts:
                continue
            if None in verdicts:
                ambiguous = True
                continue
            chosen = cls
            break
        if chosen is None:
            result.unclassified[command.name + (" (state unknown)" if ambiguous else "")] += 1
            continue
        result.counts[chosen.name] += 1
        result.variants[(chosen.name, command.name)] += 1
        slots = space.operand_slots.get(command.name, {})
        for quantity, terms in chosen.quantities.items():
            key = (chosen.name, quantity)
            if result.quantities.get(key, 0) is None:
                continue
            product: int | None = 1
            for term in terms:
                if term.startswith("fact."):
                    value = space.constants.get(term)
                elif term.startswith("state."):
                    value = state.get(term.removeprefix("state."))
                else:
                    slot = slots.get(term)
                    value = None if slot is None else slot.extract(operands[slot.operand])
                product = None if value is None or product is None else product * value
            result.quantities[key] = None if product is None else result.quantities.get(key, 0) + product
    return result
