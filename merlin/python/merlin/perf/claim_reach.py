"""Can a declared performance family reach a VERDICT at all, from its own declaration?

:mod:`merlin.perf.emitter_reach` asks whether a family's emitter can build the members its trait gate
admits. That is one of two ways a claim comes to be unsatisfiable while reading as unfinished. The
other is internal: a declaration whose own parts cannot all be true at once, or whose falsifier fires
on a quantity nobody ever declares. Neither the trait gate nor the emitter gate looks at that, because
both read one field and this reads the relation BETWEEN fields.

⚠️ MEASURED IN THIS TREE, and each rule below was written against a real declaration rather than
imagined:

``the gate demands two of what the comparand holds equal``
    One family's gate asks for ``at_least_two_separation_regimes`` while its comparand's
    ``demand_equal`` names ``separation_regime``. Both are reasonable sentences; together they say the
    measurement must vary a quantity it must also hold fixed, so no admissible pair exists. The
    contradiction is visible in the declaration and in nothing else -- the emitter builds, the traits
    pass, the capsules materialise, and the family simply never produces a pair.

``the falsifier fires on a bound the contract never declares``
    Another family's falsifier fires when a saving ``changes_beyond_the_declared_band``. No band is
    declared -- not in that family, not in the target's contract, nowhere. A firing condition that
    names its own threshold as *declared* and is not is not a strict gate: it is an unevaluable one,
    and an unevaluable falsifier cannot fail, which is the same as not existing.

**The rules are deliberately narrow.** Each fires only on a contradiction the declaration states about
ITSELF -- a quantity the contract itself calls ``declared``, or a capacity that names something the
comparand lists verbatim. A looser rule (say, flagging every unquantified "noise band") would be a
style opinion wearing a gate's authority, and would go off on families that are merely terse. What is
wanted is the case where reading the declaration honestly and reading it again cannot both succeed.

Parsing is STRUCTURAL: declaration keys and values are underscore-joined identifiers, so they split on
``_`` into tokens. No pattern matching, no opcode-style table of known family names, and nothing here
names a target or a family -- a declaration arrives as a mapping and leaves as a verdict.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "AnalyzerIdentity", "CAPACITY_CONTRADICTS_DEMAND_EQUAL", "FamilyReach",
    "UNDECLARED_FIRING_QUANTITY", "analyzer_identity", "capacity_demand", "family_reach",
    "has_decision_procedure", "tokens",
]

#: The gate asks for at least two of a quantity the comparand's ``demand_equal`` holds fixed.
CAPACITY_CONTRADICTS_DEMAND_EQUAL = "capacity_contradicts_demand_equal"
#: The falsifier's firing condition calls a quantity ``declared`` and the declaration does not carry it.
UNDECLARED_FIRING_QUANTITY = "undeclared_firing_quantity"

#: The token a capacity uses to say "this many DISTINCT values are needed". Read from the declaration's
#: own spelling rather than from a count word list, so ``at_least_three_x`` is served by the same code.
_AT, _LEAST = "at", "least"
#: The token a firing condition uses to point at a quantity it expects the contract to have fixed.
_DECLARED = "declared"
#: Tokens that carry no quantity of their own and so are skipped when reading a capacity or a firing
#: condition's subject. They are English glue, not part of any name a contract declares.
_GLUE = frozenset({"a", "an", "the", "any", "each", "of", "per", "and", "or", "its", "their", "one",
                   "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"})


def tokens(text: object) -> tuple[str, ...]:
    """Lowercase underscore-separated tokens of one declaration identifier."""
    return tuple(part for part in str(text or "").lower().split("_") if part)


def _singular(token: str) -> str:
    """``regimes`` -> ``regime``. Only the plural ``s``; nothing here needs English morphology."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def _normalised(items) -> tuple[str, ...]:
    return tuple(_singular(t) for t in items if t not in _GLUE)


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    """Is ``needle`` a contiguous subsequence of ``haystack``? Empty needles never match."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def capacity_demand(gate: Mapping | None) -> tuple[str, ...]:
    """The quantity an ``at_least_<n>_<quantity>`` capacity demands distinct values of.

    Empty when the capacity is not of that shape -- a capacity naming a completeness condition
    (``complete_..._group``) or a resource condition demands no axis, and reading one out of it would
    invent the very relation this module is checking.
    """
    parts = tokens((gate or {}).get("capacity"))
    if len(parts) < 3 or parts[0] != _AT or parts[1] != _LEAST:
        return ()
    return _normalised(parts[3:] if len(parts) > 3 else ())


def _declared_quantity(falsifier: Mapping | None) -> tuple[str, ...]:
    """The subject a firing condition calls ``declared``, as normalised tokens."""
    parts = tokens((falsifier or {}).get("fires_when"))
    if _DECLARED not in parts:
        return ()
    return _normalised(parts[parts.index(_DECLARED) + 1:])


#: The one value that may NOT count as its own declaration. ``fires_when`` is the sentence under test;
#: letting it satisfy itself makes the rule vacuous -- every firing condition mentions its own subject,
#: so ``the_declared_band`` would always find a "band" in the words ``the_declared_band``. Measured: the
#: rule silently passed the one family it was written for until this exclusion was added.
_SELF_REFERENTIAL = ("falsifier", "fires_when")


def _declaration_token_bags(performance: Mapping) -> list[tuple[str, ...]]:
    """Every key and scalar value anywhere under one family's declaration, as token tuples.

    Keys count as declarations: a contract that carries ``residual_bound: {...}`` has declared a
    residual bound whether or not the words appear in a value.
    """
    bags: list[tuple[str, ...]] = []

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                here = path + (str(key),)
                bags.append(_normalised(tokens(key)))
                walk(value, here)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, path)
        elif node is not None and not isinstance(node, bool):
            if path != _SELF_REFERENTIAL:
                bags.append(_normalised(tokens(node)))

    walk(performance, ())
    return bags


def has_decision_procedure(performance: Mapping) -> bool:
    """True when the family names the analyzer that turns its evidence into a verdict.

    A family without one is not thereby unsatisfiable -- a driver can be wired later -- but it is not
    decidable TODAY, and the two states are reported separately for exactly that reason.
    """
    acceptance = performance.get("acceptance")
    return isinstance(acceptance, Mapping) and bool(acceptance.get("analyzer"))


#: Characters a declared analyzer identifier may use.  Checked by MEMBERSHIP rather than by a
#: pattern: the parts are Python module and function names, and a set test says exactly that without
#: a regex whose narrowness would silently reject a valid-but-differently-spelled declaration.
_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


@dataclass(frozen=True)
class AnalyzerIdentity:
    """What a family's own ``acceptance.analyzer`` tells a dispatcher to import and call."""

    #: The declaration verbatim, so a refusal can quote what the family actually wrote.
    declared: str
    module: str
    function: str
    version: str


def analyzer_identity(performance: Mapping) -> AnalyzerIdentity | None:
    """Split a family's own ``acceptance.analyzer`` into module, function and version.

    The declared form is ``<module>.<function>/<version>``.  A caller resolves the decision
    procedure from the DECLARATION rather than from a table of family names, so a newly declared
    family is dispatched with no code edit, and a family that declares none is refused BY NAME
    instead of falling through to some other family's analyzer.

    Three states, deliberately distinct.  ``None`` means the family declares no analyzer -- a wiring
    state, the same one :func:`has_decision_procedure` reports.  :class:`ValueError` means it
    declares one that is not of that form, which is a malformed contract rather than an absent one.
    Anything else is the resolved identity.
    """
    acceptance = performance.get("acceptance") if isinstance(performance, Mapping) else None
    declared = acceptance.get("analyzer") if isinstance(acceptance, Mapping) else None
    if declared is None or declared == "":
        return None
    if not isinstance(declared, str):
        raise ValueError(
            f"a declared acceptance.analyzer must be a string, not {type(declared).__name__}")
    body, slash, version = declared.partition("/")
    module, dot, function = body.rpartition(".")
    if not slash or not dot:
        raise ValueError(
            f"declared analyzer {declared!r} is not <module>.<function>/<version>")
    parts = module.split(".") + [function, version]
    if any(not part or not set(part) <= _IDENTIFIER_CHARS for part in parts):
        raise ValueError(
            f"declared analyzer {declared!r} does not name a simple module, function and version")
    return AnalyzerIdentity(declared=declared, module=module, function=function, version=version)


@dataclass(frozen=True)
class FamilyReach:
    """Whether one declared family can reach a verdict, and what stops it when it cannot."""

    family: str
    satisfiable: bool
    #: One entry per contradiction found, each ``{"rule": ..., "detail": ...}``.
    obstructions: tuple[dict, ...] = ()
    #: Facts a reader needs beside the verdict, not themselves obstructions.
    decidable_today: bool = False
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        return {"family": self.family, "satisfiable": self.satisfiable,
                "decidable_today": self.decidable_today,
                "obstructions": [dict(o) for o in self.obstructions],
                "notes": list(self.notes)}


def family_reach(performance: Mapping) -> FamilyReach:
    """Decide one family from its own ``performance`` declaration.

    Reads the block a profile sweep's ``base.performance`` carries, which is the same block each
    generated capsule copies, so the audit runs identically against the shared template and against a
    materialised target corpus.
    """
    family = str(performance.get("family") or "<unnamed>")
    obstructions: list[dict] = []

    demanded = capacity_demand(performance.get("gate"))
    if demanded:
        held_equal = [entry for entry in ((performance.get("comparand") or {}).get("demand_equal") or [])]
        for entry in held_equal:
            entry_tokens = _normalised(tokens(entry))
            # A single-token entry (``K``, ``operation``) is too weak to establish the relation: it can
            # land inside an unrelated capacity by coincidence. Two or more tokens naming the same
            # quantity the capacity asks two of is the contradiction, not a coincidence.
            if len(entry_tokens) >= 2 and _contains(demanded, entry_tokens):
                obstructions.append({
                    "rule": CAPACITY_CONTRADICTS_DEMAND_EQUAL,
                    "detail": (
                        f"the gate's capacity demands at least two distinct "
                        f"{'_'.join(demanded)}, while the comparand's demand_equal holds "
                        f"{entry!r} fixed; no admissible measurement can both vary that quantity "
                        f"and hold it equal, so the family admits no pair"),
                })

    quantity = _declared_quantity(performance.get("falsifier"))
    if quantity:
        bags = _declaration_token_bags(performance)
        if not any(_contains(bag, quantity) for bag in bags):
            obstructions.append({
                "rule": UNDECLARED_FIRING_QUANTITY,
                "detail": (
                    f"the falsifier fires on a {'_'.join(quantity)} it calls declared, and nothing "
                    f"in this family's declaration carries one; the firing condition cannot be "
                    f"evaluated, and a falsifier that cannot fire cannot fail"),
            })

    decidable = has_decision_procedure(performance)
    notes: list[str] = []
    if not decidable:
        notes.append("no acceptance.analyzer is declared, so nothing computes this family's verdict "
                     "from its rows today; that is a wiring state, not a contradiction")
    return FamilyReach(family=family, satisfiable=not obstructions,
                       obstructions=tuple(obstructions), decidable_today=decidable,
                       notes=tuple(notes))
