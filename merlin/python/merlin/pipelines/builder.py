"""Build a runnable pipeline from a strategy's ``lowering_pipeline`` spec string.

The spec is a comma-separated list of pass names (e.g.
``"merlin-contract,make-resident,defer-commit,toynpu-lower"``). ``build_pipeline`` resolves
each against ``merlin.pipelines.registry`` and returns a :class:`Pipeline` that applies them in
order to a starting cost-model plan. Unknown names default to identity (``strict=False``) so a
forward-looking xDSL pass spec still runs; pass ``strict=True`` to require every pass to exist.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.pipelines import registry


@dataclass(frozen=True)
class Pipeline:
    """An ordered, runnable list of named passes over a cost-model plan."""

    names: tuple[str, ...]
    unknown: tuple[str, ...] = ()

    def run(self, plan: dict) -> dict:
        out = dict(plan)
        for name in self.names:
            fn = registry.get_pass(name)
            if fn is not None:
                out = fn(out)
        return out

    def __str__(self) -> str:
        return ",".join(self.names)


def parse_spec(spec: str) -> list[str]:
    """Split a pass spec string into pass names (whitespace/empty tolerant)."""
    return [p.strip() for p in (spec or "").split(",") if p.strip()]


def build_pipeline(spec: str, strict: bool = False) -> Pipeline:
    """Resolve ``spec`` into a runnable :class:`Pipeline`.

    Raises ``KeyError`` for unknown passes when ``strict``; otherwise records them in
    ``Pipeline.unknown`` and treats them as identity at run time.
    """
    names = parse_spec(spec)
    unknown = tuple(n for n in names if not registry.has_pass(n))
    if strict and unknown:
        raise KeyError(f"unknown passes: {', '.join(unknown)}")
    return Pipeline(names=tuple(names), unknown=unknown)
