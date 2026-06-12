"""Emit a policy_rule dict (conforming to ``policy_rule.schema.yaml``)."""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas


def emit_policy_rule(
    policy: str,
    evidence: Iterable[str],
    when: dict,
    actions: Iterable[str],
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped policy rule.

    ``evidence`` is the list of real kernel evidence-ids backing the rule; ``when`` is the
    symbolic condition (compiler-visible facts, not constants from a single kernel);
    ``actions`` are schedule/interface actions.
    """
    rule = {
        "policy": policy,
        "evidence": sorted(dict.fromkeys(evidence)),
        "when": dict(when),
        "actions": list(actions),
    }
    if extra:
        rule.update(extra)
    if validate:
        schemas.validate_or_raise(rule, "policy_rule")
    return rule
