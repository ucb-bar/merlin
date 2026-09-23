"""The declared editable surface of a forked compiler package, and the refusal when it omits one.

WHY THIS EXISTS. :mod:`merlin.perf.compiler_edit_scope` has implemented a host-frozen AST edit
authority for some time, and until this module NOTHING EVER DECLARED ONE. Searching the repository
for ``compiler_edit_contract_v1`` returns that module, its own test, :mod:`merlin.perf.agent_guidance`,
and four work-order tests that each build a throwaway contract inline. So a phase-2 fork ran either
with no authority object at all -- the agent may edit anything in the package -- or against a fixture
invented by the same caller that asked the question. Those two are different experiments and the
artifact afterwards does not say which one happened.

THE FAILURE THIS MODULE IS ACTUALLY AGAINST. A contract that names everything is not a contract, and
that failure is loud: the agent wins by editing an opcode and the win does not transfer. The opposite
failure is quiet and worse. Omit the one symbol that decides where the epilogue is placed and the
experiment is UNWINNABLE while still producing rounds, edits, verdicts and a report -- indistinguishable
from an agent that tried and failed. So a declaration here carries ``required_decisions``: for each
decision under study, the symbol that makes it. :func:`load` REFUSES a declaration whose required
decision has no owner, and names the decision and the symbol. An unwinnable experiment should fail at
load rather than at the end of a campaign.

WHAT THIS MODULE DOES NOT DO. It grants nothing and checks no edit. Deciding whether a submitted
change lies inside the surface is :func:`merlin.perf.compiler_edit_scope.inspect_compiler_edits`, and
scoring a round's mechanism is ``inspect_round_mechanism_edits``. This module only turns a reviewed
file into the object those functions already take, with the digest they already require.

THE DIGEST IS COMPUTED, NOT STORED. ``compiler_edit_scope`` binds a contract's canonical identity into
every inspection result so a candidate cannot swap contracts underneath one. That identity is derived
here from the file's own body rather than written into the file, because a digest stored beside the
bytes it covers goes stale on every edit and is then either corrected mechanically -- in which case it
proves nothing -- or left wrong. The review of the tracked file is the authority; a test pins the
digest as a literal so that changing the surface is a visible diff in two places rather than one.

TARGET-NEUTRAL. Declarations live per target and per package under ``merlin/contract/phase2_edit_contracts/``.
This module names no target and no package; both are parameters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "Phase2EditContractError",
    "contract_path",
    "load",
    "seal",
    "validate_against_package",
]

_CONTRACTS = ("contract", "phase2_edit_contracts")

#: Keys this module owns. They travel inside the sealed body -- so a change to the ARGUMENT for a
#: boundary changes the contract's identity, which is the point -- but ``compiler_edit_scope`` does
#: not read them and must not have to.
_ANNOTATIONS = frozenset({"target", "package_id", "package", "required_decisions", "excluded"})


class Phase2EditContractError(ValueError):
    """A declaration was missing, malformed, or omitted an owner for a decision under study."""


def contract_path(target: str, package_id: str) -> Path:
    """Where the declaration for one package lives. The target is a directory, never a literal."""
    from ..common.paths import merlin_dir

    return merlin_dir().joinpath(*_CONTRACTS, target, f"{package_id}.yaml")


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """Add the canonical ``sha256`` ``compiler_edit_scope`` requires, over every other key.

    Exactly the digest ``validate_edit_contract`` recomputes, so a contract sealed here and handed
    straight to it cannot disagree about its own identity.
    """
    payload = {key: value for key, value in body.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "sha256": digest}


def _owner_of(entry: Mapping[str, Any]) -> str:
    return f"{entry.get('path')}:{entry.get('symbol')}"


def _check_required(body: Mapping[str, Any]) -> None:
    """Refuse, BY NAME, a declaration whose decision under study has no owner.

    ``validate_edit_contract`` checks that every named symbol EXISTS. It cannot check the converse --
    that a symbol which needs naming was named -- because it has no idea what the experiment is about.
    That is the whole gap: a contract omitting the deciding symbol is perfectly valid and perfectly
    unwinnable.
    """
    required = body.get("required_decisions")
    if not isinstance(required, list) or not required:
        raise Phase2EditContractError(
            "the declaration names no required_decisions. A surface with nothing declared to be "
            "reachable through it cannot be shown to be reachable at all, and an agent that fails "
            "would be indistinguishable from a boundary that made success impossible."
        )
    owned = {_owner_of(entry) for entry in body.get("existing_symbols", []) if isinstance(entry, Mapping)}
    missing = []
    for entry in required:
        if not isinstance(entry, Mapping) or not entry.get("decision") or not entry.get("owner"):
            raise Phase2EditContractError("every required_decisions entry needs a 'decision' and an 'owner'")
        if str(entry["owner"]) not in owned:
            missing.append((str(entry["decision"]), str(entry["owner"])))
    if missing:
        detail = "; ".join(f"{decision} is decided by {owner}" for decision, owner in missing)
        raise Phase2EditContractError(
            f"the declared surface omits the owner of a decision under study: {detail}. "
            "An experiment run against this contract could not reach the code it is about, and "
            "would report the agent's failure rather than the boundary's."
        )


def load(target: str, package_id: str, *, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The sealed contract for one package, refused unless every decision under study has an owner.

    ``body`` injects a declaration instead of reading the tracked file -- used by tests, and by any
    caller assembling one from another source. It goes through the identical checks, because an
    injected declaration is exactly where an unchecked one would otherwise get in.
    """
    if body is None:
        import yaml

        path = contract_path(target, package_id)
        if not path.is_file():
            raise Phase2EditContractError(
                f"no phase-2 edit contract at {path}. The editable surface is DECLARED, never inferred "
                "from what the agent happened to touch."
            )
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(body, Mapping):
        raise Phase2EditContractError("a phase-2 edit contract must be a mapping")
    if body.get("schema") != "compiler_edit_contract_v1":
        raise Phase2EditContractError(
            f"schema is {body.get('schema')!r}; this must be the schema compiler_edit_scope validates, "
            "so that a declaration and the enforcement it is checked by cannot drift apart"
        )
    if not isinstance(body.get("existing_symbols"), list) or not body["existing_symbols"]:
        raise Phase2EditContractError("a phase-2 edit contract must name at least one existing symbol")
    _check_required(body)
    return seal(body)


def validate_against_package(contract: Mapping[str, Any], package_root: Path) -> dict[str, Any]:
    """Resolve every named symbol in the package's own AST, via ``compiler_edit_scope``.

    ``package_root`` is the directory the contract's paths are relative to, which is NOT always the
    package directory. Where a package keeps its ``manifest.yaml`` one level down, rooting here at the
    package directory would put the manifest outside the root-only name ``inspect_compiler_edits``
    checks, and the protection against a candidate widening its own authority would be off while
    looking exactly the same. The declaration states its own root; the caller passes it.
    """
    from .compiler_edit_scope import validate_edit_contract

    return validate_edit_contract(contract, package_root)
