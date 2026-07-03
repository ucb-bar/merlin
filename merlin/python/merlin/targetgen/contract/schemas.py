"""Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.

A single place to load + validate the contract schemas so the runner, the tests, and the
packages all enforce the same rules. Validation raises :class:`ContractViolation` with a concise
message; nothing here ever silently accepts a malformed artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

# repo root = .../oscar-merlin (this file: merlin/python/merlin/targetgen/contract/schemas.py)
_REPO = Path(__file__).resolve().parents[5]
# The experiment ABI (contract + capsule corpus) — lives under merlin/ as core infra.
# Paths are resolved repo-root-relative via contract_dir(); no compat symlink.
DEFAULT_CONTRACT_DIR = _REPO / "merlin" / "contract"


class ContractViolation(ValueError):
    """A contract artifact failed schema validation (fail-closed)."""


def contract_dir(override: str | Path | None = None) -> Path:
    """Resolve the contract dir: explicit override > $MERLIN_CONTRACT_DIR > default merlin/contract."""
    if override:
        return Path(override)
    import os
    env = os.environ.get("MERLIN_CONTRACT_DIR")
    return Path(env) if env else DEFAULT_CONTRACT_DIR


def load_schema(name: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """Load a schema by short name (``command_buffer`` -> command_buffer.schema.json)."""
    path = contract_dir(contract) / "schemas" / f"{name}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"contract schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(obj: Any, name: str, *, contract: str | Path | None = None) -> None:
    """Validate ``obj`` against the named schema; raise ContractViolation on any error."""
    schema = load_schema(name, contract=contract)
    # The contract schemas carry a RELATIVE "$id" (e.g. "merlin/bench_contract/foo.schema.json")
    # used only as a label. Under jsonschema's RefResolver that relative id becomes the base URI,
    # so an in-document fragment ref ("#/$defs/operand") resolves to a bogus URL and the resolver
    # tries to FETCH it ("unknown url type: ...merlin/bench_contract/..."), crashing validation
    # whenever an instruction carries rs1/rs2. Every "$ref" in these schemas is an in-document
    # fragment, so dropping "$id" (in-memory only) resolves refs against an empty base. This
    # ENABLES the operand-level checks that previously crashed; it does not weaken validation.
    schema.pop("$id", None)
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ContractViolation(f"{name} schema violation at {loc}: {e.message}") from e


def validate_command_buffer(cb: Any, *, contract: str | Path | None = None) -> None:
    validate(cb, "command_buffer", contract=contract)


def validate_manifest(man: Any, *, contract: str | Path | None = None) -> None:
    validate(man, "manifest", contract=contract)
