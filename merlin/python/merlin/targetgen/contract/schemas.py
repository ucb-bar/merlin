"""Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.

A single place to load + validate the contract schemas so the runner, the tests, and the
packages all enforce the same rules. Validation raises :class:`ContractViolation` with a concise
message; nothing here ever silently accepts a malformed artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from merlin.common.paths import repo_root
from typing import Any

import jsonschema

# repo root = .../merlin (this file: merlin/python/merlin/targetgen/contract/schemas.py)
_REPO = repo_root()
# The experiment ABI (contract + capsule corpus) — lives under merlin/ as core infra.
# Paths are resolved repo-root-relative via contract_dir(); no compat symlink.
DEFAULT_CONTRACT_DIR = _REPO / "merlin" / "contract"


class ContractViolation(ValueError):
    """A contract artifact failed schema validation (fail-closed)."""


def contract_dir(override: str | Path | None = None) -> Path:
    """Resolve the contract dir: explicit override > $MERLIN_CONTRACT_DIR > in-repo merlin/contract
    (or, in an installed wheel with no checkout, the bundled ``_data/contract``)."""
    if override:
        return Path(override)
    import os
    env = os.environ.get("MERLIN_CONTRACT_DIR")
    if env:
        return Path(env)
    from merlin.common.paths import data_path
    return data_path("contract")


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


# --- {target}-parameterized generic contracts --------------------------------------------------------
# The two shared ABI contracts (mlir_oot_backend_contract.yaml, oracle_runner_contract.yaml) carry NO
# target literal — they template the target-specific tokens with the ``{target}`` placeholder (the same
# convention their ``--convert-iface-to-{target}`` argv already use). These readers resolve the active
# target at load, mirroring ``generate_prompt`` (``tool_stem = f"{target}-opt"`` / ``kernel_symbol =
# f"{target}_kernel"``): a fixed contract resolves to two targets' values by the same rule, with nothing
# baked in for any one accelerator.


def _resolve_target(obj: Any, target: str) -> Any:
    """Recursively fill the ``{target}`` placeholder token in a loaded contract (structured, no regex)."""
    if isinstance(obj, str):
        return obj.replace("{target}", target)
    if isinstance(obj, list):
        return [_resolve_target(v, target) for v in obj]
    if isinstance(obj, dict):
        return {k: _resolve_target(v, target) for k, v in obj.items()}
    return obj


def render_backend_contract(target: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """The OOT backend contract resolved for ``target`` — ``kernel_abi.symbol`` becomes ``f"{target}_
    kernel"`` and the entrypoint argv templates resolve ``--convert-iface-to-{target}``, exactly the value
    ``generate_prompt`` derives. gemmini resolves byte-identically to the former hand-authored literals."""
    import yaml
    text = (contract_dir(contract) / "mlir_oot_backend_contract.yaml").read_text(encoding="utf-8")
    return _resolve_target(yaml.safe_load(text), target)


def render_oracle_runner_contract(target: str, *, contract: str | Path | None = None) -> dict[str, Any]:
    """The oracle-runner contract resolved for ``target`` — the oracle-ladder level names
    (``spike_{target}_functional`` / ``{target}_verilator_rtl``) and the ``{target}_region`` cycle window
    fill from the active target. gemmini resolves byte-identically to the former hand-authored names."""
    import yaml
    text = (contract_dir(contract) / "oracle_runner_contract.yaml").read_text(encoding="utf-8")
    return _resolve_target(yaml.safe_load(text), target)


def render_contract_text(name: str, target: str, *, contract: str | Path | None = None) -> str:
    """The raw contract text (``name`` is the yaml basename) with every ``{target}`` placeholder filled —
    including ones that live in YAML comments (e.g. the ``merlin.runtime.backends.{target}.parse_output``
    reference), for the agent-facing rendering the structured readers above cannot reach."""
    text = (contract_dir(contract) / name).read_text(encoding="utf-8")
    return text.replace("{target}", target)
