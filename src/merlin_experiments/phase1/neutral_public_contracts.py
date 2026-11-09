"""EL1 external protocol documents without the hidden interface input dialect."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

FILES = frozenset({"VERSION", "integrity_policy.md", "command_buffer_abi.yaml",
                   "mlir_oot_backend_contract.yaml", "target_dialect_contract.yaml"})
SCHEMAS = frozenset({"manifest.schema.json", "command_buffer.schema.json",
                     "mixed_program_plan.schema.json", "artifact_bundle.schema.json"})
INPUT_DOCUMENT = "neutral_input_contract.md"


def _neutral(value):
    if isinstance(value, dict):
        return {key: _neutral(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_neutral(item) for item in value]
    if isinstance(value, str):
        return value.replace("merlin_iface", "neutral math input").replace(
            "interface.mlir", "input program").replace("--convert-iface-to-", "--lower-program-to-")
    return value


def prepare(common, *, destination, repo):
    from merlin.targetgen.sandbox.bwrap import resolve_grant

    from .neutral_program_inputs import contract_text
    from .public_runtime_abi import is_projected_protocol

    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    kept, pins, seen = [], [], set()

    def publish(target, raw, alias):
        if b"merlin_iface" in raw:
            raise ValueError("neutral EL1 public protocol contains the withheld input dialect")
        target.write_bytes(raw)
        target.chmod(0o444)
        pins.append(dict(path=str(target.absolute()), sha256=hashlib.sha256(raw).hexdigest(), alias=alias))

    for row in common:
        declared = row["path"]
        if not declared.startswith("merlin/contract/") and not is_projected_protocol(row):
            kept.append(row)
            continue
        name = Path(declared).name
        source = resolve_grant(declared, Path(repo))
        if name in {"interface_dialect_contract.yaml", "interface_grammar.md"}:
            continue
        target = destination / name
        if name == "schemas":
            target.mkdir()
            for schema in sorted(SCHEMAS):
                publish(target / schema, (source / schema).read_bytes(), "schemas/" + schema)
            target.chmod(0o555)
        elif name in FILES:
            raw = source.read_bytes()
            if source.suffix == ".yaml":
                raw = yaml.safe_dump(_neutral(yaml.safe_load(raw)), sort_keys=False).encode()
            publish(target, raw, name)
        else:
            raise ValueError("unclassified public contract in neutral EL1 environment")
        seen.add(name)
        kept.append(dict(path=str(target.absolute()), mode="ro", **{"as": name},
                         note="neutral external compiler/runtime protocol"))
    if seen != FILES | {"schemas"}:
        raise ValueError("neutral EL1 lacks its complete external output protocol")
    publish(destination / INPUT_DOCUMENT, contract_text().encode(), INPUT_DOCUMENT)
    kept.append(dict(path=str((destination / INPUT_DOCUMENT).absolute()), mode="ro",
                     **{"as": INPUT_DOCUMENT}, note="neutral external compiler/runtime protocol"))
    destination.chmod(0o555)
    return kept, pins
