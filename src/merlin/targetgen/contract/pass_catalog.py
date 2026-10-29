"""Load static pass requirements; loading never evaluates IR or runs a verifier."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import ContractViolation, contract_dir, validate


class _CatalogLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ContractViolation(f"pass catalog contains a non-string or duplicate key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_CatalogLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_catalog(path: Path | None = None, *, contract: Path | None = None) -> dict:
    """Validate schema and unique role identities, preserving unknown-policy prose."""
    source = path if path is not None else contract_dir(contract) / "pass_catalog.yaml"
    try:
        document = yaml.load(source.read_text(encoding="utf-8"), Loader=_CatalogLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ContractViolation(f"cannot read pass catalog {source}: {error}") from error
    validate(document, "pass_catalog", contract=contract)
    ids = [role["id"] for role in document["roles"]]
    if len(ids) != len(set(ids)):
        raise ContractViolation("pass catalog contains duplicate role IDs")
    return document
