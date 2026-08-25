"""Anti-drift guard for the dual-maintained `command_buffer` schema.

`merlin/schemas/command_buffer.schema.yaml` is the loose cross-workstream data-model contract;
`merlin/contract/schemas/command_buffer.schema.json` is the strict, fail-closed JSON-Schema
validator that mirrors it. They are two representations of ONE object and must not drift. This test
asserts the machine-checkable invariants (the JSON validator is a consistent SUBSET of the contract):

  - every field the JSON validator *requires* is listed in the YAML contract's
    `required_top_level_fields` (JSON can't require a field the contract dropped / a typo);
  - every property the JSON validator *knows about* is a field the YAML contract documents
    (no field exists in one representation but not the other).
"""
from __future__ import annotations

import json
import re

import yaml

from merlin.common.paths import repo_root

ROOT = repo_root()
YAML_PATH = ROOT / "merlin/schemas/command_buffer.schema.yaml"
JSON_PATH = ROOT / "merlin/contract/schemas/command_buffer.schema.json"


def _yaml_documented_fields() -> set[str]:
    """Top-level field vocabulary the YAML contract documents.

    Three sources, because a field can be part of the contract without appearing in every buffer:
    the `required_top_level_fields` list, the `optional_top_level_fields` list, and the top-level keys
    of EVERY example block. More than one example exists on purpose — a declining buffer has no
    program, so it cannot be shown as a variation of the successful one, and folding `declined` into
    the success example would have documented a buffer that never occurs.
    """
    doc = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    fields = set(doc.get("required_top_level_fields", []) or [])
    fields |= set(doc.get("optional_top_level_fields", []) or [])
    # Each `*example*` is a literal block string of a sample buffer; harvest its top-level keys.
    for key, block in doc.items():
        if "example" not in key or not isinstance(block, str):
            continue
        for m in re.finditer(r'^\s{0,2}"?([a-z_]+)"?\s*:', block, re.MULTILINE):
            fields.add(m.group(1))
    return fields


def test_command_buffer_json_required_subset_of_yaml_contract():
    j = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    yaml_required = set(yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
                        .get("required_top_level_fields", []) or [])
    json_required = set(j.get("required", []))
    missing = json_required - yaml_required
    assert not missing, (
        f"command_buffer JSON requires {sorted(missing)} not in the YAML contract's "
        f"required_top_level_fields — the two schemas drifted; reconcile them.")


def test_command_buffer_json_properties_are_documented():
    j = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    json_props = set(j.get("properties", {}))
    documented = _yaml_documented_fields()
    undocumented = json_props - documented
    assert not undocumented, (
        f"command_buffer JSON has properties {sorted(undocumented)} the YAML contract never "
        f"mentions — schemas drifted; document them in the YAML or drop from the JSON.")
