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


# ------------------------------------------------------------------ epilogue vocabulary (one source)
def _epilogue_gate():
    """The gate module, imported from build_tools (it is a script, not a package)."""
    import importlib.util
    path = ROOT / "build_tools" / "scripts" / "check_epilogue_vocabulary.py"
    spec = importlib.util.spec_from_file_location("_check_epilogue_vocabulary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_epilogue_vocabulary_has_exactly_one_definition():
    """The COMMIT epilogue vocabulary is defined once and derived everywhere else.

    Six representations used to be hand-maintained and no two agreed: the strict JSON validator rejected
    `maxpool` (which both ABI documents instruct an author to emit and all three engines implement), and
    the three dialect verifiers rejected `acc_scale` (which the validator and both documents admit). The
    first of those failed a real capsule with `command_buffer schema violation ... 'maxpool' is not one
    of [...]` for following the documented ABI.
    """
    problems = _epilogue_gate().check()
    assert not problems, "epilogue vocabulary drifted:\n" + "\n".join(f"  - {p}" for p in problems)


def test_epilogue_gate_fails_closed_when_it_cannot_find_the_enum():
    """A gate that could not run must NOT report success (recurring defect in this repo)."""
    gate = _epilogue_gate()
    assert gate.schema_enum({}) is None
    assert gate.schema_enum({"properties": {"commands": {}}}) is None


def test_every_per_opcode_epilogue_list_is_a_subset_of_the_vocabulary():
    """A per-opcode line documents FEWER stages (an engine that implements fewer for that opcode); it may
    never introduce a stage the ABI-wide vocabulary does not admit."""
    gate = _epilogue_gate()
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES
    for path, _full_key in gate._ABI_DOCS:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        lines = gate.abi_epilogue_lines(doc)
        assert lines, f"{path.name}: no epilogue line found — cannot certify"
        for keys, names in lines:
            extra = sorted(set(names) - set(EPILOGUE_STAGES))
            assert not extra, f"{path.name}: {'/'.join(keys)} documents unknown stage(s) {extra}"
