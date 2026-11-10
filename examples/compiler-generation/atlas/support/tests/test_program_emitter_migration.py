"""Static provenance checks only: never import Torch/model ISA or execute the helper."""

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_emitter_is_byte_identical_to_recorded_main_source():
    record = json.loads((ROOT / "program_emitter_migration.json").read_text())
    [source] = record["files"]
    payload = (ROOT / source["destination"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == source["source_sha256"] == source["destination_sha256"]
    tree = ast.parse(payload)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"_install_itype_shim", "_layout_inputs", "main"} <= functions


def test_contract_delta_is_only_explicit_target_owned_emitter_declaration():
    record = json.loads((ROOT / "program_emitter_migration.json").read_text())["contract_delta"]
    payload = (ROOT / record["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["destination_sha256"]
    addition = b"  program_emitter:\n    path: atlas_program_emit.py\n    args: [--fix-itype-rd]\n"
    assert payload.count(addition) == 1
    assert hashlib.sha256(payload.replace(addition, b"")).hexdigest() == record["source_sha256"]
    contract = yaml.safe_load(payload)
    assert contract["runner"]["program_emitter"] == {
        "path": "atlas_program_emit.py", "args": ["--fix-itype-rd"]
    }
    original = json.loads((ROOT / "provenance.json").read_text())
    original_contract = next(row for row in original["files"] if row["path"] == record["path"])
    assert original_contract["sha256"] == record["source_sha256"]
