"""Every source_role the holdout generator can emit must be one the capsule schema admits.

A revealed capsule that cannot validate fails the run at the REVEAL step -- after every candidate
has already sealed and every trial's compute has been spent. Measured: a campaign completed three
agent trials and three functional regrades, then died with

    NO-GO: ContractViolation: capsule 'PKH00_k26': capsule schema violation at source_role:
    'generated_seeded_holdout' is not one of [... 'derived_sweep']

because the generator wrote a role the contract does not define. The two are compared here directly
rather than trusted to stay in step.
"""
from __future__ import annotations

import ast
import json

import pytest

from merlin.common.paths import repo_root


def _schema_roles() -> set[str]:
    path = repo_root() / "merlin" / "contract" / "schemas" / "capsule.schema.json"
    if not path.exists():
        pytest.skip("the capsule schema is not in this checkout")
    schema = json.loads(path.read_text(encoding="utf-8"))
    node = (schema.get("properties") or {}).get("source_role") or {}
    roles = node.get("enum")
    assert roles, "the capsule schema no longer constrains source_role; this test is now vacuous"
    return set(roles)


def _emitted_roles() -> set[str]:
    """Every string literal the generator assigns to a source_role key, read structurally."""
    path = (repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
            / "perf_holdout_corpus.py")
    if not path.exists():
        pytest.skip("the holdout generator is not in this checkout")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            # `x["source_role"] = "..."` and `x["a"]["source_role"] = "..."`
            if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "source_role"):
                found.add(node.value.value)
    assert found, "no source_role assignment found; the generator changed shape and this test is blind"
    return found


def test_the_holdout_generator_only_emits_roles_the_schema_defines():
    emitted, allowed = _emitted_roles(), _schema_roles()
    stray = emitted - allowed
    assert not stray, (
        f"the holdout generator emits source_role(s) the capsule schema rejects: {sorted(stray)}; "
        f"a revealed capsule that cannot validate fails the run AFTER every candidate has sealed. "
        f"schema admits: {sorted(allowed)}")
