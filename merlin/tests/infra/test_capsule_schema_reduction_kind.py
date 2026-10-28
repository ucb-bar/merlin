"""The capsule schema admits exactly the reduction kinds the generator can record.

corpus_synth records ``generalization.reduction_kind`` for a kind-restricted reduction row, and the capsule's
semantic block carries it to the eligibility oracle. The semantic block forbids unknown properties, so a kind the
schema does not list makes corpus discovery raise after the whole corpus has been written. The schema and the
generator's table are compared here directly rather than trusted to stay in step.
"""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.semantic_families import _OP_REDUCTION_KIND


def _semantic_block() -> dict:
    path = repo_root() / "merlin" / "contract" / "schemas" / "capsule.schema.json"
    if not path.exists():
        pytest.skip("the capsule schema is not in this checkout")
    block = json.loads(path.read_text(encoding="utf-8"))["properties"]["semantic"]
    assert block.get("additionalProperties") is False, "the semantic block no longer closes its properties"
    return block


def test_the_schema_lists_exactly_the_kinds_the_generator_can_record():
    node = _semantic_block()["properties"].get("reduction_kind") or {}
    assert node.get("type") == "string"
    assert set(node.get("enum") or ()) == set(_OP_REDUCTION_KIND.values())


def test_the_property_is_optional_so_non_reduction_capsules_are_unchanged():
    assert "reduction_kind" not in (_semantic_block().get("required") or ())
