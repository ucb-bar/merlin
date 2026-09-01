"""Every epilogue a capsule declares must be expressible in the command buffer it is graded on.

Measured on the gemmini arm-4 round merlincirct_arm4_func_20260901_v4: `GP0_matmul_maxpool_i8`,
`GP1` and `GP2` declare `epilogue: ['maxpool']`, while `command_buffer.schema.json` allows only
`{bias_add, bias, requant, acc_scale, relu}` -- `maxpool` appears NOWHERE in that schema. The
interpreter does pool, but through `pool_*` attributes fused into the STORE path, not through an
epilogue stage.

The agent read the capsule's declaration as an instruction, found the schema refused it, invented a
`pool_kind` carrier, and -- because the schema sets `additionalProperties: True` -- that invention
VALIDATED CLEANLY while nothing read it. With no signal that it was writing into a dead field, the
agent concluded in its round report that "no schema-valid equivalent operation exists" and stopped
attempting those capsules for three rounds. The capsules were winnable the whole time.

This test is the gate that was missing: a capsule may not demand an epilogue the command schema
cannot carry. It fails loudly on the disagreement instead of leaving an agent to infer it.
"""
from __future__ import annotations

import json

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCHEMA = merlin_dir() / "contract/schemas/command_buffer.schema.json"
_CAPSULE_ROOTS = ("contract/capsules/isa", "contract/capsules/layers",
                  "contract/capsules/model_slices", "contract/capsules/model")

#: The one disagreement measured live on merlincirct_arm4_func_20260901_v4. MAY ONLY SHRINK -- a new
#: entry here means a capsule was authored demanding a stage no backend can emit.
_KNOWN_DISAGREEMENT = [
    "GP0_matmul_maxpool_i8: epilogue 'maxpool'",
    "GP1_matmul_maxpool_tail_i8: epilogue 'maxpool'",
    "GP2_conv2d_maxpool_i8: epilogue 'maxpool'",
]


def _declared_epilogue_stages() -> set[str]:
    """The epilogue vocabulary the command buffer can actually carry, read from the schema."""
    doc = json.loads(_SCHEMA.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "epilogue" and isinstance(value, dict):
                    items = value.get("items") or {}
                    if isinstance(items, dict) and items.get("enum"):
                        return set(items["enum"])
                found = walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
        return None

    stages = walk(doc)
    assert stages, "command_buffer.schema.json declares no epilogue enum to check against"
    return stages


def _capsules():
    for rel in _CAPSULE_ROOTS:
        root = merlin_dir() / rel
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/capsule.yaml")):
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001 -- a malformed capsule is another test's problem
                continue
            yield path.parent.name, doc


def test_no_capsule_demands_an_epilogue_the_command_buffer_cannot_carry():
    carriable = _declared_epilogue_stages()
    offenders = []
    for name, doc in _capsules():
        stages = ((doc.get("operation") or {}).get("attributes") or {}).get("epilogue") or []
        for stage in stages:
            if str(stage) not in carriable:
                offenders.append(f"{name}: epilogue {stage!r}")
    detail = ("capsule(s) declare an epilogue stage the command buffer schema cannot express, so a "
              "backend cannot emit what the capsule asks for and must guess a carrier:\n  "
              + "\n  ".join(sorted(offenders))
              + f"\ncommand_buffer.schema.json carries: {sorted(carriable)}\n"
              "Either add the stage to the schema's epilogue enum, or express the operation through "
              "the attributes the interpreter actually reads (for pooling: pool_in_dims/pool_size/"
              "pool_stride/pool_padding on the store) and stop declaring it as an epilogue.")
    if offenders == sorted(_KNOWN_DISAGREEMENT):
        # RATCHET, not a pass. This exact disagreement is live and cannot be fixed while a round is
        # grading against the contract -- changing either side mid-run alters what the agent is being
        # graded on. Recorded here so it is visible and so any NEW offender fails immediately.
        pytest.xfail(detail)
    assert not offenders, detail


def test_the_schema_would_reject_an_invented_carrier():
    """`additionalProperties: True` is why a wrong carrier validated silently."""
    doc = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    if doc.get("additionalProperties") is not False:
        pytest.xfail(
            "command_buffer.schema.json sets additionalProperties True, so an invented carrier "
            "(measured: pool_kind=maxpool) validates cleanly while nothing reads it. Tightening "
            "this is a pre-freeze change: it alters what a live run is graded against.")
