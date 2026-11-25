"""Non-executable custody fixtures, never numerical/runtime qualification."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from merlin.common.paths import runtime_dir
from merlin.llvmlower.post_round_clamp import Contract
from merlin.llvmlower.post_round_clamp_guard import CalleePolicy

_custody = tempfile.TemporaryDirectory(prefix="post-round-custody-", dir=os.environ.get("TMPDIR"))
ROOT = Path(_custody.name)


def pin(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


SOURCE = runtime_dir() / "abi/mlir_runtime.c"
SUPPLIER = ROOT / "mlir_rt.o"
SUPPLIER.write_bytes(b"non-executable unit-test custody envelope; not a runtime implementation\n")
proof = ROOT / "proof.json"
proof.write_text(
    json.dumps(
        {
            "scope": (
                "Synthetic typed-custody/negative unit fixtures only. Actual semantics have "
                "separate native/RV/fullforward evidence."
            )
        }
    )
)
recipe = ROOT / "recipe.json"
recipe.write_text(
    json.dumps(
        {
            "commands": [
                {
                    "requested_output": str(SUPPLIER),
                    "output": pin(SUPPLIER),
                    "inputs": [pin(SOURCE)],
                    "argv": ["nonexecuted-test-compiler", "-c", str(SOURCE), "-o", str(SUPPLIER)],
                }
            ]
        }
    )
)
policy = ROOT / "callee-policy.json"
policy.write_text(
    json.dumps(
        {
            "schema": "merlin.host_value_only_callee_policy.v1",
            "caller_premise": "closed_freestanding_value_only_no_fenv_errno_observation",
            "approval_scope": "explicit_trusted_provider_policy_not_inferred",
            "fenv_observed": False,
            "errno_observed": False,
            "external_callees": ["roundevenf"],
            "bindings": [
                {
                    "symbol": "roundevenf",
                    "actual_symbol": "roundevenf",
                    "source_pins": [pin(SOURCE)],
                    "object_pins": [pin(SUPPLIER)],
                    "effect_evidence_pins": [pin(proof)],
                }
            ],
        }
    )
)
BASE = Contract(
    "merlin_ieee_words_binary32_nearest_ties_even_v1",
    str(SUPPLIER),
    pin(SUPPLIER)["sha256"],
    str(SOURCE),
    pin(SOURCE)["sha256"],
    str(recipe),
    pin(recipe)["sha256"],
    False,
    False,
    True,
    "llvm_fptosi_poison",
    "rv64gc",
    CalleePolicy(str(policy), pin(policy)["sha256"]),
)
(ROOT / "contract.json").write_text(json.dumps(asdict(BASE)))

ASSEMBLER = ROOT / "assembler"
ASSEMBLER.write_bytes(b"non-executed unit-test assembler custody fixture\n")
