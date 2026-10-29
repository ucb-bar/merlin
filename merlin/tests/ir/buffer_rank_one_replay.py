"""Fresh-process rank-one replay from raw buffers, never a saved acceptance."""

import argparse
import json
import os
from pathlib import Path

from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ALLOCATOR, ROOT, digest, prepare_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lean", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if set(request) != {"buffers", "allocator", "index_bits"}:
        raise ValueError("only actual buffers and public model settings are replay inputs")
    if request["allocator"] != ALLOCATOR or request["index_bits"] != 64 or len(gate._LIVE):
        raise ValueError("fresh empty-ledger process and unchanged public profile required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = prepare_chain(tuple(request["buffers"]), allocator=ALLOCATOR, index_bits=64)
    (output / "INTENT.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "incoming_live_ledger": 0,
                "actual_sha256": [digest(raw) for raw in request["buffers"]],
                "saved_acceptance_consumed": False,
                "prepared": candidate.describe(),
            },
            indent=2,
        )
        + "\n"
    )
    result = gate.check(candidate, lean=args.lean, destination=output / "proof", timeout=300)
    if result.status != "checked" or result.completion is None:
        raise AssertionError((result.status, result.reason))
    received = []
    gate.deliver(result.completion, candidate, received.append)
    if received != [request["buffers"][-1]]:
        raise AssertionError("fresh consumer did not receive exact actual final buffer")
    report = json.loads(Path(result.completion.report).read_text())
    assert ROOT in report["statements"]
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    (output / "delivered.mlir").write_text(received[0])
    (output / "RESULT.json").write_text(
        json.dumps(
            {
                "status": "checked",
                "pid": os.getpid(),
                "incoming_live_ledger": 0,
                "saved_acceptance_consumed": False,
                "final_sha256": digest(received[0]),
                "actual_sha256": [digest(raw) for raw in request["buffers"]],
                "roots": len(candidate.roots),
                "report": result.completion.report,
                "full_error_or_world_equivalence": False,
                "native_allocator_fidelity": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
