"""Pinned-Torch child for original-program Q/DQ source reconstruction.

The selected archive/trace are trusted operator inputs, not execution attestation.
This binds source graphs, not universal framework semantics or native execution.
"""

import json
from pathlib import Path


def reconstruct(selection):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- register PT2 operators before loading
    from m2m.capture.trace import snapshot_exported_program

    from merlin.targetgen.qdq_capture import replay
    from merlin.targetgen.qdq_capture_identity import begin, recheck

    directory = Path(selection["directory"])
    identity = begin(Path(selection["producer"]), Path(selection["torch_wheel"]))
    if identity != json.loads((directory / "qdq-producer.json").read_bytes()):
        raise ValueError("selected producer/Torch/tools differ from the source capture")
    program = torch.export.load(directory / "original.pt2")
    original = snapshot_exported_program(program, stage="original")
    trace = json.loads((directory / "qdq-frontend.json").read_bytes())

    def calls(graph):
        # Serialization can populate output-node metadata; exact returned
        # operand identities and every producing-call type remain mandatory.
        return [
            {
                key: node[key]
                for key in ("id", "op", "target", "args", "kwargs", "results")
                if key != "results" or node["op"] != "output"
            }
            for node in graph["nodes"]
        ]

    if calls(original) != calls(trace["graphs"]["original"]):
        raise ValueError("original ExportedProgram differs from the captured source graph")
    result = replay(directory)
    recheck(identity)
    return {
        **result,
        "original_program_bound": True,
        "producer_commit": identity["producer"]["commit"],
        "torch_wheel_sha256": identity["torch"]["wheel_sha256"],
        "native_execution_replayed": False,
        "source_closure_verified": False,
    }
