"""The agent-visible projection of a public object-build scalability probe.

Only public measurement scalars cross into the verdict: per extent scale, the emit outcome, the build
status, artifact and object sizes and the compile wall time. Nothing here executes code or certifies.
"""

from __future__ import annotations

import math


def projection(observation: dict, *, budget_s: int, candidate_sha256: str) -> dict:
    """Copy only public measurement scalars into the agent-visible verdict."""
    if observation.get("schema") != "merlin.codegen_scalability_probe.v2":
        raise ValueError("public object-build observation has the wrong schema")
    samples = observation.get("samples")
    if not isinstance(samples, list) or len(samples) != 4:
        raise ValueError("public object-build observation has no complete scale roster")
    projected = []
    for row, scale in zip(samples, (1, 2, 4, 8), strict=True):
        build = row.get("build_only") if isinstance(row, dict) else None
        if not isinstance(build, dict) or row.get("extent_scale") != scale:
            raise ValueError("public object-build sample is malformed")
        outcome = row.get("outcome")
        status = build.get("status")
        if outcome not in ("lowered", "empty", "declined", "error") or status not in (
            "compiled",
            "compile_error",
            "not_emitted",
            "selection_changed",
            "source_exceeds_probe_budget",
        ):
            raise ValueError("public object-build status is not recognized")
        item = {"extent_scale": scale, "emit_outcome": outcome, "build_status": status}
        for source, dest in (("artifact_bytes", "emitted_artifact_bytes"), ("object_bytes", "object_bytes")):
            value = build.get(source)
            if type(value) is int and value >= 0:
                item[dest] = value
        elapsed = build.get("compile_wall_s")
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
            item["compile_wall_s"] = elapsed
        projected.append(item)
    return {
        "ran": True,
        "schema": "merlin.public_object_build_feedback.v1",
        "scope": "public_emit_and_build_only_advisory",
        "candidate_sha256": candidate_sha256,
        "per_sample_budget_s": budget_s,
        "samples": projected,
        "note": "Public object compilation only; no execution, numerical result, or certification claim.",
    }
