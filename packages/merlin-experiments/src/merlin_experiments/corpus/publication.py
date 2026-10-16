"""Reject held-out source identities from public release and authoring inputs."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..phase0.claim_boundary import assert_no_claim_capsules, held_out_models
from ..spec import SpecError


def require_public_capsules(experiment, roots) -> None:
    """Inspect only selected public capsule metadata, never hidden source tensors."""
    held_out = held_out_models(experiment)
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            document = yaml.safe_load(path.read_bytes())
            if not isinstance(document, dict):
                raise SpecError("public capsule declaration is malformed")
            operation = document.get("operation") or {}
            if not isinstance(operation, dict):
                raise SpecError("public capsule operation declaration is malformed")
            attributes = operation.get("attributes") or {}
            capture = document.get("materialized_capture") or {}
            reference = document.get("pytorch_ref") or {}
            if any(not isinstance(value, dict) for value in (attributes, capture, reference)):
                raise SpecError("public capsule source declaration is malformed")
            identities = [document.get("model"), attributes.get("model"), reference.get("model"),
                          capture.get("workload_id"), capture.get("source_workload_id")]
            rows = [document, *({**document, "model": identity} for identity in identities if identity)]
            try:
                assert_no_claim_capsules(rows, held_out)
            except ValueError:
                raise SpecError(
                    "public corpus contains a held-out validation source; prepare a generated-only release"
                ) from None
