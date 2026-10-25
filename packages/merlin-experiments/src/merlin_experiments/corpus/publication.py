"""Reject held-out source identities from public release and authoring inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..phase0.claim_boundary import assert_no_claim_capsules, held_out_models
from ..spec import SpecError


def shape_signature(document) -> str | None:
    """Digest of the operation and operand shapes a capsule exposes, independent of name, dtype and numerics.

    Operand names are replaced by their position so renaming a tensor cannot change the digest; the
    capsule name, labels, prose provenance, dtypes and numeric policy are excluded, so a
    held-out shape cannot be republished under a different quantization scheme. ``None`` means the capsule declares no
    operation or operands, so its identity cannot be established.
    """
    operation = document.get("operation") if isinstance(document, dict) else None
    inputs = document.get("inputs") if isinstance(document, dict) else None
    if not isinstance(operation, dict) or not operation.get("op") or not isinstance(inputs, list) or not inputs:
        return None
    if any(not isinstance(item, dict) for item in inputs):
        return None
    positions = {item.get("name"): f"${index}" for index, item in enumerate(inputs) if item.get("name")}

    def canonical(value):
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items() if k != "out" and not k.endswith("dtype")}
        if isinstance(value, list):
            return [canonical(v) for v in value]
        return positions.get(value, value) if isinstance(value, str) else value

    body = {
        "operation": canonical(operation),
        "inputs": [{k: item.get(k) for k in ("role", "shape")} for item in inputs],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _excluded_shape_signatures(experiment) -> set[str]:
    """Semantic digests of every held-out capsule; exclusion binds to content, not mutable names."""
    source_root = getattr(experiment, "source_root", None)
    if not getattr(experiment, "held_out_inputs", ()):
        return set()
    if source_root is None:
        from merlin.common.paths import repo_root

        source_root = repo_root()
    digests = set()
    for relative in experiment.held_out_inputs:
        for path in (Path(source_root) / relative).rglob("capsule.yaml"):
            digest = shape_signature(yaml.safe_load(path.read_bytes()))
            if digest is None:
                raise SpecError("excluded historical capsule declaration is malformed")
            digests.add(digest)
    return digests


def require_public_capsules(experiment, roots) -> None:
    """Inspect only selected public capsule metadata, never hidden source tensors."""
    held_out = held_out_models(experiment)
    excluded = _excluded_shape_signatures(experiment)
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            document = yaml.safe_load(path.read_bytes())
            if not isinstance(document, dict):
                raise SpecError("public capsule declaration is malformed")
            digest = shape_signature(document)
            if excluded and digest is None and document.get("source_role") == "model_derived":
                raise SpecError("public model-derived capsule cannot be identified against the held-out inputs")
            if digest in excluded:
                raise SpecError(
                    "public corpus contains an explicitly excluded historical input; use generated-only preparation"
                )
            operation = document.get("operation") or {}
            if not isinstance(operation, dict):
                raise SpecError("public capsule operation declaration is malformed")
            attributes = operation.get("attributes") or {}
            capture = document.get("materialized_capture") or {}
            reference = document.get("pytorch_ref") or {}
            if any(not isinstance(value, dict) for value in (attributes, capture, reference)):
                raise SpecError("public capsule source declaration is malformed")
            identities = [
                document.get("model"),
                attributes.get("model"),
                reference.get("model"),
                capture.get("workload_id"),
                capture.get("source_workload_id"),
            ]
            rows = [document, *({**document, "model": identity} for identity in identities if identity)]
            try:
                assert_no_claim_capsules(rows, held_out)
            except ValueError:
                raise SpecError(
                    "public corpus contains a held-out validation source; prepare a generated-only release"
                ) from None


#: Labels of the cohort an author's compiler is graded on in the open.
PUBLIC_LABELS = frozenset({"public", "dev"})


def require_accelerator_placement(roots) -> int:
    """Refuse any public capsule whose frozen Phase 0 placement puts nothing on the accelerator.

    Host placement is compiled by Merlin's own generic host codegen, so such a capsule would grade none
    of the author's work. The placement read is the one Phase 0 froze into ``software_screen``; a
    capsule without that record is refused as well, because its absence is not evidence of accelerator
    work. An unscored host guard is admitted: it is host-placed by design and grades no accelerator
    work. Returns the number of public capsules checked.
    """
    from merlin.targetgen.capsule_common import is_unscored_host_guard

    from ..phase0.program_admission import accelerator_placement_refusal

    refused, checked = [], 0
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            document = yaml.safe_load(path.read_bytes())
            if not isinstance(document, dict):
                raise SpecError("public capsule declaration is malformed")
            if document.get("label") not in PUBLIC_LABELS:
                continue
            checked += 1
            if is_unscored_host_guard(document):
                continue
            reason = accelerator_placement_refusal(document.get("software_screen"))
            if reason is not None:
                refused.append(f"{document.get('name') or path.parent.name}: {reason}")
    if refused:
        raise SpecError(
            f"{len(refused)} public capsule(s) carry no accelerator placement; public capsules grade the "
            "author's accelerator compiler only, and host placement is Merlin-supplied: " + "; ".join(refused)
        )
    return checked
