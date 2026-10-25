"""Reject held-out source identities from public release and authoring inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..phase0.claim_boundary import assert_no_claim_capsules, held_out_models
from ..spec import SpecError


def shape_signature(document) -> str | None:
    """Identify an operation's role-to-shape mapping, independent of spelling.

    Attributes (including default/empty epilogues), tensor names, dtype and
    numerical policies cannot change a held-out shape's publication identity.
    Repeated roles retain their complete shape multiset. Missing roles or
    unresolved shapes have no established identity.
    """
    operation = document.get("operation") if isinstance(document, dict) else None
    inputs = document.get("inputs") if isinstance(document, dict) else None
    if not isinstance(operation, dict) or not operation.get("op") or not isinstance(inputs, list) or not inputs:
        return None
    operands = []
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"]:
            return None
        shape = item.get("shape")
        if not isinstance(shape, list) or not shape or any(type(size) is not int or size <= 0 for size in shape):
            return None
        operands.append((item["role"], shape))
    body = {"op": operation["op"], "operands": sorted(operands)}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


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
