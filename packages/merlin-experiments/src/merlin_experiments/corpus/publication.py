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
