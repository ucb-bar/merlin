"""Reuse the frozen Phase 1 common transform service through the Phase 2 broker."""

from __future__ import annotations

import json
from pathlib import Path

from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

from .broker import BrokerAction
from .contracts import StageGateError

ACTION = "common-transform"


def action() -> BrokerAction:
    return BrokerAction(
        ACTION,
        ("__host_owned_common_transform__", "{request}"),
        ("request",),
        "apply the inherited common transformations and check their actual IR changes with the "
        "same frozen Phase 1 formal policy and proof toolchain; no simulator or performance claim",
        False,
    )


def service(frozen_functional, *, candidate: Path, work_root: Path):
    """Return no capability for historical runs; otherwise reuse exact original authority."""
    from .functional_inputs import _verify_private_functional_provenance

    if frozen_functional.host_provenance is None:
        return None
    selected = _verify_private_functional_provenance(frozen_functional.host_provenance)
    if selected["bundle"].get("compiler_tool_profile") is None:
        return None
    owner = CommonTransformService(
        workspace=selected["workspace"],
        bundle_manifest=selected["bundle_path"],
        repo=selected["repo"],
    )

    def invoke(request_path: str, timeout_s: int) -> dict:
        path = Path(request_path)
        path = path if path.is_absolute() else candidate / path
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(candidate.resolve()):
            raise StageGateError("common transform request must be an ordinary candidate-owned file")
        if path.stat().st_size > 256 * 1024:
            raise StageGateError("common transform request exceeds its inherited byte budget")
        request = json.loads(path.read_bytes())
        work_root.mkdir(mode=0o700, exist_ok=True)
        return owner.apply(request, work_root=work_root, timeout_s=timeout_s)

    return invoke
