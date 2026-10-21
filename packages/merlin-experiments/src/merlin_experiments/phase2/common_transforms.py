"""Reuse the frozen Phase 1 common services through the original Phase 2 broker."""

from __future__ import annotations

import json
from pathlib import Path

from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

from .broker import BrokerAction
from .contracts import StageGateError

ACTION = "common-transform"
CHECK_ACTION = "common-check"
CHECK_SUBSTRATE = "merlin.common_compiler_substrate.v3"
_CHECK_SUBSTRATES = (
    CHECK_SUBSTRATE,
    "merlin.common_compiler_substrate.v4",
    "merlin.common_compiler_substrate.v5",
    "merlin.common_compiler_substrate.v6",
    "merlin.common_compiler_substrate.v7",
    "merlin.common_compiler_substrate.v8",
)


def action() -> BrokerAction:
    return BrokerAction(
        ACTION,
        ("__host_owned_common_transform__", "{request}"),
        ("request",),
        "apply the inherited common transformations and check their actual IR changes with the "
        "same frozen Phase 1 formal policy and proof toolchain; no simulator or performance claim",
        False,
    )


def check_action() -> BrokerAction:
    return BrokerAction(
        CHECK_ACTION,
        ("__host_owned_common_check__", "{request}"),
        ("request",),
        "check actual before/after IR under the original frozen semantics and formal coverage; "
        "unsupported effects remain unproved; no simulator or performance claim",
        False,
    )


def supports_check(profile) -> bool:
    return isinstance(profile, dict) and profile.get("common_substrate") in _CHECK_SUBSTRATES


def services(frozen_functional, *, candidate: Path, work_root: Path):
    """Share one original frozen owner; requests cannot select its proof policy."""
    from .functional_inputs import _verify_private_functional_provenance

    if frozen_functional.host_provenance is None:
        return None, None
    selected = _verify_private_functional_provenance(frozen_functional.host_provenance)
    if selected["bundle"].get("compiler_tool_profile") is None:
        return None, None
    owner = CommonTransformService(
        workspace=selected["workspace"],
        bundle_manifest=selected["bundle_path"],
        repo=selected["repo"],
    )

    def invoke(request_path: str, timeout_s: int, *, check=False) -> dict:
        path = Path(request_path)
        path = path if path.is_absolute() else candidate / path
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(candidate.resolve()):
            raise StageGateError("common transform request must be an ordinary candidate-owned file")
        if path.stat().st_size > 256 * 1024:
            raise StageGateError("common transform request exceeds its inherited byte budget")
        request = json.loads(path.read_bytes())
        work_root.mkdir(mode=0o700, exist_ok=True)
        method = owner.check if check else owner.apply
        result = method(request, work_root=work_root, timeout_s=timeout_s)
        if check:
            # The original perf transport has a dedicated connection per call;
            # bind the summary to its validated unique request-file identity.
            result = {**result, "selfcheck_request_id": path.name}
        return result

    def check(request_path: str, timeout_s: int) -> dict:
        return invoke(request_path, timeout_s, check=True)

    return invoke, check if supports_check(owner.profile) else None


def service(frozen_functional, *, candidate: Path, work_root: Path):
    """Historical apply-service API retains its original callable contract."""
    return services(frozen_functional, candidate=candidate, work_root=work_root)[0]
