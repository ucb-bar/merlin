from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ClusterRecord:
    cluster_name: str
    status: str
    address: str | None
    dashboard_url: str | None
    namespace: str
    managed_by: str
    message: str | None = None
    started_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class RunRequest:
    run_id: str
    target: str
    workflow: str
    source: str
    command: list[str]
    workdir: str
    target_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class RunRecord:
    run_id: str
    target: str
    workflow: str
    source: str
    engine: str
    status: str
    target_dir: str
    workdir: str
    command: list[str]
    cluster_address: str | None = None
    submission_id: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class ArtifactRecord:
    run_id: str
    name: str
    path: str
    kind: str
    size_bytes: int
    created_at: str | None = None


@dataclass(slots=True)
class ResourceLease:
    lease_id: str
    resource_type: str
    owner: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
