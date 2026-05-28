from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import utils

from .model import ArtifactRecord, ClusterRecord, ResourceLease, RunRecord, RunRequest

DEFAULT_STATE_ROOT = "build/generated/ray"
DEFAULT_CLUSTER_NAME = "merlin-local"
DEFAULT_CLUSTER_ADDRESS = "http://127.0.0.1:8265"


def normalize_state_root(base: str | Path | None = None) -> Path:
    path = Path(base or DEFAULT_STATE_ROOT)
    return path if path.is_absolute() else utils.REPO_ROOT / path


def start_local_cluster(
    *,
    state_root: Path,
    host: str = "127.0.0.1",
    port: int = 6379,
    dashboard_port: int = 8265,
    namespace: str = "merlin",
    dry_run: bool = False,
) -> ClusterRecord:
    state_root.mkdir(parents=True, exist_ok=True)
    cluster_dir = state_root / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    ray_binary = shutil.which("ray")
    dashboard_url = f"http://{host}:{dashboard_port}"
    address = f"http://{host}:{dashboard_port}"
    if ray_binary is None:
        record = ClusterRecord(
            cluster_name=DEFAULT_CLUSTER_NAME,
            status="unavailable",
            address=address,
            dashboard_url=dashboard_url,
            namespace=namespace,
            managed_by="merlin-raycp",
            message="`ray` CLI is not installed in the current environment.",
            started_at=None,
            updated_at=_timestamp(),
        )
        _write_json(cluster_dir / "bootstrap.json", asdict(record))
        return record
    command = [
        ray_binary,
        "start",
        "--head",
        "--node-ip-address",
        host,
        "--port",
        str(port),
        "--dashboard-host",
        host,
        "--dashboard-port",
        str(dashboard_port),
    ]
    if dry_run:
        record = ClusterRecord(
            cluster_name=DEFAULT_CLUSTER_NAME,
            status="dry_run",
            address=address,
            dashboard_url=dashboard_url,
            namespace=namespace,
            managed_by="merlin-raycp",
            message="Dry run only. No Ray cluster was started.",
            started_at=None,
            updated_at=_timestamp(),
        )
        _write_json(cluster_dir / "bootstrap.json", asdict(record))
        return record
    result = _run_command(command, utils.REPO_ROOT)
    record = ClusterRecord(
        cluster_name=DEFAULT_CLUSTER_NAME,
        status="active" if result.returncode == 0 else "error",
        address=address,
        dashboard_url=dashboard_url,
        namespace=namespace,
        managed_by="merlin-raycp",
        message=result.stdout.strip() or result.stderr.strip() or None,
        started_at=_timestamp(),
        updated_at=_timestamp(),
    )
    _write_json(cluster_dir / "bootstrap.json", asdict(record))
    return record


def get_cluster_status(*, state_root: Path) -> ClusterRecord:
    bootstrap_path = state_root / "cluster" / "bootstrap.json"
    payload = _load_json(bootstrap_path)
    if payload is not None:
        return ClusterRecord(**payload)
    ray_binary = shutil.which("ray")
    if ray_binary is None:
        return ClusterRecord(
            cluster_name=DEFAULT_CLUSTER_NAME,
            status="unavailable",
            address=DEFAULT_CLUSTER_ADDRESS,
            dashboard_url=DEFAULT_CLUSTER_ADDRESS,
            namespace="merlin",
            managed_by="merlin-raycp",
            message="`ray` CLI is not installed in the current environment.",
            updated_at=_timestamp(),
        )
    return ClusterRecord(
        cluster_name=DEFAULT_CLUSTER_NAME,
        status="not_started",
        address=DEFAULT_CLUSTER_ADDRESS,
        dashboard_url=DEFAULT_CLUSTER_ADDRESS,
        namespace="merlin",
        managed_by="merlin-raycp",
        message="No Ray bootstrap metadata exists yet. Run `merlin ray cluster start-local`.",
        updated_at=_timestamp(),
    )


def stop_local_cluster(*, state_root: Path, dry_run: bool = False) -> ClusterRecord:
    cluster_dir = state_root / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    existing = get_cluster_status(state_root=state_root)
    ray_binary = shutil.which("ray")
    if ray_binary is None:
        existing.status = "unavailable"
        existing.message = "`ray` CLI is not installed in the current environment."
        existing.updated_at = _timestamp()
        _write_json(cluster_dir / "bootstrap.json", asdict(existing))
        return existing
    if dry_run:
        existing.status = "dry_run"
        existing.message = "Dry run only. No Ray cluster was stopped."
        existing.updated_at = _timestamp()
        _write_json(cluster_dir / "bootstrap.json", asdict(existing))
        return existing
    result = _run_command([ray_binary, "stop", "--force"], utils.REPO_ROOT)
    existing.status = "stopped" if result.returncode == 0 else "error"
    existing.message = result.stdout.strip() or result.stderr.strip() or None
    existing.updated_at = _timestamp()
    _write_json(cluster_dir / "bootstrap.json", asdict(existing))
    return existing


def submit_job(
    *,
    state_root: Path,
    target: str,
    workflow: str,
    source: str,
    command: list[str] | str,
    target_dir: Path,
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> RunRecord:
    state_root.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id(target)
    command_list = shlex.split(command) if isinstance(command, str) else command
    request = RunRequest(
        run_id=run_id,
        target=target,
        workflow=workflow,
        source=source,
        command=command_list,
        workdir=str(utils.REPO_ROOT),
        target_dir=str(target_dir),
        metadata=metadata or {},
        created_at=_timestamp(),
    )
    run_dir = _run_dir(state_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_request.json", asdict(request))

    cluster = get_cluster_status(state_root=state_root)
    ray_binary = shutil.which("ray")
    if ray_binary is None:
        record = RunRecord(
            run_id=run_id,
            target=target,
            workflow=workflow,
            source=source,
            engine="ray",
            status="blocked",
            target_dir=str(target_dir),
            workdir=str(utils.REPO_ROOT),
            command=command_list,
            cluster_address=cluster.address,
            message="`ray` CLI is not installed. Install Ray and start a local cluster first.",
            metadata=metadata or {},
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
        _write_json(run_dir / "run_record.json", asdict(record))
        return record
    if cluster.status not in {"active", "dry_run"}:
        record = RunRecord(
            run_id=run_id,
            target=target,
            workflow=workflow,
            source=source,
            engine="ray",
            status="blocked",
            target_dir=str(target_dir),
            workdir=str(utils.REPO_ROOT),
            command=command_list,
            cluster_address=cluster.address,
            message="Ray cluster is not active. Run `merlin ray cluster start-local` first.",
            metadata=metadata or {},
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
        _write_json(run_dir / "run_record.json", asdict(record))
        return record
    submit_command = [
        ray_binary,
        "job",
        "submit",
        "--address",
        cluster.address or DEFAULT_CLUSTER_ADDRESS,
        "--submission-id",
        run_id,
        "--working-dir",
        str(utils.REPO_ROOT),
        "--",
        *command_list,
    ]
    if dry_run:
        record = RunRecord(
            run_id=run_id,
            target=target,
            workflow=workflow,
            source=source,
            engine="ray",
            status="dry_run",
            target_dir=str(target_dir),
            workdir=str(utils.REPO_ROOT),
            command=command_list,
            cluster_address=cluster.address,
            submission_id=run_id,
            message="Dry run only. No Ray Job was submitted.",
            metadata=metadata or {},
            created_at=_timestamp(),
            updated_at=_timestamp(),
        )
        _write_json(run_dir / "run_record.json", asdict(record))
        return record
    result = _run_command(submit_command, utils.REPO_ROOT)
    record = RunRecord(
        run_id=run_id,
        target=target,
        workflow=workflow,
        source=source,
        engine="ray",
        status="submitted" if result.returncode == 0 else "error",
        target_dir=str(target_dir),
        workdir=str(utils.REPO_ROOT),
        command=command_list,
        cluster_address=cluster.address,
        submission_id=run_id,
        message=result.stdout.strip() or result.stderr.strip() or None,
        metadata=metadata or {},
        created_at=_timestamp(),
        updated_at=_timestamp(),
    )
    _write_json(run_dir / "run_record.json", asdict(record))
    return record


def get_run_record(*, state_root: Path, run_id: str) -> RunRecord:
    payload = _load_json(_run_dir(state_root, run_id) / "run_record.json")
    if payload is None:
        raise ValueError(f"Unknown run id: {run_id}")
    record = RunRecord(**payload)
    if record.submission_id and shutil.which("ray"):
        cluster = get_cluster_status(state_root=state_root)
        status_command = [
            shutil.which("ray") or "ray",
            "job",
            "status",
            record.submission_id,
            "--address",
            cluster.address or DEFAULT_CLUSTER_ADDRESS,
        ]
        result = _run_command(status_command, utils.REPO_ROOT, check=False)
        record.status = _parse_job_status(result.stdout or result.stderr, record.status)
        record.updated_at = _timestamp()
        _write_json(_run_dir(state_root, run_id) / "run_record.json", asdict(record))
        if record.status == "completed":
            build_artifact_index(state_root=state_root, run_id=run_id)
    return record


def get_run_logs(*, state_root: Path, run_id: str) -> str:
    record = get_run_record(state_root=state_root, run_id=run_id)
    if not record.submission_id or not shutil.which("ray"):
        return record.message or ""
    cluster = get_cluster_status(state_root=state_root)
    log_command = [
        shutil.which("ray") or "ray",
        "job",
        "logs",
        record.submission_id,
        "--address",
        cluster.address or DEFAULT_CLUSTER_ADDRESS,
    ]
    result = _run_command(log_command, utils.REPO_ROOT, check=False)
    return result.stdout or result.stderr or ""


def cancel_run(*, state_root: Path, run_id: str, dry_run: bool = False) -> RunRecord:
    record = get_run_record(state_root=state_root, run_id=run_id)
    if not record.submission_id or not shutil.which("ray"):
        record.status = "blocked"
        record.message = record.message or "Ray job cannot be cancelled because it was never submitted."
        record.updated_at = _timestamp()
        _write_json(_run_dir(state_root, run_id) / "run_record.json", asdict(record))
        return record
    if dry_run:
        record.status = "dry_run"
        record.message = "Dry run only. No Ray Job was cancelled."
        record.updated_at = _timestamp()
        _write_json(_run_dir(state_root, run_id) / "run_record.json", asdict(record))
        return record
    cluster = get_cluster_status(state_root=state_root)
    stop_command = [
        shutil.which("ray") or "ray",
        "job",
        "stop",
        record.submission_id,
        "--address",
        cluster.address or DEFAULT_CLUSTER_ADDRESS,
    ]
    result = _run_command(stop_command, utils.REPO_ROOT, check=False)
    record.status = "cancelled" if result.returncode == 0 else "error"
    record.message = result.stdout.strip() or result.stderr.strip() or None
    record.updated_at = _timestamp()
    _write_json(_run_dir(state_root, run_id) / "run_record.json", asdict(record))
    return record


def build_artifact_index(*, state_root: Path, run_id: str) -> list[ArtifactRecord]:
    record = get_run_record(state_root=state_root, run_id=run_id)
    target_dir = Path(record.target_dir)
    artifacts: list[ArtifactRecord] = []
    if target_dir.exists():
        for item in sorted(target_dir.rglob("*")):
            if not item.is_file():
                continue
            artifacts.append(
                ArtifactRecord(
                    run_id=run_id,
                    name=item.relative_to(target_dir).as_posix(),
                    path=str(item),
                    kind=item.suffix.lstrip(".") or "file",
                    size_bytes=item.stat().st_size,
                    created_at=_timestamp(),
                )
            )
    _write_json(_run_dir(state_root, run_id) / "artifacts.json", [asdict(item) for item in artifacts])
    return artifacts


def list_artifacts(*, state_root: Path, run_id: str) -> list[ArtifactRecord]:
    artifacts_path = _run_dir(state_root, run_id) / "artifacts.json"
    payload = _load_json(artifacts_path)
    if payload is None:
        return build_artifact_index(state_root=state_root, run_id=run_id)
    return [ArtifactRecord(**item) for item in payload]


def fetch_artifact(*, state_root: Path, run_id: str, artifact_name: str) -> ArtifactRecord:
    artifacts = list_artifacts(state_root=state_root, run_id=run_id)
    for item in artifacts:
        if item.name == artifact_name or Path(item.path).name == artifact_name:
            return item
    raise ValueError(f"Unknown artifact {artifact_name!r} for run {run_id}")


def list_resource_leases(*, state_root: Path) -> list[ResourceLease]:
    leases_dir = state_root / "resources" / "leases"
    if not leases_dir.exists():
        return []
    leases: list[ResourceLease] = []
    for item in sorted(leases_dir.glob("*.json")):
        payload = _load_json(item)
        if payload is None:
            continue
        leases.append(ResourceLease(**payload))
    return leases


def reserve_resource(
    *,
    state_root: Path,
    resource_type: str,
    owner: str,
    metadata: dict[str, Any] | None = None,
) -> ResourceLease:
    leases_dir = state_root / "resources" / "leases"
    leases_dir.mkdir(parents=True, exist_ok=True)
    for lease in list_resource_leases(state_root=state_root):
        if lease.resource_type == resource_type and lease.status == "active":
            raise ValueError(f"Resource {resource_type!r} already has an active lease.")
    lease = ResourceLease(
        lease_id=f"lease-{uuid.uuid4().hex[:10]}",
        resource_type=resource_type,
        owner=owner,
        status="active",
        metadata=metadata or {},
        created_at=_timestamp(),
        updated_at=_timestamp(),
    )
    _write_json(leases_dir / f"{lease.lease_id}.json", asdict(lease))
    return lease


def release_resource(*, state_root: Path, lease_id: str) -> ResourceLease:
    lease_path = state_root / "resources" / "leases" / f"{lease_id}.json"
    payload = _load_json(lease_path)
    if payload is None:
        raise ValueError(f"Unknown lease id: {lease_id}")
    lease = ResourceLease(**payload)
    lease.status = "released"
    lease.updated_at = _timestamp()
    _write_json(lease_path, asdict(lease))
    return lease


def _run_dir(state_root: Path, run_id: str) -> Path:
    return state_root / "runs" / run_id


def _new_run_id(target: str) -> str:
    return f"{target}-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _parse_job_status(raw: str, fallback: str) -> str:
    text = raw.upper()
    if "SUCCEEDED" in text:
        return "completed"
    if "FAILED" in text:
        return "failed"
    if "RUNNING" in text:
        return "running"
    if "PENDING" in text:
        return "submitted"
    if "STOPPED" in text or "CANCELLED" in text:
        return "cancelled"
    return fallback


def _run_command(command: list[str], workdir: Path, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=workdir,
        check=check,
        capture_output=True,
        text=True,
    )


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()
