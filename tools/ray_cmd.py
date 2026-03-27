#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from raycp import (
    cancel_run,
    fetch_artifact,
    get_cluster_status,
    get_run_logs,
    get_run_record,
    list_artifacts,
    list_resource_leases,
    normalize_state_root,
    release_resource,
    reserve_resource,
    start_local_cluster,
    stop_local_cluster,
    submit_job,
)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-root",
        default="build/generated/ray",
        help="Directory for Merlin-owned Ray cluster, run, artifact, and lease metadata.",
    )
    subparsers = parser.add_subparsers(dest="ray_command", required=True)

    cluster_parser = subparsers.add_parser("cluster", help="Manage the local Ray cluster bootstrap for Merlin")
    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_command", required=True)
    start_parser = cluster_subparsers.add_parser("start-local", help="Start a local Ray head node")
    start_parser.add_argument("--host", default="127.0.0.1", help="Ray head node bind address")
    start_parser.add_argument("--port", type=int, default=6379, help="Ray head node port")
    start_parser.add_argument("--dashboard-port", type=int, default=8265, help="Ray dashboard and job server port")
    start_parser.add_argument("--namespace", default="merlin", help="Logical namespace for Merlin jobs")
    start_parser.set_defaults(_handler=cmd_cluster_start_local)
    cluster_subparsers.add_parser("status", help="Show the current local Ray cluster state").set_defaults(
        _handler=cmd_cluster_status
    )
    cluster_subparsers.add_parser("stop", help="Stop the local Ray cluster managed by Merlin").set_defaults(
        _handler=cmd_cluster_stop
    )

    jobs_parser = subparsers.add_parser("jobs", help="Submit and inspect Ray-backed Merlin jobs")
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command", required=True)
    submit_parser = jobs_subparsers.add_parser("submit", help="Submit a command as a Ray Job")
    submit_parser.add_argument("--target", required=True, help="Logical target name associated with the run")
    submit_parser.add_argument("--workflow", required=True, help="Workflow name for the run record")
    submit_parser.add_argument("--source", default="manual_cli", help="Source that initiated the run")
    submit_parser.add_argument("--target-dir", required=True, help="Target artifact directory for the run")
    submit_parser.add_argument("--command", required=True, help="Shell-style command string to run via Ray Jobs")
    submit_parser.set_defaults(_handler=cmd_jobs_submit)
    status_parser = jobs_subparsers.add_parser("status", help="Show a Merlin run record and current Ray job status")
    status_parser.add_argument("run_id", help="Run id to inspect")
    status_parser.set_defaults(_handler=cmd_jobs_status)
    logs_parser = jobs_subparsers.add_parser("logs", help="Show Ray job logs for a Merlin run")
    logs_parser.add_argument("run_id", help="Run id to inspect")
    logs_parser.set_defaults(_handler=cmd_jobs_logs)
    cancel_parser = jobs_subparsers.add_parser("cancel", help="Cancel a Ray job for a Merlin run")
    cancel_parser.add_argument("run_id", help="Run id to cancel")
    cancel_parser.set_defaults(_handler=cmd_jobs_cancel)

    resources_parser = subparsers.add_parser("resources", help="Manage Merlin resource leases")
    resources_subparsers = resources_parser.add_subparsers(dest="resources_command", required=True)
    resources_subparsers.add_parser("list", help="List active and released resource leases").set_defaults(
        _handler=cmd_resources_list
    )
    reserve_parser = resources_subparsers.add_parser("reserve", help="Create a new resource lease")
    reserve_parser.add_argument("resource_type", help="Specific resource key to reserve, such as firesim_u250")
    reserve_parser.add_argument("--owner", required=True, help="Lease owner identifier")
    reserve_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional KEY=VALUE metadata attached to the lease",
    )
    reserve_parser.set_defaults(_handler=cmd_resources_reserve)
    release_parser = resources_subparsers.add_parser("release", help="Release an existing resource lease")
    release_parser.add_argument("lease_id", help="Lease id to release")
    release_parser.set_defaults(_handler=cmd_resources_release)

    artifacts_parser = subparsers.add_parser("artifacts", help="Inspect artifacts captured for Merlin Ray runs")
    artifacts_subparsers = artifacts_parser.add_subparsers(dest="artifacts_command", required=True)
    artifacts_list_parser = artifacts_subparsers.add_parser("list", help="List artifacts for a run")
    artifacts_list_parser.add_argument("run_id", help="Run id to inspect")
    artifacts_list_parser.set_defaults(_handler=cmd_artifacts_list)
    artifacts_fetch_parser = artifacts_subparsers.add_parser("fetch", help="Resolve one artifact for a run")
    artifacts_fetch_parser.add_argument("run_id", help="Run id to inspect")
    artifacts_fetch_parser.add_argument("artifact_name", help="Relative artifact name or basename")
    artifacts_fetch_parser.set_defaults(_handler=cmd_artifacts_fetch)


def main(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    if handler is None:
        return 2
    return int(handler(args))


def cmd_cluster_start_local(args: argparse.Namespace) -> int:
    record = start_local_cluster(
        state_root=_state_root(args),
        host=args.host,
        port=args.port,
        dashboard_port=args.dashboard_port,
        namespace=args.namespace,
        dry_run=getattr(args, "dry_run", False),
    )
    print(f"Cluster: {record.cluster_name}")
    print(f"Status: {record.status}")
    print(f"Address: {record.address or 'none'}")
    if record.dashboard_url:
        print(f"Dashboard: {record.dashboard_url}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_cluster_status(args: argparse.Namespace) -> int:
    record = get_cluster_status(state_root=_state_root(args))
    print(f"Cluster: {record.cluster_name}")
    print(f"Status: {record.status}")
    print(f"Address: {record.address or 'none'}")
    if record.dashboard_url:
        print(f"Dashboard: {record.dashboard_url}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_cluster_stop(args: argparse.Namespace) -> int:
    record = stop_local_cluster(
        state_root=_state_root(args),
        dry_run=getattr(args, "dry_run", False),
    )
    print(f"Cluster: {record.cluster_name}")
    print(f"Status: {record.status}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_jobs_submit(args: argparse.Namespace) -> int:
    record = submit_job(
        state_root=_state_root(args),
        target=args.target,
        workflow=args.workflow,
        source=args.source,
        command=args.command,
        target_dir=_resolve_path(args.target_dir),
        dry_run=getattr(args, "dry_run", False),
    )
    print(f"Run ID: {record.run_id}")
    print(f"Status: {record.status}")
    print(f"Workflow: {record.workflow}")
    if record.submission_id:
        print(f"Submission ID: {record.submission_id}")
    if record.cluster_address:
        print(f"Cluster address: {record.cluster_address}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_jobs_status(args: argparse.Namespace) -> int:
    record = get_run_record(state_root=_state_root(args), run_id=args.run_id)
    print(f"Run ID: {record.run_id}")
    print(f"Status: {record.status}")
    print(f"Workflow: {record.workflow}")
    print(f"Target: {record.target}")
    print(f"Target dir: {record.target_dir}")
    if record.submission_id:
        print(f"Submission ID: {record.submission_id}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_jobs_logs(args: argparse.Namespace) -> int:
    print(get_run_logs(state_root=_state_root(args), run_id=args.run_id))
    return 0


def cmd_jobs_cancel(args: argparse.Namespace) -> int:
    record = cancel_run(
        state_root=_state_root(args),
        run_id=args.run_id,
        dry_run=getattr(args, "dry_run", False),
    )
    print(f"Run ID: {record.run_id}")
    print(f"Status: {record.status}")
    if record.message:
        print(f"Message: {record.message}")
    return 0


def cmd_resources_list(args: argparse.Namespace) -> int:
    leases = list_resource_leases(state_root=_state_root(args))
    if not leases:
        print("No resource leases.")
        return 0
    for lease in leases:
        print(f"{lease.lease_id} {lease.resource_type} {lease.status} owner={lease.owner}")
    return 0


def cmd_resources_reserve(args: argparse.Namespace) -> int:
    lease = reserve_resource(
        state_root=_state_root(args),
        resource_type=args.resource_type,
        owner=args.owner,
        metadata=_parse_metadata(args.metadata),
    )
    print(f"Lease ID: {lease.lease_id}")
    print(f"Resource: {lease.resource_type}")
    print(f"Status: {lease.status}")
    return 0


def cmd_resources_release(args: argparse.Namespace) -> int:
    lease = release_resource(state_root=_state_root(args), lease_id=args.lease_id)
    print(f"Lease ID: {lease.lease_id}")
    print(f"Status: {lease.status}")
    return 0


def cmd_artifacts_list(args: argparse.Namespace) -> int:
    artifacts = list_artifacts(state_root=_state_root(args), run_id=args.run_id)
    if not artifacts:
        print("No artifacts.")
        return 0
    for artifact in artifacts:
        print(f"{artifact.name} {artifact.kind} {artifact.size_bytes}B")
    return 0


def cmd_artifacts_fetch(args: argparse.Namespace) -> int:
    artifact = fetch_artifact(
        state_root=_state_root(args),
        run_id=args.run_id,
        artifact_name=args.artifact_name,
    )
    print(f"Artifact: {artifact.name}")
    print(f"Path: {artifact.path}")
    print(f"Size: {artifact.size_bytes}")
    return 0


def _state_root(args: argparse.Namespace) -> Path:
    return normalize_state_root(args.state_root)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _parse_metadata(values: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            metadata[item] = "true"
            continue
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata
