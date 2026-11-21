"""Explicit portfolio source freezing and resource-bounded worker supervision."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from merlin.development.backend.optimization import contracts as P2_CONTRACTS

# ---- portfolio_launch --------------------------------------------------------------------------


def acquire_host_resource_lease(stage_root: Path, *, lease_path: Path):
    """Exclude another heavy compiler experiment without claiming machine-wide ownership."""
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lease_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        stage_root.mkdir(parents=True, exist_ok=True)
        P2_CONTRACTS.write_json(
            stage_root / "host_resource_telemetry.json",
            {
                "schema": "host_resource_telemetry_v1",
                "status": "launch_refused",
                "reasons": ["host_resource_lease_busy"],
                "lease": str(lease_path),
                "worker_returncode": None,
            },
        )
        return None
    except BaseException:
        handle.close()
        raise
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "stage_root": str(stage_root)}) + "\n")
        handle.flush()
    except BaseException:
        handle.close()
        raise
    return handle
