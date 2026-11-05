"""Verify the allocated enclosing worker; never create resource controls."""

from pathlib import Path

from . import _kernel_resources as kernel
from .evaluation_pool import AssignedResources
from .external_resources import prepare_external_resource_boundary


def prepare_launch_boundary(selection, *, external_boundary=None):
    """Issue hard resource proof from actual mounts and kernel controls only.

    Configuration selects capacities. The original live verifier establishes
    membership and enforcement; serialized observations cannot replace it.
    Ordinary launches without a hard requirement keep their existing behavior.
    """
    selection.verify()
    declared = dict(selection.document["resources"])
    declared["cpu_ids"] = tuple(declared["cpu_ids"])
    resources = AssignedResources(**declared)
    if external_boundary is None and resources.require_hard_limits:
        mounts = kernel.mounts(Path("/proc/self/mountinfo").read_text())
        writable_roots = tuple(
            dict.fromkeys(
                Path(row[2]) for row in mounts if "rw" in row[4].split(",") and row[3] not in {"proc", "devpts"}
            )
        )
        external_boundary = prepare_external_resource_boundary(
            cpu_ids=resources.cpu_ids,
            ram_bytes=resources.ram_bytes,
            disk_bytes=resources.disk_bytes,
            writable_roots=writable_roots,
        )
    resources.verify(external_boundary=external_boundary)
    selection.verify()
    return external_boundary
