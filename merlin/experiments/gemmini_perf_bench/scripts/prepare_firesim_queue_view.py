#!/usr/bin/env python3
"""Build an immutable Chipyard view for cross-user FireSim queue jobs.

The view deliberately makes ``deploy/`` read-only.  This forces firesim-queue
to create its per-job writable overlay while retaining the exact prebuilt host
driver selected by the bitstream's deploy-quintuplet metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_children(source: Path, destination: Path, excluded: set[str]) -> None:
    for entry in source.iterdir():
        if entry.name not in excluded:
            (destination / entry.name).symlink_to(entry)


def _deploy_identity(bitstream: Path, firesim_root: Path) -> tuple[str, Path]:
    with tarfile.open(bitstream, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith("/metadata")]
        if len(members) != 1:
            raise ValueError("bitstream archive must contain exactly one */metadata member")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError("bitstream metadata member is not a regular file")
        metadata = extracted.read().decode("utf-8")
    matches = re.findall(r"firesim-deployquintuplet:([^,]+)", metadata)
    if len(matches) != 1:
        raise ValueError("bitstream metadata must declare one deploy quintuplet")
    quintuplet = matches[0]
    platform = quintuplet.split("-firesim-", 1)[0]
    driver = firesim_root / "sim" / "output" / platform / quintuplet / f"FireSim-{platform}"
    if not driver.is_file() or not os.access(driver, os.X_OK):
        raise FileNotFoundError(f"exact-config prebuilt driver is absent: {driver}")
    return quintuplet, driver


def prepare(args: argparse.Namespace) -> dict[str, str]:
    chipyard = args.chipyard.resolve(strict=True)
    bitstream = args.bitstream.resolve(strict=True)
    elf = args.elf.resolve(strict=True)
    firesim_root = chipyard / "sims" / "firesim"
    deploy = firesim_root / "deploy"
    template = deploy / "workloads" / f"{args.workload_template}.json"
    for required in (chipyard / "env.sh", firesim_root / "sourceme-manager.sh", deploy / "firesim", template):
        if not required.is_file():
            raise FileNotFoundError(f"missing driver/workload authority: {required}")

    workload_template = json.loads(template.read_text())
    if workload_template.get("workloads") is not None or workload_template.get("common_rootfs") is not None:
        raise ValueError("workload template must be a uniform bare-metal workload")

    quintuplet, driver = _deploy_identity(bitstream, firesim_root)
    elf_sha = _sha256(elf)
    workload = f"{args.workload_prefix}-{elf_sha[:16]}"
    bootbinary = f"{workload}.elf"

    args.view_root.mkdir(parents=True, exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix="view.", dir=args.view_root))
    view_firesim = view / "sims" / "firesim"
    view_deploy = view_firesim / "deploy"
    view_workloads = view_deploy / "workloads"
    view_workloads.mkdir(parents=True)

    (view / "env.sh").symlink_to(chipyard / "env.sh")
    (view / "generators").symlink_to(chipyard / "generators")
    _link_children(firesim_root, view_firesim, {"deploy"})
    _link_children(
        deploy,
        view_deploy,
        {"workloads", "logs", "results-workload", "generated-topology-diagrams"},
    )

    generated_workload = dict(workload_template)
    generated_workload["benchmark_name"] = workload
    generated_workload["common_bootbinary"] = bootbinary
    workload_path = view_workloads / f"{workload}.json"
    workload_path.write_text(json.dumps(generated_workload, indent=2) + "\n")

    identity = {
        "schema": "merlin_firesim_queue_view_identity_v1",
        "view": str(view),
        "workload": workload,
        "bootbinary": bootbinary,
        "bootbinary_sha256": elf_sha,
        "workload_template": str(template),
        "workload_template_sha256": _sha256(template),
        "generated_workload_json_sha256": _sha256(workload_path),
        "deploy_quintuplet": quintuplet,
        "bitstream": str(bitstream),
        "bitstream_sha256": _sha256(bitstream),
        "driver": str(driver),
        "driver_sha256": _sha256(driver),
        "queue_owned_lifecycle": [
            "firesim kill",
            "firesim infrasetup",
            "firesim runworkload",
            "firesim kill",
        ],
    }
    identity_path = view / "queue_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    for path in (workload_path, identity_path):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for path in (view_workloads, view_deploy, view_firesim, view / "sims", view):
        path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chipyard", type=Path, required=True)
    parser.add_argument("--bitstream", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--workload-prefix", required=True)
    parser.add_argument("--workload-template", default="merlin-l4-20-resnet50-generated")
    parser.add_argument(
        "--view-root",
        type=Path,
        default=Path("/scratch/firesim_queue/queue_chipyard_views"),
    )
    args = parser.parse_args()
    identity = prepare(args)
    print("\t".join(identity[key] for key in ("view", "workload", "bootbinary", "deploy_quintuplet", "driver")))


if __name__ == "__main__":
    main()
