"""Audit and archive host-only Phase 1 capsule simulator evidence."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
from pathlib import Path


class TimingArchiveError(ValueError):
    """The selected run has no usable gSIM capsule timing evidence."""


def audit(run_dir: Path) -> dict:
    """Count actual capsule results; a pass with no positive cycle marker is unmeasured."""
    root = run_dir.resolve()
    paths = sorted(root.glob("_qa_work/runs_*/runs/*/*/capsule_result.json"))
    rows = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TimingArchiveError(f"invalid capsule result: {path}")
        tier = (document.get("tiers") or {}).get("L3") or {}
        if not isinstance(tier, dict):
            raise TimingArchiveError(f"invalid L3 tier: {path}")
        cycles = tier.get("cycles")
        measured = type(cycles) is int and cycles > 0
        rows.append(
            {
                "capsule": document.get("capsule"),
                "result": path.relative_to(root).as_posix(),
                "status": tier.get("status"),
                "engine": tier.get("engine"),
                "cycles": cycles,
                "cycles_measured": measured,
                "timing": tier.get("timing"),
                "sim_provenance": tier.get("sim_provenance"),
            }
        )
    gsim_pass = [r for r in rows if r["engine"] == "gsim" and r["status"] == "pass"]
    return {
        "schema": "merlin.phase1.capsule_timing_archive.v1",
        "run_dir": str(root),
        "capsule_records": len(rows),
        "gsim_pass_records": len(gsim_pass),
        "gsim_pass_with_cycles": sum(r["cycles_measured"] for r in gsim_pass),
        "timing_ready": bool(gsim_pass) and all(r["cycles_measured"] for r in gsim_pass),
        "records": rows,
    }


def archive(run_dir: Path, destination: Path) -> dict:
    """Package complete capsule result directories after timing readiness succeeds."""
    index = audit(run_dir)
    if not index["timing_ready"]:
        raise TimingArchiveError(
            f"gSIM capsule timing incomplete: {index['gsim_pass_with_cycles']}/"
            f"{index['gsim_pass_records']} passing records have positive measured cycles"
        )
    if destination.exists():
        raise TimingArchiveError(f"archive already exists: {destination}")
    root = Path(index["run_dir"])
    capsule_dirs = sorted({(root / row["result"]).parent for row in index["records"]})
    files = []
    for directory in capsule_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise TimingArchiveError(f"capsule artifact is a symlink: {path}")
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise TimingArchiveError(f"capsule artifact is not a regular file: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="capsule-timing-", suffix=".tar", dir=destination.parent, delete=False
    ) as temporary:
        partial = Path(temporary.name)
    try:
        with tarfile.open(partial, "w") as tar:
            payload = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8")
            member = tarfile.TarInfo("index.json")
            member.size = len(payload)
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(payload))
            for path in files:
                member = tar.gettarinfo(path, arcname="run/" + path.relative_to(root).as_posix())
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                with path.open("rb") as source:
                    tar.addfile(member, source)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return {**index, "archive": str(destination), "artifact_files": len(files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write a host-only .tar after positive gSIM cycle validation")
    args = parser.parse_args(argv)
    try:
        result = archive(args.run_dir, args.out) if args.out else audit(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"capsule timing archive: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["timing_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
