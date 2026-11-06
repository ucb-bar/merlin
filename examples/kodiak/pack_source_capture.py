"""Package one receipted source capture without adding ambient files.

The archive is deterministic for the same capture bytes and directory name.
The published source receipt SHA256 is required; a self-consistent replacement
of the capture and its receipt is not accepted. Packaging proves materialized
bytes only, not source closure, framework parity, or target execution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import BinaryIO

from merlin.targetgen.application_inventory import verify_capture_receipt


def _sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def inspect_capture(source: Path, expected_receipt_sha256: str) -> tuple[dict, list[str]]:
    """Verify published identity and all materialized bytes before an output is created."""
    if not _valid_sha256(expected_receipt_sha256):
        raise ValueError("expected receipt SHA256 must be 64 lowercase hex digits")
    source = source.resolve()
    if not source.is_dir() or source.name in {"", ".", ".."}:
        raise ValueError(f"capture directory is absent: {source}")
    receipt_path = source / "capture_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("capture receipt is absent or symlinked")
    with receipt_path.open("rb") as stream:
        receipt_sha256 = _sha256(stream)
    if receipt_sha256 != expected_receipt_sha256:
        raise ValueError("source receipt differs from published SHA256")
    check = verify_capture_receipt(source / "model.mlir")
    if check.get("status") != "verified_materialized":
        raise ValueError(f"source capture receipt failed: {check.get('errors')}")
    receipt = json.loads(receipt_path.read_bytes())
    names = ["capture_receipt.json", *sorted(receipt["artifacts"])]
    if len(names) != len(set(names)):
        raise ValueError("capture receipt is also listed as an artifact")
    for name in names:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"unsafe archive member: {name!r}")
    return receipt, names


def write_archive(stream: BinaryIO, source: Path, names: list[str]) -> None:
    """Write a reproducible gzip/tar stream of exact receipt membership."""
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for name in names:
                path = source / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"archive member is absent or symlinked: {name}")
                info = tarfile.TarInfo(f"{source.name}/{name}")
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as member:
                    archive.addfile(info, member)


def verify_archive(path: Path, source_name: str, receipt: dict, receipt_sha256: str) -> None:
    expected = {"capture_receipt.json": receipt_sha256}
    expected.update({name: record["sha256"] for name, record in receipt["artifacts"].items()})
    seen = set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            prefix = f"{source_name}/"
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError(f"unexpected archive member: {member.name}")
            name = member.name.removeprefix(prefix)
            if name not in expected or name in seen:
                raise ValueError(f"unreceipted or duplicate archive member: {name}")
            seen.add(name)
            extracted = archive.extractfile(member)
            if extracted is None or _sha256(extracted) != expected[name]:
                raise ValueError(f"archive member digest differs: {name}")
    if seen != set(expected):
        raise ValueError(f"archive lacks receipt members: {sorted(set(expected) - seen)}")


def pack_capture(
    source: Path,
    output: Path,
    expected_receipt_sha256: str,
    *,
    min_free_gib: int = 190,
    check_only: bool = False,
) -> dict:
    source = source.resolve()
    receipt, names = inspect_capture(source, expected_receipt_sha256)
    total_bytes = sum((source / name).stat().st_size for name in names)
    report = {
        "schema": "merlin.source-capture-archive.v1",
        "capture": source.name,
        "source_receipt_sha256": expected_receipt_sha256,
        "source_closure_verified": False,
        "members": names,
        "materialized_bytes": total_bytes,
        "status": "verified_source" if check_only else "archive_verified",
    }
    if check_only:
        return report
    if min_free_gib < 0:
        raise ValueError("min-free-gib must be nonnegative")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be fresh: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent is absent: {output.parent}")
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < min_free_gib * 1024**3 + total_bytes:
        raise ValueError("insufficient scratch margin for source capture archive")
    try:
        with output.open("xb") as stream:
            write_archive(stream, source, names)
        verify_archive(output, source.name, receipt, expected_receipt_sha256)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    with output.open("rb") as stream:
        report["archive_sha256"] = _sha256(stream)
    report["archive_bytes"] = output.stat().st_size
    report["archive"] = str(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--min-free-gib", type=int, default=190)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    report = pack_capture(
        args.source,
        args.output,
        args.expected_receipt_sha256,
        min_free_gib=args.min_free_gib,
        check_only=args.check_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
