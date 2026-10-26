"""Restore and verify the committed FireSim input bundle using Python 3.12+."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "transfer-manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir.resolve()
    archive = destination / "firesim-round2-handoff.tar.gz"
    total = hashlib.sha256()
    with archive.open("xb") as output:
        for part in manifest["archive"]["parts"]:
            if Path(part["name"]).name != part["name"]:
                raise ValueError("Part name must stay within this delivery")
            path = ROOT / part["name"]
            if path.stat().st_size != part["bytes"]:
                raise ValueError("Part size differs: " + part["name"])
            observed = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    observed.update(block)
                    total.update(block)
                    output.write(block)
            if observed.hexdigest() != part["sha256"]:
                raise ValueError("Part checksum differs: " + part["name"])
    if total.hexdigest() != manifest["archive"]["sha256"] or archive.stat().st_size != manifest["archive"]["bytes"]:
        raise ValueError("Assembled archive differs from the frozen input bundle")
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    verified = 0
    for line in (destination / "PAYLOAD_SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or digest(path) != expected:
            raise ValueError("Extracted payload differs: " + name)
        verified += 1
    print(
        json.dumps(
            dict(
                status="pass",
                input_dir=str(destination),
                payload_files_verified=verified,
                archive_sha256=total.hexdigest(),
                source_policy="Use committed branch sources; do not apply archived historical patches",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
