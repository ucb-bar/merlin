"""Restore the three-model chunk catalog and relocate checked reference proofs."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked(root, row):
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise ValueError("Publication member changed: " + row["path"])
    return path


def restore(delivery, destination, merlin):
    manifest = json.loads((delivery / "transfer-manifest.json").read_text())
    dependency = manifest["resnet_dependency"]
    original = checked(merlin, dependency["manifest"]).parent
    checked(merlin, dependency["restore"])
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, str(original / "restore.py"), "--output-dir", str(destination / "resnet-release")], check=True
    )
    archive = destination / "shared-chunks.tar.gz"
    with archive.open("xb") as output:
        for row in manifest["archive"]["parts"]:
            with checked(delivery, row).open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    output.write(block)
    checked(destination, dict(manifest["archive"], path=archive.name))
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    verified = 0
    for line in (destination / "PAYLOAD_SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()) or sha(path) != expected:
            raise ValueError("Extracted payload changed: " + name)
        verified += 1
    index = json.loads((destination / "source-map.json").read_text())
    for row in index.values():
        checked(destination, row)
    proofs = destination / "relocated-proofs"
    proofs.mkdir()
    catalog = json.loads((destination / "catalog.json").read_text())
    for row in catalog["chunks"]:
        if row["model"] == "resnet":
            continue
        folder = (destination / row["program"]).parent
        original_proof = folder / "fixture-provenance.json"
        proof = json.loads(original_proof.read_text())
        if (
            sha(folder / "fixture.npz") != proof["fixture_sha256"]
            or sha(folder / "program.json") != proof["program"]["sha256"]
        ):
            raise ValueError("Chunk identity changed")
        for item in proof["reference_sources"] + [proof["policy"]]:
            item["path"] = str(checked(destination, index[item["sha256"]]))
        proof["program"]["path"] = str(folder / "program.json")
        proof["origin_proof_sha256"] = sha(original_proof)
        (proofs / (row["id"] + ".json")).write_text(json.dumps(proof, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(destination / "resnet-release/relocate_proofs.py"),
            "--output-dir",
            str(destination / "relocated-resnet-proofs"),
        ],
        check=True,
    )
    print(
        json.dumps(
            dict(status="pass", payload_files_verified=verified, chunks=len(catalog["chunks"]), root=str(destination)),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--merlin-root", type=Path)
    args = parser.parse_args()
    delivery = Path(__file__).resolve().parent
    merlin = args.merlin_root
    if merlin is None:
        merlin = next((p for p in delivery.parents if (p / "AGENT.md").exists() and (p / "examples").exists()), None)
    if merlin is None:
        parser.error("Supply --merlin-root pointing at the published Merlin checkout")
    restore(delivery, args.output_dir.resolve(), merlin.resolve())


if __name__ == "__main__":
    main()
