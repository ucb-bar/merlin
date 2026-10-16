"""Check the compact preparation archive; omitted weight/IR bytes cannot be reopened."""

import gzip
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    summary = json.loads((root / "summary.json").read_text())
    catalogs, sidecars = [], []
    for variant in ("inline", "packed"):
        catalogs.append(json.loads((root / variant / "catalog.json").read_text()))
        with gzip.open(root / variant / "dev/device_signatures.json.gz", "rb") as stream:
            data = stream.read()
        identity = summary["programs"][variant]["dev/device_signatures.json"]
        assert hashlib.sha256(data).hexdigest() == identity["sha256"] and len(data) == identity["bytes"]
        sidecars.append(data)
    assert sidecars[0] == sidecars[1]
    assert len(json.loads(sidecars[0])["routed"]) == summary["groups"] == 155
    assert catalogs[0]["interfaces"] == catalogs[1]["interfaces"] and len(catalogs[0]["interfaces"]) == 5
    assert (
        summary["programs"]["inline"]["ref/model.prepared.mlir"]
        == summary["programs"]["packed"]["ref/model.prepared.mlir"]
    )
    print("Archive consistent: identical 155 group bindings and five interfaces; packed data omitted.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
