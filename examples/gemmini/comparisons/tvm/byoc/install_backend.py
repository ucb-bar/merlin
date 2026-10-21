#!/usr/bin/env python3
"""Install this out-of-tree backend through TVM's Python contrib extension point.

No TVM C++ rebuild, exporter patch, or existing Gemmini backend change is needed.
The tracked copies in this directory are the source of truth.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = "6b376cc8895a83387ec38cc06ef8f7b09827b071"


def install(root, check=False):
    root = root.resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError(f"Expected pinned TVM {PIN}; got {commit}")
    copies = {"gemmini_byoc.py": "python/tvm/relax/backend/contrib/gemmini_byoc.py",
              "byoc_primitives.c": "apps/gemmini/byoc_primitives.c"}
    receipts = []
    for source, destination in copies.items():
        payload = Path(__file__).with_name(source).read_bytes()
        path = root / destination
        if check:
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"Installed backend differs: {path}")
        else:
            path.write_bytes(payload)
        receipts.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
    return {"tvm_commit": commit, "files": receipts, "checked": check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvm-source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.tvm_source, args.check), indent=2))
