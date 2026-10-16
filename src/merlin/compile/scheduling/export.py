"""Export the canonical standalone dense planner into a compiler bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def export_dense_planner(directory: Path) -> dict:
    """Write exact public sources and return identities for the complete release."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("dense_planning", "dense_ir"):
        source = (Path(__file__).parents[2] / "transforms" / (name + ".py")).read_bytes()
        destination = directory / ("common_" + name + ".py")
        destination.write_bytes(source)
        files[destination.name] = hashlib.sha256(source).hexdigest()
    return {"schema": "merlin.dense_planner.v1", "files_sha256": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = export_dense_planner(args.output_dir)
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
