#!/usr/bin/env python3
"""Capture the Merlin C graph through the maintained atlas-mlir capture package.

Pass --atlas-root or set ATLAS_MLIR_ROOT to a checkout containing tools/smol_capture.
All input providers, the original model, the interpreter, and the output directory
are configured by that package's command line. No sibling job files are required.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--atlas-root", default=os.environ.get("ATLAS_MLIR_ROOT"))
    args, remaining = parser.parse_known_args()
    if not args.atlas_root:
        if "-h" in remaining or "--help" in remaining:
            parser.add_argument("-h", "--help", action="help")
            parser.print_help()
            return
        parser.error("--atlas-root or ATLAS_MLIR_ROOT is required")
    path = Path(args.atlas_root).absolute()
    if ".." in path.parts:
        parser.error("restricted atlas root")
    if any(prefix.is_symlink() for prefix in reversed([path, *path.parents])):
        parser.error("symlink atlas root is unsupported")
    sys.path.insert(0, str(path / "tools"))
    from smol_capture.capture import main as capture

    capture(["--atlas-root", str(path), *remaining])


if __name__ == "__main__":
    main()
