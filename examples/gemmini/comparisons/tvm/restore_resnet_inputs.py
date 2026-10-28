"""Restore only the published ResNet payload and relocate its checked proofs.

Historical policy files and patches remain evidence, not current instructions.
The frozen shared catalog is retained verbatim; consumers select ResNet only.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from build_shared_chunk_elf import CATALOG_SHA256, digest


def restore(merlin, destination):
    releases = merlin / "out/artifacts/handoff"
    original = releases / "gemmini-firesim-dse-publication/v1/handoff_gemmini-firesim-dse-publication_v1_20261026T173908Z_238ada8"
    shared = releases / "gemmini-shared-model-chunks/v1/handoff_gemmini-shared-model-chunks_v1_20261027T231759Z_f46e73a"
    if digest(shared / "catalog.json") != CATALOG_SHA256:
        raise ValueError("Published catalog changed")
    if digest(original / "restore.py") != "187a893e495b4f02a87fc70469d53590a8512616faafc598dcbd0324f608a51d":
        raise ValueError("Published restoration entry point changed")
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(original / "restore.py"), "--output-dir",
                    str(destination / "resnet-release")], check=True)
    shutil.copyfile(shared / "catalog.json", destination / "catalog.json")
    subprocess.run([sys.executable, str(destination / "resnet-release/relocate_proofs.py"),
                    "--output-dir", str(destination / "relocated-resnet-proofs")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    restore(args.merlin_root.resolve(strict=True), args.output_dir.resolve())


if __name__ == "__main__":
    main()
