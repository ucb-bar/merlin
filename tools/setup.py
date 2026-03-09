#!/usr/bin/env python3
import argparse
import shutil
import sys
import urllib.request

import utils

# Config
CONDA_ENV_FILE = utils.REPO_ROOT / "env_linux.yml"
PIP_REQ_FILE = utils.REPO_ROOT / "requirements.txt"  # <--- ADD THIS
SPACEMIT_URL = "https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz"
TOOLCHAIN_DEST = utils.REPO_ROOT / "build" / "riscv-tools-spacemit"


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("component", choices=["all", "env", "toolchain"], default="all", nargs="?")


def setup_env(args) -> int:
    print("--- Setting up Conda Environment ---")
    if not shutil.which("conda"):
        print("Error: 'conda' not found.")
        return 1

    # 1. Sync System Toolchain (Compilers, CMake, etc.)
    print(">>> Syncing Conda (Toolchain)...")
    ret = utils.run(["conda", "env", "update", "--file", str(CONDA_ENV_FILE), "--prune"], dry_run=False)
    if ret != 0:
        return ret

    # 2. Sync Python Libraries (Torch, IREE, etc.)
    print(">>> Syncing Pip (Libraries)...")
    return utils.run([sys.executable, "-m", "pip", "install", "-r", str(PIP_REQ_FILE)], dry_run=False)


def setup_toolchain(args) -> int:
    print("--- Setting up SpacemiT Toolchain ---")
    TOOLCHAIN_DEST.mkdir(parents=True, exist_ok=True)
    tar_path = TOOLCHAIN_DEST / "toolchain.tar.xz"

    if not tar_path.exists():
        print(f"Downloading {SPACEMIT_URL}...")
        try:
            urllib.request.urlretrieve(SPACEMIT_URL, tar_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return 1

    print("Extracting...")
    # Using tar command directly as it's often more reliable for xz on linux
    return utils.run(["tar", "-xvf", str(tar_path), "-C", str(TOOLCHAIN_DEST)], dry_run=False)


def main(args: argparse.Namespace) -> int:
    ret = 0
    if args.component in ["env", "all"]:
        ret |= setup_env(args)
    if args.component in ["toolchain", "all"]:
        ret |= setup_toolchain(args)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Setup")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
