#!/usr/bin/env python3
import argparse
import shutil
import sys
import urllib.request

import utils

# Config
CONDA_ENV_FILE = utils.REPO_ROOT / "env_linux.yml"
PIP_REQ_FILE = utils.REPO_ROOT / "requirements.txt"
SPACEMIT_URL = "https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz"
TOOLCHAIN_DEST = utils.REPO_ROOT / "build" / "riscv-tools-spacemit"


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("component", choices=["all", "env", "toolchain"], default="all", nargs="?")
    parser.add_argument(
        "--env-name",
        default="merlin-dev",
        help="Conda environment name to update/install packages into (default: merlin-dev).",
    )
    parser.add_argument("--offline", action="store_true", help="Run setup in offline mode when possible.")
    parser.add_argument("--skip-conda", action="store_true", help="Skip conda environment sync.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip Python dependency sync (uv/pip).")
    parser.add_argument(
        "--python-deps",
        choices=["auto", "uv", "pip"],
        default="auto",
        help=(
            "Python dependency installer. " "'auto' prefers uv sync with uv.lock and falls back to pip requirements."
        ),
    )
    parser.add_argument(
        "--conda-no-plugins",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force CONDA_NO_PLUGINS for conda env update. "
            "If unset, setup.py retries with CONDA_NO_PLUGINS=true on failure."
        ),
    )


def run_conda_env_update(args, *, no_plugins: bool) -> int:
    cmd = ["conda", "env", "update", "--name", args.env_name, "--file", str(CONDA_ENV_FILE), "--prune"]
    if args.offline:
        cmd.append("--offline")
    env = {"CONDA_NO_PLUGINS": "true"} if no_plugins else None
    return utils.run(cmd, dry_run=False, env=env)


def setup_env(args) -> int:
    print("--- Setting up Conda Environment ---")
    if not shutil.which("conda"):
        print("Error: 'conda' not found.")
        return 1

    # 1. Sync System Toolchain (Compilers, CMake, etc.)
    if not args.skip_conda:
        print(f">>> Syncing Conda env '{args.env_name}' (Toolchain)...")
        use_no_plugins = bool(args.conda_no_plugins)
        ret = run_conda_env_update(args, no_plugins=use_no_plugins)
        if ret != 0 and args.conda_no_plugins is None:
            print(">>> Retrying conda env update with CONDA_NO_PLUGINS=true...")
            ret = run_conda_env_update(args, no_plugins=True)
        if ret != 0:
            return ret

    # 2. Sync Python Libraries (Torch, IREE, etc.)
    if not args.skip_pip:
        use_uv = args.python_deps == "uv" or (args.python_deps == "auto" and shutil.which("uv"))
        if use_uv:
            print(f">>> Syncing Python deps in '{args.env_name}' with uv...")
            uv_cmd = ["conda", "run", "-n", args.env_name, "uv", "sync", "--frozen"]
            if args.offline:
                uv_cmd.append("--offline")
            uv_ret = utils.run(uv_cmd, dry_run=False)
            if uv_ret == 0:
                return 0
            if args.python_deps == "uv":
                return uv_ret
            print(">>> uv sync failed; falling back to pip requirements install...")

        print(f">>> Syncing Pip libs in '{args.env_name}'...")
        pip_cmd = ["conda", "run", "-n", args.env_name, "python", "-m", "pip", "install", "-r", str(PIP_REQ_FILE)]
        if args.offline:
            pip_cmd.append("--no-index")
        return utils.run(pip_cmd, dry_run=False)

    return 0


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
