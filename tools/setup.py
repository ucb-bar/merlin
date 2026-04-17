#!/usr/bin/env python3
# tools/setup.py
"""Backs `./merlin setup`: bootstraps the developer environment — conda env,
uv-managed Python deps, submodule sync (per profile), and prebuilt-artifact
installation.

See docs/getting_started.md for the canonical first-time flow.
"""

import argparse
import pathlib
import platform
import shutil
import subprocess
import sys

import utils

DEFAULT_CONDA_ENV_FILE = utils.REPO_ROOT / ("env_macOS.yml" if platform.system() == "Darwin" else "env_linux.yml")
PIP_REQ_FILE = utils.REPO_ROOT / "requirements.txt"

SUBMODULE_STEP_PROFILES = {
    "core": [
        {
            "paths": ["third_party/iree_bar"],
            "recursive": True,
        },
    ],
    "npu": [
        {
            "paths": ["third_party/iree_bar"],
            "recursive": True,
        },
        {
            "paths": ["third_party/npu_model"],
            "recursive": False,
        },
    ],
    "smolvla": [
        {
            "paths": ["third_party/iree_bar"],
            "recursive": True,
        },
        {
            "paths": [
                "third_party/Understanding-PI0",
                "third_party/iree-turbine",
                "third_party/torch-mlir",
            ],
            "recursive": False,
        },
    ],
    "full": [
        {
            "paths": [],
            "recursive": True,
        },
    ],
}


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "component",
        choices=["all", "env", "toolchain", "submodules", "prebuilt"],
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "--env-name",
        default="merlin-dev",
        help="Conda environment name to update/install packages into (default: merlin-dev).",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_CONDA_ENV_FILE),
        help=("Conda environment file to use. " f"Default is platform-specific: {DEFAULT_CONDA_ENV_FILE.name}"),
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

    parser.add_argument(
        "--submodules-profile",
        choices=["core", "npu", "smolvla", "full"],
        default="core",
        help=("Which submodule profile to initialize for the current Merlin " "checkout (default: core)."),
    )
    parser.add_argument(
        "--submodule-path",
        action="append",
        default=[],
        help="Additional top-level submodule path to initialize (repeatable).",
    )
    parser.add_argument(
        "--submodule-paths-recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether extra --submodule-path entries should be initialized recursively.",
    )
    parser.add_argument(
        "--submodule-depth",
        type=int,
        default=1,
        help="Shallow depth for submodule fetches (default: 1). Use 0 for full history.",
    )
    parser.add_argument(
        "--submodule-jobs",
        type=int,
        default=8,
        help="Parallel submodule fetch jobs (default: 8).",
    )
    parser.add_argument(
        "--submodule-sync",
        action="store_true",
        help=(
            "Run `git submodule sync --recursive` before updating. "
            "Submodule SHAs still come from the current Merlin commit."
        ),
    )

    parser.add_argument(
        "--toolchain-target",
        choices=["spacemit", "firesim", "all"],
        default="spacemit",
        help="Which toolchain target to install (default: spacemit).",
    )
    parser.add_argument(
        "--with-qemu",
        action="store_true",
        help="For firesim toolchain setup, also install QEMU.",
    )
    parser.add_argument(
        "--toolchain-force",
        action="store_true",
        help="Reinstall toolchains even if the destination already exists.",
    )

    parser.add_argument(
        "--prebuilt-artifact",
        choices=["host-linux-x86_64", "host-macos", "runtime-spacemit", "runtime-saturnopu"],
        default="host-linux-x86_64",
        help="Which published Merlin prebuilt artifact to install.",
    )
    parser.add_argument(
        "--prebuilt-tag",
        default="latest",
        help="GitHub release tag to download from, or 'latest' (default: latest).",
    )
    parser.add_argument(
        "--prebuilt-repo",
        default="ucb-bar/merlin",
        help="GitHub repository containing release assets (default: ucb-bar/merlin).",
    )
    parser.add_argument(
        "--prebuilt-force",
        action="store_true",
        help="Replace an existing destination build tree when installing a prebuilt artifact.",
    )


def resolve_env_file(args) -> pathlib.Path:
    env_file = pathlib.Path(args.env_file)
    if not env_file.is_absolute():
        env_file = utils.REPO_ROOT / env_file
    return env_file


def run_conda_env_update(args, *, no_plugins: bool) -> int:
    env_file = resolve_env_file(args)
    cmd = ["conda", "env", "update", "--name", args.env_name, "--file", str(env_file), "--prune"]
    if args.offline:
        cmd.append("--offline")
    env = {"CONDA_NO_PLUGINS": "true"} if no_plugins else None
    return utils.run(cmd, dry_run=False, env=env)


def setup_env(args) -> int:
    print("--- Setting up Conda Environment ---")
    if not shutil.which("conda"):
        print("Error: 'conda' not found.")
        return 1

    env_file = resolve_env_file(args)
    if not env_file.exists():
        print(f"Error: env file not found: {env_file}")
        return 1

    if not args.skip_conda:
        print(f">>> Syncing Conda env '{args.env_name}' using {env_file}...")
        use_no_plugins = bool(args.conda_no_plugins)
        ret = run_conda_env_update(args, no_plugins=use_no_plugins)
        if ret != 0 and args.conda_no_plugins is None:
            print(">>> Retrying conda env update with CONDA_NO_PLUGINS=true...")
            ret = run_conda_env_update(args, no_plugins=True)
        if ret != 0:
            return ret

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


def run_git_submodule_update(paths, *, recursive: bool, depth: int, jobs: int) -> int:
    cmd = ["git", "-C", str(utils.REPO_ROOT), "submodule", "update", "--init", "--jobs", str(jobs)]
    if recursive:
        cmd.append("--recursive")
    if depth > 0:
        cmd.extend(["--depth", str(depth)])
    if paths:
        cmd.extend(["--", *paths])
    return utils.run(cmd, dry_run=False)


def resolve_submodule_steps(args):
    steps = list(SUBMODULE_STEP_PROFILES[args.submodules_profile])

    if args.submodule_path:
        steps.append(
            {
                "paths": list(args.submodule_path),
                "recursive": bool(args.submodule_paths_recursive),
            }
        )

    return steps


def describe_repo_ref() -> str:
    branch = subprocess.run(
        ["git", "-C", str(utils.REPO_ROOT), "symbolic-ref", "--short", "-q", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(utils.REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if branch and commit:
        return f"{branch} ({commit})"
    if commit:
        return f"detached HEAD ({commit})"
    return "unknown ref"


def setup_submodules(args) -> int:
    print("--- Setting up Git Submodules ---")
    if not shutil.which("git"):
        print("Error: 'git' not found.")
        return 1
    print(f">>> Merlin ref: {describe_repo_ref()}")
    print(
        ">>> Note: submodule SHAs follow the current Merlin commit. "
        "If you switch branches later, rerun this command."
    )

    if args.offline:
        print(
            ">>> Offline mode enabled. Submodule setup will only succeed if "
            "needed git objects are already available locally."
        )

    if args.submodule_sync:
        sync_cmd = ["git", "-C", str(utils.REPO_ROOT), "submodule", "sync", "--recursive"]
        ret = utils.run(sync_cmd, dry_run=False)
        if ret != 0:
            return ret

    steps = resolve_submodule_steps(args)

    for step in steps:
        paths = step["paths"]
        recursive = step["recursive"]
        label = "ALL SUBMODULES" if not paths else ", ".join(paths)
        print(f">>> Initializing submodules: {label} (recursive={recursive}, depth={args.submodule_depth})")
        ret = run_git_submodule_update(
            paths,
            recursive=recursive,
            depth=args.submodule_depth,
            jobs=args.submodule_jobs,
        )
        if ret != 0:
            return ret

    return 0


def run_toolchain_script(script_path: pathlib.Path, script_args: list[str]) -> int:
    if not script_path.exists():
        print(f"Error: toolchain script not found: {script_path}")
        return 1
    cmd = ["bash", str(script_path), *script_args]
    return utils.run(cmd, dry_run=False)


def setup_toolchain(args) -> int:
    spacemit_script = utils.REPO_ROOT / "build_tools" / "SpacemiT" / "setup_toolchain.sh"
    firesim_script = utils.REPO_ROOT / "build_tools" / "firesim" / "setup_toolchain.sh"

    common_args: list[str] = []
    if args.toolchain_force:
        common_args.append("--force")
    if args.offline:
        common_args.append("--offline")

    if args.toolchain_target == "spacemit":
        return run_toolchain_script(spacemit_script, common_args)

    if args.toolchain_target == "firesim":
        firesim_args = list(common_args)
        if args.with_qemu:
            firesim_args.append("--with-qemu")
        return run_toolchain_script(firesim_script, firesim_args)

    ret = run_toolchain_script(spacemit_script, common_args)
    if ret != 0:
        return ret

    firesim_args = list(common_args)
    if args.with_qemu:
        firesim_args.append("--with-qemu")
    return run_toolchain_script(firesim_script, firesim_args)


def setup_prebuilt(args) -> int:
    cmd = [
        sys.executable,
        str(utils.REPO_ROOT / "tools" / "install_prebuilt.py"),
        "--artifact",
        args.prebuilt_artifact,
        "--tag",
        args.prebuilt_tag,
        "--repo",
        args.prebuilt_repo,
    ]
    if args.prebuilt_force:
        cmd.append("--force")
    if args.offline:
        cmd.append("--offline")
    return utils.run(cmd, dry_run=False)


def main(args: argparse.Namespace) -> int:
    ret = 0
    if args.component in ["env", "all"]:
        ret |= setup_env(args)
    if args.component in ["toolchain", "all"]:
        ret |= setup_toolchain(args)
    if args.component == "submodules":
        ret |= setup_submodules(args)
    if args.component == "prebuilt":
        ret |= setup_prebuilt(args)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Setup")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
