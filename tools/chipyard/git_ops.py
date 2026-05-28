"""Chipyard git-state machine: validate, checkout, submodule backup/restore.

The big one. `cmd_checkout` switches chipyard between recipe states, which
requires preserving untracked files in chipyard's submodules (FireSim
build outputs, bitstreams) across the checkout.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile

import utils

from .config import require_chipyard_root
from .recipe import require_recipe


def _git(repo: pathlib.Path, *git_args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo)] + list(git_args), capture_output=True, text=True)
    return result.stdout.strip()


def cmd_validate(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    if _recipe_mode(recipe) == "board":
        print(f"Recipe '{recipe['name']}' is a physical board — no Chipyard validation needed.")
        return 0

    chipyard_spec = recipe.get("chipyard", {})
    errors = 0

    # Branch
    actual_branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    expected_branch = chipyard_spec.get("branch")
    if expected_branch and actual_branch != expected_branch:
        utils.eprint(f"  MISMATCH branch: expected '{expected_branch}', got '{actual_branch}'")
        errors += 1
    else:
        print(f"  OK  branch: {actual_branch}")

    # SHA
    actual_sha = _git(root, "rev-parse", "HEAD")
    expected_sha = chipyard_spec.get("sha")
    if expected_sha and actual_sha != expected_sha:
        utils.eprint(f"  MISMATCH SHA: expected {expected_sha[:10]}, got {actual_sha[:10]}")
        errors += 1
    else:
        print(f"  OK  SHA: {actual_sha[:10]}")

    # Submodules
    for sub_name, sub_spec in recipe.get("submodules", {}).items():
        expected_sub_sha = sub_spec.get("sha")
        if not expected_sub_sha:
            continue
        for prefix in ["generators/", "sims/", ""]:
            sub_path = root / f"{prefix}{sub_name}"
            if sub_path.is_dir():
                actual_sub_sha = _git(sub_path, "rev-parse", "HEAD")
                if actual_sub_sha != expected_sub_sha:
                    utils.eprint(
                        f"  MISMATCH submodule {sub_name}: expected {expected_sub_sha[:10]}, got {actual_sub_sha[:10]}"
                    )
                    errors += 1
                else:
                    print(f"  OK  submodule {sub_name}: {actual_sub_sha[:10]}")
                break
        else:
            utils.eprint(f"  WARNING submodule {sub_name}: not found")

    # Sysroot
    sysroot = root / ".conda-env" / "riscv-tools" / "riscv64-unknown-elf"
    if sysroot.is_dir():
        print(f"  OK  sysroot: {sysroot}")
    else:
        utils.eprint(f"  MISSING sysroot: {sysroot}")
        errors += 1

    if errors:
        utils.eprint(f"\n{errors} validation error(s).")
        return 1
    print("\nAll checks passed.")
    return 0


# ---------------------------------------------------------------------------
# checkout — switch chipyard state to match a recipe
# ---------------------------------------------------------------------------


def _find_submodule_path(root: pathlib.Path, sub_name: str) -> pathlib.Path | None:
    """Find a submodule directory under common chipyard prefixes."""
    for prefix in ["generators/", "sims/", ""]:
        sub_path = root / f"{prefix}{sub_name}"
        if sub_path.is_dir():
            return sub_path
    return None


# Directories inside submodules that contain user data (untracked by git)
# which must survive submodule re-init / checkout operations.
# These are relative to the submodule root (e.g. firesim/).
_PRESERVE_DIRS = [
    "deploy/results-build",
    "deploy/results-workload",
    "deploy/workloads",
    "deploy/config_build.yaml",
    "deploy/config_build_recipes.yaml",
    "deploy/config_hwdb.yaml",
    "deploy/config_runtime.yaml",
]


def _backup_untracked(sub_path: pathlib.Path) -> pathlib.Path | None:
    """Back up untracked user files/directories before a destructive git op.

    Returns the temp directory path if anything was backed up, else None.
    """
    items_to_save = []
    for relpath in _PRESERVE_DIRS:
        candidate = sub_path / relpath
        if candidate.exists():
            items_to_save.append((relpath, candidate))

    if not items_to_save:
        return None

    backup_dir = pathlib.Path(tempfile.mkdtemp(prefix="merlin-submodule-backup-"))
    for relpath, src in items_to_save:
        dst = backup_dir / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            print(f"  Backing up {src}")
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
        else:
            print(f"  Backing up {src}")
            shutil.copy2(src, dst)

    return backup_dir


def _restore_untracked(sub_path: pathlib.Path, backup_dir: pathlib.Path) -> None:
    """Restore backed-up untracked files/directories after a git operation."""
    restored = False
    for relpath in _PRESERVE_DIRS:
        src = backup_dir / relpath
        if not src.exists():
            continue
        dst = sub_path / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            print(f"  Restoring {relpath}")
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
        else:
            print(f"  Restoring {relpath}")
            shutil.copy2(src, dst)
        restored = True

    shutil.rmtree(backup_dir)
    if restored:
        print("  Backup restored and cleaned up.")


def cmd_checkout(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    if _recipe_mode(recipe) == "board":
        print(f"Recipe '{recipe['name']}' is a physical board — no Chipyard checkout needed.")
        return 0

    chipyard_spec = recipe.get("chipyard", {})
    expected_branch = chipyard_spec.get("branch")
    expected_sha = chipyard_spec.get("sha")
    dry_run = args.dry_run

    # --- Stash dirty working tree if needed ---
    stashed = False
    dirty = _git(root, "status", "--porcelain")
    if dirty and not dry_run:
        stash_msg = f"merlin-checkout: {recipe['name']}"
        print(f"Stashing local changes in chipyard ({stash_msg})...")
        rc = utils.run(["git", "-C", str(root), "stash", "push", "-m", stash_msg], dry_run=dry_run)
        if rc != 0:
            utils.eprint("Failed to stash changes. Resolve manually before switching.")
            return rc
        stashed = True
    elif dirty and dry_run:
        print(f"+ git -C {root} stash push -m 'merlin-checkout: {recipe['name']}'")

    # --- Chipyard branch/SHA ---
    actual_branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    actual_sha = _git(root, "rev-parse", "HEAD")

    if expected_branch and actual_branch != expected_branch:
        print(f"Switching chipyard branch: {actual_branch} -> {expected_branch}")
        rc = utils.run(["git", "-C", str(root), "checkout", expected_branch], dry_run=dry_run)
        if rc != 0:
            utils.eprint(f"Failed to checkout branch '{expected_branch}'")
            if stashed:
                utils.eprint("Restoring stashed changes...")
                utils.run(["git", "-C", str(root), "stash", "pop"], dry_run=False)
            return rc
    elif expected_branch:
        print(f"  OK  chipyard branch: {actual_branch}")

    if expected_sha:
        actual_sha = _git(root, "rev-parse", "HEAD")
        if actual_sha == expected_sha:
            print(f"  OK  chipyard SHA: {actual_sha[:10]}")
        elif expected_branch:
            # Branch exists but is at a different SHA than the recipe pin.
            # Warn but stay on the branch to avoid detached HEAD.
            utils.eprint(
                f"  WARNING chipyard SHA: branch '{expected_branch}' is at"
                f" {actual_sha[:10]}, recipe pins {expected_sha[:10]}"
            )
        else:
            print(f"Pinning chipyard to SHA: {expected_sha[:10]}")
            rc = utils.run(
                ["git", "-C", str(root), "checkout", expected_sha],
                dry_run=dry_run,
            )
            if rc != 0:
                utils.eprint(f"Failed to checkout SHA '{expected_sha[:10]}'")
                return rc

    # --- Submodules ---
    for sub_name, sub_spec in recipe.get("submodules", {}).items():
        expected_sub_sha = sub_spec.get("sha")
        if not expected_sub_sha:
            continue

        sub_path = _find_submodule_path(root, sub_name)
        if not sub_path:
            # Try initializing the submodule first
            print(f"Submodule {sub_name} not found, attempting init...")
            # Back up any surviving untracked content before init
            # (e.g. deploy/ inside a partially-initialized firesim)
            backup = None
            for prefix in ["generators/", "sims/", ""]:
                candidate = f"{prefix}{sub_name}"
                candidate_path = root / candidate
                if candidate_path.is_dir() and not dry_run:
                    backup = _backup_untracked(candidate_path)
                rc = utils.run(
                    ["git", "-C", str(root), "submodule", "update", "--init", candidate],
                    dry_run=dry_run,
                )
                sub_path = root / candidate
                if sub_path.is_dir():
                    if backup and not dry_run:
                        _restore_untracked(sub_path, backup)
                        backup = None
                    break
            else:
                if backup and not dry_run:
                    shutil.rmtree(backup)
                utils.eprint(f"  SKIP submodule {sub_name}: could not find or init")
                continue

        actual_sub_sha = _git(sub_path, "rev-parse", "HEAD")

        # If the recipe specifies a remote repo, add/update the remote
        sub_repo = sub_spec.get("repo")
        if sub_repo:
            current_remote = _git(sub_path, "remote", "get-url", "origin")
            if current_remote != sub_repo:
                print(f"  Updating {sub_name} remote: {sub_repo}")
                if not dry_run:
                    # Update existing origin or add it
                    subprocess.run(
                        ["git", "-C", str(sub_path), "remote", "set-url", "origin", sub_repo],
                        capture_output=True,
                    )

        if actual_sub_sha == expected_sub_sha:
            print(f"  OK  submodule {sub_name}: {actual_sub_sha[:10]}")
            continue

        # Fetch and checkout the target SHA
        # Back up untracked user data (deploy/, results-build/, etc.)
        # before checkout which may reset the working tree.
        backup = _backup_untracked(sub_path) if not dry_run else None

        sub_branch = sub_spec.get("branch")
        print(f"  Checking out {sub_name}: {actual_sub_sha[:10]} -> {expected_sub_sha[:10]}")

        if sub_repo:
            utils.run(
                ["git", "-C", str(sub_path), "fetch", "origin"],
                dry_run=dry_run,
            )

        if sub_branch and not dry_run:
            # Try to checkout the branch, then verify SHA
            rc = utils.run(
                ["git", "-C", str(sub_path), "checkout", sub_branch],
                dry_run=dry_run,
            )
            new_sha = _git(sub_path, "rev-parse", "HEAD")
            if new_sha != expected_sub_sha:
                print(f"  Branch {sub_branch} HEAD is {new_sha[:10]}, pinning to {expected_sub_sha[:10]}")
                utils.run(
                    ["git", "-C", str(sub_path), "checkout", expected_sub_sha],
                    dry_run=dry_run,
                )
        else:
            utils.run(
                ["git", "-C", str(sub_path), "checkout", expected_sub_sha],
                dry_run=dry_run,
            )

        # Restore backed-up untracked content
        if backup:
            _restore_untracked(sub_path, backup)

    print(f"\nCheckout complete for recipe '{recipe['name']}'.")
    if stashed:
        print("  Local changes were stashed. Restore with: git -C <chipyard> stash pop")
    print(f"Run 'merlin chipyard validate {args.recipe}' to verify.")
    return 0
