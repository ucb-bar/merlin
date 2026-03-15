#!/usr/bin/env python3
import argparse
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

import utils

PREBUILT_ARTIFACTS = {
    "host-linux-x86_64": {
        "asset_name": "merlin-host-linux-x86_64.tar.gz",
        "build_dir": utils.REPO_ROOT / "build" / "host-merlin-perf",
        "layout": "host-install-root",
    },
    "host-macos": {
        "asset_name": "merlin-host-macos.tar.gz",
        "build_dir": utils.REPO_ROOT / "build" / "host-merlin-perf",
        "layout": "host-install-root",
    },
    "runtime-spacemit": {
        "asset_name": "merlin-runtime-spacemit.tar.gz",
        "build_dir": utils.REPO_ROOT / "build" / "spacemit-merlin-perf",
        "layout": "runtime-tree",
    },
    "runtime-saturnopu": {
        "asset_name": "merlin-runtime-saturnopu.tar.gz",
        "build_dir": utils.REPO_ROOT / "build" / "firesim-merlin-perf",
        "layout": "runtime-tree",
    },
}

CACHE_DIR = utils.REPO_ROOT / "build" / "downloads" / "prebuilts"


def release_asset_url(repo: str, tag: str, asset_name: str) -> str:
    if tag == "latest":
        return f"https://github.com/{repo}/releases/latest/download/{asset_name}"
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def download_file(url: str, dest: pathlib.Path, *, offline: bool, force: bool) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Reusing existing download: {dest}")
        return 0

    if offline:
        print(f"Offline mode enabled and file not already present: {dest}")
        return 1

    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"Download failed: {e}")
        return 1
    return 0


def safe_extract_tar(archive_path: pathlib.Path, dest_dir: pathlib.Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                member_path = (dest_dir / member.name).resolve()
                if not str(member_path).startswith(str(dest_dir.resolve())):
                    print(f"Refusing to extract suspicious path from archive: {member.name}")
                    return 1
            tar.extractall(dest_dir)
    except Exception as e:
        print(f"Extraction failed for {archive_path}: {e}")
        return 1
    return 0


def copy_tree_contents(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        dest_child = dst / child.name
        if child.is_dir():
            shutil.copytree(child, dest_child, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest_child)


def install_prebuilt(args) -> int:
    spec = PREBUILT_ARTIFACTS[args.artifact]
    asset_name = spec["asset_name"]
    build_dir = spec["build_dir"]
    layout = spec["layout"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / asset_name
    url = release_asset_url(args.repo, args.tag, asset_name)

    ret = download_file(url, archive_path, offline=args.offline, force=args.force)
    if ret != 0:
        return ret

    with tempfile.TemporaryDirectory(prefix="merlin-prebuilt-") as tmp:
        tmp_dir = pathlib.Path(tmp)
        ret = safe_extract_tar(archive_path, tmp_dir)
        if ret != 0:
            return ret

        entries = list(tmp_dir.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            print(f"Expected a single top-level directory in {archive_path}")
            return 1
        payload_root = entries[0]

        if build_dir.exists():
            if not args.force:
                print(f"Destination already exists: {build_dir}")
                print("Re-run with --force to replace it.")
                return 1
            shutil.rmtree(build_dir)

        build_dir.mkdir(parents=True, exist_ok=True)

        if layout == "host-install-root":
            shutil.copytree(payload_root, build_dir / "install", dirs_exist_ok=True)
        elif layout == "runtime-tree":
            copy_tree_contents(payload_root, build_dir)
        else:
            print(f"Unknown prebuilt layout: {layout}")
            return 1

    print(f"Installed prebuilt artifact '{args.artifact}' into {build_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Merlin prebuilt artifacts into build/ layout.")
    parser.add_argument("--artifact", choices=sorted(PREBUILT_ARTIFACTS.keys()), required=True)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--repo", default="ucb-bar/merlin")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    return install_prebuilt(args)


if __name__ == "__main__":
    sys.exit(main())
