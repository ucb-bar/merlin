"""Chipyard root-path persistence + accessors.

The chipyard checkout location is recorded once in `.chipyard_config.json`
at the merlin repo root, then read by every subsequent `./merlin chipyard`
subaction.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import utils

CONFIG_FILE = utils.REPO_ROOT / ".chipyard_config.json"


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_chipyard_root(args: argparse.Namespace) -> pathlib.Path | None:
    """Resolve chipyard root: --chipyard-root > $CHIPYARD_ROOT > saved config."""
    if getattr(args, "chipyard_root", None):
        return pathlib.Path(args.chipyard_root).resolve()
    env_val = os.environ.get("CHIPYARD_ROOT")
    if env_val:
        return pathlib.Path(env_val).resolve()
    saved = _load_config().get("chipyard_root")
    if saved:
        return pathlib.Path(saved).resolve()
    return None


def require_chipyard_root(args: argparse.Namespace) -> pathlib.Path | None:
    root = get_chipyard_root(args)
    if not root:
        utils.eprint("Chipyard root not configured.")
        utils.eprint("  merlin chipyard set-path /path/to/chipyard")
        utils.eprint("  OR: export CHIPYARD_ROOT=/path/to/chipyard")
        return None
    if not root.is_dir():
        utils.eprint(f"Chipyard root does not exist: {root}")
        return None
    return root


def cmd_set_path(args: argparse.Namespace) -> int:
    path = pathlib.Path(args.path).resolve()
    if not path.is_dir():
        utils.eprint(f"Directory does not exist: {path}")
        return 1

    cfg = _load_config()
    cfg["chipyard_root"] = str(path)
    if not args.dry_run:
        _save_config(cfg)
    print(f"Chipyard root saved: {path}")
    os.environ["CHIPYARD_ROOT"] = str(path)
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
