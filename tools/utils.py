#!/usr/bin/env python3
# tools/utils.py

import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, Optional, Sequence

# Constants
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS_CONFIG = REPO_ROOT / "config" / "targets.json"

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)

def _shell_quote(text: str) -> str:
    if text == "":
        return "''"
    if all(ch.isalnum() or ch in "._-/:=+" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"

def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[pathlib.Path] = None,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    cmd_str = " ".join(_shell_quote(x) for x in cmd)
    print(f"+ {cmd_str}")
    if dry_run:
        return 0
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    # Flush stdout so Python output appears before subprocess output
    sys.stdout.flush()
    completed = subprocess.run(
        list(cmd), cwd=str(cwd or REPO_ROOT), env=merged_env, check=False
    )
    return completed.returncode

def resolve_repo_path(relative: str) -> pathlib.Path:
    return (REPO_ROOT / relative).resolve()

def load_targets_config() -> dict:
    if not TARGETS_CONFIG.exists():
        return {}
    with TARGETS_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)

def run_repo_script(
    relative_script: str, script_args: Sequence[str], dry_run: bool
) -> int:
    script = resolve_repo_path(relative_script)
    if not script.exists():
        eprint(f"Script not found: {script}")
        return 2
    return run(["bash", str(script), *script_args], dry_run=dry_run)