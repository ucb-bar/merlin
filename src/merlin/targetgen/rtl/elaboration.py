"""Reproduce selected FIRRTL from a pinned, explicit source checkout.

This binds a configuration invocation to exact output bytes. It cannot prove
the semantics of the elaborator or close undeclared runtime dependencies.
Nested submodule pins are resolved through their explicitly pinned parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SCHEMA = "merlin.rtl_elaboration.v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"selected Git source cannot be verified: {result.stderr.strip()}")
    return result.stdout.strip()


def _revision(value: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("elaboration source revision must be an exact Git commit")
    return value


def _source(root: Path, revision: str, config_file: str, config: str, submodules: dict[str, str]) -> dict:
    root = root.resolve(strict=True)
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("elaboration source root must be the Git checkout root")
    if _git(root, "rev-parse", "HEAD") != _revision(revision):
        raise ValueError("elaboration source checkout differs from selected revision")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("elaboration source has changed tracked files")
    if not isinstance(config_file, str) or not config_file or Path(config_file).is_absolute() or ".." in Path(config_file).parts:
        raise ValueError("configuration source must be a checkout-relative file")
    selected = root / config_file
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("configuration source is absent or indirect")
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{config_file}"],
        capture_output=True, check=False,
    )
    if committed.returncode or committed.stdout != selected.read_bytes():
        raise ValueError("configuration source differs from selected Git object")
    if not isinstance(config, str) or not config or config.encode() not in committed.stdout:
        raise ValueError("selected configuration symbol is absent from committed source")
    if not isinstance(submodules, dict):
        raise ValueError("submodule pins must be a mapping")
    pinned = {}
    for name, expected in sorted(submodules.items()):
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("submodule path must be checkout-relative")
        _revision(expected)
        parent_name = max(
            (prefix for prefix in pinned if name.startswith(prefix + "/")),
            key=len, default="",
        )
        parent = root / parent_name if parent_name else root
        relative_name = name[len(parent_name) + 1:] if parent_name else name
        parent_revision = pinned[parent_name] if parent_name else revision
        metadata, separator, selected_name = _git(parent, "ls-tree", parent_revision, "--", relative_name).partition("\t")
        if not separator or metadata.split() != ["160000", "commit", expected] or selected_name != relative_name:
            raise ValueError(f"selected submodule Git link differs: {name}")
        child = (root / name).resolve(strict=True)
        if _git(child, "rev-parse", "HEAD") != expected or _git(child, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"selected submodule checkout differs: {name}")
        pinned[name] = expected
    return {
        "root": str(root), "revision": revision, "config_file": config_file,
        "config_sha256": _sha(selected), "config": config, "submodules": pinned,
    }


def _tool(argv: list[str], inputs: list[Path] | None = None) -> dict:
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError("elaboration command must be a nonempty argument vector")
    executable = shutil.which(argv[0])
    if executable is None or not Path(executable).is_file():
        raise ValueError("elaboration executable is unavailable")
    path = Path(executable).resolve(strict=True)
    pinned_inputs = []
    seen = set()
    for input_path in inputs or []:
        selected = Path(input_path).resolve(strict=True)
        if not selected.is_file() or selected in seen:
            raise ValueError("elaboration tool inputs must be distinct existing files")
        seen.add(selected)
        pinned_inputs.append({"path": str(selected), "sha256": _sha(selected)})
    return {"path": str(path), "sha256": _sha(path), "inputs": pinned_inputs}


def _command(argv: list[str], *, config: str, output: Path) -> list[str]:
    if sum(arg.count("{firrtl}") for arg in argv) != 1 or not any(config in arg for arg in argv):
        raise ValueError("elaboration command must name the selected config and one {firrtl} output")
    return [arg.replace("{firrtl}", str(output)) for arg in argv]


def issue(
    *, source_root: Path, revision: str, config_file: str, config: str,
    command: list[str], output: Path, submodules: dict[str, str] | None = None,
    tool_inputs: list[Path] | None = None,
) -> Path:
    """Run twice into fresh roots and retain both results and exact invocation logs."""
    selected = _source(source_root, revision, config_file, config, submodules or {})
    tool = _tool(command, tool_inputs)
    output = output.resolve()
    _command(command, config=config, output=output / "run1" / "selected.fir")
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for index in (1, 2):
        run = output / f"run{index}"
        run.mkdir()
        firrtl = run / "selected.fir"
        argv = _command(command, config=config, output=firrtl)
        result = subprocess.run(argv, cwd=selected["root"], capture_output=True, check=False)
        (run / "stdout.log").write_bytes(result.stdout)
        (run / "stderr.log").write_bytes(result.stderr)
        if result.returncode or not firrtl.is_file() or not firrtl.stat().st_size:
            raise RuntimeError(f"elaboration run {index} failed or emitted no FIRRTL; inspect {run}")
        if _source(source_root, revision, config_file, config, submodules or {}) != selected or _tool(command, tool_inputs) != tool:
            raise RuntimeError("selected elaboration source or tool changed during reproduction")
        runs.append({
            "argv": argv, "returncode": result.returncode, "firrtl": str(firrtl),
            "firrtl_sha256": _sha(firrtl), "stdout_sha256": _sha(run / "stdout.log"),
            "stderr_sha256": _sha(run / "stderr.log"),
        })
    if runs[0]["firrtl_sha256"] != runs[1]["firrtl_sha256"]:
        raise RuntimeError("fresh elaborations produced different FIRRTL bytes")
    receipt = {
        "schema": SCHEMA, "status": "reproduced_exact_firrtl",
        "source": selected, "tool": tool, "command_template": command,
        "runs": runs, "firrtl_sha256": runs[0]["firrtl_sha256"],
        "qualification": "two fresh local invocations with pinned tracked source and executable bytes; not a closed build environment or hardware-semantic proof",
    }
    path = output / "elaboration.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def verify(path: Path, *, firrtl: Path, config: str | None = None) -> dict:
    """Check the selected bytes and source identity without rerunning elaboration."""
    path = path.resolve(strict=True)
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA or receipt.get("status") != "reproduced_exact_firrtl":
        raise ValueError("unsupported or unsuccessful elaboration receipt")
    source = receipt.get("source") or {}
    observed = _source(Path(source["root"]), source["revision"], source["config_file"], source["config"], source["submodules"])
    if observed != source or (config is not None and config != source["config"]):
        raise ValueError("selected elaboration source or configuration changed")
    tool_inputs = [Path(row["path"]) for row in receipt.get("tool", {}).get("inputs", [])]
    if _tool(receipt["command_template"], tool_inputs) != receipt.get("tool"):
        raise ValueError("selected elaboration tool changed")
    runs = receipt.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("elaboration receipt needs two fresh runs")
    for index, run in enumerate(runs, 1):
        expected = path.parent / f"run{index}" / "selected.fir"
        if (
            run.get("firrtl") != str(expected) or run.get("argv") != _command(
                receipt["command_template"], config=source["config"], output=expected,
            ) or run.get("returncode") != 0 or not expected.is_file() or not expected.stat().st_size
            or _sha(expected) != run.get("firrtl_sha256")
            or _sha(path.parent / f"run{index}" / "stdout.log") != run.get("stdout_sha256")
            or _sha(path.parent / f"run{index}" / "stderr.log") != run.get("stderr_sha256")
        ):
            raise ValueError(f"elaboration run {index} bytes or command changed")
    if runs[0]["firrtl_sha256"] != runs[1]["firrtl_sha256"] or _sha(firrtl) != receipt.get("firrtl_sha256") or _sha(firrtl) != runs[0]["firrtl_sha256"]:
        raise ValueError("selected FIRRTL differs from reproduced elaboration")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "revision", "config-file", "config", "command-json", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--submodule", action="append", default=[], metavar="PATH=COMMIT")
    parser.add_argument("--tool-input", action="append", default=[], metavar="FILE")
    args = parser.parse_args(argv)
    try:
        pins = dict(part.split("=", 1) for part in args.submodule)
        if len(pins) != len(args.submodule):
            raise ValueError("duplicate selected submodule")
        command = json.loads(Path(args.command_json).read_text())
        result = issue(
            source_root=Path(args.source_root), revision=args.revision,
            config_file=args.config_file, config=args.config,
            command=command, output=Path(args.output), submodules=pins,
            tool_inputs=[Path(path) for path in args.tool_input],
        )
        print(json.dumps({"status": "REPRODUCED", "receipt": str(result)}))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
