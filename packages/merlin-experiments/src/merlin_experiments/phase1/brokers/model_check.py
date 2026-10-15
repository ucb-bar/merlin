"""Host-only broker for the visible split of Claude's model-check contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from merlin_experiments.phase1.run_inputs import strip_build_state

PROGRAMS = {"resnet50": ("model",),
            "smolvla": ("prefix_encode", "flow_denoise", "action_decode"),
            "tiny_llama": ("prefill", "decode")}
_GROUP = re.compile(r"[A-Za-z0-9_]+\Z")


def _safe_report(report: object) -> bool:
    text = json.dumps(report, ensure_ascii=True).lower()
    return not any(token in text for token in (
        '"sealed"', '"split": "sealed"', '/private/', 'private_corpus', 'sealed_inputs',
    ))


def execute_visible(ws: Path, request: dict, *, timeout_s: int = 7200, stop=None,
                    snapshot_source: Path | None = None) -> dict:
    """Run one fixed CLI request on a host-only immutable source snapshot."""
    action, model = request.get("action"), request.get("model")
    if action not in {"check", "replay-kernel"} or model not in PROGRAMS or request.get("split") != "visible":
        return {"status": "rejected", "error": "only declared visible model checks are admitted"}
    group, engine = request.get("group"), request.get("engine")
    if action == "replay-kernel" and (not isinstance(group, str) or not _GROUP.fullmatch(group)
                                        or engine not in {"spike", "gsim"}):
        return {"status": "rejected", "error": "replay requires a group and spike or gsim engine"}
    if action == "check" and (group is not None or engine is not None):
        return {"status": "rejected", "error": "check accepts only a model"}
    source = snapshot_source if snapshot_source is not None else ws / "submission"
    if source.is_symlink() or not (source / "manifest.yaml").is_file():
        return {"status": "pending", "error": "no stable submission manifest"}
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir or not Path(tmpdir).is_absolute():
        return {"status": "error", "error": "host TMPDIR is not task-owned"}
    with tempfile.TemporaryDirectory(prefix="model-check-", dir=tmpdir) as scratch:
        snapshot = Path(scratch) / "submission" if snapshot_source is None else snapshot_source
        if snapshot_source is None:
            shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
            strip_build_state(snapshot)
        from merlin.common.tree_hash import hash_tree

        digest = hash_tree(snapshot)["sha256"]
        report_path = Path(scratch) / "report.json"
        argv = [sys.executable, "-m", "merlin.model_check", action, "--submission", str(snapshot),
                "--model", model, "--split", "visible"]
        if action == "check":
            argv += ["--out", str(report_path)]
        else:
            argv += ["--group", group, "--engine", engine]
        stdout_path, stderr_path = Path(scratch) / "stdout", Path(scratch) / "stderr"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(argv, cwd=ws, stdout=stdout_file, stderr=stderr_file,
                                       start_new_session=True)
            deadline = time.monotonic() + timeout_s
            while process.poll() is None:
                if (stop is not None and stop()) or time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return {"status": "cancelled" if stop is not None and stop() else "timeout",
                            "model": model, "source_sha256": digest}
                time.sleep(0.25)
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8")) if action == "check" else json.loads(stdout)
        except (OSError, ValueError):
            report = None
        if report is not None and not _safe_report(report):
            return {"status": "error", "error": "model-check report contained sealed or private data",
                    "model": model, "source_sha256": digest}
        return {"status": "pass" if process.returncode == 0 and report is not None else "fail",
                "model": model, "action": action, "source_sha256": digest,
                "returncode": process.returncode, "report": report,
                **({"error": stderr[-2000:]} if process.returncode and report is None else {})}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=0.5)
    args = parser.parse_args(argv)
    ws = args.ws.resolve(strict=True)
    channel = ws / ".model_check_channel"
    channel.mkdir(exist_ok=True)
    parent = os.getppid()
    completed = {path.stem.removeprefix("resp_") for path in channel.glob("resp_*.json")}
    while not (channel / "STOP").exists() and os.getppid() == parent:
        for path in sorted(channel.glob("req_*.json")):
            job_id = path.stem.removeprefix("req_")
            if job_id in completed:
                continue
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
                result = execute_visible(ws, request, stop=lambda: (channel / "STOP").exists()
                                         or os.getppid() != parent)
            except (OSError, ValueError, TypeError) as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            target = channel / f"resp_{job_id}.json"
            temporary = channel / f".resp_{job_id}.tmp"
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, target)
            completed.add(job_id)
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
