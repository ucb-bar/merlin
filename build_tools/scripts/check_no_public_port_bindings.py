#!/usr/bin/env python3
"""Lint gate: a container port published to the host must name an interface.

WHY THIS EXISTS, measured 2026-09-21. A SigNoz stack installed under ``/tmp`` published its UI
on ``0.0.0.0:8080`` and its OTLP collector on ``0.0.0.0:4317``/``4318`` of a host with a publicly
routable address. It carried ``restart: unless-stopped``, so it came back 16 seconds after every
reboot with nobody deciding to start it, and it had been reachable for 4.5 days when it was found.
Nothing had ever used it: one account created at install, zero dashboards, and no telemetry
ingested after the second day.

The exposure was never required by anything. The harness it existed for talks to
``localhost:4318``; publishing on every interface was purely the default spelling of ``-p``.

THE RULE. ``-p 4318:4318`` and ``ports: ["5000:5000"]`` bind every interface, including the public
one. ``-p 127.0.0.1:4318:4318`` binds loopback. Both reach a local client identically, so the
loopback form costs nothing and is what this gate requires. Reach a remote host over
``ssh -L``, which authenticates, rather than by opening the port to the network.

WHAT IS *NOT* A VIOLATION. A server process inside a container binding ``0.0.0.0`` is correct and
necessary -- a container process listening on ``127.0.0.1`` is unreachable even through its own
published port. Host exposure is decided by the publish, so that is the only thing scanned here.
Flagging in-container binds instead breaks the service and leaves the exposure in place.

Scope: tracked YAML compose files and any tracked text naming ``docker run``. Parsed structurally
(YAML loader + token walk), never by pattern-matching, per this repo's no-regex rule.

Escape hatch, for a port that genuinely must be reachable off-box: an inline
``# public-port-ok: <rationale>`` comment on the line, or a whole-file entry in
``build_tools/scripts/public_port_allowlist.txt``. The allowlist only ever shrinks.

    python build_tools/scripts/check_no_public_port_bindings.py             # full scan
    python build_tools/scripts/check_no_public_port_bindings.py --staged    # staged files only
    python build_tools/scripts/check_no_public_port_bindings.py --stop-hook # Stop-hook JSON
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "public_port_allowlist.txt"
INLINE_OK = "# public-port-ok:"

#: Host IPs that do not reach the network. An empty host IP means "every interface".
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

YAML_SUFFIXES = frozenset({".yml", ".yaml"})
TEXT_SUFFIXES = frozenset({".md", ".sh", ".bash", ".rst", ".txt", ".py"})
PUBLISH_FLAGS = frozenset({"-p", "--publish"})


class Violation:
    def __init__(self, path: Path, line: int, spec: str, why: str) -> None:
        self.path, self.line, self.spec, self.why = path, line, spec, why

    def __str__(self) -> str:
        rel = self.path.relative_to(ROOT) if self.path.is_absolute() else self.path
        return f"{rel}:{self.line}: {self.spec!r} -- {self.why}"


def _allowlist() -> set[str]:
    if not ALLOW_FILE.exists():
        return set()
    out = set()
    for raw in ALLOW_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def _host_ip_of(spec: str) -> tuple[str | None, bool]:
    """Return ``(host_ip, publishes)`` for one short-form port spec.

    ``publishes`` is False for a bare container port (``"8080"``), which exposes nothing to the
    host. IPv6 host IPs are bracketed (``[::1]:5000:5000``); a bare ``::1`` is ambiguous with the
    separator and is not valid short form, so only the bracketed spelling is recognised.
    """
    spec = spec.strip()
    if not spec:
        return None, False
    if spec.startswith("["):
        close = spec.find("]")
        if close == -1:
            return None, True  # malformed; treat as publishing so it is surfaced, not skipped
        host = spec[1:close]
        rest = spec[close + 1 :].lstrip(":")
        return host, ":" in rest or bool(rest)
    parts = spec.split(":")
    if len(parts) == 1:
        return None, False  # container port only -- no host binding
    if len(parts) == 2:
        return "", True  # "5000:5000" -- every interface
    return parts[0], True


def _judge(spec: str, host_ip: str | None) -> str | None:
    if host_ip is None:
        return None
    if host_ip == "":
        return "publishes on every interface; prefix the host port with 127.0.0.1"
    if host_ip in LOOPBACK:
        return None
    if host_ip == "0.0.0.0" or host_ip == "::":
        return "0.0.0.0 is every interface; use 127.0.0.1"
    return f"host IP {host_ip!r} is not loopback; use 127.0.0.1 unless it must be reachable off-box"


def _line_of(needle: str, lines: list[str], start: int = 0) -> int:
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return 1


def _scan_compose(path: Path, lines: list[str]) -> list[Violation]:
    try:
        doc = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError:
        return []  # not our file to validate; other gates own YAML health
    if not isinstance(doc, dict):
        return []
    services = doc.get("services")
    if not isinstance(services, dict):
        return []

    found: list[Violation] = []
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        ports = svc.get("ports")
        if not isinstance(ports, list):
            continue
        for entry in ports:
            if isinstance(entry, dict):  # long form
                if "published" not in entry:
                    continue
                host_ip = entry.get("host_ip", "")
                spec = f"{name}: published {entry['published']}"
            else:
                spec = str(entry)
                host_ip, publishes = _host_ip_of(spec)
                if not publishes:
                    continue
            why = _judge(spec, host_ip if isinstance(host_ip, str) else "")
            if why is None:
                continue
            ln = _line_of(str(entry if not isinstance(entry, dict) else entry["published"]), lines)
            if INLINE_OK in lines[ln - 1]:
                continue
            found.append(Violation(path, ln, spec, why))
    return found


def _scan_text(path: Path, lines: list[str]) -> list[Violation]:
    found: list[Violation] = []
    for i, line in enumerate(lines, start=1):
        if "docker run" not in line and "docker create" not in line:
            # a publish flag may sit on a continued line; only scan blocks that opened one
            if not any("docker run" in prev or "docker create" in prev for prev in lines[max(0, i - 6) : i]):
                continue
        if INLINE_OK in line:
            continue
        tokens = line.replace("\\", " ").split()
        for j, tok in enumerate(tokens):
            spec = None
            if tok in PUBLISH_FLAGS and j + 1 < len(tokens):
                spec = tokens[j + 1]
            elif tok.startswith("--publish="):
                spec = tok.split("=", 1)[1]
            if spec is None:
                continue
            spec = spec.strip("\"'")
            host_ip, publishes = _host_ip_of(spec)
            if not publishes:
                continue
            why = _judge(spec, host_ip)
            if why is not None:
                found.append(Violation(path, i, spec, why))
    return found


def _candidates(staged: bool) -> list[Path]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout
    paths = []
    for rel in out.splitlines():
        p = ROOT / rel
        if not p.is_file():
            continue
        if p.suffix in YAML_SUFFIXES or p.suffix in TEXT_SUFFIXES:
            paths.append(p)
    return paths


def scan(staged: bool = False) -> list[Violation]:
    allow = _allowlist()
    found: list[Violation] = []
    for path in _candidates(staged):
        rel = str(path.relative_to(ROOT))
        if rel in allow:
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if path.suffix in YAML_SUFFIXES:
            found.extend(_scan_compose(path, lines))
        if path.suffix in TEXT_SUFFIXES or path.suffix in YAML_SUFFIXES:
            found.extend(_scan_text(path, lines))
    return found


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    found = scan(staged=staged)

    if stop_hook:
        if found:
            msg = "Container ports published on every interface:\n" + "\n".join(f"  {v}" for v in found)
            print(json.dumps({"decision": "block", "reason": msg}))
        else:
            print(json.dumps({}))
        return 0

    if not found:
        return 0
    print("Container ports published on every interface (use 127.0.0.1):", file=sys.stderr)
    for v in found:
        print(f"  {v}", file=sys.stderr)
    print(
        "\nA local client reaches 127.0.0.1 identically. For a remote host use "
        "`ssh -L <port>:localhost:<port> <host>`.\n"
        f"Genuinely must be off-box? Add `{INLINE_OK} <rationale>` on the line, or the file to "
        f"{ALLOW_FILE.relative_to(ROOT)}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
