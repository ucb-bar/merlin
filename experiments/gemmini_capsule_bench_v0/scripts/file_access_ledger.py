"""Complete per-run FILE-ACCESS LEDGER — every file the agent touched and what it did with it.

Where transcript_tooling_audit is the pass/fail GATE (pattern-greps for cheats), this is the exhaustive
RECORD for human review: it walks a run's round transcripts and emits, in order, EVERY file-touching tool
call —
  • Read  -> op=read,  path, bytes returned (the content IS in the transcript; we record its size + a head)
  • Write/Edit/MultiEdit -> op=write/edit, path, bytes written
  • Bash  -> op=exec, the verbatim command, the path-like tokens it referenced, and stdout/stderr size
each classified IN-SCOPE (under the run's granted bundle paths / workspace / system+venv) or OUT-OF-SCOPE
(any other repo path or foreign tree), reusing the audit's cheat/contaminant patterns so the ledger and the
gate agree.

HONEST LIMITATION (printed in the header): this captures TOOL-level access. A path opened *inside* a
subprocess the agent spawns (e.g. files a `python build.py` reads internally) is not individually
enumerated — only the Bash command text + the paths visible in it are. Syscall-level capture would need
strace/auditd/LD_PRELOAD, which this harness does not run. For answer-surface leakage this is mitigated by
(a) the chmod-000 lockdown of answer surfaces and (b) the gate flagging the command text itself.

-> run_dir/file_access_ledger.{json,md}.  Usage: file_access_ledger.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parent.parent
sys.path.insert(0, str(EXP / "scripts"))
import transcript_tooling_audit as TTA  # reuse cheat/contaminant classification

REPO_S = str(REPO)
# path-like token extractor for Bash commands (absolute, or repo-relative with a slash + known suffix/dir)
PATH_RX = re.compile(r"(/[\w./+-]+|(?:[\w.+-]+/)+[\w.+-]+)")
SYS_PREFIXES = ("/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc", "/proc", "/sys", "/dev", "/tmp", "/opt")


def _scope_of(path: str, allowed: list[str], ws_names: list[str]) -> str:
    """Classify a referenced path: cheat / contaminant:<k> / system / in_scope / repo_other."""
    blob = path
    for k, rx in TTA.CHEAT_PATTERNS.items():
        if rx.search(blob):
            return f"CHEAT:{k}"
    for k, rx in TTA.CONTAMINANTS.items():
        if rx.search(blob):
            return f"OOB:{k}"
    if path.startswith(SYS_PREFIXES):
        return "system"
    # in-repo or workspace?
    p = path.lstrip("./")
    if path.startswith(REPO_S) or any(seg in path for seg in ws_names):
        # under an allowed bundle path?
        if any(a.rstrip("/") in path for a in allowed):
            return "in_scope"
        return "repo_in_workspace"
    if any(a and a.rstrip("/") in p for a in allowed):
        return "in_scope"
    if "/" not in path and "." in path:
        return "rel_local"   # bare filename, almost always workspace-local
    return "repo_other"


def _allowed_paths(run_dir: Path) -> list[str]:
    m = run_dir / "input_bundle_manifest.yaml"
    if not m.is_file():
        return []
    d = yaml.safe_load(m.read_text()) or {}
    out = []
    for e in d.get("allowed", []) or []:
        if isinstance(e, dict):
            if e.get("path"):
                out.append(e["path"])
            if e.get("as"):
                out.append(e["as"])
    return out


def ledger(run_dir: Path) -> dict:
    allowed = _allowed_paths(run_dir)
    ws_names = [run_dir.name, "_qa_ws", "workspace"]
    rounds_dir = run_dir / "rounds"
    events = []
    # map tool_use_id -> result size/head (results live in subsequent `user` records)
    results: dict[str, dict] = {}
    raw = []
    for tp in sorted(rounds_dir.glob("round_*.transcript.jsonl")) if rounds_dir.is_dir() else []:
        rnd = tp.name
        for ln in tp.read_text(errors="ignore").splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            raw.append((rnd, o))
    # first pass: collect tool_result outputs
    for rnd, o in raw:
        for b in (o.get("message", {}) or {}).get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b.get("content")
                s = c if isinstance(c, str) else json.dumps(c) if c else ""
                results[b.get("tool_use_id", "")] = {"bytes": len(s), "head": s[:120]}
    # second pass: enumerate tool_use file accesses
    for rnd, o in raw:
        if o.get("type") != "assistant":
            continue
        for b in (o.get("message", {}) or {}).get("content", []) or []:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            name = b.get("name"); inp = b.get("input", {}) or {}; tid = b.get("id", "")
            res = results.get(tid, {})
            if name == "Read":
                p = inp.get("file_path", "")
                events.append({"round": rnd, "op": "read", "path": p, "scope": _scope_of(p, allowed, ws_names),
                               "result_bytes": res.get("bytes")})
            elif name in ("Write", "Edit", "MultiEdit"):
                p = inp.get("file_path", "")
                events.append({"round": rnd, "op": name.lower(), "path": p,
                               "scope": _scope_of(p, allowed, ws_names),
                               "wrote_bytes": len(inp.get("content", "") or inp.get("new_string", "") or "")})
            elif name == "Bash":
                cmd = inp.get("command", "")
                toks = [t for t in PATH_RX.findall(cmd) if ("/" in t and (t.startswith("/") or "." in t))]
                refs = []
                for t in dict.fromkeys(toks):           # dedup, keep order
                    sc = _scope_of(t, allowed, ws_names)
                    if sc not in ("rel_local",):
                        refs.append({"path": t, "scope": sc})
                events.append({"round": rnd, "op": "exec", "cmd": cmd[:240],
                               "refs": refs, "stdout_bytes": res.get("bytes")})
    # rollups
    def _paths(op):
        return sorted({e["path"] for e in events if e.get("op") == op and e.get("path")})
    oob = [e for e in events if (e.get("scope", "").startswith(("CHEAT", "OOB"))
                                 or any(r["scope"].startswith(("CHEAT", "OOB")) for r in e.get("refs", [])))]
    return {
        "run_id": run_dir.name,
        "n_events": len(events),
        "files_read": _paths("read"),
        "files_written": sorted({e["path"] for e in events if e.get("op") in ("write", "edit", "multiedit")}),
        "n_bash": sum(1 for e in events if e.get("op") == "exec"),
        "out_of_scope_events": oob,
        "events": events,
    }


def _write_md(run_dir: Path, L: dict):
    out = [f"# File-access ledger — {L['run_id']}", "",
           f"- events: {L['n_events']}  ·  files read: {len(L['files_read'])}  ·  files written: "
           f"{len(L['files_written'])}  ·  bash: {L['n_bash']}  ·  **out-of-scope events: "
           f"{len(L['out_of_scope_events'])}**",
           "- NOTE: tool-level capture; paths opened *inside* subprocesses are not individually listed "
           "(see script header).", ""]
    if L["out_of_scope_events"]:
        out += ["## ⚠ Out-of-scope accesses (review)", ""]
        for e in L["out_of_scope_events"][:50]:
            if e["op"] == "exec":
                bad = ", ".join(f"{r['path']} [{r['scope']}]" for r in e["refs"]
                                if r["scope"].startswith(("CHEAT", "OOB")))
                out.append(f"- `{e['round']}` exec: `{e['cmd'][:80]}` → {bad}")
            else:
                out.append(f"- `{e['round']}` {e['op']}: `{e['path']}` [{e['scope']}]")
        out.append("")
    out += ["## Files read", ""] + [f"- {p}" for p in L["files_read"]] or []
    out += ["", "## Files written", ""] + [f"- {p}" for p in L["files_written"]]
    (run_dir / "file_access_ledger.md").write_text("\n".join(out) + "\n")


def main(argv=None):
    args = argv or sys.argv[1:]
    if not args:
        print(__doc__); return 2
    for a in args:
        rd = Path(a)
        L = ledger(rd)
        (rd / "file_access_ledger.json").write_text(json.dumps(L, indent=2))
        _write_md(rd, L)
        print(f"\n== {L['run_id']} ==")
        print(f"  events={L['n_events']}  read={len(L['files_read'])}  written={len(L['files_written'])}"
              f"  bash={L['n_bash']}  out-of-scope={len(L['out_of_scope_events'])}")
        for e in L["out_of_scope_events"][:10]:
            tag = e.get("cmd", e.get("path", ""))
            print(f"    ⚠ {e['round']} {e['op']}: {str(tag)[:80]}")
        print(f"  -> {rd/'file_access_ledger.json'} (+ .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
