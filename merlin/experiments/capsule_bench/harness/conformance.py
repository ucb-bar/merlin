"""Per-round DEV-CONFORMANCE verdict — did the agent actually DEVELOP the way its arm mandates?

This is a first-class completion gate while leaving the capsule ORACLE grade unchanged: numeric evidence
still reports when a round is non-conformant, but that round cannot be frozen as a valid arm-N result.
It catches the case where a run scores 0 by NOT using the mandated tooling (an invented encoding, a regex
parser, no lever enumeration), which the numeric grade alone cannot distinguish from a genuine capability
wall.

Every signal derives from PERSISTED artifacts — the round transcript + the frozen submission tree — so
it is computable post-hoc on any recorded run and needs no live sandbox. Fully target-agnostic (no
accelerator name, no golden); the only "regex" here is the repo's own check_no_regex AST scan, reused.

Checks (each ARM-APPLICABLE — skipped where the arm does not grant the tool, so it is never a false fail):
  * no_regex_ok    — the submission's own .py author no regex (the repo cardinal rule)
  * isa_tools_used — an assisted arm exercised the derived ISA dev tools (lint/disasm/asm/debug) at all
  * asm_used       — an external_backend arm assembled words via `isa_tools asm` (external_backend only)
  * cca_used       — the CCA lever-set enumeration ran (check_bijection / escalation_ladder, CLI or API)
  * full_selfcheck — a self-check covering ALL capsules ran (not just one capsule)
``conformant`` is the AND of the applicable checks.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

_ASSISTED = ("merlin_assisted", "merlin_rtlchecks")


def _load_scan_file():
    """Reuse build_tools/scripts/check_no_regex.py's per-file AST scanner (the canonical repo rule) rather
    than re-implement it. Driver-side only (merlin importable); returns None if unavailable."""
    try:
        from merlin.common.paths import repo_root
        p = repo_root() / "build_tools" / "scripts" / "check_no_regex.py"
        spec = importlib.util.spec_from_file_location("_check_no_regex", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._scan_file
    except Exception:  # noqa: BLE001
        return None


def _submission_regex(sub_dir: Path) -> list[dict]:
    """List every regex use in the submission's own Python (path, line, kind). Empty => clean."""
    scan = _load_scan_file()
    if scan is None or not sub_dir.exists():
        return []
    hits: list[dict] = []
    for py in sorted(sub_dir.rglob("*.py")):
        if "__pycache__" in py.parts or "build" in py.parts:
            continue
        try:
            for line, kind in scan(py):
                hits.append({"file": str(py.relative_to(sub_dir)), "line": line, "kind": kind})
        except Exception:  # noqa: BLE001
            continue
    return hits


@dataclass(frozen=True)
class ToolCall:
    """One executed tool request paired with the result the agent actually received."""

    name: str
    input: dict
    tool_use_id: str
    result_present: bool
    succeeded: bool


def _tool_calls(tpath: Path) -> list[ToolCall]:
    """Every assistant tool request, correlated with its persisted result.

    Merely mentioning or issuing a required command is not evidence that the tool was available.  Codex
    records an explicit ``is_error`` bit and is checked strictly.  Older translated drivers did not all
    persist that bit, so they retain compatibility only when a correlated result is present; a missing
    result always fails closed.
    """
    events: list[dict] = []
    if not tpath.exists():
        return []
    for ln in tpath.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(e, dict):
            events.append(e)

    driver = next((str(e.get("driver") or "") for e in events
                   if e.get("type") == "system" and e.get("subtype") == "init"), "")
    results: dict[str, dict] = {}
    for e in events:
        if e.get("type") != "user":
            continue
        for block in (e.get("message") or {}).get("content", []) or []:
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and block.get("tool_use_id")):
                results[str(block["tool_use_id"])] = block

    out: list[ToolCall] = []
    anonymous = 0
    for e in events:
        if e.get("type") != "assistant":
            continue
        message = e.get("message") or {}
        message_id = str(message.get("id") or "")
        for b in message.get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                inp = b.get("input")
                # Native Claude/OpenCode blocks carry ``id``.  The Codex translator stores the same
                # correlation id on the enclosing assistant message and on the later tool_result.
                tool_id = str(b.get("id") or message_id)
                if not tool_id:
                    anonymous += 1
                    tool_id = f"__missing_tool_id_{anonymous}"
                result = results.get(tool_id)
                explicit_error = result.get("is_error") if result is not None else None
                succeeded = bool(result is not None and (
                    explicit_error is False
                    or (driver != "codex" and explicit_error is not True)
                ))
                out.append(ToolCall(
                    name=str(b.get("name") or ""),
                    input=inp if isinstance(inp, dict) else {"_raw": inp},
                    tool_use_id=tool_id,
                    result_present=result is not None,
                    succeeded=succeeded,
                ))
    return out


def _exec_str(call: ToolCall) -> str:
    """The COMMAND actually executed by this tool call — the bash command, not a written file's content.
    (Matching the whole serialized input would count a tool named only in prose inside a write_file, which
    is exactly the false positive we must avoid: an agent that WRITES 'run isa_tools.py asm' in its plan but
    never runs it is not conformant.)"""
    cmd = (call.input.get("command") or call.input.get("cmd")
           or call.input.get("_raw") or "")
    return f"{call.name} {cmd}"


def _any(calls: list[ToolCall], *needles: str) -> bool:
    return any(call.succeeded and any(n in _exec_str(call) for n in needles) for call in calls)


def _full_selfcheck(calls: list[ToolCall]) -> bool:
    for call in calls:
        if not call.succeeded:
            continue
        ex = _exec_str(call)
        # a CLI self-check over the whole set (claude / opencode run it as bash)
        if "agent_selfcheck" in ex and ("--capsules all" in ex or '--capsules "all"' in ex):
            return True
        # the bedrock self_check tool: default (no capsules key) or an explicit "all"
        if call.name == "self_check" and call.input.get("capsules") in (None, "", "all"):
            return True
    return False


def compute(tpath: Path, sub_dir: Path, arm: str, endpoint_kind: str) -> dict:
    """Compute the conformance verdict for one round. Returns a dict with per-check booleans (or None when
    the check does not apply to this arm), the regex-hit detail, and the overall ``conformant`` flag."""
    assisted = arm in _ASSISTED
    external = endpoint_kind == "external_backend"
    calls = _tool_calls(tpath)
    regex_hits = _submission_regex(sub_dir) if assisted else []   # no-regex is an ASSISTED-arm mandate (xDSL authoring)
    has_py = bool(list(sub_dir.rglob("*.py"))) if sub_dir.exists() else False

    checks: dict = {}
    # no-regex applies only where the arm is mandated to author in xDSL AND actually wrote python
    checks["no_regex_ok"] = (not regex_hits) if (assisted and has_py) else None
    checks["isa_tools_used"] = _any(
        calls,
        "isa_tools.py asm", "isa_tools.py disasm", "isa_tools.py lint", "isa_tools.py debug",
        "isa_tools asm", "isa_tools disasm", "isa_tools lint", "isa_tools debug",
    ) if assisted else None
    checks["asm_used"] = _any(calls, "isa_tools.py asm", "isa_tools asm") if (assisted and external) else None
    checks["cca_used"] = (
        _any(calls, "cca_contract.py check-bijection", "check_bijection")
        and _any(calls, "action_catalog.py escalation-ladder", "escalation_ladder")
    ) if assisted else None
    checks["full_selfcheck"] = _full_selfcheck(calls)

    applicable = [v for v in checks.values() if v is not None]
    conformant = all(applicable) if applicable else True
    tool_evidence = {
        "n_calls": len(calls),
        "n_successful": sum(call.succeeded for call in calls),
        "n_failed": sum(call.result_present and not call.succeeded for call in calls),
        "n_missing_results": sum(not call.result_present for call in calls),
    }
    return {"conformant": conformant, "checks": checks, "regex_hits": regex_hits,
            "tool_evidence": tool_evidence, "arm": arm, "endpoint_kind": endpoint_kind}


def failing_checks(verdict: dict) -> list[str]:
    """The applicable checks that came back False — what a WARNING should name."""
    return [k for k, v in (verdict.get("checks") or {}).items() if v is False]


def main(argv=None) -> int:
    """CLI: conformance.py <run_dir> [--arm A] [--endpoint K] — replay the verdict on a recorded run."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Dev-conformance verdict for a recorded capsule-bench run.")
    ap.add_argument("run_dir")
    ap.add_argument("--arm", default=None, help="arm name (default: read run_manifest.yaml)")
    ap.add_argument("--endpoint", default=None, help="endpoint_kind (default: read run_manifest.yaml)")
    ap.add_argument("--round", type=int, default=None, help="round index (default: latest transcript)")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    arm, endpoint = a.arm, a.endpoint
    if arm is None or endpoint is None:
        try:
            import yaml
            man = yaml.safe_load((run_dir / "run_manifest.yaml").read_text())
            arm = arm or man.get("arm") or man.get("track") or ""
            endpoint = endpoint or man.get("endpoint_kind") or ""
        except Exception:  # noqa: BLE001
            arm = arm or ""
            endpoint = endpoint or ""
    rounds = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    if a.round is not None:
        tpath = run_dir / "rounds" / f"round_{a.round:02d}.transcript.jsonl"
    else:
        tpath = rounds[-1] if rounds else run_dir / "rounds" / "round_00.transcript.jsonl"
    verdict = compute(tpath, run_dir / "submission", arm, endpoint)
    print(json.dumps(verdict, indent=2))
    bad = failing_checks(verdict)
    if bad:
        print(f"\n[conformance] NOT CONFORMANT — failing: {', '.join(bad)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
