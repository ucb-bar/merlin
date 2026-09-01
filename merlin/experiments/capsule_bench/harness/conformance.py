"""Per-round DEV-CONFORMANCE verdict — did the agent actually DEVELOP the way its arm mandates?

This is a first-class completion gate while leaving the capsule ORACLE grade unchanged: numeric evidence
still reports when a round is non-conformant, but that round cannot be frozen as a valid arm-N result.
It catches the case where a run scores 0 by NOT using the mandated tooling (an invented encoding, a regex
parser, no lever enumeration), which the numeric grade alone cannot distinguish from a genuine capability
wall.

Every signal derives from PERSISTED artifacts — the round transcript + the frozen submission tree — so
it is computable post-hoc on any recorded run and needs no live sandbox. Fully target-agnostic (no
accelerator name, no golden); the only "regex" here is the repo's own check_no_regex AST scan, reused.

Checks (each TOOL-SET-APPLICABLE — skipped where the resolved bundle does not grant the tool, so it is
never a false fail):
  * no_regex_ok    — the submission's own .py author no regex (the repo cardinal rule)
  * isa_tools_used — an assisted arm exercised the derived ISA dev tools (lint/disasm/asm/debug) at all
  * asm_used       — an external_backend arm assembled words via `isa_tools asm` (external_backend only)
  * cca_used       — the CCA lever-set enumeration ran (check_bijection / escalation_ladder, CLI or API)
  * full_selfcheck — a self-check covering ALL capsules ran (not just one capsule)
  * rtl_derived_levers_used — an RTL-tools bundle actually derived its backend levers
  * rtl_facts_used — an RTL-tools bundle loaded its target's frozen RTL facts
  * scaffold_generators_used — an RTL-tools bundle ran a real scaffold generator (not merely read it)
  * rtl_checks_read — an RTL-tools bundle successfully read the harness's ``rtl_checks`` feedback
  * arm4_discovery_before_submission_mutation — CCA + RTL discovery + scaffold preceded the first edit
``conformant`` is the AND of the applicable checks.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import shlex
from collections.abc import Iterable
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
    result_text: str


def _content_text(content) -> str:
    """Flatten the persisted tool-result body without interpreting it.

    Claude-family transcripts use either a string or a list of typed content blocks.  Keeping the
    returned text lets checks which are specifically about *reading feedback* prove that the named block
    was actually returned, rather than crediting a command that merely mentioned its filename.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or block.get("content") or "")
            if isinstance(block, dict) else str(block)
            for block in content
        )
    if content is None:
        return ""
    return str(content)


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
                    result_text=_content_text(result.get("content")) if result is not None else "",
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


def _shell_tokens(command: str) -> list[str]:
    """Tokenize one simple shell command, stripping unquoted comments.

    Evidence commands deliberately reject shell composition.  That makes ``true # derived_levers(...)``
    and ``echo 'load_facts(...)'`` data, not executable API evidence, while quoted Python ``-c`` source
    remains one token.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return []


_SHELL_CONTROLS = {";", "&&", "||", "|", "&", ">", ">>", "<", "<<"}


def _executable(tokens: list[str]) -> tuple[str, list[str]] | None:
    """Return (basename, argv) for an uncomposed command, allowing env assignments and timeout."""
    if not tokens or any(t in _SHELL_CONTROLS for t in tokens):
        return None
    i = 0
    if tokens and tokens[0] == "env":
        i += 1
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith(("/", "./")):
        i += 1
    if i < len(tokens) and tokens[i] == "timeout":
        i += 2  # timeout + duration
    if i >= len(tokens):
        return None
    return Path(tokens[i]).name, tokens[i + 1:]


def _python_source(call: ToolCall) -> str | None:
    """Extract executable Python from a canonical ``python -c`` or single heredoc invocation."""
    command = str(call.input.get("command") or call.input.get("cmd") or "")
    lines = command.splitlines()
    if not lines:
        return None
    tokens = _shell_tokens(lines[0] if len(lines) > 1 else command)
    # Simple python -c.  Reject trailing shell argv: the required snippets take no arguments.
    resolved = _executable(tokens)
    if resolved is not None:
        exe, argv = resolved
        if exe in {"python", "python3"} and "-c" in argv:
            ci = argv.index("-c")
            if ci + 2 == len(argv):
                return argv[ci + 1]
    # Single Python heredoc.  The first line may contain <<, but no other composition; the delimiter must
    # close the command and there may be no trailing shell command after it.
    if len(lines) > 2 and "<<" in tokens:
        controls = [t for t in tokens if t in _SHELL_CONTROLS]
        if controls != ["<<"]:
            return None
        hi = tokens.index("<<")
        if hi + 1 >= len(tokens):
            return None
        delimiter = tokens[hi + 1]
        prefix = tokens[:hi]
        resolved = _executable(prefix)
        if resolved is None or resolved[0] not in {"python", "python3"}:
            return None
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == delimiter)
        except StopIteration:
            return None
        if any(line.strip() for line in lines[end + 1:]):
            return None
        return "\n".join(lines[1:end])
    return None


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualname(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _python_call_names(call: ToolCall) -> tuple[str, ...]:
    source = _python_source(call)
    if source is None:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    return tuple(_qualname(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call))


def _call_index(calls: list[ToolCall], predicate) -> int | None:
    return next((i for i, call in enumerate(calls) if call.succeeded and predicate(call)), None)


def _literal_string_list(text: str) -> list[str] | None:
    """Find a printed non-empty Python list/tuple/set of strings in a tool result."""
    candidates = [text.strip(), *(line.strip() for line in reversed(text.splitlines()))]
    # Tool wrappers may prefix metadata around stdout; the bracketed payload is still deterministic.
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, (list, tuple, set)) and value and all(
                isinstance(item, str) and item.strip() for item in value):
            return list(value)
    return None


def _derived_levers_evidence(call: ToolCall) -> bool:
    names = _python_call_names(call)
    invoked = any(name.endswith(".derived_levers") or name == "derived_levers" for name in names)
    return invoked and _literal_string_list(call.result_text) is not None


def _rtl_facts_evidence(call: ToolCall) -> bool:
    names = _python_call_names(call)
    invoked = any(name.endswith(".load_facts") or name == "load_facts" for name in names)
    body = call.result_text.lower()
    fact_families = sum(f'"{key}"' in body for key in ("arrays", "memories", "datapaths", "interfaces"))
    return invoked and '"target"' in body and fact_families >= 2


def _scaffold_generator_evidence(call: ToolCall) -> bool:
    names = _python_call_names(call)
    generators = (
        "mlir_scaffold.generate", "xdsl.generate", "llvm_plan.generate",
        "target_repo.generate_skeleton", "target_repo.emit_contracts", "generate_skeleton",
    )
    invoked = any(any(name.endswith(suffix) for suffix in generators) for name in names)
    result = call.result_text
    # The required command prints the returned Artifact records or their relpaths.  A generic "ok" is not
    # a generation witness; at least one concrete generated-file shape must be present.
    artifact_shape = ("Artifact(" in result and "relpath=" in result) or any(
        suffix in result for suffix in (".py", ".td", ".cpp", ".h", "CMakeLists.txt"))
    return invoked and artifact_shape


def _cca_evidence(call: ToolCall, *, script: str, subcommand: str, api: str) -> bool:
    names = _python_call_names(call)
    if any(name.endswith(f".{api}") or name == api for name in names):
        return True
    command = str(call.input.get("command") or call.input.get("cmd") or "")
    resolved = _executable(_shell_tokens(command))
    if resolved is None or resolved[0] not in {"python", "python3"}:
        return False
    argv = resolved[1]
    return len(argv) >= 2 and Path(argv[0]).name == script and argv[1] == subcommand


def _submission_mutation(call: ToolCall) -> bool:
    """Conservative first-authoring boundary used by Arm4's discovery-before-edit requirement."""
    name = call.name.lower().split(".")[-1]
    target_text = " ".join(str(call.input.get(k) or "") for k in (
        "file_path", "path", "patch", "content", "new_string", "_raw"))
    targets_submission = "submission/" in target_text or "/submission/" in target_text
    if targets_submission and name in {"write", "edit", "multiedit", "write_file", "apply_patch"}:
        return True

    command = str(call.input.get("command") or call.input.get("cmd") or "")
    tokens = _shell_tokens(command)
    mentions_submission = (
        "submission/" in command or "/submission/" in command
        or any(token.rstrip("/") == "submission" for token in tokens)
    )
    if not mentions_submission:
        return False
    if any(t in {">", ">>"} for t in tokens):
        return True
    resolved = _executable(tokens)
    if resolved is None:
        # A composed command touching submission cannot be proved read-only, so it is the boundary.
        return True
    exe, _ = resolved
    if exe in {"cp", "mv", "install", "mkdir", "touch", "rm", "tee", "truncate", "patch"}:
        return True
    if exe == "sed" and "-i" in tokens:
        return True
    names = _python_call_names(call)
    if any(name.endswith((".write_text", ".write_bytes", ".mkdir", ".unlink", ".rename"))
           or name == "open" for name in names):
        return True
    # Unknown commands touching the answer tree fail closed.  These are the simple readers/checkers the
    # workflow legitimately runs before authoring; everything else establishes the edit boundary.
    return exe not in {
        "cat", "jq", "grep", "rg", "head", "tail", "sed", "awk", "ls", "find", "stat", "file",
        "wc", "diff", "python", "python3", "isa_tools.py", "agent_selfcheck.py",
    }


def _rtl_checks_read(calls: list[ToolCall]) -> bool:
    """Prove that the final authoring round read the RTL feedback produced by the harness.

    ``qa/verdict.json`` is the only allowed arm-4 readback surface.  A successful ``jq``/Python command
    that explicitly selects ``rtl_checks`` is sufficient.  A whole-file read (``cat``/Read) is credited
    only when its persisted result contains the block, so reading a missing/stale verdict cannot pass.
    """
    for call in calls:
        if not call.succeeded:
            continue
        name = call.name.lower().split(".")[-1]
        command = str(call.input.get("command") or call.input.get("cmd") or "")
        ex = command.lower()
        file_path = str(call.input.get("file_path") or call.input.get("path") or "").lower()
        addresses_verdict = "qa/verdict.json" in ex or "qa/verdict.json" in file_path
        if not addresses_verdict:
            continue
        # Direct file-read tools are evidence.  Shell evidence must be one simple read operation; echo,
        # comments and composed commands cannot turn a fabricated string into readback evidence.
        direct_read = name in {"read", "read_file", "open_file"}
        resolved = _executable(_shell_tokens(command)) if command else None
        shell_read = resolved is not None and resolved[0] in {
            "cat", "jq", "grep", "rg", "head", "tail", "sed", "awk", "python", "python3",
        }
        if not direct_read and not shell_read:
            continue
        if shell_read and resolved[0] in {"python", "python3"}:
            calls_in_source = _python_call_names(call)
            if not any(name == "open" or name.endswith((".read_text", ".read_bytes"))
                       for name in calls_in_source):
                continue
        result = call.result_text.strip()
        # jq/awk may print just the selected list (without its key); require a non-empty structured
        # readback.  Whole-file reads must visibly contain the exact key, not ``rtl_checks_error``.
        if resolved is not None and resolved[0] in {"jq", "awk", "python", "python3"} and "rtl_checks" in ex \
                and result.lower() not in {"", "null", "none", "[]", "{}"}:
            return True
        result_lower = result.lower()
        if '"rtl_checks"' in result_lower or "'rtl_checks'" in result_lower:
            return True
    return False


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


def compute(tpath: Path, sub_dir: Path, arm: str, endpoint_kind: str,
            *, resolved_tools: Iterable[str] | None = None) -> dict:
    """Compute the conformance verdict for one round. Returns a dict with per-check booleans (or None when
    the check does not apply to this tool set), the regex-hit detail, and the overall ``conformant`` flag.

    ``resolved_tools`` is authoritative when supplied by a real run.  This is deliberately not inferred
    from ``arm``: the Arm4 launcher historically used the Arm3 spelling ``merlin_assisted`` and the old
    bundle manifest repeated that stale label.  The bundle's resolved tools are the experimental
    treatment.  ``None`` retains replay compatibility for old runs that predate the persisted tool set.
    """
    tools = None if resolved_tools is None else frozenset(str(t) for t in resolved_tools)
    assisted = arm in _ASSISTED if tools is None else bool(
        tools & {"merlin_infra", "xdsl_kit", "cca_spine", "isa_tools", "cca_tools"})
    xdsl = assisted if tools is None else "xdsl_kit" in tools
    isa = assisted if tools is None else "isa_tools" in tools
    cca = assisted if tools is None else bool(tools & {"cca_spine", "cca_tools"})
    rtl_surface = frozenset({"rtl_generators", "rtl_facts"})
    arm4 = False if tools is None else bool(tools & rtl_surface)
    external = endpoint_kind == "external_backend"
    calls = _tool_calls(tpath)
    regex_hits = _submission_regex(sub_dir) if xdsl else []   # no-regex is an xDSL-authoring mandate
    has_py = bool(list(sub_dir.rglob("*.py"))) if sub_dir.exists() else False

    checks: dict = {}
    # no-regex applies only where the arm is mandated to author in xDSL AND actually wrote python
    checks["no_regex_ok"] = (not regex_hits) if (xdsl and has_py) else None
    checks["isa_tools_used"] = _any(
        calls,
        "isa_tools.py asm", "isa_tools.py disasm", "isa_tools.py lint", "isa_tools.py debug",
        "isa_tools asm", "isa_tools disasm", "isa_tools lint", "isa_tools debug",
    ) if isa else None
    checks["asm_used"] = _any(calls, "isa_tools.py asm", "isa_tools asm") if (isa and external) else None
    checks["cca_used"] = (
        _any(calls, "cca_contract.py check-bijection", "check_bijection")
        and _any(calls, "action_catalog.py escalation-ladder", "escalation_ladder")
    ) if cca else None
    checks["full_selfcheck"] = _full_selfcheck(calls)
    # A partial RTL surface is a malformed treatment, not Arm3.  Fail closed so removing one of the two
    # Arm4 grants cannot silently downgrade the conformance contract.
    checks["arm4_toolset_complete"] = (rtl_surface <= tools) if arm4 else None
    derived_idx = _call_index(calls, _derived_levers_evidence) if arm4 else None
    facts_idx = _call_index(calls, _rtl_facts_evidence) if arm4 else None
    scaffold_idx = _call_index(calls, _scaffold_generator_evidence) if arm4 else None
    checks["rtl_derived_levers_used"] = derived_idx is not None if arm4 else None
    checks["rtl_facts_used"] = facts_idx is not None if arm4 else None
    checks["scaffold_generators_used"] = scaffold_idx is not None if arm4 else None
    checks["rtl_checks_read"] = _rtl_checks_read(calls) if arm4 else None
    if arm4:
        bijection_idx = _call_index(calls, lambda call: _cca_evidence(
            call, script="cca_contract.py", subcommand="check-bijection", api="check_bijection"))
        ladder_idx = _call_index(calls, lambda call: _cca_evidence(
            call, script="action_catalog.py", subcommand="escalation-ladder", api="escalation_ladder"))
        mutation_idx = next((i for i, call in enumerate(calls) if _submission_mutation(call)), None)
        discovery = (bijection_idx, ladder_idx, derived_idx, facts_idx, scaffold_idx)
        checks["arm4_discovery_before_submission_mutation"] = (
            all(i is not None for i in discovery)
            and (mutation_idx is None or max(i for i in discovery if i is not None) < mutation_idx)
        )
    else:
        checks["arm4_discovery_before_submission_mutation"] = None

    applicable = [v for v in checks.values() if v is not None]
    conformant = all(applicable) if applicable else True
    tool_evidence = {
        "n_calls": len(calls),
        "n_successful": sum(call.succeeded for call in calls),
        "n_failed": sum(call.result_present and not call.succeeded for call in calls),
        "n_missing_results": sum(not call.result_present for call in calls),
    }
    return {"conformant": conformant, "checks": checks, "regex_hits": regex_hits,
            "tool_evidence": tool_evidence, "arm": arm, "endpoint_kind": endpoint_kind,
            "resolved_tools": sorted(tools) if tools is not None else None,
            "arm4_tooling_required": arm4}


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
    ap.add_argument("--tool", action="append", default=None,
                    help="resolved bundle tool name (repeatable; default: read environment.yaml)")
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
    resolved_tools = a.tool
    if resolved_tools is None:
        try:
            import yaml
            environment = yaml.safe_load((run_dir / "environment.yaml").read_text()) or {}
            saved = environment.get("resolved_tools")
            resolved_tools = list(saved) if isinstance(saved, list) else None
        except Exception:  # noqa: BLE001 — old runs have no persisted resolved tool set
            resolved_tools = None
    verdict = compute(tpath, run_dir / "submission", arm, endpoint,
                      resolved_tools=resolved_tools)
    print(json.dumps(verdict, indent=2))
    bad = failing_checks(verdict)
    if bad:
        print(f"\n[conformance] NOT CONFORMANT — failing: {', '.join(bad)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
