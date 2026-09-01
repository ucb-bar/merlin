"""Per-round DEV-CONFORMANCE verdict — did the agent actually DEVELOP the way its arm mandates?

This is a first-class FLAG, never a grade gate: a non-conformant round is loudly surfaced and excluded
from a valid arm-N demonstration, but its capsule ORACLE grade still reports unchanged (the user-chosen
middle strength — see the enforced-workflow prompt block). It catches the case where a run scores 0 by
NOT using the mandated tooling (an invented encoding, a regex parser, no lever enumeration), which the
numeric grade alone cannot distinguish from a genuine capability wall.

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


def _tool_calls(tpath: Path) -> list[tuple[str, dict]]:
    """Every assistant tool_use in the transcript as (tool_name, input_dict), across all drivers."""
    out: list[tuple[str, dict]] = []
    if not tpath.exists():
        return out
    for ln in tpath.read_text(errors="ignore").splitlines():
        try:
            e = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if e.get("type") != "assistant":
            continue
        for b in e.get("message", {}).get("content", []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                inp = b.get("input")
                out.append((b.get("name") or "", inp if isinstance(inp, dict) else {"_raw": inp}))
    return out


def _exec_str(name: str, inp: dict) -> str:
    """The COMMAND actually executed by this tool call — the bash command, not a written file's content.
    (Matching the whole serialized input would count a tool named only in prose inside a write_file, which
    is exactly the false positive we must avoid: an agent that WRITES 'run isa_tools.py asm' in its plan but
    never runs it is not conformant.)"""
    cmd = inp.get("command") or inp.get("cmd") or inp.get("_raw") or ""
    return f"{name} {cmd}"


def _any(calls: list[tuple[str, dict]], *needles: str) -> bool:
    return any(any(n in _exec_str(name, inp) for n in needles) for name, inp in calls)


def _full_selfcheck(calls: list[tuple[str, dict]]) -> bool:
    for name, inp in calls:
        ex = _exec_str(name, inp)
        # a CLI self-check over the whole set (claude / opencode run it as bash)
        if "agent_selfcheck" in ex and ("--capsules all" in ex or '--capsules "all"' in ex):
            return True
        # the bedrock self_check tool: default (no capsules key) or an explicit "all"
        if name == "self_check" and inp.get("capsules") in (None, "", "all"):
            return True
    return False


def emits_word_stream(endpoint_kind: str, fourth_output_name: str = "") -> bool:
    """Does this target's agent hand-author an INSTRUCTION WORD STREAM the ISA dev tools operate on?

    Derived exactly as :func:`merlin.targetgen.generate_prompt._is_simt_mlir` derives its inverse, from the
    same fact — the 4th artifact's filename — so the mandate and the check on that mandate cannot drift:

      * ``external_backend`` + a ``.mlir`` 4th artifact  -> the agent emits an LLVM-dialect COMPILER
        LOWERING that the runner compiles fork-free. There is no word stream. The prompt tells it in so
        many words: "no prints, no ``.insn``/``.word``".
      * ``external_backend`` + anything else             -> a self-hosted kernel of literal words, which
        `isa_tools asm/lint/disasm` exist to encode and check.

    Why this is not cosmetic: the ISA-tool checks used to key on ``external_backend`` ALONE, which is true
    of both shapes. So a target on the MLIR path was required to have run `isa_tools asm` — a tool its own
    prompt forbids it to use — and every round of it was recorded NOT CONFORMANT for correctly following
    its instructions. The agent was doing the right thing and the checker said otherwise.
    """
    return endpoint_kind == "external_backend" and not str(fourth_output_name or "").endswith(".mlir")


def compute(tpath: Path, sub_dir: Path, arm: str, endpoint_kind: str,
            fourth_output_name: str = "") -> dict:
    """Compute the conformance verdict for one round. Returns a dict with per-check booleans (or None when
    the check does not apply to this arm), the regex-hit detail, and the overall ``conformant`` flag."""
    assisted = arm in _ASSISTED
    # The ISA dev tools are mandated only where the agent authors an instruction word stream. On the
    # MLIR/fork-free path the prompt mandates the LOWERING instead and forbids `.insn`/`.word`, so both
    # ISA-tool checks are N/A there — requiring them contradicted the instructions the agent was given.
    words = emits_word_stream(endpoint_kind, fourth_output_name)
    calls = _tool_calls(tpath)
    regex_hits = _submission_regex(sub_dir) if assisted else []   # no-regex is an ASSISTED-arm mandate (xDSL authoring)
    has_py = bool(list(sub_dir.rglob("*.py"))) if sub_dir.exists() else False

    checks: dict = {}
    # no-regex applies only where the arm is mandated to author in xDSL AND actually wrote python
    checks["no_regex_ok"] = (not regex_hits) if (assisted and has_py) else None
    checks["isa_tools_used"] = _any(calls, "isa_tools") if (assisted and words) else None
    checks["asm_used"] = _any(calls, "isa_tools.py asm", "isa_tools asm") if (assisted and words) else None
    checks["cca_used"] = _any(calls, "cca_contract", "action_catalog",
                              "check_bijection", "escalation_ladder") if assisted else None
    checks["full_selfcheck"] = _full_selfcheck(calls)

    applicable = [v for v in checks.values() if v is not None]
    conformant = all(applicable) if applicable else True
    return {"conformant": conformant, "checks": checks, "regex_hits": regex_hits,
            "arm": arm, "endpoint_kind": endpoint_kind,
            # Recorded so a reader can tell an N/A that is CORRECT (this target authors no word stream)
            # from an N/A that means the fact was never threaded in — the two used to look identical.
            "fourth_output_name": fourth_output_name or None, "emits_word_stream": words}


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
