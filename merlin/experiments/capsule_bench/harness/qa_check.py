#!/usr/bin/env python3
"""Redacted QA gate for the capsule_bench_v0 raw_baseline pilot.

Grades a candidate ``submission/`` against the PUBLIC pilot capsules (A0/A2/A4/B0) through the
real ladder (L0 reference==simulate, spike, verilator + trace_check), then emits ONLY a redacted
verdict the agent is allowed to see:

    {all_pass, n_passed, n_capsules, integrity_status,
     per_capsule: [{capsule, status, numeric_status, mismatch_count, trace_status,
                    trace_violations:[class-name strings], tiers:{L*:status}, failure_plane,
                    failure_category, highest_tier}],
     first_failure_planes}

It DELIBERATELY omits every answer-bearing value: golden outputs, reference/oracle outputs,
numeric diffs (max_abs_diff / first_mismatch), command buffers, lowered MLIR. The full grading
work tree (which contains numeric_report.yaml etc.) is written under an OPERATOR-ONLY runs_root
that the agent never sees; only the scrubbed verdict crosses back.

The agent uses this as a pass/fail QA signal to iterate against — never as an answer key.

Usage:
  qa_check.py --submission <dir> --out <verdict.json> [--labels public,dev]
              [--runs-root <operator-only tmp>] [--no-oracle] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common.artifacts import cache_dir  # noqa: E402 — purgeable work trees
import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_grade as CG  # noqa: E402
from merlin.targetgen import capsule_runner as CR  # noqa: E402

PILOT_PUBLIC = C.EXP / "scripts" / "pilot_capsules"

# Allowed (answer-free) numeric fields. Everything else in the numeric block is dropped.
_SAFE_NUMERIC = {"status", "policy", "mismatch_count"}


# Process-exit metadata is NOT capsule data and must survive the numeric scrub. MEASURED (gemmini arm-4,
# 2026-08-19): 19 of nemotron's 20 capsules received a failure_detail of exactly 26 characters,
# "emit_command_buffer rc=#:", for six consecutive rounds. The underlying fact was ``rc=0`` — the agent's
# compiler was exiting CLEANLY and emitting nothing — and that single digit was the whole diagnostic. The
# blanket scrub turned an actionable message into one that merely looks like information. Same class of
# bug as the tier label, which already has a carve-out below; a return code cannot echo a golden value.
_RC_KEYS = ("rc", "returncode", "return_code", "exitcode", "exit_code", "status")


def _is_ascii_letter(ch: str) -> bool:
    """True for a-zA-Z only. ``str.isalpha`` would also accept digits' unicode cousins and letters from
    scripts that cannot appear in an MLIR token, which is a wider door than this needs."""
    return len(ch) == 1 and (("a" <= ch <= "z") or ("A" <= ch <= "Z"))


def _is_path_like(emitted: str) -> bool:
    """Does the token just before a ':' look like a file path? Trailing identifier-ish run containing a
    '.' or '/' -- i.e. ``input.interface.mlir`` or ``mlir_oot/gemmini_opt.py``, never a bare number."""
    j = len(emitted)
    while j and not emitted[j - 1].isspace() and emitted[j - 1] not in "\"'(),":
        j -= 1
    tok = emitted[j:]
    return ("." in tok or "/" in tok) and any(_is_ascii_letter(c) for c in tok)


def _scrub_numbers(text: str) -> str:
    """Collapse numeric VALUES to '#', while leaving numbers that carry STRUCTURE intact.

    A golden value is a bare number: ``expected 42``, ``cos 0.9997``, ``[1, 2, 3]``. A shape, a dtype, a
    capsule name and a tier label are not values — they are tokens that happen to contain digits, and the
    agent already holds every one of them (shapes and dtypes come from ``capsule.yaml``, which the bundle
    grants; the capsule name rides UNREDACTED in the sibling ``capsule`` field of the same record). So the
    two rules are:

      * a digit run touching an ASCII letter on either side is structure — keep it;
      * a digit run directly after an allowlisted ``<key>=`` is process metadata — keep it;
      * anything else is a candidate value — collapse it to '#'.

    MEASURED (gemmini arm-4, 2026-08-19): the blanket scrub handed Nemotron this parse error —

        %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<#x#xi#>

    The shape and the element type ARE the diagnostic for a compiler task, and both were destroyed to
    protect values that were never in the string. Under the rule above the same error now reads
    ``tensor<16x16xi8>`` while ``expected 42, actual 17`` still scrubs to ``expected #, actual #``.

    Structural, not pattern-matched (repo convention: a too-narrow regex silently drops valid input).
    Walks the string once, deciding each digit run from the characters immediately around it.
    """
    out: list[str] = []
    at_source_location = False       # the previous kept run was a `:line`, so `:col` may follow
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # A digit run starts here (possibly signed). Decide whether it is an exempt return code by
        # looking BACKWARDS at what was just emitted: an allowlisted key followed by '='.
        start = i
        if ch == "-" and i + 1 < n and text[i + 1].isdigit():
            i += 1
        if text[i].isdigit():
            while i < n and text[i].isdigit():
                i += 1
            if i < n and text[i] == "." and i + 1 < n and text[i + 1].isdigit():
                i += 1
                while i < n and text[i].isdigit():
                    i += 1
            emitted = "".join(out)
            before = text[start - 1] if start else ""
            after = text[i] if i < n else ""
            keep = False
            if emitted.endswith("="):
                key = emitted[:-1]
                # take the trailing identifier-ish run before '='
                j = len(key)
                while j and (key[j - 1].isalnum() or key[j - 1] == "_"):
                    j -= 1
                keep = key[j:].lower() in _RC_KEYS
            if not keep:
                # Structure, not a value: the run is part of a token that also contains letters --
                # ``i8``, ``bf16``, ``16x16``, ``A0_config_smoke``, ``L2``, ``vlen256``. A golden value
                # never touches a letter; it sits alone between separators.
                keep = _is_ascii_letter(before) or _is_ascii_letter(after)
            if not keep and before == ":":
                # A source location: ``input.interface.mlir:12:5``. The line and column of the agent's
                # OWN input cannot echo a golden value, and without them a parse error names a file but
                # not a place. The line is kept when the ':' follows a path-like token; the column is
                # kept because it follows a line we just kept.
                keep = at_source_location or _is_path_like(emitted[:-1])
            at_source_location = bool(keep and before == ":")
            out.append(text[start:i] if keep else "#")
            continue
        if ch != ":":
            at_source_location = False
        out.append(ch)
        i += 1
    return "".join(out)


def _redact_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    # Scrub concrete numbers (could echo expected/actual values) — the anti-answer-leak guard. The length
    # cap only bounds a runaway oracle dump; it must NOT chop the structured localization hint
    # (_encoding_divergence_hint), whose ACTIONABLE tail — "decode your OWN emitted artifact via the
    # disassembler / instruction_trace.json and check each op's operands" — runs past ~240 chars. Cutting it
    # left the agent told WHERE (the encoding) but not the self-inspection METHOD. 800 fits the full hint
    # while still bounding a pathological dump.
    return _scrub_numbers(detail)[:800]


def _per_capsule_from_results(runs_root: Path) -> dict[str, dict]:
    """Read each capsule_result.json from the operator-only work tree and redact it."""
    out: dict[str, dict] = {}
    # The runner writes under <target>-capsule-bench (target-derived), NOT the gemmini default baked into
    # CR.SUITE — keying off CR.SUITE dropped every per-capsule failure detail for any non-gemmini target,
    # so the agent saw plane counts but never the reason. Glob any suite subdir so the reason always
    # surfaces (target-general).
    rr = runs_root / "runs"
    if not rr.exists():
        return out
    for cr in sorted(rr.glob("*/*/capsule_result.json")):
        try:
            r = json.loads(cr.read_text())
        except Exception:
            continue
        num = r.get("numeric") or {}
        fail = r.get("failure") or {}
        tiers = r.get("tiers") or {}
        out[r.get("capsule", cr.parent.name)] = {
            "status": r.get("status"),
            "numeric_status": num.get("status"),
            "mismatch_count": num.get("mismatch_count"),
            "trace_status": (r.get("trace_check") or {}).get("status"),
            "trace_violations": list((r.get("trace_check") or {}).get("violations") or []),
            "tiers": {t: (tiers.get(t) or {}).get("status") for t in tiers},
            "tier_cycles": {t: (tiers.get(t) or {}).get("cycles") for t in tiers
                            if (tiers.get(t) or {}).get("cycles") is not None},
            "failure_plane": fail.get("plane"),
            "failure_category": fail.get("category"),
            # The tier LABEL survives redaction. It is a harness constant ("L2"), never capsule data, and
            # the blanket numeric scrub was rewriting it to "L#" -- so an agent could not even tell WHICH
            # tier blocked it. Passed through a strict shape check so only a tier label can ever ride here.
            "failure_tier": (lambda v: v if isinstance(v, str) and len(v) <= 4 and v[:1] == "L"
                             and v[1:].isdigit() else None)(fail.get("tier")),
            "failure_detail": _redact_detail(fail.get("detail")),
        }
    return out


def _loop_target_sim_via() -> tuple[str, str]:
    """Resolve (target, sim_via) for the loop-gate oracle from THIS experiment's descriptor (honors the
    MERLIN_TARGET_EXPERIMENT override baked into C.EXP). Falls back to C.TARGET + no bespoke sim if the
    descriptor is absent, so a descriptor-less invocation still grades on the RTL-derived (arc) tier."""
    desc = C.EXP / "target_experiment.yaml"
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(desc)
        return te.target, te.sim_via
    return C.TARGET, ""


def run(submission: str, capsules_root: str, runs_root: Path, labels: set[str],
        no_oracle: bool, timeout: int) -> dict:
    # Loop gate = L0+L1+trace + the target's FASTEST RTL oracle tier ONLY — for gemmini (sim_via=chipyard)
    # that is L2 (spike); the slower cycle-accurate tier (verilator L3) is the separate bounded checkpoint
    # (run_baseline_qa_loop). Per-round verilator on 20 capsules across 3 parallel arms is infeasible (CPU
    # storm). The adapters are resolved from the descriptor's target+sim_via via the shared factory, so a
    # non-chipyard target (arc/cyclotron) grades on its own RTL-derived tier with NO gemmini-specific path.
    _target, _sim_via = _loop_target_sim_via()
    # The loop tier is chosen from the tiers THESE capsules declare, so the per-round gate always rides a
    # tier the capsule asked for. Without this the loop picks the endpoint's fastest tier, which for an
    # endpoint that exposes an additive cheap tier below its declared gold tier means grading against a
    # tier the capsule never declared.
    from merlin.targetgen.contract.materialize import declared_oracle_tiers as _declared
    _decl = _declared(capsules_root)
    _loop_adapters = ({} if no_oracle
                      else CR.qa_loop_adapters(_target, _sim_via, declared_tiers=_decl))
    # Refuse ONLY when the endpoint exposes tiers but none of them is declared — substituting one there is
    # the defect. An endpoint that reaches nothing at all is an honestly ABSENT oracle: leave the adapter
    # set empty and let each capsule report its missing tier as unavailable, exactly as before.
    if not no_oracle and not _loop_adapters:
        _reach = sorted(CR.oracle_adapters(_target, _sim_via))
        if _reach:
            raise SystemExit(
                f"capsule corpus {capsules_root} declares required oracle tiers {sorted(_decl)} but "
                f"target {_target!r} reaches {_reach} — no declared tier is reachable, so this loop "
                f"cannot grade. Refusing to substitute a tier the capsules never declared.")
    score = CG.grade(submission, capsules_root=capsules_root, runs_root=str(runs_root),
                     labels=labels, contract=str(C.REPO / "merlin/contract"),
                     oracle_adapters=_loop_adapters, timeout=timeout, target=_target,
                     no_oracle=no_oracle)
    redacted = _per_capsule_from_results(runs_root)

    per_capsule = []
    for pc in score.get("per_capsule", []):
        name = pc["capsule"]
        rich = redacted.get(name, {})
        per_capsule.append({
            "capsule": name,
            "label": pc.get("label"),
            "status": pc.get("status"),
            "numeric_status": rich.get("numeric_status", pc.get("numeric")),
            "mismatch_count": rich.get("mismatch_count"),
            "trace_status": rich.get("trace_status", pc.get("trace")),
            "trace_violations": rich.get("trace_violations", []),
            "tiers": pc.get("tiers", {}),
            "tier_cycles": rich.get("tier_cycles", {}),
            "failure_plane": rich.get("failure_plane"),
            "failure_category": rich.get("failure_category"),
            "failure_detail": rich.get("failure_detail"),
        })

    n_caps = score.get("n_capsules", 0)
    n_pass = score.get("n_passed", 0)
    if no_oracle:
        # HONEST structure-only smoke: the numeric oracle was NOT run, so a capsule that clears the
        # structural tiers reads back as `not_gradeable_no_oracle`, never `pass` (no numeric pass is ever
        # claimed). The STOP signal is therefore "every capsule is structurally clean" (L0/L1/trace), NOT
        # the numeric `all_pass` — which can never be true here and would make the agent thrash to timeout
        # chasing an unreachable numeric pass (the observed 0/11 failure). `n_passed` reports the
        # structural-clean count so the console + round summary read coherently; the note makes the scope
        # explicit and the per-capsule numeric_status stays withheld.
        n_structural = score.get("n_structural_pass",
                                 sum(1 for pc in per_capsule
                                     if pc["status"] in ("pass", "not_gradeable_no_oracle")))
        all_pass = bool(n_caps > 0 and score.get("structural_pass", n_structural == n_caps))
        verdict = {
            "qa_gate": "capsule_bench_v0_pilot",
            "gradeable": False,
            "labels_graded": score.get("labels_graded"),
            "all_pass": all_pass,
            "stop_condition": "structural_tiers_pass",
            "n_passed": n_structural,
            "n_structural_pass": n_structural,
            "n_capsules": n_caps,
            "integrity_status": score.get("integrity_status"),
            "highest_tier": score.get("highest_tier"),
            "first_failure_planes": score.get("first_failure_planes", {}),
            "per_capsule": per_capsule,
            "note": ("NOT GRADEABLE this run: the numeric/trace oracle is unavailable, so ONLY the L0/L1 "
                     "structural tiers are graded (a structure-only smoke). Do NOT chase a numeric pass — "
                     "capsules that clear the structural tiers show status `not_gradeable_no_oracle`, "
                     "which is the target here, NOT a fixable failure. Fix only real structural failures "
                     "(schema/language/L0/L1/trace planes); never hardcode outputs."),
        }
    else:
        verdict = {
            "qa_gate": "capsule_bench_v0_pilot",
            "gradeable": True,
            "labels_graded": score.get("labels_graded"),
            "all_pass": bool(n_caps > 0 and n_pass == n_caps),
            "n_passed": n_pass,
            "n_capsules": n_caps,
            "integrity_status": score.get("integrity_status"),
            "highest_tier": score.get("highest_tier"),
            "first_failure_planes": score.get("first_failure_planes", {}),
            "per_capsule": per_capsule,
            "note": ("This is a QA pass/fail signal only. It contains NO reference output values — there is no answer key. "
                     "Fix failures by capsule + failure_plane + trace_violations; never hardcode outputs."),
        }
    # top-level integrity failure (K0/K1 fail-closed)
    if "failure" in score:
        verdict["package_failure"] = {"plane": score["failure"]["plane"],
                                      "category": score["failure"]["category"],
                                      "detail": _redact_detail(score["failure"]["detail"])}
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True, help="path to write the redacted verdict JSON")
    ap.add_argument("--capsules-root", default=str(PILOT_PUBLIC))
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--runs-root", default=None,
                    help="OPERATOR-ONLY grading work tree (must NOT be inside the agent workspace)")
    ap.add_argument("--no-oracle", action="store_true", help="L0 + trace only (skip spike/verilator)")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)

    runs_root = Path(a.runs_root) if a.runs_root else (cache_dir("capsule_bench_qa") / "scratch")
    runs_root.mkdir(parents=True, exist_ok=True)
    labels = set(a.labels.split(","))
    verdict = run(a.submission, a.capsules_root, runs_root, labels, a.no_oracle, a.timeout)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    print(f"[qa_check] all_pass={verdict['all_pass']} "
          f"{verdict['n_passed']}/{verdict['n_capsules']} integrity={verdict['integrity_status']}")
    for pc in verdict["per_capsule"]:
        extra = "" if pc["status"] == "pass" else f"  <- plane={pc['failure_plane']} viol={pc['trace_violations']}"
        print(f"    [{pc['status']:10s}] {pc['capsule']}{extra}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
