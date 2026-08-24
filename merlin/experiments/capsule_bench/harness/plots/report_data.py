#!/usr/bin/env python3
"""Extract every number the multi-model comparison report cites, from the run tree, into one JSON.

The report and its figures read THIS file and nothing else, so a number can never be hand-typed into a
figure and drift from the run that produced it. Everything here is derived from persisted artifacts:
``run_manifest.yaml``, ``environment.yaml``, ``qa_loop_state.yaml``, ``qa_history/verdict_round_*.json``,
``grading_{public,hidden}/**/capsule_result.json``, ``cost_time_toolcalls.yaml``, the round transcripts
and the frozen submission tree.

Three measurement traps are handled here rather than at the figure, because every consumer needs them:

* ``first_failure_planes`` DISCARDS ``failure.category``, so a ``spike`` count silently merges
  FUNCTIONAL_MISMATCH (the model's compiler was wrong) with TOOL_CRASH (the simulator broke). We
  re-derive every plane count from the per-capsule ``capsule_result.json`` and keep the category.
* ``highest_tier`` is a suite-MINIMUM (the highest tier EVERY capsule passed), not a pass bar — one
  failing capsule pins it for the whole run. We report ``tier_reached`` counts instead.
* The pass bar is whatever the run's MATERIALIZED corpus declared, not what the committed capsule.yaml
  says. Under bwrap the materializer caps ``required_oracle_tiers`` at L2, so a failing L3 is recorded
  and then ignored by the pass computation. We read the as-graded manifest and report both.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _common as C  # noqa: E402  (the SELECTED target's descriptor; defaults per the harness)
from merlin.common.paths import repo_root  # noqa: E402  (re-exported for callers of this module)

# Both ride the descriptor-selected target, never a literal: this module is shared by every target's
# report, so a baked name here silently renders one target's runs under another target's figures.
RUNS = C.RUNS / "merlin_assisted"
SUITE = f"{C.TARGET}-capsule-bench"

# planes where the SUBMISSION's CLI / manifest / packaging is wrong (it never reached the compiler's math)
CONTRACT_PLANES = {"build", "parse", "interface_to_target", "target_to_command_buffer",
                   "command_buffer_schema", "emit_target_artifact", "command_buffer", "contract",
                   "integrity", "schema", "trace_check"}
# planes that are the HARNESS failing, never the submission
INFRA_PLANES = {"runner_internal", "oracle_unavailable", "declared_tier_unreachable",
                "not_gradeable_no_oracle"}


def _yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _blame(detail: str, run: str) -> str:
    """Whose code is in this traceback? A ``tool_crash`` is NOT automatically the harness's fault — the
    commonest case by far is the submission's own tool crashing, or a harness reader choking on MLIR the
    submission emitted. Only a traceback that never touches the submission's artifacts is ours.

    The cross-target marker is the one unambiguous harness fault here: a gemmini submission asked to
    compile an ``atlas`` / ``mx_gemmini`` / ``saturn_opu`` capsule was graded against a contaminated
    capsule set, which is a grading-directory defect and says nothing about the model."""
    d = detail or ""
    if "backend target must be" in d and "got '" in d:
        return "harness"                       # cross-target contamination of the grading dir
    if "SimulationError" in d or "RUNNER_CRASH" in d:
        return "harness"                       # the harness simulator lacks an opcode it was handed
    model_marks = (f"/{run}/submission/", "./submission/", "/_qa_ws/", "/generated/run_lowering.py",
                   "available dialects", "does not have a custom format",
                   # the MLIR "available dialects" list, which arrives mid-string after a tool prefix
                   "dlti, emitc, func", "llvm.inline_asm", "MLIRError", "Unable to parse module")
    if any(m in d for m in model_marks):
        return "model"
    return "unclear"


def _capsule_rows(base: Path, run: str) -> list[dict]:
    rows = []
    for cf in sorted(base.glob(f"*/capsule_result.json")):
        try:
            r = json.loads(cf.read_text())
        except Exception:  # noqa: BLE001
            continue
        f = r.get("failure") or {}
        tiers = {k: (v or {}).get("status") for k, v in (r.get("tiers") or {}).items()}
        mand = {k: bool((v or {}).get("mandatory")) for k, v in (r.get("tiers") or {}).items()}
        cat = (f.get("category") or "").lower()
        plane = f.get("plane")
        if r.get("status") == "pass":
            kind = "pass"
        elif cat in ("tool_crash", "runner_crash"):
            kind = f"crash:{_blame(f.get('detail'), run)}"
        elif plane in INFRA_PLANES:
            kind = "infra"
        elif plane in CONTRACT_PLANES or cat == "protocol_violation":
            kind = "contract"
        elif cat == "functional_mismatch":
            kind = "compute"
        else:
            kind = f"other:{plane}/{cat}"
        rows.append({"capsule": r.get("capsule"), "status": r.get("status"), "kind": kind,
                     "plane": plane, "category": cat, "tiers": tiers, "mandatory": mand,
                     "trace_violations": (r.get("trace_check") or {}).get("violations") or [],
                     "detail": (f.get("detail") or "")[:600]})
    return rows


def _first_fail_tier(tiers: dict) -> str:
    """The first ladder rung this capsule failed. '-' when no tier ran at all (a contract-plane death)."""
    for t in ("L0", "L1", "L2", "L3", "L4"):
        if tiers.get(t) == "fail":
            return t
    return "-" if not tiers else "pass"


def _diagnostics(d: Path) -> dict:
    """How much of what the AGENT was shown survived redaction. The digit scrubber rewrote every numeral,
    so `tensor<16x16xi8>` reached the model as `tensor<#x#xi#>` and `rc=0` as `rc=#`. This only bites a
    model that FAILS and must iterate, which is why it has to be measured per cell rather than assumed."""
    out = {"rounds": 0, "verdicts": 0, "redacted": 0, "shape_lost": 0, "identity_lost": 0,
           "rc_lost": 0, "near_empty": 0, "localized": 0, "unusable": 0}
    vs = sorted((d / "qa_history").glob("verdict_round_*.json"))
    out["rounds"] = len(vs)
    for v in vs:
        try:
            y = json.loads(v.read_text())
        except Exception:  # noqa: BLE001
            continue
        for c in (y.get("per_capsule") or []):
            # Only a FAILING capsule has a diagnostic to damage. A passing capsule carries no detail,
            # and counting its empty string as "no content" would score a converged run as the worst
            # informed of all — exactly backwards.
            if c.get("status") == "pass":
                continue
            det = c.get("failure_detail") or ""
            out["verdicts"] += 1
            if "#" in det or len(det.strip()) < 45:
                out["unusable"] += 1        # numerals destroyed, or no content at all
            if "#" in det:
                out["redacted"] += 1
            if "<#" in det or "x#" in det or "i#" in det:
                out["shape_lost"] += 1
            if any(k in det for k in ("_hxm#", "A#_", "B#_", "C#_", "runs_#", "cand_#")):
                out["identity_lost"] += 1
            if "rc=#" in det:
                out["rc_lost"] += 1
            if "Localized" in det:
                out["localized"] += 1
            if len(det.strip()) < 45:
                out["near_empty"] += 1
    return out


def _transcript_stats(d: Path) -> dict:
    """Tool mix, thrash, and message length across every round — the behavioural axis. Tool NAMES differ by
    driver (`Bash` vs `bash`), so they are lowercased into a driver-neutral bucket before counting."""
    files = sorted((d / "rounds").glob("round_*.transcript.jsonl")) or [d / "transcript.jsonl"]
    tools: dict[str, int] = {}
    cmds: list[str] = []
    texts: list[int] = []
    for f in files:
        if not f.exists():
            continue
        for ln in f.read_text(errors="replace").splitlines():
            try:
                e = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if e.get("type") != "assistant":
                continue
            for b in (e.get("message") or {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    n = (b.get("name") or "")[:40].lower()
                    tools[n] = tools.get(n, 0) + 1
                    inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                    c = inp.get("command") or inp.get("cmd") or ""
                    if isinstance(c, list):
                        c = " ".join(map(str, c))
                    if c:
                        cmds.append(c)
                elif b.get("type") == "text":
                    texts.append(len(b.get("text") or ""))
    uniq = len(set(cmds))
    return {"tools": tools, "n_shell": len(cmds), "unique_shell": uniq,
            "repeat_frac": round(1 - uniq / len(cmds), 4) if cmds else None,
            "n_text": len(texts),
            "median_text_chars": sorted(texts)[len(texts) // 2] if texts else 0}


@lru_cache(maxsize=1)
def _allowed_tool_linesets() -> tuple[tuple[str, frozenset[str]], ...]:
    """Line sets of the granted merlin files, for detecting a submission that vendored one and then
    edited it. GLM-5 x Claude Code, for example, copied ``interface_emit.py`` verbatim and appended a
    ten-line MOVEMENT handler: not authored structure, and not a pure copy either."""
    out = []
    roots = ["targetgen/contract", "targetgen/oot_starterkit", "targetgen/synthesize",
             "targetgen/generate", "targetgen/rtl", "xdsl_dialects", "kernels"]
    for r in roots:
        base = repo_root() / "merlin" / "python" / "merlin" / r
        if not base.exists():
            continue
        for f in base.rglob("*.py"):
            try:
                lines = {ln.strip() for ln in f.read_text(errors="replace").splitlines() if ln.strip()}
            except Exception:  # noqa: BLE001
                continue
            if len(lines) >= 25:
                out.append((f.name, frozenset(lines)))
    return tuple(out)


def _vendored_from(body: str) -> str | None:
    """The granted merlin file this submission module was copied from, if >=90% of that file's lines
    survive in it verbatim. Returns None for genuinely authored code."""
    mine = {ln.strip() for ln in body.splitlines() if ln.strip()}
    if len(mine) < 25:
        return None
    for name, lines in _allowed_tool_linesets():
        if len(lines & mine) / len(lines) >= 0.90:
            return name
    return None


@lru_cache(maxsize=1)
def _allowed_tool_hashes() -> frozenset[str]:
    """Content hashes of every merlin file the assisted bundle grants read access to, so a submission
    that vendors one verbatim can be told apart from one that wrote it."""
    roots = ["targetgen/contract", "targetgen/oot_starterkit", "targetgen/synthesize",
             "targetgen/generate", "targetgen/rtl", "xdsl_dialects", "kernels"]
    out = set()
    for r in roots:
        base = repo_root() / "merlin" / "python" / "merlin" / r
        if not base.exists():
            continue
        for f in base.rglob("*.py"):
            try:
                out.add(hashlib.sha256(f.read_bytes()).hexdigest())
                out.add(hashlib.sha256(f.read_text(errors="replace").encode("utf-8", "replace")).hexdigest())
            except Exception:  # noqa: BLE001
                continue
    return frozenset(out)


def _submission(d: Path) -> dict:
    """Compiler ARCHITECTURE of what was actually shipped. The strongest correlate of success in this
    corpus is not how much code a model wrote but whether it separated the ISA-encoding layer out."""
    sub = d / "submission"
    pys = [p for p in sub.rglob("*.py") if "__pycache__" not in p.parts] if sub.exists() else []
    lines = 0
    generated = 0
    for p in pys:
        try:
            t = p.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        lines += t.count("\n") + 1
        if "GENERATED from RTL facts" in t:
            generated += 1
    txt = "\n".join(p.read_text(errors="replace") for p in pys) if pys else ""

    # WHAT is in the box matters more than how many files are in it. A raw module count scores a model
    # that left `parse_fixed`, `parse_fixed2` and `emit_improved` behind as better-architected than one
    # that shipped a clean dialect + lowering split, which is backwards. Classify instead:
    #   generated  — emitted by the RTL-facts generators the arm grants (correct encoding by construction)
    #   structure  — the dialect / lowering / codegen split this arm exists to ask for
    #   monolith   — one catch-all file doing parse, lower, emit and encode together
    #   debris     — abandoned iterations and ad-hoc test scripts left in the frozen submission
    DEBRIS = ("_fixed", "_improved", "_old", "_new", "_v2", "_backup", "test_", "debug_", "scratch")
    MONO = (f"{C.TARGET}_opt", f"{C.TARGET}_backend", "backend", "main")
    kinds = {"generated": [], "structure": [], "monolith": [], "debris": [], "vendored": []}
    vendored_hashes = _allowed_tool_hashes()
    for f in pys:
        stem = f.stem
        try:
            body = f.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        # A byte-identical copy of a merlin file the bundle explicitly ALLOWS is not the model's design
        # work and must not be credited as one — the same false positive that makes the conformance
        # checker report `no_regex_ok: False` for a model that merely vendored merlin's own regex-using
        # interface_emit.py.
        if (hashlib.sha256(body.encode("utf-8", "replace")).hexdigest() in vendored_hashes
                or _vendored_from(body)):
            kinds["vendored"].append(stem)
        elif "GENERATED from RTL facts" in body:
            kinds["generated"].append(stem)
        elif any(k in stem for k in DEBRIS):
            kinds["debris"].append(stem)
        elif stem in ("__init__",):
            continue
        elif any(stem == m or stem.startswith(m) for m in MONO):
            kinds["monolith"].append(stem)
        else:
            kinds["structure"].append(stem)
    return {"n_modules": len(pys), "n_lines": lines, "rtl_generated_modules": generated,
            "uses_starterkit": "oot_starterkit" in txt, "uses_xdsl": ("xdsl" in txt or "irdl" in txt),
            "kinds": {k: sorted(v) for k, v in kinds.items()},
            "modules": sorted(str(p.relative_to(sub)) for p in pys)}


def _harness_health(d: Path, driver: str | None) -> dict:
    """Did this run's harness actually deliver what it was configured to deliver?

    Two defects silently degraded whole cells and neither shows up in any score:

    * **opencode** writes its agent config to a tempfile under ``$TMPDIR``, which bwrap tmpfs-hides, so
      inside the sandbox the file does not exist and opencode falls back to its DEFAULT agent. A cell
      that ran ``agent=build`` never received our system prompt, our per-model ``limit.output``, our
      compaction settings, or our ``task`` denial — and ``task`` is the tool that wedges a round.
      The session log is the witness, so this is checkable after the fact on every run.
    * **the LiteLLM bridge** dropped ``toolConfig`` on continuation requests carrying tool blocks, which
      terminates a codex round with rc=1 and NO usage record. Those rounds did real work and are
      recorded as though the model did nothing.
    """
    home = repo_root() / "out" / "artifacts" / "cache"
    agents: dict[str, int] = {}
    bridge_400 = 0
    for sub, pat in (("opencode_home", "opencode/log/*.log"), ("codex_home", "*.log")):
        for rd in sorted((home / sub).glob(f"{d.name}_r*")):
            for lf in rd.glob(pat):
                try:
                    txt = lf.read_text(errors="replace")
                except Exception:  # noqa: BLE001
                    continue
                for tok in txt.split("agent=")[1:]:
                    name = tok.split()[0] if tok.split() else ""
                    if name:
                        agents[name] = agents.get(name, 0) + 1
    for rf in sorted((d / "rounds").glob("*.codex_events.raw.jsonl")):
        try:
            bridge_400 += rf.read_text(errors="replace").count("toolConfig field must be defined")
        except Exception:  # noqa: BLE001
            pass
    unpriced = 0
    for sf in sorted((d / "rounds").glob("*.codex_summary.json")):
        try:
            if json.loads(sf.read_text()).get("usage_complete") is False:
                unpriced += 1
        except Exception:  # noqa: BLE001
            pass
    config_ok = None
    if driver == "opencode" and agents:
        config_ok = agents.get("merlinbench", 0) > agents.get("build", 0)
    return {"agents_seen": agents, "config_delivered": config_ok,
            "bridge_toolconfig_400s": bridge_400, "rounds_with_no_usage_record": unpriced}


def _stop_reason(d: Path) -> dict:
    """WHY this run stopped, and after how many rounds.

    Round budgets are not equal across the corpus and a 0/20 is not comparable between a cell that
    exhausted twelve attempts and one that was cut off after one. Three distinct endings appear, and
    only one of them is the harness's doing:

      converged        — it passed and the loop ended (the intended ending)
      self_declared    — the agent dropped READY_FOR_BARRIER while still scoring zero. A MODEL
                         behaviour, not a budget limit: it decided it was finished and it was not.
      cost_cap         — an external ceiling stopped it mid-campaign. The only ending that truncates
                         a cell for reasons unrelated to the model, and therefore the only one that
                         confounds a comparison of attempts.
      budget_exhausted — it used every round it was given.
    """
    log = d.parent / f"{d.name}.launch.log"
    txt = log.read_text(errors="replace") if log.exists() else ""
    rounds = None
    for line in txt.splitlines():
        if "run complete:" in line and "rounds=" in line:
            try:
                rounds = int(line.split("rounds=")[1].split()[0])
            except Exception:  # noqa: BLE001
                pass
    if "converged=True" in txt:
        why = "converged"
    elif "COST CAP" in txt:
        why = "cost_cap"
    elif "dropped READY_FOR_BARRIER" in txt:
        why = "self_declared"
    elif rounds is not None and rounds >= 12:
        why = "budget_exhausted"
    else:
        why = "unknown"
    return {"rounds_run": rounds, "why": why,
            "externally_truncated": why == "cost_cap"}


def _conformance(d: Path) -> dict:
    y = _yaml(d / "qa_loop_state.yaml")
    rounds = [r for r in (y.get("rounds") or []) if isinstance(r, dict)]
    confs = [r.get("conformance") for r in rounds if r.get("conformance")]
    ever: dict = {}
    for c in confs:
        for k, v in (c.get("checks") or {}).items():
            if v:
                ever[k] = True
            else:
                ever.setdefault(k, v)
    hits = []
    for c in confs:
        hits = c.get("regex_hits") or hits
    # The no-regex check scans every .py in the submission, including a file the model VENDORED from
    # merlin's own granted tooling -- and merlin's interface_emit.py is itself regex-based. Flagging a
    # model for merlin's regex is a false positive, so record whether every hit lands in a vendored file
    # and let the report show that as a third state rather than as a violation.
    hit_files = sorted({h.get("file") for h in hits if h.get("file")})
    vend = {m.split("/")[-1] for m in (_submission(d).get("kinds", {}).get("vendored") or [])}
    all_vendored = bool(hit_files) and all(
        pathlib.PurePath(f).stem in vend for f in hit_files)
    return {"rounds_scored": len(confs),
            "rounds_conformant": sum(1 for c in confs if c.get("conformant")),
            "ever": ever, "regex_hit_files": hit_files,
            "regex_only_in_vendored": all_vendored,
            "progression": [{"round": r.get("round"), "n_passed": r.get("n_passed"),
                             "n_capsules": r.get("n_capsules")} for r in rounds]}


def _canonical_public_set() -> set[str]:
    """The gemmini public capsule set, taken from the target's own contract rather than from whatever a
    grading directory happens to contain. Several grading dirs were contaminated by a discovery race that
    staged OTHER targets' corpora (radiance ``R*``/``RF*``, MX ``MF*``) beside gemmini's; those capsules
    ask for ops gemmini has no ISA for (``rmsnorm``, ``rope``, flash attention, bf16) and would otherwise
    inflate a denominator with work the submission was never asked to do."""
    names: set[str] = set()
    for kind in ("isa", "layers", "model_slices"):
        root = repo_root() / "merlin" / "contract" / "capsules" / kind
        if not root.exists():
            continue
        for c in sorted(root.iterdir()):
            y = _yaml(c / "capsule.yaml")
            if not y:
                continue
            tgts = y.get("targets") or y.get("target")
            tgts = [tgts] if isinstance(tgts, str) else (tgts or [])
            if not tgts or C.TARGET in tgts:
                names.add(c.name)
    return names


def collect() -> dict:
    canonical = _canonical_public_set()
    cells = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        env = _yaml(d / "environment.yaml")
        man = _yaml(d / "run_manifest.yaml")
        cost = _yaml(d / "cost_time_toolcalls.yaml")
        pub_base = d / "grading_public" / "runs" / SUITE
        if not pub_base.exists():
            continue
        pub = _capsule_rows(pub_base, d.name)
        # A grading dir can be contaminated with another target's capsules (a discovery race staged
        # atlas / mx_gemmini / saturn_opu capsules beside gemmini's). Those are not this submission's
        # work and must not dilute its denominator; they are reported separately.
        foreign = [r for r in pub
                   if "backend target must be" in (r.get("detail") or "")
                   or (canonical and r["capsule"] not in canonical)]
        pub = [r for r in pub if r not in foreign]
        hid_base = d / "grading_hidden" / "runs" / SUITE
        hid = _capsule_rows(hid_base, d.name) if hid_base.exists() else []
        hid = [r for r in hid if "backend target must be" not in (r.get("detail") or "")]
        try:
            score = json.loads((d / "grading_public" / "score_capsule.json").read_text())
        except Exception:  # noqa: BLE001
            score = {}
        # the pass bar this run was ACTUALLY graded against, from the materialized corpus
        bar = []
        for rm in sorted(pub_base.glob("*/run_manifest.yaml"))[:1]:
            bar = (_yaml(rm).get("metadata") or _yaml(rm)).get("required_oracle_tiers") or []
        cells.append({
            "run": d.name, "model": env.get("model"), "driver": env.get("driver"),
            "bundle": env.get("bundle_id"), "sandbox": env.get("sandbox"),
            "n_public": len(pub), "n_passed": sum(1 for r in pub if r["status"] == "pass"),
            "foreign_capsules_in_grading_dir": len(foreign),
            "n_hidden": len(hid), "hidden_passed": sum(1 for r in hid if r["status"] == "pass"),
            "pass_bar_tiers": bar,
            "tier_reached": score.get("tier_reached") or {},
            # What the passes REST ON, carried beside the counts rather than left to be reconstructed
            # from tier_reached by whoever reads this. `headline` is the quotable form built once in
            # capsule_grade._headline; `pass_evidence` is the rtl_backed/cheap_tier_only split behind it.
            "headline": score.get("headline"),
            "pass_evidence": score.get("pass_evidence") or {},
            "public": pub, "hidden": hid,
            "first_fail_tier": {r["capsule"]: _first_fail_tier(r["tiers"]) for r in pub},
            "kinds": {k: sum(1 for r in pub if r["kind"] == k) for k in {r["kind"] for r in pub}},
            "diagnostics": _diagnostics(d),
            "behaviour": _transcript_stats(d),
            "submission": _submission(d),
            "conformance": _conformance(d),
            "harness_health": _harness_health(d, env.get("driver")),
            "stop": _stop_reason(d),
            "tokens": {k: cost.get(k) for k in ("tokens_input", "tokens_cached", "tokens_output",
                                                "tool_calls", "usage_source", "billing_mode",
                                                "estimated_cost_usd", "subscription_notional_usd")},
            "wall_minutes": round((man.get("process") or {}).get("wall_time_seconds") or 0, 1) / 60,
            "integrity": man.get("integrity_status"),
        })
    return {"cells": cells, "runs_root": str(RUNS)}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect()
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"wrote {out}  ({len(data['cells'])} graded cells)")


if __name__ == "__main__":
    main()
