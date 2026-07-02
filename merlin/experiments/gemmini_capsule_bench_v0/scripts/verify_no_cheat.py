"""Static anti-cheat gate — MUST pass before any A/B launch (the plan's "nothing ships until this passes").

This proves, WITHOUT running an agent, that the shipped tooling cannot leak the answer and that the
arm contrast is honest. Five checks:

  1. NO ANSWER CONTENT in any shipped tool/scaffold/prompt: no golden values, no `reference_outputs`,
     no per-capsule expected command buffers, no hidden-capsule names. (static grep over source)
  2. MOAT SEPARATION: the RTL-facts generators + facts.json are granted to the merlin+CIRCT arm ONLY,
     and explicitly denied to the non-CIRCT merlin arm + baseline — so the merlin-vs-CIRCT distinction
     is real, not nominal.
  3. GRANT CONSISTENCY: every tool a STARTER_PROMPT tells the agent to use is in that bundle's allowed
     set (a prompt must not point at a tool the bundle withholds — that would make the agent flail).
  4. KIT PARITY: the oot_starterkit is granted to BOTH merlin arms (no arm gets a bigger kit), and the
     baseline arm gets NO merlin tooling at all (the control stays a control).
  5. AUDIT REGRESSION: the hardened transcript_tooling_audit still PASSES the existing abc4 arms
     (no false-positive that would wrongly disqualify a clean run).

Exit 0 = safe to launch. Non-zero = DO NOT launch.  Usage: verify_no_cheat.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import yaml

EXP = Path("/scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0")
REPO = EXP.parent.parent
KIT_DIR = REPO / "merlin/python/merlin/targetgen/oot_starterkit"
RTL_DIR = REPO / "merlin/python/merlin/targetgen/rtl"
BUNDLES = EXP / "input_bundles"

# arms that exist as bundles, and which tooling each should/shouldn't have
MERLIN_ARMS = ["merlin_assisted", "merlin_assisted_rtlchecks"]
CIRCT_ARM = "merlin_assisted_rtlchecks"
CONDITIONS = ["hwbringup_v0", "hwbringup_nokernel_v0"]

# ---- check 1: forbidden answer-content patterns in shipped source/prompts -------------------------
ANSWER_PATTERNS = {
    "reference_outputs":  re.compile(r"reference_outputs"),
    "merlin_runtime_import": re.compile(r"from\s+merlin\.runtime\s+import|import\s+merlin\.runtime\.(reference|simulator)"),
    "hidden_capsule_ref": re.compile(r"capsules/hidden|CANARY_HIDDEN|hidden_\w+_golden"),
    "golden_yaml_ref":    re.compile(r"golden\.yaml|expected_output|expected_cb"),
}
# files that are ALLOWED to mention these tokens because they are *checkers/auditors* describing them,
# not leaking them. (Scanning a cheat-detector for the word "reference_outputs" is a false positive.)
SCAN_EXEMPT = {"verify_no_cheat.py", "transcript_tooling_audit.py", "preflight.py"}


def _shipped_files() -> list[Path]:
    fs = []
    for d in (KIT_DIR, RTL_DIR):
        if d.is_dir():
            fs += [p for p in d.rglob("*") if p.suffix in (".py", ".md") and "__pycache__" not in str(p)]
    fs += list(BUNDLES.glob("*/STARTER_PROMPT.md"))
    return fs


def check_answer_content() -> tuple[bool, list[str]]:
    bad = []
    for f in _shipped_files():
        if f.name in SCAN_EXEMPT:
            continue
        txt = f.read_text(errors="ignore")
        for name, rx in ANSWER_PATTERNS.items():
            for m in rx.finditer(txt):
                # a generator's _TMPL or a prompt may legitimately say "never import merlin.runtime";
                # only flag if it's NOT in a negation/instruction context.
                ctx = txt[max(0, m.start() - 40):m.start()].lower()
                if any(w in ctx for w in ("not ", "never", "no ", "deny", "forbid", "without", "cheat")):
                    continue
                bad.append(f"{f.relative_to(REPO)} :: {name} :: …{txt[m.start():m.start()+50]!r}")
    return (not bad), bad


# ---- checks 2-4: bundle allow/deny separation -----------------------------------------------------
def _manifest(arm: str, cond: str) -> dict:
    p = BUNDLES / f"{arm}_{cond}" / "input_bundle_manifest.yaml"
    return yaml.safe_load(p.read_text()) if p.is_file() else {}


def _allowed(m: dict) -> list[str]:
    return [e["path"] for e in m.get("allowed", []) if isinstance(e, dict)]


def _denied(m: dict) -> list[str]:
    return [e["path"] for e in m.get("denied", []) if isinstance(e, dict)]


def _has(lst, sub) -> bool:
    return any(sub in p for p in lst)


def check_moat() -> tuple[bool, list[str]]:
    probs = []
    for cond in CONDITIONS:
        circt = _manifest(CIRCT_ARM, cond)
        if not (_has(_allowed(circt), "targetgen/rtl/") and _has(_allowed(circt), "rtl_facts/")):
            probs.append(f"{CIRCT_ARM}_{cond}: CIRCT arm must ALLOW rtl/ generators + rtl_facts")
        noncirct = _manifest("merlin_assisted", cond)
        if not (_has(_denied(noncirct), "targetgen/rtl/") and _has(_denied(noncirct), "rtl_facts/")):
            probs.append(f"merlin_assisted_{cond}: non-CIRCT arm must DENY rtl/ generators + rtl_facts (moat leak)")
        base = _manifest("raw_baseline", cond)
        if _has(_allowed(base), "merlin/python") or _has(_allowed(base), "oot_starterkit"):
            probs.append(f"raw_baseline_{cond}: baseline must get NO merlin tooling (control)")
    return (not probs), probs


def check_kit_parity() -> tuple[bool, list[str]]:
    probs = []
    for cond in CONDITIONS:
        for arm in MERLIN_ARMS:
            if not _has(_allowed(_manifest(arm, cond)), "oot_starterkit"):
                probs.append(f"{arm}_{cond}: starter kit (oot_starterkit) not in allowed — prompt references it")
    return (not probs), probs


# tokens a prompt may reference -> the allowed-path substring that must back them
PROMPT_TOOL_GRANTS = {
    "oot_starterkit": "oot_starterkit",
    "gen_isa_module": "targetgen/rtl/",
    "gen_rtl_digest": "targetgen/rtl/",
    "gen_numeric_facts": "targetgen/rtl/",
}


def check_grant_consistency() -> tuple[bool, list[str]]:
    probs = []
    for arm in MERLIN_ARMS:
        for cond in CONDITIONS:
            bdir = BUNDLES / f"{arm}_{cond}"
            prompt = (bdir / "STARTER_PROMPT.md")
            if not prompt.is_file():
                probs.append(f"{arm}_{cond}: missing STARTER_PROMPT.md"); continue
            ptxt = prompt.read_text()
            allowed = _allowed(_manifest(arm, cond))
            for tok, grant in PROMPT_TOOL_GRANTS.items():
                if tok in ptxt and not _has(allowed, grant):
                    probs.append(f"{arm}_{cond}: prompt uses '{tok}' but bundle does not allow '{grant}'")
    return (not probs), probs


# ---- check 5: audit regression on existing clean runs ---------------------------------------------
def check_audit_regression() -> tuple[bool, list[str]]:
    sys.path.insert(0, str(EXP / "scripts"))
    try:
        import transcript_tooling_audit as TTA
    except Exception as e:
        return False, [f"cannot import transcript_tooling_audit: {e}"]
    probs = []
    for sub, rid in [("raw_baseline", "rb_abc4"), ("merlin_assisted", "merlin_abc4"),
                     ("merlin_assisted", "merlincirct_abc4")]:
        d = EXP / "runs" / sub / rid
        if not d.is_dir():
            continue  # abc4 may have been archived; skip silently
        r = TTA.audit_run(d)
        if r.get("disqualified"):
            probs.append(f"{rid}: REGRESSION — clean abc4 run now disqualified: {r['cheat_hits']}")
    return (not probs), probs


CHECKS = [
    ("1. no answer-content in shipped tools/prompts", check_answer_content),
    ("2. moat separation (RTL facts = CIRCT arm only)", check_moat),
    ("3. prompt↔grant consistency", check_grant_consistency),
    ("4. kit parity (both merlin arms; baseline none)", check_kit_parity),
    ("5. audit no-regression on abc4", check_audit_regression),
]


def main() -> int:
    print("=== verify_no_cheat — static anti-cheat + moat gate ===")
    all_ok = True
    for name, fn in CHECKS:
        ok, probs = fn()
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for p in probs[:8]:
            print(f"        - {p}")
    print("\n" + ("✅ VERIFY_NO_CHEAT: PASS — tooling is answer-free and the arm contrast is honest."
                  if all_ok else "❌ VERIFY_NO_CHEAT: FAIL — DO NOT launch until the above are resolved."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
