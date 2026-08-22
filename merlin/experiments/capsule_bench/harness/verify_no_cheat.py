"""Static anti-cheat gate — MUST pass before any A/B launch (the plan's "nothing ships until this passes").

This proves, WITHOUT running an agent, that the shipped tooling cannot leak the answer and that the
arm contrast is honest. The checks:

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

  7. HOLDOUT NOT SPECIFIED: no file in the granted `merlin/contract/` tree names a held-out capsule
     (derived from the holdout store on disk, so a new target is covered the day it appears).

Exit 0 = safe to launch. Non-zero = DO NOT launch.  Usage: verify_no_cheat.py
"""
from __future__ import annotations
import ast
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python

EXP = C.EXP                    # the active target's experiment dir (descriptor-driven, no target literal)
REPO = C.REPO
KIT_DIR = REPO / "merlin/python/merlin/targetgen/oot_starterkit"
RTL_DIR = REPO / "merlin/python/merlin/targetgen/rtl"
BUNDLES = EXP / "input_bundles"

# arms that exist as bundles, and which tooling each should/shouldn't have
MERLIN_ARMS = ["merlin_assisted", "merlin_assisted_rtlchecks", "merlin_assisted_eqsat"]
CIRCT_ARM = "merlin_assisted_rtlchecks"
# arm5's one declared difference from the xDSL arm: the equivalence seam. Named here so the moat check
# below asserts the grant in BOTH directions — present for arm5, absent everywhere else. A new arm ships
# with zero fairness checking unless its differentiator is written down somewhere a gate reads.
EQSAT_ARM = "merlin_assisted_eqsat"
EQSAT_PATHS = ("contraction_egraph", "persistent_equivalence")
# hw-bringup conditions DERIVED from the target's materialized bundles (gemmini ships kernel+nokernel;
# atlas ships kernel-only) — not a hardcoded gemmini set, so the moat/parity/grant checks run over
# exactly the conditions this target actually launches.
CONDITIONS = C.experiment_conditions()

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


def _materialized(arm: str, cond: str) -> bool:
    """True iff this arm ships a bundle for the active target+condition. An arm named in ``MERLIN_ARMS``
    but not materialized for THIS target (e.g. a newer treatment not yet built for every target) is not
    launchable here, so the fairness checks skip it rather than fail — mirroring how ``CONDITIONS`` is
    itself derived from the target's materialized bundles. This never weakens the gate: an arm with no
    bundle cannot be run, and any arm that IS launched has a manifest and is still fully checked."""
    return (BUNDLES / f"{arm}_{cond}" / "input_bundle_manifest.yaml").is_file()


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
        # The eqsat moat, checked in both directions: arm5 must HAVE the seam (else its treatment is
        # absent and a null result means nothing), and no other arm may reach it (else the comparison
        # is against an arm that also had it).
        if _materialized(EQSAT_ARM, cond):
            eqsat = _manifest(EQSAT_ARM, cond)
            for tok in EQSAT_PATHS:
                if not _has(_allowed(eqsat), tok):
                    probs.append(f"{EQSAT_ARM}_{cond}: eqsat arm must ALLOW {tok} — it is the treatment")
        for other in ("raw_baseline", "merlin_assisted", CIRCT_ARM, "cpp_merlininfra"):
            man = _manifest(other, cond)
            for tok in EQSAT_PATHS:
                if _has(_allowed(man), tok):
                    probs.append(f"{other}_{cond}: {tok} is granted outside the eqsat arm (moat leak)")
    return (not probs), probs


def check_kit_parity() -> tuple[bool, list[str]]:
    probs = []
    for cond in CONDITIONS:
        for arm in MERLIN_ARMS:
            if not _materialized(arm, cond):
                continue
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
            if not _materialized(arm, cond):
                continue
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the harness dir (sibling module)
    try:
        import transcript_tooling_audit as TTA
    except Exception as e:
        return False, [f"cannot import transcript_tooling_audit: {e}"]
    probs = []
    for sub, rid in [("raw_baseline", "rb_abc4"), ("merlin_assisted", "merlin_abc4"),
                     ("merlin_assisted", "merlincirct_abc4")]:
        d = C.RUNS / sub / rid
        if not d.is_dir():
            continue  # abc4 may have been archived; skip silently
        r = TTA.audit_run(d)
        if r.get("disqualified"):
            probs.append(f"{rid}: REGRESSION — clean abc4 run now disqualified: {r['cheat_hits']}")
    return (not probs), probs


# ---- check 6: no workload-specific constants baked into a frozen submission ----------------------
# A generated compiler must GENERALIZE — it may carry legal target-derived thresholds (`if K >= tile*4`,
# arithmetic on the mesh dim), but NOT a dispatch keyed to a specific workload's shape or name
# (`if M == 512`, `if model_name == "llama"`, `M in (512, 4096)`). We detect the latter structurally
# (AST), never the former: only EQUALITY / membership of a shape identifier against a real-dimension
# literal, or a name identifier against a string literal, is a hit. `>=`/`<`/`*` and small structural
# guards (== 0/1/2/...) are legal and pass.
_SHAPE_IDENTS = {"m", "n", "k", "rows", "cols", "seq", "seqlen", "seq_len", "batch", "bsz",
                 "dim", "hidden", "hidden_size", "heads", "n_heads", "d_model", "shape"}
_NAME_IDENTS = {"model_name", "model", "workload", "capsule", "capsule_name", "arch", "net", "task"}
_LEGAL_SMALL_INTS = {0, 1, 2, 3, 4, 8}   # structural guards, not a hardcoded tensor dimension


def _ident_of(node: ast.AST) -> str | None:
    """Lowercased identifier for a Name or a trailing attribute (`cfg.model_name` -> 'model_name')."""
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    return None


def scan_workload_constants(root: Path) -> list[str]:
    """Structurally flag workload-specific dispatch in the .py under ``root`` (a frozen submission).

    Returns a list of ``file:line — reason`` findings; empty == clean. Uses the AST so a comment or a
    string that merely mentions a shape does not trip it, and legal target-derived thresholds pass.
    """
    hits: list[str] = []
    root = Path(root)
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        rel = p.relative_to(root) if root in p.parents else p.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            op = node.ops[0]
            operands = [node.left, node.comparators[0]]
            idents = {_ident_of(o) for o in operands}
            # shape identifier == / != a real-dimension integer literal → hardcoded shape dispatch
            if isinstance(op, (ast.Eq, ast.NotEq)) and idents & _SHAPE_IDENTS:
                for o in operands:
                    if isinstance(o, ast.Constant) and isinstance(o.value, int) \
                            and o.value not in _LEGAL_SMALL_INTS:
                        hits.append(f"{rel}:{node.lineno} — shape identifier compared to literal "
                                    f"{o.value} (hardcoded-shape dispatch)")
            # shape identifier IN a tuple/list of dimension literals → same, spelled as membership
            if isinstance(op, ast.In) and _ident_of(node.left) in _SHAPE_IDENTS \
                    and isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set)):
                lits = [e.value for e in node.comparators[0].elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                if any(v not in _LEGAL_SMALL_INTS for v in lits):
                    hits.append(f"{rel}:{node.lineno} — shape identifier tested against a set of "
                                f"dimension literals {lits} (hardcoded-shape dispatch)")
            # name/workload identifier == a string literal → model/workload branch
            if isinstance(op, (ast.Eq, ast.NotEq)) and idents & _NAME_IDENTS:
                for o in operands:
                    if isinstance(o, ast.Constant) and isinstance(o.value, str):
                        hits.append(f"{rel}:{node.lineno} — workload/model name compared to "
                                    f"{o.value!r} (workload-specific branch)")
    return hits


def check_workload_constants() -> tuple[bool, list[str]]:
    """Scan any materialized submission trees for workload-specific constants (empty == nothing to
    scan yet, which passes — the substantive scan runs post-freeze in grade_agent_run)."""
    probs: list[str] = []
    runs = C.RUNS          # out/runs/<target>/capsule-bench — the retired <experiment>/runs never
    if runs.is_dir():      # existed here, so this scan silently covered nothing and always "passed"
        for sub in sorted(runs.glob("*/*/submission")):
            for h in scan_workload_constants(sub):
                probs.append(f"{sub.parent.name}: {h}")
    return (not probs), probs



# ---- check 7: the held-out set is not SPECIFIED anywhere the agent can read ----------------------
# Derived on BOTH sides: read the holdout NAMES off disk (the grader can see them; the agent
# cannot), and read the SCAN SCOPE off the bundle manifests, so the check covers exactly what
# the sandbox actually binds read-only rather than one hardcoded tree. This is the gate
# that was missing when the holdout roster lived in the tracked `capsules/MANIFEST.yaml` and the full
# holdout spec -- op, dtype, exact shape -- lived in the tracked `capsules/profiles/<target>.yaml`,
# both inside the broad `merlin/contract/` grant every arm receives. A name match is not the only way
# to leak a holdout, but it is the way we actually leaked one, and it costs nothing to keep closed.
# The holdout stores themselves: they are the source of the names, and they are masked in the sandbox.
_HOLDOUT_STORE_PARTS = ("hidden",)


def _holdout_names() -> set[str]:
    """Every held-out capsule name on disk, across ALL targets -- the grader can see them, the agent
    cannot. Derived by walking, so a new target's holdouts are covered the day they appear."""
    caps = REPO / "merlin/contract/capsules"
    return {c.parent.name for c in caps.rglob("hidden/*/capsule.yaml")} if caps.is_dir() else set()


def _granted_roots() -> list[Path]:
    """The union of every path any bundle grants read-only -- DERIVED from the bundle manifests, not a
    hardcoded tree. The leak this check exists for lived under ``merlin/contract/``, but the merlin arms
    are additionally granted parts of ``merlin/python/`` and the target's own contract dir, and a scan
    scoped to one hardcoded root would be blind to a holdout name landing in any of them. Bundle paths
    are repo-relative, except that the harness spells the experiment dir without the ``merlin/`` prefix."""
    roots: set[Path] = set()
    for man in sorted(BUNDLES.glob("*/input_bundle_manifest.yaml")):
        try:
            m = yaml.safe_load(man.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        for a in m.get("allowed") or []:
            rel = str(a.get("path") or "").strip("/")
            if not rel:
                continue
            for cand in (REPO / rel, REPO / "merlin" / rel):
                if cand.exists():
                    roots.add(cand)
                    break
    # never let one granted root subsume the scan of another; dedupe nested roots for speed only
    return sorted(roots)


def _granted_readable_files() -> list[Path]:
    """Files under any granted root that are NOT themselves masked answer surfaces."""
    out, seen = [], set()
    for root in _granted_roots():
        out.extend(_scan_root(root, seen))
    return out


def _scan_root(root: Path, seen: set) -> list[Path]:
    out = []
    for f in ([root] if root.is_file() else root.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or f in seen:
            continue
        try:
            rel = f.relative_to(REPO)
        except ValueError:
            continue
        if any(part in _HOLDOUT_STORE_PARTS for part in rel.parts):
            continue                       # the holdout dir itself — masked
        if f.name.endswith(".hidden.yaml"):
            continue                       # the holdout spec sidecar — masked, and untracked
        if f.suffix in (".yaml", ".yml", ".json", ".md", ".py", ".mlir", ".txt", ".h", ".c", ".cpp"):
            seen.add(f)
            out.append(f)
    return out


def _holdout_dirs_present() -> list[Path]:
    """Every ``hidden/`` directory that EXISTS, whether or not its contents can be listed.

    A preflighted run answer-locks the holdout store (chmod 000), which is correct -- but it also makes
    the store unreadable to this check, and an unreadable directory yields no names. Existence survives
    the lock even when listing does not, so the two cases can be told apart.
    """
    caps = REPO / "merlin/contract/capsules"
    return [d for d in caps.glob("*/hidden") if d.is_dir()] if caps.is_dir() else []


def check_holdout_not_specified() -> tuple[bool, list[str]]:
    names = _holdout_names()
    if not names:
        present = _holdout_dirs_present()
        if present:
            # FAIL CLOSED. The store exists and could not be read -- almost certainly because the
            # preflight answer-locked it, which is exactly when this check runs. Passing here would make
            # the check's success indistinguishable from its blindness, and it is the check that exists to
            # stop a held-out spec reaching the granted tree. Run it before the lock, or from a grader
            # context that can read the store.
            return False, [f"{len(present)} holdout store(s) exist but could not be read "
                           f"({', '.join(str(x.relative_to(REPO)) for x in present)}) — refusing to "
                           f"report a pass this check cannot substantiate; run before the answer-lock"]
        return True, ["(no holdout capsules on disk — nothing to check)"]
    bad = []
    scanned = 0
    for f in _granted_readable_files():
        scanned += 1
        txt = f.read_text(errors="ignore")
        for n in sorted(names):
            if n in txt:
                bad.append(f"{f.relative_to(REPO)} names holdout capsule {n!r}")
    if bad:
        return False, bad
    # SAY WHAT WAS EXAMINED. A bare (True, []) is indistinguishable from the vacuous pass this check was
    # just fixed to stop reporting -- the reader cannot tell a real scan from one that found nothing to
    # scan. State the size of both sides, so the pass carries its own evidence.
    return True, [f"{len(names)} holdout capsule name(s) checked against {scanned} readable granted "
                  f"file(s) — none present"]


CHECKS = [
    ("1. no answer-content in shipped tools/prompts", check_answer_content),
    ("2. moat separation (RTL facts = CIRCT arm only)", check_moat),
    ("3. prompt↔grant consistency", check_grant_consistency),
    ("4. kit parity (both merlin arms; baseline none)", check_kit_parity),
    ("5. audit no-regression on abc4", check_audit_regression),
    ("6. no workload-specific constants in submissions", check_workload_constants),
    ("7. held-out set not specified in the granted tree", check_holdout_not_specified),
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
