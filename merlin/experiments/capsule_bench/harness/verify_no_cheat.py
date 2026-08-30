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

#: Patterns meaningful over the GRANTED tree as well as the shipped kit -- naming the holdout STORE is
#: an answer surface no matter which file does it and no matter what the sandbox masks, because the
#: name itself tells the agent a holdout set exists and what it is called.
#:
#: The other three are deliberately NOT here, and the reason is the distinction this check turns on.
#: ``reference_outputs`` / ``golden.yaml`` are the benchmark's own PUBLIC SPECIFICATION: the contract
#: documents and the task statement have to name the grading relation and the capsule format or the
#: task is unstatable. ``merlin_runtime_import`` fires on granted compiler code that imports the
#: oracle -- but ``merlin/python/merlin/runtime/{reference,simulator}.py`` are themselves masked
#: answer surfaces, so inside the box that import is a dead pointer, not a route. Applying those
#: three to the granted tree produced 26 findings, every one of them spec or a dead pointer; a gate
#: that cries wolf 26 times is a gate somebody turns off.
_GRANTED_SCAN_PATTERNS = ("hidden_capsule_ref",)


def _shipped_files() -> list[Path]:
    """The tooling SHIPPED INTO a bundle: the starter kit, the RTL generators, each arm's prompt.

    This is the population where every answer pattern is a leak, because these files are copied into
    the agent's workspace and run there. It is NOT the population the agent can READ -- that is the
    much larger granted tree, which :func:`_granted_readable_files` derives from the bundles' own
    ``allowed`` lists and which check 1 now also scans (see ``_GRANTED_SCAN_PATTERNS``).
    """
    fs = []
    for d in (KIT_DIR, RTL_DIR):
        if d.is_dir():
            fs += [p for p in d.rglob("*") if p.suffix in (".py", ".md") and "__pycache__" not in str(p)]
    fs += list(BUNDLES.glob("*/STARTER_PROMPT.md"))
    return fs


def check_answer_content() -> tuple[bool, list[str]]:
    bad = _scan_for_patterns(_shipped_files(), ANSWER_PATTERNS)
    # The granted tree is everything the agent can open. Only the store-naming pattern applies here --
    # see _GRANTED_SCAN_PATTERNS for why the other three would be noise rather than signal.
    bad += _scan_for_patterns(_granted_readable_files(),
                              {k: v for k, v in ANSWER_PATTERNS.items() if k in _GRANTED_SCAN_PATTERNS})
    return (not bad), bad


def _scan_for_patterns(files, patterns) -> list[str]:
    bad = []
    for f in files:
        if f.name in SCAN_EXEMPT:
            continue
        txt = f.read_text(errors="ignore")
        for name, rx in patterns.items():
            for m in rx.finditer(txt):
                # a generator's _TMPL or a prompt may legitimately say "never import merlin.runtime";
                # only flag if it's NOT in a negation/instruction context.
                ctx = txt[max(0, m.start() - 40):m.start()].lower()
                if any(w in ctx for w in ("not ", "never", "no ", "deny", "forbid", "without", "cheat")):
                    continue
                bad.append(f"{f.relative_to(REPO)} :: {name} :: …{txt[m.start():m.start()+50]!r}")
    return bad


# ---- checks 2-4: bundle allow/deny separation -----------------------------------------------------
def _manifest(arm: str, cond: str) -> dict:
    p = BUNDLES / f"{arm}_{cond}" / "input_bundle_manifest.yaml"
    return yaml.safe_load(p.read_text()) if p.is_file() else {}


def _ships(arm: str, cond: str) -> bool:
    """Does THIS target ship this arm at all?

    The arm list is the full roster the bench can run; a given target ships the subset it is being run
    with. An arm that is absent is not a moat violation -- it is an arm that is not part of this
    experiment -- and treating absence as failure made a target that ships six complete bundles read as
    NO-GO for lacking a seventh it never runs. Absence must never MASK a leak, though, so only the
    "this arm must have its treatment" direction is skipped; the "no other arm may reach it" direction
    below stays unconditional."""
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
        if _ships(EQSAT_ARM, cond):
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
            if not _ships(arm, cond):
                continue                       # arm not part of this target's experiment (see _ships)
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
            if not _ships(arm, cond):
                continue                       # arm not part of this target's experiment (see _ships)
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


#: A run whose submission is kept for the RECORD but is no longer a launch candidate carries this file.
#: It must state a reason, so retiring a submission is a documented decision and never a quiet deletion.
_SUPERSEDED_MARKER = "SUPERSEDED.txt"


def _superseded_reason(sub: Path) -> str | None:
    """The reason this submission is retired, or ``None`` if it is still a live candidate."""
    for where in (sub / _SUPERSEDED_MARKER, sub.parent / _SUPERSEDED_MARKER):
        if where.is_file():
            reason = where.read_text(encoding="utf-8", errors="replace").strip()
            return reason or f"{where.name} present but states no reason"
    return None


def check_workload_constants() -> tuple[bool, list[str]]:
    """Scan LIVE submission trees for workload-specific constants (empty == nothing to scan yet, which
    passes — the substantive scan runs post-freeze in grade_agent_run).

    Scoped to submissions that are still launch candidates. This used to scan every submission the runs
    tree had ever held, which conflates two different questions: "did a past run cheat?" (a finding, and
    a permanent one) with "is it safe to launch now?" (this gate). Once any historical run tripped it the
    gate was red forever, and the only ways out were to delete evidence or to bypass the gate — both
    worse than the problem. A retired submission is excluded by a ``SUPERSEDED.txt`` that STATES WHY, so
    the finding stays on disk and in the report instead of disappearing.
    """
    probs: list[str] = []
    notes: list[str] = []
    runs = C.RUNS          # out/runs/<target>/capsule-bench — the retired <experiment>/runs never
    if runs.is_dir():      # existed here, so this scan silently covered nothing and always "passed"
        for sub in sorted(runs.glob("*/*/submission")):
            reason = _superseded_reason(sub)
            hits = scan_workload_constants(sub)
            if reason is not None:
                if hits:   # still reported, just not blocking -- the record keeps the finding visible
                    notes.append(f"{sub.parent.name}: SUPERSEDED ({reason}) — {len(hits)} finding(s) "
                                 f"retained for the record, not blocking")
                continue
            probs.extend(f"{sub.parent.name}: {h}" for h in hits)
    return (not probs), probs + notes



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


def _holdout_stores() -> list[Path]:
    """Every ``hidden/`` store on disk, whether or not its contents can be listed.

    ``Path.exists()`` survives ``chmod 000`` while listing does not, and that difference is the only
    thing that separates "this checkout has no holdouts" from "this run answer-locked them". Walking
    for ``hidden/*/capsule.yaml`` cannot tell them apart -- both yield nothing.
    """
    caps = REPO / "merlin/contract/capsules"
    if not caps.is_dir():
        return []
    out: list[Path] = []
    for d in caps.rglob("hidden"):
        if d.is_dir():
            out.append(d)
    return sorted(out)


def _unreadable_holdout_stores() -> list[Path]:
    """The stores that EXIST but cannot be listed -- the blind case, which must never read as a pass."""
    blind: list[Path] = []
    for d in _holdout_stores():
        try:
            next(iter(d.iterdir()), None)
        except OSError:
            blind.append(d)
    return blind


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


def check_holdout_not_specified() -> tuple[bool, list[str]]:
    # FAIL CLOSED WHEN BLIND. A preflighted run answer-locks the holdout store (chmod 000) -- correctly,
    # the agent must not read it -- but that also empties the walk, and "no names" used to mean "nothing
    # to check" and report PASS. The check then went green precisely when it could not see, and its
    # success was indistinguishable from its blindness. This is the check that stops a held-out spec
    # reaching the granted tree, and a real leak of exactly that kind was found in this corpus.
    blind = _unreadable_holdout_stores()
    if blind:
        return False, [f"holdout store {d.relative_to(REPO)} exists but could not be read "
                       f"({'permission' if d.exists() else 'missing'}) — this check cannot see the "
                       f"names it exists to look for, so it reports FAIL rather than a vacuous pass"
                       for d in blind]
    names = _holdout_names()
    if not names:
        return True, ["(no holdout capsules on disk — nothing to check)"]
    bad = []
    n_scanned = 0
    for f in _granted_readable_files():
        n_scanned += 1
        txt = f.read_text(errors="ignore")
        for n in sorted(names):
            if n in txt:
                bad.append(f"{f.relative_to(REPO)} names holdout capsule {n!r}")
    if bad:
        return False, bad
    # A silent pass is not evidence of coverage: "found nothing" and "looked at nothing" print the same.
    # Say what was actually examined, so a scan that quietly stopped granting files is visible.
    return True, [f"scanned {n_scanned} granted file(s) for {len(names)} holdout capsule name(s) "
                  f"across {len(_holdout_stores())} store(s); none appear"]


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
