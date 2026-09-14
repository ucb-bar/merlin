#!/usr/bin/env python3
"""Lint gate: no hardcoded TARGET NAME string literals in the core library + build tooling.

The capsule-bench stack is target-agnostic: the target is resolved at runtime from the descriptor /
contract / manifest / registry, never baked into core logic. This gate enforces that the specific
target names this repo ships never silently reappear as *operative* string literals in the in-scope
trees:

  * ``merlin/python/merlin/**``
  * ``build_tools/scripts/**``

A violation is a **string-literal** ``ast.Constant`` (detected structurally with ``ast`` — this
checker uses no regex on itself, honoring the sibling no-regex gate) whose value contains one of the
known target names as a whole identifier (word-boundary, so ``gemmini`` inside ``gemmini_kernel``
does NOT match, but ``mx_gemmini`` matches its own entry). **Docstrings are exempt** (a target named
as documentation/example is allowed — the goal permits a name in schema/doc examples), and **comments
are not in the AST at all**, so a ``# e.g. gemmini`` note is fine. What is caught is a name used in
code: a default value, a comparison operand, a dict key/value, help/error text — the places that make
core logic operate on one specific target.

Allowed exceptions (checked in order):

  1. an inline ``# target-ok: <rationale>`` comment on the offending line (preferred — co-located);
  2. an entry in ``build_tools/scripts/target_name_allowlist.txt`` — either an exact repo-relative
     ``.py`` path, or a directory prefix ending in ``/`` (matches a whole reference-target subtree).

The allowlist only ever *shrinks*. Its two legitimate populations are (a) the in-tree REFERENCE
target implementations (a target's own backend/eval module naturally names itself) pending eviction
to a published package via ``MERLIN_TARGET_PATH`` (OV11), and (b) CLI convenience defaults where the
reference target is the documented example. As a reference target is evicted or a default removed,
delete its entry. Run::

    python build_tools/scripts/check_no_target_name.py            # full scan (exit 1 on violation)
    python build_tools/scripts/check_no_target_name.py --staged   # only git-staged files
    python build_tools/scripts/check_no_target_name.py --stop-hook # emit Claude Code Stop-hook JSON
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW_FILE = ROOT / "build_tools" / "scripts" / "target_name_allowlist.txt"
SCAN_ROOTS = ("merlin/python/merlin", "merlin/contract", "build_tools/scripts")
# Path fragments that mark BUILD-GENERATED (gitignored) trees, not source — never scanned.
EXCLUDE_FRAGMENTS = ("/_data/",)
INLINE_MARKER = "# target-ok:"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _target_roster  # noqa: E402  (sibling module, stdlib only)

# The concrete target names this repo ships -- DERIVED from the registries that declare a target (see
# _target_roster.py), never listed here. Until 2026-09-14 this was a literal set of seven, so every target
# registered after it (gemmini_universal, saturn_opu, saturn_opu_mxv256d128, ...) passed by construction.
#   * REFERENCE_DEFAULTS are declared targets deliberately NOT enforced: ``toy_npu`` is the kept onboarding
#     example (``families.DEFAULT_EXAMPLE_TARGET``), not hardware. ``rvv`` is an ISA class, not a target.
#   * RESIDUAL_NAMES are hardware names no registry declares as a target that core code must still not bake
#     in: ``npu_model`` is the external model package one target's oracle imports.
REFERENCE_DEFAULTS = frozenset({"toy_npu"})  # target-ok: the one unenforced reference example
RESIDUAL_NAMES = frozenset({"npu_model"})  # target-ok: external model package, declared by no registry
TARGET_NAMES = frozenset((_target_roster.target_names(ROOT) | RESIDUAL_NAMES) - REFERENCE_DEFAULTS)

# SUBSTRATES: boards, simulators and hardware-unit names. Not targets, but a literal naming one welds shared
# code to one machine all the same, and none was gated (measured 2026-09-14: 56 in-scope files). Each is
# attested by a registry -- a runtime BOARDS key, a sandbox SIM_TOOLCHAINS key, an RTL-facts producer, or a
# target contract -- and merlin/tests/infra/test_overfit_gate_regression.py fails if one stops being. A
# file whose own path names the substrate is that substrate's module and is exempt; the existing debt in
# generic files is recorded per file in target_substrate_ratchet.txt, which may only shrink.
SUBSTRATE_NAMES = frozenset({"k1", "spacemit", "kodiak", "cyclotron", "opu"})  # target-ok: the set hunted
SUBSTRATE_RATCHET = ROOT / "build_tools" / "scripts" / "target_substrate_ratchet.txt"


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _contains_identifier(text: str, name: str) -> bool:
    """True if ``name`` occurs in ``text`` bounded by non-word chars on both sides (a whole-identifier
    match, implemented without regex). So ``gemmini`` matches `` gemmini `` / ``"gemmini"`` but not
    ``gemmini_kernel`` or ``mx_gemmini`` — those are matched by their own entries in the name set."""
    start = 0
    n = len(name)
    while True:
        i = text.find(name, start)
        if i < 0:
            return False
        before_ok = i == 0 or not _is_word_char(text[i - 1])
        after_ok = i + n >= len(text) or not _is_word_char(text[i + n])
        if before_ok and after_ok:
            return True
        start = i + 1


def _load_allowlist() -> tuple[set[str], list[str]]:
    """Return (exact-file paths, directory prefixes). ``#`` comments and blank lines ignored; a
    trailing ``# rationale`` is dropped. A line ending in ``/`` is a subtree prefix."""
    exact: set[str] = set()
    prefixes: list[str] = []
    if ALLOW_FILE.is_file():
        for line in ALLOW_FILE.read_text(encoding="utf-8").splitlines():
            path = line.split("#", 1)[0].strip()
            if not path:
                continue
            if path.endswith("/"):
                prefixes.append(path)
            else:
                exact.add(path)
    return exact, prefixes


class _TargetNameVisitor(ast.NodeVisitor):
    """Collect (lineno, name, snippet) for string-literal Constants that name a target, skipping the
    docstring Constant of every module/class/function (documentation is allowed to name a target)."""

    def __init__(self, names: frozenset[str] | None = None) -> None:
        self.names = TARGET_NAMES if names is None else names
        self.docstrings: set[int] = set()      # id() of Constant nodes that are docstrings
        self.hits: list[tuple[int, str, str, int]] = []   # + end_lineno, so the marker may sit anywhere
        #                                                   inside a multi-line implicit concatenation

    def _mark_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            self.docstrings.add(id(body[0].value))

    def visit_Module(self, node: ast.Module) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._mark_docstring(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            for name in sorted(self.names):
                if _contains_identifier(node.value, name):
                    end = getattr(node, "end_lineno", None) or node.lineno
                    self.hits.append((node.lineno, name, node.value.strip()[:60], end))
                    break
        self.generic_visit(node)


def _scan_file(path: Path, names: frozenset[str] | None = None) -> list[tuple[int, str, str]]:
    """Target-name literals in ``path`` not silenced by an inline ``# target-ok:`` marker.

    The marker is honoured on ANY line of the offending constant, not only its first. Python reports an
    implicitly concatenated string at the line the FIRST fragment starts on, which can be many lines
    from the fragment that actually names a target -- so anchoring the marker there put the annotation
    nowhere near the thing it explains, and a reader who placed it correctly (beside the mention) saw
    the gate keep failing with no hint why.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    v = _TargetNameVisitor(names)
    v.visit(tree)
    if not v.hits:
        return []
    lines = src.splitlines()
    out = []
    for lineno, name, snippet, end in sorted(set(v.hits)):
        span = lines[max(lineno - 1, 0):min(end, len(lines))]
        if not any(INLINE_MARKER in line for line in span):
            out.append((lineno, name, snippet))
    return out


def _iter_targets(staged: bool) -> list[Path]:
    if staged:
        # FAIL CLOSED on an unreadable index. This gate's entire work list comes from `git`, so a `git`
        # that cannot run (bad GIT_DIR, no repo, no binary) yielded an EMPTY list and the gate printed
        # OK -- a green that could not have gone red. `check=True` turns that into an exception the
        # caller reports; see check_no_answer_keys.py, which fixed the same shape first.
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        rels = [ln for ln in out.splitlines() if ln.strip()]
    else:
        rels = []
        for root in SCAN_ROOTS:
            for p in sorted((ROOT / root).rglob("*.py")):
                rels.append(p.relative_to(ROOT).as_posix())
    targets = []
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        if any(frag in f"/{rel}" for frag in EXCLUDE_FRAGMENTS):
            continue  # build-generated bundle, not source
        if any(rel.startswith(r + "/") or rel == r for r in SCAN_ROOTS):
            targets.append(Path(rel))
    return targets


def _allowed(relstr: str, exact: set[str], prefixes: list[str]) -> bool:
    return relstr in exact or any(relstr.startswith(p) for p in prefixes)


def _path_tokens(relstr: str) -> set[str]:
    """The identifier tokens of a repo path: ``build_tools/scripts/k1_int8_ab.py`` -> {..., "k1", "int8", ...}."""
    out = relstr.lower()
    for sep in ("/", ".", "-"):
        out = out.replace(sep, "_")
    return {t for t in out.split("_") if t}


def _substrate_owned(relstr: str, name: str) -> bool:
    """True when the path itself names the substrate: the file is that board's/simulator's own module."""
    return name in _path_tokens(relstr)


def _load_substrate_ratchet() -> set[str]:
    if not SUBSTRATE_RATCHET.is_file():
        return set()
    out = set()
    for line in SUBSTRATE_RATCHET.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def substrate_hits(relstr: str) -> list[tuple[int, str, str]]:
    """Substrate-name literals in a generic file (the substrate's own modules are exempt)."""
    return [h for h in _scan_file(ROOT / relstr, SUBSTRATE_NAMES) if not _substrate_owned(relstr, h[1])]


def substrate_report(targets: list[Path]) -> tuple[list[str], list[str], int]:
    """(violations, healed ratchet entries, ratcheted files still carrying debt) over ``targets``."""
    ratchet = _load_substrate_ratchet()
    violations: list[str] = []
    healed: list[str] = []
    carried = 0
    for rel in targets:
        relstr = rel.as_posix()
        hits = substrate_hits(relstr)
        if relstr in ratchet:
            if hits:
                carried += 1
            else:
                healed.append(relstr)
            continue
        for lineno, name, snippet in hits:
            violations.append(f"{relstr}:{lineno}: hardcoded substrate name {name!r} in literal {snippet!r} "
                              f"(derive it from the board/simulator registry or the target contract, or add "
                              f"`# target-ok: <why>`)")
    return violations, healed, carried


# --- the coupling scan: what the literal check above cannot see ------------------------------------
# The check above inspects string-literal Constants only, and matches a target name as a WHOLE
# identifier. Both choices are deliberate and both hide real coupling:
#
#   * a whole-identifier match cannot see `gemmini_kernel`, `gemmini_fence`, `saturn_vec` or
#     `cycle_window_gemmini_region` -- vendor SYMBOL names, which are how one target's ABI leaks into
#     shared code;
#   * inspecting only literals cannot see `from ..runtime.backends import gemmini as gem`, which is how
#     a generic module acquires a hard dependency on one target. That is a bigger problem than a string,
#     and it was completely invisible.
#
# So this is a SECOND, separate check rather than a change to the first: the whole-identifier rule
# encodes a real intent (`mx_gemmini` must be its own entry, not a match for `gemmini`) and stays.
#
# A file whose OWN PATH names a target is that target's own module and is skipped -- self-reference is
# legitimate, and those files are already tracked as eviction candidates. What this reports is code with
# no target in its name that nonetheless depends on one: the inverted dependencies.
def _mentions(text: str, name: str) -> bool:
    """Case-insensitive substring test. Deliberately looser than ``_contains_identifier``."""
    return name in text.lower()


def _is_target_owned(relstr: str) -> bool:
    """True when the path itself names a target, i.e. the file is legitimately about that target."""
    return any(_mentions(relstr, name) for name in TARGET_NAMES)


def _scan_coupling(path: Path) -> list[tuple[int, str, str, str]]:
    """``(lineno, target, kind, snippet)`` for target coupling in a file that is not about a target."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))

    hits: list[tuple[int, str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parts = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            parts += [a.name for a in node.names]
            for part in parts:
                for name in TARGET_NAMES:
                    if _mentions(part, name):
                        hits.append((node.lineno, name, "import", part))
                        break
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            for name in TARGET_NAMES:
                # only what the whole-identifier check MISSES -- otherwise every literal is double-reported
                if _mentions(node.value, name) and not _contains_identifier(node.value, name):
                    hits.append((node.lineno, name, "symbol", node.value.strip()[:60]))
                    break
    lines = src.splitlines()
    return [h for h in sorted(set(hits))
            if INLINE_MARKER not in (lines[h[0] - 1] if 0 < h[0] <= len(lines) else "")]


def lift_candidates(staged: bool = False) -> list[str]:
    """Modules NAMED after a target whose code does not actually mention one.

    The third blind spot, and the one that hid the longest. A file called ``<target>_<thing>.py`` is
    exempt from the whole-identifier name check (its path is allowlisted, or the name is not a bare
    identifier) AND from the coupling scan (:func:`_is_target_owned` skips it as self-reference). Both
    exemptions are right for a module that really is about one target — but nothing distinguishes those
    from a fully general capability wearing the name of the first target that happened to exercise it,
    which is what ``muon_link.py`` and ``muon_bsp.py`` turned out to be: derived ISA facts throughout,
    ``target`` a parameter, and not one target reference outside a docstring.

    Reported, never enforced. A hit is a QUESTION — "is this general?" — and the answer is sometimes no
    (a module can be target-specific through its logic while naming nothing). What must not happen is
    the question going unasked because two exemptions cancel out.
    """
    out: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if not _is_target_owned(relstr):
            continue
        path = ROOT / rel
        if _scan_coupling(path) or _scan_file(path):
            continue                      # names a target in its code: the filename is earned
        out.append(f"{relstr}: named after a target, but its code names none — audit for a lift")
    return out


def coupling_inventory(staged: bool = False) -> list[str]:
    """Every generic in-scope module that depends on a specific target, as reportable lines."""
    out: list[str] = []
    for rel in _iter_targets(staged):
        relstr = rel.as_posix()
        if _is_target_owned(relstr):
            continue
        for lineno, name, kind, snippet in _scan_coupling(ROOT / rel):
            out.append(f"{relstr}:{lineno}: [{kind}] generic module depends on {name!r} — {snippet!r}")
    return out


#: This gate's name in its own messages.
_GATE = "no-target-name"


def _unexaminable(stop_hook: bool, exc: BaseException) -> int:
    """Refuse when the work list could not be read.

    "We could not look" is not "there is nothing to find". A `git` failure used to yield an empty
    work list and a printed OK, so an unreadable tree was indistinguishable from a clean one.
    Reported in whichever dialect the caller speaks (a Stop hook BLOCKS via JSON on stdout, not via
    the exit status), so the two cannot drift apart.
    """
    reason = (f"{_GATE}: could not list the files to examine ({exc}); NOTHING was examined, which is "
              f"not the same as clean. Fix the tree/index and re-run.")
    if stop_hook:
        print(json.dumps({"decision": "block", "reason": reason}))
        return 0  # stop-hook signals via JSON, not exit code
    print(f"[FAIL] {reason}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    exact, prefixes = _load_allowlist()

    if "--coupling" in argv:
        # The full inventory, for populating the overfit register. Advisory by design: this debt predates
        # the check and printing it is the point, so it exits 0 and lets the caller decide. An
        # unreadable work list is still a refusal: an empty inventory would read as "no coupling".
        try:
            found = coupling_inventory(staged)
        except (OSError, subprocess.CalledProcessError) as exc:
            return _unexaminable(stop_hook, exc)
        if not found:
            print("[  ok] target-coupling: no generic module depends on a specific target.")
            return 0
        print(f"[DEBT] target-coupling: {len(found)} dependency(ies) on a specific target in modules "
              "whose own name claims to be generic:")
        for line in found:
            print(f"  - {line}")
        lifts = lift_candidates(staged)
        if lifts:
            print(f"\n[NOTE] {len(lifts)} module(s) named after a target name none in their code — the "
                  "inverse case, where a general capability is filed as one vendor's plumbing:")
            for line in lifts:
                print(f"  - {line}")
        return 0

    violations: list[str] = []
    try:
        targets = _iter_targets(staged)
    except (OSError, subprocess.CalledProcessError) as exc:
        return _unexaminable(stop_hook, exc)
    for rel in targets:
        relstr = rel.as_posix()
        if _allowed(relstr, exact, prefixes):
            continue
        for lineno, name, snippet in _scan_file(ROOT / rel):
            violations.append(f"{relstr}:{lineno}: hardcoded target name {name!r} "
                              f"in literal {snippet!r} (resolve the target at runtime, add "
                              f"`# target-ok: <why>`, or allowlist the file)")

    sub_violations, healed, carried = substrate_report(targets)
    violations.extend(sub_violations)

    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": ("Hardcoded target name outside the allowlist (see "
                                         "build_tools/scripts/target_name_allowlist.txt):\n- "
                                         + "\n- ".join(violations))}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code

    if violations:
        print(f"[FAIL] no-target-name: {len(violations)} hardcoded target name(s) outside the allowlist:")
        for v in violations:
            print(f"  - {v}")
        return 1
    n_allow = len(exact) + len(prefixes)
    # Report the exemptions as DEBT, not as part of a pass. An allowlist announced on an "ok" line reads
    # as "nothing to see"; it is 36 places where the core is welded to a specific target.
    n_coupling = len(coupling_inventory(staged))
    print(f"[  ok] no-target-name: no stray target-name literal in scope "
          f"({len(TARGET_NAMES)} derived target names, {len(SUBSTRATE_NAMES)} substrates).")
    if healed:
        print(f"[note] {len(healed)} file(s) in {SUBSTRATE_RATCHET.name} no longer name a substrate; delete "
              f"their lines: {', '.join(healed)}")
    print(f"[DEBT] {n_allow} allowlisted file(s) still name a target, and {n_coupling} dependency(ies) on "
          f"a specific target sit in modules whose own name claims to be generic (--coupling to list). "
          f"Both counts may only fall.")
    if carried:
        print(f"[DEBT] {carried} generic file(s) still name a board/simulator/unit ({SUBSTRATE_RATCHET.name}); "
              f"the list may only shrink.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
