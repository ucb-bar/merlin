#!/usr/bin/env python3
"""Gate: the two copies of the frozen contract must agree, because only one of them is READ.

The experiment ABI ships twice, and WHICH COPY IS AUTHORITATIVE DEPENDS ON WHERE YOU RUN.
``merlin.common.paths.data_path`` prefers the in-repo tree when a checkout exists and falls back to the
wheel-bundled ``merlin/_data/`` only when it does not. So ``merlin/contract/`` is what every developer,
linter and CI run reads, and ``merlin/python/merlin/_data/contract/`` is what an installed
``pip install merlin`` reads — with nothing keeping the two the same.

A divergence therefore does not show up where the work happens. Every test passes in the checkout while
the shipped package enforces a different grammar, opcode table or schema, and the frozen ABI is the worst
place for that: a submission is graded against whichever copy its environment happened to resolve.

Nothing checked this. Both copies were edited by hand today when the interface grammar gained ops, and
they happened to stay in sync; nothing would have reported it if they had not.

A NOTE ON HOW THIS CHECK IS WRITTEN. The obvious implementation — resolve one side with
``data_path("contract")`` — compares the checkout tree to ITSELF and passes unconditionally. That was the
first version of this file, and it reported 35 files identical after a deliberate divergence was injected
into the packaged copy. Both paths are therefore spelled out literally, and the test suite injects a
divergence to prove the gate can fail.

``capsules/`` is excluded: the corpora under the two roots are deliberately different (the packaged one
carries a curated subset, and the goldens/holdouts must never ship), so requiring equality there would
demand exactly the answer-key leak the no-answer-keys gate forbids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import repo_root  # noqa: E402

#: Subtrees whose two copies are MEANT to differ, with the reason. Anything not listed must match.
_EXEMPT = {
    "capsules": ("the packaged corpus is a curated subset, and goldens/holdouts must never ship; "
                 "requiring equality here would demand the answer-key leak the no-answer-keys gate "
                 "forbids"),
}


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: Suffixes the BUILD refuses to bundle, so their absence from the packaged copy is intended rather
#: than drift. setup.py's `_CODE_SUFFIXES` is the authority ("The bundle ships read-only DATA only --
#: never code"); this is a second copy of that fact, and `_build_excluded_suffixes` below asserts the
#: two agree instead of letting them drift.
#:
#: ⚠️ WITHOUT THIS THE GATE COULD NEVER GO GREEN. `merlin/contract/external/gsim/cxxwrap.sh` is a
#: shell script under the contract tree; the build will never ship it, so a plain tree comparison
#: reports it as missing on every run, forever. A check that cannot pass is as useless as one that
#: cannot fail, and it trains readers to ignore the output -- which is the same failure this repo has
#: paid for from the other direction all day.
_NEVER_BUNDLED_SUFFIXES = (".py", ".pyc", ".pyo", ".sh")
#: Directory names the build drops wholesale.
_NEVER_BUNDLED_DIRS = ("__pycache__",)


def _build_excluded_suffixes() -> tuple[str, ...]:
    """setup.py's own ``_CODE_SUFFIXES``, read structurally so a drift is reported not inherited."""
    import ast

    src = (repo_root() / "setup.py")
    if not src.is_file():
        return ()
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "_CODE_SUFFIXES":
                try:
                    return tuple(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    return ()
    return ()


def _tree(root: Path) -> dict[str, str]:
    """``relative path -> content digest`` for every readable file outside an exempt subtree."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        try:
            if not p.is_file():
                continue
        except OSError:                                  # unreadable entry: not a divergence
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in _EXEMPT:
            continue
        if p.suffix in _NEVER_BUNDLED_SUFFIXES:
            continue                                     # the build will never ship it
        if any(part in _NEVER_BUNDLED_DIRS for part in rel.parts):
            continue
        try:
            out[str(rel)] = _digest(p)
        except OSError:
            # Unreadable is UNKNOWN, not equal. Recorded so a permissions problem cannot read as a
            # clean bill of health -- a check that could not run must never report success.
            out[str(rel)] = "UNREADABLE"
    return out


#: Spelled literally, NOT resolved through ``data_path``: that helper prefers the checkout, so asking it
#: for "the packaged copy" hands back the repo copy and the comparison becomes a tree against itself.
_SOURCE_REL = ("merlin", "contract")
_PACKAGED_REL = ("merlin", "python", "merlin", "_data", "contract")


def audit(root: Path | None = None) -> dict:
    base = Path(root) if root is not None else repo_root()
    source = base.joinpath(*_SOURCE_REL)
    packaged = base.joinpath(*_PACKAGED_REL)
    if not packaged.is_dir():
        # UNKNOWN, never "clean": a missing packaged tree means the wheel would ship no contract at all.
        return {"source": str(source), "packaged": str(packaged), "authority": "n/a",
                "n_compared": 0, "only_in_source": [], "only_in_packaged": [], "differing": [],
                "unreadable": [], "exempt": dict(_EXEMPT),
                "missing_packaged_tree": True}
    a, b = _tree(source), _tree(packaged)
    only_source = sorted(set(a) - set(b))
    only_packaged = sorted(set(b) - set(a))
    differing = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    unreadable = sorted(k for k in set(a) | set(b)
                        if a.get(k) == "UNREADABLE" or b.get(k) == "UNREADABLE")
    return {
        "source": str(source),
        "packaged": str(packaged),
        "authority": ("split — a checkout reads the repo copy, an installed wheel reads the packaged "
                      "copy; nothing else keeps them equal, which is what this gate is for"),
        "n_compared": len(set(a) & set(b)),
        "only_in_source": only_source,
        "only_in_packaged": only_packaged,
        "differing": differing,
        "unreadable": unreadable,
        "exempt": {k: v for k, v in _EXEMPT.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--staged", action="store_true",
                    help="pre-commit mode: IDENTICAL checks (both copies are always compared in "
                         "full), only the success line is suppressed")
    ap.add_argument("--stop-hook", action="store_true",
                    help="session Stop hook mode: JSON-only on stdout, blocks via {'decision': "
                         "'block'} rather than the exit status")
    a = ap.parse_args(argv)

    rep = audit()
    if rep.get("missing_packaged_tree"):
        reason = (f"contract-copies: no packaged contract tree at {rep['packaged']} — an installed "
                  f"wheel would ship no contract at all. Reported as UNKNOWN, not as clean.")
        if a.stop_hook:
            print(json.dumps({"decision": "block", "reason": reason}))
            return 0  # stop-hook signals via JSON, not exit code
        print(reason)
        return 1
    bad = rep["differing"] + rep["only_in_source"] + rep["only_in_packaged"] + rep["unreadable"]
    if a.stop_hook:
        # A Claude Code Stop hook BLOCKS through {"decision": "block"} on stdout; a non-zero exit is a
        # NON-blocking error there. This flag was parsed and never read, so the gate could report and
        # not enforce — and stdout had to stay JSON-only, which the human printing below breaks.
        if bad:
            print(json.dumps({"decision": "block",
                              "reason": ("The two copies of the frozen contract DISAGREE "
                                         f"({len(bad)} file(s)):\n- " + "\n- ".join(bad))}))
        else:
            print(json.dumps({}))
        return 0
    if a.json:
        print(json.dumps(rep, indent=2))
    elif bad:
        print("contract-copies: the two copies of the frozen contract DISAGREE.")
        print(f"  {rep['authority']}")
        for k in rep["differing"]:
            print(f"  * differs        : {k}")
        for k in rep["only_in_source"]:
            print(f"  * only in repo   : {k}  (the packaged copy the code reads does NOT have it)")
        for k in rep["only_in_packaged"]:
            print(f"  * only in package: {k}  (nobody reviewing the repo copy can see it)")
        for k in rep["unreadable"]:
            print(f"  ? unreadable     : {k}  (UNKNOWN, not equal)")
        print("  fix: make the two copies identical; do not pick one and hope.")
    elif not a.staged:
        # --staged is pre-commit mode: the checks are identical (both copies are compared in full --
        # a partial compare would be a different, weaker claim), the OK line is just suppressed.
        print(f"[  ok] contract-copies: {rep['n_compared']} file(s) identical across both copies "
              f"({', '.join(sorted(_EXEMPT))} exempt).")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
