#!/usr/bin/env python3
"""Gate: staged contract resources must match their reviewed source authority.

Current manifest-v2 builds keep contracts under ``src/merlin/resources/contracts``
and stage them at ``out/build/python/lib/merlin/resources/contracts``. The shared
resource audit owns selection and detects stale extras as well as changed bytes.
Editable and installed code use this same package-relative resource layout.

Explicit archived version-1 manifests and older source layouts keep their
original ``merlin/contract`` and ``_data/contract`` comparison, exemptions, and
diagnostics. Missing staging remains unknown rather than a successful comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import repo_root  # noqa: E402

#: Subtrees whose two copies are MEANT to differ, with the reason. Anything not listed must match.
_EXEMPT = {
    "capsules": (
        "the packaged corpus is a curated subset, and goldens/holdouts must never ship; "
        "requiring equality here would demand the answer-key leak the no-answer-keys gate "
        "forbids"
    ),
}


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: Suffixes the BUILD refuses to bundle, so their absence from the packaged copy is intended rather
#: than drift. setup.py's `_CODE_SUFFIXES` is the authority ("The bundle ships read-only DATA only --
#: never code"); this is a second copy of that fact, and `_build_excluded_suffixes` below asserts the
#: two agree instead of letting them drift.
#:
#: ⚠️ WITHOUT THIS THE GATE COULD NEVER GO GREEN. `src/merlin/resources/contracts/external/gsim/cxxwrap.sh` is a
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

    src = repo_root() / "setup.py"
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
        except OSError:  # unreadable entry: not a divergence
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in _EXEMPT:
            continue
        if p.suffix in _NEVER_BUNDLED_SUFFIXES:
            continue  # the build will never ship it
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
    if (base / "build_tools/package_resources.json").is_file():
        # The reviewed manifest is the build authority. Reuse the full-bundle audit;
        # do not recreate its resource selection or exempt stale private files.
        import importlib.util

        spec = importlib.util.spec_from_file_location("_merlin_bundle_audit", _HERE.with_name("check_bundled_data.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.audit(base)
        manifest = json.loads((base / "build_tools/package_resources.json").read_text(encoding="utf-8"))
        current = manifest.get("version") == 2
        contract_kind = "resources/contracts" if current else "contract"
        result = {
            "source": str(base / ("src/merlin/resources/contracts" if current else "merlin/contract")),
            "packaged": str(base / "out/build/python/lib/merlin" / contract_kind)
            if current
            else str(base / "out/build/python/lib/merlin/_data/contract"),
            "authority": "reviewed public-resource manifest",
            "n_compared": report.get("trees", {}).get(contract_kind, {}).get("compared", 0),
            "exempt": {},
        }
        for key in ("only_in_source", "only_in_packaged", "differing", "unreadable"):
            result[key] = [
                name.removeprefix(contract_kind + "/")
                for name in report.get(key, [])
                if name.startswith(contract_kind + "/")
            ]
        if report["status"] != "compared":
            result["missing_packaged_tree"] = True
            result["reason"] = report.get("reason", "bundle could not be checked")
        return result
    source = base.joinpath(*_SOURCE_REL)
    packaged = base.joinpath(*_PACKAGED_REL)
    if not packaged.is_dir():
        # UNKNOWN, never "clean": a missing packaged tree means the wheel would ship no contract at all.
        return {
            "source": str(source),
            "packaged": str(packaged),
            "authority": "n/a",
            "n_compared": 0,
            "only_in_source": [],
            "only_in_packaged": [],
            "differing": [],
            "unreadable": [],
            "exempt": dict(_EXEMPT),
            "missing_packaged_tree": True,
        }
    a, b = _tree(source), _tree(packaged)
    only_source = sorted(set(a) - set(b))
    only_packaged = sorted(set(b) - set(a))
    differing = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    unreadable = sorted(k for k in set(a) | set(b) if a.get(k) == "UNREADABLE" or b.get(k) == "UNREADABLE")
    return {
        "source": str(source),
        "packaged": str(packaged),
        "authority": (
            "split — a checkout reads the repo copy, an installed wheel reads the packaged "
            "copy; nothing else keeps them equal, which is what this gate is for"
        ),
        "n_compared": len(set(a) & set(b)),
        "only_in_source": only_source,
        "only_in_packaged": only_packaged,
        "differing": differing,
        "unreadable": unreadable,
        "exempt": {k: v for k, v in _EXEMPT.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--staged",
        action="store_true",
        help="pre-commit mode: IDENTICAL checks (both copies are always compared in "
        "full), only the success line is suppressed",
    )
    ap.add_argument(
        "--stop-hook",
        action="store_true",
        help="session Stop hook mode: JSON-only on stdout, blocks via {'decision': "
        "'block'} rather than the exit status",
    )
    a = ap.parse_args(argv)

    rep = audit()
    if rep.get("missing_packaged_tree"):
        reason = (
            f"contract-copies: no packaged contract tree at {rep['packaged']} — an installed "
            f"wheel would ship no contract at all. Reported as UNKNOWN, not as clean."
        )
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
            print(
                json.dumps(
                    {
                        "decision": "block",
                        "reason": (
                            "The two copies of the frozen contract DISAGREE "
                            f"({len(bad)} file(s)):\n- " + "\n- ".join(bad)
                        ),
                    }
                )
            )
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
        print(
            f"[  ok] contract-copies: {rep['n_compared']} file(s) identical across both copies "
            f"({', '.join(sorted(_EXEMPT))} exempt)."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
