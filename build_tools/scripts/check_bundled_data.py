#!/usr/bin/env python3
"""Gate: every tree the wheel bundles must equal the tree it was bundled FROM.

Six curated corpora live at the repo top level, outside the importable package root:
``merlin/{schemas,prompts,benchmarks,contract,targets,runtime}``. setuptools can only ship data that
sits INSIDE a package dir, so ``setup.py``'s ``build_py`` hook copies them into a gitignored
``merlin/python/merlin/_data/`` which ``[tool.setuptools.package-data]`` puts in the wheel.

``merlin.common.paths.data_path`` then resolves the SAME logical file to different bytes depending on
where you stand: the top-level tree when a checkout exists, the bundled copy via
``importlib.resources`` when it does not. A checkout and an installed wheel can therefore enforce
different contracts, and nothing about the checkout shows it -- every test, linter and review reads
the copy that is right.

WHY A WHOLE-BUNDLE GATE AND NOT JUST THE CONTRACT ONE. ``check_contract_copies.py`` compares
``merlin/contract`` only, and exempts ``capsules/`` wholesale. That exemption is what let a bundle
sitting on disk carry 1192 ``golden.yaml`` / ``expected_instruction_coverage.yaml`` /
``*.hidden.yaml`` answer surfaces that ``setup.py`` had since been taught to exclude: the narrow gate
was structurally unable to look there. This gate instead applies the BUILD'S OWN exclusion rules --
read structurally out of ``setup.py`` -- to the source side, and compares the packaged side plainly.
A file the build would never ship but which is nevertheless sitting in the bundle shows up as
"only in package", which is exactly how a stale answer key becomes visible.

THE DECLARED DIVERGENCES. Some differences are intended, and they are declared in ONE place: the
build's own ``_EXCLUDE`` / ``_EXCLUDE_SUFFIXES`` / ``_CODE_SUFFIXES`` (the heavy capture corpora, the
per-target RTL-cert data, the answer surfaces, and code-not-data). This gate DERIVES those instead of
restating them, so the two cannot drift. Anything intentionally different that a build exclusion
cannot express goes in ``_INTENDED_DIVERGENCE`` below, with a reason per entry -- reviewed, not a
blanket skip. It is currently empty, and a bare entry with no reason is refused.

FAIL-CLOSED. If ``setup.py`` cannot be parsed, or does not carry the rule names this gate derives, the
rules are UNKNOWN and the gate refuses rather than comparing under a guessed rule set. If the bundle
has never been built in this checkout (no ``_data/`` at all -- the normal state of a fresh clone and of
CI) the gate reports ``not-built`` and makes no claim; ``--require-bundle`` turns that into a failure
for a context that should have one. A partial bundle -- ``_data/`` present but a tree missing under it
whose source exists -- is a divergence, not an absence, and fails.

PRE-COMMIT SCOPE. ``--staged`` narrows WHETHER the gate runs, never WHAT it compares: a commit that
stages a file under a bundled corpus is the moment the two copies part company, so that commit gets
the full comparison, and one that stages nothing there is told so explicitly. Several sessions share
this working tree and the bundle is a build product, so a blanket per-commit comparison would fail on
another session's edit and teach people to bypass the hook. A staged list that cannot be read runs
everything -- a work list that could not be read must never shrink the work.

The fix is never to hand-edit the bundle: rebuild it from the top-level trees. ``--sync`` does that
here (the build's ``build_py`` hook does the same thing, but needs setuptools, which the repo venv
does not carry); the copy rules come from ``setup.py`` either way, and re-running the check afterwards
verifies the result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import repo_root  # noqa: E402

#: Paths that are MEANT to differ between the top-level tree and the bundle and that no build
#: exclusion expresses, as ``"<kind>/<relative path or directory prefix>" -> reason``. Reviewed, and a
#: reason is mandatory: an entry whose reason is empty makes the gate refuse, so "allowlisted" can
#: never mean "someone silenced it". Empty today -- every intended difference is currently a build
#: exclusion, which this gate derives from ``setup.py`` rather than restating here.
_INTENDED_DIVERGENCE: dict[str, str] = {}

#: Where the bundle lives inside the importable package. Spelled literally, NOT resolved through
#: ``data_path()``: that helper prefers the checkout, so asking it for "the packaged copy" hands back
#: the top-level copy and the comparison degrades into a tree against itself -- the exact defect the
#: contract-copies gate shipped with once already.
_PACKAGED_REL = ("merlin", "python", "merlin", "_data")
#: Where the curated corpora live. Also literal, same reason.
_SOURCE_REL = ("merlin",)

#: Names in ``setup.py`` this gate derives the build's copy rules from. Missing any of them is
#: UNKNOWN: the build changed shape and this gate no longer knows what it is allowed to ignore.
_RULE_NAMES = ("_EXCLUDE", "_EXCLUDE_SUFFIXES", "_CODE_SUFFIXES")
#: The name whose keys enumerate the bundled trees.
_KINDS_FROM = "_BUNDLE"


def _setup_rules(base: Path) -> dict:
    """Read the build's copy rules out of ``setup.py`` structurally (``ast``, never text matching).

    Returns ``{"kinds": (...), "exclude": {...}, "exclude_suffixes": {...}, "code_suffixes": (...)}``
    or ``{"unknown": <why>}``. The build is the authority on what it ships; this gate asks it rather
    than keeping a second copy of the answer that can fall behind.
    """
    src = base / "setup.py"
    if not src.is_file():
        return {"unknown": f"no setup.py at {src} — the build's copy rules cannot be read"}
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return {"unknown": f"setup.py is unreadable/unparseable ({exc}) — copy rules UNKNOWN"}

    literals: dict[str, object] = {}
    kinds: tuple[str, ...] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            if name in _RULE_NAMES:
                try:
                    literals[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
            elif name == _KINDS_FROM:
                # ``_BUNDLE = {kind: ... for kind in (...)}`` — the iterated tuple names the trees.
                found = [ast.literal_eval(g.iter) for g in getattr(node.value, "generators", [])
                         if isinstance(g.iter, (ast.Tuple, ast.List))]
                if found:
                    kinds = tuple(str(k) for k in found[0])

    missing = [n for n in _RULE_NAMES if n not in literals]
    if missing or not kinds:
        gap = ", ".join(missing + ([] if kinds else [_KINDS_FROM]))
        return {"unknown": f"setup.py no longer defines {gap} — the build's copy rules are UNKNOWN"}
    return {
        "kinds": kinds,
        "exclude": {k: tuple(v) for k, v in dict(literals["_EXCLUDE"]).items()},
        "exclude_suffixes": {k: tuple(v) for k, v in dict(literals["_EXCLUDE_SUFFIXES"]).items()},
        "code_suffixes": tuple(literals["_CODE_SUFFIXES"]),
    }


def _is_ignored(name: str, kind: str, rules: dict) -> bool:
    """The build's own per-name ignore decision, applied at every directory level as copytree does."""
    if name == "__pycache__":
        return True
    if name.endswith(rules["code_suffixes"]):
        return True
    if any(name.startswith(p) for p in rules["exclude"].get(kind, ())):
        return True
    return any(name.endswith(x) for x in rules["exclude_suffixes"].get(kind, ()))


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _walk_source(root: Path, kind: str, rules: dict) -> dict[str, str]:
    """``relative path -> digest`` for what the build WOULD copy out of ``root``.

    Mirrors ``shutil.copytree(ignore=...)``: an ignored name is dropped, and an ignored directory is
    not descended into.
    """
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        here, rel = stack.pop()
        try:
            entries = sorted(here.iterdir())
        except OSError:
            out[rel or "."] = "UNREADABLE"   # a directory we could not list is UNKNOWN, not empty
            continue
        for entry in entries:
            if _is_ignored(entry.name, kind, rules):
                continue
            child = f"{rel}/{entry.name}" if rel else entry.name
            try:
                if entry.is_dir():
                    stack.append((entry, child))
                    continue
                if not entry.is_file():
                    continue
                out[child] = _digest(entry)
            except OSError:
                out[child] = "UNREADABLE"
    return out


def _walk_packaged(root: Path) -> dict[str, str]:
    """``relative path -> digest`` for everything actually sitting in the bundle. No filtering: a
    file the build would not ship but which IS there is the interesting case, not one to hide."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        try:
            if not p.is_file():
                continue
            out[str(p.relative_to(root))] = _digest(p)
        except OSError:
            out[str(p.relative_to(root))] = "UNREADABLE"
    return out


def _allowed(kind: str, rel: str) -> str | None:
    """The declared reason this path may differ, or ``None``. Matches a whole path or a directory
    prefix under a kind."""
    key = f"{kind}/{rel}"
    for declared, reason in _INTENDED_DIVERGENCE.items():
        if key == declared or key.startswith(declared.rstrip("/") + "/"):
            return reason
    return None


def audit(root: Path | None = None) -> dict:
    """Compare every bundled tree against the tree it is bundled from.

    Never returns a clean report it could not earn: an unparseable build, an unreadable file and an
    absent bundle each get their own state.
    """
    base = Path(root) if root is not None else repo_root()
    bad_allowlist = sorted(k for k, v in _INTENDED_DIVERGENCE.items() if not str(v).strip())
    if bad_allowlist:
        return {"status": "unknown",
                "reason": ("declared divergences without a reason: " + ", ".join(bad_allowlist)
                           + " — an allowlist entry with no reason is a silent skip"),
                "trees": {}, "differing": [], "only_in_source": [], "only_in_packaged": [],
                "unreadable": [], "allowed": {}}

    rules = _setup_rules(base)
    if "unknown" in rules:
        return {"status": "unknown", "reason": rules["unknown"], "trees": {}, "differing": [],
                "only_in_source": [], "only_in_packaged": [], "unreadable": [], "allowed": {}}

    source_root = base.joinpath(*_SOURCE_REL)
    packaged_root = base.joinpath(*_PACKAGED_REL)
    if not packaged_root.is_dir():
        return {"status": "not-built",
                "reason": (f"no bundle at {packaged_root} — the build has never run in this checkout, "
                           "so there is nothing to compare (not a clean bill of health)"),
                "trees": {}, "differing": [], "only_in_source": [], "only_in_packaged": [],
                "unreadable": [], "allowed": {},
                "source": str(source_root), "packaged": str(packaged_root)}

    differing: list[str] = []
    only_source: list[str] = []
    only_packaged: list[str] = []
    unreadable: list[str] = []
    allowed: dict[str, str] = {}
    trees: dict[str, dict] = {}

    for kind in rules["kinds"]:
        src = _walk_source(source_root / kind, kind, rules)
        pkg = _walk_packaged(packaged_root / kind)
        if not src and not pkg:
            trees[kind] = {"source": 0, "packaged": 0, "compared": 0, "state": "absent both sides"}
            continue
        d = sorted(k for k in set(src) & set(pkg) if src[k] != pkg[k])
        os_only = sorted(set(src) - set(pkg))
        op_only = sorted(set(pkg) - set(src))
        un = sorted(k for k in set(src) | set(pkg)
                    if src.get(k) == "UNREADABLE" or pkg.get(k) == "UNREADABLE")

        for bucket, sink in ((d, differing), (os_only, only_source), (op_only, only_packaged)):
            for rel in bucket:
                reason = _allowed(kind, rel)
                if reason is None:
                    sink.append(f"{kind}/{rel}")
                else:
                    allowed[f"{kind}/{rel}"] = reason
        unreadable.extend(f"{kind}/{k}" for k in un)
        trees[kind] = {"source": len(src), "packaged": len(pkg),
                       "compared": len(set(src) & set(pkg)),
                       "differing": len(d), "only_in_source": len(os_only),
                       "only_in_packaged": len(op_only)}

    return {
        "status": "compared",
        "source": str(source_root),
        "packaged": str(packaged_root),
        "kinds": list(rules["kinds"]),
        "build_rules": {"exclude": {k: list(v) for k, v in rules["exclude"].items()},
                        "exclude_suffixes": {k: list(v)
                                             for k, v in rules["exclude_suffixes"].items()},
                        "code_suffixes": list(rules["code_suffixes"])},
        "trees": trees,
        "n_compared": sum(t.get("compared", 0) for t in trees.values()),
        "differing": sorted(differing),
        "only_in_source": sorted(only_source),
        "only_in_packaged": sorted(only_packaged),
        "unreadable": sorted(unreadable),
        "allowed": allowed,
    }


def findings(rep: dict) -> list[str]:
    """Every path the report holds against the bundle, in one list."""
    return (rep.get("differing", []) + rep.get("only_in_source", [])
            + rep.get("only_in_packaged", []) + rep.get("unreadable", []))


def staged_touches_a_bundled_tree(kinds, base: Path | None = None) -> tuple[bool, str]:
    """Does the staged change touch a tree the wheel bundles? ``(verdict, why)``.

    Pre-commit scope decision, and the ONLY thing ``--staged`` narrows: the bundle is a build product,
    so on a tree several sessions share it can be stale for reasons that have nothing to do with the
    commit in hand, and gating every commit on that would train people to bypass the hook. A commit
    that edits a bundled corpus IS the moment the two copies part company, so that one is checked --
    in full, because a partial comparison would be a weaker claim than the one this gate makes.

    If the staged list cannot be read the answer is not "no": it runs the full check. A work list that
    could not be read must never shrink the work.
    """
    import subprocess

    root = Path(base) if base is not None else repo_root()
    try:
        proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                              capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return True, f"the staged file list is unreadable ({exc}), which is not the same as clean"
    if proc.returncode != 0:
        return True, "git could not list the staged files, which is not the same as clean"
    prefixes = tuple(f"merlin/{kind}/" for kind in kinds)
    for line in proc.stdout.splitlines():
        path = line.strip()
        if path.startswith(prefixes):
            return True, f"staged {path}"
    return False, "no bundled corpus is staged"


def sync(base: Path | None = None) -> dict:
    """Rebuild the bundle from the top-level trees, using the BUILD'S rules.

    Same effect as ``setup.py``'s ``build_py`` hook (rmtree + copytree per tree, with the build's
    ignore callback), minus the setuptools dependency the repo venv does not carry. The rules — which
    trees, and what each one drops — are still read out of ``setup.py``, so this cannot ship something
    the build would not; and :func:`audit` re-run afterwards is what actually proves it.
    """
    import shutil

    root = Path(base) if base is not None else repo_root()
    rules = _setup_rules(root)
    if "unknown" in rules:
        return {"status": "unknown", "reason": rules["unknown"], "synced": []}
    done: list[str] = []
    for kind in rules["kinds"]:
        src = root.joinpath(*_SOURCE_REL, kind)
        dst = root.joinpath(*_PACKAGED_REL, kind)
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst,
                        ignore=lambda _d, names, _k=kind: {n for n in names
                                                           if _is_ignored(n, _k, rules)})
        done.append(kind)
    return {"status": "synced", "synced": done}


def _render(rep: dict, limit: int) -> None:
    print("bundled-data: the wheel bundle and the trees it is built from DISAGREE.")
    print(f"  source  : {rep.get('source')}")
    print(f"  packaged: {rep.get('packaged')}")
    for key, label in (("differing", "differs        "),
                       ("only_in_source", "only in repo   "),
                       ("only_in_packaged", "only in package"),
                       ("unreadable", "UNREADABLE     ")):
        items = rep.get(key, [])
        for k in items[:limit]:
            print(f"  * {label}: {k}")
        if len(items) > limit:
            print(f"  * {label}: … +{len(items) - limit} more")
    print("  fix: python build_tools/scripts/check_bundled_data.py --sync  "
          "(rebuilds the bundle from the top-level trees) — never hand-edit the bundle.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--staged", action="store_true",
                    help="pre-commit mode: run the FULL comparison when the commit stages a file "
                         "under a bundled corpus (the moment the copies part company), and say so "
                         "when it does not. The comparison itself is never narrowed.")
    ap.add_argument("--stop-hook", action="store_true",
                    help="session Stop hook mode: JSON-only on stdout, blocks via "
                         "{'decision': 'block'} rather than the exit status")
    ap.add_argument("--require-bundle", action="store_true",
                    help="treat an unbuilt bundle as a failure (for a context that must have one)")
    ap.add_argument("--limit", type=int, default=40,
                    help="max paths printed per category (the full list is in --json)")
    ap.add_argument("--sync", action="store_true",
                    help="rebuild the bundle from the top-level trees (the build's own rules), then "
                         "re-check — the fix for a divergence; never hand-edit the bundle")
    a = ap.parse_args(argv)

    if a.sync:
        res = sync()
        if res["status"] == "unknown":
            print(f"bundled-data: {res['reason']}")
            return 1
        print(f"bundled-data: rebuilt {', '.join(res['synced'])} from the top-level trees.")

    if a.staged and not a.sync:
        rules = _setup_rules(repo_root())
        if "unknown" in rules:
            print(f"bundled-data: {rules['unknown']}")
            return 1
        run_it, why = staged_touches_a_bundled_tree(rules["kinds"])
        if not run_it:
            print(f"[skip] bundled-data: {why} — the bundle is a build product; this commit cannot "
                  f"have changed what it is built from.")
            return 0
        # Say WHY it is running, so the widened case (git unreadable -> check everything) is visible
        # rather than looking like an ordinary silent pass.
        print(f"bundled-data: {why} — comparing in full.")

    rep = audit()
    status = rep.get("status")
    bad = findings(rep)
    blocked = ""
    if status == "unknown":
        blocked = f"bundled-data: {rep['reason']}"
    elif status == "not-built" and a.require_bundle:
        blocked = f"bundled-data: {rep['reason']}"
    elif bad:
        blocked = ("The wheel bundle and the trees it is built from DISAGREE "
                   f"({len(bad)} path(s); fix: check_bundled_data.py --sync):\n- " + "\n- ".join(bad[:a.limit])
                   + (f"\n- … +{len(bad) - a.limit} more" if len(bad) > a.limit else ""))

    if a.stop_hook:
        # A Claude Code Stop hook BLOCKS through {"decision": "block"} on stdout; a non-zero exit is a
        # NON-blocking error there, so stdout stays JSON-only in this mode.
        print(json.dumps({"decision": "block", "reason": blocked} if blocked else {}))
        return 0
    if a.json:
        print(json.dumps(rep, indent=2))
        return 1 if blocked else 0
    if status == "unknown":
        print(blocked)
        return 1
    if status == "not-built":
        # Reported as its own state, never as "[ ok]": no comparison was made, so no claim is made.
        print(f"[skip] bundled-data: {rep['reason']}")
        return 1 if a.require_bundle else 0
    if bad:
        _render(rep, a.limit)
        return 1
    if not a.staged:
        kinds = ", ".join(rep.get("kinds", []))
        print(f"[  ok] bundled-data: {rep['n_compared']} file(s) identical across the source trees "
              f"and the wheel bundle ({kinds})."
              + (f" {len(rep['allowed'])} declared divergence(s)." if rep.get("allowed") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
