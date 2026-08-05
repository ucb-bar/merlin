"""Native hookup to the published ``<target>-mlir`` out-of-tree repos.

A target definition is a self-contained OOT package — a ``<target>-mlir`` repo shipping
``contracts/target_contract.yaml`` (+ optional ``plugin.backend``) exactly as
:func:`merlin.targetgen.capability_manifests.write_oot_target` emits. ``target_registry.resolve``
already picks such a package up with zero env once it sits in the generated-target home
(``out/build/generated/<target>/``). This module is the missing verb: it FETCHES that repo into the
home, so ``merlin-target-fetch <target>`` is all it takes to plug a published target in.

Each champion experiment is a *branch* of the target's repo (the default branch is the target base,
carrying the reference backend + contract; a champion branch forks from it and adds a codegen
payload). ``--champion <branch>`` selects one.

The repo URL is NEVER hardcoded per target: it comes from a configurable template — the paths edge,
naming WHERE a target's sources live, parameterized by the target name — so a new target needs only
its published repo, not a merlin edit:

* ``MERLIN_TARGET_REPO_<TARGET>`` (target upper-cased, ``-``/``.`` → ``_``) — an exact URL override
  for one target; else
* ``MERLIN_TARGET_REPO_TEMPLATE`` — a ``{target}`` template
  (default ``https://github.com/ucb-bar/{target}-mlir.git``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .target_registry import generated_target_home, _is_target_root

# The default publish home + naming convention for target repos (the paths edge — an org + a
# ``<target>-mlir`` suffix, parameterized by target; NOT a target-name literal). Overridable so a fork
# or a private mirror plugs in with one env var.
_DEFAULT_URL_TEMPLATE = "https://github.com/ucb-bar/{target}-mlir.git"
_ENV_URL_TEMPLATE = "MERLIN_TARGET_REPO_TEMPLATE"
_ENV_URL_PREFIX = "MERLIN_TARGET_REPO_"          # + <TARGET> for a per-target exact URL
_GIT_TIMEOUT_S = 180


class FetchError(RuntimeError):
    """A fetch could not be completed (bad URL, network, or the repo is not a target package)."""


def _env_key_for(target: str) -> str:
    """The per-target URL-override env var name (``MERLIN_TARGET_REPO_<TARGET>``)."""
    norm = "".join(c if c.isalnum() else "_" for c in target).upper()
    return _ENV_URL_PREFIX + norm


def repo_url(target: str) -> str:
    """The git URL for ``target``'s OOT repo — per-target env override, else the template."""
    override = os.environ.get(_env_key_for(target))
    if override:
        return override
    template = os.environ.get(_ENV_URL_TEMPLATE, _DEFAULT_URL_TEMPLATE)
    if "{target}" not in template:
        raise FetchError(
            f"{_ENV_URL_TEMPLATE} must contain '{{target}}' (got {template!r})"
        )
    return template.format(target=target)


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except FileNotFoundError as exc:  # git not installed
        raise FetchError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"git {args[0]} timed out after {_GIT_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"git {args[0]} failed: {(exc.stderr or exc.stdout or '').strip()}") from exc


def fetch(
    target: str,
    *,
    champion: str | None = None,
    dest: Path | None = None,
    url: str | None = None,
    depth: int = 1,
    update: bool = True,
) -> Path:
    """Fetch ``target``'s OOT repo into the generated-target home and return the package root.

    ``champion`` selects the branch (default: the repo's default branch — the target base). ``update``
    re-fetches + hard-resets an already-cloned home to the requested ref; set it False to keep a local
    clone untouched. Raises :class:`FetchError` on any git failure or if the result is not a target
    package (no ``contracts/target_contract.yaml``).
    """
    url = url or repo_url(target)
    dest = Path(dest) if dest is not None else generated_target_home() / target
    ref = champion  # a branch or tag; None → clone the default branch

    if (dest / ".git").is_dir():
        if update:
            _run_git(["fetch", "--depth", str(depth), "origin", *( [ref] if ref else [] )], cwd=dest)
            checkout = ref if ref else "FETCH_HEAD"
            _run_git(["checkout", "-q", checkout], cwd=dest)
            if ref:
                _run_git(["reset", "--hard", f"origin/{ref}"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone = ["clone", "--depth", str(depth)]
        if ref:
            clone += ["--branch", ref, "--single-branch"]
        clone += [url, str(dest)]
        _run_git(clone)

    if not _is_target_root(dest):
        raise FetchError(
            f"fetched {url} into {dest} but it is not a target package "
            f"(missing contracts/target_contract.yaml)"
        )
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-target-fetch",
        description="Fetch a published <target>-mlir OOT repo into out/build/generated/<target>/.",
    )
    ap.add_argument("target", help="target name (its repo is <target>-mlir by convention)")
    ap.add_argument("--champion", help="branch/tag to check out (default: the repo's default branch)")
    ap.add_argument("--dest", type=Path, help="destination dir (default: the generated-target home)")
    ap.add_argument("--url", help="explicit repo URL (overrides the template/env)")
    ap.add_argument("--depth", type=int, default=1, help="clone depth (default 1)")
    ap.add_argument("--no-update", action="store_true", help="do not re-fetch an existing clone")
    ap.add_argument("--print-url", action="store_true", help="print the resolved URL and exit")
    args = ap.parse_args(argv)

    if args.print_url:
        print(args.url or repo_url(args.target))
        return 0
    try:
        root = fetch(
            args.target,
            champion=args.champion,
            dest=args.dest,
            url=args.url,
            depth=args.depth,
            update=not args.no_update,
        )
    except FetchError as exc:
        print(f"merlin-target-fetch: {exc}", file=sys.stderr)
        return 1
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
