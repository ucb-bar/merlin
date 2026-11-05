"""Native hookup to the published ``<target>-mlir`` out-of-tree repos.

A target definition is a self-contained OOT package — a ``<target>-mlir`` repo shipping
``contracts/target_contract.yaml`` (+ optional ``plugin.backend``) exactly as
:func:`merlin.targetgen.capability_manifests.write_oot_target` emits. ``target_registry.resolve``
already picks such a package up with zero env once it sits in the generated-target home
(``out/build/generated/<target>/``). This module is the missing verb: it FETCHES that repo into the
home, so ``merlin-target-fetch <target>`` is all it takes to plug a published target in.

A layered package (``provider.yaml`` with ``extends``, see :mod:`merlin.targetgen.provider_compose`) is
fetched together with its base: :func:`fetch_chain` follows ``extends`` and checks each base out at the
exact commit the layer pins, into the base's own generated-target home.

A target PINNED in the checkout's support manifest (:mod:`merlin.targetgen.support_pins`) is fetched by
COMMIT, never by branch: :func:`fetch_pinned` initializes ``<generated home>/pinned/<target>@<sha12>/``,
fetches exactly the pinned commit (``--depth 1``, falling back to a full fetch when a server refuses a
commit want), checks it out detached, verifies the commit and its tree against the record, and brings in
the package's submodules (a chip package carries its base as one), each verified against Merlin's single
base pin. Re-running it on a verified checkout does nothing and needs no network; a dirty or divergent
checkout is refused, never reset.

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

from .target_registry import _is_target_root, generated_target_home

# The default publish home + naming convention for target repos (the paths edge — an org + a
# ``<target>-mlir`` suffix, parameterized by target; NOT a target-name literal). Overridable so a fork
# or a private mirror plugs in with one env var.
_DEFAULT_URL_TEMPLATE = "https://github.com/ucb-bar/{target}-mlir.git"
_ENV_URL_TEMPLATE = "MERLIN_TARGET_REPO_TEMPLATE"
_ENV_URL_PREFIX = "MERLIN_TARGET_REPO_"  # + <TARGET> for a per-target exact URL
_GIT_TIMEOUT_S = 180
_FETCHED_REF = "refs/merlin/target-fetch"


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
        raise FetchError(f"{_ENV_URL_TEMPLATE} must contain '{{target}}' (got {template!r})")
    return template.format(target=target)


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
        return result.stdout.strip()
    except FileNotFoundError as exc:  # git not installed
        raise FetchError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"git {args[0]} timed out after {_GIT_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"git {args[0]} failed: {(exc.stderr or exc.stdout or '').strip()}") from exc


def _check_contract(commit: str, dest: Path) -> None:
    member = _run_git(["ls-tree", commit, "--", "contracts/target_contract.yaml"], cwd=dest)
    if not member.startswith(("100644 blob ", "100755 blob ")):
        raise FetchError("fetched commit lacks an ordinary contracts/target_contract.yaml")


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

    ``champion`` selects a branch or tag (default: the remote's default HEAD). ``update``
    adopts the exact fetched commit with a non-forced detached checkout, never resetting a local
    branch. Dirty or locally divergent checkouts refuse; set update False to keep a local
    clone untouched. Raises :class:`FetchError` on any git failure or if the result is not a target
    package (no ``contracts/target_contract.yaml``).
    """
    url = url or repo_url(target)
    dest = Path(dest) if dest is not None else generated_target_home() / target
    ref = champion  # a branch or tag; None → clone the default branch
    if ref is not None:
        if not ref or ref.startswith("-") or "\x00" in ref:
            raise FetchError("champion must be a branch or tag name, not a Git option")
        _run_git(["check-ref-format", "--allow-onelevel", ref])

    if (dest / ".git").is_dir():
        if update:
            if _run_git(["remote", "get-url", "origin"], cwd=dest) != url:
                raise FetchError("existing target origin differs from the requested URL; use a fresh destination")
            if _run_git(["status", "--porcelain", "--untracked-files=all"], cwd=dest):
                raise FetchError("target checkout has local changes; preserve them before updating")
            known = _run_git(
                ["for-each-ref", "--format=%(objectname)", "refs/remotes/origin", _FETCHED_REF], cwd=dest
            ).splitlines()
            if not known or _run_git(["rev-list", "HEAD", "--not", *known], cwd=dest):
                raise FetchError("target checkout has local or unverified history; use a fresh destination")
            _run_git(["fetch", "--depth", str(depth), "origin", ref or "HEAD"], cwd=dest)
            commit = _run_git(["rev-parse", "--verify", "FETCH_HEAD^{commit}"], cwd=dest)
            _check_contract(commit, dest)
            _run_git(["checkout", "--detach", "--no-overwrite-ignore", "-q", commit], cwd=dest)
            _run_git(["update-ref", _FETCHED_REF, commit], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone = ["clone", "--depth", str(depth)]
        if ref:
            clone += ["--branch", ref, "--single-branch"]
        clone += ["--", url, str(dest)]
        _run_git(clone)
        commit = _run_git(["rev-parse", "--verify", "HEAD^{commit}"], cwd=dest)
        _check_contract(commit, dest)
        _run_git(["update-ref", _FETCHED_REF, commit], cwd=dest)

    if not _is_target_root(dest):
        raise FetchError(
            f"fetched {url} into {dest} but it is not a target package (missing contracts/target_contract.yaml)"
        )
    return dest


# ── pinned packages: fetch by commit ─────────────────────────────────────────────────────────────


def _git_ok(args: list[str], *, cwd: Path) -> bool:
    try:
        _run_git(args, cwd=cwd)
        return True
    except FetchError:
        return False


def _file_transport(url: str) -> list[str]:
    """Git refuses the file transport for submodules unless asked; a local URL is a deliberate choice."""
    return ["-c", "protocol.file.allow=always"] if url.startswith(("file://", "/")) else []


def _head(dest: Path) -> str:
    """The checked-out commit, or "" for a repository with none yet."""
    args = ["rev-parse", "--verify", "-q", "HEAD^{commit}"]
    return _run_git(args, cwd=dest) if _git_ok(args, cwd=dest) else ""


def _verify_checkout(dest: Path, commit: str, tree: str | None, *, what: str) -> None:
    head = _head(dest)
    if head != commit:
        raise FetchError(f"{what}: {dest} has {head or 'nothing'} checked out, not the pinned {commit}")
    if tree is not None:
        actual = _run_git(["rev-parse", "--verify", f"{commit}^{{tree}}"], cwd=dest)
        if actual != tree:
            raise FetchError(f"{what}: commit {commit} carries tree {actual}, but the pin records {tree}")
    if _run_git(["status", "--porcelain", "--untracked-files=all"], cwd=dest):
        raise FetchError(f"{what}: {dest} differs from its commit; preserve those changes and remove it")


def fetch_commit(url: str, commit: str, dest: Path, *, tree: str | None = None, what: str = "fetch") -> Path:
    """Check out exactly ``commit`` of ``url`` at ``dest``: idempotent, verified, never destructive.

    A fresh ``dest`` is initialized and only that commit is fetched (``--depth 1``; a server that
    refuses a commit want gets one full fetch instead). An existing checkout already at the commit and
    clean is verified and left alone (no network). One with local changes, or whose HEAD is not a
    commit this function fetched, is refused rather than moved.
    """
    dest = Path(dest)
    if dest.exists() and not (dest / ".git").exists() and any(dest.iterdir()):
        raise FetchError(f"{what}: {dest} exists and is not a fetched checkout; remove it or choose another")
    if (dest / ".git").exists():
        if _run_git(["status", "--porcelain", "--untracked-files=all"], cwd=dest):
            raise FetchError(f"{what}: {dest} has local changes; preserve them and remove it before fetching")
        head = _head(dest)
        if head == commit:
            _verify_checkout(dest, commit, tree, what=what)
            return dest
        known = _run_git(["for-each-ref", "--format=%(objectname)", _FETCHED_REF], cwd=dest).splitlines()
        if head and head not in known:
            raise FetchError(f"{what}: {dest} is at {head}, which it was not fetched at; remove it and re-fetch")
        if _run_git(["remote", "get-url", "origin"], cwd=dest) != url:
            raise FetchError(f"{what}: {dest} fetches from another origin than {url}; remove it and re-fetch")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        _run_git(["init", "-q"], cwd=dest)
        _run_git(["remote", "add", "origin", url], cwd=dest)
    try:
        _run_git(["fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)
    except FetchError:
        # A server that does not serve arbitrary commit wants: fetch its refs, then require the commit.
        _run_git(["fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"], cwd=dest)
    if not _git_ok(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=dest):
        raise FetchError(f"{what}: {url} does not have commit {commit}")
    _run_git(["checkout", "--detach", "--no-overwrite-ignore", "-q", commit], cwd=dest)
    _run_git(["update-ref", _FETCHED_REF, commit], cwd=dest)
    _verify_checkout(dest, commit, tree, what=what)
    return dest


def submodules(root: Path) -> dict[str, tuple[str, str]]:
    """``{path: (name, gitlink commit)}`` for every submodule ``root``'s HEAD records."""
    if not (root / ".gitmodules").is_file():
        return {}
    listing = _run_git(["config", "--file", ".gitmodules", "--list"], cwd=root).splitlines()
    names: dict[str, str] = {}
    for line in listing:
        key, _, value = line.partition("=")
        head, dot, field = key.rpartition(".")
        prefix, _, name = head.partition(".")
        if dot and prefix == "submodule" and field == "path" and name:
            names[value] = name
    out: dict[str, tuple[str, str]] = {}
    for path, name in names.items():
        meta, _, listed = _run_git(["ls-tree", "HEAD", "--", path], cwd=root).partition("\t")
        fields = meta.split()
        if len(fields) == 3 and fields[1] == "commit" and listed == path:
            out[path] = (name, fields[2])
    return out


def _declares(root: Path, target: str) -> bool:
    from .providers import ProviderError, read_provider

    if not (root / "provider.yaml").is_file() and not (root / "contracts" / "target_contract.yaml").is_file():
        return False
    try:
        found = read_provider(root, materialized=False)
    except ProviderError:
        return False
    return found is not None and found.target == target


def fetch_submodules(root: Path, pins, *, base_url: str | None = None, what: str = "fetch") -> list[Path]:
    """Bring in ``root``'s submodules at the commits it records, each verified.

    A submodule that is a support provider for the pinned base target must be recorded at Merlin's base
    pin (and carry its tree): one base per build, never two. Its URL may be redirected (``base_url``,
    else the base target's ``MERLIN_TARGET_REPO_<TARGET>``) to fetch the same commit from a mirror.
    """
    from .support_pins import single_base_message

    base = pins.base
    fetched: list[Path] = []
    for path, (name, gitlink) in sorted(submodules(root).items()):
        sub = root / path

        def refuse_other_base(at: str, path: str = path) -> None:
            if base is not None and at != base.commit:
                raise FetchError(
                    single_base_message(what.partition("@")[0], base.target, at, base.commit)
                    + f" (its submodule {path} records {at})"
                )

        if base is not None and _declares(sub, base.target):
            refuse_other_base(gitlink)  # an already checked-out base at another revision: refuse early
        url = base_url or (os.environ.get(_env_key_for(base.target)) if base is not None else None)
        _run_git(["submodule", "init", "--", path], cwd=root)
        if url:
            _run_git(["config", f"submodule.{name}.url", url], cwd=root)
        configured = _run_git(["config", f"submodule.{name}.url"], cwd=root)
        transport = _file_transport(configured)
        try:
            _run_git([*transport, "submodule", "update", "--depth", "1", "--", path], cwd=root)
        except FetchError:
            _run_git([*transport, "submodule", "update", "--", path], cwd=root)
        if _head(sub) != gitlink:
            raise FetchError(f"{what}: submodule {path} is at {_head(sub)}, not the recorded {gitlink}")
        if base is not None and _declares(sub, base.target):
            refuse_other_base(gitlink)
            _verify_checkout(sub, base.commit, base.tree, what=f"{what} submodule {path}")
        fetched.append(sub)
    if _run_git(["status", "--porcelain", "--untracked-files=all"], cwd=root):
        raise FetchError(f"{what}: {root} differs from its commit after the submodule update")
    return fetched


def fetch_pinned(
    target: str,
    *,
    url: str | None = None,
    base_url: str | None = None,
    dest: Path | None = None,
    pins=None,
    with_base: bool = True,
) -> list[Path]:
    """Fetch a pinned target at exactly its pinned commit; returns the roots (the package first).

    ``url`` overrides where the commit is fetched from (a mirror, a local ``file://`` clone); the
    commit and tree are verified either way. A package is refused unless its ``provider.yaml``
    ``extends.commit`` equals Merlin's base pin. Its submodules are fetched and verified; with
    ``with_base``, the base pin is also made available (a tracked in-repo copy at the pin counts).
    """
    from .providers import ProviderError, read_provider
    from .support_pins import PinError, available_root, check_single_base, load_pins, pinned_root

    pins = pins if pins is not None else load_pins()
    pin = pins.get(target)
    if pin is None:
        raise FetchError(f"{target!r} is not pinned in {pins.source or 'any support manifest'}")
    what = f"{pin.repo_name}@{pin.short}"
    source = url or os.environ.get(_env_key_for(target)) or pin.repo
    root = fetch_commit(source, pin.commit, Path(dest) if dest else pinned_root(pin), tree=pin.tree, what=what)
    try:
        provider = read_provider(root, materialized=False)
    except ProviderError as exc:
        raise FetchError(f"{what}: {exc}") from exc
    if provider is None or provider.target != target:
        raise FetchError(f"{what}: {root} does not declare a support provider for {target!r}")
    if pin.extends is not None and (provider.extends is None or provider.extends.target != pin.extends):
        raise FetchError(f"{what}: the pin says it extends {pin.extends!r}; its provider.yaml does not")
    try:
        check_single_base(provider, pins)
    except PinError as exc:
        raise FetchError(str(exc)) from exc
    roots = [root, *fetch_submodules(root, pins, base_url=base_url, what=what)]
    if with_base and pin.extends is not None and pins.base is not None:
        try:
            present = available_root(pins.base)
        except PinError as exc:
            raise FetchError(str(exc)) from exc
        if present is None:
            present = fetch_pinned(pins.base.target, url=base_url, pins=pins, with_base=False)[0]
        roots.append(present)
    return roots


def _pin_commit(dest: Path, commit: str) -> None:
    """Move a fetched checkout to ``commit`` (detached), fetching just that commit if it is absent."""
    if _run_git(["rev-parse", "HEAD"], cwd=dest) == commit:
        return
    try:
        _run_git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=dest)
    except FetchError:
        _run_git(["fetch", "--depth", "1", "origin", commit], cwd=dest)
    _check_contract(commit, dest)
    _run_git(["checkout", "--detach", "--no-overwrite-ignore", "-q", commit], cwd=dest)
    _run_git(["update-ref", _FETCHED_REF, commit], cwd=dest)


def fetch_chain(
    target: str,
    *,
    champion: str | None = None,
    dest: Path | None = None,
    url: str | None = None,
    depth: int = 1,
    update: bool = True,
    max_depth: int = 8,
) -> list[Path]:
    """Fetch ``target`` and, following ``provider.yaml`` ``extends``, every base it is layered on.

    Returns the roots, the requested package first. A base is cloned from the ``repo`` its layer names
    (else the URL template) into its own generated-target home and checked out at the layer's pinned
    ``commit`` when one is declared. A cycle or an over-long chain raises :class:`FetchError`.
    """
    from .providers import ProviderError, read_provider

    roots = [fetch(target, champion=champion, dest=dest, url=url, depth=depth, update=update)]
    seen = {target}
    while True:
        try:
            layer = read_provider(roots[-1], materialized=False)
        except ProviderError as exc:
            raise FetchError(str(exc)) from exc
        if layer is None or layer.extends is None:
            return roots
        base = layer.extends
        if base.target in seen or len(roots) > max_depth:
            raise FetchError(f"extends chain from {target!r} loops or exceeds {max_depth}: {sorted(seen)}")
        seen.add(base.target)
        root = fetch(base.target, url=base.repo or repo_url(base.target), depth=depth, update=update)
        if base.commit is not None:
            _pin_commit(root, base.commit)
        roots.append(root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-target-fetch",
        description=(
            "Fetch a published <target>-mlir OOT repo into out/build/generated/<target>/; a target pinned in "
            "the support manifest is fetched at exactly its pinned commit into out/build/generated/pinned/."
        ),
    )
    ap.add_argument("target", help="target name (its repo is <target>-mlir by convention)")
    ap.add_argument("--champion", help="branch/tag to check out (default: the repo's default branch)")
    ap.add_argument("--dest", type=Path, help="destination dir (default: the generated-target home)")
    ap.add_argument("--url", help="explicit repo URL (overrides the template/env)")
    ap.add_argument("--depth", type=int, default=1, help="clone depth (default 1)")
    ap.add_argument("--no-update", action="store_true", help="do not re-fetch an existing clone")
    ap.add_argument("--print-url", action="store_true", help="print the resolved URL and exit")
    ap.add_argument("--no-base", action="store_true", help="do not follow provider.yaml extends to fetch bases")
    ap.add_argument("--base-url", help="pinned targets: fetch the base (and base submodules) from this URL")
    ap.add_argument(
        "--root",
        action="store_true",
        help="print the target's selected support root (fetching nothing); fail, naming the setup command, if absent",
    )
    args = ap.parse_args(argv)

    from .support_pins import PinError, load_pins

    if args.root:
        from .plugins import PluginError, resolve_support

        try:
            print(resolve_support(args.target).base)
        except (PluginError, PinError, FileNotFoundError) as exc:
            print(f"merlin-target-fetch: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        pins = load_pins()
    except PinError as exc:
        print(f"merlin-target-fetch: {exc}", file=sys.stderr)
        return 1
    pin = pins.get(args.target)
    if args.print_url:
        print(args.url or (pin.repo if pin else repo_url(args.target)))
        return 0
    if pin is not None:
        if args.champion or args.depth != 1 or args.no_update:
            print("merlin-target-fetch: a pinned target is fetched at its pinned commit only", file=sys.stderr)
            return 2
        try:
            roots = fetch_pinned(
                args.target, url=args.url, base_url=args.base_url, dest=args.dest, pins=pins, with_base=not args.no_base
            )
        except (FetchError, PinError, FileNotFoundError) as exc:
            print(f"merlin-target-fetch: {exc}", file=sys.stderr)
            return 1
        for root in roots:
            print(root)
        return 0
    kwargs = dict(champion=args.champion, dest=args.dest, url=args.url, depth=args.depth, update=not args.no_update)
    try:
        roots = [fetch(args.target, **kwargs)] if args.no_base else fetch_chain(args.target, **kwargs)
    except FetchError as exc:
        print(f"merlin-target-fetch: {exc}", file=sys.stderr)
        return 1
    for root in roots:
        print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
