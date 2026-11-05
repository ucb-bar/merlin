"""Pinned support packages: the one record of which package revision Merlin builds with.

A chip package (a layered support provider, see :mod:`merlin.targetgen.provider_compose`) lives in its
own repository and is not copied into Merlin. Merlin records, for each package, ``{target, repo,
commit, tree}`` and ONE base pin that every package layers on, in the ``pinned_support`` block of
``build_tools/upstreams/target_support.json``. ``merlin-target-fetch <target>`` (or the chip example's
``run.sh setup``) checks a package out at exactly its pinned commit into
``<generated target home>/pinned/<target>@<sha12>/``; with ``MERLIN_TARGET_PATH`` unset, that checkout is
the target's default support. Nothing here names a target: the targets are the manifest's data.

Two rules are enforced from this record:

* **single base** -- every package's ``provider.yaml`` ``extends.commit`` must equal Merlin's base pin.
  Two packages are never built against two revisions of the base; a package composed against another
  revision is refused with the command that fixes it (:func:`check_single_base`);
* **no silent fallback** -- a pinned target that is not fetched is an error naming the setup command
  (:class:`SupportNotFetched`), never a quiet resolution to some other copy.

An installed distribution has no checkout and therefore no record: it selects support only through
``MERLIN_TARGET_PATH``. ``MERLIN_SUPPORT_PINS`` names another manifest (the empty string: none).
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import checkout_root

#: The manifest schema whose ``pinned_support`` block this module reads.
SCHEMA = "merlin.upstream_support_migration.v3"
#: Where a checkout keeps the manifest, relative to its root.
MANIFEST = "build_tools/upstreams/target_support.json"
BLOCK = "pinned_support"
ENV_MANIFEST = "MERLIN_SUPPORT_PINS"
#: The fetched packages' shelf inside the generated-target home. Kept apart from the home's own
#: children so that two fetched revisions of one target never form an ambiguous shelf there.
PINNED_SHELF = "pinned"
SHORT = 12
_PIN_FIELDS = frozenset(
    {"target", "repo", "branch", "commit", "tree", "extends", "example", "in_repo", "core_references"}
)
_HEX = frozenset("0123456789abcdef")


class PinError(ValueError):
    """The pin record is malformed, or a checkout disagrees with it."""


class SingleBaseError(PinError):
    """A selected package was composed against a different revision of the base than Merlin's pin."""


class SupportNotFetched(FileNotFoundError):
    """A pinned target's support is not fetched; the message names the command that fetches it."""


@dataclass(frozen=True)
class Pin:
    """One pinned package revision. ``tree`` is the git tree id of ``commit``, checked after a fetch.

    ``branch`` records where the commit was published; the fetch is by commit, never by branch.
    ``extends`` (packages) names the base target; ``in_repo`` (the base) a tracked vendored copy of
    the same revision; ``example`` the example directory whose ``run.sh setup`` fetches it.
    """

    target: str
    repo: str
    commit: str
    tree: str | None = None
    branch: str | None = None
    extends: str | None = None
    example: str | None = None
    in_repo: str | None = None

    @property
    def short(self) -> str:
        return self.commit[:SHORT]

    @property
    def repo_name(self) -> str:
        """The repository's name (its URL's last component, without ``.git``)."""
        name = self.repo.rstrip("/").rpartition("/")[2]
        return name[: -len(".git")] if name.endswith(".git") else name

    def setup_hint(self) -> str:
        """The command(s) that put this pin in place."""
        fetch = f"`merlin-target-fetch {self.target}`"
        if self.example:
            return f"`{self.example}/run.sh setup` (or {fetch})"
        return fetch


@dataclass(frozen=True)
class Pins:
    base: Pin | None
    packages: dict[str, Pin]
    source: Path | None = None

    def get(self, target: str) -> Pin | None:
        if self.base is not None and self.base.target == target:
            return self.base
        return self.packages.get(target)

    def all(self) -> list[Pin]:
        return ([self.base] if self.base is not None else []) + list(self.packages.values())


NO_PINS = Pins(None, {}, None)


def _sha(value: Any, *, field: str, where: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or not set(value) <= _HEX:
        raise PinError(f"{where}: {field} must be a full lowercase 40-char sha, got {value!r}")
    return value


def _text(value: Any, *, field: str, where: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PinError(f"{where}: {field} must be a non-empty string")
    return value


def _pin(doc: Any, *, where: str) -> Pin:
    if not isinstance(doc, dict):
        raise PinError(f"{where}: a pin is a mapping")
    unknown = set(doc) - _PIN_FIELDS
    if unknown:
        raise PinError(f"{where}: unknown fields {sorted(unknown)}")
    target = _text(doc.get("target"), field="target", where=where)
    assert target is not None
    where = f"{where} ({target})"
    repo = _text(doc.get("repo"), field="repo", where=where)
    assert repo is not None
    if not repo.startswith(("https://", "file://")):
        raise PinError(f"{where}: repo must be an https:// or file:// URL, got {repo!r}")
    tree = doc.get("tree")
    return Pin(
        target=target,
        repo=repo,
        commit=_sha(doc.get("commit"), field="commit", where=where),
        tree=None if tree is None else _sha(tree, field="tree", where=where),
        branch=_text(doc.get("branch"), field="branch", where=where, optional=True),
        extends=_text(doc.get("extends"), field="extends", where=where, optional=True),
        example=_text(doc.get("example"), field="example", where=where, optional=True),
        in_repo=_text(doc.get("in_repo"), field="in_repo", where=where, optional=True),
    )


def parse(document: dict[str, Any], *, source: Path | None = None) -> Pins:
    """Validate a manifest's ``pinned_support`` block (fail closed) and return its pins."""
    where = str(source or "pin manifest")
    block = document.get(BLOCK)
    if block is None:
        return Pins(None, {}, source)
    if document.get("schema") != SCHEMA:
        raise PinError(f"{where}: {BLOCK} needs schema {SCHEMA}, got {document.get('schema')!r}")
    if not isinstance(block, dict):
        raise PinError(f"{where}: {BLOCK} must be a mapping")
    base = None if block.get("base") is None else _pin(block["base"], where=f"{where}: {BLOCK}.base")
    if base is not None and base.extends is not None:
        raise PinError(f"{where}: the base pin cannot extend another target")
    packages: dict[str, Pin] = {}
    entries = block.get("packages") or []
    if not isinstance(entries, list):
        raise PinError(f"{where}: {BLOCK}.packages must be a list")
    for n, entry in enumerate(entries):
        pin = _pin(entry, where=f"{where}: {BLOCK}.packages[{n}]")
        if pin.target in packages or (base is not None and pin.target == base.target):
            raise PinError(f"{where}: {pin.target!r} is pinned twice")
        if pin.in_repo is not None:
            raise PinError(f"{where}: package {pin.target!r} is fetched, never kept in-repo")
        if pin.extends is not None and (base is None or pin.extends != base.target):
            raise PinError(
                f"{where}: package {pin.target!r} extends {pin.extends!r}, but Merlin's single base is "
                f"{base.target if base else 'not pinned'!r}; every package layers on the one base"
            )
        packages[pin.target] = pin
    return Pins(base, packages, source)


def manifest_path() -> Path | None:
    """The manifest in force: ``$MERLIN_SUPPORT_PINS`` (empty: none), else the checkout's own."""
    explicit = os.environ.get(ENV_MANIFEST)
    if explicit is not None:
        return Path(explicit).expanduser() if explicit else None
    checkout = checkout_root()
    return None if checkout is None else checkout / MANIFEST


@functools.lru_cache(maxsize=8)
def _load(path: str, stamp: tuple[int, int]) -> Pins:
    del stamp  # the cache key: a rewritten manifest is read again
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PinError(f"{path}: cannot read the pin manifest: {exc}") from exc
    if not isinstance(document, dict):
        raise PinError(f"{path}: expected a JSON object")
    return parse(document, source=Path(path))


def load_pins(path: str | Path | None = None) -> Pins:
    """The pins of ``path`` (default :func:`manifest_path`); no manifest means no pins."""
    target = Path(path) if path is not None else manifest_path()
    if target is None or not target.is_file():
        return NO_PINS
    stat = target.stat()
    return _load(str(target.resolve()), (stat.st_mtime_ns, stat.st_size))


# ── where a pin lives ────────────────────────────────────────────────────────────────────────────


def pinned_shelf() -> Path:
    """Where fetched packages live: ``<generated target home>/pinned/``."""
    from .target_registry import generated_target_home

    return generated_target_home() / PINNED_SHELF


def pinned_root(pin: Pin) -> Path:
    """The fetch destination of one pin: ``pinned/<target>@<sha12>``."""
    return pinned_shelf() / f"{pin.target}@{pin.short}"


def fetched_dirs() -> tuple[Path, ...]:
    """Every directory on the pinned shelf, whatever revision or state: all of it is support bytes."""
    shelf = pinned_shelf()
    if not shelf.is_dir():
        return ()
    return tuple(sorted(child.resolve() for child in shelf.iterdir() if child.is_dir()))


def checkout_head(root: Path) -> str | None:
    """The commit a fetched checkout has checked out, read from its ``HEAD`` without running git.

    Only a detached HEAD (which is how a fetch leaves it) counts; anything else returns None.
    """
    git = root / ".git"
    head = git / "HEAD" if git.is_dir() else None
    if head is None and git.is_file():
        key, _, value = git.read_text(encoding="utf-8").strip().partition(":")
        if key.strip() == "gitdir" and value.strip():
            head = (root / value.strip()).resolve() / "HEAD"
    if head is None or not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    return text if len(text) == 40 and set(text) <= _HEX else None


def in_repo_commit(root: Path) -> str | None:
    """The source commit a tracked vendored copy records (its sibling ``SOURCE.yaml``)."""
    from .provider_compose import provider_revision

    return provider_revision(root)


def _in_repo_root(pin: Pin) -> Path | None:
    if pin.in_repo is None:
        return None
    checkout = checkout_root()
    root = None if checkout is None else checkout / pin.in_repo
    return root.resolve() if root is not None and root.is_dir() else None


def available_root(pin: Pin) -> Path | None:
    """The checkout that provides ``pin``, or None when it has not been fetched.

    A tracked in-repo copy (the base may keep one) must record exactly the pinned commit; a fetched
    checkout must have exactly the pinned commit checked out. Either disagreeing is an error, not a
    reason to look elsewhere.
    """
    root = _in_repo_root(pin)
    if root is not None:
        recorded = in_repo_commit(root)
        if recorded != pin.commit:
            raise PinError(
                f"{root} records {pin.target}@{(recorded or 'nothing')[:SHORT]} but Merlin pins "
                f"{pin.target}@{pin.short}: the in-repo copy and the pin are one source of truth; re-vendor "
                f"it at {pin.commit} or move the pin"
            )
        return root
    root = pinned_root(pin)
    if not root.is_dir():
        return None
    head = checkout_head(root)
    if head != pin.commit:
        raise PinError(
            f"{root} has {head or 'no detached commit'} checked out, not the pinned {pin.commit}; "
            f"re-run {pin.setup_hint()}"
        )
    return root.resolve()


# ── the two rules ────────────────────────────────────────────────────────────────────────────────


def single_base_message(package: str, base: str, composed: str | None, pinned: str) -> str:
    composed_at = (composed or "an unpinned revision")[:SHORT]
    return (
        f"{package} composed against {base}@{composed_at}; Merlin's base is {base}@{pinned[:SHORT]}: "
        f"recompose {package} against {pinned} and bump its pin"
    )


def check_single_base(provider: Any, pins: Pins | None = None) -> None:
    """Refuse a layered provider whose ``extends.commit`` is not Merlin's base pin.

    Applies to every selected provider that extends the pinned base target, whether fetched, vendored
    or named on ``MERLIN_TARGET_PATH``. A provider extending a target Merlin does not pin is not
    governed by this rule.
    """
    extends = getattr(provider, "extends", None)
    if extends is None:
        return
    pins = pins if pins is not None else load_pins()
    base = pins.base
    if base is None or extends.target != base.target:
        return
    if extends.commit != base.commit:
        package = pins.packages.get(provider.target)
        name = package.repo_name if package is not None else str(provider.id)
        raise SingleBaseError(single_base_message(name, base.target, extends.commit, base.commit))


def not_fetched(pin: Pin) -> SupportNotFetched:
    return SupportNotFetched(
        f"{pin.target!r}: its support package {pin.repo_name}@{pin.short} is pinned but not fetched "
        f"(expected at {pinned_root(pin)}); run {pin.setup_hint()}. MERLIN_TARGET_PATH=<checkout> selects "
        "another copy explicitly"
    )


@dataclass(frozen=True)
class Selection:
    """The default (``MERLIN_TARGET_PATH`` unset) view of the pins: what is present, what is not."""

    roots: dict[str, Path]
    missing: dict[str, Pin]


def default_selection(pins: Pins | None = None) -> Selection:
    """Each pinned target's providing checkout, or its pin when it is absent."""
    pins = pins if pins is not None else load_pins()
    roots: dict[str, Path] = {}
    missing: dict[str, Pin] = {}
    for pin in pins.all():
        root = available_root(pin)
        if root is None:
            missing[pin.target] = pin
        else:
            roots[pin.target] = root
    return Selection(roots, missing)
