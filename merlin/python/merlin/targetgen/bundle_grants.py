"""Resolve a bundle grant to the bytes it actually delivers — and say so when it delivers nothing.

A bundle manifest is the *claim* about what an arm can read. It is not the delivery. The two drifted
silently for a long time, in two different ways, and both produced the same failure: an arm credited
with a tool it never carried, in a manifest that read as if it carried it.

  * A grant naming a path that is not there was skipped by the workspace stager without a record.
  * A grant naming a path that IS there only via a symlink out of the repo delivers on the one machine
    the link was made on and dangles on every other clone -- and a dangling entry is skipped the same
    silent way. The check that "the path exists" passes on the author's machine forever.

So resolution is a library function with an explicit, reportable STATUS, used by both the stager (which
delivers) and the gate (which refuses to ship a claim that cannot be delivered). Statuses are ordered by
how much they should worry a reader: ``ok`` < ``derived`` < ``external`` < ``missing``.

The path spelling is the bundle-stager convention, which is itself a fact about the stager, not about
any target: a grant is repo-root-relative, with the documented shorthand that the leading ``merlin/``
may be elided, so the resolver tries the root first and then under ``merlin/``.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

from merlin.common.paths import merlin_dir, repo_root

OK = "ok"                # resolves to bytes inside the repo
DERIVED = "derived"      # absent, but this is a GENERATED artifact the tooling can produce on demand
EXTERNAL = "external"    # resolves only by leaving the repo (a machine-local path); dangles elsewhere
MISSING = "missing"      # resolves to nothing at all

_ORDER = {OK: 0, DERIVED: 1, EXTERNAL: 2, MISSING: 3}


def candidates(path: str) -> list[Path]:
    """The locations the stager will look in, in its own order (repo root, then under ``merlin/``)."""
    rel = path.rstrip("/")
    return [repo_root() / rel, merlin_dir() / rel]


def _link_leaves_repo(link: Path) -> bool:
    """True when a symlink's STORED target text names somewhere outside the repo.

    Judged on the recorded bytes, not on ``resolve()``. The two answer different questions and only the
    first is about committed content: a link stored as an absolute machine path is a portability defect
    wherever it sits, while a link stored repo-relative is portable even when what it finally reaches is
    an external checkout -- that is a SETUP fact about this machine (``.env`` + ``third_party/ext/``),
    reported as setup rather than as a defect in the tree.

    Following the links instead would conflate the two and make the honest, portable spelling look
    exactly as bad as the machine-path one it replaced.
    """
    try:
        target = os.readlink(link)
    except OSError:
        return True
    if os.path.isabs(target):
        return True
    landed = os.path.normpath(os.path.join(str(link.parent), target))
    return not landed.startswith(str(repo_root()) + os.sep)


@functools.lru_cache(maxsize=1)
def tracked_escaping_links() -> dict[str, str]:
    """Every TRACKED symlink in the repo whose target leaves it, as ``{repo-relative path: target}``.

    Tracked is the operative word. An untracked local link is one machine's convenience and ships
    nowhere; a tracked one is committed content that resolves on the machine it was made on and dangles
    on every clone after that -- and the stager skips a dangling entry exactly as silently as an absent
    one. Enumerated from the index rather than by walking the tree, so the answer is both exact and
    cheap (a grant may name a toolchain directory with six figures of files in it).
    """
    import subprocess
    root = repo_root()
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-s"],
                         capture_output=True, text=True, check=True).stdout
    links: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, rel = line.partition("\t")
        if not rel or meta.split(" ", 1)[0] != "120000":
            continue
        src = root / rel
        if _link_leaves_repo(src):
            try:
                links[rel] = os.readlink(src)
            except OSError:
                links[rel] = "<unreadable>"
    return links


def escaping_members(path: str) -> dict[str, str]:
    """The tracked escaping links a granted path DELIVERS -- itself, or anything beneath it."""
    rel = path.rstrip("/")
    prefixes = [rel, f"merlin/{rel}"]
    return {p: t for p, t in tracked_escaping_links().items()
            if any(p == pre or p.startswith(pre + "/") for pre in prefixes)}


def is_generated_shape(path: str) -> bool:
    """True when this grant names an artifact the tooling PRODUCES rather than one the repo stores.

    Recognized by shape alone, so a gate can classify it without paying to build it and without writing
    to a shared tree as a side effect of linting. It is a weaker claim than ``derive_grant`` succeeding
    -- "this is generated and absent", not "this can be produced here" -- and is reported as exactly
    that. Proving producibility is what ``derive=True`` is for.
    """
    parts = Path(path.rstrip("/")).parts
    return len(parts) >= 3 and parts[-1] == "rtl_facts" and parts[-2] == "contracts"


def derive_grant(path: str) -> Path | None:
    """Produce a GENERATED grant that is simply not on disk yet, and return where it now lives.

    Only the shapes this module understands are attempted; anything else returns ``None`` rather than
    guessing. Today that is a target's RTL facts, which are DERIVED from the target's own RTL and
    gitignored on purpose -- committing them would be the wrong fix for their absence.

    It returns the artifact's own location and never writes into the granted path. Populating the grant
    re-points every other consumer (the facts accessor prefers a populated pin over the purgeable
    cache), and merely creating its parent is worse: an empty ``targets/<t>`` shadows the generated
    target home and takes the residual, dialect plan and contract under it out of scope.
    """
    parts = Path(path.rstrip("/")).parts
    if len(parts) < 3 or parts[-1] != "rtl_facts" or parts[-2] != "contracts":
        return None
    try:
        from merlin.targetgen.rtl.facts import ensure_facts
        return ensure_facts(parts[-3]).parent
    except Exception:
        return None


def resolve(path: str, *, derive: bool = True) -> tuple[str, Path | None]:
    """``(status, delivered_path)`` for one granted path.

    Never returns ``EXTERNAL``: that verdict is about whether the committed bytes are portable, which
    is a question about the index rather than about this filesystem, and ``audit`` answers it from
    ``escaping_members``. Deciding it here by following symlinks would also condemn an untracked local
    convenience link -- a fact about one machine's setup, not a defect in the tree.
    """
    for cand in candidates(path):
        if cand.exists():
            return OK, cand
    if derive:
        got = derive_grant(path)
        if got is not None and got.exists():
            return DERIVED, got
        return MISSING, None
    return (DERIVED if is_generated_shape(path) else MISSING), None


def _manifests(te, *, arms: tuple[str, ...] = ()) -> dict[str, dict]:
    """Every manifest to audit for one target: the ladder as generated TODAY, plus every manifest
    MATERIALIZED beside the descriptor.

    Both, because they are different claims. The generated ones say what a run launched now would hand
    each arm. The materialized ones under ``input_bundles/`` are what actually shipped to the runs
    already on record, they are tracked, and they can name grants the generator no longer emits -- a
    variant the generator does not produce (the no-kernel arm of an ablation, say) exists only there.
    Auditing only the generated set would give a clean report while a tracked manifest still promises a
    file that is gone.
    """
    import yaml

    from merlin.targetgen.generate_bundles import generate_bundles

    out: dict[str, dict] = dict(generate_bundles(te, arms=arms))
    for p in sorted((te.path.parent / "input_bundles").glob("*/input_bundle_manifest.yaml")):
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except Exception:  # a malformed manifest is a different gate's business
            continue
        if isinstance(doc, dict):
            out[f"{p.parent.name} (materialized)"] = doc
    return out


def grant_count(te, *, arms: tuple[str, ...] = ()) -> int:
    """How many arm-level grants ``audit`` actually looked at — so a green report states its coverage
    rather than only its verdict. A check that silently reviewed nothing reports success too."""
    return sum(len(m.get("allowed", [])) for m in _manifests(te, arms=arms).values())


def audit(te, *, arms: tuple[str, ...] = (),
          derive: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Every grant of every arm of one target that is not plainly ``ok``, as ``{bundle_id: [(path, status)]}``.

    ``derive`` is off by default: a generated grant is then classified by SHAPE (absent-but-produced-by
    the tooling) rather than by actually producing it, so linting neither pays to build nor writes to a
    shared tree as a side effect. Pass ``derive=True`` to turn that weaker claim into a proof.

    Both the generated ladder and every manifest already materialized beside the descriptor are
    audited -- see ``_manifests`` for why the second set is not redundant.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for bundle_id, manifest in _manifests(te, arms=arms).items():
        bad = []
        for entry in manifest.get("allowed", []):
            status, _ = resolve(entry["path"], derive=derive)
            if status == OK and escaping_members(entry["path"]):
                status = EXTERNAL
            if status != OK:
                bad.append((entry["path"], status))
        if bad:
            out[bundle_id] = sorted(bad, key=lambda ps: (-_ORDER[ps[1]], ps[0]))
    return out
