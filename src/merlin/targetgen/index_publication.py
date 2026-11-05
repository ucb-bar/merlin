"""Publishing a target repo's DEFAULT branch: the landing page, and the base support provider it may carry.

Branch-per-version publishing puts each package on its own branch, so the default branch would be empty;
:func:`publish_index` writes a page listing the branches that exist on the remote. When the default branch
is ALSO the target's support provider -- the base that chip packages layer on with ``extends`` -- its
files are kept and the README explains the layering instead of replacing the provider with a bare page.

Split out of :mod:`merlin.targetgen.publish`, which re-exports these names; shared publishing helpers are
reached through that module (``P.<name>``) so there is one implementation of each.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import publish as P


def base_provider_root(root: str | Path | None, target: str) -> Path | None:
    """``root`` when it is ``target``'s support provider, else None."""
    from .providers import ProviderError, ProviderRole, read_provider

    if root is None:
        return None
    try:
        provider = read_provider(Path(root))
    except ProviderError:
        return None
    if provider is None or provider.role != ProviderRole.SUPPORT or provider.target != target:
        return None
    return Path(root)


def copy_base_provider(base: Path, dest: Path, *, keep_readme: bool) -> bool:
    """Copy the provider's files into ``dest``; the README too when ``keep_readme``. Returns ``keep_readme``."""
    for entry in base.iterdir():
        if entry.name == ".git" or (entry.name == "README.md" and not keep_readme):
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(entry, dest / entry.name)
    return keep_readme and (base / "README.md").is_file()


def layers_over(target: str) -> tuple[str, ...]:
    """Targets whose support providers declare ``extends: target`` (selected, generated or vendored)."""
    from . import target_registry
    from .providers import ProviderError, read_provider

    roots = {*target_registry.in_repo_support().values(), *target_registry.external_targets().values()}
    out = set()
    for root in roots:
        try:
            provider = read_provider(root, materialized=False)
        except ProviderError:
            continue
        if provider is not None and provider.extends is not None and provider.extends.target == target:
            out.add(provider.target)
    return tuple(sorted(out))


def base_layer_intro(target: str, name: str, layers: tuple[str, ...]) -> list[str]:
    """The README opening for a default branch that is ``target``'s base support provider."""
    return [
        f"# {name}",
        "",
        f"The **base `{target}` layer** for Merlin. This branch is the `{target}` support provider "
        "(`provider.yaml`, `contracts/target_contract.yaml`, `tests/`) that Merlin selects for target "
        f"`{target}`. It contains no chip-specific facts and no matrix-unit code.",
        "",
        "Chip packages build on it: their `provider.yaml` declares "
        f"`extends: {{target: {target}, commit: <sha of this repo>}}` and their `contracts/overlay.yaml` "
        "holds only the delta. `merlin-target-compose` turns base + overlay into the effective contract "
        "and records the exact base revision.",
        "",
        "```",
        f"{name}",
        *[f" {'└' if i == len(layers) - 1 else '├'}── {P.resolve_repo_name(layer)}" for i, layer in enumerate(layers)],
        "```",
        "",
        "```sh",
        f"export MERLIN_TARGET_PATH=/path/to/{name}       # or use the copy vendored in a Merlin checkout",
        "merlin-target-fetch <chip>                       # a chip package, plus this base at its pinned commit",
        "```",
        "",
        "See Merlin's `docs/guides/chip_packages.md` for writing a chip package on top of this one.",
        "",
        "Package names on the branches below describe the schedule's MLIR lowering choices "
        "(e.g. a `vector.contract` lowering strategy), not hardware units.",
        "",
    ]


def _index_readme(
    target: str, entries: list[dict[str, Any]], *, base_provider: bool = False, layers: tuple[str, ...] = ()
) -> str:
    """Describe actual published roles without inventing a universal execution path.

    ``base_provider``: the default branch also carries ``target``'s support provider at its root, the base
    that chip packages (``layers``, the targets whose providers declare ``extends: target``) build on.
    """
    name = P.resolve_repo_name(target)
    if base_provider:
        intro = base_layer_intro(target, name, layers)
    else:
        intro = [
            f"# {name}",
            "",
            f"Published artifact branches for **{target}**.",
            "",
            "This is a **branch-per-version** directory, not a compiler package. "
            "Each listed branch preserves its original payload and adds scoped provenance.",
            "",
        ]
    lines = [
        *intro,
        "## Published packages",
        "",
        "| branch | package slot | dtype | source status | role |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['branch']}` | `{entry['package_id']}` | `{entry['dtype']}` | "
            f"`{entry['status']}` | {entry['role']} |"
        )
    lines += [
        "",
        "## Using a branch",
        "",
        "```sh",
        f"git clone -b <branch> <this-repo> {P.resolve_repo_name(target)}",
        f"cd {P.resolve_repo_name(target)}",
        "```",
        "",
        "Read the original manifest and author documentation, then MERLIN_PUBLICATION.md "
        "and .merlin/provenance.yaml for role and evidence scope.",
        "",
        "- **candidate_compiler**: use its actual declared ABI commands and build recipe "
        "with the required environment. Tool presence/build observations do not establish "
        "portable buildability or numerical correctness.",
        "- **host_schedule**: use the target-owned host consumer named by the artifact's "
        "documentation. Schedule data is not a standalone compiler.",
        "- Support packages have a separate target-support workflow and are not compiler releases "
        "listed by this exporter. Cloning a branch does not register it as target support.",
        "",
        "There is no universal model-compilation command for all these roles. "
        "No generated compiler wrapper or inferred ISA/toolchain is supplied.",
        "",
        "## Provenance",
        "",
        "Source observations have scope package-payload and external dependency closure "
        "not-attested. Added publication files and portable execution are not certified by "
        "source evidence. See .merlin/certification.yaml before citing a result. "
        "Historical manifest status alone grants no new qualification.",
        "",
        "Commit/tag messages bind the package slot, run, Merlin revision and exact source "
        "payload inventory. Existing tags remain immutable.",
        "",
        f"Generated from Merlin `{P.git_sha7()}`.",
        "",
    ]
    return "\n".join(lines)


def index_entries(target: str, *, artifacts_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Describe every package that WOULD be published for ``target``, for the landing page.

    Derived from the same selection/branch rules the publish path uses, so the index cannot
    drift from what is actually on the remote.
    """
    tdir = P._targets_root(artifacts_root) / target
    out: list[dict[str, Any]] = []
    for man_path in sorted(tdir.glob("*/manifest.yaml")):
        man = P.load_yaml(man_path)
        if not isinstance(man, dict):
            continue
        sel = P._build_selection(target, man_path.parent, man)
        gate_ok, _ = P._check_gate(sel)
        if not gate_ok:
            continue  # only certified packages are published, so only they are listed
        try:
            provider_role = P._export_role(sel)
        except P.PublishError:
            continue
        is_base = P._is_baseline(sel)
        out.append(
            {
                "branch": P.resolve_branch(sel),
                "package_id": sel.package_id,
                "dtype": P.package_dtype(man_path.parent),
                "status": sel.status or P._cert_phrase(man) or "unknown",
                "role": (
                    f"{provider_role.value}: "
                    + ("frozen unoptimized control" if is_base else "observed package inputs")
                ),
            }
        )
    return sorted(out, key=lambda e: (e["dtype"], e["branch"]))


def assemble_index_tree(
    target: str,
    dest: str | Path,
    *,
    artifacts_root: str | Path | None = None,
    only_branches: set[str] | None = None,
    base_root: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the default-branch landing page (README + LICENSE) into ``dest``.

    Destination must be absent and disjoint from source inputs; existing directories
    (even empty ones) are caller-owned and are never reused.

    Deliberately NOT a package tree: the default branch must not look like one champion, or a
    consumer would build the wrong thing. Reuses the repo's LICENSE so the published repo
    carries the same terms as Merlin.

    ``only_branches`` restricts the listing to branches that ACTUALLY exist on the remote (the
    publish path passes what the clone reports). Without it the page would advertise every
    locally-certified package, including ones never pushed — a fresh clone would then be sent
    to branches that do not exist, which is worse than the empty default branch this replaces.

    ``base_root``: the current default branch when it carries ``target``'s SUPPORT provider at its root.
    Then the default branch IS a package (the base layer chip packages extend), so its files are kept
    and only the README is regenerated. Overwriting it with a bare landing page would delete the base.
    """
    sources = (P._targets_root(artifacts_root) / target, P.paths.repo_root() / "LICENSE")
    dest = P._fresh_destination(dest, sources=sources)
    entries = P.index_entries(target, artifacts_root=artifacts_root)
    if only_branches is not None:
        entries = [e for e in entries if e["branch"] in only_branches]
    dest.mkdir(parents=True)
    base = base_provider_root(base_root, target)
    if base is not None and copy_base_provider(base, dest, keep_readme=not entries):
        pass  # nothing certifiable to list: the base's own README is kept rather than emptied
    else:
        readme = (
            P._index_readme(target, entries, base_provider=True, layers=layers_over(target))
            if base is not None
            else P._index_readme(target, entries)
        )
        P._write(dest / "README.md", readme)
    lic = P.paths.repo_root() / "LICENSE"
    if lic.is_file():
        P._write(dest / "LICENSE", lic.read_text(encoding="utf-8"))
    return {"target": target, "entries": entries, "dest": str(dest)}


def publish_index(
    target: str,
    *,
    dry_run: bool = True,
    remote: str | None = None,
    config: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    branch: str = "main",
    confirm_push: str | None = None,
) -> dict[str, Any]:
    """Publish the landing page to the repo's DEFAULT branch.

    Branch-per-version publishing leaves that branch empty, so `git clone` with no `-b` gives a
    repo containing nothing — the state ucb-bar/rvv-mlir was actually in. This writes a
    directory page listing the branches that exist ON THE REMOTE (read from the clone, so the
    page cannot advertise a branch that is not there) and how to consume a package.

    Same safety posture as :func:`merlin.targetgen.publish.publish`: dry-run by default, and a non-local remote needs an
    explicit ``confirm_push`` fingerprint.
    """
    resolved = remote or P.resolve_remote(target, config=config)
    stage_root = P._new_stage_root(
        target, sources=(P._targets_root(artifacts_root) / target, P.paths.repo_root() / "LICENSE"), index=True
    )
    repo_dir = stage_root / "repo"
    actions = [f"remote: {resolved}", f"branch: {branch}"]

    clone_dir = stage_root / "clone"
    remote_branches: set[str] | None = None
    if not dry_run:
        clone_dir = P._fresh_destination(clone_dir, sources=(P._targets_root(artifacts_root) / target,))
        P._git(["clone", resolved, str(clone_dir)])
        remote_branches = {
            b.strip().removeprefix("origin/")
            for b in P._git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
            if b.strip().startswith("origin/") and "->" not in b
        }
        actions.append(f"remote branches: {sorted(remote_branches)}")

    base_root = None
    if not dry_run:
        P._checkout_branch(clone_dir, branch)
        base_root = clone_dir
    info = P.assemble_index_tree(
        target, repo_dir, artifacts_root=artifacts_root, only_branches=remote_branches, base_root=base_root
    )
    if base_provider_root(base_root, target) is not None:
        actions.append(f"{branch} carries the {target} support provider: kept; README regenerated")
    actions.append(f"assembled index listing {len(info['entries'])} package(s) at {repo_dir}")
    res: dict[str, Any] = {
        "target": target,
        "remote": resolved,
        "branch": branch,
        "entries": info["entries"],
        "repo_dir": str(repo_dir),
        "dry_run": dry_run,
        "actions": actions,
        "noop": False,
    }
    if dry_run:
        actions.append("dry-run: nothing cloned, committed or pushed")
        return res

    fingerprint = P._fingerprint(
        f"{target}-index",
        P._git_sha_full(),
        ",".join(e["branch"] for e in info["entries"]),
        P.package_records.payload_inventory(repo_dir)["sha256"],
    )
    res["fingerprint"] = fingerprint
    if P._needs_push_confirmation(resolved) and confirm_push != fingerprint:
        actions.append(f"REFUSED push to non-local remote; re-run with --confirm-push {fingerprint}")
        res["noop"] = True
        return res

    if P._head_fingerprint(clone_dir) == fingerprint:
        actions.append("no-op: remote index already matches")
        res["noop"] = True
        return res
    P._sync_tree(clone_dir, repo_dir)
    P._git(["-C", str(clone_dir), "add", "--", "."])
    P._git(
        [
            "-C",
            str(clone_dir),
            "-c",
            "user.name=merlin-target-publish",
            "-c",
            "user.email=publish@merlin.local",
            "commit",
            "-m",
            f"docs({target}): landing page for the published package branches",
            "-m",
            (
                f"Lists the branches present on this remote and how to consume a package.\n"
                f"Merlin-Sha: {P._git_sha_full()}\n\n"
                f"Merlin-Publish-Fingerprint: {fingerprint}\n"
            ),
        ]
    )
    P._git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    res["commit_sha"] = P._git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()
    actions.append(f"pushed {res['commit_sha']} to {branch}")
    return res
