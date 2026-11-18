"""Publish preserved target payloads with separate, explicitly scoped provenance.

Candidate compiler ABI/tool/build declarations must already exist. Host schedules
remain data; support packages and unknown roles are not compiler publications.
No compiler wrappers, commands, manifests or build recipes are synthesized.

Source files stay byte-identical at the export root. Only reserved .merlin/
metadata and MERLIN_PUBLICATION.md are added. Exact source and exported inventories
are recorded separately; source certification never certifies the expanded export.
Build verification uses a retained copy and establishes only build-artifact presence.
Remotes are explicitly resolved; default invocation is non-executing dry-run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common import paths
from merlin.common.artifacts import git_sha7, new_product, utc_stamp
from merlin.common.digest import canonical_sha256, write_pretty_json
from merlin.common.record_formats import dump_yaml, load_yaml
from merlin.development.targetgen.packaging import records as package_records

# ---- publish -----------------------------------------------------------------------------------


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

    Same safety posture as :func:`publish`: dry-run by default, and a non-local remote needs an
    explicit ``confirm_push`` fingerprint.
    """
    from merlin.development.targetgen.packaging.champion_selection import (
        _fresh_destination,
        _new_stage_root,
        _targets_root,
        resolve_remote,
    )
    from merlin.development.targetgen.packaging.publication_trees import (
        _checkout_branch,
        _fingerprint,
        _git,
        _git_sha_full,
        _head_fingerprint,
        _sync_tree,
        assemble_index_tree,
    )

    resolved = remote or resolve_remote(target, config=config)
    stage_root = _new_stage_root(
        target, sources=(_targets_root(artifacts_root) / target, paths.repo_root() / "LICENSE"), index=True
    )
    repo_dir = stage_root / "repo"
    actions = [f"remote: {resolved}", f"branch: {branch}"]

    clone_dir = stage_root / "clone"
    remote_branches: set[str] | None = None
    if not dry_run:
        clone_dir = _fresh_destination(clone_dir, sources=(_targets_root(artifacts_root) / target,))
        _git(["clone", resolved, str(clone_dir)])
        remote_branches = {
            b.strip().removeprefix("origin/")
            for b in _git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
            if b.strip().startswith("origin/") and "->" not in b
        }
        actions.append(f"remote branches: {sorted(remote_branches)}")

    info = assemble_index_tree(target, repo_dir, artifacts_root=artifacts_root, only_branches=remote_branches)
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

    fingerprint = _fingerprint(
        f"{target}-index",
        _git_sha_full(),
        ",".join(e["branch"] for e in info["entries"]),
        package_records.payload_inventory(repo_dir)["sha256"],
    )
    res["fingerprint"] = fingerprint
    if _needs_push_confirmation(resolved) and confirm_push != fingerprint:
        actions.append(f"REFUSED push to non-local remote; re-run with --confirm-push {fingerprint}")
        res["noop"] = True
        return res

    _checkout_branch(clone_dir, branch)
    if _head_fingerprint(clone_dir) == fingerprint:
        actions.append("no-op: remote index already matches")
        res["noop"] = True
        return res
    _sync_tree(clone_dir, repo_dir)
    _git(["-C", str(clone_dir), "add", "--", "."])
    _git(
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
                f"Merlin-Sha: {_git_sha_full()}\n\n"
                f"Merlin-Publish-Fingerprint: {fingerprint}\n"
            ),
        ]
    )
    _git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    res["commit_sha"] = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()
    actions.append(f"pushed {res['commit_sha']} to {branch}")
    return res


def _needs_push_confirmation(remote: str) -> bool:
    """True for a NON-LOCAL remote (a real network push — git@…, https://…, ssh://…). Local/file
    remotes (``file://…``, an absolute/relative path, an existing bare dir — the verification + test
    path) never need confirmation, so those flows are unchanged."""
    r = remote.strip()
    if r.startswith("file://") or r.startswith(("/", "./", "../", "~")):
        return False
    try:
        if Path(r).exists():
            return False
    except OSError:
        pass
    return r.startswith(("git@", "ssh://", "https://", "http://")) or ":" in r


def _require_push_confirmation(
    remote: str, repo_dir: Path, branch: str, fingerprint: str, confirm_push: str | None
) -> None:
    """Human gate before a real GitHub/network push: refuse unless ``confirm_push`` equals THIS publish's
    stable release fingerprint. Because the fingerprint is derived from the selected payload and
    publication record, a blind constant cannot pass — the operator must have seen the assembled
    artifact. The generated timestamp is not covered. On refusal, print the assembled repo tree (what would
    be pushed) so it can be inspected, then raise. Local/file remotes are exempt (see
    :func:`_needs_push_confirmation`)."""
    from merlin.development.targetgen.packaging.champion_selection import PublishError

    if not _needs_push_confirmation(remote) or confirm_push == fingerprint:
        return
    files = sorted(str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*") if p.is_file())
    tree = "\n".join(f"    {f}" for f in files) or "    (empty)"
    raise PublishError(
        f"push to non-local remote {remote} (branch {branch}) REFUSED without confirmation.\n"
        f"  Assembled repo tree that WOULD be pushed (inspect at {repo_dir}):\n{tree}\n"
        f"  Re-run with --confirm-push {fingerprint} (CLI) / confirm_push={fingerprint!r} (API) to push.\n"
        f"  The token must equal this publish's content fingerprint, so it cannot be passed blindly."
    )


@dataclass
class PublishResult:
    """Outcome of a :func:`publish` invocation (dry-run or real)."""

    target: str
    package_id: str
    remote: str
    dry_run: bool
    gate_ok: bool
    gate_detail: str
    fingerprint: str
    tag: str
    repo_dir: Path
    branch: str = ""
    committed: bool = False
    noop: bool = False
    commit_sha: str | None = None
    product_dir: Path | None = None
    export_identity_path: Path | None = None
    build_verification_path: Path | None = None
    actions: list[str] = field(default_factory=list)


def publish(
    target: str,
    *,
    dry_run: bool = True,
    remote: str | None = None,
    gate: bool = True,
    verify_build: bool = True,
    build_timeout: int = 1800,
    package_id: str | None = None,
    artifacts_root: str | Path | None = None,
    config: str | Path | None = None,
    branch: str | None = None,
    confirm_push: str | None = None,
) -> PublishResult:
    """Publish the champion of ``target`` as its own repo. Dry-run by default (no git/network).

    The gate refuses an uncertified champion unless ``gate=False`` (a loud warning is emitted).
    A real publish clones the resolved remote, checks out the resolved BRANCH (the frozen baseline ->
    ``baseline``; a champion -> ``stable/<package_id>``; overridable), replaces its tree with the
    assembled repo, commits (message = provenance), tags ``v<version>-<package_id>``, and pushes to
    ``refs/heads/<branch>`` — idempotently PER BRANCH. Each real publish event is recorded via
    :func:`merlin.common.artifacts.new_product`."""
    from merlin.development.targetgen.packaging.champion_selection import (
        LAYOUT_VERSION,
        PublishError,
        _new_stage_root,
        resolve_branch,
        resolve_remote,
        select_champion,
    )
    from merlin.development.targetgen.packaging.publication_trees import (
        _cert_run_id,
        _check_gate,
        _export_role,
        _fingerprint,
        _git_publish,
        _git_sha_full,
        assemble_repo_tree,
        embed_provenance,
    )

    if type(build_timeout) is not int or build_timeout <= 0:
        raise PublishError("build_timeout must be a positive integer")
    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    resolved_remote = resolve_remote(target, config=config, override=remote)
    resolved_branch = resolve_branch(sel, override=branch, config=config)
    gate_ok, gate_detail = _check_gate(sel)
    if gate and not gate_ok:
        raise PublishError(f"publish gate refused for {target}/{sel.package_id}: {gate_detail}")
    if not gate and not gate_ok:
        sys.stderr.write(f"WARNING: --no-gate publishing UNCERTIFIED {target}/{sel.package_id}: {gate_detail}\n")

    stage_root = _new_stage_root(target, sources=(sel.package_dir,))
    repo_dir = stage_root / "repo"
    source_payload = package_records.payload_inventory(sel.package_dir)
    manifest = assemble_repo_tree(sel, repo_dir, layout_version=LAYOUT_VERSION)
    embed_provenance(repo_dir, sel)
    if package_records.payload_inventory(sel.package_dir) != source_payload:
        raise PublishError("package inputs changed during export")
    if gate and not _check_gate(sel)[0]:
        raise PublishError("source certification changed during export")
    export_identity_path = stage_root / "export_identity.json"
    write_pretty_json(
        export_identity_path,
        {
            "version": 1,
            "target": target,
            "package_slot": sel.package_id,
            "compiler_package_id": manifest.get("package_id"),
            "source_payload": source_payload,
            "source_input_certification": "pass" if gate_ok else "unverified",
            "scope": "package-payload",
            "external_dependency_closure": "not-attested",
            "exported_payload": package_records.payload_inventory(repo_dir),
            "transformation": {
                "layout_version": LAYOUT_VERSION,
                "kind": "additive-publication-metadata",
                "provider_role": _export_role(sel).value,
                "preserved_source_layout": True,
            },
            "exported_payload_certification": "unverified",
        },
    )

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    fingerprint = _fingerprint(
        sel.package_id,
        merlin_sha,
        cert_run,
        source_payload["sha256"],
        publication_record=sel.publication_record,
    )
    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    result = PublishResult(
        target=target,
        package_id=sel.package_id,
        remote=resolved_remote,
        dry_run=dry_run,
        gate_ok=gate_ok,
        gate_detail=gate_detail,
        fingerprint=fingerprint,
        tag=tag,
        repo_dir=repo_dir,
        export_identity_path=export_identity_path,
    )
    result.branch = resolved_branch
    result.actions = [
        f"select champion {sel.package_id} (family={sel.family}, status={sel.status})",
        f"assemble {sel.layout_kind} repo tree at {repo_dir}",
        f"gate: {'OK' if gate_ok else 'FAILED'} ({gate_detail})",
        f"remote: {resolved_remote}",
        f"branch: {resolved_branch}",
        f"tag: {tag}",
        f"fingerprint: {fingerprint}",
    ]

    if dry_run:
        write_pretty_json(
            stage_root / "build_verification.json",
            {
                "version": 1,
                "status": "not-run",
                "reason": "dry-run",
                "requested": verify_build,
            },
        )
        result.build_verification_path = stage_root / "build_verification.json"
        result.actions.insert(0, "DRY-RUN (no clone/commit/push)")
        return result

    result.build_verification_path = stage_root / "build_verification.json"
    if _export_role(sel).value == "candidate_compiler" and verify_build:
        from merlin.development.targetgen.packaging.records import verify_export

        verification = verify_export(repo_dir, stage_root, timeout=build_timeout)
        if verification["status"] != "pass":
            raise PublishError(f"build verification failed; evidence: {result.build_verification_path}")
        result.actions.append(
            "build-artifact presence verified on retained copy; numerical certification not performed"
        )
    else:
        write_pretty_json(
            result.build_verification_path,
            {
                "version": 1,
                "status": "not-run",
                "reason": "not-requested" if not verify_build else "host-schedule",
            },
        )

    # A build check can take a long time. Its selected input record and payload
    # must still be the bytes named by the confirmation token before any push.
    try:
        if (
            package_records.payload_inventory(sel.package_dir) != source_payload
            or package_records.read_record(sel.package_dir) != sel.publication_record
        ):
            raise PublishError("selected publication inputs changed before push")
    except (OSError, ValueError) as exc:
        raise PublishError(f"selected publication inputs became invalid before push: {exc}") from exc

    # human diff-confirm gate before any real network push (local/file remotes are exempt).
    _require_push_confirmation(resolved_remote, repo_dir, resolved_branch, fingerprint, confirm_push)
    result.actions.append(
        f"push confirmed for non-local remote (fingerprint {fingerprint})"
        if _needs_push_confirmation(resolved_remote)
        else "local remote (no confirm)"
    )

    commit_sha, published_tag, noop = _git_publish(
        resolved_remote, repo_dir, sel, manifest, fingerprint, cert_run, stage_root, resolved_branch
    )
    result.committed = not noop
    result.noop = noop
    result.commit_sha = commit_sha
    result.tag = published_tag
    result.actions.append(
        "no-op (remote branch fingerprint already matches)"
        if noop
        else f"committed {commit_sha} + tag {published_tag}, pushed to remote"
    )

    # record the publish event as a versioned product
    prod = new_product(
        "publish",
        version=1,
        target=target,
        notes=f"publish {sel.package_id} -> {resolved_remote} ({'noop' if noop else 'committed'})",
    )
    event = {
        "target": target,
        "package_id": sel.package_id,
        "remote": resolved_remote,
        "branch": resolved_branch,
        "commit_sha": commit_sha,
        "tag": published_tag,
        "noop": noop,
        "fingerprint": fingerprint,
        "merlin_git_sha": merlin_sha,
        "cert_run": cert_run,
        "gate_ok": gate_ok,
        "gate_detail": gate_detail,
        "actions": result.actions,
    }
    out = prod.add_artifact("publish_event.yaml")
    out.write_text(dump_yaml(event), encoding="utf-8")
    prod.write_manifest()
    result.product_dir = prod.path
    return result


def _print_result(res: PublishResult | dict[str, Any]) -> None:
    # `index` returns a plain dict (it publishes a landing page, not a champion package), so it
    # has no package_id/tag. Print the fields it does carry instead of crashing on the ones it
    # does not -- the push had already succeeded when this raised, which is the worst kind of
    # failure to report: a real error message about work that actually completed.
    if isinstance(res, dict):
        print(f"target={res.get('target')} branch={res.get('branch')}")
        print(f"remote={res.get('remote')}")
        print(f"dry_run={res.get('dry_run')} noop={res.get('noop')}")
        if res.get("commit_sha"):
            print(f"commit={res['commit_sha']}")
        print(f"listed {len(res.get('entries') or [])} branch(es):")
        for e in res.get("entries") or []:
            print(f"  - {e['branch']}  ({e['package_id']}, {e['dtype']}, {e['status']})")
        for a in res.get("actions") or []:
            print(f"  - {a}")
        return
    print(f"target={res.target} package={res.package_id}")
    print(f"remote={res.remote} branch={res.branch}")
    print(f"dry_run={res.dry_run} committed={res.committed} noop={res.noop}")
    if res.commit_sha:
        print(f"commit={res.commit_sha} tag={res.tag}")
    print(f"fingerprint={res.fingerprint}")
    for a in res.actions:
        print(f"  - {a}")
    if res.export_identity_path:
        print(f"export identity: {res.export_identity_path}")
    if res.build_verification_path:
        print(f"build verification: {res.build_verification_path}")
    if res.product_dir:
        print(f"event recorded at {res.product_dir}")


def main(argv: list[str] | None = None) -> int:
    from merlin.development.targetgen.packaging.champion_selection import (
        PublishError,
        resolve_branch,
        resolve_remote,
        select_champion,
    )
    from merlin.development.targetgen.packaging.publication_trees import (
        _cert_run_id,
        _check_gate,
        materialize_package,
        promote,
        record_certification,
    )

    ap = argparse.ArgumentParser(
        prog="merlin-target-publish", description="Publish a target's certified champion as its own repo (WS-E)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish", help="publish the champion as a standalone repo")
    p_pub.add_argument("--target", required=True)
    p_pub.add_argument("--champion", help="explicit package_id (else the selection rules apply)")
    p_pub.add_argument("--remote", help="override the resolved remote")
    p_pub.add_argument("--branch", help="override the resolved branch (default: baseline | stable/<pkg>)")
    p_pub.add_argument("--config", help="override the publish.yaml config path")
    p_pub.add_argument("--artifacts-root", help="override the out/artifacts root (targets under it)")
    p_pub.add_argument("--dry-run", action="store_true", help="plan only (default)")
    p_pub.add_argument("--execute", action="store_true", help="actually clone/commit/push")
    p_pub.add_argument("--no-gate", action="store_true", help="publish even if uncertified (LOUD warning)")
    p_pub.add_argument("--no-verify-build", action="store_true", help="explicitly skip build-artifact verification")
    p_pub.add_argument("--build-timeout", type=int, default=1800, help="timeout in seconds per declared build step")
    p_pub.add_argument(
        "--confirm-push",
        help="content fingerprint confirming a real push to a non-local "
        "remote (printed when refused); required for a GitHub/network push",
    )

    p_prom = sub.add_parser("promote", help="mark a package the single champion for a target")
    p_prom.add_argument("--target", required=True)
    p_prom.add_argument("--champion", "--package", dest="champion", required=True)
    p_prom.add_argument("--artifacts-root")
    p_prom.add_argument("--no-gate", action="store_true")

    p_cert = sub.add_parser("record-cert", help="record certification verdicts outside the package payload")
    p_cert.add_argument("--target", required=True)
    p_cert.add_argument("--champion", "--package", dest="champion", required=True)
    p_cert.add_argument(
        "--results", nargs="+", required=True, help="one or more certify run dirs (or results.yaml paths)"
    )
    p_cert.add_argument("--artifacts-root")
    p_mat = sub.add_parser("materialize", help="install a run's submission as the target's OOT backend package")
    p_mat.add_argument("--target", required=True)
    p_mat.add_argument(
        "--from",
        dest="source",
        required=True,
        help="the run's submission/ directory (manifest.yaml + the backend tree)",
    )
    p_mat.add_argument("--package-id", default="agent_spec_v1_mlir_oot")
    p_mat.add_argument("--certified-by-run", default="")
    p_mat.add_argument("--score", help="score_capsule.json justifying the install (checked, not trusted)")
    p_mat.add_argument("--artifacts-root")
    p_mat.add_argument("--force", action="store_true")

    p_idx = sub.add_parser("index", help="publish the landing page to the repo's default branch")
    p_idx.add_argument("--target", required=True)
    p_idx.add_argument("--remote", help="override the resolved remote")
    p_idx.add_argument("--branch", default="main", help="default branch to write (default: main)")
    p_idx.add_argument("--config", help="override the publish.yaml config path")
    p_idx.add_argument("--artifacts-root")
    p_idx.add_argument("--dry-run", action="store_true", help="plan only (default)")
    p_idx.add_argument("--execute", action="store_true", help="actually clone/commit/push")
    p_idx.add_argument("--confirm-push", help="fingerprint confirming a real push to a non-local remote")

    p_ins = sub.add_parser("inspect", help="show the selected champion + plan (no git)")
    p_ins.add_argument("--target", required=True)
    p_ins.add_argument("--champion")
    p_ins.add_argument("--config")
    p_ins.add_argument("--artifacts-root")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "publish":
            res = publish(
                args.target,
                dry_run=not args.execute,
                remote=args.remote,
                gate=not args.no_gate,
                package_id=args.champion,
                artifacts_root=args.artifacts_root,
                config=args.config,
                branch=args.branch,
                confirm_push=args.confirm_push,
                verify_build=not args.no_verify_build,
                build_timeout=args.build_timeout,
            )
            _print_result(res)
            return 0
        if args.cmd == "index":
            res = publish_index(
                args.target,
                dry_run=not args.execute,
                remote=args.remote,
                config=args.config,
                artifacts_root=args.artifacts_root,
                branch=args.branch,
                confirm_push=args.confirm_push,
            )
            _print_result(res)
            return 0
        if args.cmd == "record-cert":
            pub = record_certification(args.target, args.champion, args.results, artifacts_root=args.artifacts_root)
            tier = pub.get("certification_tier") or {}
            print(
                f"certification={pub.get('certification')} "
                f"rungs={len(pub.get('certified_rungs') or [])} "
                f"derived_from_rtl={tier.get('derived_from_rtl')} "
                f"cycle_accurate={tier.get('cycle_accurate')} "
                f"oracles={','.join(tier.get('oracles') or [])}"
            )
            return 0 if pub.get("certification") == "pass" else 1

        if args.cmd == "materialize":
            dst = materialize_package(
                args.target,
                args.source,
                package_id=args.package_id,
                certified_by_run=args.certified_by_run,
                score_path=args.score,
                artifacts_root=args.artifacts_root,
                force=args.force,
            )
            print(f"installed {args.target} backend -> {dst}")
            return 0
        if args.cmd == "promote":
            promote(args.target, args.champion, gate=not args.no_gate, artifacts_root=args.artifacts_root)
            print(f"promoted {args.target}/{args.champion} to champion")
            return 0
        if args.cmd == "inspect":
            sel = select_champion(args.target, artifacts_root=args.artifacts_root, package_id=args.champion)
            remote = resolve_remote(args.target, config=args.config)
            branch = resolve_branch(sel, config=args.config)
            ok, detail = _check_gate(sel)
            print(f"target={sel.target} champion={sel.package_id}")
            print(f"family={sel.family} layout={sel.layout_kind} status={sel.status}")
            print(f"remote={remote} branch={branch}")
            print(f"gate={'OK' if ok else 'FAILED'} ({detail})")
            print(f"cert_run={_cert_run_id(sel)}")
            return 0
    except PublishError as e:
        sys.stderr.write(f"merlin-target-publish: {e}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
