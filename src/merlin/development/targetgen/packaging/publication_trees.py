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
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.common import paths
from merlin.common.artifacts import git_sha7, new_product, utc_stamp
from merlin.common.digest import canonical_sha256, write_pretty_json
from merlin.common.record_formats import dump_yaml, load_yaml
from merlin.development.targetgen.packaging import records as package_records
from merlin.development.targetgen.packaging.champion_selection import PublishError

if TYPE_CHECKING:
    from merlin.development.targetgen.packaging.champion_selection import ChampionSelection

# ---- publish -----------------------------------------------------------------------------------


def assemble_index_tree(
    target: str, dest: str | Path, *, artifacts_root: str | Path | None = None, only_branches: set[str] | None = None
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
    """
    from merlin.development.targetgen.packaging.champion_selection import (
        _fresh_destination,
        _index_readme,
        _targets_root,
        _write,
        index_entries,
    )

    dest = _fresh_destination(dest, sources=(_targets_root(artifacts_root) / target, paths.repo_root() / "LICENSE"))
    entries = index_entries(target, artifacts_root=artifacts_root)
    if only_branches is not None:
        entries = [e for e in entries if e["branch"] in only_branches]
    dest.mkdir(parents=True)
    _write(dest / "README.md", _index_readme(target, entries))
    lic = paths.repo_root() / "LICENSE"
    if lic.is_file():
        _write(dest / "LICENSE", lic.read_text(encoding="utf-8"))
    return {"target": target, "entries": entries, "dest": str(dest)}


def _export_role(sel: ChampionSelection):
    """Read declared ownership; never turn support/schedule data into a compiler."""
    from merlin.compiler.backend.providers import ProviderError, ProviderRole, read_provider
    from merlin.development.targetgen.packaging.champion_selection import ChampionSelection, PublishError

    try:
        provider = read_provider(sel.package_dir)
    except ProviderError as exc:
        raise PublishError(str(exc)) from exc
    if provider is None or provider.role == ProviderRole.SUPPORT:
        raise PublishError("publication requires a candidate compiler or host schedule, not support/unknown data")
    if provider.target != sel.target:
        raise PublishError("provider target does not match selected publication target")
    if provider.role == ProviderRole.CANDIDATE_COMPILER:
        try:
            from merlin.development.targetgen.packaging.loading import load_package
        except ModuleNotFoundError as exc:
            if exc.name != "jsonschema":
                raise
            # The ABI is schema-validated; a minimal install refuses rather than skipping that check.
            raise PublishError(
                "candidate compiler manifest validation requires jsonschema (pip install 'merlin[targetgen]')"
            ) from exc

        try:
            pkg = load_package(sel.package_dir)
        except Exception as exc:
            raise PublishError(f"candidate compiler ABI unavailable: {exc}") from exc
        if not pkg.tool.resolve().is_relative_to(sel.package_dir.resolve()):
            raise PublishError("candidate compiler tool must be inside its payload")
        if not pkg.manifest.get("build") and not pkg.tool.is_file():
            raise PublishError("candidate compiler has neither an existing tool nor a build recipe")
    return provider.role


def assemble_repo_tree(
    sel: ChampionSelection, dest: str | Path, *, layout_version: str, champion: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Preserve original payload paths/bytes; append explicitly scoped publication metadata.

    ``champion`` is the evidence summary of a champion that passed its phase profile (see
    :func:`_readme`); only :func:`merlin.development.targetgen.packaging.champions.export_champion` passes one."""
    from merlin.development.targetgen.packaging.champion_selection import (
        ChampionSelection,
        PublishError,
        _fresh_destination,
        _readme,
        _write,
    )

    dest = _fresh_destination(dest, sources=(sel.package_dir,))
    role = _export_role(sel)
    from merlin.development.targetgen.packaging.source_layout import SourceLayoutError, publication_layout

    try:
        source_layout = (
            publication_layout(sel.package_dir, layout_version)
            if role.value == "candidate_compiler"
            else {"layout_version": layout_version, "status": "host_schedule_data"}
        )
    except SourceLayoutError as exc:
        raise PublishError(str(exc)) from exc
    for reserved in (".merlin", "MERLIN_PUBLICATION.md", ".git"):
        if (sel.package_dir / reserved).exists():
            raise PublishError(f"payload collides with reserved publication path: {reserved}")
    identity = package_records.payload_inventory(sel.package_dir)
    shutil.copytree(sel.package_dir, dest)
    if (
        package_records.payload_inventory(dest) != identity
        or package_records.payload_inventory(sel.package_dir) != identity
    ):
        raise PublishError("payload changed during export copy")
    manifest = load_yaml(dest / "manifest.yaml")
    # Host metadata remains outside the byte-identical compiler payload.
    _write(dest / ".merlin/source_layout.json", json.dumps(source_layout, indent=2) + "\n")
    _write(dest / "MERLIN_PUBLICATION.md", _readme(sel, manifest, champion))
    return manifest


def _cert_run_id(sel: ChampionSelection) -> str:
    from merlin.development.targetgen.packaging.champion_selection import ChampionSelection

    return sel.cert_run or f"recorded-{sel.status or 'unknown'}"


def _fingerprint(
    package_id: str,
    merlin_sha: str,
    cert_run_id: str,
    payload_sha256: str,
    *,
    publication_record: dict[str, Any] | None = None,
) -> str:
    payload = f"{package_id}\n{merlin_sha}\n{cert_run_id}\n{payload_sha256}".encode()
    if publication_record is not None:
        payload += f"\npublication-record:{canonical_sha256(publication_record)}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _git_sha_full(root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root or paths.repo_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def embed_provenance(dest: str | Path, sel: ChampionSelection) -> None:
    """Write the committed metadata layer under ``dest/.merlin/``: a copy of the manifest, the
    provenance lineage, the recorded certification, and the one-line CHAMPION marker."""
    from merlin.development.targetgen.packaging.champion_selection import LAYOUT_VERSION, ChampionSelection, _write

    dest = Path(dest)
    meta = dest / ".merlin"
    meta.mkdir(parents=True, exist_ok=True)

    manifest = load_yaml(dest / "manifest.yaml")
    merlin_sha = _git_sha_full()
    sha7 = git_sha7()
    cert_run = _cert_run_id(sel)
    fp = _fingerprint(
        sel.package_id,
        merlin_sha,
        cert_run,
        package_records.payload_inventory(sel.package_dir)["sha256"],
        publication_record=sel.publication_record,
    )

    # 1) manifest copy (provenance record; identical to the root build manifest)
    shutil.copy2(dest / "manifest.yaml", meta / "manifest.yaml")

    # 2) provenance lineage
    try:
        src_rel = str(sel.package_dir.relative_to(paths.repo_root()))
    except ValueError:
        src_rel = str(sel.package_dir)
    source_layout = json.loads((meta / "source_layout.json").read_text(encoding="utf-8"))
    provenance = {
        "layout_version": source_layout["layout_version"],
        "source_layout": source_layout,
        "generator": "merlin-target-publish",
        "generated_at": utc_stamp(),
        "target": sel.target,
        "package_id": sel.package_id,
        "package_slot": sel.package_id,
        "compiler_package_id": manifest.get("package_id"),
        "family": sel.family,
        "provider_role": _export_role(sel).value,
        "source_package": src_rel,
        "merlin_git_sha": merlin_sha,
        "merlin_git_sha7": sha7,
        "source_status": sel.status,
        "version": sel.version,
        "lineage_depth": sel.lineage_depth,
        "run_refs": [cert_run],
        "fingerprint": fp,
        "publication_record_sha256": canonical_sha256(sel.publication_record) if sel.publication_record else None,
        "certification_scope": "package-payload",
        "external_dependency_closure": "not-attested",
        "exported_payload_certification": "unverified",
        "source_payload": sel.publication_record.get("payload") if sel.publication_record else None,
    }
    _write(meta / "provenance.yaml", dump_yaml(provenance))

    # 3) recorded certification (oot_runner.certify for an OOT accelerator / rvv spike gate)
    gate_ok, gate_detail = _check_gate(sel)
    certification = {
        "target": sel.target,
        "package_id": sel.package_id,
        "layout_kind": sel.layout_kind,
        "gate": "package-payload-observation",
        "status": "unverified",
        "source_input_status": "pass" if gate_ok else "unverified",
        "scope": "package-payload",
        "external_dependency_closure": "not-attested",
        "transformed_export_status": "unverified",
        "recorded_status": sel.status,
        "certification": "unverified",
        "source_recorded_certification": sel.cert_status,
        "certified_by_run": cert_run,
        "oracle_cycles": sel.oracle_cycles,
        "detail": gate_detail,
        # The TIER the pass was earned at. The gate treats a functional-simulator pass and a
        # cycle-accurate RTL pass alike, but they are different claims and a reader of this repo
        # must be able to tell them apart without rerunning anything. Absent = never recorded.
        "tier": (sel.manifest.get("publication") or {}).get("certification_tier") or "UNKNOWN",
        "rungs": (sel.manifest.get("publication") or {}).get("certified_rungs") or [],
    }
    _write(meta / "certification.yaml", dump_yaml(certification))

    # 4) CHAMPION marker: one line
    _write(meta / "CHAMPION", f"{sel.package_id} {sha7} {cert_run}\n")


def _check_gate(sel: ChampionSelection) -> tuple[bool, str]:
    """Admit only scoped observed tests of the selected exact package input bytes."""
    from merlin.development.targetgen.packaging.champion_selection import ChampionSelection

    try:
        record = package_records.read_record(sel.package_dir)
    except (ValueError, OSError) as exc:
        return False, str(exc)
    if (
        record
        and record["publication"].get("certification") == "pass"
        and record["publication"].get("oracle_metadata_identified") is True
    ):
        rungs = record["publication"].get("certified_rungs")
        if (
            isinstance(rungs, list)
            and rungs
            and all(
                isinstance(r, dict)
                and r.get("status") == "pass"
                and package_records.bound_inputs(
                    r.get("package_input_identity"), record["payload"], record.get("compiler_package_id")
                )
                for r in rungs
            )
        ):
            return True, (
                "observed certification of exact package-payload inputs; external dependency closure "
                "not-attested; transformed exports and standalone portability are not certified"
            )
    return False, "producer-bound certification input identity is unavailable; publication is unverified"


def record_certification(
    target: str, package_id: str, results: list[str | Path], *, artifacts_root: str | Path | None = None
) -> dict[str, Any]:
    """Record historical verdicts outside the immutable package payload.

    A certify run writes ``results.yaml`` into its own run dir and stops there. Nothing ever carried
    that verdict back to the package, so ``publication.certification`` could only be written by
    :func:`promote` -- which asks :func:`_check_gate`, which asks for the certification. Nothing can
    satisfy that loop, which is why a package carrying a real out-of-tree dialect could never be
    promoted and the only publishable champion was the hand baseline, whose repo builds a stub.

    The verdict keeps its TIER. A pass on the functional simulator and a pass on cycle-accurate RTL
    are both ``pass`` to the gate, but they are emphatically not the same claim: one says the
    lowering computes the right numbers, the other says the hardware does. Both travel here so the
    published ``certification.yaml`` states which one it is, and a reader can never mistake a
    functional pass for an RTL one.

    Any non-passing rung records failure. Missing target, run or typed successful
    oracle metadata records an unverified result, never a passing certification.
    Explicit target mismatch is refused before any record write. These identity
    checks do not yet bind the results to package input bytes.
    """
    from merlin.development.targetgen.packaging.champion_selection import PublishError, select_champion

    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    rungs: list[dict[str, Any]] = []
    identified = []
    bound = []
    payload = package_records.payload_inventory(sel.package_dir)
    for r in results:
        rp = Path(r)
        if rp.is_dir():
            rp = rp / "results.yaml"
        if not rp.is_file():
            raise PublishError(f"no certify results at {rp}")
        data = load_yaml(rp) or {}
        if not isinstance(data, dict):
            raise PublishError(f"certify results must be an object: {rp}")
        if data.get("target") is not None and data["target"] != target:
            raise PublishError(f"certify target {data['target']!r} does not match {target!r}: {rp}")
        oracle = data.get("oracle") if isinstance(data.get("oracle"), dict) else {}
        kind = oracle.get("kind")
        bound.append(
            package_records.bound_inputs(data.get("package_input_identity"), payload, sel.manifest.get("package_id"))
        )
        identified.append(
            data.get("target") == target
            and isinstance(data.get("run_id"), str)
            and bool(data["run_id"].strip())
            and isinstance(kind, str)
            and bool(kind.strip())
            and kind.strip().lower() not in {"none", "unknown", "skipped"}
            and oracle.get("result") == "pass"
            and type(oracle.get("derived_from_rtl")) is bool
            and type(oracle.get("cycle_accurate")) is bool
        )
        rungs.append(
            {
                "rung": str(data.get("rung", rp.parent.name)),
                "run_id": str(data.get("run_id", "")),
                "target": data.get("target"),
                "status": str(data.get("status", "UNKNOWN")),
                "oracle": str(oracle.get("kind", "UNKNOWN")),
                # `is True` on purpose: a missing key must not read as False, which would silently
                # downgrade an RTL pass to a functional one (or vice versa) on a malformed file.
                "derived_from_rtl": oracle.get("derived_from_rtl") is True,
                "cycle_accurate": oracle.get("cycle_accurate") is True,
                "cycles": oracle.get("cycles"),
                "package_input_identity": data.get("package_input_identity"),
            }
        )
    if not rungs:
        raise PublishError("no certify results given")

    passed = all(r["status"] == "pass" for r in rungs)
    certification = "fail" if not passed else "pass" if all(identified) and all(bound) else "unverified"
    pub = dict(sel.manifest.get("publication") or {})
    pub.update(
        {
            "certification": certification,
            "certified_by_run": rungs[0]["run_id"] or None,
            "certified_at": utc_stamp(),
            "certified_by": "merlin.development.targetgen.packaging.loading.certify",
            # The tier is the weakest rung's, not the strongest: a package is only as certified as its
            # least-certified covered rung, and quoting the best one is how a headline outruns its
            # evidence.
            "certification_tier": {
                "derived_from_rtl": all(r["derived_from_rtl"] for r in rungs),
                "cycle_accurate": all(r["cycle_accurate"] for r in rungs),
                "oracles": sorted({r["oracle"] for r in rungs}),
            },
            "certified_rungs": rungs,
            "oracle_metadata_identified": all(identified),
            "input_binding": "package-payload" if all(bound) else "unverified",
            "external_dependency_closure": "not-attested",
        }
    )
    package_records.write_record(sel.package_dir, load_yaml(sel.package_dir / "manifest.yaml"), pub)
    return pub


class MaterializeRefused(PublishError):
    """The submission or its score is not fit to install as a target's compiler."""


def _score_is_honest(score: dict) -> tuple[bool, str]:
    """Whether a capsule score may be used to justify installing a compiler.

    Mirrors the suite-level vacuous-pass guard: an empty or ungradeable run must never read as evidence.
    A row with no ``tiers`` is the specific shape that let four whole-model capsules report ``pass``
    without executing, so it is refused here too rather than trusted a second time.
    """
    if score.get("integrity_status") not in (None, "clean"):
        return False, f"integrity_status={score.get('integrity_status')!r} (want 'clean')"
    if score.get("gradeable") is False:
        return False, "the run reported gradeable=false"
    rows = score.get("per_capsule") or []
    if not rows:
        return False, "no per-capsule rows — nothing was graded"
    passed = [r for r in rows if r.get("status") == "pass"]
    if not passed:
        return False, "no capsule passed"
    hollow = [r.get("capsule") for r in passed if not (r.get("tiers") or {})]
    if hollow:
        return False, (
            f"{len(hollow)} capsule(s) report pass with no tier evidence "
            f"({', '.join(str(h) for h in hollow[:4])}) — a pass with an empty tier map is "
            f"not evidence a compiler ran"
        )
    return True, f"{len(passed)}/{len(rows)} passed with tier evidence"


def materialize_package(
    target: str,
    source: str | Path,
    *,
    package_id: str = "agent_spec_v1_mlir_oot",
    certified_by_run: str = "",
    score_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Install a run's submission as ``out/artifacts/targets/<target>/<package_id>/`` and return its path.

    Installation preserves payload bytes without inheriting publication authority. A structurally
    non-vacuous score is descriptive evidence, not a binding to the copied payload.
    Source claims remain historical payload/record metadata only. ``certified_by_run``
    is retained as a compatibility argument naming the installation's source run,
    not a certification. ``publication.champion`` is false so ordinary ranking
    does not displace an existing champion merely by sorting earlier.
    """
    import shutil

    from merlin.development.targetgen.packaging.champion_selection import PublishError, _staging_location, _targets_root

    src_dir = Path(source)
    for name, component in (("target", target), ("package_id", package_id)):
        try:
            package_records.component(component)
        except ValueError as exc:
            raise MaterializeRefused(f"{name} must be a non-reserved single path component: {component!r}") from exc
    try:
        sources = (src_dir, Path(score_path)) if score_path is not None else (src_dir,)
        dst = _staging_location(_targets_root(artifacts_root) / target / package_id, sources=sources)
        _staging_location(package_records.record_path(dst), sources=sources)
    except PublishError as exc:
        raise MaterializeRefused(str(exc)) from exc
    if dst.exists():
        if not force:
            raise MaterializeRefused(f"{dst} already exists — pass force=True to replace it")
        if not dst.is_dir():
            raise MaterializeRefused(f"existing package is not a directory: {dst}")
    if not (src_dir / "manifest.yaml").is_file():
        raise MaterializeRefused(f"{src_dir} carries no manifest.yaml — not an OOT backend package")
    if score_path:
        import json as _json

        score = _json.loads(Path(score_path).read_text(encoding="utf-8"))
        ok, detail = _score_is_honest(score)
        if not ok:
            raise MaterializeRefused(f"refusing to install {target} compiler from {src_dir}: {detail}")
        evidence = {
            "n_passed": score.get("n_passed"),
            "n_capsules": score.get("n_capsules"),
            "labels_graded": score.get("labels_graded"),
            "detail": detail,
        }
    else:
        evidence = {"detail": "no score supplied — installed unverified"}
    evidence["payload_binding"] = "unverified"

    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".materialize-", dir=dst.parent))
    prepared = stage / "package"
    previous = stage / "previous" if dst.exists() else None
    expected_payload = package_records.payload_inventory(src_dir)
    shutil.copytree(src_dir, prepared)
    if (
        package_records.payload_inventory(prepared) != expected_payload
        or package_records.payload_inventory(src_dir) != expected_payload
    ):
        raise MaterializeRefused("source payload changed during installation")

    man = load_yaml(prepared / "manifest.yaml")
    source_package_id = man.get("package_id")
    source_status = man.get("status")
    source_publication = man.get("publication")
    promotion = {
        "from": str(src_dir),
        "source_package_id": source_package_id,
        "source_status": source_status,
        "source_publication": source_publication,
        "installed_from_run": certified_by_run,
        "promoted_by": "merlin-target-publish materialize",
        "score": str(score_path) if score_path else None,
        "evidence": evidence,
        "previous_package": str(previous) if previous is not None else None,
    }
    old_record = package_records.record_path(dst)
    if old_record.exists():
        shutil.copy2(old_record, stage / "previous-publication.json")
    # Prepare completely before moving the existing tree. Retain its exact bytes
    # for recovery; neither successful replacement nor failure recursively deletes
    # operator data. This assumes a stable trusted host filesystem, not hostile
    # concurrent path replacement, and is not a crash-atomic multi-rename commit.
    if previous is not None:
        dst.rename(previous)
    try:
        prepared.rename(dst)
        package_records.write_record(dst, man, {"champion": False, "certification": "unverified"}, promotion=promotion)
    except (OSError, ValueError):
        if dst.exists():
            dst.rename(stage / "failed-package")
        if previous is not None:
            previous.rename(dst)
        raise
    if previous is None:
        try:
            stage.rmdir()
        except OSError:
            # Installation already succeeded. An empty-container cleanup failure
            # must not report a failed or rolled-back installation.
            sys.stderr.write(f"WARNING: package installed at {dst}; staging cleanup failed at {stage}\n")
    return dst


def promote(target: str, package_id: str, *, gate: bool = True, artifacts_root: str | Path | None = None) -> None:
    """Promote ``package_id`` using external records, never editing its payload manifest.

    Verifies the certification gate (unless ``gate=False``), clears any prior champion, and sets
    ``publication.champion: true`` + promotion metadata + fingerprint on the chosen package. The
    single-champion invariant is enforced: exactly one package per target ends up flagged."""
    from merlin.development.targetgen.packaging.champion_selection import (
        PublishError,
        _build_selection,
        _targets_root,
        select_champion,
    )

    sel = select_champion(target, artifacts_root=artifacts_root, package_id=package_id)
    ok, detail = _check_gate(sel)
    if gate and not ok:
        raise PublishError(f"promote gate refused for {target}/{package_id}: {detail}")
    if not ok:
        sys.stderr.write(f"WARNING: --no-gate promoting UNCERTIFIED {target}/{package_id}: {detail}\n")

    tdir = _targets_root(artifacts_root) / target
    # clear prior champions (single-champion invariant)
    for man_path in sorted(tdir.glob("*/manifest.yaml")):
        if man_path.parent == sel.package_dir:
            continue
        other = _build_selection(target, man_path.parent, load_yaml(man_path))
        pub = other.manifest.get("publication")
        if isinstance(pub, dict) and pub.get("champion"):
            pub["champion"] = False
            package_records.write_record(man_path.parent, load_yaml(man_path), pub)

    merlin_sha = _git_sha_full()
    cert_run = _cert_run_id(sel)
    man = load_yaml(sel.package_dir / "manifest.yaml")
    pub = dict(sel.manifest.get("publication") or {})
    pub.update(
        {
            "champion": True,
            "certification": "pass" if ok else "unverified",
            "certified_by_run": cert_run,
            "promoted_at": utc_stamp(),
            "promoted_by": "merlin-target-publish",
            "fingerprint": _fingerprint(
                sel.package_id, merlin_sha, cert_run, package_records.payload_inventory(sel.package_dir)["sha256"]
            ),
        }
    )
    package_records.write_record(sel.package_dir, man, pub)


def _git(args: list[str], cwd: Path | None = None, *, check: bool = True) -> subprocess.CompletedProcess:
    from merlin.development.targetgen.packaging.champion_selection import PublishError

    proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=300)
    if check and proc.returncode != 0:
        raise PublishError(f"git {' '.join(args)} failed (rc={proc.returncode}):\n{proc.stderr}")
    return proc


def _head_fingerprint(clone_dir: Path) -> str | None:
    """Parse the ``Merlin-Publish-Fingerprint:`` trailer from the clone HEAD commit, if any."""
    proc = _git(["-C", str(clone_dir), "log", "-1", "--format=%B"], check=False)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        key, sep, val = line.partition(":")
        if sep and key.strip() == "Merlin-Publish-Fingerprint":
            return val.strip()
    return None


def _sync_tree(clone_dir: Path, repo_dir: Path) -> None:
    """Replace the clone working tree (except .git) with the assembled repo tree."""
    for entry in clone_dir.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for entry in repo_dir.iterdir():
        target = clone_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            # copy2 preserves mode. `copyfile` here is why the published entrypoint arrived at
            # 100644: the assembled tree had it executable and this last hop into the clone stripped
            # it, so git recorded a script a fresh clone could not run.
            shutil.copy2(entry, target)


def _checkout_branch(clone_dir: Path, branch: str) -> bool:
    """Check out ``branch`` in the clone. If it exists on the remote, track it (so its tip is the
    per-branch idempotency reference); otherwise start it as an orphan (a fresh, unrelated history —
    each published branch is a standalone repo view, not a commit off the default branch). Returns
    True iff the branch pre-existed on the remote."""
    remote_branches = _git(["-C", str(clone_dir), "branch", "-r"], check=False).stdout.split()
    if f"origin/{branch}" in remote_branches:
        _git(["-C", str(clone_dir), "checkout", "-B", branch, "--track", f"origin/{branch}"])
        return True
    _git(["-C", str(clone_dir), "checkout", "--orphan", branch])
    # an orphan checkout keeps the old tree staged; clear it so _sync_tree writes a clean state.
    _git(["-C", str(clone_dir), "rm", "-rf", "--quiet", "."], check=False)
    return False


def _git_publish(
    remote: str,
    repo_dir: Path,
    sel: ChampionSelection,
    manifest: dict[str, Any],
    fingerprint: str,
    cert_run: str,
    stage_root: Path,
    branch: str,
) -> tuple[str, str, bool]:
    """Clone the remote, check out ``branch``, replace the tree with the assembled repo, commit +
    tag, push to ``refs/heads/<branch>``. Idempotent PER BRANCH on the **fingerprint**: a matching
    branch-tip fingerprint is a no-op. Returns (commit_sha, tag, noop).

    An existing TAG is deliberately not a veto. The tag names a package *version*, but the
    fingerprint covers package ID, Merlin revision, cert run, source payload inventory and
    the selected external publication record — so re-certifying an unchanged
    payload (``spike_verified`` -> ``k1_verified`` after a board campaign) produces new provenance
    at the same version. Letting the tag veto that meant the published ``.merlin/certification.yaml``
    kept the WEAKER status forever, silently understating the certification to every consumer. The
    branch moves; the existing tag is left exactly where it is, because consumers may have pinned
    it and rewriting a published tag is worse than not adding one."""
    from merlin.development.targetgen.packaging.champion_selection import (
        LAYOUT_VERSION,
        ChampionSelection,
        _fresh_destination,
    )

    clone_dir = _fresh_destination(stage_root / "clone", sources=(repo_dir, sel.package_dir))
    _git(["clone", remote, str(clone_dir)])
    branch_existed = _checkout_branch(clone_dir, branch)

    version = int(manifest.get("version", sel.version) or 0)
    tag = f"v{version}-{sel.package_id}"

    existing_tags = set(_git(["-C", str(clone_dir), "tag", "--list"]).stdout.split())
    # per-branch idempotency: the fingerprint is read from THIS branch's tip (not the default HEAD),
    # so publishing the baseline branch never masks a stale champion-branch tip and vice-versa.
    if branch_existed and _head_fingerprint(clone_dir) == fingerprint:
        head = _git(["-C", str(clone_dir), "rev-parse", "HEAD"], check=False)
        return (head.stdout.strip() or "unknown", tag, True)

    _sync_tree(clone_dir, repo_dir)
    _git(["-C", str(clone_dir), "add", "--", "."])

    merlin_sha = _git_sha_full()
    gate_ok, gate_detail = _check_gate(sel)
    # A --no-gate publish names itself in the SUBJECT. The history of a target repo is its
    # provenance trail; a commit that says "champion" for a package the gate refused makes that
    # trail assert the one thing that is not true about it.
    subject = (
        f"publish({sel.target}): export of tested package inputs {sel.package_id}"
        if gate_ok
        else f"publish({sel.target}): UNCERTIFIED package {sel.package_id}"
    )
    warning = (
        ""
        if gate_ok
        else (
            f"WARNING: published with --no-gate. This package did NOT pass the certification gate.\n"
            f"Gate-Refusal: {gate_detail}\n"
            f"Not-A-Champion: true\n"
        )
    )
    body = (
        warning + f"Package: {sel.package_id}\n"
        f"Target: {sel.target}\n"
        f"Family: {sel.family}\n"
        f"Merlin-Sha: {merlin_sha}\n"
        f"Internal-Run: {cert_run}\n"
        "Cert: unverified (exported tree)\n"
        f"Source-Input-Certification: {'pass' if gate_ok else 'unverified'}\n"
        "Certification-Scope: package-payload\n"
        "External-Dependency-Closure: not-attested\n"
        "Transformed-Export-Certification: unverified\n"
        f"Layout-Version: {manifest.get('layout_version', LAYOUT_VERSION)}\n"
        "\n"
        f"Merlin-Publish-Fingerprint: {fingerprint}\n"
    )
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
            subject,
            "-m",
            body,
        ]
    )
    # annotated tag (carries the fingerprint; also robust to git configs that force annotation).
    # Never re-point an existing tag: a consumer may have pinned it, and this commit is a
    # re-certification of the SAME version, not a new release.
    if tag not in existing_tags:
        _git(
            [
                "-C",
                str(clone_dir),
                "-c",
                "user.name=merlin-target-publish",
                "-c",
                "user.email=publish@merlin.local",
                "tag",
                "-a",
                tag,
                "-m",
                (
                    f"{sel.target} export of tested package inputs {sel.package_id}"
                    if gate_ok
                    else f"{sel.target} UNCERTIFIED package {sel.package_id} (published --no-gate)"
                )
                + f"\nSource-Input-Certification: {'pass' if gate_ok else 'unverified'}\n"
                "Certification-Scope: package-payload\n"
                "External-Dependency-Closure: not-attested\n"
                "Transformed-Export-Certification: unverified\n"
                f"Merlin-Publish-Fingerprint: {fingerprint}\n",
            ]
        )
    commit_sha = _git(["-C", str(clone_dir), "rev-parse", "HEAD"]).stdout.strip()

    _git(["-C", str(clone_dir), "push", "origin", f"HEAD:refs/heads/{branch}"])
    _git(["-C", str(clone_dir), "push", "origin", "--tags"])
    # NOTE: `gh release` is intentionally skipped for file:// bare remotes (the verification path).
    return commit_sha, tag, False
