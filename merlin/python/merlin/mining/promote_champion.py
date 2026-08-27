"""Promote a beam-search champion into the publishable RVV package structure.

The RVV beam (``merlin.mining.beam``) mints isolated fork packages under
``<run>/targets/rvv/<pkg>/`` (schedule.mlir + knobs.yaml + manifest.yaml) and records the whole
search in ``<run>/beam_tree.yaml``. The measured, XNNPACK-beating win lives on the tree's ``best``
node (``gate_ok``, board-measured ``k1_wall_ns`` / ``speedup``, ``attainment_vs_expert``) -- but the
fork's ``manifest.yaml`` is stamped ``status: proposed`` with no certification, so
:mod:`merlin.targetgen.publish`'s gate REFUSES it. This module is the missing bridge: it reads the
best node, verifies it FAIL-CLOSED (must be ``gate_ok``, board-measured, not inert, and -- when a
noise margin is recorded -- above the noise floor), and stamps a TRUTHFUL certification onto the
fork package so ``merlin-target-publish`` can export it.

Honest certification -- read this before changing the status string
-------------------------------------------------------------------
The beam measures a fork on the **live K1 board** (``k1_wall_ns``) with a correctness gate
(``gate_ok``). It does **not** run the spike functional simulator. So the honest recorded status is
``k1_verified`` -- we never write ``spike_verified`` for a fork that only the board measured.

``publish._check_gate`` for the ``vector_schedule`` (rvv) family now accepts
``status in {spike_verified, rtl_certified, k1_verified}``. ``k1_verified`` means the fork was
measured correct AND faster on the live K1 board — a STRONGER certification than the spike simulator
for a physical target, not a weaker one — so this module stamps the truthful ``status: k1_verified``
+ ``publication.certification: "pass"`` + ``certified_by: "k1_board"`` + ``certified_by_run: <beam
run>`` + the measured metrics, and publishes through the REAL gate (``gate=True``). No ``--no-gate``
bypass and no false ``spike_verified`` claim for a fork the board (not spike) measured. Everything
here is additive and reuses :mod:`merlin.targetgen.publish` verbatim.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import paths
from ..common.artifacts import utc_stamp
from ..common.yaml import load_yaml, write_yaml
from ..targetgen import publish as pub

# The honest recorded status for a fork the *board* measured (not the spike simulator). Never
# downgrade this to spike_verified for a K1-only champion.
K1_VERIFIED_STATUS = "k1_verified"


class PromoteError(RuntimeError):
    """The beam champion could not be promoted (missing run, failed fail-closed verification)."""


# --------------------------------------------------------------------------- read the beam tree


@dataclass
class BeamChampion:
    """The best node of a beam run + the located fork package it points at."""

    run_dir: Path
    beam_run_id: str            # the beam run folder name (provenance: certified_by_run)
    run_id: str                 # the fork package run_id (== package dir name)
    node: dict[str, Any]
    tree: dict[str, Any]
    package_dir: Path
    target: str
    speedup: float | None
    k1_wall_ns: int | None
    attainment_vs_expert: float | None
    noise_margin: float | None
    gate_ok: bool
    inert: bool

    @property
    def artifacts_root(self) -> Path:
        """The root to hand ``publish`` so it finds ``<root>/targets/<target>/<pkg>`` (the run dir)."""
        return self.run_dir


def _resolve_package_dir(run_dir: Path, node: dict[str, Any], *, target: str) -> Path:
    """Resolve the best node's fork package dir.

    Prefer the RUN-LOCAL location (``<run_dir>/targets/<target>/<run_id>``) so that stamping and
    ``publish`` (handed ``artifacts_root=<run_dir>``) always target the SAME tree even if the run
    dir has been copied/relocated (the ``package_dir`` recorded in beam_tree.yaml is an absolute
    path to where the beam originally minted it). Fall back to the recorded ``package_dir``
    (absolute or repo-root-relative) when no run-local copy exists.
    """
    run_id = node.get("run_id")
    if run_id:
        local = run_dir / "targets" / target / str(run_id)
        if local.is_dir():
            return local
    raw = node.get("package_dir")
    if not raw:
        raise PromoteError(f"best node {run_id!r} has no run-local package and no package_dir")
    p = Path(raw)
    if not p.is_absolute():
        p = paths.repo_root() / p
    if not p.is_dir():
        raise PromoteError(f"fork package dir does not exist: {p}")
    return p


def read_beam_champion(run_dir: str | Path, *, target: str = "rvv") -> BeamChampion:
    """Load ``<run_dir>/beam_tree.yaml`` and locate the full ``best`` node + its fork package."""
    run_dir = Path(run_dir)
    tree_path = run_dir / "beam_tree.yaml"
    if not tree_path.is_file():
        raise PromoteError(f"no beam_tree.yaml under {run_dir}")
    tree = load_yaml(tree_path)
    if not isinstance(tree, dict):
        raise PromoteError(f"beam_tree.yaml is not a mapping: {tree_path}")

    best = tree.get("best")
    if not isinstance(best, dict) or not best.get("run_id"):
        raise PromoteError(f"beam_tree.yaml has no 'best' node with a run_id: {tree_path}")
    best_id = str(best["run_id"])

    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    node = next((n for n in nodes if isinstance(n, dict) and str(n.get("run_id")) == best_id), None)
    if node is None:
        raise PromoteError(f"best run_id {best_id!r} not found among beam_tree nodes")

    tgt = str(tree.get("target", target))
    pkg_dir = _resolve_package_dir(run_dir, node, target=tgt)
    return BeamChampion(
        run_dir=run_dir,
        beam_run_id=run_dir.name,
        run_id=best_id,
        node=node,
        tree=tree,
        package_dir=pkg_dir,
        target=tgt,
        speedup=_as_float(node.get("speedup")),
        k1_wall_ns=_as_int(node.get("k1_wall_ns")),
        attainment_vs_expert=_as_float(node.get("attainment_vs_expert")),
        noise_margin=_as_float(tree.get("noise_margin")),
        gate_ok=bool(node.get("gate_ok")),
        inert=bool(node.get("inert")),
    )


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- fail-closed verify


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def verify_champion(champ: BeamChampion, *, require_board: bool = True,
                    require_margin: bool = True) -> Verdict:
    """Fail-closed gate for a beam champion. NEVER stamp an unverified/inert/noise fork.

    Requires: ``gate_ok`` (numerics), a real board-measured ``k1_wall_ns`` + ``speedup`` > 1,
    ``inert`` is False, and -- when a noise margin is recorded on the tree -- the measured speedup
    beats the noise floor (``speedup - 1 > noise_margin``). Missing board metrics fail closed when
    ``require_board`` is set.
    """
    reasons: list[str] = []
    if not champ.gate_ok:
        reasons.append("best node is not gate_ok (correctness gate did not pass)")
    if champ.inert:
        reasons.append("best node is marked inert (no measured emitted-code delta)")
    if require_board and champ.k1_wall_ns is None:
        reasons.append("best node has no board-measured k1_wall_ns")
    if champ.speedup is None:
        reasons.append("best node has no measured speedup")
    elif champ.speedup <= 1.0:
        reasons.append(f"best node speedup {champ.speedup} is not a win (<= 1.0)")
    if champ.noise_margin is not None and champ.speedup is not None:
        if (champ.speedup - 1.0) <= champ.noise_margin:
            reasons.append(f"speedup margin {champ.speedup - 1.0:.4f} is within the noise floor "
                           f"{champ.noise_margin} (not above noise)")
    return Verdict(ok=not reasons, reasons=reasons)


# --------------------------------------------------------------------------- stamp certification


@dataclass
class StampResult:
    package_dir: Path
    package_id: str
    beam_run_id: str
    status: str
    version: int
    branch_hint: str
    artifacts_root: Path
    manifest: dict[str, Any]


def _clear_other_champions(target_dir: Path, keep: Path) -> None:
    """Enforce the single-champion invariant within the run's targets tree (unset others)."""
    for man_path in sorted(target_dir.glob("*/manifest.yaml")):
        if man_path.parent == keep:
            continue
        man = load_yaml(man_path)
        if not isinstance(man, dict):
            continue
        p = man.get("publication")
        if isinstance(p, dict) and p.get("champion"):
            p["champion"] = False
            man["publication"] = p
            write_yaml(man_path, man)


def stamp_champion(champ: BeamChampion, *, status: str = K1_VERIFIED_STATUS,
                   certified_by: str = "k1_board", require_board: bool = True,
                   require_margin: bool = True, force: bool = False) -> StampResult:
    """Verify (fail-closed) then stamp the fork ``manifest.yaml`` in place so it is publishable.

    Adds a top-level ``status`` (default the honest ``k1_verified``), ``version`` and ``package_id``
    (from lineage / run_id) that ``publish.select_champion`` reads, and a ``publication`` block
    carrying the truthful certification (``certification: pass`` + ``certified_by`` +
    ``certified_by_run`` = the beam run id) plus the board-measured metrics for provenance. Reuses
    ``publish._fingerprint`` / ``publish._git_sha_full`` -- publish.py is untouched.
    """
    verdict = verify_champion(champ, require_board=require_board, require_margin=require_margin)
    if not verdict.ok and not force:
        raise PromoteError("champion failed fail-closed verification: " + "; ".join(verdict.reasons))
    if not verdict.ok:
        sys.stderr.write("WARNING: --force stamping a champion that FAILED verification: "
                         + "; ".join(verdict.reasons) + "\n")

    man_path = champ.package_dir / "manifest.yaml"
    man = load_yaml(man_path)
    if not isinstance(man, dict):
        raise PromoteError(f"fork manifest is not a mapping: {man_path}")

    lineage = man.get("lineage") if isinstance(man.get("lineage"), dict) else {}
    version = int(lineage.get("version", man.get("version", 0)) or 0)
    package_id = str(man.get("package_id", champ.run_id))

    merlin_sha = pub._git_sha_full()
    cert_run = champ.beam_run_id
    fingerprint = pub._fingerprint(package_id, merlin_sha, cert_run)

    # honest status: k1_verified for a board-only measured fork (NEVER spike_verified here)
    man["status"] = status
    man["version"] = version
    man["package_id"] = package_id

    publication = man.get("publication") if isinstance(man.get("publication"), dict) else {}
    publication.update({
        "champion": True,
        "certification": "pass",
        "certified_by": certified_by,
        "certified_by_run": cert_run,
        "promoted_at": utc_stamp(),
        "promoted_by": "merlin.mining.promote_champion",
        "fingerprint": fingerprint,
        "measured": {
            "k1_wall_ns": champ.k1_wall_ns,
            "speedup": champ.speedup,
            "attainment_vs_expert": champ.attainment_vs_expert,
            "expert_wall_ns": champ.tree.get("expert_wall_ns"),
            "noise_margin": champ.noise_margin,
            "beam_run_id": champ.beam_run_id,
            "verified_by": "promote_champion.verify_champion",
        },
    })
    man["publication"] = publication
    write_yaml(man_path, man)

    _clear_other_champions(champ.package_dir.parent, champ.package_dir)

    branch_hint = f"stable/{package_id}"
    return StampResult(
        package_dir=champ.package_dir,
        package_id=package_id,
        beam_run_id=cert_run,
        status=status,
        version=version,
        branch_hint=branch_hint,
        artifacts_root=champ.artifacts_root,
        manifest=man,
    )


# --------------------------------------------------------------------------- drive the publish


def promote_and_publish(run_dir: str | Path, *, target: str = "rvv", execute: bool = False,
                        remote: str | None = None, verify_build: bool = True,
                        status: str = K1_VERIFIED_STATUS, require_board: bool = True,
                        require_margin: bool = True, force: bool = False):
    """End-to-end: read the beam champion, stamp it, and drive ``publish.publish`` for it.

    Publishes through the REAL gate (``gate=True``): ``publish._check_gate`` now accepts
    ``k1_verified`` for the rvv (``vector_schedule``) family alongside ``spike_verified`` /
    ``rtl_certified`` — a board measurement is a stronger certification than the spike simulator for a
    physical target, not a weaker one. So the champion this module stamps ``k1_verified`` passes the
    gate honestly, with no ``--no-gate`` bypass and no false ``spike_verified`` claim. The stamped
    ``publication.certification: pass`` + ``certified_by: k1_board`` rides along as the truthful
    record. Returns (StampResult, PublishResult).
    """
    champ = read_beam_champion(run_dir, target=target)
    stamp = stamp_champion(champ, status=status, require_board=require_board,
                           require_margin=require_margin, force=force)
    result = pub.publish(
        champ.target,
        dry_run=not execute,
        remote=remote,
        gate=True,                         # real gate now accepts k1_verified (see module docstring)
        verify_build=verify_build,
        package_id=stamp.package_id,
        artifacts_root=str(stamp.artifacts_root),
    )
    return stamp, result


# --------------------------------------------------------------------------- payload round-trip note


def write_payload_manifest(payload_dir: str | Path, *, package_id: str,
                           target: str = "rvv") -> Path:
    """Opt-in fix for the round-trip nuance: write a minimal ``manifest.yaml`` into a published
    ``payload/`` so ``mining.registry.load_rvv_package(payload_dir)`` can read it back.

    The publish bridge writes ``payload/schedule.mlir`` + ``knobs.yaml`` but NOT an
    ``rvv_package_manifest`` (the published tree is consumed as a CMake OOT build, and the rvv load
    path normally reads the ORIGINAL out/artifacts package). This helper is provided for callers who
    want the published payload to be directly loadable; it is NOT called by the publish flow.
    """
    payload_dir = Path(payload_dir)
    man = {
        "target": target,
        "run_id": package_id,
        "package_id": package_id,
        "family": "vector_schedule",
        "schedule_format": "transform_dialect_mlir",
        "status": K1_VERIFIED_STATUS,
        "authoring": {"mode": "deterministic_generated_from_spec",
                      "generated_by_agent": False, "author": "merlin.mining.promote_champion"},
        "outputs": {"schedule": "schedule.mlir", "knobs": "knobs.yaml"},
    }
    out = payload_dir / "manifest.yaml"
    write_yaml(out, man, header="rvv package manifest (payload round-trip; promote_champion)")
    return out


# --------------------------------------------------------------------------- CLI


def _print_stamp(stamp: StampResult) -> None:
    print(f"stamped champion: {stamp.package_id}")
    print(f"  package_dir    : {stamp.package_dir}")
    print(f"  status         : {stamp.status}")
    print(f"  version        : {stamp.version}")
    print(f"  certified_by   : k1_board (run {stamp.beam_run_id})")
    print(f"  publish branch : {stamp.branch_hint}")
    print(f"  artifacts_root : {stamp.artifacts_root}  (pass to publish --artifacts-root)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-rvv-promote-champion",
        description="Certify + stamp a beam-verified RVV champion and (optionally) publish it.")
    ap.add_argument("--run", required=True, help="beam run dir containing beam_tree.yaml")
    # REQUIRED, not defaulted. A default of one target silently mislabels every run for another
    # one -- the mined artifacts are written under <target>/ and the CCA is compared against that
    # target's expert corpus, so a mislabelled run compares the wrong things and says nothing about it.
    ap.add_argument("--target", required=True,
                    help="the target whose expert corpus is mined and whose endpoint is lifted")
    ap.add_argument("--status", default=K1_VERIFIED_STATUS,
                    help="honest recorded status (default k1_verified; do NOT use spike_verified "
                         "unless spike actually verified the fork)")
    ap.add_argument("--publish", action="store_true", help="also drive merlin.targetgen.publish")
    ap.add_argument("--execute", action="store_true",
                    help="with --publish: really clone/commit/push (else dry-run)")
    ap.add_argument("--remote", help="override the publish remote (file:// bare remote only here)")
    ap.add_argument("--no-verify-build", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="stamp even if fail-closed verification fails (LOUD warning)")
    args = ap.parse_args(argv)

    try:
        if args.publish:
            stamp, result = promote_and_publish(
                args.run, target=args.target, execute=args.execute, remote=args.remote,
                verify_build=not args.no_verify_build, status=args.status, force=args.force)
            _print_stamp(stamp)
            print("--- publish ---")
            print(f"  remote     : {result.remote}")
            print(f"  branch     : {result.branch}")
            print(f"  dry_run    : {result.dry_run}  committed={result.committed}  noop={result.noop}")
            print(f"  tag        : {result.tag}")
            if result.commit_sha:
                print(f"  commit     : {result.commit_sha}")
            return 0
        champ = read_beam_champion(args.run, target=args.target)
        stamp = stamp_champion(champ, status=args.status, force=args.force)
        _print_stamp(stamp)
        print("(run again with --publish [--execute] to export via merlin-target-publish)")
        return 0
    except (PromoteError, pub.PublishError) as e:
        sys.stderr.write(f"merlin-rvv-promote-champion: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
