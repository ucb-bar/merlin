"""One-command accelerator onboarding — the ONLY thing a designer runs after dropping a descriptor.

The onboarding ideal is: drop ONE agnostic ``target_experiment.yaml`` that points at the accelerator's
RTL, run ONE command, done — no per-target Python, no hand-authored manifest, no prompt files. This
module is that one command. It orchestrates the already-agnostic machinery end to end:

1. Load the declarative descriptor (:func:`merlin.targetgen.target_experiment.load_target_experiment`).
2. Ground the RTL pointer: if the descriptor carries ``rtl.repo``, validate the pointer resolves and
   emit the exact mlc registration step (RTL→arc compilation is mlc's job — merlin consumes its
   outputs, never fakes discovery). Report how many static facts mlc actually grounds for the target.
3. Regenerate the capability manifest as a REGENERATED artifact (never hand-edited) via
   :func:`merlin.targetgen.capability_manifests.write_oot_target` into the target's OOT package path.
4. Validate it loads through the capability spine (:func:`load_capability_manifest`): its compute-unit
   ``kind`` is a known family, its endpoint kind is legal, and print a derived summary
   (target, kind, endpoint_kind, dtypes, mesh/dim).
5. Fail honestly (:class:`OnboardError`) with a precise message the moment a step cannot be grounded —
   an unresolvable RTL pointer, or a target with neither a manifest generator nor a committed contract.
   It NEVER emits a fabricated manifest.

TARGET-AGNOSTIC by construction: every routing decision flows from the descriptor + the derived
compute-unit ``kind`` through :mod:`merlin.targetgen.families`; there is no ``if target ==`` and no
per-target branch anywhere below (membership in the generator registry is a generic dict lookup).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen import capability_manifests as _cm
from merlin.targetgen import compute_units as _cu
from merlin.targetgen import families as _families
from merlin.targetgen import target_registry as _tr
from merlin.targetgen.rtl import mlc_bridge as _mlc
from merlin.targetgen.rtl.facts import target_base
from merlin.targetgen.target_experiment import (
    CapabilityManifest,
    TargetExperiment,
    load_capability_manifest,
    load_target_experiment,
)

# URL schemes we accept as a remote RTL pointer (merlin cannot clone/compile here — that is mlc's job;
# a well-formed remote still yields an honest, precise registration step rather than a fake discovery).
_URL_SCHEMES = ("http://", "https://", "git://", "ssh://", "git@", "file://")


class OnboardError(RuntimeError):
    """A step could not be grounded — raised instead of emitting a fabricated manifest."""


@dataclass
class OnboardResult:
    """The grounded outcome of an onboarding run (what the CLI renders)."""
    target: str
    manifest: CapabilityManifest
    oot_root: Path | None                 # where the manifest was regenerated (None if not regenerated)
    regenerated: bool                     # True if written via the generator; False if a committed contract
    dtypes: tuple[str, ...]
    mesh: dict | None                     # {"rows":R,"cols":C} or {"dim":D} or None (no mesh)
    rtl_notes: list[str] = field(default_factory=list)   # honest provenance/registration lines


# --------------------------------------------------------------------------- RTL pointer grounding
def _is_url(pointer: str) -> bool:
    return "://" in pointer or pointer.startswith("git@")


def _resolve_local(pointer: str) -> Path:
    """Resolve a local RTL-repo pointer relative to the repo root (bundle convention), never parents[N]."""
    p = Path(pointer).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def _mlc_registration_step(target: str) -> str:
    """The exact, honest registration step — mlc owns RTL→arc compilation; merlin only consumes its
    per-target outputs. We name the env var + the directory mlc must produce, without inventing a flag."""
    return (f"register the RTL with mlc (RTL->arc compilation is mlc's responsibility, not merlin's): "
            f"point MERLIN_MLC_DIR at an mlc checkout that has compiled this repo so it exposes "
            f"$MERLIN_MLC_DIR/runs/circt-arc/{target}/outputs — merlin.targetgen.rtl.mlc_bridge then "
            f"derives ISA/mesh/memory facts from those outputs.")


def _ground_rtl(te: TargetExperiment) -> list[str]:
    """Validate the descriptor's RTL pointer and report mlc grounding — honestly, never faking discovery.

    Raises :class:`OnboardError` only for a pointer that provably cannot resolve (a local path that does
    not exist). A remote URL or an unregistered-but-present repo is reported with the exact next step, so
    onboarding can still regenerate the manifest from the descriptor + whatever facts mlc does ground.
    """
    notes: list[str] = []
    if te.rtl_repo:
        if _is_url(te.rtl_repo):
            if not te.rtl_repo.startswith(_URL_SCHEMES):
                raise OnboardError(
                    f"rtl.repo {te.rtl_repo!r} is not a resolvable local path or a recognized remote URL "
                    f"(expected one of {_URL_SCHEMES}). Refusing to guess the RTL location.")
            notes.append(f"rtl.repo is a remote URL ({te.rtl_repo}); merlin does not clone/compile RTL — "
                         + _mlc_registration_step(te.target))
        else:
            local = _resolve_local(te.rtl_repo)
            if not local.exists():
                raise OnboardError(
                    f"rtl.repo {te.rtl_repo!r} does not resolve (looked at {local}). Fix the descriptor's "
                    f"rtl.repo pointer; refusing to onboard against a non-existent RTL location.")
            notes.append(f"rtl.repo resolves -> {local}")
            notes.append(_mlc_registration_step(te.target))
    else:
        notes.append("rtl.repo not set (legacy mode): assuming the RTL is already registered with mlc "
                     f"under target {te.target!r}.")

    # Report — never require — how much mlc actually grounds. A SIMT/prototype target legitimately
    # grounds 0/4 static facts (its facts come from the contract, not the arc decoder), so this is
    # informational, not a gate.
    ok, why = _mlc.mlc_available()
    if not ok:
        notes.append(f"mlc unavailable ({why}) — proceeding from the descriptor; RTL-derived facts absent.")
    else:
        # KIND-routed fact extraction: systolic/vector/scalar -> the CIRCT static bundle (unchanged for
        # gemmini), simt -> muon, spatial -> the OuterProductUnit state-manifest introspect. The default
        # (no kind resolved for a freshly-onboarding target) is the systolic static path — same as before.
        bundle = _mlc.fact_bundle_for(te.target)
        notes.append(f"mlc grounds {bundle['n_derived']}/{len(bundle['fields'])} static RTL facts for "
                     f"{te.target!r} "
                     f"({'arc model present' if _mlc.arc_available(te.target) else 'no arc model — SIMT/prototype'}).")
    return notes


# --------------------------------------------------------------------------- manifest regeneration
@contextmanager
def _target_path(root: Path | None):
    """Temporarily expose an OOT package root on MERLIN_TARGET_PATH so the spine can resolve it."""
    if root is None:
        yield
        return
    key = "MERLIN_TARGET_PATH"
    prev = os.environ.get(key)
    entries = [str(root)] + ([prev] if prev else [])
    os.environ[key] = os.pathsep.join(entries)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _regenerate_manifest(target: str, oot_root: Path | None) -> tuple[Path | None, bool]:
    """Regenerate the capability manifest as a fresh artifact, or accept a committed one — never fake it.

    Returns ``(written_root, regenerated)``. Routing is a generic registry lookup, not a per-target
    branch: if the target has a manifest generator we regenerate; else if a contract is already committed
    we validate that; else we fail honestly.
    """
    if target in _cm.MANIFESTS:
        root = Path(oot_root) if oot_root is not None else target_base(target)
        _cm.write_oot_target(target, root)   # validates schema + compute_units before writing
        return root, True

    # No generator entry. Only accept an ALREADY-committed contract (reference/in-tree target); never
    # synthesize one from nothing.
    with _target_path(oot_root):
        info = _tr.resolve(target)
        if info.contract_path.is_file():
            return None, False
    raise OnboardError(
        f"cannot ground a capability manifest for {target!r}: no generator in "
        f"merlin.targetgen.capability_manifests.MANIFESTS ({sorted(_cm.MANIFESTS)}) and no committed "
        f"target_contract.yaml. Refusing to fabricate a manifest — add a generator entry or commit a "
        f"contract, then re-run onboard.")


# --------------------------------------------------------------------------- summary derivation
def _derive_summary(manifest: CapabilityManifest) -> tuple[tuple[str, ...], dict | None]:
    """Derive the dtype set (union of effective compute-unit dtypes) and mesh/dim from the manifest."""
    units = _cu.compute_units(manifest.contract)
    dtypes: set[str] = set()
    for u in units:
        dtypes |= set(_cu.effective(u, units).dtypes)
    mesh = (manifest.contract.get("capabilities") or {}).get("mesh")
    if not mesh:
        dim = _mlc.discovered_dim(manifest.target)   # may be None (no mesh / mlc absent) — honest
        mesh = {"dim": dim} if dim else None
    return tuple(sorted(dtypes)), mesh


# --------------------------------------------------------------------------- entrypoint
def onboard(descriptor: str | Path, *, oot_root: str | Path | None = None) -> OnboardResult:
    """Onboard one accelerator from its descriptor. See the module docstring for the five steps.

    ``oot_root`` overrides where the manifest is regenerated (default: the target's OOT package path
    under ``out/artifacts/targets/<target>/``); pass a tmp dir for a hermetic / non-clobbering run.
    """
    te = load_target_experiment(descriptor)
    rtl_notes = _ground_rtl(te)
    written_root, regenerated = _regenerate_manifest(te.target, Path(oot_root) if oot_root else None)

    with _target_path(written_root):
        try:
            manifest = load_capability_manifest(te.target)
        except Exception as e:  # noqa: BLE001 — surface the spine failure precisely, never swallow it
            raise OnboardError(f"regenerated manifest for {te.target!r} did not load through the spine: "
                               f"{type(e).__name__}: {e}") from e
        if manifest.kind not in _families.known_kinds():
            raise OnboardError(f"{te.target!r}: derived kind {manifest.kind!r} is not a known family "
                               f"{_families.known_kinds()} — cannot route.")
        dtypes, mesh = _derive_summary(manifest)

    return OnboardResult(target=te.target, manifest=manifest, oot_root=written_root,
                         regenerated=regenerated, dtypes=dtypes, mesh=mesh, rtl_notes=rtl_notes)


def render(result: OnboardResult) -> str:
    """Render an onboarding result as a human-readable report."""
    m = result.manifest
    lines = [f"# Onboarded target: {result.target}", ""]
    lines.append("## RTL grounding")
    lines += [f"- {n}" for n in result.rtl_notes]
    lines.append("")
    lines.append("## Capability manifest")
    if result.regenerated:
        lines.append(f"- regenerated at: {result.oot_root}/contracts/target_contract.yaml")
    else:
        lines.append("- source: committed target_contract.yaml (no generator; not regenerated)")
    mesh = ("none" if result.mesh is None
            else (f"dim={result.mesh['dim']}" if "dim" in result.mesh
                  else f"{result.mesh.get('rows')}x{result.mesh.get('cols')}"))
    lines += [
        f"- target       : {m.target}",
        f"- kind         : {m.kind}",
        f"- endpoint_kind: {m.endpoint_kind}",
        f"- dtypes       : {list(result.dtypes)}",
        f"- mesh/dim     : {mesh}",
        f"- suite        : {m.suite}",
        f"- dtype token  : {m.dtype}",
        "",
        "OK — the target routes through the capability spine.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="merlin-onboard",
        description="Onboard a new accelerator from ONE descriptor: ground its RTL pointer, regenerate "
                    "the capability manifest, and validate it routes through the capability spine.")
    ap.add_argument("descriptor", help="path to the target_experiment.yaml descriptor")
    ap.add_argument("--oot-root", help="override where the manifest is regenerated "
                                       "(default: out/artifacts/targets/<target>/)")
    a = ap.parse_args(argv)
    try:
        result = onboard(a.descriptor, oot_root=a.oot_root)
    except OnboardError as e:
        print(f"onboard FAILED (honest, not fabricated): {e}")
        return 2
    print(render(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
