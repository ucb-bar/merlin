"""Resolve explicit model workload declarations and freeze their loader dependencies."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

from merlin.capture.artifacts import M2MUnavailable
from merlin.capture.demands import model_op_demands_checked
from merlin.capture.pytorch import _m2m_dir
from merlin.common.paths import repo_root

# ---- model_inputs ------------------------------------------------------------------------------


def resolve_model_loader(entry: dict, m2m_dir: str | Path | None = None) -> Path:
    """A model capsule entry names either a workload (``model: small_llama``) or an explicit ``loader``.

    A relative ``loader`` is resolved REPO-ROOT-FIRST, then against the m2m checkout. Both roots are
    needed and the order matters: a whole-model capsule whose network is defined in THIS repo (the
    self-contained case -- the capsule's own ``capsule.pytorch.py`` is the source of record) can then be
    named by the repo-root-relative path this repo already uses for every other in-tree reference, while
    a capsule that names a model2MLIR workload keeps working unchanged. Resolving only against the m2m
    root, as this did, made the in-repo case unnameable: every whole-model capsule had to keep its
    network in an external checkout, which is exactly the dependency that leaves a capsule unbuildable
    from a clean clone.
    """
    root = Path(m2m_dir) if m2m_dir else _m2m_dir()
    if entry.get("loader"):
        p = Path(entry["loader"])
        if p.is_absolute():
            return p
        in_repo = repo_root() / p
        return in_repo if in_repo.is_file() else (root / p)
    name = entry.get("model")
    if not name:
        raise ValueError("model capsule entry needs 'model' (workload name) or 'loader' (path)")
    exact = root / "workloads" / name / "loader.py"
    if exact.is_file():
        return exact
    return _workload_named(root, name) or exact


def _workload_named(root: Path, name: str) -> Path | None:
    """The workload directory a ROSTER name denotes, when it is not spelled the same.

    A roster declares a model (``resnet50``); a workload directory carries the checkpoint revision
    (``resnet50_v1_5``). ``dse_guidance.models`` already resolves the capture direction of this by
    longest-prefix match -- without which a fully captured ResNet reads as ABSENT -- and the capsule
    direction had no resolution at all, so a roster model with a revisioned workload simply could not be
    named by a synthesized capsule.

    Prefix, and only prefix: a revision suffix extends the model name, it never replaces it. Ambiguity
    is an error rather than a pick, because two revisions are two different networks and choosing by
    sort order would silently certify the wrong one. ``None`` when nothing matches, so the caller
    reports the path it expected rather than a guess.
    """
    wroot = root / "workloads"
    if not wroot.is_dir():
        return None
    hits = sorted(d for d in wroot.iterdir() if d.is_dir() and d.name.startswith(name) and (d / "loader.py").is_file())
    if len(hits) == 1:
        return hits[0] / "loader.py"
    if len(hits) > 1:
        raise ValueError(
            f"roster model {name!r} matches {len(hits)} workload directories "
            f"({[d.name for d in hits]}); name the revision in the roster -- picking one by sort order "
            f"would certify a different network than the one declared"
        )
    return None


def resolve_model_workload(entry: dict, m2m_dir: str | Path | None = None) -> str | None:
    """The model2MLIR WORKLOAD DIRECTORY behind this entry's loader, or ``None`` when it has none.

    The roster names a model (``resnet50``) and the workload directory carries the checkpoint revision
    (``resnet50_v1_5``); the entry's own ``model`` string is therefore NOT the key under which the
    model's capture declaration is filed. Resolving the loader first and reading the directory back off
    it means one resolution serves both -- the loader and its declaration always come from the same
    workload, which a second lookup by name could not guarantee.

    ``None`` for an in-repo loader (a derived micro model writes its own): it is not a workload, so it
    declares no capture environment, and inventing one for it would be a fabricated fact.
    """
    root = (Path(m2m_dir) if m2m_dir else _m2m_dir()) / "workloads"
    loader = resolve_model_loader(entry, m2m_dir)
    try:
        loader.absolute().relative_to(root.absolute())
    except ValueError:
        return None
    try:
        rel = loader.resolve().relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        raise M2MUnavailable(f"selected Model2MLIR workload loader escapes its source tree: {loader}") from exc
    return rel.parts[0] if len(rel.parts) >= 2 else None


def _model_capture_config(workload: str, m2m_dir: str | Path | None = None) -> dict:
    """Read the selected workload declaration, never an ambient Model2MLIR checkout.

    The selected source tree is byte-bound by the frozen experiment. A missing declaration
    means no settings were declared; an unreadable or malformed declaration is an error.
    """
    if not workload or Path(workload).name != workload or workload in (".", ".."):
        raise M2MUnavailable(f"invalid Model2MLIR workload name: {workload!r}")
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    directory = root / "workloads" / workload
    loader = directory / "loader.py"
    if directory.is_symlink() or not loader.is_file() or loader.is_symlink():
        raise M2MUnavailable(f"selected Model2MLIR workload loader is missing or indirect: {loader}")
    declaration = directory / "capture.toml"
    if declaration.is_symlink():
        raise M2MUnavailable(f"selected Model2MLIR capture declaration is indirect: {declaration}")
    if not declaration.exists():
        return {}
    try:
        document = tomllib.loads(declaration.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise M2MUnavailable(f"cannot read selected Model2MLIR capture declaration: {declaration}") from exc
    declared_env = document.get("env", {})
    if not isinstance(declared_env, dict) or any(not isinstance(value, str) for value in declared_env.values()):
        raise M2MUnavailable(f"selected Model2MLIR capture environment is invalid: {declaration}")
    return document


def model_capture_declaration_identity(workload: str | None, m2m_dir: str | Path | None = None) -> dict:
    """Portable byte identity of the selected workload settings (never a host path)."""
    if not workload:
        return {"status": "not_a_model2mlir_workload"}
    _model_capture_config(workload, m2m_dir)
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    relative = Path("workloads") / workload / "capture.toml"
    declaration = root / relative
    if not declaration.is_file():
        return {"status": "absent", "path": relative.as_posix()}
    raw = declaration.read_bytes()
    return {
        "status": "selected",
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def model_capture_env(workload: str | None, m2m_dir: str | Path | None = None) -> dict:
    """Read selected host locations and historical full-fidelity settings for this workload."""
    if not workload:
        return {}
    from merlin.capture import bundle as _bundle

    config = _model_capture_config(workload, m2m_dir)
    locations = {
        str(key): value
        for key, value in config.get("env", {}).items()
        if isinstance(value, str) and value and Path(value).is_dir()
    }
    return {**locations, **_bundle.full_env(workload)}


def model_capture_python(workload: str | None, m2m_dir: str | Path | None = None) -> Path | None:
    """The selected workload's pinned interpreter; reject a missing declared pin."""
    if not workload:
        return None
    config = _model_capture_config(workload, m2m_dir)
    declared = config.get("venv")
    if declared is None or declared == "":
        return None
    if not isinstance(declared, str):
        raise M2MUnavailable(f"selected Model2MLIR workload {workload!r} has an invalid venv declaration")
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    directory = root / "workloads" / workload
    value = Path(declared)
    python = (value if value.is_absolute() else directory / value) / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise M2MUnavailable(f"selected Model2MLIR workload {workload!r} pins a missing Python: {python}")
    return python


def freeze_materialized_loader_profile(selection: dict, destination: Path) -> dict:
    """Publish the exact selected adjacent loader input, for every workload role."""
    import hashlib

    selected, digest = selection.get("profile_path"), selection.get("profile_sha256")
    if selected is None and digest is None:
        return {}
    if not isinstance(selected, str) or not isinstance(digest, str) or len(digest) != 64:
        raise M2MUnavailable("materialized loader profile needs its selected path and digest")
    profile = Path(selected)
    loader = Path(selection.get("loader_path") or "")
    if (
        profile.name != "profile.json"
        or profile.parent != loader.parent
        or any(path.is_symlink() for path in (profile, *profile.parents))
        or not profile.is_file()
    ):
        raise M2MUnavailable("materialized loader profile must be the exact adjacent ordinary input")
    raw = profile.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise M2MUnavailable("materialized loader profile changed during publication")
    destination = Path(destination)
    (destination / "profile.json").write_bytes(raw)
    return {"loader_profile": "profile.json", "loader_profile_sha256": digest}


def freeze_model_loader_dependencies(
    workload: str | None, destination: Path, capture_meta: dict, m2m_dir: str | Path | None = None
) -> str | None:
    """Copy loader-imported source declared by the workload into a capsule-local Python tree.

    A frozen loader that imports an upstream module is not frozen if it still relies on an absolute
    checkout path.  Workload ``capture.toml`` already declares that upstream source.  Preserve only
    Python sources (never checkpoints, datasets, build products, or VCS metadata); the validation worker
    adds their directories to ``sys.path`` without recovering a host path from sanitized loader text.
    """
    import shutil

    from merlin.common.digest import sha256_file

    if not workload:
        return None
    root_value = _model_capture_config(workload, m2m_dir).get("upstream")
    if not root_value:
        return None
    root = Path(str(root_value))
    if not root.is_dir():
        raise M2MUnavailable(f"selected Model2MLIR workload dependency root is missing: {root}")
    sources = []
    observed_hashes = {}
    for item in capture_meta.get("loader_dependency_sources") or []:
        source = Path(str((item or {}).get("path") or ""))
        try:
            source.resolve().relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            continue
        if item.get("sha256") is not None and (not source.is_file() or source.is_symlink()):
            raise M2MUnavailable(f"loader dependency changed since capture observation: {source}")
        if source.suffix == ".py" and source.is_file() and not source.is_symlink():
            observed = item.get("sha256")
            if observed is not None:
                if sha256_file(source) != observed:
                    raise M2MUnavailable(f"loader dependency changed since capture observation: {source}")
                observed_hashes[source] = observed
            sources.append(source)
    sources = sorted(set(sources))
    if not sources:
        return None
    dependency_root = destination / "capsule.loader_deps"
    if dependency_root.exists():
        shutil.rmtree(dependency_root)
    for source in sources:
        relative = source.resolve(strict=True).relative_to(root.resolve(strict=True))
        target = dependency_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source in observed_hashes and sha256_file(target) != observed_hashes[source]:
            raise M2MUnavailable(f"loader dependency changed while staging: {source}")
    license_files = sorted(
        p for p in root.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying"))
    )
    if not license_files:
        raise M2MUnavailable(f"workload dependency source {root} declares no license; refusing to redistribute it")
    shutil.copyfile(license_files[0], dependency_root / license_files[0].name)
    return dependency_root.name


# ---- model_requirements ------------------------------------------------------------------------
# Preserve declared model accelerator, lane and semantic obligations.


def model_accelerator_demand(linalg_mlir: str, binding) -> tuple[str | None, list[str]]:
    """Derive accelerator family/classes from original source and target roles.
    Unsupported source families, absent capability or missing role census yield
    ``(None, [])``. The caller must preserve that zero *mandatory* device demand
    instead of inventing a hardware obligation. Public host programs retain
    complete source/numerical checks and optional legal accelerator lowering.
    """
    from merlin.development.targetgen.corpus.selection import classes_for_checked_source
    from merlin.target.capabilities import eligibility as _el
    from merlin.target.capabilities import semantic_families as _sf

    target = getattr(binding, "target", None)
    if not target or not linalg_mlir:
        return None, []
    try:
        cap_map = _el.capability_map_for_target(target)
    except Exception:  # noqa: BLE001 — no contract yet: demand nothing
        return None, []
    if not cap_map:
        return None, []

    in_fmt = binding.cap_dtype(binding.operand_dtype)
    try:
        # A provenance family is not authority by itself. The checked reader joins every claimed
        # contraction to a parsed, typed contraction of the same shape and refuses omissions/extras.
        demands = model_op_demands_checked(linalg_mlir, in_fmt)
    except Exception:  # noqa: BLE001 — unreadable capture: demand nothing
        return None, []

    # Ops of the model this target's hardware is declared able to run. Asked of the eligibility oracle,
    # the same independent denominator ARR uses -- not of routing, which is the thing under test.
    eligible_ops: list[str] = []
    eligible_families: list[str] = []
    for d in demands:
        if d.op in eligible_ops:
            continue
        if not d.source_formats_complete:
            continue
        desc = _el.RegionDescriptor(
            source=d.site or d.op,
            op=d.op,
            family=d.family,
            in_dtype=d.admission_input_fmt,
            weight_dtype=d.admission_weight_fmt,
            m=d.m,
            k=d.k,
            n=d.n,
            form=d.form,
        )
        if _el.is_eligible(desc, cap_map).eligible:
            eligible_ops.append(d.op)
            # ``int_matmul`` is a capture/routing spelling, not a semantic-family key. Its family
            # was independently checked against typed IR above; retain that canonical result.
            # Other operations keep the pre-existing op-vocabulary path rather than allowing an
            # unverified provenance label to create a new whole-model demand.
            family = d.family if d.family == "contraction" else _sf.from_op(d.op)
            if family:
                eligible_families.append(family)
    if not eligible_ops:
        return None, []

    # The capsule's family is the one the accelerator is actually FOR. A model is a composition and the
    # closed vocabulary has no name for it, so naming the family it owes -- rather than adding a bogus
    # ``model`` entry to the vocabulary -- is what lets the eligibility oracle resolve the capsule at all,
    # and therefore what lets must_accelerate mean anything. Contraction when present (every declared
    # matrix unit exists for it); otherwise the first eligible family, in program order.
    families = eligible_families
    if not families:
        return None, []
    family = "contraction" if "contraction" in families else families[0]

    # Classes come from the binding's OWN deriver -- the same callable every op capsule is built with, so
    # the capstone and the op capsules can never disagree about what this target's contraction sequence
    # is. (It resolves three regimes: a declared matrix unit, a self-hosted ISA taxonomy, or a RoCC
    # encoding map. Reaching past it to one of those directly is how the capstone came out empty on a
    # target whose op capsules carry the full eight-class sequence.)
    classes: list[str] = []
    for op in eligible_ops:
        try:
            got = classes_for_checked_source(binding, op=op, output_dtype=in_fmt)
        except Exception:  # noqa: BLE001 — an underivable class list demands nothing
            continue
        for c in got or ():
            if c not in classes:
                classes.append(c)
    return family, _in_issue_order(classes, binding, in_fmt)


def _in_issue_order(classes: list[str], binding, in_fmt: str) -> list[str]:
    """Put a UNION of per-op class lists back into the target's own issue order.

    The union above is built in first-encounter order, which was harmless only while every op returned
    the identical full sequence. Now that a movement op correctly owes no contraction classes, the op
    the model happens to mention first decides the order -- and a model whose first eligible op is a
    copy reported its store class BEFORE its multiply, i.e. a sequence the target cannot issue.

    The reference order is asked of the binding, not written here: a contraction is the op whose class
    list IS this target's full sequence, so its order is the canonical one, and using it keeps this
    agnostic to how the target spells or orders its classes. A class the reference does not contain is
    kept, at the end -- an unplaceable class is not evidence it is unnecessary, and dropping it would be
    the silent drop the parsing rule forbids.
    """
    try:
        reference = list(binding.classes_for(op="matmul", output_dtype=in_fmt) or ())
    except Exception:  # noqa: BLE001 -- no reference: keep the order we have
        return classes
    rank = {c: i for i, c in enumerate(reference)}
    return sorted(classes, key=lambda c: (rank.get(c, len(rank)), classes.index(c)))


def _model_semantic_block(entry: dict, family: str | None, classes) -> dict:
    """The generalization-intent block for a whole-model capsule.

    Three things are decided here, and the profile entry's ``generalization`` block overrides each --
    the same authored-override channel ``corpus_spec._semantic_block`` already honours for every other
    capsule kind. The model path used to ignore it entirely, which caused two failures at once:

    * ``must_accelerate`` was FORCED false by the mere presence of ``lanes.require``. The reasoning is
      sound for a whole real model, whose norms have nowhere but the host to go -- but it is wrong for a
      capsule whose subject is the seam itself. The host-island capsule needs BOTH: its two int8 GEMMs
      must reach the mesh (``must_accelerate``) *and* its LayerNorm island must land on the host lane
      (``lanes.require``). Forcing them apart made declaring the lane contract silently WEAKEN the mesh
      assertion, so the automatic withholding is now a default an author can override, not a mandate.
    * ``not_asserted_reason`` could only ever say "the demand could not be derived". A capsule that
      withholds the assertion ON PURPOSE had no way to say so from the profile, so the reason was
      hand-written into the generated capsule and deleted by the next regeneration.
    """
    grounded = bool(family and classes)
    authored = entry.get("generalization")
    authored = dict(authored) if isinstance(authored, dict) else {}
    interop = bool((entry.get("lanes") or {}).get("require"))
    block: dict = {}
    if family:
        block["semantic_family"] = family
    block["generalization_axis"] = authored.get("generalization_axis", "model")
    block["must_accelerate"] = bool(authored.get("must_accelerate", grounded and not interop))
    # A WITHHELD ASSERTION ALWAYS SAYS WHY. The two reasons are different facts and must not be
    # conflated: "we could not derive the demand" is about our own reach, while "host-lane work is the
    # subject" is about the capsule. Deriving both means a synthesized interop capsule carries the
    # explanation without anyone typing it, and the profile text is an override rather than the only
    # source -- silence here reads as an author who simply never made a claim.
    reason = authored.get("not_asserted_reason")
    if not reason and not grounded:
        reason = (
            "the accelerator demand could not be derived from this model's linalg against this "
            "target's declared capabilities and role census, so must_accelerate is withheld "
            "rather than asserted ungrounded"
        )
    if not reason and interop:
        reason = (
            "interop capsule: composition across lanes is the behaviour under test, so host-lane "
            "work is expected and required. The accelerator demand is expressed as lanes.require, "
            "which fails unless every named lane actually carried work."
        )
    if reason and not block["must_accelerate"]:
        block["not_asserted_reason"] = reason
    block["eligible"] = authored.get("eligible", "auto")
    return block


def _checked_lanes(entry: dict, binding) -> list[str]:
    """The lanes an interop capsule requires, REFUSED at generation time if this target cannot populate one.

    A required lane is a bar the submission must clear. A bar the target's declared compute units make
    unreachable is not a hard capability test, it is a wall: no compiler can put work on a lane the router
    has nothing to route there. Measured on the corpus this check was written for -- a capsule required
    ``in_contract_vector_scalar`` on a target whose every declared unit is a mesh kind, so the lane was
    empty by construction and the capsule was unpassable however good the backend was.

    Raising here (rather than shipping the capsule and failing it forever) keeps the failure where someone
    can fix it: at authoring time, naming the lane and what the target actually offers."""
    want = [str(x) for x in (entry.get("lanes") or {}).get("require") or []]
    target = getattr(binding, "target", None)
    if not target:
        return want  # no target to judge against: leave the declaration
    try:
        from merlin.target.capabilities.routing import reachable_lanes

        have = reachable_lanes(target)
    except Exception:  # noqa: BLE001 - never block generation on this
        return want
    missing = [ln for ln in want if ln not in have]
    if missing:
        raise ValueError(
            f"capsule {entry.get('name')!r} requires lane(s) {missing} that target {target!r} cannot "
            f"populate (its declared compute units make {sorted(have)} reachable). A required lane the "
            f"router can put nothing on is unpassable by construction -- declare a lane this target owns, "
            f"or give the target a compute unit that serves the one you want"
        )
    return want
