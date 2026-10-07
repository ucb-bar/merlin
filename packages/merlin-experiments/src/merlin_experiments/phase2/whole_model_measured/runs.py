"""A ``whole_model_measured`` run on disk: prepared, resumed and relaunched without losing what it is.

Layout (the phase-run convention, :func:`merlin.common.artifacts.start_phase_run`)::

    out/runs/<target>/phase2/<TS>_<method>_<sha7>/
        run.json                          what this run IS: target, method, roles, lineage
        resumed_seed.json                 where its seed came from (``RESUMED_SEED_SCHEMA``)
        whole_model_objective_config.json the objective config, read-only (its sha256 is the launch pin)
        inputs/<name>/...                 large declared inputs, frozen by CONTENT (hard links)
        seed/submission/                  the seed package, as committed below
        workspace/                        the agent's writable candidate
        oot/                              harness-owned git repo: one commit per candidate
        stage/                            the session loop's records

WHAT A RELAUNCH MUST KEEP.  The METHOD (it names the run and carries the no-FSM naming) and the
declared ``prohibited_instruction_roles`` are read from the run being resumed, never from a literal:
a relaunch that spelled its method by hand once dropped ``_nofsm`` and nearly started a no-FSM
campaign without its prohibition.  :func:`resume` refuses a method or role change outright; a store
the resume would move is refused unless the caller says why it may.

LARGE INPUTS ARE LINKED, NEVER COPIED.  A model capsule's weights are gigabytes and identical from one
run to the next; copying them per run once cost 35.6 GB of duplicates.  Declared inputs are frozen
through :mod:`merlin.common.content_store` -- one read-only object per distinct content, hard-linked
into each run -- which keeps the freeze (an in-place edit of the source cannot reach the frozen
bytes) without the copy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import MODE
from . import capabilities as CAP
from . import config as CFG
from .identity import package_digest, program_digest, read_json, write_json_atomic

RUN_SCHEMA = "merlin.phase2.whole_model_measured.run.v1"
RESUMED_SEED_SCHEMA = "merlin.phase2.whole_model_measured.resumed_seed.v1"
CONFIG_NAME = "whole_model_objective_config.json"
#: A token in the objective config's build options replaced by a frozen input's path.
INPUT_TOKEN = "{input:"


class RunError(RuntimeError):
    """A run cannot be prepared or resumed as declared."""


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    config_path: Path
    config_sha256: str
    method: str
    roles: tuple[str, ...]
    seed_package_sha256: str
    store_roots: Mapping[str, str]
    #: What each section's machine lacks against the others its registry declares (:mod:`.capabilities`).
    machine_warnings: tuple[str, ...] = ()


def _default_run_factory(*, target: str, method: str) -> Path:
    from merlin.common.artifacts import start_phase_run

    return Path(start_phase_run(target=target, phase=2, method=method).run_dir)


def freeze_inputs(run_dir: Path, inputs: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Each declared input under ``run_dir/inputs/<name>/``, placed by content (hard links to the store's
    read-only objects; a plain copy only where the store is disabled or on another filesystem)."""
    from merlin.common import content_store

    root = content_store.store_root()
    frozen: dict[str, dict[str, Any]] = {}
    for name, source in inputs.items():
        if not name or "/" in name or name in (".", ".."):
            raise RunError(f"an input name is one path component, not {name!r}")
        source = Path(source)
        destination = Path(run_dir) / "inputs" / name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            content_store.place_tree(source, destination, root)
        elif source.is_file():
            content_store.place_file(source, destination, root)
        else:
            raise RunError(f"declared input {name!r} does not exist: {source}")
        files = [p for p in destination.rglob("*") if p.is_file()] if destination.is_dir() else [destination]
        frozen[name] = {
            "source": str(source),
            "frozen": str(destination),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "linked_files": sum(1 for p in files if content_store.is_shared(p)),
        }
    return frozen


def _substitute(value: Any, frozen: Mapping[str, Mapping[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith(INPUT_TOKEN) and value.endswith("}"):
        name = value[len(INPUT_TOKEN) : -1]
        if name not in frozen:
            raise RunError(f"the config names input {name!r}, which the run does not declare")
        return frozen[name]["frozen"]
    if isinstance(value, Mapping):
        return {k: _substitute(v, frozen) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, frozen) for v in value]
    return value


def _require_frozen_mechanism_inputs(
    config: Mapping[str, Any], frozen: Mapping[str, Mapping[str, Any]], resumed_from: Path | None
) -> None:
    """A derived mechanism decision may inspect only a declared, frozen capsule."""
    if config.get("mechanism_policy") != CFG.DERIVED_MECHANISMS:
        return
    allowed = {Path(row["frozen"]).resolve() for row in frozen.values()}
    if resumed_from is not None:
        previous = read_json(Path(resumed_from) / "resumed_seed.json") or {}
        allowed.update(Path(row["frozen"]).resolve() for row in (previous.get("inputs") or {}).values())
    sections = [config.get(name) for name in CFG.CANDIDATE_SECTIONS]
    sections += list((config.get("held_out") or {}).values())
    for section in sections:
        if not section:
            continue
        capsule = (section.get("build_options") or {}).get("model_capsule")
        if not isinstance(capsule, str) or Path(capsule).resolve() not in allowed:
            raise RunError("derived mechanisms require every model_capsule to be a declared frozen input")


def _copy_package(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def prepare(
    *,
    target: str,
    method: str,
    objective_config: Mapping[str, Any],
    seed: Path,
    prohibited_roles: Sequence[str],
    inputs: Mapping[str, Path] | None = None,
    phase1_oot: Path | None = None,
    oot_ref: str = "frozen",
    resumed_from: Path | None = None,
    why: str,
    allow_new_store: str | None = None,
    run_factory: Callable[..., Path] = _default_run_factory,
    oot: Any = None,
    environment: Mapping[str, str] | None = None,
    import_evidence: Path | None = None,
    phase0_manifest: Path | None = None,
) -> PreparedRun:
    """Prepare one run: freeze its inputs by content, copy and commit its seed, write its objective
    config (policy stamped in, read-only) and the records that say what it is and where it came from.

    ``import_evidence`` is the measurement result (``result.json``) of a seed IMPORTED from outside a
    Phase 1 freeze -- another line's store, a champion measured elsewhere.  Its lineage is recorded as
    exactly that (``lineage_kind: imported``, ``frozen: false``) with the evidence by content, and the
    seed's bytes must be the bytes that evidence measured; an imported seed is never presented as a
    Phase 1 freeze.

    ``phase0_manifest`` is the sealed Phase 0 corpus manifest whose ``instruction_policy`` the declared
    roles are held to (:func:`.config.seal_policy`); a run that declares roles and cannot find an
    enforceable sealed policy is refused here, before anything is frozen."""
    if not str(why or "").strip():
        raise RunError("a prepared run states why it exists")
    if not method or "/" in method:
        raise RunError(f"a run's method is one path component, not {method!r}")
    roles = [str(r) for r in prohibited_roles]
    seed = Path(seed)
    if not seed.is_dir():
        raise RunError(f"the seed package {seed} is not a directory")
    previous = read_json(Path(resumed_from) / CONFIG_NAME) if resumed_from is not None else None
    config = CFG.with_policy(objective_config, roles)
    try:
        config = CFG.seal_policy(config, target=target, manifest=phase0_manifest)
        CFG.check_policy(config)
    except (CFG.ConfigError, OSError, ValueError) as exc:
        raise RunError(f"the run's instruction rule is not enforceable: {exc}") from exc
    run_dir = Path(run_factory(target=target, method=method))
    frozen = freeze_inputs(run_dir, dict(inputs or {}))
    config = _substitute(config, frozen)
    _require_frozen_mechanism_inputs(config, frozen, resumed_from)
    config = CFG.prepare_document(config, target=target)
    config = CFG.seal_exactness(config, target=target)
    CFG.check_policy(config)
    roots = {k: str(v) for k, v in CFG.store_roots(config, environment=environment).items()}
    moved = {}
    if previous is not None:
        before = {k: str(v) for k, v in CFG.store_roots(previous, environment=environment).items()}
        moved = {k: (before.get(k), roots.get(k)) for k in set(before) | set(roots) if before.get(k) != roots.get(k)}
        if moved and not str(allow_new_store or "").strip():
            raise RunError(
                f"this resume moves the measurement store ({sorted(moved)}): a new builder, machine or build "
                "options opens a new content-keyed store and loses the history; pass allow_new_store with why"
            )
    imported = _imported_lineage(import_evidence, seed) if import_evidence is not None else None
    if imported is not None and phase1_oot is not None:
        raise RunError("a seed is either imported or a Phase 1 freeze's, never both")
    submission = run_dir / "seed" / "submission"
    _copy_package(seed, submission)
    workspace = run_dir / "workspace"
    _copy_package(seed, workspace)
    oot_record = None
    if oot is not None:
        repo = run_dir / "oot"
        if phase1_oot is not None:
            oot.init_from(repo, Path(phase1_oot), ref=oot_ref, sandbox_roots=(workspace,))
        else:
            oot.init(repo, sandbox_roots=(workspace,))
        from merlin.common.artifacts import utc_stamp

        commit = oot.commit_candidate(
            repo,
            submission,
            label="imported seed" if imported is not None else "seed",
            when=utc_stamp(),
            run_id=run_dir.name,
            **(
                {"metadata": {"lineage_kind": "imported", "evidence_sha256": imported["evidence"]["sha256"]}}
                if imported
                else {}
            ),
        )
        oot.verify(repo, commit.commit, package_digest(submission))
        oot_record = commit.as_record()
    capabilities = machine_capabilities(config, environment=environment)
    write_json_atomic(run_dir / CAP.RECORD, capabilities)
    config_path = run_dir / CONFIG_NAME
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_path.chmod(0o444)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    seed_sha = package_digest(submission)
    lineage = read_json(Path(resumed_from) / "resumed_seed.json") if resumed_from is not None else None
    write_json_atomic(
        run_dir / "resumed_seed.json",
        {
            "schema": RESUMED_SEED_SCHEMA,
            "method": method,
            "prohibited_instruction_roles": roles,
            "resumed_from_run": str(resumed_from) if resumed_from is not None else None,
            "resumed_from_workspace": str(seed),
            "seed_package_sha256": seed_sha,
            "seed_program_sha256": program_digest(submission),
            "store_roots": roots,
            "store_moved": {k: {"from": a, "to": b} for k, (a, b) in moved.items()} or None,
            "store_moved_because": str(allow_new_store).strip() if moved else None,
            "inputs": frozen,
            "oot_seed_commit": oot_record,
            "oot_source": {"repo": str(phase1_oot), "ref": oot_ref} if phase1_oot is not None else None,
            "lineage": lineage,
            "lineage_kind": _lineage_kind(imported=imported, resumed_from=resumed_from, phase1_oot=phase1_oot),
            "origin_kind": _origin_kind(lineage)
            or _lineage_kind(imported=imported, resumed_from=None, phase1_oot=phase1_oot),
            "frozen": phase1_oot is not None and imported is None and resumed_from is None,
            "imported": imported,
            "why": str(why).strip(),
        },
    )
    write_json_atomic(
        run_dir / "run.json",
        {
            "schema": RUN_SCHEMA,
            "mode": MODE,
            "target": target,
            "method": method,
            "prohibited_instruction_roles": roles,
            "instruction_policy_source": (config.get(CFG.SEALED_POLICY) or {}).get("sealed_source"),
            "config_sha256": config_sha256,
            "resumed_from_run": str(resumed_from) if resumed_from is not None else None,
        },
    )
    return PreparedRun(
        run_dir,
        config_path,
        config_sha256,
        method,
        tuple(roles),
        seed_sha,
        roots,
        tuple(capabilities["warnings"]),
    )


def machine_capabilities(config: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Each candidate section's machine capability report (:func:`.capabilities.section_report`) and
    every warning they raise, prefixed with the section -- the record a launch writes and prints."""
    sections = {
        name: CAP.section_report(config[name], environment=environment)
        for name in CFG.CANDIDATE_SECTIONS
        if config.get(name)
    }
    warnings = [
        f"{name}: {warning}" for name, document in sections.items() for warning in document.get("warnings") or ()
    ]
    return {"schema": CAP.SCHEMA, "sections": sections, "warnings": warnings}


def _lineage_kind(*, imported: Any, resumed_from: Path | None, phase1_oot: Path | None) -> str:
    """How THIS run's seed came to be.  A resume carries the previous run's history over (its oot/ is
    the source), which is not a Phase 1 freeze: naming it one would let an imported seed's descendant
    read as frozen two relaunches later."""
    if imported is not None:
        return "imported"
    if resumed_from is not None:
        return "resumed"
    return "phase1_freeze" if phase1_oot is not None else "unrecorded"


def _origin_kind(lineage: Mapping[str, Any] | None) -> str | None:
    """The kind of the chain's FIRST seed, walked back through every resume (deepest record first, so
    a link that misnamed its own kind cannot rename where the chain began)."""
    if not lineage:
        return None
    return _origin_kind(lineage.get("lineage")) or lineage.get("origin_kind") or lineage.get("lineage_kind")


def _imported_lineage(evidence: Path, seed: Path) -> dict[str, Any]:
    """What an imported seed is, by content: the measurement it carries, refused unless that measurement
    is of these exact bytes (its own package digest, or its program digest when the store aliased it)."""
    evidence = Path(evidence)
    document = read_json(evidence)
    if not isinstance(document, Mapping):
        raise RunError(f"the import evidence {evidence} is not a measurement result")
    digest, program = package_digest(Path(seed)), program_digest(Path(seed))
    measured = document.get("package_sha256")
    if measured not in (digest,) and document.get("program_sha256") != program:
        raise RunError(
            f"the import evidence measured {str(measured)[:12]}, not this seed's bytes {digest[:12]}; an imported "
            "seed is recorded only with the measurement of its own bytes"
        )
    device = document.get("device") or {}
    return {
        "frozen": False,
        "evidence": {
            "path": str(evidence.resolve()),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "package_sha256": measured,
            "timing_status": document.get("timing_status"),
            "objective_cycles": document.get("objective_cycles"),
            "device": device.get("artifact"),
            "finished_at": document.get("finished_at"),
        },
        "note": "an imported seed: measured outside this line's Phase 1, never a Phase 1 freeze",
    }


def latest_workspace(run_dir: Path) -> Path:
    """The candidate a relaunch continues from: the run's workspace, else its seed."""
    for candidate in (Path(run_dir) / "workspace", Path(run_dir) / "seed" / "submission"):
        if candidate.is_dir():
            return candidate
    raise RunError(f"{run_dir} has neither a workspace nor a seed to continue from")


def resume(
    previous: Path,
    *,
    why: str,
    workspace: Path | None = None,
    method: str | None = None,
    prohibited_roles: Sequence[str] | None = None,
    objective_config: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Path] | None = None,
    allow_new_store: str | None = None,
    run_factory: Callable[..., Path] = _default_run_factory,
    oot: Any = None,
    environment: Mapping[str, str] | None = None,
    phase0_manifest: Path | None = None,
) -> PreparedRun:
    """Prepare the next run of ``previous``: its method, roles, target and config carried over, its
    latest workspace as the seed, its store kept.  A caller that names a DIFFERENT method or roles is
    refused -- a relaunch is the same experiment, and a changed policy is a new one."""
    previous = Path(previous)
    record = read_json(previous / "run.json")
    if not record or record.get("schema") != RUN_SCHEMA:
        raise RunError(f"{previous} is not a prepared {MODE} run (no run.json)")
    kept_method = str(record["method"])
    kept_roles = [str(r) for r in record.get("prohibited_instruction_roles") or ()]
    if method is not None and method != kept_method:
        raise RunError(f"the run being resumed is method {kept_method!r}; a relaunch may not rename it {method!r}")
    if prohibited_roles is not None and sorted(prohibited_roles) != sorted(kept_roles):
        raise RunError(
            f"the run being resumed prohibits {kept_roles!r}; "
            f"a relaunch may not change that to {list(prohibited_roles)!r}"
        )
    config = objective_config if objective_config is not None else read_json(previous / CONFIG_NAME)
    if not config:
        raise RunError(f"{previous} has no readable {CONFIG_NAME}")
    previous_oot, ref = None, "frozen"
    if oot is not None and (previous / "oot").is_dir():
        # THE HISTORY CONTINUES: the previous run's candidate commits are carried over by tagging its
        # head (a new, immutable tag) and starting the new repository from that tag.  The tag names the
        # run and the commit it carries, never a clock: the new repository inherits the previous one's
        # tags, so a stamp-named tag collides with itself when the next relaunch lands in the same second.
        head = oot.resolve(previous / "oot", "HEAD")
        ref = f"relaunched/{previous.name}/{head[:12]}"
        oot.tag(previous / "oot", ref, head)
        previous_oot = previous / "oot"
    carried_inputs = dict(inputs or {})
    for name, row in ((read_json(previous / "resumed_seed.json") or {}).get("inputs") or {}).items():
        carried_inputs.setdefault(name, Path(row["frozen"]))
    return prepare(
        target=str(record["target"]),
        method=kept_method,
        objective_config=config,
        seed=workspace or latest_workspace(previous),
        prohibited_roles=kept_roles,
        inputs=carried_inputs,
        phase1_oot=previous_oot,
        oot_ref=ref,
        resumed_from=previous,
        why=why,
        allow_new_store=allow_new_store,
        run_factory=run_factory,
        oot=oot,
        environment=environment,
        phase0_manifest=phase0_manifest,
    )


__all__ = [
    "CONFIG_NAME",
    "INPUT_TOKEN",
    "PreparedRun",
    "RESUMED_SEED_SCHEMA",
    "RUN_SCHEMA",
    "RunError",
    "freeze_inputs",
    "latest_workspace",
    "prepare",
    "resume",
]
