"""Export a phase-2 champion from its ``best`` tag into ``out/artifacts/targets/<target>/champions/``.

The champion is the tree the harness committed and measured, not whatever a workspace holds now:
the ``best`` commit is exported with ``git archive`` semantics (:func:`merlin.common.oot_repo.export`)
and its tree digest must equal the digest the measurements were recorded against. The standalone
layout is the one the publish bridge already produces -- :func:`publish.assemble_repo_tree` preserves
the payload bytes and :func:`publish.embed_provenance` adds ``.merlin/{manifest.yaml,provenance.yaml,
certification.yaml,CHAMPION}`` -- so a champion directory is exactly what ``merlin-target-publish``
would push. This module adds the four JSON records the payload-scoped publish layer cannot know:

* ``provenance.json`` -- lineage to the phase-1 run and its ``frozen`` commit, the phase-2 run and
  ``best`` commit, the corpus seal digest and the phase-0 evidence digest;
* ``measurements.json`` -- FireSim cycles with the machine, the parameter header and the vendor
  control measured in the same batch;
* ``certification.json`` -- the GSIM certification;
* ``isa_prohibition.json`` -- the whole-ELF prohibited-instruction scan, naming the instructions it
  prohibited (a clean verdict over an empty prohibited set is refused).

Every field listed in :data:`REQUIRED` is required and checked, never defaulted: a champion whose
cycles came without their machine, or whose scan was not clean, is refused rather than exported with
a gap a later reader would fill in by assumption. The export is retention-pinned.

Three optional provenance blocks describe a lineage honestly where it does not fit that mould:

* ``lineage.legacy`` -- a lineage that predates the sealed phase 0 has no corpus seal and no phase-0
  evidence digest. The block (see :func:`legacy_problems`) may stand in for either digest, but only
  where the caller wrote the digest as an explicit ``null``; an absent digest is still refused, and
  the export prints ``UNSEALED LEGACY LINEAGE`` in ``.merlin/CHAMPION`` and ``MERLIN_PUBLICATION.md``.
* ``composition`` -- a champion composed from other packages rather than authored in one session says
  so (see :func:`composition`), and its base must lie in the exported history.
* a ``best`` from a history :func:`merlin.common.oot_repo.reconstruct` rebuilt from stored package
  bytes is recorded ``reconstructed: true``, and the reconstruction's ``frozen`` must be the declared one.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..common import oot_repo, paths
from ..common.jsonio import write_pretty_json
from ..common.tree_hash import hash_tree
from ..common.yaml import load_yaml
from . import package_records
from . import publish as pub

SCHEMA = "merlin_champion_v1"
CHAMPIONS_DIR = "champions"
#: The storage contract's ``product_roots`` entry for the champions' concern.
CHAMPIONS_HOME = "target-champions"
RECORDS = ("provenance", "certification", "measurements", "isa_prohibition")
#: The publish layer's own files, which make the export the standalone layout.
PUBLISH_LAYER = (".merlin/manifest.yaml", ".merlin/provenance.yaml", ".merlin/certification.yaml", ".merlin/CHAMPION")
PUBLICATION_NOTE = "MERLIN_PUBLICATION.md"

#: ``record -> dotted field -> predicate name``; see :func:`_check`.
REQUIRED: dict[str, dict[str, str]] = {
    "provenance": {
        "phase1.run": "text",
        "phase1.frozen_commit": "commit",
        "corpus_seal_digest": "sha256",
        "phase0_evidence_digest": "sha256",
    },
    "measurements": {
        "package_digest": "sha256",
        "firesim.cycles": "positive",
        "firesim.machine": "text",
        "firesim.header": "text",
        "firesim.control.in_batch": "true",
    },
    "certification": {"gsim.verdict": "pass"},
    # A clean verdict under a rule that forbids nothing is no verdict: the scan must name what it held
    # the program to, at least one instruction (``{selector: name}``, the scanner's own record).
    "isa_prohibition": {
        "scope": "whole_elf",
        "verdict": "clean",
        "prohibited_roles": "nonempty_list",
        "prohibited_instructions": "nonempty_mapping",
    },
}

#: A ``provenance.lineage.legacy`` block names a lineage older than the sealed phase 0.
LEGACY_PREDATES = "sealed phase 0"
LEGACY_MARK = "UNSEALED LEGACY LINEAGE"
#: The provenance digests a legacy block may stand in for -- each only where it is an explicit null.
LEGACY_STANDS_IN_FOR = ("corpus_seal_digest", "phase0_evidence_digest")
LEGACY_REQUIRED: dict[str, str] = {
    "reason": "text",
    "predates": LEGACY_PREDATES,
    "bundle_manifest_sha": "sha256",
    "bundle_name": "text",
}


class ChampionError(RuntimeError):
    """A champion could not be exported: missing evidence, a digest mismatch or a broken lineage."""


def champions_root(target: str, *, artifacts_root: str | Path | None = None) -> Path:
    from ..common.artifacts import declared_home

    home = declared_home(CHAMPIONS_HOME, artifacts_root=artifacts_root)
    return home / package_records.component(target) / CHAMPIONS_DIR


def champion_dir(target: str, package_id: str, *, artifacts_root: str | Path | None = None) -> Path:
    return champions_root(target, artifacts_root=artifacts_root) / package_records.component(package_id)


def _field(document: dict, dotted: str):
    value: Any = document
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _check(rule: str, value) -> bool:
    if rule == "text":
        return isinstance(value, str) and bool(value.strip())
    if rule == "commit":
        return _hex(value, 40)
    if rule == "sha256":
        return _hex(value, 64)
    if rule == "positive":
        return type(value) is int and value > 0
    if rule == "true":
        return value is True
    if rule == "list":
        return isinstance(value, list) and all(isinstance(v, str) and v for v in value)
    if rule == "nonempty_list":
        return _check("list", value) and bool(value)
    if rule == "nonempty_mapping":
        return (
            isinstance(value, dict)
            and bool(value)
            and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items())
        )
    return value == rule  # a literal the field must equal (a verdict, a scope)


def missing_evidence(records: dict[str, dict]) -> list[str]:
    """Every required field that is absent or does not hold what it must, as ``record.field``."""
    problems = []
    stood_in = {f"provenance.{name}" for name in legacy_stands_in_for(records.get("provenance"))}
    for record, rules in REQUIRED.items():
        document = records.get(record)
        if not isinstance(document, dict):
            problems.append(f"{record}: not supplied")
            continue
        problems += [
            f"{record}.{name}"
            for name, rule in rules.items()
            if not _check(rule, _field(document, name)) and f"{record}.{name}" not in stood_in
        ]
    return problems + legacy_problems(records.get("provenance")) + composition_problems(records.get("provenance"))


def _legacy_block(provenance) -> Any:
    lineage = provenance.get("lineage") if isinstance(provenance, dict) else None
    return lineage.get("legacy") if isinstance(lineage, dict) else None


def legacy_stands_in_for(provenance) -> list[str]:
    """The digests a legacy block stands in for: those the caller wrote as an explicit ``null``.

    An absent key is not a null. A digest that was simply left out stays missing, so the legacy block
    can never be what fills a gap nobody decided to leave.
    """
    if _legacy_block(provenance) is None:
        return []
    return [name for name in LEGACY_STANDS_IN_FOR if name in provenance and provenance[name] is None]


def legacy_problems(provenance) -> list[str]:
    """What keeps ``provenance.lineage.legacy`` from being an honest unsealed-lineage record.

    The block is ``{reason, predates: "sealed phase 0", bundle_manifest_sha, bundle_name, run_dirs,
    dates, hops}``: why no seal exists, the input bundle the lineage was graded on (by name and by the
    sha256 of its manifest), the run directories it came from, the dates that place it before the seal
    (``{event: date}``), and per hop the ``driver`` and ``model`` that authored it. Empty when there is
    no block.
    """
    lineage = provenance.get("lineage") if isinstance(provenance, dict) else None
    if lineage is not None and not isinstance(lineage, dict):
        return ["provenance.lineage: not a mapping"]
    if not isinstance(lineage, dict) or "legacy" not in lineage:
        return []
    prefix = "provenance.lineage.legacy"
    block = lineage["legacy"]
    if not isinstance(block, dict):
        return [f"{prefix}: not a mapping"]
    problems = [f"{prefix}.{name}" for name, rule in LEGACY_REQUIRED.items() if not _check(rule, block.get(name))]
    run_dirs = block.get("run_dirs")
    if not (isinstance(run_dirs, list) and run_dirs and all(_check("text", d) for d in run_dirs)):
        problems.append(f"{prefix}.run_dirs")
    dates = block.get("dates")
    if not (
        isinstance(dates, dict) and dates and all(_check("text", k) and _check("text", v) for k, v in dates.items())
    ):
        problems.append(f"{prefix}.dates")
    hops = block.get("hops")
    if not (isinstance(hops, list) and hops):
        problems.append(f"{prefix}.hops")
    else:
        for index, hop in enumerate(hops):
            for name in ("driver", "model"):
                if not (isinstance(hop, dict) and _check("text", hop.get(name))):
                    problems.append(f"{prefix}.hops[{index}].{name}")
    if not legacy_stands_in_for(provenance):
        problems.append(f"{prefix}: stands in for no digest (none of {', '.join(LEGACY_STANDS_IN_FOR)} is null)")
    return problems


def composition(parts: dict[str, str], base: str, *, method: str = "three-way merge", **detail: Any) -> dict:
    """A ``provenance.composition`` block, with its note written from the facts it records.

    ``parts`` maps a label (the cell a winner won) to the package digest merged in; ``base`` is the
    digest they were merged onto. ``detail`` (who composed it, with what tool, whether anything was
    edited by hand) is recorded as given.
    """
    winners = ", ".join(f"{label} {digest[:8]}" for label, digest in parts.items())
    return {
        "method": method,
        "note": f"composed by {method} of cell winners {winners} onto {base[:8]}",
        "base": base,
        "parts": dict(parts),
        **detail,
    }


def composition_problems(provenance) -> list[str]:
    """What keeps ``provenance.composition`` from saying how the champion was composed (empty if absent)."""
    block = provenance.get("composition") if isinstance(provenance, dict) else None
    if block is None:
        return []
    prefix = "provenance.composition"
    if not isinstance(block, dict):
        return [f"{prefix}: not a mapping"]
    problems = [f"{prefix}.{name}" for name in ("method", "note") if not _check("text", block.get(name))]
    if not _check("sha256", block.get("base")):
        problems.append(f"{prefix}.base")
    parts = block.get("parts")
    if not (isinstance(parts, dict) and parts and all(_check("sha256", v) for v in parts.values())):
        problems.append(f"{prefix}.parts")
    return problems


def exactness_problems(measurements: dict[str, Any]) -> list[str]:
    """What keeps the measurements' exactness record from saying which contract the cycles were graded
    under: a champion's correctness is only as strict as that contract, so an export without it -- or
    with a contract nobody can identify -- is refused rather than read as bit-exact."""
    record = measurements.get("exactness")
    if not isinstance(record, dict):
        return ["measurements.exactness (the measurement recorded no exactness contract)"]
    problems = []
    if not _check("sha256", record.get("contract_sha256")):
        problems.append("measurements.exactness.contract_sha256")
    if not _check("text", record.get("label")) or record.get("label") == "unrecorded":
        problems.append("measurements.exactness.label")
    return problems


def layout_problems(root: Path) -> list[str]:
    """What keeps ``root`` from being a standalone champion tree (empty when it is one)."""
    root = Path(root)
    problems = []
    manifest = load_yaml(root / "manifest.yaml") if (root / "manifest.yaml").is_file() else None
    if not isinstance(manifest, dict):
        return ["manifest.yaml is missing or not a mapping"]
    tool = (manifest.get("entrypoints") or {}).get("tool") if isinstance(manifest.get("entrypoints"), dict) else None
    if not isinstance(tool, str) or not (root / tool).is_file():
        problems.append(f"entry point {tool!r} is not a file at the root")
    for rel in (*PUBLISH_LAYER, *(f".merlin/{name}.json" for name in RECORDS), PUBLICATION_NOTE):
        if not (root / rel).is_file():
            problems.append(f"{rel} is missing")
    if (root / ".git").exists():
        problems.append(".git must not be exported")
    return problems


def export_champion(
    target: str,
    repo: str | Path,
    *,
    package_id: str,
    provenance: dict[str, Any],
    certification: dict[str, Any],
    measurements: dict[str, Any],
    isa_prohibition: dict[str, Any],
    rev: str = oot_repo.BEST_TAG,
    phase2_run: str | Path | None = None,
    stage_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    pin: bool = True,
    update_index: bool = True,
) -> Path:
    """Export ``rev`` (default ``best``) of a phase-2 OOT repo as a retention-pinned champion.

    ``repo`` is the phase-2 run's ``oot/``; ``phase2_run`` defaults to its parent. The payload is
    staged under ``<phase2 run>/exports/<commit12>/`` (``stage_root`` overrides), which is also the
    ``source_package`` the publish layer records.
    """
    for name, value in (("target", target), ("package_id", package_id)):
        try:
            package_records.component(value)
        except ValueError as exc:
            raise ChampionError(f"{name} must be a single path component: {value!r}") from exc
    repo = Path(repo).absolute()
    try:
        commit = oot_repo.resolve(repo, rev)
        digest = oot_repo.tree_digest(repo, commit)
    except oot_repo.OotRepoError as exc:
        raise ChampionError(str(exc)) from exc
    records = {
        "provenance": dict(provenance),
        "certification": dict(certification),
        "measurements": dict(measurements),
        "isa_prohibition": dict(isa_prohibition),
    }
    problems = missing_evidence(records) + exactness_problems(records["measurements"])
    if problems:
        raise ChampionError(f"champion evidence is incomplete or not passing: {', '.join(problems)}")
    if measurements["package_digest"] != digest:
        raise ChampionError(
            f"{rev} holds package {digest}, but the measurements are of {measurements['package_digest']}"
        )
    frozen = provenance["phase1"]["frozen_commit"]
    try:
        if not oot_repo.is_ancestor(repo, frozen, commit):
            raise ChampionError(f"{rev} does not descend from the phase-1 frozen commit {frozen}")
        origin = oot_repo.origin(repo)
    except oot_repo.OotRepoError as exc:
        raise ChampionError(f"lineage to the phase-1 frozen commit cannot be established: {exc}") from exc
    if origin is not None and origin.get("commit") != frozen:
        raise ChampionError(f"the repo was started from {origin.get('commit')}, not the declared frozen {frozen}")
    rebuilt = oot_repo.reconstruction(repo)
    if rebuilt is not None and rebuilt.get("frozen") != frozen:
        raise ChampionError(f"the history was reconstructed from frozen {rebuilt.get('frozen')}, not {frozen}")
    composed = records["provenance"].get("composition")
    if composed is not None and composed["base"] not in {c.package_digest for c in oot_repo.history(repo, commit)}:
        raise ChampionError(f"the composition base {composed['base']} is not a package in {rev}'s history")

    destination = champion_dir(target, package_id, artifacts_root=artifacts_root)
    if destination.exists() or destination.is_symlink():
        raise ChampionError(f"champion already exported: {destination}")
    run_dir = Path(phase2_run).absolute() if phase2_run else repo.parent
    payload = Path(stage_root or run_dir / "exports").absolute() / commit[:12]
    if payload.exists():
        if hash_tree(payload).get("sha256") != digest:
            raise ChampionError(f"existing export {payload} is not {rev}'s tree")
    else:
        oot_repo.export(repo, commit, payload)
    manifest = load_yaml(payload / "manifest.yaml") if (payload / "manifest.yaml").is_file() else None
    if not isinstance(manifest, dict):
        raise ChampionError(f"{rev} has no manifest.yaml mapping at its root")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{package_id}.staging-{os.urandom(4).hex()}"
    try:
        try:
            selection = pub._build_selection(target, payload, manifest)
            pub.assemble_repo_tree(selection, staging, layout_version=pub.LAYOUT_VERSION)
            pub.embed_provenance(staging, selection)
        except pub.PublishError as exc:
            raise ChampionError(f"publish layer refused the champion: {exc}") from exc
        lineage = {
            **records["provenance"],
            "schema": SCHEMA,
            "target": target,
            "package_id": package_id,
            "package_digest": digest,
            "phase2": {
                **(records["provenance"].get("phase2") or {}),
                "run": _out_relative(run_dir),
                "rev": rev,
                "best_commit": commit,
                "tree": oot_repo.history(repo, commit)[-1].tree,
                "origin": origin,
                "reconstructed": rebuilt is not None,
                **({"reconstruction": _relative_sources(rebuilt)} if rebuilt is not None else {}),
            },
            "merlin_git_sha": pub._git_sha_full(),
            "exported_payload_sha256": package_records.payload_inventory(payload)["sha256"],
        }
        legacy = _legacy_block(records["provenance"])
        if legacy is not None:
            stands_in = legacy_stands_in_for(records["provenance"])
            lineage["lineage"] = {
                **records["provenance"]["lineage"],
                "legacy": {**legacy, "mark": LEGACY_MARK, "stands_in_for": stands_in},
            }
        write_pretty_json(staging / ".merlin" / "provenance.json", lineage)
        for name in ("certification", "measurements", "isa_prohibition"):
            write_pretty_json(staging / ".merlin" / f"{name}.json", {**records[name], "schema": SCHEMA})
        _annotate(staging, lineage)
        problems = layout_problems(staging)
        if problems:
            raise ChampionError(f"export is not the standalone layout: {problems}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # only the staging tree this call created
    if pin:
        from ..common.storage_lifecycle import pin as retention_pin

        retention_pin(destination, reason=f"phase-2 champion {target}/{package_id} from {commit[:12]}")
    if update_index:
        from .target_index import write_index

        write_index(target, artifacts_root=artifacts_root)
    return destination


def _annotate(root: Path, lineage: dict[str, Any]) -> None:
    """Say in the champion's own landing page (and, for a legacy lineage, its CHAMPION marker) what
    the records say: an unsealed legacy lineage, a composition, a reconstructed history.

    A caveat that lives only in ``provenance.json`` is read by nobody who clones the repo, so each
    one is printed where the reader starts. The legacy banner goes directly under the title.
    """
    legacy = (lineage.get("lineage") or {}).get("legacy")
    composed = lineage.get("composition")
    rebuilt = (lineage.get("phase2") or {}).get("reconstruction")
    sections: list[str] = []
    banner = ""
    if legacy is not None:
        stands_in = ", ".join(f"`{name}`" for name in legacy["stands_in_for"])
        absent = " or ".join(f"`{name}`" for name in legacy["stands_in_for"])
        banner = (
            f"> **{LEGACY_MARK}** \u2014 this champion predates the {legacy['predates']}. It has no "
            f"{absent}; an explicit legacy lineage stands in. Do not read it as a sealed result.\n\n"
        )
        sections += [
            f"## {LEGACY_MARK}",
            "",
            f"- Reason: {legacy['reason']}",
            f"- Predates: {legacy['predates']}",
            f"- Stands in for: {stands_in}",
            f"- Input bundle: `{legacy['bundle_name']}` (manifest sha256 `{legacy['bundle_manifest_sha']}`)",
            *(f"- Run: `{run}`" for run in legacy["run_dirs"]),
            *(f"- {event}: {date}" for event, date in legacy["dates"].items()),
            "",
            "| Hop | Driver | Model | Effort |",
            "|---|---|---|---|",
            *(
                f"| {hop.get('package') or hop.get('label') or i} | {hop['driver']} | {hop['model']} | "
                f"{hop.get('effort') or 'not recorded'} |"
                for i, hop in enumerate(legacy["hops"])
            ),
            "",
        ]
    if composed is not None:
        sections += [
            "## Composition",
            "",
            f"This champion was {composed['note']}.",
            "",
            f"- Base: `{composed['base']}`",
            *(f"- {label}: `{digest}`" for label, digest in composed["parts"].items()),
            *(f"- {key}: {value}" for key, value in composed.items() if key not in ("note", "base", "parts")),
            "",
        ]
    if rebuilt is not None:
        sections += [
            "## Reconstructed history",
            "",
            "`reconstructed: true` \u2014 no harness `oot/` repository was kept for this lineage, so its "
            f"history was rebuilt from stored package bytes, each hop digest-checked. Reason: {rebuilt['reason']}",
            "",
            *(f"- `{hop['commit'][:12]}` {hop['label']} (`{hop['package_digest']}`)" for hop in rebuilt["hops"]),
            "",
        ]
    if not sections:
        return
    note = root / PUBLICATION_NOTE
    title, _, body = note.read_text(encoding="utf-8").partition("\n\n")
    note.write_text(f"{title}\n\n{banner}{body.rstrip()}\n\n" + "\n".join(sections), encoding="utf-8")
    if legacy is not None:
        marker = root / ".merlin" / "CHAMPION"
        marker.write_text(
            marker.read_text(encoding="utf-8") + f"{LEGACY_MARK} (predates {legacy['predates']})\n", encoding="utf-8"
        )


def _relative_sources(record: dict[str, Any]) -> dict[str, Any]:
    hops = [{**hop, "source": _out_relative(Path(hop["source"]))} for hop in record.get("hops") or ()]
    return {**record, "hops": hops}


def _out_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(paths.out_dir().resolve()).as_posix()
    except ValueError:
        return str(path)


def read_champion(root: Path) -> dict[str, dict]:
    """The four JSON records of an exported champion (missing ones are empty mappings)."""
    import json

    out = {}
    for name in RECORDS:
        member = Path(root) / ".merlin" / f"{name}.json"
        out[name] = json.loads(member.read_text(encoding="utf-8")) if member.is_file() else {}
    return out
