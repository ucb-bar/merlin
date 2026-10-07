"""Check the source and mode population selected for machine-dialect authoring.

This pre-authoring check binds a decoder census to the exact RTL checkout used
by a reproduced elaboration. It does not certify instruction semantics, a
generated dialect, or execution. Those are separate Phase 1 obligations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .isa_census import _git_source_at_revision, derive_source_census
from .isa_mode_audit import audit_mode_inventory
from .rtl.source_selection import load_selection, production_consistency

SCHEMA = "merlin.dialect_source_scope.v1"


def _read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document, hashlib.sha256(raw).hexdigest()


def _selected_rtl_checkout(receipt: dict, census: dict) -> tuple[str | None, list[str]]:
    """Locate one pinned checkout containing both observed RTL source files."""
    source = receipt.get("source") or {}
    root = Path(source["root"]).resolve(strict=True)
    revisions = {".": source["revision"], **source["submodules"]}
    expected_revision = census.get("rtl_revision")
    sources = census.get("sources") or {}
    try:
        observed = [Path(sources[role]["path"]).resolve(strict=True) for role in ("patterns", "decoder")]
    except (KeyError, TypeError, OSError, ValueError):
        return None, ["selected census RTL source paths are absent"]
    matches = [
        name
        for name, revision in revisions.items()
        if revision == expected_revision
        and all(path.is_relative_to(root if name == "." else root / name) for path in observed)
    ]
    if len(matches) != 1:
        return None, ["census RTL sources do not belong to exactly one pinned elaboration checkout"]
    return matches[0], []


def _architecture_source_basis(
    receipt: dict, checkout: str, census: dict, inventory: dict
) -> tuple[list[dict], list[str]]:
    """Verify every required mode's cited RTL files against its pinned Git objects."""
    source = receipt["source"]
    root = Path(source["root"]).resolve(strict=True)
    selected = root if checkout == "." else root / checkout
    revision = census["rtl_revision"]
    members: dict[str, str] = {}
    blockers = []

    def check_member(name: str, label: str) -> None:
        if name in members:
            return
        member = selected / name
        try:
            if member.is_symlink() or not member.is_file() or not member.resolve(strict=True).is_relative_to(selected):
                raise ValueError("source is absent, indirect, or outside selected checkout")
            committed = _git_source_at_revision(member, revision)
            if committed["path_at_revision"] != Path(name).as_posix():
                raise ValueError("source belongs to a different Git checkout")
            members[name] = committed["sha256"]
        except (OSError, ValueError) as exc:
            blockers.append(f"{label}: source {name} is not pinned: {exc}")

    required_domains: set[str] = set()
    for row in inventory["variants"]:
        if row.get("required") is not True:
            continue
        identity = row["id"]
        if isinstance(row.get("parameter_domains"), list):
            required_domains.update(name for name in row["parameter_domains"] if isinstance(name, str))
        references = row.get("architecture_sources")
        if (
            not isinstance(references, list)
            or not references
            or any(
                not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts
                for name in references
            )
        ):
            blockers.append(f"{identity}: architecture source references are absent or invalid")
            continue
        if len(references) != len(set(references)):
            blockers.append(f"{identity}: architecture source references are duplicated")
            continue
        for name in references:
            check_member(name, f"{identity}: architecture")
    for domain in sorted(required_domains):
        detail = inventory["parameter_domains"].get(domain)
        if not isinstance(detail, dict) or not isinstance(detail.get("evidence_sources"), list):
            continue  # the mode audit already records this missing declaration
        for name in detail["evidence_sources"]:
            if isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts:
                check_member(name, f"parameter domain {domain}")
    return [{"path_at_revision": name, "sha256": digest} for name, digest in sorted(members.items())], blockers


def audit_dialect_source_scope(
    *,
    selection_path: Path,
    census_path: Path,
    inventory_path: Path,
    expected_config: str,
) -> dict:
    """Replay selected input observations and report exact pre-authoring blockers.

    Each path is an explicit operator input. The census is recomputed from the
    selected committed source bytes; a copied JSON receipt cannot establish
    source identity by its own assertions. A reviewed mode ledger may still
    carry open semantic, typing, and execution obligations into Phase 1.
    """
    selection = load_selection(selection_path)
    census, census_sha256 = _read_json(census_path)
    inventory, inventory_sha256 = _read_json(inventory_path)
    consistency = production_consistency(selection)
    mode_audit = audit_mode_inventory(census, inventory)
    blockers: list[str] = []
    if not isinstance(expected_config, str) or not expected_config.strip():
        raise ValueError("expected selected configuration must be explicit")
    if selection.get("config") != expected_config:
        blockers.append("selected source configuration differs from requested campaign")
    if inventory.get("target") != selection["target"]:
        blockers.append("mode ledger target differs from selected source target")
    if consistency["status"] != "verified":
        blockers.append("selected FIRRTL-to-HW production is not verified")
    elaboration = (consistency.get("elaboration") or {}).get("status")
    if elaboration != "reproduced_exact_firrtl":
        blockers.append("selected configuration-to-FIRRTL elaboration is not reproduced")
    if not mode_audit["phase1_mode_scope_ready"]:
        blockers.append("selected decoder mode population or source discrepancy review is incomplete")
    if not mode_audit["phase1_parameter_domains_ready"]:
        blockers.append("required machine parameter domains are not structured and reviewed")

    checkout = None
    architecture_sources: list[dict] = []
    if elaboration == "reproduced_exact_firrtl":
        receipt_path = Path(selection["production"]["elaboration"]["path"])
        receipt, _ = _read_json(receipt_path)
        checkout, binding_blockers = _selected_rtl_checkout(receipt, census)
        blockers.extend(binding_blockers)
    if checkout is not None:
        architecture_sources, source_blockers = _architecture_source_basis(receipt, checkout, census, inventory)
        blockers.extend(source_blockers)
        sources = census.get("sources") or {}
        try:
            replay = derive_source_census(
                pattern_file=Path(sources["patterns"]["path"]),
                decoder_file=Path(sources["decoder"]["path"]),
                model_isa_file=Path(sources["model_isa"]["path"]),
                rtl_revision=census["rtl_revision"],
                model_revision=census["source_revision_verification"]["model_revision"],
                verify_revisions=True,
            )
            if json.dumps(replay, sort_keys=True) != json.dumps(census, sort_keys=True):
                blockers.append("selected ISA census differs from a fresh pinned-source replay")
        except (KeyError, OSError, SyntaxError, ValueError) as exc:
            blockers.append(f"selected ISA census cannot be replayed: {exc}")

    return {
        "schema": SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "target": selection["target"],
        "selected_config": selection.get("config"),
        "selected_rtl_revision": census.get("rtl_revision"),
        "selected_rtl_checkout": checkout,
        "inputs_sha256": {
            "selection": selection["selection_sha256"],
            "census": census_sha256,
            "inventory": inventory_sha256,
        },
        "source_consistency_status": consistency["status"],
        "elaboration_status": elaboration or "not_selected",
        "mode_scope_ready": mode_audit["phase1_mode_scope_ready"],
        "parameter_domains_ready": mode_audit["phase1_parameter_domains_ready"],
        "mode_counts": mode_audit["counts"],
        "architecture_sources": architecture_sources,
        "unresolved_source_discrepancies": mode_audit["unresolved_source_discrepancies"],
        "blockers": blockers,
        "qualification": (
            "source and requirement population only; this is not semantic, typed-dialect, "
            "emission, numerical, or hardware-execution qualification"
        ),
    }
