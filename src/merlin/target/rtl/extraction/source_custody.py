"""A target's declared RTL checkout revision and submodule gitlinks, checked before any read.

A descriptor may pin the elaboration it selects to an exact parent commit and to the gitlinks
recorded in that commit. Both are optional: a declaration with neither is accepted unchanged. When
present they are verified against the checkout by content -- branch names move and forks share
them -- and any disagreement refuses the source rather than reading a different elaboration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import is_sha256
from merlin.common.storage.bundle_snapshots import sha256 as source_digest
from merlin.target.rtl.extraction.census import RtlSourceInvalid
from merlin.target.rtl.extraction.circt_decode import _module_port_sig
from merlin.target.rtl.extraction.source_selection import digest, load_selection

# ---- declared_revision -------------------------------------------------------------------------


def verify_declared_revision(root: Path, source_revision: str | None, gitlinks: tuple[tuple[str, str], ...]) -> None:
    """Check optional target-authored git identities before reading an elaboration.

    A gitlink is read from the parent commit, so this check does not depend on
    whether the potentially large submodule has been initialized locally.
    """
    if source_revision is None and not gitlinks:
        return
    if gitlinks and source_revision is None:
        raise RtlSourceInvalid("selected RTL gitlinks require a parent source_revision")

    def git(*args: str) -> str:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        except OSError as exc:
            raise RtlSourceInvalid(f"selected RTL checkout cannot run git at {root}") from exc
        if result.returncode:
            raise RtlSourceInvalid(f"selected RTL checkout cannot read git identity at {root}")
        return result.stdout.strip()

    if source_revision is not None:
        if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
            raise RtlSourceInvalid("selected RTL source_revision must be a full lowercase git commit")
        if git("rev-parse", "HEAD") != source_revision:
            raise RtlSourceInvalid(f"selected RTL checkout revision differs from {source_revision}")
        if git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"):
            raise RtlSourceInvalid("selected RTL checkout has modified tracked source")
    for path, expected in gitlinks:
        parts = Path(path).parts
        if (
            not parts
            or Path(path).is_absolute()
            or any(part in (".", "..") for part in parts)
            or len(expected) != 40
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise RtlSourceInvalid("selected RTL gitlink declaration is invalid")
        row = git("ls-tree", "HEAD", "--", path)
        fields = row.partition("\t")
        if fields[2] != path or fields[0] != f"160000 commit {expected}":
            raise RtlSourceInvalid(f"selected RTL gitlink {path} differs from {expected}")


def declared_revision(
    block: Mapping[str, Any], root: Path, *, where: str
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """Parse ``source_revision`` / ``gitlinks`` from an elaboration block and verify them at ``root``."""
    gitlink_rows = block.get("gitlinks", {})
    if not isinstance(gitlink_rows, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in gitlink_rows.items()
    ):
        raise RtlSourceInvalid(f"{where} declares invalid RTL gitlinks")
    revision = block.get("source_revision")
    if revision is not None and not isinstance(revision, str):
        raise RtlSourceInvalid(f"{where} declares invalid RTL source_revision")
    gitlinks = tuple(sorted(gitlink_rows.items()))
    verify_declared_revision(root, revision, gitlinks)
    return revision, gitlinks


#: Verified (path, size, mtime_ns) -> sha256, so a pinned 80 MB FIRRTL is hashed once per change.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _content_sha256(path: Path) -> str:
    import hashlib

    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def verify_artifact_digests(source) -> None:
    """Refuse a declared elaboration whose PRESENT artifacts are not the pinned bytes.

    An absent artifact is left to the caller (absence is reported as such elsewhere); an artifact that
    exists with different bytes is a different elaboration and is never read in place of the pinned one.
    """
    if not source.artifact_sha256:
        return
    found = source.artifacts()
    for role, expected in source.artifact_sha256:
        path = Path(found[role])
        if path.is_file() and _content_sha256(path) != expected:
            raise RtlSourceInvalid(
                f"{source.target}: {path} has sha256 {_content_sha256(path)}, but {source.origin} pins the "
                f"{role} of {source.config} to {expected}; refusing a different elaboration"
            )


def declared_artifact_pins(block: Mapping[str, Any], *, where: str) -> tuple[tuple[str, str], ...]:
    """Parse an elaboration block's optional ``artifact_sha256`` (``fir``/``hierarchy`` -> sha256)."""
    pins = block.get("artifact_sha256") or {}
    if not isinstance(pins, Mapping) or any(
        role not in ("fir", "hierarchy")
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for role, value in pins.items()
    ):
        raise RtlSourceInvalid(f"{where} artifact_sha256 must map fir/hierarchy to lowercase sha256")
    return tuple(sorted(pins.items()))


# ---- source_receipts ---------------------------------------------------------------------------
# Validation of extractor and consumed-source byte commitments.


def cache_pins_match(doc: dict, *, contract_path, file_digest) -> bool:
    """Check declared extractor and consumed-source commitments before cache reuse.

    This is not hardware qualification. Artifacts without these optional pins
    remain legacy, unverified evidence. A declared pin whose source cannot be
    resolved or no longer matches is not a reusable cache entry.
    """
    from merlin.common.paths import module_source_path

    inputs = doc.get("inputs") or {}
    if not isinstance(inputs, dict):
        return False

    checked = {}

    def matches(path, digest):
        if not is_sha256(digest) or not isinstance(path, (str, Path)) or not path:
            return False
        try:
            if path not in checked:
                checked[path] = file_digest(path)
            return checked[path] == digest
        except OSError:
            return False

    extractor = inputs.get("extractor_sha256")
    if extractor is not None:
        generator = doc.get("generator") or {}
        # Only these core extractors declare this commitment. Do not
        # import arbitrary generator names from an evidence document.
        owners = {"merlin.target.rtl.extraction.circt_decode", "merlin.target.rtl.extraction.census"}
        if not isinstance(generator, dict) or generator.get("name") not in owners:
            return False
        if not matches(module_source_path(generator["name"]), extractor):
            return False
    reader = inputs.get("extraction_reader_sha256")
    if reader is not None and not matches(module_source_path("merlin.target.rtl.extraction.declared_readers"), reader):
        return False
    extraction_contract = inputs.get("extraction_contract_sha256")
    if extraction_contract is not None:
        target = inputs.get("target")
        if not isinstance(target, str) or not target or not matches(contract_path(target), extraction_contract):
            return False
    bundle = inputs.get("source_bundle_path")
    if bundle is not None:
        if not matches(bundle, inputs.get("source_bundle_sha256")):
            return False
        from merlin.target.rtl.extraction.source_selection import load_selection, production_consistency

        try:
            selected = load_selection(bundle, target=inputs.get("target"))
            if production_consistency(selected)["status"] != "verified":
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    genericization = (doc.get("source_consistency") or {}).get("genericization")
    if genericization is not None:
        if not isinstance(genericization, dict):
            return False
        for key in ("input", "output", "tool"):
            member = genericization.get(key)
            if not isinstance(member, dict) or not matches(member.get("path"), member.get("sha256")):
                return False
    body = doc.get("facts") or {}
    if not isinstance(body, dict):
        return False
    sentinels = (None, "unresolved", "missing", "n/a")
    # Legacy documents lacking paths remain diagnostic evidence. New documents
    # bind all resolved paths, including the full census FIRRTL and hierarchy;
    # changing either invalidates the extraction even when features are absent.
    for owner in (inputs, body.get("source") or {}):
        if not isinstance(owner, dict):
            return False
        for path_key, digest_key in (
            ("hw_path", "hw_sha256"),
            ("core_hw_path", "core_hw_sha256"),
            ("generic_hw_path", "generic_hw_sha256"),
            ("fir_path", "fir_sha256"),
            ("hierarchy_path", "hierarchy_sha256"),
            ("isa_path", "isa_sha256"),
        ):
            if owner.get(path_key):
                digest = owner.get(digest_key)
                if digest == "missing" and Path(owner[path_key]).is_file():
                    return False
                if digest not in sentinels and not matches(owner[path_key], digest):
                    return False
        for key in ("firrtl_inputs", "reader_sources"):
            reads = owner.get(key) or []
            if not isinstance(reads, list):
                return False
            for receipt in reads:
                if not isinstance(receipt, dict):
                    return False
                if receipt.get("sha256") == "missing" and receipt.get("path") and Path(receipt["path"]).is_file():
                    return False
                if receipt.get("sha256") not in sentinels and not matches(receipt.get("path"), receipt["sha256"]):
                    return False
    interfaces = body.get("interfaces") or []
    fir_sources = []
    for item in interfaces:
        if not isinstance(item, dict):
            continue
        digest = item.get("source_sha256")
        if digest in (None, "unresolved", "missing", "n/a"):
            continue
        if not matches(item.get("source"), digest):
            return False
        if item.get("name") == "elaborated_rtl_features":
            fir_sources.append(item.get("source"))
    fir_digest = inputs.get("fir_sha256")
    if fir_digest not in (None, "unresolved", "missing", "n/a"):
        sources = [inputs["fir_path"]] if inputs.get("fir_path") else fir_sources
        if not sources or not all(matches(path, fir_digest) for path in sources):
            return False
    return True


# ---- source_relocations ------------------------------------------------------------------------
# Explicit byte-identical ISA/FIRRTL source portability, never new facts.
#
# The returned private view changes only selected input paths. Original facts and
# their byte identity remain untouched; callers must recheck the roster after any
# source observer or other callback. This establishes no semantic or execution
# qualification and does not relocate hardware, compiler or admission receipts.


def _pin(value):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("source relocation requires closed source pins")
    path, digest = value["path"], value["sha256"]
    if (
        type(path) is not str
        or not Path(path).is_absolute()
        or "\0" in path
        or type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("source relocation requires absolute paths and exact SHA-256 pins")
    return path, digest


def relocate_operand_sources(facts, source_relocations=None):
    """Reopen a closed ISA/FIRRTL relocation roster and return a detached view.

    Rows contain exactly ``original`` and ``replacement`` byte pins. At most 16
    recorded ISA/FIRRTL sources may move; unknown sources, duplicate ownership,
    indirect paths and changed bytes refuse. No original facts are rewritten.
    """
    if type(facts) is not dict:
        raise ValueError("source relocation requires original facts")
    before = copy.deepcopy(facts)
    view = copy.deepcopy(before)
    if source_relocations is None:
        return view
    if type(source_relocations) not in (list, tuple) or len(source_relocations) > 16:
        raise ValueError("source relocation requires a bounded explicit roster")
    frozen = copy.deepcopy(source_relocations)
    inputs = view.get("inputs") or {}
    eligible = {}
    if "isa_path" in inputs:
        eligible[inputs["isa_path"]] = inputs.get("isa_sha256")
    firrtls = inputs.get("firrtl_inputs") or []
    for row in firrtls:
        path, digest = _pin(row)
        if path in eligible:
            raise ValueError("source relocation has ambiguous original source ownership")
        eligible[path] = digest
    seen, destinations = set(), set()
    for row in frozen:
        if type(row) is not dict or set(row) != {"original", "replacement"}:
            raise ValueError("source relocation has unsupported fields")
        old, digest = _pin(row["original"])
        new, replacement_digest = _pin(row["replacement"])
        if (
            old in seen
            or new in destinations
            or new in eligible
            or old not in eligible
            or eligible[old] != digest
            or replacement_digest != digest
        ):
            raise ValueError("source relocation must uniquely preserve an original ISA/FIRRTL byte pin")
        path = Path(new)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError("source relocation requires a canonical ordinary replacement source")
        if source_digest(path) != digest:
            raise ValueError("source relocation replacement bytes changed")
        seen.add(old)
        destinations.add(new)
        if inputs.get("isa_path") == old:
            inputs["isa_path"] = new
        for source in firrtls:
            if source["path"] == old:
                source["path"] = new
        if inputs.get("fir_path") == old:
            inputs["fir_path"] = new
    if source_relocations != frozen or facts != before:
        raise ValueError("source relocation original facts or roster changed while reopening its sources")
    return view


# ---- recorded_hw -------------------------------------------------------------------------------
# The core HW dialect a selected RTL facts record was extracted from, read back by content.


def recorded_core_hw(target: str) -> Path | None:
    """The core HW dialect the selected facts were EXTRACTED from, when its bytes are still those.

    A facts record names ``inputs.core_hw_path`` with its ``core_hw_sha256``; reading that exact file
    keeps every downstream reader on the bytes the facts describe instead of an unrelated cache. A
    missing file or a digest that no longer matches yields None (the caller then decides), never a
    different dialect.
    """
    from merlin.target.rtl.facts import rtl_facts_path

    try:
        record = json.loads(Path(rtl_facts_path(target)).read_text(encoding="utf-8"))
        inputs = record.get("inputs") or {}
        declared = (record.get("facts") or {}).get("target")
        path, expected = inputs.get("core_hw_path"), inputs.get("core_hw_sha256")
        if declared not in (None, target) or not isinstance(path, str) or not isinstance(expected, str):
            return None
        candidate = Path(path)
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return candidate if digest.hexdigest() == expected else None
    except (OSError, ValueError, AttributeError):
        return None


# ---- source_audit ------------------------------------------------------------------------------
# Compare authored direct RTL audit questions with actual extracted artifacts.


def _register_bank_geometry(text: str, check: dict) -> tuple[str, dict]:
    """Check port geometry only; this does not establish register behavior or scale semantics."""
    signature = _module_port_sig(text, check["module"])
    if signature is None:
        return "unknown", {"reason": "module_not_found"}
    ports = {}
    for member in signature.split(","):
        lhs, separator, typ = member.strip().rpartition(" : ")
        if not separator:
            continue
        words = lhs.split()
        if len(words) == 2 and words[0] in {"in", "out"}:
            ports[words[1].lstrip("%")] = {"direction": words[0], "type": typ}
    prefix = check["output_prefix"]
    outputs = {
        name: value
        for name, value in ports.items()
        if name.startswith(prefix)
        and bool(name[len(prefix) :])
        and all("0" <= char <= "9" for char in name[len(prefix) :])
    }
    expected_count = check["expected_entries"]
    index_type = check["expected_index_type"]
    if (
        not isinstance(expected_count, int)
        or not isinstance(index_type, str)
        or not index_type.startswith("i")
        or not index_type[1:]
        or not "1" <= index_type[1] <= "9"
        or not all("0" <= char <= "9" for char in index_type[1:])
    ):
        raise ValueError("invalid register-bank audit expectation")
    if expected_count != 1 << int(index_type[1:]):
        raise ValueError("register-bank entry count must match index width")
    observed = {
        "write_index": ports.get(check["write_index_port"]),
        "write_data": ports.get(check["write_data_port"]),
        "output_names": sorted(outputs),
        "output_types": sorted({(value["direction"], value["type"]) for value in outputs.values()}),
        "entry_count": len(outputs),
    }
    matches = (
        observed["write_index"] == {"direction": "in", "type": index_type}
        and observed["write_data"] == {"direction": "in", "type": check["expected_data_type"]}
        and set(outputs) == {f"{prefix}{index}" for index in range(expected_count)}
        and observed["output_types"] == [("out", check["expected_data_type"])]
    )
    return ("verified" if matches else "mismatch"), observed


def build_source_audit(selection: dict, facts: dict, declaration: dict) -> dict:
    """Audit exact source slices, preserving disagreements instead of writing facts.

    The expected values are human-reviewed audit inputs. Observed source slices
    and extracted values are generated outputs; neither supplies hardware support
    or numerical conformance merely by agreeing.
    """
    checks = declaration.get("audit_checks") or []
    results, seen = [], set()
    text = Path(selection["sources"]["core_hw"]["path"]).read_text()
    for check in checks:
        identity = check["id"]
        if identity in seen:
            raise ValueError("duplicate RTL audit check")
        seen.add(identity)
        if check["kind"] == "hw_register_bank_geometry":
            source_status, observed = _register_bank_geometry(text, check)
            results.append(
                {
                    "id": identity,
                    "source_status": source_status,
                    "source": {
                        **selection["sources"]["core_hw"],
                        "module": check["module"],
                        "observed_geometry": observed,
                    },
                    "authored_expectation": {
                        key: check[key]
                        for key in (
                            "write_index_port",
                            "write_data_port",
                            "output_prefix",
                            "expected_entries",
                            "expected_index_type",
                            "expected_data_type",
                        )
                    },
                    "gap": check.get("gap"),
                    "qualification": check.get("qualification"),
                }
            )
            continue
        if check["kind"] != "hw_port_type":
            raise ValueError("unsupported direct RTL audit question")
        module, port = check["module"], check["port"]
        signature = _module_port_sig(text, module)
        observed, observed_direction, snippet = None, None, None
        if signature:
            for member in signature.split(","):
                lhs, separator, typ = member.strip().rpartition(" : ")
                if separator and lhs.split()[-1].lstrip("%") == port:
                    observed, snippet = typ, member.strip()
                    observed_direction = lhs.split()[0]
        projection = check.get("extraction") or {}
        records = (facts.get("facts") or {}).get(projection.get("collection")) or []
        rows = [row for row in records if row.get("name") == projection.get("name")]
        extracted = rows[0].get(projection.get("field")) if len(rows) == 1 else None
        if extracted is None and projection.get("field") == "elem_bits" and len(rows) == 1:
            from merlin.common.target_data import get as quant_format

            try:
                extracted = quant_format(rows[0].get("dtype")).element_bits
            except (KeyError, ValueError, TypeError):
                pass
        direction_matches = check.get("expected_direction") is None or observed_direction == check["expected_direction"]
        source_status = (
            "verified"
            if observed == check.get("expected_type") and direction_matches
            else "unknown"
            if observed is None
            else "mismatch"
        )
        comparable = check.get("comparison", "same_quantity") == "same_quantity"
        bits = int(observed[1:]) if observed and observed.startswith("i") and observed[1:].isdigit() else None
        extraction_status = (
            "different_quantity"
            if not comparable
            else "unknown"
            if bits is None or extracted is None
            else "agrees"
            if bits == extracted
            else "mismatch"
        )
        results.append(
            {
                "id": identity,
                "source_status": source_status,
                "source": {
                    **selection["sources"]["core_hw"],
                    "module": module,
                    "port": port,
                    "observed_type": observed,
                    "observed_direction": observed_direction,
                    "snippet": snippet,
                },
                "authored_expected_type": check.get("expected_type"),
                "authored_expected_direction": check.get("expected_direction"),
                "extraction": {**projection, "value": extracted, "status": extraction_status},
                "gap": check.get("gap"),
                "qualification": check.get("qualification"),
            }
        )
    return {
        "schema": "merlin.rtl_source_audit.v1",
        "target": selection["target"],
        "source_selection_sha256": selection.get("selection_sha256"),
        "status": "verified" if results and all(row["source_status"] == "verified" for row in results) else "unknown",
        "checks": results,
        "historical_hierarchy": selection.get("diagnostics"),
        "qualification": "manual audit checked against exact current RTL; not operation/numerical certification",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-bundle", "facts", "hardware-spec", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    declaration = yaml.safe_load(Path(args.hardware_spec).read_bytes())
    selected = load_selection(args.source_bundle, target=declaration["target"])
    facts = json.loads(Path(args.facts).read_bytes())
    if facts.get("inputs", {}).get("source_bundle_sha256") != selected["selection_sha256"]:
        raise ValueError("audit facts do not bind selected source bundle")
    report = build_source_audit(selected, facts, declaration)
    report["facts_sha256"] = digest(args.facts)
    report["hardware_spec_sha256"] = digest(args.hardware_spec)
    output = Path(args.output)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
