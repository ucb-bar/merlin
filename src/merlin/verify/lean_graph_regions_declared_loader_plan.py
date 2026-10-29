"""Offline ORIGINAL-pin ELF/loader plan; no target binary is ever executed.

The output separates static resolution from launch assurance. Missing/conflicting
pins, unsupported search rules and unpinned selectors are UNAVAILABLE. Current
matching bytes never become new expected hashes or repair a failed profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path, PurePosixPath

if __package__:
    from merlin.verify.lean_graph_regions_declared_elf import UnsupportedElf, parse
else:
    # A data-only CLI uses `python -I -S -B /absolute/this_file.py`.
    # Normal `-m merlin.verify...` runs package initializers with optional native
    # imports. Do not use those initializers for the offline evidence tool.
    # Load the SAME adjacent owned parser, not a substitute implementation.
    import importlib.util
    import sys

    parser_source = Path(__file__).with_name("lean_graph_regions_declared_elf.py")
    parser_spec = importlib.util.spec_from_file_location("_merlin_graph_regions_offline_elf", parser_source)
    if parser_spec is None or parser_spec.loader is None:
        raise ImportError("owned offline ELF parser source unavailable")
    parser_module = importlib.util.module_from_spec(parser_spec)
    sys.modules[parser_spec.name] = parser_module
    parser_spec.loader.exec_module(parser_module)
    UnsupportedElf, parse = parser_module.UnsupportedElf, parser_module.parse


class Unavailable(ValueError):
    """An unresolved premise, never permission to select replacement bytes."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def absolute_path(value):
    if (
        type(value) is not str
        or not value.startswith("/")
        or value.startswith("//")
        or "\0" in value
        or str(PurePosixPath(value)) != value
        or ".." in PurePosixPath(value).parts
    ):
        raise Unavailable("canonical absolute path required")
    return value


def strict_ref(value):
    if (
        type(value) is not dict
        or set(value) != {"path", "sha256", "bytes"}
        or type(value["bytes"]) is not int
        or value["bytes"] < 0
        or type(value["sha256"]) is not str
        or len(value["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in value["sha256"])
    ):
        raise Unavailable("malformed ORIGINAL file reference")
    absolute_path(value["path"])
    return value


def read_original(reference):
    """Check bytes only against an input ORIGINAL pin; never create a pin."""
    reference = strict_ref(reference)
    # Refuse a final symlink or FIFO/device rather than following or blocking on
    # it. Ancestor selection/rename races still require external isolation.
    descriptor = os.open(reference["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise Unavailable("ORIGINAL reference is not a regular file: " + reference["path"])
        data = stream.read()
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    stats = {
        "before": {key: getattr(before, key) for key in fields},
        "after": {key: getattr(after, key) for key in fields},
    }
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise Unavailable(
            "ORIGINAL-pinned file changed during offline read: " + reference["path"],
            dict(original_reference=reference, descriptor_stats=stats),
        )
    observed = hashlib.sha256(data).hexdigest()
    if len(data) != reference["bytes"] or observed != reference["sha256"]:
        raise Unavailable(
            "ORIGINAL pin mismatch; expected unchanged: " + reference["path"] + "; observed=" + observed,
            dict(
                original_reference=reference,
                observed_sha256=observed,
                observed_bytes=len(data),
                descriptor_stats=stats,
                observed_digest_is_not_a_new_expected_pin=True,
            ),
        )
    return data


def references(document, path=()):
    if type(document) is dict:
        if set(document) == {"path", "sha256", "bytes"}:
            yield path, strict_ref(document)
        else:
            for key, value in document.items():
                yield from references(value, path + (key,))
    elif type(document) is list:
        for index, value in enumerate(document):
            yield from references(value, path + (index,))


def parse_json(data):
    """Duplicate keys cannot silently select a different reference or policy."""

    def dictionary(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise Unavailable("duplicate JSON field: " + key)
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=dictionary)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Unavailable("malformed evidence/provenance JSON: " + str(error)) from error


class Pins:
    def __init__(self):
        self.files = {}
        self.origins = {}

    def add(self, reference, origin):
        reference = strict_ref(reference)
        name = reference["path"]
        if name in self.files and self.files[name] != reference:
            raise Unavailable("conflicting ORIGINAL pins: " + name)
        self.files[name] = reference
        self.origins.setdefault(name, []).append(origin)

    @classmethod
    def original_union(cls, provenance):
        """Follow the coordinator's explicit original F/W manifests only.

        W contributes only the listed original runtime refs, not an implicit
        adoption of its entire proof/toolchain context or qualification status.
        """
        result = cls()
        sdk = strict_ref(provenance["original_F_SDK_manifest"])
        source = parse_json(read_original(sdk))
        for entry, reference in source["files"].items():
            result.add(reference, dict(source="F SDK original", manifest=sdk, entry=["files", entry]))
        context_ref = strict_ref(provenance["original_W_fixed_context"])
        context = parse_json(read_original(context_ref))
        indexed = {}
        for location, reference in references(context):
            indexed.setdefault(reference["path"], []).append((location, reference))
        for expected in provenance["original_runtime_refs"]:
            expected = strict_ref(expected)
            if any(reference != expected for _, reference in indexed.get(expected["path"], [])):
                raise Unavailable("conflicting ORIGINAL W runtime refs: " + expected["path"])
            locations = [location for location, reference in indexed.get(expected["path"], []) if reference == expected]
            if not locations and result.files.get(expected["path"]) != expected:
                raise Unavailable("runtime ref lacks an ORIGINAL F/W source: " + expected["path"])
            if locations:
                result.add(
                    expected,
                    dict(
                        source="W fixed original runtime",
                        manifest=context_ref,
                        entries=[list(location) for location in locations],
                    ),
                )
        return result


def resolve_links(path):
    """Resolve each path component structurally; record CURRENT selectors."""
    if not PurePosixPath(path).is_absolute():
        raise Unavailable("relative library/interpreter path")
    pending = list(PurePosixPath(path).parts[1:])
    current, selectors, count = Path("/"), [], 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(candidate)
            selectors.append(dict(path=str(candidate), target=target, original_selector_pin=False))
            count += 1
            if count > 40:
                raise Unavailable("symlink cycle/depth limit")
            value = PurePosixPath(target)
            if value.is_absolute():
                current = Path("/")
                pending = list(value.parts[1:]) + pending
            else:
                pending = list(value.parts) + pending
        else:
            if pending and not stat.S_ISDIR(info.st_mode):
                raise Unavailable("non-directory path component: " + str(candidate))
            current = candidate
    return str(current), selectors


def expand_paths(value, origin):
    if value is None:
        return []
    paths = []
    for entry in value.split(":"):
        if not entry:
            raise Unavailable("empty search component implies ambient CWD")
        # Only the ELF-defined ORIGIN token is admitted. LIB/PLATFORM require
        # separately justified runtime facts; never guess architecture paths.
        expanded, position = [], 0
        while position < len(entry):
            if entry[position] != "$":
                expanded.append(entry[position])
                position += 1
                continue
            position += 1
            if position < len(entry) and entry[position] == "{":
                end = entry.find("}", position + 1)
                if end < 0:
                    raise Unavailable("unterminated dynamic search token")
                token = entry[position + 1 : end]
                position = end + 1
            else:
                end = position
                while end < len(entry) and (entry[end].isascii() and (entry[end].isalnum() or entry[end] == "_")):
                    end += 1
                token = entry[position:end]
                position = end
            if token != "ORIGIN":
                raise Unavailable("unsupported dynamic search token: " + token)
            expanded.append(origin)
        entry = "".join(expanded)
        if not PurePosixPath(entry).is_absolute():
            raise Unavailable("unsupported/relative dynamic search token: " + entry)
        paths.append(entry)
    return paths


def closed_environment(values):
    if type(values) is not dict or any(type(k) is not str or type(v) is not str for k, v in values.items()):
        raise Unavailable("complete explicit string environment required")
    forbidden = [key for key in values if key.startswith("LD_") or key in {"GLIBC_TUNABLES", "GCONV_PATH", "LOCPATH"}]
    if forbidden:
        raise Unavailable("ambient/preload/audit/tunable environment unsupported: " + ",".join(forbidden))
    allowed = {"LANG", "LC_ALL", "LEAN_PATH", "LEAN_SRC_PATH"}
    if set(values) - allowed or any(values.get(key) != "C" for key in ("LANG", "LC_ALL")):
        raise Unavailable("environment is outside the proposed closed pure-launch policy")
    if any("\0" in value for value in values.values()):
        raise Unavailable("NUL in launch environment")
    for key in ("LEAN_PATH", "LEAN_SRC_PATH"):
        if key in values:
            for part in values[key].split(":"):
                absolute_path(part)
    return values


def static_plan(pins, executable, library_directories, environment, held_paths=()):
    environment = closed_environment(environment)
    absolute_path(executable)
    if not library_directories:
        raise Unavailable("an explicit nonempty library-directory list is required")
    for directory in library_directories:
        absolute_path(directory)
        if ":" in directory or ";" in directory or "$" in directory:
            raise Unavailable("explicit absolute library directories required")
    nodes, edges, issues, pending = {}, [], [], [(executable, (), "executable")]
    names = {}
    interpreter, root_abi = None, None
    while pending:
        requested, inherited, role = pending.pop(0)
        try:
            resolved, selectors = resolve_links(requested)
            if resolved not in pins.files:
                raise Unavailable("UNPINNED_REQUIRED_FILE: " + resolved)
            if selectors:
                issues.append(dict(kind="UNPINNED_SYMLINK_SELECTOR", requested=requested, selectors=selectors))
            if resolved in nodes:
                nodes[resolved]["repeated_search_contexts"].append(dict(inherited_rpath=list(inherited), role=role))
                if inherited:
                    issues.append(
                        dict(
                            kind="REPEATED_NODE_INHERITED_RPATH_REQUIRES_REVIEW",
                            path=resolved,
                            directories=list(inherited),
                        )
                    )
                continue
            metadata = parse(read_original(pins.files[resolved]))
            abi = (metadata.bits, metadata.byte_order, metadata.machine)
            if root_abi is None:
                root_abi = abi
            if abi != root_abi or role != "executable" and metadata.kind != 3:
                raise Unavailable("incompatible ELF ABI/type in proposed closure: " + resolved)
            nodes[resolved] = dict(
                pin=pins.files[resolved], provenance=pins.origins[resolved], elf=asdict(metadata), role=role
            )
            nodes[resolved]["repeated_search_contexts"] = []
            if metadata.soname is not None:
                if "/" in metadata.soname or "$" in metadata.soname:
                    issues.append(dict(kind="SONAME_PATH_OR_TOKEN_UNSUPPORTED", path=resolved, soname=metadata.soname))
                previous = names.setdefault(metadata.soname, resolved)
                if previous != resolved:
                    issues.append(
                        dict(
                            kind="SONAME_COLLISION_REQUIRES_LOAD_ORDER_REVIEW",
                            name=metadata.soname,
                            paths=[previous, resolved],
                        )
                    )
            if metadata.unsupported_load_tags:
                issues.append(
                    dict(
                        kind="FILTER_AUDIT_AUXILIARY_LOAD_UNSUPPORTED",
                        path=resolved,
                        tags=list(metadata.unsupported_load_tags),
                    )
                )
            if role == "executable" and metadata.interpreter:
                interpreter, links = resolve_links(metadata.interpreter)
                edges.append(
                    dict(
                        kind="PT_INTERP",
                        owner=resolved,
                        requested=metadata.interpreter,
                        resolved=interpreter,
                        selectors=links,
                        status="ORIGINAL_PIN" if interpreter in pins.files and not links else "UNAVAILABLE",
                    )
                )
                if links:
                    issues.append(
                        dict(
                            kind="UNPINNED_INTERPRETER_SELECTOR",
                            selectors=links,
                            explicit_pinned_loader_proposal_bypasses_this_selector=True,
                        )
                    )
                if interpreter not in pins.files:
                    issues.append(dict(kind="UNPINNED_INTERPRETER", path=interpreter))
                else:
                    pending.append((interpreter, (), "interpreter"))
            elif role == "interpreter" and metadata.interpreter:
                issues.append(
                    dict(kind="INTERPRETER_REQUIRES_ANOTHER_INTERPRETER", path=resolved, required=metadata.interpreter)
                )
            origin = str(PurePosixPath(resolved).parent)
            rpath = expand_paths(metadata.rpath, origin)
            runpath = expand_paths(metadata.runpath, origin)
            # Inherited RPATH order is intentionally not asserted as proved.
            # Expand and record it, but refuse launch assurance for that case.
            if inherited:
                issues.append(
                    dict(kind="INHERITED_RPATH_REQUIRES_LOADER_REVIEW", path=resolved, directories=list(inherited))
                )
            directories = (
                (rpath + list(inherited) if metadata.runpath is None else []) + list(library_directories) + runpath
            )
            child_inherited = tuple((rpath if metadata.runpath is None else []) + list(inherited))
            for needed in metadata.needed:
                if "/" in needed or "$" in needed:
                    issues.append(dict(kind="NEEDED_PATH_OR_TOKEN_UNSUPPORTED", owner=resolved, needed=needed))
                    continue
                edge = dict(kind="DT_NEEDED", owner=resolved, needed=needed, probes=[], status="UNAVAILABLE")
                for directory in dict.fromkeys(directories):
                    candidate = str(PurePosixPath(directory) / needed)
                    try:
                        target, links = resolve_links(candidate)
                    except FileNotFoundError:
                        edge["probes"].append(
                            dict(path=candidate, current_observation="absent; not original absence assurance")
                        )
                        continue
                    except (OSError, Unavailable) as error:
                        edge.update(reason="search candidate cannot be examined: " + str(error), requested=candidate)
                        break
                    edge.update(requested=candidate, resolved=target, selectors=links)
                    if target not in pins.files:
                        edge["reason"] = "first existing candidate is unpinned; never fall through to a later pin"
                    else:
                        edge["pin_status"] = "ORIGINAL_PIN_BYTES"
                        edge["status"] = (
                            "UNAVAILABLE" if links or inherited else "STATIC_PIN_RESOLVED; LOAD_REVIEW_PENDING"
                        )
                        if links or inherited:
                            edge["reason"] = "unbound symlink selector or inherited RPATH semantics"
                        pending.append((target, child_inherited, "dependency"))
                        if links:
                            issues.append(dict(kind="UNPINNED_SYMLINK_SELECTOR", requested=candidate, selectors=links))
                    break
                if edge["status"] == "UNAVAILABLE":
                    edge.setdefault("reason", "would require loader cache/default directories; no fallback is assumed")
                    issues.append(dict(kind="UNRESOLVED_DEPENDENCY", edge=edge))
                edges.append(edge)
        except (Unavailable, UnsupportedElf, OSError) as error:
            issues.append(
                dict(
                    kind="UNAVAILABLE", requested=requested, reason=str(error), details=getattr(error, "details", None)
                )
            )
    overlap = sorted(set(nodes) & set(held_paths))
    if overlap:
        issues.append(
            dict(kind="HELD_HOST_CANDIDATE_PATH_OVERLAP", paths=overlap, exact_previously_failed_reference="UNKNOWN")
        )
    prerequisites = [
        "actual process loaded-file closure not observed",
        "system preload absence/isolation not established by original positive file pins",
        "glibc-hwcaps search and loader option support require original-loader review",
        "late dlopen/plugin/source-elaboration loads require bounded enforcement/evidence",
        "exact source/object/import closure and host/platform trust remain separate",
    ]
    return dict(
        schema="merlin.graph-regions.offline-pure-loader-plan.v1",
        launch_status="UNAVAILABLE",
        static_resolution_status="UNAVAILABLE" if issues else "ORIGINAL_PIN_GRAPH_ONLY; LOAD_ASSURANCE_UNAVAILABLE",
        static_nodes=nodes,
        resolution_edges=edges,
        issues=issues,
        missing_assurance=prerequisites,
        proposed_environment=environment,
        actual_loaded_closure_established=False,
        target_binary_executed=False,
        new_expected_hashes_created=False,
        current_reference_matches_are_not_a_failed_context_repair=True,
        held_reference_association="UNAVAILABLE: exact previously failed host reference UNKNOWN",
        original_pin_count=len(pins.files),
        interpreter=interpreter,
        library_directories=list(library_directories),
        requested_executable=executable,
        closed_environment_means_replacement_not_inheritance=True,
        default_directory_fallback_is_not_disabled_by_proposed_argv=True,
        load_alias_and_symbol_resolution_not_proved=True,
        proposed_diagnostic_argv=None
        if interpreter not in pins.files
        else [interpreter, "--inhibit-cache", "--library-path", ":".join(library_directories), "--list", executable],
        proposed_pure_argv=None
        if interpreter not in pins.files
        else [
            interpreter,
            "--inhibit-cache",
            "--library-path",
            ":".join(library_directories),
            executable,
            "-j1",
            "<candidate-source.lean>",
        ],
        command_status="DATA FOR REVIEW ONLY; no execution authorized",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = parse_json(args.provenance.read_bytes())
        pins = Pins.original_union(provenance)
        result = static_plan(
            pins, args.executable, provenance["proposed_runtime_directories"], parse_json(args.environment.read_bytes())
        )
        result["original_provenance_document"] = str(args.provenance)
        result["original_manifest_inputs"] = {
            key: provenance[key] for key in ("original_F_SDK_manifest", "original_W_fixed_context")
        }
        result["original_runtime_refs"] = provenance["original_runtime_refs"]
    except (Unavailable, OSError, KeyError, TypeError) as error:
        result = dict(
            schema="merlin.graph-regions.offline-pure-loader-plan.v1",
            launch_status="UNAVAILABLE",
            static_resolution_status="UNAVAILABLE",
            original_provenance_document=str(args.provenance),
            issues=[dict(kind="UNAVAILABLE", reason=str(error), details=getattr(error, "details", None))],
            actual_loaded_closure_established=False,
            target_binary_executed=False,
            new_expected_hashes_created=False,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A first observation is retained; callers must select a NEW owned output.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
