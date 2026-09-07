"""One-call, host-owned pure-build source grants; never an agent mount API.

The trusted build adapter supplies an exact source closure and request pins.
This module only checks and applies that obligation to an existing policy; it
does not discover dependencies, authorize an oracle, or execute a command.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def _path(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError("build grants require canonical absolute paths")
    return path


def _overlap(left, right):
    # Inputs have already passed canonical-path validation. Comparing component
    # prefixes avoids rebuilding deep Path.parents tuples for every grant/mask.
    left, right = str(left), str(right)
    return left == right or left.startswith(right.rstrip("/") + "/") or right.startswith(left.rstrip("/") + "/")


def _coverage_gaps(prefix, surfaces):
    """Reuse the shared one-index-per-call mount visibility implementation."""
    from .bwrap import coverage_gap
    return coverage_gap(prefix, surfaces)


def _validate_format_data(rows):
    """Authorize only the common numeric registry and its actual schema.

    This is not an extension-based YAML grant. There are exactly two canonical
    host-selected leaves, with fixed roles and freshly validated content.
    """
    import os
    from merlin.common import quant_formats
    from merlin.common.paths import schemas_dir
    from merlin.common.yaml import load_yaml
    if os.environ.get(quant_formats._ENV_OVERLAY):
        raise ValueError("build format-data grants do not authorize registry overlays")
    expected = {
        "numeric_format_registry": schemas_dir() / quant_formats._REGISTRY_FILENAME,
        "numeric_format_schema": schemas_dir() / "quant_format.schema.yaml",
    }
    if len(rows) != 2 or {row[3] for row in rows} != set(expected):
        raise ValueError("format data requires the exact numeric registry/schema pair")
    for source, _, _, role in rows:
        if _path(source) != expected[role].resolve():
            raise ValueError("format-data source is not the canonical host registry/schema")
    schema = load_yaml(expected["numeric_format_schema"])
    fields = schema.get("required_top_level_fields") if isinstance(schema, dict) else None
    raw = load_yaml(expected["numeric_format_registry"])
    if (not isinstance(fields, list) or not fields or not all(isinstance(x, str) for x in fields)
            or not isinstance(raw, dict) or set(raw) != {"version", "formats"}
            or raw["version"] != 1 or not isinstance(raw["formats"], dict) or not raw["formats"]):
        raise ValueError("format registry/schema structure is invalid")
    for name, entry in raw["formats"].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("numeric format entry is invalid")
        value = {"name": name, **entry}
        if any(field not in value or value[field] is None for field in fields):
            raise ValueError("numeric format entry does not satisfy its actual schema")
        quant_formats._validate_entry(name, entry)


@dataclass(frozen=True)
class HostBuildDependencies:
    """Exact trusted worker invocation and pure source closure for ONE call.

    ``source_root`` and ``namespace_root`` describe the same project namespace
    before/after source freezing. Grants preserve relative paths, so a harmless
    filename cannot be used to smuggle a masked backend under another name.
    Tools are pin obligations only: this capability cannot mount tools or data.
    The request and worker must be pinned files appearing in the exact argv.
    """
    source_root: str
    namespace_root: str
    argv: tuple[str, ...]
    file_pins: tuple[tuple[str, str], ...]
    source_grants: tuple[tuple[str, str, str], ...]
    request_path: str
    worker_path: str
    format_data: tuple[tuple[str, str, str, str], ...] = ()

    def revalidate(self, argv):
        if type(self) is not HostBuildDependencies or tuple(map(str, argv)) != self.argv:
            raise ValueError("build dependency capability belongs to a different worker argv")
        pins = dict(self.file_pins)
        if not pins or len(pins) != len(self.file_pins):
            raise ValueError("build dependency pins must be complete and unique")
        for name in (self.argv[0], self.request_path, self.worker_path):
            if name not in self.argv or name not in pins:
                raise ValueError("build request and worker must be exact pinned argv files")
        for name, expected in self.file_pins:
            path = Path(name)
            if not path.is_absolute() or not path.is_file() or _hash(path) != expected:
                raise ValueError("build worker/request/tool pin changed: " + name)
        for source, _, expected in (*self.source_grants, *(row[:3] for row in self.format_data)):
            path = _path(source)
            if not path.is_file() or _hash(path) != expected:
                raise ValueError("pure build source pin changed: " + source)

    def extend(self, sandbox, argv, *, overlay_root=None, overlay_builder=None):
        """Return a fresh argv; never mutate or cache an extended native policy."""
        from .answer_surfaces import AnswerSurface
        from .bwrap import _mounts
        self.revalidate(argv)
        source_root, namespace = _path(self.source_root), _path(self.namespace_root)
        if not source_root.is_dir() or not namespace.is_dir() or namespace == Path("/"):
            raise ValueError("build namespace must be an existing frozen project directory")
        prefix = list(sandbox["command_prefix"])
        if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
            raise ValueError("build extension requires the existing clear-environment bwrap policy")
        surfaces = [AnswerSurface(str(s.get("label", "answer")), _path(s["path"]),
                                 s["kind"], str(s.get("origin", "oracle")))
                    for s in sandbox["answer_surfaces"]]
        if not surfaces or _coverage_gaps(prefix, surfaces):
            raise ValueError("existing build policy has missing or exposed answer masks")
        mounts = _mounts(prefix)
        # A namespace is an existing import tree, not any filesystem directory
        # named by a flag value. Frozen policies may expose only individual leaves.
        readonly_destinations = {prefix[i+2] for i, flag in enumerate(prefix[:-2])
                                 if flag == "--ro-bind" and prefix[i+1] != "/dev/null"}
        if not any(state == "expose" and dest in readonly_destinations
                   and namespace in Path(dest).parents for state, _, dest in mounts):
            raise ValueError("build destination namespace has no existing readonly source view")
        masks = {str(s.path) for s in surfaces}
        insertion = None
        for i, value in enumerate(prefix):
            if ((value == "--tmpfs" and i+1 < len(prefix) and prefix[i+1] in masks)
                    or (value == "--ro-bind" and i+2 < len(prefix)
                        and prefix[i+1] == "/dev/null" and prefix[i+2] in masks)):
                insertion = i
                break
        if insertion is None:
            raise ValueError("build extension requires an explicit existing answer-mask boundary")
        if self.format_data:
            _validate_format_data(self.format_data)
        additions, destinations = [], set()
        grants = [(source, destination, True) for source, destination, _ in self.source_grants]
        grants += [(row[0], row[1], False) for row in self.format_data]
        for source, destination, python_source in grants:
            src, dest = _path(source), _path(destination)
            if (python_source and src.suffix != ".py") or source_root not in src.parents:
                raise ValueError("build grants are only declared pure Python source leaves")
            if dest != namespace / src.relative_to(source_root) or dest in destinations:
                raise ValueError("build source namespace mapping is not unique and identity-preserving")
            destinations.add(dest)
            for surface in surfaces:
                aliases = [surface.path]
                if namespace in surface.path.parents:
                    aliases.append(source_root / surface.path.relative_to(namespace))
                if any(_overlap(path, answer) for path in (src, dest) for answer in aliases):
                    raise ValueError("pure build grant overlaps a masked answer source or destination")
            additions.extend(("--ro-bind", str(src), str(dest)))
        if not additions:
            raise ValueError("build dependency capability has no declared source closure")
        base = prefix
        if overlay_builder is not None:
            if overlay_root is None or not callable(overlay_builder):
                raise ValueError("build overlay needs a host-owned private root and builder")
            private = _path(overlay_root)
            if not private.is_dir() or any(private.iterdir()):
                raise ValueError("build overlay root must be a fresh private directory")
            # A private source snapshot is never copied into public scratch or
            # served writable. Reuse the host's existing missing-leaf merger.
            if any(state == "expose" and _overlap(private, Path(dest))
                   for state, _, dest in mounts):
                raise ValueError("build overlay root overlaps an existing public sandbox view")
            mapped, changed_destinations = list(prefix), []
            for i, option in enumerate(prefix[:-2]):
                if option != "--ro-bind":
                    continue
                dest = Path(prefix[i+2])
                if dest == namespace or namespace in dest.parents:
                    mapped[i+2] = str(source_root / dest.relative_to(namespace))
                    changed_destinations.append(i+2)
            dependency_record = {"shared_source_root": str(source_root), "shared_sources": {
                str(Path(src).relative_to(source_root)): pin
                for src, _, pin in (*self.source_grants, *(row[:3] for row in self.format_data))}}
            merged = overlay_builder(mapped, dependency_record, private)
            base, additions = list(merged[:len(prefix)]), list(merged[len(prefix):])
            for index in changed_destinations:
                base[index] = prefix[index]
            if len(additions) % 3:
                raise ValueError("host overlay builder returned malformed direct grants")
            for i in range(0, len(additions), 3):
                if additions[i] != "--ro-bind":
                    raise ValueError("host overlay builder returned a non-readonly grant")
                src = Path(additions[i+2])
                additions[i+2] = str(namespace / src.relative_to(source_root))
        elif overlay_root is not None:
            raise ValueError("build overlay root has no trusted merger")
        result = [*base[:insertion], *additions, *base[insertion:]]
        if _coverage_gaps(result, surfaces):
            raise ValueError("build source grants expose an answer surface")
        self.revalidate(argv)
        return result
