"""Compiler choices read from a source tree's declared tuning settings, and their declaration.

A compiler can expose a choice it would otherwise hard-code -- whether to apply a library pass, an
order, a blocking option -- as one entry of a JSON settings object inside its own source tree, and
declare the entry and its allowed values in ``TUNING_KNOBS.json`` at the tree root::

    {"schema": "merlin.tuning_knobs.v1",
     "settings_file": "tuning_settings.json",
     "knobs": {"readout_overlap": {"values": ["off", "on"], "mechanism": "catalogue id (optional)",
                                   "note": "optional"}}}

The compiler reads the current value where it makes the choice::

    from merlin.compile.tuning_knobs import knob
    if knob("readout_overlap", default="off", values=("off", "on")) == "on":
        ...apply the pass...

Tooling that copies the tree may set another declared value in its copy's settings file and compare
the two compilations; the copy's compiler then reads that value through the same call. Nothing here
chooses, ranks or recommends a value, and nothing names a target or a mechanism.

The rules are the ones copying tooling applies: a plain relative ``.json`` settings path with no hidden
or ``..`` component, one to :data:`MAX_KNOBS` knobs, one to :data:`MAX_VALUES` JSON scalar values each
(``null``, booleans, integers, finite floats, strings of at most 64 characters), and values compared
with their type (``true`` is not ``1``). A value the compiler does not accept, or a malformed
declaration, raises :class:`KnobError` rather than silently compiling something else.

Which tree. :func:`knob` reads the tree given as ``root``, else the one set with :func:`use_root`,
else the nearest directory holding ``TUNING_KNOBS.json`` at or above the running program
(``sys.argv[0]``), else at or above this module's own copy. Without a declaration it returns
``default``. ``python -m merlin.compile.tuning_knobs check TREE`` validates a tree and lists every
single-knob change; ``set TREE name=value`` writes declared values (for example into a copied tree).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.tuning_knobs.v1"
DECLARATION = "TUNING_KNOBS.json"
DEFAULT_SETTINGS = "tuning_settings.json"
MAX_KNOBS = 16
MAX_VALUES = 8
MAX_NAME = 64
MAX_TEXT = 200
MAX_BYTES = 1 << 20
_SEARCH_DEPTH = 8


class KnobError(ValueError):
    """A declaration, settings file or knob call that cannot be honoured exactly."""


def scalar(value) -> bool:
    """Whether ``value`` is a JSON scalar a knob may take."""
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return type(value) is str and len(value) <= 64


def same_value(left, right) -> bool:
    """Equal JSON scalars of the same type (``True`` is not ``1``)."""
    return type(left) is type(right) and left == right


def _member(value, values) -> bool:
    return any(same_value(value, allowed) for allowed in values)


def _relative(text) -> str | None:
    if not isinstance(text, str) or not text or len(text) > MAX_TEXT or "\\" in text or "\0" in text:
        return None
    path = PurePosixPath(text)
    if path.is_absolute() or path.suffix != ".json":
        return None
    if any(part in ("", ".", "..") or part.startswith(".") for part in path.parts):
        return None
    return path.as_posix()


@dataclass(frozen=True)
class Knob:
    """One declared choice: its allowed values, the value the tree starts with, and optional labels."""

    values: tuple
    default: object
    mechanism: str | None = None
    note: str | None = None

    def __post_init__(self):
        values = tuple(self.values)
        object.__setattr__(self, "values", values)
        if not 0 < len(values) <= MAX_VALUES or not all(scalar(v) for v in values):
            raise KnobError(f"a knob takes 1 to {MAX_VALUES} JSON scalar values (strings at most 64 characters)")
        if any(_member(v, values[:i]) for i, v in enumerate(values)):
            raise KnobError("knob values must be distinct")
        if not _member(self.default, values):
            raise KnobError(f"default {self.default!r} is not one of the knob's values {list(values)!r}")
        for label in (self.mechanism, self.note):
            if label is not None and (not isinstance(label, str) or len(label) > MAX_TEXT):
                raise KnobError(f"mechanism and note are strings of at most {MAX_TEXT} characters")

    def spec(self) -> dict:
        found = {"values": list(self.values)}
        if self.mechanism:
            found["mechanism"] = self.mechanism
        if self.note:
            found["note"] = self.note
        return found


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------

_CACHE: dict[str, tuple[tuple, object]] = {}
_ROOT: list[Path] = []
_CONSULTED: dict[str, object] = {}


def _read_json(path: Path):
    """``(exists, document)``; a symlink, oversized or unparsable file raises :class:`KnobError`."""
    if not path.exists() and not path.is_symlink():
        return False, None
    if path.is_symlink() or not path.is_file():
        raise KnobError(f"{path.name} must be a regular file, not a symlink or directory")
    stat = path.stat()
    if stat.st_size > MAX_BYTES:
        raise KnobError(f"{path.name} is larger than {MAX_BYTES} bytes")
    key, stamp = str(path.resolve()), (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return True, cached[1]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise KnobError(f"{path.name} is not readable JSON: {error}") from None
    _CACHE[key] = (stamp, document)
    return True, document


def declaration_problems(document) -> list[str]:
    """Why a parsed declaration would be refused (empty when it is accepted)."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return [f"{DECLARATION} must be a JSON object with \"schema\": \"{SCHEMA}\""]
    problems = []
    if _relative(document.get("settings_file")) is None:
        problems.append("settings_file must be a relative .json path inside the tree, with no hidden or '..' "
                        f"component, at most {MAX_TEXT} characters")
    knobs = document.get("knobs")
    if not isinstance(knobs, dict) or not 0 < len(knobs) <= MAX_KNOBS:
        return problems + [f"knobs must map 1 to {MAX_KNOBS} names to specs"]
    for name, spec in knobs.items():
        values = spec.get("values") if isinstance(spec, dict) else None
        if not isinstance(name, str) or not 0 < len(name) <= MAX_NAME:
            problems.append(f"knob name {name!r} must be 1 to {MAX_NAME} characters")
        if not isinstance(values, list) or not 0 < len(values) <= MAX_VALUES or not all(scalar(v) for v in values):
            problems.append(f"knob {name!r}: values must list 1 to {MAX_VALUES} JSON scalars "
                            "(null, booleans, integers, finite floats, strings of at most 64 characters)")
    return problems


def declaration(root) -> dict | None:
    """``{"settings_file", "knobs": {name: {"values", "mechanism", "note"}}}``, or ``None`` when absent."""
    exists, document = _read_json(Path(root) / DECLARATION)
    if not exists:
        return None
    problems = declaration_problems(document)
    if problems:
        raise KnobError(f"{Path(root) / DECLARATION}: " + "; ".join(problems))
    knobs = {}
    for name, spec in document["knobs"].items():
        knobs[name] = {"values": list(spec["values"]),
                       "mechanism": spec.get("mechanism") if isinstance(spec.get("mechanism"), str) else None,
                       "note": spec.get("note")[:MAX_TEXT] if isinstance(spec.get("note"), str) else None}
    return {"settings_file": _relative(document["settings_file"]), "knobs": knobs}


def settings(root, declared: dict | None = None) -> dict:
    """The tree's current settings object (empty when the declaration or the file is absent)."""
    declared = declaration(root) if declared is None else declared
    if declared is None:
        return {}
    exists, document = _read_json(Path(root) / declared["settings_file"])
    if not exists:
        return {}
    if not isinstance(document, dict):
        raise KnobError(f"{declared['settings_file']} must hold a JSON object")
    return document


def find_root(start) -> Path | None:
    """The nearest directory at or above ``start`` (a file or directory) that holds a declaration."""
    try:
        here = Path(start).resolve()
    except (OSError, RuntimeError, TypeError):
        return None
    if not here.is_dir():
        here = here.parent
    for candidate in (here, *here.parents)[:_SEARCH_DEPTH]:
        if (candidate / DECLARATION).is_file():
            return candidate
    return None


def use_root(root) -> None:
    """Make ``root`` the tree :func:`knob` reads when no ``root`` is passed (``None`` clears it)."""
    _ROOT.clear()
    if root is not None:
        _ROOT.append(Path(root))


def current_root() -> Path | None:
    """The tree :func:`knob` reads without an explicit ``root`` (see the module docstring)."""
    if _ROOT:
        return _ROOT[0]
    starts = [sys.argv[0]] if sys.argv and sys.argv[0] not in ("", "-c", "-m") else []
    for start in (*starts, __file__):
        found = find_root(start)
        if found is not None:
            return found
    return None


def knob(name: str, *, default, values, root=None):
    """The value of knob ``name`` in the tree: its settings entry, else ``default``.

    ``values`` are the values this call site accepts. Every value the declaration lists for ``name``
    must be one of them (copying tooling may set any declared value), and so must the current setting.
    """
    values = tuple(values)
    if not values or not all(scalar(v) for v in values) or not _member(default, values):
        raise KnobError(f"knob {name!r}: default {default!r} must be one of the JSON scalar values {list(values)!r}")
    tree = Path(root) if root is not None else current_root()
    declared = declaration(tree) if tree is not None else None
    if declared is None:
        value = default
    else:
        spec = declared["knobs"].get(name)
        if spec is not None:
            extra = [v for v in spec["values"] if not _member(v, values)]
            if extra:
                raise KnobError(f"knob {name!r}: declared value(s) {extra!r} are not accepted by this call "
                                f"site {list(values)!r}")
        value = settings(tree, declared).get(name, default)
        if not _member(value, values):
            raise KnobError(f"knob {name!r}: setting {value!r} in {declared['settings_file']} is not one of "
                            f"{list(values)!r}")
    _CONSULTED[name] = value
    return value


def consulted() -> dict:
    """Every knob read in this process with the value returned last (to record beside an output)."""
    return dict(_CONSULTED)


def alternatives(root) -> list[dict]:
    """Every single-knob change from the tree's current settings, in declaration order."""
    declared = declaration(root)
    if declared is None:
        return []
    exists, _document = _read_json(Path(root) / declared["settings_file"])
    if not exists:
        return []
    current = settings(root, declared)
    found = []
    for name, spec in declared["knobs"].items():
        for value in spec["values"]:
            if name in current and same_value(current[name], value):
                continue
            found.append({"knob": name, "value": value, "current": current.get(name),
                          "mechanism": spec["mechanism"], "settings_file": declared["settings_file"]})
    return found


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------

def _write(path: Path, document: dict, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.next")
    staged.write_text(json.dumps(document, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")
    staged.replace(path)
    _CACHE.pop(str(path.resolve()), None)


def declare(root, knobs: Mapping[str, Knob], *, settings_file: str = DEFAULT_SETTINGS) -> Path:
    """Write the tree's declaration for ``knobs`` and give each knob its default where unset.

    Settings entries that already hold a value of their knob are kept, other entries are untouched.
    Returns the declaration path.
    """
    root = Path(root)
    if _relative(settings_file) is None or not isinstance(knobs, Mapping) or not 0 < len(knobs) <= MAX_KNOBS:
        raise KnobError(f"declare needs 1 to {MAX_KNOBS} knobs and a relative .json settings_file")
    if any(type(spec) is not Knob or not isinstance(name, str) or not 0 < len(name) <= MAX_NAME
           for name, spec in knobs.items()):
        raise KnobError(f"declare takes {{name: Knob}} with names of 1 to {MAX_NAME} characters")
    document = {"schema": SCHEMA, "settings_file": settings_file,
                "knobs": {name: spec.spec() for name, spec in knobs.items()}}
    exists, current = _read_json(root / settings_file)
    if exists and not isinstance(current, dict):
        raise KnobError(f"{settings_file} must hold a JSON object")
    current = dict(current or {})
    for name, spec in knobs.items():
        if name not in current or not _member(current[name], spec.values):
            current[name] = spec.default
    _write(root / settings_file, current)
    _write(root / DECLARATION, document, sort_keys=False)  # declaration order is the tuning order
    return root / DECLARATION


def set_values(root, overrides: Mapping) -> dict:
    """Set declared knobs to declared values in the tree's settings file; returns the new settings."""
    root = Path(root)
    declared = declaration(root)
    if declared is None:
        raise KnobError(f"{root} has no {DECLARATION}")
    for name, value in overrides.items():
        spec = declared["knobs"].get(name)
        if spec is None or not _member(value, spec["values"]):
            raise KnobError(f"{name}={value!r} is not a declared knob value")
    updated = {**settings(root, declared), **overrides}
    _write(root / declared["settings_file"], updated)
    return updated


def check(root) -> dict:
    """What copying tooling will see in the tree: validity, problems, current values and alternatives."""
    root = Path(root)
    report = {"tree": str(root), "valid": False, "problems": [], "current": {}, "alternatives": []}
    try:
        declared = declaration(root)
        if declared is None:
            report["problems"].append(f"no {DECLARATION}: nothing is declared, so nothing is varied")
            return report
        exists, _document = _read_json(root / declared["settings_file"])
        current = settings(root, declared)
    except KnobError as error:
        report["problems"].append(str(error))
        return report
    if not exists:
        report["problems"].append(f"{declared['settings_file']} does not exist: no value can be varied")
    for name, spec in declared["knobs"].items():
        if name in current and not _member(current[name], spec["values"]):
            report["problems"].append(f"{name}: current setting {current[name]!r} is not a declared value")
        if any(_member(v, spec["values"][:i]) for i, v in enumerate(spec["values"])):
            report["problems"].append(f"{name}: declared values repeat")
    report.update(valid=not report["problems"], settings_file=declared["settings_file"],
                  current={name: current.get(name) for name in declared["knobs"]},
                  alternatives=alternatives(root) if exists else [])
    return report


def _parse_assignment(text: str):
    name, sep, raw = text.partition("=")
    if not sep or not name:
        raise KnobError(f"expected name=value, got {text!r}")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    return name, value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m merlin.compile.tuning_knobs", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="validate a tree's declaration and list single-knob changes")
    checking.add_argument("tree")
    setting = commands.add_parser("set", help="set declared knobs to declared values (JSON or bare strings)")
    setting.add_argument("tree")
    setting.add_argument("assignments", nargs="+", metavar="name=value")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report = check(args.tree)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 1
        updated = set_values(args.tree, dict(_parse_assignment(text) for text in args.assignments))
        print(json.dumps(updated, indent=2, sort_keys=True))
        return 0
    except KnobError as error:
        print(json.dumps({"error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
