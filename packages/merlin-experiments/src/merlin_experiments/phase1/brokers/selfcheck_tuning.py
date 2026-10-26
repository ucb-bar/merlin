"""Grade a private copy of ``submission/`` with author-declared knob values changed.

An author's compiler may expose some of its own choices -- whether to apply a library mechanism, a
traversal or blocking option -- as entries of one JSON settings file inside ``submission/``, and declare
them in ``submission/TUNING_KNOBS.json``::

    {"schema": "merlin.tuning_knobs.v1",
     "settings_file": "<relative path inside submission/ of the JSON object the compiler reads>",
     "knobs": {"<name>": {"values": [<JSON scalars>], "mechanism": "<catalogue id, optional>",
                          "note": "<optional>"}}}

Declaring knobs is optional and changes nothing by itself. A self-check request may carry
``tuning_overrides`` (``{name: value}``, declared names and declared values only). The broker copies the
tree into its private snapshot exactly as for any request and verifies the copy's digest; only then are
those keys set in the snapshot's settings file. The workspace is never written. The response says what
was graded: a ``variant`` block with ``official: false``, the overrides and the digest of the graded
copy, so every reader that sets variant results aside sets these aside too, and, like a variant, the
result promotes nothing and publishes no foreground marker.

Everything here is data the author declared; nothing names a target, a mechanism or a winning value.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.tuning_knobs.v1"
DECLARATION = "TUNING_KNOBS.json"
REQUEST_KEY = "tuning_overrides"
#: Request key the broker fills after applying the overrides; a client-supplied value is replaced.
APPLIED_KEY = "_tuning_applied"
#: Bounds on what one declaration and one request may carry.
MAX_KNOBS = 16
MAX_VALUES = 8
MAX_OVERRIDES = 4
MAX_TEXT = 200
MAX_BYTES = 1 << 20
MAX_ECHO_BYTES = 8192

NOTE = ("SELF-CHECK OF A PRIVATE COPY of submission/ with the listed knob values set in its settings file; "
        "submission/ itself was not changed. Like a variant, this result is never counted toward "
        "convergence, certifies nothing and starts no promotion. To adopt a value, set it in submission/.")


def _scalar(value) -> bool:
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return type(value) is str and len(value) <= 64


def same_value(left, right) -> bool:
    """Equal JSON scalars of the same type (``True`` is not ``1``)."""
    return type(left) is type(right) and left == right


def _relative(text) -> str | None:
    """A plain relative path below the tree: no absolute path, ``..``, empty or hidden component."""
    if not isinstance(text, str) or not text or len(text) > MAX_TEXT or "\\" in text or "\0" in text:
        return None
    path = PurePosixPath(text)
    if path.is_absolute() or path.suffix != ".json":
        return None
    if any(part in ("", ".", "..") or part.startswith(".") for part in path.parts):
        return None
    return path.as_posix()


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def declaration(root) -> dict | None:
    """The validated knob declaration of the tree at ``root``, or ``None`` when absent or malformed."""
    try:
        document = _read_json(Path(root) / DECLARATION)
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return None
    settings_file, knobs = _relative(document.get("settings_file")), document.get("knobs")
    if settings_file is None or not isinstance(knobs, dict) or not 0 < len(knobs) <= MAX_KNOBS:
        return None
    declared = {}
    for name, spec in knobs.items():
        values = spec.get("values") if isinstance(spec, dict) else None
        if (not isinstance(name, str) or not 0 < len(name) <= 64 or not isinstance(values, list)
                or not 0 < len(values) <= MAX_VALUES or not all(_scalar(value) for value in values)):
            return None
        declared[name] = {"values": values,
                          "mechanism": spec.get("mechanism") if isinstance(spec.get("mechanism"), str) else None,
                          "note": spec.get("note")[:MAX_TEXT] if isinstance(spec.get("note"), str) else None}
    return {"settings_file": settings_file, "knobs": declared}


def settings(root, declared: dict) -> dict | None:
    """The settings object the declaration names, or ``None`` when it is not a JSON object file."""
    try:
        document = _read_json(Path(root) / declared["settings_file"])
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def alternatives(root) -> list[dict]:
    """Every single-knob change from the tree's current settings, in declaration order."""
    declared = declaration(root)
    current = settings(root, declared) if declared is not None else None
    if current is None:
        return []
    found = []
    for name, spec in declared["knobs"].items():
        for value in spec["values"]:
            if name in current and same_value(current[name], value):
                continue
            found.append({"knob": name, "value": value, "current": current.get(name),
                          "mechanism": spec["mechanism"], "note": spec["note"],
                          "settings_file": declared["settings_file"]})
    return found


def requested(request) -> bool:
    """Whether a request asks for knob overrides (a malformed ask counts: it is refused, never ignored)."""
    return isinstance(request, dict) and request.get(REQUEST_KEY) not in (None, {})


def _overrides(request, declared: dict) -> dict:
    overrides = request.get(REQUEST_KEY)
    if not isinstance(overrides, dict) or not 0 < len(overrides) <= MAX_OVERRIDES:
        raise ValueError(f"{REQUEST_KEY} must map one to {MAX_OVERRIDES} declared knobs to declared values")
    for name, value in overrides.items():
        spec = declared["knobs"].get(name)
        if spec is None or not any(same_value(value, allowed) for allowed in spec["values"]):
            raise ValueError(f"{REQUEST_KEY}: knob or value not declared in submission/{DECLARATION}")
    return overrides


def apply_request(snapshot_root, request: dict, *, digest) -> dict:
    """Set the requested knob values in a private snapshot; returns the request with what was applied.

    ``snapshot_root`` is the broker's verified private copy (``<snapshot>/submission``); ``digest`` is
    the broker's tree digest. A request without overrides is returned unchanged (minus any
    client-supplied applied record). Raises ``ValueError`` for anything not declared by the copy itself.
    """
    request = {key: value for key, value in request.items() if key != APPLIED_KEY}
    if not requested(request):
        return request
    root = Path(snapshot_root)
    declared = declaration(root)
    if declared is None:
        raise ValueError(f"{REQUEST_KEY} needs a valid submission/{DECLARATION} ({SCHEMA})")
    overrides = _overrides(request, declared)
    current = settings(root, declared)
    if current is None:
        raise ValueError(f"{REQUEST_KEY}: the declared settings_file is not a JSON object file")
    target = root / declared["settings_file"]
    target.write_text(json.dumps({**current, **overrides}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request[APPLIED_KEY] = {"settings_file": declared["settings_file"], "overrides": overrides,
                            "sha256": digest(root)}
    return request


def response_fields(request) -> dict:
    """What a response states about a knob-override request (empty for any other request)."""
    fields: dict = {}
    if isinstance(request, dict) and isinstance(request.get("tuning_experiment"), dict):
        echo = request["tuning_experiment"]
        if len(json.dumps(echo, default=str)) <= MAX_ECHO_BYTES:
            fields["tuning_experiment"] = echo
    if not requested(request):
        return fields
    applied = request.get(APPLIED_KEY) if isinstance(request.get(APPLIED_KEY), dict) else {}
    overrides = request.get(REQUEST_KEY)
    fields["official"] = False
    fields["variant"] = {
        "name": None, "path": "submission", "kind": "knob_overrides",
        "tuning_overrides": overrides if len(json.dumps(overrides, default=str)) <= MAX_ECHO_BYTES else None,
        "settings_file": applied.get("settings_file"), "sha256": applied.get("sha256"),
        "applied": bool(applied), "official": False, "note": NOTE,
    }
    return fields
