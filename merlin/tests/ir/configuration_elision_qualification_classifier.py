"""Conservative bound-source attribution of a native configuration refusal."""

import hashlib
from pathlib import Path

OBLIGATION = "theorem accepted : Entries.check fields before after = true := by decide"
ROOT = "ActualConfigurationElision.accepted"
UNAVAILABLE_MARKERS = (
    "heartbeat",
    "recursion",
    "out of memory",
    "stack overflow",
    "stack space",
    "panic",
    "internal error",
    "failed to import",
    "unknown constant",
    "unknown identifier",
    "object file",
    "uncaught exception",
    "error while loading",
    "failed to open",
    "traceback",
)


def classify(directory, candidate_query):
    directory = Path(directory).resolve()
    query = directory / "ConfigurationElisionQuery.lean"
    process_file = directory / "query.process.json"
    if not process_file.exists() or not query.exists():
        return {"status": "UNAVAILABLE", "reason": "no actual query/source record"}
    import json

    process = json.loads(process_file.read_text())
    stdout, stderr = (directory / "query.stdout").read_text(), (directory / "query.stderr").read_text()
    if process["returncode"] != 1 or stderr.strip():
        return {"status": "UNAVAILABLE", "reason": "not an ordinary stdout-only exit1"}
    if query.read_text() != candidate_query or str(query) not in process["command"]:
        return {"status": "UNAVAILABLE", "reason": "query command/source association mismatch"}
    if any(marker in (stdout + stderr).lower() for marker in UNAVAILABLE_MARKERS):
        return {"status": "UNAVAILABLE", "reason": "resource/import/internal failure"}
    lines = candidate_query.splitlines()
    if lines.count(OBLIGATION) != 1:
        return {"status": "UNAVAILABLE", "reason": "expected unique closed obligation unavailable"}
    line = lines.index(OBLIGATION) + 1
    headers = [item for item in stdout.splitlines() if ": error: " in item]
    expected = str(query) + ":" + str(line) + ":"
    if len(headers) != 1 or not headers[0].startswith(expected):
        return {"status": "UNAVAILABLE", "reason": "different or mixed obligation failure"}
    column, separator, message = headers[0][len(expected) :].partition(": error: ")
    if not separator or not column.isdigit() or message != "Tactic `decide` proved that the proposition":
        return {"status": "UNAVAILABLE", "reason": "unknown proof diagnostic"}
    if "is false" not in [item.strip() for item in stdout.splitlines()]:
        return {"status": "UNAVAILABLE", "reason": "missing closed-false diagnostic"}
    return {
        "status": "ACTUAL_EXPECTED_LEAN_OBLIGATION_REFUSAL",
        "root": ROOT,
        "line": line,
        "source_sha256": hashlib.sha256(query.read_bytes()).hexdigest(),
        "authority": False,
    }
