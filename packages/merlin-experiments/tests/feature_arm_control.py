"""Owned process feature diagnostic; no runtime, cycle or correctness claims."""

import hashlib
import json
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


compiler, source, inputs, request, output = map(Path, sys.argv[1:])
selection = json.loads(request.read_bytes())
prepared = inputs.read_bytes()
assert digest(prepared) == selection["inputs_sha256"]
assert source.is_file()
raw = (compiler / "compiler").read_bytes()
artifact = output.with_name("observed.compiler")
artifact.write_bytes(raw)
domain = (selection["context"] or {}).get("domain", digest(b"owned diagnostic domain"))
counts = {
    "compiler_bytes": len(raw),
    "input_sum": sum(value for row in json.loads(prepared)["inputs"] for value in row["values"]),
}
region = {
    "id": "observed",
    "stages": ["preparation"],
    "feature_ids": ["compiler_bytes"],
    "context": counts,
    "parent": None,
    "accounting": "exclusive",
}
result = {
    key: selection[key]
    for key in ("compiler_sha256", "member_sha256", "corpus_sha256", "target_sha256", "scope_sha256", "inputs_sha256")
}
result.update(
    domain_sha256=domain,
    executable_sha256=digest(raw),
    dependencies_sha256=digest(b"diagnostic only"),
    evidence_sha256s=[digest(source.read_bytes())],
    cold=[region],
    warm=[region],
    functional_status="UNKNOWN",
    legality_status="UNKNOWN",
    artifact_files=[[artifact.name, digest(raw)]],
)
output.write_text(json.dumps(result, sort_keys=True) + "\n")
print("observed one compiler with prepared input sum", counts["input_sum"])
