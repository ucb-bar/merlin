"""Fresh source-bound complete-entry requests for configuration elimination.

Preparation produces only closed obligations and explicit model premises. It
does not approve a schema/effect model or create a checked completion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .configuration_elision_bridge import canonical, digest, file_ref, project, run_reference
from .configuration_elision_emit import FINAL_ROOT, MODULE, emit

DOMAIN = "straightline-constant-integer-configuration-entries-v1"
RELATION = "equal_modeled_complete_entry_error_and_terminal_state"
PROOF_MODULE = "Merlin.GraphStateConfigurationEntries"
PREMISES = (
    "trusted complete xDSL module/typed-SSA projection and source-owned ISA resolver/bit interpretation",
    "exact nonaliasing configuration field-to-state-cell mapping distinct from SSA/caller/observation cells",
    "one common local operational interpretation of every retained instruction and external call",
    "ordered declared writes/frame/reads agree with successful instruction steps",
    "each erasable setter is a total exact overwrite with no failure/pulse/token/memory/observer effect",
    "State Value includes all relevant caller, return, memory, error-terminal and observer state",
    "no cross-invocation initial configuration assumption; each defined entry starts with unknown knowledge",
)


def proof_sources():
    """Exact canonical project source closure; no candidate/artifact fallback."""
    base = Path(__file__).parent / "lean"
    seen, sources = set(), []

    def visit(name):
        if name in seen:
            return
        if not name.startswith("Merlin.") or any(not part.isidentifier() for part in name.split(".")):
            raise ValueError("invalid configuration proof import")
        seen.add(name)
        path = base / (name.replace(".", "/") + ".lean")
        if not path.is_file() or path.is_symlink():
            raise ValueError("reviewed canonical configuration proof source unavailable: " + name)
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                for child in words[1:]:
                    if child.startswith("Merlin."):
                        visit(child)
        sources.append({"module": name, **file_ref(path)})

    visit(PROOF_MODULE)
    return sources


def _host_sources():
    parent = Path(__file__).parent
    paths = [
        parent / ("configuration_elision_" + name + ".py") for name in ("producer", "bridge", "emit", "intake", "gate")
    ]
    paths += [
        parent / "lean/Merlin/GraphStateAudit.lean",
        parent / "__init__.py",
        parent.parent / "__init__.py",
        parent.parent / "common/__init__.py",
        parent.parent / "common/paths.py",
        parent.parent / "common/owned_process.py",
    ]
    return [file_ref(path) for path in paths]


@dataclass(frozen=True)
class Candidate:
    source: str
    final: str
    request_json: str
    resolution_json: str
    projection_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    proof_sources_json: str
    domain: str = DOMAIN
    relation: str = RELATION

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        projection = json.loads(self.projection_json)
        return {
            "schema": "merlin.configuration-elision.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "final_root": FINAL_ROOT,
            "roots": list(self.roots),
            "module": MODULE,
            "entries": [entry["name"] for entry in projection["before"]["entries"]],
            "fields": projection["fields"],
            "premises": list(PREMISES),
            "authority": False,
            "native_outcome": "UNOBSERVED",
            "qualification": "UNQUALIFIED",
            "hardware_model_approved": False,
            "production_enforcement": False,
        }


def prepare(source, final, *, reference, facts, software, target):
    request = {
        "reference": str(Path(reference).resolve()),
        "facts": str(Path(facts).resolve()),
        "software": str(Path(software).resolve()),
        "target": target,
    }
    resolved = run_reference(**request)
    projection = project(source, final, isa=resolved["isa"], effects=resolved["effects"])
    generated = emit(projection)
    proofs = proof_sources()
    sources = [*proofs, *_host_sources(), resolved["facts"], resolved["software"], *resolved["source_refs"]]
    unique = {}
    for row in sources:
        value = {key: row[key] for key in ("path", "sha256", "bytes")}
        if row["path"] in unique and unique[row["path"]] != value:
            raise ValueError("conflicting configuration source bindings")
        unique[row["path"]] = value
    return Candidate(
        source,
        final,
        canonical(request),
        canonical(resolved),
        canonical(projection),
        generated["query"],
        tuple(generated["roots"]),
        canonical([unique[key] for key in sorted(unique)]),
        canonical(proofs),
    )


def revalidate(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise ValueError("exact current configuration candidate/domain/relation required")
    for row in json.loads(candidate.sources_json):
        if file_ref(row["path"]) != row:
            raise ValueError("configuration proof/host/selection source changed")
    if prepare(candidate.source, candidate.final, **json.loads(candidate.request_json)) != candidate:
        raise ValueError("stale or forged configuration buffers/SSA/fields/query/roots/sources")
    return {"fresh": True, "authority": False, "native_outcome": "UNOBSERVED"}
