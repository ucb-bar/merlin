"""Source-only phase1 preparation, with an explicit nonauthoritative proof stage."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from merlin import verify as installed_verify
from merlin.common import owned_process
from merlin.common.paths import build_dir

from .configuration_hoist_bridge import canonical, project
from .configuration_hoist_capture import capture, ref
from .configuration_hoist_emit import MODULE, ROOT, emit

DOMAIN = "one-positive-counted-loop-constant-pure-prefix-empty-completion-v1"
RELATION = "equal_conditional_complete_kernel_error_terminal_state_hoist"
NEW_MODULES = {
    "Merlin." + name
    for name in (
        "GraphStateConfigurationPartialHoist",
        "GraphStateConfigurationHoistModel",
        "GraphStateConfigurationHoistChecker",
    )
}


def digest(data):
    return hashlib.sha256(data.encode() if type(data) is str else data).hexdigest()


def lean_root():
    return Path(installed_verify.__file__).resolve().parent / "lean"


def proof_plan(candidate_proof_root=None):
    base = lean_root()
    canonical_ready = all((base / (name.replace(".", "/") + ".lean")).is_file() for name in NEW_MODULES)
    if not canonical_ready and candidate_proof_root is None:
        raise ValueError("reviewed canonical hoist proof sources unavailable; candidate roots cannot authorize checks")
    seen, rows = set(), []

    def visit(name):
        if name in seen:
            return
        if not name.startswith("Merlin.") or any(not part.isidentifier() for part in name.split(".")):
            raise ValueError("invalid exact hoist proof import")
        seen.add(name)
        selected = base if canonical_ready or name not in NEW_MODULES else Path(candidate_proof_root)
        path = selected / (name.replace(".", "/") + ".lean")
        if path.is_symlink():
            raise ValueError("hoist proof source must be an exact selected file")
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                for child in words[1:]:
                    if child.startswith("Merlin."):
                        visit(child)
        rows.append({"module": name, **ref(path)})

    visit("Merlin.GraphStateConfigurationHoistChecker")
    return ("canonical" if canonical_ready else "isolated_phase1_candidate"), rows


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
    proof_stage: str
    domain: str = DOMAIN
    relation: str = RELATION

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        return {
            "schema": "merlin.configuration-hoist.candidate.v1",
            "identity": self.identity,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "module": MODULE,
            "final_root": ROOT,
            "roots": list(self.roots),
            "domain": self.domain,
            "relation": self.relation,
            "proof_stage": self.proof_stage,
            "kernel_outcome": "UNOBSERVED",
            "authority": False,
            "ISA_effect_hardware_approved": False,
            "production_enforcement": False,
            "premises": [
                "trusted full Kernel parser/typed dataclass projection and original schema/codec",
                "exact nonaliasing configuration/world cell interpretation",
                "total exact pure setter with no other failure/observation/effect",
                "one common indexed partial retained-call meaning obeying declared ordered effects/frame",
                "all relevant caller/return/memory/error/observer state represented; no native fidelity claim",
            ],
        }


def prepare(source, final, *, reference, facts, software, target, source_manifest, candidate_proof_root=None):
    if any(type(text) is not str or len(text.encode()) > 512 * 1024 for text in (source, final)):
        raise ValueError("bounded complete actual Kernel text required")
    request = {
        "reference": str(Path(reference).resolve()),
        "facts": str(Path(facts).resolve()),
        "software": str(Path(software).resolve()),
        "target": target,
        "source_manifest": str(Path(source_manifest).resolve()),
    }
    if candidate_proof_root is not None:
        request["candidate_proof_root"] = str(Path(candidate_proof_root).resolve())
    worker_request = {key: value for key, value in request.items() if key != "candidate_proof_root"}
    worker_request.update(mode="project", source=source, final=final)
    directory = build_dir() / "configuration-hoist-preparation" / uuid.uuid4().hex
    actual = capture(worker_request, directory)
    projection = project(actual)
    generated = emit(projection)
    stage, proofs = proof_plan(candidate_proof_root)
    here = Path(__file__).parent
    host = [
        here / ("configuration_hoist_" + name + ".py")
        for name in ("worker", "capture", "bridge", "emit", "intake", "gate")
    ]
    core = Path(installed_verify.__file__).resolve().parent.parent
    host += [here / "__init__.py"]
    host += [
        core / name
        for name in (
            "__init__.py",
            "common/__init__.py",
            "common/paths.py",
            "common/owned_process.py",
            "verify/__init__.py",
            "verify/configuration_elision_emit.py",
            "verify/lean/Merlin/GraphStateAudit.lean",
        )
    ]
    helpers, pending, seen = [], [owned_process], set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(core):
            continue
        helpers.append(ref(path))
        pending.extend(
            value
            for value in vars(module).values()
            if isinstance(value, ModuleType) and getattr(value, "__file__", None)
        )
    sources = [
        *proofs,
        *(ref(path) for path in host),
        *helpers,
        actual["facts"],
        actual["software"],
        actual["source_manifest"],
        *actual["project_sources"],
    ]
    unique = {row["path"]: {key: row[key] for key in ("path", "sha256", "bytes")} for row in sources}
    return Candidate(
        source,
        final,
        canonical(request),
        canonical(actual),
        canonical(projection),
        generated["query"],
        tuple(generated["roots"]),
        canonical([unique[key] for key in sorted(unique)]),
        canonical(proofs),
        stage,
    )


def proof_sources():
    # This path intentionally has NO artifact fallback and no caller-supplied
    # source root. Only canonical post-review sources may reach native checks.
    stage, sources = proof_plan()
    if stage != "canonical":
        raise ValueError("isolated phase1 candidate is not native authority")
    return sources


def revalidate(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise ValueError("exact current hoist request required")
    for row in json.loads(candidate.sources_json):
        if ref(row["path"]) != row:
            raise ValueError("hoist bound source/schema/proof changed")
    if prepare(candidate.source, candidate.final, **json.loads(candidate.request_json)) != candidate:
        raise ValueError("changed full Kernel/calls/control/metadata/fields/query/roots/source identity")
    return {"fresh": True, "authority": False, "kernel_outcome": "UNOBSERVED"}
