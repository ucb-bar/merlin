"""Current compiler selection, complete instruction IR and configuration obligations.

This API deliberately has no historical producer revision requirement. It binds
the actual original typed selection and reopens those facts/software bytes at
every freshness check. Declarations and qualification labels remain conditional
model premises, never independent execution or hardware approval.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

from .configuration_elision_bridge import canonical, digest, file_ref, project
from .configuration_elision_emit import FINAL_ROOT, MODULE, emit
from .configuration_elision_intake import PREMISES, proof_sources

DOMAIN = "current-compiler-straightline-configuration-entries-v1"
RELATION = "equal_modeled_complete_entry_error_and_terminal_state"


def request(selection):
    if type(selection) is not ConfigurationIRSelection:
        raise ValueError("original typed ConfigurationIRSelection required")
    selection.revalidate()
    selected = selection.selected_contracts
    return {
        "schema": "merlin.compiler-configuration.selection.v1",
        "facts": file_ref(selected.facts_path),
        "software": file_ref(selected.software_path),
        "target": selected.target,
        "qualification_identity": selection.qualification_identity,
        "enabled": selection.enabled,
        "max_operations": selection.max_operations,
        "max_fields": selection.max_fields,
        "max_ir_bytes": selection.max_ir_bytes,
        "selection": selection.record(),
        "qualification_is_premise_only": True,
    }


def reopen(value):
    required = {
        "schema",
        "facts",
        "software",
        "target",
        "qualification_identity",
        "enabled",
        "max_operations",
        "max_fields",
        "max_ir_bytes",
        "selection",
        "qualification_is_premise_only",
    }
    if (
        type(value) is not dict
        or set(value) != required
        or value["schema"] != "merlin.compiler-configuration.selection.v1"
    ):
        raise ValueError("exact current compiler configuration selection required")
    if value["qualification_is_premise_only"] is not True:
        raise ValueError("a caller label cannot establish instruction/effect qualification")
    for key in ("facts", "software"):
        if file_ref(value[key]["path"]) != value[key]:
            raise ValueError("original selected configuration input bytes changed")
    selected = read_scheduling_contracts(value["facts"]["path"], value["software"]["path"], target=value["target"])
    selection = ConfigurationIRSelection(
        selected,
        value["qualification_identity"],
        enabled=value["enabled"],
        max_operations=value["max_operations"],
        max_fields=value["max_fields"],
        max_ir_bytes=value["max_ir_bytes"],
    )
    if request(selection) != value:
        raise ValueError("current configuration resolver/selection changed")
    return selection


def _sources(selection):
    modules = set(dict(selection.selected_contracts.resolution_sources))
    modules.update(
        {
            "merlin.verify.compiler_configuration_intake",
            "merlin.verify.compiler_configuration_kernel",
            "merlin.verify.configuration_hoist_bridge",
            "merlin.verify.compiler_configuration",
            "merlin.verify.configuration_elision_runner",
            "merlin.verify.configuration_elision_bridge",
            "merlin.verify.configuration_elision_emit",
            "merlin.verify.configuration_elision_intake",
            "merlin.verify.configuration_elision_gate",
            "merlin.common.owned_process",
            "merlin.common.owned_children",
            "merlin.common.paths",
            "merlin.common.owned_group",
            "merlin.verify.pass_validation",
            "merlin.verify.gate_checks",
            "merlin.verify.gate_worker",
            "merlin.verify.gate_errors",
            "merlin.verify.speculative_gate",
            "merlin.verify.lean_emission_context",
            "merlin.targetgen.capsule_common",
            "merlin.targetgen.shared_pass_contract",
            "merlin.targetgen.package_runtime",
            "merlin.llvmlower.device_build",
            "merlin.llvmlower.target_schedule",
            "merlin.compile.scheduling.pointwise_source",
            "merlin.compile.scheduling.multioutput_fusion",
            "merlin.sched.ir.events",
        }
    )
    rows = [file_ref(module_source_path(name)) for name in sorted(modules)]
    rows += [{key: row[key] for key in ("path", "sha256", "bytes")} for row in proof_sources()]
    rows.append(file_ref(Path(module_source_path("merlin.verify")).parent / "lean/Merlin/GraphStateAudit.lean"))
    return sorted({row["path"]: row for row in rows}.values(), key=lambda row: row["path"])


@dataclass(frozen=True)
class CurrentCandidate:
    source: str
    final: str
    request_json: str
    resolution_json: str
    projection_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    proof_sources_json: str
    representation: str = "instruction_mlir"
    domain: str = DOMAIN
    relation: str = RELATION

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        projection = json.loads(self.projection_json)
        return {
            "schema": "merlin.compiler-configuration.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "representation": self.representation,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "final_root": FINAL_ROOT,
            "roots": list(self.roots),
            "module": MODULE,
            "entries": [entry["name"] for entry in projection["before"]["entries"]],
            "fields": projection["fields"],
            "premises": [
                (
                    "trusted complete typed Kernel projection and source-owned ISA resolver/bit interpretation"
                    if self.representation == "schedule_kernel"
                    else PREMISES[0]
                ),
                *PREMISES[1:],
            ],
            "authority": False,
            "hardware_model_approved": False,
            "independently_qualified": False,
            "production_enforcement": False,
        }


def prepare(source, final, *, selection, representation="instruction_mlir"):
    selected = request(selection)
    if not selection.enabled or selection.qualification_identity is None:
        raise ValueError("positive configuration proof requires original enabled qualified-premise selection")
    for text in (source, final):
        if type(text) is not str or len(text.encode()) > selection.max_ir_bytes:
            raise ValueError("bounded actual configuration instruction buffers required")
    if source == final:
        raise ValueError("unchanged configuration boundary supplies no positive erasure coverage")
    isa = selection.selected_contracts.isa.record()
    effects = json.loads(selection.selected_contracts.effects_json)
    if representation == "instruction_mlir":
        projector = partial(project, allow_bare=True)
    elif representation == "schedule_kernel":
        from .compiler_configuration_kernel import project as projector
    else:
        raise ValueError("unknown configuration compiler representation")
    projection = projector(source, final, isa=isa, effects=effects)
    count = sum(len(e["code"]) for e in projection["before"]["entries"])
    if count > selection.max_operations or len(projection["fields"]) > selection.max_fields:
        raise ValueError("actual configuration proof exceeds original selection budgets")
    generated = emit(projection)
    resolved = {
        "isa": isa,
        "effects": effects,
        "instruction_byteorder": selection.selected_contracts.instruction_byteorder,
        "hardware_model_approved": False,
    }
    return CurrentCandidate(
        source,
        final,
        canonical(selected),
        canonical(resolved),
        canonical(projection),
        generated["query"],
        tuple(generated["roots"]),
        canonical(_sources(selection)),
        canonical(proof_sources()),
        representation,
    )


def revalidate(candidate):
    if type(candidate) is not CurrentCandidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise ValueError("exact current compiler configuration candidate required")
    for row in json.loads(candidate.sources_json):
        if file_ref(row["path"]) != row:
            raise ValueError("current configuration source/proof/selection changed")
    selection = reopen(json.loads(candidate.request_json))
    if (
        prepare(candidate.source, candidate.final, selection=selection, representation=candidate.representation)
        != candidate
    ):
        raise ValueError("forged/stale current configuration buffers/projection/query/roots")
    return {"fresh": True, "authority": False}
