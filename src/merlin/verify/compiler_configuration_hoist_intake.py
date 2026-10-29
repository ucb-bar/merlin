"""Current original-selection obligations for scoped complete Kernel hoisting.

Source parser/ISA/effect fidelity remains an explicit premise. Current source
buffers and original selected inputs replace historical producer reconstruction;
no retained report or caller-selected proof tree supplies acceptance authority.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from merlin.common.paths import module_source_path
from merlin.sched.ir import Kernel, parse_kernel
from merlin.sched.ir.kernel import check_structure

from .compiler_configuration_intake import _sources as configuration_sources
from .compiler_configuration_intake import reopen, request
from .compiler_configuration_kernel import tree
from .configuration_elision_bridge import canonical, digest, file_ref
from .configuration_hoist_bridge import Unsupported, project_current
from .configuration_hoist_emit import MODULE, ROOT, emit
from .configuration_hoist_intake import proof_sources

DOMAIN = "current-compiler-one-positive-loop-constant-pure-hoist-v1"
INDEXED_DOMAIN = "current-compiler-one-positive-loop-indexed-retained-data-hoist-v2"
DOMAINS = frozenset({DOMAIN, INDEXED_DOMAIN})
RELATION = "equal_conditional_complete_kernel_error_terminal_state_hoist"


def _program(text):
    kernel = parse_kernel(text)
    if type(kernel) is not Kernel or check_structure(kernel) or kernel.text() != text:
        raise ValueError("exact complete typed canonical Kernel source required")
    return tree(kernel)


def _sources(selection, *, indexed=False):
    rows = list(configuration_sources(selection))
    if indexed:
        rows.append(file_ref(module_source_path("merlin.verify.compiler_configuration_hoist_indexed")))
    rows += [
        file_ref(module_source_path(name))
        for name in (
            "merlin.verify.compiler_configuration_hoist_intake",
            "merlin.verify.compiler_configuration_hoist",
            "merlin.verify.configuration_hoist_intake",
            "merlin.verify.configuration_hoist_emit",
            "merlin.verify.configuration_hoist_bridge",
            "merlin.compile.scheduling.configuration_loops",
            "merlin.sched.codegen",
            "merlin.sched.ir.events",
        )
    ]
    rows += [{key: row[key] for key in ("path", "sha256", "bytes")} for row in proof_sources()]
    return sorted({row["path"]: row for row in rows}.values(), key=lambda row: row["path"])


@dataclass(frozen=True)
class CurrentHoistCandidate:
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
            "schema": "merlin.compiler-configuration-hoist.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "representation": "schedule_kernel",
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "module": MODULE,
            "final_root": ROOT,
            "roots": list(self.roots),
            "entries": [json.loads(projection["before"]["frame"])["name"]],
            "fields": projection["fields"],
            "authority": False,
            "hardware_model_approved": False,
            "independently_qualified": False,
            "production_enforcement": False,
            "data_argument_admission": (
                "trusted_host_index64_bounds" if self.domain == INDEXED_DOMAIN else "static_constants"
            ),
            "all_index_model_premise_retained": True,
            "premises": [
                "trusted complete typed Kernel parser/projection and original selected ISA/effect interpretation",
                "exact nonaliasing configuration/world cells and complete relevant "
                "caller/return/memory/error/observer state",
                "one common indexed partial meaning for retained ordered calls respecting declared writes and frame",
                "moved setter is a total exact overwrite with no failure/pulse/token/memory/observer effect",
            ],
        }


def prepare(source, final, *, selection):
    selected = request(selection)
    if not selection.enabled or selection.qualification_identity is None:
        raise ValueError("enabled original selected effect-premise owner required")
    if any(type(text) is not str or len(text.encode()) > selection.max_ir_bytes for text in (source, final)):
        raise ValueError("bounded actual complete Kernel text required")
    if source == final:
        raise ValueError("unchanged hoist boundary supplies no positive motion coverage")
    isa = selection.selected_contracts.isa.record()
    effects = json.loads(selection.selected_contracts.effects_json)
    source_tree, final_tree = _program(source), _program(final)
    domain = DOMAIN
    try:
        projection = project_current(source_tree, final_tree, isa=isa, effects=effects)
    except Unsupported:
        from .compiler_configuration_hoist_indexed import project

        projection = project(source_tree, final_tree, isa=isa, effects=effects)
        domain = INDEXED_DOMAIN
    if (
        len(projection["before"]["atoms"]) > selection.max_operations
        or len(projection["fields"]) > selection.max_fields
    ):
        raise ValueError("actual hoist projection exceeds original selected budgets")
    generated = emit(projection)
    resolution = {
        "isa": isa,
        "effects": effects,
        "instruction_byteorder": selection.selected_contracts.instruction_byteorder,
        "hardware_model_approved": False,
    }
    return CurrentHoistCandidate(
        source,
        final,
        canonical(selected),
        canonical(resolution),
        canonical(projection),
        generated["query"],
        tuple(generated["roots"]),
        canonical(_sources(selection, indexed=domain == INDEXED_DOMAIN)),
        canonical(proof_sources()),
        domain=domain,
    )


def revalidate(candidate):
    if (
        type(candidate) is not CurrentHoistCandidate
        or type(candidate.domain) is not str
        or candidate.domain not in DOMAINS
        or candidate.relation != RELATION
    ):
        raise ValueError("exact current complete Kernel hoist candidate required")
    for row in json.loads(candidate.sources_json):
        if file_ref(row["path"]) != row:
            raise ValueError("current hoist source/proof/selection changed")
    if prepare(candidate.source, candidate.final, selection=reopen(json.loads(candidate.request_json))) != candidate:
        raise ValueError("forged/stale complete Kernel/ISA/fields/query/roots")
    return {"fresh": True, "authority": False}
