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
from .configuration_hoist_bridge import Unsupported, project_current, sequence
from .configuration_hoist_emit import MODULE, ROOT, emit
from .configuration_hoist_intake import proof_sources as single_proof_sources

DOMAIN = "current-compiler-one-positive-loop-constant-pure-hoist-v1"
INDEXED_DOMAIN = "current-compiler-one-positive-loop-indexed-retained-data-hoist-v2"
PREFIX_DOMAIN = "current-compiler-one-positive-loop-pure-prefix-fixed-state-hoist-v3"
MULTI_DOMAIN = "current-compiler-multiple-positive-flat-loops-pure-prefix-hoist-v4"
WIDE_DOMAIN = "current-compiler-original-C64-consumer-wide-flat-prefix-hoist-v5"
NESTED_DOMAIN = "current-compiler-recursive-positive-loops-pure-prefix-hoist-v6"
DOMAINS = frozenset({DOMAIN, INDEXED_DOMAIN, PREFIX_DOMAIN, MULTI_DOMAIN, WIDE_DOMAIN, NESTED_DOMAIN})
RELATION = "equal_conditional_complete_kernel_error_terminal_state_hoist"


def _program(text):
    kernel = parse_kernel(text)
    if type(kernel) is not Kernel or check_structure(kernel) or kernel.text() != text:
        raise ValueError("exact complete typed canonical Kernel source required")
    return tree(kernel)


def proof_sources(domain=DOMAIN):
    if domain == NESTED_DOMAIN:
        from .compiler_configuration_hoist_recursive import proof_sources as recursive_sources

        return recursive_sources()
    if domain in {MULTI_DOMAIN, WIDE_DOMAIN}:
        from .compiler_configuration_hoist_multi import proof_sources as multi_sources

        return multi_sources()
    if domain == PREFIX_DOMAIN:
        from .compiler_configuration_hoist_prefix import proof_sources as prefix_sources

        return prefix_sources()
    if domain not in {DOMAIN, INDEXED_DOMAIN}:
        raise ValueError("exact finite canonical hoist profile required")
    return single_proof_sources()


def emit_profile(projection, domain):
    if domain == NESTED_DOMAIN:
        from .configuration_hoist_recursive_emit import emit as recursive_emit

        return recursive_emit(projection)
    if domain in {MULTI_DOMAIN, WIDE_DOMAIN}:
        from .configuration_hoist_multi_emit import emit as multi_emit

        return multi_emit(projection)
    if domain == PREFIX_DOMAIN:
        from .configuration_hoist_prefix_emit import emit as prefix_emit

        return prefix_emit(projection)
    if domain not in {DOMAIN, INDEXED_DOMAIN}:
        raise ValueError("exact finite canonical hoist profile required")
    return emit(projection)


def _sources(selection, *, indexed=False, prefix=False, multi=False, recursive=False, c_consumer=None):
    rows = list(configuration_sources(selection))
    if c_consumer is not None:
        rows += c_consumer.record()["sources"]
        rows.append(file_ref(module_source_path("merlin.verify.compiler_configuration_hoist_wide")))
    if indexed:
        rows.append(file_ref(module_source_path("merlin.verify.compiler_configuration_hoist_indexed")))
    if prefix:
        rows += [
            file_ref(module_source_path(name))
            for name in (
                "merlin.verify.compiler_configuration_hoist_prefix",
                "merlin.verify.configuration_hoist_prefix_emit",
            )
        ]
    if multi:
        rows += [
            file_ref(module_source_path(name))
            for name in (
                "merlin.verify.compiler_configuration_hoist_multi",
                "merlin.verify.configuration_hoist_multi_emit",
            )
        ]
    if recursive:
        rows += [
            file_ref(module_source_path(name))
            for name in (
                "merlin.verify.compiler_configuration_hoist_recursive",
                "merlin.verify.configuration_hoist_recursive_emit",
            )
        ]
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
    rows += [
        {key: row[key] for key in ("path", "sha256", "bytes")}
        for row in proof_sources(
            NESTED_DOMAIN if recursive else MULTI_DOMAIN if multi else PREFIX_DOMAIN if prefix else DOMAIN
        )
    ]
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
        request_record = json.loads(self.request_json)
        consumer = request_record.get("c_consumer") if request_record.get("schema") == REQUEST_V2 else None
        return {
            "schema": "merlin.compiler-configuration-hoist.candidate.v2"
            if consumer is not None
            else "merlin.compiler-configuration-hoist.candidate.v1",
            **(
                {"c_consumer": consumer, "index_domain_admission": "original_actual_C_consumer_index64"}
                if consumer is not None
                else {}
            ),
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
                "trusted_host_index64_bounds" if projection.get("indexed_data_analysis") else "static_constants"
            ),
            "all_index_model_premise_retained": True,
            **(
                {"all_environment_model_premise_retained": True, "max_loop_depth": projection["max_loop_depth"]}
                if self.domain == NESTED_DOMAIN
                else {}
            ),
            "moved_setter_count": (
                sum(len(stage["move"]["setups"]) for stage in projection["stages"])
                if self.domain == NESTED_DOMAIN
                else sum(len(stage["focus"]["move"]["setups"]) for stage in projection["stages"])
                if self.domain in {MULTI_DOMAIN, WIDE_DOMAIN}
                else len(projection["move"]["setups"])
                if self.domain == PREFIX_DOMAIN
                else 1
            ),
            **(
                {
                    "actual_loop_count": projection["actual_loop_count"],
                    "focused_loop_count": projection["focused_loop_count"],
                }
                if self.domain in {MULTI_DOMAIN, WIDE_DOMAIN, NESTED_DOMAIN}
                else {}
            ),
            "premises": [
                "trusted complete typed Kernel parser/projection and original selected ISA/effect interpretation",
                "exact nonaliasing configuration/world cells and complete relevant "
                "caller/return/memory/error/observer state",
                "one common all-environment indexed partial meaning for retained ordered calls "
                "respecting declared writes and frame"
                if self.domain == NESTED_DOMAIN
                else "one common indexed partial meaning for retained ordered calls "
                "respecting declared writes and frame",
                "each moved setter is a total exact overwrite with no failure/pulse/token/memory/observer effect",
            ],
        }


REQUEST_V2 = "merlin.compiler-configuration-hoist.request.v2"


def reopen_request(value):
    if type(value) is dict and value.get("schema") == REQUEST_V2:
        from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

        if set(value) != {"schema", "selection", "c_consumer"}:
            raise ValueError("exact original ISA/C-consumer request required")
        return reopen(value["selection"]), ConfigurationCConsumer.reopen(value["c_consumer"])
    return reopen(value), None


def prepare(source, final, *, selection, c_consumer=None):
    selected = request(selection)
    if c_consumer is not None:
        from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

        if type(c_consumer) is not ConfigurationCConsumer:
            raise ValueError("original typed configuration C consumer required")
        c_consumer.revalidate()
        selected = {"schema": REQUEST_V2, "selection": selected, "c_consumer": c_consumer.record()}
    if not selection.enabled or selection.qualification_identity is None:
        raise ValueError("enabled original selected effect-premise owner required")
    if any(type(text) is not str or len(text.encode()) > selection.max_ir_bytes for text in (source, final)):
        raise ValueError("bounded actual complete Kernel text required")
    if source == final:
        raise ValueError("unchanged hoist boundary supplies no positive motion coverage")
    isa = selection.selected_contracts.isa.record()
    effects = json.loads(selection.selected_contracts.effects_json)
    source_tree, final_tree = _program(source), _program(final)
    wide = c_consumer is not None and any(
        row.get("class") == "merlin.sched.ir.kernel.Loop" and row["fields"]["extent"] > 1_000_000
        for kernel in (source_tree, final_tree)
        for row in sequence(kernel["fields"]["body"])
    )
    if c_consumer is not None:
        c_consumer.admit(parse_kernel(source))
        c_consumer.admit(parse_kernel(final))
    nested = any(
        row.get("class") == "merlin.sched.ir.kernel.Loop"
        and any(child.get("class") == "merlin.sched.ir.kernel.Loop" for child in sequence(row["fields"]["body"]))
        for row in sequence(source_tree["fields"]["body"])
    )
    if nested:
        from .compiler_configuration_hoist_recursive import project_current as recursive_project

        projection = recursive_project(source, final, isa=isa, effects=effects, c_consumer=c_consumer)
        domain = NESTED_DOMAIN
    elif wide:
        from .compiler_configuration_hoist_wide import project_current as wide_project

        projection = wide_project(source, final, isa=isa, effects=effects, c_consumer=c_consumer)
        domain = WIDE_DOMAIN
    else:
        domain = DOMAIN
        try:
            projection = project_current(source_tree, final_tree, isa=isa, effects=effects)
        except Unsupported:
            from .compiler_configuration_hoist_indexed import project

            try:
                projection = project(source_tree, final_tree, isa=isa, effects=effects)
                domain = INDEXED_DOMAIN
            except Unsupported:
                from .compiler_configuration_hoist_prefix import project as prefix_project

                try:
                    projection = prefix_project(source_tree, final_tree, isa=isa, effects=effects)
                    domain = PREFIX_DOMAIN
                except Unsupported:
                    # Preserve single-loop profile refusals and diagnostics. V4 is
                    # considered only for actual multiple top-level loop scopes.
                    if (
                        sum(
                            row.get("class") == "merlin.sched.ir.kernel.Loop"
                            for row in sequence(source_tree["fields"]["body"])
                        )
                        < 2
                    ):
                        raise
                    from .compiler_configuration_hoist_multi import project as multi_project

                    projection = multi_project(source_tree, final_tree, isa=isa, effects=effects)
                    domain = MULTI_DOMAIN
    if (
        len(projection["before"]["atoms"]) > selection.max_operations
        or len(projection["fields"]) > selection.max_fields
    ):
        raise ValueError("actual hoist projection exceeds original selected budgets")
    generated = emit_profile(projection, domain)
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
        canonical(
            _sources(
                selection,
                indexed=domain != DOMAIN,
                prefix=domain in {PREFIX_DOMAIN, MULTI_DOMAIN, WIDE_DOMAIN, NESTED_DOMAIN},
                multi=domain in {MULTI_DOMAIN, WIDE_DOMAIN},
                recursive=domain == NESTED_DOMAIN,
                c_consumer=c_consumer,
            )
        ),
        canonical(proof_sources(domain)),
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
    selection, c_consumer = reopen_request(json.loads(candidate.request_json))
    if prepare(candidate.source, candidate.final, selection=selection, c_consumer=c_consumer) != candidate:
        raise ValueError("forged/stale complete Kernel/ISA/fields/query/roots")
    return {"fresh": True, "authority": False}
