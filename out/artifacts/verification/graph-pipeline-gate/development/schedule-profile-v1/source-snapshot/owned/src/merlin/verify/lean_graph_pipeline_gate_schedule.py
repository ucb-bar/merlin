"""Inactive complete-entry profile for actual sequential dependency scheduling.

Planner facts are static diagnostics, not execution laws. Full typed SSA and
opaque order are independently matched; the unchanged Scalar checker supplies
the operational obligation. No SDK/guard/old proof runner is called here.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from .lean_graph_pipeline_gate_projection import canonical, sha, source_ref, project_lean_sources, complete_query

PROFILE = "sequential_scalar_schedule_source_v1"
KIND = "sequential_dependency_schedule"
DOMAIN = "complete_regionfree_scalar_order_equal_outcome_v1"
RELATION = "all_defined_entries_equal_returns_errors_terminal_world"
PREMISES = (
    "trusted complete xDSL parsing, projection, exact source and host binding",
    "same frozen Scalar.Program/Model/evalProgram and complete-entry dispatcher",
    "closed total deterministic state-free primitive interpretation; not inferred from dependency hints",
    "same ordered opaque descriptor interpretation on explicit operands and entire persistent world",
    "opaque calls may fail; local successful result arity only; no source-success premise",
    "region-free scalar word bodies, explicit external declarations; no internal-call/CFG/region bridge",
    "private SSA stacks unobserved; native, alias/allocator/completion/poison/numerical fidelity separate",
)


@dataclass(frozen=True)
class Order:
    block_index: int
    permutation: tuple[int, ...]

    def validate(self):
        if (type(self.block_index) is not int or self.block_index <= 0
                or type(self.permutation) is not tuple or not self.permutation
                or any(type(i) is not int for i in self.permutation)):
            raise ValueError("exact positive current block and integer permutation required")


@dataclass(frozen=True)
class PreparedSchedule:
    schema: str
    profile: str
    kind: str
    domain: str
    relation: str
    premises: tuple[str, ...]
    python_version: str
    before: str
    after: str
    changed: bool
    order: Order
    matching: str
    projection: str
    query: str
    roots: tuple[str, ...]
    sources: tuple

    @property
    def identity(self):
        return sha(canonical(asdict(self)))

    def describe(self):
        return {**asdict(self), "identity": self.identity, "query_status": "UNCOMPILED",
                "outcome": "UNOBSERVED", "qualification": "UNQUALIFIED",
                "compiler_pass_chain": "not_established", "public_route": "UNSUPPORTED/HELD",
                "boundary_label": "actual_changed_schedule" if self.changed else "unchanged_order_identity"}


def _admit(module):
    from xdsl.dialects import builtin, func
    from .lean_graph_input_admission import admit_module
    admit_module(module, calls="external-only")
    ops = tuple(module.walk())
    if len(ops) > 2048:
        raise ValueError("complete schedule exceeds syntax budget2048")
    for op in ops:
        if op.successors or op.regions and type(op) not in (builtin.ModuleOp, func.FuncOp):
            raise ValueError("schedule profile refuses regions/CFG; no implicit totality bridge")
        for value in (*op.operands, *op.results, *(a for r in op.regions for b in r.blocks for a in b.args)):
            t = value.type
            if not (type(t) is builtin.IntegerType and t.signedness.data == builtin.Signedness.SIGNLESS
                    and 1 <= t.width.data <= 256):
                raise ValueError("schedule profile supports signless scalar word values only")


def _sources():
    paths = {Path(__file__), Path(__file__).with_name("lean_graph_pipeline_gate_schedule_requests.py"),
             Path(__file__).with_name("lean_graph_pipeline_gate_schedule_models.py")}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith(("merlin.", "xdsl.")) and path and Path(path).suffix == ".py":
            paths.add(Path(path))
    return tuple(source_ref(p) for p in sorted(paths)) + project_lean_sources(("Merlin.GraphStateGateEntries",))


def prepare(before, after, order):
    from merlin.transforms.certificates import parse_ir, ir_text
    from merlin.transforms.planning import apply_order, plan_dependencies
    from .lean_graph_input_admission import admit_text
    from .lean_graph_state_schedule_projection import dependencies, match_step
    from .lean_graph_state_scalar_capture import project

    if type(order) is not Order:
        raise ValueError("typed actual-order request required; saved report cannot substitute")
    order.validate()
    if any(type(text) is not str or len(text.encode()) > 4*1024*1024 for text in (before,after)):
        raise ValueError("complete bounded actual schedule buffers required")
    for text in (before,after):
        admit_text(text)
    first, last = parse_ir(before), parse_ir(after)
    _admit(first); _admit(last)
    # This independent matcher includes complete surrounding functions, exact
    # operand DAGs/multiplicity/scopes and ordered opaque anchors. It refuses a
    # commutation even if a producer dependency hint omits it.
    facts = dependencies(first, order.block_index)
    diagnostic = {**facts, "order": list(order.permutation)}
    matched = match_step(first, last, {"block_index": order.block_index,
                         "order": list(order.permutation), "witness": diagnostic})
    planned = asdict(plan_dependencies(first).witnesses[order.block_index])
    planned["order"] = order.permutation
    if canonical(planned) != canonical(diagnostic):
        raise ValueError("producer plan differs from independently recomputed dependencies")
    clone = first.clone()
    actual = apply_order(clone, order.block_index, order.permutation)
    if canonical(asdict(actual)) != canonical(diagnostic) or ir_text(clone) != after:
        raise ValueError("actual producer output/witness differs from complete requested boundary")
    projection = project(first, last, context=(first,last))
    query, roots = complete_query(projection)
    query = query.replace("merlin_transform", "merlin_schedule_transform")
    roots = tuple(r.replace("merlin_transform", "merlin_schedule_transform") for r in roots)
    if len(query.encode()) > 1024*1024 or len(set(roots)) != len(roots):
        raise ValueError("bounded complete query and exact distinct roots required")
    return PreparedSchedule("merlin.graph_schedule_prepared.v1",PROFILE,KIND,DOMAIN,RELATION,PREMISES,
        sys.version,sha(before),sha(after),ir_text(first)!=after,order,canonical(matched),canonical(projection),
        query,roots,_sources())


def public_check(*args, **kwargs):
    raise RuntimeError("scheduling UNSUPPORTED/HELD pending reviewed new source/kernel/qualification")


def compiler_eligible(*args, **kwargs):
    return False
