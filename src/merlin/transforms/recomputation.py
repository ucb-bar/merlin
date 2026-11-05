"""Bounded, explicitly permitted integer source recomputation candidates.

The original source stays unchanged. A returned proposal is legal under its
original host permissions, not a performance decision, admission or target
qualification. Existing layout/storage passes retain their original guards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import prod
from pathlib import Path
from weakref import WeakKeyDictionary, ref

from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.common.digest import sha256_file
from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput

from . import recomputation_check as check
from .certificates import ir_digest, parse_ir
from .pipeline_state import AnalysisStamp, PassSpec, PipelineContext, PipelineSession

_ISSUED = WeakKeyDictionary()


@dataclass(frozen=True)
class RecomputationPermissions:
    """Original host premises; identifiers and records grant no permission."""

    qualification_identity: str
    repeat_evaluation_no_faults: bool = field(kw_only=True)
    repeat_evaluation_unobserved: bool = field(kw_only=True)
    suppressed_materialization_no_faults: bool = field(kw_only=True)
    suppressed_materialization_unobserved: bool = field(kw_only=True)
    inputs_stable_during_region: bool = field(kw_only=True)

    def __post_init__(self):
        if (
            type(self.qualification_identity) is not str
            or not self.qualification_identity.strip()
            or any(type(value) is not bool for key, value in vars(self).items() if key != "qualification_identity")
        ):
            raise ValueError("recomputation requires original identity and five explicit Boolean permissions")

    def require(self):
        self.__post_init__()
        if not all(value for key, value in vars(self).items() if key != "qualification_identity"):
            raise UnsupportedElementwise("original repeat/removal permissions are unavailable")


@dataclass(frozen=True)
class RecomputationBudgets:
    """Construction counts include the full cloned module, not net IR growth.

    Logical evaluations count all scalar operations, including constants.
    These finite ceilings never establish repeatability or hardware support.
    """

    max_candidates: int
    max_source_elements: int
    max_repeats: int
    max_producer_scalar_ops: int
    max_added_scalar_evaluations: int
    max_constructed_operations: int
    max_scalar_ir_bytes: int

    def __post_init__(self):
        if (
            any(type(value) is not int or not 0 < value <= 1 << 30 for value in vars(self).values())
            or self.max_repeats > 2
            or self.max_candidates > 4096
            or self.max_constructed_operations > 1 << 20
            or self.max_producer_scalar_ops > 4096
            or self.max_scalar_ir_bytes > 4 << 20
        ):
            raise ValueError("recomputation requires positive bounded construction/evaluation ceilings; repeats <= 2")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _state(module):
    return ir_digest(module), tuple(
        (
            op,
            op.parent,
            tuple(op.operands),
            tuple(op.results),
            tuple(tuple((use.operation, use.index) for use in value.uses) for value in (*op.operands, *op.results)),
        )
        for op in module.walk()
    )


def _rewrite(candidate, ordinals, source):
    operations = tuple(tuple(candidate.walk())[i] for i in ordinals)
    _, _, empty = check._site(candidate, operations)
    producer = operations[0]
    pbody = producer.body.block
    made = 0
    for i, consumer in enumerate(operations[1:]):
        inputs = check._inputs(producer, consumer)
        block = Block(arg_types=[*(v.type.element_type for v in inputs), consumer.results[0].type.element_type])
        pvalues = dict(zip(tuple(pbody.args)[:-1], (block.args[inputs.index(v)] for v in producer.inputs), strict=True))
        for op in tuple(pbody.ops)[:-1]:
            block.add_op(op.clone(value_mapper=pvalues))
            made += 1
        value = pvalues[pbody.last_op.operands[0]]
        ebody = consumer.body.block
        evalues = dict(
            zip(
                tuple(ebody.args)[:-1],
                (value if v is producer.results[0] else block.args[inputs.index(v)] for v in consumer.inputs),
                strict=True,
            )
        )
        for op in tuple(ebody.ops)[:-1]:
            block.add_op(op.clone(value_mapper=evalues))
            made += 1
        block.add_op(YieldOp(evalues[ebody.last_op.operands[0]]))
        rank = len(source.shape)
        replacement = GenericOp(
            inputs=inputs,
            outputs=consumer.outputs,
            body=Region(block),
            indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(rank))] * (len(inputs) + 1),
            iterator_types=consumer.iterator_types,
            result_types=[v.type for v in consumer.results],
        )
        replacement.attributes.update(consumer.attributes)
        consumer.parent.insert_op_before(replacement, consumer)
        consumer.results[0].replace_all_uses_with(replacement.results[0])
        consumer.parent.erase_op(consumer)
        made += 2  # the Generic and its yield, not only arithmetic
    producer.parent.erase_op(producer)
    empty.parent.erase_op(empty)
    return made


def _functions():
    from merlin.compile.scheduling import fusion_semantics, multioutput_fusion, pointwise_source
    from merlin.llvmlower import layout_conformance

    from . import certificates, dense_ir, pure_scalar

    return (
        _digest,
        _state,
        _rewrite,
        _functions,
        implementation_paths,
        check.verify_source_recomputation,
        check._blocks,
        check._site,
        check._inputs,
        check._composed_graph,
        check._skeleton,
        check._observable,
        check._work,
        read_source_multioutput,
        pointwise_source.read_pointwise_source,
        fusion_semantics.scalar_graph,
        fusion_semantics.scalar_graph.__wrapped__,
        multioutput_fusion.read_pointwise_source,
        pointwise_source.scalar_graph,
        pointwise_source.checked_static_linalg_shell,
        dense_ir.checked_static_linalg_shell,
        pure_scalar.eligible_scalar_operation,
        layout_conformance._shape,
        layout_conformance._fields,
        layout_conformance._reshape,
        layout_conformance.unit_source,
        layout_conformance.pure_scalar_body,
        layout_conformance.layout_expression,
        layout_conformance.pointwise_kind,
        layout_conformance.scalar_bodies,
        check._metadata,
        check._attribute,
        check.layout_expression,
        check.read_pointwise_source,
        check.read_source_multioutput,
        certificates.ir_digest,
        certificates.ir_text,
        RecomputationPermissions.require,
        RecomputationPermissions.__post_init__,
        RecomputationBudgets.__post_init__,
        RecomputationPlan.revalidate,
        RecomputationPlan.record,
        RecomputationPlan.implementation_paths,
    )


def implementation_paths():
    """Direct semantic owners; this is not arbitrary transitive qualification."""
    import inspect

    from merlin.llvmlower import quant_round_check
    from merlin.transforms import dense_ir, pure_scalar

    return tuple(
        dict.fromkeys(
            Path(path).resolve()
            for path in (
                __file__,
                check.__file__,
                dense_ir.__file__,
                pure_scalar.__file__,
                quant_round_check.__file__,
                *(inspect.getsourcefile(inspect.unwrap(f)) for f in _functions()),
                inspect.getsourcefile(PipelineSession),
                inspect.getsourcefile(parse_ir),
            )
            if path
        )
    )


@dataclass(frozen=True, eq=False)
class RecomputationPlan:
    source: builtin.ModuleOp
    candidate: builtin.ModuleOp
    stamp: AnalysisStamp
    baseline_selection_sha256: str
    candidate_selection_sha256: str
    _guard: object
    _record: dict

    def revalidate(self):
        issued = _ISSUED.get(self)
        if issued is None:
            raise ValueError("recomputation plan is not its original issued proposal")
        guard = issued[0]()
        if guard is None or self._guard is not guard or guard.__code__ is not issued[1]:
            raise ValueError("recomputation plan is not its original issued proposal or guard code")
        guard()

    def record(self):
        self.revalidate()
        return json.loads(json.dumps(self._record))

    def implementation_paths(self):
        self.revalidate()
        return implementation_paths()


def plan_source_recomputation(module, context, permissions, budgets):
    """Return the first bounded legal candidate, or retain the original with None.

    The caller supplies original permissions before discovery. No observed
    outputs, baseline compile, cost estimate or captured identity selects it.
    """
    if (
        type(module) is not builtin.ModuleOp
        or type(context) is not PipelineContext
        or type(permissions) is not RecomputationPermissions
        or type(budgets) is not RecomputationBudgets
    ):
        raise ValueError("recomputation requires typed original module, context, permissions and budgets")
    permissions.__post_init__()
    budgets.__post_init__()
    context.__post_init__()
    try:
        permissions.require()
    except UnsupportedElementwise:
        return None
    # Bound the full clone before construction; no tensor elements are visited.
    walk = []
    for op in module.walk():
        walk.append(op)
        if len(walk) > budgets.max_constructed_operations:
            return None
    module.verify()
    original = _state(module)
    selected = (asdict(context), asdict(permissions), asdict(budgets))
    functions = tuple((f, getattr(f, "__code__", None)) for f in _functions())
    reader, reader_code = _functions, _functions.__code__
    paths = implementation_paths()
    pins = tuple((str(path), sha256_file(path)) for path in paths)

    def fresh():
        if (
            _functions is not reader
            or reader.__code__ is not reader_code
            or tuple(f for f, _ in functions) != reader()
            or any(getattr(f, "__code__", None) is not code for f, code in functions)
        ):
            raise ValueError("recomputation original semantic reader or checker code changed")
        if selected != (asdict(context), asdict(permissions), asdict(budgets)) or _state(module) != original:
            raise ValueError("recomputation original source, SSA, context or permissions changed")
        if any(
            Path(path).is_symlink() or not Path(path).is_file() or sha256_file(path) != digest for path, digest in pins
        ):
            raise ValueError("recomputation original implementation source changed")

    # Unknown whole-module coordinate/control semantics retain the baseline.
    # This is the ORIGINAL analysis only; candidate checker failures are never
    # swallowed as unsupported input or accepted as successful proposals.
    try:
        check.layout_expression(module, include_work=True)
    except ValueError:
        fresh()
        return None
    fresh()
    count = 0
    for producer in walk:
        if type(producer) is not GenericOp or producer.parent is None:
            continue
        body = tuple(producer.parent.ops)
        index = body.index(producer)
        operations = body[index : index + 3]
        if len(operations) != 3 or any(type(op) is not GenericOp for op in operations):
            continue
        count += 1
        if count > budgets.max_candidates:
            return None
        try:
            source = read_source_multioutput(
                module,
                operations,
                max_source_elements=budgets.max_source_elements,
                max_scalar_operations=budgets.max_producer_scalar_ops,
                max_scalar_ir_bytes=budgets.max_scalar_ir_bytes,
            )
            check._site(module, operations)
            added_ops = sum(
                len(source.producer.graph.operations) + len(e.graph.operations) + 2 for e in source.epilogues
            )
            if (
                budgets.max_repeats < 2
                or len(source.producer.graph.operations) == 0
                or len(walk) + added_ops > budgets.max_constructed_operations
            ):
                continue
            # Count every actual scalar-body clone, including unused scalar
            # operations. Dominating constant captures remain outside loops.
            old_count = sum(len(tuple(op.body.block.ops)) - 1 for op in operations)
            added = (added_ops - 4 - old_count) * prod(source.shape)
            if added > budgets.max_added_scalar_evaluations:
                continue
        except UnsupportedElementwise:
            continue
        fresh()
        ordinals = tuple(walk.index(op) for op in operations)
        candidate = module.clone()
        session = PipelineSession(candidate, context=context, ir_level="linalg", max_repairs=0)
        session.run(
            PassSpec("recomputation-source", 0, produces=("source",), mutates_ir=False, ir_level="linalg"),
            lambda: {"source": ir_digest(module)},
        )
        made = []

        def rewrite():
            made.append(_rewrite(candidate, ordinals, source))
            fresh()
            return {"constructed": made[0]}

        session.run(
            PassSpec(
                "source-recomputation",
                1,
                requires=("source",),
                produces=("constructed",),
                invalidates=("*",),
                ir_level="linalg",
            ),
            rewrite,
        )

        def verify():
            fresh()
            proof = check.verify_source_recomputation(module, candidate, operations, permissions, budgets)
            fresh()
            return {"conformance": proof}

        session.run(
            PassSpec(
                "recomputation-conformance",
                2,
                requires=("constructed",),
                produces=("conformance",),
                mutates_ir=False,
                ir_level="linalg",
            ),
            verify,
        )
        proof = session.analysis("conformance")
        stamp = AnalysisStamp(original[0], context, "linalg")
        common = {
            "stamp": asdict(stamp),
            "permissions": selected[1],
            "budgets": selected[2],
            "implementation_sources": pins,
        }
        # Paths remain evidence for reopening, not semantic route identity.
        # The closed ordered owner roster binds the same bytes after relocation.
        identity = {key: value for key, value in common.items() if key != "implementation_sources"}
        identity["implementation_source_sha256"] = tuple(digest for _, digest in pins)
        baseline = _digest({**identity, "source": original[0], "route": "original"})
        candidate_id = _digest({**identity, "source": ir_digest(candidate), "route": "source_recomputation"})
        record = {
            "schema": "merlin.source_recomputation_plan.v1",
            "status": "LEGAL_CANDIDATE",
            "cost": "UNKNOWN",
            **common,
            "baseline_selection_sha256": baseline,
            "candidate_selection_sha256": candidate_id,
            "candidate_sha256": ir_digest(candidate),
            "source_operation_ordinals": ordinals,
            "constructed_operations": len(walk) + made[0],
            "net_operation_delta": len(tuple(candidate.walk())) - len(walk),
            "conformance": proof,
            "passes": [asdict(item) for item in session.records],
        }
        frozen_record, current = _digest(record), _state(candidate)
        stamp_record = asdict(stamp)
        holder = []

        def guard():
            result = holder[0]()
            if (
                result is None
                or result.source is not module
                or result.candidate is not candidate
                or asdict(result.stamp) != stamp_record
                or result.baseline_selection_sha256 != baseline
                or result.candidate_selection_sha256 != candidate_id
                or _digest(result._record) != frozen_record
            ):
                raise ValueError("recomputation issued source, candidate or record changed")
            fresh()
            if _state(candidate) != current:
                raise ValueError("recomputation candidate or SSA changed")

        result = RecomputationPlan(module, candidate, stamp, baseline, candidate_id, guard, record)
        holder.append(ref(result))
        _ISSUED[result] = (ref(guard), guard.__code__)
        result.revalidate()
        return result
    return None
