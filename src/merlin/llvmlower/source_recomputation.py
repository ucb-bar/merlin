"""Cost-gated source recomputation through the ordinary lowering boundary.

The original live owner joins the selected source plan to current compilation,
runtime and complete-cost evidence. This adapter supplies correspondence and
freshness checks, not timing qualification or permission to repeat computation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary, ref

from merlin.common.digest import sha256_file
from merlin.transforms.certificates import ir_digest, ir_text

from .source_multioutput import _callback, _check_callback

_ISSUED = WeakKeyDictionary()

if TYPE_CHECKING:
    from merlin.perf.offload_profitability import OffloadCostDecision
    from merlin.transforms.recomputation import RecomputationPlan


@dataclass(frozen=True)
class SourceRecomputationSelection:
    """An original issued plan and live evidence owner; no authoring grant.

    ``verify_current`` receives keyword-only stage, source, plan, settings,
    comparison and result. It must join the actual ordinary settings, tools,
    numerical/runtime/ABI selection and cost observations to this plan, return
    None, and raise on missing or stale evidence. At return it must qualify the
    actual products, including any cost transfer from previously measured ones.
    A callback's presence alone establishes none of those premises.
    """

    plan: RecomputationPlan
    verify_current: Callable = field(kw_only=True)
    comparison: OffloadCostDecision | None = None

    def __post_init__(self):
        from merlin.perf.offload_profitability import OffloadCostDecision
        from merlin.transforms.recomputation import RecomputationPlan

        if (
            type(self.plan) is not RecomputationPlan
            or not callable(self.verify_current)
            or self.comparison is not None
            and type(self.comparison) is not OffloadCostDecision
        ):
            raise ValueError("source recomputation requires an issued plan and original live verifier")


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _module_state(module):
    return ir_digest(module), tuple(
        (operation, operation.parent, tuple(operation.operands), tuple(operation.results))
        for operation in module.walk()
    )


def _result_state(result):
    if result is None:
        return None
    from .lower import LowerResult

    if type(result) is not LowerResult:
        raise ValueError("recomputation completion requires the actual ordinary LowerResult")
    return result.workdir, result.ll_path, result.host_so, result.riscv_obj, result.audit_index, _json(result.stats)


def _product_pins(result):
    if result is None:
        return ()
    paths = tuple(
        Path(path).absolute()
        for path in (result.ll_path, result.host_so, result.riscv_obj, result.audit_index)
        if path is not None
    )
    for name in ("lowering_recipe.json", "compilation_recipe.json"):
        path = (Path(result.workdir) / name).absolute()
        if path.exists() or path.is_symlink():
            paths = (*paths, path)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("recomputation completion products must be current original files")
    return tuple((path, sha256_file(path)) for path in paths)


def _functions():
    from merlin.perf import offload_profitability as cost

    from . import lower

    return (
        _functions,
        _json,
        _module_state,
        _result_state,
        _product_pins,
        _callback,
        _check_callback,
        ir_digest,
        ir_text,
        sha256_file,
        SourceRecomputationSelection.__post_init__,
        SourceRecomputationPreparation.__init__,
        SourceRecomputationPreparation.revalidate,
        SourceRecomputationPreparation.retain,
        lower.lower_model,
        lower.lower_model.__wrapped__,
        lower.lower_model_file,
        lower.lower_model_file.__wrapped__,
        lower.preprocess_text,
        lower.lower_to_llvm_ir,
        lower.build_host_shared,
        lower.compile_ll,
        cost.compare_complete_offload_cost,
        cost._assess,
        cost._inputs,
        cost._term,
        cost.OffloadCostDecision.__post_init__,
        cost.OffloadCostDecision.record,
        cost.OffloadCostDecision.revalidate,
    )


class SourceRecomputationPreparation:
    """One current source selection, reopened through compilation and linking."""

    def __init__(self, source, selection, settings):
        from merlin.frontends.linalg_mlir import parse_mlir_text

        if type(source) is not str or type(selection) is not SourceRecomputationSelection:
            raise ValueError("source recomputation requires exact input text and an original typed selection")
        selection.__post_init__()
        plan, comparison, verifier = selection.plan, selection.comparison, selection.verify_current
        reader, reader_code = _functions, _functions.__code__
        functions = tuple((function, function.__code__) for function in reader())
        callbacks = (plan.revalidate, plan.record, verifier)
        if comparison is not None:
            callbacks = (*callbacks, comparison.record, comparison.revalidate, comparison._check)
        frozen_callbacks = tuple(_callback(callback) for callback in callbacks)
        settings_state = _json(settings)
        source_state, candidate_state = _module_state(plan.source), _module_state(plan.candidate)
        stamp = _json(asdict(plan.stamp))
        identities = plan.baseline_selection_sha256, plan.candidate_selection_sha256
        paths = tuple(
            dict.fromkeys(
                (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("lower.py"),
                    *plan.implementation_paths(),
                    *(c[3] for c in frozen_callbacks),
                )
            )
        )
        pins = tuple((path, sha256_file(path)) for path in paths)

        def fresh():
            if (
                _functions is not reader
                or reader.__code__ is not reader_code
                or any(
                    function is not original or original.__code__ is not code
                    for function, (original, code) in zip(reader(), functions, strict=True)
                )
            ):
                raise ValueError("original recomputation selection implementation changed; replan")
            if (
                selection.plan is not plan
                or selection.comparison is not comparison
                or selection.verify_current is not verifier
                or _module_state(plan.source) != source_state
                or _module_state(plan.candidate) != candidate_state
                or _json(asdict(plan.stamp)) != stamp
                or (plan.baseline_selection_sha256, plan.candidate_selection_sha256) != identities
                or _json(settings) != settings_state
            ):
                raise ValueError("original recomputation source, context, settings or cost owner changed; replan")
            current = (plan.revalidate, plan.record, selection.verify_current)
            if comparison is not None:
                current = (*current, comparison.record, comparison.revalidate, comparison._check)
            for callback, before in zip(current, frozen_callbacks, strict=True):
                _check_callback(callback, before)
            for path, digest in pins:
                if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("original recomputation implementation source changed; replan")
            plan.revalidate()

        fresh()
        actual = parse_mlir_text(source)
        actual.verify()
        if ir_digest(actual) != plan.stamp.source_sha256:
            raise ValueError("recomputation plan does not bind the actual ordinary input; replan")
        fresh()
        plan_record = plan.record()
        fresh()
        plan_record_state = _json(plan_record)
        record = None
        if comparison is not None:
            try:
                record = comparison.record()
            finally:
                fresh()
            if (
                _json(record["stamp"]) != stamp
                or record["original_selection_sha256"] != identities[0]
                or record["candidate_selection_sha256"] != identities[1]
            ):
                raise ValueError("complete recomputation costs differ from the current source/context or alternatives")
        record_state = _json(record)
        selected = record is not None and record["choose_candidate"]
        if type(selected) is not bool:
            raise ValueError("recomputation comparison must retain its original Boolean decision")
        selected_source = ir_text(plan.candidate) if selected else source
        self.source, self.selected_source = source, selected_source
        self._retained = ()
        report_state = None

        def check(*, result=None):
            # Every original owner can run callbacks, including the cost and
            # plan readers. Pin the actual return products before any of them.
            result_state = _result_state(result)
            products = _product_pins(result)
            retained = self._retained
            fresh()
            if report_state is not None and _json(self.report) != report_state:
                raise ValueError("original recomputation report changed; replan")
            if comparison is not None:
                try:
                    current_record = comparison.record()
                finally:
                    fresh()
                if _json(current_record) != record_state:
                    raise ValueError("original recomputation complete-cost decision changed; replan")
            if _json(plan_record) != plan_record_state:
                raise ValueError("original recomputation plan record changed; replan")
            if self.source != source or self.selected_source != selected_source:
                raise ValueError("selected recomputation input changed; replan")
            try:
                returned = verifier(
                    stage="entry" if result is None else "return",
                    source=source,
                    plan=plan_record,
                    settings=settings,
                    comparison=record,
                    result=result,
                )
            finally:
                fresh()
            if returned is not None:
                raise ValueError("original recomputation verifier must raise on failure and return None")
            if _json(plan_record) != plan_record_state or _json(record) != record_state:
                raise ValueError("original recomputation evidence changed during verification; replan")
            if result_state != _result_state(result) or retained != self._retained:
                raise ValueError("original recomputation output or result changed during verification")
            for path, digest in (*products, *retained):
                if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("original recomputation retained product changed during verification")

        self._check = check
        _ISSUED[self] = ref(check), check.__code__, ()
        self.report = dict(
            schema="merlin.source_recomputation_selection.v1",
            selected=selected,
            status="MISSING_COST" if record is None else record["status"],
            source_file_sha256=ir_digest(source),
            selected_file_sha256=ir_digest(selected_source),
            plan=plan_record,
            comparison=record,
            implementation_sources=[dict(path=str(path), sha256=digest) for path, digest in pins],
            scope="checked source rewrite and conditional original-owned cost selection; "
            "no machine storage, hardware, timing or authoring qualification",
        )
        report_state = _json(self.report)
        self.revalidate()

    def revalidate(self, *, result=None):
        issued = _ISSUED.get(self)
        if (
            issued is None
            or self._check is not issued[0]()
            or self._check.__code__ is not issued[1]
            or self._retained != issued[2]
        ):
            raise ValueError("original recomputation validation guard or retained identities changed; replan")
        check = self._check
        check(result=result)
        if (
            _ISSUED.get(self) is not issued
            or self._check is not check
            or check.__code__ is not issued[1]
            or self._retained != issued[2]
        ):
            raise ValueError("original recomputation validation guard or retained identities changed; replan")

    def retain(self, directory):
        self.revalidate()
        paths = []
        for name, content in (
            ("source-recomputation-input.mlir", self.source),
            ("source-recomputation-selected.mlir", self.selected_source),
            ("source-recomputation.json", _json(self.report) + "\n"),
        ):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("recomputation retention cannot replace an indirect artifact")
            path.write_text(content)
            paths.append((path.absolute(), sha256_file(path)))
        self._retained = tuple(paths)
        issued = _ISSUED[self]
        _ISSUED[self] = issued[0], issued[1], self._retained
        self.revalidate()
