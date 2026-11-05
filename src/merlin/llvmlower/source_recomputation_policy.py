"""Original-host recomputation choices bound once to actual prepared source.

This isolated engineering policy supplies no authoring access or qualification.
The caller owns the five semantic permissions and current compilation context.
Ordinary object/link/ELF continuation is deliberately outside this leaf.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from weakref import WeakKeyDictionary, ref

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms import recomputation as R
from merlin.transforms.pipeline_state import PipelineContext

from . import source_recomputation as S
from .source_multioutput import _callback, _check_callback

_ISSUED = WeakKeyDictionary()
_MAX_SOURCE_BYTES = 8 << 20
_MAX_SETTINGS_BYTES = 64 << 10


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("recomputation policy requires finite original data") from error


def _values(value):
    if set(vars(value)) != {entry.name for entry in fields(value)}:
        raise ValueError("original recomputation policy contains undeclared fields")
    return _json(asdict(value))


def _functions():
    return (
        _json,
        _values,
        _functions,
        _callback,
        _check_callback,
        sha256_file,
        parse_mlir_text,
        PipelineContext.__post_init__,
        SourceRecomputationPolicy.__post_init__,
        SourceRecomputationPolicy.bind,
        SourceRecomputationPolicy.revalidate,
        R.plan_source_recomputation,
        R._functions,
        *R._functions(),
        S._functions,
        *S._functions(),
    )


@dataclass(frozen=True, eq=False)
class SourceRecomputationPolicy:
    """One original selection and one binding attempt, including refusal/failure.

    ``comparison_for(*, plan, settings)`` may return an existing complete-cost
    decision or None. It cannot retag observations for another source/stamp.
    ``verify_current`` retains the existing selection's exact callback contract;
    entry checks may repeat while binding and during later lowering. Its presence
    is not semantic, runtime, timing or final-image qualification.

    Create a fresh original policy for another build. Revalidation never plans
    again. Missing permissions, unsupported source or exhausted core construction
    ceilings retain the original with None; stale bindings always refuse.
    """

    context: PipelineContext
    permissions: R.RecomputationPermissions
    budgets: R.RecomputationBudgets
    verify_current: Callable = field(kw_only=True)
    comparison_for: Callable | None = field(default=None, kw_only=True)
    _check: Callable = field(init=False, repr=False)
    _bind: Callable = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.context) is not PipelineContext
            or type(self.permissions) is not R.RecomputationPermissions
            or type(self.budgets) is not R.RecomputationBudgets
        ):
            raise ValueError("recomputation policy needs exact original context, permissions and budgets")
        # Context fields must be immutable scalar identities; unknown stays None.
        if any(value is not None and type(value) is not str for value in vars(self.context).values()):
            raise ValueError("recomputation context requires immutable scalar identities")
        self.context.__post_init__()
        self.permissions.__post_init__()
        self.budgets.__post_init__()
        context, permissions, budgets = self.context, self.permissions, self.budgets
        verifier, resolver = self.verify_current, self.comparison_for
        callbacks = (verifier,) if resolver is None else (verifier, resolver)
        frozen_callbacks = tuple(_callback(callback) for callback in callbacks)
        original = tuple(_values(value) for value in (context, permissions, budgets))
        if sum(map(len, original)) > _MAX_SETTINGS_BYTES:
            raise ValueError("recomputation policy exceeds its bounded context document")
        reader, reader_code = _functions, _functions.__code__
        functions = tuple(
            (
                function,
                getattr(function, "__code__", None),
                inspect.unwrap(function),
                getattr(inspect.unwrap(function), "__code__", None),
            )
            for function in reader()
        )
        paths = tuple(
            dict.fromkeys(
                (
                    Path(__file__).resolve(),
                    *R.implementation_paths(),
                    *(Path(inspect.getsourcefile(unwrapped)).resolve() for _, _, unwrapped, _ in functions),
                    *(snapshot[3] for snapshot in frozen_callbacks),
                )
            )
        )
        pins = tuple((path, sha256_file(path)) for path in paths)
        limits = _MAX_SOURCE_BYTES, _MAX_SETTINGS_BYTES
        types = (
            PipelineContext,
            R.RecomputationPermissions,
            R.RecomputationBudgets,
            R.RecomputationPlan,
            S.SourceRecomputationSelection,
            S.SourceRecomputationPreparation,
        )
        state = dict(
            phase="unbound", preparation=None, selection=None, settings=None, settings_state=None, checking=False
        )
        check_function = bind_function = check_code = bind_code = issued = None

        def fresh():
            if (
                (_MAX_SOURCE_BYTES, _MAX_SETTINGS_BYTES) != limits
                or _functions is not reader
                or reader.__code__ is not reader_code
                or any(
                    current is not original
                    for current, original in zip(
                        (
                            PipelineContext,
                            R.RecomputationPermissions,
                            R.RecomputationBudgets,
                            R.RecomputationPlan,
                            S.SourceRecomputationSelection,
                            S.SourceRecomputationPreparation,
                        ),
                        types,
                        strict=True,
                    )
                )
                or any(
                    current is not function
                    or getattr(function, "__code__", None) is not code
                    or inspect.unwrap(current) is not unwrapped
                    or getattr(unwrapped, "__code__", None) is not unwrapped_code
                    for current, (function, code, unwrapped, unwrapped_code) in zip(reader(), functions, strict=True)
                )
            ):
                raise ValueError("original recomputation policy implementation changed")
            if (
                self.context is not context
                or self.permissions is not permissions
                or self.budgets is not budgets
                or self.verify_current is not verifier
                or self.comparison_for is not resolver
                or tuple(_values(value) for value in (context, permissions, budgets)) != original
                or self._check is not check_function
                or self._bind is not bind_function
                or check_function.__code__ is not check_code
                or bind_function.__code__ is not bind_code
                or _ISSUED.get(self) is not issued
            ):
                raise ValueError("original recomputation policy selection changed")
            for callback, snapshot in zip(callbacks, frozen_callbacks, strict=True):
                _check_callback(callback, snapshot)
            for path, digest in pins:
                if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("original recomputation policy source changed")
            if state["settings"] is not None and _json(state["settings"]) != state["settings_state"]:
                raise ValueError("original prepared recomputation settings changed")
            if state["phase"] == "failed":
                raise ValueError("recomputation binding failed; select a fresh original policy")

        def checked(function, *args, **kwargs):
            fresh()
            try:
                return function(*args, **kwargs)
            finally:
                fresh()

        def check():
            fresh()
            if state["phase"] == "binding":
                state["phase"] = "failed"
                raise ValueError("recomputation binding is not reentrant")
            if state["checking"]:
                state["phase"] = "failed"
                raise ValueError("recomputation verification is not reentrant")
            state["checking"] = True
            try:
                if state["preparation"] is not None:
                    checked(S.SourceRecomputationPreparation.revalidate, state["preparation"])
            finally:
                state["checking"] = False

        def bind(source, settings):
            fresh()
            if state["phase"] != "unbound":
                raise ValueError("recomputation policy permits exactly one binding attempt")
            state["phase"] = "binding"
            try:
                if type(source) is not str or len(source.encode()) > _MAX_SOURCE_BYTES:
                    raise ValueError("recomputation binding needs bounded actual prepared source text")
                if type(settings) is not dict or len(_json(settings).encode()) > _MAX_SETTINGS_BYTES:
                    raise ValueError("recomputation binding needs bounded original prepared settings")
                state["settings"], state["settings_state"] = settings, _json(settings)
                module = checked(parse_mlir_text, source)
                proposal = checked(R.plan_source_recomputation, module, context, permissions, budgets)
                if proposal is None:
                    state["phase"] = "original"
                    return None
                # An already issued plan owns all source/candidate SSA, uses,
                # context and IDs. Bracket the resolver with that original guard.
                plan_check = proposal.revalidate
                plan_snapshot = _callback(plan_check)

                def current_plan():
                    fresh()
                    _check_callback(proposal.revalidate, plan_snapshot)
                    checked(plan_check)

                current_plan()
                comparison = None
                if resolver is not None:
                    try:
                        comparison = checked(resolver, plan=proposal, settings=settings)
                    finally:
                        current_plan()
                expected_plan = _json(checked(proposal.record))
                expected_comparison = []

                def verify(**kwargs):
                    current_plan()
                    if (
                        kwargs["source"] != source
                        or _json(kwargs["settings"]) != state["settings_state"]
                        or _json(kwargs["plan"]) != expected_plan
                    ):
                        raise ValueError("original prepared recomputation binding differs from its consumer")
                    actual = _json(kwargs["comparison"])
                    if expected_comparison and actual != expected_comparison[0]:
                        raise ValueError("original prepared recomputation comparison changed")
                    if not expected_comparison:
                        expected_comparison.append(actual)
                    try:
                        return checked(verifier, **kwargs)
                    finally:
                        current_plan()

                selection = S.SourceRecomputationSelection(proposal, verify_current=verify, comparison=comparison)
                preparation = checked(S.SourceRecomputationPreparation, source, selection, settings)
                state["preparation"], state["selection"] = preparation, selection
                state["phase"] = "bound"
                check()
                return selection
            except BaseException:
                state["phase"] = "failed"
                raise

        check_function, bind_function = check, bind
        check_code, bind_code = check.__code__, bind.__code__
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_bind", bind)
        issued = ref(check), check_code, ref(bind), bind_code
        _ISSUED[self] = issued
        self.revalidate()

    def revalidate(self):
        issued = _ISSUED.get(self)
        if (
            issued is None
            or self._check is not issued[0]()
            or self._check.__code__ is not issued[1]
            or self._bind is not issued[2]()
            or self._bind.__code__ is not issued[3]
        ):
            raise ValueError("original recomputation policy guard changed")
        self._check()
        if _ISSUED.get(self) is not issued or self._check is not issued[0]() or self._check.__code__ is not issued[1]:
            raise ValueError("original recomputation policy guard changed during validation")

    def bind(self, source: str, *, settings: dict) -> S.SourceRecomputationSelection | None:
        self.revalidate()
        return self._bind(source, settings)
