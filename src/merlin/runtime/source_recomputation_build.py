"""Original-host recomputation custody through one saved-model compilation.

Observed IR/object/link correspondence is not semantic proof of arbitrary LLVM,
a timing transfer, target admission or an authoring/resource grant. The separate
original final-image verifier owns those qualifications.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from weakref import WeakKeyDictionary, ref

from merlin.common import invocation_record as I
from merlin.common.digest import sha256_file
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower import compilation_recipe as C
from merlin.llvmlower import lower as L
from merlin.llvmlower import source_recomputation as S
from merlin.llvmlower import source_recomputation_policy as P
from merlin.llvmlower.source_multioutput import _callback, _check_callback

_ISSUED = WeakKeyDictionary()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _pin(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(member.is_symlink() for member in (path, *path.parents)) or not path.is_file():
        raise ValueError("recomputation custody needs a direct canonical regular file")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _reopen(rows):
    for row in rows:
        if _pin(row["path"]) != row:
            raise ValueError("original recomputation artifact changed")


def _file_rows(value):
    """Only explicit observed path/hash records are files; no name guessing."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from _file_rows(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _file_rows(child)


def _metadata(function):
    # Python defaults/signature metadata can change without replacing __code__.
    return (
        repr(getattr(function, "__defaults__", None)),
        repr(getattr(function, "__kwdefaults__", None)),
        repr(getattr(function, "__signature__", None)),
    )


def _functions():
    from merlin.compile import baremetal_model
    from merlin.runtime.backends import spike_model

    return (
        _json,
        _copy,
        _pin,
        _reopen,
        _functions,
        _file_rows,
        _metadata,
        freeze_source_recomputation,
        SourceRecomputationBuild.__init__,
        SourceRecomputationBuild.revalidate,
        SourceRecomputationBuild.capture,
        SourceRecomputationBuild.bind_sources,
        SourceRecomputationBuild.prepare,
        SourceRecomputationBuild.options,
        SourceRecomputationBuild.consumed,
        SourceRecomputationBuild.observer,
        SourceRecomputationBuild.identity,
        SourceRecomputationBuild.finish,
        SourceRecomputationBuild.executed,
        SourceRecomputationBuild.retain_execution,
        SourceRecomputationBuild.record,
        _callback,
        _check_callback,
        strict_tree_sha256,
        sha256_file,
        baremetal_model.compile_saved_model,
        spike_model.build,
        spike_model._run,
        spike_model._proc.run_checked,
        C.CompilationRecipe.run,
        C.CompilationRecipe._run,
        C.CompilationRecipe.bind_preparation,
        C.CompilationRecipe.completed,
        C.CompilationRecipe.completed_product,
        C.verify_completed_recipe,
        C._checked_command,
        C._checked_identity,
        C._ordinary_path,
        C._argument_path,
        C._unique_object,
        C._recheck_weight_preparation,
        C._identity,
        C.strict_json_equal,
        C.verify_link_suppliers,
        S.lowering_settings,
        L.lower_model_file,
        L.lower_model_file.__wrapped__,
        L.lower_model,
        L.lower_model.__wrapped__,
        spike_model.lower_model_file,
        I.observe_call,
        I.verify,
        I.verify_timing_document,
        I.Invocation.__init__,
        I.Invocation.complete,
        I.Invocation._write,
        I._pin,
        I._pins,
        I.CallInvocation.__init__,
        I.CallInvocation.returned,
        I.CallInvocation.complete_streams,
        P._functions,
        *P._functions(),
    )


def freeze_source_recomputation(value):
    """Retain an original checker before callbacks; None performs no library IO."""
    if value is None:
        return None
    if type(value) is not SourceRecomputationBuild:
        raise ValueError("ordinary recomputation requires its original typed build custodian")
    issued = _ISSUED.get(value)
    if issued is None:
        raise ValueError("ordinary recomputation requires an issued build custodian")
    check, code = issued[0](), issued[1]
    static, static_code = issued[2](), issued[3]

    def require(*, callbacks=True):
        if _ISSUED.get(value) is not issued or value._check is not check or check.__code__ is not code:
            raise ValueError("original recomputation build checker changed")
        if value._static is not static or static.__code__ is not static_code:
            raise ValueError("original recomputation static checker changed")
        static()
        if callbacks:
            check()
        static()
        if _ISSUED.get(value) is not issued or value._check is not check or check.__code__ is not code:
            raise ValueError("original recomputation build checker changed during callback")

    require()
    return require


class SourceRecomputationBuild:
    """One live original policy plus a separate final-image verifier.

    ``verify_final(*, stage, context, selection, products, execution)`` returns
    None or raises. Stages are linked and executed. The original owner must join
    actual ABI/runtime/tools/options and final products to the observed complete
    costs, including any transfer to a different image. A callback alone is no
    qualification. Missing cost retains the exact original prepared source.
    """

    def __init__(self, policy, *, verify_final):
        if type(policy) is not P.SourceRecomputationPolicy:
            raise ValueError("ordinary recomputation needs an original bind-once policy")
        self.policy, self.verify_final = policy, verify_final
        callback = _callback(verify_final)
        owner_callbacks = tuple(
            (function, _callback(function), _metadata(function))
            for function in (verify_final, policy.verify_current, policy.comparison_for)
            if function is not None
        )
        reader, reader_code = _functions, _functions.__code__
        functions = tuple(
            (f, getattr(f, "__code__", None), inspect.unwrap(f), getattr(inspect.unwrap(f), "__code__", None))
            for f in reader()
        )
        metadata = tuple((_metadata(f), _metadata(raw)) for f, _, raw, _ in functions)
        signature = inspect.signature(L.lower_model_file)
        defaults = _json((L.lower_model_file.__wrapped__.__defaults__, L.lower_model_file.__wrapped__.__kwdefaults__))
        paths = tuple(
            dict.fromkeys(
                (
                    Path(__file__),
                    *(Path(inspect.getsourcefile(unwrapped)).resolve() for _, _, unwrapped, _ in functions),
                    callback[3],
                    *_callback(policy.verify_current)[3:4],
                    *((_callback(policy.comparison_for)[3],) if policy.comparison_for is not None else ()),
                )
            )
        )
        sources = tuple(_pin(path) for path in paths)
        state = dict(
            capture=None,
            context={},
            compilation=None,
            prepared=None,
            selection=None,
            report=None,
            products=[],
            observations=[],
            finished=None,
            verifying=False,
            finalizing=False,
            failed=False,
            executed=False,
        )
        operations = {}
        issued = None

        def static():
            if _functions is not reader or reader.__code__ is not reader_code:
                raise ValueError("original recomputation build function reader changed")
            if any(
                current is not f
                or getattr(f, "__code__", None) is not code
                or inspect.unwrap(current) is not unwrapped
                or getattr(unwrapped, "__code__", None) is not raw_code
                for current, (f, code, unwrapped, raw_code) in zip(reader(), functions, strict=True)
            ):
                raise ValueError("original recomputation build implementation changed")
            if tuple((_metadata(f), _metadata(raw)) for f, _, raw, _ in functions) != metadata:
                raise ValueError("original recomputation callable default metadata changed")
            if (
                inspect.signature(L.lower_model_file) != signature
                or _json((L.lower_model_file.__wrapped__.__defaults__, L.lower_model_file.__wrapped__.__kwdefaults__))
                != defaults
            ):
                raise ValueError("original recomputation lowering defaults changed")
            if self.policy is not policy or self.verify_final is not verify_final or _ISSUED.get(self) is not issued:
                raise ValueError("original recomputation build owner changed")
            if any(function.__code__ is not code for function, code in zip(internal, internal_codes, strict=True)):
                raise ValueError("original recomputation private implementation changed")
            for name, (function, code) in operations.items():
                if getattr(self, name) is not function or function.__code__ is not code:
                    raise ValueError("original recomputation build operation changed")
            _check_callback(self.verify_final, callback)
            for function, snapshot, metadata_snapshot in owner_callbacks:
                _check_callback(function, snapshot)
                if _metadata(function) != metadata_snapshot:
                    raise ValueError("original recomputation owner callback defaults changed")
            _reopen(sources)
            _reopen(state["products"])
            if state["compilation"] is not None:
                compilation, path, record, methods = state["recipe_binding"]
                if compilation.path != path or compilation.record is not record:
                    raise ValueError("original recomputation recipe identity changed")
                for name, snapshot in methods:
                    _check_callback(getattr(compilation, name), snapshot)
            if "result" in state and S._result_state(state["result"]) != state["result_state"]:
                raise ValueError("original recomputation LowerResult fields or stats changed")
            if state["capture"] is not None:
                path, snapshot = state["capture"]
                if strict_tree_sha256(path) != snapshot:
                    raise ValueError("original recomputation materialized inputs changed")

        def current():
            static()
            if state["failed"]:
                raise ValueError("original recomputation build previously failed")
            if state["verifying"]:
                state["failed"] = True
                raise ValueError("original recomputation build validation is not reentrant")
            state["verifying"] = True
            try:
                policy.revalidate()
            finally:
                state["verifying"] = False
                static()

        def retain(*paths):
            for path in paths:
                row = _pin(path)
                prior = next((r for r in state["products"] if r["path"] == row["path"]), None)
                if prior is not None and prior != row:
                    raise ValueError("recomputation cannot refresh an earlier artifact")
                if prior is None:
                    state["products"].append(row)

        def capture(path, context):
            current()
            path = Path(path).absolute()
            if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("recomputation capture must be direct and canonical")
            snapshot = strict_tree_sha256(path)
            value = _copy(context)
            if len(_json(value)) > 1 << 20:
                raise ValueError("original recomputation context exceeds its bound")
            prior = state["capture"]
            if prior is not None and prior != (path, snapshot):
                raise ValueError("original recomputation capture cannot be rebound")
            if context is not None and "capture" in state["context"] and state["context"]["capture"] != value:
                raise ValueError("original recomputation capture context changed")
            state["capture"] = path, snapshot
            if context is not None:
                state["context"]["capture"] = value
            current()

        def bind_sources(compilation):
            if type(compilation) is not C.CompilationRecipe or state["compilation"] is not None:
                raise ValueError("recomputation needs one original compilation recipe")
            methods = tuple(
                (name, _callback(getattr(compilation, name)))
                for name in ("run", "_run", "bind_preparation", "completed", "completed_product")
            )
            if any(
                snapshot[0] is not compilation or snapshot[1] is not getattr(C.CompilationRecipe, name)
                for name, snapshot in methods
            ):
                raise ValueError("recomputation needs original recipe methods")
            state["recipe_binding"] = compilation, compilation.path, compilation.record, methods
            state["compilation"] = compilation
            current()
            for index, row in enumerate(sources):
                compilation.bind_preparation(f"source_recomputation_owner_{index}", Path(row["path"]))
            current()

        def prepare(path, work, options, context):
            current()
            if state["prepared"] is not None or state["compilation"] is None:
                raise ValueError("ordinary recomputation prepared input can bind only once")
            row = _pin(path)
            arguments = signature.bind(path, work, **options)
            arguments.apply_defaults()
            settings = S.lowering_settings(arguments.arguments)
            state["context"]["build"] = _copy(context)
            if len(_json(state["context"])) > 1 << 20:
                raise ValueError("original recomputation build context exceeds its bound")
            state["prepared"] = row
            retain(path)
            selection = policy.bind(Path(path).read_text(), settings=settings)
            current()
            state["selection"] = selection
            if selection is not None:
                evaluation = S.SourceRecomputationPreparation(Path(path).read_text(), selection, settings)
                current()
                state["expected_report"] = _copy(evaluation.report)
                state["expected_source"] = evaluation.selected_source
            state["context"]["lowering"] = _copy(settings)
            state["context"]["exact_ir_audit"] = True

        def options():
            current()
            if state["prepared"] is None:
                raise ValueError("ordinary recomputation has no final prepared binding")
            return {
                "ir_audit": True,
                **({"source_recomputation": state["selection"]} if state["selection"] is not None else {}),
            }

        def consumed(result, compilation):
            if (
                type(result) is not L.LowerResult
                or compilation is not state["compilation"]
                or state["report"] is not None
            ):
                raise ValueError("recomputation needs its actual original LowerResult once")
            state["result"], state["result_state"] = result, S._result_state(result)
            audit_path = Path(result.audit_index)
            retain(result.ll_path, audit_path)
            for name in (
                "source-recomputation-input.mlir",
                "source-recomputation-selected.mlir",
                "source-recomputation.json",
            ):
                if state["selection"] is not None:
                    retain(Path(result.workdir) / name)
            if _pin(audit_path)["bytes"] > 16 << 20:
                raise ValueError("ordinary recomputation audit index exceeds its bound")
            audit = json.loads(audit_path.read_bytes())
            if audit.get("outcome") != "completed" or any(
                sum(row["name"] == name for row in audit["stages"]) != 1 for name in ("input", "llvm-final")
            ):
                raise ValueError("ordinary recomputation audit is incomplete")
            if len(audit["stages"]) > 10000 or len({row["name"] for row in audit["stages"]}) != len(audit["stages"]):
                raise ValueError("ordinary recomputation audit has duplicate or unbounded stages")
            for row in audit["stages"]:
                name = row.get("file")
                if not isinstance(name, str) or Path(name).name != name or row.get("representation") != "exact-ir":
                    raise ValueError("ordinary recomputation requires exact direct audit stages")
                observed = _pin(audit_path.parent / name)
                if any(observed[key] != row.get(key) for key in ("sha256", "bytes")):
                    raise ValueError("ordinary recomputation audit stage bytes changed")
            retain(*(audit_path.parent / row["file"] for row in audit["stages"]))
            current()
            stages = {row["name"]: row for row in audit["stages"]}
            expected = Path(state["prepared"]["path"]).read_text()
            report = result.stats.get("source_recomputation")
            if state["selection"] is not None:
                if type(report) is not dict or report != state["expected_report"]:
                    raise ValueError("ordinary lowerer ignored its recomputation selection")
                expected = state["expected_source"]
                for name in (
                    "source-recomputation-input.mlir",
                    "source-recomputation-selected.mlir",
                    "source-recomputation.json",
                ):
                    retain(Path(result.workdir) / name)
                    compilation.bind_preparation(name, Path(result.workdir) / name)
                if json.loads(Path(result.workdir, "source-recomputation.json").read_bytes()) != report:
                    raise ValueError("ordinary recomputation report changed")
                if Path(result.workdir, "source-recomputation-selected.mlir").read_text() != expected:
                    raise ValueError("ordinary lowerer consumed a foreign recomputation source")
            elif report is not None:
                raise ValueError("unavailable recomputation returned a foreign plan")
            input_stage = stages["input"]
            if (audit_path.parent / input_stage["file"]).read_text() != Path(state["prepared"]["path"]).read_text():
                raise ValueError("ordinary lowerer ignored its actual prepared source")
            if bool(report is not None and report["selected"]) != ("source-recomputed" in stages):
                raise ValueError("ordinary recomputation selected audit stage is missing or unexpected")
            selected_stage = stages.get("source-recomputed", input_stage)
            if (audit_path.parent / selected_stage["file"]).read_text() != expected:
                raise ValueError("ordinary lowerer ignored the selected recomputation IR")
            if _pin(result.ll_path)["sha256"] != stages["llvm-final"]["sha256"]:
                raise ValueError("ordinary recomputation LLVM differs from its actual audit")
            state["report"] = (
                _copy(report)
                if report is not None
                else {
                    "status": "NO_PLAN",
                    "selected": False,
                    "source_file_sha256": state["prepared"]["sha256"],
                    "selected_file_sha256": state["prepared"]["sha256"],
                }
            )
            state["llvm"] = _pin(result.ll_path)
            retain(
                result.ll_path,
                audit_path,
                *(audit_path.parent / row["file"] for row in audit["stages"] if row.get("file")),
            )
            compilation.bind_preparation("source_recomputation_prepared", Path(state["prepared"]["path"]))
            build_report = Path(result.workdir) / "source-recomputation-build.json"
            build_report.write_text(_json(state["report"]) + "\n")
            retain(build_report)
            compilation.bind_preparation("source_recomputation_build", build_report)
            current()

        def observer(compilation, stage):
            current()
            if (
                compilation is not state["compilation"]
                or stage not in ("model_object", "final_link")
                or any(row[0] == stage for row in state["observations"])
            ):
                raise ValueError("recomputation needs one original object and link observation")
            original = compilation.run

            def run(argv, *, runner, inputs, output, **kwargs):
                runner_binding, runner_metadata = _callback(runner), _metadata(runner)
                retain(runner_binding[3])
                current()
                _check_callback(runner, runner_binding)
                inputs = tuple(inputs)
                if stage == "model_object" and tuple(_pin(path) for path in inputs) != (state["llvm"],):
                    raise ValueError("model object did not consume the selected recomputation LLVM")
                before = len(compilation.record["commands"])
                with I.observe_call(
                    compilation.path.parent,
                    stage="source_recomputation_" + stage,
                    function=original,
                    arguments={"argv": list(map(str, argv)), "command_index": before},
                    inputs=inputs,
                    outputs=(output,),
                    dependencies=(Path(argv[0]).resolve(),),
                ) as observation:
                    result = original(argv, runner=runner, inputs=inputs, output=output, **kwargs)
                    command = _copy(compilation.record["commands"][before])
                    observation.returned(stdout=_json(command).encode())
                _check_callback(runner, runner_binding)
                if _metadata(runner) != runner_metadata:
                    raise ValueError("original compilation runner defaults changed")
                observed = I.verify(observation.path)
                for row in _file_rows(observed):
                    actual = _pin(row["path"])
                    if actual["sha256"] != row["sha256"]:
                        raise ValueError("recomputation observation pin changed")
                    retain(row["path"])
                state["observations"].append((stage, before, _pin(observation.path), command))
                retain(
                    output,
                    observation.path,
                    observation.path.parent / "stdout.bin",
                    observation.path.parent / "stderr.bin",
                )
                current()
                return result

            return run

        def identity():
            current()
            return _json(
                {
                    "context": state["context"],
                    "prepared": state["prepared"],
                    "selection": state["report"],
                    "sources": sources,
                }
            ).encode()

        def final(stage, execution=None):
            # Capture exact original products before any repeated owner/cost callback.
            products = _copy(state["products"])
            context, selection = _copy(state["context"]), _copy(state["report"])
            original = _json((context, selection, products, execution))
            if state["finalizing"]:
                state["failed"] = True
                raise ValueError("original final-image verification is not reentrant")
            state["finalizing"] = True
            try:
                current()
                returned = verify_final(
                    stage=stage, context=context, selection=selection, products=products, execution=execution
                )
                current()
                if returned is not None or _json((context, selection, products, execution)) != original:
                    raise ValueError("original recomputation final-image verification changed or did not return None")
            except BaseException:
                state["failed"] = True
                raise
            finally:
                state["finalizing"] = False
                _reopen(state["products"])

        def finish(compilation, elf):
            if compilation is not state["compilation"] or state["finished"] is not None:
                raise ValueError("recomputation final image must complete once")
            retain(compilation.path, elf)
            current()
            recipe = C.verify_completed_recipe(
                compilation.path, executable=elf, expected_recipe_sha256=_pin(compilation.path)["sha256"]
            )
            for row in _file_rows(recipe):
                actual = _pin(row["path"])
                if any(actual[key] != row[key] for key in ("sha256", "bytes")):
                    raise ValueError("original recipe member changed during completion")
                retain(row["path"])
            observed = {row[0]: row for row in state["observations"]}
            if tuple(row[0] for row in state["observations"]) != ("model_object", "final_link"):
                raise ValueError("recomputation has no actual model-object/link observations")
            for stage, index, invocation, command in state["observations"]:
                actual = I.verify(Path(invocation["path"]))
                if (
                    recipe["commands"][index] != command
                    or json.loads(Path(actual["stdout"]["path"]).read_bytes()) != command
                ):
                    raise ValueError("recomputation command differs from its actual observed producer")
            model = observed["model_object"][3]
            link = observed["final_link"][3]
            if (
                model["output"] not in link["inputs"]
                or link != recipe["commands"][-1]
                or link["output"] != recipe["executable"]
            ):
                raise ValueError("recomputation final ELF did not consume the original model object")
            state["finished"] = _pin(elf)
            final("linked")

        def retain_execution(paths):
            # Snapshot numeric readback products before any later original callback.
            rows = tuple(_pin(path) for path in paths)
            current()
            _reopen(rows)
            for row in rows:
                retain(row["path"])
            current()

        def executed(execution):
            if state["finished"] is None or state["executed"]:
                raise ValueError("recomputation execution needs its original final image once")
            state["executed"] = True
            execution = _copy(execution)
            for row in _file_rows(execution):
                if _pin(row["path"])["sha256"] != row["sha256"]:
                    raise ValueError("original recomputation execution artifact changed")
                retain(row["path"])
            final("executed", execution)

        def record():
            current()
            if state["finished"] is None:
                raise ValueError("recomputation has no completed original image")
            return _copy(
                {
                    "schema": "merlin.source_recomputation_build.v1",
                    "context": state["context"],
                    "selection": state["report"],
                    "products": state["products"],
                    "executable": state["finished"],
                    "scope": "original-host source selection and artifact custody; qualification required separately",
                }
            )

        closures = dict(
            _check=current,
            _static=static,
            _capture=capture,
            _bind_sources=bind_sources,
            _prepare=prepare,
            _options=options,
            _consumed=consumed,
            _observer=observer,
            _identity=identity,
            _finish=finish,
            _executed=executed,
            _retain_execution=retain_execution,
            _record=record,
        )
        internal = (static, current, retain, final)
        internal_codes = tuple(function.__code__ for function in internal)
        for name, function in closures.items():
            setattr(self, name, function)
            operations[name] = function, function.__code__
        issued = ref(current), current.__code__, ref(static), static.__code__
        _ISSUED[self] = issued
        current()

    def revalidate(self):
        freeze_source_recomputation(self)()

    def capture(self, path, context=None):
        self._capture(path, context)

    def bind_sources(self, compilation):
        self._bind_sources(compilation)

    def prepare(self, path, work, options, context):
        self._prepare(path, work, options, context)

    def options(self):
        return self._options()

    def consumed(self, result, compilation):
        self._consumed(result, compilation)

    def observer(self, compilation, stage):
        return self._observer(compilation, stage)

    def identity(self):
        return self._identity()

    def finish(self, compilation, elf):
        self._finish(compilation, elf)

    def retain_execution(self, paths):
        self._retain_execution(paths)

    def executed(self, execution):
        self._executed(execution)

    def record(self):
        return self._record()
