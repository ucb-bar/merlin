"""Source-pinned materialization and source checks for sealed CPU captures.

The issuer rechecks this module's exact bytes before delegating. Original issuer
callbacks are resolved at call time so their public patch points remain intact.
No additional issuer, schema or replay policy is admitted here.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


def _capture_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Check the selected source API without importing its heavyweight runtime.

    This is only a compatibility gate. The fresh sandbox execution and replay,
    not source signatures, establish whether a selected implementation works.
    """
    from .sealed_m2m import (
        _source_api_missing,
    )

    return _source_api_missing(
        m2m_root,
        {
            "m2m/api.py": {
                "convert": {"backend", "quantization", "quantization_preapplied", "level", "func_name", "weights_path"}
            },
            "m2m/capture/bundle.py": {"write_bundle": {"source_path", "capture_trace", "conversion_result"}},
            "m2m/capture/provenance.py": {"write_capture_receipt": {"source_path"}},
        },
    )


def _frontend_trace_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report the exact optional APIs needed for frontend-op and precision evidence."""
    from .sealed_m2m import (
        _source_api_missing,
    )

    return _source_api_missing(
        m2m_root,
        {
            "m2m/api.py": {"convert": {"capture_trace", "original_frontend_snapshot"}},
            "m2m/capture/trace.py": {
                "capture_frontend_snapshot": {"stage"},
                "materialize_frontend_precision": {"dtype", "original_frontend_snapshot"},
            },
        },
    )


def _fp32_stage_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Require the opt-in typed-retarget API before a sealed stage is selected."""
    from .sealed_m2m import (
        STAGING_API_REQUIREMENTS,
        _source_api_missing,
    )

    return _source_api_missing(m2m_root, STAGING_API_REQUIREMENTS)


def _static_integer_reference_api_missing(m2m_root: Path) -> tuple[str, ...]:
    """Report APIs needed before a static W8A8 capture can claim integer arithmetic."""
    from .sealed_m2m import (
        _source_api_missing,
    )

    return _source_api_missing(
        m2m_root,
        {
            "m2m/capture/pt2e_integerize.py": {"integerize_pt2e": set()},
            "m2m/capture/pt2e_integer_reference.py": {"run_pt2e_integer_reference": set()},
        },
    )


def _source_api_missing(m2m_root: Path, required: dict[str, dict[str, set[str]]]) -> tuple[str, ...]:
    """Inspect selected source signatures only; neither execution nor provenance proof."""
    from .sealed_m2m import (
        ast,
    )

    missing: list[str] = []
    for member, functions in required.items():
        source = m2m_root / member
        if not source.is_file() or source.is_symlink():
            missing.append(member)
            continue
        try:
            module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError):
            missing.append(f"{member}: readable Python source")
            continue
        top_level = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        for function, parameters in functions.items():
            node = top_level.get(function)
            if node is None:
                missing.append(f"{member}:{function}")
                continue
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            declared = {argument.arg for argument in arguments}
            missing.extend(f"{member}:{function}({name})" for name in sorted(parameters - declared))
    return tuple(missing)


def _validate_snapshots(
    source: Path, runtime: Path, output_mount: Path, *, schema: str, runtime_reuse: dict | None = None
) -> None:
    from .sealed_m2m import (
        SCHEMA,
        SCHEMA_V1,
        SCHEMA_V3,
        SealedM2MError,
        readonly_runtime,
    )

    for name in ("source", "capture-out", "tmp", "dev", "proc"):
        path = runtime / name
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise SealedM2MError(f"guest mount point /{name} is not empty")
    executable = (
        readonly_runtime.file_path(runtime_reuse, "opt/capture-venv/bin/python")
        if runtime_reuse
        else runtime / "opt/capture-venv/bin/python"
    )
    if executable.is_symlink() or not executable.is_file() or executable.read_bytes()[:4] != b"\x7fELF":
        raise SealedM2MError("snapshotted interpreter is not a direct ELF")
    worker = (
        source / "worker.py" if schema == SCHEMA_V1 else source / "merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    )
    if not worker.is_file() or not (source / "workload/loader.py").is_file():
        raise SealedM2MError("selected source entrypoints are absent")
    if schema in {SCHEMA, SCHEMA_V3} and any(
        not (source / "merlin-src/merlin/_data/schemas" / name).is_file()
        for name in ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    ):
        raise SealedM2MError("selected Merlin package data is absent from the snapshot")
    mounted = runtime / output_mount.relative_to("/")
    if mounted.is_symlink() or not mounted.is_dir() or any(mounted.iterdir()):
        raise SealedM2MError("host-resolvable guest output mount point is not empty")


def _materialized(
    output: Path, source: Path, output_mount: Path, *, worker_member: str = "worker.py"
) -> dict[str, Any]:
    from .sealed_m2m import (
        SealedM2MError,
        _file_digest,
        json,
        strict_json,
    )

    from merlin.targetgen.application_inventory import verify_capture_receipt

    result = verify_capture_receipt(output / "model.mlir")
    if result.get("status") != "verified_materialized":
        raise SealedM2MError(f"M2M materialized receipt is unverified: {result.get('errors')}")
    mlir = (output / "model.mlir").read_text()
    pointer = "prov.weights_file = " + json.dumps(str(output_mount / "weights.safetensors"))
    if mlir.count("prov.weights_file") != 1 or pointer not in mlir:
        raise SealedM2MError("saved MLIR does not identify host-resolvable, receipt-bound weights")
    payload = strict_json.loads((output / "capture_receipt.json").read_bytes())
    loader = payload.get("source") or {}
    entry = (payload.get("tool") or {}).get("executed_entrypoint") or {}
    worker = source / worker_member
    if (
        loader.get("path") != "/source/workload/loader.py"
        or loader.get("sha256") != _file_digest(source / "workload/loader.py")
        or entry.get("path") != "/source/" + worker_member
        or entry.get("sha256") != _file_digest(worker)
    ):
        raise SealedM2MError("M2M receipt does not bind the snapshotted entrypoints")
    sources = (payload.get("tool") or {}).get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or (payload.get("tool") or {}).get("source_inventory_status") != "complete"
    ):
        raise SealedM2MError("M2M receipt lacks its complete direct source inventory")
    for name, digest in sources.items():
        if (
            not isinstance(name, str)
            or not name.startswith("m2m/")
            or PurePosixPath(name).as_posix() != name
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            or "\x00" in name
        ):
            raise SealedM2MError(f"unsafe M2M source member: {name!r}")
        path = source / "m2m-src" / name
        if not path.is_file() or _file_digest(path) != digest:
            raise SealedM2MError(f"M2M source digest differs: {name}")
    return {"status": result["status"], "receipt_sha256": result["receipt_sha256"]}


def _materialized_v2(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    from .sealed_m2m import (
        SealedM2MError,
        _FLOAT_DTYPES,
        _materialized,
        _recipe_selection,
        _worker_options,
        native_integer,
        output_staging_error,
        strict_json,
    )

    result = _materialized(
        output, source, output_mount, worker_member="merlin-src/merlin/targetgen/_m2m_capture_worker.py"
    )
    metadata = strict_json.loads((output / "meta.json").read_bytes())
    if not isinstance(metadata, dict) or metadata.get("dtype") != plan.get("dtype"):
        raise SealedM2MError("capture metadata does not identify the selected dtype")
    stage_reason = output_staging_error(
        output,
        metadata,
        selected=_worker_options(plan.get("worker_options")).get("stage_fp32", False),
        recipe=plan.get("recipe") is not None,
    )
    if stage_reason is not None:
        raise SealedM2MError(stage_reason)
    recipe = plan.get("recipe")
    if native_integer.selected(plan.get("worker_options")):
        try:
            return {**result, **native_integer.validate(output, metadata, plan)}
        except ValueError as error:
            raise SealedM2MError(str(error)) from error
    elif plan.get("dtype") in _FLOAT_DTYPES:
        if recipe is not None or metadata.get("recipe_sha256") is not None:
            raise SealedM2MError("fp32 capture unexpectedly selected a quantization recipe")
    elif plan.get("dtype") == "int8":
        selected = source / "inputs/quant_recipe.json"
        if not isinstance(recipe, dict) or not selected.is_file():
            raise SealedM2MError("int8 capture did not retain the selected recipe bytes")
        observed = _recipe_selection(selected, dtype="int8")
        if any(observed[key] != recipe.get(key) for key in ("bytes", "sha256", "recipe_sha256")):
            raise SealedM2MError("int8 recipe snapshot differs from the selected plan")
        stats = metadata.get("quantization_stats") or {}
        agreement = (metadata.get("integerization_receipt") or {}).get("golden_agreement") or {}
        if (
            metadata.get("scheme") != "int8_static_act_int8_weight"
            or metadata.get("recipe_sha256") != recipe.get("recipe_sha256")
            or stats.get("recipe_sha256") != recipe.get("recipe_sha256")
            or agreement.get("status") != "passed"
            or agreement.get("reference") != "pt2e_integer"
            or metadata.get("software_numerical_engine") not in (None, "integer_reference")
        ):
            raise SealedM2MError("int8 capture lacks selected recipe and independent integer-reference agreement")
    else:
        raise SealedM2MError("unsupported selected capture dtype")
    return result


def _dynamic_integer_census(meta: dict, stats: dict, stage: Path, emitted: int, exported: int) -> int:
    """Use the same byte-bound recipe/trace census as the application inventory."""
    from .sealed_m2m import (
        SealedM2MError,
        _file_digest,
    )

    from merlin.capture.integerization import dynamic_integerization_count, dynamic_integerization_projection
    from merlin.targetgen.frontend_trace import loads_frontend_trace

    if stats.get("dynamic_contractions") is None:
        return 0  # The caller keeps the historical emitted == exported requirement.
    trace_path, model = stage / "frontend-trace.json", stage / "model.mlir"
    if any(path.is_symlink() or not path.is_file() for path in (trace_path, model)):
        raise SealedM2MError("dynamic contraction census lacks the stage's frontend trace or program")
    try:
        raw = trace_path.read_bytes()
        import hashlib

        model_sha256 = _file_digest(model)
        evidence = dynamic_integerization_projection(
            meta,
            stats,
            trace=loads_frontend_trace(raw),
            trace_identity={"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
            program_sha256=model_sha256,
            emitted=emitted,
            exported=exported,
        )
        return dynamic_integerization_count(evidence, emitted=emitted, exported=exported, program_sha256=model_sha256)
    except ValueError as exc:
        raise SealedM2MError(str(exc)) from exc


def _integerization_census_admission(meta: dict, stage: Path):
    """Adapt sealed stage byte custody to the shared integer-export admission policy."""
    import hashlib

    from merlin.targetgen.integerization_export import admit_integerization_census, capture_dynamic_evidence

    def evidence():
        stats = meta.get("quantization_stats")
        if not isinstance(stats, dict) or stats.get("dynamic_contractions") is None:
            return None
        trace_path = stage / "frontend-trace.json"
        if trace_path.is_symlink():
            raise ValueError("dynamic contraction census has an indirect frontend trace")
        return capture_dynamic_evidence(meta, trace_path.read_bytes(), program_sha256=program_sha256)

    model = stage / "model.mlir"
    if model.is_symlink() or not model.is_file():
        raise ValueError("integer contraction census lacks the stage program")
    program_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    return admit_integerization_census(
        meta.get("integerization_receipt"), dynamic_evidence=evidence, program_sha256=program_sha256,
    )


def _materialized_v3(output: Path, source: Path, output_mount: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Verify the complete saved ABI, including every declared session program."""
    from .sealed_m2m import (
        SealedM2MError,
        _FLOAT_DTYPES,
        _file_digest,
        _materialized,
        _materialized_v2,
        _recipe_selection,
        _worker_options,
        native_integer,
        output_staging_error,
        strict_json,
    )

    from merlin.capture.integerization import contraction_partition

    native = native_integer.selected(plan.get("worker_options"))

    def native_work(meta, stage):
        from merlin.targetgen.capsule_source import _require_materialized_contraction

        try:
            native_integer.validate(stage, meta, plan)
            return _require_materialized_contraction((stage / "model.mlir").read_text(), dtype="int8")
        except ValueError as error:
            raise SealedM2MError(str(error)) from error

    def integer_partition(meta: dict, stage: Path) -> tuple[int, dict[str, int]]:
        if not isinstance(meta, dict):
            raise SealedM2MError("capture lacks verified integer contractions: invalid metadata")
        receipt = meta.get("integerization_receipt") or {}
        try:
            partition = contraction_partition(receipt)
        except ValueError as exc:
            raise SealedM2MError(f"capture lacks verified integer contractions: {exc}") from exc
        agreement = receipt.get("golden_agreement") or {}
        stats, recipe = meta.get("quantization_stats"), meta.get("recipe")
        if not all(isinstance(value, dict) for value in (agreement, stats, recipe)):
            raise SealedM2MError("capture lacks verified integer contractions: malformed numerical evidence")
        executed = agreement.get("executed_contractions") or {}
        outputs = agreement.get("outputs")
        numerical_rows = [agreement, *outputs] if isinstance(outputs, list) and outputs else []
        by_kind = receipt["quantized_by_kind"]
        count = receipt.get("exported_integer_mm_count")
        emitted = receipt.get("integer_mm_emitted")
        max_k = receipt.get("max_reduction_k")
        annotated = stats.get("annotated_contractions")
        engine = recipe.get("software_numerical_engine")
        verdict = _integerization_census_admission(meta, stage)
        if not verdict.accepted:
            raise SealedM2MError(f"capture lacks independently verified integer contractions: {verdict.reason}")
        if (
            engine != "integer_reference"
            or type(annotated) is not int
            or annotated != partition["seen"]
            or agreement.get("status") != "passed"
            or agreement.get("reference") != "pt2e_integer"
            or type(agreement.get("samples")) is not int
            or agreement["samples"] < 1
            or not numerical_rows
            or any(
                not isinstance(result, dict)
                or result.get("finite") is not True
                or (result is not agreement and result.get("within_tolerance") is not True)
                or any(
                    type(result.get(key)) not in (int, float) or result[key] != 0.0
                    for key in ("atol", "rtol", "max_abs", "max_rel")
                )
                for result in numerical_rows
            )
            or not isinstance(executed, dict)
            or any(
                type(executed.get(key)) is not int
                for key in ("total", "selected", "observed", "linear", "conv2d", "matmul")
            )
            or executed.get("total") != partition["integerized"]
            or executed.get("selected") != partition["seen"]
            or executed.get("observed") != partition["seen"]
            or any(executed.get(kind) != by_kind[kind]["integerized"] for kind in ("linear", "conv2d", "matmul"))
            or type(count) is not int
            or count <= 0
            or type(emitted) is not int
            or emitted < partition["integerized"]
            or receipt.get("accumulator_bound_checked") is not True
            or type(max_k) is not int
            or max_k < 1
            or max_k * 128 * 128 > (1 << 31) - 1
        ):
            raise SealedM2MError(
                "capture lacks independently verified integer contractions or complete precision census"
            )
        return count, partition

    report_path = output / "session-receipt.json"
    contract_path = output / "session_contract.yaml"
    if not report_path.exists():
        if report_path.is_symlink():
            raise SealedM2MError("single-program capture has an indirect session receipt")
        if (output / "stages").exists() or (output / "stages").is_symlink():
            raise SealedM2MError("single-program capture has an undeclared stages directory")
        single_contract = None
        if contract_path.exists() or contract_path.is_symlink():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise SealedM2MError("single-program session contract is indirect or absent")
            try:
                import yaml

                contract = yaml.safe_load(contract_path.read_bytes())
            except (OSError, ValueError, yaml.YAMLError) as exc:
                raise SealedM2MError("single-program session contract is unreadable") from exc
            stages = contract.get("stages") if isinstance(contract, dict) else None
            schedule = contract.get("stage_schedule") if isinstance(contract, dict) else None
            if (
                not isinstance(contract, dict)
                or contract.get("version") != 1
                or not isinstance(stages, list)
                or len(stages) != 1
                or not isinstance(stages[0], str)
                or PurePosixPath(stages[0]).name != stages[0]
                or stages[0] in {"", ".", ".."}
                or "\\" in stages[0]
                or "\x00" in stages[0]
                or not isinstance(schedule, list)
                or len(schedule) != 1
                or not isinstance(schedule[0], dict)
                or schedule[0].get("name") != stages[0]
                or any(
                    contract.get(label) is not None and not isinstance(contract[label], dict)
                    for label in ("correctness", "quality")
                )
            ):
                raise SealedM2MError("single-program session contract has an invalid stage roster")
            for member in (
                contract.get("inputs"),
                (contract.get("correctness") or {}).get("golden"),
                (contract.get("quality") or {}).get("golden"),
            ):
                if member is None:
                    continue
                if (
                    not isinstance(member, str)
                    or not member
                    or PurePosixPath(member).is_absolute()
                    or PurePosixPath(member).as_posix() != member
                    or ".." in PurePosixPath(member).parts
                    or "\\" in member
                    or "\x00" in member
                    or (output / member).is_symlink()
                    or not (output / member).is_file()
                ):
                    raise SealedM2MError("single-program session contract references an absent or unsafe file")
            single_contract = _file_digest(contract_path)
        verified = _materialized_v2(output, source, output_mount, plan)
        integer_work = 0
        if native:
            integer_work = native_work(strict_json.loads((output / "meta.json").read_bytes()), output)
        elif plan["dtype"] == "int8":
            meta = strict_json.loads((output / "meta.json").read_bytes())
            exported, partition = integer_partition(meta, output)
            # Preserve the exact historical sealed-v3 materialized summary:
            # this legacy field counts exported integer operations. The new
            # ledger distinguishes source contractions only for mixed captures,
            # which the old verifier never admitted.
            integer_work = exported
        return {
            "kind": "single",
            **verified,
            "integer_contractions": integer_work,
            **(
                {"exported_integer_operations": exported, "contraction_partition": partition}
                if not native and plan["dtype"] == "int8" and partition["preserved"]
                else {}
            ),
            **({"session_contract_sha256": single_contract} if single_contract is not None else {}),
        }
    quantized = plan["dtype"] == "int8" and not native
    if quantized:
        recipe = plan.get("recipe")
        selected_recipe = source / "inputs/quant_recipe.json"
        if not isinstance(recipe, dict) or not selected_recipe.is_file():
            raise SealedM2MError("multi-program int8 capture lacks the selected recipe")
        observed_recipe = _recipe_selection(selected_recipe, dtype="int8")
        if any(observed_recipe[key] != recipe.get(key) for key in ("bytes", "sha256", "recipe_sha256")):
            raise SealedM2MError("multi-program int8 recipe snapshot differs from the selected plan")
    elif not native and (plan["dtype"] not in _FLOAT_DTYPES or plan.get("recipe") is not None):
        raise SealedM2MError("multi-program capture has an unsupported dtype or recipe")
    if any(path.is_symlink() or not path.is_file() for path in (report_path, contract_path)):
        raise SealedM2MError("multi-program session receipt or contract is absent or indirect")
    try:
        import yaml

        report = strict_json.loads(report_path.read_bytes())
        contract = yaml.safe_load(contract_path.read_bytes())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SealedM2MError("multi-program session evidence is unreadable") from exc
    if not isinstance(report, dict) or not isinstance(contract, dict):
        raise SealedM2MError("multi-program session evidence is malformed")
    selected_stage = _worker_options(plan.get("worker_options")).get("stage_fp32", False)
    if (selected_stage or native) and report.get("source_state_unchanged") is not True:
        raise SealedM2MError("FP32 staging lacks unchanged shared source state")
    names = contract.get("stages")
    programs = contract.get("programs")
    rows = report.get("programs")
    if (
        contract.get("version") != 2
        or report.get("schema") != "merlin.model_session_capture.v1"
        or report.get("session_contract_sha256") != _file_digest(contract_path)
        or report.get("recipe_sha256") != (plan["recipe"]["recipe_sha256"] if quantized else None)
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or not isinstance(programs, list)
        or not isinstance(rows, list)
        or len(programs) != len(names)
        or len(rows) != len(names)
        or any(not isinstance(row, dict) for row in [*programs, *rows])
    ):
        raise SealedM2MError("multi-program session roster or contract digest differs")
    for index, name in enumerate(names):
        if (
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).name != name
            or name in {".", ".."}
            or "\\" in name
            or "\x00" in name
            or programs[index].get("name") != name
            or programs[index].get("bundle") != f"stages/{name}"
            or rows[index].get("name") != name
        ):
            raise SealedM2MError("multi-program session contains an unsafe or mismatched program")
    stages_root = output / "stages"
    if (
        stages_root.is_symlink()
        or not stages_root.is_dir()
        or sorted(path.name for path in stages_root.iterdir()) != sorted(names)
    ):
        raise SealedM2MError("multi-program stage directory membership differs from the contract")
    verified = []
    integer_work = 0
    for name, row in zip(names, rows):
        stage = stages_root / name
        if stage.is_symlink() or not stage.is_dir():
            raise SealedM2MError("multi-program stage is indirect or absent")
        try:
            meta = strict_json.loads((stage / "meta.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise SealedM2MError("multi-program stage metadata is unreadable") from exc
        if not isinstance(meta, dict):
            raise SealedM2MError("multi-program stage metadata is malformed")
        stage_reason = output_staging_error(
            stage,
            meta,
            selected=selected_stage,
            recipe=quantized and row.get("precision_selection") == "recipe",
        )
        if stage_reason is not None:
            raise SealedM2MError(stage_reason)
        mode = row.get("precision_selection")
        if native and mode == native_integer.MODE:
            integer_work += native_work(meta, stage)
        elif quantized and mode == "recipe":
            count, partition = integer_partition(meta, stage)
            if (
                meta.get("dtype") != "int8"
                or meta.get("scheme") != "int8_static_act_int8_weight"
                or meta.get("recipe_sha256") != recipe["recipe_sha256"]
                or (meta.get("quantization_stats") or {}).get("recipe_sha256") != recipe["recipe_sha256"]
            ):
                raise SealedM2MError("multi-program recipe stage lacks verified integer contractions")
            integer_work += count
        elif quantized and mode == "no_recipe_work":
            if (
                meta.get("dtype") not in _FLOAT_DTYPES
                or meta.get("recipe_sha256") is not None
                or meta.get("integerization_receipt") is not None
            ):
                raise SealedM2MError("multi-program host-only stage unexpectedly claims recipe precision")
        elif not native and not quantized and mode == "untransformed":
            if meta.get("dtype") not in _FLOAT_DTYPES or meta.get("recipe_sha256") is not None:
                raise SealedM2MError("multi-program FP32 stage unexpectedly claims recipe precision")
        else:
            raise SealedM2MError("multi-program stage has an unsupported precision selection")
        result = _materialized(
            stage,
            source,
            output_mount / "stages" / name,
            worker_member="merlin-src/merlin/targetgen/_m2m_capture_worker.py",
        )
        stage_receipt = strict_json.loads((stage / "capture_receipt.json").read_bytes())
        if (
            row.get("receipt_sha256") != result["receipt_sha256"]
            or (row.get("materialized_abi") or {}).get("complete") is not True
            or row.get("materialized_abi") != stage_receipt.get("materialized_abi")
        ):
            raise SealedM2MError("multi-program stage receipt differs from the session receipt")
        verified.append(
            {
                "name": name,
                "model_sha256": _file_digest(stage / "model.mlir"),
                "receipt_sha256": result["receipt_sha256"],
                "precision_selection": mode,
                **(
                    {"contraction_partition": partition, "exported_integer_operations": count}
                    if quantized and mode == "recipe" and partition["preserved"]
                    else {}
                ),
            }
        )
    if (quantized or native) and integer_work == 0:
        raise SealedM2MError("multi-program int8 capture contains no verified integer contractions")
    return {
        "kind": "session",
        "session_contract_sha256": _file_digest(contract_path),
        "session_receipt_sha256": _file_digest(report_path),
        "programs": verified,
        "integer_contractions": integer_work,
    }


def _stage_source(plan: dict[str, Any], source: Path) -> None:
    """Copy exactly the source members named by the selected v2 plan."""
    from .sealed_m2m import (
        SCHEMA_V3,
        SealedM2MError,
        _copy_source_tree,
        _snapshot_tree,
        shutil,
    )

    _copy_source_tree(
        Path(plan["m2m_root"]) / "m2m",
        source / "m2m-src/m2m",
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    if plan.get("frozen_origin") is not None:
        shutil.copy2(plan["frozen_origin"]["path"], source / "m2m-origin.json")
    _copy_source_tree(Path(plan["workload_root"]), source / "workload")
    _copy_source_tree(
        Path(plan["merlin_root"]),
        source / "merlin-src/merlin",
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    if _snapshot_tree(source / "merlin-src/merlin") != plan["selected_trees"]["merlin"]:
        raise SealedM2MError("Merlin package snapshot differs from selected bytes")
    bundled_schemas = source / "merlin-src/merlin/_data/schemas"
    if not bundled_schemas.exists():
        _copy_source_tree(Path(plan["schemas_root"]), bundled_schemas)
    if plan.get("recipe"):
        selected_recipe = source / "inputs/quant_recipe.json"
        selected_recipe.parent.mkdir()
        shutil.copy2(plan["recipe"]["path"], selected_recipe)
    if plan.get("schema") == SCHEMA_V3:
        for row in plan["selected_inputs"]:
            destination = source / "inputs" / row["guest_member"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if row["kind"] == "file":
                shutil.copy2(row["source"], destination)
            else:
                _copy_source_tree(Path(row["source"]), destination)
