"""Staged-capture application compilation behind the existing native CLI."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.c_parameters import bind_application_parameters, parameter_source_identity
from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
from merlin.frontends.compile_inputs import load_compile_inputs, load_lifted_argument_bytes
from merlin.semantic_compiler.constant_math_backend import MathUnavailable
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.source_preparation import preparation_counts, prepare_source
from merlin.semantic_compiler.target_binding import NativeCompilationError


def _source(args, binding, target_identity):
    if (
        not args.compile_inputs
        or args.mode != "diagnostic"
        or args.linalg_entry
        or args.physical_layouts
        or args.diagnostic_region
        or args.diagnostic_graph_policy
        or args.diagnostic_tile_index is not None
        or args.whole_contraction
        or args.row_softmax_tile_index is not None
        or args.ordered_k_tile is not None
        or args.ordered_m_tile is not None
        or args.ordered_n_tile is not None
        or args.outer_tile_index is not None
    ):
        raise ValueError("application compilation requires staged inputs, diagnostic mode and no region overrides")
    factory = getattr(binding, "application_adapter", None)
    if factory is None:
        raise NativeCompilationError("unsupported_semantics", "selected support has no semantic application adapter")
    directory = Path(args.compile_inputs)
    staged = load_compile_inputs(directory)
    state_path = getattr(args, "application_state", None)
    declaration = None
    if state_path:
        from merlin.compile.application_state import StateDeclaration

        declaration = StateDeclaration.from_record(json.loads(Path(state_path).read_text()), staged)
    elif any(item["kind"] == "buffer" for item in staged.argument_bindings):
        raise NativeCompilationError(
            "unsupported_semantics", "application needs an explicit state interface for lifted buffers"
        )
    external = getattr(args, "application_parameters", "embedded") == "external"
    parameters = load_lifted_argument_bytes(
        directory, staged,
        **({"kinds": ("constant", "buffer"), "max_bytes": 64 * 1024 * 1024} if external else {}),
    )
    state_arguments = {port.argument_index for port in declaration.ports} if declaration else set()
    source = translate_linalg_text(
        staged.program,
        entry=staged.entry,
        target_identity=target_identity,
        lowering_policy=args.mode,
        argument_constants={index: payload for index, payload in parameters.items() if index not in state_arguments},
    ).request
    state = None
    if external:
        source = replace(source, source_identity=parameter_source_identity(source.source_identity, staged))
    if declaration is not None:
        source = replace(
            source,
            source_identity=sha256(
                (source.source_identity + ":state:" + declaration.compilation_identity(parameters)).encode()
            ).hexdigest(),
        )
    adapter = factory(staged.precision_policy, original_source_sha256=staged.original_source_sha256)
    try:
        source, folding = prepare_source(
            source, constant_math_policy=(staged.precision_policy or {}).get("source_constant_math")
        )
    except MathUnavailable as error:
        raise NativeCompilationError(error.status, str(error)) from error
    if folding.get("limits"):
        raise NativeCompilationError("resource_limit", json.dumps(folding["limits"], sort_keys=True))
    if declaration is not None:
        state = declaration.bind(source, parameters)
    model_parameters = bind_application_parameters(directory, staged, source) if external else None
    return staged, source, adapter, state, folding, model_parameters


def compile_application_report(args, *, binding, snapshot, limits, write_status):
    if args.abi != "auto" or args.application_memory_bytes is None or args.application_memory_bytes <= 0:
        raise ValueError("application compilation needs --abi auto and an explicit application memory byte budget")
    staged, source, adapter, state, folding, parameters = _source(args, binding, snapshot.profile.target_identity)
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise ValueError("application compilation requires a fresh output directory")
    if args.status_file and Path(args.status_file).resolve().is_relative_to(output.resolve()):
        raise ValueError("application status must be outside the compilation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-application-", dir=output.parent) as temporary:
        destination = Path(temporary) / "application"
        try:
            manifest = compile_semantic_application(
                source,
                adapter=adapter,
                snapshot=snapshot,
                target_source=Path(args.target_source),
                destination=destination,
                limits=limits,
                name=args.application_name,
                capacity=args.application_memory_bytes,
                state=state,
                source_folding=folding,
                parameters=parameters,
            )
            _, current, _, current_state, current_folding, current_parameters = _source(
                args, binding, snapshot.profile.target_identity
            )
            if (current.digest() != source.digest() or current_state != state or current_folding != folding
                    or current_parameters != parameters):
                raise ValueError("staged application inputs changed during compilation")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
            if destination.exists():
                destination.rename(output)
                (output / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "merlin.semantic_application_failure.v1",
                            "status": getattr(error, "status", "compile_error"),
                            "engine": args.engine,
                            "request_digest": source.digest(),
                            "target_identity": source.target_identity,
                            "source_identity": staged.program_sha256,
                            "reason": str(error),
                            "diagnostics_only": True,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                )
            raise
        destination.rename(output)
    report = {
        "schema": "merlin.native_compilation_status.v1",
        "status": "emitted",
        "engine": args.engine,
        "support": args.support,
        "out": str(output),
        "artifact_kind": "program_set",
        "request_digest": source.digest(),
        "target_identity": source.target_identity,
        "source_identity": staged.program_sha256,
        "source_kind": "staged_semantic_application",
        "program_count": len(manifest["segments"]),
        "region_count": len(manifest["regions"]),
        "application_interface": manifest["application_interface"],
        "execution": "not_run",
        **preparation_counts(folding),
        "qualification": "diagnostic_precision_change",
    }
    if args.status_file:
        write_status(Path(args.status_file), report)
    return report


def verify_application_command(args, write_status):
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import load_native_target_binding

    artifact, output = Path(args.artifact), Path(args.out)
    safe_status = not (
        output.resolve().is_relative_to(artifact.resolve())
        or output.resolve().is_relative_to(Path(args.snapshot).resolve())
    )
    try:
        if not safe_status or not args.support:
            raise ValueError("application verification needs support and an external status file")
        binding = load_native_target_binding(args.support)
        snapshot = open_native_snapshot(Path(args.snapshot))
        if snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("application snapshot differs from selected support profile")
        _, source, adapter, state, folding, parameters = _source(args, binding, snapshot.profile.target_identity)
        report = verify_semantic_application(
            source,
            adapter=adapter,
            snapshot=snapshot,
            target_source=Path(args.target_source),
            artifact=artifact,
            state=state,
            source_folding=folding,
            parameters=parameters,
        )
        report.update(
            status="verified_artifacts", engine=args.engine, scope="application_source_and_emitted_C_identity"
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        report = {"status": getattr(error, "status", "compile_error"), "engine": args.engine, "reason": str(error)}
    if safe_status:
        write_status(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "verified_artifacts" else 2
