"""Ordinary saved-model build, explicit execution and independent verification lifecycle.

The compile service remains compile-only. This installed-core compatibility
workflow coordinates those distinct stages and retains the original receipts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.compiler.backend import host as host
from merlin.compiler.backend.isa import implementation_functions as _host_isa_functions
from merlin.compiler.backend.isa import require_host_isa_dts
from merlin.compiler.build.modeloptions import BaremetalModelError, _output_root, _sha, prepare_saved
from merlin.compiler.build.saved import build_saved, select_build
from merlin.runtime import modelinputs as MI
from merlin.runtime.backends import spike as spike_backend  # noqa: F401 -- retained operator patch surface
from merlin.runtime.backends import spike_model as spike_model
from merlin.runtime.model import (
    _native_engine,
    _retain_simulator_failure_output,
    _spike_extension_backend,
    launch_saved,
    observe_profiles,
    select_runtime,
)
from merlin.verification.model import (
    _agreement,
    _full_references,
    _judge_full_readback,
    _judge_prefix,
    _read_full_values,
    _saved_reference,
    console_text_metrics,
    read_full_outputs,
    revalidate_saved,
)

if TYPE_CHECKING:
    from merlin.compiler.lower.model.graph_optimizations import GraphOptimizationContract
    from merlin.compiler.lower.model.host_graph_policy import HostGraphPolicy
    from merlin.compiler.lower.model.source_dense import SourceDenseLowering
    from merlin.compiler.lower.model.source_pointwise import SourcePointwiseLowering
    from merlin.compiler.lower.model.source_rows import SourceRowLowering


def implementation_functions():
    """Actual workflow bindings whose code is observed by original-host custody."""
    return (
        implementation_functions,
        implementation_paths,
        compile_saved_model,
        prepare_saved,
        select_build,
        build_saved,
        select_runtime,
        launch_saved,
        observe_profiles,
        revalidate_saved,
        _native_engine,
        _spike_extension_backend,
        _retain_simulator_failure_output,
        _sha,
        _output_root,
        _saved_reference,
        _full_references,
        _agreement,
        _judge_prefix,
        console_text_metrics,
        read_full_outputs,
        _read_full_values,
        _judge_full_readback,
        require_host_isa_dts,
        _host_isa_functions,
        *_host_isa_functions(),
    )


def implementation_paths():
    """Exact defining sources, including every extracted build/run/check owner."""
    import inspect

    return tuple(
        dict.fromkeys(Path(inspect.getsourcefile(function)).resolve() for function in implementation_functions())
    )


def compile_saved_model(
    *,
    capture: str | Path,
    package: str | Path,
    board_catalog: str | Path,
    board: str,
    dts: str | Path,
    output: str | Path,
    target: str,
    run: str,
    arena_mb: int,
    timeout_s: int = 300,
    reference_file: str | None = None,
    rtl_facts: str | Path | None = None,
    device: Any | None = None,
    math_archive_symbols: Sequence[str] | None = None,
    host_math_symbols: Sequence[str] | None = None,
    readback: str = "prefix",
    readonly_input_indices: tuple[int, ...] = (),
    max_readonly_input_bytes: int = 1 << 20,
    group_profile: bool = False,
    op_profile: bool = False,
    tolerance: dict[str, float] | None = None,
    materialized_inputs=None,
    respect_captured_quantization_scope: bool = False,
    host_llvm_bitcode: bool = False,
    blob_weights: bool = True,
    weights_source: str | Path | None = None,
    shared_source_lowerings: tuple[SourceRowLowering | SourcePointwiseLowering | SourceDenseLowering, ...] = (),
    graph_optimizations: GraphOptimizationContract | HostGraphPolicy | None = None,
    source_optimizations=None,
    source_recomputation=None,
    prepack_weights: bool = False,
    share_exact_tensor_preparations: bool = False,
    dense_input_views: bool = False,
    host_math_policy: str = "native",
    host_elementwise_fusion: bool = False,
    host_bounded_rne: bool = False,
    host_roundeven: bool = False,
) -> dict[str, Any]:
    """Compile one saved model; ``none`` makes no execution or numerical claim.

    ``blob_weights`` stores each routed weight once: the device-layout copy goes to the weights blob and
    the original the device alone read is dropped (:mod:`merlin.compiler.lower.model.blob_weights`).

    ``weights_source`` reuses an immutable existing blob only after this stage
    independently derives identical complete bytes, size and hash.

    ``shared_source_lowerings`` forwards explicit qualified source-family
    bindings to the ordinary builder. Its source discovery, fallback and
    device admission checks remain authoritative. Retained provider and
    preparation records establish custody, not a new execution qualification.

    ``graph_optimizations`` accepts an exact source contract or a HostGraphPolicy
    from original versioned software permissions. The latter binds the actual
    prepared source and rechecks software custody through execution.

    ``prepack_weights=True`` pre-applies checked static immutable weight layout
    chains before both lowering and runtime ABI generation. The generated caller
    requires pure tensor source and original non-updated stored arguments;
    mutable state, unknown effects and unsupported layouts refuse the selection.
    False retains the original capture and performs no prepacking I/O.

    ``share_exact_tensor_preparations=True`` explicitly selects the existing
    immutable view/constant and exact integer-to-float preparation CSE. It keeps
    provenance, arithmetic boundaries and independent output preparations; the
    ordinary lowering receipt records each actual match. False is unchanged.

    ``readback="full"`` builds the image to print every forward result complete
    (:mod:`merlin.runtime.whole_model_readback`); execution then holds each result to the capture's
    own reference (``goldens.npz`` + ``output_order.json``, or the one ``reference_file``), exactly or
    within ``tolerance`` (``{atol, rtol}``). ``group_profile`` brackets every routed device-group call
    with cycle counts and records them. A ``spike`` run goes through the target backend with its
    accelerator extension when the backend declares one.

    ``readonly_input_indices`` explicitly asserts unchanged bytes during forward
    for those original capture arguments (inputs or stored weights). With full
    readback, actual original buffers are captured after the call and before
    declared state commit, then compared completely against source-derived input
    bytes under ``max_readonly_input_bytes``. This diagnostic changes the image
    and adds copy work inside its cycle bracket; it grants no completion, timing,
    alias or guard-region qualification. The empty tuple leaves it disabled.

    ``op_profile`` selects existing typed top-level operation markers and retains
    their table. Execution records a complete raw-counter trace; categories do
    not establish placement or hardware latency. Pair an uninstrumented control
    because markers can alter optimization and allocation.

    ``reference_file`` is required for execution because a weight-only capture's
    ``golden.npy`` is not necessarily the reference for a W8A8 execution.  This
    initial executor supports one complete float32 OUT (at most 4096 values).

    ``host_math_policy`` selects an explicit host runtime recipe. Native preserves
    the default build arguments; unsupported policies refuse before input/output IO.

    ``host_elementwise_fusion`` explicitly selects the existing second guarded
    elementwise fusion after named-op generalization for the scalar host lane.
    It changes neither accelerator lowering nor the captured numerical recipe.
    The default stays off; realized code and full outputs need paired measurement.

    ``host_bounded_rne`` explicitly selects the shared late bounded clamp/RNE
    legalizer for a scalar RV64GC host package. It preserves defined integer
    results, accepts both residual and intrinsic source forms, and records the
    actual pre-object transform. No accelerator implementation is selected.

    ``host_roundeven`` explicitly selects Merlin's existing exact scalar
    RV64GC/lp64d binary32 runtime helper. The generic C binary64 helper remains;
    neither the floating environment nor the captured math policy changes.
    """
    prepare_saved_stage = prepare_saved(
        arena_mb=arena_mb,
        blob_weights=blob_weights,
        board=board,
        board_catalog=board_catalog,
        capture=capture,
        dense_input_views=dense_input_views,
        device=device,
        dts=dts,
        graph_optimizations=graph_optimizations,
        host_bounded_rne=host_bounded_rne,
        host_elementwise_fusion=host_elementwise_fusion,
        host_llvm_bitcode=host_llvm_bitcode,
        host_math_policy=host_math_policy,
        host_roundeven=host_roundeven,
        max_readonly_input_bytes=max_readonly_input_bytes,
        op_profile=op_profile,
        output=output,
        package=package,
        prepack_weights=prepack_weights,
        readback=readback,
        readonly_input_indices=readonly_input_indices,
        respect_captured_quantization_scope=respect_captured_quantization_scope,
        rtl_facts=rtl_facts,
        run=run,
        share_exact_tensor_preparations=share_exact_tensor_preparations,
        shared_source_lowerings=shared_source_lowerings,
        source_optimizations=source_optimizations,
        source_recomputation=source_recomputation,
        target=target,
        timeout_s=timeout_s,
        tolerance=tolerance,
    )
    recomputation_check = prepare_saved_stage.recomputation_check
    full = prepare_saved_stage.full
    source_choices = prepare_saved_stage.source_choices
    host_bounded_rne = prepare_saved_stage.host_bounded_rne
    early_layout_policy = prepare_saved_stage.early_layout_policy
    storage_policy = prepare_saved_stage.storage_policy
    capture_path = prepare_saved_stage.capture_path
    package_path = prepare_saved_stage.package_path
    catalog_path = prepare_saved_stage.catalog_path
    dts_path = prepare_saved_stage.dts_path
    output_path = prepare_saved_stage.output_path
    device_path = prepare_saved_stage.device_path
    output_path.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "schema": "merlin.baremetal-saved-model.v1",
        "status": "failed",
        "scope": "one saved whole-model ELF; execution is not a Phase 0 release or static-routing proof",
        "execution_route": "host_baseline" if device is None else "device_requested_dispatch_unverified",
        "inputs": {
            "capture": str(capture_path),
            "package": str(package_path),
            "board_catalog": str(catalog_path),
            "board": board,
            "dts": str(dts_path),
            "target": target,
            "run": run,
            "reference_file": reference_file,
            "rtl_facts": str(rtl_facts) if rtl_facts else None,
            "arena_mb": arena_mb,
            "timeout_s": timeout_s,
            "readback": readback,
            **(
                {
                    "readonly_input_indices": list(readonly_input_indices),
                    "max_readonly_input_bytes": max_readonly_input_bytes,
                }
                if readonly_input_indices
                else {}
            ),
            "group_profile": bool(group_profile),
            "tolerance": tolerance,
            "blob_weights": blob_weights,
            "host_math_policy": host_math_policy,
            **({"source_optimizations": source_choices.selection_record()} if source_optimizations is not None else {}),
        },
    }
    try:
        capture_tree = MI.strict_tree_sha256(capture_path)
        package_tree = MI.strict_tree_sha256(package_path)
        catalog_sha, dts_sha = _sha(catalog_path), _sha(dts_path)
        if materialized_inputs is not None:
            from merlin.compiler.frontend.materialized import MaterializedModelInputs

            if type(materialized_inputs) is not MaterializedModelInputs:
                raise BaremetalModelError("explicit trusted materialized model binding required")
            materialized_inputs.verify(capture_path)
            receipt["inputs"]["materialization"] = materialized_inputs.record()
        golden = _saved_reference(
            capture_path,
            reference_file,
            execution=run != "none" and not full,
            materialized_inputs=materialized_inputs,
        )
        references = (
            _full_references(capture_path, reference_file, materialized_inputs=materialized_inputs)
            if full and run != "none"
            else None
        )
        select_build_stage = select_build(
            board=board,
            catalog_path=catalog_path,
            device=device,
            device_path=device_path,
            dts_path=dts_path,
            host_bounded_rne=host_bounded_rne,
            host_elementwise_fusion=host_elementwise_fusion,
            host_roundeven=host_roundeven,
            package_path=package_path,
            receipt=receipt,
            require_isa_dts=require_host_isa_dts,
            storage_policy=storage_policy,
            target=target,
        )
        selected = select_build_stage.selected
        pkg = select_build_stage.pkg
        isas = select_build_stage.isas
        marches = select_build_stage.marches
        device_tree = select_build_stage.device_tree
        select_runtime_stage = select_runtime(
            native_engine=_native_engine,
            receipt=receipt,
            rtl_facts=rtl_facts,
            run=run,
            selected=selected,
            target=target,
        )
        native = select_runtime_stage.native
        facts = select_runtime_stage.facts
        backend = select_runtime_stage.backend
        selection = select_runtime_stage.selection
        revalidate = select_runtime_stage.revalidate
        prepare = select_runtime_stage.prepare
        receipt["inputs"].update(
            {
                "capture_tree": capture_tree,
                "package_tree": package_tree,
                "board_catalog_sha256": catalog_sha,
                "dts_sha256": dts_sha,
                "host_isa": isas[0],
                "simulator_isa": marches[0],
                "golden_sha256": _sha(capture_path / reference_file)
                if golden is not None or (references is not None and reference_file)
                else None,
            }
        )
        if selection is not None:
            receipt["engine_selection"] = selection
        build_saved_stage = build_saved(
            arena_mb=arena_mb,
            blob_weights=blob_weights,
            capture_path=capture_path,
            dense_input_views=dense_input_views,
            device=device,
            early_layout_policy=early_layout_policy,
            full=full,
            graph_optimizations=graph_optimizations,
            group_profile=group_profile,
            host_bounded_rne=host_bounded_rne,
            host_elementwise_fusion=host_elementwise_fusion,
            host_llvm_bitcode=host_llvm_bitcode,
            host_math_policy=host_math_policy,
            host_math_symbols=host_math_symbols,
            host_roundeven=host_roundeven,
            isas=isas,
            math_archive_symbols=math_archive_symbols,
            max_readonly_input_bytes=max_readonly_input_bytes,
            op_profile=op_profile,
            output_path=output_path,
            pkg=pkg,
            prepack_weights=prepack_weights,
            readback=readback,
            readonly_input_indices=readonly_input_indices,
            receipt=receipt,
            recomputation_check=recomputation_check,
            respect_captured_quantization_scope=respect_captured_quantization_scope,
            selected=selected,
            share_exact_tensor_preparations=share_exact_tensor_preparations,
            shared_source_lowerings=shared_source_lowerings,
            source_choices=source_choices,
            source_optimizations=source_optimizations,
            source_recomputation=source_recomputation,
            storage_policy=storage_policy,
            weights_source=weights_source,
        )
        built = build_saved_stage.built
        readonly_record = build_saved_stage.readonly_record
        elf = build_saved_stage.elf
        elf_sha = build_saved_stage.elf_sha
        sidecar = build_saved_stage.sidecar
        profile_table = build_saved_stage.profile_table
        profile_table_sha = build_saved_stage.profile_table_sha
        graph_artifacts = build_saved_stage.graph_artifacts
        early_layout_artifacts = build_saved_stage.early_layout_artifacts
        storage_artifacts = build_saved_stage.storage_artifacts
        preparation = build_saved_stage.preparation
        preparation_sha = build_saved_stage.preparation_sha
        preparation_inputs = build_saved_stage.preparation_inputs
        provider_record = build_saved_stage.provider_record
        recheck_host_graph_artifacts = build_saved_stage.recheck_host_graph_artifacts
        receipt_borrowed_lifetime = build_saved_stage.receipt_borrowed_lifetime
        recheck_storage = build_saved_stage.recheck_storage
        revalidate_host_provider = build_saved_stage.revalidate_host_provider
        require_observation_selection = build_saved_stage.require_observation_selection
        if run != "none":
            launch_saved_stage = launch_saved(
                backend=backend,
                built=built,
                device=device,
                elf=elf,
                elf_sha=elf_sha,
                extension_backend=_spike_extension_backend,
                full=full,
                marches=marches,
                output_path=output_path,
                prepare=prepare,
                receipt=receipt,
                revalidate=revalidate,
                run=run,
                selected=selected,
                selection=selection,
                target=target,
                timeout_s=timeout_s,
            )
            console = launch_saved_stage.console
            if full:
                if readonly_input_indices and built.get("readonly_input_observation") != readonly_record:
                    raise BaremetalModelError("original readonly input observation changed during execution")
                receipt["output"].update(
                    _judge_full_readback(console, built, capture_path, references, tolerance, output_path)
                )
            else:
                _judge_prefix(console, built, golden, output_path, receipt)
            if recomputation_check is not None:
                recomputation_check(callbacks=False)
                source_recomputation.retain_execution(
                    tuple(
                        Path(receipt["output"][name])
                        for name in ("console", "outputs_npz")
                        if name in receipt["output"]
                    )
                )
                recomputation_check()
            observe_profiles(
                console=console,
                full=full,
                group_profile=group_profile,
                op_profile=op_profile,
                output_path=output_path,
                profile_table=profile_table,
                receipt=receipt,
            )
        revalidate_saved(
            built=built,
            capture_path=capture_path,
            capture_tree=capture_tree,
            catalog_path=catalog_path,
            catalog_sha=catalog_sha,
            device=device,
            device_path=device_path,
            device_tree=device_tree,
            dts_path=dts_path,
            dts_sha=dts_sha,
            early_layout_artifacts=early_layout_artifacts,
            early_layout_policy=early_layout_policy,
            elf=elf,
            elf_sha=elf_sha,
            facts=facts,
            graph_artifacts=graph_artifacts,
            graph_optimizations=graph_optimizations,
            materialized_inputs=materialized_inputs,
            max_readonly_input_bytes=max_readonly_input_bytes,
            native=native,
            native_engine=_native_engine,
            op_profile=op_profile,
            output_path=output_path,
            package_path=package_path,
            package_tree=package_tree,
            preparation=preparation,
            preparation_inputs=preparation_inputs,
            preparation_sha=preparation_sha,
            profile_table=profile_table,
            profile_table_sha=profile_table_sha,
            provider_record=provider_record,
            readonly_input_indices=readonly_input_indices,
            readonly_record=readonly_record,
            receipt=receipt,
            receipt_borrowed_lifetime=receipt_borrowed_lifetime,
            recheck_host_graph_artifacts=recheck_host_graph_artifacts,
            recheck_storage=recheck_storage,
            recomputation_check=recomputation_check,
            require_observation_selection=require_observation_selection,
            revalidate_host_provider=revalidate_host_provider,
            rtl_facts=rtl_facts,
            run=run,
            selected=selected,
            selection=selection,
            shared_source_lowerings=shared_source_lowerings,
            sidecar=sidecar,
            source_choices=source_choices,
            source_recomputation=source_recomputation,
            storage_artifacts=storage_artifacts,
            storage_policy=storage_policy,
            target=target,
        )
        receipt["status"] = "compiled" if run == "none" else "verified_complete_output"
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        (output_path / "baremetal_model.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    (output_path / "baremetal_model.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if storage_policy is not None:
        from merlin.compiler.lower.model.borrowed_arena import retain_borrowed_lifetime

        return retain_borrowed_lifetime(receipt, receipt_borrowed_lifetime(built))
    return receipt
