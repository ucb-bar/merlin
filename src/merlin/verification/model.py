"""Independent saved-model references, complete output comparison and input/artifact custody."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from merlin.compiler.build.modeloptions import BaremetalModelError, _sha
from merlin.development.targetgen.application_inventory import verify_capture_receipt
from merlin.runtime import modelinputs as MI


def _saved_reference(
    capture: Path,
    name: str | None,
    *,
    execution: bool,
    materialized_inputs=None,
) -> np.ndarray | None:
    verification = (
        {"status": "verified_materialized"}
        if materialized_inputs is not None and materialized_inputs.verify(capture)
        else verify_capture_receipt(capture / "model.mlir")
    )
    if verification.get("status") != "verified_materialized":
        raise BaremetalModelError(f"saved capture receipt is not byte-verified: {verification.get('errors')}")
    for required in ("inputs.npz", "input_order.json", "golden.npy"):
        path = capture / required
        if not path.is_file() or path.is_symlink():
            raise BaremetalModelError(f"saved capture lacks a safe {required}")
    if not execution:
        return None
    if name is None or Path(name).name != name or not name.endswith(".npy"):
        raise BaremetalModelError("execution requires one explicit in-bundle .npy reference_file")
    if (capture / "goldens.npz").exists() or (capture / "output_order.json").exists():
        raise BaremetalModelError("multi-output captures have no complete bare-metal OUT gate")
    reference = capture / name
    recorded = (
        materialized_inputs.verify(capture)
        if materialized_inputs is not None
        else json.loads((capture / "capture_receipt.json").read_text()).get("artifacts") or {}
    )
    if not reference.is_file() or reference.is_symlink() or (recorded.get(name) or {}).get("sha256") != _sha(reference):
        raise BaremetalModelError(f"reference {name!r} is absent or not bound by the capture receipt")
    golden = np.load(reference, allow_pickle=False)
    if golden.dtype != np.float32 or golden.size < 1 or not np.isfinite(golden).all():
        raise BaremetalModelError("execution reference must be a nonempty finite float32 array")
    if golden.size > 4096:
        raise BaremetalModelError("complete OUT observability is limited to 4096 elements; use run='none'")
    return golden.reshape(-1)


def _full_references(
    capture: Path,
    reference_file: str | None,
    *,
    materialized_inputs=None,
) -> list[tuple[str, np.ndarray | None]]:
    """``[(name, reference or None)]`` in forward-result order for a full-readback execution.

    A capture that states every result (``output_order.json`` with one array each in ``goldens.npz``)
    references them all. An explicit ``integer-reference.json`` selects every typed result of
    the captured quantized program, bound to both its receipt and actual forward ABI.
    Otherwise result zero is held to the one explicit ``reference_file`` and any
    further result is recorded as having no capture reference (a caller may hold it to an independent
    host execution of the same program). Every file must be bound by the capture receipt or lie in the
    capture tree whose digest this compilation records."""
    from merlin.compiler.lower.model.c_runtime import _out_specs

    recorded = (
        materialized_inputs.verify(capture)
        if materialized_inputs is not None
        else json.loads((capture / "capture_receipt.json").read_text()).get("artifacts") or {}
    )

    def present(name: str, *, receipt_bound: bool) -> Path:
        path = capture / name
        if not path.is_file() or path.is_symlink():
            raise BaremetalModelError(f"reference {name!r} is absent from the capture")
        if receipt_bound and (recorded.get(name) or {}).get("sha256") != _sha(path):
            raise BaremetalModelError(f"reference {name!r} is not bound by the capture receipt")
        return path

    result_specs = _out_specs(capture / "model.mlir")
    results = len(result_specs)
    if (capture / "output_order.json").exists() or (capture / "goldens.npz").exists():
        names = [str(x) for x in json.loads(present("output_order.json", receipt_bound=False).read_text())]
        archive = np.load(present("goldens.npz", receipt_bound=False), allow_pickle=False)
        if len(names) != results or len(set(names)) != len(names) or set(names) != set(archive.files):
            raise BaremetalModelError("output_order.json and goldens.npz disagree with the forward's results")
        return [(name, np.asarray(archive[name])) for name in names]
    if reference_file == "integer-reference.json":
        from merlin.development.targetgen.materialized_reference import complete_outputs

        metadata = json.loads(present("meta.json", receipt_bound=True).read_text())
        actual_abi = [{"shape": shape, "dtype": dtype} for shape, dtype in result_specs]
        if metadata.get("output_abi") != actual_abi:
            raise BaremetalModelError("complete reference ABI differs from the actual forward results")
        try:
            outputs = complete_outputs(metadata, present(reference_file, receipt_bound=True).read_bytes())
        except ValueError as exc:
            raise BaremetalModelError(f"invalid complete quantized-program reference: {exc}") from exc
        integers = {"i1": np.bool_, "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64}
        return [
            (f"out{i}", np.asarray(value, dtype=integers.get(row["dtype"], np.float32)))
            for i, (value, row) in enumerate(zip(outputs, actual_abi, strict=True))
        ]
    if reference_file is None or Path(reference_file).name != reference_file or not reference_file.endswith(".npy"):
        raise BaremetalModelError("execution needs one explicit in-bundle .npy reference_file for result zero")
    first = np.load(present(reference_file, receipt_bound=True), allow_pickle=False)
    return [(Path(reference_file).stem, first), *((f"out{i}", None) for i in range(1, results))]


def _agreement(value: np.ndarray, reference: np.ndarray, tolerance: dict[str, float] | None) -> dict[str, Any]:
    """Exact (bit for bit, or integer equality) without a tolerance; element-wise atol/rtol with one."""
    reference = np.asarray(reference)
    if value.size != reference.size:
        return {"passed": False, "note": f"printed {value.size} of {reference.size} elements"}
    flat, ref = value.reshape(-1), reference.reshape(-1)
    if ref.dtype.kind in "iub":
        if flat.dtype.kind not in "iub":
            return {"passed": False, "note": "integer references require exact integer output"}
        differ = int(np.count_nonzero(flat != ref))
        return {"passed": differ == 0, "mismatched_elements": differ, "of": int(ref.size)}
    if tolerance is None:
        if flat.dtype == np.float32 and ref.dtype == np.float32:
            differ = int(np.count_nonzero(flat.view(np.uint32) != ref.view(np.uint32)))
        elif flat.dtype.kind in "iub" and ref.dtype.kind in "iub":
            differ = int(np.count_nonzero(flat.astype(np.int64) != ref.astype(np.int64)))
        else:
            return {"passed": False, "note": f"{flat.dtype} cannot be exact against {ref.dtype}"}
        return {"passed": differ == 0, "mismatched_elements": differ, "of": int(ref.size)}
    from merlin.analysis.perf.float_accuracy import compare

    agreement = compare(flat.astype(np.float64), ref.astype(np.float64), tolerance["atol"], tolerance["rtol"])
    return {**agreement, "passed": agreement["within"] == agreement["of"]}


def _judge_prefix(console: str, built: dict, golden: np.ndarray, output: Path, receipt: dict) -> None:
    """The historical one-output protocol: the printed prefix, bit for bit, against the reference."""
    from merlin.runtime.backends.spike_model import parse_console

    if isinstance(console, bytes):
        console = console.decode("utf-8", errors="replace")
    console_path = output / "console.log"
    console_path.write_text(console, encoding="utf-8")
    receipt["output"].update({"console": str(console_path), "console_sha256": _sha(console_path)})
    observed = parse_console(console)
    if not isinstance(observed, dict):
        raise BaremetalModelError("whole-model console parser returned no structured result")
    if observed.get("metrics", {}).get("build_hash") != built.get("build_hash"):
        raise BaremetalModelError("whole-model console does not identify the linked build")
    if observed.get("metrics", {}).get("memref_rank_mismatch") != 0:
        raise BaremetalModelError("whole-model console lacks a clean memref-rank diagnostic")
    values = np.asarray(observed["outputs"], dtype=np.float32).reshape(-1)
    if values.size != golden.size or not np.isfinite(values).all():
        raise BaremetalModelError("whole-model OUT is partial or nonfinite")
    if not np.array_equal(values.view(np.uint32), golden.view(np.uint32)):
        mismatched = int(np.count_nonzero(values.view(np.uint32) != golden.view(np.uint32)))
        raise BaremetalModelError(f"whole-model output differs from declared reference in {mismatched} elements")
    receipt["output"].update(
        {"elements": int(golden.size), "mismatched_elements": 0, "metrics": observed.get("metrics") or {}}
    )


def console_text_metrics(text: str) -> dict[str, str]:
    """``METRIC <name> <value>`` lines as strings (a build hash is hex, not an integer)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "METRIC":
            out[parts[1]] = parts[2]
    return out


def read_full_outputs(
    console: bytes, capture: Path, built_hash: str | None, *, readonly_input_observation=None
) -> tuple[list[np.ndarray], dict]:
    """Every forward result decoded from a full-readback console, plus its text metrics.

    Refuses a console that does not identify ``built_hash`` or reports a memref-rank refusal."""
    values, metrics, _observed = _read_full_values(console, capture, built_hash, readonly_input_observation)
    return values, metrics


def _read_full_values(console, capture, built_hash, observation):
    from merlin.compiler.lower.model.c_runtime import _out_specs
    from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details
    from merlin.runtime.whole_model_readback import decode_outputs

    if not isinstance(console, bytes):
        raise BaremetalModelError("a full-readback console must be read as raw bytes")
    outputs, _numeric, frames = parse_binary_console_details(console)
    observed = None
    if observation is not None:
        from merlin.runtime.input_observation import verify_observed_inputs

        outputs, frames, observed = verify_observed_inputs(observation, outputs, frames)
    metrics = console_text_metrics(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
    if built_hash is not None and metrics.get("build_hash") != built_hash:
        raise BaremetalModelError("whole-model console does not identify the linked build")
    if metrics.get("memref_rank_mismatch") != "0":
        raise BaremetalModelError("whole-model console lacks a clean memref-rank diagnostic")
    values = decode_outputs(outputs, frames, _out_specs(capture / "model.mlir"))
    return values, metrics, observed


def _judge_full_readback(console, built, capture, references, tolerance, output: Path) -> dict[str, Any]:
    console_path = output / "console.bin"
    console_path.write_bytes(console if isinstance(console, bytes) else console.encode("utf-8"))
    values, metrics, observed = _read_full_values(
        console, capture, built.get("build_hash"), built.get("readonly_input_observation")
    )
    np.savez(output / "outputs.npz", **{f"out{i}": value for i, value in enumerate(values)})
    record: dict[str, Any] = {
        "console": str(console_path),
        "console_sha256": _sha(console_path),
        "outputs_npz": str(output / "outputs.npz"),
        "metrics": metrics,
    }
    if observed is not None:
        record["readonly_input_observation"] = observed
    if references is None:
        return record
    if len(references) != len(values):
        raise BaremetalModelError(f"the forward has {len(values)} results and the capture states {len(references)}")
    per_output = []
    for (name, reference), value in zip(references, values, strict=True):
        if reference is None:
            per_output.append({"name": name, "passed": True, "status": "no_capture_reference"})
            continue
        per_output.append({"name": name, "status": "compared", **_agreement(value, reference, tolerance)})
    record["per_output"] = per_output
    failed = [row["name"] for row in per_output if not row["passed"]]
    if failed:
        raise BaremetalModelError(f"whole-model results differ from the capture's references: {failed}")
    return record


def revalidate_saved(
    *,
    built,
    capture_path,
    capture_tree,
    catalog_path,
    catalog_sha,
    device,
    device_path,
    device_tree,
    dts_path,
    dts_sha,
    early_layout_artifacts,
    early_layout_policy,
    elf,
    elf_sha,
    facts,
    graph_artifacts,
    graph_optimizations,
    materialized_inputs,
    max_readonly_input_bytes,
    native,
    native_engine,
    op_profile,
    output_path,
    package_path,
    package_tree,
    preparation,
    preparation_inputs,
    preparation_sha,
    profile_table,
    profile_table_sha,
    provider_record,
    readonly_input_indices,
    readonly_record,
    receipt,
    receipt_borrowed_lifetime,
    recheck_host_graph_artifacts,
    recheck_storage,
    recomputation_check,
    require_observation_selection,
    revalidate_host_provider,
    rtl_facts,
    run,
    selected,
    selection,
    shared_source_lowerings,
    sidecar,
    source_choices,
    source_recomputation,
    storage_artifacts,
    storage_policy,
    target,
):
    """Reopen original inputs, code, policies and retained execution artifacts before success."""
    if (
        MI.strict_tree_sha256(capture_path) != capture_tree
        or MI.strict_tree_sha256(package_path) != package_tree
        or _sha(catalog_path) != catalog_sha
        or _sha(dts_path) != dts_sha
        or _sha(elf) != elf_sha
    ):
        raise BaremetalModelError("an input or linked ELF changed during compilation/execution")
    if device is not None and MI.strict_tree_sha256(device_path) != device_tree:
        raise BaremetalModelError("selected device package changed during compilation/execution")
    if sidecar.is_file() and _sha(sidecar) != receipt["output"]["device_sidecar"]["sha256"]:
        raise BaremetalModelError("device dispatch sidecar changed during compilation/execution")
    if op_profile and (profile_table.is_symlink() or _sha(profile_table) != profile_table_sha):
        raise BaremetalModelError("operation profile table changed during compilation/execution")
    if selection is not None and native_engine(target, run, facts)[1] != selection:
        raise BaremetalModelError("selected native RTL engine changed during execution")
    if native and MI.selected_firrtl(rtl_facts, target=target, config=selected.rtl_sim_config) != facts:
        raise BaremetalModelError("selected RTL facts or source FIRRTL changed during execution")
    if materialized_inputs is not None:
        materialized_inputs.verify(capture_path)
    if readonly_input_indices:
        if built.get("readonly_input_observation") != readonly_record:
            raise BaremetalModelError("original readonly input observation changed during execution")
        require_observation_selection(
            readonly_record, indices=readonly_input_indices, max_bytes=max_readonly_input_bytes
        )
    if graph_optimizations is not None:
        if recheck_host_graph_artifacts(output_path / "build" / "lower", graph_optimizations) != graph_artifacts:
            raise BaremetalModelError("host graph optimization artifacts changed during execution")
    if early_layout_policy is not None:
        if (
            getattr(device, "pre_group_layout", None) is not early_layout_policy
            or recheck_host_graph_artifacts(output_path / "build" / "pre_group_layout", early_layout_policy)
            != early_layout_artifacts
        ):
            raise BaremetalModelError("pre-group layout selection or artifacts changed during execution")
    if storage_policy is not None:
        if (
            getattr(device, "intermediate_storage", None) is not storage_policy
            or recheck_storage(
                output_path / "build" / "intermediate_storage",
                storage_policy,
                borrowed_lifetime=receipt_borrowed_lifetime(built),
            )
            != storage_artifacts
        ):
            raise BaremetalModelError("intermediate storage selection or artifacts changed during execution")
    if shared_source_lowerings:
        for value in shared_source_lowerings:
            value.selected_contracts.revalidate()
        if preparation.is_symlink() or not preparation.is_file() or _sha(preparation) != preparation_sha:
            raise BaremetalModelError("shared source preparation receipt changed during execution")
        if any(path.is_symlink() or not path.is_file() or _sha(path) != digest for path, digest in preparation_inputs):
            raise BaremetalModelError("shared source preparation input or selected IR changed during execution")
        revalidate_host_provider(provider_record)
    source_choices.revalidate()
    if recomputation_check is not None:
        recomputation_check()
        if run != "none":
            source_recomputation.executed({"run": run, "output": receipt["output"]})
            recomputation_check()
            receipt["output"]["source_recomputation"] = source_recomputation.record()
            recomputation_check()
