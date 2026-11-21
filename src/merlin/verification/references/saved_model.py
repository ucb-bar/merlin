"""Replay source preparation records and declared parameter payloads.

Successful checks establish source/artifact identity only. Numerical semantics,
application quality, actual execution and target qualification remain separate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.digest import sha256_file
from merlin.compiler.build.saved_model import BaremetalModelError, _sha
from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe
from merlin.compiler.quantization.contraction.constants import precision_plan_bytes, prepared_static_inputs
from merlin.development.targetgen.applications.inventory import verify_capture_receipt
from merlin.runtime.execution import model_inputs as MI
from merlin.runtime.execution.model_inputs import strict_tree_sha256

# ---- native_records ----------------------------------------------------------------------------


def check_ordered_records(
    artifact, manifest, *, source_request, k_source_request, ordered_mn_record=None, ordered_k_record=None
):
    """Require exact requested records and refuse undeclared decompositions."""
    rows = (
        ("ordered-k-tiling.json", ordered_k_record, "source_request_digest", k_source_request, "ordered-K"),
        ("ordered-mn-tiling.json", ordered_mn_record, "outer_source_request_digest", source_request, "ordered-M/N"),
    )
    for name, record, field, source, description in rows:
        path = artifact / name
        if record is None:
            if path.exists() or path.is_symlink() or field in manifest:
                raise ValueError(f"native artifact has an unrequested {description} partition")
        elif (
            manifest.get(field) != source.digest()
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != record
        ):
            raise ValueError(f"{description} partition differs from the original source")


def check_precision_plan(artifact, manifest, precision_plan, *, binding=None, staged_root=None):
    """Rebuild selected tile constants from the original staged parameter source."""
    from merlin.compiler.quantization.contraction.contract import DiagnosticMatmulTile

    path = artifact / "precision-plan.json"
    if precision_plan is None:
        if path.exists() or path.is_symlink():
            raise ValueError("native artifact has an undeclared source precision plan")
        return
    if path.is_symlink() or not path.is_file() or path.read_bytes() != precision_plan_bytes(precision_plan):
        raise ValueError("source precision plan differs from staged compiler inputs")
    if not isinstance(precision_plan, DiagnosticMatmulTile):
        return
    if binding is None or staged_root is None:
        raise ValueError("source-bound constants require their selected support and staged compiler inputs")
    prepared = prepared_static_inputs(
        binding,
        precision_plan,
        Path(staged_root),
        json.loads((artifact / "execution_plan.json").read_bytes()),
    )
    if manifest.get("constant_inputs") != [item.record() for item in prepared]:
        raise ValueError("source-bound constant inputs differ from staged compiler inputs")
    for item in prepared:
        if item.payload is not None:
            destination = artifact / item.record()["path"]
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.parent.is_symlink()
                or destination.read_bytes() != item.payload
            ):
                raise ValueError("source-bound constant payload differs from staged parameter")


# ---- native ------------------------------------------------------------------------------------
# Replay selected source/snapshot/declared native artifact identities.
#
# Successful replay establishes artifact identity only. It is not numerical,
# implementation, execution, device or hardware qualification.


def verify_report(args, *, load_request):
    """Reconstruct the original request and check the published artifact bytes."""
    from merlin.compiler.frontend.staged import (
        bind_physical_layouts,
        check_layout_artifact,
        load_selected_request,
        prepare_ordered_request,
    )
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError
    from merlin.compiler.semantic.target_binding import (
        NativeCompilationError,
        load_native_target_binding,
        open_native_snapshot,
        verify_native_publication,
    )

    artifact, status = Path(args.artifact), Path(args.out)
    safe_status = not (
        status.resolve().is_relative_to(artifact.resolve())
        or status.resolve().is_relative_to(Path(args.snapshot).resolve())
    )
    try:
        if not safe_status:
            raise ValueError("native verification status must be outside the artifact and snapshot directories")
        if getattr(args, "application", False) and getattr(args, "diagnostic_graph_policy", None):
            raise ValueError("diagnostic graph precision cannot be combined with --application")
        if getattr(args, "application", False) and getattr(args, "physical_layouts", None):
            raise ValueError("physical boundary layouts cannot be combined with --application")
        if getattr(args, "application", False) and any(
            getattr(args, name, None) is not None
            for name in (
                "diagnostic_region",
                "diagnostic_tile_index",
                "ordered_k_tile",
                "ordered_m_tile",
                "ordered_n_tile",
                "outer_tile_index",
            )
        ):
            raise ValueError("diagnostic tiles and ordered partitions cannot be combined with --application")
        if getattr(args, "application_state", None):
            raise ValueError("--application-state requires --application")
        if getattr(args, "application_parameters", "embedded") != "embedded":
            raise ValueError("--application-parameters requires --application")
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = load_native_target_binding(args.support) if args.support else None
        if binding is not None and snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        request, kind, _, precision = load_selected_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
            load_request=load_request,
        )
        request, layout_document = bind_physical_layouts(args, request, precision)
        ordered = prepare_ordered_request(args, binding, request, precision)
        request = ordered.request
        if request.target_identity != snapshot.profile.target_identity or request.lowering_policy != args.mode:
            raise ValueError("native verification source differs from selected target or compile mode")
        manifest_bytes = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("native artifact manifest must be an object")
        snapshot_digest = hashlib.sha256((snapshot.root / "manifest.json").read_bytes()).hexdigest()
        if manifest.get("native_snapshot_sha256") != snapshot_digest:
            raise ValueError("native artifact snapshot identity differs")
        if manifest.get("compile_mode") != args.mode:
            raise ValueError("native artifact compile mode differs")
        path = artifact / "row-softmax-tiling.json"
        if path.exists() or path.is_symlink() or "softmax_source_request_digest" in manifest:
            raise ValueError("native artifact has an unrequested decomposition or physical boundary layout")
        check_ordered_records(
            artifact,
            manifest,
            source_request=ordered.source,
            k_source_request=ordered.k_source,
            ordered_mn_record=ordered.mn_record,
            ordered_k_record=ordered.k_record,
        )
        check_layout_artifact(artifact, manifest, layout_document)
        verify_native_publication(
            artifact,
            manifest,
            engine=args.engine,
            request_digest=request.digest(),
            target_identity=snapshot.profile.target_identity,
        )
        check_precision_plan(
            artifact,
            manifest,
            precision,
            binding=binding,
            staged_root=getattr(args, "compile_inputs", None),
        )
        tracked = manifest.get("stage_artifacts")
        selected_path = artifact / "native-selection.json"
        if manifest.get("native_check_fingerprint") is not None:
            if not isinstance(tracked, dict) or "native-selection.json" not in tracked:
                raise ValueError("native selected graph is not tracked by the artifact manifest")
            selection = json.loads(selected_path.read_text())
            if not isinstance(selection, dict) or any(
                selection.get(key) != value
                for key, value in (
                    ("engine", args.engine),
                    ("request_digest", request.digest()),
                    ("target_identity", snapshot.profile.target_identity),
                    ("check_fingerprint", manifest.get("native_check_fingerprint")),
                )
            ):
                raise ValueError("native selected graph identity differs from artifact manifest")
        elif selected_path.exists():
            raise ValueError("native selected graph has no checked fingerprint")
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": "verified_artifacts",
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "source_kind": kind,
            "request_digest": request.digest(),
            "target_identity": snapshot.profile.target_identity,
            "binary_sha256": manifest.get("binary_sha256"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "checked_stage_artifacts": sorted(tracked) if isinstance(tracked, dict) else [],
            "selected_record_checked": manifest.get("native_check_fingerprint") is not None,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": (
                "unsupported_semantics"
                if isinstance(error, LinalgBridgeError)
                else error.status
                if isinstance(error, NativeCompilationError)
                else "compile_error"
            ),
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "reason": str(error),
        }
    return report, safe_status


# ---- completed_model ---------------------------------------------------------------------------
# Revalidate a completed link after an exact original input tree is restored.
#
# The failed compiler receipt remains failed. A separate record permits transport
# of its already completed objects; it supplies no live effects, new rewriting,
# execution, numerical verdict or instruction admission.

SCHEMA = "merlin.completed_model_link_revalidation.v1"
_FAILURE = "BaremetalModelError: selected device package changed during compilation/execution"


def _pin(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("completed link needs original direct file inputs")
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def _reopen(record):
    actual = _pin(record["path"])
    if actual["sha256"] != record["sha256"] or ("bytes" in record and actual["bytes"] != record["bytes"]):
        raise ValueError("completed link original file changed")
    return Path(actual["path"])


def _reopen_nested(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _reopen(value)
        for child in value.values():
            _reopen_nested(child)
    elif isinstance(value, list):
        for child in value:
            _reopen_nested(child)


def _original_policy(selected, receipt):
    from merlin.compiler.lower.host.graph import read_host_graph_policy

    software = _reopen(selected["software_spec"])
    if selected["target"] != receipt["inputs"]["target"]:
        raise ValueError("completed link permissions name another selected target")
    choices = {
        k: selected[k]
        for k in (
            "enabled",
            "minimum_transpose_bytes_saved",
            "maximum_added_operations",
            "physical_layout",
            "maximum_layout_assignments",
            "maximum_layout_permutations",
            "connected_layout",
        )
    }
    if "column_reduction_layout" in selected:
        choices["column_reduction_layout"] = selected["column_reduction_layout"]
    policy = read_host_graph_policy(software, target=selected["target"], entry=selected["context"]["entry"], **choices)
    policy.revalidate()
    if json.loads(json.dumps(policy.record())) != selected:
        raise ValueError("completed link selected original permissions changed")
    return policy


def _layout_permissions(receipt, recipe):
    """Replay the original pure source transform; restore no runtime effects."""
    selected = receipt["inputs"]["device"].get("pre_group_layout")
    products = receipt["output"].get("pre_group_layout")
    if selected is None:
        if products is not None:
            raise ValueError("completed link layout has no original selected permissions")
        return None
    if not isinstance(products, dict) or not products:
        raise ValueError("completed link lacks original layout replay products")
    policy = _original_policy(selected, receipt)
    for role, record in products.items():
        expected = recipe.get("preparation", {}).get("pre_group_layout_" + role)
        if expected is None or any(expected[k] != record[k] for k in ("path", "sha256")):
            raise ValueError("completed link layout replay differs from its completed recipe")
        _reopen(record)
    from merlin.compiler.lower.host.graph import recheck_host_graph_artifacts

    root = Path(receipt["output"]["elf"]).parent / "pre_group_layout"
    if recheck_host_graph_artifacts(root, policy) != products:
        raise ValueError("completed link original layout source or pure replay changed")
    return selected


def _storage_permissions(receipt, recipe):
    """Read permissions and the original successful, recipe-bound replay products.

    This consumes immutable compiled artifacts. It never reconstructs borrowed
    lifetime owners or invokes the allocator from serialized effects.
    """
    selected = receipt["inputs"]["device"].get("intermediate_storage")
    products = receipt["output"].get("intermediate_storage")
    if selected is None:
        if products is not None:
            raise ValueError("completed link storage has no original selected permissions")
        return None
    policy = _original_policy(selected, receipt)
    software = policy.software_path
    if not isinstance(products, dict):
        raise ValueError("completed link selected storage permissions changed")
    if set(products) != {"input", "selected", "report", "software_spec", "policy"}:
        raise ValueError("completed link lacks original storage replay products")
    for role, record in products.items():
        expected = recipe.get("preparation", {}).get("intermediate_storage_" + role)
        if expected is None or any(expected[k] != record[k] for k in ("path", "sha256")):
            raise ValueError("completed link storage replay differs from its completed recipe")
        _reopen(record)
    if json.loads(_reopen(products["policy"]).read_text()) != selected:
        raise ValueError("completed link stored permission selection changed")
    if _reopen(products["software_spec"]).read_bytes() != software.read_bytes():
        raise ValueError("completed link stored software permissions changed")
    return selected


def _check(failed_pin):
    receipt = json.loads(_reopen(failed_pin).read_text())
    inputs, output = receipt.get("inputs", {}), receipt.get("output", {})
    if (
        receipt.get("schema") != "merlin.baremetal-saved-model.v1"
        or receipt.get("status") != "failed"
        or receipt.get("failure") != _FAILURE
        or inputs.get("run") != "none"
        or not isinstance(inputs.get("device"), dict)
    ):
        raise ValueError("completed link revalidation requires the precise final package-tree refusal")
    unsupported = {
        "source_optimizations",
        "source_recomputation",
        "shared_source_lowerings",
        "materialized_inputs",
        "readonly_input_indices",
        "graph_optimizations",
        "pre_group_layout",
        "op_profile",
        "group_profile",
    }
    if any(inputs.get(key) for key in unsupported) or any(
        key in output
        for key in (
            "source_optimizations",
            "source_recomputation",
            "host_graph_optimization",
            "shared_source_preparation",
            "op_profile_table",
            "group_profile",
            "console",
            "outputs_npz",
        )
    ):
        raise ValueError("completed link has deferred checks outside the supported transport scope")
    device = inputs["device"]
    trees = {}
    for role, path, expected in (
        ("capture", inputs["capture"], inputs["capture_tree"]),
        ("host_package", inputs["package"], inputs["package_tree"]),
        ("device_package", device["package"], device["package_tree"]),
    ):
        actual = strict_tree_sha256(Path(path))
        if actual != expected:
            raise ValueError("completed link original input tree has not been restored exactly")
        trees[role] = dict(path=path, **actual)
    for path_key in ("board_catalog", "dts"):
        _reopen(dict(path=inputs[path_key], sha256=inputs[path_key + "_sha256"]))
    elf = _reopen(dict(path=output["elf"], sha256=output["elf_sha256"]))
    recipe_pin = output["compilation_recipe"]
    recipe = verify_completed_recipe(_reopen(recipe_pin), executable=elf, expected_recipe_sha256=recipe_pin["sha256"])
    _reopen_nested(output)
    permissions = _storage_permissions(receipt, recipe)
    layout_permissions = _layout_permissions(receipt, recipe)
    return (
        receipt,
        recipe,
        dict(
            restored_input_trees=trees,
            executable=_pin(elf),
            recipe=recipe_pin,
            selected_storage_permissions=permissions,
            **({"selected_layout_permissions": layout_permissions} if layout_permissions is not None else {}),
        ),
    )


def finalize(failed_receipt, destination, *, restoration_evidence):
    """Write a separate completed-object transport record after exact revalidation."""
    failed_pin, evidence = _pin(failed_receipt), _pin(restoration_evidence)
    _, _, checked = _check(failed_pin)
    record = dict(
        schema=SCHEMA,
        status="completed_link_inputs_revalidated",
        failed_receipt=failed_pin,
        restoration_evidence=evidence,
        implementation=_pin(Path(__file__).resolve()),
        **checked,
        scope="completed object transport only; original compiler failure retained; no live effects restored",
    )
    destination = Path(destination)
    with destination.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return record


def verify(record_pin):
    """Reopen all original inputs/products again before ordinary object transport."""
    record = json.loads(_reopen(record_pin).read_text())
    if record.get("schema") != SCHEMA or record.get("status") != "completed_link_inputs_revalidated":
        raise ValueError("not a completed link revalidation record")
    _reopen(record["restoration_evidence"])
    if record["implementation"] != _pin(Path(__file__).resolve()):
        raise ValueError("completed link revalidation implementation changed")
    receipt, recipe, checked = _check(record["failed_receipt"])
    if any(record.get(key) != value for key, value in checked.items()):
        raise ValueError("completed link original inputs or products changed")
    return receipt, recipe, [record[role] for role in ("failed_receipt", "restoration_evidence", "implementation")]


# ---- model -------------------------------------------------------------------------------------
# Independent saved-model references, complete output comparison and input/artifact custody.


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
    from merlin.compiler.lower.c.runtime import _out_specs

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
        from merlin.development.targetgen.applications.independent_references import complete_outputs

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
    from merlin.analysis.whole_model.grading.exactness import compare

    agreement = compare(flat.astype(np.float64), ref.astype(np.float64), tolerance["atol"], tolerance["rtol"])
    return {**agreement, "passed": agreement["within"] == agreement["of"]}


def _judge_prefix(console: str, built: dict, golden: np.ndarray, output: Path, receipt: dict) -> None:
    """The historical one-output protocol: the printed prefix, bit for bit, against the reference."""
    from merlin.runtime.backends.baremetal.image_link import parse_console

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
    from merlin.compiler.lower.c.runtime import _out_specs
    from merlin.runtime.harness.output_decoding import binary_console_diagnostics, parse_binary_console_details
    from merlin.runtime.harness.output_observation import decode_outputs

    if not isinstance(console, bytes):
        raise BaremetalModelError("a full-readback console must be read as raw bytes")
    outputs, _numeric, frames = parse_binary_console_details(console)
    observed = None
    if observation is not None:
        from merlin.runtime.harness.output_observation import verify_observed_inputs

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
