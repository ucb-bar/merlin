"""Compare generated regions, optionally feeding actual outputs sequentially."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from whole_model import file_hash, verify_fixture_boundaries, execution_backend_arguments, authenticate_initial_inputs, require_stateless_manifest


def verify_stage_inputs(source, links, oracle_inputs, oracle_metadata, compiled_inputs,
                        initial_activation=None, initial_receipt=None):
    """Authenticate oracle-generated seams and preserve their raw input words."""
    if (initial_activation is None) != (initial_receipt is None):
        raise ValueError("initial embedding words require their independent authentication")
    if initial_receipt is not None and (initial_receipt.get("status") != "PASS"
            or initial_receipt.get("independent_raw_gather_verified") is not True
            or initial_receipt.get("other_preparation_boundaries_token_independent") is not True
            or initial_receipt.get("weight_cast_or_requantization") is not False):
        raise ValueError("initial embedding override lacks independent unchanged-source authentication")
    linked = {item["input_index"]: item for item in links}
    overrides, rows = {}, []
    for index, boundary in enumerate(source["source_boundaries"]):
        key = f"input_{index}"
        raw = np.ascontiguousarray(oracle_inputs[key])
        metadata = oracle_metadata[key]
        digest = sha256(raw.tobytes()).hexdigest()
        if digest != metadata["sha256"] or list(raw.shape) != metadata["shape"]:
            raise ValueError(f"oracle runtime input changed: {key}")
        compiled = compiled_inputs[key]
        if raw.shape != compiled.shape or raw.dtype.itemsize != compiled.dtype.itemsize:
            raise ValueError(f"oracle runtime input ABI differs: {key}")
        row = {"input_index": index, "boundary": boundary, "sha256": digest}
        if index in linked:
            link = linked[index]
            origin = metadata["origin"]
            if origin.get("reference_stage") != link["region"] or origin.get("output_index") != link["output_index"]:
                raise ValueError(f"oracle runtime input producer differs: {key}")
            # A view changes the carrier type only, never any tensor byte.
            overrides[key] = raw.view(compiled.dtype)
            row.update(link)
        elif source.get("kind") == "layer0" and index == 0 and initial_activation is not None:
            origin = metadata["origin"]
            if (boundary != "embedding" or initial_receipt is None
                    or digest != initial_receipt["input_0"]["sha256"]
                    or digest != sha256(initial_activation.tobytes()).hexdigest()
                    or origin.get("preparation_sha256") != initial_receipt["preparation_sha256"]
                    or origin.get("tokens_sha256") != initial_receipt["tokens_sha256"]):
                raise ValueError("oracle layer0 embedding differs from the authenticated source gather")
            overrides[key] = np.ascontiguousarray(initial_activation).view(compiled.dtype)
            row.update(preparation_sha256=initial_receipt["preparation_sha256"],
                       tokens_sha256=initial_receipt["tokens_sha256"], initial_source_embedding=True)
        elif digest != source["tensors"][key]["sha256"]:
            raise ValueError(f"oracle changed immutable preparation fixture: {key}")
        rows.append(row)
    return overrides, rows


def generated_stage_inputs(source, links, captured, execution, records,
                           initial_activation=None, initial_receipt=None):
    """Read generated predecessors only; oracle runtime inputs are never opened."""
    if (initial_activation is None) != (initial_receipt is None):
        raise ValueError("initial embedding requires authentication")
    if initial_receipt is not None and (initial_receipt.get("status") != "PASS"
            or initial_receipt.get("independent_raw_gather_verified") is not True
            or initial_receipt.get("other_preparation_boundaries_token_independent") is not True
            or initial_receipt.get("weight_cast_or_requantization") is not False):
        raise ValueError("initial embedding lacks unchanged-source authentication")
    linked = {item["input_index"]: item for item in links}
    overrides, rows = {}, []
    for index, boundary in enumerate(source["source_boundaries"]):
        key = f"input_{index}"
        compiled = captured[key]
        row = {"input_index": index, "boundary": boundary}
        if index in linked:
            link = linked[index]
            previous = records[link["region"]]
            if previous["mismatches"]:
                raise ValueError("generated predecessor differs from reference")
            path = execution / link["region"] / "actual.npz"
            with np.load(path, allow_pickle=False) as actual:
                raw = np.ascontiguousarray(actual[f"output_{link['output_index']}"])
            published = previous["execution"]["outputs"][link["output_index"]]
            if (sha256(raw.tobytes()).hexdigest() != published["sha256"]
                    or list(raw.shape) != published["shape"] or raw.dtype.str != published["dtype"]):
                raise ValueError("generated predecessor publication changed")
            row.update(link, input_origin="generated_actual", predecessor_actual_sha256=file_hash(path),
                       predecessor_completed_monotonic_ns=previous["completed_monotonic_ns"])
            overrides[key] = raw
        elif source.get("kind") == "layer0" and index == 0 and initial_activation is not None:
            raw = np.ascontiguousarray(initial_activation)
            if sha256(raw.tobytes()).hexdigest() != initial_receipt["input_0"]["sha256"]:
                raise ValueError("authenticated embedding changed")
            overrides[key] = raw
            row.update(initial_source_embedding=True, input_origin="authenticated_source_gather",
                       tokens_sha256=initial_receipt["tokens_sha256"],
                       preparation_sha256=initial_receipt["preparation_sha256"])
        else:
            raw = compiled
            if sha256(raw.tobytes()).hexdigest() != source["tensors"][key]["sha256"]:
                raise ValueError("immutable preparation fixture changed")
            row["input_origin"] = "immutable_preparation_fixture"
        if raw.shape != compiled.shape or raw.dtype.itemsize != compiled.dtype.itemsize:
            raise ValueError("generated input ABI differs")
        if key in overrides:
            overrides[key] = raw.view(compiled.dtype)
        row["sha256"] = sha256(raw.tobytes()).hexdigest()
        rows.append(row)
    return overrides, rows


def close_seams(order, links, records):
    closure = []
    for kind in order:
        record = records[kind]
        if record["mismatches"]:
            raise ValueError(f"generated stage differs from independent oracle: {kind}")
        by_index = {row["input_index"]: row for row in record["inputs"]}
        for link in links[kind]:
            previous = records[link["region"]]["execution"]["outputs"][link["output_index"]]
            incoming = by_index[link["input_index"]]
            if previous["sha256"] != incoming["sha256"]:
                raise ValueError(f"generated seam does not close: {kind}")
            if incoming.get("input_origin") == "generated_actual":
                if records[link["region"]]["completed_monotonic_ns"] >= record["started_monotonic_ns"]:
                    raise ValueError("generated predecessor did not complete before consumer started")
            closure.append({"consumer": kind, **link, "sha256": previous["sha256"]})
    return closure


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--adapter-export", type=Path, help="authenticate native capsules with atlas_fp8")
    parser.add_argument("--expected-contract-sha256", default="", help="contract SHA-256 from the source receipt")
    parser.add_argument("--expected-policy-sha256", default="", help="policy SHA-256 from the source receipt")
    parser.add_argument("--sequential", action="store_true", help="consume actual generated predecessors in temporal order")
    parser.add_argument("--tools-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--compiler-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--max-instructions", type=int, default=1000000)
    parser.add_argument("--backend", choices=("functional", "qualified_native"), default="functional")
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--memory-qualification", type=Path)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--initial-inputs", type=Path)
    parser.add_argument("--initial-preparation", type=Path)
    parser.add_argument("--initial-tokens", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--stop-after", default="head")
    parser.add_argument("--oracle-timeout-seconds", type=int, default=43200)
    parser.add_argument("--softmax-mask-policy", choices=("bf16-minfinite-zero-bounded32-nonempty-v1",
        "bf16-minfinite-zero-rowspan80-nonempty-v1"), default="bf16-minfinite-zero-rowspan80-nonempty-v1")
    parser.add_argument("--allocation-timeout-ms", type=int, default=1000)
    parser.add_argument("--merge-adjacent-device-regions", action="store_true",
                        help="Opt in to Atlas checked adjacent device-region alternatives")
    parser.add_argument("--io-memory-bytes", type=int, default=67108864)
    parser.add_argument("--application-memory-bytes", type=int, default=536870912)
    from precision_options import add_reduction_policy_argument, verify_reduction_policy

    add_reduction_policy_argument(parser)
    args = parser.parse_args()
    selection = (args.adapter_export, args.expected_contract_sha256, args.expected_policy_sha256)
    if any(selection) and not all(selection):
        parser.error("--adapter-export and both expected source SHA-256 values require each other")
    if args.allocation_timeout_ms <= 0:
        parser.error("allocation timeout must be positive")
    try:
        backend_arguments = execution_backend_arguments(args.backend, args.native_qualification,
            args.memory_qualification, args.native_cache, max_instructions=args.max_instructions)
    except ValueError as error:
        parser.error(str(error))
    regions, compiled, execution = (path.resolve() for path in
        (args.regions, args.compiled, args.execution))
    oracle = args.oracle.resolve() if args.oracle else None
    if not args.compile_only and oracle is None:
        parser.error("--oracle is required for generated execution composition")
    tools, atlas = args.tools_root.resolve(), args.atlas_root.resolve()
    manifest = json.loads((regions / "manifest.json").read_text())
    require_stateless_manifest(manifest)
    whole = manifest["whole_model"]
    order = whole["regions"]
    if order != [f"layer{index}" for index in range(22)] + ["norm", "head"]:
        raise ValueError("whole-model manifest does not contain all 24 ordered regions")
    if args.stop_after not in order:
        parser.error("stop-after must name a manifest region")
    selected = order[:order.index(args.stop_after) + 1]
    links = {item["kind"]: item["inputs"] for item in whole["links"]}
    sources = {kind: json.loads((regions / kind / "source.json").read_text()) for kind in order}
    verify_fixture_boundaries(sources, links)
    initial_activation, initial_receipt = authenticate_initial_inputs(args, manifest, sources["layer0"])
    execution.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "atlas.generated_composition_closure.v1", "status": "RUNNING",
               "evidence": ("sequential generated stages consuming actual predecessor publications" if args.sequential else
                            "ordered generated stages with independent oracle inputs and closed generated seams"),
               "temporal_sequential_execution": args.sequential,
               "oracle_chain_inputs": not args.sequential,
               "source_model_sha256": manifest["source_model_sha256"],
               "manifest_sha256": file_hash(regions / "manifest.json"),
               "driver_sha256": file_hash(Path(__file__)),
               "requested_regions": selected, "regions": [],
               "requested_allocation_timeout_ms": args.allocation_timeout_ms,
               "requested_merge_adjacent_device_regions": args.merge_adjacent_device_regions,
               "requested_reduction_policy": args.reduction_policy,
               "validation_backend": args.backend, "functional_instruction_limit": args.max_instructions,
               "native_qualification_sha256": file_hash(args.native_qualification) if args.native_qualification else None,
               "memory_qualification_sha256": file_hash(args.memory_qualification) if args.memory_qualification else None,
               "native_cache": str(args.native_cache.resolve()) if args.native_cache else None,
               "embedding_execution": "authenticated_source_word_gather" if initial_receipt else whole["embedding_execution"],
               "initial_embedding_override": initial_receipt,
               "framework_activation_substitution": False, "quality": "PENDING"}
    def save():
        (execution / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    model_env = dict(os.environ)
    imports = [str(args.model_root.resolve()), str(tools), str(atlas)]
    if args.reference_root:
        imports.insert(1, str(args.reference_root.resolve()))
    if args.backend == "qualified_native":
        merlin = find_spec("merlin")
        if merlin is None or not merlin.submodule_search_locations:
            raise ValueError("qualified_native requires the Merlin runtime source package")
        imports.insert(0, str(Path(next(iter(merlin.submodule_search_locations))).parent))
    model_env["PYTHONPATH"] = os.pathsep.join(imports)

    def run_region(kind):
        started = time.monotonic()
        started_ns = time.monotonic_ns()
        destination = compiled / kind
        source = sources[kind]
        if not (destination / "compiled.json").exists():
            command = [str(args.compiler_python), str(tools / "compile_capsule.py"),
                       "--capsule", str(regions / kind), "--output", str(destination),
                       "--atlas-root", str(atlas), "--external-parameters",
                           "--softmax-mask-policy", args.softmax_mask_policy,
                           "--reduction-policy", args.reduction_policy,
                       "--io-memory-bytes", str(args.io_memory_bytes),
                           "--allocation-timeout-ms", str(args.allocation_timeout_ms),
                       "--application-memory-bytes", str(args.application_memory_bytes)]
            if args.adapter_export is not None:
                command += ["--adapter-export", str(args.adapter_export.resolve(strict=True)),
                            "--expected-contract-sha256", args.expected_contract_sha256,
                            "--expected-policy-sha256", args.expected_policy_sha256]
            if args.merge_adjacent_device_regions:
                command.append("--merge-adjacent-device-regions")
            with (execution / f"{kind}-compile.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        compiled_record = json.loads((destination / "compiled.json").read_text())
        if compiled_record.get("merge_adjacent_device_regions", False) != args.merge_adjacent_device_regions:
            raise ValueError("compiled region uses a different adjacent-device selection")
        if (compiled_record["source_model_sha256"] != source["model_sha256"]
                or compiled_record["kind"] != kind or not compiled_record["constant_bytes_identical"]):
            raise ValueError(f"compiled region differs from selected source: {kind}")
        precision = json.loads((destination / "precision.json").read_text())
        verify_reduction_policy(kind, precision, compiled_record, args.reduction_policy)
        if kind.startswith("layer") and precision.get("already_quantized_mask") != args.softmax_mask_policy:
            raise ValueError(f"compiled region uses a different selected mask policy: {kind}")
        if args.compile_only:
            return {"kind": kind, "status": "COMPILED", "source_model_sha256": source["model_sha256"],
                    "compiled_receipt_sha256": file_hash(destination / "compiled.json"),
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
        oracle_stage = oracle / kind
        deadline = time.monotonic() + args.oracle_timeout_seconds
        while True:
            try:
                global_reference = json.loads((oracle / "reference.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                global_reference = {}
            if global_reference.get("status") in {"REFUSED", "FAILED"}:
                raise ValueError("independent oracle refused: " + str(global_reference.get("error")))
            try:
                reference = json.loads((oracle_stage / "reference.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                reference = None
            if reference is not None:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"independent stage reference did not arrive: {kind}")
            time.sleep(5)
        if (reference["status"] != "EXPECTED" or reference["source_model_sha256"] != source["model_sha256"]
                or not reference["constant_bytes_identical"]):
            raise ValueError(f"oracle stage source differs: {kind}")
        with np.load(destination / "tensors.npz", allow_pickle=False) as captured:
            if args.sequential:
                overrides, inputs = generated_stage_inputs(source, links[kind], captured,
                    execution, records, initial_activation, initial_receipt)
            else:
                with np.load(oracle_stage / "runtime_inputs.npz", allow_pickle=False) as incoming:
                    overrides, inputs = verify_stage_inputs(source, links[kind], incoming,
                        json.loads((oracle_stage / "inputs.json").read_text()), captured, initial_activation, initial_receipt)
        stage = execution / kind
        stage.mkdir()
        override_path = stage / ("generated-predecessor-inputs.npz" if args.sequential else "oracle-predecessor-inputs.npz")
        np.savez(override_path, **overrides)
        command = [str(args.model_python), str(tools / "execute_graph.py"),
                   "--compiled", str(destination), "--model-root", str(args.model_root.resolve()),
                   "--atlas-root", str(atlas), "--compact-trace",
                   "--input-overrides", str(override_path), "--execution-output", str(stage)]
        command.extend(backend_arguments)
        with (execution / f"{kind}-execute.log").open("w") as stream:
            subprocess.run(command, env=model_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        actual_execution = json.loads((stage / "execution.json").read_text())
        for row in inputs:
            if actual_execution["inputs"][row["input_index"]]["sha256"] != row["sha256"]:
                raise ValueError(f"generated stage did not consume authenticated oracle input: {kind}")
        mismatches, outputs = 0, []
        with np.load(stage / "actual.npz", allow_pickle=False) as actual, \
                np.load(oracle_stage / "expected.npz", allow_pickle=False) as expected:
            for key, metadata in reference["outputs"].items():
                lhs, rhs = actual[key], expected[key]
                if sha256(rhs.tobytes()).hexdigest() != metadata["sha256"]:
                    raise ValueError(f"independent expected output changed: {kind}")
                if lhs.shape != rhs.shape or lhs.dtype.itemsize != rhs.dtype.itemsize:
                    raise ValueError(f"generated output ABI differs: {kind}")
                different = lhs.reshape(-1).view(np.uint8).reshape(-1, lhs.dtype.itemsize) != \
                            rhs.reshape(-1).view(np.uint8).reshape(-1, rhs.dtype.itemsize)
                count = int(np.any(different, axis=1).sum())
                mismatches += count
                outputs.append({"key": key, "elements": lhs.size, "mismatches": count,
                                "sha256": sha256(lhs.tobytes()).hexdigest()})
        return {"kind": kind, "source_model_sha256": source["model_sha256"],
                "compiled_receipt_sha256": file_hash(destination / "compiled.json"),
                "reference_receipt_sha256": file_hash(oracle_stage / "reference.json"),
                "inputs": inputs, "execution": actual_execution, "outputs": outputs,
                "mismatches": mismatches, "started_monotonic_ns": started_ns,
                "completed_monotonic_ns": time.monotonic_ns(),
                "elapsed_seconds": round(time.monotonic() - started, 3)}

    records = {}
    try:
        for kind in selected:
            row = run_region(kind)
            records[kind] = row
            receipt["regions"] = [records[name] for name in selected if name in records]
            save()
            print(json.dumps({"stage": "REGION_COMPARED", "kind": kind,
                              "mismatches": row.get("mismatches", "PENDING"), "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        if args.compile_only:
            receipt.update(status="COMPILED", independent_comparison="PENDING")
        else:
            oracle_record = json.loads((oracle / "reference.json").read_text())
            if (oracle_record["status"] != "EXPECTED" or oracle_record["source_model_sha256"] != manifest["source_model_sha256"]
                    or not oracle_record.get("model_arithmetic_sources") or not oracle_record.get("validation_sha256")):
                raise ValueError("independent oracle sequence lacks complete authenticated qualification")
            receipt["oracle_receipt_sha256"] = file_hash(oracle / "reference.json")
            receipt["seam_closure"] = close_seams(selected, links, records)
            receipt.update(status="PASS", mismatches=0,
                           independent_comparison="all selected stage words and generated seam hashes exact")
    except Exception as error:
        receipt.update(status="FAILED", error=repr(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
