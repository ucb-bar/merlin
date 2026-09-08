#!/usr/bin/env python3
"""Build the fail-closed whole-capture hybrid schedule and bounded-chain witness."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent"
PLAN_ROOT = ROOT / "whole_capture_plan"
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import parse_verified  # noqa: E402
from mlir_oot.hybrid_runtime import build_hybrid_schedule  # noqa: E402
from mlir_oot.host_semantics import (  # noqa: E402
    HostSemanticLane,
    LayoutBridgeLane,
    array_sha256,
)
from mlir_oot.accelerator_semantics import (  # noqa: E402
    AcceleratorContractLane,
    contract_sha256,
)
from run_capture_partition import (  # noqa: E402
    PARTITIONS,
    _load_capture_values,
    _select_partition,
    sha256_bytes,
)


QUALIFIED = frozenset(binding["partition_id"] for binding in PARTITIONS.values())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_chain_witness() -> tuple[dict, set[str]]:
    """Replay the real host bridge and bind it to retained qualified RTL outputs."""
    binding = PARTITIONS["action_time_mlp_in"]
    partition = _select_partition(binding)
    activation, _, _, source, _ = _load_capture_values(partition, CAPTURE, binding)
    calibration_path = ROOT / binding["output_dir"] / "calibration.json"
    result_path = ROOT / binding["output_dir"] / "result.json"
    calibration = load(calibration_path)
    result = load(result_path)
    source_input = source["input"]
    saved_input = calibration["source_tensors"]["input"]
    actual_hash = sha256_bytes(activation.astype("<f4", copy=False).tobytes())
    if actual_hash != saved_input["device_chain_sha256"]:
        raise ValueError("replayed bounded host bridge differs from saved p0244 calibration")
    if not result.get("acceptance", {}).get("passed"):
        raise ValueError("p0244 retained qualification no longer passes")

    bridge_regions = list(source_input["host_bridge"]["capture_regions"])
    origin = partition["abi"]["inputs"][0]["origin"]
    bridge_regions.append(origin["region_id"])
    bridge_regions.extend(item["region_id"] for item in origin.get("bridges", []))
    bridge_regions = list(dict.fromkeys(bridge_regions))
    receipts = [load(ROOT / path) for path in result["raw_gsim_receipts"]]
    if not all(r.get("assertion_clean") and r.get("stderr_observation") == "empty"
               for r in receipts):
        raise ValueError("p0244 retained dispatch evidence is not assertion-clean")
    return ({
        "schema": "atlas_bounded_real_hybrid_chain_v1",
        "status": "host_replay_matches_retained_qualified_rtl_chain",
        "claim": (
            "fresh deterministic host-bridge replay joined to retained RTL evidence; "
            "device partitions were not rerun and this is not whole-model execution"
        ),
        "partition_path": ["atlas_p0243", "atlas_p0244"],
        "host_bridge_regions": bridge_regions,
        "host_bridge_region_count": len(bridge_regions),
        "host_bridge_f32_sha256": source_input["host_bridge"]["raw_sha256"],
        "p0244_activation_shape": list(activation.shape),
        "p0244_activation_f32_sha256": actual_hash,
        "saved_calibration": {
            "path": calibration_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(calibration_path),
        },
        "retained_results": [
            {
                "partition_id": "atlas_p0243",
                "path": source_input["predecessor"]["result"],
                "sha256": source_input["predecessor"]["result_sha256"],
                "device_output_sha256": source_input["predecessor"][
                    "device_output_sha256"
                ],
            },
            {
                "partition_id": "atlas_p0244",
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(result_path),
                "device_output_sha256": result["device_output"]["raw_sha256"],
                "dispatches": len(receipts),
                "cycles": result["cycles"],
            },
        ],
        "events": [
            "retained_assertion_clean_atlas_p0243",
            "device_to_host_dequantize",
            "execute_27_region_host_bridge",
            "host_to_device_quantize",
            "retained_assertion_clean_atlas_p0244_three_dispatches",
            "device_to_host_dequantize",
        ],
    }, set(bridge_regions))


def _execute_host_witness(
    lane: HostSemanticLane, region_ids: list[str], *, label: str, selection: str,
) -> dict:
    """Execute fresh values through selected real regions and prove replay stability."""
    first_values = lane.seed_external_values(region_ids)
    seeds = [
        {
            "ordinal": index,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
        for index, value in enumerate(first_values.values())
    ]
    first_outputs = []
    for region_id in region_ids:
        value = lane.execute(region_id, first_values)
        output_record = {
            "region_id": region_id,
            "semantic": lane.programs[region_id].semantic,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
            "finite": bool(value.dtype == np.bool_ or np.all(np.isfinite(value))),
        }
        if value.dtype == np.bool_:
            output_record["true_elements"] = int(np.count_nonzero(value))
        first_outputs.append(output_record)

    second_values = lane.seed_external_values(region_ids)
    second_hashes = []
    for region_id in region_ids:
        second_hashes.append(array_sha256(lane.execute(region_id, second_values)))
    first_hashes = [row["sha256"] for row in first_outputs]
    if second_hashes != first_hashes:
        raise ValueError("fresh host semantic chain is not deterministic")

    produced = set()
    dependency_edges = 0
    for region_id in region_ids:
        program = lane.programs[region_id]
        dependency_edges += sum(value in produced for value in program.input_values)
        produced.update(value for op in program.operations for value in op.results)
    return {
        "schema": "atlas_real_capture_host_semantic_chain_v1",
        "label": label,
        "status": "fresh_numeric_execution_exactly_replayed",
        "claim": "host semantic execution only; no device execution and not whole-model E2E",
        "selection": selection,
        "region_ids": region_ids,
        "semantics": [lane.programs[region_id].semantic for region_id in region_ids],
        "region_count": len(region_ids),
        "dependency_edges": dependency_edges,
        "fresh_input_rule": "deterministic ordinal-dependent nonzero coordinate pattern",
        "fresh_inputs": seeds,
        "outputs": first_outputs,
        "replay_hashes_equal": True,
    }


def generic_host_witnesses(workload) -> tuple[dict, list[dict]]:
    """Discover dependency chains covering each newly implemented scalar family."""
    lane = HostSemanticLane(workload)
    runs = lane.discover_contiguous_runs()
    selected = []
    requirements = [
        ("pow_reciprocal", {"pow", "elementwise"}),
        ("rsqrt_normalization", {"rsqrt"}),
        ("sigmoid_gate", {"sigmoid"}),
        ("trigonometric_fanout", {"sin", "cos"}),
    ]
    for label, required in requirements:
        run = next(row for row in runs if required <= set(row["semantics"]))
        selected.append(_execute_host_witness(
            lane,
            run["region_ids"],
            label=label,
            selection=(
                "first stable-ranked consecutive qualified run containing semantic set "
                + ",".join(sorted(required))
            ),
        ))

    def constructor_feeds_successor(run: dict, semantic: str) -> bool:
        constructor_outputs = set()
        for region_id in run["region_ids"]:
            program = lane.programs[region_id]
            if any(value in constructor_outputs for value in program.input_values):
                return True
            if program.semantic == semantic:
                constructor_outputs.add(program.output_value)
        return False

    for label, semantic in (("arange_dependency", "arange"), ("fill_dependency", "fill")):
        already_selected = {tuple(row["region_ids"]) for row in selected}
        run = next(
            row for row in runs
            if (semantic in row["semantics"]
                and constructor_feeds_successor(row, semantic)
                and tuple(row["region_ids"]) not in already_selected)
        )
        selected.append(_execute_host_witness(
            lane,
            run["region_ids"],
            label=label,
            selection=(
                "first stable-ranked consecutive qualified run where a captured "
                f"{semantic} output feeds a successor"
            ),
        ))

    reduction_requirements = [
        ("cumsum_reduce_mean", {"cumsum", "reduce_mean"}),
        ("masked_softmax", {"softmax"}),
        ("argmin_successor", {"aten_min_dim"}),
        ("reduce_sum_successor", {"reduce_sum"}),
    ]
    for label, required in reduction_requirements:
        already_selected = {tuple(row["region_ids"]) for row in selected}
        run = next(
            row for row in runs
            if required <= set(row["semantics"])
            and tuple(row["region_ids"]) not in already_selected
        )
        selected.append(_execute_host_witness(
            lane,
            run["region_ids"],
            label=label,
            selection=(
                "first stable-ranked consecutive qualified run containing reduction set "
                + ",".join(sorted(required))
            ),
        ))

    # Native layer-norm instances are isolated from other qualified host
    # regions by accelerator partitions.  Exercise the smallest real instance
    # without inventing a host-host dependency.
    layer_norm = min(
        (program for program in lane.programs.values() if program.semantic == "layer_norm"),
        key=lambda program: (
            sum(int(np.prod(shape, dtype=np.int64))
                for shape in program.signature["input_shapes"]),
            program.region_id,
        ),
    )
    selected.append(_execute_host_witness(
        lane,
        [layer_norm.region_id],
        label="layer_norm_standalone",
        selection="smallest qualified real layer norm; adjacent regions are accelerator partitions",
    ))

    movement_requirements = [
        ("static_split_slice", {"split", "slice"}),
        ("slice_scatter_successor", {"slice_scatter"}),
        ("concat_successor", {"cat"}),
        ("bitwise_reduction", {"bitwise", "cumsum"}),
        ("bucketize_chain", {"bucketize"}),
    ]
    for label, required in movement_requirements:
        run = next(
            row for row in runs
            if required <= set(row["semantics"])
        )
        selected.append(_execute_host_witness(
            lane,
            run["region_ids"],
            label=label,
            selection=(
                "first stable-ranked consecutive qualified run containing movement set "
                + ",".join(sorted(required))
            ),
        ))

    static_select_run = next(
        row for row in runs
        if any(
            lane.programs[region_id].semantic == "select"
            and lane.programs[region_id].signature["schema"]
            == "atlas_host_movement_signature_v1"
            for region_id in row["region_ids"]
        )
    )
    selected.append(_execute_host_witness(
        lane,
        static_select_run["region_ids"],
        label="static_select_slice_chain",
        selection="first stable-ranked run containing an exact static aten.select.int chain",
    ))

    # GELU instances are isolated by accelerator partitions in this capture.
    # Select the smallest real one by tensor extent and execute it standalone;
    # do not manufacture a false dependency chain around it.
    gelu = min(
        (program for program in lane.programs.values() if program.semantic == "gelu"),
        key=lambda program: (
            int(np.prod(program.signature["output_shape"], dtype=np.int64)),
            program.region_id,
        ),
    )
    selected.append(_execute_host_witness(
        lane,
        [gelu.region_id],
        label="gelu_standalone",
        selection="smallest qualified real GELU; capture has no adjacent qualified dependency",
    ))
    return selected[0], selected


def final_indexed_host_witnesses(workload) -> list[dict]:
    """Execute every formerly missing real region against an independent oracle."""
    lane = HostSemanticLane(workload)
    witnesses = []

    def record(label: str, programs: list, actual: np.ndarray, expected: np.ndarray,
               input_rule: str) -> None:
        if not np.array_equal(actual, expected):
            raise ValueError(f"independent numeric oracle failed for {label}")
        witnesses.append({
            "schema": "atlas_real_capture_indexed_numeric_witness_v1",
            "label": label,
            "status": "fresh_numeric_execution_matches_independent_oracle",
            "claim": "host semantic evidence only; not device or whole-model E2E execution",
            "region_ids": [program.region_id for program in programs],
            "semantics": [program.semantic for program in programs],
            "input_rule": input_rule,
            "output_shape": list(actual.shape),
            "output_dtype": str(actual.dtype),
            "output_sha256": array_sha256(actual),
            "oracle_sha256": array_sha256(expected),
        })

    embeddings = sorted(
        (program for program in lane.programs.values() if program.semantic == "embedding"),
        key=lambda program: program.signature["indexed_generic"]["table_shape"],
    )
    for ordinal, program in enumerate(embeddings):
        table_shape = program.signature["indexed_generic"]["table_shape"]
        index_shape = program.signature["indexed_generic"]["input_shapes"][0]
        indices = (
            np.arange(np.prod(index_shape), dtype=np.int64) * (101 + ordinal)
        ).reshape(index_shape) % table_shape[0]
        if table_shape[0] <= 2048:
            table = (
                np.arange(table_shape[0], dtype=np.float32)[:, None] * np.float32(1000)
                + np.arange(table_shape[1], dtype=np.float32)[None, :]
            )
        else:
            row = np.arange(table_shape[1], dtype=np.float32) % np.float32(32)
            table = np.broadcast_to(row, table_shape)
        actual = lane.execute(
            program.region_id,
            {program.input_values[0]: indices, program.input_values[1]: table},
        )
        record(
            f"embedding_{ordinal}", [program], actual, table[indices],
            "bounded ordinal indices and a coordinate-derived exact table",
        )

    gather = next(
        program for program in lane.programs.values() if program.semantic == "index_gather"
    )
    row_indices = np.zeros((1, 1, 1, 1), dtype=np.int64)
    column_indices = np.arange(1023, -1, -1, dtype=np.int64).reshape(1, 1, 1, 1024)
    bool_table = ((np.arange(1024) * 7) % 11 < 5).reshape(1, 1024)
    actual = lane.execute(
        gather.region_id,
        {
            gather.input_values[0]: row_indices,
            gather.input_values[1]: column_indices,
            gather.input_values[2]: bool_table,
        },
    )
    record(
        "index_gather", [gather], actual,
        bool_table[row_indices, column_indices],
        "zero row plus reverse column permutation over a patterned boolean table",
    )

    mask_gather = next(
        program for program in lane.programs.values() if program.semantic == "mask_gather"
    )
    index_put = next(
        program for program in lane.programs.values() if program.semantic == "index_put"
    )
    data = (np.arange(1024, dtype=np.int64) * 13 - 7).reshape(1, 1024)
    mask = ((np.arange(1024) * 5) % 17 < 6).reshape(1, 1024)
    values = {mask_gather.input_values[0]: data, mask_gather.input_values[1]: mask}
    compact = lane.execute(mask_gather.region_id, values)
    destination = np.full((1, 1024), -99, dtype=np.int64)
    values[index_put.input_values[0]] = destination
    values[index_put.input_values[1]] = mask
    actual = lane.execute(index_put.region_id, values)
    expected = destination.copy()
    expected[mask] = data[mask]
    record(
        "mask_gather_index_put", [mask_gather, index_put], actual, expected,
        "patterned mask compacts coordinate data then scatters it into a fresh destination",
    )

    convolution = next(
        program for program in lane.programs.values()
        if program.semantic == "convolution_im2col_matmul"
    )
    image_shape, weight_shape, bias_shape = convolution.signature["input_shapes"]
    image = np.arange(np.prod(image_shape), dtype=np.float32).reshape(image_shape)
    weight = np.zeros(weight_shape, dtype=np.float32)
    weight[0, 0, 0, 0] = np.float32(2)
    weight[1, 2, 15, 15] = np.float32(-1)
    bias = (
        np.arange(np.prod(bias_shape), dtype=np.float32) - np.float32(384)
    ).reshape(bias_shape) / np.float32(8)
    actual = lane.execute(
        convolution.region_id,
        {
            convolution.input_values[0]: image,
            convolution.input_values[1]: weight,
            convolution.input_values[2]: bias,
        },
    )
    expected = np.broadcast_to(bias.reshape(1, 768, 1, 1), actual.shape).copy()
    expected[0, 0] += np.float32(2) * image[0, 0, 0::16, 0::16]
    expected[0, 1] -= image[0, 2, 15::16, 15::16]
    record(
        "patch_embedding_convolution", [convolution], actual, expected,
        "two sparse kernel taps plus per-channel bias over the full captured 512x512 input",
    )
    return witnesses


def layout_bridge_witnesses(workload) -> list[dict]:
    """Numerically qualify one real bridge per exact map/dtype/topology class."""
    lane = LayoutBridgeLane(workload)
    classes = {}
    for program in lane.programs.values():
        signature = program.signature
        key = (
            signature["semantic"], signature["dtype"],
            signature["materialization"], len(signature["output_shape"]),
            tuple((item["kind"], item.get("position"), item.get("value"))
                  for item in signature["input_map"]),
        )
        classes.setdefault(key, []).append(program)
    witnesses = []
    for key, programs in sorted(classes.items()):
        program = min(programs, key=lambda item: (
            int(np.prod(item.signature["output_shape"], dtype=np.int64)),
            item.region_id,
        ))
        signature = program.signature
        count = int(np.prod(signature["input_shape"], dtype=np.int64))
        ordinal = np.arange(count, dtype=np.int64).reshape(signature["input_shape"])
        if signature["dtype"] == "i1":
            source = (ordinal % 3) == 0
        else:
            source = ((ordinal % 31).astype(np.float32) - np.float32(7)) / np.float32(4)
        actual = lane.execute(program.region_id, {program.input_values[0]: source})

        index = tuple(
            0 if item["kind"] == "constant" else slice(None)
            for item in signature["input_map"]
        )
        reduced = source[index]
        mapped = [
            item["position"] for item in signature["input_map"]
            if item["kind"] == "dim"
        ]
        if mapped:
            reduced = np.transpose(reduced, axes=tuple(int(v) for v in np.argsort(mapped)))
        reshape = [1] * len(signature["output_shape"])
        for axis, dimension in enumerate(sorted(mapped)):
            reshape[dimension] = reduced.shape[axis]
        expected = np.array(
            np.broadcast_to(reduced.reshape(reshape), signature["output_shape"]),
            copy=True,
            order="C",
        )
        if not np.array_equal(actual, expected):
            raise ValueError(f"independent layout oracle failed for {program.region_id}")
        if not actual.flags.c_contiguous or np.shares_memory(actual, source):
            raise ValueError(f"layout bridge did not create distinct contiguous storage")
        witnesses.append({
            "schema": "atlas_real_capture_layout_bridge_numeric_witness_v1",
            "label": "_".join((key[0], key[1], key[2], f"rank{key[3]}")),
            "status": "fresh_numeric_execution_matches_independent_oracle",
            "claim": (
                "host materialization evidence only; physical DMA/event execution is absent"
            ),
            "representative_region_id": program.region_id,
            "class_region_count": len(programs),
            "semantic": signature["semantic"],
            "dtype": signature["dtype"],
            "input_shape": signature["input_shape"],
            "output_shape": signature["output_shape"],
            "input_map": signature["input_map"],
            "materialization": signature["materialization"],
            "output_sha256": array_sha256(actual),
            "oracle_sha256": array_sha256(expected),
            "distinct_contiguous_storage": True,
        })
    return witnesses


def _independent_bf16_rne(value: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def accelerator_contract_witnesses(
    workload, plan: dict, command_buffers: dict[str, dict],
) -> list[dict]:
    """Exercise one real-shape contract per exact dtype/epilogue topology."""
    lane = AcceleratorContractLane(workload, plan, command_buffers)
    classes = {}
    for contract in lane.contracts.values():
        signature = contract.signature
        key = (signature["source_dtype"], signature["bias_fused"])
        classes.setdefault(key, []).append(contract)
    witnesses = []
    for (source_dtype, bias_fused), contracts in sorted(classes.items()):
        contract = min(
            contracts,
            key=lambda row: (
                row.signature["geometry"]["M"] * row.signature["geometry"]["N"],
                row.signature["geometry"]["K"], row.partition_id,
            ),
        )
        geometry = contract.signature["geometry"]
        m, k, n = (geometry[key] for key in ("M", "K", "N"))
        activation = np.zeros((m, k), dtype=np.float32)
        selected = (np.arange(m, dtype=np.int64) * 17 + 3) % k
        activation[np.arange(m), selected] = np.float32(1)
        rows = np.arange(k, dtype=np.int64)[:, None]
        columns = np.arange(n, dtype=np.int64)[None, :]
        weight = (((rows * 3 + columns) % 5) - 2).astype(np.float32)
        values = {"A0": activation, "W": weight}
        expected = weight[selected].copy()
        if bias_fused:
            bias = ((np.arange(n, dtype=np.int64) % 3) - 1).astype(np.float32)
            values["B"] = bias
            expected = np.asarray(expected + bias, dtype=np.float32)
        actual = lane.execute_device_domain(contract.partition_id, dict(values))
        expected = _independent_bf16_rne(expected)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"independent accelerator oracle failed for {contract.partition_id}"
            )
        converted = lane.prepare_capture_inputs(contract.partition_id, values)
        expected_preload_bytes = m * k + k * n + (2 * n if bias_fused else 0)
        actual_preload_bytes = sum(len(raw) for raw in converted["preloads"].values())
        if actual_preload_bytes != expected_preload_bytes:
            raise ValueError(
                f"conversion byte extent failed for {contract.partition_id}"
            )
        published = lane.publish_device_output(
            contract.partition_id, actual, converted["record"]["output_scale"]
        )
        published_oracle = np.asarray(
            expected * np.float32(converted["record"]["output_scale"]),
            dtype=np.float32,
        )
        if source_dtype == "bf16":
            published_oracle = _independent_bf16_rne(published_oracle)
        if not np.array_equal(published, published_oracle):
            raise ValueError(
                f"independent output conversion oracle failed for {contract.partition_id}"
            )
        witnesses.append({
            "schema": "atlas_real_shape_rank2_command_contract_witness_v1",
            "label": f"rank2_{source_dtype}_{'bias' if bias_fused else 'no_bias'}",
            "status": "fresh_device_domain_execution_matches_independent_oracle",
            "claim": (
                "static command-contract and host conversion evidence only; encoded image, "
                "physical partition, DMA/event runtime, and E2E are not qualified"
            ),
            "representative_partition_id": contract.partition_id,
            "class_partition_count": len(contracts),
            "geometry": geometry,
            "source_dtype": source_dtype,
            "bias_fused": bias_fused,
            "output_sha256": array_sha256(actual),
            "oracle_sha256": array_sha256(expected),
            "conversion_preload_bytes": actual_preload_bytes,
            "published_capture_sha256": array_sha256(published),
            "published_oracle_sha256": array_sha256(published_oracle),
            "command_contract_sha256": contract_sha256(contract.signature),
        })
    return witnesses


def main() -> int:
    source_path = CAPTURE / "model.mlir"
    source = source_path.read_text(encoding="utf-8")
    plan = load(PLAN_ROOT / "partition_plan.json")
    command_buffers = {
        row["kernel_id"]: load(ROOT / row["command_buffer"])
        for row in plan["kernel_library"]
    }
    inventory = load(ROOT / "full_capture_partition_inventory.json")
    workload = parse_verified(source)
    chain, bounded_regions = bounded_chain_witness()
    host_chain, host_witnesses = generic_host_witnesses(workload)
    indexed_witnesses = final_indexed_host_witnesses(workload)
    bridge_witnesses = layout_bridge_witnesses(workload)
    accelerator_witnesses = accelerator_contract_witnesses(
        workload, plan, command_buffers
    )
    schedule = build_hybrid_schedule(
        workload, plan, inventory,
        qualified_partitions=set(QUALIFIED),
        bounded_host_regions=bounded_regions,
        command_buffers=command_buffers,
        alignment=32,
    )
    schedule["capture"].update({
        "path": source_path.relative_to(REPO).as_posix(),
        "bytes": len(source.encode()),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
    })
    schedule["bounded_chain"] = chain
    schedule["generic_host_chain"] = host_chain
    schedule["generic_host_numeric_witnesses"] = host_witnesses
    schedule["final_indexed_host_numeric_witnesses"] = indexed_witnesses
    schedule["layout_bridge_numeric_witnesses"] = bridge_witnesses
    schedule["accelerator_contract_numeric_witnesses"] = accelerator_witnesses
    schedule["device_activation_arena"]["alignment_source"] = (
        "the existing Atlas command-buffer allocator's 32-byte tensor-base alignment"
    )
    schedule_path = PLAN_ROOT / "hybrid_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "atlas_hybrid_capture_schedule_summary_v1",
        "status": schedule["status"],
        "runnable_e2e": schedule["runnable_e2e"],
        "claim": schedule["claim"],
        "coverage": schedule["coverage"],
        "fail_closed": {
            key: value for key, value in schedule["fail_closed"].items()
            if not key.endswith("_ids")
        },
        "conversion_boundaries": schedule["conversion_boundaries"],
        "device_activation_arena": {
            key: value for key, value in schedule["device_activation_arena"].items()
            if key != "allocations"
        },
        "event_count": len(schedule["events"]),
        "bounded_chain": chain,
        "generic_host_chain": host_chain,
        "generic_host_numeric_witnesses": host_witnesses,
        "final_indexed_host_numeric_witnesses": indexed_witnesses,
        "layout_bridge_census": schedule["layout_bridge_census"],
        "layout_bridge_numeric_witnesses": bridge_witnesses,
        "accelerator_contract_census": schedule["accelerator_contract_census"],
        "accelerator_contract_numeric_witnesses": accelerator_witnesses,
        "full_schedule": {
            "path": schedule_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(schedule_path),
        },
    }
    summary_path = PLAN_ROOT / "hybrid_schedule_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
