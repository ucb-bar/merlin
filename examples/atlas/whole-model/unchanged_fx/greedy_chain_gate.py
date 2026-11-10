"""Compare a generated prefill64/decode65/decode66 chain with independent replay.

This example gate authenticates both publications, checks every raw BF16 output,
and checks both token choices and every generated cache binding. It grants no
hardware qualification and never supplies device output words to the reference.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


GREEDY_POLICY = "bf16-last-row-highest-value-lowest-index-v1"
REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "rotary-host" / "reference"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def info(value):
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=sha256(value.tobytes()).hexdigest())


def metadata(row):
    return {key: row[key] for key in ("dtype", "shape", "sha256")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def greedy_tokens(words):
    """Gate-owned selection, independent of either chain's token selector."""
    require(words.dtype == np.uint16 and words.ndim == 3 and all(words.shape),
            "greedy logits must be nonempty raw BF16 [batch, sequence, vocabulary]")
    require(not np.any((words & 0x7fff) > 0x7f80), "greedy logits contain NaN")
    rows = (words[:, -1, :].astype(np.uint32) << 16).view(np.float32)
    # Find the first equal maximum explicitly to make the tie rule reviewable.
    return np.array([[int(np.flatnonzero(row == np.max(row))[0])] for row in rows], dtype=np.int64)


def load_step(application, execution, reference):
    application, execution, reference = map(Path, (application, execution, reference))
    compiled, actual_receipt, expected_receipt = (read(application / "compiled.json"),
        read(execution / "execution.json"), read(reference / "reference.json"))
    require(compiled.get("status") == "COMPILED", "step application is not compiled")
    require(actual_receipt.get("status") == "EXECUTED_UNCOMPARED"
            and actual_receipt.get("immutable_inputs") is True,
            "step generated execution is incomplete or inputs were mutable")
    require(actual_receipt.get("compiled_sha256") == digest(application / "compiled.json"),
            "step generated compiled publication differs")
    for field in ("source_request_digest", "source_model_sha256"):
        require(bool(compiled.get(field)) and actual_receipt.get(field) == compiled[field],
                "step generated source identity differs: " + field)
    require(expected_receipt.get("status") == "EXPECTED"
            and expected_receipt.get("reference_version") == 3
            and expected_receipt.get("oracle_mode") == "independent_whole_cache_source"
            and expected_receipt.get("compiler_or_emitted_program_inputs") is False
            and expected_receipt.get("ordinary_framework_kernels_called") is False,
            "step reference is not an independent whole-cache publication")
    for field, name in (("orchestration_sha256", "cache_decode_reference.py"),
                        ("trace_rebuild_sha256", "source_trace.py"),
                        ("adapter_sha256", "whole_reference.py")):
        require(expected_receipt.get(field) == digest(REFERENCE_ROOT / name),
                "step independent reference implementation differs: " + field)
    require(isinstance(compiled.get("capture"), str) and bool(compiled["capture"]),
            "step compiled application lacks its source capture")
    capture = Path(compiled["capture"])
    if not capture.is_absolute():
        capture = application / capture
    require(compiled.get("capture_mlir_sha256") == digest(capture / "model.mlir"),
            "step compiled source capture MLIR differs")
    for field, name in (("source_trace_sha256", "frontend-trace.json"),
                        ("source_weights_sha256", "weights.safetensors")):
        require(expected_receipt.get(field) == digest(capture / name),
                "step reference/compiled source capture differs: " + field)
    require(expected_receipt.get("expected_sha256") == digest(reference / "expected.npz"),
            "step expected archive publication differs")
    interface = compiled["interface"]
    count = len(interface["outputs"])
    require(count >= 3 and count % 2 == 1
            and len(interface["output_types"]) == count,
            "step requires logits and complete K/V outputs")
    keys = {f"output_{i}" for i in range(count)}
    require(set(expected_receipt.get("outputs", {})) == keys
            and len(actual_receipt.get("outputs", [])) == count,
            "step output receipt roster differs")
    actual, expected = {}, {}
    with np.load(execution / "actual.npz", allow_pickle=False) as aa, \
            np.load(reference / "expected.npz", allow_pickle=False) as ee:
        require(set(aa.files) == keys and len(aa.files) == count
                and set(ee.files) == keys and len(ee.files) == count,
                "step output archive roster differs")
        for index in range(count):
            key, spec = f"output_{index}", interface["output_types"][index]
            a, e = aa[key], ee[key]
            require(a.dtype == np.uint16 and e.dtype == np.uint16 and spec["dtype"] == "bf16"
                    and list(a.shape) == spec["shape"] and a.shape == e.shape,
                    "step BF16 output signature differs: " + key)
            require(info(a) == metadata(actual_receipt["outputs"][index])
                    and info(e) == metadata(expected_receipt["outputs"][key])
                    and expected_receipt["outputs"][key].get("logical_dtype") == "bf16",
                    "step output words differ from publication: " + key)
            require(np.array_equal(a, e), "step exact output comparison failed: " + key)
            actual[key], expected[key] = a.copy(), e.copy()
    return dict(application=application, execution=execution, reference=reference,
                compiled=compiled, generated=actual_receipt, independent=expected_receipt,
                actual=actual, expected=expected)


def check_inputs(step, past, query, count):
    generated, independent = step["generated"], step["independent"]
    actual_inputs, expected_inputs = generated.get("inputs", []), independent.get("inputs", {})
    expected_count = 3 if past == 0 else count + 2
    require(len(actual_inputs) == expected_count
            and set(expected_inputs) == {f"input_{i}" for i in range(expected_count)}
            and len(step["compiled"]["interface"]["inputs"]) == expected_count,
            "step complete cache input roster differs")
    batch = actual_inputs[0]["shape"][0]
    require(actual_inputs[0]["shape"] == [batch, query] and batch > 0
            and actual_inputs[0]["dtype"] == np.dtype(np.int64).str,
            "step token input signature differs")
    positions = np.broadcast_to(np.arange(past, past + query, dtype=np.int64), (batch, query))
    keep = np.arange(past + query)[None, :] <= np.arange(past, past + query)[:, None]
    mask = np.broadcast_to(np.where(keep, 0, 0xff7f).astype(np.uint16),
                           (batch, 1, query, past + query))
    require(metadata(actual_inputs[1]) == info(positions) and metadata(actual_inputs[2]) == info(mask),
            "step causal positions or mask differ")
    for index, row in enumerate(actual_inputs):
        require(metadata(row) == metadata(expected_inputs[f"input_{index}"]),
                f"step generated/reference input differs: input_{index}")
        spec = step["compiled"]["interface"]["input_types"][index]
        require(spec["shape"] == row["shape"] and spec["dtype"] == ("i64" if index < 2 else "bf16"),
                f"step compiled input signature differs: input_{index}")
    for key, words in step["actual"].items():
        if key == "output_0":
            require(list(words.shape[:2]) == [batch, query], "step logits row boundary differs")
        else:
            require(words.ndim == 4 and words.shape[0] == batch and words.shape[2] == past + query,
                    "step cache append boundary differs")


def check_transition(previous, current):
    old, new = previous["independent"], current["independent"]
    require(new.get("parent_reference_sha256") == digest(previous["reference"] / "reference.json"),
            "independent parent reference link differs")
    for field in ("source_constants", "original_export", "selected_policies", "qualification"):
        require(field in old and new.get(field) == old[field],
                "independent chain arithmetic or constants differ: " + field)
    generated_token = greedy_tokens(previous["actual"]["output_0"])
    independent_token = greedy_tokens(previous["expected"]["output_0"])
    require(np.array_equal(generated_token, independent_token), "generated/reference greedy token differs")
    require(metadata(current["generated"]["inputs"][0]) == info(generated_token),
            "generated greedy token input differs")
    require(metadata(new["inputs"]["input_0"]) == info(independent_token),
            "independent greedy token input differs")
    wanted = dict(policy=GREEDY_POLICY, source="independent_parent_expected_logits",
                  parent_reference_sha256=digest(previous["reference"] / "reference.json"),
                  parent_expected_sha256=old["expected_sha256"],
                  parent_logits=old["outputs"]["output_0"], tokens=info(independent_token),
                  token_ids=independent_token.tolist())
    require(new.get("greedy_selection") == wanted and new.get("explicit_tokens_sha256") is None,
            "independent greedy selection provenance differs")
    bindings = current["generated"].get("generated_bindings", [])
    count = len(previous["actual"]) - 1
    require(len(bindings) == count + 1, "complete generated cache and token binding roster differs")
    by_input = {row["input_index"]: row for row in bindings}
    require(set(by_input) == {0, *range(3, count + 3)} and len(by_input) == count + 1,
            "complete generated cache binding indices differ")
    identities = dict(producer_compiled_sha256=digest(previous["application"] / "compiled.json"),
                      producer_execution_sha256=digest(previous["execution"] / "execution.json"),
                      producer_actual_sha256=digest(previous["execution"] / "actual.npz"),
                      source_request_digest=previous["compiled"]["source_request_digest"],
                      source_model_sha256=previous["compiled"]["source_model_sha256"])
    wanted_token = {**identities, "operation": "greedy_last_token", "input_index": 0, "output_index": 0,
                    "dtype": "i64", "shape": [1, 1], "sha256": info(generated_token)["sha256"],
                    "producer_output_sha256": info(previous["actual"]["output_0"])["sha256"],
                    "source_dtype": "bf16", "source_shape": list(previous["actual"]["output_0"].shape),
                    "sequence_index": previous["actual"]["output_0"].shape[1] - 1,
                    "token_id": int(generated_token[0, 0]), "tie_break": "lowest_index"}
    require(by_input[0] == wanted_token, "generated greedy selection provenance differs")
    for index in range(count):
        key = f"output_{index + 1}"
        row = by_input[index + 3]
        wanted = {**identities, "input_index": index + 3, "output_index": index + 1,
                  "dtype": "bf16", "shape": list(previous["actual"][key].shape),
                  "sha256": info(previous["actual"][key])["sha256"]}
        require(row == wanted, f"generated cache binding provenance differs: input_{index + 3}")
        require(metadata(current["generated"]["inputs"][index + 3]) == info(previous["actual"][key]),
                f"generated cache input does not bind predecessor words: input_{index + 3}")
        require(metadata(new["inputs"][f"input_{index + 3}"]) == info(previous["expected"][key]),
                f"independent cache input does not bind predecessor words: input_{index + 3}")
        require(np.array_equal(current["actual"][key][:, :, :previous["actual"][key].shape[2]],
                               previous["actual"][key]), "generated cache prefix differs")
    return dict(generated_token_ids=generated_token.tolist(), independent_token_ids=independent_token.tolist(),
                cache_bindings=count)


def compare_chain(steps):
    """Each step is (compiled_application, generated_execution, independent_reference)."""
    require(len(steps) == 3, "greedy chain requires exactly prefill64, decode65, decode66")
    loaded = [load_step(*step) for step in steps]
    count = len(loaded[0]["actual"])
    transitions = []
    for index, (step, past, query) in enumerate(zip(loaded, (0, 64, 65), (64, 1, 1))):
        require(len(step["actual"]) == count, "chain output roster differs")
        check_inputs(step, past, query, count)
        if index:
            transitions.append(check_transition(loaded[index - 1], step))
        else:
            require(step["independent"].get("parent_reference_sha256") is None
                    and not step["generated"].get("generated_bindings")
                    and not step["independent"].get("greedy_selection"),
                    "prefill unexpectedly has parent bindings")
    return dict(schema="atlas.greedy_chain_comparison.v1", status="PASS", mismatches=0,
                steps=[dict(name=name, outputs_compared=count,
                            compiled_sha256=digest(step["application"] / "compiled.json"),
                            execution_sha256=digest(step["execution"] / "execution.json"),
                            reference_sha256=digest(step["reference"] / "reference.json"))
                       for name, step in zip(("prefill64", "decode65", "decode66"), loaded)],
                transitions=transitions, hardware_qualification="PENDING")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--step", nargs=3, metavar=("APPLICATION", "EXECUTION", "REFERENCE"),
                        action="append", required=True, help="repeat in prefill64/decode65/decode66 order")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare_chain(args.step)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
