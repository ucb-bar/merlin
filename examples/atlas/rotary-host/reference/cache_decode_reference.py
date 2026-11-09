"""Independent reference v3 replay of captured whole-model cache source FX.

Only original source operators, constants, token IDs, positions, and masks are
read from capture. Decode replaces every past cache with authenticated expected
words from an independent prefill reference. Generated words are never inputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import whole_reference as reference
import source_trace


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def program_from_trace(capture):
    """Rebuild original FX source structure; no lowering or compiler evaluator."""
    from safetensors.torch import load_file

    capture = Path(capture)
    nodes = read(capture / "frontend-trace.json")["graphs"]["quantized"]["nodes"]
    archive = load_file(str(capture / "weights.safetensors"))
    return source_trace.rebuild_program(nodes, lambda record: archive[record["input_target"]])


def authenticate_weights(program, capture, export_dir):
    """Compare every captured constant with authenticated original export bytes."""
    from safetensors import safe_open
    import torch

    export_dir = Path(export_dir)
    manifest = read(export_dir / "manifest.json")
    files = manifest["files"]
    if set(files) != {"model.safetensors", "policy.json"}:
        raise ValueError("original export file roster differs")
    for name, record in files.items():
        if digest(export_dir / name) != record["sha256"]:
            raise ValueError("original export file hash differs: " + name)
    state = {**program.state_dict, **program.constants}
    with safe_open(str(Path(capture) / "weights.safetensors"), framework="pt") as captured:
        if set(captured.keys()) != set(state):
            raise ValueError("capture weight archive and source state roster differ")
    with safe_open(str(export_dir / "model.safetensors"), framework="pt") as original:
        names = set(original.keys())
        for name, value in state.items():
            source_name = name.removeprefix("model.")
            if source_name not in names:
                raise ValueError("capture weight missing from original export: " + name)
            source = original.get_tensor(source_name)
            if (value.dtype != source.dtype or value.shape != source.shape
                    or not np.array_equal(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy(),
                                          source.detach().contiguous().reshape(-1).view(torch.uint8).numpy())):
                raise ValueError("capture weight differs from original export: " + name)
    return dict(export_manifest_sha256=digest(export_dir / "manifest.json"),
                model_safetensors_sha256=files["model.safetensors"]["sha256"],
                policy_sha256=files["policy.json"]["sha256"], authenticated_tensor_count=len(state))


class Interpreter(reference.Interpreter):
    def call(self, node):
        if str(node.target) == "aten.embedding.default":
            args, kwargs = self.resolve(node.args), self.resolve(node.kwargs)
            weight, indices, *options = args
            if (kwargs or weight.dtype != "bf16" or indices.dtype != "i64"
                    or weight.raw.ndim != 2 or self.dtype(node) != "bf16"
                    or any(value not in {-1, False} for value in options)
                    or np.any(indices.raw < 0) or np.any(indices.raw >= weight.raw.shape[0])):
                raise ValueError("source embedding requires unchanged BF16 gather and valid token IDs")
            return reference.Tensor(weight.raw[indices.raw].copy(), "bf16")
        return super().call(node)


def decode_inputs(capture, users, parent=None, tokens=None):
    """Only the first three source inputs can originate in a capture archive."""
    capture = Path(capture)
    if len(users) < 3 or (len(users) - 3) % 2:
        raise ValueError("cache source requires IDs, positions, mask, and complete K/V pairs")
    inputs = []
    with np.load(capture / "inputs.npz", allow_pickle=False) as arrays:
        for index, user in enumerate(users[:3]):
            value = np.ascontiguousarray(arrays[f"in{index}"])
            expected = user["results"][0]
            dtype = expected["dtype"]
            if dtype == "int64":
                if value.dtype != np.int64:
                    raise ValueError("source IDs and positions must retain int64")
                result = reference.Tensor(value.copy(), "i64")
            elif dtype == "bfloat16":
                if value.dtype == np.float32 and not np.any(value.view(np.uint32) & 0xffff):
                    value = (value.view(np.uint32) >> 16).astype(np.uint16)
                if value.dtype != np.uint16:
                    raise ValueError("source mask must contain exact BF16 words")
                result = reference.Tensor(value.copy(), "bf16")
            else:
                raise ValueError("cache source preparation dtype differs")
            if list(result.raw.shape) != expected["shape"]:
                raise ValueError("source preparation shape differs")
            inputs.append(result)
    if tokens is not None:
        tokens = np.load(tokens, allow_pickle=False)
        if tokens.dtype != np.int64 or tokens.shape != inputs[0].raw.shape:
            raise ValueError("explicit source token selection must retain int64 source shape")
        inputs[0] = reference.Tensor(tokens.copy(), "i64")
    ids, positions, mask = inputs
    if ids.dtype != "i64" or positions.dtype != "i64" or mask.dtype != "bf16" or ids.raw.ndim != 2:
        raise ValueError("source preparation ABI differs")
    batch, query = ids.raw.shape
    past = int(positions.raw[0, 0])
    wanted_positions = np.broadcast_to(np.arange(past, past + query, dtype=np.int64), (batch, query))
    keep = np.arange(past + query)[None, :] <= np.arange(past, past + query)[:, None]
    wanted_mask = np.broadcast_to(np.where(keep, 0, 0xff7f).astype(np.uint16), (batch, 1, query, past + query))
    if not np.array_equal(positions.raw, wanted_positions) or not np.array_equal(mask.raw, wanted_mask):
        raise ValueError("source positions or mask differ from the exact causal append boundary")
    if past == 0:
        if len(users) != 3 or parent is not None:
            raise ValueError("empty prefill must have three source inputs and no parent reference")
        return inputs, None
    if parent is None:
        raise ValueError("decode requires an independent parent reference")
    parent = Path(parent)
    receipt = read(parent / "reference.json")
    if (receipt.get("status") != "EXPECTED" or receipt.get("reference_version") != 3
            or receipt.get("oracle_mode") != "independent_whole_cache_source"
            or receipt.get("compiler_or_emitted_program_inputs") is not False
            or receipt.get("ordinary_framework_kernels_called") is not False
            or receipt.get("orchestration_sha256") != digest(__file__)
            or receipt.get("trace_rebuild_sha256") != digest(source_trace.__file__)
            or receipt.get("expected_sha256") != digest(parent / "expected.npz")):
        raise ValueError("parent is not an authenticated independent whole-cache reference")
    if len(receipt.get("outputs", {})) != len(users) - 2:
        raise ValueError("parent independent cache output roster differs")
    with np.load(parent / "expected.npz", allow_pickle=False) as arrays:
        for index, user in enumerate(users[3:], 1):
            key = f"output_{index}"
            value = np.ascontiguousarray(arrays[key])
            if (value.dtype != np.uint16 or list(value.shape) != user["results"][0]["shape"]
                    or reference.info(value) != {field: receipt["outputs"][key][field] for field in ("dtype", "shape", "sha256")}):
                raise ValueError("parent independent cache words differ from their publication")
            inputs.append(reference.Tensor(value.copy(), "bf16"))
    return inputs, receipt


def run(capture, out, policy, *, export_dir, parent=None, engine="numpy", proof=None, tokens=None):
    capture, out = Path(capture), Path(out)
    program, users = program_from_trace(capture)
    original_export = authenticate_weights(program, capture, export_dir)
    inputs, parent_receipt = decode_inputs(capture, users, parent, tokens)
    constants = {key: reference.info(reference.raw_tensor(value).raw)
                 for key, value in {**program.state_dict, **program.constants}.items()}
    frequency = constants.get("model.model.rotary_emb.inv_freq")
    if frequency is not None and frequency["sha256"] != reference.rotary_exact.INV_FREQ_SHA256:
        raise ValueError("source rotary inverse-frequency words differ from qualification")
    if parent_receipt is not None and parent_receipt.get("source_constants") != constants:
        raise ValueError("decode weights differ from independent prefill weights")
    if parent_receipt is not None and (parent_receipt.get("qualification") != proof
            or parent_receipt.get("selected_policies") != policy
            or parent_receipt.get("adapter_sha256") != digest(reference.__file__)):
        raise ValueError("decode independent arithmetic differs from parent reference")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(schema="atlas.independent_whole_cache_source.v3", status="RUNNING", reference_version=3,
        oracle_mode="independent_whole_cache_source", orchestration_sha256=digest(__file__),
        trace_rebuild_sha256=digest(source_trace.__file__),
        adapter_sha256=digest(reference.__file__), source_trace_sha256=digest(capture / "frontend-trace.json"),
        source_weights_sha256=digest(capture / "weights.safetensors"), source_constants=constants,
        original_export=original_export,
        compiler_or_emitted_program_inputs=False, ordinary_framework_kernels_called=False,
        selected_policies=policy, inputs={f"input_{i}": reference.info(value.raw) for i, value in enumerate(inputs)},
        parent_reference_sha256=digest(Path(parent) / "reference.json") if parent else None, qualification=proof)
    record["explicit_tokens_sha256"] = digest(tokens) if tokens is not None else None
    write(out / "reference.json", record)
    try:
        interpreter = Interpreter(program, policy, engine=engine, threads=1,
            linear_policy=reference.BF16_LINEAR_POLICY,
            progress=lambda name, **rest: print(json.dumps(dict(node=name, **rest)), flush=True))
        values = interpreter.run(inputs)
        declared = len(list(program.graph.nodes)[-1].args[0])
        if (len(values) != declared or len(values) < 3 or (len(values) - 1) % 2
                or any(value.dtype != "bf16" for value in values)
                or parent_receipt is not None and len(values) != len(users) - 2):
            raise ValueError("whole-cache reference requires logits and all BF16 K/V pairs")
        if parent_receipt is not None:
            for index, value in enumerate(values[1:]):
                prefix = inputs[3 + index].raw
                if not np.array_equal(value.raw[:, :, :prefix.shape[2]], prefix):
                    raise ValueError("decode changed independent cache prefix words")
        arrays = {f"output_{i}": value.raw for i, value in enumerate(values)}
        np.savez(out / "expected.npz", **arrays)
        record.update(status="EXPECTED", outputs={key: {**reference.info(value), "logical_dtype": "bf16"}
            for key, value in arrays.items()}, expected_sha256=digest(out / "expected.npz"),
            constant_bytes_identical=True, cache_prefix_preserved=True,
            elapsed_seconds=round(time.monotonic() - started, 6))
        write(out / "reference.json", record)
        return record
    except Exception as error:
        record.update(status="REFUSED", error=str(error))
        write(out / "reference.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("capture", "out", "export-dir", "model-root", "reference-root", "policy", "validation", "mask-validation", "bf16-linear-validation", "rotary-validation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parent-reference", type=Path)
    parser.add_argument("--tokens", type=Path, help="explicit original token IDs with the captured static shape")
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--native-qualification", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.model_root), str(args.reference_root)]
    import torch
    from atlas_ref.ops import register
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    register()
    before = reference.validate_helpers(args.validation, args.model_root)
    policy = read(args.policy)
    mask, linear = read(args.mask_validation), read(args.bf16_linear_validation)
    if (mask.get("status") != "PASS" or mask.get("mismatches") != 0
            or mask.get("selected_policy") != reference.rowspan_softmax.POLICY
            or mask.get("model_arithmetic_sources") != before
            or mask.get("oracle_sha256") != reference.rowspan_softmax.source_hash()
            or policy.get("already_quantized_mask") != reference.rowspan_softmax.POLICY):
        raise ValueError("independent source mask arithmetic lacks matching qualification")
    if (linear.get("status") != "PASS" or linear.get("mismatches") != 0
            or linear.get("selected_policy") != reference.BF16_LINEAR_POLICY
            or linear.get("model_arithmetic_sources") != before
            or linear.get("bf16_linear_oracle_sha256") != sha256(inspect.getsource(reference.bf16_linear).encode()).hexdigest()):
        raise ValueError("independent source head arithmetic lacks matching qualification")
    rotary = read(args.rotary_validation)
    if (rotary.get("status") != "PASS" or rotary.get("mismatches") != 0
            or rotary.get("selected_policy") != reference.ROTARY_POLICY
            or rotary.get("oracle_sha256") != digest(reference.rotary_exact.__file__)
            or rotary.get("whole_interpreter_sha256") != digest(reference.__file__)
            or rotary.get("archive_gate_sha256") != digest(rotary["archive_gate_path"])):
        raise ValueError("independent source rotary arithmetic lacks matching qualification")
    engine = reference.primitives.configure_engine(args.engine, 1, args.native_qualification)
    result = run(args.capture, args.out, policy, export_dir=args.export_dir, parent=args.parent_reference, tokens=args.tokens, engine=args.engine,
        proof=dict(engine=engine, model_arithmetic_sources=before,
            validation_sha256=digest(args.validation), mask_validation_sha256=digest(args.mask_validation),
            bf16_linear_validation_sha256=digest(args.bf16_linear_validation),
            rotary_validation_sha256=digest(args.rotary_validation)))
    if reference.primitives.authenticate_model(args.model_root) != before:
        raise ValueError("instruction-model arithmetic changed during independent cache replay")
    print(json.dumps(dict(status=result["status"], elapsed_seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
