"""Capture source-framework prefill4 and two one-token BF16-cache phases.

Requires the completed authenticated raw64 parity/reload gate. This command
performs source framework execution and native uint8 FX capture only. It does
not compile or execute generated device arithmetic, evaluate quality, or replace
weights. Each source fixture follows its own actual cache chain.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys

import torch

import raw_capture as raw
from cache_source import (TensorCacheLogits, TensorPrefillLogits, phase_positions_and_mask,
                          validate_phase, validate_prefill_phase)
from cache_roles import derive_cache_phase_abi

PHASES = (("prefill4", 0, 4), ("decode5", 4, 1), ("decode6", 5, 1))


def record_hash(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def module_inventory(model):
    """Include nonpersistent raw buffers and shape-only meta placeholders."""
    result = {}
    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in values:
            result[f"{kind}.{name}"] = (dict(dtype=str(value.dtype), shape=list(value.shape), device="meta")
                                          if value.device.type == "meta" else raw.tensor_record(value))
    return result


def fixture_sequences(fixtures, parity):
    """Use the first six IDs from two declared original no-cache cases."""
    if len(parity.get("cases", ())) < 2:
        raise ValueError("two original source fixtures are required")
    sequences, origins = [], []
    for index, case in enumerate(parity["cases"][:2]):
        if len(case.get("inputs", ())) != 1:
            raise ValueError("original source fixture must expose token IDs only")
        key = case["inputs"][0]
        value = fixtures[key]
        if value.dtype != torch.int64 or value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 6:
            raise ValueError("original source fixture needs int64 [1,S] with S >= 6")
        selected = value[:, :6].detach().clone().contiguous()
        sequences.append(selected)
        origins.append(dict(source_case=index, source_input=key, original=raw.tensor_record(value),
                            selected=raw.tensor_record(selected), selection="positions 0 through 5"))
    if raw.tensor_record(sequences[0][:, :4]) == raw.tensor_record(sequences[1][:, :4]):
        raise ValueError("the two original prefill prefixes must differ")
    return sequences, origins


def phase_interface(layer_count):
    inputs = [dict(index=0, role="input_ids"), dict(index=1, role="position_ids"),
              dict(index=2, role="attention_mask")]
    outputs = [dict(index=0, role="logits")]
    for layer in range(layer_count):
        for offset, operand in enumerate(("key", "value")):
            inputs.append(dict(index=3 + 2 * layer + offset, role=f"past_{operand}", layer=layer))
            outputs.append(dict(index=1 + 2 * layer + offset, role=f"updated_{operand}", layer=layer))
    links = [dict(producer=PHASES[i - 1][0], consumer=PHASES[i][0], layer=layer,
                  operand=operand, output_index=1 + 2 * layer + offset, input_index=3 + 2 * layer + offset)
             for i in range(1, len(PHASES)) for layer in range(layer_count)
             for offset, operand in enumerate(("key", "value"))]
    phases = {name: dict(inputs=inputs[:3] if past == 0 else inputs, outputs=outputs,
                        **({"initial_cache_empty": True} if past == 0 else {})) for name, past, _ in PHASES}
    return dict(inputs=inputs, outputs=outputs, source_cache_links=links, phases=phases,
                default_input_layout="decode_explicit_cache")


def collect_parity_cases(model, sequences):
    """Require source wrapper/original-forward parity before any capture."""
    from transformers import DynamicCache

    if len(sequences) != 2 or any(value.dtype != torch.int64 or tuple(value.shape) != (1, 6)
                                  for value in sequences):
        raise ValueError("source cache parity requires two six-token int64 [1,6] sequences")
    decode = TensorCacheLogits(model).eval()
    boundaries = {name: TensorPrefillLogits(model).eval() if past == 0 else decode for name, past, _ in PHASES}
    before = module_inventory(model)
    cases = {name: [] for name, _, _ in PHASES}
    comparisons = []
    with raw.prohibit_weight_conversion(), torch.no_grad():
        for case_index, sequence in enumerate(sequences):
            tensors = None
            original_cache = DynamicCache(config=model.config)
            for name, past, query in PHASES:
                if past and (tensors is None or tensors[0].shape[-2] != past):
                    raise ValueError("source cache chain length differs from phase plan")
                ids = sequence[:, past:past + query].contiguous()
                positions, mask = phase_positions_and_mask(ids, past)
                if past == 0:
                    validate_prefill_phase(ids, positions, mask)
                    args = (ids, positions, mask)
                else:
                    validate_phase(ids, positions, mask, tensors)
                    args = (ids, positions, mask, *tensors)
                incoming = raw.inventory({f"input_{i}": value for i, value in enumerate(args)})
                actual = boundaries[name](*args)
                original = model(ids, position_ids=positions, attention_mask=mask,
                                 past_key_values=original_cache, use_cache=True)
                expected = (original.logits, *(value for layer in original_cache.layers
                                               for value in (layer.keys, layer.values)))
                if len(actual) != 1 + 2 * model.config.num_hidden_layers or len(actual) != len(expected):
                    raise ValueError("source cache output roster differs")
                outputs = raw.inventory({f"output_{i}": value for i, value in enumerate(actual)})
                wanted = raw.inventory({f"output_{i}": value for i, value in enumerate(expected)})
                if outputs != wanted:
                    raise ValueError(f"source cache parity differs: case {case_index}, phase {name}")
                if incoming != raw.inventory({f"input_{i}": value for i, value in enumerate(args)}):
                    raise ValueError("source phase mutated its consumed inputs")
                for index, updated in enumerate(actual[1:]):
                    if updated.dtype != torch.bfloat16 or updated.shape[-2] != past + query:
                        raise ValueError("source updated cache dtype or length differs")
                    if past and raw.tensor_record(tensors[index]) != raw.tensor_record(updated[:, :, :past]):
                        raise ValueError("source cache append changed old BF16 words")
                cases[name].append(tuple(value.detach().clone() for value in args))
                comparisons.append(dict(case=case_index, phase=name, status="SOURCE_WORDS_EXACT",
                                        past_length=past, query_length=query, total_length=past + query,
                                        inputs=incoming, outputs=outputs,
                                        original_forward_cache_outputs=wanted, input_mutation=False))
                tensors = actual[1:]
    if module_inventory(model) != before:
        raise ValueError("source state changed during cache parity")
    for name, args in cases.items():
        if len(args) != 2 or record_hash(raw.inventory(dict(enumerate(args[0])))) == record_hash(
                raw.inventory(dict(enumerate(args[1])))):
            raise ValueError(f"phase requires two distinct original-source cases: {name}")
    return boundaries, cases, comparisons, before


def construct_source(bundle, export_dir, architecture, source):
    """Mirror raw intake's architecture setup, reusing its frozen auth helpers."""
    from safetensors.torch import load_file
    import algos
    import atlas_fx_ops  # noqa: F401, register original saved source operators.
    import load_quantized
    import ref_attention
    import ref_profile
    from transformers import AutoConfig, AutoModelForCausalLM

    policy = raw.read_json(export_dir / "policy.json")
    tensors = load_file(str(export_dir / "model.safetensors"))
    raw.validate_policy(policy, tensors)
    frozen = raw.inventory(tensors)
    source_program = torch.export.load(bundle / "model.pt2")
    source_state = raw.authenticate_state(source_program, tensors, policy)
    del source_program
    config = AutoConfig.from_pretrained(str(architecture), local_files_only=True)
    if config.model_type != "llama" or PHASES[-1][1] + PHASES[-1][2] > config.max_position_embeddings:
        raise ValueError("cache phases require the unchanged supported Llama RoPE configuration")
    config.use_cache, config._attn_implementation = False, "eager"
    with raw.prohibit_weight_conversion(), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    replay = load_quantized.PolicyAlgo(policy, tensors)
    with raw.prohibit_weight_conversion(), algos.active(replay):
        raw.reconstruct(model, policy, tensors, replay=replay, linear_class=ref_profile.RefQuantLinear,
                        patch_attention=ref_attention.patch_attention)
    if sha256(json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest() != source[
            "provenance"]["config_sha256"]:
        raise ValueError("reconstructed effective config differs from original source")
    return model, policy, tensors, frozen, source_state


def capture_phase(boundary, cases, directory, *, policy, provenance, tensors, source_state, layer_count):
    """Publish and reauthenticate one original native uint8 source phase."""
    fx = importlib.import_module("fx_export")
    implicit_prefill = len(cases[0]) == 3
    past = 0 if implicit_prefill else cases[0][3].shape[2]
    query = cases[0][0].shape[1]
    expected_inputs = 3 if past == 0 else 3 + 2 * layer_count
    if (past == 0 and not implicit_prefill or provenance.get("past_length", past) != past
            or provenance.get("seqlen", query) != query):
        raise ValueError("prefill capture requires three inputs and phase geometry must match")
    for case in cases:
        if len(case) != expected_inputs or case[0].shape[1] != query:
            raise ValueError("source phase positional input roster differs")
        if implicit_prefill:
            validate_prefill_phase(case[0], case[1], case[2])
        else:
            validate_phase(case[0], case[1], case[2], case[3:])
            if case[3].shape[2] != past:
                raise ValueError("source phase cache length differs between cases")
    incoming = [raw.inventory({f"input_{i}": value for i, value in enumerate(case)}) for case in cases]
    with raw.prohibit_weight_conversion():
        verification = fx.capture_fx(boundary, cases, directory, policy=policy, provenance=provenance,
                                     native_fp8="uint8", fresh_process=True)
    if incoming != [raw.inventory({f"input_{i}": value for i, value in enumerate(case)}) for case in cases]:
        raise ValueError("source capture changed its admitted phase inputs")
    native = verification.get("native_fp8_detail", {})
    fresh = verification.get("fresh_process_reload", {})
    if (verification.get("status") != "passed" or not verification.get("fresh_process_requested")
            or native.get("storage") != "uint8" or native.get("code_bytes_equal_uint8_source") is not True
            or native.get("outputs_identical_to_decode_graph") is not True
            or fresh.get("status") != "passed" or fresh.get("all_outputs_byte_identical") is not True
            or fresh.get("code_scale_bytes_preserved") is not True or fresh.get("model_replay_modules_imported") != []):
        raise ValueError("phase capture lacks original native uint8 and fresh-process qualification")
    if raw.read_json(directory / "fresh_process_verification.json") != fresh:
        raise ValueError("phase fresh-process verification publication differs")
    manifest = raw.read_json(directory / "fx_manifest.json")
    raw.verify_files(directory, manifest["files"], fx.BUNDLE_FILES)
    program = torch.export.load(directory / "model.pt2")
    if raw.authenticate_state(program, tensors, policy) != source_state:
        raise ValueError("phase source state differs from authenticated original source")
    user_inputs = [spec for spec in program.graph_signature.input_specs if spec.kind.name == "USER_INPUT"]
    user_outputs = [spec for spec in program.graph_signature.output_specs if spec.kind.name == "USER_OUTPUT"]
    if len(user_inputs) != expected_inputs or len(user_outputs) != 1 + 2 * layer_count:
        raise ValueError("saved phase tensor interface differs from the explicit cache ABI")
    abi = derive_cache_phase_abi(program, layer_count=layer_count, past_length=past,
        query_length=query, kv_heads=getattr(boundary, "kv_heads", None),
        head_dim=getattr(boundary, "head_dim", None), batch_size=cases[0][0].shape[0])
    model_hash = raw.file_hash(directory / "model.pt2")
    manifest_hash = raw.file_hash(directory / "fx_manifest.json")
    binding = dict(schema="atlas.source_cache_phase_binding.v1", model_pt2_sha256=model_hash,
                   source_fx_manifest_sha256=manifest_hash, abi=abi)
    (directory / "cache_phase_abi.json").write_text(json.dumps(binding, indent=2) + "\n")
    return dict(status="SOURCE_CAPTURED", manifest_sha256=manifest_hash, model_pt2_sha256=model_hash,
                verification_sha256=raw.file_hash(directory / "verification.json"),
                fresh_process_verification_sha256=raw.file_hash(directory / "fresh_process_verification.json"),
                cache_phase_abi_sha256=raw.file_hash(directory / "cache_phase_abi.json"),
                source_state_sha256=record_hash(source_state), input_count=len(user_inputs),
                output_count=len(user_outputs), generated_execution="UNRUN", initial_cache_empty=implicit_prefill)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ("bundle", "export-dir", "architecture", "source-shape-gate", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--parity-only", action="store_true", help="stop after two source cache chains; no FX export")
    args = parser.parse_args(argv)
    bundle, export_dir, architecture, gate, output = map(raw.checked_path, (
        args.bundle, args.export_dir, args.architecture, args.source_shape_gate, args.out))
    if output.exists():
        parser.error("a fresh output directory is required")
    # Refuse an absent or still-running raw64 gate before large source intake.
    preliminary = raw.read_json(gate)
    if preliminary.get("status") != "SOURCE_PARITY_CAPTURED" or preliminary.get("seqlen") != 64:
        raise ValueError("completed raw64 source parity capture is required before cache phases")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source, identity = raw.source_identity(bundle, export_dir, architecture)
    if source["provenance"]["seqlen"] != 64:
        raise ValueError("cache capture currently requires the original fixed64 source bundle")
    raw.verify_source_gate(gate, identity, 64)
    output.mkdir(parents=True)
    result = dict(schema="atlas.source_cache_capture.v1", status="RUNNING", raw_identity=identity,
                  source_shape_gate_sha256=raw.file_hash(gate),
                  cache_capture_sha256=raw.file_hash(__file__),
                  cache_wrapper_sha256=raw.file_hash(importlib.import_module("cache_source").__file__),
                  cache_roles_sha256=raw.file_hash(importlib.import_module("cache_roles").__file__),
                  execution_scope="original source framework only", native_fx_scope="source framework emulation only",
                  generated_execution="UNRUN", independent_atlas_exactness="UNRUN", quality="UNRUN",
                  weight_conversion_performed=False, pack_fold_performed=False, phases={})
    def save():
        (output / "cache_capture.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        print(json.dumps(dict(stage="AUTHENTICATED_RAW64_GATE")), flush=True)
        model, policy, tensors, frozen, source_state = construct_source(bundle, export_dir, architecture, source)
        frozen_policy = record_hash(policy)
        result["source_policy_sha256"] = frozen_policy
        from safetensors.torch import load_file
        sequences, origins = fixture_sequences(load_file(str(bundle / "fixtures.safetensors")),
                                              raw.read_json(bundle / "parity.json"))
        result.update(source_fixture_sequences=origins, source_state=source_state,
                      interface=phase_interface(model.config.num_hidden_layers))
        print(json.dumps(dict(stage="SOURCE_CACHE_PARITY")), flush=True)
        boundaries, cases, comparisons, before = collect_parity_cases(model, sequences)
        result.update(source_cache_parity=comparisons, source_cache_parity_cases=2,
                      source_phase_comparisons=len(comparisons), status="SOURCE_CACHE_PARITY")
        save()
        if not args.parity_only:
            for name, past, query in PHASES:
                print(json.dumps(dict(stage="CAPTURE_SOURCE_PHASE", phase=name)), flush=True)
                result["phases"][name] = capture_phase(boundaries[name], cases[name], output / name, policy=policy,
                    provenance={**source["provenance"], "boundary": "causal_lm_tensor_bf16_cache",
                        "seqlen": query, "past_length": past, "total_length": past + query, "phase": name,
                        "raw_capture": identity, "cache_interface": result["interface"]["phases"][name],
                        "source_shape_gate_sha256": result["source_shape_gate_sha256"],
                        "cache_capture_sha256": result["cache_capture_sha256"],
                        "cache_wrapper_sha256": result["cache_wrapper_sha256"],
                        "cache_roles_sha256": result["cache_roles_sha256"]},
                    tensors=tensors, source_state=source_state, layer_count=model.config.num_hidden_layers)
                if (module_inventory(model) != before or raw.inventory(tensors) != frozen
                        or record_hash(policy) != frozen_policy):
                    raise ValueError("immutable state changed during source phase capture")
                save()
        if (module_inventory(model) != before or raw.inventory(tensors) != frozen
                or record_hash(policy) != frozen_policy):
            raise ValueError("immutable state changed during source cache work")
        result["status"] = "SOURCE_CACHE_PARITY_ONLY" if args.parity_only else "SOURCE_CACHE_CAPTURED"
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        save()
        raise
    save()
    print(json.dumps({key: result[key] for key in ("schema", "status", "generated_execution")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
