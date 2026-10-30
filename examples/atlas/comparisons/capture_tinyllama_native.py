"""Capture a source-native layer-zero projection without loading the full model.

Only the embedding, first RMSNorm and query weights are materialized. The pinned
native decoder is stopped by a query forward hook, before attention or later
layers. This is an experiment input capture, not a model-quality evaluation.
"""

from merlin.common.jsonio import write_pretty_json
import argparse
import importlib.util
import inspect
import json
import os
import resource
import stat
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
from source_relocation import physical_path as safe_path
from atlas_provider_files import (
    LEGACY_HELPER_MODULES, LEGACY_PRODUCER_FILES, LEGACY_PROVIDER_REQUIREMENT, LEGACY_VALIDATOR_FILE,
)

import numpy as np

from merlin.common.digest import sha256_bytes, sha256_file
from merlin.common.paths import artifacts_dir


REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
WEIGHT_SHA = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
CORPUS_SHA = "573e1d244efc95bd7bb4ee3df0735bfdd688d3abf7f70a849ac2061c6de8a4d9"
TOKEN_SHA = "24e1d064a71f5aa75b560cb0881cab1fac5312cb3c534d8e6c0eaa0528f4231a"
LLAMA_SHA = "46d313ec0f1116bcc1bb7f2379467ab249c140415c64fb758e2d5915cd8b89a5"
CACHE_SHA = "6ee7388c7bf9e270df19c33d32b963895ff16e5c08ffd068eb3ffd45d147f700"


def admit_archive_members(members):
    """Screen central-directory metadata before hashing or opening member data."""
    seen = set()
    for member in members:
        name = member.filename
        parts = name.rstrip("/").split("/")
        if (not name or getattr(member, "orig_filename", name) != name
                or name in seen or name.startswith("/") or "\\" in name
                or any(part in ("", ".", "..") or ":" in part for part in parts)
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError("Unsafe or duplicate archive member metadata")
        seen.add(name)


def digest(path):
    path = safe_path(path)
    # Torch checkpoints may be ZIPs despite their .pt/.pth suffix. Detect the
    # container, rather than relying on a filename extension or extracting it.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            admit_archive_members(archive.infolist())
    return sha256_file(path)


def identity(path, expected=None):
    path = safe_path(path)
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError(f"Source hash differs: {path}")
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def artifact_output(path):
    path, root = safe_path(path), safe_path(artifacts_dir())
    if path == root or not path.is_relative_to(root):
        raise ValueError("Output must be new and beneath the configured artifact root")
    return path


def load_provider_validator(producer_root, validator_sha256, receipt_path, receipt_sha256):
    """Bind the explicit producer tuple before importing its admission code."""
    def pinned(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    if not pinned(validator_sha256) or not pinned(receipt_sha256):
        raise ValueError("Require explicit validator and receipt SHA256 pins")
    root = safe_path(producer_root) / "tools"
    receipt_id = identity(receipt_path, receipt_sha256)
    receipt = json.loads(Path(receipt_id["path"]).read_text())
    code = receipt.get("code_before")
    if (receipt.get("status") != "passed_model_only" or not isinstance(code, dict)
            or set(code) != set(LEGACY_PRODUCER_FILES)
            or code != receipt.get("code_after") or not all(pinned(sha) for sha in code.values())):
        raise ValueError(LEGACY_PROVIDER_REQUIREMENT + "; require an unchanged complete producer source tuple")
    validator_id = identity(root / LEGACY_VALIDATOR_FILE, validator_sha256)
    bindings = [receipt_id, validator_id, *[identity(root / name, sha) for name, sha in code.items()]]
    inserted = []
    try:
        # These are the validator's only local imports. Refuse a cached module
        # from another invocation/root instead of trusting Python's name cache.
        for name in LEGACY_HELPER_MODULES:
            record = identity(root / (name + ".py"), code[name + ".py"])
            module = sys.modules.get(name)
            if module is not None:
                if getattr(module, "_native_source_identity", None) != record:
                    raise ValueError("Cached provider helper lacks the selected source binding")
                continue
            spec = importlib.util.spec_from_file_location(name, record["path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            inserted.append(name)
            spec.loader.exec_module(module)
            module._native_source_identity = record
        spec = importlib.util.spec_from_file_location("native_physical_admission", validator_id["path"])
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        if any(not callable(getattr(validator, name, None)) for name in ("bind", "recheck", "admit_contraction")):
            raise ValueError(LEGACY_PROVIDER_REQUIREMENT)
        # This invocation's existing validator keeps its physical/schema logic;
        # every delegated bind/recheck now uses the same strict archive-first IO.
        validator.sha256 = digest
        validator.physical_path = safe_path
        original_recheck = validator.recheck

        def recheck(records):
            original_recheck(records)
            verify_declared_sources(bindings)

        validator.recheck = recheck
        recheck([])
        return validator, validator_id
    except BaseException:
        for name in reversed(inserted):
            sys.modules.pop(name, None)
        raise


def verify_declared_sources(value):
    """Recheck explicit metadata records, without discovering any new paths."""
    count = 0
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            identity(value["path"], value["sha256"])
            count += 1
        else:
            for child in value.values():
                count += verify_declared_sources(child)
    elif isinstance(value, list):
        for child in value:
            count += verify_declared_sources(child)
    return count


def array_identity(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": sha256_bytes(array.tobytes())}


def bf16_words_to_float(words):
    return (words.astype(np.uint32) << 16).view(np.float32)


def round_bf16_words(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source or diagnostic")
    words = values.view(np.uint32)
    return ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def projection_bounds(full_query=False):
    return (128, 2048) if full_query else (32, 32)


def validate_arrays(arrays, full_query=False):
    rows, columns = projection_bounds(full_query)
    shapes = {"A": (rows, 2048), "B": (2048, columns), "native_C": (rows, columns)}
    for name, shape in shapes.items():
        value, words = arrays[name], arrays[name + "_bf16"]
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"Invalid {name} float ABI")
        if words.shape != shape or words.dtype != np.uint16:
            raise ValueError(f"Invalid {name} raw-word ABI")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}")
        if not np.array_equal(value.view(np.uint32), bf16_words_to_float(words).view(np.uint32)):
            raise ValueError(f"BF16 words disagree with {name}")


def check_memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > 2 * 1024**3:
        raise MemoryError("Selective capture exceeded its 2 GiB RSS lease")
    return peak


def capture(args):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, cache_utils
    from transformers.models.llama import modeling_llama
    from merlin.common.paths import artifacts_dir

    started = time.monotonic()
    full_query = getattr(args, "full_query", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if transformers.__version__ != "5.4.0":
        raise ValueError("The frozen Transformers 5.4.0 implementation is required")
    source = {
        "checkpoint_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "revision": REVISION,
        "weights": identity(args.checkpoint / "model.safetensors", WEIGHT_SHA),
        "corpus": identity(args.corpus, CORPUS_SHA),
        "modeling_llama": identity(modeling_llama.__file__, LLAMA_SHA),
        "cache_utils": identity(cache_utils.__file__, CACHE_SHA),
        "capture_script": identity(__file__),
        "native_linear": identity(inspect.getfile(torch.nn.Linear)),
        "torch_cpu_library": identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so"),
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "source_policies": {"precision": "bfloat16", "quantized": False,
                            "arithmetic_policy": "native", "rotary_policy": "native",
                            "silu_export_policy": "native", "chat_template": False},
    }
    checkpoint_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    source["checkpoint_files"] = [identity(args.checkpoint / name) for name in checkpoint_files]
    config_json = json.loads((args.checkpoint / "config.json").read_text())
    expected = {"hidden_size": 2048, "num_hidden_layers": 22, "num_attention_heads": 32,
                "num_key_value_heads": 4, "vocab_size": 32000, "attention_bias": False,
                "rms_norm_eps": 1e-5, "torch_dtype": "bfloat16"}
    if any(config_json.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned checkpoint configuration differs")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    tokens = tokenizer(args.corpus.read_text(encoding="utf-8"), return_tensors="pt",
                       add_special_tokens=True)["input_ids"][:, :160].contiguous()
    if tokens.shape != (1, 160) or array_identity(tokens.numpy())["sha256"] != TOKEN_SHA:
        raise ValueError("Frozen raw-text development token identity differs")
    source["tokens"] = array_identity(tokens.numpy())
    token_before = tokens.clone()
    config = modeling_llama.LlamaConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        layer = modeling_llama.LlamaDecoderLayer(config, 0)
    tensor_keys = ("model.embed_tokens.weight", "model.layers.0.input_layernorm.weight",
                   "model.layers.0.self_attn.q_proj.weight")
    with safe_open(str(args.checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        loaded = {name: handle.get_tensor(name) for name in tensor_keys}
    if any(value.dtype != torch.bfloat16 for value in loaded.values()):
        raise ValueError("Selected native parameters must retain BF16 checkpoint words")
    embedding.weight = torch.nn.Parameter(loaded[tensor_keys[0]], requires_grad=False)
    layer.input_layernorm.weight = torch.nn.Parameter(loaded[tensor_keys[1]], requires_grad=False)
    layer.self_attn.q_proj.weight = torch.nn.Parameter(loaded[tensor_keys[2]], requires_grad=False)
    embedding.eval()
    layer.eval()
    parameter_before = {name: array_identity(value.view(torch.uint16).numpy()) for name, value in loaded.items()}
    observed = {}

    class ProjectionCaptured(Exception):
        pass

    def stop_after_query(_module, inputs, output):
        observed["input"] = inputs[0].detach().clone()
        observed["output"] = output.detach().clone()
        raise ProjectionCaptured()

    hook = layer.self_attn.q_proj.register_forward_hook(stop_after_query)
    try:
        with torch.inference_mode():
            hidden = embedding(tokens[:, :128])
            hidden_before = hidden.clone()
            for _ in range(2):
                try:
                    layer(hidden)
                except ProjectionCaptured:
                    pass
                else:
                    raise AssertionError("Native decoder did not execute its query projection")
                current = {name: array_identity(value.view(torch.uint16).numpy()) for name, value in observed.items()}
                if _ == 0:
                    first = current
                elif current != first:
                    raise AssertionError("Native repeated projection changed")
    finally:
        hook.remove()
    if not torch.equal(hidden, hidden_before) or not torch.equal(tokens, token_before):
        raise AssertionError("Native capture changed caller inputs")
    parameter_after = {name: array_identity(value.view(torch.uint16).numpy()) for name, value in loaded.items()}
    if parameter_before != parameter_after:
        raise AssertionError("Native capture changed source parameters")
    rows, columns = projection_bounds(full_query)
    a = observed["input"][0, :rows].contiguous()
    b = loaded[tensor_keys[2]][:columns].T.contiguous()
    c = observed["output"][0, :rows, :columns].contiguous()
    arrays = {}
    for name, value in (("A", a), ("B", b), ("native_C", c)):
        arrays[name] = value.float().numpy().copy()
        arrays[name + "_bf16"] = value.view(torch.uint16).numpy().copy()
    validate_arrays(arrays, full_query)
    # The exact BF16 operands make a separately accumulated FP64 dot a useful
    # diagnostic. Native CPU accumulation/rounding remains the reference.
    ideal = arrays["A"].astype(np.float64) @ arrays["B"].astype(np.float64)
    ideal_words = round_bf16_words(ideal)
    mismatches = int(np.count_nonzero(ideal_words != arrays["native_C_bf16"]))
    native_error = np.abs(arrays["native_C"].astype(np.float64) - ideal)
    wrong_b = arrays["B"].copy()
    wrong_b[[0, 1]] = wrong_b[[1, 0]]
    wrong_k = arrays["A"].astype(np.float64) @ wrong_b.astype(np.float64)
    controls = {
        "wrong_k_order_detected": bool(np.any(round_bf16_words(wrong_k) != arrays["native_C_bf16"])),
        "wrong_output_channels_detected": not torch.equal(c, torch.roll(c, shifts=1, dims=1)),
        "wrong_token_order_detected": array_identity(tokens.flip(1).numpy())["sha256"] != TOKEN_SHA,
    }
    for key, change in (("raw_word_mismatch_refused", "word"), ("wrong_shape_refused", "shape"),
                        ("nonfinite_refused", "nonfinite")):
        altered = dict(arrays)
        if change == "word":
            altered["A_bf16"] = arrays["A_bf16"].copy()
            altered["A_bf16"][0, 0] ^= np.uint16(1)
        elif change == "shape":
            altered["B"] = arrays["B"][:, :-1].copy()
        else:
            altered["A"] = arrays["A"].copy()
            altered["A"][0, 0] = np.nan
        try:
            validate_arrays(altered, full_query)
        except ValueError:
            controls[key] = True
        else:
            controls[key] = False
    if not all(controls.values()):
        raise AssertionError(f"A capture negative control failed: {controls}")
    verified_sources = verify_declared_sources(source)
    capture_name = "tinyllama-native-q-full" if full_query else "tinyllama-native-t1"
    out = artifact_output(args.out if args.out else artifacts_dir() / "compare" / "workloads" / capture_name)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "operands.npz", **arrays)
    report = {
        "schema_version": 1, "status": "native_selective_capture_passed",
        "npz_sha256": digest(out / "operands.npz"), "source_dtype": "bfloat16", "source": source,
        "site": {"module": "model.layers.0.self_attn.q_proj", "batch": 0,
                 "positions": [0, rows], "output_channels": [0, columns], "k_range": [0, 2048],
                 "native_execution_shape": [1, 128, 2048], "full_k_reduction": True},
        "capture": {"route": "pinned_native_Embedding_then_LlamaDecoderLayer_stopped_after_q_proj",
                    "materialized_parameter_keys": list(tensor_keys), "parameters": parameter_before,
                    "native_repeats_raw_exact": True, "caller_inputs_preserved": True,
                    "native_query_input_raw_bf16": first["input"],
                    "native_query_output_raw_bf16": first["output"],
                    "full_query_captured": full_query,
                    "parameters_preserved": True, "source_artifacts_preserved": True,
                    "declared_source_bookends_verified": verified_sources,
                    "full_model_executed": False, "full_session_executed": False},
        "arrays": {key: array_identity(value) for key, value in arrays.items()},
        "independent_dot": {"kind": "numpy_FP64_from_original_BF16_operands_diagnostic",
                            "rounded_bf16_mismatches": mismatches,
                            "max_abs_native_vs_ideal": float(native_error.max()),
                            "native_reference_replaced": False},
        "negative_controls": controls, "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": check_memory(), "cpu_threads": 1,
        "claim_limits": ["selective_native_projection_capture", "no_target_execution",
                         "no_full_model_or_language_quality_claim"],
    }
    write_pretty_json(out / "manifest.json", report, sort_keys=False)
    print(json.dumps({"status": report["status"], "out": str(out),
                      "npz_sha256": report["npz_sha256"], "independent_dot": report["independent_dot"],
                      "negative_controls": controls, "wall_seconds": report["wall_seconds"],
                      "peak_rss_bytes": report["peak_rss_bytes"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=safe_path, required=True)
    parser.add_argument("--corpus", type=safe_path, required=True)
    parser.add_argument("--out", type=artifact_output)
    parser.add_argument("--full-query", action="store_true",
                        help="Capture all 128 prefill positions and 2048 query outputs; default is a 32x32 tile")
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
