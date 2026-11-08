"""Recapture a causal LM from authenticated raw codes, without weight conversion.

Intake is CPU-only and refuses unsupported policy features. A successful intake is
not source parity, generated execution, or quality evidence. Capturing a different
length requires a separately completed source-shape parity capture first.
"""
from __future__ import annotations

import argparse
import contextlib
from hashlib import sha256
import importlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import torch
from torch import nn


def checked_path(path):
    path = Path(path).absolute()
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ValueError("source and destination paths must not use symlinks")
    return path.resolve(strict=False)


def file_hash(path):
    digest = sha256()
    with checked_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(value):
    if value.device.type == "meta":
        raise ValueError("meta tensor has no authenticated payload")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(value.dtype), shape=list(value.shape), bytes=len(raw), sha256=sha256(raw).hexdigest())


def inventory(tensors):
    return {name: tensor_record(value) for name, value in sorted(tensors.items())}


def read_json(path):
    return json.loads(checked_path(path).read_text())


def verify_files(root, files, allowed):
    root = checked_path(root)
    if set(files) != set(allowed):
        raise ValueError("unexpected or incomplete source manifest file roster")
    for name, record in files.items():
        if Path(name).name != name or file_hash(root / name) != record["sha256"]:
            raise ValueError(f"source file hash mismatch: {name}")


def validate_policy(policy, tensors):
    if policy.get("exported_mode") != "ref_conv" or policy.get("aliases"):
        raise ValueError("only unfolded ref_conv with no parameter aliases is supported")
    if policy.get("pack_fold") or policy.get("pack_fold_applied"):
        raise ValueError("folded policy is unsupported")
    count = 0
    for name, site in policy["sites"].items():
        if site.get("status") != "quantized":
            if site.get("status") != "skipped":
                raise ValueError(f"unsupported site status: {name}")
            continue
        if site.get("pv_pop_shift", 0) != 0 or site.get("residual_passes"):
            raise ValueError(f"folded PV or residual policy is unsupported: {name}")
        if site["kind"] == "attention":
            if site.get("variant") != "text" or set(site["operands"]) != {"q", "k", "p", "v"}:
                raise ValueError(f"unsupported attention policy: {name}")
            rules = site["operands"].values()
        elif site["kind"] == "linear":
            weight = site["weight"]
            if weight.get("block", 0) or site.get("weight_dtype") != "bfloat16" or site.get("bias") is not None:
                raise ValueError(f"unsupported weight block, dtype, or bias: {name}")
            codes, exponent = tensors[weight["codes"]], tensors[weight["scale_exp"]]
            shape = tuple(site["weight_shape"])
            if codes.dtype != torch.uint8 or exponent.dtype != torch.int32 or codes.ndim != 2:
                raise ValueError(f"unsupported code/exponent dtype or rank: {name}")
            if tuple(codes.shape) != shape or tuple(exponent.shape) not in ((), (1,), (shape[0], 1)):
                raise ValueError(f"weight code/exponent shape mismatch: {name}")
            rules = [site["activation"]]
            count += 1
        else:
            raise ValueError(f"unsupported quantized site kind: {name}")
        for rule in rules:
            if rule.get("mode") not in ("static", "dynamic"):
                raise ValueError(f"mutable or unsupported activation rule: {name}")
            if rule.get("mode") == "static" and type(rule.get("exp")) is not int:
                raise ValueError(f"invalid static activation exponent: {name}")
    if not count:
        raise ValueError("policy contains no raw quantized linear sites")


def raw_linear(linear, name, site, tensors, replay, linear_class):
    """Bypass the source constructor, whose ordinary behavior quantizes weights."""
    if type(linear) is not nn.Linear or list(linear.weight.shape) != site["weight_shape"]:
        raise ValueError(f"architecture linear mismatch: {name}")
    if linear.weight.dtype != torch.bfloat16 or linear.bias is not None:
        raise ValueError(f"architecture dtype or bias mismatch: {name}")
    result = linear_class.__new__(linear_class)
    nn.Module.__init__(result)
    result.fm, result.algo, result.layer_name = None, replay, name
    result.in_features, result.out_features = linear.in_features, linear.out_features
    # Only dtype/shape are used by source forwards. The placeholder has no payload.
    result.weight = nn.Parameter(torch.empty(tuple(linear.weight.shape), dtype=linear.weight.dtype, device="meta"),
                                 requires_grad=False)
    result.register_parameter("bias", None)
    result.weight_block, result.residual = 0, None
    result.register_buffer("weight_codes", tensors[site["weight"]["codes"]].clone(), persistent=False)
    result.register_buffer("weight_scale_exp", tensors[site["weight"]["scale_exp"]].clone(), persistent=False)
    return result


def get_module(model, path):
    return model.get_submodule(path) if path else model


def reconstruct(model, policy, tensors, *, replay, linear_class, patch_attention):
    """Install raw state on an architecture-only model; perform no forwards."""
    validate_policy(policy, tensors)
    before = inventory(tensors)
    sites = {name: site for name, site in policy["sites"].items()
             if site.get("status") == "quantized" and site["kind"] == "linear"}
    quant_keys = {f"{name}.weight" for name in sites}
    suffixes = (".weight_codes", ".weight_scale_exp", ".input_scale_exp")
    ordinary = {name: value for name, value in tensors.items() if not name.endswith(suffixes)}
    expected = set(model.state_dict()) - quant_keys
    if expected != set(ordinary):
        raise ValueError(f"architecture tensor roster mismatch: missing={sorted(expected - set(ordinary))}, "
                         f"extra={sorted(set(ordinary) - expected)}")
    for name, value in ordinary.items():
        parent, _, attr = name.rpartition(".")
        module = get_module(model, parent)
        original = getattr(module, attr)
        if original.dtype != value.dtype or tuple(original.shape) != tuple(value.shape):
            raise ValueError(f"architecture tensor dtype/shape mismatch: {name}")
        if isinstance(original, nn.Parameter):
            setattr(module, attr, nn.Parameter(value.clone(), requires_grad=False))
        elif attr in module._buffers:
            module._buffers[attr] = value.clone()
        else:
            raise ValueError(f"architecture state is not a parameter or buffer: {name}")
    for name, site in sites.items():
        parent, _, attr = name.rpartition(".")
        module = get_module(model, parent)
        setattr(module, attr, raw_linear(getattr(module, attr), name, site, tensors, replay, linear_class))
    actual_attention = patch_attention(model)
    declared_attention = {name for name, site in policy["sites"].items()
                          if site.get("status") == "quantized" and site["kind"] == "attention"}
    if {site["site"] for site in actual_attention} != declared_attention:
        raise ValueError("architecture attention policy roster mismatch")
    if inventory(tensors) != before:
        raise ValueError("source tensors changed during raw reconstruction")
    model.eval()
    return model


def authenticate_state(program, tensors, policy):
    """Bind all non-placeholder source state to original exported tensor bytes."""
    expected = {}
    # Export keeps nonpersistent raw buffers in lifted constants, not state_dict.
    state = dict(program.state_dict)
    for name, value in program.named_buffers():
        if name in state and tensor_record(value) != tensor_record(state[name]):
            raise ValueError(f"source buffer and state disagree: {name}")
        state[name] = value
    for name, value in state.items():
        if value.device.type == "meta":
            source_name = name.removeprefix("model.")
            site = policy["sites"].get(source_name.removesuffix(".weight"))
            if not name.endswith(".weight") or not site or site.get("status") != "quantized":
                raise ValueError(f"unexpected source meta placeholder: {name}")
            continue
        source_name = name.removeprefix("model.")
        if source_name not in tensors or tensor_record(value) != tensor_record(tensors[source_name]):
            raise ValueError(f"original FX state differs from raw export: {name}")
        expected[name] = tensor_record(value)
    quantized = {f"model.{site['weight'][key]}" for site in policy["sites"].values()
                 if site.get("status") == "quantized" and site["kind"] == "linear"
                 for key in ("codes", "scale_exp")}
    if not quantized <= set(expected):
        raise ValueError(f"original FX graph omits declared code/exponent state: {sorted(quantized - set(expected))}")
    ordinary = {f"model.{name}" for name in tensors if not name.endswith(
        (".weight_codes", ".weight_scale_exp", ".input_scale_exp"))}
    if set(expected) != ordinary | quantized:
        raise ValueError("original FX graph omits or adds raw exported state")
    return expected


@contextlib.contextmanager
def prohibit_weight_conversion():
    """Tripwires cover source entry points that would rederive or fold weights."""
    targets = ("algos.base.weight_cast", "quant_modes.quantize", "load_quantized.apply_export",
               "pack_fold.apply_folds", "pack_fold.fold_export_dir")
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, side_effect=RuntimeError(f"immutable weight conversion forbidden: {target}")))
        yield


def source_identity(bundle, export_dir, architecture):
    fx = importlib.import_module("fx_export")
    source = read_json(bundle / "fx_manifest.json")
    verify_files(bundle, source["files"], fx.BUNDLE_FILES)
    exported = read_json(export_dir / "manifest.json")
    verify_files(export_dir, exported["files"], ("model.safetensors", "policy.json"))
    export_hash = file_hash(export_dir / "manifest.json")
    if source["provenance"]["export_manifest_sha256"] != export_hash:
        raise ValueError("raw export is not the original FX source export")
    config_hash = file_hash(architecture / "config.json")
    if exported["model"]["checkpoint"]["config_json_sha256"] != config_hash:
        raise ValueError("architecture config differs from authenticated raw export")
    if exported["model"]["kind"] != "causal_lm" or source["provenance"]["boundary"] != "causal_lm":
        raise ValueError("only causal-LM source intake is supported")
    if source["native_fp8"]["storage"] != "uint8":
        raise ValueError("raw source capture currently requires original uint8 code carriers")
    modules = ("fx_export", "load_quantized", "ref_profile", "ref_attention", "atlas_fx_ops", "algos.base")
    return source, dict(source_manifest_sha256=file_hash(bundle / "fx_manifest.json"),
        export_manifest_sha256=export_hash, config_file_sha256=config_hash,
        loader_sha256=file_hash(__file__),
        source_modules={name: file_hash(importlib.import_module(name).__file__) for name in modules},
        python=platform.python_version(), torch=torch.__version__,
        transformers=importlib.import_module("transformers").__version__)


def verify_source_gate(path, identity, source_length):
    gate = read_json(path)
    if (gate.get("schema") != "atlas.raw_source_capture.v1" or gate.get("status") != "SOURCE_PARITY_CAPTURED"
            or gate.get("identity") != identity or gate.get("seqlen") != source_length
            or gate.get("source_shape_parity_cases", 0) < 2):
        raise ValueError("missing, stale, or incomplete original-shape source parity gate")
    parent = checked_path(path).parent
    if file_hash(parent / "fx_manifest.json") != gate["capture_manifest_sha256"]:
        raise ValueError("source parity capture manifest changed")
    manifest = read_json(parent / "fx_manifest.json")
    verify_files(parent, manifest["files"], importlib.import_module("fx_export").BUNDLE_FILES)
    if file_hash(parent / "verification.json") != gate["verification_sha256"]:
        raise ValueError("source parity verification changed")
    verification = read_json(parent / "verification.json")
    if verification.get("status") != "passed" or not verification.get("fresh_process_requested"):
        raise ValueError("source parity capture was not freshly verified")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-shape-gate", type=Path)
    parser.add_argument("--intake-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    bundle, export_dir, architecture, output = map(checked_path, (args.bundle, args.export_dir, args.architecture, args.out))
    if output.exists() or args.seqlen < 2:
        parser.error("a fresh output path and seqlen >= 2 are required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(dict(stage="AUTHENTICATE_FILES")), flush=True)
    source, identity = source_identity(bundle, export_dir, architecture)
    source_length = source["provenance"]["seqlen"]
    if args.seqlen != source_length and not args.intake_only:
        if args.source_shape_gate is None:
            parser.error("different-length capture requires --source-shape-gate from original-length capture")
        verify_source_gate(args.source_shape_gate, identity, source_length)
    from safetensors.torch import load_file
    import algos
    import atlas_fx_ops  # noqa: F401, register original source operators before load.
    import fx_export
    import load_quantized
    import ref_attention
    import ref_profile
    from transformers import AutoConfig, AutoModelForCausalLM
    policy = read_json(export_dir / "policy.json")
    tensors = load_file(str(export_dir / "model.safetensors"))
    validate_policy(policy, tensors)
    frozen = inventory(tensors)
    print(json.dumps(dict(stage="LOAD_ORIGINAL_SOURCE_ARCHIVE")), flush=True)
    program = torch.export.load(bundle / "model.pt2")
    source_state = authenticate_state(program, tensors, policy)
    print(json.dumps(dict(stage="BUILD_RAW_ARCHITECTURE", authenticated_tensors=len(source_state))), flush=True)
    config = AutoConfig.from_pretrained(str(architecture), local_files_only=True)
    if args.seqlen > config.max_position_embeddings:
        raise ValueError("capture would require changing the original RoPE configuration")
    config.use_cache, config._attn_implementation = False, "eager"
    with prohibit_weight_conversion(), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    replay = load_quantized.PolicyAlgo(policy, tensors)
    with prohibit_weight_conversion(), algos.active(replay):
        reconstruct(model, policy, tensors, replay=replay, linear_class=ref_profile.RefQuantLinear,
                    patch_attention=ref_attention.patch_attention)
    boundary = fx_export.CausalLogits(model).eval()
    effective_hash = sha256(json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    if effective_hash != source["provenance"]["config_sha256"]:
        raise ValueError("reconstructed effective config differs from original source capture")
    print(json.dumps(dict(stage="RAW_INTAKE_AUTHENTICATED")), flush=True)
    result = dict(schema="atlas.raw_source_capture.v1", status="INTAKE_ONLY", identity=identity,
                  seqlen=args.seqlen, source_shape_parity_cases=0,
                  weight_conversion_performed=False, pack_fold_performed=False,
                  source_state=source_state, raw_export_tensors=frozen)
    if args.intake_only:
        output.mkdir(parents=True)
    else:
        fixtures = load_file(str(bundle / "fixtures.safetensors"))
        parity = read_json(bundle / "parity.json")
        if args.seqlen == source_length:
            cases = [tuple(fixtures[name] for name in case["inputs"]) for case in parity["cases"]]
            with prohibit_weight_conversion(), torch.no_grad():
                for inputs, case in zip(cases, parity["cases"], strict=True):
                    actual = boundary(*inputs)
                    expected = [fixtures[name] for name in case["outputs"]]
                    if len(expected) != 1 or tensor_record(actual) != tensor_record(expected[0]):
                        raise ValueError("raw reconstruction differs from original source fixture")
            result["source_shape_parity_cases"] = len(cases)
        else:
            cases = [(torch.randint(model.config.vocab_size, (1, args.seqlen),
                      generator=torch.Generator().manual_seed(args.seed + index)),) for index in range(2)]
        with prohibit_weight_conversion():
            fx_export.capture_fx(boundary, cases, output, policy=policy, native_fp8="uint8",
                provenance={**source["provenance"], "seqlen": args.seqlen, "raw_capture": identity,
                            "seed": args.seed}, fresh_process=True)
        result.update(status="SOURCE_PARITY_CAPTURED" if args.seqlen == source_length else "RECAPTURED_UNEXECUTED",
                      capture_manifest_sha256=file_hash(output / "fx_manifest.json"),
                      verification_sha256=file_hash(output / "verification.json"))
        authenticate_state(torch.export.load(output / "model.pt2"), tensors, policy)
    if inventory(tensors) != frozen:
        raise ValueError("raw export tensors changed during capture")
    (output / "raw_capture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("schema", "status", "seqlen", "source_shape_parity_cases")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
