#!/usr/bin/env python3
"""Validate a frozen model capsule against its loader and recover its export ABI.

This is deliberately *not* a capture worker.  It never invokes model2MLIR, a model hub, or a
quantizer and it writes no model artifact.  The operator-side capsule runner invokes it with the
Python that owns torch because the ordinary Merlin environment intentionally need not contain
torch.  Its only inputs are files in one already-frozen capsule plus the input/golden arrays that
the parent decoded from that capsule's ``golden.yaml``.

Version-2 capsules carry the manifest produced by model2MLIR from the exact exported graph it lowered.
The worker validates that manifest against the frozen MLIR signature, safetensors bytes, loader inputs,
and every golden result.  Legacy capsules may still recover an ABI from ``torch.export``; newly generated
capsules never do, because rebuilding an unquantized loader can have a different placeholder list.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_module(loader: Path):
    spec = importlib.util.spec_from_file_location("_frozen_capsule_loader", loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen loader {loader}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model_and_inputs"):
        raise RuntimeError("frozen loader has no get_model_and_inputs()")
    return module


def _torch_dtype_name(tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args(argv)
    req_path = Path(args.request).resolve(strict=True)
    req = json.loads(req_path.read_text(encoding="utf-8"))

    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.export.graph_signature import InputKind

    loader = Path(req["loader"]).resolve(strict=True)
    weights_path = Path(req["weights"]).resolve(strict=True)
    inputs_path = Path(req["inputs_npz"]).resolve(strict=True)
    golden_path = Path(req["golden_npy"]).resolve(strict=True)
    signature = list(req["signature"])

    dependency_root_name = req.get("loader_dependencies")
    if dependency_root_name:
        dependency_root = Path(dependency_root_name).resolve(strict=True)
        dependency_dirs = {dependency_root, *(p.parent for p in dependency_root.rglob("*.py"))}
        for dependency_dir in sorted(dependency_dirs, key=lambda p: (len(p.parts), str(p))):
            if str(dependency_dir) not in sys.path:
                sys.path.append(str(dependency_dir))

    module = _load_module(loader)
    torch.manual_seed(int(req.get("torch_seed", 0)))
    model, loader_inputs = module.get_model_and_inputs()
    model = model.eval()
    loader_inputs = tuple(loader_inputs)
    captured_manifest_name = req.get("captured_manifest")
    if captured_manifest_name:
        captured_manifest_path = Path(captured_manifest_name).resolve(strict=True)
        captured_manifest = json.loads(captured_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(captured_manifest, dict) or set(captured_manifest) != {
                str(i) for i in range(len(signature))}:
            raise RuntimeError("capture-time manifest does not cover every frozen interface argument")
        frozen_weights = load_file(str(weights_path), device="cpu")
        supplied_inputs = np.load(inputs_path, allow_pickle=False)
        input_order: dict[str, int] = {}
        user_index = 0
        for index, sig in enumerate(signature):
            meta = captured_manifest[str(index)]
            sig_shape = [int(v) for v in sig["shape"]]
            if list(meta.get("shape") or sig_shape) != sig_shape:
                raise RuntimeError(
                    f"capture manifest shape disagrees with interface argument {index}")
            if meta.get("kind") in ("param", "buffer"):
                key = meta.get("weight")
                if key:
                    if key not in frozen_weights:
                        raise RuntimeError(
                            f"capture manifest argument {index} names absent frozen tensor {key!r}")
                    stored = frozen_weights[key]
                    # Tensor-subclass arguments are represented by a one-element dead-argument stub.
                    if not meta.get("stub") and list(stored.shape) != sig_shape:
                        raise RuntimeError(
                            f"frozen tensor {key!r} shape disagrees with interface argument {index}")
                continue
            key = f"in{user_index}"
            if key not in supplied_inputs.files or user_index >= len(loader_inputs):
                raise RuntimeError(f"missing frozen user input {key} for interface argument {index}")
            loader_arr = loader_inputs[user_index].detach().cpu().numpy()
            frozen_arr = supplied_inputs[key]
            if list(loader_arr.shape) != sig_shape or list(frozen_arr.shape) != sig_shape:
                raise RuntimeError(
                    f"user input {user_index} shape disagrees with interface argument {index}")
            if not np.array_equal(loader_arr, frozen_arr):
                raise RuntimeError(f"frozen input {user_index} disagrees with frozen loader")
            name = str(meta.get("name") or f"input_{user_index}")
            input_order[name] = user_index
            user_index += 1
        if user_index != len(loader_inputs) or user_index != len(supplied_inputs.files):
            raise RuntimeError(
                f"input cardinality mismatch: manifest={user_index}, loader={len(loader_inputs)}, "
                f"frozen={len(supplied_inputs.files)}")

        output_signature = list(req.get("output_signature") or [])
        output_order = list(req.get("output_order") or [])
        goldens_path = Path(req["goldens_npz"]).resolve(strict=True)
        goldens = np.load(goldens_path, allow_pickle=False)
        if len(output_signature) != len(output_order) or set(goldens.files) != set(output_order):
            raise RuntimeError("multi-output golden order does not match the frozen result signature")
        for index, (name, sig) in enumerate(zip(output_order, output_signature)):
            if list(goldens[name].shape) != [int(v) for v in sig["shape"]]:
                raise RuntimeError(f"frozen golden {name!r} shape disagrees with result {index}")

        report = {
            "version": 2,
            "manifest": captured_manifest,
            "input_order": input_order,
            "output_order": output_order,
            "loader_sha256": _sha256(loader),
            "weights_sha256": _sha256(weights_path),
            "manifest_sha256": _sha256(captured_manifest_path),
            "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "torch_export": False,
            "capture_manifest_validated": True,
            "loader_input_count": user_index,
            "golden_validated": True,
            "golden_value_replay": False,
            "weights_validated_exact": True,
        }
        print("__CAPSULE_BUNDLE_ABI__ " + json.dumps(report, sort_keys=True))
        return 0

    exported = torch.export.export(model, loader_inputs)
    specs = list(exported.graph_signature.input_specs)
    if len(specs) != len(signature):
        raise RuntimeError(
            f"loader export has {len(specs)} arguments but frozen interface has {len(signature)}")

    frozen_weights = load_file(str(weights_path), device="cpu")
    state = dict(model.state_dict())
    if set(frozen_weights) != set(state):
        missing = sorted(set(state) - set(frozen_weights))
        extra = sorted(set(frozen_weights) - set(state))
        raise RuntimeError(
            f"frozen weights do not match loader state_dict keys (missing={missing}, extra={extra})")

    supplied_inputs = np.load(inputs_path, allow_pickle=False)
    expected_golden = np.load(golden_path, allow_pickle=False)
    manifest: dict[str, dict] = {}
    input_order: dict[str, int] = {}
    user_index = 0
    for index, (spec, sig) in enumerate(zip(specs, signature)):
        sig_shape = [int(v) for v in sig["shape"]]
        sig_dtype = str(sig["dtype"])
        arg_name = str(spec.arg.name)
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            key = str(spec.target)
            if key not in frozen_weights or key not in state:
                raise RuntimeError(f"export argument {index} refers to absent frozen weight {key!r}")
            actual = state[key].detach().cpu()
            frozen = frozen_weights[key].detach().cpu()
            if list(actual.shape) != sig_shape or list(frozen.shape) != sig_shape:
                raise RuntimeError(
                    f"weight {key!r} shape disagrees with interface argument {index}: "
                    f"loader={list(actual.shape)} frozen={list(frozen.shape)} interface={sig_shape}")
            if actual.dtype != frozen.dtype or not torch.equal(actual, frozen):
                raise RuntimeError(f"frozen weight {key!r} bytes/values disagree with frozen loader")
            manifest[str(index)] = {
                "weight": key,
                "kind": "param" if spec.kind == InputKind.PARAMETER else "buffer",
                "dtype": _torch_dtype_name(frozen),
                "shape": sig_shape,
            }
            continue
        if spec.kind != InputKind.USER_INPUT:
            raise RuntimeError(
                f"unsupported export argument kind at {index}: {spec.kind}; capsule bundles may not "
                "guess lifted/custom arguments")
        key = f"in{user_index}"
        if user_index >= len(loader_inputs) or key not in supplied_inputs.files:
            raise RuntimeError(f"missing frozen user input {key} for export argument {index}")
        loader_arr = loader_inputs[user_index].detach().cpu().numpy()
        frozen_arr = supplied_inputs[key]
        if list(loader_arr.shape) != sig_shape or list(frozen_arr.shape) != sig_shape:
            raise RuntimeError(
                f"user input {user_index} shape disagrees with interface argument {index}: "
                f"loader={list(loader_arr.shape)} frozen={list(frozen_arr.shape)} interface={sig_shape}")
        if not np.array_equal(loader_arr, frozen_arr):
            raise RuntimeError(f"frozen input {user_index} disagrees with frozen loader")
        manifest[str(index)] = {"kind": "input", "name": arg_name}
        input_order[arg_name] = user_index
        user_index += 1
    if user_index != len(loader_inputs) or user_index != len(supplied_inputs.files):
        raise RuntimeError(
            f"input cardinality mismatch: export={user_index}, loader={len(loader_inputs)}, "
            f"frozen={len(supplied_inputs.files)}")

    with torch.no_grad():
        observed = model(*loader_inputs)
    if isinstance(observed, (tuple, list)):
        if len(observed) != 1:
            raise RuntimeError("multi-output loader cannot be represented by this single-golden bundle")
        observed = observed[0]
    observed = observed.detach().float().cpu().numpy()
    policy = dict(req.get("numeric_policy") or {})
    atol = float(policy.get("atol", 0.0) or 0.0)
    rtol = float(policy.get("rtol", 0.0) or 0.0)
    if list(observed.shape) != list(expected_golden.shape):
        raise RuntimeError(
            f"frozen golden shape {list(expected_golden.shape)} disagrees with loader "
            f"{list(observed.shape)}")
    if not np.allclose(observed, expected_golden, atol=atol, rtol=rtol, equal_nan=True):
        delta = float(np.max(np.abs(observed.astype(np.float64) -
                                    expected_golden.astype(np.float64))))
        raise RuntimeError(
            f"frozen golden disagrees with frozen loader (max_abs={delta}, atol={atol}, rtol={rtol})")

    report = {
        "version": 1,
        "manifest": manifest,
        "input_order": input_order,
        "loader_sha256": _sha256(loader),
        "weights_sha256": _sha256(weights_path),
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_export": True,
        "loader_input_count": user_index,
        "golden_validated": True,
        "weights_validated_exact": True,
    }
    print("__CAPSULE_BUNDLE_ABI__ " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
