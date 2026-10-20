"""Verify complete FP32 or explicit integer ResNet50 v1.5 development sessions."""

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback


RTOL, ATOL = 1e-4, 1e-4


def parse_sample_indices(value):
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Device samples require comma-separated integer indices") from error
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("Device sample indices must be distinct and nonnegative")
    return indices


def safe_path(value):
    """Validate the helper's import path before importing its path validator."""
    path = Path(os.path.abspath(value))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("Restricted path component")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            target = Path(os.readlink(current))
            current = safe_path(target if target.is_absolute() else current.parent / target)
    return current


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_identity(path, allowed_path):
    path = allowed_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def load_images(path, np):
    identity = file_identity(path, safe_path)
    with np.load(identity["path"], allow_pickle=False) as data:
        if "images" not in data.files:
            raise ValueError("Input NPZ must contain an images array")
        images = data["images"]
    if images.dtype != np.float32:
        raise ValueError("Input images must be float32; preprocessing and dtype conversion must be explicit")
    if images.ndim == 4:
        images = images[:, None, :, :, :]
    if images.ndim != 5 or images.shape[0] < 1 or images.shape[1:] != (1, 3, 224, 224):
        raise ValueError("Expected images shaped [N,3,224,224] or [N,1,3,224,224], with N positive")
    if not np.isfinite(images).all():
        raise ValueError("Input images contain nonfinite values")
    return np.ascontiguousarray(images), identity


def load_checkpoint(model, path, torch):
    identity = file_identity(path, safe_path)
    with Path(identity["path"]).open("rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint must be a plain state_dict with exactly the model's parameter and buffer keys")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"Checkpoint entry {name} must be a dense tensor")
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint shape or dtype mismatch for {name}; implicit conversion is not permitted")
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint entry {name} contains nonfinite values")
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[name], value) for name, value in state.items()):
        raise AssertionError("Loaded model does not preserve the supplied state_dict")
    if file_identity(identity["path"], safe_path) != identity:
        raise ValueError("Checkpoint file changed while loading")
    return identity


def load_labels(path, input_identity, count):
    identity = file_identity(path, safe_path)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result

    with Path(identity["path"]).open(encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=unique_keys)
    if not isinstance(manifest, dict) or set(manifest) != {"input_sha256", "class_ids", "samples"}:
        raise ValueError("Labels manifest must contain input_sha256, class_ids and samples")
    if not isinstance(manifest["input_sha256"], str) or manifest["input_sha256"] != input_identity["sha256"]:
        raise ValueError("Labels input_sha256 must match the supplied NPZ file")
    class_ids = manifest["class_ids"]
    if not isinstance(class_ids, list) or len(class_ids) != 1000:
        raise ValueError("class_ids must list 1000 class identifiers in logit order")
    if any(not isinstance(value, str) or not value.strip() for value in class_ids) or len(set(class_ids)) != 1000:
        raise ValueError("class_ids must be unique nonempty strings")
    samples = manifest["samples"]
    if not isinstance(samples, list) or len(samples) != count:
        raise ValueError("Labels samples must match the complete image stream in order")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"id", "label"}:
            raise ValueError("Each sample must contain id and label")
        if not isinstance(sample["id"], str) or not sample["id"].strip():
            raise ValueError("Sample id must be a nonempty string")
        if type(sample["label"]) is not int or not 0 <= sample["label"] < 1000:
            raise ValueError("Sample label must be an integer class index from 0 to 999")
    if file_identity(identity["path"], safe_path) != identity:
        raise ValueError("Labels file changed while loading")
    return manifest, identity


def classify_logits(logits, label, np):
    if logits.shape != (1, 1000) or not np.isfinite(logits).all():
        raise ValueError("Classification requires finite logits shaped [1,1000]")
    ranking = np.lexsort((np.arange(1000), -logits[0]))[:5].tolist()
    return {"top5": ranking, "top1_hit": ranking[0] == label, "top5_hit": label in ranking}


def accuracy_counts(classifications):
    count = len(classifications)
    if count < 1:
        raise ValueError("Accuracy requires a nonempty complete stream")
    result = {"samples": count, "top1_disagreements": sum(row["torch"]["top5"][0] != row["relax"]["top5"][0] for row in classifications)}
    for backend in ("torch", "relax"):
        hits = {key: sum(row[backend][key + "_hit"] for row in classifications) for key in ("top1", "top5")}
        result[backend] = {key: {"correct": value, "rate": value / count} for key, value in hits.items()}
    return result


def top1_rate_drop(accuracy):
    """Compute a paired loss from counts without subtracting rounded rates."""
    return (accuracy["torch"]["top1"]["correct"] - accuracy["relax"]["top1"]["correct"]) / accuracy["samples"]


def load_model(loader, args, torch, np):
    supplied, artifacts = None, {}
    if args.inputs is not None:
        supplied, artifacts["inputs"] = load_images(args.inputs, np)
    labels = None
    if getattr(args, "labels", None) is not None:
        labels, artifacts["labels"] = load_labels(args.labels, artifacts["inputs"], len(supplied))
    isolated = {"M2M_RESNET_RANDOM": "1", "M2M_RESNET_PRETRAINED": "0", "M2M_RESNET_PAPER_READY": "0", "M2M_SESSION_STEPS": str(len(supplied) if supplied is not None else 2)}
    if supplied is not None:
        isolated.update({"M2M_RESNET_INPUT_NPZ": artifacts["inputs"]["path"], "M2M_RESNET_INPUT_SOURCE": args.input_source, "M2M_RESNET_PREPROCESSING": args.preprocessing})
    inherited = {key: value for key, value in os.environ.items() if key.startswith("M2M_RESNET_") or key == "M2M_SESSION_STEPS"}
    for key in inherited:
        os.environ.pop(key)
    os.environ.update(isolated)
    try:
        model, _ = loader.get_model_and_inputs()
    finally:
        for key in isolated:
            os.environ.pop(key, None)
        os.environ.update(inherited)
    model = model.cpu().eval()
    if supplied is not None:
        if not np.array_equal(model.session_images.numpy(), supplied):
            raise AssertionError("Loader changed the supplied image stream")
        artifacts["checkpoint"] = load_checkpoint(model.model, args.checkpoint, torch)
        model.session_provenance.update({"checkpoint": "supplied_state_dict", "checkpoint_sha256": loader._state_dict_sha256(model.model), "checkpoint_file_sha256": artifacts["checkpoint"]["sha256"], "full_checkpoint": True, "training_provenance": "unverified", "synthetic_inputs": None, "input_declarations_verified": False, "calibration_source": "not_used"})
    if model.paper_ready:
        raise AssertionError("Frontend verification must not claim paper qualification")
    return model, {"artifacts": artifacts, "loader_environment": isolated, "labels": labels}


def onnx_inventory(model, onnx):
    def tensor_info(tensor):
        return {"dtype": onnx.TensorProto.DataType.Name(tensor.data_type), "shape": list(tensor.dims)}

    def attribute(value):
        if isinstance(value, onnx.TensorProto):
            return tensor_info(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [attribute(item) for item in value]
        return value

    values = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        kind = value.type.tensor_type
        shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None for dim in kind.shape.dim]
        values[value.name] = {"dtype": onnx.TensorProto.DataType.Name(kind.elem_type), "shape": shape}
    initializers = {value.name: tensor_info(value) for value in model.graph.initializer}
    values.update(initializers)
    nodes, counts = [], Counter()
    for index, node in enumerate(model.graph.node):
        op = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        counts[op] += 1
        attrs = {attr.name: attribute(onnx.helper.get_attribute_value(attr)) for attr in node.attribute}
        nodes.append({"index": index, "name": node.name, "operator": op, "attributes": attrs, "inputs": [{"name": name, **values.get(name, {})} for name in node.input], "outputs": [{"name": name, **values.get(name, {})} for name in node.output]})
        if node.op_type == "Conv" and len(node.input) > 1 and node.input[1] in initializers:
            initializers[node.input[1]]["layout"] = "OIHW"
        if node.op_type == "Gemm" and len(node.input) > 1 and node.input[1] in initializers:
            initializers[node.input[1]]["layout"] = "OI" if attrs.get("transB", 0) else "IO"
    return {"operator_counts": dict(sorted(counts.items())), "nodes": nodes, "initializers": initializers, "values": values}


def relax_inventory(mod, tvm):
    calls, counts = [], Counter()
    functions, function_counts = [], Counter()

    def visit(expr):
        if isinstance(expr, tvm.relax.Call):
            op = getattr(expr.op, "name", getattr(expr.op, "name_hint", str(expr.op)))
            counts[op] += 1
            calls.append({"operator": op, "inputs": [str(arg.struct_info) for arg in expr.args], "output": str(expr.struct_info), "attributes": str(expr.attrs)})

    for name, function in mod.functions.items():
        kind = "relax" if isinstance(function, tvm.relax.Function) else "tir" if isinstance(function, tvm.tir.PrimFunc) else type(function).__name__
        function_counts[kind] += 1
        functions.append({"name": name.name_hint, "kind": kind, "parameters": len(function.params), "attributes": str(function.attrs)})
        if isinstance(function, tvm.relax.Function):
            tvm.relax.analysis.post_order_visit(function.body, visit)
    return {"operator_counts": dict(sorted(counts.items())), "calls": calls,
            "function_counts": dict(sorted(function_counts.items())), "functions": sorted(functions, key=lambda item: item["name"])}


def prepare_graph(mod, mode, tvm):
    """Apply the selected graph passes, then expose the actual VM build IR."""
    if mode not in ("baseline", "optimized"):
        raise ValueError("Expected baseline or optimized graph mode")
    with tvm.target.Target("llvm"):
        graph = tvm.relax.get_pipeline("zero")(mod) if mode == "optimized" else mod
        lowered = tvm.relax.get_pipeline("default_build")(graph)
    return graph, lowered


def compare_output(actual, expected, np):
    if not np.isfinite(actual).all():
        raise AssertionError("Relax output contains nonfinite values")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"Output mismatch: {actual.shape}/{actual.dtype} versus {expected.shape}/{expected.dtype}")
    delta = np.abs(actual - expected)
    return {"max_absolute_error": float(delta.max()), "max_relative_error": float((delta / np.maximum(np.abs(expected), ATOL)).max()),
            "passed": bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False))}


def fold_resnet_batchnorm(model, torch, state_hash):
    """Derive an FP32 evaluation copy for explicitly paired ResNet Conv/BN sites."""
    if any(module.training for module in model.modules()):
        raise ValueError("BatchNorm folding requires every module in evaluation mode")
    modules = dict(model.named_modules())
    pairs = []
    for name, bn in modules.items():
        if not isinstance(bn, torch.nn.modules.batchnorm._BatchNorm):
            continue
        parent, _, leaf = name.rpartition(".")
        if leaf.startswith("bn") and leaf[2:].isdigit():
            conv_name = ".".join(filter(None, (parent, "conv" + leaf[2:])))
        elif leaf == "1" and parent.endswith("downsample") and type(modules[parent]) is torch.nn.Sequential:
            conv_name = parent + ".0"
        else:
            raise ValueError(f"Unsupported BatchNorm pairing: {name}")
        conv = modules.get(conv_name)
        if type(conv) is not torch.nn.Conv2d or type(bn) is not torch.nn.BatchNorm2d:
            raise ValueError(f"BatchNorm folding requires standard Conv2d/BatchNorm2d: {name}")
        if not bn.affine or not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
            raise ValueError(f"BatchNorm folding requires affine parameters and running statistics: {name}")
        if conv.out_channels != bn.num_features or not math.isfinite(bn.eps) or bn.eps <= 0:
            raise ValueError(f"Incompatible BatchNorm channels or epsilon: {name}")
        tensors = [conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var]
        if conv.bias is not None:
            tensors.append(conv.bias)
        if any(value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all() for value in tensors):
            raise ValueError(f"BatchNorm folding requires finite CPU FP32 parameters: {name}")
        if (bn.running_var < 0).any():
            raise ValueError(f"BatchNorm variance must be nonnegative: {name}")
        pairs.append((conv_name, name, bn.eps))
    if not pairs:
        raise ValueError("BatchNorm folding requires at least one supported pair")
    original_hash = state_hash(model)
    derived = copy.deepcopy(model)
    sites = []
    with torch.no_grad():
        for conv_name, bn_name, epsilon in pairs:
            conv, bn = derived.get_submodule(conv_name), derived.get_submodule(bn_name)
            scale = bn.weight / torch.sqrt(bn.running_var + epsilon)
            bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            weight = conv.weight * scale[:, None, None, None]
            bias = bn.bias + scale * (bias - bn.running_mean)
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError(f"BatchNorm folding produced nonfinite parameters: {bn_name}")
            conv.weight = torch.nn.Parameter(weight, requires_grad=False)
            conv.bias = torch.nn.Parameter(bias, requires_grad=False)
            parent, _, leaf = bn_name.rpartition(".")
            derived.get_submodule(parent)._modules[leaf] = torch.nn.Identity().eval()
            sites.append({"conv": conv_name, "batchnorm": bn_name, "epsilon": epsilon,
                          "weight_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                          "bias_sha256": hashlib.sha256(bias.numpy().tobytes()).hexdigest()})
    if state_hash(model) != original_hash:
        raise AssertionError("BatchNorm folding changed the source model")
    return derived, {"source_state_dict_sha256": original_hash, "derived_state_dict_sha256": state_hash(derived),
                     "contract": {"mode": "evaluation", "dtype": "float32", "device": "cpu",
                                  "scale": "gamma / sqrt(running_var + epsilon)",
                                  "weight": "weight * scale", "bias": "beta + scale * (bias - running_mean)",
                                  "missing_conv_bias": "zero", "rtol": RTOL, "atol": ATOL},
                     "sites": sites, "images": []}


def verify_integer_resnet(args, model, expected_outputs, labels, report, output, stage, write_json, np, torch, tvm, state_hash):
    """Qualify the complete development integer graph and optional static export."""
    import resnet_quantized as recipe
    from verify_session import load_module as load_registered_module

    report.update(integer_reference_verified=False, device_execution=False, timing_qualified=False, model_quality_qualified=False)
    report["integer_recipe_source"] = file_identity(recipe.__file__, safe_path)
    stage("load_distinct_calibration_stream")
    calibration, identity = load_images(args.calibration_inputs, np)
    report["artifacts"]["calibration_inputs"] = identity
    ids = [f"{identity['sha256']}:{index}" for index in range(len(calibration))]
    plan = recipe.calibrate_resnet50(model.model, calibration, calibration_ids=ids,
        checkpoint_declaration="local supplied state_dict" if args.checkpoint else "random diagnostic; not pretrained",
        activation_estimator=args.activation_estimator, residual_policy=args.residual_policy)
    report["quantization"] = plan.manifest()
    if args.frozen_quantization:
        frozen_identity = file_identity(args.frozen_quantization, safe_path)
        frozen = json.loads(Path(frozen_identity["path"]).read_text())
        if frozen != report["quantization"]:
            raise AssertionError("TRAIN calibration differs from the frozen quantization manifest")
        report["artifacts"]["frozen_quantization"] = frozen_identity
        report["frozen_quantization_verified"] = True
    report["evaluation_role"] = args.evaluation_role
    report["quantization"]["calibration_source"] = args.calibration_source
    write_json("quantization.json", report["quantization"])
    expected = [recipe.reference(plan, image.numpy(), return_integer_logits=True) for image in model.session_images]
    modes = list(report["graph_modes"])
    exporter_dir = safe_path(args.tvm_source) / "apps/gemmini"
    sys.path.insert(0, str(exporter_dir))
    exporter = load_registered_module("tvm_resnet_baremetal_exporter", exporter_dir / "baremetal.py")
    report["baremetal_exporter_source"] = file_identity(exporter.__file__, safe_path)
    for mode in modes:
        stage(f"integer_prepare_{mode}")
        semantic, prepared, coverage = recipe.prepare_device_graph(plan, optimize=mode == "optimized",
            tile_i=args.tile_i, tile_j=args.tile_j, return_integer_logits=True)
        row = report["graph_modes"][mode]
        row["device_coverage"] = coverage
        memory = exporter.preflight_graph(prepared, memory_limit_bytes=args.memory_limit_bytes)
        memory_name = f"memory_{mode}.json"
        write_json(memory_name, memory)
        row["memory_preflight"] = {key: value for key, value in memory.items() if not isinstance(value, (dict, list))}
        row["memory_preflight"].update(allocated=memory["allocated"], artifact=file_identity(output / memory_name, safe_path))
        row["memory_scope"] = "explicit tensors only; executable, stack and platform reservations additional"
        semantic_path, prepared_path = output / "integer_semantic.relax.py", output / f"integer_{mode}.relax.py"
        semantic_path.write_text(semantic.script())
        prepared_path.write_text(prepared.script())
        row["semantic_ir"], row["device_ir"] = file_identity(semantic_path, safe_path), file_identity(prepared_path, safe_path)
        stage(f"integer_build_cpu_{mode}")
        _, lowered = prepare_graph(semantic, mode, tvm)
        vm = tvm.relax.VirtualMachine(tvm.relax.build(lowered, "llvm", pipeline=None), tvm.cpu())
        for index, image in enumerate(model.session_images):
            stage(f"integer_execute_{mode}_image_{index}")
            image_array = image.numpy()
            result = vm["main"](tvm.nd.array(recipe.quantize(image_array, plan.input_exponent), tvm.cpu()))
            integer_actual, actual = result[0].numpy(), result[1].numpy()
            integer_expected, expected_logits = expected[index]
            for candidate, reference in ((integer_actual, integer_expected), (actual, expected_logits)):
                if candidate.dtype != reference.dtype or candidate.shape != reference.shape or not np.array_equal(candidate, reference):
                    raise AssertionError(f"Integer recipe/CPU disagreement for image {index}")
            if mode == modes[0]:
                report["images"].append({"index": index, "input_sha256": recipe.array_digest(image_array), "graph_mode_comparisons": {}})
            comparison = {"integer_exact": True, "dequantized_exact": True,
                "integer_sha256": recipe.array_digest(integer_actual), "dequantized_sha256": recipe.array_digest(actual),
                "versus_original_fp32_max_absolute_error": float(np.max(np.abs(actual.astype(np.float64) - expected_outputs[index])))}
            if labels is not None:
                sample = labels["samples"][index]
                comparison["classification"] = {"sample_id": sample["id"], "label": sample["label"],
                    "torch": classify_logits(expected_outputs[index], sample["label"], np), "relax": classify_logits(actual, sample["label"], np)}
            report["images"][index]["graph_mode_comparisons"][mode] = comparison
            np.savez(output / f"integer_{mode}_image_{index}.npz", integer_reference=integer_expected,
                integer_actual=integer_actual, dequantized_reference=expected_logits, dequantized_actual=actual)
            row["checked_images"] = index + 1
            report["session"]["checked_images"] = min(value["checked_images"] for value in report["graph_modes"].values())
        if labels is not None:
            row["descriptive_accuracy"] = accuracy_counts([image["graph_mode_comparisons"][mode]["classification"] for image in report["images"]])
            rates = row["descriptive_accuracy"]
            row["development_accuracy_gate"] = {"maximum_top1_rate_drop": args.max_top1_drop,
                "observed_top1_rate_drop": top1_rate_drop(rates),
                "paper_approved": False, "dataset_truth_verified": False}
            row["development_accuracy_gate"]["passed"] = row["development_accuracy_gate"]["observed_top1_rate_drop"] <= args.max_top1_drop
            if not row["development_accuracy_gate"]["passed"]:
                raise AssertionError("Quantized model exceeds the declared development top-1 loss gate")
        if args.export_baremetal:
            stage(f"integer_export_baremetal_{mode}")
            directory = output / f"baremetal_{mode}"
            manifest = exporter.export_graph(prepared, directory, memory_limit_bytes=args.memory_limit_bytes)
            row["baremetal_export"] = {key: value for key, value in manifest.items() if not isinstance(value, (dict, list))}
            row["baremetal_export"]["manifest"] = file_identity(directory / "graph.json", safe_path)
        write_json("results.json", report)
    references = [report[key] for key in ("verifier_source", "integer_recipe_source", "loader_source", "torchvision_resnet_source")]
    references.extend(report["artifacts"].values())
    for identity in [*references, report["baremetal_exporter_source"]]:
        if file_identity(identity["path"], safe_path) != identity:
            raise AssertionError("Selected integer source or supplied artifact changed")
    if state_hash(model.model) != report["model"]["state_dict_sha256"]:
        raise AssertionError("Quantization changed original FP32 checkpoint parameters")
    report["session"]["complete"] = all(value["checked_images"] == len(model.session_images) for value in report["graph_modes"].values())
    if not report["session"]["complete"]:
        raise AssertionError("Integer verification omitted declared images")
    if args.export_baremetal:
        selected = list(args.device_sample_indices) if args.device_sample_indices is not None else list(range(len(expected)))
        if max(selected) >= len(expected):
            raise ValueError("Device sample index exceeds the complete checked host session")
        fixture = output / "reference.npz"
        np.savez(fixture, input_0=np.stack([recipe.quantize(model.session_images[index].numpy(), plan.input_exponent) for index in selected]), output_0=np.stack([expected[index][0] for index in selected]), output_1=np.stack([expected[index][1] for index in selected]))
        identity = file_identity(fixture, safe_path)
        sample_ids = [f"{index}:{labels['samples'][index]['id'] if labels else recipe.array_digest(model.session_images[index].numpy())}" for index in selected]
        provenance = {"fixture_sha256": identity["sha256"], "sample_ids": sample_ids, "reference_sources": references,
            "reference_contract": report["quantization"]["recipe"], "oracle": "independent integer reference; no exported or accelerated contraction", "paper_quality_approved": False,
            "selected_session_indices": selected, "complete_host_session_images": len(expected)}
        write_json("reference.json", provenance)
        report["device_fixture"] = {"fixture": identity, "provenance": file_identity(output / "reference.json", safe_path), "samples": len(selected), "selected_session_indices": selected, "complete_host_session_images": len(expected)}
    report.update(status="passed", integer_reference_verified=True)
    stage("complete")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--tvm-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Local plain torchvision ResNet50 state_dict; no download")
    parser.add_argument("--inputs", type=Path, help="NPZ with an images array; every supplied image is checked")
    parser.add_argument("--input-source", help="Declared origin of the supplied image stream")
    parser.add_argument("--preprocessing", help="Declared preprocessing already applied to the images")
    parser.add_argument("--labels", type=Path, help="Optional JSON class order and labels bound to the supplied NPZ")
    parser.add_argument("--quantized", action="store_true", help="Verify the explicit full integer development recipe; no automatic paper qualification")
    parser.add_argument("--calibration-inputs", type=Path, help="Distinct FP32 NPZ stream required by --quantized")
    parser.add_argument("--calibration-source", help="Declared origin of the distinct calibration stream")
    parser.add_argument("--activation-estimator", choices=("maxabs", "mse"), default="maxabs", help="TRAIN-only activation scale estimator; mse admits deterministic bounded clipping candidates")
    parser.add_argument("--residual-policy", choices=("legacy", "accumulator"), default="legacy", help="Opt-in proved unique-consumer residual accumulators and joint one-round addition")
    parser.add_argument("--frozen-quantization", type=Path, help="Previously frozen plan.manifest() JSON; recomputed TRAIN plan must match before integer evaluation")
    parser.add_argument("--evaluation-role", choices=("development", "held-out"), default="development", help="Explicit evaluation use; held-out requires --frozen-quantization")
    parser.add_argument("--max-top1-drop", type=float, default=0.02, help="Development top-1 rate-loss gate versus unchanged FP32 on supplied labels (default 0.02)")
    parser.add_argument("--export-baremetal", action="store_true", help="Export complete integer device graph; does not execute or time it")
    parser.add_argument("--device-sample-indices", type=parse_sample_indices, help="Comma-separated checked host-session indices for the simulator fixture; every host image still undergoes numerical/quality checks")
    parser.add_argument("--memory-limit-bytes", type=int, default=256 * 1024 * 1024, help="Explicit tensor-storage budget; additional platform reservations are required")
    parser.add_argument("--tile-i", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--tile-j", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--batchnorm-folding", action="store_true",
                        help="Validate an evaluation-only FP32 folded copy against the unchanged reference, then export that copy")
    parser.add_argument("--graph-mode", choices=("baseline", "optimized", "both"), default="baseline",
                        help="Opt-in graph diagnostic: optimized runs zero_pipeline before the ordinary VM build; default preserves the baseline")
    args = parser.parse_args()
    if args.evaluation_role == "held-out" and not args.frozen_quantization:
        parser.error("Held-out evaluation requires a previously frozen quantization manifest")
    if args.frozen_quantization and not args.quantized:
        parser.error("Frozen quantization requires --quantized")
    if args.residual_policy != "legacy" and not args.quantized:
        parser.error("Accumulator residual policy requires --quantized")
    if args.device_sample_indices is not None and not args.export_baremetal:
        parser.error("Device sample selection requires --export-baremetal")
    supplied = (args.checkpoint, args.inputs, args.input_source, args.preprocessing)
    if any(value is not None for value in supplied) and not all(value is not None and str(value).strip() for value in supplied):
        parser.error("Supplied-artifact mode requires --checkpoint, --inputs, --input-source and --preprocessing together")
    if args.labels is not None and args.inputs is None:
        parser.error("--labels requires supplied-artifact mode with all four artifact options")
    if args.quantized and (not args.calibration_inputs or not args.calibration_source or args.batchnorm_folding):
        parser.error("--quantized requires distinct --calibration-inputs and --calibration-source and performs its own BatchNorm folding")
    if not args.quantized and (args.calibration_inputs or args.calibration_source or args.export_baremetal):
        parser.error("Calibration and baremetal export options require --quantized")
    if not 0 <= args.max_top1_drop <= 1 or args.memory_limit_bytes <= 0:
        parser.error("Invalid accuracy loss or tensor-memory budget")
    sys.dont_write_bytecode = True
    output = safe_path(args.output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output-dir must be a new directory to preserve existing artifacts")
    allowed_path = safe_path
    report = {"status": "failed", "stage": "initialization", "scope": __doc__, "mode": "supplied_artifacts" if args.checkpoint else "random_diagnostic", "trained_model": None if args.checkpoint else False, "paper_validation": False, "seed": 194, "rtol": RTOL, "atol": ATOL, "images": [], "verifier_source": file_identity(__file__, allowed_path)}
    modes = ("baseline", "optimized") if args.graph_mode == "both" else (args.graph_mode,)
    report.update(graph_mode=args.graph_mode, primary_graph_mode=modes[0], graph_modes={mode: {"checked_images": 0} for mode in modes})

    def write_json(name, data):
        allowed_path(output / name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def stage(name):
        report["stage"] = name
        write_json("results.json", report)
        print(name, flush=True)

    try:
        stage("load_frontend_helpers")
        helper = load_module("resnet_frontend_helpers", safe_path(args.tvm_source) / "apps/gemmini/verify_onnx.py")
        allowed_path = helper.allowed_path
        report["frontend_helper_source"] = helper.identity(helper.__file__)
        stage("dependencies")
        import numpy as np
        import onnx
        import torch
        import torchvision
        import torchvision.models.resnet as resnet_source
        import tvm

        torch.set_num_threads(1)
        torch.manual_seed(194)
        from_onnx, metadata = helper.provenance(tvm, torch, onnx, np, None)
        report.update(metadata)
        report["versions"]["torchvision"] = torchvision.__version__
        report["torchvision_resnet_source"] = helper.identity(resnet_source.__file__)
        loader_path = allowed_path(allowed_path(args.model2mlir_root) / "workloads/resnet50_v1_5/loader.py")
        report["loader_source"] = helper.identity(loader_path)
        stage("load_supplied_artifacts" if args.checkpoint else "load_random_initialized_model")
        loader = load_module("resnet_structural_loader", loader_path)
        model, loaded = load_model(loader, args, torch, np)
        report["artifacts"] = loaded["artifacts"]
        labels = loaded["labels"]
        if labels is not None:
            report["label_declarations"] = {"input_sha256": labels["input_sha256"], "class_ids": labels["class_ids"], "samples": labels["samples"], "dataset_truth_verified": False, "checkpoint_class_mapping_verified": False, "tie_order": "ascending_class_index", "accuracy_threshold": None}
        images = model.session_images
        if images.dtype != torch.float32 or images.ndim != 5 or tuple(images.shape[1:]) != (1, 3, 224, 224) or len(images) < 1 or not torch.isfinite(images).all():
            raise AssertionError("Expected a nonempty finite FP32 NCHW image stream")
        if args.checkpoint is None and (len(images) != 2 or torch.equal(images[0], images[1]) or model.session_provenance["checkpoint"] != "random_init"):
            raise AssertionError("Expected random initialization and two distinct diagnostic images")
        report["session"] = {"declared_images": len(images), "checked_images": 0, "complete": False}
        backbone = model.model
        blocks = [len(getattr(backbone, f"layer{i}")) for i in range(1, 5)]
        strides = {name: {"stride": list(module.stride), "weight_shape": list(module.weight.shape), "weight_dtype": str(module.weight.dtype), "input_layout": "NCHW", "weight_layout": "OIHW"} for name, module in backbone.named_modules() if isinstance(module, torch.nn.Conv2d)}
        if blocks != [3, 4, 6, 3] or any(strides[f"layer{i}.0.conv1"]["stride"] != [1, 1] or strides[f"layer{i}.0.conv2"]["stride"] != [2, 2] for i in (2, 3, 4)):
            raise AssertionError("Expected ResNet50 v1.5 stage structure and stride in the 3x3 convolution")
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise AssertionError("Expected original full FP32 model parameters")
        state_hash = loader._state_dict_sha256(backbone)
        report["model"] = {"architecture": "ResNet50 v1.5", "state_dict_sha256": state_hash, "parameter_count": sum(value.numel() for value in backbone.parameters()), "module_count": sum(1 for _ in backbone.modules()), "stage_blocks": blocks, "conv_modules": strides, "loader_provenance": model.session_provenance, "loader_environment": loaded["loader_environment"], "input_layout": "NCHW", "parameter_perturbation": False}
        export_model = model
        if args.batchnorm_folding:
            stage("derive_batchnorm_folded_model")
            export_model = copy.deepcopy(model)
            export_model.model, report["batchnorm_folding"] = fold_resnet_batchnorm(backbone, torch, loader._state_dict_sha256)
        expected_outputs = []
        for index, image in enumerate(images):
            stage(f"torch_execute_image_{index}")
            with torch.no_grad():
                expected = model(image).numpy()
            if expected.dtype != np.float32 or expected.shape != (1, 1000) or not np.isfinite(expected).all():
                raise AssertionError("Expected finite FP32 classifier logits [1,1000]")
            expected_outputs.append(expected)
            values = {"image": image.numpy(), "torch": expected}
            if args.batchnorm_folding:
                with torch.no_grad():
                    folded = export_model(image).numpy()
                comparison = compare_output(folded, expected, np)
                report["batchnorm_folding"]["images"].append({"index": index, "folded_vs_torch": comparison})
                values["torch_folded"] = folded
                if not comparison["passed"]:
                    raise AssertionError(f"BatchNorm-folded PyTorch mismatch for image {index}")
            np.savez(allowed_path(output / f"image_{index}.npz"), **values)
        if args.quantized:
            return verify_integer_resnet(args, model, expected_outputs, labels, report, output, stage, write_json, np, torch, tvm, loader._state_dict_sha256)
        stage("torch_legacy_onnx_export")
        graph_path = allowed_path(output / "model.onnx")
        torch.onnx.export(export_model, (images[0],), str(graph_path), input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=True)
        report["onnx_graph"] = file_identity(graph_path, allowed_path)
        stage("onnx_check_and_shape_inference")
        graph = onnx.load(str(graph_path))
        onnx.checker.check_model(graph, full_check=True)
        graph = onnx.shape_inference.infer_shapes(graph, check_type=True, strict_mode=True, data_prop=True)
        onnx.checker.check_model(graph, full_check=True)
        inferred_path = allowed_path(output / "inferred.onnx")
        onnx.save(graph, str(inferred_path))
        report["inferred_onnx_graph"] = file_identity(inferred_path, allowed_path)
        inventory = onnx_inventory(graph, onnx)
        write_json("onnx_inventory.json", inventory)
        report["onnx_operator_counts"] = inventory["operator_counts"]
        stage("relax_import_frozen_parameters")
        mod = from_onnx(graph, shape_dict={"image": [1, 3, 224, 224]}, keep_params_in_input=False)
        ir_path = allowed_path(output / "imported_relax.py")
        ir_path.write_text(mod.script(), encoding="utf-8")
        report["relax_ir"] = file_identity(ir_path, allowed_path)
        inventory = relax_inventory(mod, tvm)
        write_json("relax_inventory.json", inventory)
        report["relax_operator_counts"] = inventory["operator_counts"]
        image_arrays = [{"image": image.numpy(), "torch": expected} for image, expected in zip(images, expected_outputs)]
        if args.batchnorm_folding:
            for index, values in enumerate(image_arrays):
                with np.load(allowed_path(output / f"image_{index}.npz"), allow_pickle=False) as saved:
                    values["torch_folded"] = saved["torch_folded"]
        for mode in modes:
            stage(f"relax_prepare_{mode}")
            graph_mod, vm_mod = prepare_graph(mod, mode, tvm)
            mode_report = report["graph_modes"][mode]
            mode_report["pipelines"] = ["zero", "default_build"] if mode == "optimized" else ["default_build"]
            for phase, phase_mod in (("graph", graph_mod), ("vm", vm_mod)):
                path = allowed_path(output / f"{mode}_{phase}.relax.py")
                path.write_text(phase_mod.script(), encoding="utf-8")
                phase_inventory = relax_inventory(phase_mod, tvm)
                inventory_name = f"{mode}_{phase}_inventory.json"
                write_json(inventory_name, phase_inventory)
                mode_report[phase] = {"ir": file_identity(path, allowed_path), "inventory": inventory_name,
                                      "operator_counts": phase_inventory["operator_counts"], "function_counts": phase_inventory["function_counts"]}
            stage(f"relax_build_llvm_{mode}")
            # The recorded default_build IR is already lowered; do not apply it twice.
            executable = tvm.relax.build(vm_mod, target="llvm", pipeline=None)
            vm = tvm.relax.VirtualMachine(executable, tvm.cpu())
            for index, image in enumerate(images):
                stage(f"relax_execute_{mode}_image_{index}")
                result = vm["main"](tvm.nd.array(image.numpy(), tvm.cpu()))
                actual = result.numpy() if hasattr(result, "numpy") else result[0].numpy()
                expected = expected_outputs[index]
                image_arrays[index]["relax_" + mode] = actual
                if mode == modes[0]:
                    image_arrays[index]["relax"] = actual
                np.savez(allowed_path(output / f"image_{index}.npz"), **image_arrays[index])
                comparison = compare_output(actual, expected, np)
                mode_values = {"relax_output": helper.tensors([actual])[0], "relax_vs_torch": comparison}
                if mode == modes[0]:
                    report["images"].append({"index": index, "input_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                                             "torch_output": helper.tensors([expected])[0], "finite": True, "graph_mode_comparisons": {}})
                    report["images"][index].update(mode_values)
                if labels is not None:
                    sample = labels["samples"][index]
                    mode_values["classification"] = {"sample_id": sample["id"], "label": sample["label"],
                                                       "torch": classify_logits(expected, sample["label"], np), "relax": classify_logits(actual, sample["label"], np)}
                    if mode == modes[0]:
                        report["images"][index]["classification"] = mode_values["classification"]
                report["images"][index]["graph_mode_comparisons"][mode] = mode_values
                mode_report["checked_images"] = index + 1
                report["session"]["checked_images"] = min(row["checked_images"] for row in report["graph_modes"].values())
                write_json("results.json", report)
            if labels is not None:
                mode_report["descriptive_accuracy"] = accuracy_counts([row["graph_mode_comparisons"][mode]["classification"] for row in report["images"]])
        if labels is not None:
            report["descriptive_accuracy"] = report["graph_modes"][modes[0]]["descriptive_accuracy"]
        if loader._state_dict_sha256(backbone) != state_hash:
            raise AssertionError("Model parameters or buffers changed during verification")
        if args.batchnorm_folding and loader._state_dict_sha256(export_model.model) != report["batchnorm_folding"]["derived_state_dict_sha256"]:
            raise AssertionError("BatchNorm-folded parameters changed during verification")
        if not all(row["graph_mode_comparisons"][mode]["relax_vs_torch"]["passed"] for row in report["images"] for mode in modes):
            raise AssertionError("Relax/PyTorch output mismatch at fixed thresholds")
        for identity in report["artifacts"].values():
            if file_identity(identity["path"], allowed_path) != identity:
                raise AssertionError("Supplied artifact changed during verification")
        report["session"]["complete"] = all(row["checked_images"] == len(images) for row in report["graph_modes"].values())
        if not report["session"]["complete"]:
            raise AssertionError("Verification did not cover the entire image stream")
        report["status"] = "passed"
        stage("complete")
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_json("results.json", report)
        print(f"failed at {report['stage']}: {error}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
