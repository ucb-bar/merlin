"""ResNet50 (torchvision) with its 53 convolutions and classifier on the emulated MXU.

--artifact-dir is a PR #26 provider directory: host-plan.json gives each contraction's padded shape, and batch norm
uses its folded FP32 parameters. --input is one raw float32 NCHW image or an NPZ images array, and --labels a JSON
{"samples": [{"label": ...}]}. --reference is the provider arithmetic receipt for one image, for the default recipe.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from . import arithmetic


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class Runner(torch.fx.Interpreter):
    """Replace the contractions and keep the provider's host arithmetic."""

    def __init__(self, graph, nodes, batchnorm, linear):
        super().__init__(graph)
        self.nodes, self.batchnorm, self.linear, self.outputs = nodes, batchnorm, linear, []

    def call_module(self, target, args, kwargs):
        module, node = self.fetch_attr(target), self.nodes[target]
        if isinstance(module, torch.nn.Conv2d):
            left = torch.nn.functional.unfold(args[0], module.kernel_size, dilation=module.dilation,
                                              padding=module.padding, stride=module.stride)[0].T
            self.outputs.append(self.linear(node["node"], left, lambda: module.weight.flatten(1).T))
            return self.outputs[-1].T.reshape(node["output_shape"])
        if isinstance(module, torch.nn.Linear):
            left = args[0].reshape(-1, module.in_features)
            self.outputs.append(self.linear(node["node"], left, lambda: module.weight.T))
            return self.outputs[-1].reshape(node["output_shape"]) + module.bias
        if isinstance(module, torch.nn.BatchNorm2d):
            scale, bias = self.batchnorm[target]
            return args[0] * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            data = args[0].flatten(2)
            total = torch.zeros(data.shape[:2])
            for i in range(data.shape[2]):
                total = total + data[:, :, i]
            return (total / data.shape[2]).reshape(*data.shape[:2], 1, 1)
        return module(*args, **kwargs)


def run(args):
    plan = json.loads((args.artifact_dir / "host-plan.json").read_text())
    nodes = {node["target"]: node for node in plan["nodes"] if node["kind"] == "call_module"}
    batchnorm = {target: torch.from_numpy(np.fromfile(
        args.artifact_dir / Path(node["host_params"]["scale_then_bias"]["path"]).name, np.float32).reshape(2, -1))
        for target, node in nodes.items() if node["operator"] == "BatchNorm2d"}
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    graph = torch.fx.symbolic_trace(model.eval())
    images = np.load(args.input)["images"] if args.input.suffix == ".npz" else np.fromfile(args.input, np.float32)
    labels = [s["label"] for s in json.loads(args.labels.read_text())["samples"]] if args.labels else None
    reference = json.loads(args.reference.read_text()) if args.reference else None
    linear = arithmetic.Linear(args)
    records = []
    for index, image in enumerate(torch.from_numpy(images.reshape(-1, 1, 3, 224, 224))):
        native = model(image).flatten()
        runner = Runner(graph, nodes, batchnorm, linear)
        logits = runner.run(image).flatten()
        records.append(dict(top1=int(logits.argmax()), native_top1=int(native.argmax()), logits_sha256=sha(logits),
                            native_logits_sha256=sha(native), relative_l2=float(
                                torch.linalg.vector_norm(logits.double() - native.double())
                                / torch.linalg.vector_norm(native.double()))))
        if labels:
            records[-1]["label"] = labels[index]
        if reference:
            records[-1]["reference_exact"] = sha(logits) == reference["final_logits"]["sha256"] and [
                sha(x) for x in runner.outputs] == [kernel["target_output"]["sha256"] for kernel in reference["kernels"]]
    # Extra MXU K32 steps and padded weight bytes of two-term sites; K split adds partial stores instead.
    shapes = [(node["node"], node["padded_mkn"]) for node in nodes.values() if "padded_mkn" in node]
    extra = [arithmetic.settings(args, site)[3] for site, _ in shapes]
    work = [(m * k * n, k * n) for _, (m, k, n) in shapes]
    return dict(model="resnet50", records=records,
                top1_agreement=float(np.mean([r["top1"] == r["native_top1"] for r in records])),
                label_top1=float(np.mean([r["top1"] == r["label"] for r in records])) if labels else None,
                mean_relative_l2=float(np.mean([r["relative_l2"] for r in records])),
                extra_mxu_work=sum(w for (w, _), e in zip(work, extra) if e) / sum(w for w, _ in work),
                extra_weight_bytes=sum(b for (_, b), e in zip(work, extra) if e) / sum(b for _, b in work))
