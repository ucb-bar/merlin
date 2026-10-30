#!/usr/bin/env python3
"""Declare a ResNet host/device partition from the actual native FX graph.

This is an experiment inventory. It does not claim automatic device compilation
or turn an estimated MAC share into measured runtime coverage.
"""

from merlin.common.jsonio import write_pretty_json
from merlin.common.digest import sha256_bytes
import argparse
import hashlib
import json
import math
import operator
from pathlib import Path
import sys

from resnet_native import safe_path, file_identity, load_checkpoint, load_images


def digest_tensor(value):
    a = value.detach().contiguous().numpy()
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_bytes(a.tobytes())}


def state_identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().contiguous().numpy()
        digest.update(
            name.encode()
            + b"\0"
            + str(array.dtype).encode()
            + b"\0"
            + json.dumps(list(array.shape)).encode()
            + b"\0"
            + array.tobytes()
        )
    return digest.hexdigest()


def inventory(model, image, torch, tile_shape):
    if len(tile_shape) != 3 or any(type(x) is not int or x <= 0 for x in tile_shape):
        raise ValueError("Require explicit positive M/K/N alignments")
    graph = torch.fx.symbolic_trace(model)
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(graph).propagate(image)
    records = []
    for node in graph.graph.nodes:
        meta = node.meta.get("tensor_meta")
        shape = list(meta.shape) if meta is not None else None
        record = {
            "node": node.name,
            "kind": node.op,
            "target": (
                node.target.__module__ + "." + node.target.__qualname__ if callable(node.target) else str(node.target)
            ),
            "output_shape": shape,
            "inputs": [x.name for x in node.all_input_nodes],
            "partition": "host",
            "macs": 0,
        }
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            record["operator"] = type(module).__name__
            if type(module) is torch.nn.Conv2d:
                if module.groups != 1 or module.bias is not None or module.padding_mode != "zeros":
                    raise ValueError("Inventory supports only the declared bias-free dense convolutions")
                m = math.prod(shape) // shape[1]
                k = module.in_channels * math.prod(module.kernel_size)
                n = module.out_channels
                record.update(
                    partition="device",
                    macs=m * k * n,
                    matrix_mkn=[m, k, n],
                    padded_mkn=[
                        ((x + alignment - 1) // alignment) * alignment for x, alignment in zip((m, k, n), tile_shape)
                    ],
                    host_preparation=(
                        "NCHW im2col with explicit stride, dilation and zero padding, "
                        "FP8 conversion and tiled packing"
                    ),
                    geometry={x: list(getattr(module, x)) for x in ("kernel_size", "stride", "padding", "dilation")},
                    weight_shape=list(module.weight.shape),
                )
            elif type(module) is torch.nn.Linear:
                m = math.prod(shape[:-1])
                k = module.in_features
                n = module.out_features
                record.update(
                    partition="device",
                    macs=m * k * n,
                    matrix_mkn=[m, k, n],
                    padded_mkn=[
                        ((x + alignment - 1) // alignment) * alignment for x, alignment in zip((m, k, n), tile_shape)
                    ],
                    host_preparation="FP8 conversion, tiled packing; bias remains host",
                    weight_shape=list(module.weight.shape),
                    host_bias=module.bias is not None,
                )
            elif type(module) not in (
                torch.nn.BatchNorm2d,
                torch.nn.ReLU,
                torch.nn.MaxPool2d,
                torch.nn.AdaptiveAvgPool2d,
            ):
                raise ValueError("Undeclared module " + type(module).__name__)
        elif node.op == "call_function":
            if node.target is operator.add:
                record["operator"] = "residual_add"
            elif node.target is torch.flatten:
                record["operator"] = "flatten"
            else:
                raise ValueError("Undeclared function " + str(node.target))
        else:
            record["operator"] = node.op
        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=safe_path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--relocate", action="append", default=[])
    p.add_argument("--tile-shape", type=int, nargs=3, required=True, metavar=("M", "K", "N"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from resnet_provider import read_source, artifact_output, source_bookend, finish_bookend

    out = artifact_output(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import torchvision
    import torchvision.models.resnet as implementation

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    manifest_path = args.source_manifest
    source, sourceid, relocation = read_source(args, torch, torchvision, np)
    bookend = source_bookend(args, [sourceid], source)
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    if versions != source["versions"]:
        raise ValueError("Native environment differs from frozen source")
    model = torchvision.models.resnet50(weights=None).eval()
    checkpoint = load_checkpoint(model, source["checkpoint"]["path"], torch)
    if checkpoint != source["checkpoint"]:
        raise ValueError("Checkpoint identity differs")
    images, input_identity = load_images(source["input_stream"]["path"], np)
    if input_identity != source["input_stream"]:
        raise ValueError("Image stream differs")
    image = torch.from_numpy(images[source["sample_index"]].copy())
    if digest_tensor(image)["sha256"] != source["input"]["raw_sha256"]:
        raise ValueError("Frozen development image differs")
    logits = model(image)
    if digest_tensor(logits)["sha256"] != source["native_logits"]["raw_sha256"]:
        raise ValueError("Native reference differs")
    records = inventory(model, image, torch, args.tile_shape)
    devices = [x for x in records if x["partition"] == "device"]
    hosts = [x for x in records if x["kind"].startswith("call_") and x["partition"] == "host"]
    from collections import Counter

    counts = dict(Counter(x["operator"] for x in records if x["kind"].startswith("call_")))
    data = {
        "schema_version": 1,
        "status": "passed_native_inventory",
        "source_manifest": sourceid,
        "checkpoint": checkpoint,
        "image_stream": input_identity,
        "source_script": file_identity(__file__, safe_path),
        "torchvision_resnet_source": file_identity(implementation.__file__, safe_path),
        "versions": versions,
        "workload": source["workload"],
        "counts": counts,
        "operators": records,
        "partition": {
            "device": "All 53 convolutions and final classifier matrix product through padded ordered M/N/K contraction routes",
            "host": (
                "BatchNorm, ReLU, residual additions, max and average pooling, flatten, classifier bias, "
                "im2col, conversion and packing"
            ),
            "contraction_macs": sum(x["macs"] for x in devices),
            "contraction_mac_share": 1.0,
            "share_meaning": (
                "All dense convolution and linear MAC work is assigned to the device; this excludes host "
                "scalar work, padded work, conversion and traffic, and is not a speedup estimate"
            ),
            "device_operator_count": len(devices),
            "host_operator_count": len(hosts),
        },
        "checks": {"native_logits_raw_exact": True, "checkpoint_raw_exact": True, "full_graph_declared": True},
        "implementation_status": "Declared full partition only; executable coverage is recorded by separate per-scope receipts",
        "native_gate": {"rtol": 1e-4, "atol": 1e-4, "unchanged": True},
        "task_quality_policy": "No task-quality evaluation is performed by the inventory",
    }
    data["source_bookends"] = finish_bookend(bookend)
    data["source_relocation"] = relocation.report()
    data["tile_shape"] = list(args.tile_shape)
    write_pretty_json(out / "partition.json", data)
    np.save(out / "native-logits.npy", logits.numpy())
    lines = [
        "# ResNet50 declared partition",
        "",
        data["partition"]["device"] + ".",
        "",
        data["partition"]["host"] + ".",
        "",
        (
            f"The device is assigned {data['partition']['contraction_macs']:,} MACs across {len(devices)} "
            "contraction sites; this is 100% of convolution/linear MACs, and is not a runtime share estimate."
        ),
        "",
        "| Node | Operator | Output | Partition | M,K,N | MACs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for x in records:
        if x["kind"].startswith("call_"):
            lines.append(
                (
                    f"| {x['target']} | {x['operator']} | {x['output_shape']} | {x['partition']} | "
                    f"{x.get('matrix_mkn','')} | {x['macs']} |"
                )
            )
    (out / "partition.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": data["status"], "counts": counts, "macs": data["partition"]["contraction_macs"]}))


if __name__ == "__main__":
    main()
