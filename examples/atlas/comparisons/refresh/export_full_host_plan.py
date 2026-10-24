#!/usr/bin/env python3
"""Export source-bound full ResNet graph and offline parameters; no runtime gate."""

from merlin.common.jsonio import write_pretty_json
from merlin.common.digest import sha256_bytes
import argparse, hashlib, json, math, sys
from pathlib import Path
from resnet_native import safe_path, file_identity, load_checkpoint, load_images
from partition_resnet import digest_tensor, inventory, state_identity
from resnet_graph_inputs import explicit_patches, raw_identity
from resnet_provider import (
    Provider,
    read_source,
    load_provider,
    load_packing,
    validate_plan,
    add_export_arguments,
    source_bookend,
    finish_bookend,
    artifact_output,
)


def pair(value):
    return list(value) if isinstance(value, (tuple, list)) else [value, value]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_export_arguments(p, reference=False)
    a = p.parse_args()
    import numpy as np
    import torch, torchvision

    source, sourceid, relocation = read_source(a, torch, torchvision, np)
    vector, vectorid = load_provider("full_export_vector", a.vector, a.vector_sha256)
    scalar, scalarid = load_provider("full_export_scalar", a.scalar, a.scalar_sha256)
    packing, packingid, packing_sources = load_packing(a)
    bookend = source_bookend(a, [sourceid, vectorid, scalarid, packingid, *packing_sources], source)
    out = artifact_output(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sourcepath = a.source_manifest
    provider = Provider(vector, scalar, np)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    checkpoint = load_checkpoint(model, source["checkpoint"]["path"], torch)
    assert checkpoint == source["checkpoint"]
    before = state_identity(model)
    images, stream = load_images(source["input_stream"]["path"], np)
    assert stream == source["input_stream"]
    image = torch.from_numpy(images[source["sample_index"]].copy())
    assert digest_tensor(model(image))["sha256"] == source["native_logits"]["raw_sha256"]
    nodes = inventory(model, image, torch, a.tile_shape)
    original, inventoryid = relocation.read_json(a.inventory, a.inventory_sha256)
    assert nodes == original["operators"]
    memory = {"allocations": []}
    memoryid = None
    if a.memory_plan is not None:
        if not a.memory_plan_sha256:
            raise ValueError("--memory-plan requires --memory-plan-sha256")
        memory, memoryid = relocation.read_json(a.memory_plan, a.memory_plan_sha256)
    offsets = {x["node"]: x for x in memory["allocations"]}
    records = []
    params = []
    for node in nodes:
        record = dict(node)
        if node["node"] in offsets:
            record["activation_allocation"] = offsets[node["node"]]
        if node["kind"] == "call_module":
            module = model.get_submodule(node["target"])
            name = node["target"]
            if type(module) in (torch.nn.Conv2d, torch.nn.Linear):
                b = module.weight.detach().flatten(1).t().contiguous().numpy().copy()
                encoded, exponent = provider.encode(b)
                pk, pn = node["padded_mkn"][1:]
                padded = np.zeros((pk, pn), dtype=np.uint8)
                padded[: b.shape[0], : b.shape[1]] = encoded
                path = out / ("weight-" + name + ".fp8-tiled.bin")
                path.write_bytes(packing.pack_weights(padded))
                record["weight"] = {
                    "raw_fp32": {"shape": list(b.shape), "sha256": sha256_bytes(b.tobytes())},
                    "padded_shape": [pk, pn],
                    "b_scale_exponent": exponent,
                    "fp8_tiled": file_identity(path, safe_path),
                    "layout": "Ntile-major then Ktile-major, row-major 32x32 tile",
                }
                if module.bias is not None:
                    bias = module.bias.detach().numpy().copy()
                    path = out / ("bias-" + name + ".f32.bin")
                    path.write_bytes(bias.astype("<f4").tobytes())
                    record["host_bias"] = file_identity(path, safe_path)
            elif type(module) is torch.nn.BatchNorm2d:
                var = module.running_var.numpy()
                mean = module.running_mean.numpy()
                gamma = module.weight.detach().numpy()
                beta = module.bias.detach().numpy()
                scale = np.divide(
                    gamma, np.sqrt(np.add(var, np.float32(module.eps), dtype=np.float32)), dtype=np.float32
                )
                bias = np.subtract(beta, np.multiply(mean, scale, dtype=np.float32), dtype=np.float32)
                path = out / ("batchnorm-" + name + ".f32.bin")
                path.write_bytes(scale.astype("<f4").tobytes() + bias.astype("<f4").tobytes())
                record["host_params"] = {
                    "channels": len(scale),
                    "scale_then_bias": file_identity(path, safe_path),
                    "arithmetic": "FP32 multiply then FP32 add, no FMA",
                }
            elif type(module) is torch.nn.MaxPool2d:
                record["host_params"] = {
                    "kernel": pair(module.kernel_size),
                    "stride": pair(module.stride),
                    "padding": pair(module.padding),
                    "dilation": pair(module.dilation),
                    "ceil_mode": module.ceil_mode,
                    "boundary": "padding is negative infinity; maximum value only, no argmax",
                }
            elif type(module) is torch.nn.AdaptiveAvgPool2d:
                record["host_params"] = {
                    "output_size": pair(module.output_size),
                    "qualification": (
                        "Version-bound native Torch pooling reference; C54 host conformance is a separate "
                        "execution gate, not performed by this exporter"
                    ),
                }
            elif type(module) is torch.nn.ReLU:
                record["host_params"] = {
                    "inplace": module.inplace,
                    "arithmetic": (
                        "FP32 clamp negative values to positive zero; signed-zero behavior must be "
                        "retained or explicitly qualified"
                    ),
                }
        records.append(record)
    assert state_identity(model) == before and file_identity(source["checkpoint"]["path"], safe_path) == checkpoint
    (out / "image-nchw.f32.bin").write_bytes(image.numpy().astype("<f4").tobytes())
    report = {
        "status": "exported_compile_plan_only",
        "scope": (
            "Entire native ResNet graph with 53 convolutions and classifier assigned to device, other "
            "operators on host; exporter does not execute that partition"
        ),
        "source_manifest": sourceid,
        "checkpoint": checkpoint,
        "input_stream": stream,
        "image": digest_tensor(image),
        "source_script": file_identity(__file__, safe_path),
        "providers": [vectorid, scalarid, packingid],
        "inventory": inventoryid,
        "memory_plan": memoryid,
        "nodes": records,
        "contraction_count": sum(x["partition"] == "device" for x in nodes),
        "memory_projection": {k: v for k, v in memory.items() if k.endswith("bytes")},
        "execution_boundary": [
            "Host C conformance and signed-zero behavior are checked by separate C54 gates",
            "Program templates require the dynamic A exponent patch before launch",
            "This exporter supplies no instruction-model, RTL or SoC execution gate",
        ],
    }
    report["source_bookends"] = finish_bookend(bookend)
    report["source_relocation"] = relocation.report()
    report["tile_shape"] = list(a.tile_shape)
    report["artifacts"] = {x.name: file_identity(x, safe_path)["sha256"] for x in out.iterdir() if x.is_file()}
    write_pretty_json(out / "host-plan.json", report, allow_nan=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "contraction_count": report["contraction_count"],
                "artifacts": len(report["artifacts"]),
            }
        )
    )


if __name__ == "__main__":
    main()
