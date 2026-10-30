#!/usr/bin/env python3
"""Export one full target-arithmetic development graph for host C conformance.

This exporter performs arithmetic emulation; this is not a fresh
instruction-model, RTL or SoC execution and does not use saved intermediates.
"""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_export_arguments(p, reference=True)
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
    planpath = a.host_plan
    plan, planid = relocation.read_json(planpath, a.host_plan_sha256)
    validate_plan(plan, source, sourceid, [vectorid, scalarid, packingid])
    ids = [vectorid, scalarid, packingid]
    provider = Provider(vector, scalar, np)
    plan_sites = {x["target"]: x for x in plan["nodes"] if x["partition"] == "device"}
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
    assert digest_tensor(image) == plan["image"]
    assert digest_tensor(model(image))["sha256"] == source["native_logits"]["raw_sha256"]
    graph = torch.fx.symbolic_trace(model)
    critical = set()
    expected_sites = {
        name for name, module in graph.named_modules() if type(module) in (torch.nn.Conv2d, torch.nn.Linear)
    }
    for node in graph.graph.nodes:
        if node.op == "call_module" and type(graph.get_submodule(node.target)) in (
            torch.nn.Conv2d,
            torch.nn.Linear,
            torch.nn.MaxPool2d,
            torch.nn.AdaptiveAvgPool2d,
        ):
            critical.add(node.name)
            critical.update(x.name for x in node.all_input_nodes)
        if node.op == "output":
            critical.update(x.name for x in node.all_input_nodes)
    kernels = []
    nodes = []

    class Interpreter(torch.fx.Interpreter):
        def call_module(self, target, args, kwargs):
            module = self.fetch_attr(target)
            native = module(*args, **kwargs)
            if type(module) in (torch.nn.Conv2d, torch.nn.Linear):
                activation = args[0].detach().contiguous().numpy().copy()
                if type(module) is torch.nn.Conv2d:
                    left = explicit_patches(activation, module, native.shape, np)
                    unfold = (
                        torch.nn.functional.unfold(
                            args[0],
                            module.kernel_size,
                            dilation=module.dilation,
                            padding=module.padding,
                            stride=module.stride,
                        )[0]
                        .t()
                        .contiguous()
                        .numpy()
                    )
                    assert np.array_equal(left.view(np.uint32), unfold.view(np.uint32))
                    right = module.weight.detach().flatten(1).t().contiguous().numpy()
                else:
                    left = activation.reshape(-1, module.in_features)
                    right = module.weight.detach().t().contiguous().numpy()
                ea, sa = provider.encode(left)
                eb, sb = provider.encode(right)
                weight = plan_sites[target]["weight"]
                if (
                    weight["raw_fp32"]["sha256"] != sha256_bytes(right.tobytes())
                    or weight["b_scale_exponent"] != sb
                ):
                    raise ValueError("Host plan weight source/scale differs")
                pk, pn = weight["padded_shape"]
                padded_b = np.zeros((pk, pn), dtype=np.uint8)
                padded_b[: eb.shape[0], : eb.shape[1]] = eb
                weightid = weight["fp8_tiled"]
                if file_identity(weightid["path"], safe_path) != weightid:
                    raise ValueError("Host plan weight payload digest differs")
                if packing.pack_weights(padded_b) != safe_path(weightid["path"]).read_bytes():
                    raise ValueError("Regenerated weight payload differs from host plan")
                bits = vector.rescale_bf16_array(vector.ordered_matmul(ea, eb), sa + sb)
                value = (bits.astype(np.uint32) << 16).view(np.float32)
                m, k = left.shape
                n = right.shape[1]
                pm, pk, pn = plan_sites[target]["padded_mkn"]
                assert plan_sites[target]["matrix_mkn"] == [m, k, n]
                pa = np.zeros((pm, pk), dtype=np.uint8)
                pa[:m, :k] = ea
                pc = np.zeros((pm, pn), dtype=np.uint16)
                pc[:m, :n] = bits
                directory = out / f"kernel-{len(kernels):02d}-{target}"
                directory.mkdir()
                path_a = directory / "a-fp8-tiled.bin"
                path_c = directory / "expected-bf16-tiles.bin"
                path_a.write_bytes(packing.pack_activation(pa))
                path_c.write_bytes(packing.pack_output(pc))
                kernels.append(
                    {
                        "site": target,
                        "matrix_mkn": [m, k, n],
                        "padded_mkn": [pm, pk, pn],
                        "activation": raw_identity(activation),
                        "a_matrix_fp32": raw_identity(left),
                        "a_scale_exponent": sa,
                        "b_scale_exponent": sb,
                        "rescale_exponent": sa + sb,
                        "encoded_a_tiled": file_identity(path_a, safe_path),
                        "expected_physical_output": file_identity(path_c, safe_path),
                        "target_output": raw_identity(value),
                    }
                )
                if type(module) is torch.nn.Conv2d:
                    return torch.from_numpy(value.T.reshape(tuple(native.shape)).copy())
                result = torch.from_numpy(value.reshape(tuple(native.shape)).copy())
                return result + module.bias if module.bias is not None else result
            if type(module) is torch.nn.BatchNorm2d:
                x = args[0].detach().contiguous().numpy()
                scale = np.divide(
                    module.weight.detach().numpy(),
                    np.sqrt(np.add(module.running_var.numpy(), np.float32(module.eps), dtype=np.float32)),
                    dtype=np.float32,
                )
                bias = np.subtract(
                    module.bias.detach().numpy(),
                    np.multiply(module.running_mean.numpy(), scale, dtype=np.float32),
                    dtype=np.float32,
                )
                return torch.from_numpy(
                    np.add(
                        np.multiply(x, scale.reshape(1, -1, 1, 1), dtype=np.float32),
                        bias.reshape(1, -1, 1, 1),
                        dtype=np.float32,
                    ).copy()
                )
            return native

        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                entry = {
                    "node": node.name,
                    "kind": node.op,
                    "target": (
                        node.target.__module__ + "." + node.target.__qualname__
                        if callable(node.target)
                        else str(node.target)
                    ),
                    "output": digest_tensor(result),
                    "inputs": [x.name for x in node.all_input_nodes],
                }
                if node.name in critical:
                    path = out / ("node-" + node.name + ".f32.bin")
                    path.write_bytes(result.detach().contiguous().numpy().astype("<f4").tobytes())
                    entry["payload"] = file_identity(path, safe_path)
                nodes.append(entry)
            return result

    result = Interpreter(graph).run(image).detach().clone()
    assert len(kernels) == 54 and {x["site"] for x in kernels} == expected_sites
    assert state_identity(model) == before
    path = out / "target-logits.f32.bin"
    path.write_bytes(result.numpy().astype("<f4").tobytes())
    report = {
        "status": "exported_arithmetic_reference_only",
        "scope": (
            "one selected frozen development image, full 54-contraction target arithmetic graph, no "
            "saved intermediate substitution"
        ),
        "source_script": file_identity(__file__, safe_path),
        "source_manifest": sourceid,
        "host_plan": planid,
        "checkpoint": checkpoint,
        "input_stream": stream,
        "image": digest_tensor(image),
        "providers": ids,
        "kernels": kernels,
        "nodes": nodes,
        "final_logits": digest_tensor(result),
        "state_before_after": before,
        "pool_contract": (
            "Torch source-version native operators for maxpool and adaptiveavgpool; exact host C "
            "conformance requires independent checks on exported inputs/outputs"
        ),
        "not_claimed": [
            "instruction-model execution",
            "RTL or SoC execution",
            "hostC correctness",
            "heldout task quality",
        ],
    }
    report["source_bookends"] = finish_bookend(bookend)
    report["source_relocation"] = relocation.report()
    write_pretty_json(out / "receipt.json", report, allow_nan=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "nodes": len(nodes),
                "payload_nodes": len(critical),
                "kernels": len(kernels),
                "logits": report["final_logits"],
            }
        )
    )


if __name__ == "__main__":
    main()
