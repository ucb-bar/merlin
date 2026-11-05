#!/usr/bin/env python3
"""Generate an explicit batch1 host graph with a replaceable device callback."""
from atlas_host_layout import ATLAS_DRAM_BASE
from atlas_host_build import GRAPH_PROGRAM_WORDS
from pathlib import Path
import json
import math
import argparse
import struct

try:
    from .atlas_path_guard import checked
except ImportError:
    import kernel_library  # noqa: E402,F401
    from atlas_path_guard import checked
from atlas_tooling import write_json, sha


def shape(value):
    if (not isinstance(value, list) or len(value) not in (2, 4) or value[0] != 1
        or any(type(x) is not int or x <= 0 for x in value)):
        raise ValueError("require positive batch1 rank2/rank4 shapes")
    return value


def validate_graph(plan, templates, read_asset, *, require_full=True):
    nodes = plan.get("nodes", [])
    if not nodes or (require_full and (len(nodes) != 177 or len(templates) != 54)):
        raise ValueError("unexpected complete graph inventory")
    index = {}
    uses = {}
    kernels = []
    params = {}
    total = 0
    for i, n in enumerate(nodes):
        name = n.get("node")
        op = n.get("operator")
        out = shape(n.get("output_shape"))
        if not isinstance(name, str) or not name or name in index:
            raise ValueError("duplicate or missing node identity")
        ins = n.get("inputs")
        arity = 0 if op == "placeholder" else 2 if op == "residual_add" else 1
        if not isinstance(ins, list) or len(ins) != arity or any(x not in index for x in ins):
            raise ValueError("arity/topology differs")
        incoming = [nodes[index[x]]["output_shape"] for x in ins]
        for x in ins:
            uses[x] = uses.get(x, 0) + 1
        count = math.prod(out)
        if count > 2048 * 224 * 224:
            raise ValueError("tensor exceeds bounded host geometry")
        total += 0 if op == "output" else count * 4
        if total > 192 * 1024**2:
            raise ValueError("nonalias host activation arena exceeds192MiB")
        if n.get("partition") != ("device" if op in ("Conv2d", "Linear") else "host"):
            raise ValueError("operator partition differs")
        param = None
        if op == "placeholder":
            if i != 0 or out != ([1, 3, 224, 224] if require_full else out):
                raise ValueError("one leading input required")
        elif op in ("Conv2d", "Linear"):
            if len(kernels) >= len(templates):
                raise ValueError("missing contraction template")
            t = templates[len(kernels)]
            source = incoming[0]
            if op == "Conv2d":
                if len(source) != 4 or len(out) != 4:
                    raise ValueError("convolution rank differs")
                _, c, h, w = source
                g = n.get("geometry", {})
                for key in ("kernel_size", "stride", "padding", "dilation"):
                    pair = g.get(key)
                    if (not isinstance(pair, list) or len(pair) != 2 or pair[0] != pair[1]
                        or any(type(x) is not int for x in pair)):
                        raise ValueError("convolution geometry must be symmetric")
                kernel, stride, padding = g["kernel_size"][0], g["stride"][0], g["padding"][0]
                if (kernel not in (1, 3, 7) or stride not in (1, 2) or not 0 <= padding <= 3 or g["dilation"] != [1, 1]
                    or g.get("groups", n.get("groups", 1)) != 1 or not 1 <= c <= 2048 or not 1 <= h <= 224
                    or not 1 <= w <= 224 or min(h + 2 * padding, w + 2 * padding) < kernel):
                    raise ValueError("unsupported convolution geometry/groups")
                oh = (h + 2 * padding - kernel) // stride + 1
                ow = (w + 2 * padding - kernel) // stride + 1
                channels = out[1]
                if out != [1, channels, oh, ow] or n.get("weight_shape") != [channels, c, kernel, kernel]:
                    raise ValueError("convolution output geometry or groups 1 weight proof differs")
                m, k, nn = oh * ow, c * kernel * kernel, channels
            else:
                if len(source) != 2 or len(out) != 2 or n.get("weight_shape") != [out[1], source[1]]:
                    raise ValueError("linear geometry/weight shape differs")
                c, h, w = source[1], 1, 1
                kernel, stride, padding = 1, 1, 0
                oh, ow, channels = 1, 1, out[1]
                m, k, nn = 1, c, channels
                param = n.get("host_bias")
                if not param:
                    raise ValueError("explicit classifier bias required")
                params[i] = (param, nn * 4)
            if channels > 2048:
                raise ValueError("output channel bound exceeded")
            padded = [(x + 31) // 32 * 32 for x in (m, k, nn)]
            mp, kp, np = padded
            if kp > 8192 or mp * np > 1048576:
                raise ValueError("contraction resource bound exceeded")
            weight = n.get("weight", {})
            if (n.get("matrix_mkn") != [m, k, nn] or n.get("padded_mkn") != padded or n.get("macs") != m * k * nn
                or weight.get("padded_shape") != [kp, np] or weight.get("raw_fp32", {}).get("shape") != [k, nn]
                or t.get("site") != n.get("target") or t.get("matrix_mkn") != [m, k, nn] or t.get("padded_mkn") != padded
                or t.get("weight") != weight or t.get("b_scale_exponent") != weight.get("b_scale_exponent")
                or type(t.get("b_scale_exponent")) is not int or not -126 <= t["b_scale_exponent"] <= 127):
                raise ValueError("logical/padded contraction metadata or weights differ")
            extents = {"input_bytes": mp * kp + kp * np, "output_bytes": mp * np * 2,
                "output_offset": mp * kp + kp * np + 256, "total_dram_bytes": mp * kp + kp * np + mp * np * 2 + 512}
            if (t.get("dram_base") != ATLAS_DRAM_BASE or any(t.get(key) != value for key, value in extents.items())
                or extents["total_dram_bytes"] > 64 * 1024**2 or type(t.get("program_words")) is not int
                or not 1 <= t["program_words"] <= GRAPH_PROGRAM_WORDS or type(t.get("rescale_patch_word_index")) is not int
                or not 0 <= t["rescale_patch_word_index"] < t["program_words"] - 1):
                raise ValueError("template physical extents/patch bounds differ")
            raw = read_asset(weight["fp8_tiled"])
            if len(raw) != kp * np:
                raise ValueError("tiled weight extent differs")
            kernels.append(dict(kernel_index=len(kernels), node_index=i, node=name, site=n["target"], input_node=ins[0],
                geometry=[c, h, w, kernel, stride, padding], template=t, groups_proven=1,
                groups_proof="weight shape equals full input channels (no grouped contraction)"))
        elif op == "BatchNorm2d":
            hp = n.get("host_params", {})
            if (len(out) != 4 or out != incoming[0] or out[1] > 2048 or hp.get("channels") != out[1]
                or hp.get("arithmetic") != "FP32 multiply then FP32 add, no FMA"):
                raise ValueError("batchnorm shape/arithmetic differs")
            params[i] = (hp["scale_then_bias"], out[1] * 8)
        elif op == "ReLU":
            if out != incoming[0]:
                raise ValueError("unary shape differs")
        elif op == "residual_add":
            if incoming[0] != out or incoming[1] != out:
                raise ValueError("residual shape/broadcast differs")
        elif op == "MaxPool2d":
            hp = n.get("host_params", {})
            if (len(incoming[0]) != 4 or hp.get("kernel") != [3, 3] or hp.get("stride") != [2, 2]
                or hp.get("padding") != [1, 1] or hp.get("dilation") != [1, 1] or hp.get("ceil_mode") is not False):
                raise ValueError("only exact maxpool3/2/1 no-ceil is admitted")
            _, c, h, w = incoming[0]
            if out != [1, c, (h + 1) // 2, (w + 1) // 2]:
                raise ValueError("maxpool output shape differs")
        elif op == "AdaptiveAvgPool2d":
            if (len(incoming[0]) != 4 or n.get("host_params", {}).get("output_size") != [1, 1]
                or out != [1, incoming[0][1], 1, 1]):
                raise ValueError("adaptive average shape differs")
        elif op == "flatten":
            if len(incoming[0]) != 4 or out != [1, math.prod(incoming[0][1:])]:
                raise ValueError("flatten shape/order differs")
        elif op == "output":
            if i != len(nodes) - 1 or out != incoming[0] or (require_full and out != [1, 1000]):
                raise ValueError("one final output with exact shape required")
        else:
            raise ValueError("unsupported operator " + str(op))
        index[name] = i
    if (nodes[-1]["operator"] != "output" or len(kernels) != len(templates)
        or any(n["node"] not in uses for n in nodes[:-1])):
        raise ValueError("unused graph nodes or incomplete contraction inventory")
    for item, extent in params.values():
        raw = read_asset(item)
        if len(raw) != extent or any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
            raise ValueError("parameter size/finite FP32 contract differs")
    return kernels, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    planpath = checked(args.plan)
    planraw = planpath.read_bytes()
    plan = json.loads(planraw)
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh generated graph required")
    if any(ch in str(out) for ch in ('"', "\\", "\n", "\r")):
        raise ValueError("unsafe assembly output path")
    inputs = {str(planpath): sha(planraw), str(Path(__file__).absolute()): sha(Path(__file__).read_bytes())}
    templatespath = checked(args.templates)
    traw = templatespath.read_bytes()
    template_receipt = json.loads(traw)
    templates = template_receipt["kernels"]
    inputs[str(templatespath)] = sha(traw)
    if (plan.get("status") != "exported_compile_plan_only" or template_receipt.get("status") != "passed_compile_only"
        or template_receipt["host_plan"]["sha256"] != sha(planraw)):
        raise ValueError("plan/template provenance differs")

    def read_asset(item):
        path = checked(Path(item["path"]))
        raw = path.read_bytes()
        if sha(raw) != item["sha256"]:
            raise ValueError("asset digest differs")
        inputs[str(path)] = sha(raw)
        return raw

    validated_kernels, validated_params = validate_graph(plan, templates, read_asset)
    out.mkdir(parents=True)
    nodes = plan["nodes"]
    index = {n["node"]: i for i, n in enumerate(nodes)}
    if len(index) != len(nodes) or sum(n["partition"] == "device" for n in nodes) != 54:
        raise ValueError("unexpected graph inventory")
    code = ["#include <stdint.h>", "#include <stddef.h>",
        "typedef int (*atlas_graph_device)(void*,unsigned,const uint8_t*,size_t,int,uint8_t*,size_t);",
        "int atlas_graph_pack(const float*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,uint8_t*,size_t,int*);",
        "int atlas_graph_decode(const uint8_t*,size_t,float*,unsigned,unsigned,unsigned);",
        "int atlas_graph_affine(const float*,float*,const float*,const float*,unsigned,size_t);",
        "int atlas_graph_relu(const float*,float*,size_t);", "int atlas_graph_add(const float*,const float*,float*,size_t);",
        "int atlas_graph_maxpool(const float*,float*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned);",
        "int atlas_graph_average(const float*,float*,unsigned,size_t);"]
    assembly = ['.section .rodata,"a",@progbits']
    params = {}
    for i, n in enumerate(nodes):
        count = math.prod(n["output_shape"])
        if not 0 < count <= 2048 * 224 * 224:
            raise ValueError("unadmitted tensor extent")
        if n["operator"] != "output":
            code.append(f"static float tensor_{i}[{count}] __attribute__((aligned(64)));")
        source = None
        if n["operator"] == "BatchNorm2d":
            source = n["host_params"]["scale_then_bias"]
        elif n["operator"] == "Linear":
            source = n.get("host_bias")
        if source:
            path = checked(Path(source["path"]))
            data = path.read_bytes()
            if sha(data) != source["sha256"]:
                raise ValueError("changed parameter asset")
            inputs[str(path)] = sha(data)
            local = out / f"params-{i}.bin"
            local.write_bytes(data)
            label = f"params_{i}"
            params[i] = label
            code.append(f"extern const float {label}[];")
            assembly += [".balign 64", f".global {label}", f"{label}:", f'.incbin "{local}"', f".size {label}, .-{label}"]
    max_a = max(t["padded_mkn"][0] * t["padded_mkn"][1] for t in templates)
    max_output = max(t["output_bytes"] for t in templates)
    code += [f"static uint8_t packed_a[{max_a}] __attribute__((aligned(64)));",
        f"static uint8_t physical_output[{max_output}] __attribute__((aligned(64)));",
        "int atlas_resnet_graph_run(const float *input,float *output,void *context,atlas_graph_device device) {",
        "  if (!input || !output || !device) return -1;", "  int scale=0;"]
    kernels = []
    ki = 0
    for i, n in enumerate(nodes):
        op = n["operator"]
        count = math.prod(n["output_shape"])
        ins = [index[name] for name in n["inputs"]]
        if any(j >= i for j in ins):
            raise ValueError("graph is not topologically ordered")
        dst = f"tensor_{i}"
        if op == "placeholder":
            code.append(f"  for (size_t j=0;j<{count};++j) {dst}[j]=input[j];")
        elif op in ("Conv2d", "Linear"):
            t = templates[ki]
            if n["target"] != t["site"]:
                raise ValueError("changed contraction site")
            shape = nodes[ins[0]]["output_shape"]
            if op == "Conv2d":
                _, channels, h, w = shape
                g = n["geometry"]
                if g["dilation"] != [1, 1] or any(g[key][0] != g[key][1] for key in ("kernel_size", "stride", "padding")):
                    raise ValueError("unsupported convolution geometry")
                kernel, stride, padding = g["kernel_size"][0], g["stride"][0], g["padding"][0]
                _, out_channels, oh, ow = n["output_shape"]
            else:
                if len(shape) != 2 or shape[0] != 1:
                    raise ValueError("unsupported classifier input")
                channels, h, w = shape[1], 1, 1
                kernel, stride, padding = 1, 1, 0
                out_channels, oh, ow = n["output_shape"][1], 1, 1
            a_bytes = t["padded_mkn"][0] * t["padded_mkn"][1]
            code += [
                f"  if (atlas_graph_pack(tensor_{ins[0]},{channels},{h},{w},{kernel},{stride},{
                padding}," f"packed_a,sizeof packed_a,&scale)) return {i+1};",
                f'  if (device(context,{ki},packed_a,{a_bytes},scale,physical_output,{t["output_bytes"]})) return {i+1};',
                f'  if (atlas_graph_decode(physical_output,{t["output_bytes"]},{dst},{out_channels},{oh},{ow})) return {i
                +1};']
            if op == "Linear" and i in params:
                code.append(f"  for (size_t j=0;j<{count};++j) {dst}[j]={dst}[j]+{params[i]}[j];")
            kernels.append({"kernel_index": ki, "node_index": i, "node": n["node"], "site": n["target"],
                "input_node": nodes[ins[0]]["node"], "geometry": [channels, h, w, kernel, stride, padding], "template": t})
            ki += 1
        elif op == "BatchNorm2d":
            _, channels, h, w = n["output_shape"]
            code.append(f"  if (atlas_graph_affine(tensor_{ins[0]},{dst},{params[i]},{params[i]}+{channels}," f"{channels},{h
                *w})) return {i+1};")
        elif op == "ReLU":
            code.append(f"  if (atlas_graph_relu(tensor_{ins[0]},{dst},{count})) return {i+1};")
        elif op == "residual_add":
            code.append(f"  if (atlas_graph_add(tensor_{ins[0]},tensor_{ins[1]},{dst},{count})) return {i+1};")
        elif op == "MaxPool2d":
            _, channels, h, w = nodes[ins[0]]["output_shape"]
            g = n["host_params"]
            if g["ceil_mode"] or g["dilation"] != [1, 1]:
                raise ValueError("unsupported maxpool")
            code.append(f"  if (atlas_graph_maxpool(tensor_{ins[0]},{dst},{channels},{h},{w},3,2,1)) return {i+1};")
        elif op == "AdaptiveAvgPool2d":
            _, channels, h, w = nodes[ins[0]]["output_shape"]
            if n["host_params"]["output_size"] != [1, 1]:
                raise ValueError("unsupported adaptive pool")
            code.append(f"  if (atlas_graph_average(tensor_{ins[0]},{dst},{channels},{h*w})) return {i+1};")
        elif op == "flatten":
            code.append(f"  for (size_t j=0;j<{count};++j) {dst}[j]=tensor_{ins[0]}[j];")
        elif op == "output":
            code.append(f"  for (size_t j=0;j<{count};++j) output[j]=tensor_{ins[0]}[j];")
        else:
            raise ValueError("unsupported host node " + op)
    code += ["  return 0;", "}", "const float *atlas_resnet_graph_node(unsigned index,size_t *count) {",
        "  if (!count) return (const float*)0;", "  switch(index) {"]
    for i, n in enumerate(nodes):
        if n["operator"] != "output":
            code += [f"    case {i}:", f'      *count={math.prod(n["output_shape"])};', f"      return tensor_{i};"]
    code += ["    default:", "      *count=0;", "      return (const float*)0;", "  }", "}"]
    (out / "graph.c").write_text("\n".join(code) + "\n")
    (out / "constants.S").write_text("\n".join(assembly) + "\n")
    result = {"scope": "generated batch1 complete host graph; no execution by generator", "nodes": nodes, "kernels": kernels,
        "input_sha256": inputs,
        "host_average_policy": (
        "sequential FP32 sum then FP32 divide; device internal words differ from Torch, " "no cohort equivalence claim"),
        "geometry_admission": "all operators/shapes/arity/topology/54 contractions/groups 1 proven before generation",
        "validated_kernels": validated_kernels,
        "params_sha256": {f"params-{i}.bin": sha((out / f"params-{i}.bin").read_bytes()) for i in params},
        "activation_array_bytes": sum(math.prod(n["output_shape"]) * 4 for n in nodes if n["operator"] != "output"),
        "packed_a_capacity_bytes": max_a, "physical_output_capacity_bytes": max_output,
        "generated_sha256": {name: sha((out / name).read_bytes()) for name in ("graph.c", "constants.S")}}
    if any(sha(checked(Path(path)).read_bytes()) != digest for path, digest in inputs.items()):
        raise ValueError("generator source/input bookends changed")
    write_json(out / "graph.json", result)
    print(json.dumps({k: result[k] for k in ("activation_array_bytes", "packed_a_capacity_bytes",
        "physical_output_capacity_bytes")}))


if __name__ == "__main__":
    main()
