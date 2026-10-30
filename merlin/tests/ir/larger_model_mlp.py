"""A complete runtime-weight MLP and an independent ordered-policy reference.

This is a newly constructed network, not a pretrained or recaptured model.
Every contraction contributes to the final logits. The reference follows the
selected draft f32-accumulation policy, not the original half accumulator.
"""

from __future__ import annotations

import hashlib


def fixture(*, layers=8, width=128, batch=8, classes=31):
    if any(type(value) is not int or value < 1 for value in (layers, width, batch, classes)):
        raise ValueError("positive static network dimensions required")
    arguments = [{"name": "x", "shape": [batch, width], "dtype": "float16"}]
    for layer in range(layers):
        output = classes if layer == layers - 1 else width
        arguments.extend(
            [
                {"name": f"w{layer}", "shape": [width, output], "dtype": "float16"},
                {"name": f"b{layer}", "shape": [output], "dtype": "float16"},
            ]
        )

    def tensor(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf16>"

    signature = ", ".join(f"%{row['name']}: {tensor(row['shape'])}" for row in arguments)
    returned = tensor([batch, classes])
    lines = [
        "module {",
        f"  func.func @forward({signature}) -> {returned} attributes {{llvm.emit_c_interface}} {{",
        "    %zero = arith.constant 0.000000e+00 : f16",
    ]
    current = "%x"
    for layer in range(layers):
        output = classes if layer == layers - 1 else width
        incoming, weight, result, bias = map(tensor, ([batch, width], [width, output], [batch, output], [output]))
        lines.extend(
            [
                f"    %init{layer} = tensor.splat %zero : {result}",
                f"    %mm{layer} = linalg.matmul ins({current}, %w{layer} : {incoming}, {weight}) "
                f"outs(%init{layer} : {result}) -> {result}",
                f"    %empty{layer} = tensor.empty() : {result}",
                f"    %out{layer} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
                "affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], "
                'iterator_types = ["parallel", "parallel"]} '
                f"ins(%mm{layer}, %b{layer} : {result}, {bias}) outs(%empty{layer} : {result}) {{",
                f"    ^bb0(%m{layer}: f16, %bias{layer}: f16, %unused{layer}: f16):",
                f"      %add{layer} = arith.addf %m{layer}, %bias{layer} : f16",
            ]
        )
        if layer < layers - 1:
            lines.append(f"      %relu{layer} = arith.maximumf %add{layer}, %zero : f16")
        selected = f"%relu{layer}" if layer < layers - 1 else f"%add{layer}"
        lines.extend([f"      linalg.yield {selected} : f16", f"    }} -> {result}"])
        current = f"%out{layer}"
    lines.extend([f"    func.return {current} : {returned}", "  }", "}", ""])
    source = "\n".join(lines)
    for row in arguments:
        row["canonical_input_name"] = row["name"]
    return {
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "arguments": arguments,
        "outputs": [{"name": "logits", "shape": [batch, classes], "dtype": "float16"}],
        "model": {
            "kind": "constructed_connected_runtime_weight_mlp",
            "layers": layers,
            "width": width,
            "batch": batch,
            "classes": classes,
            "weight_scale_denominator": 16,
            "activation": "half bias addition and ReLU after each nonfinal contraction",
            "pretrained": False,
            "captured_framework_model": False,
            "policy": "draft ordered f32 accumulation from positive zero, then one f16 truncation",
        },
    }


def inputs_and_reference(value, *, seed=20261029):
    """Independent mathematical execution in explicit reduction order.

    Inputs are small dyadic values. Each multiply and accumulator update is
    rounded to binary32, followed by the actual per-layer half truncation and
    separate half bias operation. No compiler operation or output is inspected.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    inputs = {}
    for row in value["arguments"]:
        name = row["name"]
        scale = value["model"]["weight_scale_denominator"] if name.startswith("w") else 32
        bound = 2 if name.startswith("w") else 4
        inputs[name] = (rng.integers(-bound, bound + 1, size=row["shape"]).astype(np.float32) / scale).astype(
            np.float16
        )
    current = inputs["x"]
    stages = []
    for layer in range(value["model"]["layers"]):
        weight, bias = inputs[f"w{layer}"], inputs[f"b{layer}"]
        accumulator = np.zeros((current.shape[0], weight.shape[1]), dtype=np.float32)
        for index in range(current.shape[1]):
            product = np.multiply(
                current[:, index, None].astype(np.float32), weight[None, index, :].astype(np.float32), dtype=np.float32
            )
            accumulator = np.add(accumulator, product, dtype=np.float32)
        contracted = accumulator.astype(np.float16)
        added = np.add(contracted.astype(np.float32), bias.astype(np.float32), dtype=np.float32).astype(np.float16)
        current = np.maximum(added, np.float16(0)) if layer < value["model"]["layers"] - 1 else added
        if not np.isfinite(current).all():
            raise ValueError("demo's independently evaluated network must remain finite")
        stages.append(
            {
                "layer": layer,
                "shape": list(current.shape),
                "sha256": array_sha256(current),
                "distinct_batch_rows": len(np.unique(current, axis=0)),
            }
        )
    return inputs, {"logits": current}, stages


def array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
