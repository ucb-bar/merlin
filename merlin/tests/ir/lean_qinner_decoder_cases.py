"""Raw archive patterns and native MLIR/C-generation observations for the concrete decoder."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from merlin.verify.lean.differential import Tools, ValueType, execute_mlir
from merlin.verify.lean.qinner import write
from merlin.verify.lean.qinner_decoder import decode_records, payload

KINDS = ("i1", "i8", "i16", "i32", "i64", "f16", "f32", "f64")
SHAPES = ((2, 8), (1, 2, 1, 8), (0, 3))
DTYPES = {
    "i1": np.bool_,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}


def patterns(kind):
    width = int(kind[1:])
    if kind == "i1":
        return [0, 1, 1, 0, 1, 0]
    if kind.startswith("i"):
        mask = (1 << width) - 1
        asymmetric = int.from_bytes(bytes(range(1, width // 8 + 1)), "little")
        return [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), mask, asymmetric]
    fraction, exponent = {"f16": (10, 5), "f32": (23, 8), "f64": (52, 11)}[kind]
    inf = ((1 << exponent) - 1) << fraction
    sign = 1 << (width - 1)
    quiet = 1 << (fraction - 1)
    return [
        0,
        sign,
        1,
        (1 << fraction) - 1,
        1 << fraction,
        inf - 1,
        inf,
        inf | sign,
        inf | quiet | 19,
        inf | 7,
        inf | sign | quiet | 23,
        {"f16": 0x3C00, "f32": 0x3F800000, "f64": 0x3FF0000000000000}[kind],
    ]


def array_from_words(kind, shape, words=None):
    words = patterns(kind) if words is None else words
    count = math.prod(shape)
    values = [words[i % len(words)] for i in range(count)]
    storage = np.uint8 if kind == "i1" else np.dtype(f"u{int(kind[1:]) // 8}")
    # View, never float conversion: signaling/quiet NaNs and signed zero keep their storage bits.
    return np.asarray(values, dtype=storage).view(DTYPES[kind]).reshape(shape), values


def record(kind, shape):
    array, words = array_from_words(kind, shape)
    blob = payload("values", array)
    return {**blob, "expected_type": copy.deepcopy(blob["type"])}, words


def c_source(kind, shape):
    ty = f"tensor<{''.join(f'{d}x' for d in shape)}{kind}>"
    return f"""module {{
 func.func @forward(%x: tensor<1xf32>) -> ({ty}, {ty}, tensor<1xf32>) {{
  %a = tensor.empty() : {ty}
  %b = tensor.empty() : {ty}
  %ac = "tensor.cast"(%a) {{prov.quant_inner_0 = "left"}} : ({ty}) -> {ty}
  %bc = "tensor.cast"(%b) {{prov.quant_inner_0 = "right"}} : ({ty}) -> {ty}
  func.return %ac, %bc, %x : {ty}, {ty}, tensor<1xf32>
 }}
}}"""


def archive_compare(kind, shape, destination):
    """Exercise actual np.savez/load, production resolve and C generation; record every byte."""
    from merlin.llvmlower import c_runtime, qinner

    destination.mkdir(parents=True, exist_ok=True)
    source = c_source(kind, shape)
    (destination / "model.mlir").write_text(source)
    (destination / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"x"}}\n')
    (destination / "input_order.json").write_text('{"x":0}\n')
    np.savez(destination / "inputs.npz", in0=np.array([3], dtype=np.float32))
    left, left_words = array_from_words(kind, shape)
    right, right_words = array_from_words(kind, shape, list(reversed(patterns(kind))))
    np.savez(destination / "extra.npz", **{"qinner::left": left, "qinner::right": right})
    plan = qinner.plan_for_bundle(destination / "model.mlir")
    with np.load(destination / "extra.npz", allow_pickle=False) as archive:
        resolved = qinner.resolve(archive, plan)
        records = []
        for entry, array in zip(plan, resolved, strict=True):
            blob = payload(entry.key, array)
            records.append({**blob, "expected_type": copy.deepcopy(blob["type"])})
    decoded = decode_records(records, destination / "decoder", archive_path=destination / "extra.npz")
    assert all(p["checked"] for p in decoded["payloads"]), decoded
    observed = [[int(x) for x in p["bits"]] for p in decoded["payloads"]]
    assert observed == [left_words, right_words]
    generated = destination / "generated"
    info = c_runtime.generate(destination, generated, destination / "inputs.npz")
    assert info["n_qinner"] == 2
    rows = [
        line
        for line in (generated / "model_gen.h").read_text().splitlines()
        if line.strip().startswith("{MERLIN_WEIGHT")
    ]
    data = (generated / "weights.bin").read_bytes()
    blobs = []
    for row, entry, array in zip(rows, plan, resolved, strict=True):
        offset = int(row.split(",")[1].strip().removesuffix("L"))
        dimensions = ",".join(str(d) for d in shape) or "0"
        expected_row = f"{{MERLIN_WEIGHT, {offset}L, {len(shape)}, {{{dimensions}}}, {c_runtime.DT_BYTES[kind]}}},"
        assert row.strip() == expected_row
        want = array.tobytes(order="C")
        actual_bytes = data[offset : offset + len(want)]
        assert actual_bytes == want
        blobs.append({"key": entry.key, "offset": offset, "hex": actual_bytes.hex(), "bytes": len(want)})
    report = {
        "kind": kind,
        "shape": shape,
        "raw_words": [left_words, right_words],
        "decoded_words": observed,
        "c_generated_blob_bytes_match": True,
        "blobs": blobs,
        "production_resolve_exercised": True,
        "archive_authenticity_proved": False,
        "actual_runtime_delivery_proved": False,
    }
    write(destination / "archive-comparison.json", report)
    return report


def native_compare(kind, shape, compiler_python, destination):
    """Compare Lean-decoded archive cells with bit-preserving native MLIR tensor observations."""
    archived = archive_compare(kind, shape, destination)
    ty = ValueType(kind, shape)
    mlir = destination / "identity.mlir"
    mlir.write_text(f"module {{ func.func @identity(%a: {ty.mlir}) -> {ty.mlir} {{ return %a : {ty.mlir} }} }}\n")
    samples = [[words] for words in archived["raw_words"]]
    tools = Tools(compiler_python, Path("unused"), Path("unused"))
    observation = execute_mlir(
        mlir, samples, destination / "native", tools, inputs=[ty], outputs=[ty], function="identity"
    )
    # Native requests/results retain all input and output bits, including exceptional float values.
    native_bits = [[int(word) for word in sample] for sample in observation["results"]]
    assert native_bits == archived["decoded_words"], (archived, observation)
    report = {**archived, "native_words": native_bits, "all_output_bits_match": True, "samples": len(samples)}
    write(destination / "native-comparison.json", report)
    return report
