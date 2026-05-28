#!/usr/bin/env python3
"""Materialize readonly constant subspans in a per-dispatch MLIR module.

IREE's HAL dispatch wrappers pass model constants as readonly subspans of a
large packed arena. QNN_HTA validates Conv2d weights as static tensors, not as
runtime APP_WRITE tensors, so feeding the constant arena as a dispatch input is
not sufficient. This tool rewrites loads from readonly, non-indirect
`hal.interface.binding.subspan` values into `arith.constant dense<...>` values
using bytes sliced from an extracted constant arena.

The rewrite is intentionally narrow:
  - only `hal.interface.binding.subspan` ops with `flags(ReadOnly)` are eligible;
  - subspans containing `Indirect` are left as runtime inputs/outputs;
  - only immediately loaded tensor subspans are materialized;
  - the output is verified later by `iree-compile`.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import struct

_INDEX_CONST_RE = re.compile(r"%(?P<name>c-?\d+)\s*=\s*arith\.constant\s+(?P<value>-?\d+)\s*:\s*index")
_SUBSPAN_RE = re.compile(
    r"^\s*(?P<ssa>%[\w\d_]+)\s*=\s*hal\.interface\.binding\.subspan"
    r".*?binding\((?P<binding>\d+)\).*?offset\(%(?P<offset>[\w\d_]+)\)"
    r".*?flags\((?P<flags>[^)]*)\).*?:\s*(?P<dispatch_ty>!iree_tensor_ext\.dispatch\.tensor<[^>]+>)",
    re.MULTILINE,
)
_LOAD_RE = re.compile(
    r"^(?P<indent>\s*)(?P<ssa>%[\w\d_]+)\s*=\s*iree_tensor_ext\.dispatch\.tensor\.load\s+"
    r"(?P<src>%[\w\d_]+),.*?->\s*(?P<tensor_ty>tensor<[^>]+>)\s*$",
    re.MULTILINE,
)
_TENSOR_RE = re.compile(r"tensor<(?P<body>[^>]+)>")


def _parse_tensor_type(tensor_ty: str) -> tuple[list[int], str]:
    match = _TENSOR_RE.fullmatch(tensor_ty)
    if not match:
        raise ValueError(f"unsupported tensor type: {tensor_ty}")
    parts = match.group("body").split("x")
    elem = parts[-1]
    dims = [int(p) for p in parts[:-1]]
    return dims, elem


def _elem_size(elem: str) -> int:
    if elem in ("i8", "si8", "ui8"):
        return 1
    if elem in ("i16", "si16", "ui16", "f16"):
        return 2
    if elem in ("i32", "si32", "ui32", "f32"):
        return 4
    if elem in ("i64", "si64", "ui64", "f64"):
        return 8
    raise ValueError(f"unsupported element type: {elem}")


def _num_elements(dims: list[int]) -> int:
    n = 1
    for dim in dims:
        n *= dim
    return n


def _dense_attr(raw: bytes, tensor_ty: str) -> str:
    dims, elem = _parse_tensor_type(tensor_ty)
    count = _num_elements(dims)
    expected = count * _elem_size(elem)
    if len(raw) != expected:
        raise ValueError(f"{tensor_ty}: got {len(raw)} bytes, expected {expected}")

    if elem in ("i8", "si8"):
        return f'dense<"0x{raw.hex().upper()}"> : {tensor_ty}'
    if elem == "ui8":
        return f'dense<"0x{raw.hex().upper()}"> : {tensor_ty}'
    if elem in ("i32", "si32"):
        values = struct.unpack("<" + "i" * count, raw)
        return f"dense<{list(values)}> : {tensor_ty}".replace("[", "[").replace("]", "]")
    if elem == "ui32":
        values = struct.unpack("<" + "I" * count, raw)
        return f"dense<{list(values)}> : {tensor_ty}".replace("[", "[").replace("]", "]")
    raise ValueError(f"materializing {elem} constants is not implemented")


def _constant_line(indent: str, ssa: str, raw: bytes, tensor_ty: str) -> str:
    return f"{indent}{ssa} = arith.constant {_dense_attr(raw, tensor_ty)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mlir", type=pathlib.Path, required=True)
    parser.add_argument("--constant-arena", type=pathlib.Path, required=True)
    parser.add_argument("--out-mlir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--constant-binding",
        type=int,
        default=None,
        help="Only materialize this binding id. Default: all readonly non-indirect bindings.",
    )
    args = parser.parse_args()

    text = args.input_mlir.read_text()
    arena = args.constant_arena.read_bytes()

    index_consts = {m.group("name"): int(m.group("value")) for m in _INDEX_CONST_RE.finditer(text)}
    subspans: dict[str, tuple[int, int]] = {}
    for match in _SUBSPAN_RE.finditer(text):
        flags = match.group("flags").strip('"')
        if "ReadOnly" not in flags or "Indirect" in flags:
            continue
        binding = int(match.group("binding"))
        if args.constant_binding is not None and binding != args.constant_binding:
            continue
        offset_name = match.group("offset")
        if offset_name not in index_consts:
            raise SystemExit(f"missing index constant %{offset_name}")
        subspans[match.group("ssa")] = (binding, index_consts[offset_name])

    replacements = 0

    def replace_load(match: re.Match[str]) -> str:
        nonlocal replacements
        src = match.group("src")
        if src not in subspans:
            return match.group(0)
        _binding, offset = subspans[src]
        tensor_ty = match.group("tensor_ty")
        dims, elem = _parse_tensor_type(tensor_ty)
        size = _num_elements(dims) * _elem_size(elem)
        end = offset + size
        if end > len(arena):
            raise ValueError(f"{src} slice [{offset}, {end}) exceeds arena size {len(arena)}")
        replacements += 1
        return _constant_line(match.group("indent"), match.group("ssa"), arena[offset:end], tensor_ty)

    rewritten = _LOAD_RE.sub(replace_load, text)
    args.out_mlir.parent.mkdir(parents=True, exist_ok=True)
    args.out_mlir.write_text(rewritten)
    print(f"wrote {args.out_mlir}; materialized {replacements} constant loads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
