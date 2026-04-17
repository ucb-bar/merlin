#!/usr/bin/env python3
"""Extract dense_resource metadata and inline payloads from an MLIR file."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

RESOURCE_REF_RE = re.compile(r"dense_resource<([^>]+)>\s*:\s*tensor<([^>]+)>")
RESOURCE_BLOCK_RE = re.compile(r"\{-#(?P<body>.*)#-\}", re.DOTALL)
RESOURCE_PAYLOAD_RE = re.compile(r'([A-Za-z0-9_.$]+):\s*"0x([0-9A-Fa-f]*)"')


def parse_tensor_type(resource_type: str) -> tuple[list[int], str]:
    parts = resource_type.split("x")
    if len(parts) == 1:
        return [], parts[0]
    dims = [int(part) for part in parts[:-1]]
    return dims, parts[-1]


def dtype_nbytes(dtype: str) -> int:
    if dtype in {"bf16", "f16"}:
        return 2
    if dtype in {"f32", "i32"}:
        return 4
    if dtype == "i64":
        return 8
    if dtype in {"f8E4M3FN", "i8", "xi8", "ui8", "i1"}:
        return 1
    raise ValueError(f"Unsupported dense_resource dtype: {dtype}")


def expected_nbytes(resource_type: str) -> int:
    dims, dtype = parse_tensor_type(resource_type)
    return math.prod(dims or [1]) * dtype_nbytes(dtype)


def extract_payloads(text: str) -> dict[str, str]:
    match = RESOURCE_BLOCK_RE.search(text)
    if not match:
        return {}
    return {name: hex_payload for name, hex_payload in RESOURCE_PAYLOAD_RE.findall(match.group("body"))}


def build_manifest(text: str) -> dict[str, Any]:
    refs: dict[str, dict[str, Any]] = {}
    for name, resource_type in RESOURCE_REF_RE.findall(text):
        dims, dtype = parse_tensor_type(resource_type)
        entry = refs.setdefault(
            name,
            {
                "name": name,
                "tensor_type": resource_type,
                "shape": dims,
                "dtype": dtype,
                "uses": 0,
                "expected_nbytes": expected_nbytes(resource_type),
            },
        )
        entry["uses"] += 1

    payloads = extract_payloads(text)
    for name, payload in payloads.items():
        entry = refs.setdefault(
            name,
            {
                "name": name,
                "tensor_type": None,
                "shape": None,
                "dtype": None,
                "uses": 0,
                "expected_nbytes": None,
            },
        )
        entry["payload_nbytes"] = len(payload) // 2
        entry["payload_hex"] = payload
        expected = entry.get("expected_nbytes")
        if expected is None:
            entry["payload_data_offset"] = None
            entry["payload_data_nbytes"] = entry["payload_nbytes"]
            entry["payload_matches_expected"] = True
        elif expected == entry["payload_nbytes"]:
            entry["payload_data_offset"] = 0
            entry["payload_data_nbytes"] = entry["payload_nbytes"]
            entry["payload_matches_expected"] = True
        elif expected + 4 == entry["payload_nbytes"]:
            # MLIR byte blobs in these dumps carry a 4-byte prefix before the
            # actual tensor payload. Keep the prefix visible so the streaming
            # memory packer can skip it explicitly.
            entry["payload_data_offset"] = 4
            entry["payload_data_nbytes"] = expected
            entry["payload_matches_expected"] = True
        else:
            entry["payload_data_offset"] = None
            entry["payload_data_nbytes"] = entry["payload_nbytes"]
            entry["payload_matches_expected"] = False

    return {
        "resource_count": len(refs),
        "payload_count": len(payloads),
        "resources": [refs[name] for name in sorted(refs)],
    }


# Mapping from MLIR element-type names to (torch dtype, element-size bytes).
# Kept narrow on purpose — extend only when a new dtype actually appears in the
# demoted payloads.
_TORCH_DTYPE_MAP: dict[str, tuple[Any, int]] = {}


def _build_torch_dtype_map() -> dict[str, tuple[Any, int]]:
    """Lazy: torch import lives outside the script's CLI fast path."""
    import torch  # local import: keeps `--summary-only` runs torch-free

    return {
        "bf16": (torch.bfloat16, 2),
        "f16": (torch.float16, 2),
        "f32": (torch.float32, 4),
        "i32": (torch.int32, 4),
        "i64": (torch.int64, 8),
        "f8E4M3FN": (torch.float8_e4m3fn, 1),
        "i8": (torch.int8, 1),
        "xi8": (torch.int8, 1),
        "ui8": (torch.uint8, 1),
        "i1": (torch.bool, 1),
    }


def load_dense_resources(mlir_path: Path, base_addr: int = 0) -> list[tuple[int, Any]]:
    """Parse an MLIR file's ``dense_resource`` payloads into addressed tensors.

    Returns a list of ``(dram_address, tensor)`` pairs in IR-declaration order,
    laid out contiguously starting at ``base_addr``. Each tensor's element type
    matches the MLIR resource type; the 4-byte MLIR blob header that prefixes
    most payloads is stripped automatically (build_manifest already detects
    this and exposes it as ``payload_data_offset``).

    Resources without payload bytes (uses-only references) are skipped.
    """
    import torch  # noqa: F401  (used implicitly via dtype map)

    global _TORCH_DTYPE_MAP
    if not _TORCH_DTYPE_MAP:
        _TORCH_DTYPE_MAP = _build_torch_dtype_map()

    text = Path(mlir_path).read_text(errors="ignore")
    manifest = build_manifest(text)

    out: list[tuple[int, Any]] = []
    cursor = base_addr
    for resource in manifest["resources"]:
        payload_hex = resource.get("payload_hex")
        if not payload_hex:
            continue
        offset = resource.get("payload_data_offset")
        if offset is None:
            continue
        nbytes = resource["payload_data_nbytes"]
        # Slice off the MLIR header bytes (offset == 4 for normal blobs, 0 when
        # the payload size already matches `expected_nbytes`).
        data_hex = payload_hex[offset * 2 : (offset + nbytes) * 2]
        raw = bytes.fromhex(data_hex)
        dtype = resource.get("dtype")
        shape = resource.get("shape") or []
        if dtype not in _TORCH_DTYPE_MAP:
            raise ValueError(f"unsupported dtype {dtype!r} for resource {resource['name']!r}")
        torch_dtype, _ = _TORCH_DTYPE_MAP[dtype]
        # Build tensor from raw bytes via uint8 view + .view(target_dtype).
        import torch as _torch

        flat = _torch.frombuffer(bytearray(raw), dtype=_torch.uint8).clone()
        tensor = flat.view(torch_dtype)
        if shape:
            tensor = tensor.reshape(shape)
        out.append((cursor, tensor))
        cursor += nbytes
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MLIR dense_resource metadata and inline payloads")
    parser.add_argument("mlir", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--no-payload-hex",
        action="store_true",
        help="Omit payload_hex from JSON while keeping payload sizes",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print resource/payload counts and payload validation summary",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.mlir.read_text(errors="ignore"))
    if args.no_payload_hex:
        for resource in manifest["resources"]:
            resource.pop("payload_hex", None)

    mismatches = [
        resource["name"] for resource in manifest["resources"] if resource.get("payload_matches_expected") is False
    ]
    offset4_count = sum(1 for resource in manifest["resources"] if resource.get("payload_data_offset") == 4)

    if args.summary_only:
        print(f"resource_count: {manifest['resource_count']}")
        print(f"payload_count: {manifest['payload_count']}")
        print(f"payload_offset_4_count: {offset4_count}")
        print(f"payload_mismatch_count: {len(mismatches)}")
    else:
        encoded = json.dumps(manifest, indent=2, sort_keys=True)
        if args.out:
            args.out.write_text(encoded + "\n")
        else:
            print(encoded)

    if mismatches:
        print("FAILED: dense_resource payload size mismatch")
        for name in mismatches[:20]:
            print(f"- {name}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
