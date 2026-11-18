"""Dry-run an exported tensor inventory; this does not compile a body boundary.

The inventory preserves exported bytes. Runtime state, physical packing, staging,
images, and scratch must be supplied separately before claiming application fit.
"""
from argparse import ArgumentParser
import json
from pathlib import Path

from merlin.llvmlower.weights_pack import load_safetensors_header
from .residency import ResidencyRequest, ResidencyOverflow, plan_residency


def inventory(path, *, alignment, exclude=(), prefix=""):
    header, _ = load_safetensors_header(path)
    missing = set(exclude) - set(header)
    if missing:
        raise ValueError("excluded tensors absent from export: " + ", ".join(sorted(missing)))
    return [
        ResidencyRequest(name, "immutable_parameter", row["data_offsets"][1] - row["data_offsets"][0],
                         alignment, "application", "once_at_load")
        for name, row in sorted(header.items())
        if name != "__metadata__" and name not in exclude and name.startswith(prefix)
    ]


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--prefix", default="")
    for name, help_text in (
        ("capacity-base", "base of the usable arena, excluding host ELF, stack, and heap memory"),
        ("capacity-bytes", "usable arena size in bytes"),
        ("dma-base", "provider DMA aperture base"),
        ("dma-capacity", "provider DMA aperture size in bytes"),
        ("alignment", "provider allocation alignment in bytes"),
    ):
        parser.add_argument("--" + name, type=lambda text: int(text, 0), required=True, help=help_text)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    requests = inventory(args.weights, alignment=args.alignment, exclude=args.exclude, prefix=args.prefix)
    result = {
        "schema": "merlin.export_residency_inventory.v1",
        "exported_bytes": sum(request.bytes for request in requests),
        "tensors": len(requests), "excluded": args.exclude, "prefix": args.prefix,
        "scope": "exported tensor bytes only; no source-derived body boundary or runtime memory proof",
    }
    try:
        plan = plan_residency(requests, base=args.capacity_base, capacity=args.capacity_bytes,
                              dma_base=args.dma_base, dma_capacity=args.dma_capacity)
        result.update(status="fits_export_inventory", plan=plan.record(),
                      headroom_bytes=plan.capacity - plan.required_bytes)
        code = 0
    except ResidencyOverflow as error:
        result.update(status="refused", reason=str(error), required_bytes=error.required_bytes,
                      overflow_bytes=error.overflow_bytes)
        code = 2
    Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
