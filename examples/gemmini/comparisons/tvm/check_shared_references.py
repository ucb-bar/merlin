"""Recalculate every published chunk output without TVM or a simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_exported_chunks import compare, independent_values


def check(root, threads=2):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    torch.set_num_threads(threads)
    rows = []
    for program in sorted(root.glob("*/*/program.pt2")):
        folder = program.parent
        print("recalculating " + folder.name, flush=True)
        with np.load(folder / "fixture.npz", allow_pickle=False) as data:
            names = sorted((n for n in data if n.startswith("input_")), key=lambda n: int(n.split("_")[1]))
            inputs = tuple(torch.from_numpy(data[n].copy()) for n in names)
            names = (
                ["expected"]
                if "expected" in data
                else sorted((n for n in data if n.startswith("expected_")), key=lambda n: int(n.split("_")[1]))
            )
            expected = tuple(torch.from_numpy(data[n].copy()) for n in names)
        module = torch.export.load(program).module()
        observed = independent_values(module, inputs)
        rows.append(dict(chunk=folder.name, status="pass", comparison=compare(observed, expected)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.root, args.threads), indent=2) + "\n")


if __name__ == "__main__":
    main()
