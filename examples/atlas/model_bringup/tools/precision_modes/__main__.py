"""Offline emulation of Atlas precision recipes; prints a JSON report. With no recipe flag, runs the frozen default."""

import argparse
import contextlib
import importlib
import json
from pathlib import Path
import sys

PATHS = {"tiny": ["capture", "checkpoint", "inputs"], "smol": ["checkpoint", "backbone", "observation", "reference"],
         "resnet": ["artifact-dir", "checkpoint", "input", "reference", "labels"]}
OPTIONAL = ("inputs", "reference", "labels")

parser = argparse.ArgumentParser(prog="python -m precision_modes", description=__doc__)
models = parser.add_subparsers(dest="model", required=True)
for name, paths in PATHS.items():
    model = models.add_parser(name)
    for path in paths:
        model.add_argument("--" + path, type=Path, required=path not in OPTIONAL)
    model.add_argument("--finer-scales", nargs="?", const="both", choices=("activation", "weight", "both"),
                       help="per-row activation and/or per-column weight scales, at every site")
    model.add_argument("--k-split", type=int, default=0, metavar="K", help="K chunk, a multiple of 32")
    model.add_argument("--two-term", action="store_true", help="add an FP8 weight residual term")
    model.add_argument("--sites", default="*", help="comma-separated site globs, or @file, for K split and two-term")
models.choices["tiny"].add_argument("--steps", type=int, default=3)
args = parser.parse_args()
if args.k_split % 32 or args.k_split < 0:
    parser.error("--k-split must be a positive multiple of 32")
args.sites = Path(args.sites[1:]).read_text().split() if args.sites.startswith("@") else args.sites.split(",")

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):  # model libraries print progress
    report = importlib.import_module(f".{args.model}", __package__).run(args)
report["recipe"] = dict(finer_scales=args.finer_scales, k_split=args.k_split, two_term=args.two_term, sites=args.sites)
print(json.dumps(report, indent=2))
